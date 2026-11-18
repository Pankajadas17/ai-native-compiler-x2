"""Phase 3a: chip flow runner (build any design for classic_fabric_reram).

Offline tests use real tool logs recorded in the sandbox (tests/fixtures/chip_flow/).
The live test runs the examples if yosys/nextpnr are on PATH (Ubuntu), else it skips.
"""
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from x2c.chip import chip_flow as cf  # noqa: E402

FIX = ROOT / "tests" / "fixtures" / "chip_flow"
PROFILE = ROOT / "specifications" / "chip_profile.json"


def text(name):
    return (FIX / name).read_text()


def profile_or_skip():
    if not PROFILE.is_file():
        pytest.skip("specifications/chip_profile.json missing (run x2c chip profile)")
    return json.loads(PROFILE.read_text())


# ---------------------------------------------------------------- parsers on real logs
def test_utilisation_parsed_from_real_nextpnr_log():
    u = cf.parse_utilisation(text("counter.pnr.log"))
    assert u["FABULOUS_LC"] == {"used": 9, "available": 2160}
    assert u["IOBUF"] == {"used": 11, "available": 36}
    assert u["RERAM_CIM_32x32"]["available"] == 2
    assert set(u) == {"FABULOUS_LC", "IOBUF", "RAM_32x4_2R_1W", "MACC_8x8_20", "RERAM_CIM_32x32", "GBUF", "SYS_RESET"}


def test_over_capacity_visible_in_utilisation():
    u = cf.parse_utilisation(text("macc_over.pnr.log"))
    assert u["MACC_8x8_20"] == {"used": 20, "available": 18}


def test_ports_and_pin_names_from_real_netlist():
    ports = cf.ports_from_netlist(json.loads(text("counter.netlist.json")), "counter")
    assert sum(w for _, _, w in ports) == 11
    bits = cf.port_bits(ports)
    assert len(bits) == 11 and len(set(bits)) == 11


def test_pin_plan_uses_all_36_sites_and_refuses_37():
    sites = [f"X0Y{i}/{ab}" for i in range(1, 19) for ab in "AB"]
    assert len(sites) == 36
    fits = cf.pin_plan([("a", "input", 35), ("clk", "input", 1)], sites)
    assert fits["fits"] and fits["needed"] == 36 and fits["assignment"]["clk"] == sites[-1]
    too_many = cf.pin_plan([("a", "input", 36), ("clk", "input", 1)], sites)
    assert not too_many["fits"] and too_many["assignment"] == {}


def test_pin_plan_against_real_profile():
    p = profile_or_skip()
    sites = p["io"]["iobuf_site_list"]
    assert len(sites) == 36 and len(set(sites)) == 36
    plan = cf.pin_plan([("count", "input", 14), ("k", "input", 3), ("clk", "input", 1), ("rst", "input", 1),
                        ("en", "input", 1), ("acc", "output", 22)], sites)
    assert plan["needed"] == 42 and not plan["fits"]


def test_write_pcf_roundtrip(tmp_path):
    cf.write_pcf(tmp_path / "d.pcf", {"clk": "X0Y1/A", "a[0]": "X0Y1/B"})
    assert (tmp_path / "d.pcf").read_text().splitlines()[1:] == ["set_io clk X0Y1/A", "set_io a[0] X0Y1/B"]


def test_expected_bitstream_size_matches_real_bitstream():
    # 21 columns x 20 frames x (4 + 20 rows x 32 bits / 8) + 24 = 35,304 bytes (the counter.bit size)
    assert cf.expected_bitstream_bytes(21, 20, 20, 32) == 35304


def test_frame_geometry_from_repo():
    from x2c.chip.chip_repo import ChipRepo, ChipRepoError
    try:
        repo = ChipRepo()
    except ChipRepoError as e:
        pytest.skip(str(e))
    assert cf.frame_geometry(repo) == {"FrameBitsPerRow": 32, "MaxFramesPerCol": 20}


def test_constrained_pins_and_silent_unconstrained_pin():
    full = cf.constrained_pins(text("counter.pnr.log"))
    one_missing = cf.constrained_pins(text("counter_one_pin_missing.pnr.log"))
    assert len(full) == 11 and len(one_missing) == 10


# ---------------------------------------------------------------- failure classification (A3 input)
@pytest.mark.parametrize("stage,log,extra,cause", [
    ("synth", "bitslice_acc_bad.synth.log", None, "syntax_error"),
    ("synth", "missing_mod.synth.log", None, "unknown_module"),
    ("pnr", "macc_over.pnr.log", None, "over_capacity"),
    ("pins", "", None, "pin_budget_exceeded"),
    ("check", "", {"fasm_features": 0}, "empty_fasm"),
    ("check", "", {"unconstrained": ["c[3]"]}, "unconstrained_io"),
    ("check", "", {"audit": {"missing": 1}}, "bitstream_audit_failed"),
    ("bitstream", "python_missing.bitstream.log", None, "python_module_missing"),
])
def test_classify_real_failures(stage, log, extra, cause):
    tail = "\n".join(text(log).splitlines()[-25:]) if log else ""
    assert cf.classify(stage, tail, extra) == cause


def test_classify_never_guesses():
    assert cf.classify("pnr", "Info: something new happened", None) == "unknown"
    assert cf.classify("synth", "ERROR: brand new message", None) == "unknown"


def test_every_rule_has_recorded_evidence():
    causes = {"pin_budget_exceeded", "syntax_error", "unknown_module", "ff_type_unsupported", "over_capacity",
              "unconstrained_io", "routing_failed", "empty_fasm", "bitstream_size_mismatch",
              "bitstream_audit_failed", "python_module_missing"}
    assert set(cf.RULE_EVIDENCE) == causes


@pytest.mark.parametrize("name,status,cause", [
    ("counter", "ok", None), ("bitslice_acc_mux", "ok", None),
    ("bitslice_acc_parallel", "failed", "pin_budget_exceeded"), ("bitslice_acc_bad", "failed", "syntax_error"),
    ("missing_mod", "failed", "unknown_module"), ("macc_over", "failed", "over_capacity"),
])
def test_recorded_reports(name, status, cause):
    r = json.loads(text(f"{name}.flow_report.json"))
    assert r["status"] == status and (r["failure"] or {}).get("cause") == cause
    assert r["chip_repo_read_only"] is True
    if status == "ok":
        assert r["fasm_features"] > 0 and r["bitstream_bytes"] == r["bitstream_expected_bytes"] == 35304
        assert r["pins"]["needed"] == len(r["pins"]["assignment"])


# ---------------------------------------------------------------- live (Ubuntu)
def test_live_examples(tmp_path):
    if not all(shutil.which(t) for t in cf.TOOLS):
        pytest.skip("yosys/nextpnr not on PATH (expected on Windows; run x2c build --examples in Ubuntu)")
    profile_or_skip()
    import x2c.commands.build as run_chip_flow
    assert run_chip_flow.main(["--examples", "--out", str(tmp_path)]) == 0
