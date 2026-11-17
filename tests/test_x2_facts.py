"""chip/x2_facts.py: every X2/chip number that silicon has not confirmed, in one place, replaceable without code changes.

* every fact says its value, status, source, who uses it and when to replace it; 'missing' facts refuse to be used;
* specifications/x2_overrides.json changes what the schedule and the timing check use; malformed overrides are rejected;
* the project clock is the highest clock every chip design met (with margin), and the chain judges timing against it;
* chip_flow puts the chip toolchain first and refuses an old Yosys (the 0.66 'Unknown option' failure of 2026-11-16).
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chip import chip_flow as cf  # noqa: E402
from chip import x2_facts as xf  # noqa: E402


@pytest.fixture
def overrides(tmp_path, monkeypatch):
    p = tmp_path / "x2_overrides.json"
    monkeypatch.setattr(xf, "OVERRIDES", p)
    return p


def test_every_fact_is_complete():
    for k, f in xf.FACTS.items():
        assert f["status"] in xf.STATUSES and f["source"] and f["unit"] and f["replace_when"] and isinstance(f["used_by"], list), k
        assert (f["value"] is None) == (f["status"] == "missing"), k
    assert xf.FACTS["cell_bits"]["value"] == 4 and xf.FACTS["blocks"]["value"] == 2


def test_missing_facts_refuse_to_be_used(overrides):
    with pytest.raises(xf.FactMissing, match="x2_overrides.json"):
        xf.value("cell_compute_4bit")


def test_no_override_file_means_the_defaults(overrides):
    assert not overrides.exists() and xf.load_overrides() == {}
    assert xf.value("program_cycles_per_cell") == 365 and xf.value("clock_mhz") == 14.0


def test_an_override_reaches_the_schedule(overrides):
    from compiler import x2_schedule as xs
    from compiler.partition import load_hardware, propose_partition_baseline
    sys.path.insert(0, str(ROOT / "tests"))
    from x2_schedule_graphs import mini
    g, hw = mini(), load_hardware()
    chip = propose_partition_baseline(g, hw)["chip"]
    before = xs.tile_costs(chip, hw)["c0"][0]["program"]
    overrides.write_text(json.dumps({"program_cycles_per_cell": {"value": 730, "status": "measured", "source": "test"}}))
    assert xs.tile_costs(chip, hw)["c0"][0]["program"] == 2 * before
    assert xf.summary()["program_cycles_per_cell"] == {"value": 730, "status": "measured", "overridden_by": "x2_overrides.json"}
    overrides.write_text(json.dumps({"cell_compute_4bit": {"value": "sum(pwm*level)", "status": "decision", "source": "team"}}))
    assert xf.value("cell_compute_4bit") == "sum(pwm*level)"                      # a missing fact can be filled in


@pytest.mark.parametrize("bad,msg", [
    ([1], "must be an object"), ({"nope": {"value": 1, "status": "model", "source": "x"}}, "unknown fact"),
    ({"clock_mhz": 15}, "needs value, status"), ({"clock_mhz": {"value": 15, "status": "guess", "source": "x"}}, "needs value"),
    ({"clock_mhz": {"value": 15, "status": "measured", "source": ""}}, "needs value"),
])
def test_malformed_overrides_are_rejected(overrides, bad, msg):
    overrides.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match=msg):
        xf.value("clock_mhz")


def test_clock_is_met_by_every_design_and_used_by_the_chain(overrides):
    from verification import chip_chain as cc
    recorded = {"x2_engine2_4bit": 14.45, "x2_engine2": 16.67, "x2_engine": 17.51, "relu8": 151.52, "add8_sat": 42.19}
    assert xf.value("clock_mhz") <= min(recorded.values())
    rep = {"status": "ok", "chip_repo_read_only": True, "bitstream_audit": {"set_bits": 1, "required_bits": 1}}
    sim = {"status": "pass", "reason": "ok"}
    fast = {"clk": {"fmax_mhz": 14.2, "target_mhz": 12.0, "ok": True}}
    slow = {"clk": {"fmax_mhz": 13.9, "target_mhz": 12.0, "ok": True}}           # nextpnr says PASS at its 12 MHz ...
    assert cc.judge(rep, fast, sim)[0] == "pass"
    st, detail, facts = cc.judge(rep, slow, sim)
    assert st == "unverified" and "13.9 MHz < 14.0 MHz" in detail                 # ... but not at the project clock


def test_schedule_reports_microseconds(overrides):
    from compiler import x2_schedule as xs
    from compiler.partition import load_hardware, propose_partition_baseline
    sys.path.insert(0, str(ROOT / "tests"))
    from x2_schedule_graphs import mini
    g, hw = mini(), load_hardware()
    chip = propose_partition_baseline(g, hw)["chip"]
    _, m = xs.validate_schedule(g, chip, hw, xs.materialize(g, chip, hw, [], "earliest_free"))
    assert m["clock_mhz"] == 14.0 and m["latency_us"] == round(m["latency_cycles"] / 14.0, 1)


def test_tool_env_puts_the_chip_toolchain_first(tmp_path):
    (tmp_path / "chipenv" / "bin").mkdir(parents=True)
    (tmp_path / "tools" / "chip" / "oss-cad-suite" / "bin").mkdir(parents=True)
    import os
    old_path = os.pathsep.join(["/home/u/tools/oss-cad-suite/bin", "/usr/bin"])
    parts = cf.tool_env({"HOME": str(tmp_path), "PATH": old_path})["PATH"].split(os.pathsep)   # ';' on Windows (C:\...)
    assert parts[:2] == [str(tmp_path / "chipenv" / "bin"), str(tmp_path / "tools" / "chip" / "oss-cad-suite" / "bin")]
    assert parts[2:] == old_path.split(os.pathsep)                                   # the old suite stays, but after ours
    assert cf.tool_env({"HOME": str(tmp_path / "nothing"), "PATH": "/usr/bin"})["PATH"].split(os.pathsep)[-1] == "/usr/bin"


def test_yosys_version_check():
    assert cf.yosys_version("Yosys 0.66+154 (git sha1 23aadd92a)") == (0, 66) < cf.MIN_YOSYS
    assert cf.yosys_version("Yosys 0.70+368 (git sha1 7cf88aca6)") == (0, 70) >= cf.MIN_YOSYS
    assert cf.yosys_version("bash: yosys: not found") is None
