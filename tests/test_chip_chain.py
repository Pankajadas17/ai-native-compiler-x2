"""Flow step 5: the chain on the BM Labs chip (run_fab_chain.py --target chip, verification/chip_chain.py).

* which operators fit the chip's 36 pins, and the wrapper for each passes Yosys vs NumPy (faults are caught);
* recorded netlist simulations (real chip flow, sandbox) are judged right: relu8 / add8_sat pass, requant fails with the
  hard-multiplier cause named;
* the check stage: timing met / not met / no register paths, a failed flow, a modified chip repo;
* old results can never be read as this run's (cleared report status), Ubuntu-side faults block or fail, never pass;
* the full six-stage chain with a fake Ubuntu gives PASS only when every stage passed.
"""
import json
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from verification import chip_chain as cc  # noqa: E402
from verification import fab_chain as fc  # noqa: E402
from verification.rtl_check import _yosys  # noqa: E402

FIX = ROOT / "tests" / "fixtures" / "chip_chain"


def have_yosys():
    try:
        _yosys()
        return True
    except RuntimeError:
        return False


needs_yosys = pytest.mark.skipif(not have_yosys(), reason="Yosys (yowasp-yosys) not installed")


def reference(op):
    from agents.rtl_agent import REFERENCE_VERILOG
    return REFERENCE_VERILOG[op]


# ------------------------------------------------------------------ pins and wrappers
def test_pins_and_operators():
    assert cc.chip_pins() == 36
    ok, no = cc.wrappable_ops()
    assert ok == ["relu8", "add8_sat", "requant_i32_i8"]
    assert "x2_engine" in no["x2_acc_step"] and "84" in no["x2_acc_step"]
    assert cc.chip_layout("relu8")["pins"] == 17 and cc.chip_layout("add8_sat")["pins"] == 25
    lay = cc.chip_layout("requant_i32_i8")
    assert lay["mode"] == "two_step" and lay["pins"] == 26


@needs_yosys
@pytest.mark.parametrize("op,flops", [("relu8", 16), ("add8_sat", 24), ("requant_i32_i8", 40)])
def test_wrapper_passes(op, flops):
    r = cc.check_chip_wrapper(op, cc.make_chip_wrapper(op, reference(op)))
    assert r["status"] == "pass" and r["flop_bits"] == flops, r


@needs_yosys
def test_wrapper_faults_are_caught():
    bad_core = reference("relu8").replace("8'sd0 : x", "8'sd1 : x")
    assert cc.check_chip_wrapper("relu8", cc.make_chip_wrapper("relu8", bad_core))["status"] == "fail"
    two = cc.make_chip_wrapper("requant_i32_i8", reference("requant_i32_i8"))
    broken = two.replace("assign lo_n = sel ? lo : d;", "assign lo_n = d;")        # the high tick overwrites the low half
    r = cc.check_chip_wrapper("requant_i32_i8", broken)
    assert r["status"] == "fail" and r["stage"] == "function"
    renamed = cc.make_chip_wrapper("add8_sat", reference("add8_sat")).replace("input  wire [7:0] b,", "input  wire [6:0] b,", 1)
    assert cc.check_chip_wrapper("add8_sat", renamed)["status"] == "fail"


def test_testbench_uses_pins_only():
    for op in ("relu8", "add8_sat", "requant_i32_i8"):
        tb = cc.netlist_tb(op)
        inst = re.search(r"\n\s*" + cc.top_name(op) + r" dut \((.*?)\);", tb).group(1)
        assert ".clk(clk)" in inst and "step" not in inst and "core" not in inst
        assert f"DONE %0d\", {len(next(iter(cc.vectors_for(op).values())))}" in tb


def test_scripts_put_chipenv_before_the_cad_suite():
    """The repo Makefile runs bit_gen.py with the first python3 on PATH; OSS CAD Suite ships one without fasm."""
    for text in (cc.PREFLIGHT_SCRIPT, cc.STEP_SCRIPT):
        assert 'export PATH="$(dirname "$PY"):$CAD/bin:$PATH' in text.replace("{{", "{").replace("}}", "}")
    assert "MISSING fasm or numpy in the Makefile's python3" in cc.PREFLIGHT_SCRIPT


def test_missing_fasm_is_named():
    from chip.chip_flow import classify
    tail = "    import fasm\nModuleNotFoundError: No module named 'fasm'\nmake: *** [makefile.include:90: bitstream] Error 1"
    assert classify("bitstream", tail) == "python_module_missing"


def test_iobuf_stand_in_has_chip_polarity():
    text = (ROOT / "sim" / "IOBUF_chip_sim.v").read_text()
    assert "assign PAD = EN_mux ? IN_mux : 1'bz;" in text and "module IOBUF" in text


# ------------------------------------------------------------------ recorded netlist simulations
def test_recorded_netlist_sims():
    assert cc.parse_netlist_log("relu8", (FIX / "relu8_sim.log").read_text())["status"] == "pass"
    assert cc.parse_netlist_log("add8_sat", (FIX / "add8_sat_sim.log").read_text())["status"] == "pass"
    r = cc.parse_netlist_log("requant_i32_i8", (FIX / "requant_i32_i8_sim.log").read_text())
    assert r["status"] == "fail" and r["reason"].startswith("219/485")
    assert all(v < 0 for b in r["all_mismatches"] for v in b["inputs"].values())
    why = cc.explain_mismatch("requant_i32_i8", r, "  MACC_8x8_20 #(\n" * 4)
    assert why and "4 MACC_8x8_20" in why and "MACC_8x8_20_map_signed.v" in why
    assert cc.explain_mismatch("requant_i32_i8", r, "no multipliers here") is None


def test_corrupted_and_unfinished_logs():
    text = (FIX / "relu8_sim.log").read_text()
    m = re.search(r"^VEC 200 ([01]{8})$", text, re.M)
    flipped = text.replace(m.group(0), "VEC 200 " + ("1" if m.group(1)[0] == "0" else "0") + m.group(1)[1:])
    assert cc.parse_netlist_log("relu8", flipped)["status"] == "fail"
    undefined = re.sub(r"^VEC 5 [01]{8}$", "VEC 5 xxxxxxxx", text, flags=re.M)
    r = cc.parse_netlist_log("relu8", undefined)
    assert r["status"] == "fail" and "1 undefined" in r["reason"]
    assert cc.parse_netlist_log("relu8", text.split("DONE")[0])["status"] == "unverified"
    assert cc.parse_netlist_log("relu8", "")["status"] == "unverified"


# ------------------------------------------------------------------ check stage
def report():
    return json.loads((FIX / "flow_report_relu8.json").read_text())


def sim_pass():
    return cc.parse_netlist_log("relu8", (FIX / "relu8_sim.log").read_text())


def test_timing_lines():
    t = cc.final_timing((FIX / "pnr_relu8.log").read_text())
    assert list(t.values()) == [{"fmax_mhz": 151.52, "target_mhz": 12.0, "ok": True}]           # the LAST line per clock
    late = cc.final_timing((FIX / "pnr_timing_fail.log").read_text())
    assert late["engine.clk"] == {"fmax_mhz": 11.2, "target_mhz": 12.0, "ok": False}
    assert cc.final_timing((FIX / "pnr_no_paths.log").read_text()) == {}


def test_judge():
    rep, t = report(), cc.final_timing((FIX / "pnr_relu8.log").read_text())
    assert cc.judge(rep, t, sim_pass())[0] == "pass"
    st, detail, facts = cc.judge(rep, cc.final_timing((FIX / "pnr_timing_fail.log").read_text()), sim_pass())
    assert st == "unverified" and "NOT met" in detail
    st, _, facts = cc.judge(rep, {}, sim_pass())
    assert st == "pass" and "register-to-register" in facts["not_checked"][0]
    assert cc.judge(rep, t, None)[0] == "unverified"
    assert cc.judge(rep, t, {"status": "fail", "reason": "x"})[0] == "fail"
    assert cc.judge({**rep, "status": "failed", "failure": {"cause": "pins"}}, t, sim_pass())[0] == "fail"
    assert cc.judge({**rep, "chip_repo_read_only": False}, t, sim_pass())[0] == "fail"
    assert cc.judge({"design": "chain_relu8", "status": "cleared"}, t, sim_pass())[0] == "fail"


# ------------------------------------------------------------------ stages with a fake Ubuntu
def cfg():
    return SimpleNamespace(runner="native", distro="Ubuntu", heartbeat_s=30, timeout_s=60, from_stage="rtl")


PREFLIGHT_OK = "\n".join(["FOUND yosys", "FOUND nextpnr-generic", "FOUND iverilog", "FOUND vvp", "FOUND make", "FOUND chipenv",
                          "FOUND fasm+numpy", "FOUND make python3", "FOUND chip repo", "FOUND chip profile", "VERSION yosys Yosys 0.70+368 (git sha1 x)",
                          "VERSION nextpnr-generic \"nextpnr-generic\" -- (Version nextpnr-0.11.1-89-gabc)",
                          "VERSION iverilog Icarus Verilog version 14.0 (devel)", "PREFLIGHT DONE"])


class FakeUbuntu:
    """Answers the preflight, and for flow / sim writes what the real Ubuntu side would (from the fixtures)."""
    def __init__(self, op, flow_ok=True, sim_log=None, preflight=PREFLIGHT_OK):
        self.op, self.flow_ok, self.sim_log, self.preflight, self.calls = op, flow_ok, sim_log, preflight, []

    def __call__(self, cmd, timeout, hb, say):
        self.calls.append(cmd)
        stage = cmd[-1]
        if stage not in ("flow", "sim"):
            return {"started": True, "returncode": 0, "output": self.preflight, "seconds": 0.1, "timed_out": False}
        b = Path("chip_build") / f"chain_{self.op}"
        if stage == "flow" and self.flow_ok:
            b.mkdir(parents=True, exist_ok=True)
            (b / "flow_report.json").write_text((FIX / "flow_report_relu8.json").read_text())
            (b / "pnr.log").write_text((FIX / "pnr_relu8.log").read_text())
        if stage == "sim" and self.sim_log is not None:
            (Path("reports/chip_jobs") / self.op / "sim.log").write_text(self.sim_log)
        return {"started": True, "returncode": 0, "output": "", "seconds": 0.2, "timed_out": False}


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "reports" / "rtl").mkdir(parents=True)
    (tmp_path / "reports" / "rtl" / "relu8.v").write_text(reference("relu8"))
    return tmp_path


def chain(ubuntu, op="relu8", rtl=lambda: ("pass", "reference core")):
    return fc.run_chain(op, cc.build_chip_steps(op, cfg(), ubuntu, {}, rtl, say=lambda m: None), "rtl", say=lambda m: None)


@needs_yosys
def test_full_chain_passes_with_fake_ubuntu(work):
    u = FakeUbuntu("relu8", sim_log=(FIX / "relu8_sim.log").read_text())
    r = chain(u)
    assert r["verdict"] == "PASS", r["stages"]
    assert [c[-1] for c in u.calls][-2:] == ["flow", "sim"]
    job = Path("reports/chip_jobs/relu8")
    assert (job / "relu8_chip.v").is_file() and (job / "relu8_chip_tb.v").is_file()
    sh = (job / "run_chip_step.sh").read_text()
    assert "--name \"chain_relu8\" --top \"relu8_chip\"" in sh and "\r" not in sh
    assert json.loads((job / "chip_check.json").read_text())["status"] == "pass"


@needs_yosys
def test_stale_report_is_never_this_runs(work):
    chain(FakeUbuntu("relu8", sim_log=(FIX / "relu8_sim.log").read_text()))          # an earlier passing run
    removed = cc.clean_job("relu8", "rtl")
    assert any("status cleared" in x for x in removed) and "sim.log" in removed
    r = chain(FakeUbuntu("relu8", flow_ok=False, sim_log=(FIX / "relu8_sim.log").read_text()))   # this run's flow writes nothing
    rows = {x["stage"]: x for x in r["stages"]}
    assert r["verdict"] == "FAIL" and rows["flow"]["status"] == "fail" and "cleared" in rows["flow"]["detail"]
    assert rows["sim"]["status"] == "not_run"


@needs_yosys
def test_missing_tools_block(work):
    r = chain(FakeUbuntu("relu8", preflight=PREFLIGHT_OK.replace("FOUND chipenv", "MISSING chipenv python (/home/u/chipenv/bin/python)")))
    rows = {x["stage"]: x for x in r["stages"]}
    assert r["verdict"] == "NOT RUN" and rows["env"]["status"] == "blocked" and "chipenv" in rows["env"]["detail"]


@needs_yosys
def test_wrong_netlist_fails(work):
    text = (FIX / "relu8_sim.log").read_text().replace("VEC 0 00000000", "VEC 0 00000001")
    r = chain(FakeUbuntu("relu8", sim_log=text))
    assert r["verdict"] == "FAIL" and {x["stage"]: x["status"] for x in r["stages"]}["sim"] == "fail"


def test_no_sim_log_fails(work):
    r = chain(FakeUbuntu("relu8"), rtl=lambda: ("pass", "x"))
    assert not r["verdict"].startswith("PASS")


# ------------------------------------------------------------------ the CLI
def test_plan_through_run_fab_chain(capsys):
    import run_fab_chain
    assert run_fab_chain.main(["--op", "all", "--target", "chip", "--plan", "--runner", "native"]) == 0
    out = capsys.readouterr().out
    assert "skipping x2_acc_step" in out and "chip plan for requant_i32_i8" in out and "run_chip_step.sh flow" in out


def test_live_chain(tmp_path):
    import os
    tools = all(shutil.which(t) for t in ("yosys", "nextpnr-generic", "iverilog", "vvp", "make"))
    chipenv = Path(os.environ.get("CHIP_PY", Path.home() / "chipenv" / "bin" / "python")).is_file()
    if not (tools and chipenv) or sys.platform.startswith("win"):
        pytest.skip("live chip chain runs in Ubuntu: python3 run_fab_chain.py --op relu8 --target chip --runner native")
    import run_fab_chain
    assert run_fab_chain.main(["--op", "relu8", "--target", "chip", "--runner", "native"]) == 0
