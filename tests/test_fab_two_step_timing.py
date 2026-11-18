"""Two-step load (requant_i32_i8 does not fit the 28 pins) and timing as a real result.
The timing lines are copied from a real nextpnr 0.10-82 log of this design on the FABulous 2.2.0 demo fabric (run in a Linux sandbox with the
same tool versions as the project machine): 7.19 MHz against the 12 MHz default -> nextpnr stops; with --freq 5 it passes."""
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

import x2c.commands.chain as chain
from x2c.commands.demo_prep import FLOW_SCRIPT
from tests.test_fab_chain import FakeUbuntu, cfg, proj, result, run_main  # noqa: F401  (proj is a fixture)
from x2c.verification import chain_debug as cd
from x2c.verification import fab_chain as fc
from x2c.verification.fab_sim import make_fab_testbench, sim_vectors, tb_vector_count
from x2c.verification.fab_wrap import wrapper_layout
from x2c.verification.rtl_check import OPS, vectors_for

needs_bash = pytest.mark.skipif(sys.platform == "win32" or not (Path("/bin/bash").exists() or Path("/usr/bin/bash").exists()),
                                reason="runs real bash (on Windows `bash` is WSL's launcher and cannot open Windows temp paths)")
REAL_FAIL = """Info: Max frequency for clock 'clk': 4.38 MHz (FAIL at 12.00 MHz)
Info: Slack histogram:
ERROR: Max frequency for clock 'clk': 7.19 MHz (FAIL at 12.00 MHz)
0 warnings, 1 error
"""
REAL_PASS = """Info: Max frequency for clock 'clk': 4.38 MHz (FAIL at 5.00 MHz)
Info: Max frequency for clock 'clk': 7.35 MHz (PASS at 5.00 MHz)
"""


# ------------------------------------------------------------ the testbench for a two-step load
def test_two_step_vectors_are_the_two_halves_and_python_arithmetic():
    lay, vals = wrapper_layout("requant_i32_i8"), vectors_for("requant_i32_i8")["acc"]
    rows = sim_vectors("requant_i32_i8")
    assert len(rows) == len(vals) == 485
    for (lo, exp, hi), acc in zip(rows, vals):
        acc = int(acc)
        assert lo == acc & 0xFFFF and hi == (1 << 16) | ((acc >> 16) & 0xFFFF)                    # select pin 0 for the low word, 1 for the high word
        assert exp == max(-128, min(127, (acc * 19 + 512) >> 10)) & 0xFF                          # ASSUMED convention, plain Python integers
    assert lay["mode"] == "two_step"


def test_two_step_testbench_text():
    tb = make_fab_testbench("requant_i32_i8")
    assert tb_vector_count(tb) == 485 and "NVEC = 485" in tb and "I_top[24:17]" in tb and "reg [27:0] vec_hi  [0:NVEC-1]" in tb
    assert tb.index("O_top = vec_in[i];") < tb.index("O_top = vec_hi[i];") < tb.index("repeat (3) @(posedge CLK);")       # low half, high half, then wait
    rows = re.findall(r"vec_in\[(\d+)\] = 28'h([0-9A-F]{7}); vec_exp\[\d+\] = 8'h([0-9A-F]{2}); vec_hi\[\d+\] = 28'h([0-9A-F]{7});", tb)
    assert [(int(lo, 16), int(e, 16), int(hi, 16)) for _, lo, e, hi in rows] == sim_vectors("requant_i32_i8")
    direct = make_fab_testbench("add8_sat")
    assert "vec_hi" not in direct and tb_vector_count(direct) == 500                              # the one-step testbench is unchanged


# ------------------------------------------------------------ timing
def test_timing_failure_is_read_from_the_real_nextpnr_lines(tmp_path):
    log = tmp_path / "x_npnr_log.txt"
    log.write_text(REAL_FAIL)
    assert chain.timing_failure(log) == {"clock": "clk", "fmax_mhz": 7.19, "target_mhz": 12.0}       # the final ERROR line, not the first estimate
    log.write_text(REAL_PASS)
    assert chain.timing_failure(log) is None                                                        # an early 'Info: ... FAIL' estimate is not the result
    assert chain.timing_failure(tmp_path / "missing.txt") is None


class Flow(FakeUbuntu):
    """The flow script as the real one behaves for this design: nextpnr's timing error -> exit code 1, the log is copied, no bitstream."""
    def __init__(self, log=REAL_FAIL, code=1):
        super().__init__()
        self.log, self.code = log, code

    def __call__(self, cmd, timeout_s, heartbeat_s, say):
        script = next(c for c in cmd if c.endswith(".sh"))
        if Path(script).name != "run_fab_flow.sh":
            return super().__call__(cmd, timeout_s, heartbeat_s, say)
        self.cmds.append(cmd)
        job = Path(script).parent
        op = cmd[cmd.index(script) + 1]
        (job / "exit_code.txt").write_text(f"{self.code}\n")
        (job / f"{op}_fab_npnr_log.txt").write_text(self.log)
        return result(output="ERROR | Command 'compile_design' execution failed with exit code 1")


@pytest.fixture
def job(proj):
    d = proj / "reports" / "fab_jobs" / "requant_i32_i8"
    d.mkdir(parents=True, exist_ok=True)
    (d / "run_fab_flow.sh").write_text(FLOW_SCRIPT)
    return d


def flow_stage(fake, **over):
    return chain.build_steps("requant_i32_i8", cfg(**over), exec_fn=fake, preflight={})["flow"]()


def test_flow_reports_timing_not_met_with_both_numbers(job):
    fake = Flow()
    status, detail = flow_stage(fake)[:2]
    assert status == "fail" and "timing not met" in detail and "at most 7.19 MHz for clock 'clk'" in detail and "target is 12 MHz" in detail
    assert "--freq-mhz" in detail and "for example 5" in detail and "not measured hardware" in detail and "no bitstream" in detail
    assert fake.cmds[-1][-1] == "requant_i32_i8"                                                    # no clock target given: the script gets the op only
    ev = cd.collect_evidence("requant_i32_i8", {"op": "requant_i32_i8", "verdict": "FAIL", "stages": [{"stage": "flow", "status": status, "detail": detail}]}, job.parent)
    assert cd.rule_candidates(ev) == ["timing_not_met"] and "12 MHz default" in cd.CAUSES["timing_not_met"]["seen"]


def test_another_flow_error_is_not_called_timing(job):
    status, detail = flow_stage(Flow(log="ERROR: something else\n"))[:2]
    assert status == "fail" and "timing" not in detail and "FABulous exited with code 1" in detail


def test_clock_target_is_passed_to_the_script_and_named_in_the_result(job):
    fake = Flow(log=REAL_PASS, code=0)
    (job / "requant_i32_i8_fab.bin").write_bytes(b"x" * 12024)
    status, detail = flow_stage(fake, freq_mhz=5.0)[:2]
    assert status == "pass" and "clock target 5 MHz (--freq-mhz)" in detail and fake.cmds[-1][-2:] == ["requant_i32_i8", "5"]
    status, detail = flow_stage(Flow(log=REAL_PASS, code=0))[:2]
    assert status == "pass" and "nextpnr's default clock target" in detail
    status, detail = flow_stage(Flow(), freq_mhz=7.5)[:2]
    assert status == "fail" and "timing not met" in detail


def test_freq_option_accepts_only_a_positive_number():
    import argparse
    assert chain.positive_mhz("5") == 5.0 and chain.positive_mhz("7.5") == 7.5
    for bad in ("0", "-5", "5; rm -rf ~", "fast", "", "5e1", "$(id)"):
        with pytest.raises(argparse.ArgumentTypeError):
            chain.positive_mhz(bad)


def test_summary_shows_timing_when_nextpnr_reported_it():
    base = {"op": "x", "verdict": "PASS", "stages": [], "facts": {"fmax": {"clk": {"fmax_mhz": 7.35, "target_mhz": 5.0, "ok": True}}}}
    text = "\n".join(fc.format_summary(base))
    assert "clock clk can run at most 7.35 MHz; target 5.0 MHz: met" in text and "NOT measured hardware" in text and "printed no fmax" not in text
    base["facts"]["fmax"]["clk"]["ok"] = False
    assert "target 5.0 MHz: NOT met" in "\n".join(fc.format_summary(base))
    none = {"op": "x", "verdict": "PASS", "stages": [], "facts": {"fmax": {}}}
    assert "printed no fmax" in "\n".join(fc.format_summary(none)) and "can run at most" not in "\n".join(fc.format_summary(none))


def test_main_records_the_clock_target_and_names_the_cause(proj, capsys):
    (proj / "hardware" / "operators").mkdir(parents=True, exist_ok=True)
    from x2c.agents.rtl_agent import REFERENCE_VERILOG
    (proj / "hardware" / "operators" / "requant_i32_i8.v").write_text(REFERENCE_VERILOG["requant_i32_i8"])
    rc = run_main(["--op", "requant_i32_i8"], Flow())
    out = capsys.readouterr().out
    assert rc == 1 and "wrapper check (requant_i32_i8_fab): PASS at function | 485 vectors" in out and "40 flip-flop bits" in out
    assert "timing not met" in out and "likely cause [timing_not_met]" in out and "CHAIN RESULT requant_i32_i8: FAIL" in out
    assert json.loads(chain.REPORT.read_text())["clock_target_mhz"] is None
    run_main(["--op", "requant_i32_i8", "--freq-mhz", "5"], Flow())
    assert json.loads(chain.REPORT.read_text())["clock_target_mhz"] == 5.0
    with pytest.raises(SystemExit):
        run_main(["--op", "requant_i32_i8", "--freq-mhz", "5;ls"], Flow())


# ------------------------------------------------------------ the real flow script with a fake FABulous
@needs_bash
def test_flow_script_passes_the_clock_target_to_nextpnr_only_when_given(tmp_path):
    home = tmp_path / "home"
    (home / "fabenv" / "bin").mkdir(parents=True)
    (home / "fabenv" / "bin" / "activate").write_text("")
    demo = home / "work" / "demo"
    (demo / "user_design").mkdir(parents=True)
    tools = home / "tools" / "oss-cad-suite" / "bin"
    tools.mkdir(parents=True)
    fab = tools / "FABulous"
    fab.write_text('#!/bin/bash\necho "$@" >> "$HOME/calls.txt"\n')
    fab.chmod(fab.stat().st_mode | stat.S_IEXEC)
    root = tmp_path / "proj"
    job = root / "reports" / "fab_jobs" / "add8_sat"
    job.mkdir(parents=True)
    (root / "src" / "x2c" / "verification").mkdir(parents=True)
    (root / "src" / "x2c" / "verification" / "fab_connect.py").write_text("print('fab_connect: ok')\n")
    (job / "run_fab_flow.sh").write_text(FLOW_SCRIPT, newline="\n")
    (job / "add8_sat_fab.v").write_text("module add8_sat_fab; endmodule\n")
    env = {**os.environ, "HOME": str(home)}
    subprocess.run(["bash", str(job / "run_fab_flow.sh"), "add8_sat"], capture_output=True, text=True, env=env)
    subprocess.run(["bash", str(job / "run_fab_flow.sh"), "add8_sat", "5"], capture_output=True, text=True, env=env)
    calls = [l for l in (home / "calls.txt").read_text().splitlines() if "compile_design" in l]
    assert len(calls) == 2 and "--nextpnr-extra-args" not in calls[0] and calls[0].rstrip().endswith("compile_design ./user_design/add8_sat_fab.v; exit")
    assert "compile_design ./user_design/add8_sat_fab.v --nextpnr-extra-args '--freq 5'; exit" in calls[1]
    r = subprocess.run(["bash", str(job / "run_fab_flow.sh"), "add8_sat", "5;id"], capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "must be a number" in r.stdout
    assert len([l for l in (home / "calls.txt").read_text().splitlines() if "compile_design" in l]) == 2          # nothing was run for the bad value
    assert subprocess.run(["bash", "-n"], input=FLOW_SCRIPT.encode(), capture_output=True).returncode == 0
