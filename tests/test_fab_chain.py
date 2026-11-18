"""Step 24: the one-command chain. Logic is tested with a FAKE Ubuntu (no WSL, no FABulous needed); the commands it WOULD run are checked as text.
What this cannot show: that the real flow passes. That is what `x2c chain --op add8_sat` on your machine shows."""
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import x2c.commands.chain as chain
from x2c.agents.rtl_agent import REFERENCE_VERILOG
from x2c.verification.fab_sim import tb_vector_count
from x2c.verification import fab_chain as fc

FIX = Path(__file__).parent / "fixtures"
PASS_LINE = "FAB_SIM RESULT: PASS vectors=500 numpy_mismatches=0 rtl_mismatches=0\n"


# ------------------------------------------------------------ paths and commands
def test_windows_path_becomes_a_mnt_path():
    assert fc.to_wsl_path(r"E:\work\ai-native-compiler-x2") == "/mnt/e/work/ai-native-compiler-x2"
    assert fc.to_wsl_path("e:/vm/x/a.sh") == "/mnt/e/vm/x/a.sh"
    assert fc.to_wsl_path("E:\\") == "/mnt/e/"
    assert fc.to_wsl_path("/home/u/x") == "/home/u/x"
    with pytest.raises(ValueError):
        fc.to_wsl_path("relative\\path")


def test_wsl_refuses_paths_it_could_split():
    assert fc.script_path("wsl", r"E:\work\ai-native-compiler-x2\reports\fab_jobs\add8_sat\run_fab_flow.sh") == \
        "/mnt/e/work/ai-native-compiler-x2/reports/fab_jobs/add8_sat/run_fab_flow.sh"
    with pytest.raises(ValueError, match="spaces"):
        fc.script_path("wsl", r"E:\my projects\ai-native-compiler-x2\run.sh")
    assert fc.script_path("native", "/home/u/my projects/run.sh") == "/home/u/my projects/run.sh"     # native uses an argument list: no splitting


def test_runner_choice():
    assert fc.resolve_runner("auto", "win32") == "wsl" and fc.resolve_runner("auto", "linux") == "native"
    assert fc.resolve_runner("native", "win32") == "native"


def test_commands_have_the_exact_shape_that_worked_by_hand():
    s = "/mnt/e/work/ai-native-compiler-x2/reports/fab_jobs/add8_sat/run_fab_flow.sh"
    assert fc.build_command("wsl", s, ["add8_sat"], "Ubuntu") == ["wsl", "-d", "Ubuntu", "--", "bash", s, "add8_sat"]
    assert fc.build_command("native", s, ["add8_sat"]) == ["bash", s, "add8_sat"]
    with pytest.raises(ValueError):
        fc.build_command("docker", s)


# ------------------------------------------------------------ running a command
def test_decode_handles_wsl_utf16_and_utf8():
    msg = "There is no distribution with the supplied name."
    assert fc.decode_output(msg.encode("utf-16-le")) == msg
    assert fc.decode_output(b"\xff\xfe" + msg.encode("utf-16-le")).startswith("There is no")
    assert fc.decode_output("caf\u00e9".encode("utf-8")) == "caf\u00e9"
    assert fc.decode_output(b"") == ""


def test_decode_handles_a_stream_that_mixes_wsl_and_ubuntu_text():
    mixed = "FOUND yosys\n".encode("utf-8") + "wsl: some warning\n".encode("utf-16-le") + "PREFLIGHT DONE\n".encode("utf-8")
    text = fc.decode_output(mixed)
    assert "FOUND yosys" in text and "wsl: some warning" in text and "PREFLIGHT DONE" in text
    assert fc.parse_preflight(text)["reached_end"]


def test_a_tool_that_asks_a_question_gets_end_of_input_instead_of_hanging():
    r = fc.run_command([sys.executable, "-c", "import sys; print(repr(sys.stdin.read()))"], timeout_s=20)
    assert r["output"].strip() == "''" and r["returncode"] == 0


def test_run_command_captures_output_and_exit_code():
    r = fc.run_command([sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)"], timeout_s=60)
    assert r["started"] and r["returncode"] == 3 and "out" in r["output"] and "err" in r["output"] and not r["timed_out"]


def test_run_command_reports_a_command_that_cannot_start():
    r = fc.run_command(["definitely-not-a-real-program-xyz"], timeout_s=5)
    assert r["started"] is False and r["returncode"] is None and "cannot start" in r["output"]


def test_run_command_kills_at_the_timeout():
    r = fc.run_command([sys.executable, "-c", "import time; time.sleep(60)"], timeout_s=1, heartbeat_s=0.2)
    assert r["timed_out"] is True and r["seconds"] < 30


def test_run_command_prints_a_heartbeat_while_it_waits():
    said = []
    fc.run_command([sys.executable, "-c", "import time; print('x', flush=True); time.sleep(1.5)"], timeout_s=30, heartbeat_s=0.3, say=said.append)
    assert len(said) >= 2 and "still running" in said[0] and "lines of tool output" in said[0]


# ------------------------------------------------------------ preflight
GOOD_PF = ("FOUND FABulous\nFOUND yosys\nFOUND nextpnr-generic\nFOUND iverilog\nFOUND demo project\n"
           "VERSION yosys Yosys 0.66+154\nVERSION nextpnr-generic nextpnr-0.10-82\nVERSION iverilog Icarus Verilog version 14.0\nPREFLIGHT DONE\n")


def test_preflight_parsing():
    i = fc.parse_preflight(GOOD_PF)
    assert i["ok"] and i["versions"]["yosys"] == "Yosys 0.66+154" and not i["missing"]
    bad = fc.parse_preflight(GOOD_PF.replace("FOUND iverilog", "MISSING iverilog"))
    assert not bad["ok"] and bad["missing"] == ["iverilog"]
    assert not fc.parse_preflight("FOUND FABulous\n")["ok"]                      # never reached the end: not ok
    assert not fc.parse_preflight("")["ok"]
    assert "VERSION" not in fc.parse_preflight("VERSION yosys \nPREFLIGHT DONE")["versions"]      # empty version is not recorded


def test_preflight_explanations_name_the_fix():
    assert "wsl -l -v" in fc.explain_preflight({"started": True, "output": "no distribution"}, fc.parse_preflight(""), "wsl", "Ubuntu")
    assert "iverilog" in fc.explain_preflight({"started": True, "output": ""}, fc.parse_preflight("MISSING iverilog\nPREFLIGHT DONE"), "wsl", "Ubuntu")
    assert "--runner native" in fc.explain_preflight({"started": False, "output": "cannot start 'wsl'"}, fc.parse_preflight(""), "wsl", "Ubuntu")


def test_preflight_script_is_lf_only_valid_bash_and_changes_nothing():
    assert "\r" not in fc.PREFLIGHT_SCRIPT
    for danger in ("rm ", "mv ", "pip ", "apt", "cp ", "> /"):
        assert danger not in fc.PREFLIGHT_SCRIPT.replace(">/dev/null", "")
    if shutil.which("bash") is None:
        pytest.skip("bash not available")
    r = subprocess.run(["bash", "-n"], input=fc.PREFLIGHT_SCRIPT.encode(), capture_output=True)     # stdin: WSL bash cannot open Windows temp paths
    assert r.returncode == 0, r.stderr.decode(errors="replace")


# ------------------------------------------------------------ stale results
def test_which_files_a_start_stage_deletes():
    assert "add8_sat_fab.bin" in fc.downstream_files("add8_sat", "rtl") and "sim_console.log" in fc.downstream_files("add8_sat", "rtl")
    assert "add8_sat_fab.bin" in fc.downstream_files("add8_sat", "flow")
    sim = fc.downstream_files("add8_sat", "sim")
    assert "add8_sat_fab.bin" not in sim and "sim_console.log" in sim and "flow_check.json" in sim
    assert fc.downstream_files("add8_sat", "check") == ["flow_check.json"]


def test_clean_job_removes_old_results_but_not_the_inputs(tmp_path):
    for n in ("add8_sat_fab.bin", "sim_console.log", "flow_check.json", "add8_sat_fab.v", "run_fab_flow.sh", "add8_sat_fab_tb.v"):
        (tmp_path / n).write_text("x")
    gone = fc.clean_job(tmp_path, "add8_sat", "rtl")
    assert set(gone) == {"add8_sat_fab.bin", "sim_console.log", "flow_check.json"}
    assert all((tmp_path / n).exists() for n in ("add8_sat_fab.v", "run_fab_flow.sh", "add8_sat_fab_tb.v"))
    assert fc.clean_job(tmp_path / "does_not_exist", "add8_sat", "rtl") == []


def test_resuming_from_sim_keeps_the_bitstream(tmp_path):
    for n in ("add8_sat_fab.bin", "sim_console.log"):
        (tmp_path / n).write_text("x")
    fc.clean_job(tmp_path, "add8_sat", "sim")
    assert (tmp_path / "add8_sat_fab.bin").exists() and not (tmp_path / "sim_console.log").exists()


# ------------------------------------------------------------ orchestration
def ok(detail="fine", extra=None):
    return lambda: ("pass", detail, extra) if extra else ("pass", detail)


def steps(**over):
    base = {s: ok(f"{s} done") for s in fc.STAGES}
    base.update(over)
    return base


def test_all_stages_pass_gives_pass_in_order():
    said, calls = [], []
    st = {s: (lambda s=s: (calls.append(s), ("pass", f"{s} done"))[1]) for s in fc.STAGES}
    r = fc.run_chain("add8_sat", st, say=said.append)
    assert r["verdict"] == "PASS" and calls == list(fc.STAGES) and [x["stage"] for x in r["stages"]] == list(fc.STAGES)
    assert len(said) == 6 and "[1/6] rtl" in said[0] and "[6/6] check" in said[5]


def test_a_failed_stage_stops_the_chain_and_later_stages_are_not_run():
    r = fc.run_chain("add8_sat", steps(prep=lambda: ("fail", "wrapper check failed")), say=lambda m: None)
    s = {x["stage"]: x["status"] for x in r["stages"]}
    assert r["verdict"] == "FAIL" and s["rtl"] == "pass" and s["prep"] == "fail"
    assert all(s[k] == "not_run" for k in ("env", "flow", "sim", "check"))
    assert "prep did not pass" in r["stages"][2]["detail"]


def test_failed_flow_skips_sim_but_check_still_explains():
    calls = []
    r = fc.run_chain("op", steps(flow=lambda: ("fail", "FABulous exited with code 1"), sim=lambda: calls.append("sim"),
                                 check=lambda: (calls.append("check"), ("fail", "FLOW STATUS SUSPECT"))[1]), say=lambda m: None)
    s = {x["stage"]: x["status"] for x in r["stages"]}
    assert r["verdict"] == "FAIL" and s["sim"] == "not_run" and s["check"] == "fail" and calls == ["check"]


def test_blocked_environment_means_not_run_not_failed_and_check_is_not_run():
    calls = []
    r = fc.run_chain("op", steps(env=lambda: ("blocked", "Ubuntu is missing: iverilog"), check=lambda: calls.append("check")), say=lambda m: None)
    assert r["verdict"] == "NOT RUN" and not calls
    assert {x["stage"]: x["status"] for x in r["stages"]}["flow"] == "not_run"


def test_a_crashing_stage_is_a_failure_never_a_pass():
    def boom():
        raise KeyError("x")
    r = fc.run_chain("op", steps(sim=boom), say=lambda m: None)
    s = {x["stage"]: x for x in r["stages"]}
    assert r["verdict"] == "FAIL" and s["sim"]["status"] == "fail" and "KeyError" in s["sim"]["detail"] and s["check"]["status"] == "pass"


def test_unverified_check_is_not_verified_not_pass():
    r = fc.run_chain("op", steps(check=lambda: ("unverified", "FLOW STATUS SUSPECT")), say=lambda m: None)
    assert r["verdict"] == "NOT VERIFIED"


def test_resume_from_sim_says_what_was_not_rerun():
    calls = []
    st = {s: (lambda s=s: (calls.append(s), ("pass", "x"))[1]) for s in fc.STAGES}
    r = fc.run_chain("op", st, from_stage="sim", say=lambda m: None)
    assert calls == ["sim", "check"]
    assert r["verdict"] == "PASS (partial: rtl, prep, env, flow not re-run)" and r["stages"][0]["status"] == "skipped"


def test_resume_from_sim_failing_still_runs_check():
    calls = []
    r = fc.run_chain("op", steps(sim=lambda: ("fail", "mismatch"), check=lambda: (calls.append("check"), ("fail", "x"))[1]), from_stage="sim", say=lambda m: None)
    assert r["verdict"] == "FAIL" and calls == ["check"]


def test_unknown_stage_is_rejected():
    with pytest.raises(ValueError):
        fc.run_chain("op", steps(), from_stage="deploy")


def test_summary_pass_and_fail_wording():
    facts = {"bitstream_sha256": "ab" * 32, "set_bits": 653, "logic_cells": "25/672", "fasm_features": 875, "fmax": {}, "not_checked": [], "findings": ["[warning] w: x"]}
    r = fc.run_chain("add8_sat", steps(check=ok("c", facts)), say=lambda m: None)
    text = "\n".join(fc.format_summary(r))
    assert "CHAIN RESULT add8_sat: PASS" in text and "set bits 653" in text and "25/672" in text and "finding: [warning] w: x" in text
    assert "timing (nextpnr printed no fmax" in text and "silicon" in text and "shows:" in text
    f = fc.run_chain("add8_sat", steps(sim=lambda: ("fail", "m")), say=lambda m: None)
    assert "shows:" not in "\n".join(fc.format_summary(f)) and "FAIL" in fc.format_summary(f)[0]


def test_nothing_is_said_about_timing_when_the_flow_never_ran():
    r = fc.run_chain("op", steps(env=lambda: ("blocked", "Ubuntu is missing: iverilog")), say=lambda m: None)
    text = "\n".join(fc.format_summary(r))
    assert "NOT RUN" in text and "nextpnr printed no fmax" not in text and "silicon" in text


def test_timing_line_is_dropped_only_when_fmax_exists():
    facts = {"fmax": {"clk": {"fmax_mhz": 30.6, "ok": True}}}
    r = fc.run_chain("op", steps(check=ok("c", facts)), say=lambda m: None)
    assert "no fmax" not in "\n".join(fc.format_summary(r))


def test_utilisation_text():
    assert fc.utilisation_text({"used": 25, "total": 672}) == "25/672" and fc.utilisation_text(None) == "None"


# ------------------------------------------------------------ the real stage functions, with a fake Ubuntu
def cfg(**o):
    d = dict(runner="native", distro="Ubuntu", llm="mock", model=None, regen_rtl=False, timeout_s=60, heartbeat_s=30, from_stage="rtl")
    d.update(o)
    return SimpleNamespace(**d)


def result(**o):
    d = {"started": True, "returncode": 0, "output": "", "seconds": 2.0, "timed_out": False}
    d.update(o)
    return d


class FakeUbuntu:
    """Stands in for `wsl ... bash <script> <op>`: writes the files the real scripts write. `mode` picks a failure to simulate."""
    def __init__(self, mode="ok", preflight=GOOD_PF):
        self.mode, self.preflight, self.cmds = mode, preflight, []

    def __call__(self, cmd, timeout_s, heartbeat_s, say):
        self.cmds.append(cmd)
        script = next(c for c in cmd if c.endswith(".sh"))
        name, job = Path(script).name, Path(script).parent
        op = cmd[-1]
        if name == "_preflight.sh":
            return result(output=self.preflight)
        if name == "run_fab_flow.sh":
            if self.mode == "not_started":
                return result(started=False, returncode=None, output="cannot start 'wsl'")
            if self.mode == "timeout":
                return result(timed_out=True, returncode=None)
            if self.mode == "no_exit_file":
                return result(output="line1\nline2")
            code = {"exit1": 1, "exit97": 97}.get(self.mode, 0)
            (job / "exit_code.txt").write_text(f"{code}\n")
            if code == 0 and self.mode != "no_bin":
                shutil.copy(FIX / "real_flow_run/nextpnr_log.txt", job / f"{op}_fab_npnr_log.txt")
                shutil.copy(FIX / "real_flow_run/design.fasm", job / f"{op}_fab.fasm")
                shutil.copy(FIX / "fabulous_demo/design.bin", job / f"{op}_fab.bin")
                shutil.copy(FIX / "fabulous_demo/fabric.csv", job / "fabric.csv")
                (job / "flow_console.log").write_text("INFO | Compile flow completed successfully.\n")
            return result(returncode=1 if code == 97 else 0, output="flow output")
        if name == "run_fab_sim.sh":
            n = tb_vector_count((job / f"{op}_fab_tb.v").read_text())          # the real testbench's table size: 500 for add8_sat, 256 for relu8
            text = {"sim_mismatch": f"FAB_SIM RESULT: FAIL vectors={n} numpy_mismatches=3 rtl_mismatches=3\n",
                    "sim_empty": "INFO | Compile flow completed successfully.\n"}.get(
                        self.mode, f"FAB_SIM RESULT: PASS vectors={n} numpy_mismatches=0 rtl_mismatches=0\n")
            if self.mode != "no_sim_log":
                (job / "sim_console.log").write_text(text)
            return result()
        raise AssertionError(f"unexpected command {cmd}")


@pytest.fixture
def proj(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "hardware" / "operators").mkdir(parents=True)
    for op in ("add8_sat", "relu8"):
        (tmp_path / "hardware" / "operators" / f"{op}.v").write_text(REFERENCE_VERILOG[op])
    return tmp_path


def run_main(args, fake):
    return chain.main(["--runner", "native", *args], exec_fn=fake)


@pytest.mark.parametrize("op", ["add8_sat", "relu8"])
def test_whole_chain_with_a_fake_ubuntu_passes(proj, capsys, op):
    fake = FakeUbuntu()
    assert run_main(["--op", op], fake) == 0
    out = capsys.readouterr().out
    assert f"CHAIN RESULT {op}: PASS" in out and "[6/6] check" in out
    rep = json.loads((proj / "reports" / "fab_chain_report.json").read_text())
    assert rep["results"][0]["verdict"] == "PASS" and rep["tool_versions"]["yosys"] == "Yosys 0.66+154"
    assert "NOT silicon" in rep["evidence_level"] and (proj / "reports" / "fab_jobs" / op / "flow_check.json").exists()
    scripts = [Path(next(c for c in cmd if c.endswith(".sh"))).name for cmd in fake.cmds]
    assert scripts == ["_preflight.sh", "run_fab_flow.sh", "run_fab_sim.sh"]
    assert b"\r" not in (proj / "reports" / "fab_jobs" / "_preflight.sh").read_bytes()


def test_stale_results_cannot_pass_a_failed_flow(proj, capsys):
    job = proj / "reports" / "fab_jobs" / "add8_sat"
    job.mkdir(parents=True)
    (job / "add8_sat_fab.bin").write_bytes(b"STALE")
    (job / "sim_console.log").write_text(PASS_LINE)
    (job / "exit_code.txt").write_text("0\n")
    assert run_main(["--op", "add8_sat"], FakeUbuntu("no_bin")) == 1
    out = capsys.readouterr().out
    assert "cleared old results" in out and "FABulous exited 0 but wrote no bitstream" in out and "CHAIN RESULT add8_sat: FAIL" in out
    assert not (job / "add8_sat_fab.bin").exists() and not (job / "sim_console.log").exists()


@pytest.mark.parametrize("mode,words", [
    ("exit1", "exited with code 1"), ("exit97", "could not connect the pins"), ("no_exit_file", "did not finish"),
    ("timeout", "timed out"), ("sim_mismatch", "numpy mismatch"), ("no_sim_log", "no sim_console.log")])
def test_each_failure_is_named_and_the_verdict_is_fail(proj, capsys, mode, words):
    assert run_main(["--op", "add8_sat"], FakeUbuntu(mode)) == 1
    out = capsys.readouterr().out
    assert words in out and "CHAIN RESULT add8_sat: FAIL" in out and "PASS ===" not in out


def test_a_simulation_that_compared_nothing_is_not_verified(proj, capsys):
    assert run_main(["--op", "add8_sat"], FakeUbuntu("sim_empty")) == 1
    out = capsys.readouterr().out
    assert "NOT VERIFIED" in out and "sim   UNVERIFIED" in out and "sim   PASS" not in out


def test_missing_tools_in_ubuntu_block_before_the_flow_starts(proj, capsys):
    fake = FakeUbuntu(preflight=GOOD_PF.replace("FOUND iverilog", "MISSING iverilog"))
    assert run_main(["--op", "add8_sat"], fake) == 1
    out = capsys.readouterr().out
    assert "BLOCKED" in out and "iverilog" in out and "CHAIN RESULT add8_sat: NOT RUN" in out
    assert len(fake.cmds) == 1                                           # only the preflight ran; the flow was never started


def test_wsl_cannot_start_is_blocked_with_the_fix(proj, capsys):
    assert run_main(["--op", "add8_sat"], FakeUbuntu("ok", preflight="")) == 1
    assert "wsl -l -v" in capsys.readouterr().out


def test_two_operators_share_one_preflight(proj, capsys):
    fake = FakeUbuntu()
    assert run_main(["--op", "add8_sat,relu8"], fake) == 0
    out = capsys.readouterr().out
    assert "ALL OPERATORS: add8_sat PASS | relu8 PASS" in out
    assert sum(Path(next(c for c in cmd if c.endswith(".sh"))).name == "_preflight.sh" for cmd in fake.cmds) == 1


WIDE2 = {"top": "wide2", "inputs": {"a": 20, "b": 20}, "output": ("y", 8), "text": "two wide inputs", "golden": lambda v: v["a"]}     # fits no layout


def test_all_skips_what_does_not_fit_the_pins(proj, capsys, monkeypatch):
    from x2c.verification.rtl_check import OPS
    monkeypatch.setitem(OPS, "wide2", WIDE2)
    run_main(["--op", "all"], FakeUbuntu())
    out = capsys.readouterr().out
    assert "skipping wide2" in out and "CHAIN RESULT relu8: PASS" in out and "CHAIN RESULT add8_sat: PASS" in out
    assert "FAB CHAIN  op=requant_i32_i8" in out and "skipping requant" not in out                     # loaded in two steps now, so it is run


def test_asking_for_an_operator_that_cannot_be_wrapped_is_refused(proj):
    from x2c.verification.rtl_check import OPS
    OPS["wide2"] = WIDE2
    try:
        with pytest.raises(SystemExit) as e:
            run_main(["--op", "wide2"], FakeUbuntu())
    finally:
        del OPS["wide2"]
    assert "cannot wrap wide2" in str(e.value)
    with pytest.raises(SystemExit) as e:
        run_main(["--op", "nope"], FakeUbuntu())
    assert "unknown op" in str(e.value)


def test_a_bad_existing_verilog_file_is_not_trusted(proj, capsys):
    (proj / "hardware" / "operators" / "relu8.v").write_text("module relu8(input signed [7:0] x, output signed [7:0] y); assign y = x; endmodule\n")
    assert run_main(["--op", "relu8"], FakeUbuntu()) == 0              # re-check failed -> A2 (mock) wrote the reference again
    assert "existing file failed its re-check" in capsys.readouterr().out


def test_real_llm_without_a_key_is_blocked_not_run(proj, capsys, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    (proj / "hardware" / "operators" / "relu8.v").unlink()
    assert run_main(["--op", "relu8", "--llm", "groq"], FakeUbuntu()) == 1
    out = capsys.readouterr().out
    assert "GROQ_API_KEY is not set" in out and "NOT RUN" in out


def test_resume_from_sim_keeps_the_bitstream_and_reruns_only_sim_and_check(proj, capsys):
    assert run_main(["--op", "add8_sat"], FakeUbuntu()) == 0
    capsys.readouterr()
    fake = FakeUbuntu("sim_mismatch")
    assert run_main(["--op", "add8_sat", "--from-stage", "sim"], fake) == 1
    out = capsys.readouterr().out
    assert "numpy mismatch" in out and (proj / "reports" / "fab_jobs" / "add8_sat" / "add8_sat_fab.bin").exists()
    assert [Path(next(c for c in cmd if c.endswith(".sh"))).name for cmd in fake.cmds] == ["run_fab_sim.sh"]


def _wsl_path_is_plain():
    """True when the current folder gives a path wsl.exe can take (no spaces). pytest's temp folder on Windows may be under 'C:\\Users\\First Last'."""
    try:
        fc.script_path("wsl", str((chain.JOBS / "x").resolve()))
        return True
    except ValueError:
        return False


def test_plan_runs_nothing_and_shows_the_commands(proj, capsys):
    fake = FakeUbuntu()
    assert chain.main(["--op", "add8_sat", "--runner", "native", "--plan"], exec_fn=fake) == 0
    out = capsys.readouterr().out
    assert fake.cmds == [] and "run_fab_flow.sh add8_sat" in out and "run_fab_sim.sh add8_sat" in out and "deleted" in out


def test_plan_with_the_wsl_runner_works_without_wsl_installed(proj, capsys):
    assert chain.main(["--op", "add8_sat", "--runner", "wsl", "--plan"]) == 0           # default exec_fn: plan must not need wsl.exe
    out = capsys.readouterr().out
    assert ("wsl -d Ubuntu -- bash" in out) if _wsl_path_is_plain() else ("CANNOT RUN" in out and "spaces" in out)


def test_plan_says_cannot_run_instead_of_crashing_on_a_path_with_spaces(proj, monkeypatch):
    def refuse(runner, p):
        raise ValueError("'/mnt/c/My Folder/run.sh' has spaces or special characters that wsl.exe may split")
    monkeypatch.setattr(chain.fc, "script_path", refuse)
    text = "\n".join(chain.plan_text("add8_sat", cfg(runner="wsl")))
    assert "CANNOT RUN" in text and "spaces" in text and "run  rtl" in text


def test_script_paths_are_made_for_the_runner(proj):
    cmd = fc.build_command("wsl", fc.script_path("wsl", "/mnt/e/vm/a/run_fab_flow.sh"), ["relu8"], "Ubuntu")
    assert cmd[:5] == ["wsl", "-d", "Ubuntu", "--", "bash"] and cmd[-1] == "relu8"


# ------------------------------------------------------------ readable summary
LONG = {"yosys": "Yosys 0.66+154 (git sha1 23aadd92a, Release, Clang /usr/bin/clang++ 18.1.8)",
        "nextpnr-generic": '"nextpnr-generic" -- Next Generation Place and Route (Version nextpnr-0.10-82-g2b560ad0)',
        "iverilog": "Icarus Verilog version 14.0 (devel) (s20260301-220-g78750c51d-dirty)"}


def test_versions_are_shortened_for_the_console_and_kept_whole_in_the_report():
    assert fc.short_versions(LONG) == {"yosys": "0.66+154", "nextpnr-generic": "0.10-82", "iverilog": "14.0"}
    assert fc.short_versions({"nextpnr-generic": "nextpnr-0.10-82"})["nextpnr-generic"] == "0.10-82"          # plainer wording also works
    assert fc.short_versions({"other": "x" * 80})["other"] == "x" * 30 and fc.short_versions(None) == {}


def test_repeated_warnings_are_shown_once_with_a_count():
    w = [f"[warning] yosys_warning: Wire top_wrapper.\\OutPass4_frame_config_mux_I{i} [{j}] is used but has no driver." for i in range(4) for j in range(2)]
    g = fc.group_findings(w + ["[error] fabric_sim_mismatch: 3 numpy mismatch(es)"])
    assert len(g) == 2 and g[0].startswith("x8 ") and "first of 8 similar" in g[0] and "mismatch" in g[1]
    assert fc.group_findings(["[warning] a: b"]) == ["[warning] a: b"] and fc.group_findings(None) == []
    many = [f"[warning] kind{chr(97 + i)}: m" for i in range(8)]
    assert len(fc.group_findings(many)) == 6 and "3 more kinds" in fc.group_findings(many)[-1]


def test_summary_uses_grouped_findings_and_the_env_line_is_short(proj, capsys):
    real_pf = "FOUND FABulous\nFOUND demo project\n" + "".join(f"VERSION {k} {v}\n" for k, v in LONG.items()) + "PREFLIGHT DONE\n"
    fake = FakeUbuntu(preflight=real_pf)
    assert run_main(["--op", "add8_sat"], fake) == 0
    out = capsys.readouterr().out
    assert "(Yosys 0.66+154, nextpnr 0.10-82, Icarus 14.0)" in out and "git sha1" not in out
    rep = json.loads((proj / "reports" / "fab_chain_report.json").read_text())
    assert "git sha1" in rep["tool_versions"]["yosys"]                          # the report keeps the full text
