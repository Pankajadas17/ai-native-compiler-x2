"""The simulated bitstream must be the job folder's (the audited one). Found on a real run: after an Ubuntu crash the copy inside Ubuntu was
0 bytes, the simulation loaded an empty bitstream, and every fabric output was unknown (500/500 mismatches) although the design was right.
The real run_fab_sim.sh is run by real bash against a FAKE FABulous that records the bitstream it was given."""
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

import x2c.commands.chain as chain
from x2c.commands.demo_prep import SIM_SCRIPT
from tests.test_fab_chain import PASS_LINE, cfg, proj, result  # noqa: F401  (proj is a fixture)

needs_bash = pytest.mark.skipif(sys.platform == "win32" or not (Path("/bin/bash").exists() or Path("/usr/bin/bash").exists()),
                                reason="runs real bash (on Windows `bash` is WSL's launcher and cannot open Windows temp paths)")
GOOD = bytes(range(256)) * 4


def setup(tmp_path, ubuntu_copy, job_copy=GOOD):
    """A fake home with fabenv, a demo project and a FABulous that prints the size of the bitstream it finds, then a PASS line."""
    home, job = tmp_path / "home", tmp_path / "job"
    (home / "fabenv" / "bin").mkdir(parents=True)
    (home / "fabenv" / "bin" / "activate").write_text("")
    demo = home / "work" / "demo"
    (demo / "user_design").mkdir(parents=True)
    (demo / "Test").mkdir()
    tools = home / "tools" / "oss-cad-suite" / "bin"
    tools.mkdir(parents=True)
    fab = tools / "FABulous"
    fab.write_text('#!/bin/bash\necho "SIMULATED_BYTES $(stat -c %s "$HOME/work/demo/user_design/add8_sat_fab.bin")"\n'
                   'echo "FAB_SIM RESULT: PASS vectors=500 numpy_mismatches=0 rtl_mismatches=0"\n')
    fab.chmod(fab.stat().st_mode | stat.S_IEXEC)
    job.mkdir()
    (job / "run_fab_sim.sh").write_text(SIM_SCRIPT, newline="\n")
    (job / "add8_sat_fab_tb.v").write_text("module tb; endmodule\n")
    if job_copy is not None:
        (job / "add8_sat_fab.bin").write_bytes(job_copy)
    if ubuntu_copy is not None:
        (demo / "user_design" / "add8_sat_fab.bin").write_bytes(ubuntu_copy)
    r = subprocess.run(["bash", str(job / "run_fab_sim.sh"), "add8_sat"], capture_output=True, text=True, env={**os.environ, "HOME": str(home)})
    log = (job / "sim_console.log").read_text() if (job / "sim_console.log").exists() else ""
    return r, log, demo / "user_design" / "add8_sat_fab.bin"


@needs_bash
def test_an_empty_copy_in_ubuntu_is_replaced_before_simulating(tmp_path):
    r, log, ubuntu_bin = setup(tmp_path, b"")                                    # what the crash left on your machine
    assert f"SIMULATED_BYTES {len(GOOD)}" in log and ubuntu_bin.read_bytes() == GOOD
    note = f"FAB_SIM NOTE: the bitstream copy in Ubuntu (0 bytes) differed from the job folder's ({len(GOOD)} bytes); replaced it"
    assert note in r.stdout and note in log and "FAB_SIM RESULT: PASS" in log


@needs_bash
def test_a_different_or_missing_copy_is_replaced_too(tmp_path):
    r, log, ubuntu_bin = setup(tmp_path / "a", GOOD[:-1] + b"\x00")              # same size, one byte different
    assert ubuntu_bin.read_bytes() == GOOD and f"({len(GOOD)} bytes) differed" in r.stdout
    r, log, ubuntu_bin = setup(tmp_path / "b", None)
    assert ubuntu_bin.read_bytes() == GOOD and "(missing bytes) differed" in r.stdout and f"SIMULATED_BYTES {len(GOOD)}" in log


@needs_bash
def test_an_identical_copy_is_left_alone_and_nothing_is_said(tmp_path):
    r, log, ubuntu_bin = setup(tmp_path, GOOD)
    assert "FAB_SIM NOTE" not in r.stdout + log and f"SIMULATED_BYTES {len(GOOD)}" in log and r.returncode == 0


@needs_bash
def test_no_usable_bitstream_anywhere_stops_before_fabulous(tmp_path):
    for i, (ubuntu, jobc) in enumerate([(b"", None), (b"", b""), (None, None)]):
        r, log, _ = setup(tmp_path / str(i), ubuntu, jobc)
        assert r.returncode == 1 and "no bitstream (missing or empty)" in r.stdout and log == ""


def test_script_text_has_the_guard():
    assert "\r" not in SIM_SCRIPT and 'cmp -s "$HERE/${OP}_fab.bin" "$DEMO/user_design/${OP}_fab.bin"' in SIM_SCRIPT
    assert SIM_SCRIPT.index("cmp -s") < SIM_SCRIPT.index("run_simulation vcd") and '[ ! -s "$DEMO/user_design/${OP}_fab.bin" ]' in SIM_SCRIPT


# ------------------------------------------------------------ the chain's sim stage
class Sim:
    def __init__(self, output=""):
        self.output, self.cmds = output, []

    def __call__(self, cmd, timeout_s, heartbeat_s, say):
        self.cmds.append(cmd)
        (chain.JOBS / "add8_sat" / "sim_console.log").write_text(PASS_LINE)
        return result(output=self.output)


@pytest.fixture
def job(proj):
    d = proj / "reports" / "fab_jobs" / "add8_sat"
    d.mkdir(parents=True, exist_ok=True)
    (d / "add8_sat_fab.bin").write_bytes(GOOD)
    return d


def sim_stage(fake):
    return chain.build_steps("add8_sat", cfg(), exec_fn=fake, preflight={})["sim"]()


def test_chain_refuses_an_empty_bitstream_in_the_job_folder(job):
    (job / "add8_sat_fab.bin").write_bytes(b"")
    fake = Sim()
    res = sim_stage(fake)
    assert res[0] == "fail" and "empty (0 bytes)" in res[1] and "--from-stage flow" in res[1] and fake.cmds == []


def test_chain_reports_that_the_ubuntu_copy_was_replaced(job):
    note = "FAB_SIM NOTE: the bitstream copy in Ubuntu (0 bytes) differed from the job folder's (1024 bytes); replaced it with the job folder's before simulating"
    res = sim_stage(Sim(output="INFO | x\n" + note + "\nmore\n"))
    assert res[0] == "pass" and "| NOTE: the bitstream copy in Ubuntu (0 bytes) differed" in res[1]
    assert "NOTE" not in sim_stage(Sim(output="INFO | nothing special\n"))[1]


@needs_bash
def test_an_empty_job_copy_never_overwrites_a_good_copy_in_ubuntu(tmp_path):
    r, log, ubuntu_bin = setup(tmp_path, GOOD, b"")
    assert ubuntu_bin.read_bytes() == GOOD and "FAB_SIM NOTE" not in r.stdout and f"SIMULATED_BYTES {len(GOOD)}" in log
