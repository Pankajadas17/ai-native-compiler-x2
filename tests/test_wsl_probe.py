"""WSL probe v2: the real probe.sh is run by real bash against FAKE tools (an `iverilog` that works / fails / kills the script mid-compile, a
`python3` that kills the script mid-memory-test, like a crashed Ubuntu), and the summary logic is checked on the logs they leave. Windows free
memory is checked with fake readings, and the real reader is called once (on Windows that calls GlobalMemoryStatusEx).
The real Icarus/Ubuntu run is yours."""
import ctypes
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

import x2c.commands.doctor_wsl as probe
from x2c.commands.demo_prep import SIM_SCRIPT
from x2c.verification import wsl_probe as wp

needs_bash = pytest.mark.skipif(sys.platform == "win32" or not (Path("/bin/bash").exists() or Path("/usr/bin/bash").exists()),
                                reason="runs real bash (on Windows `bash` is WSL's launcher and cannot open Windows temp paths)")
REAL_TOP = wp.top_memory_users          # the real function, kept for the one test that calls it
REAL_DISKS = wp.read_disk_free          # same: the disk reader is tested for real once


@pytest.fixture(autouse=True)
def no_real_process_listing(monkeypatch):
    """The runner must not list this machine's programs or disks during the unit tests (it would be a different answer every time)."""
    monkeypatch.setattr(wp, "top_memory_users", lambda *a, **k: [])
    monkeypatch.setattr(wp, "read_disk_free", lambda *a, **k: {})


@pytest.fixture(autouse=True)
def waited(monkeypatch):
    """No test really waits between tries; the waits the runner asked for are recorded here (a test can ask for this fixture and read them)."""
    asked = []
    monkeypatch.setattr(probe, "wait", asked.append)
    return asked


# ------------------------------------------------------------ the script text
def test_script_is_lf_only_and_valid_bash():
    assert "\r" not in wp.PROBE_SCRIPT
    if sys.platform != "win32":
        r = subprocess.run(["bash", "-n"], input=wp.PROBE_SCRIPT.encode(), capture_output=True)
        assert r.returncode == 0, r.stderr.decode(errors="replace")


def test_script_order_logging_place_and_what_it_touches():
    s = wp.PROBE_SCRIPT
    assert 'LOG="$HERE/probe.log"' in s and "rm -f build/probe_A.vvp build/probe_B.vvp" in s
    for danger in ("rm -rf", "pip ", "apt", "mv ", "> /dev", "sudo"):
        assert danger not in s
    assert s.index("PHASE R_start") < s.index("PHASE A_start") < s.index("PHASE B_start") < s.index("PHASE C_start") < s.index('say "DONE"')
    assert "ulimit -v 2500000" in s and "alive()" in s and ">/dev/null 2>&1 &" in s                                 # sampler stops by itself, holds no pipes
    assert "ps -C iverilog,ivl,ivlpp,vvp,python3,FABulous,task" in s and "ps -eo rss=,comm= --sort=-rss" in s       # the programs that really use the memory
    assert "${RAMP_STEPS:-16}" in s and "${RAMP_STEP_MB:-128}" in s                                                 # real runs: 16 x 128 MB = 2 GB
    assert 'bash "$HERE/phaseC/run_fab_sim.sh"' in s and '"$JOB/run_fab_sim.sh"' in s                                # phase C runs a COPY: results stay out of the job folder


# ------------------------------------------------------------ real bash with fake tools
FAKE_IVERILOG = """#!/usr/bin/env bash
sleep 0.7
case "$FAKE_MODE" in
  fail)  echo "iverilog: error: out of memory" >&2; exit 3;;
  die_b) n=$(cat "$FAKE_COUNT" 2>/dev/null || echo 0); echo $((n+1)) > "$FAKE_COUNT"
         if [ "$n" -ge 1 ]; then kill -9 "$(sed -n 's/.*START pid=\\([0-9]*\\).*/\\1/p' "$PROBE_LOG" | head -1)"; fi;;
  die_a) kill -9 "$(sed -n 's/.*START pid=\\([0-9]*\\).*/\\1/p' "$PROBE_LOG" | head -1)";;
esac
exit 0
"""

# stands in for python3: in die_ramp mode it logs two steps like the real memory test and then kills the probe script
FAKE_PYTHON = """#!/usr/bin/env bash
if [ "$FAKE_MODE" = "die_ramp" ]; then
  for n in 4 8; do echo "$(date +%H:%M:%S) RAMP $n MB ok" >> "$2"; sleep 0.3; done
  kill -9 "$(sed -n 's/.*START pid=\\([0-9]*\\).*/\\1/p' "$PROBE_LOG" | head -1)"
fi
exec "$REAL_PYTHON" "$@"
"""

# stands in for FABulous, which the REAL run_fab_sim.sh text (taken from run_fab_prep) calls
FAKE_FABULOUS = """#!/usr/bin/env bash
case "$FAKE_MODE" in
  die_c)     kill -9 "$(sed -n 's/.*START pid=\\([0-9]*\\).*/\\1/p' "$PROBE_LOG" | head -1)";;
  no_result) echo "FABulous: finished, nothing useful printed";;
  *)         echo "FAB_SIM RESULT PASS vectors=3 mismatches=0";;
esac
exit 0
"""


def _exe(path, text):
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def make_world(tmp_path, bitstream=True):
    home, demo, proj = tmp_path / "home", tmp_path / "demo", tmp_path / "proj"
    fake_bin = tmp_path / "fakebin"
    fake_bin.mkdir()
    (home / "fabenv" / "bin").mkdir(parents=True)
    (home / "fabenv" / "bin" / "activate").write_text(f'export PATH="{fake_bin}:$PATH"\n')      # like a venv: puts its programs first on PATH
    _exe(fake_bin / "iverilog", FAKE_IVERILOG)
    _exe(fake_bin / "python3", FAKE_PYTHON)
    _exe(fake_bin / "FABulous", FAKE_FABULOUS)
    (home / "tools" / "oss-cad-suite" / "bin").mkdir(parents=True)
    for d in ("Test/build", "Tile/x", "Fabric", "user_design"):
        (demo / d).mkdir(parents=True)
    (demo / "Tile" / "x" / "t.v").write_text("module t; endmodule")
    (demo / "Fabric" / "f.v").write_text("module f; endmodule")
    if bitstream:
        (demo / "user_design" / "add8_sat_fab.bin").write_bytes(b"\x01\x02")
    out = proj / "reports" / "wsl_probe"
    out.mkdir(parents=True)
    job = proj / "reports" / "fab_jobs" / "add8_sat"
    job.mkdir(parents=True)
    (job / "add8_sat_fab_tb.v").write_text("module add8_sat_fab_tb; endmodule")
    (job / "add8_sat_fab.v").write_text("module add8_sat_fab; endmodule")
    (job / "run_fab_sim.sh").write_text(SIM_SCRIPT, newline="\n")                                  # the real script text, not a lookalike
    (out / "probe.sh").write_text(wp.PROBE_SCRIPT, newline="\n")
    return home, demo, out


def run_probe(tmp_path, mode, bitstream=True):
    home, demo, out = make_world(tmp_path, bitstream)
    env = dict(os.environ, HOME=str(home), DEMO=str(demo), FAKE_MODE=mode, FAKE_COUNT=str(tmp_path / "count"), PROBE_LOG=str(out / "probe.log"),
               REAL_PYTHON=sys.executable, RAMP_STEPS="3", RAMP_STEP_MB="4")                    # 3 x 4 MB: the tests must not eat gigabytes
    r = subprocess.run(["bash", str(out / "probe.sh"), "add8_sat"], env=env, capture_output=True, text=True, timeout=60)
    return r, (out / "probe.log").read_text(), demo


@needs_bash
def test_a_healthy_run_logs_all_four_phases_and_done(tmp_path):
    r, log, demo = run_probe(tmp_path, "ok")
    s = wp.summarize_log(log)
    assert r.returncode == 0 and s["done"] and all(s["phases"][p]["exit"] == 0 for p in "RABC")
    assert s["phases"]["C"]["result_line"] is True and s["phases"]["C"]["sim_exit"] == "0"      # the REAL run_fab_sim.sh text ran, with the fake FABulous
    assert s["ramp_max_mb"] == 12 and s["ramp_step_mb"] == 4                                   # 3 real steps of 4 MB were taken by real Python
    assert s["samples"] >= 1 and s["mem_total_mb"] > 0 and s["cpus"] >= 1
    text = " ".join(wp.verdict(s, False))
    assert "Ubuntu stayed up this time" in text and "printed a result line" in text and "run the chain's simulation step again" in text
    assert not list((demo / "Test" / "build").glob("probe_*.vvp"))                              # its temporary files are gone
    assert (demo / "Test" / "build" / "fabric_files" / "t.v").exists()                           # same copy step the Taskfile does
    assert (demo / "user_design" / "add8_sat_fab.v").exists()                                     # copied from the job, as the flow step does


@needs_bash
def test_phase_c_writes_its_results_away_from_the_job_folder(tmp_path):
    run_probe(tmp_path, "ok")
    job, out = tmp_path / "proj" / "reports" / "fab_jobs" / "add8_sat", tmp_path / "proj" / "reports" / "wsl_probe"
    assert not (job / "sim_console.log").exists() and not (job / "sim_exit_code.txt").exists()
    assert "FAB_SIM RESULT" in (out / "phaseC" / "sim_console.log").read_text() and (out / "phaseC" / "sim_exit_code.txt").read_text().strip() == "0"
    assert sorted(p.name for p in job.iterdir()) == ["add8_sat_fab.v", "add8_sat_fab_tb.v", "run_fab_sim.sh"]     # the job folder is exactly as it was


@needs_bash
def test_no_bitstream_skips_phase_c_and_says_why(tmp_path):
    r, log, _ = run_probe(tmp_path, "ok", bitstream=False)
    s = wp.summarize_log(log)
    text = " ".join(wp.verdict(s, False))
    assert r.returncode == 0 and s["done"] and s["c_skipped"] and "C" not in s["phases"]
    assert "skipped" in text and "flow stage" in text and "Ubuntu stayed up this time" in text


@needs_bash
def test_the_real_script_finishing_without_a_result_line_is_flagged(tmp_path):
    _, log, _ = run_probe(tmp_path, "no_result")
    s = wp.summarize_log(log)
    text = " ".join(wp.verdict(s, False))
    assert s["phases"]["C"]["result_line"] is False and "WITHOUT a result line" in text and "phaseC/console.out" in text


@needs_bash
def test_a_script_killed_during_the_real_simulation_blames_what_the_script_adds(tmp_path):
    r, log, _ = run_probe(tmp_path, "die_c")
    s = wp.summarize_log(log)
    text = " ".join(wp.verdict(s, True))
    assert not s["done"] and s["last_phase_started"] == "C" and {"R", "A", "B"} <= set(s["phases"]) and "C" not in s["phases"]
    assert "the real simulation script (phase C)" in text and "Both bare compiles had finished" in text and "FABulous itself" in text


@needs_bash
def test_a_compile_that_fails_under_the_cap_is_reported_with_its_message(tmp_path):
    r, log, _ = run_probe(tmp_path, "fail")
    s = wp.summarize_log(log)
    text = " ".join(wp.verdict(s, False))
    assert s["done"] and s["phases"]["A"]["exit"] == 3 and "CAPPED compile" in text and "out of memory" in text


@needs_bash
def test_a_script_killed_during_phase_b_is_reported_as_a_death_with_the_last_samples(tmp_path):
    r, log, _ = run_probe(tmp_path, "die_b")
    s = wp.summarize_log(log)
    text = " ".join(wp.verdict(s, True))
    assert r.returncode != 0 and not s["done"] and s["last_phase_started"] == "B" and {"R", "A"} <= set(s["phases"]) and "B" not in s["phases"]
    assert "uncapped compile (phase B)" in text and "plain memory test had passed (12 MB)" in text and "VM restarted" in text and "Last sample" in text


@needs_bash
def test_a_script_killed_during_phase_a_names_phase_a(tmp_path):
    _, log, _ = run_probe(tmp_path, "die_a")
    s = wp.summarize_log(log)
    assert not s["done"] and s["last_phase_started"] == "A" and "capped compile (phase A)" in " ".join(wp.verdict(s, None))


@needs_bash
def test_a_script_killed_during_the_plain_memory_test_says_no_compiler_is_needed(tmp_path):
    r, log, _ = run_probe(tmp_path, "die_ramp")
    s = wp.summarize_log(log)
    text = " ".join(wp.verdict(s, True))
    assert not s["done"] and s["last_phase_started"] == "R" and s["ramp_max_mb"] == 8 and not s["phases"]
    assert "PLAIN MEMORY TEST" in text and "after Python had taken 8 MB" in text and "Icarus is not needed" in text and "past 8 MB" in text


@needs_bash
def test_the_sampler_does_not_outlive_a_killed_script(tmp_path):
    _, log, _ = run_probe(tmp_path, "die_a")
    n1 = log.count("SAMPLE")
    time.sleep(1.5)
    out = tmp_path / "proj" / "reports" / "wsl_probe"
    assert (out / "probe.log").read_text().count("SAMPLE") == n1                              # nothing keeps writing after the script died


@needs_bash
def test_the_sampler_stops_even_while_the_dead_script_is_still_an_unreaped_zombie(tmp_path):
    """Nobody calls wait() here, so the killed script stays a zombie (its /proc entry still exists). The sampler must notice that and stop."""
    home, demo, out = make_world(tmp_path)
    env = dict(os.environ, HOME=str(home), DEMO=str(demo), FAKE_MODE="die_a", FAKE_COUNT=str(tmp_path / "count"), PROBE_LOG=str(out / "probe.log"),
               REAL_PYTHON=sys.executable, RAMP_STEPS="2", RAMP_STEP_MB="2")
    p = subprocess.Popen(["bash", str(out / "probe.sh"), "add8_sat"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        status = Path(f"/proc/{p.pid}/status")
        for _ in range(100):
            if "\nState:\tZ" in status.read_text():
                break
            time.sleep(0.1)
        else:
            pytest.fail("the fake compile did not kill the probe script")
        time.sleep(1.2)                                                       # a few sampler intervals to notice
        n1 = (out / "probe.log").read_text().count("SAMPLE")
        time.sleep(1.5)
        assert status.exists() and (out / "probe.log").read_text().count("SAMPLE") == n1        # still a zombie, and the sampler has stopped
    finally:
        p.wait()


def test_the_host_sampler_really_stops_when_told_to():
    sm = wp.HostSampler(reader=lambda: {"total_gb": 8, "free_gb": 3.0, "commit_headroom_gb": 5.0}, interval=0.01).start()
    time.sleep(0.1)
    got = sm.stop()
    n = len(sm.samples)
    time.sleep(0.15)
    assert n > 0 and len(sm.samples) == n and len(got) == n and not sm._thread.is_alive()


# ------------------------------------------------------------ summary and verdict logic on handwritten logs
GOOD = """10:00:00 START pid=42 kernel=6.6.0 cpus=4 mem_total_kB=3145728 swap_total_kB=8388608 uptime_s=5.1
10:00:01 FILES fabric_files=120 build_size=100M
10:00:01 PHASE R_start (plain memory test: Python takes 128 MB at a time, 16 steps, no compiler)
10:00:01 RAMP 128 MB ok
10:00:02 SAMPLE avail_kB=2900000 swap_free_kB=8388608 work_rss_kB=300000 load=0.1 top=python3:292MB,Xorg:50MB,bash:4MB,
10:00:02 RAMP 256 MB ok
10:00:03 PHASE R_end exit=0 secs=2
10:00:03 PHASE A_start (compile capped at 2.5 GB of memory)
10:00:03 SAMPLE avail_kB=2000000 swap_free_kB=8388608 work_rss_kB=500000 load=0.9 top=ivl:488MB,bash:4MB,sh:1MB,
10:00:04 SAMPLE avail_kB=1500000 swap_free_kB=8388608 work_rss_kB=1200000 load=1.0 top=ivl:1171MB,bash:4MB,sh:1MB,
10:00:05 PHASE A_end exit=0 secs=2
10:00:05 PHASE B_start (compile with no cap, as FABulous runs it)
10:00:05 SAMPLE avail_kB=1400000 swap_free_kB=8388608 work_rss_kB=1300000 load=1.0 top=ivl:1269MB,bash:4MB,sh:1MB,
10:00:06 PHASE B_end exit=0 secs=1
10:00:06 PHASE C_start (the real run_fab_sim.sh, as the chain runs it; its results go to reports/wsl_probe/phaseC, not to your job folder)
10:00:06 SAMPLE avail_kB=1300000 swap_free_kB=8388608 work_rss_kB=1500000 load=1.0 top=ivl:1270MB,FABulous:230MB,task:20MB,
10:00:08 PHASE C_end exit=0 secs=2 sim_exit=0 result_line=1
10:00:08 DONE
"""
HOST_LOW = {"samples": 20, "total_gb": 7.8, "start_free_gb": 2.6, "min_free_gb": 0.3, "min_free_at_s": 5.5, "end_free_gb": 2.5, "min_commit_headroom_gb": 0.4}
HOST_OK = {**HOST_LOW, "start_free_gb": 5.2, "min_free_gb": 3.9, "min_commit_headroom_gb": 6.1}
HOST_MID = {**HOST_LOW, "min_free_gb": 1.0, "min_commit_headroom_gb": 2.5}


def test_summary_numbers():
    s = wp.summarize_log(GOOD)
    assert s["done"] and s["mem_total_mb"] == 3072 and s["swap_total_mb"] == 8192 and s["samples"] == 5
    assert s["min_avail_mb"] == 1269 and s["peak_work_rss_mb"] == 1464 and s["last_phase_started"] == "C" and not s["c_skipped"]
    assert s["ramp_max_mb"] == 256 and s["ramp_step_mb"] == 128 and s["last_top"] == "ivl:1270MB,FABulous:230MB,task:20MB"
    assert s["phases"] == {"R": {"exit": 0, "secs": 2}, "A": {"exit": 0, "secs": 2}, "B": {"exit": 0, "secs": 1},
                           "C": {"exit": 0, "secs": 2, "sim_exit": "0", "result_line": True}}


def test_old_format_sample_lines_without_top_still_parse():
    s = wp.summarize_log("10:00:00 START pid=1 kernel=k cpus=2 mem_total_kB=2048 swap_total_kB=0 uptime_s=1\n"
                         "10:00:01 SAMPLE avail_kB=1024 swap_free_kB=0 work_rss_kB=2048 load=0.5\n")
    assert s["samples"] == 1 and s["last_top"] is None and s["last_work_rss_mb"] == 2


def test_log_cut_off_in_phase_b_is_a_death_and_names_the_biggest_program():
    cut = GOOD[:GOOD.index("10:00:06 PHASE B_end")]
    text = " ".join(wp.verdict(wp.summarize_log(cut), True))
    assert "uncapped compile (phase B)" in text and "Last sample at 10:00:05: 1367 MB available, test programs using 1269 MB" in text
    assert "biggest programs then: ivl:1269MB,bash:4MB,sh:1MB." in text and "VM restarted" in text and "Both bare compiles" not in text


def test_log_cut_off_in_phase_c_blames_what_the_real_script_adds():
    cut = GOOD[:GOOD.index("10:00:08 PHASE C_end")]
    s = wp.summarize_log(cut)
    text = " ".join(wp.verdict(s, True))
    assert s["last_phase_started"] == "C" and "C" not in s["phases"]
    assert "the real simulation script (phase C)" in text and "Both bare compiles had finished" in text and "FABulous:230MB" in text


def test_one_sample_is_not_called_samples():
    cut = GOOD[:GOOD.index("10:00:02 RAMP 256")]
    assert "1 sample." in " ".join(wp.verdict(wp.summarize_log(cut), True)) and "1 samples" not in " ".join(wp.verdict(wp.summarize_log(cut), True))


def test_log_cut_off_during_the_ramp():
    cut = GOOD[:GOOD.index("10:00:02 RAMP 256")]
    s = wp.summarize_log(cut)
    text = " ".join(wp.verdict(s, True))
    assert s["last_phase_started"] == "R" and s["ramp_max_mb"] == 128 and "PLAIN MEMORY TEST" in text and "past 128 MB" in text


def test_nothing_started_and_missing_project_have_their_own_messages():
    assert "never started" in wp.verdict(wp.summarize_log(""), None)[0]
    log = "10:00:00 START pid=1 kernel=k cpus=1 mem_total_kB=1 swap_total_kB=1 uptime_s=1\n10:00:00 NO_DEMO x"
    assert "demo project was not found" in wp.verdict(wp.summarize_log(log), None)[0]


def test_uncapped_failure_without_a_death_is_not_called_a_crash():
    log = GOOD.replace("PHASE B_end exit=0 secs=1", "PHASE B_end exit=1 secs=1")
    text = " ".join(wp.verdict(wp.summarize_log(log), False))
    assert "uncapped compile failed with exit 1" in text and "without Ubuntu dying" in text and "did not restart" in text


def test_ubuntu_stopping_its_own_memory_test_is_normal():
    log = GOOD.replace("PHASE R_end exit=0 secs=2", "PHASE R_end exit=137 secs=2")
    text = " ".join(wp.verdict(wp.summarize_log(log), False))
    assert "stopped by Ubuntu itself at 256 MB" in text and "normal behaviour" in text


def test_vm_restart_detection_with_your_real_numbers():
    assert wp.vm_restarted(4.11, 3.68, 20) is True               # what your machine showed
    assert wp.vm_restarted(4.11, 24.1, 20) is False              # a normal run: uptime keeps counting
    assert wp.vm_restarted(4.0, 18.0, 20) is True                # restarted early in the run: 18 s is MORE than the 4 s before, but far less than 4 + 20
    assert wp.vm_restarted(4.0, 3.0, 3) is None                  # too short to tell
    assert wp.vm_restarted(None, 5.0, 30) is None and wp.vm_restarted(5.0, None, 30) is None


def test_uptime_parsing():
    assert wp.parse_uptime("4.11 23.84\n") == 4.11 and wp.parse_uptime("") is None and wp.parse_uptime("garbage") is None


# ------------------------------------------------------------ Windows side: verdict from host numbers
def test_windows_nearly_out_of_ram_is_called_the_likely_cause():
    cut = GOOD[:GOOD.index("10:00:06 PHASE B_end")]
    text = " ".join(wp.verdict(wp.summarize_log(cut), True, HOST_LOW))
    assert "lowest 0.3 GB" in text and "nearly out of commit headroom (0.4 GB)" in text and "likely cause" in text and "commit headroom lowest 0.4 GB" in text
    passed = {**HOST_LOW, "min_free_gb": 0.4, "min_commit_headroom_gb": 4.1}                     # your passing probe: free RAM 0.4 GB, headroom 4.1 GB
    text = " ".join(wp.verdict(wp.summarize_log(cut), True, passed))
    assert "likely cause" not in text and "does not explain it" in text
    no_commit = {**HOST_LOW, "min_commit_headroom_gb": None}                                     # headroom unknown: free RAM is all there is
    assert "nearly out of free RAM (0.3 GB)" in " ".join(wp.verdict(wp.summarize_log(cut), True, no_commit))


def test_plenty_of_windows_ram_points_away_from_windows_memory():
    cut = GOOD[:GOOD.index("10:00:06 PHASE B_end")]
    text = " ".join(wp.verdict(wp.summarize_log(cut), True, HOST_OK))
    assert "does not explain it" in text and "likely cause" not in text and "WSL itself" in text


def test_in_between_is_not_called_proof():
    cut = GOOD[:GOOD.index("10:00:06 PHASE B_end")]
    text = " ".join(wp.verdict(wp.summarize_log(cut), True, HOST_MID))
    assert "possible contributor, not proof" in text and "likely cause" not in text


def test_a_healthy_run_makes_no_claim_about_causes():
    text = " ".join(wp.verdict(wp.summarize_log(GOOD), False, HOST_LOW))
    assert "Windows memory during the run" in text and "likely cause" not in text and "did not restart" in text


# ------------------------------------------------------------ Windows side: readings
MEMINFO = """MemTotal:        8000000 kB
MemFree:         1000000 kB
MemAvailable:    2000000 kB
CommitLimit:    10000000 kB
Committed_AS:    7000000 kB
"""


def test_meminfo_parsing():
    r = wp.parse_meminfo(MEMINFO)
    assert round(r["free_gb"], 2) == 1.91 and round(r["total_gb"], 2) == 7.63 and round(r["commit_headroom_gb"], 2) == 2.86
    assert wp.parse_meminfo("nonsense") is None and wp.parse_meminfo("") is None
    assert wp.parse_meminfo("MemTotal: 100 kB\nMemAvailable: 50 kB\n")["commit_headroom_gb"] is None


def test_the_windows_api_structure_has_the_documented_size():
    assert ctypes.sizeof(wp.MEMORYSTATUSEX) == 64          # two 32-bit DWORDs + seven 64-bit values: the layout GlobalMemoryStatusEx expects, on every platform


def test_the_real_reader_works_on_this_machine():
    """On Windows this calls GlobalMemoryStatusEx for real; if it returns None there, the host-memory lines of the probe cannot be trusted."""
    if not (sys.platform.startswith("win") or sys.platform.startswith("linux")):
        pytest.skip("only Windows and Linux are supported")
    r = wp.read_host_memory()
    assert r is not None, "the host memory reader returned nothing on this machine"
    assert 0 < r["free_gb"] <= r["total_gb"] < 4096 and r["total_gb"] > 0.5
    assert r["commit_headroom_gb"] is None or r["commit_headroom_gb"] >= 0


def test_sampler_collects_readings_and_stops_when_the_reader_gives_up():
    vals = iter([{"total_gb": 8, "free_gb": 3.0, "commit_headroom_gb": 5.0}, {"total_gb": 8, "free_gb": 1.0, "commit_headroom_gb": 4.0},
                 {"total_gb": 8, "free_gb": 2.0, "commit_headroom_gb": 4.5}, None])
    sm = wp.HostSampler(reader=lambda: next(vals), interval=0.01).start()
    time.sleep(0.3)
    samples = sm.stop()
    assert [s["free_gb"] for s in samples] == [3.0, 1.0, 2.0]
    h = wp.host_summary(samples)
    assert h["start_free_gb"] == 3.0 and h["min_free_gb"] == 1.0 and h["end_free_gb"] == 2.0 and h["min_commit_headroom_gb"] == 4.0 and h["samples"] == 3
    assert wp.host_summary([]) is None


def test_the_windows_branch_reads_the_right_fields_in_the_right_units(monkeypatch):
    """A fake kernel32 fills the real ctypes structure the way GlobalMemoryStatusEx does, so the field names, the length field and the GB conversion are exercised here."""
    class Kernel32:
        ok = 1

        def GlobalMemoryStatusEx(self, ref):
            st = ref._obj
            assert st.dwLength == 64
            st.ullTotalPhys, st.ullAvailPhys, st.ullAvailPageFile = 8 * wp.GB, 3 * wp.GB, 5 * wp.GB
            return self.ok

    k = Kernel32()
    monkeypatch.setattr(wp.ctypes, "windll", type("W", (), {"kernel32": k}), raising=False)
    assert wp.read_host_memory("win32") == {"total_gb": 8.0, "free_gb": 3.0, "commit_headroom_gb": 5.0}
    k.ok = 0                                                                             # the API says it failed
    assert wp.read_host_memory("win32") is None


def test_a_reader_that_raises_is_handled_by_the_reader_not_the_sampler(monkeypatch):
    monkeypatch.setattr(wp.ctypes, "windll", None, raising=False)
    assert wp.read_host_memory("win32") is None             # no windll (as on Linux) -> None, never an exception


# ------------------------------------------------------------ the runner, with a fake Ubuntu and fake Windows readings
def result(**o):
    d = {"started": True, "returncode": 0, "output": "", "seconds": 1.0, "timed_out": False}
    d.update(o)
    return d


class FakeUbuntu:
    def __init__(self, uptimes=("4.11 23.8\n", "24.5 40.0\n"), log=GOOD, up_ok=True):
        self.uptimes, self.log, self.cmds, self.up_ok = list(uptimes), log, [], up_ok

    def __call__(self, cmd, timeout_s, heartbeat_s, say):
        self.cmds.append(cmd)
        if cmd[-2:] == ["cat", "/proc/uptime"]:
            return result(output=self.uptimes.pop(0)) if self.up_ok else result(output="", returncode=1)
        time.sleep(0.15)                                                  # lets the host sampler take a few readings
        Path("reports/wsl_probe/probe.log").write_text(self.log)
        return result()


def CLOCK():
    return iter([0.0, 20.0, 20.0, 20.0]).__next__


def host_reader(free):
    vals = iter(free)
    last = [free[-1]]

    def read():
        try:
            last[0] = next(vals)
        except StopIteration:
            pass
        return {"total_gb": 7.8, "free_gb": last[0], "commit_headroom_gb": last[0]}
    return read


@pytest.fixture
def jobdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "reports" / "fab_jobs" / "add8_sat").mkdir(parents=True)
    (tmp_path / "reports" / "fab_jobs" / "add8_sat" / "add8_sat_fab_tb.v").write_text("module x; endmodule")
    return tmp_path


def test_runner_order_and_a_clean_report(jobdir, capsys):
    fake = FakeUbuntu()
    assert probe.main(["--runner", "native"], exec_fn=fake, clock=CLOCK(), host_reader=host_reader([5.0])) == 0
    out = capsys.readouterr().out
    assert fake.cmds[0] == ["cat", "/proc/uptime"] and fake.cmds[1][0] == "bash" and fake.cmds[1][1].endswith("probe.sh") and fake.cmds[1][2] == "add8_sat"
    assert "Windows right now: 5.0 GB free RAM of 7.8 GB" in out and "close browsers" in out
    assert "=== WSL PROBE RESULT ===" in out and "Ubuntu stayed up this time" in out and "did not restart" in out and "Windows memory during the run" in out
    assert (jobdir / "reports" / "wsl_probe" / "probe_summary.json").exists()
    assert b"\r" not in (jobdir / "reports" / "wsl_probe" / "probe.sh").read_bytes()


def test_runner_reports_a_vm_restart_with_windows_nearly_out_of_ram(jobdir, capsys):
    fake = FakeUbuntu(uptimes=("4.11 23.8\n", "3.68 20.6\n"), log=GOOD[:GOOD.index("10:00:06 PHASE B_end")])
    probe.main(["--runner", "native"], exec_fn=fake, clock=CLOCK(), host_reader=host_reader([2.6, 0.3]))        # 1st reading = "right now", then the sampler sees 0.3 GB
    out = capsys.readouterr().out
    assert "uncapped compile (phase B)" in out and "the whole Ubuntu VM restarted" in out and "uptime before 4.1 s, after 3.7 s" in out
    assert "nearly out of commit headroom (0.3 GB)" in out and "likely cause" in out


def test_runner_says_so_when_ubuntu_does_not_answer_after_the_run(jobdir, capsys):
    fake = FakeUbuntu(uptimes=("4.6 10.0\n", ""), log=GOOD[:GOOD.index("10:00:06 PHASE B_end")])           # your first real probe: uptime after = None
    probe.main(["--runner", "native"], exec_fn=fake, clock=CLOCK(), host_reader=host_reader([2.6]))
    out = capsys.readouterr().out
    assert "uptime before 4.6 s, after no answer (Ubuntu was not responding)" in out and "uptime counter went back" not in out and "Ubuntu died during" in out


def test_runner_needs_the_job_folder_and_a_working_ubuntu(jobdir):
    (jobdir / "reports" / "fab_jobs" / "add8_sat" / "add8_sat_fab_tb.v").unlink()
    with pytest.raises(SystemExit, match="x2c chain"):
        probe.main(["--runner", "native"], exec_fn=FakeUbuntu(), host_reader=host_reader([5.0]))
    (jobdir / "reports" / "fab_jobs" / "add8_sat" / "add8_sat_fab_tb.v").write_text("x")
    with pytest.raises(SystemExit, match="did not answer"):
        probe.main(["--runner", "native"], exec_fn=FakeUbuntu(up_ok=False), host_reader=host_reader([5.0]))


def test_old_log_is_removed_so_a_dead_run_cannot_show_an_old_success(jobdir, capsys):
    out = jobdir / "reports" / "wsl_probe"
    out.mkdir(parents=True)
    (out / "probe.log").write_text(GOOD)                                  # left by an earlier, healthy run

    class Dies(FakeUbuntu):
        def __call__(self, cmd, *a):
            if cmd[0] == "bash":
                self.cmds.append(cmd)
                return result(returncode=1)                               # the probe wrote nothing this time
            return super().__call__(cmd, *a)
    probe.main(["--runner", "native"], exec_fn=Dies(), clock=CLOCK(), host_reader=host_reader([5.0]))
    assert "never started" in capsys.readouterr().out


def test_runner_works_when_the_host_cannot_be_read(jobdir, capsys):
    assert probe.main(["--runner", "native"], exec_fn=FakeUbuntu(), clock=CLOCK(), host_reader=lambda: None) == 0
    out = capsys.readouterr().out
    assert "=== WSL PROBE RESULT ===" in out and "Windows memory during the run" not in out and "Windows right now" not in out


def test_wsl_uptime_command_shape():
    assert probe.uptime_command("wsl", "Ubuntu") == ["wsl", "-d", "Ubuntu", "--", "cat", "/proc/uptime"]
    assert probe.uptime_command("native", "Ubuntu") == ["cat", "/proc/uptime"]


# ------------------------------------------------------------ step 26: the memory gate and what is using the memory
def host(free, head=None, total=7.8):
    """One number means: this much free RAM and the same commit headroom."""
    return {"total_gb": total, "free_gb": free, "commit_headroom_gb": free if head is None else head}


def test_memory_gate_levels_with_your_real_numbers():
    gate, text = wp.memory_gate(host(0.6, 1.5))                                  # what your machine showed just before the probe that never started
    assert gate == "block" and "only 1.5 GB of commit headroom (free RAM 0.6 GB)" in text and "wsl --shutdown" in text
    gate, text = wp.memory_gate(host(2.6))
    assert gate == "warn" and "about 4 GB" in text and "Windows has 2.6 GB of commit headroom" in text
    assert wp.memory_gate(host(5.0)) == ("ok", "") and wp.memory_gate(None) == ("unknown", "")
    assert wp.memory_gate(host(2.0))[0] == "warn" and wp.memory_gate(host(1.99))[0] == "block"      # below 2 GB of commit headroom is refused
    assert wp.memory_gate(host(0.4, 3.1))[0] == "warn"                                               # your passing run: free RAM 0.4 GB, headroom 3.1 GB
    no_commit = {"total_gb": 7.8, "free_gb": 0.9}                                                    # headroom unknown: free RAM decides, as before
    assert wp.memory_gate(no_commit)[0] == "block" and "only 0.9 GB of free RAM:" in wp.memory_gate(no_commit)[1]
    assert wp.memory_gate({"total_gb": 7.8, "free_gb": 1.0})[0] == "warn"
    assert wp.memory_gate(host(4.0))[0] == "ok" and wp.memory_gate(host(3.99))[0] == "warn"          # 4 GB is comfortable


def test_commit_headroom_can_be_the_limit():
    gate, text = wp.memory_gate(host(3.0, 0.8))
    assert gate == "block" and "only 0.8 GB of commit headroom (free RAM 3.0 GB)" in text
    assert wp.tightest_gb(host(3.0, 5.0)) == (5.0, "commit headroom") and wp.tightest_gb({"free_gb": 3.0}) == (3.0, "free RAM")
    assert wp.tightest_gb(None) == (None, None)


TASKLIST = '''"chrome.exe","1234","Console","1","412,340 K"
"chrome.exe","1300","Console","1","98,000 K"
"Code.exe","2000","Console","1","300 000 K"
"vmmemWSL","3000","Console","0","2.900.000 K"
"System","4","Services","0","144 K"
'''


def test_tasklist_parsing_adds_up_programs_and_ignores_number_formats():
    t = wp.parse_tasklist(TASKLIST)
    assert round(t["chrome.exe"][0]) == 498 and t["chrome.exe"][1] == 2                    # (412,340 + 98,000) K
    assert round(t["Code.exe"][0]) == 293 and round(t["vmmemWSL"][0]) == 2832             # '300 000 K' and '2.900.000 K': other languages write numbers differently
    assert wp.parse_tasklist("") == {} and wp.parse_tasklist("garbage\n") == {}


def test_top_memory_users_with_canned_output_for_windows_and_linux():
    seen = []

    def fake_run(cmd):
        seen.append(cmd)
        return TASKLIST
    top = REAL_TOP(n=3, platform="win32", run_text=fake_run)
    assert seen == [["tasklist", "/fo", "csv", "/nh"]] and [u["name"] for u in top] == ["vmmemWSL", "chrome.exe", "Code.exe"]
    assert top[1] == {"name": "chrome.exe", "mb": 498, "count": 2}
    text = wp.format_users(top)
    assert "chrome.exe 498 MB (2 processes)" in text and "Code.exe 293 MB (1 process)" in text and "wsl --shutdown" in text      # vmmem is listed: says what it is
    assert "vmmem" not in wp.format_users(top[1:])
    ps = REAL_TOP(n=2, platform="linux", run_text=lambda cmd: " 204800 chrome\n  1024 bash\n 102400 chrome\n")
    assert ps == [{"name": "chrome", "mb": 300, "count": 2}, {"name": "bash", "mb": 1, "count": 1}]

    def broken(cmd):
        raise OSError("no tasklist")
    assert REAL_TOP(platform="win32", run_text=broken) == []              # a hint must never stop a run


def test_the_real_process_listing_works_on_this_machine():
    """On Windows this runs the real `tasklist`; if it returns nothing there, the 'biggest memory users' hint cannot be trusted."""
    top = REAL_TOP()
    assert top and all(u["name"] and u["mb"] >= 0 and u["count"] >= 1 for u in top)
    assert [u["mb"] for u in top] == sorted((u["mb"] for u in top), reverse=True)


def test_simulation_memory_sentences():
    h = {"samples": 5, "total_gb": 7.8, "start_free_gb": 2.6, "min_free_gb": 0.4, "min_free_at_s": 5.0, "end_free_gb": 2.5, "min_commit_headroom_gb": 1.0}
    low = wp.sim_memory_sentence(h)
    assert "commit headroom fell to 1.0 GB" in low and "most likely stopped for lack of memory" in low and "--op add8_sat --from-stage sim" in low
    assert "--op relu8" in wp.sim_memory_sentence(h, "relu8")
    mid = wp.sim_memory_sentence({**h, "min_commit_headroom_gb": 2.5})
    assert "commit headroom got low (2.5 GB" in mid and "may be the reason" in mid and "most likely" not in mid and "x2c doctor wsl" in mid
    assert "may be the reason" in wp.sim_memory_sentence({**h, "min_commit_headroom_gb": 2.0}) and "most likely" in wp.sim_memory_sentence({**h, "min_commit_headroom_gb": 1.99})
    assert "probably not" in wp.sim_memory_sentence({**h, "min_commit_headroom_gb": 3.0}) and "may be" in wp.sim_memory_sentence({**h, "min_commit_headroom_gb": 2.99})
    old = {**h, "min_commit_headroom_gb": None}                                                  # no headroom reading: free RAM, with the old lines
    assert "free RAM fell to 0.4 GB" in wp.sim_memory_sentence(old) and "may be" in wp.sim_memory_sentence({**old, "min_free_gb": 1.2})
    plenty = wp.sim_memory_sentence({**h, "min_free_gb": 0.4, "min_commit_headroom_gb": 4.1})         # your passing runs
    assert "probably not the reason" in plenty and "most likely" not in plenty and "wsl --shutdown" not in plenty
    assert wp.sim_memory_sentence(None) == ""


def test_never_started_with_low_windows_memory_names_the_likely_cause():
    low = {**HOST_LOW, "start_free_gb": 0.6, "min_free_gb": 0.6, "min_commit_headroom_gb": 1.5, "samples": 1}     # your machine: 0.6 GB free, 1.5 GB headroom, Ubuntu gone before the log existed
    text = " ".join(wp.verdict(wp.summarize_log(""), None, low))
    assert "never started" in text and "free RAM 0.6 GB at the start" in text and "nearly out of commit headroom (1.5 GB)" in text and "most likely stopped at once" in text
    ok = " ".join(wp.verdict(wp.summarize_log(""), None, HOST_OK))
    assert "never started" in ok and "nearly out" not in ok
    assert len(wp.verdict(wp.summarize_log(""), None, None)) == 1                                           # without readings: the plain message only


class NeverStarts(FakeUbuntu):
    """wsl.exe answers the uptime question but the probe script itself dies at once and leaves no log (as on your machine)."""
    def __call__(self, cmd, timeout_s, heartbeat_s, say):
        if cmd[0] == "bash":
            self.cmds.append(cmd)
            time.sleep(0.15)                                                         # long enough for the Windows sampler to take a reading
            return result(returncode=-1, output="Wsl/Service/E_UNEXPECTED\r\n", seconds=0.0)
        return super().__call__(cmd, timeout_s, heartbeat_s, say)


def test_probe_refuses_to_start_when_windows_has_almost_no_memory(jobdir, capsys):
    fake = FakeUbuntu()
    users = [{"name": "chrome.exe", "mb": 1800, "count": 14}, {"name": "Code.exe", "mb": 900, "count": 7}]
    rc = probe.main(["--runner", "native"], exec_fn=fake, clock=CLOCK(), host_reader=lambda: host(0.6, 1.5), users_reader=lambda: users)
    out = capsys.readouterr().out
    assert rc == 2 and fake.cmds == []                                              # Ubuntu was never touched
    assert "Windows right now: 0.6 GB free RAM of 7.8 GB, commit headroom 1.5 GB" in out and "chrome.exe 1800 MB (14 processes)" in out
    assert "=== WSL PROBE NOT STARTED ===" in out and "only 1.5 GB of commit headroom" in out and "--force" in out
    assert not (jobdir / "reports" / "wsl_probe" / "probe.sh").exists() and "WSL PROBE RESULT" not in out


def test_force_runs_anyway_and_a_probe_that_never_started_shows_what_wsl_printed(jobdir, capsys):
    fake = NeverStarts(uptimes=("8.2 9.0\n", ""))
    rc = probe.main(["--runner", "native", "--force"], exec_fn=fake, clock=CLOCK(), host_reader=lambda: host(0.6, 1.5), users_reader=lambda: [])
    out = capsys.readouterr().out
    assert rc == 0 and "--force given: running anyway" in out and fake.cmds[1][0] == "bash"
    assert "The probe never started in Ubuntu" in out and "What wsl.exe printed (exit code -1): Wsl/Service/E_UNEXPECTED" in out
    assert "nearly out of commit headroom (1.5 GB)" in out and "uptime before 8.2 s, after no answer" in out                  # your first probe output, now with the reason


def test_warning_level_lists_the_biggest_users_and_still_runs(jobdir, capsys):
    fake = FakeUbuntu()
    rc = probe.main(["--runner", "native"], exec_fn=fake, clock=CLOCK(), host_reader=lambda: host(2.6),
                    users_reader=lambda: [{"name": "chrome.exe", "mb": 1800, "count": 14}])
    out = capsys.readouterr().out
    assert rc == 0 and "Biggest memory users right now: chrome.exe 1800 MB (14 processes)" in out and "NOT STARTED" not in out and len(fake.cmds) == 3


def test_enough_memory_does_not_list_anything(jobdir, capsys):
    def must_not_be_called():
        raise AssertionError("the process list is only needed when memory is short")
    assert probe.main(["--runner", "native"], exec_fn=FakeUbuntu(), clock=CLOCK(), host_reader=lambda: host(5.0), users_reader=must_not_be_called) == 0
    assert "Biggest memory users" not in capsys.readouterr().out


def test_runner_looks_up_the_process_lister_when_none_is_given(jobdir, capsys, monkeypatch):
    monkeypatch.setattr(wp, "top_memory_users", lambda *a, **k: [{"name": "edge.exe", "mb": 700, "count": 3}])
    probe.main(["--runner", "native"], exec_fn=FakeUbuntu(), clock=CLOCK(), host_reader=lambda: host(2.0))
    assert "edge.exe 700 MB (3 processes)" in capsys.readouterr().out


# ------------------------------------------------------------ Ubuntu does not answer its first question (your run at 2.6 GB free)
class Scripted:
    """A fake Ubuntu whose answers to the uptime question come from a list (the last answer repeats); everything else works."""
    def __init__(self, answers, log=GOOD):
        self.answers, self.log, self.cmds = list(answers), log, []

    def __call__(self, cmd, timeout_s, heartbeat_s, say):
        self.cmds.append(cmd)
        if cmd[-2:] == ["cat", "/proc/uptime"]:
            return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        time.sleep(0.15)                                                  # lets the host sampler take a few readings
        Path("reports/wsl_probe/probe.log").write_text(self.log)
        return result()


DOWN = result(returncode=1, output="Wsl/Service/CreateInstance/E_UNEXPECTED\r\n", seconds=2.0)


def up(text="4.11 23.8\n"):
    return result(output=text)


def test_uptime_is_found_even_when_wsl_prints_warnings_above_it():
    warning = "wsl: Failed to start the systemd user session for 'pankaja'. See journalctl for more details."
    noisy = warning + "\r\n12.50 99.10\r\n"
    assert wp.parse_uptime(noisy) == 12.5 and wp.parse_uptime("4.11 23.84\n") == 4.11
    assert wp.parse_uptime("wsl: something 3 4\n") is None and wp.parse_uptime("") is None and wp.parse_uptime(None) is None
    assert wp.uptime_extras(noisy) == [warning] and wp.uptime_extras("4.11 23.84\n") == [] and wp.uptime_extras("") == []


def test_first_question_is_asked_again_when_ubuntu_does_not_answer_once(jobdir, capsys, waited):
    fake = Scripted([DOWN, up("5.0 9.0\n"), up("25.0 40.0\n")])
    assert probe.main(["--runner", "native"], exec_fn=fake, clock=CLOCK(), host_reader=host_reader([5.0])) == 0
    out = capsys.readouterr().out
    assert waited == [5]
    assert "Ubuntu did not answer (try 1 of 3; exit code 1, 2 s). wsl.exe printed: Wsl/Service/CreateInstance/E_UNEXPECTED" in out
    assert "Trying again in 5 s" in out and "Ubuntu is up (uptime 5.0 s, after 2 tries)" in out
    assert sum(1 for c in fake.cmds if c[-2:] == ["cat", "/proc/uptime"]) == 3                      # two before the run, one after
    data = json.loads((jobdir / "reports" / "wsl_probe" / "probe_summary.json").read_text())
    assert data["uptime_tries"] == 2 and data["wsl_facts"] == []


def test_ubuntu_never_answers_shows_every_try_and_the_facts_about_wsl(jobdir, capsys, waited):
    fake = Scripted([DOWN])
    facts = ["WSL: WSL version: 2.4.13.0 / Kernel version: 5.15.167.4-1", "Distros: NAME STATE VERSION / * Ubuntu Stopped 2", "Free disk space: C: 3.1 GB, E: 80.0 GB"]
    with pytest.raises(SystemExit, match=r"did not answer `cat /proc/uptime` \(3 tries"):
        probe.main(["--runner", "native"], exec_fn=fake, host_reader=host_reader([5.0]), snapshot_fn=lambda: facts)
    out = capsys.readouterr().out
    assert waited == [5, 15] and len(fake.cmds) == 3 and all(c[-2:] == ["cat", "/proc/uptime"] for c in fake.cmds)        # nothing else was asked of the broken Ubuntu
    assert out.count("Ubuntu did not answer (try") == 3 and "try 3 of 3" in out and "Trying again in 15 s" in out and out.count("Trying again") == 2
    for f in facts:
        assert f in out
    assert not (jobdir / "reports" / "wsl_probe" / "probe_summary.json").exists()


def test_a_missing_wsl_exe_is_not_tried_again(jobdir, capsys, waited):
    gone = result(started=False, returncode=None, output="cannot start 'wsl': [WinError 2] The system cannot find the file specified", seconds=0.0)
    with pytest.raises(SystemExit, match="did not answer"):
        probe.main(["--runner", "native"], exec_fn=Scripted([gone]), host_reader=host_reader([5.0]), snapshot_fn=lambda: [])
    out = capsys.readouterr().out
    assert waited == [] and "try 1 of 3" in out and "cannot start 'wsl'" in out and "try 2" not in out


def test_a_warning_printed_above_the_uptime_is_not_a_failure(jobdir, capsys, waited):
    warning = "wsl: Failed to start the systemd user session for 'pankaja'. See journalctl for more details."
    fake = Scripted([up(warning + "\r\n5.0 9.0\r\n"), up("25.0 40.0\n")])
    assert probe.main(["--runner", "native"], exec_fn=fake, clock=CLOCK(), host_reader=host_reader([5.0])) == 0
    out = capsys.readouterr().out
    assert waited == [] and "Ubuntu is up (uptime 5.0 s)." in out and f"(wsl.exe also printed: {warning})" in out and "did not answer" not in out


def test_no_answer_after_the_run_shows_what_wsl_printed_and_the_facts(jobdir, capsys):
    gone = result(returncode=1, output="The Windows Subsystem for Linux instance has terminated.\r\n")
    fake = Scripted([up("4.6 10.0\n"), gone], log=GOOD[:GOOD.index("10:00:06 PHASE B_end")])
    probe.main(["--runner", "native"], exec_fn=fake, clock=CLOCK(), host_reader=host_reader([2.6]),
               snapshot_fn=lambda: ["WSL: WSL version: 2.4.13.0", "Free disk space: C: 40.0 GB"])
    out = capsys.readouterr().out
    assert "Asked again after the run, wsl.exe printed (exit code 1): The Windows Subsystem for Linux instance has terminated." in out
    assert "WSL: WSL version: 2.4.13.0" in out and "Free disk space: C: 40.0 GB" in out and "after no answer" in out
    assert json.loads((jobdir / "reports" / "wsl_probe" / "probe_summary.json").read_text())["wsl_facts"][0] == "WSL: WSL version: 2.4.13.0"


def test_a_probe_that_never_started_also_shows_the_wsl_facts(jobdir, capsys):
    fake = NeverStarts(uptimes=("8.2 9.0\n", "8.9 9.0\n"))                           # Ubuntu answers, but the probe script dies at once
    probe.main(["--runner", "native"], exec_fn=fake, clock=CLOCK(), host_reader=host_reader([5.0]), snapshot_fn=lambda: ["Distros: * Ubuntu Running 2"])
    out = capsys.readouterr().out
    assert "The probe never started in Ubuntu" in out and "Distros: * Ubuntu Running 2" in out and "Asked again after the run" not in out


def test_a_healthy_run_does_not_ask_for_the_wsl_facts(jobdir):
    def must_not_be_called():
        raise AssertionError("the WSL facts are only for a probe that cannot reach Ubuntu")
    assert probe.main(["--runner", "native"], exec_fn=FakeUbuntu(), clock=CLOCK(), host_reader=host_reader([5.0]), snapshot_fn=must_not_be_called) == 0


def test_without_a_given_function_the_facts_come_from_wsl_snapshot(jobdir, capsys, monkeypatch):
    monkeypatch.setattr(wp, "read_disk_free", lambda *a, **k: {"C:": 2.5, "E:": 90.0})
    with pytest.raises(SystemExit):
        probe.main(["--runner", "native"], exec_fn=Scripted([DOWN]), host_reader=host_reader([5.0]))
    out = capsys.readouterr().out
    assert "Free disk space: C: 2.5 GB, E: 90.0 GB" in out and "C: is almost full (under 5 GB)" in out


class WslExe:
    """The two read-only wsl.exe commands the facts use, with the text a real WSL prints (the runner has already turned its UTF-16 into text)."""
    def __init__(self, version="WSL version: 2.4.13.0\r\nKernel version: 5.15.167.4-1\r\nWSLg version: 1.0.65\r\n",
                 distros="  NAME      STATE           VERSION\r\n* Ubuntu    Stopped         2\r\n"):
        self.out, self.cmds = {("wsl", "--version"): version, ("wsl", "-l", "-v"): distros}, []

    def __call__(self, cmd, timeout_s, heartbeat_s, say):
        self.cmds.append(cmd)
        return result(output=self.out[tuple(cmd)])


def test_wsl_facts_show_version_distro_state_and_disk_space():
    fake = WslExe()
    lines = wp.wsl_snapshot("wsl", fake, disks={"C:": 41.2, "E:": 120.5})
    assert fake.cmds == [["wsl", "--version"], ["wsl", "-l", "-v"]]
    assert lines == ["WSL: WSL version: 2.4.13.0 / Kernel version: 5.15.167.4-1",                       # the first two lines only
                     "Distros: NAME STATE VERSION / * Ubuntu Stopped 2",
                     "Free disk space: C: 41.2 GB, E: 120.5 GB"]


def test_wsl_facts_warn_about_a_nearly_full_drive():
    one = wp.wsl_snapshot("wsl", WslExe(), disks={"C:": 3.1, "E:": 120.5})
    assert one[-1].startswith("C: is almost full (under 5 GB)") and "swap file" in one[-1] and len(one) == 4
    both = wp.wsl_snapshot("native", WslExe(), disks={"C:": 1.0, "E:": 2.0})
    assert both[-1].startswith("C: and E: are almost full")
    assert len(wp.wsl_snapshot("native", WslExe(), disks={"C:": 5.0})) == 1                     # exactly 5 GB is not "almost full"


def test_wsl_facts_for_the_native_runner_do_not_call_wsl_exe():
    fake = WslExe()
    assert wp.wsl_snapshot("native", fake, disks={"/": 50.0}) == ["Free disk space: / 50.0 GB"] and fake.cmds == []
    assert wp.wsl_snapshot("native", fake, disks={}) == []


def test_wsl_facts_say_when_wsl_does_not_answer():
    def silent(cmd, *a):
        return result(started=False, returncode=None, output="")
    def refuses(cmd, *a):
        return result(started=False, returncode=None, output="cannot start 'wsl': nope")
    assert wp.wsl_snapshot("wsl", silent, disks={}) == ["WSL: no answer", "Distros: no answer"]
    assert wp.wsl_snapshot("wsl", refuses, disks={})[0] == "WSL: cannot start 'wsl': nope"


def test_wsl_facts_read_the_disks_themselves_when_none_are_given(monkeypatch):
    monkeypatch.setattr(wp, "read_disk_free", lambda *a, **k: {"C:": 9.0})
    assert wp.wsl_snapshot("native", WslExe()) == ["Free disk space: C: 9.0 GB"]


def test_disk_reader_reads_real_drives_and_skips_the_ones_that_do_not_exist():
    here = str(Path.cwd().anchor or "/")
    got = REAL_DISKS([here, "Z:\\no\\such\\drive\\folder"])
    assert len(got) == 1 and list(got.values())[0] >= 0 and isinstance(list(got)[0], str)
    assert REAL_DISKS(["Z:\\no\\such\\drive\\folder"]) == {}
    assert isinstance(REAL_DISKS(), dict)                                                       # the default pair (C: and this drive) never raises


def test_disk_reader_looks_at_c_and_at_this_drive_by_default():
    asked = []

    def usage(path):
        asked.append(path)
        return type("U", (), {"free": 5 * wp.GB})()
    got = REAL_DISKS(usage=usage)
    assert asked == ["C:\\", str(Path.cwd().anchor or "/")] and got and all(v == 5.0 for v in got.values())
