"""Step 26: the chain looks at Windows' free memory (wsl runner only) before the Ubuntu stages, and says whether memory was short when a simulation
ends without a result. The memory numbers here are FAKE readings; what your machine really has is what the chain and `x2c doctor wsl` print."""
import sys
import threading
import time

import pytest

import x2c.commands.chain as chain
from tests.test_fab_chain import FakeUbuntu, PASS_LINE, cfg, proj, result   # noqa: F401  (proj is a fixture)
from x2c.verification import wsl_probe as wp
from x2c.verification.fab_sim import make_fab_testbench

FAIL_LINE = "FAB_SIM RESULT: FAIL vectors=500 numpy_mismatches=3 rtl_mismatches=3\n"


def reading(r):
    """A float means 'this much free RAM and the same commit headroom'; a dict is used as it is; None means the reading failed."""
    if r is None or isinstance(r, dict):
        return r
    return {"total_gb": 7.8, "free_gb": r, "commit_headroom_gb": r}


class Reader:
    """A fake Windows memory reader: the first call gives the first value, the next call the next one ... the last value repeats. Counts its calls."""
    def __init__(self, *values):
        self.values, self.calls = list(values), 0

    def __call__(self):
        v = self.values[min(self.calls, len(self.values) - 1)]
        self.calls += 1
        return reading(v)


class SimFake:
    """Stands in for Ubuntu running run_fab_sim.sh: takes a moment (so the Windows sampler runs), then writes sim_console.log (or does not)."""
    def __init__(self, log=PASS_LINE, started=True):
        self.log, self.started, self.cmds = log, started, []

    def __call__(self, cmd, timeout_s, heartbeat_s, say):
        self.cmds.append(cmd)
        time.sleep(0.1)
        if self.log is not None:
            (chain.JOBS / "add8_sat" / "sim_console.log").write_text(self.log)
        return result(started=self.started, output="" if self.started else "cannot start 'wsl'")


@pytest.fixture
def job(proj):
    d = proj / "reports" / "fab_jobs" / "add8_sat"
    d.mkdir(parents=True)
    (d / "add8_sat_fab.bin").write_bytes(b"\x01\x02")
    (d / "add8_sat_fab_tb.v").write_text(make_fab_testbench("add8_sat"))
    return d


def sim_stage(reader, fake, **over):
    c = cfg(check_memory=True, host_reader=reader, sample_s=0.02, **over)
    return chain.build_steps("add8_sat", c, exec_fn=fake, preflight={})["sim"]()


# ------------------------------------------------------------ the sim stage
def test_sim_is_not_started_when_windows_has_almost_no_memory(job):
    fake = SimFake()
    res = sim_stage(Reader(0.6), fake)
    assert res[0] == "blocked" and "only 0.6 GB of commit headroom" in res[1] and "wsl --shutdown" in res[1] and "--ignore-memory" in res[1]
    assert fake.cmds == [] and not (job / "sim_console.log").exists()                 # Ubuntu was never asked


def test_commit_headroom_can_block_the_sim_too(job):
    res = sim_stage(Reader({"total_gb": 7.8, "free_gb": 3.0, "commit_headroom_gb": 0.5}), SimFake())
    assert res[0] == "blocked" and "only 0.5 GB of commit headroom" in res[1]


def test_a_tight_but_workable_memory_runs_the_sim_and_a_pass_stays_quiet(job):
    reader, fake = Reader(3.0), SimFake()
    res = sim_stage(reader, fake)
    assert res[0] == "pass" and "Windows" not in res[1] and len(fake.cmds) == 1
    assert res[2]["windows_memory"]["min_free_gb"] == 3.0 and res[2]["windows_memory"]["samples"] >= 2          # kept in the report


def test_a_dead_simulation_is_blamed_on_memory_only_when_windows_ran_low(job):
    res = sim_stage(Reader(3.0, 0.3), SimFake(log="INFO | starting\n"))                 # Windows: 3.0 GB at the start, 0.3 GB while Ubuntu ran
    assert res[0] == "unverified" and "no FAB_SIM RESULT line" in res[1]
    assert "fell to 0.3 GB" in res[1] and "most likely stopped for lack of memory" in res[1] and "--op add8_sat --from-stage sim" in res[1]
    assert res[2]["windows_memory"]["min_free_gb"] == 0.3


class DuringRunLow:
    """Windows memory as the main thread sees it: 3.0 GB before the run, 4.0 GB after it (the VM is gone). Only the background sampler, which reads
    while Ubuntu runs, sees 0.3 GB."""
    def __init__(self):
        self.main_calls = 0

    def __call__(self):
        if threading.current_thread() is threading.main_thread():
            self.main_calls += 1
            return reading(3.0 if self.main_calls == 1 else 4.0)
        return reading(0.3)


def test_the_lowest_reading_during_the_run_counts_even_if_memory_recovers_afterwards(job):
    res = sim_stage(DuringRunLow(), SimFake(log=""))
    assert res[0] == "unverified" and "commit headroom fell to 0.3 GB" in res[1] and res[2]["windows_memory"]["min_free_gb"] == 0.3


def test_a_dead_simulation_with_plenty_of_memory_does_not_blame_memory(job):
    res = sim_stage(Reader(6.0), SimFake(log=""))
    assert res[0] == "unverified" and "probably not the reason" in res[1] and "most likely" not in res[1] and "wsl --shutdown" not in res[1]


def test_a_missing_log_also_gets_the_memory_sentence(job):
    res = sim_stage(Reader(3.0, 0.3), SimFake(log=None))
    assert res[0] == "fail" and "no sim_console.log was written" in res[1] and "fell to 0.3 GB" in res[1]


def test_a_real_mismatch_is_never_blamed_on_memory(job):
    res = sim_stage(Reader(3.0, 0.3), SimFake(log=FAIL_LINE))
    assert res[0] == "fail" and "numpy mismatch" in res[1] and "Windows" not in res[1]


def test_wsl_that_cannot_start_stays_blocked_without_a_memory_story(job):
    res = sim_stage(Reader(3.0, 0.3), SimFake(started=False))
    assert res[0] == "blocked" and res[1] == "cannot start 'wsl'"


def test_a_reading_that_fails_never_stops_the_simulation(job):
    reader, fake = Reader(None), SimFake(log="")
    res = sim_stage(reader, fake)
    assert len(fake.cmds) == 1 and res[0] == "unverified" and "Windows" not in res[1] and len(res) == 3 and res[2] is None


def test_the_memory_check_is_off_unless_asked_for(job):
    for c in (cfg(), cfg(check_memory=False, host_reader=Reader(0.1))):                # the native runner / --ignore-memory
        reader = Reader(0.1)
        c.host_reader = reader
        fake = SimFake()
        res = chain.build_steps("add8_sat", c, exec_fn=fake, preflight={})["sim"]()
        assert res[0] == "pass" and reader.calls == 0 and len(fake.cmds) == 1


# ------------------------------------------------------------ the env stage
def env_stage(reader, fake, preflight=None, **over):
    c = cfg(check_memory=True, host_reader=reader, **over)
    pre = {} if preflight is None else preflight
    return chain.build_steps("add8_sat", c, exec_fn=fake, preflight=pre)["env"](), pre


def test_env_is_blocked_before_touching_ubuntu_when_memory_is_very_low(proj):
    fake = FakeUbuntu()
    (res, pre) = env_stage(Reader(0.6), fake)
    assert res[0] == "blocked" and "only 0.6 GB of commit headroom" in res[1] and "--ignore-memory" in res[1]
    assert fake.cmds == [] and "info" not in pre                                       # nothing was asked of Ubuntu and nothing is remembered


def test_env_warns_and_still_passes_when_memory_is_tight(proj):
    fake = FakeUbuntu()
    (res, pre) = env_stage(Reader(2.0), fake)
    assert res[0] == "pass" and "Ubuntu has FABulous" in res[1] and "WARNING: Windows has 2.0 GB of commit headroom" in res[1] and res[2]["tool_versions"]["yosys"]
    assert pre["memory"].startswith("Windows has 2.0 GB")


def test_env_with_enough_memory_has_no_warning(proj):
    (res, pre) = env_stage(Reader(6.0), FakeUbuntu())
    assert res[0] == "pass" and "WARNING" not in res[1] and pre["memory"] == ""


def test_env_failure_mentions_memory_when_it_was_tight(proj):
    (res, _) = env_stage(Reader(2.0), FakeUbuntu(preflight=""))                         # Ubuntu said nothing at all
    assert res[0] == "blocked" and "wsl -l -v" in res[1] and "Windows has 2.0 GB of commit headroom" in res[1]


def test_two_operators_read_memory_once_for_the_shared_preflight(proj):
    fake, pre, reader = FakeUbuntu(), {}, Reader(2.0)
    c = cfg(check_memory=True, host_reader=reader)
    chain.build_steps("add8_sat", c, exec_fn=fake, preflight=pre)["env"]()
    chain.build_steps("relu8", c, exec_fn=fake, preflight=pre)["env"]()
    assert reader.calls == 1 and len(fake.cmds) == 1


# ------------------------------------------------------------ the whole command
def test_main_with_the_wsl_runner_stops_at_env_when_memory_is_very_low(proj, capsys):
    fake, reader = FakeUbuntu(), Reader(0.6)
    rc = chain.main(["--runner", "wsl", "--op", "add8_sat"], exec_fn=fake, host_reader=reader)
    out = capsys.readouterr().out
    assert rc == 1 and fake.cmds == [] and "BLOCKED" in out and "only 0.6 GB of commit headroom" in out and "CHAIN RESULT add8_sat: NOT RUN" in out


@pytest.mark.skipif(sys.platform == "win32", reason="the fake Ubuntu writes next to the script path, which is a /mnt/... path for the wsl runner on Windows")
def test_main_ignore_memory_never_reads_memory_and_the_chain_passes(proj, capsys):
    reader = Reader(0.1)
    assert chain.main(["--runner", "wsl", "--ignore-memory", "--op", "add8_sat"], exec_fn=FakeUbuntu(), host_reader=reader) == 0
    assert reader.calls == 0 and "CHAIN RESULT add8_sat: PASS" in capsys.readouterr().out


@pytest.mark.skipif(sys.platform == "win32", reason="the fake Ubuntu writes next to the script path, which is a /mnt/... path for the wsl runner on Windows")
def test_main_with_enough_memory_passes_and_records_the_readings(proj, capsys):
    import json
    assert chain.main(["--runner", "wsl", "--op", "add8_sat"], exec_fn=FakeUbuntu(), host_reader=Reader(6.0)) == 0
    out = capsys.readouterr().out
    assert "CHAIN RESULT add8_sat: PASS" in out and "WARNING" not in out
    rep = json.loads((proj / "reports" / "fab_chain_report.json").read_text())
    assert rep["results"][0]["facts"]["windows_memory"]["min_free_gb"] == 6.0


def test_the_native_runner_never_checks_memory(proj, capsys):
    reader = Reader(0.1)
    assert chain.main(["--runner", "native", "--op", "add8_sat"], exec_fn=FakeUbuntu(), host_reader=reader) == 0
    assert reader.calls == 0


def test_the_plan_says_when_memory_is_checked(proj):
    on = "\n".join(chain.plan_text("add8_sat", cfg(runner="wsl", check_memory=True)))
    assert "free memory is checked before env and sim" in on and "below 1 GB" in on and "--ignore-memory" in on
    assert "free memory is checked" not in "\n".join(chain.plan_text("add8_sat", cfg(runner="wsl")))


# ------------------------------------------------------------ the check applies to Ubuntu's FIRST use in a run
def test_sim_is_not_blocked_when_an_earlier_stage_of_this_run_already_used_ubuntu(job):
    """Found on a real run: right after a passing flow stage the headroom read 2.0 GB (Ubuntu's VM was holding its memory) and the simulation
    was blocked, although it passed when run. The low reading only means 'do not start' while Ubuntu has not been used yet."""
    fake = SimFake()
    steps = chain.build_steps("add8_sat", cfg(check_memory=True, host_reader=Reader(0.6), sample_s=0.02), exec_fn=fake, preflight={})
    steps["flow"]()                                                     # an earlier stage of the same run ran in Ubuntu
    assert len(fake.cmds) == 1
    res = steps["sim"]()
    assert res[0] == "pass" and len(fake.cmds) == 2                     # not blocked: the simulation ran
    fresh = chain.build_steps("add8_sat", cfg(check_memory=True, host_reader=Reader(0.6), sample_s=0.02), exec_fn=SimFake(), preflight={})
    assert fresh["sim"]()[0] == "blocked"                               # a run that starts at sim is still checked
