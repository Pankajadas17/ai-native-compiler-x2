"""Debug agent (A3) for the FABulous chain. The cases use the exact stage messages and log lines of real runs where there was one
(the empty-bitstream failure, the memory block, Ubuntu not answering); the others use the messages the chain itself prints.
The LLMs here are fakes: an honest one, and several that lie, ramble or break, to show that code decides the answer."""
import json

import pytest

import run_chain_debug as cli
from agents.chain_debug_agent import MockChainDebugLLM, run_chain_debug_agent
from agents.llm_client import Action, LLMUnavailable
from verification import chain_debug as cd

REAL_XX_LOG = """Read bitstream hex from build/add8_sat_fab.hex
FAB_SIM MISMATCH vector 0: io_in=0x0008080 expected=0x80 fabric=0xxx
FAB_SIM MISMATCH vector 1: io_in=0x0008081 expected=0x80 fabric=0xxx
FAB_SIM MISMATCH vector 2: io_in=0x00080c0 expected=0x80 fabric=0xxx
FAB_SIM MISMATCH vector 3: io_in=0x00080ff expected=0x80 fabric=0xxx
FAB_SIM MISMATCH vector 4: io_in=0x0008000 expected=0x80 fabric=0xxx
FAB_SIM RESULT: FAIL vectors=500 numpy_mismatches=500 rtl_mismatches=500
"""
MEMORY_BLOCK = ("Windows has only 0.8 GB of free RAM (commit headroom 3.1 GB): too little for Ubuntu's VM, which would most likely stop. "
                "Close browsers and other big programs, run `wsl --shutdown`, then try again. (Ignore this check with --ignore-memory.)")
ORDER = ["rtl", "prep", "env", "flow", "sim", "check"]


def result(bad_stage, status, detail, op="add8_sat", verdict="FAIL"):
    rows = []
    for s in ORDER:
        if ORDER.index(s) < ORDER.index(bad_stage):
            rows.append({"stage": s, "status": "pass", "detail": "ok", "seconds": 1.0})
        elif s == bad_stage:
            rows.append({"stage": s, "status": status, "detail": detail, "seconds": 1.0})
        else:
            rows.append({"stage": s, "status": "not_run", "detail": f"not run: {bad_stage} did not pass", "seconds": 0.0})
    return {"op": op, "verdict": verdict, "stages": rows, "facts": {}}


@pytest.fixture
def jobs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "reports" / "fab_jobs" / "add8_sat"
    d.mkdir(parents=True)
    (d / "add8_sat_fab.bin").write_bytes(b"\x01" * 12024)
    return d


def cands(jobs, r):
    return cd.rule_candidates(cd.collect_evidence("add8_sat", r, jobs.parent))


# ------------------------------------------------------------ evidence and rules
def test_the_real_empty_bitstream_failure_is_recognised(jobs):
    (jobs / "sim_console.log").write_text(REAL_XX_LOG)
    r = result("sim", "fail", "500 numpy mismatch(es), 500 rtl mismatch(es) in 500 vectors")
    ev = cd.collect_evidence("add8_sat", r, jobs.parent)
    assert ev["first_bad_stage"] == "sim" and ev["sim_vectors"] == 500 and ev["sim_numpy_mismatches"] == 500 and ev["sim_rtl_mismatches"] == 500
    assert ev["sim_fabric_outputs_unknown"] is True and ev["sim_fabric_outputs_defined"] is False and ev["job_bitstream_bytes"] == 12024
    assert ev["sim_first_mismatches"][0] == "expected=0x80 fabric=0xxx" and ev["sim_mismatch_lines_shown"] == 5
    assert cd.rule_candidates(ev) == ["bitstream_not_loaded"]


def test_wrong_but_defined_outputs_are_a_different_cause(jobs):
    defined = REAL_XX_LOG.replace("fabric=0xxx", "fabric=0x7f").replace("numpy_mismatches=500 rtl_mismatches=500", "numpy_mismatches=3 rtl_mismatches=3")
    (jobs / "sim_console.log").write_text(defined)
    assert cands(jobs, result("sim", "fail", "3 numpy mismatch(es), 3 rtl mismatch(es) in 500 vectors")) == ["fabric_differs_from_rtl"]
    (jobs / "sim_console.log").write_text(defined.replace("rtl_mismatches=3", "rtl_mismatches=0"))
    assert cands(jobs, result("sim", "fail", "3 numpy mismatch(es), 0 rtl mismatch(es) in 500 vectors")) == ["golden_convention_mismatch"]
    some_x = REAL_XX_LOG.replace("fabric=0xxx", "fabric=0x7f", 2).replace("=500 rtl", "=40 rtl").replace("rtl_mismatches=500", "rtl_mismatches=40")
    (jobs / "sim_console.log").write_text(some_x)                                         # a few vectors, some unknown, some defined: no rule claims it
    assert cands(jobs, result("sim", "fail", "40 numpy mismatch(es), 40 rtl mismatch(es) in 500 vectors")) == []
    (jobs / "sim_console.log").write_text(REAL_XX_LOG.replace("fabric=0xxx", "fabric=0x7f", 2))        # EVERY vector wrong, but only some outputs unknown
    assert cands(jobs, result("sim", "fail", "500 numpy mismatch(es), 500 rtl mismatch(es) in 500 vectors")) == []
    (jobs / "sim_console.log").write_text(REAL_XX_LOG.replace("=500 rtl", "=40 rtl").replace("rtl_mismatches=500", "rtl_mismatches=40"))
    assert cands(jobs, result("sim", "fail", "40 numpy mismatch(es), 40 rtl mismatch(es) in 500 vectors")) == []      # all unknown, but only 40 of 500 vectors


def test_an_empty_bitstream_in_the_job_folder(jobs):
    (jobs / "add8_sat_fab.bin").write_bytes(b"")
    assert cands(jobs, result("sim", "fail", "the bitstream in the job folder is empty (0 bytes): run again with --from-stage flow")) == ["bitstream_not_loaded"]


def test_memory_block_and_dead_simulation(jobs):
    assert cands(jobs, result("sim", "blocked", MEMORY_BLOCK, verdict="NOT VERIFIED")) == ["windows_memory_low"]
    assert cands(jobs, result("env", "blocked", MEMORY_BLOCK, verdict="NOT RUN")) == ["windows_memory_low"]
    (jobs / "sim_console.log").write_text("INFO | Running simulation\n")                  # Ubuntu died: the log has no result line
    assert cands(jobs, result("sim", "unverified", "no FAB_SIM RESULT line: the simulation did not finish or did not compare anything")) == ["ubuntu_died_during_run"]
    (jobs / "sim_console.log").unlink()
    assert cands(jobs, result("sim", "fail", "no sim_console.log was written. Last output: nothing")) == ["ubuntu_died_during_run"]


def test_flow_env_and_rtl_causes(jobs):
    assert cands(jobs, result("flow", "fail", "FABulous could not connect the pins of the wrapper (tools/fab_connect.py refused); see flow_console.log")) == ["pins_not_connected"]
    (jobs / "exit_code.txt").write_text("97\n")
    assert cands(jobs, result("flow", "fail", "something else")) == ["pins_not_connected"]
    (jobs / "exit_code.txt").write_text("1\n")
    assert cands(jobs, result("flow", "fail", "FABulous exited with code 1; see reports/fab_jobs/add8_sat/flow_console.log. Last output: ERROR")) == ["flow_tool_error"]
    assert cands(jobs, result("flow", "fail", "the flow script did not finish (no exit_code.txt). Last output: x")) == ["ubuntu_died_during_run"]
    assert cands(jobs, result("env", "blocked", "Ubuntu did not answer the tool check (no output). Check the distro name with: wsl -l -v")) == ["ubuntu_not_answering"]
    assert cands(jobs, result("rtl", "blocked", "GROQ_API_KEY is not set in this window (set it with Read-Host -AsSecureString) or use --llm mock")) == ["llm_key_missing"]
    assert cands(jobs, result("rtl", "fail", "A2 produced no passing Verilog (3 mismatches)")) == ["rtl_not_verified"]


def test_no_rule_means_no_candidate_and_a_pass_has_nothing_to_explain(jobs):
    assert cands(jobs, result("check", "fail", "flow check wrote no report")) == []
    ok = {"op": "add8_sat", "verdict": "PASS", "stages": [{"stage": s, "status": "pass", "detail": "ok"} for s in ORDER]}
    ev = cd.collect_evidence("add8_sat", ok, jobs.parent)
    assert ev["first_bad_stage"] is None and cd.rule_candidates(ev) == []
    assert cd.collect_evidence("add8_sat", None, jobs.parent)["first_bad_stage"] is None          # no result at all: still no exception


def test_every_cause_has_text_and_says_where_it_comes_from():
    for cid, c in cd.CAUSES.items():
        assert c["suspect"] and c["fix"] and c["seen"], cid
    assert "unknown" in cd.CAUSE_IDS and "unknown" not in cd.CAUSES


# ------------------------------------------------------------ the agent: code decides
class Says:
    """A fake LLM that asks for the evidence once and then answers with fixed text."""
    def __init__(self, text, ask=True):
        self.text, self.ask, self.seen = text, ask, None

    def start(self, goal, tool_schemas, system_prompt=None):
        self.prompt = system_prompt
        return Action("tool_calls", calls=[("chain_evidence", {"op": "add8_sat"})]) if self.ask else Action("final", text=self.text)

    def send_tool_results(self, results, force_final=False):
        self.seen = results[0][1]
        return Action("final", text=self.text)


def answer(cause, **more):
    return json.dumps({"cause_id": cause, "confidence": "high", "explanation": "because", "proposed_fix": "fix", "next_check": "check", **more})


@pytest.fixture
def xx(jobs):
    (jobs / "sim_console.log").write_text(REAL_XX_LOG)
    return result("sim", "fail", "500 numpy mismatch(es), 500 rtl mismatch(es) in 500 vectors"), jobs.parent


def test_mock_llm_is_validated_and_the_fix_is_the_known_one(xx):
    r, jd = xx
    d = run_chain_debug_agent(MockChainDebugLLM(), "add8_sat", r, jd)
    assert d["final_cause"] == "bitstream_not_loaded" and d["final_cause_source"] == "llm_validated" and d["validated"] is True
    assert "empty or wrong bitstream" in d["suspect"] and "--from-stage flow" in d["known_fix"] and "0-byte" in d["seen"] and d["steps"] == 1


def test_a_wrong_llm_claim_is_rejected_and_the_rules_answer(xx):
    r, jd = xx
    d = run_chain_debug_agent(Says(answer("pins_not_connected")), "add8_sat", r, jd)                    # a real cause, but not what the evidence shows
    assert d["final_cause"] == "bitstream_not_loaded" and d["final_cause_source"] == "rules_fallback (llm claim rejected)"
    assert d["validation"] == {"json_ok": True, "cause_known": True, "supported_by_evidence": False} and d["validated"] is False
    d = run_chain_debug_agent(Says(answer("cosmic_rays")), "add8_sat", r, jd)                           # an invented cause
    assert d["final_cause"] == "bitstream_not_loaded" and d["validation"]["cause_known"] is False


def test_unknown_rambling_and_a_list_instead_of_an_object(xx):
    r, jd = xx
    assert run_chain_debug_agent(Says(answer("unknown")), "add8_sat", r, jd)["final_cause_source"] == "rules_fallback (llm said unknown)"
    d = run_chain_debug_agent(Says("I think the fabric is broken, sorry."), "add8_sat", r, jd)
    assert d["final_cause"] == "bitstream_not_loaded" and d["final_cause_source"] == "rules_fallback (no usable llm answer)" and d["raw_answer"].startswith("I think")
    assert run_chain_debug_agent(Says('["bitstream_not_loaded"]'), "add8_sat", r, jd)["validation"]["json_ok"] is False


def test_when_the_rules_do_not_know_the_answer_is_unknown(jobs):
    r = result("check", "fail", "flow check wrote no report")
    d = run_chain_debug_agent(Says(answer("flow_tool_error")), "add8_sat", r, jobs.parent)              # a guess the evidence does not support
    assert d["final_cause"] == "unknown" and d["final_cause_source"] == "unknown_fallback (llm claim rejected)" and d["suspect"] is None
    d = run_chain_debug_agent(Says(answer("unknown")), "add8_sat", r, jobs.parent)
    assert d["final_cause"] == "unknown" and d["final_cause_source"] == "llm_unknown_confirmed_by_rules" and d["validated"] is True


def test_llm_down_or_broken(xx):
    r, jd = xx

    class Down:
        def start(self, *a, **k):
            raise LLMUnavailable("all models are rate-limited")
    d = run_chain_debug_agent(Down(), "add8_sat", r, jd)
    assert d["final_cause"] == "bitstream_not_loaded" and d["final_cause_source"] == "rules_fallback (llm unavailable)" and "rate-limited" in d["llm_error"]

    class Bug:
        def start(self, *a, **k):
            raise KeyError("a programming error must not be hidden")
    with pytest.raises(KeyError):
        run_chain_debug_agent(Bug(), "add8_sat", r, jd)


def test_an_llm_that_never_answers_hits_the_step_limit(xx):
    r, jd = xx

    class Loops:
        calls = 0

        def start(self, goal, tool_schemas, system_prompt=None):
            return Action("tool_calls", calls=[("chain_evidence", {"op": "add8_sat"})])

        def send_tool_results(self, results, force_final=False):
            Loops.calls += 1
            return Action("tool_calls", calls=[("chain_evidence", {"op": "add8_sat"})])
    d = run_chain_debug_agent(Loops(), "add8_sat", r, jd)
    assert d["stopped"] == "step_limit_reached" and Loops.calls == 3 and d["final_cause"] == "bitstream_not_loaded"


def test_the_tool_is_read_only_and_confined_to_this_operator(xx):
    r, jd = xx

    class Pokes(Says):
        def start(self, goal, tool_schemas, system_prompt=None):
            assert [t["name"] for t in tool_schemas] == ["chain_evidence"]
            return Action("tool_calls", calls=[("delete_files", {}), ("chain_evidence", {"op": "relu8"}), ("chain_evidence", {"op": "add8_sat", "path": "C:/"})])

        def send_tool_results(self, results, force_final=False):
            self.all = results
            return Action("final", text=answer("bitstream_not_loaded"))
    llm = Pokes("")
    d = run_chain_debug_agent(llm, "add8_sat", r, jd)
    assert all("error" in res for _, res in llm.all) and d["steps"] == 3 and d["final_cause_source"] == "llm_validated"


def test_blind_mode_hides_the_rules_from_the_llm_but_not_from_the_validation(xx):
    r, jd = xx
    open_llm, blind_llm = Says(answer("bitstream_not_loaded")), Says(answer("bitstream_not_loaded"))
    run_chain_debug_agent(open_llm, "add8_sat", r, jd)
    d = run_chain_debug_agent(blind_llm, "add8_sat", r, jd, blind=True)
    assert open_llm.seen["rule_based_candidates"] == ["bitstream_not_loaded"] and "rule_based_candidates" not in blind_llm.seen
    assert blind_llm.seen["sim_fabric_outputs_unknown"] is True and d["final_cause_source"] == "llm_validated"
    assert "'x' means UNKNOWN" in open_llm.prompt and "bitstream_not_loaded:" in open_llm.prompt and "silicon" in open_llm.prompt


def test_an_llm_that_answers_without_looking_is_still_checked(xx):
    r, jd = xx
    d = run_chain_debug_agent(Says(answer("windows_memory_low"), ask=False), "add8_sat", r, jd)
    assert d["steps"] == 0 and d["final_cause"] == "bitstream_not_loaded" and "rejected" in d["final_cause_source"]


# ------------------------------------------------------------ the command
def write_report(results):
    cd.REPORT.parent.mkdir(parents=True, exist_ok=True)
    cd.REPORT.write_text(json.dumps({"results": results}))


def test_cli_diagnoses_what_did_not_pass(xx, capsys):
    r, _ = xx
    ok = {"op": "relu8", "verdict": "PASS", "stages": []}
    write_report([ok, r])
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert "CHAIN DEBUG  op=add8_sat  chain verdict FAIL  llm=mock (mock: NOT a real model)" in out and "relu8" not in out
    assert "first stage that did not pass: sim (fail)" in out and "first mismatches: expected=0x80 fabric=0xxx" in out
    assert "ROOT CAUSE: bitstream_not_loaded  [llm_validated]" in out and "fix:  compare the bitstream" in out and "basis: real run" in out
    assert "NOT a new verification result" in out
    rep = json.loads(cli.OUT.read_text())
    assert rep["diagnoses"][0]["final_cause"] == "bitstream_not_loaded" and rep["llm"] == "mock"


def test_cli_edges(xx, capsys, monkeypatch):
    r, _ = xx
    with pytest.raises(SystemExit, match="run_fab_chain.py"):
        cli.main([])
    write_report([{"op": "relu8", "verdict": "PASS (partial: rtl not re-run)", "stages": []}])
    assert cli.main([]) == 0 and "nothing to diagnose" in capsys.readouterr().out
    assert cli.main(["--op", "relu8"]) == 0 and "relu8: the chain passed" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="not in the chain report: add8_sat"):
        cli.main(["--op", "add8_sat"])
    write_report([r])
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="GROQ_API_KEY is not set"):
        cli.main(["--llm", "groq"])
    cd.REPORT.write_text("{not json")
    with pytest.raises(SystemExit, match="not readable JSON"):
        cli.main([])


def test_cli_shows_a_rejected_claim_and_an_unknown(xx, capsys, jobs):
    r, _ = xx
    write_report([r])
    cli.main([], llm_factory=lambda name, model: Says(answer("pins_not_connected")))
    out = capsys.readouterr().out
    assert "ROOT CAUSE: bitstream_not_loaded  [rules_fallback (llm claim rejected)]" in out and "llm said: pins_not_connected | rules support: bitstream_not_loaded" in out
    assert "llm explanation" not in out                                                                  # a rejected claim's wording is not shown as an explanation
    write_report([result("check", "fail", "flow check wrote no report")])
    cli.main([], llm_factory=lambda name, model: Says("???"))
    out = capsys.readouterr().out
    assert "ROOT CAUSE: unknown" in out and "does not match any known cause" in out and "llm said: no usable answer | rules support: nothing" in out


def test_load_result_reads_one_operator(xx):
    r, _ = xx
    assert cd.load_result("add8_sat") is None
    write_report([r])
    assert cd.load_result("add8_sat")["verdict"] == "FAIL" and cd.load_result("relu8") is None
