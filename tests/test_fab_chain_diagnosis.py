"""The chain names the likely root cause of a run that did not pass (rules only, no LLM), and never for a run that passed."""
import json

import x2c.commands.chain as chain
from tests.test_fab_chain import FakeUbuntu, proj, run_main  # noqa: F401  (proj is a fixture)
from tests.test_fab_memory import Reader


def test_a_memory_block_is_named_with_its_fix(proj, capsys):
    rc = chain.main(["--runner", "wsl", "--op", "add8_sat"], exec_fn=FakeUbuntu(), host_reader=Reader(0.6))
    out = capsys.readouterr().out
    assert rc == 1 and "likely cause [windows_memory_low]: Windows had too little commit headroom" in out and "fix: close browsers" in out
    assert "from the evidence rules, no LLM" in out and "x2c agent a3 --llm groq" in out
    rep = json.loads(chain.REPORT.read_text())
    assert rep["results"][0]["diagnosis"] == {"rule_based_candidates": ["windows_memory_low"], "source": "rules (no LLM)"}


def test_a_failed_flow_is_named_after_the_chain_result(proj, capsys):
    assert run_main(["--op", "add8_sat"], FakeUbuntu("no_bin")) == 1                       # "FABulous exited 0 but wrote no bitstream"
    out = capsys.readouterr().out
    assert "likely cause [flow_tool_error]: FABulous, Yosys or nextpnr stopped with an error during the flow" in out and "flow_console.log" in out
    assert "CHAIN RESULT add8_sat: FAIL" in out and out.index("CHAIN RESULT") < out.index("likely cause")


def test_diagnose_function_edges(proj, monkeypatch):
    assert chain.diagnose("add8_sat", {"op": "add8_sat", "verdict": "PASS", "stages": []}) == []
    assert chain.diagnose("add8_sat", {"op": "add8_sat", "verdict": "PASS (partial: rtl not re-run)", "stages": []}) == []
    r = {"op": "add8_sat", "verdict": "FAIL", "stages": [{"stage": "check", "status": "fail", "detail": "flow check wrote no report"}]}
    lines = chain.diagnose("add8_sat", r)
    assert len(lines) == 1 and "not one of the known causes" in lines[0] and r["diagnosis"]["rule_based_candidates"] == []
    r = {"op": "add8_sat", "verdict": "NOT RUN", "stages": [{"stage": "rtl", "status": "blocked", "detail": "GROQ_API_KEY is not set in this window"}]}
    assert "[llm_key_missing]" in chain.diagnose("add8_sat", r)[0]

    def boom(*a, **k):
        raise RuntimeError("broken")
    monkeypatch.setattr(chain.chain_debug, "collect_evidence", boom)                       # a broken diagnosis never breaks the chain
    r = {"op": "add8_sat", "verdict": "FAIL", "stages": []}
    assert chain.diagnose("add8_sat", r) == [] and "RuntimeError: broken" in r["diagnosis"]["error"]


def test_a_passing_chain_prints_no_cause(proj, capsys):
    import sys
    if sys.platform == "win32":
        return                                                                             # the fake Ubuntu needs Linux paths for a full pass
    assert run_main(["--op", "add8_sat"], FakeUbuntu()) == 0
    out = capsys.readouterr().out
    assert "likely cause" not in out and "diagnosis" not in json.loads(chain.REPORT.read_text())["results"][0]
