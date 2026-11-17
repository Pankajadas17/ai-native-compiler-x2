import json
from verification import tools

def test_run_suite_healthy(tmp_path):
    rp = str(tmp_path / "r.json")
    r = tools.run_suite(report_path=rp)
    assert r["total"] == 25 and r["passed"] == 25 and r["failed"] == []
    json.dumps(r)                                   # must be JSON-serialisable for an LLM

def test_run_case_named_and_custom():
    assert tools.run_case("fixed_2x2")["passed"] is True
    r = tools.run_case(A=[[1, 2]], B=[[5, 6], [7, 8]])
    assert r["passed"] and r["actual"] == [[19.0, 22.0]]
    assert "known_cases" in tools.run_case("nope")

def test_summarize_and_get_failures_healthy(tmp_path):
    rp = str(tmp_path / "r.json"); tools.run_suite(report_path=rp)
    assert tools.summarize_report(rp)["failed"] == 0
    assert tools.get_failures(rp)["failures"] == []

def test_call_tool_entry_point_is_safe():
    assert "error" in tools.call_tool("rm_rf", {})
    assert "error" in tools.call_tool("run_case", {"bogus": 1})
    assert tools.call_tool("run_case", {"name": "fixed_2x2"})["passed"]

def test_schemas_match_tools():
    assert {s["name"] for s in tools.TOOL_SCHEMAS} == set(tools.TOOLS)
