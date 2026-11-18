"""Step 17: the target-mapping agent (A1). The LLM proposes; compiler.partition.validate_partition alone decides."""
import copy
import json

import pytest

from x2c.agents import llm_client as lc
from x2c.agents.llm_client import Action, MockMappingLLM
from x2c.agents.mapping_agent import _Session, run_mapping_agent
from x2c.compiler.ir_builder import build_ir_from_chain_workload, build_ir_from_compiler_graph
from x2c.compiler.partition import (allowed_targets, load_hardware, plan_from_targets, propose_partition_baseline,
                                search_space_size, validate_partition)
from x2c.inputs.bmsemi_source import ReadOnlyRepo
from x2c.inputs.normalize import compiler_v2_workload, deepgate_workload


@pytest.fixture(scope="module")
def repo():
    return ReadOnlyRepo()


@pytest.fixture(scope="module")
def ir(repo):
    return build_ir_from_compiler_graph(repo, compiler_v2_workload(repo))


@pytest.fixture(scope="module")
def hw():
    return load_hardware()


@pytest.fixture
def log(tmp_path):
    return str(tmp_path / "log.jsonl")


class Scripted:
    """Fake LLM: first asks for the graph, then proposes whatever `make(targets_by_node)` returns."""
    def __init__(self, make):
        self.make = make

    def start(self, goal, schemas, system_prompt=None):
        return Action("tool_calls", calls=[("describe_graph", {})])

    def send_tool_results(self, results, force_final=False):
        name, res = results[0]
        if name == "describe_graph":
            allowed = {n["name"]: n["allowed_targets"][0] for n in res["nodes"]}
            return Action("tool_calls", calls=[("propose_assignment", {"targets": self.make(allowed)})])
        return Action("final", text="done")


# ------------------------------------------------------------ helpers
def test_allowed_targets_and_search_space(ir):
    by_op = {}
    for n in ir.nodes:
        by_op.setdefault(n.op, set()).add(tuple(allowed_targets(n)))
    assert by_op["input"] == {("io",)} and by_op["output"] == {("io",)}
    assert by_op["conv2d"] == {("x2",)} and by_op["add"] == {("efpga",)}
    assert search_space_size(ir) == 1


def test_plan_from_targets_equals_the_baseline(ir, hw):
    base = propose_partition_baseline(ir, hw)
    plan = plan_from_targets(ir, hw, {k: v["target"] for k, v in base["assignments"].items()})
    assert plan["totals"] == base["totals"]
    assert validate_partition(ir, plan, hw) == []


# ------------------------------------------------------------ validator: new checks, never crashes
def test_compute_node_cannot_be_io(ir, hw):
    base = propose_partition_baseline(ir, hw)
    t = {k: v["target"] for k, v in base["assignments"].items()}
    t[next(n.name for n in ir.nodes if n.op == "add")] = "io"
    errs = validate_partition(ir, plan_from_targets(ir, hw, t), hw)
    assert any("only input and output nodes may be 'io'" in e for e in errs)


def test_tampered_totals_are_caught(ir, hw):
    plan = copy.deepcopy(propose_partition_baseline(ir, hw))
    plan["totals"]["x2_nodes"] += 1
    plan["totals"]["boundary_edges"] += 1
    errs = validate_partition(ir, plan, hw)
    assert any("x2_nodes" in e for e in errs) and any("boundary_edges" in e for e in errs)


@pytest.mark.parametrize("bad", [None, [], "x", 3, {}, {"assignments": None}, {"assignments": [1]},
                                 {"assignments": {"a": 5}}, {"assignments": {"a": None}, "totals": 4}])
def test_malformed_plans_are_errors_not_exceptions(ir, hw, bad):
    errs = validate_partition(ir, bad, hw)
    assert isinstance(errs, list) and errs


# ------------------------------------------------------------ the agent loop
def test_mock_agent_is_validated_and_matches_baseline(ir, hw, log):
    out = run_mapping_agent(MockMappingLLM(), ir, hw, log_path=log)
    assert out["final_source"] == "llm_validated" and out["llm_plan_legal"] and out["matches_baseline"]
    assert out["stopped"] == "finished" and out["search_space"]["legal_plans"] == 1
    assert "NOT silicon" in out["evidence_level"]
    t = out["final_plan"]["totals"]
    assert (t["x2_nodes"], t["efpga_nodes"], t["x2_tiles"], t["boundary_edges"], t["boundary_bytes"]) == (17, 32, 24, 32, 2480)


def test_audit_log_is_written(ir, hw, log):
    run_mapping_agent(MockMappingLLM(), ir, hw, log_path=log)
    rows = [json.loads(l) for l in open(log, encoding="utf-8")]
    assert [r["tool"] for r in rows] == ["describe_graph", "propose_assignment"] and rows[1]["legal"] is True


def test_illegal_llm_plan_falls_back_to_the_baseline(ir, hw, log):
    def everything_on_x2(allowed):
        return {k: "x2" for k in allowed}
    out = run_mapping_agent(Scripted(everything_on_x2), ir, hw, log_path=log)
    assert not out["llm_plan_legal"] and out["llm_errors"]
    assert out["final_source"].startswith("baseline_fallback") and "rejected" in out["final_source"]
    assert out["final_plan"]["totals"]["x2_nodes"] == 17


def test_no_proposal_falls_back(ir, hw, log):
    class Silent:
        def start(self, *a, **k):
            return Action("final", text="nothing to do")
    out = run_mapping_agent(Silent(), ir, hw, log_path=log)
    assert out["final_source"].startswith("baseline_fallback") and "no proposal" in out["final_source"]


def test_llm_unavailable_falls_back(ir, hw, log):
    class Down:
        def start(self, *a, **k):
            raise lc.LLMUnavailable("all models down")
    out = run_mapping_agent(Down(), ir, hw, log_path=log)
    assert out["stopped"] == "llm_unavailable" and out["final_source"].startswith("baseline_fallback")


def test_provider_429_is_survivable_but_bugs_and_auth_raise(ir, hw, log):
    class Err(Exception):
        def __init__(self, code):
            super().__init__("x")
            self.status_code = code

    def failing(exc):
        class L:
            def start(self, *a, **k):
                raise exc
        return L()

    out = run_mapping_agent(failing(Err(429)), ir, hw, log_path=log)
    assert out["stopped"] == "llm_error" and out["final_plan"] is not None
    with pytest.raises(Err):
        run_mapping_agent(failing(Err(401)), ir, hw, log_path=log)
    with pytest.raises(ZeroDivisionError):
        run_mapping_agent(failing(ZeroDivisionError()), ir, hw, log_path=log)


def test_session_rejects_unknown_tools_and_arguments(ir, hw):
    s = _Session(ir, hw)
    assert "unknown tool" in s.call("rm_rf", {})["error"]
    assert "unknown arguments" in s.call("describe_graph", {"path": "/etc/passwd"})["error"]
    assert "error" in s.call("propose_assignment", {"targets": "x2"})


def test_garbage_targets_never_crash_the_session(ir, hw):
    s = _Session(ir, hw)
    for bad in ({"conv": None}, {"conv": 5}, {"conv": {"a": 1}}, {}):
        res = s.call("propose_assignment", {"targets": bad})
        assert res.get("legal") is False or "error" in res


def test_deepgate_chain_end_to_end(repo, hw, log):
    g = build_ir_from_chain_workload(deepgate_workload(repo))
    out = run_mapping_agent(MockMappingLLM(), g, hw, log_path=log)
    t = out["final_plan"]["totals"]
    assert out["final_source"] == "llm_validated" and t["efpga_nodes"] == 0 and t["boundary_edges"] == 0
