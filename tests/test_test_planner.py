import numpy as np
import pytest
from x2c.agents.llm_client import Action, MockPlannerLLM
from x2c.agents.test_agent import plan_tests
from x2c.agents.verification_agent import run_agent
from x2c.commands.bmsemi_verify import run_all
from x2c.verification import test_planner as tp, tools
from tests import faults


@pytest.fixture(autouse=True)
def _clean():
    tools.reset_generated(); yield; tools.reset_generated(); tools.set_x2_backend(None)


def good(**kw):
    s = {"name": "t1", "m": 2, "k": 3, "n": 2, "a_pattern": "random", "b_pattern": "random", "seed": 1}
    s.update(kw); return s


# ---------------- validation: a spec can never carry an answer key ----------------
@pytest.mark.parametrize("bad", [
    good(expected=[[1, 2], [3, 4]]), good(m=0), good(k=13), good(a_pattern="evil"), good(name="has space"),
    good(name=""), good(lo=5, hi=1), good(hi=300), good(lo=-300), "not a dict", {"name": "x"}])
def test_bad_specs_rejected(bad):
    spec, err = tp.validate_spec(bad)
    assert spec is None and err


def test_plan_validation_limits_and_duplicates():
    acc, rej = tp.validate_plan([good(name="a"), good(name="a"), good(name="b", m=99)])
    assert [s["name"] for s in acc] == ["a"] and len(rej) == 2
    acc, rej = tp.validate_plan([good(name=f"n{i}") for i in range(tp.MAX_SPECS + 5)])
    assert len(acc) == tp.MAX_SPECS and any("first" in r["reason"] for r in rej)
    assert tp.validate_plan("nope")[0] == []


def test_materialize_is_deterministic_and_seed_sensitive():
    s = tp.validate_spec(good(m=4, k=4, n=4))[0]
    A1, B1 = tp.materialize(s); A2, B2 = tp.materialize(s)
    assert np.array_equal(A1, A2) and np.array_equal(B1, B2)
    A3, _ = tp.materialize(dict(s, seed=2))
    assert not np.array_equal(A1, A3)


# ---------------- coverage is measured on the materialised matrices ----------------
def test_fixed_suite_has_blind_spots_and_baseline_closes_them():
    missing = set(tp.coverage_gaps()["missing"])
    assert {"extreme_values", "int8_min", "long_k"} <= missing          # the fixed suite never goes there
    plan, rej = tp.validate_plan(tp.baseline_plan())
    assert not rej and tp.coverage_gaps(plan)["missing"] == []


def test_classify_examples():
    A = np.array([[-128.0, 127.0], [0.0, 0.0]]); B = np.eye(2)
    c = tp.classify(A, B)
    assert {"int8_min", "extreme_values", "identity_like", "square"} <= c
    assert "long_k" in tp.classify(np.ones((2, 9)), np.ones((9, 2)))
    assert "all_zero_operand" in tp.classify(np.zeros((2, 2)), np.ones((2, 2)))


# ---------------- the oracle always decides ----------------
def test_healthy_generated_run_passes_and_expected_comes_from_the_oracle():
    out = tools.run_generated_tests(tp.baseline_plan())
    assert out["failed"] == [] and out["passed"] == out["total"] and out["coverage_still_missing"] == []
    import json
    rep = json.load(open(out["report_path"]))
    for r in rep["results"]:
        A, B = tp.materialize(r["spec"])
        assert np.allclose(r["expected"], A @ B)                       # expected == independent NumPy result


@pytest.mark.parametrize("fault", [faults.latent_wrap16, faults.latent_long_k, faults.latent_int8_min_clamp])
def test_latent_bugs_are_invisible_to_fixed_suite_but_caught_by_generated_tests(fault):
    fixed = run_all(x2_fn=fault, report_path="reports/_t_fixed.json")
    assert fixed["passed"] == fixed["total"]                           # the fixed suite CANNOT see these
    gen = tools.run_generated_tests(tp.baseline_plan(), report_path="reports/_t_gen.json", x2_fn=fault)
    assert gen["failed"], "generated tests failed to expose a latent bug"


def test_generated_tests_still_catch_ordinary_bugs():
    gen = tools.run_generated_tests(tp.baseline_plan(), report_path="reports/_t_gen2.json", x2_fn=faults.sign_flip)
    assert len(gen["failed"]) > 0


# ---------------- tool / agent safety ----------------
def test_tool_rejects_answer_keys_bad_paths_and_backend_injection():
    r = tools.call_tool("run_generated_tests", {"specs": [good(expected=[[1]])]})
    assert "error" in r and r["rejected"][0]["reason"].startswith("unknown keys")
    assert "error" in tools.call_tool("run_generated_tests", {"specs": [good()], "report_path": "../x.json"})
    assert "error" in tools.call_tool("run_generated_tests", {"specs": [good()], "x2_fn": "evil"})
    assert tools.call_tool("run_generated_tests", {"specs": [good()]})["total"] == 1


def test_coverage_gaps_tool_updates_after_generation():
    before = tools.call_tool("coverage_gaps", {})
    tools.call_tool("run_generated_tests", {"specs": tp.baseline_plan()})
    after = tools.call_tool("coverage_gaps", {})
    assert before["missing"] and after["missing"] == [] and "spec_fields" in after


def test_planner_agent_with_mock_and_with_a_rogue_llm(tmp_path):
    out = plan_tests(MockPlannerLLM(), log_path=str(tmp_path / "l.jsonl"))
    assert out["num_specs"] >= 8 and out["coverage_still_missing"] == [] and out["stopped"] == "finished"
    class Rogue:
        def start(self, g, s, system_prompt=None): return Action("tool_calls", calls=[("run_suite", {})])
        def send_tool_results(self, r): return Action("tool_calls", calls=[("rm_rf", {})])
    rg = plan_tests(Rogue(), max_steps=2, log_path=str(tmp_path / "r.jsonl"))
    assert rg["num_specs"] == 0 and rg["stopped"] == "step_limit_reached"


def test_verification_agent_reports_generated_test_facts(tmp_path):
    class L:
        def start(self, g, s, system_prompt=None):
            return Action("tool_calls", calls=[("run_generated_tests", {"specs": tp.baseline_plan()})])
        def send_tool_results(self, r): return Action("final", text="done")
    out = run_agent(L(), "verify", log_path=str(tmp_path / "l.jsonl"))
    f = out["verified_generated_test_facts"]
    assert f["total"] >= 8 and f["failed"] == 0


def test_on_the_harness_the_planner_must_look_first_and_close_every_gap(tmp_path):
    """Google's harness with a stand-in model (no key): specs travel as a JSON string; the end-of-turn check sends the
    planner back until it looked at the gaps first and no coverage class is left open."""
    pytest.importorskip("google.antigravity", reason="the agent harness is an optional extra: uv sync --extra agents")
    import json as _json
    from tests.fake_model import FakeModel
    from x2c.agents import harness
    harness.reset_skipped()
    plan = tp.baseline_plan()
    steps = [("run_generated_tests", {"specs": _json.dumps(plan[:1])}), ("say", "Planned."),
             ("coverage_gaps", {}), ("say", "Still planned."),
             ("run_generated_tests", {"specs": _json.dumps(plan)}), ("say", "All classes covered.")]

    def script(messages, request):
        n = sum(1 for m in messages if m.get("role") == "assistant")
        kind, value = steps[min(n, len(steps) - 1)]
        return ("text", value) if kind == "say" else ("tool", kind, value)

    with FakeModel(script) as fm:
        out = plan_tests(harness.Gemini(model="fake", base_url=fm.url), log_path=str(tmp_path / "l.jsonl"))
    told = " ".join(str(m.get("content")) for r in fm.requests for m in r["messages"])
    assert fm.tools_offered() == ["coverage_gaps", "run_generated_tests"]
    assert "Call coverage_gaps first" in told and "still uncovered" in told
    assert out["end_of_turn_sendbacks"] == 2 and out["num_specs"] >= len(plan) and out["coverage_still_missing"] == []
