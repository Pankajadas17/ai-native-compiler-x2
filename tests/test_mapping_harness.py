"""A1 (mapping + chip schedule) on OpenAI's Agents SDK with a stand-in for OpenRouter (no key, no BMsemi_Model): the
end-of-turn check sends the agent back until the VALIDATOR accepts its plan and, on the chip, its schedule; the system's
plan and schedule are still decided by code afterwards. Graph: tests/x2_schedule_graphs.mini() with the chip config."""
import json

import pytest

pytest.importorskip("agents", reason="the agent harness is an optional extra: uv sync --extra agents")

from tests.fake_model import FakeModel, calls_so_far  # noqa: E402
from tests.x2_schedule_graphs import mini  # noqa: E402
from x2c.agents import harness as h  # noqa: E402
from x2c.agents.mapping_agent import end_check, _Session, run_mapping_agent  # noqa: E402
from x2c.compiler.partition import has_chip, load_hardware, propose_partition_baseline  # noqa: E402


@pytest.fixture
def g_hw():
    h.reset_skipped()
    g, hw = mini(), load_hardware()
    assert has_chip(hw)
    return g, hw


def entries(g, hw, wrong=None):
    t = {n: v["target"] for n, v in propose_partition_baseline(g, hw)["assignments"].items()}
    if wrong:
        t[wrong] = "efpga"                                   # a matrix op on the fabric: the validator rejects it
    return t


def script_for(steps):
    """Each step: (tool, args) or ('say', text), in order; the last one repeats."""
    def s(messages, request):
        n = sum(1 for m in messages if m.get("role") == "assistant")
        kind, value = steps[min(n, len(steps) - 1)]
        return ("text", value) if kind == "say" else ("tool", kind, value)
    return s


def test_the_check_itself(g_hw):
    g, hw = g_hw
    s = _Session(g, hw)
    assert "describe_graph first" in end_check(s, [])
    assert "Call propose_assignment" in end_check(s, ["describe_graph"])
    s.propose_assignment(entries(g, hw, wrong="c0"))
    assert "validator rejected your last assignment" in end_check(s, ["describe_graph", "propose_assignment"])
    s.propose_assignment(entries(g, hw))
    assert "describe_schedule" in end_check(s, ["describe_graph", "propose_assignment"])
    s.propose_schedule([], "earliest_free")
    assert end_check(s, ["describe_graph", "propose_assignment", "propose_schedule"]) is None


def test_an_illegal_plan_and_a_missing_schedule_are_sent_back(g_hw, tmp_path):
    g, hw = g_hw
    steps = [("describe_graph", {}), ("propose_assignment", {"targets": entries(g, hw, wrong="c0")}), ("say", "Plan done."),
             ("propose_assignment", {"targets": entries(g, hw)}), ("say", "Legal plan."),
             ("describe_schedule", {}), ("propose_schedule", {"resident": [], "policy": "earliest_free"}), ("say", "Done.")]
    with FakeModel(script_for(steps)) as fm:
        out = run_mapping_agent(h.OpenRouter(model="fake", base_url=fm.url), g, hw, log_path=str(tmp_path / "l.jsonl"))
    assert fm.tools_offered() == ["describe_graph", "describe_schedule", "propose_assignment", "propose_schedule"]
    assert out["end_of_turn_sendbacks"] == 2 and out["model"] == "fake"
    assert out["llm_plan_legal"] and out["final_source"] == "llm_validated" and out["proposals"] == 2
    assert out["schedule"]["source"].startswith(("llm_validated", "code_best"))       # code still picks the faster schedule
    told = " ".join(str(m.get("content")) for m in fm.all_chat())
    assert "validator rejected your last assignment" in told and "Now call describe_schedule" in told


def test_malformed_arguments_are_answered_by_the_tool(g_hw, tmp_path):
    g, hw = g_hw
    steps = [("describe_graph", {}), ("propose_assignment", {"targets": ["c0 x2"]}), ("propose_assignment", {"targets": entries(g, hw)}),
             ("describe_schedule", {}), ("propose_schedule", {"resident": [], "tiles": "[[not a list"}),
             ("propose_schedule", {"resident": []}), ("say", "Done.")]
    with FakeModel(script_for(steps)) as fm:
        out = run_mapping_agent(h.OpenRouter(model="fake", base_url=fm.url), g, hw, log_path=str(tmp_path / "l.jsonl"))
    tool_results = " ".join(str(m.get("content")) for m in fm.all_chat() if m.get("role") == "tool")
    assert "targets must be an object" in tool_results and '"legal": false' in tool_results    # the validator answers
    assert out["final_source"] == "llm_validated" and out["end_of_turn_sendbacks"] == 0
