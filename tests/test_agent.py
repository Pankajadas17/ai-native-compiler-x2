import json, pytest
from x2c.agents.llm_client import MockLLM, Action
from x2c.agents.verification_agent import run_agent
from x2c.verification import tools
from tests import faults
from x2c.agents.llm_client import GroqLLM as _GroqLLM


class _GroqAdapter(_GroqLLM):
    """GroqLLM's own mechanics (protocol, retries, fallbacks) without the end-of-turn check's replies;
    those are tested in test_groq_end_of_turn_reply and tests/test_agent_harness.py."""
    send_text = None


@pytest.fixture(autouse=True)
def _reset_backend():
    yield
    tools.set_x2_backend(None)


def test_agent_healthy_run(tmp_path):
    out = run_agent(MockLLM(), "verify", log_path=str(tmp_path / "log.jsonl"))
    assert out["stopped"] == "finished" and out["steps"] == 1
    assert out["verified_facts_from_report"]["failed"] == 0


def test_agent_investigates_failures(tmp_path):
    tools.set_x2_backend(faults.sign_flip)
    out = run_agent(MockLLM(), "verify", log_path=str(tmp_path / "log.jsonl"))
    assert out["steps"] == 2                                   # run_suite, then get_failures
    facts = out["verified_facts_from_report"]
    assert facts["failed"] > 0 and "sign_inverted" in facts["failure_categories"]
    lines = (tmp_path / "log.jsonl").read_text().splitlines()
    assert [json.loads(l)["tool"] for l in lines] == ["run_suite", "get_failures"]


class _Rogue:
    """An LLM that misbehaves: invents a tool, passes bad args, then loops forever."""
    def start(self, goal, schemas): return Action("tool_calls", calls=[("delete_everything", {})])
    def send_tool_results(self, results): return Action("tool_calls", calls=[("run_suite", {"n_random": 10**9})])


def test_rogue_llm_is_contained(tmp_path):
    out = run_agent(_Rogue(), "verify", max_steps=3, log_path=str(tmp_path / "log.jsonl"))
    assert out["stopped"] == "step_limit_reached"
    entries = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert all(e["ok"] is False for e in entries)               # every bad call was rejected


def test_call_tool_blocks_path_escape_and_unknown_args():
    assert "error" in tools.call_tool("get_failures", {"report_path": "../../etc/passwd"})
    assert "error" in tools.call_tool("get_failures", {"report_path": "C:/Windows/x.json"})
    assert "error" in tools.call_tool("run_suite", {"x2_fn": "evil"})
    assert "error" in tools.call_tool("run_suite", {"n_random": 99999})


# ---------------- Groq adapter (fake client; real tool-call message shapes) ----------------
import types as _t


def _gmsg(content=None, calls=None):
    m = _t.SimpleNamespace(content=content, tool_calls=[
        _t.SimpleNamespace(id=cid, function=_t.SimpleNamespace(name=n, arguments=a)) for cid, n, a in (calls or [])] or None)
    return _t.SimpleNamespace(choices=[_t.SimpleNamespace(message=m)])


class _GroqClient:
    def __init__(self, seq):
        self.seq, self.requests = list(seq), []
        self.chat = _t.SimpleNamespace(completions=_t.SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.requests.append({**kw, "messages": list(kw["messages"])})   # snapshot, not a live reference
        x = self.seq.pop(0)
        if isinstance(x, Exception): raise x
        return x


class _HttpErr(Exception):
    def __init__(self, status): super().__init__(f"http {status}"); self.status_code = status


def test_groq_full_loop_and_message_protocol(tmp_path):
    GroqLLM = _GroqAdapter
    c = _GroqClient([_gmsg(calls=[("call_1", "run_suite", '{"n_random": 3}')]), _gmsg(content="All good.")])
    out = run_agent(GroqLLM(model="m", client=c), "verify", log_path=str(tmp_path / "l.jsonl"))
    assert out["llm_summary"] == "All good." and out["verified_facts_from_report"]["total"] == 8
    second = c.requests[1]["messages"]                                   # protocol check
    assert second[-2]["role"] == "assistant" and second[-2]["tool_calls"][0]["id"] == "call_1"
    assert second[-1]["role"] == "tool" and second[-1]["tool_call_id"] == "call_1"
    assert c.requests[0]["tools"][0]["type"] == "function" and c.requests[0]["tool_choice"] == "auto"


def test_groq_bad_json_arguments_are_rejected_safely(tmp_path):
    GroqLLM = _GroqAdapter
    c = _GroqClient([_gmsg(calls=[("c1", "run_suite", "{not json")]), _gmsg(content="gave up")])
    out = run_agent(GroqLLM(model="m", client=c), "verify", log_path=str(tmp_path / "l.jsonl"))
    entries = [json.loads(l) for l in (tmp_path / "l.jsonl").read_text().splitlines()]
    assert entries[0]["ok"] is False and out["llm_summary"] == "gave up"


def test_groq_retry_and_fallback(tmp_path):
    GroqLLM = _GroqAdapter
    c = _GroqClient([_HttpErr(503), _HttpErr(429), _gmsg(content="ok")])
    naps = []
    llm = GroqLLM(model="p", client=c, fallback_models=[], sleep=naps.append)
    assert run_agent(llm, "v", log_path=str(tmp_path / "l.jsonl"))["llm_summary"] == "ok" and naps == [2, 4]
    c2 = _GroqClient([_HttpErr(503), _gmsg(content="fb")])
    llm2 = GroqLLM(model="p", client=c2, fallback_models=["b"], retries=0, sleep=lambda s: None)
    run_agent(llm2, "v", log_path=str(tmp_path / "l2.jsonl"))
    assert llm2.model_used == "b" and c2.requests[-1]["model"] == "b"


def test_groq_auth_error_not_retried(tmp_path):
    GroqLLM = _GroqAdapter
    c = _GroqClient([_HttpErr(401), _gmsg(content="never")])
    with pytest.raises(_HttpErr):
        run_agent(GroqLLM(model="p", client=c, fallback_models=["b"], sleep=lambda s: None), "v",
                  log_path=str(tmp_path / "l.jsonl"))
    assert len(c.requests) == 1


def test_reduced_coverage_is_flagged(tmp_path):
    class Lazy:                                   # an LLM that quietly runs only the 5 fixed cases
        def start(self, g, sch): return Action("tool_calls", calls=[("run_suite", {"n_random": 0})])
        def send_tool_results(self, r): return Action("final", text="all 5 passed")
    out = run_agent(Lazy(), "verify", log_path=str(tmp_path / "l.jsonl"))
    assert out["verified_facts_from_report"]["coverage"] == {"tests_run": 5, "full_suite_size": 25}
    assert any("REDUCED COVERAGE" in w for w in out["warnings"])


def test_full_coverage_has_no_warning(tmp_path):
    out = run_agent(MockLLM(), "verify", log_path=str(tmp_path / "l.jsonl"))
    assert out["warnings"] == []


def test_groq_retired_model_404_falls_through_to_next_model(tmp_path):
    GroqLLM = _GroqAdapter
    c = _GroqClient([_HttpErr(404), _gmsg(content="ok via fallback")])
    naps = []
    llm = GroqLLM(model="retired-model", client=c, fallback_models=["good-model"], sleep=naps.append)
    out = run_agent(llm, "v", log_path=str(tmp_path / "l.jsonl"))
    assert out["llm_summary"] == "ok via fallback" and llm.model_used == "good-model" and naps == []   # no retries on 404


def test_agent_reports_workload_facts(tmp_path):
    class Two:
        def start(self, g, sch): return Action("tool_calls", calls=[("run_suite", {})])
        def send_tool_results(self, r):
            if r[0][0] == "run_suite": return Action("tool_calls", calls=[("run_workloads", {})])
            return Action("final", text="done")
    out = run_agent(Two(), "verify", log_path=str(tmp_path / "l.jsonl"))
    assert out["steps"] == 2 and out["verified_workload_facts"]["total"] == 5
    assert out["verified_workload_facts"]["failed"] == 0 and out["warnings"] == []


# ------------------------------------------------------------ the agent on OpenAI's Agents SDK (stand-in model, no key)
def _scripted(steps):
    from tests.fake_model import calls_so_far
    def s(messages, request):
        n = sum(1 for m in messages if m.get("role") == "assistant")
        kind, value = steps[min(n, len(steps) - 1)]
        return ("text", value) if kind == "say" else ("tool", kind, value)
    return s


@pytest.fixture
def harness_sdk():
    pytest.importorskip("agents", reason="the agent harness is an optional extra: uv sync --extra agents")
    from tests.fake_model import FakeModel
    from x2c.agents import harness
    harness.reset_skipped()
    return FakeModel, harness


def test_on_the_harness_nothing_verified_and_reduced_coverage_are_sent_back(tmp_path, harness_sdk):
    FakeModel, harness = harness_sdk
    steps = [("say", "Everything passes."), ("run_suite", {"n_random": 1}), ("say", "Suite passed."),
             ("run_suite", {}), ("say", "The full suite passed.")]
    with FakeModel(_scripted(steps)) as fm:
        out = run_agent(harness.OpenRouter(model="fake", base_url=fm.url), "verify", log_path=str(tmp_path / "log.jsonl"))
    told = " ".join(str(m.get("content")) for m in fm.all_chat())
    assert "Nothing has been verified yet" in told and "standard tests" in told
    assert out["end_of_turn_sendbacks"] == 2 and out["model"] == "fake" and not out["warnings"]
    facts = out["verified_facts_from_report"]
    assert facts["failed"] == 0 and facts["coverage"]["tests_run"] == facts["coverage"]["full_suite_size"]
    assert "run_suite" in fm.tools_offered() and "get_failures" in fm.tools_offered()


def test_on_the_harness_a_failing_suite_must_be_diagnosed(tmp_path, harness_sdk):
    FakeModel, harness = harness_sdk
    tools.set_x2_backend(faults.sign_flip)
    steps = [("run_suite", {}), ("say", "Some tests failed."), ("get_failures", {}), ("say", "The output sign is inverted.")]
    with FakeModel(_scripted(steps)) as fm:
        out = run_agent(harness.OpenRouter(model="fake", base_url=fm.url), "verify", log_path=str(tmp_path / "log.jsonl"))
    assert out["end_of_turn_sendbacks"] == 1
    assert out["verified_facts_from_report"]["failed"] > 0 and "sign_inverted" in out["verified_facts_from_report"]["failure_categories"]


def test_groq_end_of_turn_reply(tmp_path):
    """The real Groq adapter in the small fallback loop gets the same end-of-turn check as OpenAI's Agents SDK:
    a reduced suite is sent back, and the reply reaches the model as a user message."""
    from x2c.agents.llm_client import GroqLLM
    c = _GroqClient([_gmsg(calls=[("c1", "run_suite", '{"n_random": 3}')]), _gmsg(content="All good."),
                     _gmsg(calls=[("c2", "run_suite", "{}")]), _gmsg(content="All good, full suite.")])
    out = run_agent(GroqLLM(model="m", client=c, fallback_models=[]), "verify", log_path=str(tmp_path / "l.jsonl"))
    assert out["end_of_turn_sendbacks"] == 1 and out["llm_summary"] == "All good, full suite." and not out["warnings"]
    third = c.requests[2]["messages"]
    assert third[-1]["role"] == "user" and "standard tests" in third[-1]["content"]
