import json, pytest
from agents.llm_client import MockLLM, Action
from agents.verification_agent import run_agent
from verification import tools
from tests import faults


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


def test_gemini_adapter_with_fake_client(tmp_path):
    pytest.importorskip("google.genai")
    from agents.llm_client import GeminiLLM

    class FC:  # fake function call
        def __init__(s, n, a): s.name, s.args = n, a
    class Resp:
        def __init__(s, calls=None, text=None):
            s.function_calls, s.text = calls, text
            s.candidates = [type("C", (), {"content": object()})()]
    seq = [Resp(calls=[FC("run_suite", {"n_random": 3})]), Resp(text="All good.")]
    class FakeModels:
        def generate_content(self, **kw): return seq.pop(0)
    class FakeClient: models = FakeModels()

    # the real google.genai types validate our contents/config; only the network call is faked
    llm = GeminiLLM(model="test-model", client=FakeClient())
    out = run_agent(llm, "verify", log_path=str(tmp_path / "log.jsonl"))
    assert out["steps"] == 1 and out["llm_summary"] == "All good."
    assert out["verified_facts_from_report"]["total"] == 8        # 5 fixed + 3 random


def _server_error(code=503):
    from google.genai import errors
    return errors.ServerError(code, {"error": {"code": code, "message": "high demand", "status": "UNAVAILABLE"}}, None)


def _resp(text="done"):
    r = type("R", (), {})(); r.function_calls = None; r.text = text
    r.candidates = [type("C", (), {"content": object()})()]
    return r


def test_gemini_retries_then_succeeds(tmp_path):
    pytest.importorskip("google.genai")
    from agents.llm_client import GeminiLLM
    seq = [_server_error(), _server_error(), _resp("ok")]
    class M:
        def generate_content(self, **kw):
            x = seq.pop(0)
            if isinstance(x, Exception): raise x
            return x
    class C: models = M()
    naps = []
    llm = GeminiLLM(model="primary", client=C(), fallback_models=[], sleep=naps.append)
    out = run_agent(llm, "verify", log_path=str(tmp_path / "l.jsonl"))
    assert out["llm_summary"] == "ok" and naps == [2, 4]


def test_gemini_falls_back_to_second_model(tmp_path):
    pytest.importorskip("google.genai")
    from agents.llm_client import GeminiLLM
    seen = []
    class M:
        def generate_content(self, model, **kw):
            seen.append(model)
            if model == "primary": raise _server_error()
            return _resp("from fallback")
    class C: models = M()
    llm = GeminiLLM(model="primary", client=C(), fallback_models=["backup"], retries=1, sleep=lambda s: None)
    out = run_agent(llm, "verify", log_path=str(tmp_path / "l.jsonl"))
    assert out["llm_summary"] == "from fallback" and llm.model_used == "backup" and seen[-1] == "backup"


def test_gemini_total_outage_raises_clean_error(tmp_path):
    pytest.importorskip("google.genai")
    from agents.llm_client import GeminiLLM, LLMUnavailable
    class M:
        def generate_content(self, **kw): raise _server_error()
    class C: models = M()
    llm = GeminiLLM(model="p", client=C(), fallback_models=["b"], retries=1, sleep=lambda s: None)
    with pytest.raises(LLMUnavailable):
        run_agent(llm, "verify", log_path=str(tmp_path / "l.jsonl"))


def test_auth_errors_fail_fast_not_retried(tmp_path):
    pytest.importorskip("google.genai")
    from google.genai import errors
    from agents.llm_client import GeminiLLM
    calls = []
    class M:
        def generate_content(self, **kw):
            calls.append(1); raise errors.ClientError(403, {"error": {"code": 403, "message": "bad key"}}, None)
    class C: models = M()
    llm = GeminiLLM(model="p", client=C(), fallback_models=["b"], retries=3, sleep=lambda s: None)
    with pytest.raises(errors.ClientError):
        run_agent(llm, "verify", log_path=str(tmp_path / "l.jsonl"))
    assert len(calls) == 1


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
    from agents.llm_client import GroqLLM
    c = _GroqClient([_gmsg(calls=[("call_1", "run_suite", '{"n_random": 3}')]), _gmsg(content="All good.")])
    out = run_agent(GroqLLM(model="m", client=c), "verify", log_path=str(tmp_path / "l.jsonl"))
    assert out["llm_summary"] == "All good." and out["verified_facts_from_report"]["total"] == 8
    second = c.requests[1]["messages"]                                   # protocol check
    assert second[-2]["role"] == "assistant" and second[-2]["tool_calls"][0]["id"] == "call_1"
    assert second[-1]["role"] == "tool" and second[-1]["tool_call_id"] == "call_1"
    assert c.requests[0]["tools"][0]["type"] == "function" and c.requests[0]["tool_choice"] == "auto"


def test_groq_bad_json_arguments_are_rejected_safely(tmp_path):
    from agents.llm_client import GroqLLM
    c = _GroqClient([_gmsg(calls=[("c1", "run_suite", "{not json")]), _gmsg(content="gave up")])
    out = run_agent(GroqLLM(model="m", client=c), "verify", log_path=str(tmp_path / "l.jsonl"))
    entries = [json.loads(l) for l in (tmp_path / "l.jsonl").read_text().splitlines()]
    assert entries[0]["ok"] is False and out["llm_summary"] == "gave up"


def test_groq_retry_and_fallback(tmp_path):
    from agents.llm_client import GroqLLM
    c = _GroqClient([_HttpErr(503), _HttpErr(429), _gmsg(content="ok")])
    naps = []
    llm = GroqLLM(model="p", client=c, fallback_models=[], sleep=naps.append)
    assert run_agent(llm, "v", log_path=str(tmp_path / "l.jsonl"))["llm_summary"] == "ok" and naps == [2, 4]
    c2 = _GroqClient([_HttpErr(503), _gmsg(content="fb")])
    llm2 = GroqLLM(model="p", client=c2, fallback_models=["b"], retries=0, sleep=lambda s: None)
    run_agent(llm2, "v", log_path=str(tmp_path / "l2.jsonl"))
    assert llm2.model_used == "b" and c2.requests[-1]["model"] == "b"


def test_groq_auth_error_not_retried(tmp_path):
    from agents.llm_client import GroqLLM
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
    from agents.llm_client import GroqLLM
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
