"""The agents' harness (OpenAI's Agents SDK, models through OpenRouter) and its end-of-turn check, tested OFFLINE with a
stand-in for OpenRouter (tests/fake_model.py): no key, no network, no cost. The real loop, tools and check run; only the model is scripted.
Shown here: only our tools are offered, a reply that breaks the answer contract is sent back, the send-backs are limited,
code still decides whether a cause is supported, a busy model falls back to the next one, a used-up credit goes straight
to the last fallback, the spending cap stops a task, a wrong key is raised."""
import json

import pytest

from tests.fake_model import FakeModel, calls_so_far, texts_so_far
from tests.test_chain_debug import REAL_XX_LOG, answer, result
from x2c.agents import harness as h
from x2c.agents.chain_debug_agent import run_chain_debug_agent

KEYS = ("cause_id", "confidence", "explanation", "proposed_fix", "next_check")


@pytest.fixture(autouse=True)
def _fresh_skips():
    h.reset_skipped()
    yield
    h.reset_skipped()


def test_errors_are_classified():
    assert h.classify(503) == "busy" and h.classify(500) == "busy" and h.classify(None) == "busy"
    assert h.classify(429, "rate_limit_exceeded") == "quota" and h.classify(429, "insufficient_quota") == "billing"
    assert h.classify(404, "model_not_found") == "missing"
    assert h.classify(401, "invalid_api_key") == "auth" and h.classify(403) == "auth"
    assert h.classify(400, "invalid_request_error") == "other"


def test_cost_estimate():
    assert h.estimate_usd("openai/gpt-5.6-luna", 1_000_000, 1_000_000) == 0.20 + 1.20
    assert h.estimate_usd("gpt-5.6-luna", 1_000_000, 0) == 0.20                # OpenAI's own name works too
    assert h.estimate_usd("some-new-model", 10, 10) is None        # unknown price: only the tokens are reported


# ------------------------------------------------------------------ the end-of-turn check itself (pure code)
def test_the_json_answer_check():
    chk = lambda text, used, allowed={"x", "unknown"}: h.check_json_answer(text, used, first_tool="ev", keys=("a", "b"),
                                                                            field_name="a", allowed=allowed)
    assert "Call ev first" in chk('{"a": "x", "b": 1}', [])
    assert "ONLY one JSON object" in chk("the fabric is broken", ["ev"])
    assert "ONLY one JSON object" in chk('["x"]', ["ev"])
    assert "missing: b" in chk('{"a": "x"}', ["ev"])
    assert "not allowed" in chk('{"a": "cosmic_rays", "b": 1}', ["ev"]) and "'unknown'" in chk('{"a": "y", "b": 1}', ["ev"])
    assert "'unknown'" not in chk('{"a": "y", "b": 1}', ["ev"], allowed={"x"})
    assert chk('Here it is: {"a": "x", "b": 1}', ["ev"]) is None


def test_model_order(monkeypatch):
    monkeypatch.delenv("BM_LLM_MODEL", raising=False)
    monkeypatch.delenv("BM_LLM_FALLBACKS", raising=False)
    assert h.OpenRouter().models() == [h.DEFAULT_MODEL, *h.DEFAULT_FALLBACKS]
    monkeypatch.setenv("BM_LLM_MODEL", "m1")
    monkeypatch.setenv("BM_LLM_FALLBACKS", "m2, m1 ,m3")
    assert h.OpenRouter().models() == ["m1", "m2", "m3"]
    assert h.OpenRouter(model="a", fallbacks=("b",)).models() == ["a", "b"]


def test_spending_cap_setting(monkeypatch):
    monkeypatch.delenv("BM_LLM_MAX_USD", raising=False)
    assert h.OpenRouter().budget() == h.DEFAULT_MAX_USD
    monkeypatch.setenv("BM_LLM_MAX_USD", "0.25")
    assert h.OpenRouter().budget() == 0.25 and h.OpenRouter(max_usd=2).budget() == 2


# ------------------------------------------------------------------ the real harness with a stand-in model
sdk = pytest.importorskip("agents", reason="the agent harness is an optional extra: uv sync --extra agents")


@pytest.fixture
def xx(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "reports" / "fab_jobs" / "add8_sat"
    d.mkdir(parents=True)
    (d / "add8_sat_fab.bin").write_bytes(b"\x01" * 12024)
    (d / "sim_console.log").write_text(REAL_XX_LOG)
    return result("sim", "fail", "500 numpy mismatch(es), 500 rtl mismatch(es) in 500 vectors"), d.parent


def chain(script_answers, model="fake"):
    """A model that asks for the evidence once, then gives the scripted replies in order (the last one repeats)."""
    def script(messages, request):
        if "chain_evidence" not in calls_so_far(messages):
            return ("tool", "chain_evidence", {"op": "add8_sat"})
        n = len(texts_so_far(messages))
        return ("text", script_answers[min(n, len(script_answers) - 1)])
    return script


def test_only_our_tool_and_a_contract_breaking_reply_is_sent_back(xx):
    r, jd = xx
    with FakeModel(chain(["I think the fabric is broken, sorry.", answer("bitstream_not_loaded")])) as fm:
        d = run_chain_debug_agent(h.OpenRouter(model="fake", base_url=fm.url), "add8_sat", r, jd)
    assert fm.tools_offered() == ["chain_evidence"]              # none of the SDK's own file / shell / web tools
    assert d["end_of_turn_sendbacks"] == 1 and d["steps"] == 1 and d["model"] == "fake"
    assert d["final_cause"] == "bitstream_not_loaded" and d["final_cause_source"] == "llm_validated" and d["validated"]
    sent_back = [m for m in fm.all_chat() if m["role"] == "user" and "ONLY one JSON object" in str(m.get("content"))]
    assert sent_back, "the end-of-turn check's reason must reach the model"


def test_an_off_menu_cause_is_sent_back(xx):
    r, jd = xx
    with FakeModel(chain([answer("cosmic_rays"), answer("bitstream_not_loaded")])) as fm:
        d = run_chain_debug_agent(h.OpenRouter(model="fake", base_url=fm.url), "add8_sat", r, jd)
    assert d["end_of_turn_sendbacks"] == 1 and d["final_cause_source"] == "llm_validated"


def test_code_still_decides_whether_the_evidence_supports_the_cause(xx):
    r, jd = xx
    with FakeModel(chain([answer("pins_not_connected")])) as fm:       # on the menu, keeps the contract, but wrong
        d = run_chain_debug_agent(h.OpenRouter(model="fake", base_url=fm.url), "add8_sat", r, jd)
    assert d["end_of_turn_sendbacks"] == 0
    assert d["final_cause"] == "bitstream_not_loaded" and d["final_cause_source"] == "rules_fallback (llm claim rejected)"


def test_send_backs_are_limited_then_the_rules_answer(xx):
    r, jd = xx
    with FakeModel(chain(["still thinking"])) as fm:
        d = run_chain_debug_agent(h.OpenRouter(model="fake", base_url=fm.url), "add8_sat", r, jd)
    assert d["end_of_turn_sendbacks"] == 2 and d["validation"]["json_ok"] is False
    assert d["final_cause"] == "bitstream_not_loaded" and d["final_cause_source"] == "rules_fallback (no usable llm answer)"


def busy_then(script, busy=("busy",)):
    def s(messages, request):
        if request.get("model") in busy:
            return ("http", 503, {"error": {"message": "overloaded", "type": "server_error"}})
        return script(messages, request)
    return s


def test_a_busy_model_falls_back_to_the_next(xx):
    r, jd = xx
    with FakeModel(busy_then(chain([answer("bitstream_not_loaded")]))) as fm:
        d = run_chain_debug_agent(h.OpenRouter(model="busy", fallbacks=("fake",), base_url=fm.url, api_retries=0), "add8_sat", r, jd)
    assert d["model"] == "fake" and d["final_cause_source"] == "llm_validated"
    assert {req["model"] for req in fm.requests} == {"busy", "fake"}


def test_every_model_busy_means_the_rules_answer(xx):
    r, jd = xx
    with FakeModel(busy_then(chain([answer("bitstream_not_loaded")]), busy=("busy", "busy2"))) as fm:
        d = run_chain_debug_agent(h.OpenRouter(model="busy", fallbacks=("busy2",), base_url=fm.url, api_retries=0), "add8_sat", r, jd)
    assert d["stopped"] == "llm_unavailable" and "busy" in d["llm_error"]
    assert d["final_cause"] == "bitstream_not_loaded" and d["final_cause_source"] == "rules_fallback (llm unavailable)"


def test_a_wrong_key_is_raised_not_hidden(xx):
    r, jd = xx
    bad = lambda messages, request: ("http", 401, {"error": {"message": "Incorrect API key", "code": "invalid_api_key"}})
    with FakeModel(bad) as fm, pytest.raises(Exception, match="401"):
        run_chain_debug_agent(h.OpenRouter(model="fake", fallbacks=("fake2",), base_url=fm.url, api_retries=0), "add8_sat", r, jd)


def test_no_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        h.converse(h.OpenRouter(), "s", "t", [], None)


EVIDENCE = {"name": "chain_evidence", "description": "Stand-in evidence tool.",
            "input_schema": {"type": "object", "properties": {"op": {"type": "string"},
                                                              "detail": {"type": "object", "properties": {"lines": {"type": "integer"}}}}}}


def test_the_tool_log_and_the_schema_reaches_the_model_unchanged(xx, tmp_path):
    r, jd = xx
    log_path, calls = tmp_path / "log.jsonl", []
    with FakeModel(chain(["no", answer("bitstream_not_loaded")])) as fm:
        run = h.converse(h.OpenRouter(model="fake", base_url=fm.url), "system", "task op=add8_sat", [EVIDENCE],
                         lambda name, args: calls.append((name, args)) or {"op": args.get("op")}, log=h.jsonl_logger(log_path),
                         end_check=lambda text, used: h.check_json_answer(text, used, first_tool="chain_evidence", keys=KEYS,
                                                                         field_name="cause_id", allowed={"bitstream_not_loaded"}))
    kinds = [json.loads(line)["kind"] for line in log_path.read_text().splitlines()]
    assert kinds[0] == "start" and "tool" in kinds and kinds.count("end_of_turn") == 2
    assert calls == [("chain_evidence", {"op": "add8_sat"})] and run["continues"] == 1 and run["stopped"] == "finished"
    (offered,) = [t["function"] for t in fm.requests[0]["tools"] if t["function"]["name"] == "chain_evidence"]
    assert offered["parameters"] == EVIDENCE["input_schema"]          # objects too: the schema goes to the model as it is
    assert fm.requests[0]["messages"][0] == {"role": "system", "content": "system"}   # our system prompt reaches the model
    assert run["tokens"] == {"input": 10 * len(fm.requests), "output": 5 * len(fm.requests)}
    assert {req.get("max_tokens") for req in fm.requests} == {h.DEFAULT_MAX_TOKENS}    # OpenRouter reserves credit for it
    assert run["text"] == answer("bitstream_not_loaded")               # the accepted reply only, not every reply joined


def failing(script, models, status, message, code_for={404: "model_not_found", 429: "rate_limit_exceeded"}):
    def s(messages, request):
        if request.get("model") in models:
            return ("http", status, {"error": {"message": message, "code": code_for.get(status)}})
        return script(messages, request)
    return s


def test_a_model_that_does_not_exist_is_skipped_for_the_rest_of_the_run(xx):
    r, jd = xx
    with FakeModel(failing(chain([answer("bitstream_not_loaded")]), {"gone"}, 404, "model: gone")) as fm:
        spec = h.OpenRouter(model="gone", fallbacks=("fake",), base_url=fm.url, api_retries=0)
        first = run_chain_debug_agent(spec, "add8_sat", r, jd)
        asked = sum(req.get("model") == "gone" for req in fm.requests)
        second = run_chain_debug_agent(spec, "add8_sat", r, jd)
    assert first["model"] == second["model"] == "fake" and first["final_cause_source"] == second["final_cause_source"] == "llm_validated"
    assert asked >= 1 and sum(req.get("model") == "gone" for req in fm.requests) == asked      # not asked again


def test_used_up_credit_stops_at_once(xx):
    r, jd = xx
    with FakeModel(failing(chain([answer("bitstream_not_loaded")]), {"a", "b"}, 402,
                           "Insufficient credits. Add more using https://openrouter.ai/credits")) as fm:
        d = run_chain_debug_agent(h.OpenRouter(model="a", fallbacks=("b",), base_url=fm.url, api_retries=0, groq_fallback=False),
                                  "add8_sat", r, jd)
    assert {req["model"] for req in fm.requests} == {"a"}                  # the other model uses the same credit: not asked
    assert d["stopped"] == "llm_unavailable" and "credit used up" in d["llm_error"]
    assert d["final_cause_source"] == "rules_fallback (llm unavailable)"


def test_when_no_model_is_left_the_rules_answer(xx):
    r, jd = xx
    with FakeModel(failing(chain([answer("bitstream_not_loaded")]), {"a", "b"}, 429, "rate limit")) as fm:
        d = run_chain_debug_agent(h.OpenRouter(model="a", fallbacks=("b",), base_url=fm.url, api_retries=0), "add8_sat", r, jd)
    assert d["stopped"] == "llm_unavailable" and "rate limit" in d["llm_error"]
    assert d["final_cause"] == "bitstream_not_loaded" and d["final_cause_source"] == "rules_fallback (llm unavailable)"


# ------------------------------------------------------------------ the small fallback loop and Groq as the last resort
class _Replying:
    """A stand-in for Groq in plain_loop: asks for the evidence, then gives the scripted replies; it can take the
    end-of-turn check's reply (send_text), like GroqLLM."""
    model_used = "groq-test"

    def __init__(self, replies):
        self.replies, self.texts_received = list(replies), []

    def start(self, goal, schemas, system_prompt=None):
        from x2c.agents.llm_client import Action
        return Action("tool_calls", calls=[("chain_evidence", {"op": "add8_sat"})])

    def send_tool_results(self, results, force_final=False):
        return self._next()

    def send_text(self, text):
        self.texts_received.append(text)
        return self._next()

    def _next(self):
        from x2c.agents.llm_client import Action
        return Action("final", text=self.replies.pop(0) if len(self.replies) > 1 else self.replies[0])


def test_plain_loop_applies_the_same_end_of_turn_check(xx):
    r, jd = xx
    groq = _Replying(["no idea", answer("bitstream_not_loaded")])
    d = run_chain_debug_agent(groq, "add8_sat", r, jd)
    assert d["end_of_turn_sendbacks"] == 1 and "ONLY one JSON object" in groq.texts_received[0]
    assert d["final_cause_source"] == "llm_validated" and d["model"] == "groq-test"


def test_the_no_model_stand_ins_behave_as_before(xx):
    from x2c.agents.chain_debug_agent import MockChainDebugLLM
    r, jd = xx
    d = run_chain_debug_agent(MockChainDebugLLM(), "add8_sat", r, jd)
    assert d["end_of_turn_sendbacks"] == 0 and d["model"] is None and d["final_cause"] == "bitstream_not_loaded"


def test_groq_is_the_last_resort_when_no_model_is_left(xx, monkeypatch):
    from x2c.agents import llm_client
    r, jd = xx
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setattr(llm_client, "GroqLLM", lambda *a, **k: _Replying([answer("bitstream_not_loaded")]))
    with FakeModel(failing(chain([answer("bitstream_not_loaded")]), {"a", "b"}, 429, "rate limit")) as fm:
        d = run_chain_debug_agent(h.OpenRouter(model="a", fallbacks=("b",), base_url=fm.url, api_retries=0), "add8_sat", r, jd)
    assert d["model"] == "groq:groq-test" and d["final_cause_source"] == "llm_validated" and d["llm_error"] is None


@pytest.mark.parametrize("why", ["no_groq_key", "groq_fallback_off"])
def test_without_groq_the_rules_answer(xx, monkeypatch, why):
    from x2c.agents import llm_client
    r, jd = xx
    if why == "no_groq_key":
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
    else:
        monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setattr(llm_client, "GroqLLM", lambda *a, **k: pytest.fail("Groq must not be used"))
    with FakeModel(failing(chain([answer("bitstream_not_loaded")]), {"a"}, 429, "rate limit")) as fm:
        spec = h.OpenRouter(model="a", fallbacks=(), base_url=fm.url, api_retries=0, groq_fallback=(why != "groq_fallback_off"))
        spec.fallbacks = ("a",)                       # only one model
        d = run_chain_debug_agent(spec, "add8_sat", r, jd)
    assert d["stopped"] == "llm_unavailable" and d["final_cause_source"] == "rules_fallback (llm unavailable)"


def test_the_spending_cap_stops_a_task_and_the_rules_answer(xx):
    r, jd = xx
    with FakeModel(chain(["not json", answer("bitstream_not_loaded")])) as fm:
        d = run_chain_debug_agent(h.OpenRouter(model="openai/gpt-5.6-luna", base_url=fm.url, max_usd=0.000001), "add8_sat", r, jd)
    assert len(fm.requests) == 1                                             # stopped after the first model call
    assert d["stopped"] == "budget_reached" and d["final_cause"] == "bitstream_not_loaded"
    assert d["cost_usd"] > 0.000001
