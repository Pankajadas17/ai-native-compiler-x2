"""The agents' harness (Google's Antigravity SDK) and its end-of-turn check, tested OFFLINE with a stand-in model
(tests/fake_model.py): no key and no network. The real loop, tools and hook run; only the model is scripted.
Shown here: only our tools are offered, a reply that breaks the answer contract is sent back, the send-backs are limited,
code still decides whether a cause is supported, a busy model falls back to the next one, a wrong key is raised."""
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


# the exact provider messages of a real run (2026-11-18): daily free quota used up, a retired model, a busy model
REAL_429 = ("AntigravityExecutionError: agent executor error: Error 429, Message: You exceeded your current quota, please check your "
            "plan and billing details. * Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, "
            "limit: 20, model: gemini-3.6-flash, description: Generate content requests per day limit exceeded on Free Tier. "
            "Please retry in 20h53m28.800033716s., Status: RESOURCE_EXHAUSTED")
REAL_404 = ("AntigravityExecutionError: Error 404, Message: This model models/gemini-2.5-flash is no longer available to new users. "
            "Please update your code to use models/gemini-3.8-flash for the latest features and improvements., Status: NOT_FOUND")
REAL_503 = ("Error 503, Message: This model is currently experiencing high demand. Spikes in demand are usually temporary. "
            "Please try again later., Status: UNAVAILABLE")


def test_the_real_provider_errors_are_classified():
    assert h.classify(REAL_429) == "quota_daily" and h.classify(REAL_404) == "missing" and h.classify(REAL_503) == "busy"
    assert h.classify("Error 429: rate limit, Status: RESOURCE_EXHAUSTED") == "quota"
    assert h.classify("Error 401: API key not valid. Please pass a valid API key. Status: UNAUTHENTICATED") == "auth"
    assert h.classify("KeyError: 'cause_id'") == "other"


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
    assert h.Gemini().models() == [h.DEFAULT_MODEL, *h.DEFAULT_FALLBACKS]
    monkeypatch.setenv("BM_LLM_MODEL", "m1")
    monkeypatch.setenv("BM_LLM_FALLBACKS", "m2, m1 ,m3")
    assert h.Gemini().models() == ["m1", "m2", "m3"]
    assert h.Gemini(model="a", fallbacks=("b",)).models() == ["a", "b"]


def test_tools_from_schemas():
    """Our JSON tool schemas -> plain functions for the harness; objects travel as JSON strings (Gemini tool schemas
    cannot hold a free-form object); the arguments the model gave reach our existing call() unchanged."""
    import inspect
    calls = []
    schemas = [{"name": "t1", "description": "A tool.", "input_schema": {"type": "object", "required": ["n"], "properties": {
        "n": {"type": "integer"}, "xs": {"type": "array", "items": {"type": "string"}},
        "specs": {"type": "array", "items": {"type": "object", "properties": {"a": {"type": "integer"}}}}}}}]
    (f,) = h.tools_from_schemas(schemas, lambda name, args: calls.append((name, args)) or {"ok": True})
    sig = inspect.signature(f)
    assert f.__name__ == "t1" and list(sig.parameters) == ["n", "xs", "specs"]
    assert sig.parameters["n"].default is inspect.Parameter.empty and sig.parameters["xs"].default is None
    assert "JSON array of objects with the fields a" in f.__doc__
    assert json.loads(f(n=1, specs='[{"a": 2}]')) == {"ok": True} and calls[-1] == ("t1", {"n": 1, "specs": [{"a": 2}]})
    assert "not valid JSON" in f(n=1, specs="[oops") and len(calls) == 1
    gt = pytest.importorskip("google.genai.types")
    props = gt.FunctionDeclaration.from_callable_with_api_option(callable=f, api_option="GEMINI_API").parameters.properties
    assert props["specs"].type == "STRING" and props["xs"].type == "ARRAY" and props["n"].type == "INTEGER"


# ------------------------------------------------------------------ the real harness with a stand-in model
sdk = pytest.importorskip("google.antigravity", reason="the agent harness is an optional extra: uv sync --extra agents")


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
        d = run_chain_debug_agent(h.Gemini(model="fake", base_url=fm.url), "add8_sat", r, jd)
    assert fm.tools_offered() == ["chain_evidence"]              # none of the harness's own file / shell / web tools
    assert d["end_of_turn_sendbacks"] == 1 and d["steps"] == 1 and d["model"] == "fake"
    assert d["final_cause"] == "bitstream_not_loaded" and d["final_cause_source"] == "llm_validated" and d["validated"]
    sent_back = [m for req in fm.requests for m in req["messages"] if "ONLY one JSON object" in str(m.get("content"))]
    assert sent_back, "the end-of-turn check's reason must reach the model"


def test_an_off_menu_cause_is_sent_back(xx):
    r, jd = xx
    with FakeModel(chain([answer("cosmic_rays"), answer("bitstream_not_loaded")])) as fm:
        d = run_chain_debug_agent(h.Gemini(model="fake", base_url=fm.url), "add8_sat", r, jd)
    assert d["end_of_turn_sendbacks"] == 1 and d["final_cause_source"] == "llm_validated"


def test_code_still_decides_whether_the_evidence_supports_the_cause(xx):
    r, jd = xx
    with FakeModel(chain([answer("pins_not_connected")])) as fm:       # on the menu, keeps the contract, but wrong
        d = run_chain_debug_agent(h.Gemini(model="fake", base_url=fm.url), "add8_sat", r, jd)
    assert d["end_of_turn_sendbacks"] == 0
    assert d["final_cause"] == "bitstream_not_loaded" and d["final_cause_source"] == "rules_fallback (llm claim rejected)"


def test_send_backs_are_limited_then_the_rules_answer(xx):
    r, jd = xx
    with FakeModel(chain(["still thinking"])) as fm:
        d = run_chain_debug_agent(h.Gemini(model="fake", base_url=fm.url), "add8_sat", r, jd)
    assert d["end_of_turn_sendbacks"] == 2 and d["validation"]["json_ok"] is False
    assert d["final_cause"] == "bitstream_not_loaded" and d["final_cause_source"] == "rules_fallback (no usable llm answer)"


def busy_then(script, busy=("busy",)):
    def s(messages, request):
        if request.get("model") in busy:
            return ("http", 503, {"error": {"message": "This model is currently experiencing high demand.", "code": 503,
                                            "status": "UNAVAILABLE"}})
        return script(messages, request)
    return s


def test_a_busy_model_falls_back_to_the_next(xx):
    r, jd = xx
    with FakeModel(busy_then(chain([answer("bitstream_not_loaded")]))) as fm:
        d = run_chain_debug_agent(h.Gemini(model="busy", fallbacks=("fake",), base_url=fm.url, api_retries=0), "add8_sat", r, jd)
    assert d["model"] == "fake" and d["final_cause_source"] == "llm_validated"


def test_every_model_busy_means_the_rules_answer(xx):
    r, jd = xx
    with FakeModel(busy_then(chain([answer("bitstream_not_loaded")]), busy=("busy", "busy2"))) as fm:
        d = run_chain_debug_agent(h.Gemini(model="busy", fallbacks=("busy2",), base_url=fm.url, api_retries=0), "add8_sat", r, jd)
    assert d["stopped"] == "llm_unavailable" and "busy" in d["llm_error"]
    assert d["final_cause"] == "bitstream_not_loaded" and d["final_cause_source"] == "rules_fallback (llm unavailable)"


def test_a_wrong_key_is_raised_not_hidden(xx):
    r, jd = xx
    bad = lambda messages, request: ("http", 401, {"error": {"message": "API key not valid. Please pass a valid API key.",
                                                              "code": 401, "status": "UNAUTHENTICATED"}})
    with FakeModel(bad) as fm, pytest.raises(Exception, match="(?i)401|api key|unauthenticated"):
        run_chain_debug_agent(h.Gemini(model="fake", fallbacks=("fake2",), base_url=fm.url, api_retries=0), "add8_sat", r, jd)


def test_no_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        h.converse(h.Gemini(), "s", "t", [])


def test_the_tool_log(xx, tmp_path):
    r, jd = xx
    log_path = tmp_path / "log.jsonl"
    with FakeModel(chain(["no", answer("bitstream_not_loaded")])) as fm:
        h.converse(h.Gemini(model="fake", base_url=fm.url), "system", "task op=add8_sat",
                   [_evidence], log=h.jsonl_logger(log_path),
                   end_check=lambda text, used: h.check_json_answer(text, used, first_tool="chain_evidence", keys=KEYS,
                                                                   field_name="cause_id", allowed={"bitstream_not_loaded"}))
    kinds = [json.loads(line)["kind"] for line in log_path.read_text().splitlines()]
    assert kinds[0] == "start" and "tool" in kinds and kinds.count("end_of_turn") == 2


def _evidence(op: str = "add8_sat") -> str:
    """Stand-in evidence tool.

    Args:
        op: The operator.
    """
    return json.dumps({"op": op})


_evidence.__name__ = "chain_evidence"


def failing(script, models, status, message):
    def s(messages, request):
        if request.get("model") in models:
            return ("http", status, {"error": {"message": message, "code": status}})
        return script(messages, request)
    return s


@pytest.mark.parametrize("status,message", [(404, REAL_404), (429, REAL_429)], ids=["retired_model", "daily_quota_used_up"])
def test_a_model_that_cannot_answer_today_is_skipped_for_the_rest_of_the_run(xx, status, message):
    r, jd = xx
    with FakeModel(failing(chain([answer("bitstream_not_loaded")]), {"gone"}, status, message)) as fm:
        spec = h.Gemini(model="gone", fallbacks=("fake",), base_url=fm.url, api_retries=0)
        first = run_chain_debug_agent(spec, "add8_sat", r, jd)
        asked = sum(req.get("model") == "gone" for req in fm.requests)
        second = run_chain_debug_agent(spec, "add8_sat", r, jd)
    assert first["model"] == second["model"] == "fake" and first["final_cause_source"] == second["final_cause_source"] == "llm_validated"
    assert asked >= 1 and sum(req.get("model") == "gone" for req in fm.requests) == asked      # not asked again


def test_when_no_model_is_left_the_rules_answer(xx):
    r, jd = xx
    with FakeModel(failing(chain([answer("bitstream_not_loaded")]), {"a", "b"}, 429, REAL_429)) as fm:
        d = run_chain_debug_agent(h.Gemini(model="a", fallbacks=("b",), base_url=fm.url, api_retries=0), "add8_sat", r, jd)
    assert d["stopped"] == "llm_unavailable" and "daily free quota used up" in d["llm_error"]
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


def test_groq_is_the_last_resort_when_every_gemini_model_is_out(xx, monkeypatch):
    from x2c.agents import llm_client
    r, jd = xx
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setattr(llm_client, "GroqLLM", lambda *a, **k: _Replying([answer("bitstream_not_loaded")]))
    with FakeModel(failing(chain([answer("bitstream_not_loaded")]), {"a", "b"}, 429, REAL_429)) as fm:
        d = run_chain_debug_agent(h.Gemini(model="a", fallbacks=("b",), base_url=fm.url, api_retries=0), "add8_sat", r, jd)
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
    with FakeModel(failing(chain([answer("bitstream_not_loaded")]), {"a"}, 429, REAL_429)) as fm:
        spec = h.Gemini(model="a", fallbacks=(), base_url=fm.url, api_retries=0, groq_fallback=(why != "groq_fallback_off"))
        spec.fallbacks = ("a",)                       # only one model
        d = run_chain_debug_agent(spec, "add8_sat", r, jd)
    assert d["stopped"] == "llm_unavailable" and d["final_cause_source"] == "rules_fallback (llm unavailable)"
