"""Step 13: waiting must be visible ([llm] progress lines) and no single request may outlive the time budget."""
import pytest

from x2c.agents import llm_client as lc


class Err(Exception):
    def __init__(self, status, msg):
        super().__init__(msg)
        self.status_code = status


def test_retry_is_announced():
    seen, calls = [], []

    def attempt(m):
        calls.append(m)
        if len(calls) == 1:
            raise Err(429, "Rate limit. Please try again in 3s.")
        return "ok"

    lc.call_with_fallback(["m1"], attempt, 3, lambda s: None, notify=seen.append)
    assert len(seen) == 1 and "m1" in seen[0] and "retry 1/3" in seen[0] and "3.5s" in seen[0]


def test_skip_and_give_up_are_announced():
    seen = []

    def attempt(m):
        if m == "audio":
            raise Err(400, "The model `audio` does not support chat completions")
        raise Err(503, "overloaded")

    with pytest.raises(lc.LLMUnavailable):
        lc.call_with_fallback(["audio", "m2"], attempt, 1, lambda s: None, notify=seen.append)
    assert any("audio: skipped" in t for t in seen)
    assert any("m2: giving up after 2 attempts" in t for t in seen)


def test_without_notify_nothing_is_printed(capsys):
    lc.call_with_fallback(["m1"], lambda m: "ok", 1, lambda s: None)
    assert capsys.readouterr().out == ""


def test_print_progress_flushes_a_labelled_line(capsys):
    lc.print_progress("hello")
    assert "[llm] hello" in capsys.readouterr().out


def test_compound_models_are_not_chat_tool_models():
    kept, dropped = lc.filter_chat_models(["openai/gpt-oss-120b", "groq/compound", "whisper-large-v3"])
    assert kept == ["openai/gpt-oss-120b"] and set(dropped) == {"groq/compound", "whisper-large-v3"}


def test_groq_request_timeout_shrinks_with_the_budget(monkeypatch):
    monkeypatch.delenv("BM_GROQ_FALLBACKS", raising=False)
    g = lc.GroqLLM(client=object())
    assert g._timeout() == lc.REQUEST_TIMEOUT_S              # no conversation started yet
    now = [100.0]
    g.clock, g._deadline = (lambda: now[0]), 100.0 + 10.0
    assert g._timeout() == 10.0                              # only 10 s of budget left
    now[0] = 109.5
    assert g._timeout() == 1.0                               # never below 1 s
    now[0] = 500.0
    assert g._timeout() == 1.0


def test_groq_uses_notify_and_timeout_in_the_real_call():
    """The Groq client must receive timeout=..., and retries must reach the notify sink."""
    seen_kwargs = []

    class Msg:
        content, tool_calls = "done", None

    class Resp:
        choices = [type("C", (), {"message": Msg()})()]

    class FakeCompletions:
        def create(self, **kw):
            seen_kwargs.append(kw)
            if len(seen_kwargs) == 1:
                raise Err(429, "Please try again in 1s.")
            return Resp()

    class FakeClient:
        chat = type("X", (), {"completions": FakeCompletions()})()

    g = lc.GroqLLM(client=FakeClient(), sleep=lambda s: None)
    lines = []
    g.notify = lines.append
    act = g.start("goal", [{"name": "t", "description": "d", "input_schema": {"type": "object", "properties": {}}}])
    assert act.kind == "final" and act.text == "done"
    assert all("timeout" in kw and kw["timeout"] > 0 for kw in seen_kwargs)
    assert len(lines) == 1 and "retry 1/3" in lines[0]
