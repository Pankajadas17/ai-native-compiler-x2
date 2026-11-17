"""Step 12: rate-limit (429) waits follow the provider's own hint; the default Groq fallback is a model that exists."""
import pytest

from agents import llm_client as lc


class Err(Exception):
    def __init__(self, status, msg, headers=None):
        super().__init__(msg)
        self.status_code = status
        if headers is not None:
            class R:  # minimal stand-in for an HTTP response
                pass
            self.response = R()
            self.response.headers = headers


GROQ_429 = ("Error code: 429 - Rate limit reached for model `openai/gpt-oss-120b` on tokens per minute (TPM): "
            "Limit 8000, Used 5850, Requested 2438. Please try again in 2.159999999s.")


def test_429_waits_the_hinted_time():
    assert lc._retry_delay(Err(429, GROQ_429), 0) == pytest.approx(2.66, abs=0.01)   # 2.16 + 0.5 margin


def test_429_milliseconds_hint():
    assert lc._retry_delay(Err(429, "Please try again in 250ms."), 0) == 1.0          # floor of 1s


def test_429_hint_is_capped():
    assert lc._retry_delay(Err(429, "Please try again in 600s."), 0) == 30.0


def test_429_retry_after_header():
    assert lc._retry_delay(Err(429, "slow down", {"retry-after": "4"}), 0) == 4.5


def test_429_without_hint_uses_backoff():
    assert [lc._retry_delay(Err(429, "slow down"), n) for n in range(3)] == [2, 4, 8]


def test_other_errors_keep_plain_backoff():
    # a 503 that happens to contain the words must NOT use the hint
    assert [lc._retry_delay(Err(503, GROQ_429), n) for n in range(3)] == [2, 4, 8]


def test_call_with_fallback_sleeps_the_hint_and_succeeds():
    calls, slept = [], []

    def attempt(model):
        calls.append(model)
        if len(calls) == 1:
            raise Err(429, GROQ_429)
        return "ok"

    model, resp = lc.call_with_fallback(["m1"], attempt, 3, slept.append)
    assert (model, resp) == ("m1", "ok")
    assert slept == [pytest.approx(2.66, abs=0.01)]


def test_deadline_uses_the_real_delay():
    now = [0.0]
    with pytest.raises(lc.LLMUnavailable):
        lc.call_with_fallback(["m1"], lambda m: (_ for _ in ()).throw(Err(429, "Please try again in 20s.")),
                              3, lambda s: None, deadline=10.0, clock=lambda: now[0])


def test_default_groq_fallback_is_gpt_oss_20b(monkeypatch):
    monkeypatch.delenv("BM_GROQ_FALLBACKS", raising=False)
    monkeypatch.delenv("BM_GROQ_MODEL", raising=False)

    class FakeClient:
        pass

    g = lc.GroqLLM(client=FakeClient())
    assert g.model == "openai/gpt-oss-120b"
    assert g.fallbacks == ["openai/gpt-oss-20b"]
    assert not any(p in m for m in g.fallbacks for p in lc.NON_CHAT_PATTERNS)
