"""Step 15: say WHICH rate limit was hit, and do not sit through waits that will not clear soon."""
import pytest

from agents import llm_client as lc


class Err(Exception):
    def __init__(self, status, msg):
        super().__init__(msg)
        self.status_code = status


TPM = ("Error code: 429 - Rate limit reached for model `openai/gpt-oss-120b` in organization `org_x` service tier `on_demand` "
       "on tokens per minute (TPM): Limit 8000, Used 5850, Requested 2438. Please try again in 2.159999999s.")
TPD = ("Error code: 429 - Rate limit reached for model `openai/gpt-oss-120b` on tokens per day (TPD): Limit 200000, "
       "Used 199000, Requested 2400. Please try again in 9m12.5s.")


@pytest.mark.parametrize("text,seconds", [("try again in 2.16s", 2.16), ("try again in 9m12.5s", 552.5),
                                          ("try again in 1h2m", 3720), ("try again in 250ms", 0.25),
                                          ("try again in 1m30s.", 90), ("nothing here", None)])
def test_duration_parsing(text, seconds):
    got = lc._retry_hint_seconds(Err(429, text))
    assert got == pytest.approx(seconds) if seconds is not None else got is None


def test_brief_names_the_limit_and_the_numbers():
    b = lc._brief(Err(429, TPM))
    assert "TPM" in b and "limit 8000" in b and "used 5850" in b and "requested 2438" in b and "2.16" in b
    assert "TPD" in lc._brief(Err(429, TPD))


def test_brief_for_other_errors_is_unchanged():
    assert lc._brief(Err(503, "overloaded")).startswith("503: overloaded")


def test_short_tpm_wait_is_still_retried():
    assert not lc._is_quota_exhausted(Err(429, TPM))


def test_daily_limit_or_very_long_wait_is_exhausted():
    assert lc._is_quota_exhausted(Err(429, TPD))
    assert lc._is_quota_exhausted(Err(429, "Rate limit. Please try again in 120s."))
    assert lc._is_quota_exhausted(Err(429, "requests per day exceeded"))
    assert not lc._is_quota_exhausted(Err(503, TPD))


def test_exhausted_model_is_skipped_without_sleeping_and_fallback_is_used():
    slept, seen, tried = [], [], []

    def attempt(m):
        tried.append(m)
        if m == "big":
            raise Err(429, TPD)
        return "ok"

    model, resp = lc.call_with_fallback(["big", "small"], attempt, 3, slept.append, notify=seen.append)
    assert (model, resp) == ("small", "ok") and tried == ["big", "small"] and slept == []
    assert any("not worth waiting" in t and "TPD" in t for t in seen)


def test_all_models_exhausted_fails_fast():
    slept = []
    with pytest.raises(lc.LLMUnavailable):
        lc.call_with_fallback(["a", "b"], lambda m: (_ for _ in ()).throw(Err(429, TPD)), 3, slept.append)
    assert slept == []
