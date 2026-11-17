"""Step 14: stay under the provider's tokens-per-minute limit instead of hitting 429s, and send fewer tokens per request."""
import json

import pytest

from agents import llm_client as lc


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    lc._PACERS.clear()
    for k in ("BM_GROQ_TPM", "BM_GROQ_REASONING", "BM_GROQ_FALLBACKS", "BM_GROQ_MODEL"):
        monkeypatch.delenv(k, raising=False)
    yield
    lc._PACERS.clear()


class FakeTime:
    def __init__(self):
        self.now, self.slept = 0.0, []

    def clock(self):
        return self.now

    def sleep(self, s):
        self.slept.append(s)
        self.now += s


# ---------------- TokenPacer ----------------
def test_pacer_does_not_wait_when_there_is_room():
    t = FakeTime()
    p = lc.TokenPacer(7000, t.clock, t.sleep)
    p.record(3000)
    assert p.wait(2000) == 0 and t.slept == []


def test_pacer_waits_until_old_tokens_leave_the_window():
    t = FakeTime()
    p = lc.TokenPacer(7000, t.clock, t.sleep)
    p.record(6000)                       # at t=0
    t.now = 10.0
    waited = p.wait(2000)                # 6000+2000 > 7000: must wait until t=60
    assert waited == pytest.approx(50.0, abs=1.0) and t.now >= 60.0


def test_pacer_tells_the_user_once(capsys):
    t = FakeTime()
    p = lc.TokenPacer(7000, t.clock, t.sleep)
    p.record(7000)
    seen = []
    p.wait(1000, seen.append)
    assert len(seen) == 1 and "pacing" in seen[0] and "429" in seen[0]


def test_pacer_limit_zero_is_off():
    t = FakeTime()
    p = lc.TokenPacer(0, t.clock, t.sleep)
    p.record(10 ** 6)
    assert p.wait(10 ** 6) == 0


def test_pacer_never_blocks_forever_on_an_oversized_request():
    t = FakeTime()
    p = lc.TokenPacer(7000, t.clock, t.sleep)
    assert p.wait(50_000) == 0           # nothing recorded yet: send it, let the provider decide


def test_one_pacer_per_model_is_shared_between_instances():
    assert lc._pacer_for("m1") is lc._pacer_for("m1")
    assert lc._pacer_for("m1") is not lc._pacer_for("m2")


def test_tpm_env_changes_the_limit(monkeypatch):
    monkeypatch.setenv("BM_GROQ_TPM", "5000")
    assert lc._pacer_for("m1").limit == 5000


# ---------------- compact results ----------------
def test_compact_drops_nulls_everywhere_but_keeps_zero_false_empty():
    r = {"a": None, "b": 0, "c": False, "d": [], "e": {"x": None, "y": 1}, "f": [{"z": None, "w": "k"}]}
    assert lc._compact(r) == {"b": 0, "c": False, "d": [], "e": {"y": 1}, "f": [{"w": "k"}]}


# ---------------- GroqLLM request building ----------------
class Msg:
    content, tool_calls = "done", None


class Resp:
    choices = [type("C", (), {"message": Msg()})()]

    def __init__(self, total=None):
        self.usage = type("U", (), {"total_tokens": total})() if total else None


def make_groq(create, **kw):
    class Comp:
        pass
    Comp.create = staticmethod(create)
    client = type("Cl", (), {"chat": type("Ch", (), {"completions": Comp()})()})()
    t = FakeTime()
    g = lc.GroqLLM(client=client, sleep=t.sleep, **kw)
    g.clock = t.clock
    g.notify = lambda m: None
    return g, t


SCHEMA = [{"name": "t", "description": "d", "input_schema": {"type": "object", "properties": {}}}]


def test_gpt_oss_gets_low_reasoning_effort_by_default():
    seen = []
    g, _ = make_groq(lambda **kw: (seen.append(kw), Resp(500))[1])
    g.start("goal", SCHEMA)
    assert seen[0]["extra_body"] == {"reasoning_effort": "low"}


def test_reasoning_effort_can_be_changed_or_turned_off(monkeypatch):
    seen = []
    g, _ = make_groq(lambda **kw: (seen.append(kw), Resp(500))[1])
    monkeypatch.setenv("BM_GROQ_REASONING", "medium")
    g.start("goal", SCHEMA)
    monkeypatch.setenv("BM_GROQ_REASONING", "off")
    g.start("goal", SCHEMA)
    assert seen[0]["extra_body"] == {"reasoning_effort": "medium"} and "extra_body" not in seen[1]


def test_other_models_never_get_reasoning_effort():
    seen = []
    g, _ = make_groq(lambda **kw: (seen.append(kw), Resp(500))[1], model="llama-3.3-70b-versatile")
    g.start("goal", SCHEMA)
    assert "extra_body" not in seen[0]


def test_if_the_provider_rejects_reasoning_effort_we_retry_without_it_and_remember():
    class BadReq(Exception):
        status_code = 400

    seen = []

    def create(**kw):
        seen.append(kw)
        if "extra_body" in kw:
            raise BadReq("unknown parameter reasoning_effort")
        return Resp(500)

    g, _ = make_groq(create)
    g.start("goal", SCHEMA)
    g.start("goal", SCHEMA)
    assert ["extra_body" in kw for kw in seen] == [True, False, False]       # rejected once, never sent again


def test_real_usage_is_recorded_and_next_request_waits_when_the_window_is_full(monkeypatch):
    monkeypatch.setenv("BM_GROQ_TPM", "7000")
    calls = []
    g, t = make_groq(lambda **kw: (calls.append(t.now), Resp(6900))[1])
    g.start("goal", SCHEMA)                       # records 6900 real tokens at t=0
    g.start("goal", SCHEMA)                       # estimate pushes it over 7000: must wait for the window to empty
    assert t.slept and calls[1] - calls[0] >= 59.0


def test_missing_usage_is_not_counted_so_fakes_never_wait():
    g, t = make_groq(lambda **kw: Resp(None))
    for _ in range(30):
        g.start("goal", SCHEMA)
    assert t.slept == []


def test_tool_results_are_sent_compactly():
    sent = []

    class TC:
        id = "c1"

        class function:
            name, arguments = "t", "{}"

    class MsgTool:
        content, tool_calls = "", [TC]

    class RespTool:
        choices = [type("C", (), {"message": MsgTool()})()]
        usage = None

    def create(**kw):
        sent.append(kw["messages"])
        return RespTool() if len(sent) == 1 else Resp(500)

    g, _ = make_groq(create)
    g.start("goal", SCHEMA)
    g.send_tool_results([("t", {"error": None, "n": 3})])
    tool_msg = [m for m in sent[1] if m["role"] == "tool"][0]
    assert tool_msg["content"] == '{"n":3}'
    assert json.loads(tool_msg["content"]) == {"n": 3}


def test_pacing_is_off_by_default():
    assert lc._pacer_for("m1").limit == 0
    g, t = make_groq(lambda **kw: Resp(6900))
    g.start("goal", SCHEMA)
    g.start("goal", SCHEMA)
    assert t.slept == []
