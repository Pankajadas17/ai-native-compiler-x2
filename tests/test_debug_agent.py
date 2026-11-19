import json, os
import pytest
from x2c.agents.debug_agent import run_debug_agent
from x2c.agents.llm_client import Action, MockDebugLLM
from x2c.verification import debug_tools, tools, workloads
from tests import faults

BENCH = [("sign_flip", "matmul", faults.sign_flip, "output_sign_inverted"),
         ("off_by_one", "matmul", faults.off_by_one, "constant_offset"),
         ("doubled", "matmul", faults.doubled, "output_scaled"),
         ("transposed_output", "matmul", faults.transposed_output, "output_transposed"),
         ("transposed_weights", "matmul", faults.transposed_weights, "weight_orientation_transposed"),
         ("drop_fold", "workload", faults.drop_last_fold, "mapper_dropped_fold"),
         ("drop_channel", "workload", faults.drop_last_output_channel, "mapper_dropped_output_channel")]


@pytest.fixture
def make_report():
    made = []
    debug_tools.set_blind_mode(False)
    def make(label, kind, fault):
        path = f"reports/_t_{label}.json"; made.append(path)
        if kind == "matmul":
            tools.run_suite(report_path=path, x2_fn=fault)
        else:
            tools.run_workloads(report_path=path, mapping_transform=fault)
        return path
    yield make
    tools.set_x2_backend(None); workloads.set_mapping_transform(None)
    for p in made:
        if os.path.exists(p): os.remove(p)


@pytest.mark.parametrize("label,kind,fault,truth", BENCH)
def test_agent_finds_each_injected_bug(label, kind, fault, truth, make_report, tmp_path):
    path = make_report(label, kind, fault)
    out = run_debug_agent(MockDebugLLM(), path, log_path=str(tmp_path / "l.jsonl"))
    assert out["diagnosis"]["cause_id"] == truth, out
    assert out["validated"] and out["agrees_with_rules"]


class Scripted:
    def __init__(self, answer): self.answer = answer
    def start(self, goal, schemas, system_prompt=None):
        self.path = goal.split("report_path=")[1]
        return Action("tool_calls", calls=[("diagnose_report", {"report_path": self.path})])
    def send_tool_results(self, r): return Action("final", text=self.answer)


def test_cause_inconsistent_with_evidence_is_flagged(make_report, tmp_path):
    path = make_report("off_by_one", "matmul", faults.off_by_one)
    wrong = json.dumps({"cause_id": "output_sign_inverted", "confidence": "high", "explanation": "x",
                        "proposed_fix": "y", "next_check": "z"})
    out = run_debug_agent(Scripted(wrong), path, log_path=str(tmp_path / "l.jsonl"))
    assert out["validation"]["cause_known"] and not out["validation"]["supported_by_evidence"] and not out["validated"]
    assert out["agrees_with_rules"] is False


def test_invented_cause_and_non_json_are_rejected(make_report, tmp_path):
    path = make_report("sign_flip", "matmul", faults.sign_flip)
    inv = run_debug_agent(Scripted('{"cause_id": "gremlins_in_the_chip"}'), path, log_path=str(tmp_path / "a.jsonl"))
    assert not inv["validation"]["cause_known"] and not inv["validated"]
    nj = run_debug_agent(Scripted("it is probably the sign"), path, log_path=str(tmp_path / "b.jsonl"))
    assert nj["diagnosis"] is None and not nj["validated"] and nj["raw_answer"]


def test_unknown_is_an_allowed_honest_answer(make_report, tmp_path):
    path = make_report("sign_flip", "matmul", faults.sign_flip)
    out = run_debug_agent(Scripted('{"cause_id": "unknown", "confidence": "low"}'), path, log_path=str(tmp_path / "l.jsonl"))
    assert out["validated"] is True and out["agrees_with_rules"] is False


def test_debug_tools_reject_bad_calls():
    assert "error" in debug_tools.call_debug_tool("diagnose_report", {"report_path": "../../etc/passwd"})
    assert "error" in debug_tools.call_debug_tool("diagnose_report", {"report_path": "C:/x.json"})
    assert "error" in debug_tools.call_debug_tool("rm_rf", {})
    assert "error" in debug_tools.call_debug_tool("check_hypothesis", {"report_path": "reports/x.json", "evil": 1})


def test_hypothesis_and_reproduce_tools(make_report):
    path = make_report("sign_flip", "matmul", faults.sign_flip)
    ok = debug_tools.check_hypothesis(path, "fixed_2x2", "negated")
    assert ok["explains_failure"] is True
    assert debug_tools.check_hypothesis(path, "fixed_2x2", "transposed")["explains_failure"] is False
    assert "error" in debug_tools.check_hypothesis(path, "fixed_2x2", "magic")
    tools.set_x2_backend(faults.sign_flip)                           # same bug active -> reproducible
    assert debug_tools.reproduce_failure(path, "fixed_2x2")["reproduced"] is True
    tools.set_x2_backend(None)                                       # bug "fixed" -> no longer reproduces
    assert debug_tools.reproduce_failure(path, "fixed_2x2")["reproduced"] is False
    assert "error" in debug_tools.reproduce_failure(path, "no_such_test")


# ---------- Groq "final answer as a fake tool call" recovery ----------
import types as _types
from x2c.agents.llm_client import GroqLLM as _GroqLLM


class GroqLLM(_GroqLLM):
    """GroqLLM's own mechanics (protocol, retries, fallbacks, recoveries) without the end-of-turn check's replies;
    those are tested in test_agent.py::test_groq_end_of_turn_reply and tests/test_agent_harness.py."""
    send_text = None
from tests.test_agent import _GroqClient, _gmsg


class _ToolUseFailed(Exception):
    def __init__(self, failed_generation):
        super().__init__("Error code: 400 - tool_use_failed")
        self.status_code = 400
        self.body = {"error": {"code": "tool_use_failed", "message": "bad", "failed_generation": failed_generation}}


def _debug_client(path, failure):
    return _GroqClient([_gmsg(calls=[("c1", "diagnose_report", json.dumps({"report_path": path}))]), failure])


def test_groq_pseudo_tool_answer_is_recovered_and_still_validated(make_report, tmp_path):
    path = make_report("sign_flip", "matmul", faults.sign_flip)
    answer = {"cause_id": "output_sign_inverted", "confidence": "high", "explanation": "e", "proposed_fix": "f", "next_check": "n"}
    fail = _ToolUseFailed(json.dumps({"name": "json", "arguments": answer}))
    llm = GroqLLM(model="m", client=_debug_client(path, fail), fallback_models=[], sleep=lambda s: None)
    out = run_debug_agent(llm, path, log_path=str(tmp_path / "l.jsonl"))
    assert out["diagnosis"]["cause_id"] == "output_sign_inverted" and out["validated"] and out["llm_recovered_answers"] == 1


def test_recovered_wrong_answer_is_still_rejected_by_validator(make_report, tmp_path):
    path = make_report("off_by_one", "matmul", faults.off_by_one)
    fail = _ToolUseFailed(json.dumps({"name": "json", "arguments": {"cause_id": "output_sign_inverted"}}))
    llm = GroqLLM(model="m", client=_debug_client(path, fail), fallback_models=[], sleep=lambda s: None)
    out = run_debug_agent(llm, path, log_path=str(tmp_path / "l.jsonl"))
    assert out["llm_recovered_answers"] == 1 and not out["validated"]


def test_unrecoverable_tool_use_failures_are_recorded_not_guessed(make_report, tmp_path):
    path = make_report("sf_unrec", "matmul", faults.sign_flip)
    for fg in (json.dumps({"name": "diagnose_report", "arguments": "garbage"}),   # a REAL tool, malformed: we do not guess its answer
               "not json at all"):
        llm = GroqLLM(model="m", client=_debug_client(path, _ToolUseFailed(fg)), fallback_models=[], sleep=lambda s: None)
        out = run_debug_agent(llm, path, log_path=str(tmp_path / "l.jsonl"))
        assert out["diagnosis"] is None and out["stopped"] == "llm_error"        # nothing was invented from the bad text
        assert out["final_cause"] == "output_sign_inverted" and "llm unavailable" in out["final_cause_source"]


def test_recovery_works_on_a_real_groq_sdk_error_object():
    httpx = pytest.importorskip("httpx"); groq = pytest.importorskip("groq")
    fg = json.dumps({"name": "json", "arguments": {"cause_id": "constant_offset", "confidence": "high"}})
    payload = {"error": {"message": "Tool call validation failed", "type": "invalid_request_error",
                         "code": "tool_use_failed", "failed_generation": fg}}
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    err = groq.Groq(api_key="x")._make_status_error_from_response(httpx.Response(400, json=payload, request=req))
    llm = GroqLLM(model="m", client=_GroqClient([]), fallback_models=[])
    llm._tools = [{"type": "function", "function": {"name": "diagnose_report"}}]
    assert json.loads(llm._recover_answer(err))["cause_id"] == "constant_offset"


# ---------- stricter rules, blind mode, unseen bugs ----------
UNSEEN = [("clipped_output", faults.clipped_output), ("stale_accumulator", faults.stale_accumulator)]


@pytest.mark.parametrize("label,fault", UNSEEN)
def test_rules_say_unknown_for_unseen_bugs(label, fault, make_report):
    path = make_report("u_" + label, "matmul", fault)
    assert debug_tools.diagnose_report(path)["rule_based_candidates"] == []


@pytest.mark.parametrize("label,fault", UNSEEN)
def test_overconfident_llm_on_unseen_bug_is_not_validated(label, fault, make_report, tmp_path):
    path = make_report("u_" + label, "matmul", fault)
    for claim in ("mapper_dropped_output_channel", "weight_orientation_transposed"):
        out = run_debug_agent(Scripted(json.dumps({"cause_id": claim, "confidence": "high"})), path,
                              log_path=str(tmp_path / "l.jsonl"))
        assert not out["validated"] and not out["validation"]["supported_by_evidence"]
    honest = run_debug_agent(Scripted(json.dumps({"cause_id": "unknown", "confidence": "low"})), path,
                             log_path=str(tmp_path / "m.jsonl"))
    assert honest["validated"]


def test_mock_agent_answers_unknown_on_unseen_bug(make_report, tmp_path):
    path = make_report("u_clip", "matmul", faults.clipped_output)
    out = run_debug_agent(MockDebugLLM(), path, log_path=str(tmp_path / "l.jsonl"))
    assert out["diagnosis"]["cause_id"] == "unknown" and out["validated"]


def test_blind_mode_hides_the_answer_key(make_report):
    path = make_report("blind", "matmul", faults.sign_flip)
    debug_tools.set_blind_mode(True)
    try:
        d = debug_tools.diagnose_report(path)
        assert "rule_based_candidates" not in d and "candidate_details" not in d
        assert d["failure_categories"] and d["examples"]          # the evidence is still there
    finally:
        debug_tools.set_blind_mode(False)
    assert "rule_based_candidates" in debug_tools.diagnose_report(path)


def test_prompt_contains_cause_menu_but_no_ranking():
    from x2c.agents.debug_agent import DEBUG_PROMPT
    assert "mapper_dropped_fold:" in DEBUG_PROMPT and "unknown:" in DEBUG_PROMPT


def test_mapper_causes_never_proposed_for_plain_matmul_reports(make_report):
    path = make_report("sf", "matmul", faults.sign_flip)
    assert not [c for c in debug_tools.diagnose_report(path)["rule_based_candidates"] if c.startswith("mapper_")]


# ---------- system-level decision: validated LLM > rules fallback > unknown ----------
def _claim(c): return json.dumps({"cause_id": c, "confidence": "high"})


def test_system_overrides_overconfident_llm_on_unseen_bug(make_report, tmp_path):
    path = make_report("u_stale", "matmul", faults.stale_accumulator)
    out = run_debug_agent(Scripted(_claim("mapper_dropped_output_channel")), path, log_path=str(tmp_path / "l.jsonl"))
    assert out["diagnosis"]["cause_id"] == "mapper_dropped_output_channel"      # what the raw LLM said
    assert out["final_cause"] == "unknown" and out["final_cause_source"].startswith("unknown_fallback")


def test_system_falls_back_to_rules_when_llm_is_wrong_on_a_known_bug(make_report, tmp_path):
    path = make_report("off1", "matmul", faults.off_by_one)
    out = run_debug_agent(Scripted(_claim("output_sign_inverted")), path, log_path=str(tmp_path / "l.jsonl"))
    assert out["final_cause"] == "constant_offset" and out["final_cause_source"].startswith("rules_fallback")


def test_system_uses_llm_answer_when_validated_and_honest_unknown_is_confirmed(make_report, tmp_path):
    p1 = make_report("sf", "matmul", faults.sign_flip)
    ok = run_debug_agent(Scripted(_claim("output_sign_inverted")), p1, log_path=str(tmp_path / "a.jsonl"))
    assert ok["final_cause"] == "output_sign_inverted" and ok["final_cause_source"] == "llm_validated"
    p2 = make_report("u_clip", "matmul", faults.clipped_output)
    unk = run_debug_agent(Scripted(_claim("unknown")), p2, log_path=str(tmp_path / "b.jsonl"))
    assert unk["final_cause"] == "unknown" and unk["final_cause_source"] == "llm_unknown_confirmed_by_rules"


def test_no_json_falls_back_to_rules(make_report, tmp_path):
    path = make_report("sf2", "matmul", faults.sign_flip)
    out = run_debug_agent(Scripted("probably the sign"), path, log_path=str(tmp_path / "l.jsonl"))
    assert out["final_cause"] == "output_sign_inverted" and "no usable llm answer" in out["final_cause_source"]


def test_report_kind_and_menu_applicability(make_report):
    from x2c.agents.debug_agent import DEBUG_PROMPT
    assert debug_tools.diagnose_report(make_report("k1", "matmul", faults.sign_flip))["report_kind"].startswith("matmul_suite")
    assert debug_tools.diagnose_report(make_report("k2", "workload", faults.drop_last_fold))["report_kind"].startswith("workload_report")
    assert "applies only to" in DEBUG_PROMPT and "NEVER for plain matmul reports" in DEBUG_PROMPT


# ---------- last-chance turn (force_final) and holdout bugs ----------
@pytest.mark.parametrize("name", ["swap_first_two_columns", "int8_wraparound", "zero_last_row"])
def test_rules_say_unknown_for_holdout_bugs(name, make_report):
    path = make_report("h_" + name, "matmul", getattr(faults, name))
    assert debug_tools.diagnose_report(path)["rule_based_candidates"] == []


def test_groq_last_round_forbids_tools_and_agent_gets_an_answer(make_report, tmp_path):
    path = make_report("sf_last", "matmul", faults.sign_flip)
    call = lambda cid: _gmsg(calls=[(cid, "diagnose_report", json.dumps({"report_path": path}))])
    answer = json.dumps({"cause_id": "output_sign_inverted", "confidence": "high"})
    c = _GroqClient([call("c1"), call("c2"), _gmsg(content=answer)])
    out = run_agent_for_test(GroqLLM(model="m", client=c, fallback_models=[], sleep=lambda s: None), path, tmp_path)
    assert [r["tool_choice"] for r in c.requests] == ["auto", "auto", "none"]
    assert out["stopped"] == "finished" and out["final_cause"] == "output_sign_inverted"


def run_agent_for_test(llm, path, tmp_path, max_steps=2):
    return run_debug_agent(llm, path, max_steps=max_steps, log_path=str(tmp_path / "l.jsonl"))


# ---------- LLM outage: 404 semantics and graceful degradation ----------
class _Http(Exception):
    def __init__(self, status): super().__init__(f"http {status}"); self.status_code = status


def test_404_on_a_model_that_worked_before_is_retried_not_skipped():
    from x2c.agents.llm_client import call_with_fallback
    calls, naps = [], []
    def attempt(m):
        calls.append(m)
        if m == "good" and calls.count("good") == 2:          # the 2nd call to a previously-good model 404s once
            raise _Http(404)
        return "ok"
    good = set()
    assert call_with_fallback(["good", "other"], attempt, 3, naps.append, good)[0] == "good"
    out = call_with_fallback(["good", "other"], attempt, 3, naps.append, good)   # 404 once, then success
    assert out[0] == "good" and naps == [2]


def test_404_on_a_model_that_never_worked_skips_immediately():
    from x2c.agents.llm_client import call_with_fallback
    naps, seen = [], []
    def attempt(m):
        seen.append(m)
        if m == "retired": raise _Http(404)
        return "ok"
    assert call_with_fallback(["retired", "alive"], attempt, 3, naps.append, set())[0] == "alive"
    assert naps == [] and seen == ["retired", "alive"]


def test_llm_outage_still_yields_a_rule_based_system_answer(make_report, tmp_path):
    path = make_report("sf_out", "matmul", faults.sign_flip)
    c = _GroqClient([_Http(404)] * 20)
    llm = GroqLLM(model="a", client=c, fallback_models=["b"], retries=1, sleep=lambda s: None)
    out = run_debug_agent(llm, path, log_path=str(tmp_path / "l.jsonl"))
    assert out["stopped"] == "llm_unavailable" and out["llm_error"]
    assert out["final_cause"] == "output_sign_inverted" and "llm unavailable" in out["final_cause_source"]
    path2 = make_report("clip_out", "matmul", faults.clipped_output)
    llm2 = GroqLLM(model="a", client=_GroqClient([_Http(503)] * 20), fallback_models=[], retries=0, sleep=lambda s: None)
    out2 = run_debug_agent(llm2, path2, log_path=str(tmp_path / "m.jsonl"))
    assert out2["final_cause"] == "unknown" and "llm unavailable" in out2["final_cause_source"]


def test_auth_errors_still_raise_instead_of_being_hidden(make_report, tmp_path):
    path = make_report("sf_auth", "matmul", faults.sign_flip)
    llm = GroqLLM(model="a", client=_GroqClient([_Http(401)]), fallback_models=[], sleep=lambda s: None)
    with pytest.raises(_Http):
        run_debug_agent(llm, path, log_path=str(tmp_path / "l.jsonl"))


# ---------- time budget, request timeouts, live progress ----------
def test_time_budget_stops_endless_retries():
    from x2c.agents.llm_client import LLMUnavailable, call_with_fallback
    clock = {"t": 0.0}
    def fake_clock(): return clock["t"]
    def sleeper(s): clock["t"] += s
    def attempt(m): clock["t"] += 30; raise _Http(503)           # every call burns 30 s and fails
    with pytest.raises(LLMUnavailable) as e:
        call_with_fallback(["a", "b"], attempt, 5, sleeper, set(), deadline=100.0, clock=fake_clock)
    assert "time budget exceeded" in str(e.value) and clock["t"] < 140      # stopped near the 100 s budget


def test_groq_client_gets_explicit_timeout_and_no_hidden_retries(monkeypatch):
    groq = pytest.importorskip("groq")
    monkeypatch.setenv("GROQ_API_KEY", "dummy")
    llm = GroqLLM()
    assert llm.client.max_retries == 0 and float(llm.client.timeout) == 45.0


def test_budget_is_enforced_per_conversation(make_report, tmp_path):
    path = make_report("sf_budget", "matmul", faults.sign_flip)
    clock = {"t": 0.0}
    class Slow(_GroqClient):
        def _create(self, **kw):
            clock["t"] += 50; raise _Http(503)
    llm = GroqLLM(model="a", client=Slow([]), fallback_models=["b"], retries=5, sleep=lambda s: clock.__setitem__("t", clock["t"] + s))
    llm.clock = lambda: clock["t"]; llm.time_budget_s = 120
    out = run_debug_agent(llm, path, log_path=str(tmp_path / "l.jsonl"))
    assert out["stopped"] == "llm_unavailable" and "time budget" in out["llm_error"]
    assert out["final_cause"] == "output_sign_inverted"            # the rule-based system still answers


# ---------- candidate-model sanity (audio models, terms acceptance, over-long chains) ----------
class _Bad400(Exception):
    def __init__(self, msg): super().__init__(msg); self.status_code = 400


def test_non_chat_models_are_dropped_and_the_chain_is_capped():
    from x2c.agents.llm_client import filter_chat_models
    llm = GroqLLM(model="openai/gpt-oss-120b", client=_GroqClient([]),
                  fallback_models=["whisper-large-v3", "canopylabs/orpheus-v1-english", "openai/gpt-oss-20b", "x", "y"])
    assert llm.fallbacks == ["openai/gpt-oss-20b", "x"] and len(llm.notes) == 2
    assert "whisper" in llm.describe() or "ignored" in llm.describe()
    assert filter_chat_models(["a", "whisper-large-v3-turbo"]) == (["a"], ["whisper-large-v3-turbo"])


def test_explicit_audio_model_is_rejected_with_a_clear_message():
    with pytest.raises(ValueError, match="not a chat"):
        GroqLLM(model="whisper-large-v3-turbo", client=_GroqClient([]))


@pytest.mark.parametrize("msg", ["The model `x` does not support chat completions",
                                 "The model `y` requires terms acceptance. Please have an admin accept",
                                 "`tool calling` is not supported with this model"])
def test_400_model_unsuitable_skips_to_next_model_instead_of_crashing(msg, make_report, tmp_path):
    path = make_report("sf_unsuit", "matmul", faults.sign_flip)
    answer = json.dumps({"cause_id": "output_sign_inverted", "confidence": "high"})
    seen = []
    class C(_GroqClient):
        def _create(self, **kw):
            seen.append(kw["model"])
            if kw["model"] == "bad-primary": raise _Bad400(msg)
            return super()._create(**kw)
    c = C([_gmsg(calls=[("c1", "diagnose_report", json.dumps({"report_path": path}))]), _gmsg(content=answer)])
    llm = GroqLLM(model="bad-primary", client=c, fallback_models=["good"], sleep=lambda s: None)
    out = run_debug_agent(llm, path, log_path=str(tmp_path / "l.jsonl"))
    assert out["final_cause"] == "output_sign_inverted" and llm.model_used == "good" and seen[0] == "bad-primary"


def test_other_provider_errors_are_recorded_and_the_rules_still_answer(make_report, tmp_path):
    path = make_report("sf_400", "matmul", faults.sign_flip)
    llm = GroqLLM(model="m", client=_GroqClient([_Bad400("some unrelated bad request")]), fallback_models=["n"], sleep=lambda s: None)
    out = run_debug_agent(llm, path, log_path=str(tmp_path / "l.jsonl"))
    assert out["stopped"] == "llm_error" and "unrelated bad request" in out["llm_error"]       # visible, not hidden
    assert out["final_cause"] == "output_sign_inverted" and "llm unavailable" in out["final_cause_source"]


def test_programming_errors_are_never_swallowed(make_report, tmp_path):
    path = make_report("sf_prog", "matmul", faults.sign_flip)
    class Broken(_GroqClient):
        def _create(self, **kw): raise TypeError("a real bug in our code")
    llm = GroqLLM(model="m", client=Broken([]), fallback_models=[], sleep=lambda s: None)
    with pytest.raises(TypeError):
        run_debug_agent(llm, path, log_path=str(tmp_path / "l.jsonl"))


# ---------- transposition evidence in interface errors (ambiguity found in the real Groq run) ----------
def test_transposed_weights_report_points_to_the_specific_cause_only(make_report):
    path = make_report("tw_ev", "matmul", faults.transposed_weights)
    c = debug_tools.diagnose_report(path)["rule_based_candidates"]
    assert c[0] == "weight_orientation_transposed" and "adapter_interface_mismatch" not in c


def test_transposition_evidence_parser():
    from x2c.verification.knowledge_base import transposed_weight_evidence as ev
    assert ev("ValueError: incompatible shapes (2, 3) @ (4, 3)") is True          # inner dim fits B's 2nd dim
    assert ev("ValueError: incompatible shapes (8, 1) @ (8, 1)") is True
    assert ev("ValueError: incompatible shapes (2, 3) @ (5, 4)") is False         # fits neither orientation
    assert ev("dimension mismatch: weights (4, 3), vec (3,)") is True
    assert ev("dimension mismatch: weights (4, 5), vec (3,)") is False
    assert ev("something else") is None


def test_generic_shape_errors_still_map_to_the_adapter_cause():
    from x2c.verification.knowledge_base import rule_based_candidates
    f = [{"diagnosis": {"category": "interface_error"}, "error": "ValueError: incompatible shapes (2, 3) @ (5, 4)"}]
    assert rule_based_candidates(f) == ["adapter_interface_mismatch"]
    f = [{"diagnosis": {"category": "interface_error"}, "error": "ValueError: boom"}]       # unparseable: both stay possible
    assert set(rule_based_candidates(f)) == {"adapter_interface_mismatch", "weight_orientation_transposed"}


def test_vague_llm_answer_is_rejected_when_a_specific_cause_explains_the_evidence(make_report, tmp_path):
    path = make_report("tw_vague", "matmul", faults.transposed_weights)
    out = run_debug_agent(Scripted(json.dumps({"cause_id": "adapter_interface_mismatch", "confidence": "high"})), path,
                          log_path=str(tmp_path / "l.jsonl"))
    assert not out["validated"] and out["final_cause"] == "weight_orientation_transposed"


# ---------- 'Tool choice is none, but model called a tool' (seen with gpt-oss on the forced-final turn) ----------
def test_tool_choice_none_violation_recovers_from_failed_generation(make_report, tmp_path):
    path = make_report("tc1", "matmul", faults.sign_flip)
    answer = {"cause_id": "output_sign_inverted", "confidence": "high"}
    err = type("E", (Exception,), {})("Error code: 400 - Tool choice is none, but model called a tool")
    err.status_code = 400
    err.body = {"error": {"message": "Tool choice is none, but model called a tool", "code": "tool_use_failed",
                          "failed_generation": json.dumps({"name": "json", "arguments": answer})}}
    c = _GroqClient([_gmsg(calls=[("c1", "diagnose_report", json.dumps({"report_path": path}))]), err])
    llm = GroqLLM(model="m", client=c, fallback_models=[], sleep=lambda s: None)
    out = run_debug_agent(llm, path, max_steps=1, log_path=str(tmp_path / "l.jsonl"))
    assert out["final_cause"] == "output_sign_inverted" and out["llm_recovered_answers"] == 1
    assert c.requests[1]["tool_choice"] == "none"


def test_tool_choice_none_violation_without_text_retries_without_tools(make_report, tmp_path):
    path = make_report("tc2", "matmul", faults.sign_flip)
    answer = json.dumps({"cause_id": "output_sign_inverted", "confidence": "high"})
    err = type("E", (Exception,), {})("Error code: 400 - Tool choice is none, but model called a tool")
    err.status_code = 400; err.body = {"error": {"message": "Tool choice is none, but model called a tool"}}
    c = _GroqClient([_gmsg(calls=[("c1", "diagnose_report", json.dumps({"report_path": path}))]), err, _gmsg(content=answer)])
    llm = GroqLLM(model="m", client=c, fallback_models=[], sleep=lambda s: None)
    out = run_debug_agent(llm, path, max_steps=1, log_path=str(tmp_path / "l.jsonl"))
    assert out["final_cause"] == "output_sign_inverted" and out["validated"]
    assert "tools" not in c.requests[2] and c.requests[2]["messages"][-1]["role"] == "user"


# ---------- all-zero output is its own category (found by the real-geometry benchmark) ----------
def test_all_zero_output_is_not_called_scaled():
    from x2c.verification.failure_classifier import classify_failure
    r = {"passed": False, "reason": "mismatch", "expected": [[1.0, 2.0], [3.0, 4.0]], "actual": [[0.0, 0.0], [0.0, 0.0]]}
    assert classify_failure(r)["category"] == "zero_output"
    r2 = {"passed": False, "reason": "mismatch", "expected": [[1.0, 2.0], [3.0, 4.0]], "actual": [[2.0, 4.0], [6.0, 8.0]]}
    assert classify_failure(r2)["category"] == "scaled_output"                   # real scaling is unaffected


def test_zero_output_needs_mapper_evidence_to_blame_the_mapper():
    from x2c.verification.knowledge_base import rule_based_candidates
    plain = [{"diagnosis": {"category": "zero_output"}}]                           # matmul report: no pass counts
    assert rule_based_candidates(plain) == []
    wl = [{"diagnosis": {"category": "zero_output"}, "mapping_check": {"passes_executed": 0, "mvm_ops_from_mapper": 4, "ok": False}}]
    assert rule_based_candidates(wl) == ["mapper_dropped_fold"]


def test_the_agent_on_the_harness_is_sent_back_until_it_keeps_the_contract(make_report, tmp_path):
    """Google's harness with a stand-in model (no key): the end-of-turn check sends back a reply that is not JSON and one
    that misses keys; the agent sees only the three A3 tools; code still validates the cause against the evidence."""
    pytest.importorskip("google.antigravity", reason="the agent harness is an optional extra: uv sync --extra agents")
    from tests.fake_model import FakeModel, calls_so_far, texts_so_far
    from x2c.agents.harness import Gemini
    path = make_report("sign_flip", "matmul", faults.sign_flip)
    good = json.dumps({"cause_id": "output_sign_inverted", "confidence": "high", "explanation": "actual == -expected",
                       "proposed_fix": "fix the sign", "next_check": "rerun"})
    replies = ["it is probably the sign", '{"cause_id": "output_sign_inverted"}', good]

    def script(messages, request):
        if "diagnose_report" not in calls_so_far(messages):
            return ("tool", "diagnose_report", {"report_path": path})
        return ("text", replies[min(len(texts_so_far(messages)), len(replies) - 1)])

    with FakeModel(script) as fm:
        out = run_debug_agent(Gemini(model="fake", base_url=fm.url), path, log_path=str(tmp_path / "l.jsonl"))
    assert fm.tools_offered() == ["check_hypothesis", "diagnose_report", "reproduce_failure"]
    assert out["end_of_turn_sendbacks"] == 2 and out["model"] == "fake"
    assert out["final_cause"] == "output_sign_inverted" and out["final_cause_source"] == "llm_validated" and out["validated"]
