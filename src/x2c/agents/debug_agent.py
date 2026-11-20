"""
Debug agent (A3). Reads a failing verification report through deterministic tools and answers with a
STRUCTURED root-cause diagnosis chosen from a fixed list. Its answer is then VALIDATED by code:
a cause is only 'supported' if it is consistent with the failure categories actually observed.
It never edits code or re-runs flows by itself: it proposes, humans/other tools act.
"""
import json
import re
from pathlib import Path

from x2c.agents import harness
from x2c.verification import debug_tools
from x2c.verification.knowledge_base import CAUSES, CAUSE_IDS, rule_based_candidates

MENU = "\n".join(f"   - {k}: {v['suspect']}" + (f" [applies only to: {v['only_for']}]" if v.get("only_for") else "")
                 for k, v in CAUSES.items()) + "\n   - unknown: none of the above fits the evidence"

DEBUG_PROMPT = f"""You are the debug agent for a compiler-to-hardware verification flow (X2 behavioral model, mapper, adapters).
You receive a FAILING verification report. Use the tools to investigate, then answer.
Rules:
1. Start with diagnose_report. Optionally use check_hypothesis / reproduce_failure to confirm.
2. Base every claim on tool results. Do not invent numbers, tests or causes.
3. Check report_kind in diagnose_report: a cause marked "applies only to" must not be chosen for other report kinds.
   Choose cause_id ONLY from this menu (id: what it means). Use "unknown" if the evidence does not clearly support one;
   "unknown" is a GOOD answer when the failure does not match any menu entry.
{MENU}
4. Final answer: ONLY a JSON object with keys cause_id, confidence ("low"|"medium"|"high"), explanation, proposed_fix, next_check.
   Write it as your normal reply text. Do NOT call a tool (e.g. one named "json") to deliver the answer.
5. Evidence level is Python behavioral simulation; never claim RTL or silicon findings."""


def _parse_json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except ValueError:
        return None


ANSWER_KEYS = ("cause_id", "confidence", "explanation", "proposed_fix", "next_check")


def check_answer(text, tools_used, first_tool, menu_ids):
    """End-of-turn check of both A3 agents: the answer CONTRACT (looked at the evidence, one JSON object with every key,
    a cause from the menu). Whether that cause is supported by the evidence is still decided AFTER the run by code
    (_finish here, chain_debug.decide for the chain agent), exactly as before. Returns the send-back reason or None."""
    return harness.check_json_answer(text, tools_used, first_tool=first_tool, keys=ANSWER_KEYS, field_name="cause_id",
                                     allowed=set(menu_ids))


def run_debug_agent(llm, report_path, max_steps=6, log_path="reports/debug_agent_log.jsonl"):
    """llm: harness.OpenRouter(...) (OpenAI's Agents SDK via OpenRouter, then Groq if no model is left), or an object with start /
    send_tool_results (the no-model stand-in MockDebugLLM, GroqLLM, test doubles). Code validates the answer either way."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    goal = f"Diagnose the failures in the verification report. report_path={report_path}"
    log = harness.jsonl_logger(log_path)
    run = harness.run(llm, DEBUG_PROMPT, goal, max_steps=max_steps, max_model_calls=max_steps + 6,
                      end_check=lambda text, used: check_answer(text, used, "diagnose_report", CAUSE_IDS), log=log,
                      tools=dict(schemas=debug_tools.DEBUG_TOOL_SCHEMAS, call=debug_tools.call_debug_tool,
                                  on_tool=lambda step, name, args, res: log("tool", step=step, tool=name, args=args,
                                                                            ok="error" not in res)))
    out = _finish(run["text"], run["stopped"], run["steps"], llm, report_path, log_path, run["llm_error"])
    return {**out, "end_of_turn_sendbacks": run["continues"], "model": run["model"], "cost_usd": run.get("cost_usd")}


def _finish(final_text, stopped, steps, llm, report_path, log_path, llm_error):
    # ---- validation by code, independent of the LLM ----
    rep = debug_tools._load(report_path)
    fails = debug_tools._failures(rep)
    observed = sorted({f["diagnosis"]["category"] for f in fails})
    baseline = rule_based_candidates(fails)
    parsed = _parse_json(final_text)
    checks = {"json_ok": parsed is not None, "cause_known": False, "supported_by_evidence": False}
    if parsed:
        cid = parsed.get("cause_id")
        checks["cause_known"] = cid in CAUSE_IDS
        # strict: a specific cause counts only if the deterministic evidence rules also support it
        checks["supported_by_evidence"] = (cid == "unknown") or (cid in baseline)
    # ---- the SYSTEM's answer is decided by code: validated LLM answer > rule fallback > unknown ----
    validated = all(checks.values())
    said = (parsed or {}).get("cause_id")
    if validated and said != "unknown":
        final_cause, source = said, "llm_validated"
    elif validated and said == "unknown" and not baseline:
        final_cause, source = "unknown", "llm_unknown_confirmed_by_rules"
    elif baseline:
        final_cause = baseline[0]
        source = "rules_fallback (" + ("llm said unknown" if said == "unknown" else "llm claim rejected" if parsed else "llm unavailable" if llm_error else "no usable llm answer") + ")"
    else:
        final_cause, source = "unknown", "unknown_fallback (" + ("llm claim rejected" if parsed else "llm unavailable" if llm_error else "no usable llm answer") + ")"
    return {"final_cause": final_cause, "final_cause_source": source,
            "stopped": stopped, "steps": steps, "diagnosis": parsed, "raw_answer": None if parsed else final_text[:300],
            "validation": checks, "validated": validated,
            "observed_failure_categories": observed, "rule_based_baseline": baseline[:3],
            "agrees_with_rules": bool(parsed and baseline and parsed.get("cause_id") == baseline[0]),
            "known_fix_for_cause": CAUSES.get((parsed or {}).get("cause_id"), {}).get("fix"),
            "llm_error": llm_error, "llm_recovered_answers": getattr(llm, "recoveries", 0),
            "audit_log": log_path, "evidence_level": rep.get("level")}
