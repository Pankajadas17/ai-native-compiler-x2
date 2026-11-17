"""
Debug agent (A3). Reads a failing verification report through deterministic tools and answers with a
STRUCTURED root-cause diagnosis chosen from a fixed list. Its answer is then VALIDATED by code:
a cause is only 'supported' if it is consistent with the failure categories actually observed.
It never edits code or re-runs flows by itself: it proposes, humans/other tools act.
"""
import json
import re
import time
from pathlib import Path

from agents.llm_client import LLMUnavailable, is_provider_error
from verification import debug_tools
from verification.knowledge_base import CAUSES, CAUSE_IDS, rule_based_candidates

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


def run_debug_agent(llm, report_path, max_steps=6, log_path="reports/debug_agent_log.jsonl"):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    goal = f"Diagnose the failures in the verification report. report_path={report_path}"
    log, steps = open(log_path, "w", encoding="utf-8"), 0
    final_text, stopped, llm_error = "", "finished", None
    try:
        final_text, stopped, steps = _converse(llm, goal, max_steps, log)
    except LLMUnavailable as e:        # the LLM service is down: the deterministic part must still answer
        llm_error, stopped = str(e)[:400], "llm_unavailable"
    except Exception as e:  # noqa: BLE001
        if not is_provider_error(e):   # programming errors and auth problems (401/403) must still raise
            raise
        llm_error, stopped = f"{type(e).__name__}: {str(e)[:350]}", "llm_error"
    log.close()
    return _finish(final_text, stopped, steps, llm, report_path, log_path, llm_error)


def _converse(llm, goal, max_steps, log):
    steps = 0
    action = llm.start(goal, debug_tools.DEBUG_TOOL_SCHEMAS, system_prompt=DEBUG_PROMPT)
    final_text, stopped = "", "finished"
    for step in range(1, max_steps + 1):
        if action.kind == "final":
            final_text = action.text; break
        results = []
        for name, args in action.calls:
            res = debug_tools.call_debug_tool(name, args)
            steps += 1
            log.write(json.dumps({"step": step, "tool": name, "args": args, "ok": "error" not in res,
                                  "t": time.strftime("%H:%M:%S")}) + "\n"); log.flush()
            results.append((name, res))
        last = step == max_steps          # last allowed tool round: the next turn must be a final answer
        if last and getattr(llm, "supports_force_final", False):
            action = llm.send_tool_results(results, force_final=True)
        else:
            action = llm.send_tool_results(results)
    else:
        if action.kind == "final":
            final_text = action.text
        else:
            stopped = "step_limit_reached"
    return final_text, stopped, steps


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
