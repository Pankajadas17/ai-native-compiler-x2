"""
Verification agent (A4) loop. The LLM decides WHICH tool to call; deterministic tools do the work.
Guard rails: step limit, validated tool calls, bounded tool output, full JSONL audit log,
and a final block of facts recomputed from the report by code (not by the LLM).
"""
import json
from pathlib import Path

from x2c.agents.llm_client import SYSTEM_PROMPT
from x2c.verification import tools
from x2c.verification.matmul_check import make_cases

MAX_RESULT_CHARS = 6000
LOG_PATH = "reports/agent_log.jsonl"


def _compact(result):
    s = json.dumps(result)
    if len(s) <= MAX_RESULT_CHARS:
        return result
    if isinstance(result, dict) and isinstance(result.get("failures"), list):   # trim, keep it honest
        r = dict(result); keep = r["failures"][:5]
        r["failures"] = [{k: f.get(k) for k in ("test", "shape", "diagnosis", "error")} for f in keep]
        r["note"] = f"truncated: showing {len(keep)} of {result.get('failed_count')} failures"
        return r
    return {"truncated": True, "preview": s[:MAX_RESULT_CHARS]}


def _chip_facts():
    """Chip facts recomputed by code from the reports at the end (never the LLM's wording)."""
    from x2c.verification.chip_tools import chip_status
    try:
        return chip_status()
    except Exception as e:  # noqa: BLE001  -- a broken report must not break the agent's result
        return {"error": f"{type(e).__name__}: {e}"}


REPORT_KEYS = {"run_suite": "suite", "run_workloads": "workloads", "run_benchmark": "benchmark",
               "run_generated_tests": "generated", "run_model_workloads": "model"}


def end_check(state, full_size):
    """A4's end-of-turn check (the verification agent): it may stop only after the FULL standard suite ran
    (default arguments, never fewer tests) and, if that suite failed, after it read the diagnoses (get_failures).
    The facts in the result are still recomputed from the reports by code. Returns the send-back reason or None."""
    suite = state["reports"].get("suite")
    if suite is None:
        return "Nothing has been verified yet: call run_suite with its default arguments (no arguments)."
    facts = tools.summarize_report(suite)
    total = facts.get("total", 0)
    if total < full_size:
        return (f"The suite ran only {total} of {full_size} standard tests. Call run_suite again with its default arguments; "
                "never lower n_random.")
    if facts.get("failed", 0) and "get_failures" not in state["tools"]:
        return "The suite has failures: call get_failures and explain the likely causes from its diagnoses before you finish."
    return None


def run_agent(llm, goal, max_steps=6, log_path=LOG_PATH):
    """llm: harness.OpenRouter(...) (OpenAI's Agents SDK via OpenRouter, then Groq if no model is left), or an object with start /
    send_tool_results (the stand-in MockLLM, GroqLLM, test doubles). The facts are recomputed from the reports by code."""
    from x2c.agents import harness
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    log = harness.jsonl_logger(log_path)
    state = {"steps": [], "reports": {}, "tools": [], "chip_used": False}

    def call(name, args):
        res = tools.call_tool(name, args)
        state["steps"].append({"tool": name, "args": args, "ok": "error" not in res})
        state["tools"].append(name)
        if name in REPORT_KEYS and "report_path" in res:
            state["reports"][REPORT_KEYS[name]] = res["report_path"]
        if name in ("chip_status", "chip_schedule", "run_chip_layers") and "error" not in res:
            state["chip_used"] = True
        return _compact(res)

    full = len(make_cases())
    run = harness.run(llm, SYSTEM_PROMPT, goal, max_steps=max_steps, max_model_calls=2 * max_steps + 6, log=log, catch_llm_errors=False,
                      end_check=lambda reply, used: end_check(state, full),
                      tools=dict(schemas=tools.TOOL_SCHEMAS, call=call, system=None, force_final=False,
                                  on_tool=lambda step, name, args, res: log("tool", step=step, tool=name, args=args,
                                                                            ok="error" not in res)))
    final_text, stopped, llm_error = run["text"], run["stopped"], run["llm_error"]
    extra = {"end_of_turn_sendbacks": run["continues"], "model": run["model"], "cost_usd": run.get("cost_usd"), "llm_error": llm_error}
    last_report, last_workload = state["reports"].get("suite"), state["reports"].get("workloads")
    last_bench, last_gen, last_model = state["reports"].get("benchmark"), state["reports"].get("generated"), state["reports"].get("model")
    steps, chip_used = state["steps"], state["chip_used"]

    facts = tools.summarize_report(last_report) if last_report else {"note": "run_suite was never called"}
    warnings = []
    if "total" in facts:
        facts["coverage"] = {"tests_run": facts["total"], "full_suite_size": full}
        if facts["total"] < full:
            warnings.append(f"REDUCED COVERAGE: only {facts['total']} of {full} standard tests were run "
                            f"(the LLM chose fewer tests). Re-run the full suite before claiming a pass.")
    else:
        warnings.append("run_suite was never called: nothing was verified.")
    return {"stopped": stopped, "steps": len(steps), "llm_summary": final_text,
            "verified_facts_from_report": facts,
            "verified_workload_facts": tools.summarize_report(last_workload) if last_workload else None,
            "verified_benchmark_facts": tools.summarize_report(last_bench) if last_bench else None,
            "verified_generated_test_facts": tools.summarize_report(last_gen) if last_gen else None,
            "verified_model_workload_facts": tools.summarize_report(last_model) if last_model else None,
            "verified_chip_facts": _chip_facts() if chip_used else None,
            "warnings": warnings, "audit_log": log_path, **extra}
