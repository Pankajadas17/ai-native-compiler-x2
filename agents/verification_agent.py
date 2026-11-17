"""
Verification agent (A4) loop. The LLM decides WHICH tool to call; deterministic tools do the work.
Guard rails: step limit, validated tool calls, bounded tool output, full JSONL audit log,
and a final block of facts recomputed from the report by code (not by the LLM).
"""
import json
import time
from pathlib import Path

from verification import tools
from verification.test_generator import make_cases

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
    from verification.chip_tools import chip_status
    try:
        return chip_status()
    except Exception as e:  # noqa: BLE001  -- a broken report must not break the agent's result
        return {"error": f"{type(e).__name__}: {e}"}


def run_agent(llm, goal, max_steps=6, log_path=LOG_PATH):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    steps, last_report, last_workload, last_model, last_gen, last_bench = [], None, None, None, None, None
    chip_used = False
    log = open(log_path, "w", encoding="utf-8")

    def record(entry):
        entry["t"] = time.strftime("%H:%M:%S"); steps.append(entry)
        log.write(json.dumps(entry) + "\n"); log.flush()

    action = llm.start(goal, tools.TOOL_SCHEMAS)
    final_text, stopped = "", "finished"
    for step in range(1, max_steps + 1):
        if action.kind == "final":
            final_text = action.text; break
        results = []
        for name, args in action.calls:
            res = tools.call_tool(name, args)
            record({"step": step, "tool": name, "args": args, "ok": "error" not in res})
            if name == "run_suite" and "report_path" in res:
                last_report = res["report_path"]
            if name == "run_workloads" and "report_path" in res:
                last_workload = res["report_path"]
            if name == "run_benchmark" and "report_path" in res:
                last_bench = res["report_path"]
            if name == "run_generated_tests" and "report_path" in res:
                last_gen = res["report_path"]
            if name == "run_model_workloads" and "report_path" in res:
                last_model = res["report_path"]
            if name in ("chip_status", "chip_schedule", "run_chip_layers") and "error" not in res:
                chip_used = True
            results.append((name, _compact(res)))
        action = llm.send_tool_results(results)
    else:
        if action.kind == "final":
            final_text = action.text
        else:
            stopped = "step_limit_reached"
    log.close()

    facts = tools.summarize_report(last_report) if last_report else {"note": "run_suite was never called"}
    full = len(make_cases())
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
            "warnings": warnings, "audit_log": log_path}
