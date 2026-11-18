"""
A4 test-generation agent. The LLM proposes test SPECS to close coverage gaps; code validates them, builds the matrices
and the Python oracle supplies expected results. The plan is made against the HEALTHY backend and from coverage
information only (never from RTL or from knowledge of any bug), then replayed unchanged against any backend.
"""
import json
import time
from pathlib import Path

from x2c.verification import tools

PLANNER_TOOLS = ("coverage_gaps", "run_generated_tests")

TEST_PROMPT = """You are the test-generation agent (A4) for a hardware verification flow.
Goal: propose INDEPENDENT randomized and corner-case tests that exercise input classes the fixed suite misses.
Rules:
1. Call coverage_gaps first. It lists classes not yet covered and the spec format.
2. Call run_generated_tests with specs aimed at the missing classes. Keep specs small and varied
   (dims 1..12, int8 values -128..127). Use different seeds. You may call it again for remaining gaps.
3. A spec has NO expected-value field. The Python oracle computes expected results; never try to supply them.
4. Think like a hardware tester: overflow, extreme values, long accumulations, empty/sparse inputs, sign patterns.
   Do not assume anything about the design under test; plan from coverage only.
5. Finish with a short plain-text summary of what you covered and what remains."""


def call_planner_tool(name, arguments):
    if name not in PLANNER_TOOLS:
        return {"error": f"tool '{name}' is not available to the test agent", "available": list(PLANNER_TOOLS)}
    return tools.call_tool(name, arguments)


def plan_tests(llm, max_steps=5, log_path="reports/test_agent_log.jsonl"):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    tools.reset_generated()
    schemas = [s for s in tools.TOOL_SCHEMAS if s["name"] in PLANNER_TOOLS]
    log = open(log_path, "w", encoding="utf-8")
    action = llm.start("Plan independent tests to close the coverage gaps.", schemas, system_prompt=TEST_PROMPT)
    final_text, stopped = "", "finished"
    for step in range(1, max_steps + 1):
        if action.kind == "final":
            final_text = action.text; break
        results = []
        for name, args in action.calls:
            res = call_planner_tool(name, args)
            log.write(json.dumps({"step": step, "tool": name, "ok": "error" not in res, "t": time.strftime("%H:%M:%S")}) + "\n")
            results.append((name, res))
        action = llm.send_tool_results(results)
    else:
        if action.kind == "final":
            final_text = action.text
        else:
            stopped = "step_limit_reached"
    log.close()
    from x2c.verification import test_planner as tp
    specs = tools.get_generated_specs()
    return {"stopped": stopped, "summary": final_text, "specs": specs, "num_specs": len(specs),
            "coverage_still_missing": tp.coverage_gaps(specs)["missing"], "audit_log": log_path}
