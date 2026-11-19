"""
A4 test-generation agent. The LLM proposes test SPECS to close coverage gaps; code validates them, builds the matrices
and the Python oracle supplies expected results. The plan is made against the HEALTHY backend and from coverage
information only (never from RTL or from knowledge of any bug), then replayed unchanged against any backend.
"""
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


def end_check(tools_used):
    """A4's end-of-turn check (the test planner): it looked at the coverage gaps first, at least one spec was accepted,
    and no coverage class is left open (at most 2 send-backs). The oracle and the replay are code, as before."""
    from x2c.verification import test_planner as tp
    if "coverage_gaps" not in tools_used:
        return "Call coverage_gaps first; it lists the classes the fixed suite misses and the spec format."
    specs = tools.get_generated_specs()
    if not specs:
        return "No spec has been accepted yet. Call run_generated_tests with specs aimed at the missing classes."
    missing = tp.coverage_gaps(specs)["missing"]
    if missing:
        return (f"These classes are still uncovered: {', '.join(map(str, missing[:10]))}. Call run_generated_tests again "
                "with specs aimed at them.")
    return None


def plan_tests(llm, max_steps=5, log_path="reports/test_agent_log.jsonl"):
    """llm: harness.Gemini(...) (Google's harness, then Groq if every Gemini model is out), or an object with start /
    send_tool_results (the stand-in MockPlannerLLM, GroqLLM, test doubles). The oracle and the replay are code."""
    from x2c.agents import harness
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    tools.reset_generated()
    schemas = [s for s in tools.TOOL_SCHEMAS if s["name"] in PLANNER_TOOLS]
    log = harness.jsonl_logger(log_path)
    run = harness.run(llm, TEST_PROMPT, "Plan independent tests to close the coverage gaps.",
                      sdk_tools=harness.tools_from_schemas(schemas, call_planner_tool), max_steps=max_steps,
                      max_model_calls=2 * max_steps + 4, log=log, catch_llm_errors=False,
                      end_check=lambda reply, used: end_check(used),
                      legacy=dict(schemas=schemas, call=call_planner_tool, force_final=False,
                                  on_tool=lambda step, name, args, res: log("tool", step=step, tool=name, ok="error" not in res)))
    final_text, stopped = run["text"], run["stopped"]
    extra = {"end_of_turn_sendbacks": run["continues"], "model": run["model"], "llm_error": run["llm_error"]}
    from x2c.verification import test_planner as tp
    specs = tools.get_generated_specs()
    return {"stopped": stopped, "summary": final_text, "specs": specs, "num_specs": len(specs),
            "coverage_still_missing": tp.coverage_gaps(specs)["missing"], "audit_log": log_path, **extra}
