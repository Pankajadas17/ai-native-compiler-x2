"""
RTL agent (A2). The LLM writes Verilog for one eFPGA-side operator; verification.rtl_check decides pass/fail by running it in Yosys
against a NumPy golden model. The LLM sees the checker's verdict (stage, errors, a few mismatching vectors) and may retry.

Rule: only a checker 'pass' counts. Nothing the LLM says about its own code is evidence. Verilog is saved only after a pass.
Evidence level: Yosys RTL evaluation vs NumPy golden (NOT nextpnr, NOT bitstream, NOT silicon).
"""
import json
from pathlib import Path

from x2c.agents import harness
from x2c.agents.llm_client import Action
from x2c.verification.rtl_check import EVIDENCE_LEVEL, OPS, check_rtl, describe_op

MAX_SUBMISSIONS = 4

RTL_PROMPT = """You are the RTL agent (A2) for an eFPGA compiler flow. You write small, synthesizable, purely combinational Verilog.
Rules:
1. Call get_spec first. Use exactly the module name and ports it lists (names, directions, widths); declare them signed.
2. Call submit_verilog with the complete module text. The checker runs it in Yosys on test vectors against a golden model.
3. If it fails, read the stage and errors, fix the code, and submit again. Only a checker 'pass' counts.
4. No clocks, registers, latches, memories, `include, or system tasks. Mind widths: sums and products need extra bits before saturating.
5. Never claim the design is verified, synthesized or tested on hardware; say only what the checker reported.
6. Finish with one plain sentence on the outcome."""

RTL_TOOL_SCHEMAS = [
    {"name": "get_spec", "description": "Return the operator specification (module name, ports, behaviour).",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "submit_verilog", "description": "Check a complete Verilog module; returns pass/fail with stage and errors.",
     "input_schema": {"type": "object", "properties": {"verilog": {"type": "string"}}, "required": ["verilog"]}},
]

# Reference solutions: used ONLY by the mock stand-in and by tests (never shown to a real LLM).
REFERENCE_VERILOG = {
    "relu8": "module relu8(input signed [7:0] x, output signed [7:0] y);\n  assign y = x[7] ? 8'sd0 : x;\nendmodule\n",
    "add8_sat": """module add8_sat(input signed [7:0] a, input signed [7:0] b, output signed [7:0] y);
  wire signed [8:0] s = a + b;
  assign y = (s > 9'sd127) ? 8'sd127 : (s < -9'sd128) ? -8'sd128 : s[7:0];
endmodule
""",
    "requant_i32_i8": """module requant_i32_i8(input signed [31:0] acc, output signed [7:0] y);
  wire signed [47:0] p = acc * 48'sd19 + 48'sd512;
  wire signed [47:0] q = p >>> 10;
  assign y = (q > 48'sd127) ? 8'sd127 : (q < -48'sd128) ? -8'sd128 : q[7:0];
endmodule
""",
    "x2_acc_step": """module x2_acc_step(input signed [31:0] acc_in, input [13:0] count, input [3:0] shift, input neg,
                   output signed [31:0] acc_out);
  wire [31:0] v = {18'd0, count} << shift;
  assign acc_out = neg ? acc_in - $signed(v) : acc_in + $signed(v);
endmodule
""",
}


class MockRTLLLM:
    """Rule-based stand-in (NOT a real model): submits the reference Verilog for the op."""
    def __init__(self, op):
        self.op = op

    def start(self, goal, tool_schemas, system_prompt=None):
        return Action("tool_calls", calls=[("get_spec", {})])

    def send_tool_results(self, results, force_final=False):
        name, res = results[0]
        if name == "get_spec":
            return Action("tool_calls", calls=[("submit_verilog", {"verilog": REFERENCE_VERILOG[self.op]})])
        return Action("final", text=f"[mock] submitted reference Verilog for {self.op}")


class _Session:
    def __init__(self, op):
        self.op, self.attempts, self.last, self.best_pass = op, 0, None, None

    def get_spec(self):
        return describe_op(self.op)

    def submit_verilog(self, verilog):
        if self.best_pass is not None:
            return {"status": "pass", "note": "already passed; no more submissions needed"}
        if self.attempts >= MAX_SUBMISSIONS:
            return {"status": "fail", "errors": [f"submission limit ({MAX_SUBMISSIONS}) reached"]}
        self.attempts += 1
        res = check_rtl(self.op, verilog)
        self.last = {"verilog": verilog if isinstance(verilog, str) else "", **res}
        if res["status"] == "pass":
            self.best_pass = self.last
        return {k: v for k, v in res.items() if k != "evidence_level"}

    def call(self, name, arguments):
        """Untrusted-LLM entry point: unknown tools and arguments are rejected."""
        schema = next((s for s in RTL_TOOL_SCHEMAS if s["name"] == name), None)
        if schema is None:
            return {"error": f"unknown tool '{name}'", "available": [s["name"] for s in RTL_TOOL_SCHEMAS]}
        args = dict(arguments or {})
        allowed = schema["input_schema"]["properties"]
        if set(args) - set(allowed):
            return {"error": f"unknown arguments {sorted(set(args) - set(allowed))}; allowed: {sorted(allowed)}"}
        try:
            return getattr(self, name)(**args)
        except (TypeError, ValueError, KeyError, AttributeError) as e:
            return {"error": f"{type(e).__name__}: {e}"}


def end_check(session, tools_used):
    """A2's end-of-turn check: the agent may stop only when the CHECKER passed one of its submissions (or no submission
    is left). What the model says about its own Verilog is never evidence. Returns the send-back reason or None."""
    if session.best_pass is not None or session.attempts >= MAX_SUBMISSIONS:
        return None
    if "submit_verilog" not in tools_used or session.last is None:
        return "You have not submitted any Verilog. Call submit_verilog with the complete module."
    errors = "; ".join(str(e) for e in (session.last.get("errors") or [])[:2]) or "see the mismatches it returned"
    return (f"The checker rejected your last submission (stage '{session.last.get('stage')}': {errors[:300]}). "
            f"Fix the module and submit it again with submit_verilog ({MAX_SUBMISSIONS - session.attempts} submission(s) "
            "left). Only a checker 'pass' counts.")


def _sdk_tools(session):
    """A2's two tools as plain functions for the harness; the session enforces the submission limit."""
    def get_spec() -> str:
        """Return the operator specification: module name, ports (names, directions, widths, signedness) and behaviour."""
        return json.dumps(session.get_spec())

    def submit_verilog(verilog: str) -> str:
        """Check one complete Verilog module in Yosys against the golden model.

        Args:
            verilog: The complete module text.

        Returns:
            JSON with status (pass or fail), the stage that failed, errors and a few mismatching vectors.
        """
        return json.dumps(session.submit_verilog(verilog), default=str)[:6000]
    return [get_spec, submit_verilog]


def run_rtl_agent(llm, op, out_dir="hardware/operators", max_steps=10, log_path="reports/rtl_agent_log.jsonl"):
    """llm: harness.Gemini(...) (Google's harness, then Groq if every Gemini model is out), or an object with start /
    send_tool_results (the stand-in MockRTLLLM the pipeline uses, GroqLLM, test doubles). Only a checker pass is saved."""
    if op not in OPS:
        raise ValueError(f"unknown op '{op}'; known: {sorted(OPS)}")
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    session = _Session(op)
    log = harness.jsonl_logger(log_path)
    run = harness.run(llm, RTL_PROMPT, f"Write Verilog for operator '{op}'.", sdk_tools=_sdk_tools(session), max_steps=max_steps,
                      max_model_calls=max_steps + 4, max_continues=MAX_SUBMISSIONS - 1, log=log,
                      end_check=lambda reply, used: end_check(session, used),
                      legacy=dict(schemas=RTL_TOOL_SCHEMAS, call=session.call,
                                  on_tool=lambda step, name, args, res: log("tool", step=step, tool=name, status=res.get("status"),
                                                                            stage=res.get("stage"))))
    text, stopped, steps, llm_error = run["text"], run["stopped"], run["steps"], run["llm_error"]
    extra = {"end_of_turn_sendbacks": run["continues"], "model": run["model"]}
    ok = session.best_pass is not None
    path = None
    if ok:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        path = str(Path(out_dir) / f"{op}.v")
        Path(path).write_text(session.best_pass["verilog"], encoding="utf-8")
    last = session.last or {}
    return {"op": op, "verified": ok, "verilog_path": path, "attempts": session.attempts, "stopped": stopped, "steps": steps,
            "llm_error": llm_error, "last_stage": last.get("stage"), "last_errors": (last.get("errors") or [])[:4],
            "vectors": (session.best_pass or {}).get("vectors"), "llm_summary_unverified": text[:600],
            "evidence_level": EVIDENCE_LEVEL, "audit_log": log_path, **extra}
