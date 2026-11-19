"""
Debug agent (A3) for the FABulous chain. Given a chain result that did not pass, the LLM reads the evidence through ONE read-only tool and
answers with a root cause chosen from a fixed list. Code then validates the answer: a cause counts only if the deterministic evidence rules
support it (verification/chain_debug.py). If the LLM is unavailable or wrong, the rules answer; if the rules do not know, the answer is "unknown".
The agent never edits files and never re-runs anything: it proposes, the person acts.
Evidence level: reading of chain reports and logs (NOT a new verification result; nothing about timing or silicon).
"""
import json
import re

from x2c.agents import harness
from x2c.agents.llm_client import Action
from x2c.verification import chain_debug as cd

MENU = "\n".join(f"   - {k}: {v['suspect']}" for k, v in cd.CAUSES.items()) + "\n   - unknown: none of the above fits the evidence"

PROMPT = f"""You are the debug agent for a hardware compiler flow: operator Verilog -> FABulous (Yosys, nextpnr) -> bitstream -> simulation of
the configured fabric, compared with a NumPy golden model. A chain run did NOT pass. Find the most likely root cause.
Rules:
1. Call chain_evidence first. Base every claim on what it returns. Do not invent numbers, file contents or causes.
2. Choose cause_id ONLY from this menu. "unknown" is a GOOD answer when the evidence does not clearly support an entry.
{MENU}
3. A fabric output shown as 'x' means UNKNOWN (never configured or never driven), not a wrong number.
4. Final answer: ONLY a JSON object with keys cause_id, confidence ("low"|"medium"|"high"), explanation, proposed_fix, next_check.
   Write it as your normal reply text. Do NOT call a tool to deliver the answer.
5. All evidence is simulation and tool logs. Never claim timing, gate-level or silicon findings."""

CHIP_MENU = "\n".join(f"   - {k}: {cd.cause_info(k)['suspect']}" for k in cd.chip_menu_ids()) + "\n   - unknown: none of the above fits the evidence"

CHIP_PROMPT = f"""You are the debug agent for a hardware compiler flow on the BM Labs chip (sky130-reram-fpga): operator Verilog -> chip-pin
wrapper -> the chip repo's flow (Yosys with the repo techmaps, nextpnr) -> bitstream checked by the V0 audit -> simulation of the SYNTHESIZED
NETLIST, compared with a NumPy golden model. A chain run did NOT pass. Find the most likely root cause.
Rules:
1. Call chain_evidence first. Base every claim on what it returns. Do not invent numbers, file contents or causes.
2. Choose cause_id ONLY from this menu. "unknown" is a GOOD answer when the evidence does not clearly support an entry.
{CHIP_MENU}
3. A netlist output shown as None means UNDEFINED (x or z), not a wrong number.
4. Final answer: ONLY a JSON object with keys cause_id, confidence ("low"|"medium"|"high"), explanation, proposed_fix, next_check.
   Write it as your normal reply text. Do NOT call a tool to deliver the answer.
5. All evidence is netlist simulation and tool logs. Timing is nextpnr's estimate. Never claim configured-fabric or silicon findings."""

ANSWER_KEYS = ("cause_id", "confidence", "explanation", "proposed_fix", "next_check")
TOOL_SCHEMAS = [{"name": "chain_evidence",
                 "description": "Facts about the failed chain run of one operator: stage results, bitstream size, simulation result and first mismatch lines.",
                 "input_schema": {"type": "object", "properties": {"op": {"type": "string"}}}}]


def _parse_json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
    except ValueError:
        return None
    return v if isinstance(v, dict) else None


class MockChainDebugLLM:
    """Rule-based stand-in for the LLM (NOT a real model): asks for the evidence and answers with the first rule candidate."""
    def start(self, goal, tool_schemas, system_prompt=None):
        return Action("tool_calls", calls=[("chain_evidence", {"op": re.search(r"op=(\S+)", goal).group(1)})])

    def send_tool_results(self, results, force_final=False):
        cands = results[0][1].get("rule_based_candidates") or []
        cause = cands[0] if cands else "unknown"
        return Action("final", text=json.dumps({"cause_id": cause, "confidence": "medium" if cands else "low",
                                                "explanation": f"[mock] first rule-based candidate: {cause}",
                                                "proposed_fix": cd.cause_info(cause).get("fix", "none known"), "next_check": "run the chain again"}))


def run_chain_debug_agent(llm, op, result, jobs_dir=None, max_steps=3, blind=False, target="demo", build_dir=None):
    """result: one entry of fab_chain_report.json["results"] (target "chip": chip_chain_report.json). blind=True hides the rule candidates
    from the LLM (to measure the LLM itself)."""
    if target == "chip":
        ev = cd.collect_chip_evidence(op, result, jobs_dir or cd.CHIP_JOBS, build_dir or cd.CHIP_BUILD)
        cands, prompt = cd.chip_rule_candidates(ev), CHIP_PROMPT
    else:
        ev = cd.collect_evidence(op, result, jobs_dir or cd.JOBS)
        cands, prompt = cd.rule_candidates(ev), PROMPT

    def tool(name, args):
        if name != "chain_evidence":
            return {"error": f"unknown tool '{name}'", "available": ["chain_evidence"]}
        if set(args or {}) - {"op"} or (args or {}).get("op", op) != op:
            return {"error": f"this run is about op '{op}' only; allowed argument: op"}
        return {**ev, **({} if blind else {"rule_based_candidates": cands})}

    def chain_evidence(op: str = op) -> str:
        """Facts about the failed chain run of one operator: stage results, bitstream size, simulation result and first mismatch lines.

        Args:
            op: The operator named in the task.
        """
        return json.dumps(tool("chain_evidence", {"op": op}), default=str)

    menu = (set(cd.chip_menu_ids()) if target == "chip" else set(cd.CAUSES)) | {"unknown"}
    run = harness.run(llm, prompt, f"Diagnose why the chain run did not pass. op={op}", sdk_tools=[chain_evidence],
                      max_steps=max_steps, max_model_calls=max_steps + 5, legacy=dict(schemas=TOOL_SCHEMAS, call=tool),
                      end_check=lambda text, used: harness.check_json_answer(
                          text, used, first_tool="chain_evidence", keys=ANSWER_KEYS, field_name="cause_id", allowed=menu))
    final_text, stopped, steps, llm_error = run["text"], run["stopped"], run["steps"], run["llm_error"]
    extra = {"end_of_turn_sendbacks": run["continues"], "model": run["model"]}
    parsed = _parse_json(final_text)
    cause, source, checks = cd.decide(parsed, cands, llm_error)
    info = cd.cause_info(cause)
    return {"op": op, "target": target, "final_cause": cause, "final_cause_source": source, "suspect": info.get("suspect"),
            "known_fix": info.get("fix"), "seen": info.get("seen"),
            "diagnosis": parsed, "raw_answer": None if parsed else (final_text or "")[:300], "validation": checks, "validated": all(checks.values()),
            "rule_based_candidates": cands, "evidence": ev, "stopped": stopped, "steps": steps, "llm_error": llm_error,
            "evidence_level": "reading of chain reports and logs (NOT a new verification result; NOT timing, NOT silicon)", **extra}
