"""
RTL agent (A2): an LLM writes Verilog for an eFPGA-side operator; Yosys + a NumPy golden model decide if it is right.

    x2c agent a2                       # mock LLM, all ops
    x2c agent a2 --llm openrouter --op relu8   # OpenRouter model on OpenAI's Agents SDK, with the end-of-turn check
    x2c agent a2 --llm groq --op relu8     # a real LLM through the plain loop (GROQ_API_KEY must be set)
    x2c agent a2 --list

Verilog is saved to hardware/operators/<op>.v ONLY after the checker passes. Op conventions are ASSUMED (see verification/rtl_check.py).
Evidence level: Yosys RTL evaluation vs NumPy golden (NOT nextpnr, NOT bitstream, NOT silicon).
"""
import argparse
import json
import os
import sys

from x2c.agents.harness import OpenRouter
from x2c.agents.llm_client import GroqLLM
from x2c.agents.rtl_agent import MockRTLLLM, run_rtl_agent
from x2c.verification.rtl_check import OPS


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", choices=["mock", "openrouter", "groq"], default="mock")
    ap.add_argument("--model", default=None)
    ap.add_argument("--op", default="all")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    if a.list:
        print("\n".join(OPS)); return 0
    ops = list(OPS) if a.op == "all" else [a.op]
    if any(o not in OPS for o in ops):
        sys.exit(f"unknown op '{a.op}'. Available: {', '.join(OPS)}")
    keys = {"groq": "GROQ_API_KEY", "openrouter": "OPENROUTER_API_KEY"}
    if a.llm in keys and not os.environ.get(keys[a.llm]):
        sys.exit(f"{keys[a.llm]} is not set in THIS terminal window. Set it with Read-Host -AsSecureString, or use --llm mock")
    results = []
    for op in ops:
        llm = {"mock": lambda: MockRTLLLM(op), "openrouter": lambda: OpenRouter(model=a.model), "groq": lambda: GroqLLM(model=a.model)}[a.llm]()
        r = run_rtl_agent(llm, op)
        results.append(r)
        tag = "VERIFIED" if r["verified"] else "NOT VERIFIED"
        print(f"{op:16} {tag:13} attempts={r['attempts']} vectors={r['vectors']} stopped={r['stopped']}"
              + (f" | last: {r['last_stage']}: {r['last_errors'][0]}" if not r["verified"] and r["last_errors"] else "")
              + (f" | sent back {r['end_of_turn_sendbacks']}x" if r.get("end_of_turn_sendbacks") else "")
              + (f" | model: {r['model']}" if r.get("model") else "")
              + (f" | cost ${r['cost_usd']:.3f}" if r.get("cost_usd") else "")
              + (f" | LLM problem: {r['llm_error']}" if r["llm_error"] else ""))
    os.makedirs("reports", exist_ok=True)
    with open("reports/rtl_agent_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    n = sum(r["verified"] for r in results)
    cost = sum(r.get("cost_usd") or 0 for r in results)
    print(f"{n}/{len(results)} operators verified | LLM: {a.llm}" + (f" | total cost ${cost:.3f}" if cost else "")
          + " | report: reports/rtl_agent_report.json")
    print("[yosys_rtl_evaluation_vs_numpy_golden (ASSUMED op conventions; NOT nextpnr, NOT silicon)]")
    return 0 if n == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
