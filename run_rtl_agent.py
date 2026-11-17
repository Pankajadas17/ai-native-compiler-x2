"""
RTL agent (A2): an LLM writes Verilog for an eFPGA-side operator; Yosys + a NumPy golden model decide if it is right.

    python run_rtl_agent.py                       # mock LLM, all ops
    python run_rtl_agent.py --llm groq --op relu8 # a real LLM (GROQ_API_KEY must be set in this window)
    python run_rtl_agent.py --list

Verilog is saved to reports/rtl/<op>.v ONLY after the checker passes. Op conventions are ASSUMED (see verification/rtl_check.py).
Evidence level: Yosys RTL evaluation vs NumPy golden (NOT nextpnr, NOT bitstream, NOT silicon).
"""
import argparse
import json
import os
import sys

from agents.llm_client import GeminiLLM, GroqLLM
from agents.rtl_agent import MockRTLLLM, run_rtl_agent
from verification.rtl_check import OPS


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", choices=["mock", "gemini", "groq"], default="mock")
    ap.add_argument("--model", default=None)
    ap.add_argument("--op", default="all")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    if a.list:
        print("\n".join(OPS)); return 0
    ops = list(OPS) if a.op == "all" else [a.op]
    if any(o not in OPS for o in ops):
        sys.exit(f"unknown op '{a.op}'. Available: {', '.join(OPS)}")
    keys = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY"}
    if a.llm in keys and not os.environ.get(keys[a.llm]):
        sys.exit(f"{keys[a.llm]} is not set in THIS terminal window. Set it with Read-Host -AsSecureString, or use --llm mock")
    results = []
    for op in ops:
        llm = {"mock": lambda: MockRTLLLM(op), "gemini": lambda: GeminiLLM(model=a.model), "groq": lambda: GroqLLM(model=a.model)}[a.llm]()
        r = run_rtl_agent(llm, op)
        results.append(r)
        tag = "VERIFIED" if r["verified"] else "NOT VERIFIED"
        print(f"{op:16} {tag:13} attempts={r['attempts']} vectors={r['vectors']} stopped={r['stopped']}"
              + (f" | last: {r['last_stage']}: {r['last_errors'][0]}" if not r["verified"] and r["last_errors"] else "")
              + (f" | LLM problem: {r['llm_error']}" if r["llm_error"] else ""))
    os.makedirs("reports", exist_ok=True)
    with open("reports/rtl_agent_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    n = sum(r["verified"] for r in results)
    print(f"{n}/{len(results)} operators verified | LLM: {a.llm} | report: reports/rtl_agent_report.json")
    print("[yosys_rtl_evaluation_vs_numpy_golden (ASSUMED op conventions; NOT nextpnr, NOT silicon)]")
    return 0 if n == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
