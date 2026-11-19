"""
Ask the debug agent (A3) why a chain run did not pass.

    x2c agent a3                      # every operator in reports/fab_chain_report.json that did not pass
    x2c agent a3 --op add8_sat --llm groq
    x2c agent a3 --target chip         # the chain on the BM Labs chip: reports/chip_chain_report.json, reports/chip_jobs/, chip_build/

Run it right after x2c chain (it reads that run's report and the job folders; it changes nothing and runs nothing in Ubuntu).
The LLM proposes a root cause from a fixed list; code accepts it only if the evidence rules support it, otherwise the rules answer.
--llm mock (default) needs no key and is NOT a real model. For groq/gemini the key must be set in this window with Read-Host -AsSecureString.
Report: reports/chain_debug_report.json. Evidence level: reading of chain reports and logs (NOT a new verification result).
"""
import argparse
import json
import os
import sys
from pathlib import Path

from x2c.agents.chain_debug_agent import MockChainDebugLLM, run_chain_debug_agent
from x2c.verification import chain_debug as cd

OUT = Path("reports/chain_debug_report.json")
CHIP_OUT = Path("reports/chip_chain_debug_report.json")
KEYS = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY"}


def make_llm(name, model=None):
    if name == "mock":
        return MockChainDebugLLM()
    if name == "gemini":                                   # Google's harness with the end-of-turn check
        from x2c.agents.harness import Gemini
        return Gemini(model=model)
    from x2c.agents.llm_client import GroqLLM
    return GroqLLM(model=model)


def main(argv=None, llm_factory=make_llm):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--op", action="append", help="operator (repeat); default: every operator in the report that did not pass")
    ap.add_argument("--llm", choices=["mock", "groq", "gemini"], default="mock")
    ap.add_argument("--model", default=None)
    ap.add_argument("--blind", action="store_true", help="hide the rules' candidates from the LLM (measures the LLM itself)")
    ap.add_argument("--target", choices=["demo", "chip"], default="demo", help="which chain's report: demo fabric or the BM Labs chip")
    a = ap.parse_args(argv)
    chip = a.target == "chip"
    report, out_path = (cd.CHIP_REPORT, CHIP_OUT) if chip else (cd.REPORT, OUT)
    if not report.exists():
        sys.exit(f"{report.as_posix()} not found: run  x2c chain --op <op>{' --target chip' if chip else ''}  first")
    try:
        results = json.loads(report.read_text(encoding="utf-8")).get("results", [])
    except ValueError:
        sys.exit(f"{report.as_posix()} is not readable JSON: run the chain again")
    by_op = {r.get("op"): r for r in results}
    ops = a.op or [r["op"] for r in results if not str(r.get("verdict", "")).startswith("PASS")]
    missing = [o for o in ops if o not in by_op]
    if missing:
        sys.exit(f"not in the chain report: {', '.join(missing)} (it has: {', '.join(by_op) or 'nothing'})")
    if not ops:
        print("Every operator in the chain report passed: nothing to diagnose.")
        return 0
    if a.llm in KEYS and not os.environ.get(KEYS[a.llm]):
        sys.exit(f"{KEYS[a.llm]} is not set in this window (set it with Read-Host -AsSecureString) or use --llm mock")
    rows = []
    for op in ops:
        r = by_op[op]
        if str(r.get("verdict", "")).startswith("PASS"):
            print(f"\n{op}: the chain passed, nothing to diagnose")
            continue
        d = run_chain_debug_agent(llm_factory(a.llm, a.model), op, r, blind=a.blind, target=a.target)
        rows.append(d)
        ev = d["evidence"]
        print(f"\n{'CHIP ' if chip else ''}CHAIN DEBUG  op={op}  chain verdict {ev['verdict']}  llm={a.llm}" + (" (mock: NOT a real model)" if a.llm == "mock" else ""))
        print(f"  first stage that did not pass: {ev['first_bad_stage']} ({ev['first_bad_status']}): {ev['first_bad_detail'][:220]}")
        if ev["sim_first_mismatches"]:
            print(f"  first mismatches: {' | '.join(ev['sim_first_mismatches'])}")
        print(f"  ROOT CAUSE: {d['final_cause']}  [{d['final_cause_source']}]")
        if d["suspect"]:
            print(f"    what: {d['suspect']}")
            print(f"    fix:  {d['known_fix']}")
            print(f"    basis: {d['seen']}")
        else:
            print("    the evidence does not match any known cause; read the stage detail above and the logs in the job folder")
        said = (d["diagnosis"] or {}).get("cause_id")
        print(f"  llm said: {said if d['diagnosis'] else 'no usable answer'} | rules support: {', '.join(d['rule_based_candidates']) or 'nothing'}"
              + (f" | sent back by the end-of-turn check: {d['end_of_turn_sendbacks']}x" if d.get("end_of_turn_sendbacks") else "")
              + (f" | model: {d['model']}" if d.get("model") else "")
              + (f" | llm error: {d['llm_error'][:120]}" if d["llm_error"] else ""))
        if d["diagnosis"] and d["final_cause_source"] == "llm_validated" and d["diagnosis"].get("explanation"):
            print(f"  llm explanation (validated cause, wording unchecked): {str(d['diagnosis']['explanation'])[:300]}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"llm": a.llm, "blind": a.blind, "target": a.target, "diagnoses": rows}, indent=2, default=str), encoding="utf-8")
    print(f"\nreport: {out_path.as_posix()}\n[reading of chain reports and logs: NOT a new verification result, NOT timing, NOT silicon]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
