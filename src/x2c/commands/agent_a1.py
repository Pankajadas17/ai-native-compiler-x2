"""
Target-mapping agent (A1): an LLM proposes which hardware (X2 or eFPGA) runs each node of a model graph;
code derives the numbers and compiler.partition.validate_partition alone decides if the plan is legal.

    x2c agent a1                          # mock LLM, the Compiler V2 MobileNetV2 graph
    x2c agent a1 --llm gemini             # Gemini on Google's agent harness, with the end-of-turn check
    x2c agent a1 --llm groq               # a real LLM through the plain loop (GROQ_API_KEY must be set)
    x2c agent a1 --workload bm_deepgate_ad01_int8
    x2c agent a1 --list

HONEST LIMIT: with no fabric budget every node has exactly one legal target, so today the only legal plan IS the
baseline. This run proves the loop and its gate; real placement choices need the fabric profile and the X2 ABI.
Evidence level: static plan check against ASSUMED hardware (NOT RTL, NOT silicon). Report: reports/mapping_plan.json
"""
import argparse
import json
import os
import sys

from x2c.agents.harness import Gemini
from x2c.agents.llm_client import GroqLLM, MockMappingLLM
from x2c.agents.mapping_agent import run_mapping_agent
from x2c.compiler.ir_builder import build_ir_from_chain_workload, build_ir_from_compiler_graph
from x2c.compiler.partition import load_hardware
from x2c.inputs.bmsemi_source import ReadOnlyRepo
from x2c.inputs.normalize import all_workloads


def build_ir(repo, wl):
    """Compiler V2 comes with a graph artifact; the other workloads are plain layer chains."""
    return build_ir_from_compiler_graph(repo, wl) if "compiler_v2" in wl.id else build_ir_from_chain_workload(wl)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", choices=["mock", "gemini", "groq"], default="mock")
    ap.add_argument("--model", default=None)
    ap.add_argument("--workload", default="bm_compiler_v2_tiny_mobilenetv2")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    repo = ReadOnlyRepo()
    wls = {w.id: w for w in all_workloads(repo)}
    if a.list:
        print("\n".join(wls)); return 0
    if a.workload not in wls:
        sys.exit(f"unknown workload '{a.workload}'. Available: {', '.join(wls)}")
    keys = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY"}
    if a.llm in keys and not os.environ.get(keys[a.llm]):
        sys.exit(f"{keys[a.llm]} is not set in THIS terminal window. Set it with Read-Host -AsSecureString, or use --llm mock")
    llm = {"mock": MockMappingLLM, "gemini": lambda: Gemini(model=a.model), "groq": lambda: GroqLLM(model=a.model)}[a.llm]()
    ir = build_ir(repo, wls[a.workload])
    out = run_mapping_agent(llm, ir, load_hardware())
    plan = out["final_plan"]
    os.makedirs("reports", exist_ok=True)
    with open("reports/mapping_plan.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"graph: {ir.name} ({len(ir.nodes)} nodes) | LLM: {a.llm} | stopped: {out['stopped']} after {out['steps']} tool calls")
    print(f"LLM plan legal (validator): {out['llm_plan_legal']} | same as baseline: {out['matches_baseline']}")
    if out["llm_errors"]:
        print("validator errors on the LLM plan:", *out["llm_errors"][:5], sep="\n  ")
    if out.get("end_of_turn_sendbacks") or out.get("model"):
        print(f"model: {out.get('model')} | sent back by the end-of-turn check: {out.get('end_of_turn_sendbacks', 0)}x")
    if out["llm_error"]:
        print("LLM problem:", out["llm_error"])
    print(f"FINAL PLAN SOURCE: {out['final_source']}")
    if plan:
        t = plan["totals"]
        print(f"  X2 nodes {t['x2_nodes']} ({t['x2_tiles']} tiles) | eFPGA nodes {t['efpga_nodes']} | "
              f"X2<->eFPGA crossings {t['boundary_edges']} = {t['boundary_bytes']} bytes (assumes INT8 activations)")
    print(f"search space: {out['search_space']['legal_plans']} legal plan(s): {out['search_space']['note']}")
    print(f"[{out['evidence_level']}] report: reports/mapping_plan.json")
    return 0 if plan else 1


if __name__ == "__main__":
    sys.exit(main())
