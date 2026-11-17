"""
Benchmark for the debug agent: inject 7 known bugs, let the agent diagnose each failing report, score against
the ground-truth cause.   python evaluate_debug_agent.py [--llm mock|gemini|groq] [--model NAME]
(The mock LLM just echoes the rule-based top candidate, so it scores what the RULES can do. A real LLM is
 interesting where it differs from the rules, or explains better; check 'validated' and 'agrees_with_rules'.)
"""
import argparse, json, sys, time
from agents.debug_agent import run_debug_agent
from agents.llm_client import MockDebugLLM, GeminiLLM, GroqLLM
from verification import tools, workloads, debug_tools
from tests import faults

BENCH = [  # (label, kind, fault function, ground-truth cause)
    ("sign_flip",                       "matmul",   faults.sign_flip,                 "output_sign_inverted"),
    ("off_by_one",                      "matmul",   faults.off_by_one,                "constant_offset"),
    ("doubled",                         "matmul",   faults.doubled,                   "output_scaled"),
    ("transposed_output",               "matmul",   faults.transposed_output,         "output_transposed"),
    ("transposed_weights",              "matmul",   faults.transposed_weights,        "weight_orientation_transposed"),
    ("mapper_drop_last_fold",           "workload", faults.drop_last_fold,            "mapper_dropped_fold"),
    ("mapper_drop_last_output_channel", "workload", faults.drop_last_output_channel,  "mapper_dropped_output_channel"),
]
UNSEEN = [  # bugs the knowledge base has never seen: the honest answer is "unknown"
    ("UNSEEN_clipped_output",       "matmul", faults.clipped_output,    "unknown"),
    ("UNSEEN_stale_accumulator",    "matmul", faults.stale_accumulator, "unknown"),
]

import numpy as np

HOLDOUT = [  # created AFTER the prompt/rules were tuned on UNSEEN: do NOT tune on these
    ("HOLDOUT_swap_first_two_columns", "matmul", faults.swap_first_two_columns, "unknown"),
    ("HOLDOUT_int8_wraparound",        "matmul", faults.int8_wraparound,        "unknown"),
    ("HOLDOUT_zero_last_row",          "matmul", faults.zero_last_row,          "unknown"),
]

ap = argparse.ArgumentParser()
ap.add_argument("--llm", choices=["mock", "gemini", "groq"], default="mock")
ap.add_argument("--model", default=None)
ap.add_argument("--budget", type=float, default=None, help="max seconds of LLM time per bug (default 120, env BM_LLM_BUDGET_S)")
ap.add_argument("--blind", action="store_true", help="hide the rule engine's ranked answer from the agent")
ap.add_argument("--holdout", action="store_true", help="add 3 fresh unseen bugs that were NEVER used to tune prompts/rules (truth = unknown)")
ap.add_argument("--unseen", action="store_true", help="add bugs the knowledge base does not cover (truth = unknown)")
a = ap.parse_args()
import os
_KEYS = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY"}
if a.llm in _KEYS and not os.environ.get(_KEYS[a.llm]):
    sys.exit(f"{_KEYS[a.llm]} is not set in THIS terminal window (it only lasts for the window where you set it).\n"
             f'Set it, then rerun:  $env:{_KEYS[a.llm]} = "your-key"      (or use --llm mock)')
make = {"mock": lambda: MockDebugLLM(), "gemini": lambda: GeminiLLM(model=a.model),
        "groq": lambda: GroqLLM(model=a.model)}[a.llm]

if a.blind:
    debug_tools.set_blind_mode(True)
bench = BENCH + (UNSEEN if a.unseen else []) + (HOLDOUT if a.holdout else [])
probe = make()
if hasattr(probe, "describe"):
    print("LLM config:", probe.describe(), flush=True)
rows = []
print(f"{len(bench)} bugs to diagnose with [{a.llm}]; progress is printed as each one finishes (Ctrl+C to stop)", flush=True)
t_all = time.time()
for i, (label, kind, fault, truth) in enumerate(bench, 1):
    t0 = time.time()
    print(f"[{i}/{len(bench)}] {label} ...", end=" ", flush=True)
    path = f"reports/eval_{label}.json"
    if kind == "matmul":
        tools.set_x2_backend(fault); tools.run_suite(report_path=path)
    else:
        workloads.set_mapping_transform(fault); tools.run_workloads(report_path=path)
    out, err = None, None
    try:
        llm_obj = make()
        if a.budget is not None and hasattr(llm_obj, "time_budget_s"):
            llm_obj.time_budget_s = a.budget
        out = run_debug_agent(llm_obj, path, log_path=f"reports/eval_{label}_log.jsonl")
    except Exception as e:  # noqa: BLE001 - one failing bug must not hide the others
        err = f"{type(e).__name__}: {str(e)[:120]}"
    finally:
        tools.set_x2_backend(None); workloads.set_mapping_transform(None)
    if out is None:
        rows.append((label, truth, "ERROR", False, False, False, 0, False, err))
        print(f"ERROR after {time.time() - t0:.0f}s: {err}", flush=True); continue
    got = (out["diagnosis"] or {}).get("cause_id")
    if out["llm_error"]:
        got = "LLM-DOWN"
    note = (f"recovered x{out['llm_recovered_answers']} " if out["llm_recovered_answers"] else "") + \
           (f"[{out['llm_error'][:160]}] " if out["llm_error"] else "") + \
           ("STEP LIMIT " if out["stopped"] == "step_limit_reached" else "") + \
           (f"-> system: {out['final_cause']} [{out['final_cause_source']}]" if out["final_cause_source"] not in ("llm_validated", "llm_unknown_confirmed_by_rules") else "")
    rows.append((label, truth, got, got == truth, out["validated"], out["agrees_with_rules"], out["steps"],
                 out["final_cause"] == truth, note))
    print(f"{got} ({time.time() - t0:.0f}s)" + (f"  [{out['llm_error'][:100]}]" if out["llm_error"] else ""), flush=True)
print(f"total time {time.time() - t_all:.0f}s", flush=True)

print(f"\n{'injected bug':34} {'truth':30} {'LLM said':30} LLM  valid rules steps SYSTEM")
for r in rows:
    print(f"{r[0]:34} {r[1]:30} {str(r[2]):30} {'Y' if r[3] else 'N'}    {'Y' if r[4] else 'N'}     {'Y' if r[5] else 'N'}     {r[6]}     {'Y' if r[7] else 'N'}   {r[8]}")
n = len(rows)
if all(r[2] == "ERROR" for r in rows):
    sys.exit(f"\nEvery run failed - this is a setup/connection problem, not a benchmark result. First error: {rows[0][8]}")
down = [r[0] for r in rows if r[2] == "LLM-DOWN"]
wrong_confident = [r[0] for r in rows if not r[3] and r[2] not in ("unknown", "ERROR", "LLM-DOWN", None)]
no_answer = [r[0] for r in rows if r[2] is None]
print(f"\nLLM named a specific WRONG cause: {len(wrong_confident)} {wrong_confident}")
print(f"LLM service unavailable on {len(down)} row(s): {down}  (excluded from the raw LLM score; the system still answered from rules)")
print(f"LLM gave no usable answer (step limit / not JSON): {len(no_answer)} {no_answer}")
print(f"mode: blind={a.blind} unseen={a.unseen} holdout={a.holdout}")
answered = n - len(down)
print(f"\nRAW LLM correct {sum(r[3] for r in rows)}/{answered} answered | answers validated {sum(r[4] for r in rows)}/{n} | SYSTEM (validated pipeline) correct {sum(r[7] for r in rows)}/{n}"
      f"   [LLM: {a.llm}; evidence level: Python behavioral simulation]")
