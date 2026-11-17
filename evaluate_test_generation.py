"""
Does generated testing find bugs the fixed suite cannot?   python evaluate_test_generation.py [--llm mock|groq|gemini]
Plan = generated ONCE against the healthy backend from coverage gaps only, then replayed against every injected bug.
(--llm mock uses the deterministic baseline planner; groq/gemini let a real model write the specs.)
"""
import argparse, os, sys
import numpy as np
from agents.llm_client import MockPlannerLLM, GeminiLLM, GroqLLM
from agents.test_agent import plan_tests
from run_verification import run_all
from verification import tools
from tests import faults

ap = argparse.ArgumentParser()
ap.add_argument("--llm", choices=["mock", "gemini", "groq"], default="mock")
ap.add_argument("--model", default=None)
a = ap.parse_args()
_KEYS = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY"}
if a.llm in _KEYS and not os.environ.get(_KEYS[a.llm]):
    sys.exit(f"{_KEYS[a.llm]} is not set in THIS terminal window. Set it first (see README), or use --llm mock")

CATALOGUE = [  # (label, fault function, family)
    ("sign_flip", faults.sign_flip, "known"), ("off_by_one", faults.off_by_one, "known"), ("doubled", faults.doubled, "known"),
    ("transposed_output", faults.transposed_output, "known"), ("transposed_weights", faults.transposed_weights, "known"),
    ("clipped_output", faults.clipped_output, "unseen"), ("stale_accumulator", faults.stale_accumulator, "unseen"),
    ("swap_first_two_columns", faults.swap_first_two_columns, "holdout"), ("int8_wraparound", faults.int8_wraparound, "holdout"),
    ("zero_last_row", faults.zero_last_row, "holdout"),
    ("LATENT_wrap16", faults.latent_wrap16, "latent"), ("LATENT_long_k", faults.latent_long_k, "latent"),
    ("LATENT_int8_min_clamp", faults.latent_int8_min_clamp, "latent")]

llm = {"mock": lambda: MockPlannerLLM(), "gemini": lambda: GeminiLLM(model=a.model), "groq": lambda: GroqLLM(model=a.model)}[a.llm]()
plan = plan_tests(llm)
specs = plan["specs"]
print(f"plan from [{a.llm}]: {plan['num_specs']} specs accepted; classes still uncovered: {plan['coverage_still_missing'] or 'none'}")
if not specs:
    sys.exit("no valid specs were produced")

rows = []
for label, fault, family in CATALOGUE:
    fixed = run_all(x2_fn=fault, report_path=f"reports/_tg_fixed_{label}.json")
    tools.reset_generated()
    gen = tools.run_generated_tests(specs, report_path=f"reports/_tg_gen_{label}.json", x2_fn=fault)
    rows.append((label, family, fixed["total"] - fixed["passed"], fixed["total"], len(gen["failed"]), gen["total"]))

print(f"\n{'injected bug':24} {'family':8} {'fixed suite':>12} {'generated':>11}")
for r in rows:
    print(f"{r[0]:24} {r[1]:8} {r[2]:>6}/{r[3]:<5} {r[4]:>5}/{r[5]:<4}" + ("   <- ONLY generated tests catch it" if r[2] == 0 and r[4] > 0 else "") +
          ("   <- MISSED by both" if r[2] == 0 and r[4] == 0 else ""))
n = len(rows)
f_ok = [r[0] for r in rows if r[2] > 0]; g_ok = [r[0] for r in rows if r[4] > 0]
print(f"\nbugs detected - fixed suite: {len(f_ok)}/{n} | generated tests: {len(g_ok)}/{n} | either: {len(set(f_ok) | set(g_ok))}/{n}")
print(f"missed by the fixed suite but caught by generated tests: {sorted(set(g_ok) - set(f_ok))}")
print(f"missed by both: {sorted(set(r[0] for r in rows) - set(f_ok) - set(g_ok))}")
print("[evidence level: Python behavioral simulation; the bug catalogue was written by us, so this measures test-suite "
      "sensitivity to OUR bugs, not to real hardware faults]")
