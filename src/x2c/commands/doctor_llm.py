"""Quick connectivity check (a few seconds): does the LLM answer, with which model, how fast?
   x2c doctor llm [--llm gemini|groq] [--model NAME]
gemini goes through Google's agent harness (the agents' main path) and uses ONE request of the free quota."""
import argparse, os, sys, time
from x2c.agents.llm_client import GroqLLM, LLMUnavailable

ap = argparse.ArgumentParser()
ap.add_argument("--llm", choices=["gemini", "groq"], default="gemini")
ap.add_argument("--model", default=None)
a = ap.parse_args()
key = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY"}[a.llm]
if not os.environ.get(key):
    sys.exit(f"{key} is not set in THIS terminal window.")
t = time.time()
if a.llm == "gemini":
    from x2c.agents import harness
    spec = harness.Gemini(model=a.model, groq_fallback=False)
    print("config:", spec.describe(), flush=True)
    print("asking ...", flush=True)
    run = harness.converse(spec, "You are a connectivity check.", "Reply with the single word OK.", [], max_model_calls=2)
    if run["llm_error"]:
        sys.exit(f"FAILED after {time.time() - t:.0f}s: {run['llm_error']}")
    print(f"answered by {run['model']} in {time.time() - t:.1f}s -> {run['text'][:60]!r}")
    sys.exit(0)
llm = GroqLLM(model=a.model)
llm.time_budget_s = 40
print("config:", llm.describe(), flush=True)
print("asking ...", flush=True)
try:
    act = llm.start("Reply with the single word OK.", [{"name": "noop", "description": "unused", "input_schema": {"type": "object", "properties": {}}}])
except LLMUnavailable as e:
    sys.exit(f"FAILED after {time.time() - t:.0f}s: {e}")
print(f"answered by {llm.model_used} in {time.time() - t:.1f}s -> {act.kind}: {act.text[:60]!r}")
