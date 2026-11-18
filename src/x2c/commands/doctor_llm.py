"""Quick connectivity check (a few seconds): does the LLM answer, with which model, how fast?
   x2c doctor llm [--llm groq|gemini] [--model NAME]"""
import argparse, os, sys, time
from x2c.agents.llm_client import GeminiLLM, GroqLLM, LLMUnavailable

ap = argparse.ArgumentParser()
ap.add_argument("--llm", choices=["groq", "gemini"], default="groq")
ap.add_argument("--model", default=None)
a = ap.parse_args()
key = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY"}[a.llm]
if not os.environ.get(key):
    sys.exit(f"{key} is not set in THIS terminal window.")
llm = (GroqLLM if a.llm == "groq" else GeminiLLM)(model=a.model)
llm.time_budget_s = 40
t = time.time()
print("config:", llm.describe(), flush=True)
print("asking ...", flush=True)
try:
    act = llm.start("Reply with the single word OK.", [{"name": "noop", "description": "unused", "input_schema": {"type": "object", "properties": {}}}])
except LLMUnavailable as e:
    sys.exit(f"FAILED after {time.time() - t:.0f}s: {e}")
print(f"answered by {llm.model_used} in {time.time() - t:.1f}s -> {act.kind}: {act.text[:60]!r}")
