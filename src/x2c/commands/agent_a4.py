"""Run the verification agent.
  x2c agent a4                         # mock LLM (no key), healthy X2
  x2c agent a4 --inject-fault sign_flip
  x2c agent a4 --llm openrouter            # OpenAI's Agents SDK via OpenRouter, with the end-of-turn check (needs OPENROUTER_API_KEY)
  x2c agent a4 --llm groq              # needs GROQ_API_KEY
"""
import argparse, json
import sys
from x2c.agents.harness import OpenRouter
from x2c.agents.llm_client import MockLLM, GroqLLM, LLMUnavailable
from x2c.agents.verification_agent import run_agent
from x2c.verification import tools

ap = argparse.ArgumentParser()
ap.add_argument("--llm", choices=["mock", "openrouter", "groq"], default="mock")
ap.add_argument("--model", default=None)
ap.add_argument("--inject-fault", default=None,
                help="sign_flip | off_by_one | doubled | transposed_output | transposed_weights")
ap.add_argument("--goal", default="Verify the X2 behavioral model against the Python reference and report any problems.")
a = ap.parse_args()
import os
_KEYS = {"groq": "GROQ_API_KEY", "openrouter": "OPENROUTER_API_KEY"}
if a.llm in _KEYS and not os.environ.get(_KEYS[a.llm]):
    sys.exit(f"{_KEYS[a.llm]} is not set in THIS terminal window (it only lasts for the window where you set it).\n"
             f'Set it, then rerun:  $env:{_KEYS[a.llm]} = "your-key"      (or use --llm mock)')

if a.inject_fault:
    from tests import faults
    tools.set_x2_backend(getattr(faults, a.inject_fault))
    print(f"[demo] deliberately broken X2 injected: {a.inject_fault}")
llm = {"mock": lambda: MockLLM(), "openrouter": lambda: OpenRouter(model=a.model),          # openrouter: OpenAI's Agents SDK + end-of-turn check
       "groq": lambda: GroqLLM(model=a.model)}[a.llm]()
if hasattr(llm, "describe"):
    print("LLM config:", llm.describe(), flush=True)
try:
    out = run_agent(llm, a.goal)
except LLMUnavailable as e:
    sys.exit(f"LLM unavailable (this is the provider, not your code): {e}\n"
             "Try again in a few minutes, or use --llm mock, or set BM_LLM_MODEL / --model to another model.")
if hasattr(llm, "model_used"):
    out["llm_model_used"] = llm.model_used
print(json.dumps(out, indent=2))
