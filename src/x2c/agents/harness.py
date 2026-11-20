"""
The agents' harness: OpenAI's Agents SDK (the agent loop for GPT models), wrapped in ONE function. The models are
reached through OpenRouter (one key, OpenAI-compatible API, many models).

This is the only file that imports the SDK. An agent gives it a system prompt, a task, its tools (our JSON tool schemas
and the function that runs them) and an END-OF-TURN CHECK. The SDK runs the loop (model call -> tool calls -> model
call ...). When the agent ends its turn with a final answer, the check runs: it returns None ("done, accepted") or a
reason; with a reason the conversation continues with that reason as the next message (at most max_continues times).
The check is code, not the model's choice.

    run = converse(OpenRouter(), system, task, schemas, call, end_check=check)
    run["text"], run["stopped"], run["tools_used"], run["continues"], run["llm_error"], run["cost_usd"]

Only our tools exist for the agent, and the SDK's tracing (which would send the run to OpenAI's trace dashboard) is off.
Models: BM_LLM_MODEL (default openai/gpt-5.6-luna, OpenRouter's name for it), then BM_LLM_FALLBACKS (none by default).
The next model is tried when one is busy (5xx) or rate limited (429); a model that does not exist (404) is skipped for
the rest of the run. Every request asks for at most BM_LLM_MAX_TOKENS output tokens (default 8000): OpenRouter reserves
credit for the maximum, so an unset limit could refuse a request on a small balance. When the OpenRouter credit is used
up, or no model is left, and GROQ_API_KEY is set, the task runs once more on Groq through
plain_loop (our small fallback loop, same tools, same end-of-turn check). With no model at all the agent returns
llm_unavailable and its code decides (rules, baseline, or "not verified"). A wrong or missing key is NOT that: it raises.
Each task has a spending cap: BM_LLM_MAX_USD (default 1.00 US dollar), checked after every model call from the tokens
the provider reports and the list prices below (an estimate; the exact amount is on openrouter.ai/activity).

run(...) is what the agents call: an OpenRouter(...) spec goes to the Agents SDK; any object with start /
send_tool_results (the no-model stand-ins the pipeline uses, Groq, test doubles) goes to plain_loop.
Key: OPENROUTER_API_KEY in the environment (never in a file of this project).
"""
import asyncio
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

OPENROUTER_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-5.6-luna"
DEFAULT_FALLBACKS = ()
DEFAULT_MAX_USD = 1.00
DEFAULT_MAX_TOKENS = 8000
PRICES = {"openai/gpt-5.6-luna": (0.20, 1.20), "openai/gpt-5.6-terra": (2.00, 12.00),     # US$ per 1M tokens in / out
          "openai/gpt-5.6-sol": (5.00, 30.00)}
_SKIP = {}                                   # model -> why it is skipped for the rest of this run (not available)


def reset_skipped():
    """Forget the skipped models (tests; a new run)."""
    _SKIP.clear()


def estimate_usd(model, input_tokens, output_tokens):
    """List-price estimate for a model in PRICES, else None (unknown model: only the tokens are reported)."""
    price = PRICES.get(model) or PRICES.get(f"openai/{model}")
    return None if price is None else (input_tokens * price[0] + output_tokens * price[1]) / 1e6


def classify(status, code=""):
    """auth | billing | quota | missing | busy | other, from the HTTP status and the error code.
    auth and other are raised; billing (402: no credit left) ends the run on OpenRouter; quota and busy try the next
    model; missing skips it."""
    if status in (401, 403):
        return "auth"
    if code == "insufficient_quota" or status == 402:
        return "billing"
    if status == 429:
        return "quota"
    if status == 404 or code == "model_not_found":
        return "missing"
    if status is None or status >= 500:              # None: no answer at all (connection, timeout)
        return "busy"
    return "other"


WHY = {"billing": "OpenRouter credit used up", "quota": "rate limit (429)", "missing": "not available (404)",
       "busy": "busy or not reachable"}


class HarnessUnavailable(RuntimeError):
    """The SDK is not installed in this environment."""


class BudgetReached(RuntimeError):
    """The task reached its spending cap (BM_LLM_MAX_USD)."""


@dataclass
class OpenRouter:
    """Which model(s) to use (OpenRouter names, e.g. openai/gpt-5.6-luna). base_url: an offline stand-in for OpenRouter
    (tests only; no key needed)."""
    model: str | None = None
    fallbacks: tuple = ()
    base_url: str | None = None
    api_retries: int = 2
    max_usd: float | None = None
    max_tokens: int | None = None
    groq_fallback: bool = True          # last resort when no model is left: Groq through plain_loop
    extra: dict = field(default_factory=dict)

    def models(self):
        first = self.model or os.environ.get("BM_LLM_MODEL") or DEFAULT_MODEL
        env_fb = tuple(m.strip() for m in os.environ.get("BM_LLM_FALLBACKS", "").split(",") if m.strip())
        rest = self.fallbacks or env_fb or DEFAULT_FALLBACKS
        return [first] + [m for m in rest if m != first]

    def budget(self):
        return float(self.max_usd if self.max_usd is not None else os.environ.get("BM_LLM_MAX_USD", DEFAULT_MAX_USD))

    def output_limit(self):
        return int(self.max_tokens or os.environ.get("BM_LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS))

    def describe(self):
        return {"harness": "openai-agents via OpenRouter", "models": self.models(), "max_usd_per_task": self.budget(),
                "max_output_tokens": self.output_limit(), **({"base_url": self.base_url} if self.base_url else {})}


def _sdk():
    try:
        import agents
        from openai import AsyncOpenAI
    except ImportError as e:
        raise HarnessUnavailable("the agent harness is not installed: uv sync --extra agents") from e
    agents.set_tracing_disabled(True)
    return agents, AsyncOpenAI


def converse(spec, system, task, schemas, call, end_check=None, *, max_model_calls=12, max_continues=2, log=None,
             on_tool=None):
    """Run one agent task. end_check(text, tools_used) -> None (accept) or a reason (send the agent back).
    Returns {text, stopped, steps, tools_used, continues, llm_error, model, tokens, cost_usd}. stopped: finished |
    step_limit_reached | budget_reached | llm_unavailable. Raises on a missing/wrong key and on programming errors."""
    sdk, AsyncOpenAI = _sdk()
    log = log or (lambda kind, **data: None)
    if not spec.base_url and not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not set in this terminal (see docs/agents.md)")
    errors, tokens = [], {"input": 0, "output": 0}
    cost = 0.0
    candidates = spec.models()
    for model in candidates:
        if model in _SKIP:
            errors.append(f"{model}: {_SKIP[model]} (skipped)")
            continue
        state = {"tools_used": [], "continues": 0, "tokens": {"input": 0, "output": 0}, "cost": 0.0}
        try:
            with _quiet_sdk(log):
                text, stopped = asyncio.run(_one_run(sdk, AsyncOpenAI, spec, model, system, task, schemas, call, end_check,
                                                     state, max_model_calls, max_continues, log, on_tool))
        except Exception as e:  # noqa: BLE001  (classified: a wrong key or a bug is raised, never hidden behind a fallback)
            status, code = getattr(e, "status_code", None), (getattr(e, "code", None) or "")
            if status is None and not _is_connection_error(e):
                raise
            why = classify(status, code)
            _add(tokens, state["tokens"])
            cost += state["cost"]
            if why in ("auth", "other"):
                raise
            log("model_unavailable", model=model, why=why, error=f"{type(e).__name__}: {str(e)[:300]}")
            errors.append(f"{model}: {WHY[why]}")
            if why == "billing":
                print(f"({WHY[why]}: add credit on openrouter.ai/credits)", end=" ", file=sys.stderr, flush=True)
                break                                             # every model draws on the same credit
            if why == "missing":
                _SKIP[model] = WHY[why]
            rest = [m for m in candidates[candidates.index(model) + 1:] if m not in _SKIP]
            print(f"({model}: {WHY[why]}{'; trying ' + rest[0] if rest else '; no model left'})",
                  end=" ", file=sys.stderr, flush=True)
            continue
        _add(tokens, state["tokens"])
        cost += state["cost"]
        return {"text": text, "stopped": stopped, "steps": len(state["tools_used"]), "tools_used": state["tools_used"],
                "continues": state["continues"], "llm_error": None, "model": model, "tokens": tokens,
                "cost_usd": round(cost, 6)}
    return {"text": "", "stopped": "llm_unavailable", "steps": 0, "tools_used": [], "continues": 0,
            "llm_error": "no model available: " + " | ".join(errors), "model": None, "tokens": tokens,
            "cost_usd": round(cost, 6)}


def _add(total, part):
    for k in total:
        total[k] += part[k]


def _is_connection_error(e):
    return type(e).__name__ in ("APIConnectionError", "APITimeoutError", "ModelTimeoutError")


async def _one_run(sdk, AsyncOpenAI, spec, model, system, task, schemas, call, end_check, state, max_model_calls,
                   max_continues, log, on_tool):
    client = AsyncOpenAI(base_url=spec.base_url or OPENROUTER_URL, api_key=os.environ.get("OPENROUTER_API_KEY") or "test",
                         max_retries=spec.api_retries)
    agent = sdk.Agent(name="x2c", instructions=system, tools=[_sdk_tool(sdk, s, call, state, log, on_tool) for s in schemas],
                      model=sdk.OpenAIChatCompletionsModel(model, openai_client=client),      # OpenRouter: chat completions
                      model_settings=sdk.ModelSettings(max_tokens=spec.output_limit()), **spec.extra)
    budget, calls = spec.budget(), [0]

    class Meter(sdk.RunHooks):
        """After every model call: count the tokens, estimate the cost, stop the task at the spending cap."""
        async def on_llm_end(self, context, agent_, response):
            calls[0] += 1
            usage = getattr(response, "usage", None)
            state["tokens"]["input"] += int(getattr(usage, "input_tokens", 0) or 0)
            state["tokens"]["output"] += int(getattr(usage, "output_tokens", 0) or 0)
            state["cost"] = estimate_usd(model, state["tokens"]["input"], state["tokens"]["output"]) or 0.0
            if state["cost"] > budget:
                raise BudgetReached(f"estimated ${state['cost']:.3f} > cap ${budget:.2f} (BM_LLM_MAX_USD)")

    log("start", model=model, tools=[s["name"] for s in schemas], max_usd=budget)
    items = task
    while True:
        try:
            result = await sdk.Runner.run(agent, items, max_turns=max(1, max_model_calls - calls[0]), hooks=Meter())
        except sdk.MaxTurnsExceeded:
            return "", "step_limit_reached"
        except BudgetReached as e:
            log("budget_reached", detail=str(e))
            return "", "budget_reached"
        reply = str(result.final_output or "")
        reason = None
        if end_check is not None and state["continues"] < max_continues:
            reason = end_check(reply, list(state["tools_used"]))
        log("end_of_turn", continuation=state["continues"], accepted=reason is None, reason=reason)
        if not reason:
            return reply, "finished"
        if calls[0] >= max_model_calls:
            return reply, "step_limit_reached"
        state["continues"] += 1
        items = result.to_input_list() + [{"role": "user", "content": reason}]


def _sdk_tool(sdk, schema, call, state, log, on_tool):
    """One of our tools for the SDK: the JSON schema goes to the model as it is; call(name, args) runs it and checks
    the arguments exactly as for Groq and the stand-ins. Each call is recorded (the end-of-turn checks use the names)."""
    name = schema["name"]
    params = dict(schema.get("input_schema") or {})
    params.setdefault("type", "object")
    params.setdefault("properties", {})

    async def invoke(ctx, args_json):
        try:
            args = json.loads(args_json or "{}")
        except ValueError as e:
            return json.dumps({"error": f"the arguments are not valid JSON: {e}"})
        state["tools_used"].append(name)
        t0 = time.time()
        res = call(name, args if isinstance(args, dict) else {"_invalid_arguments": args})
        if on_tool:
            on_tool(len(state["tools_used"]), name, args, res)
        log("tool", tool=name, seconds=round(time.time() - t0, 2))
        return res if isinstance(res, str) else json.dumps(res, default=str)
    return sdk.FunctionTool(name=name, description=schema.get("description", name), params_json_schema=params,
                            on_invoke_tool=invoke, strict_json_schema=False)


class _quiet_sdk:
    """The SDK logs every failed request as an ERROR with a traceback. Keep those in the agent's log, not on the screen."""
    def __init__(self, log):
        self.log, self.logger, self.level = log, logging.getLogger("openai.agents"), None

    def __enter__(self):
        self.level = self.logger.level
        self.logger.setLevel(logging.CRITICAL)

    def __exit__(self, *exc):
        self.logger.setLevel(self.level)


def plain_loop(llm, task, schemas, call, *, system=None, end_check=None, max_steps=6, max_continues=2, force_final=True,
               on_tool=None, catch_llm_errors=True):
    """The small loop for objects with start / send_tool_results: the no-model stand-ins the pipeline uses, Groq, and
    test doubles. The same end-of-turn check is applied when the object can take a reply (send_text, e.g. Groq); the
    stand-ins cannot and behave exactly as before. Returns the same fields as converse."""
    from x2c.agents.llm_client import LLMUnavailable, is_provider_error
    used, steps, continues, step = [], 0, 0, 0
    text, stopped, llm_error = "", "finished", None
    try:
        action = llm.start(task, schemas) if system is None else llm.start(task, schemas, system_prompt=system)
        while True:
            if action.kind == "final":
                reason = None
                if end_check is not None and callable(getattr(llm, "send_text", None)) and continues < max_continues:
                    reason = end_check(action.text or "", list(used))
                if reason:
                    continues += 1
                    action = llm.send_text(reason)
                    continue
                text = action.text
                break
            step += 1
            if step > max_steps:
                stopped = "step_limit_reached"
                break
            results = []
            for name, args in action.calls:
                res = call(name, args)
                steps += 1
                used.append(name)
                if on_tool:
                    on_tool(step, name, args, res)
                results.append((name, res))
            if step == max_steps and force_final and getattr(llm, "supports_force_final", False):
                action = llm.send_tool_results(results, force_final=True)     # the last tool round: answer now
            else:
                action = llm.send_tool_results(results)
    except LLMUnavailable as e:
        if not catch_llm_errors:
            raise
        llm_error, stopped = str(e)[:400], "llm_unavailable"
    except Exception as e:  # noqa: BLE001
        if not catch_llm_errors or not is_provider_error(e):     # programming errors and auth problems must still raise
            raise
        llm_error, stopped = f"{type(e).__name__}: {str(e)[:350]}", "llm_error"
    return {"text": text, "stopped": stopped, "steps": steps, "tools_used": used, "continues": continues,
            "llm_error": llm_error, "model": getattr(llm, "model_used", None), "cost_usd": None}


def run(llm, system, task, *, tools, end_check=None, max_steps=6, max_continues=2, max_model_calls=None, log=None,
        catch_llm_errors=True):
    """One agent task, whatever the model. tools = dict(schemas, call, system (for plain_loop; None = the model's
    default), force_final, on_tool). The same tools and end-of-turn check are used on every path."""
    loop_args = dict(system=tools.get("system", system), end_check=end_check, max_steps=max_steps, max_continues=max_continues,
                     force_final=tools.get("force_final", True), on_tool=tools.get("on_tool"))
    if not isinstance(llm, OpenRouter):
        return plain_loop(llm, task, tools["schemas"], tools["call"], catch_llm_errors=catch_llm_errors, **loop_args)
    out = converse(llm, system, task, tools["schemas"], tools["call"], end_check,
                   max_model_calls=max_model_calls or 2 * max_steps + 4, max_continues=max_continues, log=log,
                   on_tool=tools.get("on_tool"))
    if out["stopped"] != "llm_unavailable" or not llm.groq_fallback or not os.environ.get("GROQ_API_KEY"):
        return out
    try:
        from x2c.agents.llm_client import GroqLLM
        groq = GroqLLM()
    except Exception as e:  # noqa: BLE001  (groq not installed or misconfigured: report it, the agent's code still decides)
        out["llm_error"] += f" | groq fallback not available: {type(e).__name__}: {str(e)[:120]}"
        return out
    print("(trying Groq)", end=" ", file=sys.stderr, flush=True)
    (log or (lambda *a, **k: None))("groq_fallback", reason=out["llm_error"][:300])
    g = plain_loop(groq, task, tools["schemas"], tools["call"], catch_llm_errors=True, **loop_args)
    g["model"] = f"groq:{g['model']}"
    g["cost_usd"] = out["cost_usd"]
    if g["llm_error"]:
        g["llm_error"] = f"{out['llm_error']} | groq: {g['llm_error']}"
    return g


def parse_json_object(text):
    """The first {...} object in a reply, or None."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
    except ValueError:
        return None
    return v if isinstance(v, dict) else None


def check_json_answer(text, tools_used, *, first_tool, keys, field_name, allowed):
    """A ready-made end-of-turn check for agents that must answer with ONE JSON object: the agent looked at the
    evidence first (first_tool), the reply is a JSON object with all keys, and answer[field_name] is in allowed.
    Returns the reason to send the agent back, or None."""
    if first_tool and first_tool not in tools_used:
        return f"Call {first_tool} first and base your answer on what it returns."
    parsed = parse_json_object(text)
    if parsed is None:
        return "Your final answer must be ONLY one JSON object with keys " + ", ".join(keys) + ", written as your reply text (no tool call)."
    missing = [k for k in keys if k not in parsed]
    if missing:
        return f"Your JSON answer is missing: {', '.join(missing)}. Reply again with the complete JSON object."
    if allowed is not None and parsed.get(field_name) not in allowed:
        return (f"{field_name} '{parsed.get(field_name)}' is not allowed. Choose one from the list in your instructions"
                + (", or 'unknown' if the evidence does not clearly support one." if "unknown" in allowed else "."))
    return None


def jsonl_logger(path):
    """A log(kind, **data) function that appends one JSON line per event to path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("", encoding="utf-8")

    def log(kind, **data):
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.strftime("%H:%M:%S"), "kind": kind, **data}, default=str) + "\n")
    return log
