"""
The agents' harness: Google's Antigravity SDK (the harness Gemini models are trained with), wrapped in ONE function.

This is the only file that imports the SDK. An agent gives it a system prompt, a task, its tools (plain Python functions
with type hints and docstrings) and an END-OF-TURN CHECK. The SDK runs the loop (model call -> tool calls -> model call ...);
when the agent says it is done, the harness calls the check (a StopHook). The check returns None ("done, accepted") or a
reason; with a reason the agent must continue (at most max_continues times). The check is code, not the model's choice.

    run = converse(Gemini(), system, task, tools=[...], end_check=check)
    run["text"], run["stopped"], run["tools_used"], run["continues"], run["llm_error"]

Only our tools exist for the agent: the SDK's own file / shell / web tools are switched off.
Models: BM_LLM_MODEL (default gemini-3.5-flash), then BM_LLM_FALLBACKS (default gemini-3.8-flash, gemini-3.5-flash-lite).
The next model is tried when one is busy (503), out of quota (429) or not available (404). A model whose DAILY free quota
is used up, or that does not exist, is skipped for the rest of the run (no wasted requests). When no Gemini model is left
and GROQ_API_KEY is set, the task runs once more on Groq through plain_loop (our small fallback loop, same tools, same
end-of-turn check). With no model at all the agent returns llm_unavailable and its code decides (rules, baseline, or
"not verified"). A wrong or missing key is NOT that: it raises.

run(...) is what the agents call: a Gemini(...) spec goes to Google's harness; any object with start / send_tool_results
(the no-model stand-ins the pipeline uses, Groq, test doubles) goes to plain_loop.
Key: GEMINI_API_KEY in the environment (never in a file of this project).
"""
import asyncio
import json
import logging
import os
import sys
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "gemini-3.5-flash"
DEFAULT_FALLBACKS = ("gemini-3.8-flash", "gemini-3.5-flash-lite")
AUTH = re.compile(r"\b(401|403)\b|PERMISSION_DENIED|UNAUTHENTICATED|API key not valid|API_KEY_INVALID|invalid API key", re.I)
QUOTA = re.compile(r"\b429\b|RESOURCE_EXHAUSTED|quota|rate.?limit", re.I)
DAILY = re.compile(r"per ?day|PerDay", re.I)
MISSING = re.compile(r"\b404\b|NOT_FOUND|no longer available|is not found|not supported", re.I)
BUSY = re.compile(r"\b503\b|UNAVAILABLE|high demand|overloaded|try again later|network issue|failed to send HTTP request"
                  r"|deadline|timed? ?out", re.I)
_SKIP = {}                                   # model -> why it is skipped for the rest of this run (daily quota, not available)


def reset_skipped():
    """Forget the skipped models (tests; a new day)."""
    _SKIP.clear()


def classify(msg):
    """auth | quota_daily | quota | missing | busy | other  (auth and other are raised, the rest move to the next model)."""
    if AUTH.search(msg) and not QUOTA.search(msg):
        return "auth"
    if QUOTA.search(msg):
        return "quota_daily" if DAILY.search(msg) else "quota"
    if MISSING.search(msg):
        return "missing"
    if BUSY.search(msg):
        return "busy"
    return "other"


WHY = {"quota_daily": "daily free quota used up", "quota": "rate limit (429)", "missing": "not available (404)",
       "busy": "busy (503)"}


class HarnessUnavailable(RuntimeError):
    """The SDK is not installed in this environment."""


@dataclass
class Gemini:
    """Which model(s) to use. base_url: an OpenAI-compatible stand-in model (tests only; no key needed)."""
    model: str | None = None
    fallbacks: tuple = ()
    base_url: str | None = None
    api_retries: int = 2
    groq_fallback: bool = True          # last resort when every Gemini model is out: Groq through plain_loop
    extra: dict = field(default_factory=dict)

    def models(self):
        first = self.model or os.environ.get("BM_LLM_MODEL") or DEFAULT_MODEL
        env_fb = tuple(m.strip() for m in os.environ.get("BM_LLM_FALLBACKS", "").split(",") if m.strip())
        rest = self.fallbacks or env_fb or DEFAULT_FALLBACKS
        return [first] + [m for m in rest if m != first]

    def describe(self):
        return {"harness": "google-antigravity", "models": self.models(), **({"base_url": self.base_url} if self.base_url else {})}


def _sdk():
    try:
        from google import antigravity as ag
        from google.antigravity import types as agt
        from google.antigravity.hooks import hooks
    except ImportError as e:
        raise HarnessUnavailable("the agent harness is not installed: uv sync --extra agents") from e
    return ag, agt, hooks


def converse(spec, system, task, tools, end_check=None, *, max_model_calls=12, max_continues=2, log=None):
    """Run one agent task. end_check(text, tools_used) -> None (accept) or a reason (send the agent back).
    Returns {text, stopped, steps, tools_used, continues, llm_error, model}. stopped: finished | step_limit_reached |
    llm_unavailable. Raises on a missing/wrong key and on programming errors."""
    ag, agt, hooks = _sdk()
    log = log or (lambda kind, **data: None)
    if not spec.base_url and not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not set in this terminal (see docs/agents.md)")
    errors = []
    candidates = spec.models()
    for model in candidates:
        if model in _SKIP:
            errors.append(f"{model}: {_SKIP[model]} (skipped)")
            continue
        state = {"tools_used": [], "continues": 0, "stop_reason": None, "last_reply": None}
        try:
            with _quiet_sdk(log):
                text = asyncio.run(_one_run(ag, agt, hooks, spec, model, system, task, tools, end_check, state,
                                            max_model_calls, max_continues, log))
        except Exception as e:  # noqa: BLE001  (classified: a wrong key or a bug is raised, never hidden behind a fallback)
            msg = f"{type(e).__name__}: {e}"
            kind = classify(msg)
            if kind in ("auth", "other"):
                raise
            if kind in ("quota_daily", "missing"):
                _SKIP[model] = WHY[kind]
            errors.append(f"{model}: {WHY[kind]}")
            log("model_unavailable", model=model, why=kind, error=msg[:300])
            rest = [m for m in candidates[candidates.index(model) + 1:] if m not in _SKIP]
            print(f"({model}: {WHY[kind]}{'; trying ' + rest[0] if rest else '; no Gemini model left'})",
                  end=" ", file=sys.stderr, flush=True)
            continue
        stopped = "step_limit_reached" if "EXCEEDED" in str(state["stop_reason"] or "") else "finished"
        return {"text": text, "stopped": stopped, "steps": len(state["tools_used"]), "tools_used": state["tools_used"],
                "continues": state["continues"], "llm_error": None, "model": model}
    return {"text": "", "stopped": "llm_unavailable", "steps": 0, "tools_used": [], "continues": 0,
            "llm_error": "no model available: " + " | ".join(errors), "model": None}


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
            "llm_error": llm_error, "model": getattr(llm, "model_used", None)}


def run(llm, system, task, *, sdk_tools, legacy, end_check=None, max_steps=6, max_continues=2, max_model_calls=None,
        log=None, catch_llm_errors=True):
    """One agent task, whatever the model. legacy = dict(schemas, call, system (may differ: None = the model's default),
    force_final, on_tool) for plain_loop; sdk_tools are the same tools as plain functions for Google's harness."""
    loop_args = dict(system=legacy.get("system", system), end_check=end_check, max_steps=max_steps, max_continues=max_continues,
                     force_final=legacy.get("force_final", True), on_tool=legacy.get("on_tool"))
    if not isinstance(llm, Gemini):
        return plain_loop(llm, task, legacy["schemas"], legacy["call"], catch_llm_errors=catch_llm_errors, **loop_args)
    out = converse(llm, system, task, sdk_tools, end_check, max_model_calls=max_model_calls or 2 * max_steps + 4,
                   max_continues=max_continues, log=log)
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
    g = plain_loop(groq, task, legacy["schemas"], legacy["call"], catch_llm_errors=True, **loop_args)
    g["model"] = f"groq:{g['model']}"
    if g["llm_error"]:
        g["llm_error"] = f"{out['llm_error']} | groq: {g['llm_error']}"
    return g


class _QuietSDK(logging.Filter):
    """The SDK prints every retry as a long WARNING. Keep those in the agent's log file, not on the screen."""
    NOISE = ("System step error", "use_json_schema is not True")

    def __init__(self, log):
        super().__init__()
        self.log = log

    def filter(self, record):
        msg = record.getMessage()
        if any(n in msg for n in self.NOISE):
            self.log("sdk_warning", message=msg[:400])
            return False
        return True


class _quiet_sdk:
    def __init__(self, log):
        self.f, self.targets = _QuietSDK(log), []

    def __enter__(self):
        root = logging.getLogger()
        self.targets = [root, *root.handlers, logging.getLogger("google_genai.types")]
        for t in self.targets:
            t.addFilter(self.f)

    def __exit__(self, *exc):
        for t in self.targets:
            t.removeFilter(self.f)


async def _one_run(ag, agt, hooks, spec, model, system, task, tools, end_check, state, max_model_calls, max_continues, log):
    wrapped = [_recording(t, state, log) for t in tools]

    @hooks.stop
    def end_of_turn(data: agt.StopArgs) -> agt.StopHookResult:
        state["stop_reason"] = data.stop_reason
        reason = None
        if not data.error_message and data.continuation_count < max_continues and end_check is not None:
            reason = end_check(data.response_text or "", list(state["tools_used"]))
        log("end_of_turn", continuation=data.continuation_count, accepted=reason is None, reason=reason,
            stop_reason=str(data.stop_reason), error=data.error_message or None)
        if reason:
            state["continues"] += 1
            return agt.StopHookResult(decision=agt.StopDecision.CONTINUE, reason=reason)
        state["last_reply"] = data.response_text or ""     # the accepted reply (resp.text() joins every reply of the turn)
        return agt.StopHookResult(decision=agt.StopDecision.ALLOW_STOP)

    common = dict(
        system_instructions=system,
        tools=wrapped,
        hooks=[end_of_turn],
        capabilities=agt.CapabilitiesConfig(enabled_tools=[], enable_subagents=False),     # only our tools
        budget_config=agt.BudgetConfig(max_model_calls=max_model_calls),
        retry_config=agt.RetryConfig(api_retry=agt.ModelAPIRetryConfig(max_retries=spec.api_retries,
                                                                       initial_sleep_duration_ms=1000)),
        **spec.extra,
    )
    with tempfile.TemporaryDirectory(prefix="x2c_agent_") as save_dir:
        if spec.base_url:
            config = ag.LocalOpenAIAgentConfig(model=model, base_url=spec.base_url, save_dir=save_dir, **common)
        else:
            config = ag.LocalAgentConfig(model=model, save_dir=save_dir, **common)
        log("start", model=model, tools=[t.__name__ for t in tools])
        async with ag.Agent(config) as agent:
            resp = await agent.chat(task)
            joined = await resp.text()
        return state["last_reply"] if state["last_reply"] is not None else joined


def _recording(fn, state, log):
    """Same name, signature and docstring (the SDK builds the tool schema from them); records each call."""
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        state["tools_used"].append(fn.__name__)
        t0 = time.time()
        out = fn(*args, **kwargs)
        log("tool", tool=fn.__name__, args={k: (v if len(str(v)) < 200 else str(v)[:200] + "...") for k, v in kwargs.items()},
            seconds=round(time.time() - t0, 2))
        return out if isinstance(out, str) else json.dumps(out, default=str)
    return wrapper


_SIMPLE = {"string": str, "integer": int, "number": float, "boolean": bool}


def _py_type(prop):
    """JSON-schema property -> (Python annotation, takes_json). Gemini tool schemas hold simple values and arrays of them;
    an object, or an array of objects, is passed as a JSON string and parsed back here."""
    kind = prop.get("type")
    if kind in _SIMPLE:
        return _SIMPLE[kind], False
    if kind == "array":
        item = prop.get("items") or {}
        if item.get("type") in _SIMPLE:
            return list[_SIMPLE[item["type"]]], False
        inner = item.get("items") or {}
        if item.get("type") == "array" and inner.get("type") in _SIMPLE:
            return list[list[_SIMPLE[inner["type"]]]], False
    return str, True


def _json_hint(prop):
    item = prop.get("items") or {}
    fields = ", ".join((item.get("properties") or prop.get("properties") or {}).keys())
    shape = "a JSON array of objects" if prop.get("type") == "array" else "a JSON object"
    return f"Pass {shape}" + (f" with the fields {fields}." if fields else ".")


def tools_from_schemas(schemas, call):
    """Plain functions for the harness from our existing JSON tool schemas ({name, description, input_schema}).
    Each function calls call(name, args) with only the arguments the model gave, so the existing argument checks run.
    Every argument is optional unless the schema requires it."""
    import inspect
    fns = []
    for s in schemas:
        props = s.get("input_schema", {}).get("properties", {}) or {}
        required = set(s.get("input_schema", {}).get("required") or [])
        params, json_args, doc_args = [], set(), []
        for name, prop in props.items():
            ann, takes_json = _py_type(prop)
            if takes_json:
                json_args.add(name)
            optional = name not in required
            params.append(inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=None if optional else inspect.Parameter.empty,
                                            annotation=(ann | None) if optional else ann))
            desc = prop.get("description", "")
            if takes_json:
                desc = (desc + " " if desc else "") + _json_hint(prop)
            doc_args.append(f"    {name}: {desc or name.replace('_', ' ')}")

        def make(tool_name, jargs):
            def fn(**kwargs):
                args = {}
                for k, v in kwargs.items():
                    if v is None:
                        continue
                    if k in jargs and isinstance(v, str):
                        try:
                            v = json.loads(v)
                        except ValueError as e:
                            return json.dumps({"error": f"{k} is not valid JSON: {e}"})
                    args[k] = v
                out = call(tool_name, args)
                return out if isinstance(out, str) else json.dumps(out, default=str)
            return fn
        fn = make(s["name"], json_args)
        fn.__name__ = fn.__qualname__ = s["name"]
        fn.__doc__ = s.get("description", s["name"]) + ("\n\nArgs:\n" + "\n".join(doc_args) if doc_args else "")
        fn.__signature__ = inspect.Signature(params, return_annotation=str)
        fn.__annotations__ = {p.name: p.annotation for p in params} | {"return": str}
        fns.append(fn)
    return fns


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
