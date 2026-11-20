"""
The small fallback side of the agents (the main path is OpenAI's Agents SDK, x2c.agents.harness).

Objects with this tiny interface run in harness.plain_loop:
    llm.start(goal, tool_schemas, system_prompt=None) -> Action
    llm.send_tool_results([(name, result), ...])      -> Action
    llm.send_text(text)                               -> Action   (optional: lets the end-of-turn check reply)
An Action is either tool calls (kind="tool_calls") or a final answer (kind="final").

Here: GroqLLM (the last fallback when no OpenRouter model is left, or --llm groq) and the no-model stand-ins (MockLLM,
MockDebugLLM, MockMappingLLM, MockPlannerLLM): fixed, rule-based behaviour the pipeline and the tests use. They are NOT
models and are labelled as such in every report.
"""
import os
import time
from dataclasses import dataclass, field

SYSTEM_PROMPT = """You are the verification agent for a compiler-to-hardware flow (BM Labs X2 / FABulous project).
Your tools run deterministic checks of a hardware behavioral model against an independent Python reference.
Rules:
1. Call run_suite first. If tests fail, call get_failures and use the diagnoses to explain likely causes.
2. Never state a number, pass/fail result or cause that did not come from a tool result.
3. Pass/fail is decided by the tools, never by you.
4. The evidence level is Python behavioral simulation. Never claim RTL, bitstream or silicon was verified.
5. Call run_suite with its DEFAULT arguments first (no arguments). Never reduce test coverage (do not lower n_random).
6. If run_suite shows no failures, also call run_workloads (conv layers tiled onto the PE array) and run_model_workloads (real BMsemi model layers).
7. When run_suite shows no failures, call coverage_gaps, then run_generated_tests with specs that target the missing
   classes (small, varied; never include expected values, the oracle computes them).
8. Then call list_workloads and run_benchmark to check real BMsemi workloads (read-only source). Report the findings it returns.
9. Finish with a short plain-text summary: what ran, counts, failure categories, suggested next check."""


@dataclass
class Action:
    kind: str                                  # "tool_calls" | "final"
    calls: list = field(default_factory=list)  # [(tool_name, args_dict), ...]
    text: str = ""


class MockLLM:
    """Rule-based stand-in for an LLM (NOT a real model). Lets us test the whole loop with no API key."""
    def start(self, goal, tool_schemas, system_prompt=None):
        self.last = None
        return Action("tool_calls", calls=[("run_suite", {})])

    def send_tool_results(self, results):
        name, res = results[0]
        if name == "run_suite" and res.get("failed"):
            return Action("tool_calls", calls=[("get_failures", {"report_path": res["report_path"]})])
        if name == "get_failures":
            cats = sorted({f["diagnosis"]["category"] for f in res.get("failures", [])})
            return Action("final", text=f"[mock] {res['failed_count']} of {res['total']} tests failed. "
                                        f"Diagnosed categories: {cats}. Suggest inspecting the X2 adapter/interface.")
        return Action("final", text="[mock] Suite finished with no failures reported by the tools.")


NON_CHAT_PATTERNS = ("whisper", "orpheus", "tts", "playai", "guard", "embed", "transcribe", "image", "lyria", "robotics",
                     "computer-use", "live", "translate", "native-audio", "compound")
MAX_CANDIDATES = 3          # primary + at most 2 fallbacks: a long chain only hides problems and burns time


def filter_chat_models(models):
    """Split model names into (usable, dropped): speech / image / embedding / safety models cannot run a tool-calling chat."""
    kept, dropped = [], []
    for m in models:
        (dropped if any(p in m.lower() for p in NON_CHAT_PATTERNS) else kept).append(m)
    return kept, dropped


class LLMUnavailable(RuntimeError):
    """The LLM provider could not be reached or kept failing (overload, rate limit, outage)."""


RETRYABLE = {429, 500, 502, 503, 504}
REQUEST_TIMEOUT_S = float(os.environ.get("BM_LLM_TIMEOUT_S", "45"))      # one HTTP request
CONVERSATION_BUDGET_S = float(os.environ.get("BM_LLM_BUDGET_S", "120"))  # all LLM calls of one agent run

_TRANSIENT_NAMES = {"APIConnectionError", "APITimeoutError", "ConnectError", "ReadTimeout", "ConnectTimeout"}


def _is_retryable(e):
    code = getattr(e, "status_code", None) or getattr(e, "code", None)
    return code in RETRYABLE or type(e).__name__ in _TRANSIENT_NAMES


_UNSUITABLE_MSGS = ("does not support chat completions", "requires terms acceptance", "not supported with this model")


def _is_model_unsuitable(e):
    """A 400 that says THIS model cannot do the job (audio model, terms not accepted, no tool calling)."""
    return getattr(e, "status_code", None) == 400 and any(m in str(e).lower() for m in _UNSUITABLE_MSGS)


def is_provider_error(e):
    """An HTTP error from the LLM provider (has a numeric status), excluding auth problems (401/403), which must be fixed by a human."""
    code = getattr(e, "status_code", None)
    return isinstance(code, int) and code not in (401, 403)


def _is_model_missing(e):
    return (getattr(e, "status_code", None) or getattr(e, "code", None)) == 404


def print_progress(msg):
    """Default progress sink: without this, retries look like a hang (one bug can take minutes of silent waiting)."""
    print(f"\n    [llm] {msg}", flush=True)


def _brief(e):
    """One-line error summary for progress output. A 429 says WHICH limit was hit and how much was used."""
    code = getattr(e, "status_code", None) or getattr(e, "code", None) or type(e).__name__
    text = " ".join(str(e).split())
    if code == 429:
        import re
        kind = re.search(r"\((TPM|TPD|RPM|RPD)\)", text)
        nums = re.search(r"Limit\s+(\d+),\s*Used\s+(\d+),\s*Requested\s+(\d+)", text)
        hint = _retry_hint_seconds(e)
        if kind or nums:
            return (f"429 rate limit {kind.group(1) if kind else '?'}"
                    + (f" (limit {nums.group(1)}, used {nums.group(2)}, requested {nums.group(3)})" if nums else "")
                    + (f", provider says wait {hint:g}s" if hint is not None else ""))
    return f"{code}: " + text[:100]


MAX_WAIT_S = float(os.environ.get("BM_LLM_MAX_WAIT_S", "45"))   # a rate-limit wait longer than this is not worth sitting through


def _retry_hint_seconds(e):
    """How long the provider says to wait ("try again in 2.16s", "9m12.5s", "1h2m", "250ms", or a Retry-After header), else None."""
    import re
    m = re.search(r"try again in\s+((?:\d+(?:\.\d+)?(?:ms|h|m|s))+)", str(e), re.I)
    if m:
        scale = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
        return sum(float(v) * scale[u.lower()] for v, u in re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", m.group(1), re.I))
    headers = getattr(getattr(e, "response", None), "headers", None)
    try:
        return float(headers.get("retry-after")) if headers is not None and headers.get("retry-after") else None
    except (TypeError, ValueError):
        return None


def _is_quota_exhausted(e):
    """A 429 that will not clear soon (a daily limit, or a wait far longer than we are willing to sit through):
    retrying only burns time, so the caller should skip this model at once."""
    if getattr(e, "status_code", None) != 429:
        return False
    hint = _retry_hint_seconds(e)
    return "per day" in str(e).lower() or "(TPD)" in str(e) or (hint is not None and hint > MAX_WAIT_S)


def _retry_delay(e, n):
    """Seconds to wait before retry n+1. Normally 2/4/8s backoff. A 429 (rate limit) usually says how long to wait,
    and waiting exactly that long beats guessing. Capped at 30s."""
    backoff = 2 ** (n + 1)
    if getattr(e, "status_code", None) != 429:
        return backoff
    hint = _retry_hint_seconds(e)
    if hint is None:
        return backoff
    return min(30.0, max(1.0, hint + 0.5))


def call_with_fallback(models, attempt, retries, sleep, known_good=None, deadline=None, clock=time.monotonic, notify=None):
    """Provider-neutral resilience: for each model, retry transient errors with 2s/4s/8s backoff, then move to
    the next model. Auth / bad-request errors are NOT retried (they raise immediately).
    A 404 means two different things: a model that has NEVER answered in this session is probably retired or
    unavailable to this account (skip to the next model at once); a model that HAS answered before is much more
    likely hitting a transient routing/capacity problem (retry with backoff first).
    notify(text) is told about every skip, retry and give-up, so waiting is visible.
    attempt(model) -> response. Returns (model_used, response)."""
    known_good = known_good if known_good is not None else set()
    say = notify or (lambda text: None)
    last = None
    for model in models:
        for n in range(retries + 1):
            if deadline is not None and clock() >= deadline:
                raise LLMUnavailable(f"time budget exceeded while calling {list(models)}; last error: {str(last)[:300]}")
            try:
                resp = attempt(model)
                known_good.add(model)
                return model, resp
            except Exception as e:  # noqa: BLE001 - classified below
                if not (_is_retryable(e) or _is_model_missing(e) or _is_model_unsuitable(e)):
                    raise
                last = e
                if _is_quota_exhausted(e):
                    say(f"{model}: skipped, {_brief(e)} - not worth waiting for")
                    break
                if _is_model_unsuitable(e) or (_is_model_missing(e) and model not in known_good):
                    say(f"{model}: skipped ({_brief(e)})")
                    break                                   # unusable here (retired / audio model / terms not accepted)
                if n < retries:
                    delay = _retry_delay(e, n)
                    if deadline is not None and clock() + delay >= deadline:
                        raise LLMUnavailable(f"time budget exceeded while calling {list(models)}; last error: {str(last)[:300]}")
                    say(f"{model}: {_brief(e)}; retry {n + 1}/{retries} in {delay:g}s")
                    sleep(delay)
                else:
                    say(f"{model}: giving up after {retries + 1} attempts ({_brief(e)})")
    raise LLMUnavailable(f"LLM unavailable after retries and fallbacks ({list(models)}): {str(last)[:400]}")


class TokenPacer:
    """Keeps the tokens sent to ONE model under its tokens-per-minute (TPM) limit by waiting BEFORE a request,
    instead of sending it, being refused with a 429 and retrying. (Analogy: a rate limiter in front of an API.)
    wait(estimate) blocks until `estimate` tokens fit in the last `window` seconds; record(n) adds what was used."""
    def __init__(self, limit, clock=time.monotonic, sleep=time.sleep, window=60.0):
        self.limit, self.clock, self._sleep, self.window, self.events = limit, clock, sleep, window, []

    def _used(self):
        now = self.clock()
        self.events = [(t, n) for t, n in self.events if now - t < self.window]
        return now, sum(n for _, n in self.events)

    def wait(self, estimate, notify=None):
        waited = 0.0
        while self.limit > 0:
            now, used = self._used()
            if used + estimate <= self.limit or not self.events:
                break
            need, acc, delay = used + estimate - self.limit, 0, 1.0
            for t, n in self.events:                              # oldest first: when will enough tokens expire?
                acc += n
                if acc >= need:
                    delay = t + self.window - now
                    break
            delay = min(max(delay, 0.5), 30.0)
            if notify and waited == 0:
                notify(f"pacing: waiting ~{delay:.0f}s to stay under the {self.limit} tokens/min limit (avoids a 429)")
            self._sleep(delay)
            waited += delay
            if waited > 120:                                      # never stall forever on a bad estimate
                break
        return waited

    def record(self, tokens):
        self.events.append((self.clock(), int(tokens)))


_PACERS = {}      # one pacer per model, shared by every GroqLLM in this process (the evaluation makes one per bug)


def _pacer_for(model, clock=None, sleep=None):
    limit = int(os.environ.get("BM_GROQ_TPM", "0"))           # tokens/min to stay under; 0 = off (the default: pacing measured SLOWER than retrying)
    if model not in _PACERS or _PACERS[model].limit != limit:
        _PACERS[model] = TokenPacer(limit, clock or time.monotonic, sleep or time.sleep)
    return _PACERS[model]


def _compact(obj):
    """Drop null values (the LLM learns nothing from `"error": null`); every token sent counts against the TPM limit."""
    if isinstance(obj, dict):
        return {k: _compact(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_compact(v) for v in obj]
    return obj


class GroqLLM:
    """Groq (OpenAI-compatible chat completions + tool calling). Reads GROQ_API_KEY from the environment."""
    def __init__(self, model=None, client=None, fallback_models=None, retries=3, sleep=time.sleep):
        if client is None:
            from groq import Groq
            # SDK defaults are a 60 s read timeout AND 2 hidden retries per call; we control retries ourselves
            client = Groq(timeout=REQUEST_TIMEOUT_S, max_retries=0)
        self.client = client
        self.time_budget_s, self.clock = CONVERSATION_BUDGET_S, time.monotonic
        self._deadline = None
        self.model = model or os.environ.get("BM_GROQ_MODEL", "openai/gpt-oss-120b")
        env_fb = os.environ.get("BM_GROQ_FALLBACKS", "openai/gpt-oss-20b")
        fb = fallback_models if fallback_models is not None else [m.strip() for m in env_fb.split(",") if m.strip()]
        kept, dropped = filter_chat_models([self.model] + list(fb))
        if self.model in dropped:
            raise ValueError(f"'{self.model}' is not a chat / tool-calling model (speech, image, embedding or safety model). "
                             f"Check $env:BM_GROQ_MODEL and --model.")
        self.notes = []
        if dropped:
            self.notes.append(f"ignored non-chat fallback models: {dropped}")
        if len(kept) > MAX_CANDIDATES:
            self.notes.append(f"fallback chain capped at {MAX_CANDIDATES - 1}; ignored: {kept[MAX_CANDIDATES:]}")
        self.fallbacks = kept[1:MAX_CANDIDATES]
        self.retries, self._sleep = retries, sleep
        self.notify = print_progress
        self.model_used = self.model
        self._no_reasoning = set()   # models that rejected the reasoning_effort setting
        self._good = set()           # models that have answered at least once in this session
        self.recoveries = 0          # answers rescued from a rejected pseudo tool call (see _recover_answer)

    def _create(self, model, messages, **kw):
        """The ONE place a Groq chat request is made: paces to the TPM limit, lowers gpt-oss reasoning effort
        (hidden reasoning tokens dominate the cost of a simple classification), and records the real token usage."""
        import json
        pacer = _pacer_for(model, self.clock, self._sleep)
        est = int((len(json.dumps(messages)) + len(json.dumps(kw.get("tools") or []))) / 3) + 400
        pacer.wait(est, self.notify)
        effort = os.environ.get("BM_GROQ_REASONING", "low").strip().lower()
        extra = {}
        if effort not in ("", "off", "none") and "gpt-oss" in model and model not in self._no_reasoning:
            extra = {"extra_body": {"reasoning_effort": effort}}
        try:
            resp = self.client.chat.completions.create(model=model, messages=messages, timeout=self._timeout(), **kw, **extra)
        except Exception as e:  # noqa: BLE001
            if extra and getattr(e, "status_code", None) == 400 and "reasoning" in str(e).lower():
                self._no_reasoning.add(model)               # this model/SDK does not take it: never send it again
                resp = self.client.chat.completions.create(model=model, messages=messages, timeout=self._timeout(), **kw)
            else:
                raise
        usage = getattr(resp, "usage", None)
        used = getattr(usage, "total_tokens", None)
        if isinstance(used, (int, float)) and used > 0:     # only REAL usage reported by the provider is counted
            pacer.record(used)
        return resp

    def _timeout(self):
        """Seconds for the next HTTP request: never lets one request run past the conversation budget."""
        if self._deadline is None:
            return REQUEST_TIMEOUT_S
        return max(1.0, min(REQUEST_TIMEOUT_S, self._deadline - self.clock()))

    def _answer_without_tools(self):
        """Last resort for 'Tool choice is none, but model called a tool': ask once more with tools omitted entirely."""
        msgs = self._messages + [{"role": "user", "content":
                                  "Give your final answer now as plain JSON text. Do not call any tool."}]
        _, resp = call_with_fallback([self.model] + self.fallbacks,
                                     lambda m: self._create(m, msgs),
                                     self.retries, self._sleep, self._good, self._deadline, self.clock, self.notify)
        return resp.choices[0].message.content or ""

    def describe(self):
        return (f"model={self.model} fallbacks={self.fallbacks} request_timeout={REQUEST_TIMEOUT_S:.0f}s "
                f"budget={self.time_budget_s:.0f}s" + (f" NOTES: {'; '.join(self.notes)}" if self.notes else ""))

    def _recover_answer(self, e):
        """Some models (e.g. gpt-oss) deliver their FINAL answer as a call to a tool that does not exist
        (typically 'json'); Groq rejects it with 400 tool_use_failed but returns the text in failed_generation.
        If that text is a call to an UNDECLARED tool, its arguments ARE the model's answer: return them as text.
        This is safe because every final answer is validated by code afterwards. Anything else: not recoverable."""
        import json
        import re
        if getattr(e, "status_code", None) != 400:
            return None
        body = getattr(e, "body", None)
        body = body if isinstance(body, dict) else {}
        err = body.get("error", body) if isinstance(body.get("error", body), dict) else {}
        if err.get("code") != "tool_use_failed" and "tool_use_failed" not in str(e) and "tool choice is none" not in str(e).lower():
            return None
        fg = err.get("failed_generation")
        if fg is None:
            m = re.search(r"failed_generation'?:\s*'(\{.*\})'", str(e), re.S)
            fg = m.group(1) if m else None
        try:
            call = json.loads(fg)
            name, args = call["name"], call["arguments"]
        except (TypeError, ValueError, KeyError):
            return None
        declared = {t["function"]["name"] for t in self._tools}
        if name in declared:
            return None                                   # a real tool call that was malformed: do not guess
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                return args or None
        return json.dumps(args) if isinstance(args, dict) else None

    def start(self, goal, tool_schemas, system_prompt=None):
        self._deadline = self.clock() + self.time_budget_s
        self._tools = [{"type": "function",
                        "function": {"name": t["name"], "description": t["description"],
                                     "parameters": t["input_schema"]}} for t in tool_schemas]
        self._messages = [{"role": "system", "content": system_prompt or SYSTEM_PROMPT},
                          {"role": "user", "content": goal}]
        return self._step()

    supports_force_final = True

    def send_text(self, text):
        """The end-of-turn check's reply (harness.plain_loop): one more user message, then the model continues."""
        self._messages.append({"role": "user", "content": text})
        return self._step()

    def send_tool_results(self, results, force_final=False):
        import json
        for call_id, (name, result) in zip(self._pending_ids, results):
            self._messages.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(_compact(result), separators=(",", ":"))})
        return self._step(force_final)

    def _step(self, force_final=False):
        import json
        choice = "none" if force_final else "auto"       # "none" = the model may not call tools this turn
        try:
            self.model_used, resp = call_with_fallback(
                [self.model] + self.fallbacks,
                lambda m: self._create(m, self._messages, tools=self._tools, tool_choice=choice),
                self.retries, self._sleep, self._good, self._deadline, self.clock, self.notify)
        except Exception as e:  # noqa: BLE001
            text = self._recover_answer(e)
            if text is None and "tool choice is none" in str(e).lower():
                text = self._answer_without_tools()          # forced-final turn, but the model tried a tool anyway
            if text is None:
                raise
            self.recoveries += 1
            self._messages.append({"role": "assistant", "content": text})
            return Action("final", text=text)
        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []
        if not tool_calls:
            self._messages.append({"role": "assistant", "content": msg.content or ""})
            return Action("final", text=msg.content or "")
        self._messages.append({"role": "assistant", "content": msg.content or "",
                               "tool_calls": [{"id": tc.id, "type": "function",
                                               "function": {"name": tc.function.name,
                                                            "arguments": tc.function.arguments or "{}"}}
                                              for tc in tool_calls]})
        self._pending_ids, calls = [], []
        for tc in tool_calls:
            self._pending_ids.append(tc.id)
            try:
                args = json.loads(tc.function.arguments or "{}")
                if not isinstance(args, dict):
                    raise ValueError
            except ValueError:
                args = {"_invalid_arguments": str(tc.function.arguments)}   # call_tool will reject it safely
            calls.append((tc.function.name, args))
        return Action("tool_calls", calls=calls)


class MockDebugLLM:
    """Rule-based stand-in for the debug LLM (NOT a real model): runs diagnose_report, answers with the top candidate."""
    def start(self, goal, tool_schemas, system_prompt=None):
        import re
        self.path = re.search(r"report_path=(\S+)", goal).group(1)
        return Action("tool_calls", calls=[("diagnose_report", {"report_path": self.path})])

    def send_tool_results(self, results):
        import json
        name, res = results[0]
        cands = res.get("rule_based_candidates") or []
        cause = cands[0] if cands else "unknown"
        return Action("final", text=json.dumps({
            "cause_id": cause, "confidence": "medium" if cands else "low",
            "explanation": f"[mock] top rule-based candidate from the diagnosis: {cause}",
            "proposed_fix": "see knowledge-base fix for this cause", "next_check": "rerun the failing test after the fix"}))


class MockMappingLLM:
    """Rule-based stand-in for the mapping LLM (NOT a real model): reads the graph, proposes the first allowed target per node."""
    def start(self, goal, tool_schemas, system_prompt=None):
        return Action("tool_calls", calls=[("describe_graph", {})])

    def send_tool_results(self, results, force_final=False):
        name, res = results[0]
        if name == "describe_graph":
            targets = {n["name"]: n["allowed_targets"][0] for n in res["nodes"]}
            return Action("tool_calls", calls=[("propose_assignment", {"targets": targets, "rationale": "[mock] first allowed target"})])
        return Action("final", text="[mock] proposed the first allowed target for every node")


class MockPlannerLLM:
    """Rule-based stand-in for the test-planning LLM (NOT a real model): coverage_gaps, then the baseline plan."""
    def start(self, goal, tool_schemas, system_prompt=None):
        return Action("tool_calls", calls=[("coverage_gaps", {})])

    def send_tool_results(self, results, force_final=False):
        from x2c.verification.test_planner import baseline_plan
        if results[0][0] == "coverage_gaps":
            return Action("tool_calls", calls=[("run_generated_tests", {"specs": baseline_plan()})])
        return Action("final", text="[mock] generated the baseline plan covering the listed gaps")