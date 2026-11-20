"""
Mapping agent (A1). The LLM picks a target for each IR node; code derives tile counts and boundary traffic, and
validate_partition alone decides whether the plan is legal. The system's plan: validated LLM plan > validated baseline > none.

HONESTY: each node still has exactly one legal target (matrix op -> X2, other ops -> eFPGA), so the only legal target
plan IS the baseline (the report says so in search_space). With the chip hardware config (v1.0) every plan also carries
a chip view (32 x 32 X2, 2 blocks): which layers stay programmed is chosen by code (chip_x2.best_resident_set) and
checked by validate_partition. Moving matrix ops to the eFPGA needs per-operator resource estimates (RTL).

SCHEDULE (chip config only): the real decision on the chip is the X2 tile schedule (compiler/x2_schedule.py): which layers
stay programmed and which of the 2 blocks programs / computes each streamed tile. The LLM may propose one; validate_schedule
decides legality and computes the cycles (calibrated on the X2 engine simulations); code computes its own best_schedule.
The system keeps the LLM's schedule only if it is legal and at least as fast as code's; otherwise code's.
"""
import copy
from pathlib import Path

from x2c.agents import harness
from x2c.compiler import x2_schedule as xs
from x2c.compiler.partition import (allowed_targets, chip_view, has_chip, load_hardware, plan_from_targets,
                                propose_partition_baseline, search_space_size, validate_partition)

EVIDENCE_LEVEL = "static_plan_check_against_chip_profile_and_behavioral_X2_timing (NOT RTL, NOT silicon)"
MAX_ERRORS_SHOWN = 20

MAPPING_PROMPT = """You are the target-mapping agent (A1) for a compiler flow onto X2 compute-in-memory and a FABulous eFPGA.
Rules:
1. Call describe_graph first. It lists every node, the targets it may use, and the hardware facts (from the chip repo;
   the weight scheme is marked with its source).
2. Call propose_assignment with a target for EVERY node. Only the last proposal counts. If it reports errors, fix them and propose again.
3. You choose targets only. Code computes tile counts and traffic; never state numbers that did not come from a tool.
4. A plan is legal only if the validator says so. Never claim legality yourself.
5. Evidence level is a static check against the chip profile; never claim RTL, bitstream or silicon results.
6. Finish with a short plain-text summary of the plan."""

SCHEDULE_PROMPT = """
Chip schedule (after a legal propose_assignment):
7. Call describe_schedule. It lists every X2 layer, the program and compute cycles of each tile (calibrated cost model),
   which layers each layer depends on, the resident sets that can be legal, the policies, and the latency of today's schedule.
8. Call propose_schedule with a resident list and a policy (or an explicit tile list). Programming is far slower than
   computing, so using both blocks so that one programs while the other computes usually wins; but you decide.
   The validator computes the latency; code also computes its own best schedule and keeps the faster legal one.
9. Never state cycle numbers that did not come from a tool. Timing is a calibrated behavioral-model estimate, not silicon."""

MAPPING_TOOL_SCHEMAS = [
    {"name": "describe_graph", "description": "List the IR nodes with their allowed targets, and the hardware facts.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "propose_assignment", "description": "Propose a target for every node; returns the validator's verdict. The last proposal counts.",
     "input_schema": {"type": "object", "properties": {
         "targets": {"type": "object", "description": "node name -> 'x2', 'efpga' or 'io'", "additionalProperties": {"type": "string"}},
         "rationale": {"type": "string"}}, "required": ["targets"]}},
    {"name": "describe_schedule", "description": "Chip only: X2 layers, per-tile program/compute cycles, dependencies, legal resident sets, policies, today's latency.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "propose_schedule", "description": "Chip only: propose which layers stay programmed and how the 2 blocks are used; returns the validator's verdict and cycles. The last proposal counts.",
     "input_schema": {"type": "object", "properties": {
         "resident": {"type": "array", "items": {"type": "string"}, "description": "X2 layers kept programmed"},
         "policy": {"type": "string", "description": "earliest_free | round_robin | single_block (ignored if tiles is given)"},
         "tiles": {"type": "array", "description": "optional explicit issue order: [[layer, tile_index, block], ...] covering every X2 tile"},
         "rationale": {"type": "string"}}, "required": ["resident"]}},
]


class _Session:
    """The agent's tools for one graph. All deterministic; the LLM only decides what to call."""
    def __init__(self, g, hw):
        self.g, self.hw = g, hw
        self.plan, self.errors, self.rationale, self.proposals = None, None, None, 0
        self.schedule, self.schedule_errors, self.schedule_metrics, self.schedule_rationale, self.schedule_proposals = None, None, None, None, 0

    def describe_graph(self):
        return {"graph": self.g.name, "search_space_size": search_space_size(self.g),
                "hardware": {k: self.hw[k] for k in ("version", "status", "x2", "efpga", "weights") if k in self.hw},
                "nodes": [{"name": n.name, "op": n.op, "inputs": n.inputs, "elements": n.out.numel(),
                           "allowed_targets": allowed_targets(n)} for n in self.g.nodes]}

    def propose_assignment(self, targets, rationale=""):
        if not isinstance(targets, dict):
            return {"error": "targets must be an object mapping node name to target"}
        self.proposals += 1
        self.plan = plan_from_targets(self.g, self.hw, targets, basis="llm proposal")
        self.errors = validate_partition(self.g, self.plan, self.hw)
        self.rationale = str(rationale)[:1000]
        out = {"legal": not self.errors, "errors": self.errors[:MAX_ERRORS_SHOWN], "errors_total": len(self.errors)}
        if not self.errors:
            out["totals"] = self.plan["totals"]
        return out

    def _chip_plan(self):
        if not has_chip(self.hw):
            raise ValueError("the hardware config has no chip X2: there is no schedule to choose")
        if self.plan is None or self.errors:
            raise ValueError("propose a legal assignment first (propose_assignment)")
        return self.plan

    def describe_schedule(self):
        chip = self._chip_plan()["chip"]
        costs, deps = xs.tile_costs(chip, self.hw), xs.x2_deps(self.g, chip["layers"])
        today = xs.materialize(self.g, chip, self.hw, chip["resident"], "single_block")
        _, m = xs.validate_schedule(self.g, chip, self.hw, today)
        return {"blocks": self.hw["x2"]["blocks"], "policies": list(xs.POLICIES), "resident_options": xs.resident_options(chip, self.hw),
                "layers": [{"name": k, "depends_on": deps[k], "tiles": [{"program_cycles_upper": c["program"], "compute_cycles": c["compute"]}
                                                                       for c in costs[k]]} for k in xs.layer_order(self.g, costs)],
                "today": {"resident": chip["resident"], "policy": "single_block", "latency_cycles": (m or {}).get("latency_cycles")},
                "cost_model": xs.CALIBRATION, "assumptions": xs.ASSUMPTIONS}

    def propose_schedule(self, resident, policy="earliest_free", tiles=None, rationale=""):
        plan = self._chip_plan()
        if not isinstance(resident, list) or not all(isinstance(r, str) for r in resident):
            return {"error": "resident must be a list of layer names"}
        self.schedule_proposals += 1
        chip = chip_view(self.g, self.hw, plan["assignments"], resident)
        sched = {"resident": sorted(resident), "tiles": tiles, "policy": "explicit"} if tiles is not None else \
            xs.materialize(self.g, chip, self.hw, resident, policy)
        trial = dict(plan, chip=chip)
        errs = validate_partition(self.g, trial, self.hw)
        s_errs, metrics = xs.validate_schedule(self.g, chip, self.hw, sched)
        self.schedule, self.schedule_errors, self.schedule_metrics = sched, errs + s_errs, metrics
        self.schedule_rationale = str(rationale)[:1000]
        out = {"legal": not self.schedule_errors, "errors": self.schedule_errors[:MAX_ERRORS_SHOWN], "errors_total": len(self.schedule_errors)}
        if not self.schedule_errors:
            out["metrics"] = metrics
        return out

    def call(self, name, arguments):
        """Untrusted-LLM entry point: unknown tools and arguments are rejected."""
        schema = next((s for s in MAPPING_TOOL_SCHEMAS if s["name"] == name), None)
        if schema is None:
            return {"error": f"unknown tool '{name}'", "available": [s["name"] for s in MAPPING_TOOL_SCHEMAS]}
        args = dict(arguments or {})
        allowed = schema["input_schema"]["properties"]
        if set(args) - set(allowed):
            return {"error": f"unknown arguments {sorted(set(args) - set(allowed))}; allowed: {sorted(allowed)}"}
        try:
            return getattr(self, name)(**args)
        except (TypeError, ValueError, KeyError, AttributeError) as e:
            return {"error": f"{type(e).__name__}: {e}"}


def end_check(session, tools_used):
    """A1's end-of-turn check: the agent may stop only with a plan the VALIDATOR accepts (validate_partition) and, on the
    chip, a schedule it accepts (validate_schedule). Which plan and schedule the system keeps is still decided after the
    run by code (llm plan > baseline; llm schedule only if at least as fast as code's best). Returns the reason or None."""
    if "describe_graph" not in tools_used:
        return "Call describe_graph first; it lists every node, its allowed targets and the hardware facts."
    if session.plan is None:
        return "Call propose_assignment with a target for every node (one entry per node: 'node_name=target')."
    if session.errors:
        return (f"The validator rejected your last assignment ({len(session.errors)} error(s); first: {str(session.errors[0])[:200]}). "
                "Fix it and call propose_assignment again. Only the validator decides what is legal.")
    if has_chip(session.hw) and (session.plan.get("chip") or {}).get("layers"):
        if session.schedule is None:
            return "The assignment is legal. Now call describe_schedule, then propose_schedule with a resident list and a policy."
        if session.schedule_errors:
            return (f"The validator rejected your last schedule (first error: {str(session.schedule_errors[0])[:200]}). "
                    "Fix it and call propose_schedule again.")
    return None


def final_plan_chip_summary(plan):
    c = plan.get("chip")
    return None if c is None else {"totals": c["totals"], "resident": c["resident"], "findings": c["findings"]}


def choose_schedule(g, hw, plan, session):
    """The SYSTEM's schedule for a final plan (chip config only): the LLM's if it is legal for this plan and at least as fast
    as code's best_schedule, else code's. Returns (plan with chip.resident set to the chosen schedule's, schedule report)."""
    chip = plan["chip"]
    best, best_m, tried = xs.best_schedule(g, chip, hw)
    today = xs.materialize(g, chip, hw, chip["resident"], "single_block")
    _, today_m = xs.validate_schedule(g, chip, hw, today)
    llm = {"proposed": session.schedule is not None, "proposals": session.schedule_proposals,
           "rationale_unverified": session.schedule_rationale}
    llm_ok = False
    if session.schedule is not None:
        res = session.schedule.get("resident") if isinstance(session.schedule, dict) else None
        lchip = chip_view(g, hw, plan["assignments"], res if isinstance(res, list) else [])
        errs = validate_partition(g, dict(plan, chip=lchip), hw)
        s_errs, lm = xs.validate_schedule(g, lchip, hw, session.schedule)
        llm.update(legal=not (errs or s_errs), errors=(errs + s_errs)[:MAX_ERRORS_SHOWN], metrics=lm,
                   policy=session.schedule.get("policy") if isinstance(session.schedule, dict) else None)
        llm_ok = not (errs or s_errs)
    if llm_ok and (best_m is None or llm["metrics"]["latency_cycles"] <= best_m["latency_cycles"]):
        sched, metrics, source = session.schedule, llm["metrics"], "llm_validated"
    elif best is not None:
        sched, metrics = best, best_m
        why = ("llm proposed no schedule" if session.schedule is None else "llm schedule rejected" if not llm_ok else
               f"llm schedule legal but {round(100 * (llm['metrics']['latency_cycles'] / best_m['latency_cycles'] - 1), 1)} % slower")
        source = f"code_best ({why})"
    else:
        return plan, {"source": "none (no legal schedule)", "candidates_tried": tried}
    final = copy.deepcopy(plan)
    final["chip"] = chip_view(g, hw, final["assignments"], sched["resident"])
    if validate_partition(g, final, hw):                     # never hand out a plan the validator rejects
        return plan, {"source": "none (chosen schedule made the plan illegal)", "candidates_tried": tried}
    gain = None if not today_m else round(100 * (1 - metrics["latency_cycles"] / today_m["latency_cycles"]), 1)
    return final, {"source": source, "resident": sched["resident"], "policy": sched.get("policy"), "metrics": metrics,
                   "tiles": sched["tiles"], "today": {"resident": chip["resident"], "policy": "single_block", "metrics": today_m},
                   "latency_reduction_vs_today_percent": gain, "code_best": {"resident": best["resident"], "policy": best["policy"],
                                                                             "metrics": best_m}, "candidates_tried": tried,
                   "llm": llm, "cost_model": xs.CALIBRATION, "assumptions": xs.ASSUMPTIONS, "x2_facts": xs.x2_facts.summary(),
                   "evidence_level": "calibrated behavioral-model schedule estimate (NOT a simulation of the schedule, NOT silicon)"}


def run_mapping_agent(llm, g, hw=None, max_steps=5, log_path="reports/mapping_agent_log.jsonl"):
    """llm: harness.OpenRouter(...) (OpenAI's Agents SDK via OpenRouter, then Groq if no model is left), or an object with start /
    send_tool_results (the stand-in MockMappingLLM the deploy stage uses, GroqLLM, test doubles). Code decides the plan."""
    hw = hw or load_hardware()
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    session, chip = _Session(g, hw), has_chip(hw)
    goal = f"Choose a target for every node of graph {g.name}." + (" Then choose the X2 schedule." if chip else "")
    log = harness.jsonl_logger(log_path)
    run = harness.run(llm, MAPPING_PROMPT + (SCHEDULE_PROMPT if chip else ""), goal,
                      max_steps=max_steps, max_model_calls=max_steps + 7, log=log,
                      end_check=lambda reply, used: end_check(session, used),
                      tools=dict(schemas=MAPPING_TOOL_SCHEMAS, call=session.call,
                                  on_tool=lambda step, name, args, res: log("tool", step=step, tool=name, ok="error" not in res,
                                                                            legal=res.get("legal"))))
    text, stopped, steps, llm_error = run["text"], run["stopped"], run["steps"], run["llm_error"]
    extra = {"end_of_turn_sendbacks": run["continues"], "model": run["model"], "cost_usd": run.get("cost_usd")}

    baseline = propose_partition_baseline(g, hw)
    baseline_errors = validate_partition(g, baseline, hw)
    llm_legal = session.plan is not None and not session.errors
    reason = ("llm unavailable" if llm_error else "no proposal" if session.plan is None else "llm plan rejected")
    if llm_legal:
        final, source = session.plan, "llm_validated"
    elif not baseline_errors:
        final, source = baseline, f"baseline_fallback ({reason})"
    else:
        final, source = None, f"no_legal_plan ({reason}; baseline also illegal)"
    schedule = None
    if final is not None and has_chip(hw) and final.get("chip", {}).get("layers"):
        final, schedule = choose_schedule(g, hw, final, session)
    same = llm_legal and {k: v["target"] for k, v in session.plan["assignments"].items()} == \
        {k: v["target"] for k, v in baseline["assignments"].items()}
    space = search_space_size(g)
    return {"final_plan": final, "final_source": source, "stopped": stopped, "steps": steps, "llm_error": llm_error,
            "proposals": session.proposals, "llm_plan_legal": llm_legal, "llm_errors": (session.errors or [])[:MAX_ERRORS_SHOWN],
            "matches_baseline": same, "llm_rationale_unverified": session.rationale, "llm_summary_unverified": text[:1000],
            "search_space": {"legal_plans": space, "note": "exactly one legal plan exists: there is no placement decision until a fabric budget is known"
                             if space == 1 else "more than one legal plan exists"},
            "hardware_profile": {"version": hw["version"], "status": "assumed" if "ASSUMED" in hw.get("version", "") else "chip_profile"},
            "chip": (final_plan_chip_summary(final) if final else None), "schedule": schedule,
            "evidence_level": EVIDENCE_LEVEL, "audit_log": log_path, **extra}
