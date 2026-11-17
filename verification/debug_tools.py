"""Tools for the debug agent (A3). All deterministic; the LLM only decides which to call and how to interpret."""
import json
from pathlib import Path

import numpy as np

from verification import tools, workloads
from verification.failure_classifier import classify_failure
from verification.knowledge_base import CAUSES, rule_based_candidates

MAX_SHOWN = 6
_BLIND = {"on": False}


def set_blind_mode(on=True):
    """Evaluation only (NOT exposed to the LLM): hide the rule engine's ranked answer from diagnose_report,
    so a benchmark measures the LLM's own reasoning instead of its ability to copy the answer key."""
    _BLIND["on"] = bool(on)


def _load(report_path):
    return json.loads(Path(tools._safe_report_path(report_path)).read_text())


def _failures(rep):
    return [dict(r, diagnosis=classify_failure(r)) for r in rep["results"] if not r["passed"]]


def diagnose_report(report_path: str):
    """Group failures by category, show the most informative ones and the knowledge-base candidate causes."""
    rep = _load(report_path)
    fails = _failures(rep)
    cats = {}
    for f in fails:
        c = f["diagnosis"]["category"]; cats[c] = cats.get(c, 0) + 1
    shown = [{k: f.get(k) for k in ("test", "shape", "diagnosis", "error", "mapping_check", "first_failing_layer", "reason")}
             for f in fails[:MAX_SHOWN]]
    kind = "workload_report (has mapper pass counts)" if any("mapping_check" in r for r in rep["results"]) \
        else "matmul_suite_report (no mapper involved)"
    out = {"report_kind": kind, "total": rep["total"], "failed": len(fails), "failure_categories": cats,
           "examples": shown, "examples_shown": len(shown), "evidence_level": rep.get("level")}
    if not _BLIND["on"]:
        cands = rule_based_candidates(fails)
        out["rule_based_candidates"] = cands
        out["candidate_details"] = {c: {k: CAUSES[c][k] for k in ("suspect", "fix", "next_check")} for c in cands[:3]}
    return out


def reproduce_failure(report_path: str, test_name: str):
    """Re-run one failing test with the CURRENT backend to see whether the failure is deterministic."""
    names = {r["test"] for r in _load(report_path)["results"]}
    if test_name not in names:
        return {"error": f"'{test_name}' is not in that report", "available": sorted(names)[:30]}
    if test_name in workloads.WORKLOADS:
        r = workloads.run_workload(test_name)
    else:
        r = tools.run_case(test_name)
        if "error" in r:
            return {"error": f"cannot rerun '{test_name}' (only matmul cases and synthetic conv workloads supported)"}
    return {"test": test_name, "reproduced": not r["passed"], "diagnosis": classify_failure(r),
            "mapping_check": r.get("mapping_check")}


def check_hypothesis(report_path: str, test_name: str, hypothesis: str):
    """Numerically test a hypothesis about expected vs actual of a failing test."""
    row = next((r for r in _load(report_path)["results"] if r["test"] == test_name), None)
    if row is None or row.get("actual") is None:
        return {"error": f"no comparable numbers for '{test_name}'"}
    e, a = np.asarray(row["expected"], float), np.asarray(row["actual"], float)
    if hypothesis == "negated":
        ok = e.shape == a.shape and bool(np.allclose(a, -e)); ev = "actual == -expected"
    elif hypothesis == "transposed":
        ok = e.ndim == 2 and a.T.shape == e.shape and bool(np.allclose(a.T, e)); ev = "actual.T == expected"
    elif hypothesis == "constant_offset":
        d = a - e if e.shape == a.shape else None
        ok = d is not None and bool(np.allclose(d, d.flat[0])) and not np.isclose(d.flat[0], 0)
        ev = f"actual - expected is constant ({d.flat[0]:.4g})" if ok else "difference is not constant"
    elif hypothesis == "scaled":
        m = e != 0
        r = a[m] / e[m] if e.shape == a.shape and m.any() else None
        ok = r is not None and bool(np.allclose(r, r[0])) and not np.isclose(r[0], 1)
        ev = f"actual = {r[0]:.4g} x expected" if ok else "ratio is not constant"
    elif hypothesis == "localized_rows":
        bad = [i for i in range(e.shape[0]) if e.shape == a.shape and not np.allclose(a[i], e[i])] if e.ndim == 2 else []
        ok = 0 < len(bad) < e.shape[0]; ev = f"differing rows/channels: {bad}"
    else:
        return {"error": "unknown hypothesis", "allowed": ["negated", "transposed", "constant_offset", "scaled", "localized_rows"]}
    return {"test": test_name, "hypothesis": hypothesis, "explains_failure": ok, "evidence": ev}


DEBUG_TOOLS = {"diagnose_report": diagnose_report, "reproduce_failure": reproduce_failure,
               "check_hypothesis": check_hypothesis}
_RP = {"report_path": {"type": "string"}}
DEBUG_TOOL_SCHEMAS = [
    {"name": "diagnose_report", "description": "Group the failures in a report by category, show examples and candidate root causes from the knowledge base.",
     "input_schema": {"type": "object", "properties": dict(_RP)}},
    {"name": "reproduce_failure", "description": "Re-run one failing test with the current backend to check that the failure is deterministic.",
     "input_schema": {"type": "object", "properties": dict(_RP, test_name={"type": "string"})}},
    {"name": "check_hypothesis", "description": "Numerically test a hypothesis (negated, transposed, constant_offset, scaled, localized_rows) on one failing test.",
     "input_schema": {"type": "object", "properties": dict(_RP, test_name={"type": "string"}, hypothesis={"type": "string"})}},
]


def call_debug_tool(name, arguments):
    """Untrusted-LLM entry point: unknown tools/args rejected, report paths confined to reports/*.json."""
    if name not in DEBUG_TOOLS:
        return {"error": f"unknown tool '{name}'", "available": sorted(DEBUG_TOOLS)}
    allowed = next(x for x in DEBUG_TOOL_SCHEMAS if x["name"] == name)["input_schema"]["properties"]
    args = dict(arguments or {})
    if set(args) - set(allowed):
        return {"error": f"unknown arguments {sorted(set(args) - set(allowed))}; allowed: {sorted(allowed)}"}
    try:
        return DEBUG_TOOLS[name](**args)
    except (TypeError, ValueError, FileNotFoundError, KeyError) as e:
        return {"error": f"{type(e).__name__}: {e}"}
