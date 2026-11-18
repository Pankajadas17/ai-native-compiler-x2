"""
Agent-facing tools. Every function takes plain Python/JSON values and returns a
JSON-serialisable dict, so an LLM agent can call them via function calling.
The LLM chooses WHICH tool to call; these deterministic functions do ALL the work
and decide pass/fail. TOOL_SCHEMAS describes them to the LLM.
"""
import json
from pathlib import Path

import numpy as np

from x2c.bmsemi.x2_adapter import x2_matmul_report
from x2c.commands.bmsemi_verify import run_all, run_one
from x2c.verification.failure_classifier import classify_failure
from x2c.verification.matmul_check import make_cases

DEFAULT_REPORT = "reports/report.json"
WORKLOAD_REPORT = "reports/workload_report.json"
MODEL_WORKLOAD_REPORT = "reports/model_workload_report.json"
GENERATED_REPORT = "reports/generated_report.json"
BENCHMARK_REPORT = "reports/benchmark_report.json"
_GENERATED = []     # test specs accepted so far (state used by coverage_gaps)
REPORTS_DIR = Path("reports").resolve()
_BACKEND = {"fn": x2_matmul_report}


def set_x2_backend(fn=None):
    """Choose which X2 implementation the tools use (default: the real simulator adapter).
    Used by tests / demos to plug in a deliberately broken X2. NOT exposed to the LLM."""
    _BACKEND["fn"] = fn or x2_matmul_report


def run_suite(seed: int = 0, n_random: int = 20, report_path: str = DEFAULT_REPORT, x2_fn=None):
    """Run the whole golden-vs-X2 suite. Returns counts and the failing test names."""
    s = run_all(x2_fn=x2_fn or _BACKEND["fn"], seed=seed, n_random=n_random, report_path=report_path)
    failed = [r["test"] for r in s["results"] if not r["passed"]]
    return {"level": s["level"], "total": s["total"], "passed": s["passed"],
            "failed": failed, "report_path": report_path}


def run_case(name: str = None, A=None, B=None, x2_fn=None):
    """Run ONE case: a named built-in case, or custom matrices A (MxK) and B (KxN)."""
    if A is None:
        known = {n: (a, b) for n, a, b in make_cases(n_random=50)}
        if name not in known:
            return {"error": f"unknown case '{name}'", "known_cases": sorted(known)}
        A, B = known[name]
    else:
        name = name or "custom"
        A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    r = run_one(name, A, B, x2_fn or _BACKEND["fn"])
    r["diagnosis"] = classify_failure(r)
    return r


def run_workloads(report_path: str = WORKLOAD_REPORT, mapping_transform=None):
    """Run conv-layer workloads executed in tiles on the X2 simulator vs an independent direct convolution."""
    from x2c.verification.workloads import run_workload_suite
    from x2c.verification.matmul_check import write_report
    s = write_report(run_workload_suite(mapping_transform), path=report_path)
    return {"level": s["level"], "total": s["total"], "passed": s["passed"],
            "failed": [r["test"] for r in s["results"] if not r["passed"]], "report_path": report_path}


def run_model_workloads(report_path: str = MODEL_WORKLOAD_REPORT, mapping_transform=None, vecmat=None, full=False):
    """Run workloads taken from real BMsemi_Model operations (MNIST edge filter, DeepGate INT8 autoencoder)
    on the X2 simulator and compare with the repository's own operation output."""
    from x2c.verification import model_workloads as mw
    from x2c.verification.matmul_check import write_report
    from x2c.bmsemi.x2_adapter import x2_vecmat
    if not mw.data_available():
        return {"error": "BMsemi_Model example data not found (projects/2_building_blocks, projects/3_example_models)"}
    res = mw.run_model_workload_suite(mapping_transform=mapping_transform, vecmat=vecmat or x2_vecmat, full=full)
    s = write_report(res, path=report_path)
    return {"level": s["level"], "total": s["total"], "passed": s["passed"],
            "failed": [r["test"] for r in s["results"] if not r["passed"]], "report_path": report_path}


def list_workloads():
    """Workloads discovered in the read-only BMsemi_Model repository, normalised to the local schema, plus the stage table."""
    from x2c.inputs.bmsemi_source import ReadOnlyRepo
    from x2c.inputs.catalog import discover
    from x2c.inputs.normalize import all_workloads
    from x2c.bmsemi.benchmark import STAGE_STATUS
    repo = ReadOnlyRepo()
    cat = discover(repo)
    return {"repo_commit": cat["repo_commit"], "operations": len(cat["operations"]), "assets": len(cat["assets"]),
            "workloads": [{"id": w.id, "title": w.title, "data_source": w.data_source, "layers": len(w.layers),
                           "loader": w.loader, "stages": w.stages} for w in all_workloads(repo)],
            "stage_status": [{k: x[k] for k in ("id", "status")} for x in STAGE_STATUS]}


def run_benchmark(workload_id: str = None, report_path: str = BENCHMARK_REPORT, vecmat=None, mapping_transform=None):
    """Run real BMsemi workloads (read-only source) through the verification pipeline. workload_id: one id from
    list_workloads, or omit for all. Returns counts, failing test names, findings and the result digest."""
    from x2c.bmsemi.benchmark import run_benchmark as _run
    from x2c.bmsemi.x2_adapter import x2_vecmat
    try:
        rep = _run(workload_ids=[workload_id] if workload_id else None, vecmat=vecmat or x2_vecmat,
                   mapping_transform=mapping_transform, report_path=report_path)
    except ValueError as e:
        return {"error": str(e)}
    return {"level": rep["level"], "total": rep["total"], "passed": rep["passed"],
            "failed": [r["test"] for r in rep["results"] if not r["passed"]][:20], "report_path": report_path,
            "findings": [f["id"] for f in rep["findings"]], "result_digest_sha256": rep["result_digest_sha256"],
            "repo_read_only_check": rep["repo_read_only_check"]}


def reset_generated():
    _GENERATED.clear()


def get_generated_specs():
    return list(_GENERATED)


def coverage_gaps():
    """Which input classes (shapes, value patterns, INT8 extremes...) are NOT yet exercised by the fixed suite
    plus any generated tests so far. Also returns the spec format for run_generated_tests."""
    from x2c.verification import test_planner as tp
    g = tp.coverage_gaps(_GENERATED)
    g["spec_fields"] = tp.SPEC_HELP
    g["max_specs_per_call"] = tp.MAX_SPECS
    return g


def run_generated_tests(specs=None, report_path: str = GENERATED_REPORT, x2_fn=None):
    """Run proposed test specs. Code validates each spec and builds the matrices; the Python ORACLE computes the
    expected results (a spec cannot contain expected values). Returns pass/fail counts and remaining coverage gaps."""
    from x2c.verification import test_planner as tp
    from x2c.verification.matmul_check import write_report
    acc, rej = tp.validate_plan(specs)
    if not acc:
        return {"error": "no valid specs", "rejected": rej}
    results = []
    for sp in acc:
        A, B = tp.materialize(sp)
        r = run_one(sp["name"], A, B, x2_fn or _BACKEND["fn"])
        r["spec"] = sp
        r["coverage_classes"] = sorted(tp.classify(A, B))
        results.append(r)
    rep = write_report(results, path=report_path)
    known = {x["name"] for x in _GENERATED}
    _GENERATED.extend(sp for sp in acc if sp["name"] not in known)
    return {"level": rep["level"], "total": rep["total"], "passed": rep["passed"],
            "failed": [r["test"] for r in results if not r["passed"]], "report_path": report_path,
            "rejected_specs": rej, "coverage_still_missing": tp.coverage_gaps(_GENERATED)["missing"]}


def get_failures(report_path: str = DEFAULT_REPORT):
    """List failing tests from a saved report, each with a deterministic diagnosis."""
    rep = json.loads(Path(report_path).read_text())
    fails = [dict(r, diagnosis=classify_failure(r)) for r in rep["results"] if not r["passed"]]
    return {"total": rep["total"], "failed_count": len(fails), "failures": fails}


def summarize_report(report_path: str = DEFAULT_REPORT):
    """Counts and failure categories computed from the report (numbers never come from an LLM)."""
    g = get_failures(report_path)
    cats = {}
    for f in g["failures"]:
        c = f["diagnosis"]["category"]; cats[c] = cats.get(c, 0) + 1
    return {"total": g["total"], "passed": g["total"] - g["failed_count"],
            "failed": g["failed_count"], "failure_categories": cats,
            "evidence_level": "python_behavioral_simulation (NOT RTL, NOT silicon)"}


def chip_status():
    """BM Labs chip: chain verdicts, chip designs (logic cells, X2 blocks, Fmax vs the project clock), two-block engine,
    real-layer end-to-end results, X2 facts and what is still missing. Reads reports only; runs nothing."""
    from x2c.verification.chip_tools import chip_status as _s
    return _s()


def chip_schedule():
    """BM Labs chip: today's vs the best X2 tile schedule for the real tiny MobileNetV2 (cycles and microseconds)."""
    from x2c.verification.chip_tools import chip_schedule as _s
    return _s()


def run_chip_layers(positions: int = 2):
    """BM Labs chip: 4-bit end-to-end check of every real layer in Python (ASSUMED 4-bit cell rule)."""
    from x2c.verification.chip_tools import run_chip_layers as _r
    return _r(positions=positions)


TOOLS = {"chip_status": chip_status, "chip_schedule": chip_schedule, "run_chip_layers": run_chip_layers,
         "list_workloads": list_workloads, "run_benchmark": run_benchmark, "coverage_gaps": coverage_gaps, "run_generated_tests": run_generated_tests, "run_suite": run_suite, "run_workloads": run_workloads, "run_model_workloads": run_model_workloads, "run_case": run_case,
         "get_failures": get_failures, "summarize_report": summarize_report}

TOOL_SCHEMAS = [
    {"name": "chip_status", "description": "BM Labs chip: what has been verified (chain verdicts, chip builds with logic cells and Fmax vs the project clock, two-block X2 engine, real-layer end-to-end results, X2 facts and what is still missing). Reads reports; runs nothing.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "chip_schedule", "description": "BM Labs chip: today's vs the best X2 tile schedule for the real tiny MobileNetV2, in cycles and microseconds.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "run_chip_layers", "description": "BM Labs chip: run the 4-bit end-to-end check of every real layer in Python (assumed 4-bit cell rule) and return the summary.",
     "input_schema": {"type": "object", "properties": {"positions": {"type": "integer", "minimum": 1, "maximum": 16}}}},
    {"name": "run_suite", "description": "Run the full golden-vs-X2 verification suite and return pass/fail counts.",
     "input_schema": {"type": "object", "properties": {
         "seed": {"type": "integer"}, "n_random": {"type": "integer", "minimum": 0, "maximum": 500}}}},
    {"name": "run_workloads", "description": "Run convolution-layer workloads, tiled onto the PE array and executed on the X2 simulator, against an independent reference.",
     "input_schema": {"type": "object", "properties": {"report_path": {"type": "string"}}}},
    {"name": "run_model_workloads", "description": "Run workloads from real BMsemi models (MNIST edge filter, DeepGate INT8 autoencoder) on the X2 simulator and compare with the repository's own operation output.",
     "input_schema": {"type": "object", "properties": {"report_path": {"type": "string"}}}},
    {"name": "list_workloads", "description": "List the real workloads found in the read-only BMsemi_Model repository (normalised to the local schema) and which pipeline stages exist.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "run_benchmark", "description": "Run real BMsemi workloads through graph validation, mapping consistency and golden-vs-X2 checks. Pass workload_id from list_workloads, or omit for all.",
     "input_schema": {"type": "object", "properties": {"workload_id": {"type": "string"}, "report_path": {"type": "string"}}}},
    {"name": "coverage_gaps", "description": "List input classes (shapes, value patterns, INT8 extremes) not yet covered by the fixed suite plus generated tests, and the spec format for run_generated_tests.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "run_generated_tests", "description": "Run proposed test specs (shape + value pattern + seed). Code builds the matrices and the Python oracle computes expected results; specs cannot contain expected values.",
     "input_schema": {"type": "object", "properties": {"specs": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "m": {"type": "integer"}, "k": {"type": "integer"}, "n": {"type": "integer"},
                          "a_pattern": {"type": "string"}, "b_pattern": {"type": "string"},
                          "lo": {"type": "integer"}, "hi": {"type": "integer"}, "seed": {"type": "integer"}}}},
                                                         "report_path": {"type": "string"}}}},
    {"name": "run_case", "description": "Run one named case, or custom matrices A (MxK) and B (KxN).",
     "input_schema": {"type": "object", "properties": {
         "name": {"type": "string"},
         "A": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
         "B": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}}}},
    {"name": "get_failures", "description": "List failed tests from the latest report with deterministic diagnoses.",
     "input_schema": {"type": "object", "properties": {"report_path": {"type": "string"}}}},
    {"name": "summarize_report", "description": "Counts and failure categories from the latest report.",
     "input_schema": {"type": "object", "properties": {"report_path": {"type": "string"}}}},
]


def _safe_report_path(p):
    """LLM-supplied paths may only point to a .json file inside the reports/ folder."""
    q = Path(p).resolve()
    if REPORTS_DIR not in q.parents or q.suffix != ".json":
        raise ValueError("report_path must be a .json file inside the reports/ folder")
    return str(Path(p))


def call_tool(name: str, arguments: dict):
    """Single entry point an agent loop uses. The LLM is UNTRUSTED input:
    unknown tools, unknown arguments, out-of-range values and paths outside reports/ are rejected."""
    if name not in TOOLS:
        return {"error": f"unknown tool '{name}'", "available": sorted(TOOLS)}
    schema = next(x for x in TOOL_SCHEMAS if x["name"] == name)["input_schema"]["properties"]
    args = dict(arguments or {})
    unknown = set(args) - set(schema)
    if unknown:
        return {"error": f"unknown arguments {sorted(unknown)}; allowed: {sorted(schema)}"}
    try:
        if "n_random" in args:
            args["n_random"] = int(args["n_random"])
            if not 0 <= args["n_random"] <= 500:
                return {"error": "n_random must be between 0 and 500"}
        if "seed" in args:
            args["seed"] = int(args["seed"])
        if "positions" in args:
            args["positions"] = int(args["positions"])
            if not 1 <= args["positions"] <= 16:
                return {"error": "positions must be between 1 and 16"}
        if "report_path" in args:
            args["report_path"] = _safe_report_path(args["report_path"])
        return TOOLS[name](**args)
    except (TypeError, ValueError, FileNotFoundError, ImportError) as e:
        return {"error": f"{type(e).__name__}: {e}"}
