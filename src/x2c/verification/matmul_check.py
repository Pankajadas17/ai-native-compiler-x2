"""The first experiment (2x2 matmul on the BMsemi X2 simulator): golden result, comparison, test cases, report.

Used by x2c bmsemi verify, the A4 verification agent and the BMsemi workloads.
"""
import json
import time
from pathlib import Path

import numpy as np


def golden_matmul(A, B):
    """Float reference A @ B."""
    return np.asarray(A, dtype=float) @ np.asarray(B, dtype=float)


def compare(expected, actual, atol=0.0):
    e, a = np.asarray(expected, float), np.asarray(actual, float)
    if e.shape != a.shape:
        return {"passed": False, "reason": f"shape {e.shape} != {a.shape}", "max_abs_err": None}
    err = float(np.max(np.abs(e - a))) if e.size else 0.0
    return {"passed": bool(err <= atol), "reason": "ok" if err <= atol else "mismatch", "max_abs_err": err}


def make_cases(seed=0, n_random=20):
    rng = np.random.default_rng(seed)
    cases = [("fixed_2x2", np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])),
             ("zeros", np.zeros((3, 3)), np.ones((3, 3))),
             ("identity", np.eye(4), np.arange(16).reshape(4, 4)),
             ("negatives", np.array([[-1, 2], [3, -4]]), np.array([[5, -6], [-7, 8]])),
             ("non_square", np.arange(6).reshape(2, 3), np.arange(12).reshape(3, 4))]
    for i in range(n_random):
        m, k, n = rng.integers(1, 9, 3)
        cases.append((f"random_{i}", rng.integers(-8, 9, (m, k)), rng.integers(-8, 9, (k, n))))
    return cases


def write_report(results, path="reports/report.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    summary = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "level": "python_behavioral_simulation (NOT RTL, NOT silicon)",
               "total": len(results), "passed": sum(r["passed"] for r in results),
               "results": results}
    Path(path).write_text(json.dumps(summary, indent=2))
    return summary
