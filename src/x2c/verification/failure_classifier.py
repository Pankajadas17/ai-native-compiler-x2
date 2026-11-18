"""Deterministic failure classifier = the seed of the future debug agent (A3).
Pure numeric pattern matching; no LLM. Returns a category plus the evidence for it."""
import numpy as np


def classify_failure(result):
    det = _classify(result)
    fl = result.get("first_failing_layer")
    if fl is not None and det["category"] != "none":
        det = dict(det, evidence=det["evidence"] + f" | first failing layer: {fl}")
    return det


def _classify(result):
    if result.get("passed"):
        return {"category": "none", "evidence": "test passed"}
    if result.get("reason") == "exception":
        return {"category": "interface_error", "evidence": result.get("error", "")}
    if result.get("reason") == "pass_count_mismatch":
        return {"category": "mapping_inconsistency",
                "evidence": f"numbers match but X2 passes executed != mapper mvm_ops: {result.get('mapping_check')}"}
    e = np.asarray(result["expected"], dtype=float)
    a = np.asarray(result["actual"], dtype=float)
    if e.shape != a.shape:
        hint = " (actual.T has the expected shape: likely transposed output)" if a.T.shape == e.shape else ""
        return {"category": "shape_mismatch", "evidence": f"expected {e.shape}, got {a.shape}{hint}"}
    if e.shape[0] == e.shape[1] and np.allclose(a, e.T):
        return {"category": "transposed_output", "evidence": "actual equals expected transposed"}
    if np.any(e != 0) and np.allclose(a, -e):
        return {"category": "sign_inverted", "evidence": "actual equals -expected"}
    if np.any(e != 0) and np.all(a == 0):
        return {"category": "zero_output", "evidence": "actual is all zeros while expected is not (nothing was computed, or every contribution was dropped)"}
    mask = e != 0
    if mask.any():
        ratio = a[mask] / e[mask]
        if np.allclose(ratio, ratio[0]) and not np.isclose(ratio[0], 1.0) and np.allclose(a[~mask], 0):
            return {"category": "scaled_output", "evidence": f"actual = {ratio[0]:.4g} x expected"}
    diff = a - e
    if np.allclose(diff, diff.flat[0]):
        return {"category": "constant_offset", "evidence": f"actual = expected + {diff.flat[0]:.4g}"}
    if e.ndim == 2 and e.shape[0] > 1:
        bad = [i for i in range(e.shape[0]) if not np.allclose(a[i], e[i])]
        if 0 < len(bad) < e.shape[0]:
            return {"category": "localized_mismatch",
                    "evidence": f"only output rows/channels {bad} differ (of {e.shape[0]}); check the tile/chunk covering them"}
    return {"category": "numerical_mismatch", "evidence": f"max abs error {np.max(np.abs(diff)):.4g}"}
