import numpy as np
def compare(expected, actual, atol=0.0):
    e, a = np.asarray(expected, float), np.asarray(actual, float)
    if e.shape != a.shape:
        return {"passed": False, "reason": f"shape {e.shape} != {a.shape}", "max_abs_err": None}
    err = float(np.max(np.abs(e - a))) if e.size else 0.0
    return {"passed": bool(err <= atol), "reason": "ok" if err <= atol else "mismatch", "max_abs_err": err}
