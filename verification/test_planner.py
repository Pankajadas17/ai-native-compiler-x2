"""
A4 test-generation core. An LLM (or the deterministic baseline planner) proposes TEST SPECS; code validates them,
materialises the matrices, and the Python oracle computes the expected result. A spec has NO field for an expected
value, so a model can never supply, bias or "round" the answer key. Plans are generated from the spec/coverage ONLY
(never from the RTL or the backend under test) and are replayed unchanged against any backend.

Coverage is measured on the MATERIALISED matrices, not on what a spec claims to be.
"""
import numpy as np

PATTERNS = ["random", "zeros", "ones", "identity_like", "sparse", "max_magnitude", "alternating_sign", "single_nonzero"]
ALLOWED_KEYS = {"name", "m", "k", "n", "a_pattern", "b_pattern", "lo", "hi", "seed"}
MAX_DIM, MAX_SPECS = 12, 40
INT8_MIN, INT8_MAX = -128, 127

SPEC_HELP = {"name": "short id: letters, digits, underscore (<= 40 chars)", "m": f"rows of A, 1..{MAX_DIM}",
             "k": f"inner dimension, 1..{MAX_DIM}", "n": f"columns of B, 1..{MAX_DIM}",
             "a_pattern": PATTERNS, "b_pattern": PATTERNS, "lo": f"min value, >= {INT8_MIN}",
             "hi": f"max value, <= {INT8_MAX}", "seed": "integer"}

SHAPE_CLASSES = ["single_row_A", "single_col_out", "square", "non_square", "long_k", "tall_m"]
VALUE_CLASSES = ["all_zero_operand", "all_ones_operand", "identity_like", "sparse_operand", "extreme_values",
                 "int8_min", "alternating_sign", "single_nonzero"]
ALL_CLASSES = SHAPE_CLASSES + VALUE_CLASSES


def validate_spec(raw):
    """Return (clean_spec, None) or (None, reason). Unknown keys (e.g. 'expected') are rejected."""
    if not isinstance(raw, dict):
        return None, "spec must be an object"
    extra = set(raw) - ALLOWED_KEYS
    if extra:
        return None, f"unknown keys {sorted(extra)} (specs cannot contain expected values)"
    try:
        s = {"name": str(raw["name"]), "m": int(raw["m"]), "k": int(raw["k"]), "n": int(raw["n"]),
             "a_pattern": raw.get("a_pattern", "random"), "b_pattern": raw.get("b_pattern", "random"),
             "lo": int(raw.get("lo", INT8_MIN)), "hi": int(raw.get("hi", INT8_MAX)), "seed": int(raw.get("seed", 0))}
    except (KeyError, TypeError, ValueError) as e:
        return None, f"bad or missing field: {e}"
    if not s["name"] or len(s["name"]) > 40 or not s["name"].replace("_", "").isalnum():
        return None, "name must be 1-40 chars: letters, digits, underscore"
    for d in ("m", "k", "n"):
        if not 1 <= s[d] <= MAX_DIM:
            return None, f"{d} must be in 1..{MAX_DIM}"
    for p in ("a_pattern", "b_pattern"):
        if s[p] not in PATTERNS:
            return None, f"{p} must be one of {PATTERNS}"
    if not (INT8_MIN <= s["lo"] <= s["hi"] <= INT8_MAX):
        return None, f"need {INT8_MIN} <= lo <= hi <= {INT8_MAX}"
    return s, None


def _make(pattern, rows, cols, lo, hi, rng):
    if pattern == "zeros":
        return np.zeros((rows, cols))
    if pattern == "ones":
        return np.ones((rows, cols))
    if pattern == "identity_like":
        return np.eye(rows, cols)
    if pattern == "single_nonzero":
        x = np.zeros((rows, cols)); x[rng.integers(rows), rng.integers(cols)] = hi if hi != 0 else 1; return x
    if pattern == "alternating_sign":
        sign = np.where((np.add.outer(np.arange(rows), np.arange(cols)) % 2) == 0, 1, -1)
        return sign * float(max(abs(lo), abs(hi), 1) if hi >= -lo else max(abs(hi), 1))
    if pattern == "max_magnitude":
        x = rng.choice([lo, hi], size=(rows, cols)).astype(float); x[0, 0] = lo; return x   # always includes the minimum
    if pattern == "sparse":
        x = rng.integers(lo, hi + 1, (rows, cols)).astype(float); x[rng.random((rows, cols)) < 0.8] = 0; return x
    return rng.integers(lo, hi + 1, (rows, cols)).astype(float)                              # random


def materialize(spec):
    rng = np.random.default_rng(spec["seed"])
    A = _make(spec["a_pattern"], spec["m"], spec["k"], spec["lo"], spec["hi"], rng)
    B = _make(spec["b_pattern"], spec["k"], spec["n"], spec["lo"], spec["hi"], rng)
    return A, B


def classify(A, B):
    """Coverage classes actually present in this (A, B) pair."""
    m, k = A.shape; n = B.shape[1]
    c = set()
    if m == 1: c.add("single_row_A")
    if n == 1: c.add("single_col_out")
    c.add("square" if m == k == n else "non_square")
    if k > 8: c.add("long_k")
    if m > 8: c.add("tall_m")
    for X in (A, B):
        nz = np.count_nonzero(X)
        if nz == 0: c.add("all_zero_operand")
        if np.all(X == 1): c.add("all_ones_operand")
        if X.shape[0] > 1 and X.shape[1] > 1 and np.array_equal(X, np.eye(*X.shape)): c.add("identity_like")
        if 0 < nz and (X == 0).mean() >= 0.6: c.add("sparse_operand")
        if np.abs(X).max() >= 127: c.add("extreme_values")
        if (X == -128).any(): c.add("int8_min")
        if nz == 1: c.add("single_nonzero")
        vals = np.abs(X[X != 0])
        if vals.size > 1 and np.all(vals == vals[0]) and (X > 0).any() and (X < 0).any(): c.add("alternating_sign")
    return c


def coverage_of(pairs):
    seen = set()
    for A, B in pairs:
        seen |= classify(np.asarray(A, float), np.asarray(B, float))
    return seen


def fixed_suite_pairs():
    from verification.test_generator import make_cases
    return [(A, B) for _, A, B in make_cases()]


def coverage_gaps(extra_specs=()):
    pairs = fixed_suite_pairs() + [materialize(s) for s in extra_specs]
    seen = coverage_of(pairs)
    return {"covered": sorted(seen), "missing": [c for c in ALL_CLASSES if c not in seen],
            "fixed_suite_tests": len(fixed_suite_pairs()), "extra_tests_counted": len(list(extra_specs))}


def baseline_plan():
    """Deterministic planner (no LLM): one spec per class the fixed suite does not cover + two combined stress cases."""
    miss = set(coverage_gaps()["missing"])
    plan, i = [], 0
    def add(**kw):
        nonlocal i; i += 1
        plan.append({"name": f"gen_{i:02d}_{kw.pop('tag')}", "seed": 100 + i, **kw})
    if "single_row_A" in miss: add(tag="single_row", m=1, k=4, n=4, a_pattern="random", b_pattern="random")
    if "single_col_out" in miss: add(tag="single_col", m=4, k=4, n=1, a_pattern="random", b_pattern="random")
    if "long_k" in miss: add(tag="long_k", m=2, k=12, n=3, a_pattern="random", b_pattern="random")
    if "tall_m" in miss: add(tag="tall_m", m=10, k=3, n=3, a_pattern="random", b_pattern="random")
    if "all_ones_operand" in miss: add(tag="ones", m=3, k=3, n=3, a_pattern="ones", b_pattern="ones")
    if "identity_like" in miss: add(tag="identity", m=4, k=4, n=4, a_pattern="identity_like", b_pattern="random")
    if "sparse_operand" in miss: add(tag="sparse", m=5, k=5, n=5, a_pattern="sparse", b_pattern="sparse")
    if "single_nonzero" in miss: add(tag="single_nz", m=4, k=4, n=4, a_pattern="single_nonzero", b_pattern="random")
    if "alternating_sign" in miss: add(tag="alt_sign", m=4, k=4, n=4, a_pattern="alternating_sign", b_pattern="alternating_sign")
    if {"extreme_values", "int8_min"} & miss: add(tag="extremes", m=3, k=3, n=3, a_pattern="max_magnitude", b_pattern="max_magnitude")
    add(tag="extremes_long_k", m=3, k=12, n=3, a_pattern="max_magnitude", b_pattern="max_magnitude")
    add(tag="extremes_tall", m=10, k=4, n=2, a_pattern="max_magnitude", b_pattern="max_magnitude")
    return plan


def validate_plan(raw_specs):
    """Validate a list of raw specs. Returns (accepted_specs, rejected [{index, reason}])."""
    if not isinstance(raw_specs, list):
        return [], [{"index": None, "reason": "specs must be a list"}]
    acc, rej, names = [], [], set()
    for i, raw in enumerate(raw_specs[:MAX_SPECS]):
        s, err = validate_spec(raw)
        if err is None and s["name"] in names:
            err = "duplicate name"
        if err:
            rej.append({"index": i, "reason": err})
        else:
            names.add(s["name"]); acc.append(s)
    if len(raw_specs) > MAX_SPECS:
        rej.append({"index": None, "reason": f"only the first {MAX_SPECS} specs are used"})
    return acc, rej
