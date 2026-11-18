"""
Workload-level verification: a convolution layer is executed the way the compiler would map it
(tiled onto PE-sized blocks, each block = one X2 pass) and compared with an INDEPENDENT direct convolution.

ASSUMPTIONS (confirm with the compiler team):
  * valid convolution, stride 1; Y, X are OUTPUT height/width, so the input is (C, Y+R-1, X+S-1)
  * patch (im2col) order is channel-major: (c, r, s) flattened
  * partial sums from different row folds are added OUTSIDE the PE, in software
Existing functionality used: map_layer (tiling rule) and the real X2 simulator via x2_adapter.
New here: the executor, the independent golden convolution and the pass-count cross-check.
"""
import numpy as np

from x2c.bmsemi.platform_ops import map_layer
from x2c.bmsemi.x2_adapter import x2_vecmat
from x2c.verification.matmul_check import compare

MAX_PASSES = 3000   # keep every workload quick
_ACTIVE = {"mapping_transform": None}


def set_mapping_transform(fn=None):
    """Plug in a deliberately wrong tiling plan (fault injection / demos). NOT exposed to any LLM."""
    _ACTIVE["mapping_transform"] = fn



def conv_golden(ifm, w):
    """Independent reference: plain nested loops, no im2col, no tiling. ifm (C,H,W), w (K,C,R,S) -> (K,Y,X)."""
    C, H, W = ifm.shape
    K, C2, R, S = w.shape
    assert C == C2
    Y, X = H - R + 1, W - S + 1
    out = np.zeros((K, Y, X))
    for k in range(K):
        for y in range(Y):
            for x in range(X):
                acc = 0.0
                for c in range(C):
                    for r in range(R):
                        for s in range(S):
                            acc += ifm[c, y + r, x + s] * w[k, c, r, s]
                out[k, y, x] = acc
    return out


def conv_on_x2_tiled(ifm, w, mapping, vecmat=x2_vecmat, region=None):
    """Execute the layer tile by tile on the X2 simulator following `mapping` (a map_layer result dict).
    region=((y0,y1),(x0,x1)) restricts which OUTPUT positions are computed (default: all).
    Returns (output (K, y1-y0, x1-x0), number of X2 passes executed)."""
    C, H, W = ifm.shape
    K, _, R, S = w.shape
    Y, X = H - R + 1, W - S + 1
    patch_len = C * R * S
    W2 = w.reshape(K, patch_len).T                       # (patch_len, K) - the weight matrix the PE holds
    folds, start = [], 0
    for size in mapping["fold_sizes"]:
        folds.append((start, min(start + size, patch_len))); start += size
    (y0, y1), (x0, x1) = region if region else ((0, Y), (0, X))
    out = np.zeros((K, y1 - y0, x1 - x0))
    passes = 0
    for y in range(y0, y1):
        for x in range(x0, x1):
            patch = ifm[:, y:y + R, x:x + S].reshape(-1)
            for (r0, r1) in folds:
                if r0 >= r1:
                    continue
                for (k0, k1) in mapping["k_chunks"]:
                    if k0 >= k1:
                        continue
                    part, _cyc = vecmat(W2[r0:r1, k0:k1], patch[r0:r1])
                    out[k0:k1, y - y0, x - x0] += part             # software accumulation across folds
                    passes += 1
    return out, passes


# name -> (K, C, R, S, Y, X, pe_rows, pe_cols)
WORKLOADS = {
    "conv3x3_multi_fold":   (8, 2, 3, 3, 4, 4, 8, 4),     # 3 row folds x 2 chunks
    "pointwise_1x1":        (16, 24, 1, 1, 3, 3, 16, 8),
    "exact_fit_single_tile": (4, 2, 2, 2, 2, 2, 8, 4),
    "ragged_edges":         (5, 3, 3, 3, 3, 3, 10, 4),    # last fold and last chunk are partial
    "canvas_64x32_geometry": (64, 32, 3, 3, 2, 2, 64, 32),  # the canvas example's tiling, shrunken output map
}


def run_workload(name, mapping_transform=None, seed=0, vecmat=x2_vecmat):
    K, C, R, S, Y, X, pr, pc = WORKLOADS[name]
    rng = np.random.default_rng(seed)
    ifm = rng.integers(-3, 4, (C, Y + R - 1, X + S - 1)).astype(float)
    w = rng.integers(-3, 4, (K, C, R, S)).astype(float)
    expected = conv_golden(ifm, w).reshape(K, Y * X)
    mapping = map_layer({"K": K, "C": C, "R": R, "S": S, "Y": Y, "X": X, "pe_rows": pr, "pe_cols": pc})
    base = {"test": name, "kind": "conv_workload", "dims": dict(K=K, C=C, R=R, S=S, Y=Y, X=X, pe_rows=pr, pe_cols=pc),
            "shape": [K, Y * X], "expected": expected.tolist(), "mapper_pe_tiles": mapping["pe_tiles"]}
    if mapping["mvm_ops"] > MAX_PASSES:
        return {**base, "actual": None, "passed": False, "reason": "exception", "error": "workload too large", "max_abs_err": None}
    mapping_transform = mapping_transform or _ACTIVE["mapping_transform"]
    if mapping_transform:                                   # fault injection hook (a deliberately wrong mapping)
        mapping = mapping_transform(dict(mapping))
    try:
        actual, passes = conv_on_x2_tiled(ifm, w, mapping, vecmat)
    except Exception as e:  # noqa: BLE001
        return {**base, "actual": None, "passed": False, "reason": "exception",
                "error": f"{type(e).__name__}: {e}", "max_abs_err": None}
    actual = actual.reshape(K, Y * X)
    cmp = compare(expected, actual)
    expected_passes = map_layer(dict(base["dims"]))["mvm_ops"]       # from the UNMODIFIED mapper
    check = {"passes_executed": passes, "mvm_ops_from_mapper": expected_passes, "ok": passes == expected_passes}
    if cmp["passed"] and not check["ok"]:
        cmp = {"passed": False, "reason": "pass_count_mismatch", "max_abs_err": cmp["max_abs_err"]}
    return {**base, "actual": actual.tolist(), "mapping_check": check, **cmp}


def run_workload_suite(mapping_transform=None):
    return [run_workload(n, mapping_transform) for n in WORKLOADS]
