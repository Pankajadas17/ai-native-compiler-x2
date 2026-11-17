"""
Workloads taken from REAL BMsemi_Model operations, executed on the X2 simulator and compared with the
repository's own operation output plus an independent reference.

What is tested: the X2 behavioral simulator + the mapper's tiling (map_layer), on real model data.
What is NOT tested: the application models themselves (they are NumPy "correctness paths" in the repo,
YOLO runs on a TFLite CPU path), RTL, bitstreams or silicon.

  mnist_edge_digit5/9  <- models.mnist.edge_filter : 5x5 SAME conv, 2 crossbar columns (+ and -), out = pos - neg
  deepgate_ad01_case0  <- models.deepgate.infer    : 10 INT8 fully-connected layers; X2 computes each layer's
                          int32 dot product tile-by-tile, then the repo's OWN requantization finishes the layer.

ASSUMPTIONS (confirm): PE geometry 32x32 (per the repo's mapper/README); DeepGate activations enter as
(x - input_zp); accumulation across row folds happens in software; the simulator's ideal arithmetic is exact
for these integer magnitudes.
"""
from pathlib import Path

import numpy as np

from emulators.platform_ops import map_layer
from emulators.x2_adapter import x2_vecmat
from verification.comparator import compare
from verification.workloads import conv_golden, conv_on_x2_tiled

ROOT = Path(__file__).resolve().parents[1] / "BMsemi_Model"
PE = {"pe_rows": 32, "pe_cols": 32}


def data_available():
    return (ROOT / "projects/3_example_models/Deep_gate/schema.json").exists() and \
           (ROOT / "projects/2_building_blocks/Neuromorphic_compiler/mnist5_bin_28x28.txt").exists()


# ----------------------------------------------------------------------------- MNIST
def run_mnist(digit, crop=10, full=False, mapping_transform=None, vecmat=x2_vecmat):
    from BMsemi_Model.bm_models.models.mnist import edge_filter, default_kernels, R, S
    name = f"mnist_edge_digit{digit}" + ("_full" if full else f"_crop{crop}")
    op = edge_filter({"digit": digit})                       # the repo's own operation result
    repo_map = np.asarray(op["map"], dtype=float)            # (28, 28)
    H, W = repo_map.shape
    if full:
        region = ((0, H), (0, W))
    else:                                                     # the crop x crop block with most non-zero outputs
        best, region = -1, None
        for y0 in range(0, H - crop + 1):
            for x0 in range(0, W - crop + 1):
                nz = int(np.count_nonzero(repo_map[y0:y0 + crop, x0:x0 + crop]))
                if nz > best:
                    best, region = nz, ((y0, y0 + crop), (x0, x0 + crop))
    (y0, y1), (x0, x1) = region
    kp, kn = default_kernels()
    img = np.asarray(op_image(digit), dtype=float)
    ifm = np.pad(img, (R // 2, S // 2))[None]                # (1, 32, 32) SAME padding, as the repo does
    w = np.stack([kp, kn])[:, None]                          # (2, 1, 5, 5): column 0 = '+', column 1 = '-'
    mp = map_layer({"K": 2, "C": 1, "R": R, "S": S, "Y": H, "X": W, **PE})
    use = mapping_transform(dict(mp)) if mapping_transform else mp
    base = {"test": name, "kind": "bm_model_workload", "source_model": "models.mnist.edge_filter",
            "region": [[y0, y1], [x0, x1]], "shape": [y1 - y0, x1 - x0],
            "expected": repo_map[y0:y1, x0:x1].tolist(), "mapper_pe_tiles": mp["pe_tiles"]}
    try:
        out2, passes = conv_on_x2_tiled(ifm, w, use, vecmat, region=region)
    except Exception as e:  # noqa: BLE001
        return {**base, "actual": None, "passed": False, "reason": "exception",
                "error": f"{type(e).__name__}: {e}", "max_abs_err": None}
    actual = out2[0] - out2[1]
    golden = (lambda g: g[0] - g[1])(conv_golden(ifm, w))[y0:y1, x0:x1]       # independent nested-loop reference
    cmp = compare(base["expected"], actual)
    exp_passes = mp["pe_tiles"] * (y1 - y0) * (x1 - x0)
    check = {"passes_executed": passes, "expected_passes": exp_passes,
             "repo_reported_pe_tiles": op["pe_tiles_32x32"], "mapper_pe_tiles": mp["pe_tiles"],
             "independent_reference_agrees": bool(np.array_equal(golden, np.asarray(base["expected"]))),
             "ok": passes == exp_passes and op["pe_tiles_32x32"] == mp["pe_tiles"]}
    if cmp["passed"] and not check["ok"]:
        cmp = {"passed": False, "reason": "pass_count_mismatch", "max_abs_err": cmp["max_abs_err"]}
    if cmp["passed"] and not check["independent_reference_agrees"]:
        cmp = {"passed": False, "reason": "references_disagree", "max_abs_err": cmp["max_abs_err"]}
    return {**base, "actual": actual.tolist(), "mapping_check": check, **cmp}


def op_image(digit):
    from BMsemi_Model.bm_models.models.mnist import _digit
    return _digit(digit)


# ----------------------------------------------------------------------------- DeepGate
def run_deepgate(case=0, mapping_transform=None, vecmat=x2_vecmat, stop_on_first_failure=True):
    from BMsemi_Model.bm_models.models import deepgate as dg
    fwd, layers, cases = dg._load()
    ex = cases[case]
    x_repo = np.array(ex[0]["output"], dtype=np.int8)
    x_x2 = x_repo.copy()
    per_layer, first_fail, max_ref_err = [], None, 0.0
    for i, L in enumerate(layers):
        n_in, n_out = L["n_in"], L["n_out"]
        # repo path (reference chain)
        y_repo = fwd(x_repo.astype(np.int32) - L["in_zp"], L["W"], L["bias"], L["mult"], L["shift"], L["out_zp"], np.int8)
        # X2 path: the int32 dot product is computed on the X2 simulator, tile by tile
        xin = x_x2.astype(np.int64) - L["in_zp"]
        mp = map_layer({"K": n_out, "C": n_in, "R": 1, "S": 1, "Y": 1, "X": 1, **PE})
        use = mapping_transform(dict(mp)) if mapping_transform else mp
        try:
            acc, passes = conv_on_x2_tiled(xin.reshape(n_in, 1, 1).astype(float),
                                           L["W"].astype(float).reshape(n_out, n_in, 1, 1), use, vecmat)
        except Exception as e:  # noqa: BLE001
            return {"test": f"deepgate_ad01_case{case}", "kind": "bm_model_workload", "source_model": "models.deepgate.infer",
                    "expected": None, "actual": None, "passed": False, "reason": "exception",
                    "error": f"layer {i}: {type(e).__name__}: {e}", "max_abs_err": None, "first_failing_layer": i}
        acc_x2 = np.rint(acc.reshape(n_out)).astype(np.int64)
        acc_np = L["W"].astype(np.int64) @ xin                                   # independent integer reference
        # finish the layer with the repo's OWN requantization: bias + I @ acc, multiplier, shift, zero-point, clamp
        y_x2 = fwd(acc_x2, np.eye(n_out, dtype=np.int8), L["bias"], L["mult"], L["shift"], L["out_zp"], np.int8)
        yf = (y_x2.astype(np.float32) - L["out_zp"]) * L["out_scale"]
        ref = np.array(ex[i + 1]["output"], dtype=np.float32)
        ref_err = float(np.abs(yf - ref).max()); max_ref_err = max(max_ref_err, ref_err)
        entry = {"layer": i, "n_in": n_in, "n_out": n_out, "passes": passes, "mapper_mvm_ops": map_layer(
                     {"K": n_out, "C": n_in, "R": 1, "S": 1, "Y": 1, "X": 1, **PE})["mvm_ops"],
                 "accumulators_exact": bool(np.array_equal(acc_x2, acc_np)),
                 "output_matches_repo": bool(np.array_equal(y_x2, y_repo)), "ref_err_vs_test_case": ref_err}
        entry["ok"] = entry["accumulators_exact"] and entry["output_matches_repo"] and \
            entry["passes"] == entry["mapper_mvm_ops"] and ref_err < 0.02
        per_layer.append(entry)
        if not entry["ok"] and first_fail is None:
            first_fail = i
            if stop_on_first_failure:      # later layers only carry the error forward; report the culprit layer
                return {"test": f"deepgate_ad01_case{case}", "kind": "bm_model_workload",
                        "source_model": "models.deepgate.infer", "shape": [1, n_out],
                        "expected": y_repo.astype(float).reshape(1, n_out).tolist(),
                        "actual": y_x2.astype(float).reshape(1, n_out).tolist(),
                        "layers": per_layer, "first_failing_layer": i, "stopped_early": True,
                        "total_x2_passes": sum(e["passes"] for e in per_layer),
                        "passed": False, "reason": "layer_check_failed",
                        "max_abs_err": float(np.max(np.abs(y_repo.astype(float) - y_x2.astype(float))))}
        x_repo, x_x2 = y_repo, y_x2
    exp = x_repo.astype(float).reshape(20, 32); act = x_x2.astype(float).reshape(20, 32)
    cmp = compare(exp, act)
    if cmp["passed"] and first_fail is not None:
        cmp = {"passed": False, "reason": "layer_check_failed", "max_abs_err": cmp["max_abs_err"]}
    return {"test": f"deepgate_ad01_case{case}", "kind": "bm_model_workload", "source_model": "models.deepgate.infer",
            "shape": [20, 32], "expected": exp.tolist(), "actual": act.tolist(),
            "layers": per_layer, "first_failing_layer": first_fail,
            "total_x2_passes": sum(e["passes"] for e in per_layer),
            "reference_max_abs_err": max_ref_err, **cmp}


def run_model_workload_suite(mapping_transform=None, vecmat=x2_vecmat, full=False):
    return [run_mnist(5, full=full, mapping_transform=mapping_transform, vecmat=vecmat),
            run_mnist(9, full=full, mapping_transform=mapping_transform, vecmat=vecmat),
            run_deepgate(0, mapping_transform=mapping_transform, vecmat=vecmat)]
