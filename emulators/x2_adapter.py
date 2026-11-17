"""
X2 adapter - the ONLY file in bm_verif that talks to the BMsemi X2 simulator.

It calls the ORIGINAL simulator inside the (read-only) BMsemi_Model clone.
There is deliberately NO fallback: if the real simulator cannot be imported,
we crash loudly instead of silently using a NumPy stand-in.

Level of verification: Python behavioral simulation (NOT RTL, NOT silicon).
"""
import numpy as np

try:
    from BMsemi_Model.bm_models.core.pe import simulate
except ModuleNotFoundError as e:  # no silent fallback on purpose
    raise ModuleNotFoundError(
        "Cannot import the real BMsemi simulator. Run this from the bm_verif folder "
        "and make sure the BMsemi_Model folder is inside it (see README)."
    ) from e


def x2_vecmat_full(weights, vec):
    """One 'vector x matrix' pass. weights (K,N), vec (K,). Returns the full simulator dict."""
    W = np.asarray(weights, dtype=float)
    v = np.asarray(vec, dtype=float)
    if W.ndim != 2 or v.ndim != 1:
        raise ValueError("weights must be 2D and vec 1D")
    if W.shape[0] != v.shape[0]:
        raise ValueError(f"dimension mismatch: weights {W.shape}, vec {v.shape}")
    r = simulate({"rows": W.shape[0], "cols": W.shape[1],
                  "weights": W.tolist(), "input": v.tolist()})
    if r.get("backend") != "behavioral_simulator" or r.get("physical_hardware") is not False:
        raise RuntimeError(f"Unexpected backend, not the original simulator: {r.get('backend')}")
    return r


def x2_vecmat(weights, vec):
    r = x2_vecmat_full(weights, vec)
    return np.asarray(r["output"], dtype=float), r["total_cycles"]


def x2_matmul(A, B):
    """A (M,K) @ B (K,N): one simulator pass per row of A."""
    A = np.asarray(A, dtype=float); B = np.asarray(B, dtype=float)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[0]:
        raise ValueError(f"incompatible shapes {A.shape} @ {B.shape}")
    outs, cycles = [], 0
    for row in A:
        o, c = x2_vecmat(B, row)
        outs.append(o); cycles += c
    return np.asarray(outs, dtype=float), cycles


def x2_matmul_report(A, B):
    """Like x2_matmul but keeps simulator metadata for the verification report."""
    A = np.asarray(A, dtype=float); B = np.asarray(B, dtype=float)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[0]:
        raise ValueError(f"incompatible shapes {A.shape} @ {B.shape}")
    rows = [x2_vecmat_full(B, r) for r in A]
    return {
        "output": np.asarray([r["output"] for r in rows], dtype=float),
        "total_cycles": int(sum(r["total_cycles"] for r in rows)),
        "energy_configured_subtotal_J": float(sum(r["energy"]["configured_subtotal_J"] for r in rows)),
        "energy_status": rows[-1]["energy"]["status"],  # unset costs are None, never 0
        "final_status": rows[-1]["status"],
        "backend": rows[-1]["backend"],
        "physical_hardware": rows[-1]["physical_hardware"],
        "source_commit": rows[-1]["source_commit"],
    }
