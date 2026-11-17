"""
Generalised layer execution for real-geometry workloads: stride, zero padding and groups (depthwise).
Two independent paths:
  grouped_conv_golden  : plain nested loops (the oracle; no tiling, no im2col)
  grouped_conv_on_x2   : each group is tiled with the repo mapper (map_layer) and executed block by block on the
                         X2 behavioural simulator; partial sums over row folds are added in software.
ASSUMPTIONS: zero padding; groups executed one after another (our own execution plan; the compiler's packing
of groups onto PE columns is NOT modelled); input size chosen so the OUTPUT is exactly (Y, X).
"""
import numpy as np

from emulators.platform_ops import map_layer
from emulators.x2_adapter import x2_vecmat


def input_hw(layer):
    """Smallest input height/width that yields the layer's Y x X output with its stride/padding."""
    h = (layer.Y - 1) * layer.stride + layer.R - 2 * layer.padding
    w = (layer.X - 1) * layer.stride + layer.S - 2 * layer.padding
    if h < 1 or w < 1:
        raise ValueError(f"layer {layer.name}: geometry gives a non-positive input size ({h}x{w})")
    return h, w


def grouped_conv_golden(ifm, w, stride, padding, groups):
    """ifm (C,H,W), w (K, C/groups, R, S) -> (K, Y, X). Zero padding."""
    C, H, W = ifm.shape
    K, Cg, R, S = w.shape
    assert C == Cg * groups and K % groups == 0
    p = np.pad(ifm, ((0, 0), (padding, padding), (padding, padding)))
    Y = (H + 2 * padding - R) // stride + 1
    X = (W + 2 * padding - S) // stride + 1
    Kg = K // groups
    out = np.zeros((K, Y, X))
    for k in range(K):
        g = k // Kg
        for y in range(Y):
            for x in range(X):
                acc = 0.0
                for c in range(Cg):
                    for r in range(R):
                        for s in range(S):
                            acc += p[g * Cg + c, y * stride + r, x * stride + s] * w[k, c, r, s]
                out[k, y, x] = acc
    return out


def grouped_conv_on_x2(ifm, w, stride, padding, groups, pe_rows, pe_cols, vecmat=x2_vecmat, mapping_transform=None):
    """Returns (output (K,Y,X), X2 passes executed, X2 passes the mapper says are needed)."""
    C, H, W = ifm.shape
    K, Cg, R, S = w.shape
    Kg = K // groups
    p = np.pad(ifm, ((0, 0), (padding, padding), (padding, padding)))
    Y = (H + 2 * padding - R) // stride + 1
    X = (W + 2 * padding - S) // stride + 1
    patch_len = Cg * R * S
    out = np.zeros((K, Y, X))
    passes = expected = 0
    for g in range(groups):
        mp = map_layer({"K": Kg, "C": Cg, "R": R, "S": S, "Y": Y, "X": X, "pe_rows": pe_rows, "pe_cols": pe_cols})
        expected += mp["mvm_ops"]
        use = mapping_transform(dict(mp)) if mapping_transform else mp
        W2 = w[g * Kg:(g + 1) * Kg].reshape(Kg, patch_len).T            # (patch_len, Kg): what the PE holds
        folds, start = [], 0
        for size in use["fold_sizes"]:
            folds.append((start, min(start + size, patch_len))); start += size
        for y in range(Y):
            for x in range(X):
                patch = p[g * Cg:(g + 1) * Cg, y * stride:y * stride + R, x * stride:x * stride + S].reshape(-1)
                for (r0, r1) in folds:
                    if r0 >= r1:
                        continue
                    for (k0, k1) in use["k_chunks"]:
                        if k0 >= k1:
                            continue
                        part, _ = vecmat(W2[r0:r1, k0:k1], patch[r0:r1])
                        out[g * Kg + k0:g * Kg + k1, y, x] += part
                        passes += 1
    return out, passes, expected
