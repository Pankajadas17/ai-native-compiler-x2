"""
A whole network as the chip runs it: INT8 activations, matrix layers on the X2, the rest as an "epilogue" and host steps.

1. Module kinds (infer_kinds): the compiler artifact does not record what its 27 'shape_preserving' modules are. The
   structure decides: one directly after a matrix layer is BatchNorm, one directly after a BatchNorm is ReLU6
   (torchvision MobileNetV2: Conv -> BN -> ReLU6; the real graph gives exactly 16 + 11). Recorded as INFERRED.
2. INT8 lowering (build_mobilenet / build_chain):
     conv/linear + folded BN  ->  X2 matmul, then epilogue  y = clamp((acc + bias + 2^(s-1)) >> s, lo, hi)
     ReLU6                    ->  the epilogue's clamp to [0, 127]  (activation scale chosen so 6.0 <-> 127)
     residual add             ->  saturating INT8 add (same scale on both inputs; the add8_sat operator's rule)
     global average pool      ->  floor((sum + n/2) / n) over H x W
   The X2 only takes unsigned 8-bit inputs: a signed tensor enters as x + 128 and the layer's bias subtracts
   128 * sum(W) back (exact integer arithmetic; padding uses the same zero point).
   Weights: the compiler artifact has shapes only, so MobileNetV2 weights / folded-BN biases are random INT8 / INT32,
   seeded by layer name; the shift s of each layer is calibrated on the input (s = smallest shift that fits INT8).
3. golden(): an independent nested-loop integer model (no im2col, no packing) of the same network.
4. run(): the chip path: im2col -> X2 matmul (any executor: the 4-bit Python model, or the 1-bit engine in Icarus)
   -> epilogue -> host steps; every activation is compared with golden().
"""
import zlib
from dataclasses import dataclass, field

import numpy as np

from compiler.ir import MATRIX_OPS

LO, HI = -128, 127


# ---------------------------------------------------------------- 1. module kinds
def infer_kinds(g):
    """{node name: 'batchnorm' | 'relu6'} for the shape_preserving nodes, from the graph structure."""
    by, kinds = g.by_name(), {}
    for n in g.nodes:
        if n.op != "shape_preserving":
            continue
        src = by[n.inputs[0]]
        if src.op in MATRIX_OPS:
            kinds[n.name] = "batchnorm"
        elif src.op == "shape_preserving" and kinds.get(src.name) == "batchnorm":
            kinds[n.name] = "relu6"
        else:
            raise ValueError(f"{n.name}: cannot infer its kind from the structure (after {src.op} {src.name})")
    return kinds


# ---------------------------------------------------------------- 2. the INT8 network
@dataclass
class Layer:
    name: str
    attrs: dict
    W: np.ndarray                        # (K, C/groups, R, S) int8
    bias: np.ndarray                     # (K,) int32 (folded BN; before the zero-point correction)
    src: str                             # tensor this layer reads
    in_zp: int                           # zero point added so the X2 sees uint8: 128 for signed tensors, 0 for >= 0
    shift: int = 0
    lo: int = LO
    hi: int = HI
    requant: object = None               # optional callable acc(K,) -> int8 (DeepGate: the repo's own fwd)
    model_zp: int = 0                    # the MODEL's input zero point: the layer computes W . (x - model_zp)
    out: str = ""
    requant_spec: dict = None            # the same requantization as data, so a deployment bundle can replay it


@dataclass
class Net:
    name: str
    input_shape: tuple                   # (C, H, W) or (C,)
    input_signed: bool
    steps: list = field(default_factory=list)      # Layer | ("add", out, a, b) | ("pool", out, a) | ("diff", out, a)
    output: str = ""
    notes: list = field(default_factory=list)


def _rng(name, salt):
    return np.random.default_rng(zlib.crc32(f"{salt}:{name}".encode()))


def build_mobilenet(g):
    """INT8 lowering of the compiler-V2 graph. Random seeded weights (the artifact has none)."""
    kinds, by = infer_kinds(g), g.by_name()
    alias, signed = {}, {}                                 # IR node -> tensor name it produces; tensor -> may be negative
    x = next(n for n in g.nodes if n.op == "input")
    net = Net(g.name, tuple(x.out.shape[1:]), False,
              notes=["module kinds INFERRED from structure (BatchNorm after a matrix op, ReLU6 after BatchNorm)",
                     "weights and folded-BN biases are random INT8/INT32 seeded by layer name (the artifact has none)",
                     "ReLU6 -> clamp [0, 127]; residual add -> saturating INT8 add (same scale); global average pool",
                     "requant: shift only, calibrated per layer on this input (no trained scales exist)"])
    alias[x.name], signed["input"] = "input", False
    for n in g.nodes:
        if n.op in MATRIX_OPS:
            a = {k: n.attrs[k] for k in ("K", "C", "R", "S", "groups", "stride", "padding")}
            a["op"] = n.op
            r = _rng(n.name, "w")
            W = r.integers(-128, 128, size=(a["K"], a["C"] // a["groups"], a["R"], a["S"]))
            bias = _rng(n.name, "b").integers(-2048, 2048, size=a["K"])
            src = alias[n.inputs[0]]
            L = Layer(n.name, a, W, bias, src, 128 if signed[src] else 0, out=n.name)
            net.steps.append(L)
            alias[n.name], signed[n.name] = n.name, True
        elif n.op == "shape_preserving":
            k = kinds[n.name]
            alias[n.name] = alias[n.inputs[0]]
            if k == "relu6":                                       # fold the clamp into the producing layer's epilogue
                L = next(s for s in reversed(net.steps) if isinstance(s, Layer) and s.out == alias[n.inputs[0]])
                L.lo, L.hi = 0, HI
                signed[alias[n.name]] = False
        elif n.op == "add":
            a, b = (alias[i] for i in n.inputs)
            net.steps.append(("add", n.name, a, b))
            alias[n.name], signed[n.name] = n.name, signed[a] or signed[b]
        elif n.op == "pool":
            net.steps.append(("pool", n.name, alias[n.inputs[0]]))
            alias[n.name], signed[n.name] = n.name, signed[alias[n.inputs[0]]]
        elif n.op in ("flatten", "output"):
            alias[n.name] = alias[n.inputs[0]]
            signed.setdefault(alias[n.name], signed[alias[n.inputs[0]]])
    net.output = alias[next(n for n in g.nodes if n.op == "output").name]
    return net


def build_chain(name, layers, input_shape, input_signed=True, notes=()):
    """A chain of linear layers given as dicts {W (K, C), bias, requant(acc)->int8, in_zp} (DeepGate with the repo's data)."""
    net, src = Net(name, input_shape, input_signed, notes=list(notes)), "input"
    for i, d in enumerate(layers):
        W = np.asarray(d["W"], dtype=np.int64)
        K, C = W.shape
        a = {"K": K, "C": C, "R": 1, "S": 1, "groups": 1, "stride": 1, "padding": 0, "op": "linear"}
        net.steps.append(Layer(d.get("name", f"fc{i}"), a, W.reshape(K, C, 1, 1), np.asarray(d.get("bias", np.zeros(K)), dtype=np.int64),
                               src, 128 if input_signed or i else 0, requant=d.get("requant"), out=d.get("name", f"fc{i}"),
                               model_zp=int(d.get("model_zp", 0)), requant_spec=d.get("requant_spec")))
        src = net.steps[-1].out
    net.output = src
    return net


# ---------------------------------------------------------------- 3. the golden model (independent: nested loops)
def _conv_ref(x, L):
    """x: signed int tensor (C, H, W). Plain loops over outputs: no im2col, no packing."""
    a = L.attrs
    C, H, Wd = x.shape
    g, Cg, Kg = a["groups"], a["C"] // a["groups"], a["K"] // a["groups"]
    st, pd, R, S = a["stride"], a["padding"], a["R"], a["S"]
    Ho, Wo = (H + 2 * pd - R) // st + 1, (Wd + 2 * pd - S) // st + 1
    out = np.zeros((a["K"], Ho, Wo), dtype=np.int64)
    for k in range(a["K"]):
        gi = k // Kg
        for oy in range(Ho):
            for ox in range(Wo):
                s = 0
                for c in range(Cg):
                    for r in range(R):
                        for t in range(S):
                            iy, ix = oy * st - pd + r, ox * st - pd + t
                            if 0 <= iy < H and 0 <= ix < Wd:
                                s += int(L.W[k, c, r, t]) * int(x[gi * Cg + c, iy, ix])
                out[k, oy, ox] = s
    return out


def epilogue(acc, L):
    """acc: exact signed dot products (K, ...) -> INT8 activations."""
    if L.requant is not None:
        return np.asarray(L.requant(acc), dtype=np.int64)
    b = L.bias.reshape((-1,) + (1,) * (acc.ndim - 1))
    v = acc + b
    if L.shift:
        v = (v + (1 << (L.shift - 1))) >> L.shift
    return np.clip(v, L.lo, L.hi)


# ---- requantization as data (for deployment bundles) ------------------------------------------------------------
def _trunc_div(x, d):
    return np.where(x >= 0, x // d, -((-x) // d))


def _gemmlowp(v, m, sh):
    """TFLite/gemmlowp: SaturatingRoundingDoublingHighMul(v, m) then RoundingDivideByPOT(sh); m is a Q31 int32."""
    ab = v * m
    nudge = np.where(ab >= 0, 1 << 30, 1 - (1 << 30))
    hi = _trunc_div(ab + nudge, 1 << 31)
    mask = (np.int64(1) << sh) - 1
    rem = hi & mask
    thr = (mask >> 1) + (hi < 0)
    return (hi >> sh) + (rem > thr)


FORMULAS = {                         # candidate integer requantizations y = f(acc + bias, mult, shift) + out_zp, clamped
    "floor": lambda v, m, sh: (v * m) >> sh,
    "round_half_up": lambda v, m, sh: (v * m + np.where(sh > 0, np.int64(1) << np.maximum(sh - 1, 0), 0)) >> sh,
    "float_round_even": lambda v, m, sh: np.round(v * np.asarray(m, dtype=np.float64) / np.exp2(sh)).astype(np.int64),
    "float_round_away": lambda v, m, sh: (np.sign(v * np.asarray(m, dtype=np.float64))
                                          * np.floor(np.abs(v * np.asarray(m, dtype=np.float64) / np.exp2(sh)) + 0.5)).astype(np.int64),
    "gemmlowp": _gemmlowp,
}


def requant_from_spec(spec):
    """Callable acc -> int8 from a requant_spec dict; None for kinds that need the model's own code."""
    k = spec.get("kind")
    if k == "identity":
        return lambda acc: np.asarray(acc, dtype=np.int64)
    if k == "shift_clip":
        sh, lo, hi = int(spec["shift"]), int(spec.get("lo", LO)), int(spec.get("hi", HI))
        rnd = (1 << (sh - 1)) if spec.get("round") and sh > 0 else 0
        return lambda acc: np.clip((np.asarray(acc, dtype=np.int64) + rnd) >> sh, lo, hi)
    if k == "fixed_point":
        f = FORMULAS[spec["formula"]]
        bias, m, sh, zp = (np.asarray(spec[n]) for n in ("bias", "mult", "shift", "out_zp"))
        m = m.astype(np.int64) if spec["formula"] in ("floor", "round_half_up", "gemmlowp") else m
        lo, hi = int(spec.get("lo", LO)), int(spec.get("hi", HI))

        def col(p, a):                   # per-channel parameters run along the first axis of acc (K, ...)
            return p.reshape((-1,) + (1,) * (a.ndim - 1)) if p.ndim == 1 and a.ndim > 1 else p

        def rq(acc):
            a = np.asarray(acc, dtype=np.int64)
            return np.clip(f(a + col(bias, a).astype(np.int64), col(m, a), col(sh, a).astype(np.int64)) + col(zp, a).astype(np.int64), lo, hi)
        return rq
    return None


def match_formula(reference, params, accs):
    """Which FORMULAS entry reproduces reference(acc) exactly on every array in accs; None if none does."""
    for name, _ in FORMULAS.items():
        if name in ("floor", "round_half_up", "gemmlowp") and not np.all(np.asarray(params["mult"]) == np.round(params["mult"])):
            continue
        f = requant_from_spec({"kind": "fixed_point", "formula": name, **params})
        try:
            if all(np.array_equal(f(a), np.asarray(reference(a), dtype=np.int64)) for a in accs):
                return name
        except (ValueError, OverflowError, TypeError):
            continue
    return None


def calibrate(net, x):
    """Choose each layer's shift on this input (smallest shift that brings acc + bias into INT8)."""
    for_t = {"input": x}
    for s in net.steps:
        if isinstance(s, Layer):
            acc = _matmul_ref(_src(for_t, s) - s.model_zp, s)
            if s.requant is None:
                m = int(np.abs(acc + s.bias.reshape((-1,) + (1,) * (acc.ndim - 1))).max())
                s.shift = 0 if m <= HI else int(np.ceil(np.log2(m / HI)))
            for_t[s.out] = epilogue(acc, s)
        else:
            for_t[s[1]] = _host_step(s, for_t)
    return net


def _matmul_ref(x, L):
    if x.ndim == 1:                                        # linear on a vector
        return L.W.reshape(L.attrs["K"], -1).astype(np.int64) @ x.astype(np.int64)
    return _conv_ref(x, L)


def _host_step(s, t):
    if s[0] == "add":
        return np.clip(t[s[2]] + t[s[3]], LO, HI)
    if s[0] == "pool":
        x = t[s[2]]
        n = x.shape[1] * x.shape[2]
        return ((x.reshape(x.shape[0], -1).sum(axis=1) + n // 2) // n).reshape(-1, 1, 1)
    if s[0] == "diff":
        x = t[s[2]]
        return x[0] - x[1]
    raise ValueError(s)


def golden(net, x):
    t = {"input": np.asarray(x, dtype=np.int64)}
    for s in net.steps:
        if isinstance(s, Layer):
            t[s.out] = epilogue(_matmul_ref(_src(t, s) - s.model_zp, s), s)
        else:
            t[s[1]] = _host_step(s, t)
    return t


def _src(t, L):
    """The tensor a layer reads; a (C, 1, 1) map feeding a linear layer is used as a vector (flatten)."""
    x = t[L.src]
    if L.attrs["op"] == "linear" and x.ndim == 3:
        return x.reshape(-1)
    return x


def _is_linear(L):
    return L.attrs["R"] == L.attrs["S"] == 1 and L.attrs.get("stride", 1) == 1 and L.attrs.get("padding", 0) == 0


# ---------------------------------------------------------------- 4. the chip path
def im2col(x, L):
    """uint8 patches (positions, groups, taps) for the X2, taps in (c, r, s) order; padding = the zero point."""
    a = L.attrs
    if x.ndim == 1:
        return (x + L.in_zp).reshape(1, 1, -1), None
    C, H, Wd = x.shape
    g, Cg = a["groups"], a["C"] // a["groups"]
    st, pd, R, S = a["stride"], a["padding"], a["R"], a["S"]
    Ho, Wo = (H + 2 * pd - R) // st + 1, (Wd + 2 * pd - S) // st + 1
    xp = np.pad(x + L.in_zp, ((0, 0), (pd, pd), (pd, pd)), constant_values=L.in_zp)
    P = np.zeros((Ho * Wo, g, Cg * R * S), dtype=np.int64)
    for oy in range(Ho):
        for ox in range(Wo):
            win = xp[:, oy * st:oy * st + R, ox * st:ox * st + S]
            for gi in range(g):
                P[oy * Wo + ox, gi] = win[gi * Cg:(gi + 1) * Cg].reshape(-1)
    return P, (Ho, Wo)


def run(net, x, matmul, gold=None):
    """matmul(layer, patches uint8 (P, g, taps)) -> exact sums over the uint8 inputs, shape (P, K).
    Returns (tensors, per-step check rows). Raises nothing on a mismatch: the rows say where it went wrong."""
    gold = gold or golden(net, x)
    t, rows = {"input": np.asarray(x, dtype=np.int64)}, []
    for s in net.steps:
        if isinstance(s, Layer):
            P, hw = im2col(_src(t, s), s)
            if P.min() < 0 or P.max() > 255:
                raise ValueError(f"{s.name}: X2 inputs outside uint8 ({P.min()}..{P.max()})")
            y_u = np.asarray(matmul(s, P), dtype=np.int64)                         # (positions, K)
            corr = (s.in_zp + s.model_zp) * s.W.reshape(s.attrs["K"], -1).sum(axis=1)   # both offsets, exactly
            acc = (y_u - corr).T                                                   # (K, positions)
            acc = acc.reshape(-1) if hw is None else acc.reshape(-1, *hw)
            t[s.out] = epilogue(acc, s)
            rows.append({"step": s.name, "kind": "x2_layer", "exact": bool(np.array_equal(t[s.out], gold[s.out]))})
        else:
            t[s[1]] = _host_step(s, t)
            rows.append({"step": s[1], "kind": s[0], "exact": bool(np.array_equal(t[s[1]], gold[s[1]]))})
    return t, rows


def build_mnist(name, kernels_pos, kernels_neg, H=28, W=28):
    """BMsemi MNIST edge filter: one 5x5 SAME conv with a '+' and a '-' binary kernel; output = '+' minus '-'."""
    w = np.stack([np.asarray(kernels_pos), np.asarray(kernels_neg)])[:, None].astype(np.int64)      # (2, 1, R, S)
    R = w.shape[2]
    a = {"K": 2, "C": 1, "R": R, "S": w.shape[3], "groups": 1, "stride": 1, "padding": R // 2, "op": "conv2d"}
    net = Net(name, (1, H, W), False, notes=["binary image and kernels from the repo; output = '+' map minus '-' map"])
    net.steps = [Layer("edge", a, w, np.zeros(2, dtype=np.int64), "input", 0, requant=lambda acc: acc, out="edge",
                       requant_spec={"kind": "identity"}),
                 ("diff", "edge_map", "edge")]
    net.output = "edge_map"
    return net
