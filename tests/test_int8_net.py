"""compiler/int8_net.py: module kinds from structure, INT8 lowering, the independent golden model, and the chip path
(im2col -> X2 matmul with uint8 offsets -> exact correction -> epilogue -> host steps)."""
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from compiler import int8_net as N
from compiler import x2_pack as xp
from compiler.ir import IRGraph
from compiler.partition import load_hardware

IR = Path(__file__).resolve().parent / "fixtures" / "ir"


def graph(name):
    return IRGraph.from_dict(json.loads((IR / f"{name}.json").read_text()))


def numpy_x2(L, P):
    """Exact integer stand-in for the X2: per group, uint8 patch . weights."""
    return xp.reference(L.W, P, L.attrs)


@pytest.fixture(scope="module")
def mnv2():
    net = N.build_mobilenet(graph("mobilenetv2"))
    x = np.random.default_rng(7).integers(0, 128, size=net.input_shape)
    N.calibrate(net, x)
    return net, x, N.golden(net, x)


def test_module_kinds_on_the_real_graph():
    kinds = N.infer_kinds(graph("mobilenetv2"))
    assert Counter(kinds.values()) == {"batchnorm": 16, "relu6": 11}


def test_kinds_refuse_an_unexplained_node():
    g = graph("mobilenetv2")
    sp = [n for n in g.nodes if n.op == "shape_preserving"]
    relu = next(n for n in sp if N.infer_kinds(g)[n.name] == "relu6")
    relu.inputs = [next(n.name for n in g.nodes if n.op == "add")]      # a shape_preserving node after an add: unknown
    with pytest.raises(ValueError, match="cannot infer"):
        N.infer_kinds(g)


def test_mobilenet_lowering(mnv2):
    net, x, gold = mnv2
    layers = [s for s in net.steps if isinstance(s, N.Layer)]
    assert len(layers) == 17 and sum(1 for s in net.steps if not isinstance(s, N.Layer)) == 4      # 3 adds + pool
    assert sum(1 for L in layers if L.lo == 0) == 11                                                # ReLU6 folded
    assert all(L.in_zp in (0, 128) for L in layers)
    for s in net.steps:
        v = gold[s.out if isinstance(s, N.Layer) else s[1]]
        assert v.min() >= -128 and v.max() <= 127                                                   # INT8 everywhere
    assert gold[net.output].shape == (10,)


def test_chip_path_is_exact_with_a_numpy_x2(mnv2):
    net, x, gold = mnv2
    _, rows = N.run(net, x, numpy_x2, gold)
    assert len(rows) == 21 and all(r["exact"] for r in rows)


def test_chip_path_is_exact_on_the_4bit_x2_model(mnv2):
    net, x, gold = mnv2
    hw = load_hardware()
    mm = lambda L, P: xp.run_layer(xp.pack_layer(L.name, {k: L.attrs[k] for k in ("K", "C", "R", "S", "groups")}, L.W, hw), P)[0]
    _, rows = N.run(net, x, mm, gold)
    assert all(r["exact"] for r in rows)


def test_a_wrong_x2_result_is_reported_at_its_layer(mnv2):
    net, x, gold = mnv2
    bad = net.steps[5].name

    def mm(L, P):
        y = numpy_x2(L, P)
        if L.name == bad:
            y = y.copy(); y[0, 0] += 1 << 12
        return y
    _, rows = N.run(net, x, mm, gold)
    first = next(r for r in rows if not r["exact"])
    assert first["step"] == bad


def test_offsets_are_corrected_exactly():
    """Signed input (+128 for the X2) and a model zero point: W . (x - zp) exactly."""
    rng = np.random.default_rng(3)
    ds = [{"name": "a", "W": rng.integers(-128, 128, size=(6, 9)), "requant": lambda acc: np.clip(acc >> 9, -128, 127), "model_zp": -5},
          {"name": "b", "W": rng.integers(-128, 128, size=(4, 6)), "requant": lambda acc: np.clip(acc >> 7, -128, 127), "model_zp": 3}]
    net = N.build_chain("t", ds, (9,), True)
    x = rng.integers(-128, 128, size=9)
    gold = N.golden(net, x)
    assert np.array_equal(gold["a"], np.clip((ds[0]["W"] @ (x + 5)) >> 9, -128, 127))
    _, rows = N.run(net, x, numpy_x2, gold)
    assert all(r["exact"] for r in rows) and net.steps[0].in_zp == 128


def test_padding_uses_the_zero_point():
    L = N.Layer("p", {"K": 1, "C": 1, "R": 3, "S": 3, "groups": 1, "stride": 1, "padding": 1, "op": "conv2d"},
                np.ones((1, 1, 3, 3), dtype=np.int64), np.zeros(1, dtype=np.int64), "input", 128)
    P, hw = N.im2col(-np.ones((1, 2, 2), dtype=np.int64), L)
    assert hw == (2, 2) and P.shape == (4, 1, 9) and set(np.unique(P)) == {127, 128}


def test_mnist_edge_filter_matches_its_definition():
    rng = np.random.default_rng(5)
    kp, kn = rng.integers(0, 2, size=(5, 5)), rng.integers(0, 2, size=(5, 5))
    img = rng.integers(0, 2, size=(1, 28, 28))
    net = N.build_mnist("m", kp, kn)
    gold = N.golden(net, img)
    pad = np.pad(img[0], 2)
    want = np.array([[np.sum(pad[y:y + 5, x:x + 5] * (kp - kn)) for x in range(28)] for y in range(28)])
    assert np.array_equal(gold["edge_map"], want)
    _, rows = N.run(net, img, numpy_x2, gold)
    assert all(r["exact"] for r in rows)
