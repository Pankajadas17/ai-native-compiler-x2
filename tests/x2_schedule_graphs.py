"""Small IR graphs for tests/test_x2_schedule.py (shapes like tiny MobileNetV2 blocks; built with compiler.ir)."""
from compiler.ir import IRGraph, IRNode, TensorType as T

def conv(name, inp, op, K, C, R, Y, groups=1):
    return IRNode(name, op, [inp], T((1, K, Y, Y)), {"K": K, "C": C, "R": R, "S": R, "Y": Y, "X": Y, "groups": groups,
                                                      "stride": 1, "padding": R // 2})

def mini():
    n = [IRNode("x", "input", [], T((1, 3, 16, 16))),
         conv("c0", "x", "conv2d", 16, 3, 3, 16), IRNode("a0", "shape_preserving", ["c0"], T((1, 16, 16, 16))),
         conv("dw1", "a0", "depthwise_conv2d", 16, 16, 3, 16, groups=16), IRNode("a1", "shape_preserving", ["dw1"], T((1, 16, 16, 16))),
         conv("pw1", "a1", "pointwise_conv2d", 16, 16, 1, 16), IRNode("add1", "add", ["pw1", "a0"], T((1, 16, 16, 16))),
         conv("pw2", "add1", "pointwise_conv2d", 32, 16, 1, 16), IRNode("pool", "pool", ["pw2"], T((1, 32, 1, 1))),
         IRNode("flat", "flatten", ["pool"], T((1, 32))),
         IRNode("fc", "linear", ["flat"], T((1, 10)), {"K": 10, "C": 32, "R": 1, "S": 1, "Y": 1, "X": 1, "groups": 1}),
         IRNode("y", "output", ["fc"], T((1, 10)))]
    return IRGraph("mini", n)


def with_small_layer():
    """mini() plus a 1-tile first conv (K=4, 3x3 on 3 channels: 27 rows x 16 cells), like the real features_0_0."""
    g = mini()
    n = g.nodes
    n[1] = conv("c0", "x", "conv2d", 4, 3, 3, 16)
    n[2] = IRNode("a0", "shape_preserving", ["c0"], T((1, 4, 16, 16)))
    n[3] = conv("dw1", "a0", "depthwise_conv2d", 4, 4, 3, 16, groups=4)
    n[4] = IRNode("a1", "shape_preserving", ["dw1"], T((1, 4, 16, 16)))
    n[5] = conv("pw1", "a1", "pointwise_conv2d", 4, 4, 1, 16)
    n[6] = IRNode("add1", "add", ["pw1", "a0"], T((1, 4, 16, 16)))
    n[7] = conv("pw2", "add1", "pointwise_conv2d", 32, 4, 1, 16)
    return g
