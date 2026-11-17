"""
Public stand-in model graphs for clones without the team's private BMsemi_Model.

    python tools/make_public_fixtures.py          # writes tests/fixtures/ir_public/*.json (deterministic)

They are written HERE, by hand-chosen shapes, and contain nothing from BMsemi_Model:
  invres_net.json  a small inverted-residual CNN in the MobileNetV2 style: conv3x3 -> BN -> ReLU6, two blocks
                   (pointwise expand -> BN -> ReLU6 -> depthwise 3x3 -> BN -> ReLU6 -> pointwise project -> BN, the
                   first with a residual add), global pool, flatten, linear. BN / ReLU6 appear as 'shape_preserving'
                   nodes, as in the compiler's graphs, so module-kind inference is exercised.
  mlp_chain.json   five linear layers (a DeepGate-style chain, own sizes)
  edge5x5.json     one 5x5 'same' convolution with two output maps (an MNIST-style edge filter)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "ir_public"
NOTE = "PUBLIC stand-in written by tools/make_public_fixtures.py; shapes chosen by hand, no data from any team repository"


def node(name, op, inputs, shape, **attrs):
    return {"name": name, "op": op, "inputs": inputs, "shape": shape, "dtype": "unspecified", "attrs": attrs, "origin": {"note": NOTE}}


def conv(name, src, cin, cout, k, stride, groups, hw_out, op):
    return node(name, op, [src], [1, cout, hw_out, hw_out], K=cout, C=cin, R=k, S=k, stride=stride, padding=k // 2,
                groups=groups, weight_dtype="int8", tile_width=min(cout, 16))


def invres_net():
    n = [node("x", "input", [], [1, 3, 8, 8])]
    n += [conv("stem", "x", 3, 8, 3, 2, 1, 4, "conv2d"), node("stem_bn", "shape_preserving", ["stem"], [1, 8, 4, 4]),
          node("stem_act", "shape_preserving", ["stem_bn"], [1, 8, 4, 4])]
    # block 1 (8 -> 8, residual)
    n += [conv("b1_expand", "stem_act", 8, 16, 1, 1, 1, 4, "pointwise_conv2d"), node("b1_expand_bn", "shape_preserving", ["b1_expand"], [1, 16, 4, 4]),
          node("b1_expand_act", "shape_preserving", ["b1_expand_bn"], [1, 16, 4, 4]),
          conv("b1_dw", "b1_expand_act", 16, 16, 3, 1, 16, 4, "depthwise_conv2d"), node("b1_dw_bn", "shape_preserving", ["b1_dw"], [1, 16, 4, 4]),
          node("b1_dw_act", "shape_preserving", ["b1_dw_bn"], [1, 16, 4, 4]),
          conv("b1_project", "b1_dw_act", 16, 8, 1, 1, 1, 4, "pointwise_conv2d"), node("b1_project_bn", "shape_preserving", ["b1_project"], [1, 8, 4, 4]),
          node("b1_add", "add", ["stem_act", "b1_project_bn"], [1, 8, 4, 4])]
    # block 2 (8 -> 12, stride 2, no residual)
    n += [conv("b2_expand", "b1_add", 8, 24, 1, 1, 1, 4, "pointwise_conv2d"), node("b2_expand_bn", "shape_preserving", ["b2_expand"], [1, 24, 4, 4]),
          node("b2_expand_act", "shape_preserving", ["b2_expand_bn"], [1, 24, 4, 4]),
          conv("b2_dw", "b2_expand_act", 24, 24, 3, 2, 24, 2, "depthwise_conv2d"), node("b2_dw_bn", "shape_preserving", ["b2_dw"], [1, 24, 2, 2]),
          node("b2_dw_act", "shape_preserving", ["b2_dw_bn"], [1, 24, 2, 2]),
          conv("b2_project", "b2_dw_act", 24, 12, 1, 1, 1, 2, "pointwise_conv2d"), node("b2_project_bn", "shape_preserving", ["b2_project"], [1, 12, 2, 2])]
    n += [node("pool", "pool", ["b2_project_bn"], [1, 12, 1, 1]), node("flat", "flatten", ["pool"], [1, 12]),
          node("classifier", "linear", ["flat"], [1, 10], K=10, C=12, R=1, S=1, stride=1, padding=0, groups=1, weight_dtype="int8", tile_width=10),
          node("output", "output", ["classifier"], [1, 10])]
    return {"name": "public_invres_net", "meta": {"source": NOTE, "assumptions": ["seeded INT8 weights; shapes chosen here"]}, "nodes": n}


def mlp_chain():
    dims = [96, 48, 16, 48, 96, 96]
    n, src = [node("input", "input", [], [1, dims[0]])], "input"
    for i in range(5):
        n.append(node(f"fc{i}", "linear", [src], [1, dims[i + 1]], K=dims[i + 1], C=dims[i], R=1, S=1, stride=1, padding=0,
                      groups=1, weight_dtype="int8"))
        src = f"fc{i}"
    n.append(node("output", "output", [src], [1, dims[-1]]))
    return {"name": "public_mlp_chain", "meta": {"source": NOTE, "assumptions": ["chain of matrix ops only"]}, "nodes": n}


def edge5x5():
    return {"name": "public_edge5x5", "meta": {"source": NOTE, "assumptions": ["binary image and kernels"]},
            "nodes": [node("input", "input", [], [1, 1, 28, 28]),
                      node("edge_5x5", "conv2d", ["input"], [1, 2, 28, 28], K=2, C=1, R=5, S=5, stride=1, padding=2, groups=1,
                           weight_dtype="binary"),
                      node("output", "output", ["edge_5x5"], [1, 2, 28, 28])]}


GRAPHS = {"invres_net": invres_net, "mlp_chain": mlp_chain, "edge5x5": edge5x5}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in GRAPHS.items():
        (OUT / f"{name}.json").write_text(json.dumps(fn(), indent=1) + "\n", encoding="utf-8")
        print(f"written: {(OUT / f'{name}.json').relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
