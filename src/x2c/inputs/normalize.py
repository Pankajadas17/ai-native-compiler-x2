"""
Normalise BMsemi_Model assets into local WorkloadSpec objects. Read-only; every spec carries provenance
(repo commit + fingerprints of the exact files it was derived from).

ASSUMPTIONS (confirm with the compiler team):
  * Compiler V2 tile_width is the number of PE columns used for that layer; PE rows = 64.
  * Depthwise layers are described with groups == C (as in maestro_layer_dataflow.json).
  * The TinyMobileNetV2 checkpoint (.pt) is NOT loaded (needs torch): layer VALUES are synthetic,
    only the GEOMETRY is real. This is recorded in data_source and repeated in every report.
"""
from x2c.inputs.catalog import COMPILER_V2, EXAMPLES
from x2c.inputs.schemas import LayerSpec, WorkloadSpec

DATAFLOW = f"{COMPILER_V2}/artifacts/maestro_layer_dataflow.json"
GRAPH = f"{COMPILER_V2}/artifacts/compiler_graph.json"
SCHEDULE = f"{COMPILER_V2}/artifacts/maestro_schedule.json"
DEEP = f"{EXAMPLES}/Deep_gate"
MNIST_DIR = "projects/2_building_blocks/Neuromorphic_compiler"
_OP = {"CONV": "conv2d", "DEPTHWISE_CONV": "depthwise_conv2d", "POINTWISE_CONV": "pointwise_conv2d", "GEMM": "linear"}


def _prov(repo, rels):
    return {"repo_commit": repo.commit(), "files": [repo.fingerprint(r) for r in rels]}


def compiler_v2_workload(repo):
    layers = []
    for l in repo.read_json(DATAFLOW)["layers"]:
        d = l["dimensions"]
        if l["stride_x"] != l["stride_y"] or l["padding_x"] != l["padding_y"]:
            raise ValueError(f"{l['layer_name']}: non-square stride/padding is not supported by the local schema")
        layers.append(LayerSpec(
            name=l["layer_name"], op=_OP[l["layer_type"]], K=d["K"], C=d["C"], R=d["R"], S=d["S"], Y=d["Y"], X=d["X"],
            stride=l["stride_x"], padding=l["padding_x"], groups=l["groups"],
            extra={"compiler_expected_tiles": l["expected_tile_count"], "tile_width": l["tile_width"],
                   "compiler_layer_type": l["layer_type"], "module": l["module_name"]}))
    return WorkloadSpec(
        id="bm_compiler_v2_tiny_mobilenetv2", title="Compiler V2: TinyMobileNetV2 (digits), 17 mapped layers",
        source_model="Neuromorphic_compiler_V2 (maestro_layer_dataflow.json)", kind="conv_net_geometry",
        data_source="synthetic_values_real_geometry", loader="compiler_v2_layers", layers=layers,
        pe={"rows": 64, "cols": 32}, provenance=_prov(repo, [DATAFLOW, GRAPH, SCHEDULE]),
        stages=["S01", "S02", "S03", "S04"],
        notes="Geometry (K,C,R,S,Y,X,stride,padding,groups) and compiler tile counts are real; weights/inputs are seeded synthetic integers.")


def deepgate_workload(repo):
    S = repo.read_json(f"{DEEP}/schema.json")
    layers = [LayerSpec(name=f"fc{i}", op="linear", K=L["shape_out"][0], C=L["shape_in"][0], dtype="int8")
              for i, L in enumerate(S)]
    return WorkloadSpec(
        id="bm_deepgate_ad01_int8", title="DeepGate AD01 autoencoder: 10 INT8 fully-connected layers",
        source_model="bm_models.models.deepgate:infer", kind="int8_fc_network", data_source="repo_bundled_vectors",
        loader="deepgate_int8", layers=layers, pe={"rows": 32, "cols": 32},
        provenance=_prov(repo, [f"{DEEP}/schema.json", f"{DEEP}/test_cases.json"]), stages=["S01", "S02", "S04"],
        notes="Real INT8 weights and bundled test vectors; X2 computes the integer dot products, the repo's own requantisation finishes each layer.")


def mnist_workload(repo):
    imgs = [f"{MNIST_DIR}/mnist5_bin_28x28.txt", f"{MNIST_DIR}/mnist9_bin_28x28.txt"]
    layer = LayerSpec(name="edge_5x5", op="conv2d", K=2, C=1, R=5, S=5, Y=28, X=28, stride=1, padding=2, dtype="binary",
                      extra={"columns": "positive kernel, negative kernel; output = pos - neg"})
    return WorkloadSpec(
        id="bm_mnist_edge_filter", title="MNIST multi-PE edge filter (digits 5 and 9)",
        source_model="bm_models.models.mnist:edge_filter", kind="crossbar_conv", data_source="repo_bundled_vectors",
        loader="mnist_edge", layers=[layer], pe={"rows": 32, "cols": 32},
        provenance=_prov(repo, [r for r in imgs if repo.exists(r)] or [f"{MNIST_DIR}/mnist5_bin_28x28.txt"]), stages=["S01", "S02", "S04"],
        notes="Real bundled digit images and kernels; compared with the repo's own operation output.")


def all_workloads(repo):
    out = []
    for f in (compiler_v2_workload, deepgate_workload, mnist_workload):
        try:
            out.append(f(repo))
        except FileNotFoundError:
            continue                       # an asset family missing from this checkout is simply not offered
    return out
