"""Build the local IR from external artifacts / normalised workloads (read-only inputs)."""
from x2c.compiler.ir import IRGraph, IRNode, TensorType
from x2c.inputs.graph_check import GRAPH

ASSUMPTIONS = [
    "dtype is not recorded in compiler_graph.json: tensors are 'unspecified' (INT8 is a property of the matrix ops' weights, per the V2 README)",
    "modules that are not in the MAESTRO layer table are classified 'shape_preserving' (norm/activation); their true kind is NOT in the artifact",
    "a module whose target name ends with 'pool' is classified 'pool' (kind of pooling not recorded)",
]


def build_ir_from_compiler_graph(repo, workload, graph_rel=GRAPH):
    g = repo.read_json(graph_rel)["nodes"]
    layers = {l.name: l for l in workload.layers}
    producers = {n["name"]: [] for n in g}
    for n in g:                                              # invert 'users' into ordered producer lists
        for u in n.get("users", []):
            if u in producers:
                producers[u].append(n["name"])
    nodes = []
    for n in g:
        name, fx_op, sh = n["name"], n["op"], n.get("shape")
        origin = {"artifact": graph_rel, "fx_op": fx_op, "target": str(n.get("target"))}
        ins = producers[name]
        if fx_op == "placeholder":
            op, attrs = "input", {}
        elif fx_op == "output":
            op, attrs = "output", {}
            sh = sh or (g[[x["name"] for x in g].index(ins[0])].get("shape") if ins else None)
        elif fx_op == "call_function":
            op, attrs = ("add", {}) if "add" in str(n.get("target")) else ("shape_preserving", {})
            if op != "add":
                origin["caveat"] = "unrecognised function"
        elif fx_op == "call_method":
            op, attrs = ("flatten", {}) if str(n.get("target")) == "flatten" else ("shape_preserving", {})
        elif name in layers:                                   # a mapped matrix layer: attributes come from the MAESTRO table
            L = layers[name]
            op = L.op
            attrs = {"K": L.K, "C": L.C, "R": L.R, "S": L.S, "stride": L.stride, "padding": L.padding, "groups": L.groups,
                     "weight_dtype": L.dtype, "tile_width": L.extra.get("tile_width"),
                     "compiler_expected_tiles": L.extra.get("compiler_expected_tiles")}
            origin["layer_table"] = "maestro_layer_dataflow.json"
        elif str(n.get("target")).endswith("pool"):
            op, attrs = "pool", {}
            origin["caveat"] = "pooling kind not recorded in the artifact"
        else:
            op, attrs = "shape_preserving", {}
            origin["caveat"] = "module kind not recorded in the artifact; classified by shape behaviour only"
        nodes.append(IRNode(name, op, ins, TensorType(tuple(sh) if sh else (), "unspecified"), attrs, origin))
    return IRGraph(workload.id, nodes, {"source": graph_rel, "repo_commit": workload.provenance["repo_commit"], "assumptions": ASSUMPTIONS})


def build_ir_from_chain_workload(workload):
    """Sequential models (DeepGate, MNIST edge filter): input -> layer0 -> layer1 -> ... -> output."""
    L0 = workload.layers[0]
    if workload.kind == "int8_fc_network":
        in_shape = (1, L0.C)
    else:
        h = (L0.Y - 1) * L0.stride + L0.R - 2 * L0.padding
        in_shape = (1, L0.C, h, (L0.X - 1) * L0.stride + L0.S - 2 * L0.padding)
    nodes = [IRNode("input", "input", [], TensorType(in_shape))]
    prev = "input"
    for L in workload.layers:
        out = (1, L.K) if L.op == "linear" else (1, L.K, L.Y, L.X)
        nodes.append(IRNode(L.name, L.op, [prev], TensorType(out),
                            {"K": L.K, "C": L.C, "R": L.R, "S": L.S, "stride": L.stride, "padding": L.padding,
                             "groups": L.groups, "weight_dtype": L.dtype},
                            {"workload": workload.id, "note": "layer sequence from the normalised workload"}))
        prev = L.name
    nodes.append(IRNode("output", "output", [prev], nodes[-1].out))
    return IRGraph(workload.id, nodes, {"repo_commit": workload.provenance["repo_commit"],
                                       "assumptions": ["requantisation / activation between layers is not modelled in this IR (chain of matrix ops only)"]})
