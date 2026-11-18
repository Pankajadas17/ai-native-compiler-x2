"""S03-style validation of an EXTERNAL compiler graph (compiler_graph.json) and its consistency with the
MAESTRO per-layer table. Deterministic, read-only."""
from collections import Counter, deque

from x2c.inputs.catalog import COMPILER_V2

GRAPH = f"{COMPILER_V2}/artifacts/compiler_graph.json"
LAYERS = f"{COMPILER_V2}/artifacts/maestro_layer_dataflow.json"


def validate_compiler_graph(repo, graph_rel=GRAPH, layers_rel=LAYERS):
    errors = []
    g = repo.read_json(graph_rel)
    nodes = g.get("nodes", [])
    by_name, dup = {}, []
    for n in nodes:
        (dup.append(n["name"]) if n["name"] in by_name else by_name.__setitem__(n["name"], n))
    if dup:
        errors.append(f"duplicate node names: {sorted(set(dup))[:5]}")
    ops = Counter(n.get("op") for n in nodes)
    if ops.get("placeholder", 0) != 1:
        errors.append(f"expected exactly 1 placeholder, found {ops.get('placeholder', 0)}")
    if ops.get("output", 0) < 1:
        errors.append("no output node")
    for n in nodes:                                         # edges and shapes
        for u in n.get("users", []):
            if u not in by_name:
                errors.append(f"node '{n['name']}' lists unknown user '{u}'")
        sh = n.get("shape")
        if n.get("op") == "call_module":
            if not (isinstance(sh, list) and sh and all(isinstance(v, int) and v > 0 for v in sh)):
                errors.append(f"call_module '{n['name']}' has an invalid shape {sh}")
    # acyclic + reachability (Kahn on the users graph)
    indeg = Counter({n["name"]: 0 for n in nodes})
    for n in nodes:
        for u in n.get("users", []):
            if u in by_name:
                indeg[u] += 1
    q = deque(sorted(k for k, v in indeg.items() if v == 0)); seen = 0
    d = dict(indeg)
    while q:
        k = q.popleft(); seen += 1
        for u in by_name[k].get("users", []):
            if u in d:
                d[u] -= 1
                if d[u] == 0:
                    q.append(u)
    if seen != len(nodes):
        errors.append(f"graph has a cycle or unreachable part ({len(nodes) - seen} nodes not ordered)")
    roots = [k for k, v in indeg.items() if v == 0]
    if len(roots) != 1:
        errors.append(f"expected a single root, found {len(roots)}: {sorted(roots)[:5]}")
    # consistency with the MAESTRO layer table
    layer_mismatch = []
    for l in repo.read_json(layers_rel)["layers"]:
        n = by_name.get(l["layer_name"])
        if n is None:
            layer_mismatch.append(f"{l['layer_name']}: not in graph")
        elif n.get("shape") != l.get("output_shape"):
            layer_mismatch.append(f"{l['layer_name']}: graph shape {n.get('shape')} != layer output_shape {l.get('output_shape')}")
    errors += layer_mismatch
    return {"ok": not errors, "errors": errors, "node_count": len(nodes), "ops": dict(sorted(ops.items())),
            "layers_checked": len(repo.read_json(layers_rel)["layers"])}
