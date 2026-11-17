"""
Discover what the BMsemi_Model repository offers, without modifying it.
Output is deterministic (sorted) so two runs on the same commit produce identical catalogs.
"""
from collections import Counter

COMPILER_V2 = "projects/2_building_blocks/Neuromorphic_compiler_V2"
EXAMPLES = "projects/3_example_models"

# Which asset kinds we can already consume, and with which local loader.
LOADERS = {
    "operation_catalog": "catalog_json",
    "compiler_layer_dataflow": "compiler_v2_normalizer",
    "compiler_graph": "graph_check",
    "compiler_schedule": "none (cataloged)",
    "model_schema": "deepgate_int8",
    "model_test_vectors": "deepgate_int8",
    "bundled_image": "mnist_edge",
}


def _kind(rel):
    n = rel.lower()
    if n.endswith("maestro_layer_dataflow.json"): return "compiler_layer_dataflow"
    if n.endswith("compiler_graph.json"): return "compiler_graph"
    if n.endswith("maestro_schedule.json"): return "compiler_schedule"
    if n.endswith("fx_graph.json"): return "compiler_fx_graph"
    if "deep_gate/schema.json" in n: return "model_schema"
    if "deep_gate/test_cases.json" in n: return "model_test_vectors"
    if n.endswith(".pt"): return "checkpoint"
    if n.endswith(".tflite") or n.endswith(".onnx"): return "exported_model"
    if "mnist" in n and n.endswith(".txt"): return "bundled_image"
    if n.endswith(".json"): return "json_artifact"
    return "other"


def _record(repo, rel):
    fp = repo.fingerprint(rel)
    k = _kind(rel)
    loader = LOADERS.get(k, "none (cataloged)")
    if k == "checkpoint":
        loader = "none (needs torch)"
    return {"path": rel, "kind": k, "size": fp["size"], "sha256_lf": fp["sha256_lf"], "loader": loader}


def discover(repo):
    cat = repo.read_json("catalog.json")
    ops = cat["operations"]
    operations = [{"name": o.get("name", f"op{i}"), "module": o.get("module", ""), "status": o.get("status", ""),
                   "entrypoint": o.get("entrypoint", "")} for i, o in enumerate(ops if isinstance(ops, list) else ops.values())]
    # small, directly consumable assets
    wanted = []
    for base in (f"{COMPILER_V2}/artifacts", f"{COMPILER_V2}/checkpoints"):
        if repo.exists(base):
            wanted += [f"{base}/{n}" for n in repo.list_dir(base)]
    deep = f"{EXAMPLES}/Deep_gate"
    if repo.exists(deep):
        wanted += [f"{deep}/schema.json", f"{deep}/test_cases.json"]
    wanted += [r for r in repo.walk("projects/2_building_blocks/Neuromorphic_compiler") if "mnist" in r.lower() and r.endswith(".txt")]
    assets = [_record(repo, r) for r in sorted(set(wanted)) if repo.exists(r)]
    # large corpora: summarised only (counted by extension), not hashed file by file
    corpora = []
    for d in sorted(repo.list_dir(EXAMPLES)) if repo.exists(EXAMPLES) else []:
        files = list(repo.walk(f"{EXAMPLES}/{d}"))
        ext = Counter(f.rsplit(".", 1)[-1].lower() if "." in f.rsplit("/", 1)[-1] else "(none)" for f in files)
        size = sum(repo.path(f).stat().st_size for f in files)
        corpora.append({"project": f"{EXAMPLES}/{d}", "files": len(files), "bytes": size, "extensions": dict(sorted(ext.items())),
                        "loader": "deepgate_int8 (schema + vectors)" if d == "Deep_gate" else "none (cataloged only)"})
    return {"repo_commit": repo.commit(), "operations": operations, "assets": assets, "corpora": corpora}
