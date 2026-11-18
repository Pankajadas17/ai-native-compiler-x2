import copy
import json

import pytest

from x2c.compiler.ir import IRGraph, IRNode, TensorType
from x2c.compiler.ir_builder import build_ir_from_chain_workload, build_ir_from_compiler_graph
from x2c.compiler.ir_validate import validate_ir
from x2c.compiler.partition import load_hardware, propose_partition_baseline, validate_partition
from x2c.inputs.bmsemi_source import ReadOnlyRepo
from x2c.inputs.graph_check import GRAPH
from x2c.inputs.normalize import compiler_v2_workload, deepgate_workload, mnist_workload


@pytest.fixture(scope="module")
def repo():
    return ReadOnlyRepo()


@pytest.fixture(scope="module")
def ir(repo):
    return build_ir_from_compiler_graph(repo, compiler_v2_workload(repo))


@pytest.fixture(scope="module")
def hw():
    return load_hardware()


# ---------------------------------------------------------------- the IR of the real compiler graph
def test_real_graph_builds_a_valid_ir(ir):
    errs, warns = validate_ir(ir)
    assert errs == [] and len(ir.nodes) == 51
    ops = {}
    for n in ir.nodes:
        ops[n.op] = ops.get(n.op, 0) + 1
    assert ops == {"input": 1, "conv2d": 1, "depthwise_conv2d": 5, "pointwise_conv2d": 10, "linear": 1, "add": 3,
                   "pool": 1, "flatten": 1, "shape_preserving": 27, "output": 1}
    assert any("27 node(s)" in w for w in warns)                       # honest: kinds not recorded in the artifact
    assert all(n.out.dtype == "unspecified" for n in ir.nodes)          # dtype is never silently guessed
    assert "dtype is not recorded" in ir.meta["assumptions"][0]


def test_ir_round_trips_through_json_with_a_stable_digest(ir):
    back = IRGraph.from_dict(json.loads(json.dumps(ir.to_dict())))
    assert back.digest() == ir.digest() and validate_ir(back)[0] == []


def test_chain_irs_for_sequential_models(repo):
    dg = build_ir_from_chain_workload(deepgate_workload(repo)); mn = build_ir_from_chain_workload(mnist_workload(repo))
    assert validate_ir(dg)[0] == [] and dg.nodes[0].out.shape == (1, 640) and dg.nodes[-2].out.shape == (1, 640)
    assert validate_ir(mn)[0] == [] and mn.nodes[0].out.shape == (1, 1, 28, 28) and mn.nodes[1].out.shape == (1, 2, 28, 28)


# ---------------------------------------------------------------- shape inference rejects corrupted IRs
def _mut(ir, fn):
    g = copy.deepcopy(ir); fn(g.by_name()); return validate_ir(g)[0]


@pytest.mark.parametrize("label,fn,msg", [
    ("wrong declared conv shape", lambda b: setattr(b["features_0_0"], "out", TensorType((1, 8, 5, 5))), "shape inference gives"),
    ("wrong input channels", lambda b: b["features_1_conv_0_0"].attrs.update(C=7), "expects input channels"),
    ("shape-changing norm", lambda b: setattr(b["features_0_1"], "out", TensorType((1, 8, 2, 2))), "must preserve shape"),
    ("add with mismatched input", lambda b: setattr(b["add"], "out", TensorType((1, 9, 4, 4))), "add inputs"),
    ("bad flatten", lambda b: setattr(b["flatten"], "out", TensorType((1, 31))), "flatten of"),
    ("linear mismatch", lambda b: b["classifier"].attrs.update(C=33), "linear C="),
    ("depthwise groups", lambda b: b["features_2_conv_1_0"].attrs.update(groups=1), "depthwise needs groups"),
    ("pool grows the map", lambda b: setattr(b["pool"], "out", TensorType((1, 32, 9, 9))), "pool must keep"),
    ("unknown input", lambda b: b["features_0_1"].inputs.append("ghost"), "unknown input"),
    ("second input node", lambda b: setattr(b["features_0_1"], "op", "input"), "exactly one input"),
])
def test_ir_validator_finds_each_corruption(ir, label, fn, msg):
    errs = _mut(ir, fn)
    assert any(msg in e for e in errs), (label, errs)


def test_ir_validator_catches_ordering_and_duplicates(ir):
    g = copy.deepcopy(ir); g.nodes[1], g.nodes[2] = g.nodes[2], g.nodes[1]
    assert any("not topologically ordered" in e for e in validate_ir(g)[0])
    g = copy.deepcopy(ir); g.nodes.append(copy.deepcopy(g.nodes[3]))
    assert any("duplicate node name" in e for e in validate_ir(g)[0])


# ---------------------------------------------------------------- hardware config: chip numbers, compiler geometry kept apart
def test_hardware_config_is_derived_from_the_chip_and_keeps_the_compiler_geometry_apart(hw):
    assert hw["version"] == "1.0-chip" and "sky130-reram-fpga" in hw["status"]
    assert (hw["x2"]["rows"], hw["x2"]["cols"], hw["x2"]["blocks"]) == (32, 32, 2)
    assert hw["x2"]["cell_bits"] == 4          # decision following BMsemi DeepGate; the chip model computes 1-bit
    assert (hw["efpga"]["luts"], hw["efpga"]["macc_blocks"], hw["efpga"]["io_pins"]) == (2160, 18, 36)
    assert (hw["compiler_v2"]["pe_rows"], hw["compiler_v2"]["default_tile_cols"]) == (64, 32)   # compiler view only


# ---------------------------------------------------------------- baseline partition and its independent validator
def test_baseline_partition_is_legal_and_matches_the_compiler(ir, hw):
    plan = propose_partition_baseline(ir, hw)
    assert validate_partition(ir, plan, hw) == []
    t = plan["totals"]
    assert (t["x2_nodes"], t["efpga_nodes"], t["x2_tiles"], t["boundary_edges"], t["boundary_bytes"]) == (17, 32, 24, 32, 2480)
    assert t["x2_tiles"] == sum(n.attrs["compiler_expected_tiles"] for n in ir.nodes if n.op in ("conv2d", "depthwise_conv2d", "pointwise_conv2d", "linear"))
    assert all(v["resource_estimate"] is None for v in plan["assignments"].values() if v["target"] == "efpga")


def test_chain_models_go_entirely_to_x2(repo, hw):
    dg = build_ir_from_chain_workload(deepgate_workload(repo)); plan = propose_partition_baseline(dg, hw)
    assert plan["totals"]["efpga_nodes"] == 0 and plan["totals"]["x2_nodes"] == 10 and plan["totals"]["boundary_edges"] == 0
    assert validate_partition(dg, plan, hw) == []


@pytest.mark.parametrize("label,fn,msg", [
    ("matrix op moved to eFPGA", lambda p: p["assignments"]["features_0_0"].update(target="efpga"), "matrix op on eFPGA"),
    ("non-matrix op moved to X2", lambda p: p["assignments"]["features_0_1"].update(target="x2"), "X2 only executes matrix ops"),
    ("node missing", lambda p: p["assignments"].pop("features_0_1"), "is not assigned"),
    ("unknown node", lambda p: p["assignments"].update(ghost={"target": "efpga"}), "unknown node"),
    ("wrong tile count", lambda p: p["assignments"]["features_0_0"].update(x2_tiles=9), "the mapper gives"),
    ("wrong total tiles", lambda p: p["totals"].update(x2_tiles=99), "totals.x2_tiles"),
    ("wrong boundary bytes", lambda p: p["totals"].update(boundary_bytes=1), "totals.boundary_bytes"),
    ("invalid target", lambda p: p["assignments"]["features_0_0"].update(target="gpu"), "invalid target"),
    ("input not io", lambda p: p["assignments"]["x"].update(target="efpga"), "must be 'io'"),
    ("plan for another IR", lambda p: p.update(ir_digest="0" * 64), "digest mismatch"),
])
def test_validator_rejects_every_tampered_plan(ir, hw, label, fn, msg):
    plan = copy.deepcopy(propose_partition_baseline(ir, hw)); fn(plan)
    errs = validate_partition(ir, plan, hw)
    assert any(msg in e for e in errs), (label, errs)


def test_a_different_but_legal_justification_text_is_still_accepted(ir, hw):
    plan = copy.deepcopy(propose_partition_baseline(ir, hw))
    for v in plan["assignments"].values():
        v["basis"] = "an LLM wrote this rationale"                       # free text never changes legality
    assert validate_partition(ir, plan, hw) == []


# ---------------------------------------------------------------- benchmark integration
def test_benchmark_reports_ir_and_partition_stages(tmp_path):
    from x2c.bmsemi.benchmark import run_benchmark
    rep = run_benchmark(workload_ids=["bm_compiler_v2_tiny_mobilenetv2"], report_path=str(tmp_path / "r.json"))
    by = {x["test"].split("/")[-1]: x for x in rep["results"]}
    assert by["ir"]["stage"] == "S03" and by["ir"]["passed"] and by["ir"]["details"]["nodes"] == 51
    assert by["partition"]["stage"] == "S04" and by["partition"]["passed"] and by["partition"]["details"]["x2_tiles"] == 24
    assert "plans" in rep and "ir_digest" in by["ir"]["details"]


def test_a_corrupted_graph_fails_the_ir_and_graph_stages(tmp_path):
    import shutil
    from x2c.inputs.catalog import COMPILER_V2
    from x2c.inputs.graph_check import LAYERS
    from x2c.bmsemi.benchmark import run_benchmark
    real = ReadOnlyRepo()
    for rel in ("catalog.json", GRAPH, LAYERS, f"{COMPILER_V2}/artifacts/maestro_schedule.json"):
        dst = tmp_path / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy(real.path(rel), dst)
    gp = tmp_path / GRAPH; g = json.loads(gp.read_text())
    next(n for n in g["nodes"] if n["name"] == "features_1_conv_0_0")["shape"] = [1, 8, 5, 5]
    gp.write_text(json.dumps(g))
    rep = run_benchmark(ReadOnlyRepo(tmp_path), ["bm_compiler_v2_tiny_mobilenetv2"], report_path=None)
    by = {x["test"].split("/")[-1]: x for x in rep["results"]}
    assert not by["graph"]["passed"] and not by["ir"]["passed"]
    assert any("features_1_conv_0_0" in e for e in by["ir"]["errors"])
