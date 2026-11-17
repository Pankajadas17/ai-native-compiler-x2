import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from emulators.x2_adapter import x2_vecmat
from inputs.bmsemi_source import ReadOnlyRepo, RepoAccessError
from inputs.catalog import COMPILER_V2, discover
from inputs.graph_check import GRAPH, LAYERS, validate_compiler_graph
from inputs.normalize import all_workloads, compiler_v2_workload, deepgate_workload, mnist_workload
from inputs.schemas import WorkloadSpec, validate_workload
from orchestration.benchmark import naive_core_mapper_tiles, reproduce_compiler_tiles
from verification.layer_exec import grouped_conv_golden, grouped_conv_on_x2, input_hw
from tests import faults


@pytest.fixture(scope="module")
def repo():
    return ReadOnlyRepo()


def mini_repo(tmp_path):
    """A tiny copy holding only the compiler JSON files, so tests can corrupt them without touching the real repo."""
    real = ReadOnlyRepo()
    for rel in ("catalog.json", GRAPH, LAYERS):
        dst = tmp_path / rel; dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(real.path(rel), dst)
    return ReadOnlyRepo(tmp_path)


# ---------------------------------------------------------------- read-only source
def test_paths_cannot_escape_the_repo(repo):
    with pytest.raises(RepoAccessError):
        repo.read_text("../run_agent.py")
    with pytest.raises(RepoAccessError):
        repo.path("projects/../../x")
    assert repo.exists("../run_agent.py") is False


def test_source_has_no_write_api(repo):
    forbidden = ("write", "write_text", "write_bytes", "open_write", "delete", "remove", "mkdir", "save")
    assert not [n for n in forbidden if hasattr(repo, n)]


def test_missing_repo_gives_a_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Clone it"):
        ReadOnlyRepo(tmp_path)


def test_fingerprint_ignores_windows_line_endings(tmp_path):
    (tmp_path / "catalog.json").write_text('{"operations": []}')
    (tmp_path / "a.txt").write_bytes(b"line1\nline2\n")
    (tmp_path / "b.txt").write_bytes(b"line1\r\nline2\r\n")
    r = ReadOnlyRepo(tmp_path)
    assert r.fingerprint("a.txt")["sha256_lf"] == r.fingerprint("b.txt")["sha256_lf"]


def test_assert_unchanged_detects_modification_and_deletion(tmp_path):
    (tmp_path / "catalog.json").write_text('{"operations": []}'); (tmp_path / "f.json").write_text("{}")
    r = ReadOnlyRepo(tmp_path); snap = r.snapshot(["f.json"])
    assert r.assert_unchanged(snap)
    (tmp_path / "f.json").write_text('{"changed": 1}')
    with pytest.raises(RepoAccessError, match="content changed"):
        r.assert_unchanged(snap)
    (tmp_path / "f.json").unlink()
    with pytest.raises(RepoAccessError, match="missing"):
        r.assert_unchanged(snap)


# ---------------------------------------------------------------- schemas
def test_schema_accepts_real_workloads_and_round_trips(repo):
    for w in all_workloads(repo):
        d = w.to_dict()
        assert validate_workload(d) == []
        assert WorkloadSpec.from_dict(json.loads(json.dumps(d))).to_dict() == d


@pytest.mark.parametrize("mutate,expected", [
    (lambda d: d["layers"][0].update(op="mystery"), "op must be"),
    (lambda d: d["layers"][0].update(K=-1), "K must be"),
    (lambda d: d["layers"][1].update(groups=3), "must divide"),
    (lambda d: d["layers"][1].update(groups=2), "depthwise_conv2d needs groups == C"),
    (lambda d: d["layers"][1].update(name=d["layers"][0]["name"]), "duplicate name"),
    (lambda d: d.update(pe={"rows": 0, "cols": 4}), "pe must be"),
    (lambda d: d.update(provenance={}), "provenance"),
    (lambda d: d.update(data_source="magic"), "data_source"),
    (lambda d: d.pop("layers"), "missing field"),
])
def test_schema_rejects_bad_workloads(repo, mutate, expected):
    d = copy.deepcopy(compiler_v2_workload(repo).to_dict()); mutate(d)
    assert any(expected in e for e in validate_workload(d))
    with pytest.raises(ValueError):
        WorkloadSpec.from_dict(d)


# ---------------------------------------------------------------- catalog + normalisation
def test_catalog_is_deterministic_and_complete(repo):
    a, b = discover(repo), discover(repo)
    assert a == b
    names = {o["name"] for o in a["operations"]}
    assert {"core.map_layer", "core.pe_simulate", "models.deepgate.infer", "models.mnist.edge_filter"} <= names
    kinds = {x["path"].rsplit("/", 1)[-1]: x for x in a["assets"]}
    assert kinds["compiler_graph.json"]["kind"] == "compiler_graph" and kinds["compiler_graph.json"]["sha256_lf"]
    assert kinds["tiny_mobilenetv2_digits_best.pt"]["loader"] == "none (needs torch)"
    assert {c["project"].rsplit("/", 1)[-1] for c in a["corpora"]} >= {"Deep_gate", "FP4LLM", "Yolo26_INT8_mac"}
    assert all("none" in c["loader"] for c in a["corpora"] if "FP4LLM" in c["project"] or "Yolo26" in c["project"])


def test_normalised_workloads_match_the_source_artifacts(repo):
    c = compiler_v2_workload(repo)
    assert len(c.layers) == 17 and sum(l.op == "depthwise_conv2d" for l in c.layers) == 5
    assert c.data_source == "synthetic_values_real_geometry" and c.provenance["repo_commit"]
    assert {f["path"].rsplit("/", 1)[-1] for f in c.provenance["files"]} >= {"maestro_layer_dataflow.json", "compiler_graph.json"}
    d = deepgate_workload(repo)
    assert (d.layers[0].C, d.layers[0].K) == (640, 128) and (d.layers[-1].C, d.layers[-1].K) == (128, 640) and len(d.layers) == 10
    m = mnist_workload(repo)
    assert (m.layers[0].K, m.layers[0].R, m.layers[0].padding) == (2, 5, 2)


# ---------------------------------------------------------------- S03 graph validation
def test_real_graph_is_valid(repo):
    g = validate_compiler_graph(repo)
    assert g["ok"] and g["node_count"] == 51 and g["layers_checked"] == 17


def _edit_graph(tmp_path, fn):
    r = mini_repo(tmp_path); p = r.path(GRAPH); g = json.loads(p.read_text()); fn(g); p.write_text(json.dumps(g))
    return r


def test_graph_check_finds_corruptions(tmp_path):
    cases = {
        "dangling user": (lambda g: g["nodes"][1]["users"].append("ghost_node"), "unknown user"),
        "second placeholder": (lambda g: g["nodes"][2].update(op="placeholder"), "exactly 1 placeholder"),
        "bad shape": (lambda g: next(n for n in g["nodes"] if n["op"] == "call_module").update(shape=[1, 0, 4]), "invalid shape"),
        "duplicate name": (lambda g: g["nodes"][3].update(name=g["nodes"][2]["name"]), "duplicate node names"),
        "no output": (lambda g: [n.update(op="call_function") for n in g["nodes"] if n["op"] == "output"], "no output node"),
    }
    for label, (fn, msg) in cases.items():
        d = tmp_path / label.replace(" ", "_"); d.mkdir()
        res = validate_compiler_graph(_edit_graph(d, fn))
        assert not res["ok"] and any(msg in e for e in res["errors"]), (label, res["errors"])


def test_graph_check_detects_a_cycle(tmp_path):
    def cyc(g):
        first, last = g["nodes"][0], next(n for n in g["nodes"] if n["op"] == "output")
        last["users"] = [first["name"]]
    res = validate_compiler_graph(_edit_graph(tmp_path, cyc))
    assert not res["ok"] and any("cycle" in e or "root" in e for e in res["errors"])


def test_graph_check_detects_shape_disagreement_with_maestro_table(tmp_path):
    def reshape(g):
        n = next(n for n in g["nodes"] if n["name"] == "features_0_0"); n["shape"] = [1, 9, 4, 4]
    res = validate_compiler_graph(_edit_graph(tmp_path, reshape))
    assert not res["ok"] and any("features_0_0" in e and "output_shape" in e for e in res["errors"])


# ---------------------------------------------------------------- generalised layer execution
def test_golden_conv_hand_examples():
    ones = np.ones((1, 2, 2))
    out = grouped_conv_golden(ones, np.ones((1, 1, 2, 2)), stride=1, padding=1, groups=1)
    assert out.tolist() == [[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]
    ifm = np.stack([np.ones((3, 3)), 2 * np.ones((3, 3))])                      # depthwise: channels do not mix
    dw = grouped_conv_golden(ifm, np.ones((2, 1, 3, 3)), stride=2, padding=1, groups=2)
    assert dw.shape == (2, 2, 2) and dw[0].tolist() == [[4, 4], [4, 4]] and dw[1].tolist() == [[8, 8], [8, 8]]


def test_every_real_layer_matches_golden_on_x2_with_mapper_pass_counts(repo):
    for L in compiler_v2_workload(repo).layers:
        h, w = input_hw(L); rng = np.random.default_rng(7)
        ifm = rng.integers(-3, 4, (L.C, h, w)).astype(float)
        wt = rng.integers(-3, 4, (L.K, L.C // L.groups, L.R, L.S)).astype(float)
        gold = grouped_conv_golden(ifm, wt, L.stride, L.padding, L.groups)
        out, passes, exp = grouped_conv_on_x2(ifm, wt, L.stride, L.padding, L.groups, 64, L.extra["tile_width"])
        assert gold.shape == (L.K, L.Y, L.X) and np.array_equal(gold, out), L.name
        assert passes == exp, L.name


def test_layer_execution_catches_a_wrong_mapping_and_a_wrong_x2(repo):
    L = next(l for l in compiler_v2_workload(repo).layers if l.op == "depthwise_conv2d")
    h, w = input_hw(L); rng = np.random.default_rng(3)
    ifm = rng.integers(-3, 4, (L.C, h, w)).astype(float); wt = rng.integers(1, 4, (L.K, 1, L.R, L.S)).astype(float)
    gold = grouped_conv_golden(ifm, wt, L.stride, L.padding, L.groups)
    bad, passes, exp = grouped_conv_on_x2(ifm, wt, L.stride, L.padding, L.groups, 64, 16, mapping_transform=faults.drop_last_fold)
    assert not np.array_equal(gold, bad) and passes < exp
    neg = lambda W, v: (-x2_vecmat(W, v)[0], 0)
    out, _, _ = grouped_conv_on_x2(ifm, wt, L.stride, L.padding, L.groups, 64, 16, vecmat=neg)
    assert np.array_equal(out, -gold)


# ---------------------------------------------------------------- S04 mapping consistency
def test_compiler_tile_counts_are_reproduced_and_the_naive_mapper_differs(repo):
    wl = compiler_v2_workload(repo)
    assert all(reproduce_compiler_tiles(L) == L.extra["compiler_expected_tiles"] for L in wl.layers)      # 17 / 17
    differ = {L.name for L in wl.layers if naive_core_mapper_tiles(L, wl.pe) != L.extra["compiler_expected_tiles"]}
    assert {L.name for L in wl.layers if L.op == "depthwise_conv2d"} <= differ and len(differ) == 9
