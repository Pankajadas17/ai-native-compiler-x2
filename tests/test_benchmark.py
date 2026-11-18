import json
import os

import pytest

from x2c.agents.debug_agent import run_debug_agent
from x2c.agents.llm_client import Action, MockDebugLLM
from x2c.agents.verification_agent import run_agent
from x2c.bmsemi.x2_adapter import x2_vecmat
from x2c.inputs.bmsemi_source import ReadOnlyRepo
from x2c.bmsemi.benchmark import STAGE_STATUS, run_benchmark
from x2c.verification import debug_tools, tools
from tests import faults

CV2 = "bm_compiler_v2_tiny_mobilenetv2"


@pytest.fixture(scope="module")
def good_report(tmp_path_factory):
    return run_benchmark(workload_ids=[CV2], report_path=str(tmp_path_factory.mktemp("b") / "r.json"))


def test_compiler_workload_passes_all_stages_it_covers(good_report):
    r = good_report
    assert r["passed"] == r["total"] == 37                         # graph + IR + partition + 17 tile checks + 17 layers on X2
    assert {x["stage"] for x in r["results"]} == {"S02", "S03", "S04"}
    assert r["repo_read_only_check"] == {"consumed_files_unchanged": True, "no_new_pycache_in_repo": True}
    assert [f["id"] for f in r["findings"]] == ["F2_boundary_traffic_baseline", "F1_naive_core_mapper_vs_compiler"] and len(r["findings"][1]["layers"]) == 9
    assert r["plans"][CV2]["totals"]["x2_tiles"] == 24 and r["plans"][CV2]["totals"]["boundary_bytes"] == 2480


def test_stage_table_is_honest():
    st = {s["id"]: s["status"] for s in STAGE_STATUS}
    assert st["S01"] == st["S02"] == st["S10"] == "implemented"         # S10: every chip-flow bitstream is audited
    assert all(st[k] == "partial" for k in ("S03", "S04", "S05", "S06", "S07-S09", "S11"))
    assert st["S12"] == "not_implemented"
    assert set(st.values()) <= {"implemented", "partial", "not_implemented"}


def test_stage_table_points_at_code_that_exists_and_says_what_is_missing():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for s in STAGE_STATUS:
        if s["status"] != "not_implemented":
            assert s.get("where"), s["id"]
            for p in s["where"]:
                assert (root / p).is_file(), (s["id"], p)
        if s["status"] != "implemented":
            assert s.get("missing"), s["id"]                            # a partial stage must say what it lacks


def test_report_is_deterministic_and_auditable(good_report, tmp_path):
    again = run_benchmark(workload_ids=[CV2], report_path=str(tmp_path / "r2.json"))
    assert again["result_digest_sha256"] == good_report["result_digest_sha256"]
    src = good_report["source"]
    assert src["repo_commit"] and all(f["path"] and f["size"] > 0 for f in src["consumed_files"])
    assert any(f["sha256_lf"] for f in src["consumed_files"])
    assert good_report["level"].endswith("NOT silicon)")


def test_run_never_modifies_the_repository(tmp_path):
    repo = ReadOnlyRepo()
    def state():
        return {p: (repo.path(p).stat().st_size, repo.path(p).stat().st_mtime_ns)
                for d in ("projects/2_building_blocks/Neuromorphic_compiler_V2", "projects/3_example_models/Deep_gate") for p in repo.walk(d)}
    before, caches = state(), repo.pycache_dirs()
    run_benchmark(repo, workload_ids=[CV2], report_path=str(tmp_path / "r.json"))
    assert state() == before and repo.pycache_dirs() == caches


def test_unknown_workload_is_rejected():
    with pytest.raises(ValueError, match="unknown workload ids"):
        run_benchmark(workload_ids=["nope"], report_path=None)


# ---------------------------------------------------------------- failures flow into the existing agents
def test_wrong_x2_is_detected_and_the_debug_agent_diagnoses_it():
    flipped = lambda W, v: (-x2_vecmat(W, v)[0], 0)
    rep = run_benchmark(workload_ids=[CV2], vecmat=flipped, report_path="reports/_bench_flip.json")
    try:
        assert rep["passed"] < rep["total"] and any(not r["passed"] for r in rep["results"] if r["stage"] == "S02")
        d = debug_tools.diagnose_report("reports/_bench_flip.json")
        assert d["report_kind"].startswith("workload_report") and "sign_inverted" in d["failure_categories"]
        out = run_debug_agent(MockDebugLLM(), "reports/_bench_flip.json", log_path="reports/_bench_flip_log.jsonl")
        assert out["final_cause"] == "output_sign_inverted"
    finally:
        for p in ("reports/_bench_flip.json", "reports/_bench_flip_log.jsonl"):
            if os.path.exists(p): os.remove(p)


def test_wrong_mapping_is_detected_as_a_dropped_fold():
    rep = run_benchmark(workload_ids=[CV2], mapping_transform=faults.drop_last_fold, report_path="reports/_bench_fold.json")
    try:
        assert rep["passed"] < rep["total"]
        assert "mapper_dropped_fold" in debug_tools.diagnose_report("reports/_bench_fold.json")["rule_based_candidates"]
    finally:
        os.remove("reports/_bench_fold.json")


def test_compiler_tile_mismatch_would_be_reported(monkeypatch, tmp_path):
    import x2c.bmsemi.benchmark as b
    monkeypatch.setattr(b, "reproduce_compiler_tiles", lambda L, rows=64: 99)
    rep = b.run_benchmark(workload_ids=[CV2], report_path=str(tmp_path / "r.json"))
    bad = [r for r in rep["results"] if r["stage"] == "S04" and not r["passed"]]
    assert len(bad) == 17 and bad[0]["reason"] == "tile_count_mismatch"


# ---------------------------------------------------------------- tools for the agents
def test_list_workloads_tool():
    out = tools.call_tool("list_workloads", {})
    assert {w["id"] for w in out["workloads"]} == {CV2, "bm_deepgate_ad01_int8", "bm_mnist_edge_filter"}
    assert out["repo_commit"] and {s["id"]: s["status"] for s in out["stage_status"]}["S05"] == "partial"


def test_run_benchmark_tool_is_safe_and_works(tmp_path):
    assert "error" in tools.call_tool("run_benchmark", {"workload_id": "../../etc"})
    assert "error" in tools.call_tool("run_benchmark", {"report_path": "../x.json"})
    assert "error" in tools.call_tool("run_benchmark", {"vecmat": "evil"})
    assert "error" in tools.call_tool("run_benchmark", {"mapping_transform": "evil"})
    out = tools.call_tool("run_benchmark", {"workload_id": CV2, "report_path": "reports/_bench_tool.json"})
    try:
        assert out["total"] == out["passed"] == 37 and out["failed"] == [] and out["findings"] == ["F2_boundary_traffic_baseline", "F1_naive_core_mapper_vs_compiler"]
    finally:
        os.remove("reports/_bench_tool.json")


def test_verification_agent_reports_benchmark_facts(tmp_path):
    class L:
        def start(self, g, s, system_prompt=None): return Action("tool_calls", calls=[("run_benchmark", {"workload_id": CV2, "report_path": "reports/_bench_agent.json"})])
        def send_tool_results(self, r): return Action("final", text="done")
    try:
        out = run_agent(L(), "verify", log_path=str(tmp_path / "l.jsonl"))
        assert out["verified_benchmark_facts"]["total"] == 37 and out["verified_benchmark_facts"]["failed"] == 0
    finally:
        if os.path.exists("reports/_bench_agent.json"): os.remove("reports/_bench_agent.json")


@pytest.mark.slow
def test_full_benchmark_all_three_workloads(tmp_path):
    rep = run_benchmark(report_path=str(tmp_path / "full.json"))
    assert rep["passed"] == rep["total"] == 44 and len(rep["workloads"]) == 3


# ---------------------------------------------------------------- digest must not depend on the OS line endings
def _mini_repo(tmp_path, crlf, mutate=None):
    import shutil
    from x2c.inputs.graph_check import GRAPH, LAYERS
    from x2c.inputs.catalog import COMPILER_V2
    real = ReadOnlyRepo()
    for rel in ("catalog.json", GRAPH, LAYERS, f"{COMPILER_V2}/artifacts/maestro_schedule.json"):
        data = real.path(rel).read_bytes().replace(b"\r\n", b"\n")
        if mutate and rel == LAYERS:
            data = mutate(data)
        dst = tmp_path / rel; dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data.replace(b"\n", b"\r\n") if crlf else data)
    return ReadOnlyRepo(tmp_path)


def test_digest_is_identical_for_lf_and_crlf_checkouts(tmp_path):
    (tmp_path / "lf").mkdir(); (tmp_path / "crlf").mkdir()
    a = run_benchmark(_mini_repo(tmp_path / "lf", False), [CV2], report_path=None)
    b = run_benchmark(_mini_repo(tmp_path / "crlf", True), [CV2], report_path=None)
    sizes = lambda r: [f["size"] for f in r["source"]["consumed_files"]]
    assert sizes(a) != sizes(b)                                         # the on-disk bytes really differ ...
    assert a["results"] == b["results"] and a["result_digest_sha256"] == b["result_digest_sha256"]   # ... the audit identity does not


def test_digest_changes_when_the_source_content_changes(tmp_path):
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    base = run_benchmark(_mini_repo(tmp_path / "a", False), [CV2], report_path=None)
    changed = run_benchmark(_mini_repo(tmp_path / "b", False, mutate=lambda d: d.replace(b'"expected_tile_count": 1', b'"expected_tile_count": 1 ', 1)),
                            [CV2], report_path=None)
    assert changed["result_digest_sha256"] != base["result_digest_sha256"]
