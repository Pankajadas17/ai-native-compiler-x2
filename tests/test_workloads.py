import pytest
from x2c.verification import workloads, tools
from x2c.verification.failure_classifier import classify_failure
from tests import faults


def test_all_workloads_pass_on_real_x2():
    res = workloads.run_workload_suite()
    assert [r["test"] for r in res if not r["passed"]] == []
    assert all(r["mapping_check"]["ok"] for r in res)           # X2 passes == mapper mvm_ops


def test_pass_count_matches_mapper_for_known_case():
    r = workloads.run_workload("canvas_64x32_geometry")
    assert r["mapping_check"] == {"passes_executed": 40, "mvm_ops_from_mapper": 40, "ok": True}   # 10 tiles x 2x2


def test_dropped_fold_is_caught():
    r = workloads.run_workload("conv3x3_multi_fold", mapping_transform=faults.drop_last_fold)
    assert r["passed"] is False and classify_failure(r)["category"] != "none"


def test_dropped_output_channel_is_localized():
    r = workloads.run_workload("conv3x3_multi_fold", mapping_transform=faults.drop_last_output_channel)
    d = classify_failure(r)
    assert r["passed"] is False and d["category"] == "localized_mismatch" and "[7]" in d["evidence"]


def test_mapping_inconsistency_classification():
    fake = {"passed": False, "reason": "pass_count_mismatch", "mapping_check": {"ok": False}}
    assert classify_failure(fake)["category"] == "mapping_inconsistency"


def test_tool_and_agent_entrypoint(tmp_path):
    rp = str(tmp_path / "w.json")
    out = tools.run_workloads(report_path=rp)
    assert out["total"] == 5 and out["passed"] == 5
    assert tools.summarize_report(rp)["failed"] == 0
    assert "error" in tools.call_tool("run_workloads", {"report_path": "../x.json"})


def test_golden_conv_matches_hand_example():
    import numpy as np
    ifm = np.arange(9, dtype=float).reshape(1, 3, 3); w = np.ones((1, 1, 2, 2))
    assert workloads.conv_golden(ifm, w).tolist() == [[[8.0, 12.0], [20.0, 24.0]]]
