import numpy as np
import pytest
from x2c.verification import model_workloads as mw, tools
from x2c.verification.failure_classifier import classify_failure
from tests import faults

pytestmark = pytest.mark.skipif(not mw.data_available(), reason="BMsemi_Model example data not present")


@pytest.mark.parametrize("digit", [5, 9])
def test_mnist_crop_matches_repo_operation(digit):
    r = mw.run_mnist(digit)
    assert r["passed"], r.get("reason")
    mc = r["mapping_check"]
    assert mc["ok"] and mc["independent_reference_agrees"] and mc["repo_reported_pe_tiles"] == mc["mapper_pe_tiles"] == 1
    assert np.count_nonzero(r["expected"]) > 0                      # the crop really contains edges


@pytest.mark.slow
def test_mnist_full_map_digit5():
    r = mw.run_mnist(5, full=True)
    assert r["passed"] and r["mapping_check"]["passes_executed"] == 784


@pytest.mark.slow
def test_deepgate_all_ten_layers_exact_on_x2():
    r = mw.run_deepgate(0)
    assert r["passed"] and r["first_failing_layer"] is None
    assert len(r["layers"]) == 10 and r["total_x2_passes"] == 264
    assert all(L["accumulators_exact"] and L["output_matches_repo"] for L in r["layers"])
    assert r["reference_max_abs_err"] < 0.02                         # same acceptance rule the repo uses


def test_deepgate_fault_dropped_fold_points_at_first_layer():
    r = mw.run_deepgate(0, mapping_transform=faults.drop_last_fold)
    d = classify_failure(r)
    assert r["passed"] is False and r["first_failing_layer"] == 0
    assert "first failing layer: 0" in d["evidence"]


def test_deepgate_fault_sign_flipped_x2():
    def flipped(W, v):
        from x2c.bmsemi.x2_adapter import x2_vecmat
        o, c = x2_vecmat(W, v); return -o, c
    r = mw.run_deepgate(0, vecmat=flipped)
    assert r["passed"] is False and r["first_failing_layer"] == 0


def test_mnist_fault_dropped_negative_column():
    r = mw.run_mnist(5, mapping_transform=faults.drop_last_output_channel)
    assert r["passed"] is False and classify_failure(r)["category"] != "none"


@pytest.mark.slow
def test_tool_and_path_safety(tmp_path):
    out = tools.run_model_workloads(report_path=str(tmp_path / "m.json"))
    assert out["total"] == 3 and out["passed"] == 3
    assert tools.summarize_report(str(tmp_path / "m.json"))["failed"] == 0
    assert "error" in tools.call_tool("run_model_workloads", {"report_path": "../m.json"})


@pytest.mark.slow
def test_agent_reports_model_workload_facts(tmp_path):
    from x2c.agents.llm_client import Action
    from x2c.agents.verification_agent import run_agent
    class L:
        def start(self, g, s): return Action("tool_calls", calls=[("run_model_workloads", {})])
        def send_tool_results(self, r): return Action("final", text="done")
    out = run_agent(L(), "verify", log_path=str(tmp_path / "l.jsonl"))
    assert out["verified_model_workload_facts"]["total"] == 3 and out["verified_model_workload_facts"]["failed"] == 0
