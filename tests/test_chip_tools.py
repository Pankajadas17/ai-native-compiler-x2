"""A4 on the BM Labs chip: verification/chip_tools.py and the three tools registered in verification/tools.py.

* chip_status reads the chip reports (recorded real ones) and computes every number itself; a report that does not exist
  is listed with the command that makes it; Fmax is judged against the project clock; missing X2 facts are listed;
* the tools reject bad arguments; the agent adds chip facts recomputed by code when a chip tool was used;
* chip_schedule and run_chip_layers on the real tiny MobileNetV2 (need BMsemi_Model: project machine only).
"""
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from verification import chip_tools as ct  # noqa: E402

FIX = ROOT / "tests" / "fixtures" / "chip_tools"


def tree(tmp_path, fmax_line=None):
    """A project tree holding the recorded reports where the chip tools look for them."""
    (tmp_path / "reports").mkdir()
    shutil.copy(FIX / "chain.json", tmp_path / "reports" / "chip_chain_report.json")
    for rel, src in (("chip_build/x2_engine2_sim/report.json", "engine2_report.json"),
                     ("chip_build/x2_layer_e2e/report.json", "layers_report.json"),
                     ("chip_build/x2_engine2/flow_report.json", "flow_report_x2_engine2.json")):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(FIX / src, tmp_path / rel)
    pnr = (FIX / "pnr_x2_engine2.log").read_text()
    (tmp_path / "chip_build/x2_engine2/pnr.log").write_text(pnr if fmax_line is None else fmax_line)
    return tmp_path


def test_status_from_recorded_reports(tmp_path):
    s = ct.chip_status(tree(tmp_path))
    assert s["project_clock_mhz"] == 14.0 and "NOT silicon" in s["evidence_level"]
    assert s["chain"]["all_pass"] and set(s["chain"]["verdicts"]) == {"relu8", "add8_sat", "requant_i32_i8"}
    d = s["designs"]["x2_engine2"]
    assert d["status"] == "ok" and d["x2_blocks"] == 2 and d["pins"] == 35 and d["meets_project_clock"] is True
    assert d["v0_audit"].split("/")[0] == d["v0_audit"].split("/")[1]
    assert s["designs"]["x2_engine2_4bit"] == {"status": "not run", "how": "python3 run_chip_flow.py --example x2_engine2_4bit   (Ubuntu terminal)"}
    e = s["two_block_engine"]
    assert e["all_pass"] and set(e["latency_cut_percent"]) == {"conv", "dw", "pw", "fc"} and e["max_model_error_percent"] < 0.1
    L = s["layers_simulated"]
    assert L["all_exact"] and L["layers"] == L["4bit_exact"] == L["1bit_simulated"] == 4 and L["latency_cut_percent_range"][0] > 40
    assert s["layers_4bit_python"]["status"] == "not run"
    assert "cell_compute_4bit" in s["x2_facts_missing"] and s["x2_facts"]["cell_bits"] == {"value": 4, "status": "decision"}


def test_nothing_run_yet(tmp_path):
    s = ct.chip_status(tmp_path)
    assert s["chain"]["status"] == "not run" and "--target chip" in s["chain"]["how"]
    assert all(v["status"] == "not run" for v in s["designs"].values())
    assert s["two_block_engine"]["how"].startswith("python3 run_x2_engine2_check.py")


def test_fmax_is_judged_against_the_project_clock(tmp_path):
    slow = "Info: Max frequency for clock 'clk': 13.90 MHz (PASS at 12.00 MHz)\n"         # nextpnr says PASS at its 12 MHz
    d = ct.chip_status(tree(tmp_path, slow))["designs"]["x2_engine2"]
    assert d["fmax_mhz"] == 13.9 and d["meets_project_clock"] is False


def test_bad_layer_positions():
    with pytest.raises(ValueError):
        ct.run_chip_layers(positions=0)


# ---------------------------------------------------------------- registered tools and the agent (project machine)
def test_tools_are_registered_and_guarded():
    pytest.importorskip("emulators.x2_adapter")
    from verification import tools
    assert {"chip_status", "chip_schedule", "run_chip_layers"} <= set(tools.TOOLS)
    assert {s["name"] for s in tools.TOOL_SCHEMAS} == set(tools.TOOLS)
    assert "positions must be between 1 and 16" in tools.call_tool("run_chip_layers", {"positions": 99})["error"]
    assert "unknown arguments" in tools.call_tool("chip_status", {"path": "/etc"})["error"]
    assert "chain" in tools.call_tool("chip_status", {})


def test_agent_adds_chip_facts_recomputed_by_code(tmp_path):
    pytest.importorskip("emulators.x2_adapter")
    from agents.llm_client import Action
    from agents.verification_agent import run_agent

    class L:
        def start(self, goal, schemas):
            return Action("tool_calls", calls=[("chip_status", {})])

        def send_tool_results(self, results):
            return Action("final", text="the chip is verified on silicon")      # an overclaim: the facts block stays code's
    out = run_agent(L(), "check the chip", log_path=str(tmp_path / "l.jsonl"))
    assert out["verified_chip_facts"]["project_clock_mhz"] == 14.0 and "NOT silicon" in out["verified_chip_facts"]["evidence_level"]


def test_real_schedule_and_layers():
    pytest.importorskip("inputs.bmsemi_source")
    try:
        s = ct.chip_schedule()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"BMsemi_Model not available: {e}")
    assert s["x2_tiles"] == 64 and s["best"]["latency_cycles"] <= s["today"]["latency_cycles"]
    assert s["best"]["latency_us"] == round(s["best"]["latency_cycles"] / 14.0, 1)
    r = ct.run_chip_layers(positions=1)
    assert r["exit_code"] == 0 and r["layers"] == r["4bit_exact"] == 17
