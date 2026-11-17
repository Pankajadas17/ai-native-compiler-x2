"""The X2 tile schedule (compiler/x2_schedule.py) and A1 choosing it (agents/mapping_agent.py).

* the cost model's constants re-fit from the four recorded X2 engine simulations (flow step 4);
* the validator rejects every kind of illegal schedule and computes a latency checked by hand;
* code's best schedule is never worse than any candidate, and on a graph with a 1-tile layer it is NOT today's choice;
* A1: an LLM schedule is kept only if legal and at least as fast as code's; the final plan always passes validate_partition;
* the real tiny MobileNetV2 graph (needs BMsemi_Model, so it runs on the project machine only).
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents.llm_client import Action  # noqa: E402
from agents.mapping_agent import _Session, run_mapping_agent  # noqa: E402
from compiler import x2_schedule as xs  # noqa: E402
from compiler.partition import chip_view, load_hardware, propose_partition_baseline, validate_partition  # noqa: E402
from x2_schedule_graphs import mini, with_small_layer  # noqa: E402

FIX = ROOT / "tests" / "fixtures" / "x2_engine"


@pytest.fixture(scope="module")
def hw():
    h = load_hardware()
    if "rows" not in h.get("x2", {}):
        pytest.skip("hardware_config.json has no chip X2")
    return h


def plan_of(g, hw):
    return propose_partition_baseline(g, hw)


# ------------------------------------------------------------ calibration
def test_cost_model_refits_from_the_engine_simulations():
    import run_x2_engine_check as ec
    hw1 = load_hardware()
    hw1 = copy.deepcopy(hw1); hw1["x2"]["cell_bits"] = 1
    rows, meas = [], []
    for k, pkg, X, ref, cfg in ec.cases(hw1, 2, 2026):
        text, _, _ = ec.script(pkg, X, cfg)
        b = [l.split() for l in text.splitlines() if l.startswith("B ")]
        prog = sum(1 for f in b if f[1] in ("0", "3"))
        beats = sum(1 for f in b if f[1] == "2")
        mvms = sum(1 for f in b if f[1] == "2" and f[5] == "1")
        rows.append([prog, mvms, beats])
        meas.append(int((FIX / f"engine_{k}.log").read_text().split("DONE")[1].split()[2]))
    coef, *_ = np.linalg.lstsq(np.array(rows, float), np.array(meas, float), rcond=None)
    assert len(rows) == 4 and abs(np.array(rows, float) @ coef - meas).max() < 5
    assert abs(coef[0] - xs.PROGRAM_CMD) < 5 and abs(coef[1] - xs.MVM) < 5 and abs(coef[2] - xs.BEAT) < 0.5


def test_tile_costs_come_from_the_placement(hw):
    chip = plan_of(mini(), hw)["chip"]
    costs = xs.tile_costs(chip, hw)
    for k, v in chip["layers"].items():
        assert sum(c["cells"] for c in costs[k]) == v["cells"]
        assert all(c["program"] == c["cells"] * xs.PROGRAM_CMD and c["compute"] == v["mvms_per_tile"] * (xs.MVM + xs.BEAT * c["rows"])
                   for c in costs[k])


def test_dependencies_look_through_non_x2_nodes(hw):
    g = mini()
    assert xs.x2_deps(g, plan_of(g, hw)["chip"]["layers"]) == {"c0": [], "dw1": ["c0"], "pw1": ["dw1"], "pw2": ["c0", "pw1"], "fc": ["pw2"]}


# ------------------------------------------------------------ validator
def test_latency_by_hand(hw):
    """Two streamed layers c0 (2 tiles) then dw1 on the mini graph, one block: nothing overlaps, latency = the plain sum."""
    g = mini(); chip = plan_of(g, hw)["chip"]
    costs = xs.tile_costs(chip, hw)
    errs, m = xs.validate_schedule(g, chip, hw, xs.materialize(g, chip, hw, [], "single_block"))
    total = sum(c["program"] + c["compute"] for k in costs for c in costs[k])
    assert errs == [] and m["latency_cycles"] == total and m["hidden_by_overlap_cycles"] == 0 and m["runs_on_built_x2_engine"]
    # two blocks: c0 tile 0 on block 0, tile 1 on block 1 -> both program at t=0 and compute in parallel
    sched = xs.materialize(g, chip, hw, [], "round_robin")
    _, m2 = xs.validate_schedule(g, chip, hw, sched)
    c = costs["c0"]
    assert sched["tiles"][:2] == [["c0", 0, 0], ["c0", 1, 1]]
    assert m2["latency_cycles"] < total and m2["blocks_used"] == 2 and not m2["runs_on_built_x2_engine"]
    assert max(c[0]["program"] + c[0]["compute"], c[1]["program"] + c[1]["compute"]) <= m2["latency_cycles"]


def _good(g, hw):
    chip = plan_of(g, hw)["chip"]
    return chip, xs.materialize(g, chip, hw, [], "earliest_free")


@pytest.mark.parametrize("label,edit,msg", [
    ("not an object", lambda s: "x", "must be an object"),
    ("resident not a list", lambda s: dict(s, resident="c0"), "resident must be a list"),
    ("tiles not a list", lambda s: dict(s, tiles=None), "schedule.tiles must be a list"),
    ("duplicate tile", lambda s: dict(s, tiles=s["tiles"] + [s["tiles"][0]]), "scheduled twice"),
    ("missing tile", lambda s: dict(s, tiles=s["tiles"][:-1]), "never scheduled"),
    ("no such block", lambda s: dict(s, tiles=[[l, t, 7] for l, t, b in s["tiles"]]), "does not exist"),
    ("no such tile", lambda s: dict(s, tiles=s["tiles"] + [["c0", 99, 0]]), "no tile 99"),
    ("bad entry", lambda s: dict(s, tiles=s["tiles"] + [["c0", "0", True]]), "must be [layer, tile_index, block]"),
    ("dependency order", lambda s: dict(s, tiles=s["tiles"][2:] + s["tiles"][:2]), "before every tile"),
    ("resident differs from the plan", lambda s: dict(s, resident=["c0"]), "differs from the plan"),
])
def test_validator_rejects(hw, label, edit, msg):
    g = mini(); chip, s = _good(g, hw)
    errs, m = xs.validate_schedule(g, chip, hw, edit(copy.deepcopy(s)))
    assert m is None and any(msg in e for e in errs), (label, errs)


def test_resident_blocks_cannot_also_stream(hw):
    g = with_small_layer(); plan = plan_of(g, hw)
    chip = chip_view(g, hw, plan["assignments"], ["c0"])
    s = xs.materialize(g, chip, hw, ["c0"], "single_block")
    assert xs.validate_schedule(g, chip, hw, s)[0] == []
    s["tiles"] = [[l, t, 0] for l, t, b in s["tiles"]]                      # everything on the resident block
    assert any("resident tile and are also used" in e for e in xs.validate_schedule(g, chip, hw, s)[0])


def test_resident_filling_every_block_while_streaming_is_illegal(hw):
    g = with_small_layer(); plan = plan_of(g, hw)
    chip = chip_view(g, hw, plan["assignments"], ["c0", "pw1"])
    s = {"resident": ["c0", "pw1"], "tiles": [[l, t, (0 if l == "c0" else 1 if l == "pw1" else 1)] for l, t, b in
                                              xs.materialize(g, chip, hw, [], "single_block")["tiles"]]}
    errs = xs.validate_schedule(g, chip, hw, s)[0]
    assert any("fill all 2 blocks" in e for e in errs) or any("also used" in e for e in errs)


# ------------------------------------------------------------ code's own search
@pytest.mark.parametrize("make", [mini, with_small_layer])
def test_best_is_never_worse_than_any_candidate(hw, make):
    g = make(); chip = plan_of(g, hw)["chip"]
    best, bm, tried = xs.best_schedule(g, chip, hw)
    lat = []
    for res in xs.resident_options(chip, hw):
        for pol in xs.POLICIES:
            e, m = xs.validate_schedule(g, dict(chip, resident=res), hw, xs.materialize(g, chip, hw, res, pol))
            assert e == []
            lat.append(m["latency_cycles"])
    assert tried == len(lat) and bm["latency_cycles"] == min(lat)


def test_with_a_one_tile_layer_streaming_both_blocks_beats_keeping_it_resident(hw):
    g = with_small_layer(); chip = plan_of(g, hw)["chip"]
    assert chip["resident"] == ["c0"] and xs.resident_options(chip, hw) == [[], ["c0"], ["pw1"]]
    _, today = xs.validate_schedule(g, chip, hw, xs.materialize(g, chip, hw, ["c0"], "single_block"))
    best, bm, _ = xs.best_schedule(g, chip, hw)
    assert best["resident"] == [] and bm["latency_cycles"] < today["latency_cycles"]
    assert bm["program_cycles_per_inference_upper"] > today["program_cycles_per_inference_upper"]     # more programming, less latency
    assert not today["runs_on_built_x2_engine"]                     # even today's plan needs both blocks (one holds c0)


# ------------------------------------------------------------ A1
class Scripted:
    def __init__(self, sched):
        self.sched = sched

    def start(self, goal, schemas, system_prompt=None):
        self.prompt, self.schemas = system_prompt, schemas
        return Action("tool_calls", calls=[("describe_graph", {})])

    def send_tool_results(self, results, force_final=False):
        name, res = results[0]
        if name == "describe_graph":
            return Action("tool_calls", calls=[("propose_assignment", {"targets": {n["name"]: n["allowed_targets"][0] for n in res["nodes"]}})])
        if name == "propose_assignment" and self.sched is not None:
            return Action("tool_calls", calls=[("describe_schedule", {})])
        if name == "describe_schedule":
            self.described = res
            return Action("tool_calls", calls=[("propose_schedule", self.sched)])
        if name == "propose_schedule":
            self.verdict = res
        return Action("final", text="done")


@pytest.mark.parametrize("sched,source", [
    (None, "code_best (llm proposed no schedule)"),
    ({"resident": ["c0"], "policy": "single_block"}, "code_best (llm schedule legal but"),
    ({"resident": [], "policy": "round_robin"}, "llm_validated"),
    ({"resident": ["ghost"]}, "code_best (llm schedule rejected)"),
    ({"resident": [], "tiles": [["c0", 0, 0]]}, "code_best (llm schedule rejected)"),
])
def test_a1_keeps_only_a_legal_schedule_that_is_as_fast_as_code(hw, tmp_path, sched, source):
    g = with_small_layer()
    llm = Scripted(sched)
    out = run_mapping_agent(llm, g, hw, log_path=str(tmp_path / "l.jsonl"))
    s = out["schedule"]
    assert s["source"].startswith(source), s["source"]
    assert validate_partition(g, out["final_plan"], hw) == []
    assert out["final_plan"]["chip"]["resident"] == s["resident"] == []            # the plan follows the chosen schedule
    assert s["metrics"]["latency_cycles"] == s["code_best"]["metrics"]["latency_cycles"] < s["today"]["metrics"]["latency_cycles"]
    assert s["latency_reduction_vs_today_percent"] > 20 and "NOT silicon" in s["evidence_level"] and s["assumptions"]
    assert "describe_schedule" in llm.prompt and any(x["name"] == "propose_schedule" for x in llm.schemas)
    if sched is not None:
        assert "today" in llm.described and all("tiles" in L for L in llm.described["layers"])


def test_describe_schedule_does_not_leak_code_best(hw, tmp_path):
    llm = Scripted({"resident": [], "policy": "round_robin"})
    run_mapping_agent(llm, with_small_layer(), hw, log_path=str(tmp_path / "l.jsonl"))
    assert "best" not in str(sorted(llm.described))


def test_schedule_tools_need_a_legal_assignment_and_a_chip(hw):
    s = _Session(mini(), hw)
    assert "propose a legal assignment first" in s.call("describe_schedule", {})["error"]
    assert "propose a legal assignment first" in s.call("propose_schedule", {"resident": []})["error"]
    assert "unknown arguments" in s.call("propose_schedule", {"resident": [], "rm": 1})["error"]
    old = copy.deepcopy(hw)
    for k in ("rows",):
        old["x2"].pop(k)
    s2 = _Session(mini(), old)
    assert "no chip X2" in s2.call("describe_schedule", {})["error"]


def test_real_tiny_mobilenetv2(hw, tmp_path):
    pytest.importorskip("inputs.bmsemi_source")
    from agents.llm_client import MockMappingLLM
    from compiler.ir_builder import build_ir_from_compiler_graph
    from inputs.bmsemi_source import ReadOnlyRepo
    from inputs.normalize import compiler_v2_workload
    try:
        repo = ReadOnlyRepo()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"BMsemi_Model not available: {e}")
    g = build_ir_from_compiler_graph(repo, compiler_v2_workload(repo))
    out = run_mapping_agent(MockMappingLLM(), g, hw, log_path=str(tmp_path / "l.jsonl"))
    s = out["schedule"]
    assert validate_partition(g, out["final_plan"], hw) == [] and out["final_plan"]["totals"]["x2_tiles"] == 24
    assert s["metrics"]["tiles"] == out["final_plan"]["chip"]["totals"]["tiles"]
    assert s["metrics"]["latency_cycles"] <= s["today"]["metrics"]["latency_cycles"]
    assert s["code_best"]["metrics"]["latency_cycles"] == s["metrics"]["latency_cycles"]
