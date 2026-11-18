"""Phase 3b: chip view of the partition (32 x 32 X2, 1-bit cells, 2 blocks) and its independent validation.

Runs on a small synthetic graph shaped like the BMsemi tiny MobileNetV2 (conv, depthwise, pointwise, linear),
so it needs neither BMsemi_Model nor an LLM. The real 51-node graph is covered by test_ir_partition.py and the
benchmark, whose compiler-view numbers (24 tiles) are unchanged.
"""
import copy
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from x2c.compiler import chip_x2  # noqa: E402
from x2c.compiler.ir import IRGraph, IRNode, TensorType  # noqa: E402
from x2c.compiler.partition import (has_chip, load_hardware, plan_from_targets, propose_partition_baseline,  # noqa: E402
                                validate_partition)

SPEC = ROOT / "specifications"


@pytest.fixture(scope="module")
def hw_file():
    hw = load_hardware()
    if not has_chip(hw):
        pytest.skip("hardware_config.json is the old 0.1 file: run x2c chip config")
    return hw


def with_cells(hw, bits):
    h = copy.deepcopy(hw); h["x2"]["cell_bits"] = bits
    return h


@pytest.fixture(scope="module", params=[4, 1], ids=["4bit_cells", "1bit_cells"])
def hw(request, hw_file):
    """Every chip test runs with the configured 4-bit cells AND the chip model's 1-bit cells."""
    return with_cells(hw_file, request.param)


# expected numbers on the synthetic graph for each cell size
EXP = {4: {"cpw": 4, "tiles": [1, 3, 2, 2], "total": 8, "resident": ["conv"], "conv_cols_tiles": 1, "low_use": ["dw"]},
       1: {"cpw": 16, "tiles": [4, 4, 8, 5], "total": 21, "resident": [], "conv_cols_tiles": 4, "low_use": []}}


def _m(name, op, inp, shape, K, C, R=1, S=1, groups=1, tile_width=16):
    patch = (C // groups) * R * S
    exp = -(-K // tile_width) * -(-patch // 64)                     # compiler view: 64-row geometry
    return IRNode(name, op, [inp], TensorType(shape), {"K": K, "C": C, "R": R, "S": S, "groups": groups,
                                                       "tile_width": tile_width, "compiler_expected_tiles": exp})


def graph():
    n = [IRNode("x", "input", [], TensorType((1, 3, 32, 32))),
         _m("conv", "conv2d", "x", (1, 8, 16, 16), K=8, C=3, R=3, S=3),
         IRNode("act0", "shape_preserving", ["conv"], TensorType((1, 8, 16, 16))),
         _m("dw", "depthwise_conv2d", "act0", (1, 8, 16, 16), K=8, C=8, R=3, S=3, groups=8),
         _m("pw", "pointwise_conv2d", "dw", (1, 16, 16, 16), K=16, C=8),
         IRNode("pool", "pool", ["pw"], TensorType((1, 16, 1, 1))),
         IRNode("flatten", "flatten", ["pool"], TensorType((1, 16))),
         _m("fc", "linear", "flatten", (1, 10), K=10, C=16),
         IRNode("out", "output", ["fc"], TensorType((1, 10)))]
    return IRGraph("synthetic_tiny_mbv2", n, {})


# ---------------------------------------------------------------- generated hardware config
def test_hardware_config_comes_from_the_chip_profile(hw_file):
    hw = hw_file
    prof = json.loads((SPEC / "chip_profile.json").read_text())
    assert (hw["x2"]["rows"], hw["x2"]["cols"], hw["x2"]["blocks"]) == (prof["x2"]["rows"], prof["x2"]["cols"], prof["x2"]["count"]) == (32, 32, 2)
    assert hw["x2"]["cell_bits"] == 4 and "DECISION" in hw["x2"]["basis"]["cell_bits"] and "1-bit" in hw["x2"]["basis"]["cell_bits"]
    assert hw["x2"]["input_bits"] == 8 and hw["x2"]["result_count_bits"] == 14
    assert hw["x2"]["timing_cycles"] == {"read": 160, "program": 360, "compute": 450}
    assert (hw["efpga"]["luts"], hw["efpga"]["macc_blocks"], hw["efpga"]["io_pins"]) == (2160, 18, 36)
    assert hw["compiler_v2"] == {**hw["compiler_v2"], "pe_rows": 64, "default_tile_cols": 32}
    assert "deep_gate/Crossbar_array.py" in hw["weights"]["basis"] and "cim32x32.sv" in hw["x2"]["basis"]["cell_bits"]


def test_hardware_config_is_current():
    import x2c.commands.chip_config as run_hardware_config
    assert run_hardware_config.main(["--check"]) == 0


def test_generator_can_model_the_chips_1bit_cells():
    import x2c.commands.chip_config as run_hardware_config
    prof = json.loads((SPEC / "chip_profile.json").read_text())
    one = run_hardware_config.build(prof, cell_bits=1)
    assert one["x2"]["cell_bits"] == 1 and "DECISION" not in one["x2"]["basis"]["cell_bits"]
    assert "16 cells per weight" in one["weights"]["basis"]


# ---------------------------------------------------------------- chip maths
def test_cells_per_weight(hw):
    # 4-bit cells: 2 signs x 2 banks (DeepGate) = 4.  1-bit cells: 2 signs x 8 bit planes = 16.
    assert chip_x2.cells_per_weight(hw) == EXP[hw["x2"]["cell_bits"]]["cpw"]


def test_known_layer_tiles(hw):
    g = graph().by_name()
    # conv: 27 taps -> 1 row fold; 8 outputs x 16 cells = 128 columns -> 4 tiles
    # dw: 8 groups of 9 taps x 16 columns; block-diagonal 2 per tile (32 cols / 16) -> 4 tiles
    # pw: 8 taps; 16 x 16 = 256 columns -> 8 tiles.  fc: 16 taps; 10 x 16 = 160 columns -> 5 tiles
    # 4-bit: conv 32 cols -> 1; dw 3 groups per tile -> 3; pw 64 cols -> 2; fc 40 cols -> 2
    assert [chip_x2.layer_tiles(g[k].attrs, hw) for k in ("conv", "dw", "pw", "fc")] == EXP[hw["x2"]["cell_bits"]]["tiles"]


def test_placements_are_valid_on_random_layers(hw):
    rnd = random.Random(13)
    for _ in range(400):
        groups = rnd.choice([1, 1, 2, 4, 8, 16, 32])
        a = {"K": groups * rnd.randint(1, 6), "C": groups * rnd.randint(1, 12), "R": rnd.choice([1, 3, 5]),
             "S": rnd.choice([1, 3]), "groups": groups}
        p = chip_x2.place_groups(a, hw)
        assert chip_x2.check_placement(a, hw, p) == [], a
        assert len(p) >= chip_x2.area_lower_bound(a, hw)
        if groups == 1:                                               # ungrouped: exactly folds x chunks
            g, taps, outs = chip_x2.layer_shape(a)
            assert len(p) == -(-taps // 32) * -(-(outs * chip_x2.cells_per_weight(hw)) // 32)


@pytest.mark.parametrize("label,mutate,msg", [
    ("block shifted onto another", lambda p: p[0][1].__setitem__(3, 0), "share rows"),
    ("block dropped", lambda p: p[0].pop(), "not covered exactly once"),
    ("block outside array", lambda p: p[0][0].__setitem__(3, 30), "outside the 32 x 32"),
    ("garbage", lambda p: p.__setitem__(0, "x"), "not covered exactly once"),
])
def test_check_placement_catches_bad_placements(hw, label, mutate, msg):
    a = graph().by_name()["dw"].attrs                                 # 2 groups per tile, so tiles hold 2 blocks
    p = copy.deepcopy(chip_x2.place_groups(a, hw)); mutate(p)
    errs = chip_x2.check_placement(a, hw, p)
    assert any(msg in e for e in errs), (label, errs)


def test_resident_set_keeps_one_block_free_when_streaming():
    layers = {"a": {"tiles": 1, "program_upper": 10}, "b": {"tiles": 1, "program_upper": 30},
              "c": {"tiles": 3, "program_upper": 99}}
    assert chip_x2.best_resident_set(layers, 2) == ["b"]              # 1 block resident, 1 free to stream into
    assert chip_x2.best_resident_set({"a": layers["a"], "b": layers["b"]}, 2) == ["a", "b"]  # everything fits


# ---------------------------------------------------------------- plans
def test_baseline_plan_is_legal_and_has_both_views(hw):
    g = graph()
    plan = propose_partition_baseline(g, hw)
    assert validate_partition(g, plan, hw) == []
    assert plan["totals"]["x2_tiles"] == 4                            # compiler view (64 rows): 1+1+1+1
    c = plan["chip"]
    e = EXP[hw["x2"]["cell_bits"]]
    assert c["totals"]["tiles"] == e["total"] and c["resident"] == e["resident"]
    assert c["totals"]["resident_tiles"] + c["totals"]["streamed_tiles"] == e["total"]
    assert c["geometry"] == {"rows": 32, "cols": 32, "blocks": 2, "cells_per_weight": e["cpw"]}
    assert any("64-row" in f for f in c["findings"]) and any("reprogrammed every inference" in f for f in c["findings"])
    assert plan["hardware_profile"]["status"] == "chip_profile"


def test_cycle_estimates(hw):
    plan = propose_partition_baseline(graph(), hw)
    L = plan["chip"]["layers"]
    assert L["conv"]["mvms_per_tile"] == 256 and L["fc"]["mvms_per_tile"] == 1
    e = EXP[hw["x2"]["cell_bits"]]
    assert L["conv"]["compute_cycles"] == e["conv_cols_tiles"] * 256 * (32 + 450 + 32)
    assert L["conv"]["program_upper"] == 8 * 27 * e["cpw"] * 360      # every weight cell SET once


def test_cell_use_and_low_use_finding(hw):
    plan = propose_partition_baseline(graph(), hw)
    L, e = plan["chip"]["layers"], EXP[hw["x2"]["cell_bits"]]
    for k, v in L.items():
        assert v["cell_use"] == round(v["cells"] / (v["tiles"] * 32 * 32), 4) and 0 < v["cell_use"] <= 1
    # 4-bit: depthwise = 8 groups x 9 taps x 4 cells = 288 cells on 3 tiles (3072 cells) = 9%
    low = sorted(k for k, v in L.items() if v["cell_use"] < 0.25)
    assert low == e["low_use"]
    finding = [f for f in plan["chip"]["findings"] if "fill under 25%" in f]
    assert (finding != []) == bool(e["low_use"])
    if finding:
        assert "dw (9%)" in finding[0] and "take 3 of 8 X2 tiles" in finding[0]


def test_plan_from_targets_matches_baseline_and_accepts_a_resident_choice(hw):
    g = graph()
    base = propose_partition_baseline(g, hw)
    targets = {k: v["target"] for k, v in base["assignments"].items()}
    same = plan_from_targets(g, hw, targets)
    assert validate_partition(g, same, hw) == [] and same["chip"] == base["chip"]
    tiny = IRGraph("one_tile", [IRNode("x", "input", [], TensorType((1, 4))),
                                _m("fc", "linear", "x", (1, 2), K=2, C=4), IRNode("y", "output", ["fc"], TensorType((1, 2)))])
    p = propose_partition_baseline(tiny, hw)
    assert p["chip"]["resident"] == ["fc"] and p["chip"]["totals"]["program_cycles_per_inference_upper"] == 0
    assert validate_partition(tiny, p, hw) == []


@pytest.mark.parametrize("label,fn,msg", [
    ("wrong chip tiles", lambda p: p["chip"]["layers"]["pw"].update(tiles=1), "its placement has"),
    ("placement removed", lambda p: p["chip"]["layers"]["pw"].pop("placement"), "placement must be a list"),
    ("too many resident", lambda p: p["chip"].update(resident=["dw", "pw"], streamed=["conv", "fc"]), "only 1 are free"),
    ("resident not on X2", lambda p: p["chip"].update(resident=["pool"]), "resident layers not on X2"),
    ("streamed list wrong", lambda p: p["chip"].update(streamed=[]), "plan.chip.streamed"),
    ("program cycles hidden", lambda p: p["chip"]["totals"].update(program_cycles_per_inference_upper=0), "program_cycles_per_inference_upper"),
    ("compute cycles wrong", lambda p: p["chip"]["totals"].update(compute_cycles_lower=1), "compute_cycles_lower"),
    ("chip view removed", lambda p: p.pop("chip"), "plan.chip is missing"),
    ("resident not a list", lambda p: p["chip"].update(resident="all"), "must be a list"),
])
def test_validator_rejects_tampered_chip_views(hw, label, fn, msg):
    g = graph()
    plan = copy.deepcopy(propose_partition_baseline(g, hw)); fn(plan)
    errs = validate_partition(g, plan, hw)
    assert any(msg in e for e in errs), (label, errs)


def test_old_config_still_works_without_a_chip_view():
    old = json.loads((SPEC / "hardware_config.compiler_v2.json").read_text()) if (SPEC / "hardware_config.compiler_v2.json").is_file() else None
    if old is None:
        pytest.skip("no old config kept")
    g = graph()
    plan = propose_partition_baseline(g, old)
    assert "chip" not in plan and validate_partition(g, plan, old) == [] and plan["totals"]["x2_tiles"] == 4


def test_mapping_agent_reports_the_chip_view(hw, tmp_path):
    from x2c.agents.mapping_agent import run_mapping_agent
    from x2c.agents.llm_client import Action

    class Scripted:
        def start(self, goal, schemas, system_prompt=None):
            return Action("tool_calls", calls=[("describe_graph", {})])

        def send_tool_results(self, results, force_final=False):
            name, res = results[0]
            if name == "describe_graph":
                assert res["hardware"]["x2"]["rows"] == 32 and "weights" in res["hardware"]
                return Action("tool_calls", calls=[("propose_assignment", {"targets": {n["name"]: n["allowed_targets"][0] for n in res["nodes"]}})])
            return Action("final", text="done")

    out = run_mapping_agent(Scripted(), graph(), hw, log_path=str(tmp_path / "l.jsonl"))
    assert out["final_source"] == "llm_validated" and out["matches_baseline"]
    assert out["chip"]["totals"]["tiles"] == EXP[hw["x2"]["cell_bits"]]["total"] and out["hardware_profile"]["status"] == "chip_profile"
    assert "NOT silicon" in out["evidence_level"]
