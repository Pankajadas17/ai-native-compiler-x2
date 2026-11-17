"""Regression tests for canvas operations (values reproduced from the repo code; confirm
them against the GUI result panel once you have access - a mismatch means a different commit)."""
import pytest
from emulators.platform_ops import map_layer, schedule_layer, energy_estimate

def test_map_layer_conv_64x32_3x3():
    r = map_layer({"K": 64, "C": 32, "R": 3, "S": 3, "Y": 16, "X": 16, "pe_rows": 64, "pe_cols": 32})
    assert [list(c) for c in r["k_chunks"]] == [[0, 32], [32, 64]]
    assert list(r["fold_sizes"]) == [64, 64, 64, 64, 32]
    assert (r["patch_len"], r["pe_tiles"], r["mvm_ops"]) == (288, 10, 2560)
    assert r["utilization"] == pytest.approx(0.9)

SMALL = {"K": 32, "C": 1, "R": 1, "S": 1, "Y": 1, "X": 1, "xy_tile": 1}

def test_schedule_layer_small():
    r = schedule_layer(SMALL)
    assert (r["pes_needed"], r["k_chunks"], r["patch_len"]) == (1, 1, 1)
    assert r["loop_order"] == ["K", "XY", "CRS_fold"]

def test_energy_estimate_small():
    r = energy_estimate(dict(SMALL, weight_density=0.5, input_density=0.5))
    assert r["program_J"] == pytest.approx(9.2176e-06, rel=1e-3)
    assert r["total_program_plus_compute_J"] == pytest.approx(9.3307e-06, rel=1e-3)
    assert r["assumptions"]["basis"] == "expected value, binary 0/1"
