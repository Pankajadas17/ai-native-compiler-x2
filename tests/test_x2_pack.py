"""Step 1 of the flow: X2 packing (weights -> cells, command stream, recombination).

Every layer's output rebuilt from X2 counts must equal NumPy exactly. 1-bit streams were also run on the BM Labs
X2 behavioral model in Icarus; those logs are bundled (tests/fixtures/x2_pack/) and replayed here.
"""
import copy
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from x2c.chip.x2_check import parse_log  # noqa: E402
from x2c.compiler import chip_x2  # noqa: E402
from x2c.compiler import x2_pack as xp  # noqa: E402
from x2c.compiler.partition import has_chip, load_hardware  # noqa: E402
import x2c.commands.sim_pack as rc  # noqa: E402

FIX = ROOT / "tests" / "fixtures" / "x2_pack"


@pytest.fixture(scope="module")
def hw4():
    hw = load_hardware()
    if not has_chip(hw):
        pytest.skip("hardware_config.json is the old 0.1 file: run x2c chip config")
    return hw


@pytest.fixture(scope="module")
def hw1(hw4):
    h = copy.deepcopy(hw4); h["x2"]["cell_bits"] = 1
    return h


@pytest.mark.parametrize("bits", [4, 1])
def test_every_int8_weight_rebuilds_exactly(hw4, bits):
    hw = copy.deepcopy(hw4); hw["x2"]["cell_bits"] = bits
    w = np.arange(-128, 128).reshape(-1, 1)
    cols = xp.weight_columns(w, hw)
    assert cols.shape == (256, chip_x2.cells_per_weight(hw)) and cols.min() >= 0 and cols.max() < (1 << bits)
    back = sum(s * scale * cols[:, i] for i, (s, _, scale) in enumerate(xp.digits(hw)))
    assert np.array_equal(back, w[:, 0])


@pytest.mark.parametrize("layer", list(rc.LAYERS))
@pytest.mark.parametrize("cells", ["4-bit", "1-bit"])
@pytest.mark.parametrize("seed", [1, 7, 42])
def test_packed_layer_equals_numpy(hw4, hw1, layer, cells, seed):
    hw = hw4 if cells == "4-bit" else hw1
    a, W, X = rc.make_data(layer, 3, seed)
    pkg = xp.pack_layer(layer, a, W, hw)
    y, st = xp.run_layer(pkg, X)
    assert np.array_equal(y, xp.reference(W, X, a))
    assert pkg.tiles == chip_x2.layer_tiles(a, hw)              # same placement as the partition
    assert st["computes"] == pkg.tiles * 3 * -(-8 // pkg.input_split_bits)


@pytest.mark.parametrize("wval", [-128, 127])
def test_extreme_weights_and_inputs(hw4, wval):
    a = {"K": 4, "C": 32, "R": 1, "S": 1, "groups": 1}              # all 32 rows used
    W, X = np.full((4, 32, 1, 1), wval), np.full((2, 1, 32), 255)
    pkg = xp.pack_layer("x", a, W, hw4)
    assert pkg.input_split_bits == 4
    assert np.array_equal(xp.run_layer(pkg, X)[0], xp.reference(W, X, a))


def test_8bit_inputs_on_4bit_cells_are_refused_when_they_could_saturate(hw4):
    a = {"K": 2, "C": 32, "R": 1, "S": 1, "groups": 1}
    W = np.full((2, 32, 1, 1), -128)        # magnitude 128 -> high digit 8 in all 32 rows: 32 x 8 x 255 = 65280
    with pytest.raises(ValueError, match="14-bit result"):
        xp.pack_layer("x", a, W, hw4, input_split_bits=8)
    assert xp.pack_layer("x", a, W, hw4).max_count == 32 * 8 * 15  # two 4-bit slices: 3840, fits


def test_required_tdc():
    assert xp.required_tdc(255) == 0 and xp.required_tdc(256) == 1 and xp.required_tdc(7200) == 28
    assert xp.required_tdc(16383) == 63 and xp.required_tdc(16384) is None


def test_1bit_stream_uses_only_chip_commands_and_4bit_uses_write_level(hw4, hw1):
    a, W, X = rc.make_data("pw", 1, 3)
    one = xp.program_commands(xp.pack_layer("pw", a, W, hw1), 0)
    four = xp.program_commands(xp.pack_layer("pw", a, W, hw4), 0)
    assert {c[0] for c in one} <= {xp.MODE_SET, xp.MODE_RESET}
    assert {c[0] for c in four} == {xp.WRITE_LEVEL}


def test_reprogramming_only_touches_changed_cells(hw1):
    a, W, X = rc.make_data("pw", 1, 3)
    pkg = xp.pack_layer("pw", a, W, hw1)
    assert xp.program_commands(pkg, 1, pkg.cells[1]) == []
    diff = int((pkg.cells[1] != pkg.cells[0]).sum())
    assert len(xp.program_commands(pkg, 1, pkg.cells[0])) == diff


def test_bad_inputs_are_rejected(hw4):
    a = rc.attrs("pw")
    with pytest.raises(ValueError):
        xp.pack_layer("pw", a, np.full((16, 8, 1, 1), 200), hw4)          # not INT8
    with pytest.raises(ValueError):
        xp.pack_layer("pw", a, np.zeros((16, 9, 1, 1)), hw4)              # wrong shape
    with pytest.raises(ValueError):
        xp.input_slices(np.array([256]), 4)                               # not uint8


@pytest.mark.parametrize("layer", list(rc.LAYERS))
def test_bm_x2_model_logs_rebuild_numpy(hw1, layer):
    """Replays the Icarus logs of the BM Labs X2 model running the 1-bit stream (recorded with --sim, 2 positions)."""
    i = list(rc.LAYERS).index(layer)
    a, W, X = rc.make_data(layer, 2, 2026 + i)
    pkg = xp.pack_layer(layer, a, W, hw1)
    beats, plan = rc.stream_for_sim(pkg, X)
    results, failures, done = parse_log((FIX / f"pack_{layer}.sim.txt").read_text())
    assert failures == [] and done["beats"] == len(beats)
    y = rc.rebuild_from_results(pkg, X, plan, results, hw1["x2"]["cols"])
    assert np.array_equal(y, xp.reference(W, X, a))


def test_live_simulation(tmp_path):
    if not (shutil.which("iverilog") and shutil.which("vvp")):
        pytest.skip("iverilog not on PATH (expected on Windows; run x2c sim pack --sim in Ubuntu)")
    assert rc.main(["--sim", "--out", str(tmp_path)]) == 0
