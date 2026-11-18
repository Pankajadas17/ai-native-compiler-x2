"""One-block X2 engine (bridge + on-fabric recombination).

* the fabric column rule (accumulator = col / CPW, sign and digit from col % CPW) holds for every placement;
* the engine's rule rebuilds NumPy exactly for 4-bit AND 1-bit cells (Python model of the RTL rule);
* bundled Icarus logs (host -> engine pins -> chip-wired BM X2 model) match the model and rebuild NumPy;
* the simulation stand-in delays all 32 X2 outputs (models clock-to-output; without it the engine missed results).
"""
import copy
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import x2c.commands.sim_engine1 as ec  # noqa: E402
import x2c.commands.sim_pack as pc  # noqa: E402
from x2c.compiler import x2_pack as xp  # noqa: E402
from x2c.compiler.partition import has_chip, load_hardware  # noqa: E402

FIX = ROOT / "tests" / "fixtures" / "x2_engine"


@pytest.fixture(scope="module")
def hws():
    hw4 = load_hardware()
    if not has_chip(hw4):
        pytest.skip("hardware_config.json is the old 0.1 file: run x2c chip config")
    hw1 = copy.deepcopy(hw4); hw1["x2"]["cell_bits"] = 1
    return {4: hw4, 1: hw1}


@pytest.mark.parametrize("bits", [4, 1])
@pytest.mark.parametrize("layer", list(pc.LAYERS))
@pytest.mark.parametrize("seed", [3, 99])
def test_fabric_rule_rebuilds_numpy(hws, bits, layer, seed):
    a, W, X = pc.make_data(layer, 3, seed)
    pkg = xp.pack_layer(layer, a, W, hws[bits])
    layout = ec.engine_layout(pkg)                          # raises if a block breaks the column rule
    order, nacc = ec.read_order(pkg, X), pkg.hw["x2"]["cols"] // xp.chip_x2.cells_per_weight(pkg.hw)
    _, _, counts = xp.run_layer(pkg, X, return_counts=True)
    accs = ec.engine_model_by_order(pkg, X, counts, order)
    assert nacc == (8 if bits == 4 else 2)
    assert np.array_equal(ec.host_rebuild(pkg, X, order, accs, layout), xp.reference(W, X, a))


@pytest.mark.parametrize("layer", list(pc.LAYERS))
def test_recorded_engine_runs(hws, layer):
    k, pkg, X, ref, cfg = next(c for c in ec.cases(hws[1], 2, 2026) if c[0] == layer)
    _, order, nacc = ec.script(pkg, X, cfg)
    row = ec.check(pkg, X, ref, (FIX / f"engine_{k}.log").read_text(), order, nacc)
    assert row["pass"] and row["model_match"] and row["numpy_exact"], row


def test_a_missed_result_is_caught(hws):
    """Exactly the failure the clock-to-output delay fixed: one column count missing from one accumulator."""
    k, pkg, X, ref, cfg = next(c for c in ec.cases(hws[1], 2, 2026) if c[0] == "pw")
    _, order, nacc = ec.script(pkg, X, cfg)
    lines = (FIX / f"engine_{k}.log").read_text().splitlines()
    i = next(n for n, l in enumerate(lines) if l.startswith("ACC 1 ") and l != "ACC 1 0")
    f = lines[i].split(); f[2] = str(int(f[2]) + (241 << 7)); lines[i] = " ".join(f)
    row = ec.check(pkg, X, ref, "\n".join(lines) + "\n", order, nacc)
    assert not row["pass"] and not row["model_match"] and not row["numpy_exact"]


def test_stand_in_delays_every_output():
    text = (ROOT / "hardware" / "sim" / "RERAM_CIM_32x32_sim.v").read_text()
    outs = set(re.findall(r"output\s+wire\s+([A-Z_0-9]+)", text))
    delayed = set(re.findall(r"assign\s+#1\s+([A-Z_0-9]+)\s*=", text))
    assert len(outs) == 32 and outs == delayed     # 21 data + 11 status/handshake


def test_engine_rtl_states_the_same_rule():
    rtl = (ROOT / "hardware" / "designs" / "x2_engine.v").read_text()
    for frag in ("CPW   = 2 * ((8 + CELL_BITS - 1) / CELL_BITS)", "slot    = r_col % CPW", "r_neg   = (slot >= HALF)",
                 "r_shift = (slot % HALF) * CELL_BITS + (slice ? SPLIT : 0)", "r_acc   = r_col / CPW"):
        assert frag in rtl, frag


def test_live_simulation(tmp_path):
    if not (shutil.which("iverilog") and shutil.which("vvp")):
        pytest.skip("iverilog not on PATH (expected on Windows; run x2c sim engine1 --sim in Ubuntu)")
    assert ec.main(["--sim", "--out", str(tmp_path)]) == 0
