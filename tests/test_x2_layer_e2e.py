"""x2c sim layers: real layer shapes end to end (4-bit in Python with the ASSUMED cell rule; 1-bit simulated, Ubuntu)."""
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import x2c.commands.sim_layers as le  # noqa: E402

SHAPES = ["16,3,3,3,1", "16,16,3,3,16", "24,16,1,1,1", "10,32,1,1,1"]


def test_4bit_layers_are_exact_in_python(tmp_path):
    argv = sum([["--shape", s] for s in SHAPES], []) + ["--out", str(tmp_path)]
    assert le.main(argv) == 0
    rep = json.loads((tmp_path / "report.json").read_text())
    assert rep["all_exact"] and len(rep["rows"]) == 4
    assert all(r["bits4"]["exact"] and r["bits4"]["cell_rule"].startswith("ASSUMED") for r in rep["rows"])
    assert [r["bits4"]["tiles"] for r in rep["rows"]] == [2, 6, 3, 2] and [r["bits1"]["tiles"] for r in rep["rows"]] == [8, 8, 12, 5]


def test_weights_are_seeded_and_replaceable(tmp_path):
    import numpy as np
    a = {"K": 4, "C": 2, "R": 1, "S": 1, "groups": 1}
    w1, x1 = le.data("L", a, 2)
    w2, x2 = le.data("L", a, 2)
    assert (w1 == w2).all() and (x1 == x2).all()
    np.savez(tmp_path / "w.npz", L=np.ones((4, 2, 1, 1), dtype=np.int64))
    w3, _ = le.data("L", a, 2, np.load(tmp_path / "w.npz"))
    assert (w3 == 1).all()


def test_too_big_layers_are_skipped_not_hidden(tmp_path):
    assert le.main(["--shape", "24,16,1,1,1", "--max-tiles", "4", "--sim", "--out", str(tmp_path)]) == 0 if shutil.which("iverilog") \
        else le.main(["--shape", "24,16,1,1,1", "--max-tiles", "4", "--out", str(tmp_path)]) == 0
    row = json.loads((tmp_path / "report.json").read_text())["rows"][0]
    assert "skipped" in row["bits1"]


def test_real_ir_layers():
    pytest.importorskip("x2c.inputs.bmsemi_source")
    try:
        layers = le.layers_from_ir()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"BMsemi_Model not available: {e}")
    assert len(layers) == 17 and "features_0_0" in layers


def test_live_simulation(tmp_path):
    if not (shutil.which("iverilog") and shutil.which("vvp")):
        pytest.skip("iverilog not on PATH (expected on Windows; run x2c sim layers --sim in Ubuntu)")
    assert le.main(["--sim", "--shape", "10,32,1,1,1", "--out", str(tmp_path)]) == 0
    row = json.loads((tmp_path / "report.json").read_text())["rows"][0]
    assert row["bits1"]["alternate"]["pass"] and row["bits1"]["latency_cut_percent"] > 40
