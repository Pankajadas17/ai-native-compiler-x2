"""Flow step 2: the eFPGA recombiner (x2_acc_step operator + x2_recombiner design).

* the operator passes rtl_check (Yosys eval vs NumPy) like the other A2 operators (also via test_rtl_agent.py);
* rtl_check now reads Yosys' decimal result form (needed for 32-bit outputs);
* X2 counts -> recombiner steps -> NumPy, with counts from the 4-bit model and from the BM Labs X2 model logs;
* the recombiner RTL's Icarus logs (recorded with --sim) are bundled and replayed here.
"""
import copy
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_x2_recombine_check as rr  # noqa: E402
from compiler import x2_pack as xp  # noqa: E402
from compiler.partition import has_chip, load_hardware  # noqa: E402
from verification.rtl_check import OPS, golden_x2_acc_step, parse_eval_results, vectors_for  # noqa: E402

FIX = ROOT / "tests" / "fixtures" / "x2_recombine"


@pytest.fixture(scope="module")
def hws():
    hw4 = load_hardware()
    if not has_chip(hw4):
        pytest.skip("hardware_config.json is the old 0.1 file: run run_hardware_config.py")
    hw1 = copy.deepcopy(hw4); hw1["x2"]["cell_bits"] = 1
    return hw4, hw1


@pytest.fixture(scope="module")
def all_cases(hws):
    return list(rr.cases(*hws, 2, 2026))


def test_operator_spec_and_golden():
    s = OPS["x2_acc_step"]
    assert s["inputs"] == {"acc_in": 32, "count": 14, "shift": 4, "neg": 1} and s["output"] == ("acc_out", 32)
    g = golden_x2_acc_step(np.array([10, 2**31 - 1, -(2**31)]), np.array([3, 1, 1]), np.array([2, 0, 0]), np.array([1, 0, 1]))
    assert list(g) == [10 - 12, -(2**31), 2**31 - 1]                       # wraps like 32-bit hardware
    v = vectors_for("x2_acc_step")
    assert len(v["acc_in"]) == 636 and v["count"].max() <= 16383 and v["shift"].max() <= 15


def test_eval_parser_reads_bits_and_decimal_forms():
    out = ("Eval result: \\acc_out = 32'11111111111111111111111111111110.\n"
           "Eval result: \\acc_out = 2147483647.\n"
           "Eval result: \\y = 8'00000101.\n")
    assert parse_eval_results(out, 32) == [("32", "1" * 31 + "0"), ("32", "0" + "1" * 31), ("8", "00000101")]


def test_eight_cases_cover_both_count_sources(all_cases):
    labels = [(k, label) for k, label, *_ in all_cases]
    assert len(labels) == 8 and sum("BM X2 log" in l for _, l in labels) == 4


@pytest.mark.parametrize("i", range(8))
def test_steps_rebuild_numpy(all_cases, i):
    k, label, pkg, X, ref, counts = all_cases[i]
    steps = [xp.recombine_steps(pkg, counts[p]) for p in range(X.shape[0])]
    assert np.array_equal(rr.python_model(steps), ref), (k, label)
    shifts = {s for per in steps for st in per for _, s, _ in st}
    assert shifts <= ({0, 4, 8, 12} if label.startswith("4") else set(range(8)))


@pytest.mark.parametrize("i", range(8))
def test_recorded_recombiner_rtl_logs(all_cases, i):
    k, label, pkg, X, ref, counts = all_cases[i]
    tag = f"{k}_{label.split()[0].replace('-', '')}"
    vals, done = rr.parse_acc((FIX / f"{tag}.log").read_text())
    assert done == ref.size and np.array_equal(np.array(vals).reshape(ref.shape), ref), (k, label)


def test_a_wrong_step_is_caught(all_cases):
    k, label, pkg, X, ref, counts = all_cases[0]
    steps = [xp.recombine_steps(pkg, counts[p]) for p in range(X.shape[0])]
    c, s, n = steps[0][0][0]
    steps[0][0][0] = (c, s, 1 - n)                                       # flip one sign
    assert not np.array_equal(rr.python_model(steps), ref) or c == 0


def test_live_simulation(tmp_path):
    if not (shutil.which("iverilog") and shutil.which("vvp")):
        pytest.skip("iverilog not on PATH (expected on Windows; run run_x2_recombine_check.py --sim in Ubuntu)")
    assert rr.main(["--sim", "--out", str(tmp_path)]) == 0
