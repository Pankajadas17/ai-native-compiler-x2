"""Two-block X2 engine (examples/chip_designs/x2_engine2.v): both blocks used at once, measured in simulation.

* recorded Icarus runs (host -> engine pins -> two chip-wired BM X2 models), both layouts, all 4 layers: every accumulator
  equals the engine rule and every output equals NumPy;
* the measured cycles match compiler/x2_schedule.simulate (what A1 uses) within 0.1 %, and two blocks cut the latency 40 %+;
* a lost result is caught; the RTL has the pins, the arbiter and the forwarding it claims; a live run if Icarus is installed.
"""
import copy
import re
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_x2_engine2_check as e2  # noqa: E402
import run_x2_engine_check as ec  # noqa: E402
from compiler.partition import load_hardware  # noqa: E402

FIX = ROOT / "tests" / "fixtures" / "x2_engine2"


@pytest.fixture(scope="module")
def runs():
    hw1 = copy.deepcopy(load_hardware()); hw1["x2"]["cell_bits"] = 1
    out = {}
    for k, pkg, X, ref, cfg in ec.cases(hw1, 2, 2026):
        for layout in e2.LAYOUTS:
            scripts, entries, costs, nacc = e2.queues(pkg, X, layout)
            text = (FIX / f"engine2_{k}_{layout}.log").read_text()
            out[(k, layout)] = (pkg, X, ref, nacc, entries, costs, text)
    return out


def test_recorded_runs_are_exact(runs):
    assert len(runs) == 8
    for key, (pkg, X, ref, nacc, entries, costs, text) in runs.items():
        row = e2.check(pkg, X, ref, text, nacc)
        assert row["pass"] and row["model_match"] and row["numpy_exact"], key


def test_measured_cycles_match_the_schedule_model(runs):
    for key, (pkg, X, ref, nacc, entries, costs, text) in runs.items():
        measured = e2.check(pkg, X, ref, text, nacc)["cycles"]
        assert abs(e2.predicted(entries, costs) / measured - 1) < 0.001, key
    for k in ("conv", "dw", "pw", "fc"):
        one = e2.check(*[runs[(k, "single")][i] for i in (0, 1, 2)], runs[(k, "single")][6], runs[(k, "single")][3])["cycles"]
        two = e2.check(*[runs[(k, "alternate")][i] for i in (0, 1, 2)], runs[(k, "alternate")][6], runs[(k, "alternate")][3])["cycles"]
        assert two < 0.6 * one, k                                      # two blocks: at least a 40 % cut on every layer


def test_layouts_put_tiles_where_they_say(runs):
    _, _, _, _, entries, _, _ = runs[("pw", "alternate")]
    assert [b for _, _, b in entries] == [0, 1] * 4
    _, _, _, _, entries, _, _ = runs[("pw", "single")]
    assert {b for _, _, b in entries} == {0}


def test_a_lost_result_is_caught(runs):
    pkg, X, ref, nacc, _, _, text = runs[("conv", "alternate")]
    line = next(l for l in text.splitlines() if l.startswith("ACC 1 ") and not l.endswith(" 0"))
    f = line.split(); f[-1] = str(int(f[-1]) + (37 << 3))
    row = e2.check(pkg, X, ref, text.replace(line, " ".join(f)), nacc)
    assert not row["pass"] and not row["model_match"]
    assert not e2.check(pkg, X, ref, text.split("DONE")[0], nacc)["pass"]


def test_rtl_states_its_pins_arbiter_and_forwarding():
    rtl = (ROOT / "examples" / "chip_designs" / "x2_engine2.v").read_text()
    ports = re.findall(r"^\s*(input|output)\s+wire\s*(\[(\d+):0\])?\s*(\w+)", rtl, re.M)
    assert sum(int(w) + 1 if w else 1 for _, _, w, _ in ports) == 35
    for frag in ("wire        g0 = rv[0] && (!rv[1] || last_g1);", ".RESULT_READY(g0)", ".RESULT_READY(g1)",
                 "s2_old   <= (s2_writes && s2_idx == s1_idx) ? nxt_full : acc[s1_idx];", "x2_0 (", "x2_1 ("):
        assert frag in rtl, frag


def test_live_simulation(tmp_path):
    if not (shutil.which("iverilog") and shutil.which("vvp")):
        pytest.skip("iverilog not on PATH (expected on Windows; run run_x2_engine2_check.py --sim in Ubuntu)")
    assert e2.main(["--sim", "--out", str(tmp_path)]) == 0
