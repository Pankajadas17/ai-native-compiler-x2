"""Flow step 3: the X2 controller (host bridge) on the eFPGA.

* the bridge connects every port of the chip's RERAM_CIM_32x32 primitive;
* the simulation stand-in connects the BM Labs macro exactly like the chip's src/fabric_wrapper.sv;
* bundled Icarus logs (host -> bridge pins -> X2 model) equal the X2 golden model, flow step 1's direct run, and NumPy.
"""
import copy
import re
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_x2_bridge_check as bc  # noqa: E402
from chip.chip_repo import ChipRepo, ChipRepoError  # noqa: E402
from compiler.partition import has_chip, load_hardware  # noqa: E402

FIX = ROOT / "tests" / "fixtures" / "x2_bridge"
PRIM = "ip/extra-tiles/primitives/RERAM_CIM_32x32/yosys/primitives/RERAM_CIM_32x32.v"


@pytest.fixture(scope="module")
def repo():
    try:
        r = ChipRepo()
    except ChipRepoError as e:
        pytest.skip(str(e))
    return r


@pytest.fixture(scope="module")
def hw1():
    hw = load_hardware()
    if not has_chip(hw):
        pytest.skip("hardware_config.json is the old 0.1 file: run run_hardware_config.py")
    hw = copy.deepcopy(hw); hw["x2"]["cell_bits"] = 1
    return hw


def _connections(text, instance_marker):
    body = text[text.index(instance_marker):]
    body = body[:body.index(");")]
    return set(re.findall(r"\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", body))


def test_bridge_connects_every_primitive_port(repo):
    prim = set(re.findall(r"(?:input|output)\s+wire\s+([A-Z_0-9]+)", repo.read_text(PRIM)))
    assert len(prim) == 58
    bridge = (ROOT / "examples" / "chip_designs" / "x2_bridge.v").read_text()
    conns = _connections(bridge, ") x2 (")
    assert prim == conns


def test_sim_stand_in_wires_the_macro_like_the_chip(repo):
    chip = repo.read_text("src/fabric_wrapper.sv")
    want = _connections(chip, "sky130_bmlabs_reram__cim32x32 reram_0 (")
    stand_in = (ROOT / "sim" / "RERAM_CIM_32x32_sim.v").read_text()
    got = _connections(stand_in, "sky130_bmlabs_reram__cim32x32 macro (")
    assert got == want and len(want) == 44


def test_primitive_exposes_two_of_three_error_bits(repo):
    """Finding: fabric_wrapper.sv wires ERROR0..2, the Yosys primitive declares only ERROR0, ERROR1."""
    assert "EXT_ERROR2" in repo.read_text("src/fabric_wrapper.sv")
    assert "ERROR2" not in repo.read_text(PRIM) and "ERROR1" in repo.read_text(PRIM)


@pytest.mark.parametrize("layer", ["conv", "dw", "pw", "fc"])
def test_recorded_bridge_runs(hw1, layer):
    case = next(c for c in bc.layer_cases(hw1) if c[0] == layer)
    k, pkg, X, ref, sc, plan = case
    row = bc.check_log(k, pkg, X, ref, sc, plan, (FIX / f"bridge_{k}.log").read_text(), hw1["x2"]["cols"])
    assert row["golden_match"] and row["same_as_direct"] and row["numpy_exact"], row


def test_a_corrupted_log_is_caught(hw1):
    k, pkg, X, ref, sc, plan = next(bc.layer_cases(hw1))
    lines = (FIX / f"bridge_{k}.log").read_text().splitlines()
    i = next(n for n, l in enumerate(lines) if l.startswith("R 2 "))       # first COMPUTE result
    f = lines[i].split(); f[3] = str(int(f[3]) + 1); lines[i] = " ".join(f)
    row = bc.check_log(k, pkg, X, ref, sc, plan, "\n".join(lines) + "\n", hw1["x2"]["cols"])
    assert not row["pass"] and not row["golden_match"] and not row["numpy_exact"]


def test_live_simulation(tmp_path):
    if not (shutil.which("iverilog") and shutil.which("vvp")):
        pytest.skip("iverilog not on PATH (expected on Windows; run run_x2_bridge_check.py --sim in Ubuntu)")
    assert bc.main(["--sim", "--out", str(tmp_path)]) == 0
