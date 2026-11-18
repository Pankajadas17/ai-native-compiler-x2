"""verification/formal.py + x2c formal (review item 4). Parsing and setup run everywhere; the proofs need the chip
toolchain (Ubuntu) and are skipped elsewhere."""
import json
import shutil
from pathlib import Path

import pytest

from x2c.verification import formal as F

ROOT = Path(__file__).resolve().parents[1]

PROVED = "Import proof-constraint: \\trigger = 1'0\nSAT proof finished - no model found: SUCCESS!\n"
FAILED = """ERROR: Called with -verify and proof did fail!
SAT proof finished - model found: FAIL!
   ______                   ___       ___       _ _            _ _
  \\gate_y                 127        7f                            01111111
  \\gold_y                 128        80                            10000000
  \\in_acc         -2147483648  80000000    10000000000000000000000000000000
  \\trigger                  1         1                                   1
"""


def test_parse_sat():
    assert F.parse_sat(PROVED) == {"result": "proved"}
    r = F.parse_sat(FAILED)
    assert r["result"] == "counterexample" and r["values"] == {"gate.y": 127, "gold.y": -128, "acc": -2147483648}
    assert F.parse_sat("ERROR: Module `x' not found!")["result"] == "error"


def test_strip_pads_turns_pad_cells_into_wires():
    net = {"modules": {"op": {"ports": {"a": {"direction": "input", "bits": [2]}, "y": {"direction": "output", "bits": [3]}},
                              "cells": {"pin_a": {"type": "IOBUF", "connections": {"EN": ["0"], "PAD": [2], "OUT": [10]}},
                                        "pin_y": {"type": "IOBUF", "connections": {"EN": ["1"], "PAD": [3], "IN": [11]}},
                                        "lut": {"type": "LUT1", "connections": {"I0": [10], "O": [11]}}},
                              "netnames": {}},
                       "LUT1": {"attributes": {"blackbox": "1"}}}}
    g = F.strip_pads(net, "op")
    assert list(g["modules"]) == ["op"] and list(g["modules"]["op"]["cells"]) == ["lut"]
    assert g["modules"]["op"]["cells"]["lut"]["connections"] == {"I0": [2], "O": [11]}
    assert g["modules"]["op"]["ports"]["y"]["bits"] == [11]


def test_specs_mutants_and_engine_hooks_match_the_sources():
    for op in F.OPS:
        assert f"module {op}_spec(" in F.SPECS[op]
        assert F.MUTANTS[op][0] in (ROOT / "hardware" / "operators" / f"{op}.v").read_text()
    eng = (ROOT / "hardware" / "designs" / "x2_engine2.v").read_text()
    assert F.ENGINE_MUTANT[0] in eng and '`ifdef FORMAL\n`include "x2_engine2_props.vh"\n`endif' in eng
    props = (F.FORMAL_DIR / "x2_engine2_props.vh").read_text()
    assert all(f"// {p}" in props for p in ("P1", "P2", "P3"))
    assert F._prop_name("x2_engine2_props.vh:" + str(1 + props.splitlines().index(
        next(l for l in props.splitlines() if "// P3" in l and "assert" in l)))).startswith("P3")


def _chip_tools():
    from x2c.chip import chip_flow as cf
    env = cf.tool_env()
    if not all(shutil.which(t, path=env["PATH"]) for t in ("yosys", "make", "verilator", "sby")):
        pytest.skip("chip toolchain not on PATH (Ubuntu: x2c formal)")
    import subprocess
    v = cf.yosys_version(subprocess.run([shutil.which("yosys", path=env["PATH"]), "-V"], capture_output=True, text=True).stdout)
    if v is None or v < cf.MIN_YOSYS:
        pytest.skip(f"Yosys {v} < {cf.MIN_YOSYS}")
    return env


def test_live_operator_proofs_and_lint(tmp_path, monkeypatch):
    _chip_tools()
    import x2c.commands.formal as run_formal
    monkeypatch.setattr(run_formal, "WORK", tmp_path / "work")
    monkeypatch.setattr(run_formal, "REPORT", tmp_path / "report.json")
    assert run_formal.main(["--part", "spec,netlist,lint"]) == 0
    rep = json.loads((tmp_path / "report.json").read_text())
    assert rep["ok"] and rep["chip_repo_unchanged"] and len(rep["rows"]) == 18
    bug = next(r for r in rep["rows"] if "REPO's multiplier map" in r["check"])
    assert bug["got"] == "counterexample"                        # the repo map's signed-multiply bug, found formally


def test_live_engine_mutant_is_caught(tmp_path):
    env = _chip_tools()
    r = F.engine_bmc(4, 8, tmp_path / "mutant", env, mutate=True)
    assert r["result"] == "fail" and r["failed_asserts"] and r["failed_asserts"][0].startswith("P3")
