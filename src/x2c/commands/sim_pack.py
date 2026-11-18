"""Check X2 packing end to end: pack layers, run the command streams, rebuild the outputs, compare with NumPy.

Any terminal (Python models only):
    x2c sim pack
Ubuntu terminal (also runs the 1-bit command streams on the BM Labs X2 behavioral model in Icarus):
    x2c sim pack --sim

Layers: the synthetic tiny-MobileNetV2 shapes (conv, depthwise, pointwise, linear) with random INT8 weights and
random uint8 inputs (seeded). 4-bit cells (the configured decision) run on MultiLevelX2 only; 1-bit cells run on
the chip-semantics golden model and, with --sim, on the chip repo's own X2 model.
Writes only to chip_build/x2_pack/.
"""
from x2c import paths
import argparse
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = paths.ROOT

from x2c.chip.chip_repo import INPUT_FILES, ChipRepo  # noqa: E402
from x2c.chip.x2_check import Scenario, compare, parse_log, write_vectors  # noqa: E402
from x2c.chip.x2_golden import X2Config  # noqa: E402
from x2c.compiler import chip_x2  # noqa: E402
from x2c.compiler import x2_pack as xp  # noqa: E402
from x2c.compiler.partition import load_hardware  # noqa: E402

OUT = ROOT / "chip_build" / "x2_pack"
TB = ROOT / "hardware" / "sim" / "x2_vector_tb.sv"
LAYERS = {  # name: (K, C, R, S, groups)  -- the synthetic tiny-MobileNetV2 shapes used by tests/test_chip_partition.py
    "conv": (8, 3, 3, 3, 1), "dw": (8, 8, 3, 3, 8), "pw": (16, 8, 1, 1, 1), "fc": (10, 16, 1, 1, 1)}


def attrs(k):
    K, C, R, S, g = LAYERS[k]
    return {"K": K, "C": C, "R": R, "S": S, "groups": g}


def make_data(k, positions, seed):
    a = attrs(k)
    g, taps, outs = chip_x2.layer_shape(a)
    rng = np.random.default_rng(seed)
    W = rng.integers(-128, 128, size=(a["K"], a["C"] // g, a["R"], a["S"]))
    X = rng.integers(0, 256, size=(positions, g, taps))
    return a, W, X


def stream_for_sim(pkg, X):
    """Full 1-bit command stream for one block (program tile, compute every position, next tile), plus a map
    telling which simulator results belong to which (position, tile, slice)."""
    beats, plan, state = [], [], np.zeros_like(pkg.cells[0])
    n_slices = -(-8 // pkg.input_split_bits)
    for ti in range(pkg.tiles):
        prog = xp.program_commands(pkg, ti, state)
        beats += prog
        plan += [("program", None)] * len(prog)            # each SET/RESET returns one readback result
        state = pkg.cells[ti].copy()
        for p in range(X.shape[0]):
            for si in range(n_slices):
                beats += xp.compute_beats(pkg, ti, X[p], si)
                plan.append(("compute", (p, ti, si)))      # all_columns COMPUTE returns 32 results
    return beats, plan


def rebuild_from_results(pkg, X, plan, results, cols):
    counts = counts_from_results(pkg, X, plan, results, cols)
    return np.stack([np.concatenate(xp.recombine(pkg, counts[p])) for p in range(X.shape[0])])


def counts_from_results(pkg, X, plan, results, cols):
    """Simulator results in stream order -> per position {(tile, slice): 32 column counts}."""
    counts = [dict() for _ in range(X.shape[0])]
    i = 0
    for kind, key in plan:
        if kind == "program":
            i += 1
            continue
        chunk = results[i:i + cols]
        i += cols
        col = [0] * cols
        for r in chunk:
            col[r.col] = r.count
        p, ti, si = key
        counts[p][(ti, si)] = col
    if i != len(results):
        raise ValueError(f"simulator returned {len(results)} results, the stream implies {i}")
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true", help="also run the 1-bit streams on the BM Labs X2 model (Ubuntu)")
    ap.add_argument("--positions", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    hw4 = load_hardware()
    hw1 = copy.deepcopy(hw4); hw1["x2"]["cell_bits"] = 1
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows, ok = [], True
    repo = vvp = None
    if a.sim:
        for tool in ("iverilog", "vvp"):
            if not shutil.which(tool):
                raise SystemExit(f"{tool} not on PATH (Ubuntu terminal, after the export line)")
        repo = ChipRepo()
        before = repo.tree_fingerprint()
        vvp = out / "x2_tb.vvp"
        r = subprocess.run(["iverilog", "-g2012", "-o", str(vvp), str(repo.path(INPUT_FILES["x2_behavioral"])), str(TB)],
                           capture_output=True, text=True)
        if r.returncode:
            raise SystemExit("iverilog failed:\n" + r.stderr[-2000:])
    print(f"{'layer':5s} {'cells':6s} {'tiles':>5s} {'TDC>=':>5s} {'prog cmds':>9s} {'computes':>8s}  python model  BM X2 model (Icarus)")
    for i, k in enumerate(LAYERS):
        at, W, X = make_data(k, a.positions, a.seed + i)
        ref = xp.reference(W, X, at)
        for label, hw in (("4-bit", hw4), ("1-bit", hw1)):
            pkg = xp.pack_layer(k, at, W, hw)
            y, st = xp.run_layer(pkg, X)
            py_ok = bool(np.array_equal(y, ref)) and pkg.tiles == chip_x2.layer_tiles(at, hw)
            sim = "-"
            if a.sim and label == "1-bit":
                beats, plan = stream_for_sim(pkg, X)
                sc = Scenario(f"pack_{k}", "packed layer", X2Config(tdc_time_out=pkg.required_tdc_time_out), beats)
                vec, log = out / f"pack_{k}.vec", out / f"pack_{k}.sim.txt"
                write_vectors(vec, sc)
                subprocess.run(["vvp", "-n", str(vvp), f"+VECTORS={vec}", f"+LOG={log}"], capture_output=True, text=True)
                rep = compare(sc, log.read_text())
                sim_y = rebuild_from_results(pkg, X, plan, parse_log(log.read_text())[0], hw["x2"]["cols"])
                sim_ok = rep["pass"] and bool(np.array_equal(sim_y, ref))
                sim = f"{'PASS' if sim_ok else 'FAIL'} ({rep['simulated_results']} results, {rep['cycles']} cycles)"
                ok &= sim_ok
            ok &= py_ok
            rows.append({"layer": k, "cells": label, "tiles": pkg.tiles, "required_tdc_time_out": pkg.required_tdc_time_out,
                         "python_exact": py_ok, "sim": sim, **st})
            print(f"{k:5s} {label:6s} {pkg.tiles:5d} {pkg.required_tdc_time_out:5d} {st['program_commands']:9d} "
                  f"{st['computes']:8d}  {'PASS' if py_ok else 'FAIL':12s}  {sim}")
    if a.sim:
        untouched = before == repo.tree_fingerprint()
        print(f"chip repo read-only: {untouched}")
        ok &= untouched
    (out / "report.json").write_text(json.dumps({"all_pass": ok, "rows": rows}, indent=2) + "\n")
    print(f"X2 PACK CHECK: {'PASS' if ok else 'FAIL'}  (outputs == NumPy for every layer and cell size)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
