"""Recombiner check: X2 counts -> eFPGA recombiner RTL -> layer outputs, compared with NumPy.

Any terminal (Python model of the recombiner steps only):
    x2c sim recombine
Ubuntu terminal (also simulates hardware/designs/x2_recombiner.v in Icarus):
    x2c sim recombine --sim

Counts come from two sources:
  * 4-bit cells: MultiLevelX2 (the 4-bit decision model) running the packed command stream
  * 1-bit cells: the BM Labs X2 behavioral model's own results (Icarus logs recorded by the packing check,
    tests/fixtures/x2_pack/), so this links the chip's X2 model to the recombiner RTL
Writes only to chip_build/x2_recombine/.
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

import x2c.commands.sim_pack as pc  # noqa: E402
from x2c.chip.x2_check import parse_log  # noqa: E402
from x2c.compiler import x2_pack as xp  # noqa: E402
from x2c.compiler.partition import load_hardware  # noqa: E402
from x2c.verification.rtl_check import golden_x2_acc_step  # noqa: E402

OUT = ROOT / "chip_build" / "x2_recombine"
RTL = ROOT / "hardware" / "designs" / "x2_recombiner.v"
TB = ROOT / "hardware" / "sim" / "x2_recombiner_tb.v"
PACK_FIX = ROOT / "tests" / "fixtures" / "x2_pack"


def stimulus(steps_per_position):
    """[[steps per output] per position] -> stimulus text (clear, steps, read) and expected order of reads."""
    lines = []
    for per_out in steps_per_position:
        for steps in per_out:
            lines.append("C")
            lines += [f"S {c} {s} {n}" for c, s, n in steps]
            lines.append("R")
    return "\n".join(lines) + "\n"


def python_model(steps_per_position):
    """Runs the steps through golden_x2_acc_step (the operator's golden model in rtl_check)."""
    out = []
    for per_out in steps_per_position:
        row = []
        for steps in per_out:
            acc = np.array([0], dtype=np.int64)
            for c, s, n in steps:
                acc = golden_x2_acc_step(acc, np.array([c]), np.array([s]), np.array([n]))
            row.append(int(acc[0]))
        out.append(row)
    return np.array(out, dtype=np.int64)


def parse_acc(text):
    vals = [int(l.split()[1]) for l in text.splitlines() if l.startswith("ACC ")]
    done = [int(l.split()[1]) for l in text.splitlines() if l.startswith("DONE ")]
    return vals, (done[0] if done else None)


def cases(hw4, hw1, positions, seed):
    """(name, cells, pkg, X, reference, counts per position)."""
    for i, k in enumerate(pc.LAYERS):
        a, W, X = pc.make_data(k, positions, seed + i)
        pkg = xp.pack_layer(k, a, W, hw4)
        _, _, counts = xp.run_layer(pkg, X, return_counts=True)
        yield k, "4-bit (model)", pkg, X, xp.reference(W, X, a), counts
        a1, W1, X1 = pc.make_data(k, 2, 2026 + i)                      # the exact data of the recorded BM-model runs
        pkg1 = xp.pack_layer(k, a1, W1, hw1)
        beats, plan = pc.stream_for_sim(pkg1, X1)
        results, failures, done = parse_log((PACK_FIX / f"pack_{k}.sim.txt").read_text())
        if failures or not done or done["beats"] != len(beats):
            raise SystemExit(f"recorded BM X2 log for {k} does not match this stream")
        counts1 = pc.counts_from_results(pkg1, X1, plan, results, hw1["x2"]["cols"])
        yield k, "1-bit (BM X2 log)", pkg1, X1, xp.reference(W1, X1, a1), counts1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--positions", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    hw4 = load_hardware()
    hw1 = copy.deepcopy(hw4); hw1["x2"]["cell_bits"] = 1
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    vvp = None
    if a.sim:
        for tool in ("iverilog", "vvp"):
            if not shutil.which(tool):
                raise SystemExit(f"{tool} not on PATH (Ubuntu terminal, after the export line)")
        vvp = out / "recombiner_tb.vvp"
        r = subprocess.run(["iverilog", "-g2012", "-o", str(vvp), str(RTL), str(TB)], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit("iverilog failed:\n" + r.stderr[-2000:])
    ok, rows = True, []
    print(f"{'layer':5s} {'counts from':18s} {'outputs':>7s} {'steps':>6s}  python model  recombiner RTL (Icarus)")
    for k, label, pkg, X, ref, counts in cases(hw4, hw1, a.positions, a.seed):
        steps = [xp.recombine_steps(pkg, counts[p]) for p in range(X.shape[0])]
        n_steps = sum(len(s) for per in steps for s in per)
        py = python_model(steps)
        py_ok = bool(np.array_equal(py, ref))
        sim = "-"
        if vvp is not None:
            tag = f"{k}_{label.split()[0].replace('-', '')}"
            stim, log = out / f"{tag}.stim", out / f"{tag}.log"
            stim.write_text(stimulus(steps))
            subprocess.run(["vvp", "-n", str(vvp), f"+STIM={stim}", f"+LOG={log}"], capture_output=True, text=True)
            vals, done = parse_acc(log.read_text())
            sim_ok = done == ref.size and np.array_equal(np.array(vals).reshape(ref.shape), ref)
            sim = "PASS" if sim_ok else "FAIL"
            ok &= sim_ok
        ok &= py_ok
        rows.append({"layer": k, "counts": label, "outputs": int(ref.size), "steps": n_steps, "python": py_ok, "sim": sim})
        print(f"{k:5s} {label:18s} {ref.size:7d} {n_steps:6d}  {'PASS' if py_ok else 'FAIL':12s}  {sim}")
    (out / "report.json").write_text(json.dumps({"all_pass": ok, "rows": rows}, indent=2) + "\n")
    print(f"X2 RECOMBINE CHECK: {'PASS' if ok else 'FAIL'}  (rebuilt outputs == NumPy for every layer)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
