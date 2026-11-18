"""One-block engine check: the X2 engine (bridge + on-fabric recombination) end to end.

A host drives hardware/designs/x2_engine.v through its pins only: it streams the packed command stream
(the packing), sets the input slice of each COMPUTE pass, reads the finished partial outputs from the on-fabric
accumulators, and adds row folds. The X2 block is the BM Labs behavioral model wired like the chip
(hardware/sim/RERAM_CIM_32x32_sim.v). 1-bit cells only: the chip's commands cannot program 4-bit cells.

Ubuntu terminal:   x2c sim engine1 --sim
Any terminal:      x2c sim engine1           (re-checks logs already in chip_build/x2_engine_sim/)
Per layer: (1) every accumulator read equals the Python model of the engine, (2) the layer output rebuilt by the host
equals NumPy. Writes only to chip_build/x2_engine_sim/.
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
from x2c.chip.chip_repo import INPUT_FILES, ChipRepo  # noqa: E402
from x2c.chip.x2_golden import X2Config  # noqa: E402
from x2c.compiler import chip_x2  # noqa: E402
from x2c.compiler import x2_pack as xp  # noqa: E402
from x2c.compiler.partition import load_hardware  # noqa: E402

OUT = ROOT / "chip_build" / "x2_engine_sim"
SOURCES = [ROOT / "hardware" / "sim" / "RERAM_CIM_32x32_sim.v", ROOT / "hardware" / "designs" / "x2_engine.v",
           ROOT / "hardware" / "sim" / "x2_engine_tb.v"]
PARAMS = ("TARGET_SET1", "TARGET_SET2", "TARGET_RESET1", "TARGET_RESET2", "NO_OF_CLK_CYCLES", "COUNTER_VALUE",
          "TDC_TIME_OUT", "TDC_DEAD_TIME")


def engine_layout(pkg):
    """Per tile: {accumulator index: (group, output channel)} for every column group that holds weights."""
    cpw = chip_x2.cells_per_weight(pkg.hw)
    out = []
    for tile in pkg.placement:
        m = {}
        for grp, ro, co, r0, c0, nr, nc in tile:
            if c0 % cpw or co % cpw or nc % cpw:
                raise ValueError("block not aligned to cells_per_weight: the engine's column rule would not hold")
            for j in range(0, nc, cpw):
                m[(c0 + j) // cpw] = (grp, (co + j) // cpw)
        out.append(m)
    return out


def read_order(pkg, X):
    """Order of accumulator reads: every position of tile 0, then of tile 1, ... [(position, tile)]."""
    return [(p, ti) for ti in range(pkg.tiles) for p in range(X.shape[0])]


def script(pkg, X, cfg):
    """Host script for the testbench and the order of accumulator reads [(position, tile)]."""
    nacc = pkg.hw["x2"]["cols"] // chip_x2.cells_per_weight(pkg.hw)
    n_slices = -(-8 // pkg.input_split_bits)
    lines = ["CFG " + " ".join(str(v) for v in cfg.as_list())]
    order, state = [], np.zeros_like(pkg.cells[0])
    for ti in range(pkg.tiles):
        lines += ["B %d %d %d %d %d %d" % (m, r, c, p, int(l), int(a)) for m, r, c, p, l, a in xp.program_commands(pkg, ti, state)]
        state = pkg.cells[ti].copy()
        for p in range(X.shape[0]):
            lines.append("C")
            for si in range(n_slices):
                lines.append(f"P {si}")
                lines += ["B %d %d %d %d %d %d" % (m, r, c, pw, int(l), int(a)) for m, r, c, pw, l, a in xp.compute_beats(pkg, ti, X[p], si)]
            lines.append(f"A {nacc}")
            order.append((p, ti))
    assert order == read_order(pkg, X)
    return "\n".join(lines) + "\n", order, nacc


def engine_model(pkg, counts):
    """What the engine's accumulators hold after one (position, tile): the fabric rule, written in Python."""
    cb, cpw = pkg.hw["x2"]["cell_bits"], chip_x2.cells_per_weight(pkg.hw)
    half, nacc = cpw // 2, pkg.hw["x2"]["cols"] // cpw
    out = {}
    for ti in range(pkg.tiles):
        acc = [0] * nacc
        for (t, si), cols in counts.items():
            if t != ti:
                continue
            for c, count in enumerate(cols):
                slot = c % cpw
                v = count << ((slot % half) * cb + si * pkg.input_split_bits)
                acc[c // cpw] += -v if slot >= half else v
        out[ti] = acc
    return out


def host_rebuild(pkg, X, order, accs, layout):
    g, taps, outs = chip_x2.layer_shape(pkg.attrs)
    y = np.zeros((X.shape[0], g * outs), dtype=np.int64)
    for (p, ti), vals in zip(order, accs):
        for a, (grp, k) in layout[ti].items():
            y[p, grp * outs + k] += vals[a]                      # row folds of the same outputs add up here
    return y


def parse(text, nacc):
    vals = [int(l.split()[2]) for l in text.splitlines() if l.startswith("ACC ")]
    done = next((l.split()[1:] for l in text.splitlines() if l.startswith("DONE ")), None)
    return [vals[i:i + nacc] for i in range(0, len(vals), nacc)], done


def cases(hw1, positions, seed):
    for i, k in enumerate(pc.LAYERS):
        a, W, X = pc.make_data(k, positions, seed + i)
        pkg = xp.pack_layer(k, a, W, hw1)
        yield k, pkg, X, xp.reference(W, X, a), X2Config(tdc_time_out=pkg.required_tdc_time_out)


def check(pkg, X, ref, text, order, nacc):
    accs, done = parse(text, nacc)
    _, _, counts = xp.run_layer(pkg, X, return_counts=True)
    model = engine_model_by_order(pkg, X, counts, order)
    model_ok = accs == model
    y = host_rebuild(pkg, X, order, accs, engine_layout(pkg)) if len(accs) == len(order) else None
    numpy_ok = y is not None and bool(np.array_equal(y, ref))
    return {"pass": bool(done) and model_ok and numpy_ok, "reads": len(accs) * nacc, "model_match": model_ok,
            "numpy_exact": numpy_ok, "cycles": int(done[2]) if done else None}


def engine_model_by_order(pkg, X, counts, order):
    per = [engine_model(pkg, counts[p]) for p in range(X.shape[0])]
    return [per[p][ti] for p, ti in order]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--positions", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    hw1 = copy.deepcopy(load_hardware()); hw1["x2"]["cell_bits"] = 1
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    repo = before = None
    if a.sim:
        for tool in ("iverilog", "vvp"):
            if not shutil.which(tool):
                raise SystemExit(f"{tool} not on PATH (Ubuntu terminal, after the export line)")
        repo = ChipRepo(); before = repo.tree_fingerprint()
    ok, rows = True, []
    print(f"{'layer':5s} {'tiles':>5s} {'reads':>6s} {'cycles':>8s}  {'= engine model':14s} {'output = NumPy':14s}")
    for k, pkg, X, ref, cfg in cases(hw1, a.positions, a.seed):
        text_script, order, nacc = script(pkg, X, cfg)
        log = out / f"engine_{k}.log"
        if a.sim:
            scr = out / f"engine_{k}.script"
            scr.write_text(text_script)
            vvp = out / f"engine_{k}.vvp"
            params = [f"-Px2_engine_tb.{n}={v}" for n, v in zip(PARAMS, cfg.as_list())] + ["-Px2_engine_tb.CELL_BITS=1"]
            r = subprocess.run(["iverilog", "-g2012", "-o", str(vvp), *params, str(repo.path(INPUT_FILES["x2_behavioral"])),
                                *map(str, SOURCES)], capture_output=True, text=True)
            if r.returncode:
                raise SystemExit("iverilog failed:\n" + r.stderr[-2000:])
            subprocess.run(["vvp", "-n", str(vvp), f"+SCRIPT={scr}", f"+LOG={log}"], capture_output=True, text=True)
        if not log.is_file():
            ok = False
            rows.append({"layer": k, "pass": False, "problem": "no log: run with --sim in the Ubuntu terminal"})
            print(f"{k:5s} no log (run --sim in Ubuntu)")
            continue
        row = {"layer": k, **check(pkg, X, ref, log.read_text(), order, nacc)}
        rows.append(row)
        ok &= row["pass"]
        yn = lambda v: "yes" if v else "NO"
        print(f"{k:5s} {pkg.tiles:5d} {row['reads']:6d} {str(row['cycles']):>8s}  {yn(row['model_match']):14s} {yn(row['numpy_exact']):14s}")
    if a.sim:
        untouched = before == repo.tree_fingerprint()
        print(f"chip repo read-only: {untouched}")
        ok &= untouched
    (out / "report.json").write_text(json.dumps({"all_pass": ok, "rows": rows}, indent=2) + "\n")
    print(f"X2 ENGINE CHECK: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
