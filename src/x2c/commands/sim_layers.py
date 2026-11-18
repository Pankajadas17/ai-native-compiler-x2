"""
Real tiny-MobileNetV2 layers end to end on the chip's X2: IR layer -> pack -> two-block schedule -> engine -> NumPy.

For every chosen matrix layer of the BMsemi compiler graph (shapes from the IR):
  4-bit (the project's cells per weight): packed and run in Python with the ASSUMED 4-bit cell rule (count = sum pwm * level;
         chip/x2_facts.py 'cell_compute_4bit' is missing because the chip's X2 model has no 4-bit cells) -> exact vs NumPy?
  1-bit (what the chip's X2 model computes): simulated in Icarus on hardware/designs/x2_engine2.v with tiles alternating
         over both blocks and with all tiles on one block -> every output exact vs NumPy? measured vs predicted cycles?
Weights: random INT8, seeded by the layer name (the compiler artifact gives shapes, not trained weights);
         --weights FILE.npz with arrays named like the layers replaces them later. Inputs: random UINT8 patches.

    x2c sim layers --sim                      # Ubuntu terminal (needs BMsemi_Model, Icarus, the chip repo)
    x2c sim layers --sim --layer features_0_0 --positions 4
    x2c sim layers --shape 16,16,1,1,1         # a layer shape without BMsemi (for testing)
Evidence: Python model (4-bit, ASSUMED cell rule) + RTL simulation with the BM behavioral X2 (1-bit); NOT silicon.
"""
from x2c import paths
import argparse
import copy
import json
import shutil
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = paths.ROOT

import x2c.commands.sim_engine2 as e2  # noqa: E402
import x2c.commands.sim_engine1 as ec  # noqa: E402
from x2c.chip.x2_golden import X2Config  # noqa: E402
from x2c.compiler import chip_x2  # noqa: E402
from x2c.compiler import x2_pack as xp  # noqa: E402
from x2c.compiler.ir import MATRIX_OPS  # noqa: E402
from x2c.compiler.partition import load_hardware  # noqa: E402

OUT = ROOT / "chip_build" / "x2_layer_e2e"


def layers_from_ir():
    from x2c.compiler.ir_builder import build_ir_from_compiler_graph
    from x2c.inputs.bmsemi_source import ReadOnlyRepo
    from x2c.inputs.normalize import compiler_v2_workload
    r = ReadOnlyRepo()
    g = build_ir_from_compiler_graph(r, compiler_v2_workload(r))
    return {n.name: {k: n.attrs[k] for k in ("K", "C", "R", "S", "groups")} for n in g.nodes if n.op in MATRIX_OPS}


def data(name, a, positions, weights=None):
    g, taps, outs = chip_x2.layer_shape(a)
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    W = weights[name] if weights is not None and name in weights.files else rng.integers(-128, 128, size=(a["K"], a["C"] // g, a["R"], a["S"]))
    X = rng.integers(0, 256, size=(positions, g, taps))
    return np.asarray(W, dtype=np.int64), X


def run_sim(pkg, X, cfg, layout, out, tag, repo):
    scripts, entries, costs, nacc = e2.queues(pkg, X, layout)
    paths = []
    for b in (0, 1):
        p = out / f"{tag}_{layout}_b{b}.script"
        p.write_text(scripts[b])
        paths.append(p)
    vvp, log = out / f"{tag}.vvp", out / f"{tag}_{layout}.log"
    params = [f"-Px2_engine2_tb.{n}={v}" for n, v in zip(ec.PARAMS, cfg.as_list())] + ["-Px2_engine2_tb.CELL_BITS=1"]
    from x2c.chip.chip_repo import INPUT_FILES
    r = subprocess.run(["iverilog", "-g2012", "-o", str(vvp), *params, str(repo.path(INPUT_FILES["x2_behavioral"])),
                        *map(str, e2.SOURCES)], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("iverilog failed:\n" + r.stderr[-2000:])
    log.unlink(missing_ok=True)
    subprocess.run(["vvp", "-n", str(vvp), f"+SCRIPT0={paths[0]}", f"+SCRIPT1={paths[1]}", f"+LOG={log}"],
                   capture_output=True, text=True, timeout=7200)
    return (log.read_text() if log.is_file() else ""), entries, costs, nacc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true", help="run the 1-bit two-block engine in Icarus (Ubuntu)")
    ap.add_argument("--layer", action="append", help="layer name (repeat); default: every matrix layer up to --max-tiles")
    ap.add_argument("--shape", action="append", help="K,C,R,S,groups instead of the BMsemi graph (repeat)")
    ap.add_argument("--positions", type=int, default=2, help="output positions per layer (a slice of the layer)")
    ap.add_argument("--max-tiles", type=int, default=16, help="skip layers needing more 1-bit tiles in simulation")
    ap.add_argument("--weights", help=".npz with real weights, one array per layer name (K, C/groups, R, S)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    hw4 = load_hardware()
    hw1 = copy.deepcopy(hw4); hw1["x2"]["cell_bits"] = 1
    if a.shape:
        layers = {f"shape_{s.replace(',', '_')}": dict(zip(("K", "C", "R", "S", "groups"), map(int, s.split(",")))) for s in a.shape}
    else:
        layers = layers_from_ir()
        if a.layer:
            unknown = [n for n in a.layer if n not in layers]
            if unknown:
                raise SystemExit(f"unknown layer(s) {unknown}; matrix layers: {sorted(layers)}")
            layers = {n: layers[n] for n in a.layer}
    weights = np.load(a.weights) if a.weights else None
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    repo = before = None
    if a.sim:
        for tool in ("iverilog", "vvp"):
            if not shutil.which(tool):
                raise SystemExit(f"{tool} not on PATH (Ubuntu terminal, chip toolchain)")
        from x2c.chip.chip_repo import ChipRepo
        repo = ChipRepo(); before = repo.tree_fingerprint()
    rows, ok = [], True
    print(f"{'layer':22s} {'K,C,R,S,g':>14s} {'4b tiles':>8s} {'4-bit exact':>11s} {'1b tiles':>8s} "
          f"{'one block':>10s} {'two blocks':>10s} {'cut':>6s} {'model err':>9s} {'sim exact':>9s}")
    for name, at in layers.items():
        W, X = data(name, at, a.positions, weights)
        ref = xp.reference(W, X, at)
        row = {"layer": name, "attrs": at, "positions": a.positions,
               "weights": "file" if weights is not None and name in weights.files else "random INT8 (seeded)"}
        p4 = xp.pack_layer(name, at, W, hw4)
        y4, _ = xp.run_layer(p4, X)
        row["bits4"] = {"tiles": p4.tiles, "exact": bool(np.array_equal(y4, ref)), "cell_rule": "ASSUMED sum(pwm*level)"}
        ok &= row["bits4"]["exact"]
        p1 = xp.pack_layer(name, at, W, hw1)
        row["bits1"] = {"tiles": p1.tiles}
        shape = ",".join(str(at[k]) for k in ("K", "C", "R", "S", "groups"))
        if p1.tiles > a.max_tiles:
            row["bits1"]["skipped"] = f"{p1.tiles} tiles > --max-tiles {a.max_tiles}"
            print(f"{name:22s} {shape:>14s} {p4.tiles:8d} {str(row['bits4']['exact']):>11s} {p1.tiles:8d}   skipped (--max-tiles)")
        elif a.sim:
            cfg = X2Config(tdc_time_out=p1.required_tdc_time_out)
            res = {}
            for layout in ("single", "alternate"):
                text, entries, costs, nacc = run_sim(p1, X, cfg, layout, out, name, repo)
                chk = e2.check(p1, X, ref, text, nacc)
                chk["predicted"] = e2.predicted(entries, costs)
                res[layout] = chk
            s, t = res["single"], res["alternate"]
            exact = s["pass"] and t["pass"]
            cut = round(100 * (1 - t["cycles"] / s["cycles"]), 1) if exact else None
            err = max(abs(r["predicted"] / r["cycles"] - 1) for r in res.values()) * 100 if exact else None
            row["bits1"].update(single=s, alternate=t, latency_cut_percent=cut, max_model_error_percent=None if err is None else round(err, 3))
            ok &= exact
            print(f"{name:22s} {shape:>14s} {p4.tiles:8d} {str(row['bits4']['exact']):>11s} {p1.tiles:8d} "
                  f"{s['cycles']:10d} {t['cycles']:10d} {str(cut) + ' %':>6s} {round(err, 3) if err is not None else '-':>8} % {str(exact):>9s}")
        else:
            print(f"{name:22s} {shape:>14s} {p4.tiles:8d} {str(row['bits4']['exact']):>11s} {p1.tiles:8d}   (add --sim to simulate)")
        rows.append(row)
    if a.sim:
        untouched = before == repo.tree_fingerprint()
        print(f"chip repo read-only: {untouched}")
        ok &= untouched
    (out / "report.json").write_text(json.dumps({"all_exact": ok, "rows": rows}, indent=2, default=str) + "\n")
    print(f"X2 LAYER END-TO-END: {'PASS' if ok else 'FAIL'}   (4-bit: Python, ASSUMED cell rule; 1-bit: RTL sim, BM X2; NOT silicon)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
