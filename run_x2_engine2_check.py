"""
Two-block X2 engine (examples/chip_designs/x2_engine2.v): does using both X2 blocks really cut the latency the schedule
predicts? Each layer of flow step 4 runs twice on the SAME design, with one command queue per block driven at once:
  single:     every tile on block 0 (block 1 idle)                  -- what the one-block engine can do
  alternate:  tile i on block i % 2, so one block programs while the other computes
Both runs must reproduce every accumulator (the engine rule) and NumPy exactly. The cycles are MEASURED in the BM behavioral
X2 model and compared with compiler/x2_schedule.simulate (the latency model A1 uses), fed with this run's exact command counts.

    python3 run_x2_engine2_check.py --sim         # Ubuntu terminal (Icarus + the chip repo); without --sim: the scripts only
Evidence level: RTL simulation with the BM behavioral X2 (1-bit cells), NOT the configured fabric, NOT silicon.
"""
import argparse
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import run_x2_engine_check as ec  # noqa: E402
from chip.chip_repo import INPUT_FILES, ChipRepo  # noqa: E402
from compiler import chip_x2  # noqa: E402
from compiler import x2_pack as xp  # noqa: E402
from compiler import x2_schedule as xs  # noqa: E402
from compiler.partition import load_hardware  # noqa: E402

OUT = ROOT / "chip_build" / "x2_engine2_sim"
SOURCES = [ROOT / "sim" / "RERAM_CIM_32x32_sim.v", ROOT / "examples" / "chip_designs" / "x2_engine2.v",
           ROOT / "sim" / "x2_engine2_tb.v"]
LAYOUTS = {"single": lambda ti: 0, "alternate": lambda ti: ti % 2}


def queues(pkg, X, layout):
    """Per-block scripts, the issue order [(tile, block)] and per-tile costs from the exact commands sent."""
    nacc = pkg.hw["x2"]["cols"] // chip_x2.cells_per_weight(pkg.hw)
    n_slices = -(-8 // pkg.input_split_bits)
    lines, state, costs, entries = {0: [], 1: []}, {0: np.zeros_like(pkg.cells[0]), 1: np.zeros_like(pkg.cells[0])}, [], []
    for ti in range(pkg.tiles):
        b = LAYOUTS[layout](ti)
        prog = list(xp.program_commands(pkg, ti, state[b]))
        state[b] = pkg.cells[ti].copy()
        q = lines[b]
        q += ["B %d %d %d %d %d %d" % (m, r, c, p, int(l), int(a)) for m, r, c, p, l, a in prog]
        mvms = beats = 0
        for p in range(X.shape[0]):
            q.append("C")
            for si in range(n_slices):
                q.append(f"P {si}")
                cb = list(xp.compute_beats(pkg, ti, X[p], si))
                q += ["B %d %d %d %d %d %d" % (m, r, c, pw, int(l), int(a)) for m, r, c, pw, l, a in cb]
                beats += len(cb)
                mvms += sum(1 for x in cb if x[4])
            q.append(f"A {nacc} {ti} {p}")
        costs.append({"program": xs.PROGRAM_CMD * len(prog), "compute": xs.MVM * mvms + xs.BEAT * beats})
        entries.append(("L", ti, b))
    return {b: "\n".join(v) + "\n" for b, v in lines.items()}, entries, {"L": costs}, nacc


def predicted(entries, costs):
    return xs.simulate(costs, {"L": []}, entries, [], 2)["latency_cycles"]


def parse(text, nacc):
    got = {}
    for line in text.splitlines():
        f = line.split()
        if f and f[0] == "ACC":
            b, ti, p, k, v = map(int, f[1:])
            got.setdefault((p, ti), [0] * nacc)[k] = v
    done = next((l.split()[1:] for l in text.splitlines() if l.startswith("DONE ")), None)
    return got, done


def check(pkg, X, ref, text, nacc):
    got, done = parse(text, nacc)
    order = ec.read_order(pkg, X)
    accs = [got.get(o) for o in order]
    _, _, counts = xp.run_layer(pkg, X, return_counts=True)
    model_ok = None not in accs and accs == ec.engine_model_by_order(pkg, X, counts, order)
    y = ec.host_rebuild(pkg, X, order, accs, ec.engine_layout(pkg)) if None not in accs else None
    numpy_ok = y is not None and bool(np.array_equal(y, ref))
    return {"pass": bool(done) and model_ok and numpy_ok, "model_match": model_ok, "numpy_exact": numpy_ok,
            "cycles": int(done[2]) if done else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    print(f"{'layer':5s} {'tiles':>5s} {'layout':9s} {'measured':>9s} {'predicted':>9s} {'model err':>9s}  exact")
    for k, pkg, X, ref, cfg in ec.cases(hw1, a.positions, a.seed):
        res = {}
        for layout in LAYOUTS:
            scripts, entries, costs, nacc = queues(pkg, X, layout)
            log = out / f"engine2_{k}_{layout}.log"
            if a.sim:
                paths = []
                for b in (0, 1):
                    pth = out / f"engine2_{k}_{layout}_b{b}.script"
                    pth.write_text(scripts[b])
                    paths.append(pth)
                vvp = out / f"engine2_{k}.vvp"
                params = [f"-Px2_engine2_tb.{n}={v}" for n, v in zip(ec.PARAMS, cfg.as_list())] + ["-Px2_engine2_tb.CELL_BITS=1"]
                r = subprocess.run(["iverilog", "-g2012", "-o", str(vvp), *params, str(repo.path(INPUT_FILES["x2_behavioral"])),
                                    *map(str, SOURCES)], capture_output=True, text=True)
                if r.returncode:
                    raise SystemExit("iverilog failed:\n" + r.stderr[-2000:])
                subprocess.run(["vvp", "-n", str(vvp), f"+SCRIPT0={paths[0]}", f"+SCRIPT1={paths[1]}", f"+LOG={log}"],
                               capture_output=True, text=True, timeout=3600)
            if not log.is_file():
                ok = False
                print(f"{k:5s} {layout}: no log (run with --sim in the Ubuntu terminal)")
                continue
            row = check(pkg, X, ref, log.read_text(), nacc)
            row.update(layer=k, layout=layout, tiles=pkg.tiles, predicted=predicted(entries, costs))
            row["model_error_percent"] = None if not row["cycles"] else round(100 * (row["predicted"] / row["cycles"] - 1), 2)
            rows.append(row)
            res[layout] = row
            ok &= row["pass"]
            print(f"{k:5s} {pkg.tiles:5d} {layout:9s} {str(row['cycles']):>9s} {row['predicted']:9d} "
                  f"{str(row['model_error_percent']) + ' %':>9s}  {'yes' if row['pass'] else 'NO'}")
        if len(res) == 2 and res["single"]["cycles"] and res["alternate"]["cycles"]:
            cut = round(100 * (1 - res["alternate"]["cycles"] / res["single"]["cycles"]), 1)
            pcut = round(100 * (1 - res["alternate"]["predicted"] / res["single"]["predicted"]), 1)
            print(f"{'':5s} {'':5s} two blocks: measured latency cut {cut} %, predicted {pcut} %")
            res["alternate"]["measured_cut_percent"], res["alternate"]["predicted_cut_percent"] = cut, pcut
    if a.sim:
        untouched = before == repo.tree_fingerprint()
        print(f"chip repo read-only: {untouched}")
        ok &= untouched
    (out / "report.json").write_text(json.dumps({"all_pass": ok, "rows": rows}, indent=2) + "\n")
    print(f"X2 TWO-BLOCK ENGINE CHECK: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
