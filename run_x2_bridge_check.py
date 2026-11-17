"""Flow step 3 check: the X2 controller (host bridge) on the eFPGA.

A host drives examples/chip_designs/x2_bridge.v through its 30 pins only; the bridge feeds the chip's X2 primitive,
simulated by sim/RERAM_CIM_32x32_sim.v = the BM Labs X2 behavioral model wired exactly as src/fabric_wrapper.sv does.
The command streams are flow step 1's packed layers (1-bit cells: the only cells the chip's X2 commands can program).

Ubuntu terminal:   python3 run_x2_bridge_check.py --sim
Any terminal:      python run_x2_bridge_check.py          (re-checks logs already in chip_build/x2_bridge/)
Per layer it checks: (1) every result equals the X2 golden model, (2) the results are identical to flow step 1's
direct-to-X2 run, (3) the rebuilt layer output equals NumPy. Writes only to chip_build/x2_bridge_sim/.
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

import run_x2_pack_check as pc  # noqa: E402
from chip.chip_repo import INPUT_FILES, ChipRepo  # noqa: E402
from chip.x2_check import Scenario, compare, parse_log, write_vectors  # noqa: E402
from chip.x2_golden import X2Config  # noqa: E402
from compiler import x2_pack as xp  # noqa: E402
from compiler.partition import load_hardware  # noqa: E402

OUT = ROOT / "chip_build" / "x2_bridge_sim"
SOURCES = [ROOT / "sim" / "RERAM_CIM_32x32_sim.v", ROOT / "examples" / "chip_designs" / "x2_bridge.v",
           ROOT / "sim" / "x2_bridge_tb.v"]
DIRECT = ROOT / "tests" / "fixtures" / "x2_pack"
PARAMS = ("TARGET_SET1", "TARGET_SET2", "TARGET_RESET1", "TARGET_RESET2", "NO_OF_CLK_CYCLES", "COUNTER_VALUE",
          "TDC_TIME_OUT", "TDC_DEAD_TIME")


def layer_cases(hw1):
    for i, k in enumerate(pc.LAYERS):
        a, W, X = pc.make_data(k, 2, 2026 + i)                 # the same data as flow step 1's recorded runs
        pkg = xp.pack_layer(k, a, W, hw1)
        beats, plan = pc.stream_for_sim(pkg, X)
        sc = Scenario(f"bridge_{k}", "packed layer through the bridge", X2Config(tdc_time_out=pkg.required_tdc_time_out), beats)
        yield k, pkg, X, xp.reference(W, X, a), sc, plan


def check_log(k, pkg, X, ref, sc, plan, log_text, cols):
    rep = compare(sc, log_text)
    results, _, done = parse_log(log_text)
    same_as_direct = None
    direct = DIRECT / f"pack_{k}.sim.txt"
    if direct.is_file():
        same_as_direct = [l for l in log_text.splitlines() if l.startswith("R ")] == \
                         [l for l in direct.read_text().splitlines() if l.startswith("R ")]
    try:
        y = pc.rebuild_from_results(pkg, X, plan, results, cols)
        numpy_ok = bool(np.array_equal(y, ref))
    except ValueError:
        numpy_ok = False
    ok = rep["pass"] and numpy_ok and same_as_direct is not False
    return {"layer": k, "pass": ok, "golden_match": rep["pass"], "same_as_direct": same_as_direct, "numpy_exact": numpy_ok,
            "beats": len(sc.beats), "results": len(results), "mismatches": rep["mismatches"][:3], "problems": rep["bench_problems"][:3]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    hw1 = copy.deepcopy(load_hardware()); hw1["x2"]["cell_bits"] = 1
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    repo = before = None
    if a.sim:
        for tool in ("iverilog", "vvp"):
            if not shutil.which(tool):
                raise SystemExit(f"{tool} not on PATH (Ubuntu terminal, after the export line)")
        repo = ChipRepo()
        before = repo.tree_fingerprint()
    rows, ok = [], True
    print(f"{'layer':5s} {'beats':>6s} {'results':>7s}  {'= X2 golden':11s} {'= direct X2 run':15s} {'output = NumPy':14s}")
    for k, pkg, X, ref, sc, plan in layer_cases(hw1):
        log = out / f"bridge_{k}.log"
        if a.sim:
            vec = out / f"bridge_{k}.vec"
            write_vectors(vec, sc)
            vvp = out / f"bridge_{k}.vvp"
            params = [f"-Px2_bridge_tb.{n}={v}" for n, v in zip(PARAMS, sc.cfg.as_list())]
            r = subprocess.run(["iverilog", "-g2012", "-o", str(vvp), *params, str(repo.path(INPUT_FILES["x2_behavioral"])),
                                *map(str, SOURCES)], capture_output=True, text=True)
            if r.returncode:
                raise SystemExit("iverilog failed:\n" + r.stderr[-2000:])
            subprocess.run(["vvp", "-n", str(vvp), f"+VECTORS={vec}", f"+LOG={log}"], capture_output=True, text=True)
        if not log.is_file():
            rows.append({"layer": k, "pass": False, "problems": ["no log: run with --sim in the Ubuntu terminal"]})
            ok = False
            print(f"{k:5s} no log (run --sim in Ubuntu)")
            continue
        row = check_log(k, pkg, X, ref, sc, plan, log.read_text(), hw1["x2"]["cols"])
        rows.append(row)
        ok &= row["pass"]
        yn = lambda v: "-" if v is None else ("yes" if v else "NO")
        print(f"{k:5s} {row['beats']:6d} {row['results']:7d}  {yn(row['golden_match']):11s} {yn(row['same_as_direct']):15s} {yn(row['numpy_exact']):14s}")
    if a.sim:
        untouched = before == repo.tree_fingerprint()
        print(f"chip repo read-only: {untouched}")
        ok &= untouched
    (out / "report.json").write_text(json.dumps({"all_pass": ok, "rows": rows}, indent=2) + "\n")
    print(f"X2 BRIDGE CHECK: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
