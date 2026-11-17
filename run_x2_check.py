"""Check the X2 golden model against the BM Labs X2 behavioral model (Icarus Verilog).

Ubuntu terminal (needs iverilog/vvp on PATH):
    python3 run_x2_check.py --sim
Any terminal (re-compare logs already in the output folder, no simulator):
    python run_x2_check.py

Writes only to the output folder (default chip_build/x2_check). The chip repo is
read (the behavioral model source is passed to iverilog) and fingerprinted before
and after; the run fails if anything in it changed.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from chip.chip_repo import INPUT_FILES, ChipRepo  # noqa: E402
from chip.x2_check import all_scenarios, compare, write_results, write_vectors  # noqa: E402

TB = ROOT / "sim" / "x2_vector_tb.sv"
OUT = ROOT / "chip_build" / "x2_check"


def simulate(repo, out, scenarios):
    for tool in ("iverilog", "vvp"):
        if not shutil.which(tool):
            raise SystemExit(f"{tool} not found on PATH (run this in the Ubuntu terminal after the export line)")
    model = repo.path(INPUT_FILES["x2_behavioral"])
    vvp = out / "x2_tb.vvp"
    cmd = ["iverilog", "-g2012", "-o", str(vvp), str(model), str(TB)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    (out / "compile.log").write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        raise SystemExit(f"iverilog failed (see {out / 'compile.log'}):\n{r.stderr[-2000:]}")
    timings = {}
    for sc in scenarios:
        t0 = time.time()
        r = subprocess.run(["vvp", "-n", str(vvp), f"+VECTORS={out / (sc.name + '.vec')}",
                            f"+LOG={out / (sc.name + '.sim.txt')}"], capture_output=True, text=True)
        (out / (sc.name + ".vvp.txt")).write_text(r.stdout + r.stderr)
        timings[sc.name] = round(time.time() - t0, 1)
        print(f"  simulated {sc.name:22s} {timings[sc.name]:6.1f} s")
    return timings


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true", help="run Icarus (Ubuntu); otherwise only compare")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--repo")
    a = ap.parse_args(argv)

    repo = ChipRepo(a.repo)
    before = repo.tree_fingerprint()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scenarios = all_scenarios()
    for sc in scenarios:
        write_vectors(out / (sc.name + ".vec"), sc)
        write_results(out / (sc.name + ".expected.txt"), sc.expected())

    timings = simulate(repo, out, scenarios) if a.sim else {}

    reports = []
    for sc in scenarios:
        log = out / (sc.name + ".sim.txt")
        if not log.is_file():
            reports.append({"scenario": sc.name, "pass": False,
                            "bench_problems": ["no simulator log: run with --sim in the Ubuntu terminal"]})
            continue
        rep = compare(sc, log.read_text())
        rep["sim_seconds"] = timings.get(sc.name)
        reports.append(rep)

    untouched = before == repo.tree_fingerprint()
    summary = {"chip_repo_read_only": untouched, "source": repo.provenance()["input_digest"],
               "all_pass": untouched and all(r["pass"] for r in reports), "scenarios": reports}
    (out / "report.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"{'scenario':22s} {'beats':>6s} {'results':>15s} {'cycles':>8s}  verdict")
    for r in reports:
        res = f"{r.get('simulated_results', '-')}/{r.get('expected_results', '-')}"
        print(f"{r['scenario']:22s} {r.get('beats', '-'):>6} {res:>15s} {str(r.get('cycles', '-')):>8s}  "
              f"{'PASS' if r['pass'] else 'FAIL'}")
        n = r.get("numeric_check")
        if n:
            print(f"{'':22s} numeric: {n['what']}: {n['ok']} ({n['values']} values, max {n['max_value']})")
        for m in r.get("mismatches", [])[:3]:
            print(f"{'':22s} result #{m['index']}: expected {m['expected']} | simulated {m['simulated']}")
        for p in r.get("bench_problems", [])[:3]:
            print(f"{'':22s} problem: {p}")
    print(f"chip repo read-only: {untouched}")
    print(f"X2 CHECK: {'PASS' if summary['all_pass'] else 'FAIL'}  (report: {out / 'report.json'})")
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
