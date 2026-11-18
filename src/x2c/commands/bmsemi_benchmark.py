"""x2c bmsemi benchmark [--list] [--workload ID]
Runs real BMsemi_Model workloads (read-only) through the verification pipeline and prints the stage table,
per-workload results, findings and the audit digest. Report: reports/benchmark_report.json"""
import argparse, json, sys
from x2c.inputs.bmsemi_source import ReadOnlyRepo
from x2c.inputs.catalog import discover
from x2c.inputs.normalize import all_workloads
from x2c.bmsemi.benchmark import run_benchmark

ap = argparse.ArgumentParser()
ap.add_argument("--list", action="store_true", help="only list what was discovered")
ap.add_argument("--workload", default=None, help="run a single workload id")
a = ap.parse_args()
repo = ReadOnlyRepo()
cat = discover(repo)
print(f"BMsemi_Model @ {cat['repo_commit'][:12]} (read-only): {len(cat['operations'])} operations, {len(cat['assets'])} cataloged assets, "
      f"corpora: " + ", ".join(f"{c['project'].rsplit('/', 1)[-1]} ({c['files']} files)" for c in cat["corpora"]))
wls = all_workloads(repo)
print("\nworkloads normalised to the local schema:")
for w in wls:
    print(f"  {w.id:34} {len(w.layers):>3} layers  data: {w.data_source}")
if a.list:
    sys.exit(0)
try:
    rep = run_benchmark(repo, [a.workload] if a.workload else None)
except ValueError as e:
    sys.exit(str(e))
print("\nstage table:")
for s in rep["stage_status"]:
    print(f"  {s['id']:8} {s['status']:16} {s['name']}")
print("\nresults:")
for w in rep["workloads"]:
    print(f"  {w['id']:34} {w['passed']:>3}/{w['tests']:<3} tests   [{w['data_source']}]")
bad = [r["test"] for r in rep["results"] if not r["passed"]]
if bad:
    print("  FAILED:", bad[:10])
for f in rep["findings"]:
    print(f"\nFINDING {f['id']}: {f['text']}" + (f"\n   layers: {f['layers']}" if f["layers"] else ""))
print(f"\n{rep['passed']}/{rep['total']} passed | read-only check: {rep['repo_read_only_check']}")
print(f"result digest (deterministic): {rep['result_digest_sha256']}")
print(f"[{rep['level']}] report: reports/benchmark_report.json")
