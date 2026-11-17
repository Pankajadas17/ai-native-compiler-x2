"""
Flow check (S05/S10): read the files a FABulous run left behind and decide ok / suspect / failed / inconclusive.
An exit code of 0 or "Compile flow completed successfully" is never trusted on its own.

    python run_flow_check.py --demo                       # the real run stored in tests/fixtures/real_flow_run
    python run_flow_check.py --job reports/fab_jobs/add8_sat   # a job folder made by run_fab_prep.py + run_fab_flow.sh
    python run_flow_check.py --nextpnr-log demo/user_design/x_npnr_log.txt --fasm demo/user_design/x.fasm \
        --bin demo/user_design/x.bin --fabric-csv demo/user_design/x.csv --exit-code 0 [--yosys-log y.log] [--flow-log f.txt]

Report: reports/flow_check.json. Evidence level: tool log and output file inspection; with a simulation log also
RTL simulation of the FABulous fabric with the bitstream loaded vs NumPy (NOT gate-level, NOT timing, NOT silicon).
"""
import argparse
import json
import sys
from pathlib import Path

from verification.bitstream_audit import profile_from_fabric_csv
from verification.fab_sim import parse_sim_log, tb_vector_count
from verification.flow_logs import check_flow


def _read(p):
    return Path(p).read_text(encoding="utf-8", errors="replace") if p else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--job", help="job folder from run_fab_flow.sh / run_fab_sim.sh (files are found by name)")
    ap.add_argument("--sim-log", help="console log of run_fab_sim.sh (found automatically with --job)")
    ap.add_argument("--tb", help="the generated testbench (found automatically with --job); gives the expected vector count")
    for n in ("yosys-log", "nextpnr-log", "flow-log", "fasm", "bin", "fabric-csv"):
        ap.add_argument(f"--{n}")
    ap.add_argument("--exit-code", type=int)
    a = ap.parse_args(argv)
    if a.demo:
        d = Path(__file__).parent / "tests" / "fixtures" / "real_flow_run"
        a.nextpnr_log, a.fasm, a.exit_code = str(d / "nextpnr_log.txt"), str(d / "design.fasm"), 0
        a.flow_log_text = "INFO | Compile flow completed successfully."
    if a.job:
        j = Path(a.job)
        op = next((p.name[:-len("_fab.v")] for p in j.glob("*_fab.v")), None)
        if op is None:
            sys.exit(f"no *_fab.v in {j}: run run_fab_prep.py first")
        for attr, name in (("nextpnr_log", f"{op}_fab_npnr_log.txt"), ("fasm", f"{op}_fab.fasm"), ("bin", f"{op}_fab.bin"),
                           ("fabric_csv", "fabric.csv"), ("flow_log", "flow_console.log")):
            if (j / name).exists():
                setattr(a, attr, str(j / name))
        if (j / "sim_console.log").exists():
            a.sim_log = str(j / "sim_console.log")
        tb = j / f"{op}_fab_tb.v"
        if tb.exists():
            a.tb = str(tb)
        if (j / "exit_code.txt").exists():
            a.exit_code = int((j / "exit_code.txt").read_text().strip())
        if a.flow_log:
            a.yosys_log = a.flow_log          # FABulous prints Yosys output in its console; errors there are still caught
    flow = getattr(a, "flow_log_text", None) or _read(a.flow_log)
    data = Path(a.bin).read_bytes() if a.bin else None
    prof = profile_from_fabric_csv(_read(a.fabric_csv), Path(a.bin).stem) if (a.bin and a.fabric_csv) else None
    r = check_flow(exit_code=a.exit_code, yosys_log=_read(a.yosys_log), nextpnr_log=_read(a.nextpnr_log), flow_log=flow,
                   fasm_text=_read(a.fasm), bitstream=data, profile=prof)
    sim = None
    if a.sim_log:
        expected = tb_vector_count(_read(a.tb)) if a.tb else None
        sim = parse_sim_log(_read(a.sim_log), expected)
        r["facts"]["fabric_simulation"] = sim
        if sim["status"] == "fail":
            r["findings"].append({"stage": "fabric_sim", "severity": "error", "code": "fabric_sim_mismatch", "message": sim["reason"], "evidence": sim["mismatch_lines"]})
            r["status"] = "failed"
        elif sim["status"] == "inconclusive":
            r["findings"].append({"stage": "fabric_sim", "severity": "suspect", "code": "fabric_sim_inconclusive", "message": sim["reason"], "evidence": sim.get("tool_errors", [])})
            if r["status"] in ("ok", "inconclusive"):
                r["status"] = "suspect"
    else:
        r["not_checked"].append("fabric_simulation")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/flow_check.json").write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
    f = r["facts"]
    print(f"FLOW STATUS: {r['status'].upper()}")
    print(f"  tool claimed success: {f['tool_claimed_success']} | FASM features: {f.get('fasm_features')} | "
          f"FABULOUS_LC: {f.get('utilisation', {}).get('FABULOUS_LC')} | fmax: {f.get('fmax')}")
    if sim:
        print(f"FABRIC SIMULATION: {sim['status'].upper()} - {sim['reason']}")
    if "bitstream_sha256" in f:
        print(f"  bitstream: sha256 {f['bitstream_sha256'][:16]}... set bits {f['bitstream_set_bits']}")
    for x in r["findings"][:10]:
        print(f"  [{x['severity']}] {x['code']}: {x['message']}")
    if r["not_checked"]:
        print("  not checked (so not claimed):", ", ".join(r["not_checked"]))
    print(f"  {r['limits']}\n[{r['evidence_level']}] report: reports/flow_check.json")
    return 0 if r["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
