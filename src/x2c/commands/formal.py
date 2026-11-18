"""
Formal checks and lint (review item 4). Ubuntu terminal, with the chip toolchain on PATH:

    x2c formal                       # everything (a few minutes)       also: x2c formal
    x2c formal --part spec,netlist   # only the operator proofs (seconds)
    x2c formal --part engine --depth 14

Parts (see verification/formal.py):
  spec     A2 operators == independent specs for EVERY input; a broken copy must give a counterexample
  netlist  chip-synthesized operators == their RTL for EVERY input (repo primitive models); the repo's own multiplier map
           must give a counterexample for requant_i32_i8 (why chip/techmap/MACC_8x8_20_map_signed.v exists)
  lint     Verilator -Wall: no errors, none of the serious warning classes
  engine   SymbiYosys BOUNDED check of the two-block engine (1-bit and 4-bit) against an unconstrained X2; a copy without
           the pipeline's forwarding must fail
Report: reports/formal/report.json; work folders: chip_build/formal/.
"""
from x2c import paths
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = paths.ROOT

from x2c.verification import formal as F  # noqa: E402

PARTS = ("spec", "netlist", "lint", "engine")
REPORT = ROOT / "reports" / "formal" / "report.json"
WORK = ROOT / "chip_build" / "formal"


def row(name, expect, got, detail=""):
    ok = got == expect
    print(f"  {'PASS' if ok else 'FAIL'}  {name:46s} expected {expect:14s} got {got:14s} {detail}")
    return {"check": name, "expected": expect, "got": got, "ok": ok, "detail": detail}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default=",".join(PARTS), help=f"comma list of {', '.join(PARTS)}")
    ap.add_argument("--depth", type=int, default=10, help="engine: clocks after reset to check (bounded)")
    ap.add_argument("--solver", default="bitwuzla", help="engine: SMT solver for SymbiYosys (bitwuzla, boolector, yices)")
    a = ap.parse_args(argv)
    parts = [p.strip() for p in a.part.split(",") if p.strip()]
    bad = [p for p in parts if p not in PARTS]
    if bad:
        ap.error(f"unknown part(s) {bad}")
    from x2c.chip import chip_flow as cf
    from x2c.chip.chip_repo import ChipRepo
    env = cf.tool_env()
    need = {"spec": ["yosys"], "netlist": ["yosys", "make"], "lint": ["verilator"], "engine": ["sby", "yosys-smtbmc", a.solver]}
    missing = sorted({t for p in parts for t in need[p] if not shutil.which(t, path=env["PATH"])})
    if missing:
        print(f"tools not found: {missing}. Run in the Ubuntu terminal after your chip toolchain export line.")
        return 2
    repo = ChipRepo()
    before = repo.tree_fingerprint()
    rows, t0 = [], time.time()
    if "spec" in parts:
        print("SPEC: A2 operators vs independent specs, every input (Yosys SAT)")
        for op in F.OPS:
            rtl = (ROOT / "hardware" / "operators" / f"{op}.v").read_text(encoding="utf-8")
            r = F.prove_spec(op, rtl, WORK / "spec" / op, env)
            rows.append(row(f"spec {op}", "proved", r["result"], f"{r['seconds']} s"))
            old, new = F.MUTANTS[op]
            r = F.prove_spec(op, rtl.replace(old, new, 1), WORK / "spec" / f"{op}_mutant", env)
            rows.append(row(f"spec {op} (broken copy: {old} -> {new})", "counterexample", r["result"],
                            json.dumps(r.get("values", {}))))
    if "netlist" in parts:
        print("NETLIST: chip synthesis of each operator vs its RTL, every input (repo primitive models)")
        for op in F.OPS:
            rtl = ROOT / "hardware" / "operators" / f"{op}.v"
            net = F.synth_core(op, rtl, WORK / "netlist" / op / "synth", env, repo)
            r = F.prove_netlist(op, rtl, net, WORK / "netlist" / op / "prove", env, repo)
            rows.append(row(f"netlist {op} (signed multiplier map)", "proved", r["result"], f"{r['seconds']} s"))
        op = "requant_i32_i8"
        net = F.synth_core(op, ROOT / "hardware" / "operators" / f"{op}.v", WORK / "netlist" / "repo_map" / "synth", env, repo,
                           multiplier=(str(repo.path(F.REPO_MULTIPLIER)), "8 8 2 2 12"))
        r = F.prove_netlist(op, ROOT / "hardware" / "operators" / f"{op}.v", net, WORK / "netlist" / "repo_map" / "prove", env, repo)
        rows.append(row(f"netlist {op} (the REPO's multiplier map)", "counterexample", r["result"], json.dumps(r.get("values", {}))))
    if "lint" in parts:
        print("LINT: Verilator -Wall")
        for top, files in F.LINT_DESIGNS.items():
            r = F.lint(top, files, WORK / "lint" / top, env, repo)
            detail = (f"errors {r['errors']} serious {r['serious']}" if r["result"] != "clean"
                      else (", ".join(f"{k} {v}" for k, v in sorted(r["warnings"].items())) or "no warnings"))
            rows.append(row(f"lint {top}", "clean", r["result"], detail))
    if "engine" in parts:
        print(f"ENGINE: SymbiYosys bounded check, {a.depth} clocks after reset (this is the slow part)")
        for cb in (1, 4):
            r = F.engine_bmc(cb, a.depth, WORK / "engine" / f"bit{cb}", env, solver=a.solver)
            rows.append(row(f"engine x2_engine2 CELL_BITS={cb} (P1-P3)", "pass", r["result"], f"{r['seconds']} s"))
        r = F.engine_bmc(4, a.depth, WORK / "engine" / "mutant_no_forwarding", env, mutate=True, solver=a.solver)
        rows.append(row("engine without forwarding (broken copy)", "fail", r["result"],
                        f"{r['seconds']} s, failed {', '.join(r['failed_asserts']) or '-'}, trace {r['trace']}"))
    unchanged = repo.tree_fingerprint() == before
    ok = all(r["ok"] for r in rows) and unchanged
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "parts": parts,
                                  "engine_depth": a.depth if "engine" in parts else None, "ok": ok, "rows": rows,
                                  "chip_repo_unchanged": unchanged, "seconds": round(time.time() - t0, 1),
                                  "evidence_level": F.EVIDENCE}, indent=1), encoding="utf-8")
    print(f"\nFORMAL {'PASS' if ok else 'FAIL'}: {sum(r['ok'] for r in rows)}/{len(rows)} checks as expected"
          f"{'' if unchanged else ' (CHIP REPO CHANGED!)'}   report: {REPORT.as_posix()}")
    if "engine" in parts:
        print(f"  engine result is BOUNDED: every X2 behaviour that keeps the handshake rules, for {a.depth} clocks after reset")
    print(f"[{F.EVIDENCE}]")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
