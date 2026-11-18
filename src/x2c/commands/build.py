"""Build a design for the real chip and report utilisation, pins and failures.

Ubuntu terminal (needs yosys, nextpnr-generic, make, python3 with fasm on PATH):
    x2c build --examples                         # all bundled examples
    x2c build --example bitslice_acc_mux
    x2c build --name my_op --top my_op path/to/my_op.v
Any terminal (print reports already written):
    x2c build --show

Outputs: chip_build/<name>/ (Makefile, design.pcf, logs, bitstream, flow_report.json).
"""
from x2c import paths
import argparse
import json
import sys
from pathlib import Path

ROOT = paths.ROOT

from x2c.chip.chip_flow import DEFAULT_OUT, run_flow  # noqa: E402
from x2c.chip.chip_repo import ChipRepo  # noqa: E402

PROFILE = ROOT / "specifications" / "chip_profile.json"
EX = ROOT / "hardware" / "designs"
TEST = ROOT / "hardware" / "test_designs"      # small and deliberately broken designs for the flow tests
# name -> (rtl source or list of sources, top, expected status, expected failure cause)
EXAMPLES = {
    "repo_counter": ("REPO:user_designs/designs/classic/counter/counter.v", "counter", "ok", None),
    "bitslice_acc_mux": (TEST / "bitslice_acc_mux.v", "bitslice_acc_mux", "ok", None),
    "bitslice_acc_parallel": (TEST / "bitslice_acc_parallel.v", "bitslice_acc_parallel", "failed", "pin_budget_exceeded"),
    "bitslice_acc_bad": (TEST / "bitslice_acc_bad.v", "bitslice_acc_bad", "failed", "syntax_error"),
    "missing_mod": (TEST / "missing_mod.v", "missing_mod", "failed", "unknown_module"),
    "macc_over": (TEST / "macc_over.v", "macc_over", "failed", "over_capacity"),
    "repo_reram_cim_32x32": ("REPO:user_designs/designs/classic/reram_cim_32x32/reram_cim_32x32.sv", "reram_cim_32x32", "ok", None),
    "repo_macc_8x8_20": ("REPO:user_designs/designs/classic/macc_8x8_20/macc_8x8_20.sv", "macc_8x8_20", "ok", None),
    "repo_ram_32x4_2r_1w": ("REPO:user_designs/designs/classic/ram_32x4_2r_1w/ram_32x4_2r_1w.sv", "ram_32x4_2r_1w", "ok", None),
    "repo_register_file": ("REPO:user_designs/designs/classic/register_file/register_file.sv", "register_file", "ok", None),
    "x2_vendor_cfg": (EX / "x2_vendor_cfg.sv", "x2_vendor_cfg", "ok", None),
    "x2_recombiner": (EX / "x2_recombiner.v", "x2_recombiner", "ok", None),
    "x2_bridge": (EX / "x2_bridge.v", "x2_bridge", "ok", None),
    "x2_engine": (EX / "x2_engine.v", "x2_engine", "ok", None),
    "x2_engine_4bit": ([EX / "x2_engine_4bit.v", EX / "x2_engine.v"], "x2_engine_4bit", "ok", None),
    "x2_engine2": (EX / "x2_engine2.v", "x2_engine2", "ok", None),
    "x2_engine2_4bit": ([EX / "x2_engine2_4bit.v", EX / "x2_engine2.v"], "x2_engine2_4bit", "ok", None),
}


def line(r):
    u = r.get("utilisation") or {}
    used = ", ".join(f"{k} {v['used']}/{v['available']}" for k, v in u.items() if v["used"])
    pins = r.get("pins")
    p = f"pins {pins['needed']}/{pins['available']}" if pins else "pins -"
    f = r.get("failure")
    why = f"{f['stage']}: {f['cause']}" if f else ""
    a = r.get("bitstream_audit")
    aud = f" | V0 {a['set_bits']}/{a['required_bits']} bits" if a and a.get("required_bits") is not None else ""
    return f"{r['design']:24s} {r['status']:7s} {p:12s} {why:32s} {used}{aud}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--examples", action="store_true")
    g.add_argument("--example", choices=sorted(EXAMPLES))
    g.add_argument("--show", action="store_true")
    g.add_argument("--name")
    ap.add_argument("--top")
    ap.add_argument("rtl", nargs="*")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    a = ap.parse_args(argv)
    out = Path(a.out)

    if a.show:
        reps = sorted(out.glob("*/flow_report.json"))
        for p in reps:
            print(line(json.loads(p.read_text())))
        return 0 if reps else 1

    if not PROFILE.is_file():
        raise SystemExit("specifications/chip_profile.json missing: run x2c chip profile first")
    profile = json.loads(PROFILE.read_text())
    repo = ChipRepo()
    jobs = []
    if a.name:
        if not (a.top and a.rtl):
            raise SystemExit("--name needs --top and at least one RTL file")
        jobs.append((a.name, [Path(x) for x in a.rtl], a.top, None, None))
    else:
        for n in (sorted(EXAMPLES) if a.examples else [a.example]):
            src, top, st, cause = EXAMPLES[n]
            srcs = src if isinstance(src, list) else [src]
            paths = [repo.path(x[5:]) if str(x).startswith("REPO:") else Path(x) for x in srcs]
            jobs.append((n, paths, top, st, cause))

    ok = True
    print(f"{'design':24s} {'status':7s} {'pins':12s} {'failure':32s} utilisation (used/available)")
    for name, rtl, top, want_status, want_cause in jobs:
        r = run_flow(name, rtl, top, profile, repo, out)
        print(line(r))
        if want_status is not None:
            got = (r["status"], (r["failure"] or {}).get("cause"))
            good = got == (want_status, want_cause)
            ok &= good
            if not good:
                print(f"{'':24s} UNEXPECTED: wanted {want_status}/{want_cause}")
        if not r["chip_repo_read_only"]:
            ok = False
    if a.examples or a.example:
        print(f"EXAMPLES AS EXPECTED: {ok}")
    print(f"reports: {out}{'/'}<design>/flow_report.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
