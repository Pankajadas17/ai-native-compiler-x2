"""Audit chip bitstreams (V0 / S10) against the chip's real frame layout and bitstream spec.

    x2c audit --all                      # every successful chip_build/<design>/
    x2c audit --design bitslice_acc_mux
    x2c audit --bit X.bit [--fasm X.fasm]

Works in either terminal (no hardware tools needed). Reads the chip repo, never writes to it.
"""
from x2c import paths
import argparse
import json
import sys
from pathlib import Path

ROOT = paths.ROOT

from x2c.chip.chip_bitstream import audit_chip_bitstream, chip_fabric_profile  # noqa: E402
from x2c.chip.chip_repo import ChipRepo  # noqa: E402

BUILD = ROOT / "chip_build"
PROFILE = ROOT / "specifications" / "chip_profile.json"


def row(name, r):
    s = r["summary"]
    dec = (f"{s['set_bits']}/{s['required_bits']} bits, missing {s['missing']}, unexplained {s['unexplained']}, "
           f"unparsed {s['unparsed_fasm_lines']}") if s["required_bits"] is not None else "no FASM: structure only"
    bad = f" failed {s['failed_checks']}" if s["failed_checks"] else ""
    return f"{name:24s} {'PASS' if r['ok'] else 'FAIL'}  structure {'ok' if s['structural_ok'] else 'BAD'}{bad}  {dec}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true")
    g.add_argument("--design")
    g.add_argument("--bit")
    ap.add_argument("--fasm")
    ap.add_argument("--build", default=str(BUILD))
    a = ap.parse_args(argv)

    repo = ChipRepo()
    before = repo.tree_fingerprint()
    chip_profile = json.loads(PROFILE.read_text()) if PROFILE.is_file() else None
    profile, spec = chip_fabric_profile(repo, chip_profile=chip_profile)
    print(f"profile: {profile.columns} columns x {profile.rows_per_frame} rows x {profile.frames_per_col} frames, "
          f"first row word = Y{profile.top_row_y}, exact size {profile.max_bytes} bytes (chip loader)")

    jobs = []
    if a.bit:
        jobs.append((Path(a.bit).stem, Path(a.bit), Path(a.fasm) if a.fasm else None, None))
    else:
        reports = sorted(Path(a.build).glob("*/flow_report.json"))
        for rp in reports:
            fr = json.loads(rp.read_text())
            if a.design and fr["design"] != a.design:
                continue
            if fr["status"] != "ok":
                continue
            d = rp.parent
            jobs.append((fr["design"], d / f"{fr['top']}.bit", d / f"{fr['top']}.fasm", d))
        if a.design and not jobs:
            raise SystemExit(f"no successful flow report for design {a.design} in {a.build}")

    ok = True
    for name, bit, fasm, folder in jobs:
        r = audit_chip_bitstream(bit.read_bytes(), fasm.read_text() if fasm else None, profile=profile, spec=spec)
        print(row(name, r))
        ok &= r["ok"]
        if folder is not None:
            (folder / "bitstream_audit.json").write_text(json.dumps(r, indent=2) + "\n")
    untouched = before == repo.tree_fingerprint()
    print(f"chip repo read-only: {untouched}")
    print(f"BITSTREAM AUDIT: {'PASS' if ok and untouched and jobs else 'FAIL'} ({len(jobs)} bitstreams)")
    return 0 if ok and untouched and jobs else 1


if __name__ == "__main__":
    sys.exit(main())
