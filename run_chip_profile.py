"""Derive specifications/chip_profile.json from the chip repo (read-only).

  python run_chip_profile.py            # write/refresh the profile, print a summary
  python run_chip_profile.py --check    # exit 1 if the saved profile is stale or missing
  python run_chip_profile.py --repo PATH

The chip repo is never written to: a fingerprint of every file in it is taken
before and after, and the run fails if they differ.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from chip.chip_repo import ChipRepo  # noqa: E402
from chip.fabric_profile import build_chip_profile  # noqa: E402

OUT = ROOT / "specifications" / "chip_profile.json"


def dumps(p):
    return json.dumps(p, indent=2, sort_keys=False) + "\n"


def summary(p):
    s, f, x2, io = p["source"], p["fabric"], p["x2"], p["io"]
    bm = x2["behavioral_model"]
    lines = [
        f"chip repo      : {s['repo']} @ {(s['commit'] or 'unknown')[:12]}",
        *[f"  submodule    : {k} @ {(v or 'NOT CHECKED OUT')[:12]}" for k, v in s["submodules"].items()],
        f"input digest   : {s['input_digest'][:16]}",
        f"fabric grid    : {f['grid']['columns']} columns x {f['grid']['rows']} rows",
        "resources      : " + ", ".join(f"{k} {v}" for k, v in f["resources"].items()),
        f"general pins   : {len(io['general_pins'])} ({io['general_pins'][0]} .. {io['general_pins'][-1]}) on {io['iobuf_sites']} IOBUF sites",
        f"X2             : {x2['count']} x {x2['primitive']} at {', '.join(x2['sites'])}",
        f"X2 array       : {x2['rows']} rows x {x2['cols']} cols, PWM {x2['pwm_bits']} bits, result {x2['result_bits']} bits",
        "X2 result     : " + (" ".join(f"{r['field']}[{r['width']}]" for r in bm["result_format_msb_first"]) if bm["result_format_msb_first"] else "format not found"),
        "X2 modes       : " + ", ".join(f"{k}={v}" for k, v in bm["modes"].items()),
        f"X2 config bits : {x2['config_bits_per_block']} per block ({len(x2['parameters'])} parameters)",
    ]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)

    repo = ChipRepo(a.repo)
    before = repo.tree_fingerprint()
    profile = build_chip_profile(repo)
    after = repo.tree_fingerprint()
    untouched = before == after
    print(summary(profile))
    print(f"chip repo read-only: {untouched} ({after['files']} files)")
    if not untouched:
        print("FAIL: the chip repo changed during the run")
        return 2

    out = Path(a.out)
    text = dumps(profile)
    if a.check:
        if not out.is_file():
            print(f"FAIL: {out} does not exist; run without --check first")
            return 1
        same = out.read_text(encoding="utf-8") == text
        print(f"saved profile up to date: {same}")
        return 0 if same else 1
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
