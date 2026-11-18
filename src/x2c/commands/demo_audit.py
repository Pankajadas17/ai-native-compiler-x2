"""V0 bitstream audit (S10) for a FABulous .bin.
   x2c demo audit --demo
   x2c demo audit --bin design.bin --fabric-csv fabric.csv [--fasm design.fasm --spec bitStreamSpec.csv|spec_reduced.json]
Exit code 0 = every error-level check passed (and the decode matched, if requested). Report: reports/bitstream_audit.json"""
from x2c import paths
import argparse, json, sys
from pathlib import Path
from x2c.verification.bitstream_audit import audit_against_fasm, audit_structure, load_spec, profile_from_fabric_csv

ap = argparse.ArgumentParser()
ap.add_argument("--demo", action="store_true", help="audit the bundled real FABulous demo bitstream")
ap.add_argument("--bin"); ap.add_argument("--fabric-csv"); ap.add_argument("--fasm"); ap.add_argument("--spec")
a = ap.parse_args()
if a.demo:
    d = paths.ROOT / "tests" / "fixtures" / "fabulous_demo"
    a.bin, a.fabric_csv, a.fasm, a.spec = (str(d / n) for n in ("design.bin", "fabric.csv", "design.fasm", "spec_reduced.json"))
if not (a.bin and a.fabric_csv):
    sys.exit("need --demo, or --bin and --fabric-csv")
data = Path(a.bin).read_bytes()
prof = profile_from_fabric_csv(Path(a.fabric_csv).read_text(), Path(a.bin).stem)
print(f"profile from fabric.csv: {prof.rows_per_frame} rows/frame x {prof.columns} columns x {prof.frames_per_col} frames = {prof.expected_frames} frames")
s = audit_structure(data, prof)
print("\nSTRUCTURAL AUDIT (V0)")
for c in s["checks"]:
    print(f"  {'PASS' if c['ok'] else ('WARN' if c['severity'] == 'warning' else 'FAIL')}  {c['id']:6} {c['title']}" + ("" if c["ok"] else f"\n          {c['detail'][:150]}"))
st = s["stats"]
print(f"  {st['bytes']} bytes, sha256 {st['sha256'][:16]}..., {st.get('frames', '?')} frames, {st.get('set_bits', '?')} set bits, {st.get('zero_frames', '?')} all-zero frames")
print(f"  note: rows_per_frame values under which the stream would also parse: {st['rows_per_frame_that_parse']} (consistency only; the profile must come from fabric.csv)")
out = {"structural": s}
ok = s["ok"]
if a.fasm and a.spec:
    d = audit_against_fasm(data, prof, Path(a.fasm).read_text(), load_spec(a.spec))
    out["semantic"] = d; ok = ok and d["ok"]
    print("\nSEMANTIC DECODE vs FASM")
    if "error" in d:
        print("  FAIL ", d["error"])
    else:
        print(f"  {'PASS' if d['ok'] else 'FAIL'}  bits set in .bin: {d['set_bits']} | required by FASM: {d['required_bits']} | "
              f"missing: {d['missing_count']} | unexplained: {d['unexplained_count']} | FASM features not in spec: {len(d['fasm_features_not_in_spec'])}")
        for m in d["missing"][:5]: print("   missing    ", m)
        for u in d["unexplained"][:5]: print("   unexplained", u)
    print(f"  layout assumption: {d.get('layout', {}).get('verified_on', '')}")
Path("reports").mkdir(exist_ok=True)
Path("reports/bitstream_audit.json").write_text(json.dumps(out, indent=2, default=str))
print("\nRESULT:", "PASS" if ok else "FAIL", "  [python-level audit of a file: NOT a hardware test; does not prove the fabric behaves correctly]")
sys.exit(0 if ok else 1)
