"""Create a small, self-contained audit fixture from a FABulous project that has been compiled.
   python tools/extract_bitstream_fixture.py <project_dir> <design_name> <out_dir> [--note TEXT]
Copies the .bin, .fasm, fabric.csv and the nextpnr log, and writes a REDUCED spec (only the features the FASM uses)
as JSON so the fixture stays small. A MANIFEST.json records provenance and sha256 of every file."""
import argparse, hashlib, json, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verification.bitstream_audit import load_spec, parse_fasm

ap = argparse.ArgumentParser(); ap.add_argument("project"); ap.add_argument("design"); ap.add_argument("out")
ap.add_argument("--note", default=""); ap.add_argument("--tool", action="append", default=[], help="name=version (repeatable)")
a = ap.parse_args()
proj, out = Path(a.project), Path(a.out); out.mkdir(parents=True, exist_ok=True)
ud = proj / "user_design"
shutil.copy(ud / f"{a.design}.bin", out / "design.bin"); shutil.copy(ud / f"{a.design}.fasm", out / "design.fasm")
shutil.copy(proj / "fabric.csv", out / "fabric.csv")
log = ud / f"{a.design}_npnr_log.txt"
if log.exists(): shutil.copy(log, out / "nextpnr_log.txt")
spec = load_spec(str(proj / ".FABulous" / "bitStreamSpec.csv"))
feats, _ = parse_fasm((ud / f"{a.design}.fasm").read_text())
keep = {}
for tile, base, rng, val in feats:
    names = [base] if rng is None else [f"{base}[{rng[1] + k}]" for k in range(rng[0] - rng[1] + 1)]
    for n in names + ([n[:-3] for n in names if n.endswith("[0]")]):
        if n in spec.get(tile, {}): keep.setdefault(tile, {})[n] = {str(k): v for k, v in spec[tile][n].items()}
(out / "spec_reduced.json").write_text(json.dumps(keep, sort_keys=True))
files = ["design.bin", "design.fasm", "fabric.csv", "spec_reduced.json"] + (["nextpnr_log.txt"] if log.exists() else [])
(out / "MANIFEST.json").write_text(json.dumps({
    "what": "FABulous demo design compiled to a real bitstream", "design": a.design, "note": a.note,
    "tools": dict(t.split("=", 1) for t in a.tool), "spec_reduced": "only features referenced by the FASM (full spec is ~9 MB)",
    "files": {f: {"size": (out / f).stat().st_size, "sha256": hashlib.sha256((out / f).read_bytes()).hexdigest()} for f in files}}, indent=2))
print("fixture written to", out, files)
