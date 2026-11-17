"""
cleanup_move.py: move the old backup copies out of bm_verif, so that only the project remains. Nothing is deleted.

    python tools/cleanup_move.py              # VS Code terminal: show what WOULD move (changes nothing)
    python tools/cleanup_move.py --apply      # move them to ..\\bm_verif_to_delete\\ (same sub-folders)
    python tools/cleanup_move.py --restore    # put everything back exactly where it was

What moves (decided by tools/inventory.py, category downloads_and_backups):
  * backup_*/ folders               (copies the overlays kept before changing files)
  * *.before_* files                (the same, single files)
  * *.bak, "file (1).py", *_old ... (copies)
  * the old top-level inventory.py   (replaced by tools/inventory.py)
What is relocated (kept, but not part of the code repository):
  * *.docx at the top level          -> the folder ABOVE bm_verif (team documents do not go to GitHub)
Never touched: code and data any command or test uses, tests/, config, docs, generated outputs (chip_build/, reports/,
deploy/: keep them, they are git-ignored later), BMsemi_Model/, sky130-reram-fpga/, venv/, .venv/.
The holding folder is OUTSIDE bm_verif, so pytest and imports cannot see the moved files. After both terminals' tests
pass, delete it:  Remove-Item -Recurse -Force ..\\bm_verif_to_delete
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import inventory as inv  # noqa: E402

ROOT = inv.ROOT
HOLD = ROOT.parent / f"{ROOT.name}_to_delete"
MANIFEST = HOLD / "MANIFEST.json"


def plan():
    files, _, _ = inv.walk(ROOT)
    rows, _, _ = inv.classify(files)
    moves = set()
    for r in rows:
        p = r["path"]
        if r["category"] == "downloads_and_backups":
            top = p.split("/", 1)[0]
            moves.add(top if "/" in p and top.startswith("backup") else p)          # whole backup folder at once
    if (ROOT / "inventory.py").is_file() and (HERE / "inventory.py").is_file() and HERE != ROOT:
        moves.add("inventory.py")
    relocate = sorted(f["path"] for f in files if "/" not in f["path"] and f["path"].lower().endswith(".docx"))
    zips = sorted(r["path"] for r in rows if r["path"].lower().endswith(tuple(inv.ARCHIVE_EXT)))
    moves -= set(zips)                                    # archives are only listed: you may still need them
    return sorted(moves), relocate, zips


def size(rel):
    p = ROOT / rel
    return p.stat().st_size if p.is_file() else inv.dir_size(p)


def apply(moves, relocate):
    HOLD.mkdir(exist_ok=True)
    done = json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else []
    for rel in moves:
        src, dst = ROOT / rel, HOLD / rel
        if dst.exists():
            sys.exit(f"stop: {dst} already exists (an earlier --apply?); --restore first or remove it by hand")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        done.append({"from": rel, "to": str(dst), "when": time.strftime("%Y-%m-%d %H:%M")})
    for rel in relocate:
        src, dst = ROOT / rel, ROOT.parent / rel
        if dst.exists():
            print(f"  keep {rel}: {dst} already exists, not moved")
            continue
        shutil.move(str(src), str(dst))
        done.append({"from": rel, "to": str(dst), "when": time.strftime("%Y-%m-%d %H:%M"), "relocated": True})
    MANIFEST.write_text(json.dumps(done, indent=1))


def restore():
    if not MANIFEST.is_file():
        sys.exit(f"nothing to restore: {MANIFEST} not found")
    done = json.loads(MANIFEST.read_text())
    for m in reversed(done):
        src, dst = Path(m["to"]), ROOT / m["from"]
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            print(f"  restored {m['from']}")
    MANIFEST.unlink()
    print("restored everything listed in the manifest")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--apply", action="store_true")
    g.add_argument("--restore", action="store_true")
    a = ap.parse_args(argv)
    if a.restore:
        return restore()
    moves, relocate, zips = plan()
    total = sum(size(m) for m in moves)
    print(f"{'MOVE' if a.apply else 'WOULD MOVE'} to {HOLD}  ({len(moves)} items, {total / 1e6:.2f} MB)")
    for m in moves:
        print(f"  {m}{'/' if (ROOT / m).is_dir() else ''}")
    for r in relocate:
        print(f"{'RELOCATE' if a.apply else 'WOULD RELOCATE'} (kept, outside the repo): {r} -> {ROOT.parent / r}")
    for z in zips:
        print(f"  archive left in place (delete by hand if applied already): {z}")
    if not moves and not relocate:
        print("nothing to move: the folder is clean")
        return 0
    if a.apply:
        apply(moves, relocate)
        print(f"\ndone. Now run the tests in BOTH terminals. If all pass: Remove-Item -Recurse -Force {HOLD}"
              f"\nIf something broke: python tools/cleanup_move.py --restore")
    else:
        print("\nnothing changed (dry run). To move them: python tools/cleanup_move.py --apply")
    return 0


if __name__ == "__main__":
    sys.exit(main())
