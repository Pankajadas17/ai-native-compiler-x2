"""
export_public.py: make the PUBLIC copy of this project in a separate folder. This repository is never changed.

    python tools/export_public.py ..\\ai-native-compiler-x2-public          # VS Code terminal (about 5 min: runs the tests)
    python tools/export_public.py ..\\ai-native-compiler-x2-public --skip-tests

What it does, in order (any failure stops it and the folder is not ready to push):
  1 copies every file git tracks here, EXCEPT the private ones (PRIVATE below: the team's model graphs)
  2 if the folder is a clone of the public repo, its old files are replaced (its .git history is kept)
  3 checks the copy: no private path, no fixture carrying the team repo's data, no API key or token
  4 runs the whole test suite IN THE COPY, where BMsemi_Model does not exist: it must pass in public mode
Then it prints the git commands that publish the folder.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

PRIVATE = ("tests/fixtures/ir/",)                    # the team's model graphs (copied from BMsemi_Model)
PRIVATE_MARKERS = ("repo_commit", "Neuromorphic_compiler_V2/artifacts")   # BMsemi data inside a fixture
NEVER = ("BMsemi_Model/", "sky130-reram-fpga/", "venv/", ".venv/", "chip_build/", "deploy/")


def tracked():
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"], capture_output=True, text=True, check=True).stdout
    return sorted(p for p in out.split("\0") if p)


def dirty():
    return subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout.strip()


def prepare(target):
    if target.resolve() == ROOT or ROOT in target.resolve().parents:
        sys.exit("the public copy must be OUTSIDE this project folder")
    if target.exists():
        items = [p for p in target.iterdir() if p.name != ".git"]
        if items and not (target / ".git").exists():
            sys.exit(f"{target} is not empty and not a git clone: pick an empty folder or a clone of the public repo")
        for p in items:                                   # replace the previous export, keep the public repo's history
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    target.mkdir(parents=True, exist_ok=True)


def check(target, files):
    from tools.prepush_check import SECRETS, PLACEHOLDER, BINARY_EXT
    problems = []
    for rel in files:
        if rel.startswith(PRIVATE) or rel.startswith(NEVER):
            problems.append(f"private path exported: {rel}")
        f = target / rel
        if f.suffix.lower() in BINARY_EXT or f.stat().st_size > 5_000_000:
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")
        if rel.startswith("tests/fixtures/") and any(m in text for m in PRIVATE_MARKERS):
            problems.append(f"fixture carries the team repo's data: {rel}")
        for n, line in enumerate(text.splitlines(), 1):
            for what, rx in SECRETS:
                m = rx.search(line)
                if m and not PLACEHOLDER.search(m.group(0)):
                    problems.append(f"{what} in {rel}:{n}")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="folder for the public copy (empty, or a clone of the public repo)")
    ap.add_argument("--skip-tests", action="store_true", help="do not run the test suite in the copy")
    a = ap.parse_args(argv)
    target = Path(a.target)
    if dirty():
        print("note: this folder has uncommitted changes; they are exported as they are on disk")
    prepare(target)
    files = [f for f in tracked() if not f.startswith(PRIVATE) and (ROOT / f).is_file()]
    skipped = [f for f in tracked() if f.startswith(PRIVATE)]
    for rel in files:
        dst = target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)
    print(f"1-2 copied {len(files)} files to {target}  (left out, private: {', '.join(skipped) or 'none'})")
    problems = check(target, files)
    if problems:
        print("3   CHECK FAILED: the copy is NOT ready to publish")
        for p in problems:
            print(f"      {p}")
        return 1
    print("3   check passed: no private path, no private fixture data, no keys or tokens")
    if not a.skip_tests:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        chip = ROOT / "sky130-reram-fpga"
        if chip.is_dir() and "BM_CHIP_REPO" not in env:
            env["BM_CHIP_REPO"] = str(chip)               # read-only, as everywhere else
        print("4   running the test suite in the copy (public mode: no BMsemi_Model there) ...", flush=True)
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=target, env=env,
                           capture_output=True, text=True)
        lines = [l for l in (r.stdout + r.stderr).splitlines() if l.strip()]
        print(f"    {lines[-1] if lines else 'no output'}   (the BMsemi-only and private-graph test files are not collected)")
        if r.returncode != 0:
            (target.parent / "export_public_tests.log").write_text(r.stdout + r.stderr, encoding="utf-8")
            print(f"    TESTS FAILED in the copy: NOT ready to publish (log: {target.parent / 'export_public_tests.log'})")
            return 1
    print("\nREADY TO PUBLISH. In the copy folder:")
    if not (target / ".git").exists():
        print(f"  cd {target}\n  git init\n  git branch -M main\n  git remote add origin https://github.com/Pankajadas17/ai-native-compiler-x2.git")
    else:
        print(f"  cd {target}")
    print('  git add -A\n  git commit -m "Public release: <what changed>"\n  git push -u origin main')
    return 0


if __name__ == "__main__":
    sys.exit(main())
