"""
prepush_check.py: run before every push. It reads what git WOULD commit and refuses when something must not go to
GitHub. It changes nothing in the folder except, with --pin, specifications/external_repos.json.

    git init                                 # once (inside the project folder)
    python tools/prepush_check.py --pin      # records which commits of the two cloned repos this work was tested on
    python tools/prepush_check.py            # before every push: must end with "PRE-PUSH CHECK PASSED"

ERRORS (exit 1, do not push):
  * an API key or secret in a file that would be committed (Groq gsk_..., Google AIza..., sk-..., key = "...")
  * a .env file, a cloned repo (BMsemi_Model, sky130-reram-fpga), a virtual environment, chip_build/, deploy/,
    generated reports or caches that would be committed (.gitignore is missing or wrong)
  * a file of 95 MB or more (GitHub refuses files over 100 MB)
WARNINGS (push allowed, read them):
  * third-party imports that pyproject.toml does not list, files over 20 MB, .docx/.pdf, no .gitattributes
"""
import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent if _HERE.name == "tools" else _HERE
CLONES = ("BMsemi_Model", "sky130-reram-fpga")
FORBIDDEN = CLONES + ("venv", ".venv", "env", "chip_build", "deploy", "__pycache__", ".pytest_cache")
REPORTS_ALLOWED = ()                                  # reports/ is generated output only (A2's operators: hardware/operators/)
BINARY_EXT = {".bit", ".bin", ".npz", ".npy", ".png", ".jpg", ".pdf", ".docx", ".zip", ".vvp", ".pyc"}
SECRETS = [
    ("Groq API key", re.compile(r"gsk_[A-Za-z0-9]{20,}")),
    ("Google API key", re.compile(r"AIza[0-9A-Za-z_\-]{30,}")),
    ("OpenAI-style key", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}")),
    ("hard-coded secret", re.compile(r"""(?i)\b(api[_-]?key|secret|token|password)\b\s*[:=]\s*["']([^"'\s]{16,})["']""")),
]
PLACEHOLDER = re.compile(r"(?i)(your|example|dummy|fake|test|xxx|<|\.\.\.|changeme|placeholder)")
IMPORT_NAME = {"google-antigravity": "google", "google-genai": "google", "google-generativeai": "google", "pyyaml": "yaml", "scikit-learn": "sklearn",
               "opencv-python": "cv2", "pillow": "PIL", "beautifulsoup4": "bs4", "python-dotenv": "dotenv"}


def git(*args):
    r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr.strip() or f"git {' '.join(args)} failed")
    return r.stdout


def would_commit():
    out = git("ls-files", "--cached", "--others", "--exclude-standard", "-z")
    return sorted(p for p in out.split("\0") if p)


def forbidden(p):
    parts = p.split("/")
    if any(x in FORBIDDEN for x in parts) or any(x.endswith("_to_delete") for x in parts):
        return "cloned repo / environment / generated folder"
    if parts[0] == "reports" and not p.startswith(REPORTS_ALLOWED):
        return "generated report"
    if parts[-1] == ".env" or parts[-1].endswith(".env"):
        return "environment file (holds keys)"
    if parts[-1] in ("inventory.json", "inventory_report.txt"):
        return "inventory output (generated)"
    return None


def scan_secrets(paths):
    hits = []
    for p in paths:
        f = ROOT / p
        if f.suffix.lower() in BINARY_EXT or not f.is_file() or f.stat().st_size > 5_000_000:
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            for what, rx in SECRETS:
                m = rx.search(line)
                if m and not PLACEHOLDER.search(m.group(0)):
                    v = m.group(m.lastindex or 0)
                    hits.append(f"{p}:{n}: {what} ({v[:4]}...{v[-2:]})")
    return hits


def third_party(paths):
    local = {p.split("/", 1)[0].removesuffix(".py") for p in paths} | {Path(p).stem for p in paths if p.endswith(".py")} \
        | set(CLONES) | {"tests", "conftest", "x2c"}
    std = set(getattr(sys, "stdlib_module_names", ())) | {"__future__"}
    found = {}
    for p in paths:
        if not p.endswith(".py"):
            continue
        try:
            tree = ast.parse((ROOT / p).read_text(encoding="utf-8", errors="ignore"))
        except (SyntaxError, OSError):
            continue
        for node in ast.walk(tree):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module] if isinstance(node, ast.ImportFrom) and node.module and not node.level else []
            for m in mods:
                top = m.split(".")[0]
                if top not in std and top not in local:
                    found.setdefault(top, p)
    listed = set()
    import tomllib
    pp = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = list(pp.get("project", {}).get("dependencies", []))
    for group in (*pp.get("project", {}).get("optional-dependencies", {}).values(), *pp.get("dependency-groups", {}).values()):
        deps += [d for d in group if isinstance(d, str)]
    for line in deps:
        name = re.split(r"[<>=!~\[;\s]", line.strip(), maxsplit=1)[0].lower()
        if name:
            listed |= {name, name.replace("-", "_"), IMPORT_NAME.get(name, name)}
    return {m: p for m, p in sorted(found.items()) if m.lower() not in listed}


def pin():
    out = {"note": "the cloned repos are NOT in this repository; clone these commits next to the code (read-only)"}
    for c in CLONES:
        d = ROOT / c
        if not (d / ".git").exists():
            out[c] = {"status": "not a git clone here"}
            continue
        g = lambda *a: subprocess.run(["git", "-C", str(d), *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
        out[c] = {"remote": g("remote", "get-url", "origin"), "commit": g("rev-parse", "HEAD"),
                  "submodules": [l.strip() for l in g("submodule", "status").splitlines() if l.strip()],
                  "local_changes": bool(g("status", "--porcelain", "--untracked-files=no"))}
    dst = ROOT / "specifications" / "external_repos.json"
    dst.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"pinned: {dst.relative_to(ROOT).as_posix()}")
    for c in CLONES:
        info = out[c]
        print(f"  {c}: {info.get('commit', info.get('status'))}" + ("   (LOCAL CHANGES in the clone!)" if info.get("local_changes") else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pin", action="store_true", help="write specifications/external_repos.json first")
    a = ap.parse_args(argv)
    try:
        git("rev-parse", "--is-inside-work-tree")
    except (RuntimeError, FileNotFoundError):
        print("not a git repository yet (or git is not installed): run  git init  inside the project folder first")
        return 1
    if a.pin:
        pin()
    paths = would_commit()
    errors, warnings = [], []
    bad = [(p, why) for p in paths if (why := forbidden(p))]
    groups = {}
    for p, why in bad:
        groups.setdefault((p.split("/", 1)[0] + "/" if "/" in p else p, why), 0)
        groups[(p.split("/", 1)[0] + "/" if "/" in p else p, why)] += 1
    errors += [f"would commit {k} ({n} files): {why}" for (k, why), n in sorted(groups.items())]
    errors += [f"secret: {h}" for h in scan_secrets([p for p in paths if not forbidden(p)])]
    sizes = {p: (ROOT / p).stat().st_size for p in paths if (ROOT / p).is_file()}
    errors += [f"too large for GitHub: {p} ({s / 1e6:.0f} MB)" for p, s in sizes.items() if s >= 95e6]
    warnings += [f"large file: {p} ({s / 1e6:.0f} MB)" for p, s in sizes.items() if 20e6 <= s < 95e6]
    warnings += [f"document in the repository (team material?): {p}" for p in paths if p.lower().endswith((".docx", ".pdf"))]
    if not (ROOT / ".gitattributes").is_file():
        warnings.append("no .gitattributes: Windows and Ubuntu may commit different line endings")
    if not (ROOT / "specifications" / "external_repos.json").is_file():
        warnings.append("cloned repos not pinned yet: run with --pin")
    for m, p in third_party([p for p in paths if not forbidden(p)]).items():
        warnings.append(f"import '{m}' (first in {p}) is not a dependency in pyproject.toml")
    total = sum(sizes.get(p, 0) for p in paths if not forbidden(p))
    print(f"git would commit {len(paths)} files ({total / 1e6:.1f} MB)")
    for w in warnings:
        print(f"  WARNING {w}")
    for e in errors:
        print(f"  ERROR   {e}")
    print("PRE-PUSH CHECK " + ("FAILED: fix the errors above, nothing was committed" if errors else "PASSED"))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
