"""
inventory.py: a READ-ONLY map of the bm_verif folder: which files the project really uses, which are generated,
and which nothing refers to. It deletes, moves and changes nothing.

    python tools/inventory.py      # VS Code terminal, inside bm_verif
Writes inventory_report.txt (short, to read or share) and inventory.json (every file, with the reason).

How "used" is decided:
  roots   = every command (run_*.py, deploy_chip.py, check_*.py, other top-level .py with a __main__ block)
            and every test (tests/test_*.py, conftest.py)
  a file is USED if a root reaches it through
     - a Python import (also imports inside functions), or
     - its name written in a used file (e.g. "x2_engine2_tb.v", "run_fab_chain.py", "verification.tools"),
       which is how Verilog, JSON, fixtures and scripts started by subprocess are found
  tests/fixtures/<folder>/ is used when a used file names <folder>.
Categories: used_code, used_data, generated (rebuilt by a command), docs, unused_code, unused_data,
            downloads_and_backups (zips, copies, *_old, *.bak ...), external_clones (never touched).
"Unused" means nothing refers to it: a candidate to review, not an automatic delete.
"""
import ast
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent if _HERE.name == "tools" else _HERE
EXTERNAL = {"BMsemi_Model", "sky130-reram-fpga"}                       # read-only clones: listed, never analysed
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "venv", ".venv", "env", ".env", "node_modules", ".mypy_cache",
             ".ruff_cache", ".ipynb_checkpoints", ".idea", ".vscode"}
GENERATED_DIRS = {"chip_build", "reports", "deploy", "build", "out", "dist", "sim_build", "_to_delete"}
GENERATED_EXT = {".vvp", ".log", ".fasm", ".bit", ".bin", ".hex", ".vcd", ".fst", ".pyc", ".jsonl"}
BACKUP_PAT = re.compile(r"(\(\d+\)|[ _-]copy\b|[ _-]old\b|_backup|\.bak$|\.orig$|\.rej$|~$|\.tmp$|^tmp_|_v\d+\.py$)", re.I)
ARCHIVE_EXT = {".zip", ".7z", ".rar", ".tar", ".gz", ".tgz"}
CONFIG_NAMES = {".gitignore", ".gitattributes", "pytest.ini", "setup.cfg", "pyproject.toml", "tox.ini", "conftest.py",
                ".env.example", "LICENSE"}


def is_backup(path):
    """Copies kept by earlier overlays: backup_*/ folders, *.before_*, *.bak, 'file (1).py' ..."""
    name = path.rsplit("/", 1)[-1]
    return path.split("/", 1)[0].startswith("backup") or ".before_" in name or bool(BACKUP_PAT.search(name))
NAME_TOKEN = re.compile(r"[A-Za-z0-9_\-.]+\.(?:py|v|sv|vh|json|jsonl|md|txt|pcf|csv|npz|npy|yaml|yml|toml|ini|cfg|sh|ps1|tcl|xdc)\b")
DOTTED = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b")
WORD = re.compile(r"[A-Za-z0-9_\-]+")


def walk(root):
    files, external, skipped = [], {}, defaultdict(int)
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        keep = []
        for d in sorted(dirnames):
            if rel_dir == Path(".") and d in EXTERNAL:
                external[d] = dir_size(Path(dirpath) / d)
            elif d in SKIP_DIRS:
                skipped[d] += dir_size(Path(dirpath) / d)
            else:
                keep.append(d)
        dirnames[:] = keep
        for f in filenames:
            p = Path(dirpath) / f
            try:
                st = p.stat()
            except OSError:
                continue
            files.append({"path": p.relative_to(root).as_posix(), "bytes": st.st_size,
                          "modified": time.strftime("%Y-%m-%d", time.localtime(st.st_mtime))})
    return files, external, dict(skipped)


def dir_size(p):
    total = 0
    for dp, _, fns in os.walk(p):
        for f in fns:
            try:
                total += (Path(dp) / f).stat().st_size
            except OSError:
                pass
    return total


def module_index(files):
    """dotted module name -> path, for every .py we own (packages and top-level scripts)."""
    idx = {}
    for f in files:
        p = f["path"]
        if not p.endswith(".py"):
            continue
        parts = p[:-3].split("/")
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if parts:
            idx[".".join(parts)] = p
    return idx


def read(path):
    try:
        return (ROOT / path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def py_imports(path, text, idx):
    """Local files imported by this module (absolute and relative imports, anywhere in the file)."""
    out = set()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return out, True
    pkg = path[:-3].split("/")[:-1]
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                up = pkg[:len(pkg) - node.level + 1] if node.level > 1 else pkg
                base = ".".join(up + ([base] if base else []))
            names = [base] + [f"{base}.{a.name}" if base else a.name for a in node.names]
        for n in names:
            parts = n.split(".")
            for i in range(len(parts), 0, -1):          # the module itself and every parent package (__init__.py runs)
                m = ".".join(parts[:i])
                if m in idx:
                    out.add(idx[m])
                sib = ".".join(pkg + parts[:i])          # a module next to this file (tests/ helpers, scripts' sys.path)
                if pkg and sib in idx:
                    out.add(idx[sib])
    return out, False


def is_root(f, text):
    p = f["path"]
    name = p.rsplit("/", 1)[-1]
    if is_backup(p) or not p.endswith(".py"):
        return False
    if p.startswith("tests/") and (name.startswith("test_") or name == "conftest.py"):
        return True
    return "/" not in p or p.startswith("tools/")        # every top-level script and every tool is a command


def classify(files):
    idx = module_index(files)
    by_name = defaultdict(list)
    for f in files:
        if not is_backup(f["path"]):
            by_name[f["path"].rsplit("/", 1)[-1]].append(f["path"])
    fixture_dirs = defaultdict(list)                      # tests/fixtures/<folder>/... -> files
    for f in files:
        parts = f["path"].split("/")
        if len(parts) > 3 and parts[0] == "tests" and parts[1] == "fixtures":
            fixture_dirs[parts[2]].append(f["path"])
    files_live = [f for f in files if not is_backup(f["path"])]
    texts = {f["path"]: read(f["path"]) for f in files_live
             if f["path"].endswith((".py", ".v", ".sv", ".json", ".md", ".txt", ".toml", ".cfg", ".ini", ".sh", ".ps1", ".tcl"))
             and f["bytes"] < 2_000_000}
    why, syntax_errors = {}, []
    queue = [f["path"] for f in files if is_root(f, texts.get(f["path"], ""))]
    for r in queue:
        why[r] = "root: a command or a test"
    roots = list(queue)
    seen = set()
    while queue:
        p = queue.pop()
        if p in seen:
            continue
        seen.add(p)
        text = texts.get(p, "")
        if not text:
            continue
        found = set()
        if p.endswith(".py"):
            imps, bad = py_imports(p, text, idx)
            if bad:
                syntax_errors.append(p)
            found |= {(q, f"imported by {p}") for q in imps}
            for m in set(DOTTED.findall(text)):            # "verification.tools", importlib strings
                if m in idx:
                    found.add((idx[m], f"named as module in {p}"))
        if p.endswith((".py", ".v", ".sv", ".json", ".toml", ".cfg", ".ini", ".sh", ".ps1", ".tcl")):
            for tok in set(NAME_TOKEN.findall(text)):
                for q in by_name.get(tok.rsplit("/", 1)[-1], []):
                    found.add((q, f"named in {p}"))
            if p.startswith("tests/") or "fixture" in text:
                words = set(WORD.findall(text))
                for d, members in fixture_dirs.items():
                    if d in words:
                        found |= {(q, f"fixture folder '{d}' named in {p}") for q in members}
        for q, reason in found:
            if q != p and q not in why:
                why[q] = reason
                queue.append(q)
    rows = []
    for f in files:
        p, name = f["path"], f["path"].rsplit("/", 1)[-1]
        top = p.split("/", 1)[0] if "/" in p else ""
        ext = Path(name).suffix.lower()
        if is_backup(p):
            cat, reason = "downloads_and_backups", "backup or copy (backup_*/ folder, .before_*, .bak, (1) ...)"
        elif name in CONFIG_NAMES or (name.startswith("requirements") and ext == ".txt"):
            cat, reason = "config", "project configuration: keep"
        elif top != "tests" and (top in GENERATED_DIRS or ext in GENERATED_EXT):
            cat = "generated"
            reason = f"in {top}/ (made by a command)" if top in GENERATED_DIRS else f"{ext} output file"
        elif ext in ARCHIVE_EXT:
            cat, reason = "downloads_and_backups", "archive"
        elif p in why:
            cat, reason = ("used_code" if ext == ".py" else "used_data"), why[p]
        elif ext in (".md", ".diff", ".rst") or top == "docs":
            cat, reason = "docs", "documentation (nothing needs it to run)"
        elif ext == ".py":
            cat, reason = "unused_code", "no command or test imports or names it"
        else:
            cat, reason = "unused_data", "no used file names it"
        rows.append({**f, "category": cat, "reason": reason})
    return rows, roots, syntax_errors


def mb(n):
    return f"{n / 1e6:9.2f} MB"


def report(rows, roots, syntax_errors, external, skipped):
    by = defaultdict(list)
    for r in rows:
        by[r["category"]].append(r)
    order = ["used_code", "used_data", "config", "generated", "docs", "unused_code", "unused_data", "downloads_and_backups"]
    L = [f"bm_verif inventory  {time.strftime('%Y-%m-%d %H:%M')}  root={ROOT}", "(read-only: nothing was changed)", "",
         "SUMMARY"]
    for c in order:
        L.append(f"  {c:24s} {len(by[c]):6d} files {mb(sum(r['bytes'] for r in by[c]))}")
    for d, n in sorted(external.items()):
        L.append(f"  external clone {d + '/':17s}              {mb(n)}   (never touched)")
    for d, n in sorted(skipped.items()):
        L.append(f"  skipped {d + '/':24s}              {mb(n)}   (environments / caches)")
    tops = defaultdict(lambda: [0, 0])
    for r in rows:
        t = r["path"].split("/", 1)[0] if "/" in r["path"] else "(top level)"
        tops[t][0] += 1; tops[t][1] += r["bytes"]
    L += ["", "TOP-LEVEL FOLDERS (files, size)"] + [f"  {t:28s} {n:6d} {mb(s)}" for t, (n, s) in sorted(tops.items(), key=lambda kv: -kv[1][1])]
    L += ["", f"ROOTS: {len(roots)} commands/tests"] + [f"  {p}" for p in sorted(r for r in roots if "/" not in r)]
    L.append(f"  + {sum(1 for r in roots if '/' in r)} test files")
    for c, title in (("unused_code", "UNUSED CODE (no command or test reaches it)"),
                     ("unused_data", "UNUSED DATA (no used file names it)"),
                     ("downloads_and_backups", "DOWNLOADS / COPIES / BACKUPS")):
        items = sorted(by[c], key=lambda r: r["path"])
        L += ["", f"{title}: {len(items)}"] + [f"  {r['path']:70s} {r['bytes']:>10d} B  {r['modified']}" for r in items[:300]]
        if len(items) > 300:
            L.append(f"  ... {len(items) - 300} more in inventory.json")
    gen = defaultdict(lambda: [0, 0])
    for r in by["generated"]:
        k = "/".join(r["path"].split("/")[:2]) if "/" in r["path"] else r["path"]
        gen[k][0] += 1; gen[k][1] += r["bytes"]
    L += ["", "GENERATED (rebuildable), by folder"] + [f"  {k:50s} {n:6d} {mb(s)}" for k, (n, s) in sorted(gen.items(), key=lambda kv: -kv[1][1])[:60]]
    L += ["", "DOCS"] + [f"  {r['path']}" for r in sorted(by["docs"], key=lambda r: r["path"])]
    L += ["", "USED CODE"] + [f"  {r['path']}" for r in sorted(by["used_code"], key=lambda r: r["path"])]
    big = sorted(rows, key=lambda r: -r["bytes"])[:25]
    L += ["", "25 LARGEST FILES"] + [f"  {r['path']:70s} {mb(r['bytes'])}  [{r['category']}]" for r in big]
    if syntax_errors:
        L += ["", "PYTHON FILES THAT DO NOT PARSE (their imports were not followed)"] + [f"  {p}" for p in syntax_errors]
    return "\n".join(L) + "\n"


def main():
    files, external, skipped = walk(ROOT)
    files = [f for f in files if f["path"] not in ("inventory_report.txt", "inventory.json")]
    rows, roots, bad = classify(files)
    (ROOT / "inventory.json").write_text(json.dumps({"root": str(ROOT), "external": external, "skipped": skipped,
                                                     "roots": roots, "files": rows}, indent=1), encoding="utf-8")
    text = report(rows, roots, bad, external, skipped)
    (ROOT / "inventory_report.txt").write_text(text, encoding="utf-8")
    print(text.split("\nTOP-LEVEL")[0])
    print("written: inventory_report.txt (send this) and inventory.json (send this too if it is under a few MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
