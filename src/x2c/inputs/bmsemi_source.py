"""
ReadOnlyRepo: the ONLY way our code touches the BMsemi_Model clone.

* There is no write method. Every path is resolved inside the repo root (no '..' escapes).
* Identity for auditing: git commit + per-file size + sha256 of the content with CRLF normalised to LF
  (Windows checkouts convert line endings, which would otherwise make hashes differ between machines).
* snapshot() / assert_unchanged() let a run PROVE it did not modify the files it consumed.
"""
import hashlib
import json
import subprocess
from pathlib import Path

from x2c import paths

TEXT_SUFFIXES = {".json", ".txt", ".md", ".py", ".m", ".log", ".csv", ".yaml", ".yml", ".toml"}
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache"}
HASH_LIMIT = 8 * 1024 * 1024          # files above this are identified by size only (e.g. model binaries)


class RepoAccessError(Exception):
    pass


class ReadOnlyRepo:
    def __init__(self, root=None):
        root = Path(root) if root else paths.ROOT / "BMsemi_Model"
        self.root = root.resolve()
        if not (self.root / "catalog.json").exists():
            raise FileNotFoundError(f"BMsemi_Model not found at {self.root} (expected catalog.json inside). "
                                    f"Clone it into the project folder; it is used read-only.")

    # ---- path safety -------------------------------------------------------------------
    def path(self, rel):
        p = (self.root / rel).resolve()
        if self.root != p and self.root not in p.parents:
            raise RepoAccessError(f"path escapes the repository: {rel}")
        return p

    # ---- read-only access --------------------------------------------------------------
    def exists(self, rel):
        try:
            return self.path(rel).exists()
        except RepoAccessError:
            return False

    def read_bytes(self, rel):
        return self.path(rel).read_bytes()

    def read_text(self, rel, encoding="utf-8"):
        return self.path(rel).read_text(encoding=encoding)

    def read_json(self, rel):
        return json.loads(self.read_text(rel))

    def list_dir(self, rel=""):
        return sorted(x.name for x in self.path(rel).iterdir() if x.name not in SKIP_DIRS)

    def walk(self, rel=""):
        base = self.path(rel)
        for p in sorted(base.rglob("*")):
            if p.is_file() and not (set(p.relative_to(self.root).parts) & SKIP_DIRS):
                yield p.relative_to(self.root).as_posix()

    # ---- identity / audit --------------------------------------------------------------
    def commit(self):
        try:
            r = subprocess.run(["git", "-C", str(self.root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
            return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else "unknown"
        except (OSError, subprocess.SubprocessError):
            return "unknown"

    def fingerprint(self, rel):
        p = self.path(rel)
        size = p.stat().st_size
        out = {"path": rel, "size": size}              # raw on-disk size: differs between LF and CRLF checkouts
        if size <= HASH_LIMIT:
            data = p.read_bytes()
            if p.suffix.lower() in TEXT_SUFFIXES:
                data = data.replace(b"\r\n", b"\n")
            out["sha256_lf"] = hashlib.sha256(data).hexdigest()
            out["size_lf"] = len(data)                 # size after line-ending normalisation: identical on every OS
        else:
            out["sha256_lf"] = None                    # large binary: identified by size + git commit only
            out["size_lf"] = size
        return out

    def snapshot(self, rels):
        return {r: self.fingerprint(r) for r in sorted(set(rels))}

    def assert_unchanged(self, snap):
        """Raise if any previously fingerprinted file changed or disappeared."""
        bad = []
        for rel, before in snap.items():
            if not self.exists(rel):
                bad.append(f"{rel}: missing"); continue
            if self.fingerprint(rel) != before:
                bad.append(f"{rel}: content changed")
        if bad:
            raise RepoAccessError("read-only repository was modified: " + "; ".join(bad))
        return True

    def pycache_dirs(self):
        return sorted(p.relative_to(self.root).as_posix() for p in self.root.rglob("__pycache__"))
