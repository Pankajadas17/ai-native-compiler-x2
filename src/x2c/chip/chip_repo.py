"""Read-only view of the chip repo (sky130-reram-fpga, with its submodules).

Rules this module enforces:
  * files are only ever opened for reading ("rb");
  * text is LF-normalised before hashing, so a Windows (CRLF) checkout and a
    Linux (LF) checkout of the same commit give the same digest;
  * git metadata is read from the .git files directly (no git command is run).

A tree fingerprint (path, size, mtime) taken before and after a run proves that
the run did not create, delete or modify any file in the repo.
"""
import hashlib
import os
from pathlib import Path

from x2c import paths

PROJECT_ROOT = paths.ROOT
DEFAULT_REPO = PROJECT_ROOT / "sky130-reram-fpga"
ENV_VAR = "BM_CHIP_REPO"

# Files the profile is derived from (relative to the repo root).
FABRIC_DIR = "user_designs/fabrics/classic_fabric_reram"
FABULOUS_DIR = FABRIC_DIR + "/macro/sky130A/fabulous"
INPUT_FILES = {
    "bels": FABULOUS_DIR + "/.FABulous/bel.txt",
    "bels_v2": FABULOUS_DIR + "/.FABulous/bel.v2.txt",
    "pins": FABRIC_DIR + "/constraints.pcf",
    "x2_primitive": "ip/extra-tiles/primitives/RERAM_CIM_32x32/yosys/primitives/RERAM_CIM_32x32.v",
    "x2_behavioral": "ip/sky130_bmlabs_reram/verilog/behavioral/sky130_bmlabs_reram__cim32x32.sv",
}
SUBMODULES = ("ip/fabulous-tiles", "ip/sky130_bmlabs_reram")

# A separate X2 model repo (when the X2 team publishes a new model): $BM_X2_REPO points at that repo or at the .sv file.
# Every reader of INPUT_FILES["x2_behavioral"] (simulations, profile, hardware config) then uses it instead of the copy
# inside the chip repo. Re-check after switching:  x2c recheck  (Ubuntu).
X2_ENV_VAR = "BM_X2_REPO"
X2_MODEL_IN_IP = "verilog/behavioral/sky130_bmlabs_reram__cim32x32.sv"     # path inside the BM Labs ReRAM IP repo


class ChipRepoError(RuntimeError):
    pass


def x2_model_override():
    """The X2 behavioral model from $BM_X2_REPO (a checkout of the X2/IP repo, or the .sv file itself); None if unset."""
    v = os.environ.get(X2_ENV_VAR)
    if not v:
        return None
    p = Path(v).expanduser().resolve()
    if p.is_file():
        return p
    if (p / X2_MODEL_IN_IP).is_file():
        return p / X2_MODEL_IN_IP
    hits = sorted(p.rglob("*cim32x32*.sv")) if p.is_dir() else []
    if len(hits) == 1:
        return hits[0]
    raise ChipRepoError(f"${X2_ENV_VAR}={v}: expected the X2 model .sv file, or a repo containing {X2_MODEL_IN_IP}"
                        + (f" ({len(hits)} *cim32x32*.sv files found: name the file)" if hits else " (not found)"))


def x2_model_label():
    """Where the X2 model is read from, for reports and documents."""
    o = x2_model_override()
    return f"${X2_ENV_VAR}/{o.name}" if o else INPUT_FILES["x2_behavioral"]


def locate_repo(path=None):
    """Explicit path > $BM_CHIP_REPO > <project>/sky130-reram-fpga."""
    p = Path(path or os.environ.get(ENV_VAR) or DEFAULT_REPO).resolve()
    if not (p / "user_designs" / "makefile.include").is_file():
        raise ChipRepoError(f"not a sky130-reram-fpga checkout: {p}")
    return p


def _read_git_head(git_dir):
    """Commit hash HEAD points to, read from the git files; None if unreadable."""
    try:
        head = (git_dir / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head or None
        ref = head.split(":", 1)[1].strip()
        loose = git_dir / ref
        if loose.is_file():
            return loose.read_text().strip()
        packed = git_dir / "packed-refs"
        if packed.is_file():
            for line in packed.read_text().splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref:
                    return parts[0]
    except OSError:
        pass
    return None


def _git_dir(work_dir):
    g = work_dir / ".git"
    if g.is_dir():
        return g
    if g.is_file():  # submodule: ".git" is a file "gitdir: <path>"
        text = g.read_text().strip()
        if text.startswith("gitdir:"):
            return (work_dir / text.split(":", 1)[1].strip()).resolve()
    return None


def lf_sha256(data):
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


class ChipRepo:
    def __init__(self, path=None):
        self.root = locate_repo(path)

    def path(self, rel):
        if rel == INPUT_FILES["x2_behavioral"]:
            o = x2_model_override()
            if o is not None:
                return o
        p = (self.root / rel).resolve()
        if self.root not in p.parents and p != self.root:
            raise ChipRepoError(f"path escapes the repo: {rel}")
        return p

    def read_bytes(self, rel):
        p = self.path(rel)
        if not p.is_file():
            raise ChipRepoError(f"missing file in chip repo: {rel}"
                                + (" (submodule not checked out?)" if rel.startswith("ip/") else ""))
        with open(p, "rb") as f:
            return f.read()

    def read_text(self, rel):
        """UTF-8 text with line endings normalised to LF."""
        return self.read_bytes(rel).replace(b"\r\n", b"\n").decode("utf-8")

    def commit(self, rel=""):
        gd = _git_dir(self.path(rel) if rel else self.root)
        return _read_git_head(gd) if gd else None

    def provenance(self, files=None):
        files = dict(INPUT_FILES if files is None else files)
        hashes = {rel: lf_sha256(self.read_bytes(rel)) for rel in sorted(files.values())}
        h = hashlib.sha256()
        for rel, digest in hashes.items():
            h.update(f"{rel}\0{digest}\n".encode())
        return {
            "repo": "sky130-reram-fpga",
            "commit": self.commit(),
            "submodules": {s: self.commit(s) for s in SUBMODULES},
            "files": hashes,
            "input_digest": h.hexdigest(),
            "x2_model": {"source": f"${X2_ENV_VAR}" if x2_model_override() else "chip repo", "path": x2_model_label()},
        }

    def tree_fingerprint(self):
        """Hash of every file's (path, size, mtime) outside .git folders."""
        h = hashlib.sha256()
        count = 0
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(d for d in dirnames if d != ".git")
            for name in sorted(filenames):
                if name == ".git":
                    continue
                full = Path(dirpath) / name
                st = full.lstat()
                rel = full.relative_to(self.root).as_posix()
                h.update(f"{rel}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
                count += 1
        return {"files": count, "fingerprint": h.hexdigest()}
