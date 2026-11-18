"""V0 bitstream audit for the BM Labs chip (classic_fabric_reram).

Uses verification/bitstream_audit.py (the existing audit) with a FabricProfile derived from the
chip repo instead of the FABulous demo defaults:

  value                     source in the chip repo                          cross-checked against
  ------------------------  -----------------------------------------------  ------------------------------
  rows, columns             src/chip_core.sv FABRIC_NUM_ROWS / _COLUMNS      chip profile grid (bel.txt)
  frames/col, bits/row      bitStreamSpec.bin ArchSpecs                      chip_core.sv localparams
  sync header, desync bit   bitStreamSpec.bin ArchSpecs                      the audit's defaults
  border rows in stream     bitStreamSpec.bin ArchSpecs IncludeBorderRows    -> top_row_y = rows - 1
  exact bitstream size      chip_core.sv BITSTREAM_LENGTH_WORDS * 4          the frame layout formula

The spec is a Python pickle. It is loaded with an unpickler that refuses every class/function
reference, so only plain data (dict/str/int/bool) can come out of it.
"""
import pickle
import re

from x2c.chip.chip_repo import ChipRepo
from x2c.verification.bitstream_audit import (FabricProfile, audit_against_fasm, audit_structure)

SPEC = "user_designs/fabrics/classic_fabric_reram/macro/sky130A/fabulous/bitStreamSpec.bin"
CHIP_CORE = "src/chip_core.sv"
LAYOUT = {"column": "X", "frame": "idx // 32", "row_word": "rows_per_frame - 1 - Y", "bit": "idx % 32",
          "verified_on": "classic_fabric_reram (sky130-reram-fpga @ e0790eb): 7 designs, every FASM-required bit "
                         "set and no other bit set (counter, bitslice_acc_mux, reram_cim_32x32, x2_vendor_cfg, "
                         "macc_8x8_20, ram_32x4_2r_1w, register_file)"}


class _PlainDataOnly(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}: the spec must be plain data")


def load_spec_pickle(path):
    with open(path, "rb") as f:
        spec = _PlainDataOnly(f).load()
    if not isinstance(spec, dict) or "TileSpecs" not in spec or "ArchSpecs" not in spec:
        raise ValueError("not a FABulous bitStreamSpec: TileSpecs/ArchSpecs missing")
    return spec


def chip_core_params(text):
    def lp(name):
        m = re.search(r"localparam\s+" + name + r"\s*=\s*(\d+)\s*;", text)
        if not m:
            raise ValueError(f"{CHIP_CORE}: localparam {name} not found")
        return int(m.group(1))
    m = re.search(r"\.BITSTREAM_LENGTH_WORDS\s*\(\s*32'h([0-9A-Fa-f_]+)\s*\)", text)
    if not m:
        raise ValueError(f"{CHIP_CORE}: BITSTREAM_LENGTH_WORDS not found")
    return {"rows": lp("FABRIC_NUM_ROWS"), "columns": lp("FABRIC_NUM_COLUMNS"),
            "frames_per_col": lp("MAX_FRAMES_PER_COL"), "frame_bits_per_row": lp("FRAME_BITS_PER_ROW"),
            "bitstream_words": int(m.group(1).replace("_", ""), 16)}


def chip_fabric_profile(repo=None, spec=None, chip_profile=None):
    """FabricProfile for the chip, with every value cross-checked. Raises on any disagreement."""
    repo = repo if isinstance(repo, ChipRepo) else ChipRepo(repo)
    spec = spec or load_spec_pickle(repo.path(SPEC))
    arch = spec["ArchSpecs"]
    core = chip_core_params(repo.read_text(CHIP_CORE))
    top_row_y = core["rows"] - 1 if arch.get("IncludeBorderRows") else None
    p = FabricProfile(name="classic_fabric_reram", rows_per_frame=core["rows"], columns=core["columns"],
                      frames_per_col=arch["MaxFramesPerCol"], frame_bits_per_row=arch["FrameBitsPerRow"],
                      max_bytes=core["bitstream_words"] * 4, sync_header_hex=arch["SyncHeaderHex"],
                      desync_bit=arch["DesyncBit"], layout=dict(LAYOUT), top_row_y=top_row_y)
    errors = []
    if (arch["MaxFramesPerCol"], arch["FrameBitsPerRow"]) != (core["frames_per_col"], core["frame_bits_per_row"]):
        errors.append(f"spec frames/bits {arch['MaxFramesPerCol']}/{arch['FrameBitsPerRow']} != chip_core "
                      f"{core['frames_per_col']}/{core['frame_bits_per_row']}")
    size = 20 + p.columns * p.frames_per_col * (4 + p.rows_per_frame * p.frame_bits_per_row // 8) + 4
    if size != p.max_bytes:
        errors.append(f"frame layout implies {size} bytes, chip loader expects {p.max_bytes}")
    if not arch.get("IncludeBorderRows"):
        errors.append("spec says border rows are excluded: the chip layout above was verified with them included")
    if chip_profile is not None:
        g = chip_profile["fabric"]["grid"]
        if (g["columns"], g["rows"]) != (p.columns, p.rows_per_frame):
            errors.append(f"chip_profile grid {g} != chip_core {p.columns}x{p.rows_per_frame}")
    if errors:
        raise ValueError("chip bitstream profile inconsistent: " + "; ".join(errors))
    return p, spec


def audit_chip_bitstream(bit_bytes, fasm_text=None, repo=None, profile=None, spec=None):
    """Both audit levels. Pass = structure ok AND (no FASM given, or decoded bits == FASM-required bits)."""
    if profile is None or spec is None:
        profile, spec = chip_fabric_profile(repo, spec)
    s = audit_structure(bit_bytes, profile)
    d = audit_against_fasm(bit_bytes, profile, fasm_text, spec["TileSpecs"]) if fasm_text is not None else None
    ok = s["ok"] and (d is None or d["ok"])
    return {"ok": ok, "structural": s, "semantic": d,
            "summary": {"structural_ok": s["ok"],
                        "failed_checks": [c["id"] for c in s["checks"] if not c["ok"] and c["severity"] == "error"],
                        "required_bits": d["required_bits"] if d else None,
                        "set_bits": d.get("set_bits") if d else s["stats"].get("set_bits"),
                        "missing": d["missing_count"] if d else None,
                        "unexplained": d["unexplained_count"] if d else None,
                        "unparsed_fasm_lines": len(d["unparsed_fasm_lines"]) if d else None}}
