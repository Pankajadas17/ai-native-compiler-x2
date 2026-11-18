"""Phase 5: V0 bitstream audit on the real chip (classic_fabric_reram).

Fixtures are real bitstreams + FASM from the chip flow (tests/fixtures/chip_bitstream/).
The decode needs the chip's bitStreamSpec.bin, so those tests skip if the chip repo is absent.
"""
import pickle
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from x2c.chip import chip_bitstream as cb  # noqa: E402
from x2c.chip.chip_repo import ChipRepo, ChipRepoError  # noqa: E402
from x2c.verification.bitstream_audit import (audit_against_fasm, audit_structure, parse_fasm,  # noqa: E402
                                          required_bits)

FIX = ROOT / "tests" / "fixtures" / "chip_bitstream"
DESIGNS = ["repo_counter", "bitslice_acc_mux", "x2_vendor_cfg"]


@pytest.fixture(scope="module")
def chip():
    try:
        repo = ChipRepo()
    except ChipRepoError as e:
        pytest.skip(str(e))
    if not repo.path(cb.SPEC).is_file():
        pytest.skip("bitStreamSpec.bin missing")
    profile, spec = cb.chip_fabric_profile(repo)
    return profile, spec


def bit(name):
    return (FIX / f"{name}.bit").read_bytes()


def fasm(name):
    return (FIX / f"{name}.fasm").read_text()


# ---------------------------------------------------------------- no repo needed
def test_unsized_fasm_literal_is_parsed():
    """nextpnr writes X2 parameters as unsized literals ('b0111110); these used to be 'unparsed'."""
    feats, bad = parse_fasm("X20Y6.A.TDC_TIME_OUT[6:0] = 'b0111110\nX1Y1.A.INIT[7:0] = 8'h0F\n")
    assert bad == []
    assert feats[0] == ("X20Y6", "A.TDC_TIME_OUT", (6, 0), "0111110")
    assert feats[1][3] == "00001111"


def test_unpickler_refuses_code(tmp_path):
    p = tmp_path / "evil.bin"
    p.write_bytes(pickle.dumps({"TileSpecs": {}, "ArchSpecs": {}, "x": date(2026, 1, 1)}))
    with pytest.raises(pickle.UnpicklingError):
        cb.load_spec_pickle(p)


def test_chip_core_params_parse():
    text = ("localparam FRAME_BITS_PER_ROW = 32;\nlocalparam MAX_FRAMES_PER_COL = 20;\n"
            "localparam FABRIC_NUM_COLUMNS = 21;\nlocalparam FABRIC_NUM_ROWS = 20;\n"
            ".BITSTREAM_LENGTH_WORDS (32'h227A),\n")
    assert cb.chip_core_params(text) == {"rows": 20, "columns": 21, "frames_per_col": 20,
                                         "frame_bits_per_row": 32, "bitstream_words": 0x227A}


# ---------------------------------------------------------------- real chip
def test_chip_profile_values(chip):
    p, spec = chip
    assert (p.columns, p.rows_per_frame, p.frames_per_col, p.frame_bits_per_row) == (21, 20, 20, 32)
    assert p.top_row_y == 19 and p.max_bytes == 35304
    assert spec["ArchSpecs"]["IncludeBorderRows"] is True
    assert p.layout["row_word"] == "rows_per_frame - 1 - Y"


@pytest.mark.parametrize("name", DESIGNS)
def test_real_chip_bitstreams_pass_both_levels(chip, name):
    p, spec = chip
    r = cb.audit_chip_bitstream(bit(name), fasm(name), profile=p, spec=spec)
    s = r["summary"]
    assert r["ok"], s
    assert s["missing"] == 0 and s["unexplained"] == 0 and s["unparsed_fasm_lines"] == 0
    assert s["set_bits"] == s["required_bits"] > 0


def test_demo_row_numbering_fails_on_the_chip(chip):
    """Locks in the finding: the demo-fabric layout (no border rows) does NOT fit this chip."""
    p, spec = chip
    demo = replace(p, top_row_y=None)
    d = audit_against_fasm(bit("repo_counter"), demo, fasm("repo_counter"), spec["TileSpecs"])
    assert not d["ok"] and d["missing_count"] > 0


def test_old_size_limit_would_reject_every_chip_bitstream(chip):
    p, _ = chip
    s = audit_structure(bit("repo_counter"), replace(p, max_bytes=16384))
    assert any(c["id"] == "V0.1b" and not c["ok"] for c in s["checks"])


def _byte_and_mask(p, tile, idx):
    x, y = (int(v) for v in tile[1:].split("Y"))
    f, b = divmod(idx, p.frame_bits_per_row)
    r = p.top_row_y - y
    frame_bytes = 4 + p.rows_per_frame * p.frame_bits_per_row // 8
    word_off = 20 + (x * p.frames_per_col + f) * frame_bytes + 4 + r * 4
    return word_off + (3 - b // 8), 1 << (b % 8)          # big-endian word


def test_flipped_x2_parameter_bit_is_caught_and_named(chip):
    p, spec = chip
    req, _, _ = required_bits(fasm("x2_vendor_cfg"), spec["TileSpecs"])
    key = next(k for k, v in sorted(req.items()) if any("TDC_TIME_OUT" in n for n in v))
    data = bytearray(bit("x2_vendor_cfg"))
    off, mask = _byte_and_mask(p, *key)
    assert data[off] & mask, "layout arithmetic should land on a set bit"
    data[off] ^= mask
    s = audit_structure(bytes(data), p)
    assert s["ok"], "structure cannot see a payload flip (known limit)"
    d = audit_against_fasm(bytes(data), p, fasm("x2_vendor_cfg"), spec["TileSpecs"])
    assert not d["ok"] and d["missing_count"] == 1
    assert any("TDC_TIME_OUT" in n for n in d["missing"][0]["needed_by"])


def test_a_stray_bit_is_reported_as_unexplained(chip):
    p, spec = chip
    data = bytearray(bit("repo_counter"))
    got = set(required_bits(fasm("repo_counter"), spec["TileSpecs"])[0])
    key = next((t, i) for t, feats in spec["TileSpecs"].items() for d in feats.values() for i, v in d.items()
               if v == "1" and (t, i) not in got)
    off, mask = _byte_and_mask(p, *key)
    data[off] ^= mask
    d = audit_against_fasm(bytes(data), p, fasm("repo_counter"), spec["TileSpecs"])
    assert not d["ok"] and d["unexplained_count"] == 1


def test_cross_check_rejects_a_mismatching_chip_profile(chip):
    _, spec = chip
    bad = {"fabric": {"grid": {"columns": 21, "rows": 19}}}
    with pytest.raises(ValueError, match="grid"):
        cb.chip_fabric_profile(spec=spec, chip_profile=bad)
