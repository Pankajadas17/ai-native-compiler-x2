"""Phase 1: chip profile derived from the chip repo (sky130-reram-fpga).

Unit tests use small inline fixtures (no repo needed). The repo tests run against
the real checkout and are skipped if it is not present (set BM_CHIP_REPO to point
elsewhere).
"""
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chip import fabric_profile as fp  # noqa: E402
from chip.chip_repo import INPUT_FILES, ChipRepo, ChipRepoError, lf_sha256, locate_repo  # noqa: E402

BELS = """# BEL descriptions
#Tile_X0Y0
#Tile_X1Y0
#Tile_X0Y1
X0Y1,X0,Y1,A,IOBUF,A_CLK,A_IN,A_EN,A_OUT
X0Y1,X0,Y1,B,IOBUF,B_CLK,B_IN,B_EN,B_OUT
#Tile_X1Y1
X1Y1,X1,Y1,A,FABULOUS_LC,LA_I0
X1Y1,X1,Y1,B,FABULOUS_LC,LB_I0
X1Y1,X1,Y1,C
#Tile_X2Y1
X2Y1,X2,Y1,A,RERAM_CIM_32x32,CLK
"""

PRIM = """(* blackbox, keep *)
module RERAM_CIM_32x32 #(
    parameter [15:0] TARGET_SET1 = '0,
    parameter [ 1:0] TDC_DEAD_TIME = '0
)(
    (* clkbuf_sink *) input wire CLK,
    input  wire MODE0,
    input  wire MODE1,
    input  wire ROW_INDEX0,
    input  wire ROW_INDEX1,
    output wire RESULT_DATA0,
    output wire RESULT_DATA1,
    output wire RESULT_LAST // trailing comment
);
endmodule
"""


# ---------------------------------------------------------------- unit tests
def test_parse_bels_counts_sites_and_grid():
    b = fp.parse_bels(BELS)
    assert b["resources"] == {"FABULOUS_LC": 2, "IOBUF": 2, "RERAM_CIM_32x32": 1}
    assert b["sites"]["IOBUF"] == ["X0Y1/A", "X0Y1/B"]
    assert b["grid"] == {"columns": 3, "rows": 2}


def test_parse_bels_rejects_text_without_tiles():
    with pytest.raises(ValueError):
        fp.parse_bels("not a bel file\n")


def test_parse_bel_cfg_takes_first_matching_bel_only():
    text = ("BelBegin,X0Y0,A,OTHER,\nCFG,X\nBelEnd\n"
            "BelBegin,X1Y0,A,RERAM_CIM_32x32,\nCFG,P\nCFG,P[1]\nCFG,Q\nBelEnd\n"
            "BelBegin,X2Y0,A,RERAM_CIM_32x32,\nCFG,ZZ\nBelEnd\n")
    assert fp.parse_bel_cfg(text, "RERAM_CIM_32x32") == (3, {"P": 2, "Q": 1})


def test_parse_pcf_groups_by_heading():
    g = fp.parse_pcf("# All I/Os\nset_io all[0] X0Y1/A\n\n# Tests\nset_io clk1 X0Y1/A\n")
    assert g["All I/Os"] == {"all[0]": "X0Y1/A"}
    assert g["Tests"] == {"clk1": "X0Y1/A"}


def test_primitive_ports_params_and_buses():
    ports, params = fp.parse_primitive(PRIM, "RERAM_CIM_32x32")
    assert params == {"TARGET_SET1": 16, "TDC_DEAD_TIME": 2}
    buses, single = fp.group_buses(ports)
    assert buses["MODE"] == {"direction": "input", "width": 2}
    assert buses["ROW_INDEX"]["width"] == 2
    assert buses["RESULT_DATA"] == {"direction": "output", "width": 2}
    assert single == {"CLK": "input", "RESULT_LAST": "output"}


def test_group_buses_rejects_missing_bit():
    with pytest.raises(ValueError):
        fp.group_buses([("input", "PWM_DATA0"), ("input", "PWM_DATA2")])


def test_behavioral_parse():
    text = ("module sky130_bmlabs_reram__cim32x32 #(\n  parameter integer READ_DELAY = 160,\n"
            "  parameter logic [13:0] INITIAL_CELL_LEVEL = 14'h0E23\n)(\n"
            "  // Result stream.  Data format is {mode[1:0], column[4:0], count[13:0]}.\n"
            "  localparam logic [1:0] MODE_READ = 2'b01;\n  localparam logic [1:0] MODE_SET = 2'b11;\n")
    b = fp.parse_behavioral(text)
    assert b["parameters"] == {"READ_DELAY": 160, "INITIAL_CELL_LEVEL": 0x0E23}
    assert b["modes"] == {"READ": 1, "SET": 3}
    assert [f["width"] for f in b["result_format_msb_first"]] == [2, 5, 14]


def test_lf_hash_ignores_windows_line_endings():
    assert lf_sha256(b"a\r\nb\r\n") == lf_sha256(b"a\nb\n")
    assert lf_sha256(b"a\nb\n") != lf_sha256(b"a\nc\n")


def test_locate_repo_rejects_wrong_folder(tmp_path):
    with pytest.raises(ChipRepoError):
        locate_repo(tmp_path)


# ---------------------------------------------------------------- real chip repo
def _repo_or_skip():
    try:
        repo = ChipRepo()
    except ChipRepoError as e:
        pytest.skip(str(e))
    missing = [r for r in INPUT_FILES.values() if not (repo.root / r).is_file()]
    if missing:
        pytest.skip(f"chip repo incomplete (submodules?): {missing}")
    return repo


@pytest.fixture(scope="module")
def real():
    repo = _repo_or_skip()
    before = repo.tree_fingerprint()
    profile = fp.build_chip_profile(repo)
    after = repo.tree_fingerprint()
    return repo, profile, before, after


def test_real_profile_numbers(real):
    _, p, _, _ = real
    assert p["fabric"]["grid"] == {"columns": 21, "rows": 20}
    assert p["fabric"]["resources"] == {"FABULOUS_LC": 2160, "GBUF": 4, "IOBUF": 36, "MACC_8x8_20": 18,
                                        "RAM_32x4_2R_1W": 36, "RERAM_CIM_32x32": 2, "SYS_RESET": 1}
    x2 = p["x2"]
    assert (x2["count"], x2["rows"], x2["cols"]) == (2, 32, 32)
    assert x2["sites"] == ["X20Y6/A", "X20Y14/A"]
    assert (x2["pwm_bits"], x2["result_bits"], x2["config_bits_per_block"]) == (8, 21, 93)
    assert [f["width"] for f in x2["behavioral_model"]["result_format_msb_first"]] == [2, 5, 14]
    assert x2["behavioral_model"]["modes"] == {"RESET": 0, "READ": 1, "COMPUTE": 2, "SET": 3}
    assert p["io"]["general_pins"] == [f"all[{i}]" for i in range(28)]


def test_real_run_leaves_repo_untouched(real):
    _, _, before, after = real
    assert before == after


def test_real_profile_is_deterministic(real):
    repo, p, _, _ = real
    assert json.dumps(fp.build_chip_profile(repo)) == json.dumps(p)


def test_self_check_catches_inconsistent_repo(real, tmp_path):
    """Copy only the input files, break the X2 config field list, expect a loud failure."""
    repo, _, _, _ = real
    for rel in INPUT_FILES.values():
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo.root / rel, dst)
    (tmp_path / "user_designs" / "makefile.include").write_text("# stub\n")
    bels_v2 = tmp_path / INPUT_FILES["bels_v2"]
    bels_v2.write_bytes(bels_v2.read_bytes().replace(b"CFG,TDC_DEAD_TIME[1]", b"", 1))
    with pytest.raises(ValueError, match="config"):
        fp.build_chip_profile(ChipRepo(tmp_path))
