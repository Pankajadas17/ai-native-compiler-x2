"""The corrected multiplier map (chip/techmap/MACC_8x8_20_map_signed.v) that chip_flow passes to the repo Makefile.

The repo map (fabulous-tiles 0651993f, primitives/MACC_8x8_20/yosys/techmap/MACC_8x8_20_map.v) ignores A_SIGNED/B_SIGNED:
every signed multiply wider than 8 bits was wrong for negative operands (requant 219/485 on the chip chain). Neither repo
is edited; the generated Makefile in chip_build/<design>/ overrides MULTIPLIER_MAP_PATH.

* the correction rule, block model in Python, exhaustively: every A/B width 1..8, both signednesses, several Y widths;
* the Verilog map on the REAL tile RTL (fabulous/MACC_8x8_20.v), exhaustively, in Icarus (Ubuntu only);
* the override is in every generated Makefile, and refuses to run if the repo's multiplier line ever changes;
* requant through the chip chain: the recorded netlist simulation with the corrected map passes (it failed with the repo's).
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from x2c.chip import chip_flow as cf  # noqa: E402
from x2c.chip.chip_repo import ChipRepo, ChipRepoError  # noqa: E402
from x2c.verification import chip_chain as cc  # noqa: E402

MAP = ROOT / "src" / "x2c" / "chip" / "techmap" / "MACC_8x8_20_map_signed.v"
FIX = ROOT / "tests" / "fixtures" / "chip_chain"
TILE = "ip/fabulous-tiles/primitives/MACC_8x8_20/fabulous/MACC_8x8_20.v"
CASES = [(aw, bw, sa, sb, yw) for aw in range(1, 9) for bw in range(1, 9) for sa in (0, 1) for sb in (0, 1)
         for yw in sorted({aw + bw, 20, max(1, aw + bw - 3)})]


def repo_or_skip():
    try:
        return ChipRepo()
    except ChipRepoError as e:
        pytest.skip(str(e))


# ---------------------------------------------------------------- the rule, in Python
def block(a8, b8, c):
    """MACC_8x8_20 as configured by the map (tile RTL): unsigned 8x8 product + C, 20 bits."""
    return (a8 * b8 + c) % (1 << 20)


def correction(a, b, aw, bw, sa, sb):
    """C exactly as the map computes it."""
    am = (a >> (aw - 1)) & 1 if sa else np.zeros_like(a)
    bm = (b >> (bw - 1)) & 1 if sb else np.zeros_like(b)
    corr_a = np.where(am == 1, b << aw, 0)
    corr_b = np.where(bm == 1, a << bw, 0)
    corr_ab = np.where((am & bm) == 1, 1 << (aw + bw), 0)
    return (corr_ab - corr_a - corr_b) % (1 << 20)


def test_correction_rule_exhaustive():
    total = 0
    for aw, bw, sa, sb, yw in CASES:
        a, b = np.meshgrid(np.arange(1 << aw, dtype=np.int64), np.arange(1 << bw, dtype=np.int64), indexing="ij")
        a, b = a.ravel(), b.ravel()
        va = np.where(sa & (a >> (aw - 1)), a - (1 << aw), a)
        vb = np.where(sb & (b >> (bw - 1)), b - (1 << bw), b)
        want = (va * vb) % (1 << yw)
        got = block(a, b, correction(a, b, aw, bw, sa, sb)) % (1 << yw)
        assert np.array_equal(got, want), (aw, bw, sa, sb, yw)
        total += a.size
    assert total == 3121200


def test_unsigned_pieces_get_c_zero_like_the_repo_map():
    for aw in range(1, 9):
        for bw in range(1, 9):
            a = np.arange(1 << aw, dtype=np.int64).repeat(1 << bw)
            b = np.tile(np.arange(1 << bw, dtype=np.int64), 1 << aw)
            assert not correction(a, b, aw, bw, 0, 0).any()


def test_map_text():
    text = MAP.read_text()
    assert "module \\$__FABULOUS_MUL" in text
    for frag in ("wire [7:0] a8 = A;", "wire [7:0] b8 = B;", "wire am = A_SIGNED ? A[A_WIDTH-1] : 1'b0;",
                 "wire bm = B_SIGNED ? B[B_WIDTH-1] : 1'b0;", "assign C = corr_ab - corr_a - corr_b;", ".SIGN_EXT (1'b0)",
                 "assign Y = Q[Y_WIDTH-1:0];"):
        assert frag in text, frag
    assert len(re.findall(r"\.A\d \(a8\[\d\]\)", text)) == 8 and len(re.findall(r"\.B\d \(b8\[\d\]\)", text)) == 8


# ---------------------------------------------------------------- the override
def test_override_is_in_the_generated_makefile():
    mk = cf.MAKEFILE.format(name="d", top="d", rtl="rtl/d.v", multiplier_map="/x/project/src/x2c/chip/techmap/MACC_8x8_20_map_signed.v",
                            multiplier_args="8 8 2 2 12")
    lines = mk.splitlines()
    fixed = lines.index("MULTIPLIER_MAP_PATH := $(call fix,$(MULTIPLIER_MAP_PATH))")
    ours = lines.index("MULTIPLIER_MAP_PATH := -multiplier-map /x/project/src/x2c/chip/techmap/MACC_8x8_20_map_signed.v 8 8 2 2 12")
    assert ours > fixed > lines.index("include $(UD)/makefile.include")       # the later assignment wins


def test_override_matches_the_repo():
    path, args = cf.multiplier_override(repo_or_skip())
    assert path.endswith("chip/techmap/MACC_8x8_20_map_signed.v") and args == "8 8 2 2 12"


class FakeRepo:
    def __init__(self, text):
        self.text = text

    def path(self, rel):
        return self

    def read_text(self, errors=None):
        return self.text


def test_override_refuses_a_changed_repo_line():
    line = "MULTIPLIER_MAP_PATH \t = -multiplier-map ../../../../ip/fabulous-tiles/primitives/MACC_8x8_20/yosys/techmap/MACC_8x8_20_map.v 8 8 2 2 12\r\n"
    assert cf.multiplier_override(FakeRepo("A = 1\r\n" + line))[1] == "8 8 2 2 12"          # CRLF clone is fine
    for changed in (line.replace("8 8 2 2 12", "8 8 2 2 16"), line.replace("MACC_8x8_20_map.v", "MACC_map_v2.v"), "A = 1\n"):
        with pytest.raises(RuntimeError, match="changed"):
            cf.multiplier_override(FakeRepo(changed))


# ---------------------------------------------------------------- on the chip chain
def test_requant_recorded_with_both_maps():
    old = cc.parse_netlist_log("requant_i32_i8", (FIX / "requant_i32_i8_sim.log").read_text())
    new = cc.parse_netlist_log("requant_i32_i8", (FIX / "requant_i32_i8_signed_map_sim.log").read_text())
    assert old["status"] == "fail" and old["reason"].startswith("219/485")
    assert new["status"] == "pass" and new["vectors"] == 485


# ---------------------------------------------------------------- the Verilog map on the real tile RTL
def tile_shim():
    bits = lambda p, n: ", ".join(f"input {p}{i}" for i in range(n))
    cat = lambda p, n: "{" + ", ".join(f"{p}{i}" for i in reversed(range(n))) + "}"
    return (f"module MACC_8x8_20 #(parameter A_REG=1'b0, B_REG=1'b0, C_REG=1'b0, SIGN_EXT=1'b0, ACC_SEL=1'b0, ACC_OUT=1'b0, "
            f"CLK_INV=1'b0)\n (input CLK, input EN, input ACC_RST, input ACC_CI, output ACC_CO, {bits('A', 8)}, {bits('B', 8)}, "
            f"{bits('C', 20)}, " + ", ".join(f"output Q{i}" for i in range(20)) + ");\n"
            "  wire [6:0] cfg = {CLK_INV, ACC_OUT, ACC_SEL, SIGN_EXT, C_REG, B_REG, A_REG};   // the tile's BelMap order\n"
            "  wire [19:0] q;\n"
            f"  TILE_MACC tile (.CLK(CLK), .EN(EN), .ACC_RST(ACC_RST), .ACC_CI(ACC_CI), .ACC_CO(ACC_CO), .A({cat('A', 8)}), "
            f".B({cat('B', 8)}), .C({cat('C', 20)}), .Q(q), .ConfigBits(cfg));\n  assign {cat('Q', 20)} = q;\nendmodule\n")


def exhaustive_tb():
    L = ["`timescale 1ns/1ps", "module t;", "integer bad = 0, n = 0, i, j;", "reg signed [40:0] va, vb, p; reg [40:0] e;"]
    for k, (aw, bw, sa, sb, yw) in enumerate(CASES):
        L += [f"reg [{aw-1}:0] a{k}; reg [{bw-1}:0] b{k}; wire [{yw-1}:0] y{k};",
              f"\\$__FABULOUS_MUL #(.A_SIGNED({sa}),.B_SIGNED({sb}),.A_WIDTH({aw}),.B_WIDTH({bw}),.Y_WIDTH({yw})) m{k} (.A(a{k}),.B(b{k}),.Y(y{k}));"]
    L.append("initial begin")
    for k, (aw, bw, sa, sb, yw) in enumerate(CASES):
        sv = lambda x, w, s: f"(({s}) && {x}[{w-1}]) ? $signed({{1'b0,{x}}}) - (41'sd1 <<< {w}) : $signed({{1'b0,{x}}})"
        L += [f"  for (i = 0; i < {1 << aw}; i = i + 1) for (j = 0; j < {1 << bw}; j = j + 1) begin",
              f"    a{k} = i; b{k} = j; #1; va = {sv(f'a{k}', aw, sa)}; vb = {sv(f'b{k}', bw, sb)}; p = va * vb; e = p;",
              f"    n = n + 1; if (y{k} !== e[{yw-1}:0]) bad = bad + 1;", "  end"]
    L += ['  $display("CHECKED %0d WRONG %0d", n, bad); $finish;', "end", "endmodule"]
    return "\n".join(L) + "\n"


def test_verilog_map_on_the_tile_rtl(tmp_path):
    if not (shutil.which("iverilog") and shutil.which("vvp")):
        pytest.skip("iverilog not on PATH (expected on Windows; run this test in Ubuntu)")
    repo = repo_or_skip()
    tile = repo.path(TILE).read_text(errors="replace")
    (tmp_path / "tile.v").write_text(re.sub(r"\bmodule MACC_8x8_20\b", "module TILE_MACC", tile))
    (tmp_path / "shim.v").write_text(tile_shim())
    (tmp_path / "tb.v").write_text(exhaustive_tb())
    r = subprocess.run(["iverilog", "-g2012", "-s", "t", "-o", str(tmp_path / "x.vvp"), str(tmp_path / "tb.v"), str(MAP),
                        str(tmp_path / "shim.v"), str(tmp_path / "tile.v")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    out = subprocess.run(["vvp", "-n", str(tmp_path / "x.vvp")], capture_output=True, text=True, timeout=600).stdout
    assert "CHECKED 3121200 WRONG 0" in out, out[-400:]
