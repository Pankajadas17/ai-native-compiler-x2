# Fix: signed multiplies on the chip's hard multipliers (no change to either chip repo)

## The problem
The chip has 18 hard multiplier blocks (MACC_8x8_20). Each one multiplies two **unsigned** 8-bit numbers and adds a 20-bit
`C` (tile RTL `ip/fabulous-tiles/primitives/MACC_8x8_20/fabulous/MACC_8x8_20.v`: `product = OPA * OPB`,
`sum = product + C`).

Yosys splits a wider multiply into 8x8 pieces and marks the pieces that hold a sign bit "signed". The adapter between
Yosys and the block, `ip/fabulous-tiles/primitives/MACC_8x8_20/yosys/techmap/MACC_8x8_20_map.v` (mole99/fabulous-tiles @
0651993f), ignores that mark. So a negative piece is read as a large positive number, like
`np.int8(-3).view(np.uint8) == 253`. It also wires bits 7..0 even when a piece is narrower, which leaves undefined bits.

## The fix
`chip/techmap/MACC_8x8_20_map_signed.v` is a copy of that adapter with two changes:
1. The operands are zero-extended to 8 bits (no undefined bits).
2. Signed pieces get a correction through the block's own `C` input. A W-bit operand with sign bit m is
   `unsigned - m*2^W`, so
   `a*b = a_u*b_u - am*(b_u << AW) - bm*(a_u << BW) + am*bm*2^(AW+BW)`.
   The block still computes `a_u*b_u + C`; the correction is `C`. For unsigned pieces `C = 0`, as in the original.

`chip/chip_flow.py` puts it into the Makefile it generates in `chip_build/<design>/`. That file is ours; the line comes
after the repo recipe is included, so it wins:
```make
MULTIPLIER_MAP_PATH := -multiplier-map <bm_verif>/chip/techmap/MACC_8x8_20_map_signed.v 8 8 2 2 12
```
Before each build, `chip_flow` checks that `user_designs/makefile.include` still has exactly
`MULTIPLIER_MAP_PATH = -multiplier-map ../../../../ip/fabulous-tiles/primitives/MACC_8x8_20/yosys/techmap/MACC_8x8_20_map.v 8 8 2 2 12`.
If the repo changes it, the build stops with "re-check ... before building", so nothing is overridden silently.
Each `flow_report.json` records `multiplier_map` (which file, what it replaces, why).

## Evidence

| check | repo map | corrected map |
|---|---|---|
| the adapter alone, every width 1..8, both signednesses, 3 result widths (3,121,200 products), on the **real tile RTL** | 2,674,222 wrong | **0** |
| synthesized netlist, 16x8 unsigned (453,632) | 0 | 0 |
| synthesized netlist, 16x8 signed (453,632) | 339,210 | **0** |
| synthesized netlist, 8x16 signed (409,344) | 306,017 | **0** |
| synthesized netlist, 16x16 signed (196,572 incl. edge rows), 4 blocks both | 114,404 | **0** |
| requant through the chip chain (485 vectors) | 219 wrong, FAIL | **PASS**, 134 LC, 4 MACC, V0 3549/3549 |
| all 15 chip examples: status, LC, MACC, V0 bits | | **identical** to before |

The fix costs no extra logic cells for requant: the correction terms fold into the block's `C` input. Timing through a
hard multiplier is not reported: nextpnr found no register-to-register path in requant.

## Exact changes
- **`chip/chip_flow.py`:**
  - The `MAKEFILE` template gets 2 lines after `CLKBUF_MAP_PATH := ...` (the comment and the line above).
  - New: `REPO_MULTIPLIER_LINE`, `SIGNED_MAP`, `multiplier_override(repo)`.
  - In `run_flow`, from
    ```python
    (work / "Makefile").write_text(MAKEFILE.format(name=name, top=top, rtl=" ".join(rtl_local)), ...)
    ```
    to
    ```python
    mul_map, mul_args = multiplier_override(repo)
    (work / "Makefile").write_text(MAKEFILE.format(name=name, top=top, rtl=" ".join(rtl_local),
                                                   multiplier_map=mul_map, multiplier_args=mul_args), ...)
    ```
  - The report gains `"multiplier_map": {...}`.
- **`verification/chip_chain.py`:** `MACC_SIGNED_NOTE` text only. If that pattern ever shows again, it says to check the
  build's multiplier override.
- **`tests/test_chip_chain.py`:** one assertion follows that text.

New files:
- chip/techmap/MACC_8x8_20_map_signed.v
- tests/test_macc_signed_map.py
- tests/fixtures/chip_chain/requant_i32_i8_signed_map_sim.log

Full diff: `docs/fix_signed_multiplier.diff`. The corrected adapter is also the fix fabulous-tiles itself needs; once
fabulous-tiles is fixed upstream, the override can be removed.
