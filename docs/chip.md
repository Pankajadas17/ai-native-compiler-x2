# The chip: profile, flow and bitstream audit

Where every chip number comes from, how a design becomes a bitstream, how the bitstream is audited, and the signed-multiplier fix.

- [Chip profile: `x2c chip profile`](#chip-profile-x2c-chip-profile)
- [Chip flow: `x2c build`](#chip-flow-x2c-build)
- [V0 bitstream audit: `x2c audit`](#v0-bitstream-audit-x2c-audit)
- [Signed multiplies on the hard multipliers](#signed-multiplies-on-the-hard-multipliers)
- [The FABulous demo fabric (before the chip repo)](#the-fabulous-demo-fabric-before-the-chip-repo)

## Chip profile: `x2c chip profile`

`specifications/chip_profile.json` describes the real BM Labs chip, derived from the chip repo `sky130-reram-fpga`
(cloned inside the project folder, with submodules). No chip number is typed by hand. `x2c chip config` turns the
profile into `specifications/hardware_config.json` for A1 ([deploy.md](deploy.md#the-partition-on-the-chip)).

### Rules
- The chip repo is a read-only input, like `BMsemi_Model`. Code only opens its files with `"rb"`.
- `x2c chip profile` fingerprints every file in the repo (path, size, mtime) before and after
  the run and fails if anything changed.
- Text is LF-normalised before hashing, so Windows (CRLF) and Linux (LF) checkouts of the
  same commit give the same `input_digest` and the same JSON.
- Builds of chip designs go to `chip_build/<design>/`, never inside the repo.

### Where each number comes from
| Profile field | Source file in the chip repo |
|---|---|
| `fabric.grid`, `fabric.resources`, X2 sites, IOBUF sites | `user_designs/fabrics/classic_fabric_reram/macro/sky130A/fabulous/.FABulous/bel.txt` |
| `x2.config_bits_per_block`, `x2.config_fields` | same folder, `bel.v2.txt` (CFG entries of the X2 BEL) |
| `io.*` pins | `user_designs/fabrics/classic_fabric_reram/constraints.pcf` |
| `x2.buses`, `x2.parameters`, `x2.rows/cols` (= 2^width of ROW/COL_INDEX) | `ip/extra-tiles/primitives/RERAM_CIM_32x32/yosys/primitives/RERAM_CIM_32x32.v` |
| `x2.behavioral_model` (modes, delays, result format) | `ip/sky130_bmlabs_reram/verilog/behavioral/sky130_bmlabs_reram__cim32x32.sv` |

### Cross-checks (the build fails if the repo disagrees with itself)
- X2 count = number of X2 sites.
- Bitstream config fields and widths = the primitive's parameters and widths (93 bits = 16+16+16+16+10+10+7+2).
- Behavioral-model result format width (2+5+14) = primitive `RESULT_DATA` width (21).
- Every pin in `constraints.pcf` sits on an IOBUF site.

### Not in the repo (still a team answer)
- How INT8 activations/weights map to the 8-bit PWM input, and the 14-bit count back to an output value.
- What values to choose for the eight X2 parameters.
- Analog effects (the behavioral model does not simulate them).

### Commands
```console
x2c chip profile           # write the profile
x2c chip profile --check   # exit 1 if it is stale
uv run pytest -q tests/test_chip_profile.py
```

## Chip flow: `x2c build`

`x2c build` builds any Verilog design for the real chip (`classic_fabric_reram`) with the chip
repo's own recipe (`user_designs/makefile.include`, unchanged), in `chip_build/<name>/`, and writes
`flow_report.json`. It is the hard check behind A1's choices and the source of real failures for A3.

### Steps
1. Copy the RTL to `chip_build/<name>/rtl/`, write a Makefile that includes the repo recipe.
2. `make synth`, then read the top module's ports from the netlist.
3. **Pin budget**: one IOBUF site per port bit; the fabric has **36**. (The 28 `all[...]` names in
   the repo's `constraints.pcf` are example assignments only, as the team confirmed.) Over budget
   -> stop with `pin_budget_exceeded`: the operator needs a narrower transfer protocol
   (serialised or multiplexed), which costs extra transfers.
4. Write `design.pcf` (port bits -> IOBUF sites in site order), `make pnr bitstream hex`.
5. Checks: device utilisation (used/available per primitive), every pin really constrained,
   FASM not empty, bitstream size = what the frame layout implies (35,304 bytes).

### Failure causes (for A3)
`failure = {stage, cause, evidence}`; `cause` comes from fixed rules and is `unknown` when no rule
matches (never guessed). `RULE_EVIDENCE` in `src/x2c/chip/chip_flow.py` records which rules were checked on
real tool output: all except `routing_failed` and `bitstream_size_mismatch`.

Two real tool behaviours found while calibrating:
* nextpnr places a pin that has no constraint by itself and still exits 0. The pin then sits on a
  site nobody chose, so the runner treats it as a failure (`unconstrained_io`).
* Over capacity shows up both in the utilisation table (`MACC_8x8_20: 20/18`) and as
  `no BELs remaining to implement cell type ...`.

### Examples (`hardware/designs/` and `hardware/test_designs/`; `x2c build --examples` checks each gives the expected result)
| Example | Expected | Why |
|---|---|---|
| repo_counter (counter.v from the repo) | ok | 11 pins, 9 logic cells |
| bitslice_acc_mux | ok | shift-and-add accumulator for bit-sliced X2 weights, result read 1 byte at a time: 30 pins |
| bitslice_acc_parallel | pin_budget_exceeded | same accumulator, 22-bit result in parallel: 42 pins |
| bitslice_acc_bad | syntax_error | deliberately broken |
| missing_mod | unknown_module | instantiates a module that does not exist |
| macc_over | over_capacity | 20 multipliers, 18 MACC blocks |

The bit-slice accumulator is an example operator that illustrates the pin trade-off; it is not a
decided design.

## V0 bitstream audit: `x2c audit`

`src/x2c/chip/chip_bitstream.py` runs the existing audit (`src/x2c/verification/bitstream_audit.py`) with a profile
derived from the chip repo, and the chip flow runs it as its last stage. A build is `ok` only if
the bitstream passes both audit levels.

### Profile (every value from a repo file, cross-checked)
| Value | Chip | Source | Demo-fabric default it replaces |
|---|---|---|---|
| rows per frame | 20 | `sky130-reram-fpga/src/chip_core.sv` FABRIC_NUM_ROWS | fabric.csv rows - 2 border rows (gave 18: wrong here) |
| first row word | Y19 (`rows - 1 - Y`) | spec `ArchSpecs.IncludeBorderRows = True` | `rows - Y` |
| exact size | 35,304 bytes | `chip_core.sv` BITSTREAM_LENGTH_WORDS = 0x227A words (the chip's SPI loader) | 16,384 (simulation default) |
| frames, bits/row, header, desync | 20, 32, sync header, bit 20 | spec `ArchSpecs`, checked against `chip_core.sv` | same values |

### Evidence
7 real chip bitstreams (counter, bitslice_acc_mux, reram_cim_32x32, x2_vendor_cfg, macc_8x8_20,
ram_32x4_2r_1w, register_file): every FASM-required bit set, no other bit set, 0 unparsed lines.
With the demo layout, the same bitstreams miss 40-90% of their bits.

Tests (`tests/test_chip_bitstream_audit.py`) flip one `TDC_TIME_OUT` bit in a real bitstream: the
structural level still passes (known limit), the decode level reports exactly 1 missing bit and names
`TDC_TIME_OUT`. A stray extra bit is reported as unexplained.

### Safety
`bitStreamSpec.bin` is a pickle. It is loaded with an unpickler that refuses every class or function
reference, so only plain data can come out of it (checked: the file contains none).

## Signed multiplies on the hard multipliers

### The problem
The chip has 18 hard multiplier blocks (MACC_8x8_20). Each one multiplies two **unsigned** 8-bit numbers and adds a 20-bit
`C` (tile RTL `ip/fabulous-tiles/primitives/MACC_8x8_20/fabulous/MACC_8x8_20.v`: `product = OPA * OPB`,
`sum = product + C`).

Yosys splits a wider multiply into 8x8 pieces and marks the pieces that hold a sign bit "signed". The adapter between
Yosys and the block, `ip/fabulous-tiles/primitives/MACC_8x8_20/yosys/techmap/MACC_8x8_20_map.v` (mole99/fabulous-tiles @
0651993f), ignores that mark. So a negative piece is read as a large positive number, like
`np.int8(-3).view(np.uint8) == 253`. It also wires bits 7..0 even when a piece is narrower, which leaves undefined bits.

### The fix
`src/x2c/chip/techmap/MACC_8x8_20_map_signed.v` is a copy of that adapter with two changes:
1. The operands are zero-extended to 8 bits (no undefined bits).
2. Signed pieces get a correction through the block's own `C` input. A W-bit operand with sign bit m is
   `unsigned - m*2^W`, so
   `a*b = a_u*b_u - am*(b_u << AW) - bm*(a_u << BW) + am*bm*2^(AW+BW)`.
   The block still computes `a_u*b_u + C`; the correction is `C`. For unsigned pieces `C = 0`, as in the original.

`src/x2c/chip/chip_flow.py` puts it into the Makefile it generates in `chip_build/<design>/`. That file is ours; the line comes
after the repo recipe is included, so it wins:
```make
MULTIPLIER_MAP_PATH := -multiplier-map <project>/src/x2c/chip/techmap/MACC_8x8_20_map_signed.v 8 8 2 2 12
```
Before each build, `chip_flow` checks that `user_designs/makefile.include` still has exactly
`MULTIPLIER_MAP_PATH = -multiplier-map ../../../../ip/fabulous-tiles/primitives/MACC_8x8_20/yosys/techmap/MACC_8x8_20_map.v 8 8 2 2 12`.
If the repo changes it, the build stops with "re-check ... before building", so nothing is overridden silently.
Each `flow_report.json` records `multiplier_map` (which file, what it replaces, why).

### Evidence

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

**Upstream:** checked on 2026-11-17, the latest `mole99/fabulous-tiles` (`4fcdd52`) still has the same map file, unchanged
since it was added; Yosys only passes the map through (`synth_fabulous -multiplier-map`), and FABulous ships no map of its
own for this block. The corrected adapter is the fix the upstream map needs; once it is fixed upstream, the override can be removed.

## The FABulous demo fabric (before the chip repo)

Before the chip repo was the target, the chain ran on FABulous's own demo fabric; `x2c chain --target demo` and the
`x2c demo …` commands still use it. Reproduced on Ubuntu 24.04 (Python 3.12, 1 CPU, 4 GB RAM). Everything below produced the fixture in
`tests/fixtures/fabulous_demo/`. **Not tested:** native Windows (use WSL2, Docker or Codespaces; check the FABulous docs).

1. `python3 -m venv fabenv && . fabenv/bin/activate && pip install fabulous-fpga`   -> `FABulous --version` = 2.2.0
2. CAD tools: OSS CAD Suite build **2026-06-25** (the FABulous docs warn against Yosys nightlies after 29 June; this build has Yosys 0.66+154,
   nextpnr-generic 0.10-82, Icarus 14.0). Linux x64 archive (717 MB):
   `https://github.com/YosysHQ/oss-cad-suite-build/releases/download/2026-06-25/oss-cad-suite-linux-x64-20260625.tgz`, extracted (~2.4 GB);
   then `export PATH=<extract dir>/oss-cad-suite/bin:$PATH`. (`FABulous install oss-cad-suite` installs the LATEST build instead.)
3. `FABulous create-project demo`
4. **Offline quirk:** at start-up FABulous 2.2 tries to download a chip PDK (`FAB_PDK='ihp-sg13g2'` in `demo/.FABulous/.env`) from
   fossi-foundation.github.io. A bitstream does not need a PDK; with no access to that host the download fails (`Name or service not known`).
   Deleting the `FAB_PDK` line avoided it. On a normal internet connection FABulous simply downloads the PDK; no edit is needed.
5. `FABulous -p demo run "load_fabric; run_FABulous_fabric; gen_user_design_wrapper user_design/sequential_16bit_en.v user_design/top_wrapper.v; compile_design ./user_design/sequential_16bit_en.v; exit"`
   -> `Compile flow completed successfully.` Produces `sequential_16bit_en.bin` (12,024 bytes), `.fasm`, `.json`, nextpnr log, `.FABulous/bitStreamSpec.csv`.
   (`run FABulous.tcl` does NOT work: `run` takes commands, not a filename.)
6. `uv run python tools/extract_bitstream_fixture.py demo sequential_16bit_en tests/fixtures/<name> --tool FABulous=2.2.0 ...` builds a small audit fixture.
7. `x2c demo audit --demo` (or `--bin ... --fabric-csv ... [--fasm ... --spec ...]` for your own build).
