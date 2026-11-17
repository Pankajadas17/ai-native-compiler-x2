# Chip profile (Phase 1)

`specifications/chip_profile.json` describes the real BM Labs chip, derived from the
chip repo `sky130-reram-fpga` (checked out at `bm_verif/sky130-reram-fpga`, with submodules).
It replaces hand-typed hardware numbers. It does not yet replace `hardware_config.json`:
the agents are switched over in later phases.

## Rules
- The chip repo is a read-only input, like `BMsemi_Model`. Code only opens its files with `"rb"`.
- `run_chip_profile.py` fingerprints every file in the repo (path, size, mtime) before and after
  the run and fails if anything changed.
- Text is LF-normalised before hashing, so Windows (CRLF) and Linux (LF) checkouts of the
  same commit give the same `input_digest` and the same JSON.
- Builds of chip designs go to `bm_verif/chip_build/<design>/`, never inside the repo.

## Where each number comes from
| Profile field | Source file in the chip repo |
|---|---|
| `fabric.grid`, `fabric.resources`, X2 sites, IOBUF sites | `user_designs/fabrics/classic_fabric_reram/macro/sky130A/fabulous/.FABulous/bel.txt` |
| `x2.config_bits_per_block`, `x2.config_fields` | same folder, `bel.v2.txt` (CFG entries of the X2 BEL) |
| `io.*` pins | `user_designs/fabrics/classic_fabric_reram/constraints.pcf` |
| `x2.buses`, `x2.parameters`, `x2.rows/cols` (= 2^width of ROW/COL_INDEX) | `ip/extra-tiles/primitives/RERAM_CIM_32x32/yosys/primitives/RERAM_CIM_32x32.v` |
| `x2.behavioral_model` (modes, delays, result format) | `ip/sky130_bmlabs_reram/verilog/behavioral/sky130_bmlabs_reram__cim32x32.sv` |

## Cross-checks (the build fails if the repo disagrees with itself)
- X2 count = number of X2 sites.
- Bitstream config fields and widths = the primitive's parameters and widths (93 bits = 16+16+16+16+10+10+7+2).
- Behavioral-model result format width (2+5+14) = primitive `RESULT_DATA` width (21).
- Every pin in `constraints.pcf` sits on an IOBUF site.

## Not in the repo (still a team answer)
- How INT8 activations/weights map to the 8-bit PWM input, and the 14-bit count back to an output value.
- What values to choose for the eight X2 parameters.
- Analog effects (the behavioral model does not simulate them).

## Commands
```powershell
python run_chip_profile.py           # write the profile
python run_chip_profile.py --check   # exit 1 if it is stale
python -m pytest -q tests/test_chip_profile.py
```
