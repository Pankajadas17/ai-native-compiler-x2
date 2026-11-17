# V0 bitstream audit on the real chip (Phase 5)

`chip/chip_bitstream.py` runs the existing audit (`verification/bitstream_audit.py`) with a profile
derived from the chip repo, and the chip flow now runs it as its last stage. A build is `ok` only if
the bitstream passes both audit levels.

## Profile (every value from a repo file, cross-checked)
| Value | Chip | Source | Demo-fabric default it replaces |
|---|---|---|---|
| rows per frame | 20 | `src/chip_core.sv` FABRIC_NUM_ROWS | fabric.csv rows - 2 border rows (gave 18: wrong here) |
| first row word | Y19 (`rows - 1 - Y`) | spec `ArchSpecs.IncludeBorderRows = True` | `rows - Y` |
| exact size | 35,304 bytes | `chip_core.sv` BITSTREAM_LENGTH_WORDS = 0x227A words (the chip's SPI loader) | 16,384 (simulation default) |
| frames, bits/row, header, desync | 20, 32, sync header, bit 20 | spec `ArchSpecs`, checked against `chip_core.sv` | same values |

## Changes to `verification/bitstream_audit.py` (diff: `docs/bitstream_audit_phase5.diff`)
1. **Bug fix:** FASM literals without a width (`'b0111110`) were reported as unparsed. nextpnr writes
   the X2 parameters this way, so their bits were skipped. Width now defaults to the bit range.
2. New `FabricProfile.top_row_y` (default `None` = old behaviour, so the demo fixtures are unaffected).
3. Docstring note on the chip layout.

## Evidence
7 real chip bitstreams (counter, bitslice_acc_mux, reram_cim_32x32, x2_vendor_cfg, macc_8x8_20,
ram_32x4_2r_1w, register_file): every FASM-required bit set, no other bit set, 0 unparsed lines.
With the demo layout, the same bitstreams miss 40-90% of their bits.

Tests (`tests/test_chip_bitstream_audit.py`) flip one `TDC_TIME_OUT` bit in a real bitstream: the
structural level still passes (known limit), the decode level reports exactly 1 missing bit and names
`TDC_TIME_OUT`. A stray extra bit is reported as unexplained.

## Safety
`bitStreamSpec.bin` is a pickle. It is loaded with an unpickler that refuses every class or function
reference, so only plain data can come out of it (checked: the file contains none).
