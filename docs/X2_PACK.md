# X2 packing (flow step 1, S04 "X2 package")

`compiler/x2_pack.py` turns one matrix layer into what the X2 receives and rebuilds the layer output from the
X2's 14-bit column counts. `run_x2_pack_check.py` checks it end to end.

| Part | What it does |
|---|---|
| `weight_columns` | signed INT8 -> positive/negative banks (BMsemi deep_gate `Crossbar_array.py` l.29-30), each split into `cell_bits` digits (l.127, 165): 4 cells per weight with 4-bit cells, 16 with 1-bit |
| `pack_layer` | places blocks exactly like the partition (`chip_x2.place_groups`), fills each 32 x 32 tile, and refuses a layer whose worst-case COMPUTE could pass the 14-bit cap (16383) |
| inputs | each 8-bit input is sent as 4-bit slices (two COMPUTE passes) with 4-bit cells: 32 x 15 x 15 = 7200 fits; one 8-bit pass would reach 122400 |
| `program_commands` | the commands that change the block's cells into a tile's cells (only changed cells). 1-bit: SET / RESET, the chip's own commands. 4-bit: `WRITE_LEVEL`, which is NOT a chip command (the chip repo's X2 stores 1 bit) |
| `compute_beats` | one COMPUTE beat per used row, pwm = that row's input slice, `compute_all_columns` |
| `recombine` | `y = sum(2^(p*4) * sign * 2^(d*cell_bits) * count)`, adding row folds of the same columns |
| `required_tdc` | smallest TDC_TIME_OUT whose ceiling `(t << 8) | 0xFF` covers the layer's worst count (data dependent; 28 covers any full 32-row tile with 4-bit cells and 4-bit slices) |

Execution: 4-bit streams run on `MultiLevelX2` (a model of the 4-bit decision, checked against NumPy only).
1-bit streams run on `chip.x2_golden.X2Golden` and, with `--sim`, on the BM Labs X2 behavioral model in Icarus.

Evidence (2026-11-13): conv, depthwise, pointwise and linear layers with random INT8 weights, both cell sizes:
outputs equal NumPy exactly; the 1-bit streams on the BM Labs X2 model (Icarus) also equal NumPy exactly.

ASSUMED: patch order c-r-s. Not covered yet: stride/padding im2col (the packing works on patch vectors), and the
eFPGA side that sends these commands (flow step 3).
