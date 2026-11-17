# Flow step 2: eFPGA side of the X2 path

## The recombiner (built)
The X2 returns 14-bit column counts. To get a layer's real output, the eFPGA adds them back up:
`acc += (neg ? -1 : +1) * (count << shift)`, one step per (input slice, sign bank, weight digit, row fold).
`compiler/x2_pack.py recombine_steps()` produces the step list from a packed layer.

| Piece | File | Check |
|---|---|---|
| operator `x2_acc_step` (one step, combinational) | `verification/rtl_check.py` OPS, reference in `agents/rtl_agent.py` | Yosys eval vs NumPy, 636 vectors (edges incl. 32-bit wrap); A2 can be asked to write it |
| design `x2_recombiner` (clocked, 32-bit acc, read 1 byte at a time) | `examples/chip_designs/x2_recombiner.v` | Icarus: X2 counts -> outputs == NumPy for conv, depthwise, pointwise, linear |
| on the chip | `run_chip_flow.py --example x2_recombiner` | 32/36 pins, 227/2160 logic cells, V0 audit exact (5392/5392 bits); nextpnr estimate 21-26 MHz (delay model, not silicon) |

The 1-bit cases use counts produced by the BM Labs X2 behavioral model (flow step 1 logs), so
BM X2 model -> recombiner RTL -> NumPy is checked end to end.

Fix found on the way: `rtl_check` only read Yosys results printed as `<width>'<bits>`. For 32-bit outputs Yosys
prints fully defined values as plain decimals (`= 2147483647.`), so 337 of 636 results were lost.
`parse_eval_results` now reads both forms.

## The other eFPGA operators in the tiny MobileNetV2 graph
| Graph node(s) | Count | Operator | Status |
|---|---|---|---|
| after every X2 layer | 17 | `x2_acc_step` / `x2_recombiner` | built (this step) |
| requantise the 32-bit result to INT8 | 17 | `requant_i32_i8` | exists; scale 19/2^10 is ASSUMED, real per-layer scales are not in compiler_graph.json |
| `add` (residual) | 3 | `add8_sat` | exists; saturating INT8 is ASSUMED |
| `flatten` | 1 | none | no logic: only the order of values |
| `pool` | 1 | not built | kind and size are not recorded in compiler_graph.json |
| `shape_preserving` | 27 | not built | kinds are not recorded in compiler_graph.json (in MobileNetV2 usually BatchNorm / ReLU6, unconfirmed) |
