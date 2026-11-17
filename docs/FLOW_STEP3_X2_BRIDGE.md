# Flow step 3: X2 controller (host bridge) on the eFPGA

## Why a bridge
The fabric's memory is 36 x RAM_32x4 = 4608 bits; one 4-bit-cell tile is 4096 bits of weights. The command stream
from `compiler/x2_pack.py` therefore comes from a host through the pins, and `examples/chip_designs/x2_bridge.v`
hands it to the X2 block (`RERAM_CIM_32x32`, the chip's own primitive) and returns its results.

| Host pins (30 of 36) | Meaning |
|---|---|
| `cmd_data[11:0]`, `cmd_hi`, `cmd_strobe` | 22-bit command `{all_cols, last, pwm, col, row, mode}` loaded as two 12-bit halves; the second half sends it |
| `cmd_ready` | the previous command was accepted by the X2 |
| `res_valid`, `res_sel[1:0]`, `res_byte[7:0]`, `res_last` | the X2's 21-bit result `{mode, column, count}`, read one byte at a time |
| `res_pop` | consume the result (drives the X2's `RESULT_READY`) |
| `clk`, `rst` | |

X2 settings (`TDC_TIME_OUT`, targets, ...) are parameters of the bridge, i.e. bitstream settings.

## Checks
| Check | Result (2026-11-13) |
|---|---|
| bridge connects all 58 ports of `RERAM_CIM_32x32` | yes |
| simulation stand-in `sim/RERAM_CIM_32x32_sim.v` connects the BM Labs macro like `src/fabric_wrapper.sv` (l.259-312, 566-624) | same 44 connections |
| host -> bridge pins -> X2 model, flow step 1's packed conv / depthwise / pointwise / linear | every result = X2 golden model; identical line for line to step 1's direct X2 run; rebuilt outputs = NumPy |
| cost of the bridge | conv: 376,514 cycles vs 375,082 direct (+0.4%); programming dominates |
| on the chip (`run_chip_flow.py --example x2_bridge`) | ok: 30/36 pins, 42 logic cells, 1 of 2 X2 blocks (X20Y6), V0 audit exact (1813/1813 bits, X2 settings included); nextpnr estimate 80-85 MHz (not silicon) |

## Limits
* Only 1-bit cells can be programmed with the chip's commands (SET/RESET). The 4-bit `WRITE_LEVEL` stream (flow step 1)
  has no chip command, so it cannot go through the bridge.
* The bridge and the recombiner (flow step 2) are separate designs; joining them on the chip is a later step.
* The host is the testbench here.

## Finding
`src/fabric_wrapper.sv` wires a 3-bit X2 error code into the fabric (`EXT_ERROR0..2`, l.259-261), but the Yosys
primitive user designs see declares only `ERROR0`, `ERROR1` (`RERAM_CIM_32x32.v` l.60-61): error bit 2 cannot be read
from fabric logic.
