# X2 golden model (Phase 2)

`chip/x2_golden.py` is a Python model of ONE X2 block (`sky130_bmlabs_reram__cim32x32`), written
from the BM Labs behavioral model in the chip repo and checked against it in Icarus Verilog.
It models what the block returns (mode, column, count, last), not cycle timing.

## What the X2 computes (from the behavioral model)
| Command | Result |
|---|---|
| COMPUTE (one beat per row, last beat has `compute_last`) | per column: `sum(pwm[row] for selected rows where cell(row, col) is SET)`, clamped |
| READ | `cell level + pwm`, clamped |
| SET / RESET | cell state 1 / 0, level = `((t1 + t2) >> 1) + pwm` (mod 2^14); returns `level + pwm of the previous READ`, clamped |

* Clamp ceiling = `(TDC_TIME_OUT[5:0] << 8) | 0xFF`.
* COMPUTE columns: all 32 with `compute_all_columns`, otherwise the set of beat columns; emitted
  in ascending order, `last` on the highest.
* Same row twice in one stream: the later pwm wins. A non-COMPUTE command cancels an unfinished stream.

## Consequences for the compiler (findings, not assumptions)
1. **Weights are binary.** COMPUTE uses only the SET/RESET state; the 14-bit level does not enter
   COMPUTE. A multi-bit (e.g. INT8) weight therefore needs several cells or several passes
   (bit slicing), plus a scheme for signs. How the team wants this done is a team decision.
2. **Inputs are unsigned 8-bit** (pwm 0..255).
3. **Saturation:** a full 32-row sum is at most 32 x 255 = 8160. It fits only if
   `TDC_TIME_OUT >= 31` (ceiling 8191). With all parameters 0 (as in the chip repo's
   `reram_cim_32x32` example design) the ceiling is 255.

## Check
`run_x2_check.py --sim` (Ubuntu terminal) writes 4 scenarios to `chip_build/x2_check/`, simulates the
behavioral model with `sim/x2_vector_tb.sv`, and compares every result with the golden model.

| Scenario | Covers |
|---|---|
| directed | each rule once |
| binary_matmul_32x32 | full array, 0/1 weights, `y = x @ W` for 4 vectors (also checked with plain Python) |
| zero_parameters | all parameters 0: ceiling 255, saturation |
| random_mixed | 120 random operations with random parameters (53 saturated results) |

`tests/test_x2_golden.py` replays bundled simulator logs (`tests/fixtures/x2_check/`) so it runs on
Windows without Icarus, and plants 9 bugs in the golden model to show the logs catch each of them.
`test_fixture_was_recorded_from_this_model_version` fails if the chip repo's model changes.

## Not covered
* Timing (delays, dead time) and error codes; the checker compares values and order only.
* Analog behavior (the behavioral model does not simulate it).
* The X2 inside the fabric (that needs the fabric simulation, Phase 6).
