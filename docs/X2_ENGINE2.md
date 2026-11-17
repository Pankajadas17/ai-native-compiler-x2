# Two-block X2 engine: both X2 blocks at once, measured

A1's fastest schedules program one X2 block while the other computes, but the flow-step-4 engine drove only one block.
`examples/chip_designs/x2_engine2.v` drives both. It has the same column rule, the same command format, and per-block
accumulator banks.

## Pins (35 of 36)
- `clk`, `rst`, `cmd_data[11:0]`, `cmd_hi`, `cmd_strobe`, `acc_clear`
- `cmd_ready[1:0]` and `idle[1:0]`: one each per block, so the host feeds one block while the other is busy
- `acc_sel[3:0]`: bit 3 picks the block's bank for reading and clearing
- `byte_sel[1:0]` and `acc_byte[7:0]`

The high command beat carries the block (`cmd_data[11]`) and the input slice (`cmd_data[10]`) in bits that were unused,
so the old `slice` pin is gone.

## Correctness points
- **Results** from the two blocks can arrive in the same clock. A round-robin arbiter grants one and holds the other
  with `RESULT_READY`. The BM model keeps a result stable until it is accepted (behavioral `.sv` l.365, l.690).
- **Slice of a result:** the slice of the command its block accepted last. The model takes no command while a result
  is pending (`o_ready = !busy && !result_valid`, l.533).
- **Accumulate pipeline:** decode+shift, read, add+write. A result for the accumulator being written in the same clock
  is forwarded; neighbouring columns share an accumulator.

## Measured (BM behavioral X2, 1-bit cells; `python3 run_x2_engine2_check.py --sim`)

| layer | tiles | one block | two blocks | latency cut | predicted cut | model error |
|---|---|---|---|---|---|---|
| conv | 4 | 374,437 | 172,450 | 53.9 % | 53.9 % | ≤ 0.01 % |
| dw | 4 | 135,146 | 68,315 | 49.5 % | 49.5 % | ≤ 0.03 % |
| pw | 8 | 238,406 | 122,496 | 48.6 % | 48.6 % | ≤ 0.04 % |
| fc | 5 | 309,211 | 174,880 | 43.4 % | 43.4 % | ≤ 0.02 % |

- Both layouts reproduce every accumulator (the engine rule) and NumPy exactly.
- "Predicted" is `compiler/x2_schedule.simulate`, the model A1 uses, fed with each run's exact command counts.
- So A1's two-block assumption is now a measured result, not an assumption.

## Chip builds (`run_chip_flow.py --example x2_engine2` / `x2_engine2_4bit`)

| design | logic cells | X2 | Fmax (nextpnr, 12 MHz target) | V0 audit |
|---|---|---|---|---|
| x2_engine2 (1-bit) | 710 / 2160 | 2 / 2 | 16.7 MHz | 18177/18177 |
| x2_engine2_4bit | 1792 / 2160 | 2 / 2 | 14.5 MHz | 47236/47236 |

The first version failed 4-bit timing at 7.8 MHz: an accumulator multiplexer and a 32-bit adder in one clock. The read
stage with forwarding fixed it. 4-bit is size- and timing-checked only, because the chip's X2 model computes 1-bit cells.

## Changes
- **New:**
  - examples/chip_designs/x2_engine2.v and x2_engine2_4bit.v
  - sim/x2_engine2_tb.v (one command queue per block, sharing the host port)
  - run_x2_engine2_check.py
  - tests/test_x2_engine2.py and tests/fixtures/x2_engine2/
- `run_chip_flow.py`: two EXAMPLES lines after `x2_engine_4bit`:
  ```python
      "x2_engine2": (EX / "x2_engine2.v", "x2_engine2", "ok", None),
      "x2_engine2_4bit": ([EX / "x2_engine2_4bit.v", EX / "x2_engine2.v"], "x2_engine2_4bit", "ok", None),
  ```
- `compiler/x2_schedule.py`: `simulate()` split out of `validate_schedule` (same numbers). The two-block note in the
  docstring and `ASSUMPTIONS` now states the measured result.
