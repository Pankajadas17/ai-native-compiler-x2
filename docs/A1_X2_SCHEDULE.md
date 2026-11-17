# A1 chooses the X2 schedule

## The problem in one paragraph
The chip has 2 X2 blocks, and a layer needs more tiles than that (tiny MobileNetV2: 64 tiles on the chip). So tiles are
programmed, used, and replaced. Programming a tile is far slower than using it: per weight cell about 365 cycles, while an
MVM takes about 511 + 3 per row. The schedule decides which layers stay programmed, which block takes each tile, and in
what order; that decides the latency. Today's code kept the layer that saves the most programming resident and added the
cycles up as if one block did everything in turn. Using both blocks, so that one programs while the other computes, can
hide much of the work. It's like prefetching the next batch while the GPU runs this one.

## Cost model, checked against real simulations
The four X2 engine simulations from flow step 4 fit one formula within 1.6 cycles:
`cycles = 365 x (SET + RESET commands) + 511 x MVMs + 3.1 x input beats`.
`tests/test_x2_schedule.py` re-fits it from your recorded logs every run.
- **Programming:** every weight cell of a tile programmed once. This is an upper bound: the real count depends on the
  values, and the calibration runs used 21–29 % of it.
- **Not modelled** (listed in every result):
  - eFPGA operators between X2 layers;
  - host transfers.
- **Both blocks at once is built and measured:** `x2_engine2.v` (docs/X2_ENGINE2.md) cut the latency of the 4 engine
  layers by 43–54 %, and `simulate()` predicted every run within 0.05 %. A schedule says `runs_on_built_x2_engine: true`
  only if it fits the one-block `x2_engine.v`.

## How A1 decides (same contract as before)
1. The LLM picks targets as before. Then it may call `describe_schedule`: per-tile program and compute cycles, the
   dependencies, the legal resident sets, the policies and today's latency. It does not see code's own best answer.
2. The LLM calls `propose_schedule` with a resident list and a policy (`earliest_free`, `round_robin`, `single_block`),
   or an explicit `[[layer, tile, block], ...]` list.
3. `validate_schedule` decides legality and computes the cycles. It rejects:
   - a tile scheduled twice, a missing tile, or a block that doesn't exist;
   - a resident block that is also used for streaming;
   - resident tiles that fill every block while layers still stream;
   - a tile issued before the layers it depends on (dependencies are traced through the graph, so residual adds count);
   - a resident set different from the plan's.
4. `best_schedule` tries every legal resident set with every policy. The system keeps the LLM's schedule only if it is
   legal and at least as fast as code's; otherwise code's, and the report says why (for example "llm schedule legal but 89.8 % slower").
5. The final plan's `chip.resident` follows the chosen schedule, and the final plan is re-validated by `validate_partition`.

## Results (sandbox, graphs built with your compiler.ir)

| graph | today (resident + one streaming block) | chosen | latency cut |
|---|---|---|---|
| mini (5 layers, 16 tiles, no 1-tile layer) | 4,487,870 cycles | no resident, earliest_free | 47.3 % |
| with a 1-tile first layer (like features_0_0) | 1,802,066 (c0 resident) | no resident, round_robin | 27.5 % |

In the second case code chooses MORE programming per inference (887,680 vs 730,000 cycles) for LESS latency,
because the two blocks overlap it. Your real MobileNetV2 graph is tested on your machine
(`test_real_tiny_mobilenetv2`): it checks legality and that the result is no slower than today, not exact numbers.

## Exact changes
- `compiler/x2_schedule.py`: new.
- `agents/mapping_agent.py`:
  - Docstring: one SCHEDULE paragraph. Imports: `copy`, `x2_schedule as xs`, and `chip_view, has_chip` from partition.
  - New `SCHEDULE_PROMPT`, added to the prompt only with the chip config.
  - Two tool schemas: `describe_schedule`, `propose_schedule`.
  - `_Session` gains `_chip_plan`, `describe_schedule`, `propose_schedule` and five schedule fields.
  - The goal string adds " Then choose the X2 schedule." with the chip config.
  - New `choose_schedule(g, hw, plan, session)`.
  - In `run_mapping_agent`, before `same = ...`:
    ```python
    schedule = None
    if final is not None and has_chip(hw) and final.get("chip", {}).get("layers"):
        final, schedule = choose_schedule(g, hw, final, session)
    ```
    and the result gains `"schedule": schedule`.
- `compiler/partition.py` and `compiler/chip_x2.py` are NOT changed. Neither is the partition validator or anything
  `test_ir_partition.py` checks.

New tests: `tests/test_x2_schedule.py`, `tests/x2_schedule_graphs.py`. Full diff: `docs/a1_x2_schedule.diff`.
