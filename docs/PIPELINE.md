# The pipeline and the deployment bundle

How `x2c pipeline` runs every stage in order, what each stage checks, and how a deployment bundle is replayed from
its own files.

- [The pipeline: `x2c pipeline`](#the-pipeline-x2c-pipeline)
- [Replaying a bundle: `x2c replay`](#replaying-a-bundle-x2c-replay)

## The pipeline: `x2c pipeline`

Runs on Linux (or WSL), because stages 2–6 need the chip tools.

| # | key | runs | PASS means |
|---|---|---|---|
| 1 | `environment` | tool versions, chip repo (fingerprint taken), BMsemi_Model, X2 model in use | everything the later stages need is present |
| 2 | `operators` | `x2c chain --op all --target chip --runner native` | relu8, add8_sat and requant all PASS: Yosys vs NumPy, chip flow, bitstream, V0 audit, netlist simulation, timing ≥ 14 MHz. On a failure, A3's rule diagnosis appears in the row |
| 3 | `formal` | `x2c formal` | 21/21: operators proved for all inputs, chip netlists equivalent, lint clean, engine bounded check, broken copies caught |
| 4 | `engine` | reuses `chip_build/x2_engine2_4bit` if its source is unchanged (the `FORMAL` block is ignored), else `x2c build --example x2_engine2_4bit` | flow ok, V0 audit complete, fmax ≥ 14 MHz |
| 5 | `x2-model` | `x2c sim x2 --sim` | Python X2 golden == the X2 developer's Verilog model |
| 6 | `engine-sim` | `x2c sim engine2 --sim` | two-block engine exact on all layer kinds; cost-model error reported |
| 7 | `deploy` | `x2c deploy --workload <m> --sim` for mobilenetv2, mnist, deepgate | every tensor exact (4-bit path + engine RTL), bundle replays, golden == repo results |
| 8 | `replay` | `x2c replay deploy/*` | each bundle reproduces every tensor from its files alone |
| 9 | `status` | A4's `chip_status()` + the chip repo fingerprint again | chip repo unchanged; missing X2 facts listed |
| (10) | `tests` | `pytest -q` (only with `--with-tests`) | all pass |

Options: `--rebuild` rebuilds the engine bitstream, `--with-tests` adds the test suite, and `--only`/`--skip` take a
comma list of stage keys (for example `--only environment,operators`).

**Output:** each stage prints PASS or FAIL and where its results are:
- `saved here: <file>` for a file written by this run;
- `reused from <date>: <file>` for a file kept from an earlier run, such as the engine bitstream when its source is
  unchanged;
- `NOT FOUND: <file>` for a file the stage should have produced.

The summary goes to `reports/pipeline/REPORT.md`:
- a table with each stage's result, what it found, and its log;
- the saved files per stage;
- the list of what the run does **not** prove;
- the facts each stage reported.

`report.json` holds the same data for machines. A failed stage does not stop the rest. The verdict is PASS, FAIL, or
PARTIAL when `--only` or `--skip` was used.

## Replaying a bundle: `x2c replay`

`x2c replay deploy/<model>` re-runs a deployment bundle from its own files: `host_program.json`, `x2_program.npz` and
`expected.npz`. No compiler, agent or model source is involved.

- It programs the stored cell levels into the 4-bit X2 model, runs every step, and compares every tensor with
  `expected.npz`. The result goes to `replay_report.json`.
- The input-offset correction reads the weight sums **back from the cells** (an all-ones input), not from the model.
- Every `x2c deploy` replays its own bundle before it reports PASS (`bundle_replays_exactly` in `report.json`).
- Requantization is stored as data (`epilogue.requant` in `host_program.json`), not as code. `x2c deploy` finds which
  integer formula reproduces the model's own requantization **exactly**, on the layer's real accumulators plus 64
  random ones: `floor`, `round_half_up`, `float_round_even`, `float_round_away` or `gemmlowp`. If none matches, the
  bundle names the BMsemi `fwd` function instead, and the replay refuses to run without BMsemi_Model rather than guess.
- For DeepGate, the real BMsemi `fwd` matches `gemmlowp` (TFLite-style SaturatingRoundingDoublingHighMul +
  RoundingDivideByPOT) on every layer, so DeepGate bundles replay without BMsemi_Model.

Tests (`tests/test_run_bundle.py`):
- all three models replay exactly;
- **one cell programmed one level too low** is caught at its layer;
- a changed expected tensor is caught;
- requant specs and formula matching work;
- a bundle that cannot be replayed is refused.
