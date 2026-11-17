# Item 11: replaying a deployment bundle from its files alone (`run_bundle.py`)

Before this change, a bundle held cell levels and a step list, but nothing could execute it, and DeepGate's
requantization was stored only as the words "repo requant". Now:

| file | from | to |
|---|---|---|
| `run_bundle.py` (new) | – | `replay(folder)`: reads `host_program.json`, `x2_program.npz` and `expected.npz` only. It programs the stored cells into the 4-bit X2 model, runs every step and compares every tensor. The offset correction reads the weight sums **back from the cells** (an all-ones input), not from the model. Writes `replay_report.json` |
| `deploy_chip.py` | the bundle was written and not re-checked | every deploy replays its own bundle: new check `bundle_replays_exactly`, plus `bundle_replay` in `report.json` |
| `deploy_chip.py` `write_bundle()` | `"epilogue": "repo requant"` | `"epilogue": {"requant": {...}}` as data. Each X2 layer also stores what the package needs: cell bits, rows, cols, input/magnitude bits, input split, TDC setting |
| `deploy_chip.py` `_deepgate_spec()` (new) | – | finds which integer formula reproduces BMsemi's `fwd` **exactly** on the layer's real accumulators plus 64 random ones (`floor`, `round_half_up`, `float_round_even`, `float_round_away`, `gemmlowp`). If none matches, the bundle says `bmsemi_deepgate_fwd` and the replay imports the repo's `fwd`; without BMsemi_Model it refuses with a clear message instead of guessing |
| `compiler/int8_net.py` | requantization only as a Python function | `Layer.requant_spec`, `requant_from_spec()`, `match_formula()`, `FORMULAS`. MNIST is marked `identity` |
| `bm.py` | – | `python bm.py replay deploy/<model>` |

Tests (`tests/test_run_bundle.py`):
- all three models replay exactly;
- **one cell programmed one level too low** is caught at its layer;
- a changed expected tensor is caught;
- requant specs and formula matching work;
- an unreplayable bundle is refused.

Checked in the sandbox with a stand-in BMsemi `fwd`: DeepGate's requantization was matched as `floor`. A deliberately
odd `fwd` matched no formula; the bundle then used the repo's `fwd`, and without BMsemi_Model it refused. **On your
machine the report shows which formula matched the real `fwd`** (`host_program.json` → `epilogue.requant.formula`).

**Measured on your machine (2026-11-16):** the real BMsemi DeepGate `fwd` matches the **`gemmlowp`** formula (TFLite-style
SaturatingRoundingDoublingHighMul + RoundingDivideByPOT) on every layer, so DeepGate bundles replay without BMsemi_Model.
