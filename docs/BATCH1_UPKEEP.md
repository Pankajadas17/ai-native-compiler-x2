# Batch 1: upkeep fixes and one entry point (2026-11-16)

This batch answers items 5, 6, 7, 13, 14, 15 and 16 of the project review, and adds `bm.py`.

| # | file | from | to |
|---|---|---|---|
| 13 | `run_fab_chain.py` `write_report()` | each run **overwrote** the report, so `--op relu8` left only relu8 | the run **replaces only the operators it ran**; the others stay. Each result has its own `run_at`, and `ops_this_run` lists what this run did |
| 14 | same | `"clock_target_mhz": 12.0` for the chip; time without a zone | `clock_target_mhz` = the project clock from `x2_facts` (14.0, what PASS is judged against), plus `nextpnr_target_mhz: 12.0` and a note. Times are ISO with the UTC offset (`+0530` on Windows; WSL shows `+0000` unless its timezone is set) |
| 6 | `chip/chip_repo.py` | the X2 model was only read from inside the chip repo | new `BM_X2_REPO` (a repo or the `.sv` file). `ChipRepo.path()` redirects the X2 model path, so **every** reader follows it: simulations, profile, hardware config, chain. `provenance()` records which model was used. New `x2_model_override()`, `x2_model_label()` |
| 6 | `run_hardware_config.py` line 24 | `BEHAV = "ip/sky130_bmlabs_reram/…cim32x32.sv"` (hard-coded) | `BEHAV = x2_model_label()` |
| 5 | `chip/fabric_costs.py` (new) | eFPGA use was known only after a build | per-block costs **measured** from `chip_build/*/flow_report.json` + `pnr.log` (engine, relu, add, requant) → `specifications/fabric_costs.json`; `estimate(blocks)` sums them against the budget. Missing builds are reported, never guessed |
| 5 | `deploy_chip.py` | – | `report.json` and `plan.json` gain `fabric`: the engine alone, and an estimate of the engine with the host steps moved onto the fabric |
| 7 | `deploy_chip.py` | MobileNetV2 used seeded random weights only | `--weights file.npz`: one INT8 array per layer name, shape (K, C/groups, R, S), optional `<layer>__bias`. Shapes and the INT8 range are checked, shifts are re-calibrated, and the report says how many layers are real |
| 15 | `README.md` | – | commands section rewritten around `bm.py`; `BM_X2_REPO` added; known limits listed |
| 16 | `bm.py x2-recheck` | a manual note | one command (Ubuntu): profile → hardware config → golden vs model → packing → engine → deploys |
| – | `bm.py` (new) | 30 scripts to remember | one entry point with groups, terminal hints and workflows (`all`, `x2-recheck`). The scripts themselves are unchanged |

New tests: `test_bm_cli.py`, `test_fabric_costs.py`, `test_chip_repo_x2.py`, `test_chain_report_merge.py`, plus 3 in
`test_deploy_chip.py` (real weights, bad weights refused, fabric estimate).

Measured costs (sandbox builds, the same designs as yours):

| block | design | logic cells | pins | MACC | fmax |
|---|---|---|---|---|---|
| x2_engine | x2_engine2_4bit | 1,792 | 35 | 0 | 14.45 MHz |
| relu | chain_relu8 | 15 | 17 | 0 | 151.5 MHz |
| add | chain_add8_sat | 35 | 25 | 0 | 42.2 MHz |
| requant | chain_requant_i32_i8 | 134 | 26 | 4 | combinational |

The estimate for engine + relu + add + requant in one design is **1,976 / 2,160 cells, 4 / 18 MACC, 35 pins**, so it would fit.
This is a sum of measurements, not a build. Building that combined design is the real test, and it is part of review item 2/3.
