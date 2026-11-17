# One command for the whole pipeline: `python3 bm.py pipeline` (Ubuntu)

| # | stage | runs | PASS means |
|---|---|---|---|
| 1 | environment | tool versions, chip repo (fingerprint taken), BMsemi_Model, X2 model in use | everything the later stages need is present |
| 2 | operators | `run_fab_chain.py --op all --target chip --runner native` | relu8, add8_sat and requant all PASS: Yosys vs NumPy, chip flow, bitstream, V0 audit, netlist sim, timing ≥ 14 MHz. On a failure, A3's rule diagnosis appears in the row |
| 3 | formal | `run_formal.py` | 21/21: operators proved for all inputs, chip netlists equivalent, lint clean, engine bounded check, broken copies caught |
| 4 | engine | reuses `chip_build/x2_engine2_4bit` if its source is unchanged (the FORMAL block is ignored), else `run_chip_flow.py --example x2_engine2_4bit` | flow ok, V0 audit complete, fmax ≥ 14 MHz |
| 5 | x2-model | `run_x2_check.py --sim` | Python X2 golden == the X2 developer's Verilog model |
| 6 | engine-sim | `run_x2_engine2_check.py --sim` | two-block engine exact on all layer kinds; cost-model error reported |
| 7 | deploy | `deploy_chip.py --workload <m> --sim` for mobilenetv2, mnist, deepgate | every tensor exact (4-bit path + engine RTL), bundle replays, golden == repo results |
| 8 | replay | `run_bundle.py deploy/*` | each bundle reproduces every tensor from its files alone |
| 9 | status | A4's `chip_status()` + chip repo fingerprint again | chip repo unchanged; missing X2 facts listed |
| (10) | tests | `pytest -q` (only with `--with-tests`) | all pass |

**Output:** for each stage, a console line with PASS or FAIL and where its results are saved:
`saved here: <file>` (written by this run), `reused from <date>: <file>` (for example, the engine bitstream when its
source is unchanged), or `NOT FOUND: <file>`. The summary is saved in `reports/pipeline/REPORT.md`, which has:
- a table with result, what was found and log for each stage;
- the saved files per stage;
- the "What this does NOT prove" list;
- the facts each stage reported.

`report.json` holds the same for machines. A stage failure doesn't stop the rest. The verdict is
PASS, FAIL, or PARTIAL when you used `--only`/`--skip`.

On Windows, `python bm.py pipeline` tells you to run it in Ubuntu: the chip tools are only there.
