# ai-native-compiler-x2

[![ci](https://github.com/Pankajadas17/ai-native-compiler-x2/actions/workflows/ci.yml/badge.svg)](https://github.com/Pankajadas17/ai-native-compiler-x2/actions/workflows/ci.yml)

**AI agents plus a verification framework that compile INT8 neural networks onto a real chip design and check every step.**
The target is the open sky130 ReRAM-FPGA chip ([mole99/sky130-reram-fpga](https://github.com/mole99/sky130-reram-fpga),
`classic_fabric_reram`). It has two **X2** compute-in-memory crossbars (32×32, they do the matrix multiplies) next to a
**FABulous eFPGA** (2,160 logic cells, 36 pins). The pipeline takes a model all the way to an audited bitstream plus a
deployment bundle. Each result is compared with an independent reference, and that check decides PASS, not the AI agents.

```text
$ python3 bm.py pipeline
[1/10] Environment and inputs ............... PASS
[2/10] eFPGA operators on the chip .......... PASS   relu8, add8_sat, requant: chip flow, bitstream audit, netlist sim
[3/10] Formal proofs and lint ............... PASS   21/21: operators proved for ALL inputs, chip netlists equivalent
[4/10] Deployment engine bitstream .......... PASS   1792/2160 cells, 35/36 pins, both X2, audit 47236/47236, 14.45 MHz
[5/10] X2 golden vs the X2 developer's model  PASS
[6/10] Two-block engine in simulation ....... PASS   exact, 43-54 % faster than one block, cost model within 0.04 %
[7/10] Model deployment (3 models) .......... PASS   every tensor exact, layers run on the engine RTL
[8/10] Bundles replayed from their files .... PASS
[9/10] Chip status and read-only check ...... PASS
PIPELINE PASS: 9/9 stages in 4.7 min        summary saved here: reports/pipeline/REPORT.md
```

| what | result |
|---|---|
| Agents | **A1** partitions and schedules across both X2 blocks (LLM proposes, code validates); **A2** writes eFPGA operators in Verilog; **A3** diagnoses failed runs from evidence; **A4** generates tests and reports the chip status |
| Formal | the operators equal independent specs for every input; the chip-synthesized netlists equal their RTL; the proof finds a real signed-multiply bug in the chip's stock multiplier mapping (worked around without editing the chip repo) |
| Deployment | MobileNetV2-style CNN, an MLP chain and a 5×5 edge filter: INT8 lowering (BN folded, ReLU6 clamp, residual add, pool), 4-bit X2 cells, every tensor bit-exact, bundles replay from their own files |
| Not proven yet | the configured fabric (the chip's tile RTL loaded with our bitstream), timing-annotated simulation, silicon. See [Not done yet](#not-done-yet) |

## Try it (Linux or WSL)

```bash
git clone https://github.com/Pankajadas17/ai-native-compiler-x2 && cd ai-native-compiler-x2
git clone https://github.com/mole99/sky130-reram-fpga                     # the chip (read-only)
git -C sky130-reram-fpga checkout e0790eb11a1e95914f2beb8f175396a0924144ed && git -C sky130-reram-fpga submodule update --init --recursive
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
python3 -m pytest -q                                   # the tests (Verilog checks need Yosys: pip install yowasp-yosys)
# the full pipeline needs the OSS CAD Suite (Yosys 0.70+, nextpnr, Icarus, SymbiYosys, Verilator) and ~/chipenv with fasm:
python3 -m venv ~/chipenv && ~/chipenv/bin/pip install fasm numpy
python3 bm.py pipeline --rebuild
```

**Public mode.** The development team also uses a private model repository (BMsemi_Model) with its own trained models.
Without it, everything still runs. The deployment stage uses public stand-in graphs (`tests/fixtures/ir_public/`, made
by `tools/make_public_fixtures.py`) and seeded weights. The test files built on that private repository are not
collected, and the test header and the pipeline report say "PUBLIC mode". The CI badge above runs exactly this.

**Feedback is welcome.** Open an issue on anything unclear, unconvincing or wrong: the verification approach, the
agents' role, the evidence, or the code.

## The hardware in software terms

| part | analogy | budget (fixed by the team) |
|---|---|---|
| X2 crossbar | A tensor core that **stores the weights inside itself**. Loading weights is slow (about 365 cycles per cell); a 32×32 matrix-vector product is fast (about 511 cycles) | 2 blocks, 32×32 cells, 4 bits per cell (project decision) |
| eFPGA (FABulous) | A blank, reprogrammable circuit | 2,160 LUT4+FF, 36 IOBUF pins, 36 RAM 32×4, 18 MACC 8×8/20, 4 GBUF, 1 SYS_RESET |
| Bitstream | The compiled binary that configures the eFPGA (35,304 bytes) | – |
| Verilog / Yosys / nextpnr / Icarus | Source code / compiler / linker (placement and wires) / emulator | 14 MHz project clock |

## The pipeline

```
model (BMsemi_Model, read-only)
  1 IR ............ typed graph: layers, shapes, edges                       compiler/ir*.py, inputs/
  2 A1 ............ partition (matrix ops -> X2, rest -> eFPGA/host) and the  agents/mapping_agent.py,
                    X2 tile schedule over both blocks; LLM proposes, code     compiler/partition.py, x2_schedule.py
                    validates and keeps the faster legal one
  3 INT8 network .. BN folded, ReLU6 -> clamp, residual add, pool             compiler/int8_net.py
  4 X2 packing .... 8-bit weights -> 4-bit cells per 32x32 tile, inputs as    compiler/chip_x2.py, x2_pack.py
                    two 4-bit halves, command stream
  5 engine ........ Verilog -> Yosys -> nextpnr (pins, timing) -> bitstream   examples/chip_designs/x2_engine2_4bit.v,
                    -> V0 audit (every required bit decoded and checked)      chip/chip_flow.py, chip/chip_bitstream.py
  6 verification .. each tensor against an independent golden model; 4-bit   verification/, sim/, run_x2_*.py
                    Python X2 + 1-bit two-block engine RTL in Icarus
  7 bundle ........ deploy/<model>/: plan, host program, X2 cell images,      deploy_chip.py
                    expected tensors, bitstream + pin map, report
  8 replay ........ the bundle re-run from its files alone (programmed cells,   run_bundle.py
                    host steps, epilogue as data) must give every tensor
A2 (agents/rtl_agent.py) writes the eFPGA operators (relu8, add8_sat, requant_i32_i8 -> reports/rtl/)
A3 (agents/chain_debug_agent.py, verification/chain_debug.py) names the cause of any failed stage from its evidence
A4 (agents/verification_agent.py, agents/test_agent.py, verification/chip_tools.py) generates independent tests and
   reports the chip status and the missing X2 facts
```

## Folder layout

| folder | contents |
|---|---|
| `inputs/` | read-only adapters for BMsemi_Model (`ReadOnlyRepo`, catalog, normalizers, graph check) |
| `compiler/` | IR, partition, X2 packing, schedule, INT8 network |
| `emulators/` | `x2_adapter.py` (BMsemi's PE/X2 simulator), `platform_ops.py` (`map_layer`) |
| `chip/` | chip repo reader + profile, chip flow, X2 golden model, bitstream audit, `x2_facts.py`, signed MACC techmap |
| `verification/` | comparators, golden references, workloads, chain runners, debug rules, agent tools |
| `agents/` | A1 mapping, A2 RTL, A3 debug, A4 verification/test agents, LLM client (Groq / Gemini / mock) |
| `orchestration/` | benchmark over all BMsemi workloads (with an OS-independent digest) |
| `examples/chip_designs/` | hand-written Verilog: X2 bridge, recombiner, one- and two-block engines |
| `sim/` | testbenches and simulation-only models (chip-pin X2 adapter, IOBUF with the chip's polarity) |
| `specifications/` | hardware_config.json, chip_profile.json, external_repos.json (pinned clones) |
| `tests/` | about 920 tests; `tests/fixtures/` holds real logs, bitstreams and IRs |
| `docs/` | one document per step (`FLOW_STEP*`, `A1_*`…`A4_*`, `X2_*`, `DEPLOY_CHIP.md`) plus the exact diffs |
| `tools/` | `inventory.py`, `cleanup_move.py`, `prepush_check.py`, fixture helpers |
| not in git | `BMsemi_Model/`, `sky130-reram-fpga/` (read-only clones), `chip_build/`, `deploy/`, `reports/` (generated, except `reports/rtl/`) |

## The X2 models

| model | where | role |
|---|---|---|
| BM Labs behavioral X2 (Verilog) | `sky130-reram-fpga/ip/sky130_bmlabs_reram/verilog/behavioral/sky130_bmlabs_reram__cim32x32.sv` | ground truth: commands SET/RESET/READ/COMPUTE, 1-bit cells, realistic delays |
| Python golden | `chip/x2_golden.py` | same rules without timing; must equal the Verilog model on the same commands (`chip/x2_check.py`, `run_x2_check.py --sim`) |
| chip-pin adapter | `sim/RERAM_CIM_32x32_sim.v` | wires the fabric primitive's pins to the Verilog model the way the chip does |
| 4-bit model | `compiler/x2_pack.py` `MultiLevelX2` | 4-bit cells; the cell rule sum(pwm·level) is ASSUMED (`x2_facts.cell_compute_4bit` is missing) |
| BMsemi PE/X2 simulator | inside BMsemi_Model, via `emulators/x2_adapter.py` | ideal arithmetic; used by the early workload and benchmark stages |
| cost model | `compiler/x2_schedule.py` | cycles fitted to the Verilog-model simulations (365 / 511 / 3); A1 schedules with it |

## Setup

Two terminals work on the same folder (`E:\vm\KLOK\bm_verif` = `/mnt/e/vm/KLOK/bm_verif`):

* **VS Code PowerShell (Windows)**: Python work, agents, tests. Tool tests are skipped here.
  ```powershell
  python -m venv venv; .\venv\Scripts\Activate.ps1
  pip install -r requirements.txt -r requirements-agent.txt
  ```
* **Ubuntu (WSL)**: everything that needs the chip tools: OSS CAD Suite with Yosys ≥ 0.70, nextpnr and Icarus on PATH,
  and the `fasm` Python package for bitstream generation. `chip/chip_flow.py` checks the Yosys version and names a
  missing tool or module.
* Clone the two read-only repos next to the code, at the commits in `specifications/external_repos.json`. **Never edit
  them**: every run checks that they are unchanged.
* A separate X2 model repo: set `BM_X2_REPO` to that repo or to its `.sv` file. Every simulation, the profile and the
  hardware config then use it instead of the copy inside the chip repo. Re-check with `python3 bm.py x2-recheck`.
* LLM keys come **only from environment variables** (see `agents/llm_client.py`; test with `python check_llm.py`). Every
  agent also runs with the mock LLM or `--llm none`.

## The whole pipeline in one command

```bash
python3 bm.py pipeline            # Ubuntu terminal, about 10 minutes  (--rebuild: also rebuild the engine; --with-tests)
```

It runs these stages in order: environment, eFPGA operators on the chip, formal proofs and lint, engine bitstream,
X2 golden vs model, engine simulation, deployment of the 3 models, bundle replay, chip status. Each stage prints PASS
or FAIL and where its results are saved. The summary is saved in **`reports/pipeline/REPORT.md`**: one row per stage
(PASS/FAIL, the key numbers, its log), the saved files per stage, and what is not proven.

## Commands: one entry point, `bm.py`

`python bm.py` lists every command. (In Ubuntu use `python3`.) Arguments pass straight through, and
`python bm.py <command> --help` shows the script's own help. Commands marked [Ubuntu] need the chip tools. On Windows
they are refused with the right Ubuntu command instead of failing halfway.

| group | commands (`python bm.py …`) |
|---|---|
| checks | `tests`, `env`, `llm`, `status` (A4's view: builds, timing, audits, chain, missing X2 facts) |
| early pipeline | `verify`, `workloads`, `model-workloads`, `benchmark` |
| agents | `a1`, `a2`, `a3 --target chip`, `a4`, `eval-a3`, `eval-a4` |
| chip description | `chip-profile`, `hw-config`, `costs` (measured eFPGA cost per block), `wsl-probe` |
| chip build and audit | `build --example x2_engine2_4bit` [Ubuntu], `audit`, `chain --op all --target chip --runner native` [Ubuntu] |
| formal and lint | `formal` [Ubuntu]: operator proofs over all inputs (RTL and chip netlist), Verilator lint, bounded engine check (`--part`, `--depth`) |
| X2 on the chip | `x2-check`, `x2-pack`, `x2-recombine`, `x2-bridge`, `x2-engine`, `x2-engine2`, `layers` (all with `--sim`) [Ubuntu] |
| deployment | `deploy --workload mobilenetv2\|deepgate\|mnist` (`--sim` in Ubuntu, `--weights file.npz` for real MobileNetV2 weights), `replay deploy/<model>` (re-runs a bundle from its files alone) |
| repository | `inventory`, `cleanup`, `prepush` |

**Workflows** run several commands and print one PASS/FAIL summary:

| workflow | what it runs |
|---|---|
| `python bm.py all` | **Windows:** tests, deploy of all 3 models, status. **Ubuntu:** X2 golden vs model, engine, chip chain, deploy with `--sim`, costs, status |
| `python3 bm.py x2-recheck` | after a new X2 model (`BM_X2_REPO`): profile, hardware config, golden vs the new model, packing, engine, deploys |

The individual `run_*.py` scripts are unchanged; `bm.py` only names and groups them.

## Results (Windows + Ubuntu, 2026-11-16)

| what | result |
|---|---|
| Deployment, 4-bit X2 path | MobileNetV2 (real graph, seeded weights) 21/21 steps exact; DeepGate (real INT8) 10/10, equal to the repo's own chain; MNIST equal to the repo's edge map |
| Deployment, 1-bit engine RTL | MobileNetV2 17/17 layers exact, 4,295,897 cycles measured vs 4,296,056 predicted; MNIST full 28×28 exact |
| A1 schedule | MobileNetV2 254 ms per inference, 46 % below one block; DeepGate 13.8 s, dominated by reprogramming (its weights need far more than 2 blocks) |
| Engine bitstream `x2_engine2_4bit` | 35/36 pins, both X2, 1,792 logic cells, 14.45 MHz (nextpnr), V0 audit 47,236/47,236 bits |
| A2 operators on the chip | relu8, add8_sat, requant_i32_i8: netlist simulation = NumPy, timing met at 14 MHz |
| Bugs found in the chip flow | signed MACC techmap wrong for signed products wider than 8 bits (fixed by an override, repo untouched); IOBUF simulation model has the enable inverted |
| Formal (`bm.py formal`) | relu8, add8_sat and requant_i32_i8 equal independent specs for **every input**, and their chip-synthesized netlists equal their RTL for every input. With the repo's own multiplier map, the prover finds the signed bug (acc = -2^31 gives +127 instead of -128). Verilator: no errors and no serious warnings. Engine: P1-P3 hold for every X2 behaviour that keeps the handshake rules, **bounded to 10 clocks**, and a copy without forwarding fails P3 |
| A3 / A4 benchmarks | debug system 12/12 on injected bugs (validator rejected every LLM overclaim); generated tests caught 13/13 planted bugs (fixed suite 10/13) |

## Not done yet

The evidence comes from Python models, synthesized netlists and RTL simulation with the X2 developer's behavioural
model. It is **not** from the configured fabric and **not** from silicon.

Known limits:
- 1-bit RTL simulation skips layers above `--max-tiles` (8 of DeepGate's 10).
- `run_x2_layer_e2e.py` checks 2 output positions per layer by default; `deploy_chip.py --sim` checks every position.
- A2 builds a fixed operator set. `python bm.py costs` shows that engine + relu + add + requant would fit in one design
  (about 1,976 / 2,160 cells, 4 / 18 MACC): an ESTIMATE until that design is built.


1. **Configured-fabric simulation on the chip (S11/V1/V2):** load the exact `.bit` through the chip's configuration
   interface into the fabric RTL with the behavioral X2. This is done only on the FABulous demo fabric so far.
2. **Stress tests (V4):** stalls, late results, reset mid-run, reprogramming between models.
3. **Facts only the team can give** (`chip/x2_facts.py`, filled in via `specifications/x2_overrides.json`):
   - the 4-bit cell rule;
   - a 4-bit programming command;
   - silicon variation;
   - trained MobileNetV2 weights.
4. **Silicon** loading and testing (S12), the GUI path (S01), and Qwen for A2 if required.

## Publishing (maintainers)

The public repository is an export of the team's private working repository. `python tools/export_public.py <folder>`
copies the tracked files without the private ones (the team's model graphs), checks the copy for private data and keys,
and runs the whole test suite in the copy in public mode. Only then does it print the push commands.

## License

No license has been chosen yet. Until one is added, the code is visible for review and feedback, but is not licensed for
reuse. The chip repository and the tools keep their own licenses.
