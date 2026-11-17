# ai-native-compiler-x2

[![ci](https://github.com/Pankajadas17/ai-native-compiler-x2/actions/workflows/ci.yml/badge.svg)](https://github.com/Pankajadas17/ai-native-compiler-x2/actions/workflows/ci.yml)

AI agents and a verification framework that compile INT8 neural networks onto the **sky130 ReRAM FPGA**
([mole99/sky130-reram-fpga](https://github.com/mole99/sky130-reram-fpga), fabric `classic_fabric_reram`). One command
takes a model through partitioning, scheduling, Verilog, bitstream and simulation. Every stage is checked against an
independent reference, and that check, not an agent, decides PASS.

The chip's resources and what the deployment engine uses of them:

|    Primitive    | Available | Used by the engine |                       Role here                        |
|:---------------:|:---------:|:------------------:|:------------------------------------------------------:|
| RERAM_CIM_32x32 | 2         | 2                  | the X2 crossbars: matrix-vector products, 4-bit cells   |
| FABULOUS_LC     | 2160      | 1792               | command sequencer, read pipeline, accumulator banks    |
| IOBUF           | 36        | 35                 | host interface: clock, reset, command beats, results   |
| MACC_8x8_20     | 18        | 0                  | used by the requantize operator (4)                    |
| GBUF            | 4         | 1                  | clock                                                  |

The engine bitstream is the same for every model. A model is deployed by programming the X2 cells and running a host program.

# The agents

| Agent | Job | Decided by |
|:-----:|-----|------------|
| A1 | partition the model (matrix ops → X2, the rest → eFPGA/host) and schedule tiles over both X2 blocks | a validator and a cycle model fitted to simulation |
| A2 | write eFPGA operators in Verilog (`relu8`, `add8_sat`, `requant_i32_i8`) | Yosys evaluation vs NumPy, chip build, formal proof |
| A3 | name the cause of a failed run from its logs | rules first; LLM answers are checked against the evidence |
| A4 | generate tests and report the chip status | the Python oracle |

Agents can use Groq or Gemini, or run with a mock or no LLM; the results do not depend on which.

## The pipeline

| # | Stage | What PASS means |
|:-:|-------|-----------------|
| 1 | Environment | tool versions recorded, chip repo found and fingerprinted |
| 2 | eFPGA operators | each A2 operator through the chip flow: bitstream, V0 audit (every required bit decoded), netlist simulation = NumPy, timing ≥ 14 MHz |
| 3 | Formal | operators equal independent specs for **all** inputs; chip-synthesized netlists equal their RTL; Verilator lint clean; engine handshake bounded-checked |
| 4 | Engine bitstream | `x2_engine2_4bit`: built (or reused if unchanged), audited, meets 14 MHz |
| 5 | X2 model | the Python X2 model equals the X2 developer's Verilog model on the same commands |
| 6 | Engine simulation | the two-block engine is exact; measured cycles match A1's cost model |
| 7 | Deploy | MobileNetV2-style CNN, MLP chain, edge filter: every tensor bit-exact, layers run on the engine RTL |
| 8 | Replay | each deployment bundle reproduces its results from its own files |
| 9 | Status | chip repo unchanged; open X2 facts listed |

## Requirements

> [!NOTE]
> Linux, or WSL on Windows. Clone the chip repository **next to this one's code**, with submodules, at the pinned commit
> in `specifications/external_repos.json`. It is only read, never modified:
>
> ```console
> git clone https://github.com/Pankajadas17/ai-native-compiler-x2 && cd ai-native-compiler-x2
> git clone https://github.com/mole99/sky130-reram-fpga
> git -C sky130-reram-fpga checkout e0790eb11a1e95914f2beb8f175396a0924144ed
> git -C sky130-reram-fpga submodule update --init --recursive
> ```

- Python 3.11+: `python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt`
- [OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build) with Yosys 0.70 or newer (nextpnr, Icarus, SymbiYosys, Verilator included). Put it on `PATH`, or unpack it to `~/tools/chip/oss-cad-suite` (or set `CHIP_CAD`).
- A Python environment for the chip's bitstream generator: `python3 -m venv ~/chipenv && ~/chipenv/bin/pip install fasm numpy`

## Run the whole pipeline

```console
python3 bm.py pipeline
```

```text
[4/10] Deployment engine bitstream ... PASS
      1792/2160 cells, 35/36 pins, both X2, V0 audit 47236/47236, 14.45 MHz >= 14
        saved here: chip_build/x2_engine2_4bit/x2_engine2_4bit.bit
        ...
PIPELINE PASS: 9/9 stages in 4.7 min
summary saved here: reports/pipeline/REPORT.md
```

Each stage prints PASS or FAIL and where its results are saved. `reports/pipeline/REPORT.md` collects all of it, plus
what the run does not prove. Add `--rebuild` to rebuild the engine bitstream, `--with-tests` to also run the test suite,
and `--only`/`--skip` to pick stages.

## Run single steps

`python3 bm.py` lists every command. Arguments pass through to the underlying script.

#### Build the engine bitstream

```console
python3 bm.py build --example x2_engine2_4bit
```

The bitstream, pin map (`design.pcf`), audit and timing log go to `chip_build/x2_engine2_4bit/`.

#### Run the A2 operators on the chip

```console
python3 bm.py chain --op all --target chip --runner native
```

#### Formal proofs and lint

```console
python3 bm.py formal                      # or --part spec,netlist,lint,engine and --depth N
```

#### Deploy a model and replay its bundle

```console
python3 bm.py deploy --workload mobilenetv2 --ir tests/fixtures/ir_public/invres_net.json --sim
python3 bm.py replay deploy/mobilenetv2
```

The bundle in `deploy/<model>/` holds `plan.json`, `host_program.json`, `x2_program.npz` (cell levels per tile),
`expected.npz` and `report.json`. `--weights file.npz` loads real INT8 weights.

#### Tests

```console
python3 -m pytest -q
```

## Results

| | |
|---|---|
| Pipeline | 9/9 stages in under 5 minutes |
| Formal | 21/21. The proof finds a signed-multiply bug in the chip's stock `MACC_8x8_20` multiplier map (acc = −2³¹ gives +127, not −128); a corrected map in `chip/techmap/` is used without editing the chip repo |
| Engine | 1792/2160 cells, 35/36 pins, 14.45 MHz; two X2 blocks give 43–54 % lower latency than one; cycle model within 0.04 % |
| Deployment | MobileNetV2: 17/17 layers exact on the engine RTL, 4.30 M cycles (model error 0.004 %) |

**Not proven yet:**
- the configured fabric (the chip's tile RTL loaded with the bitstream) in simulation;
- timing-annotated simulation;
- silicon;
- the 4-bit cell rule, which is assumed (the chip's X2 model is 1-bit);
- the engine beyond 10 clocks in the bounded check.

## Repository layout

```
agents/         A1-A4 and the LLM client          compiler/   IR, partition, X2 packing, schedule, INT8 network
chip/           chip flow, X2 model, audits        verification/  checkers, chain runner, formal, debug rules
examples/       engine and bridge Verilog          formal/     X2 stand-in and engine properties
sim/            testbenches                        tests/      ~900 tests and fixtures
bm.py           one entry point                    run_pipeline.py  the whole pipeline
```

## Public mode

The development team also uses a private model repository with its own trained models. Without it, everything above
still runs. Deployment uses the public stand-in graphs in `tests/fixtures/ir_public/`, and the tests that need the
private repository are not collected; the test header and the pipeline report say "PUBLIC mode". The CI badge runs
exactly this.

## Feedback

Issues are welcome. Report anything unclear, unconvincing or wrong in the verification approach, the agents' role,
the evidence, or the code.

## License

No license has been chosen yet. The code is visible for review and feedback, but is not yet licensed for reuse.
