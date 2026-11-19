# ai-native-compiler-x2

[![ci](https://github.com/Pankajadas17/ai-native-compiler-x2/actions/workflows/ci.yml/badge.svg)](https://github.com/Pankajadas17/ai-native-compiler-x2/actions/workflows/ci.yml)

AI agents and a verification framework that compile INT8 neural networks onto the **sky130 ReRAM FPGA**
([mole99/sky130-reram-fpga](https://github.com/mole99/sky130-reram-fpga), fabric `classic_fabric_reram`). One command
takes a model through partitioning, scheduling, Verilog, bitstream and simulation. Every stage is checked against an
independent reference, and that check, not an agent, decides PASS.

Developed during an internship at BM Labs.

The chip's resources and what the deployment engine uses of them:

|    Primitive    | Available | Used by the engine |                       Role here                        |
|:---------------:|:---------:|:------------------:|:------------------------------------------------------:|
| RERAM_CIM_32x32 | 2         | 2                  | the X2 crossbars: matrix-vector products, 4-bit cells   |
| FABULOUS_LC     | 2160      | 1792               | command sequencer, read pipeline, accumulator banks    |
| IOBUF           | 36        | 35                 | host interface: clock, reset, command beats, results   |
| MACC_8x8_20     | 18        | 0                  | used by the requantize operator (4)                    |
| GBUF            | 4         | 1                  | clock                                                  |

The engine bitstream is the same for every model. A model is deployed by programming the X2 cells and running a host
program.

## The agents

| Agent | Job | Decided by |
|:-----:|-----|------------|
| A1 | partition the model (matrix ops → X2, the rest → fabric/host) and schedule tiles over both X2 blocks | a validator and a cycle model fitted to simulation |
| A2 | write fabric operators in Verilog (`relu8`, `add8_sat`, `requant_i32_i8`) | Yosys evaluation vs NumPy, chip build, formal proof |
| A3 | name the cause of a failed run from its logs | rules first; LLM answers are checked against the evidence |
| A4 | generate tests and report the chip status | the Python oracle |

Agents run on Google's agent harness with Gemini (Groq as the last fallback), or with a fixed no-model stand-in;
an end-of-turn check makes each agent continue until a checker or validator accepts its work, and code decides the result.

## The pipeline

| # | Stage | What PASS means |
|:-:|-------|-----------------|
| 1 | Environment | tool versions recorded, chip repo found and fingerprinted |
| 2 | Fabric operators | each A2 operator through the chip flow: bitstream, V0 audit (every required bit decoded), netlist simulation = NumPy, timing ≥ 14 MHz |
| 3 | Formal | operators equal independent specs for **all** inputs; chip-synthesized netlists equal their RTL; Verilator lint clean; engine handshake bounded-checked |
| 4 | Engine bitstream | `x2_engine2_4bit`: built (or reused if unchanged), audited, meets 14 MHz |
| 5 | X2 model | the Python X2 model equals the X2 developer's Verilog model on the same commands |
| 6 | Engine simulation | the two-block engine is exact; measured cycles match A1's cost model |
| 7 | Deploy | MobileNetV2-style CNN, MLP chain, edge filter: every tensor bit-exact, layers run on the engine RTL |
| 8 | Replay | each deployment bundle reproduces its results from its own files |
| 9 | Status | chip repo unchanged; open X2 facts listed |

## Setup

> [!NOTE]
> Linux, or WSL on Windows. Clone the chip repository **inside this one**, with submodules, at the pinned commit in
> `specifications/external_repos.json`. It is only read, never modified:
>
> ```console
> git clone https://github.com/Pankajadas17/ai-native-compiler-x2 && cd ai-native-compiler-x2
> git clone https://github.com/mole99/sky130-reram-fpga
> git -C sky130-reram-fpga checkout e0790eb11a1e95914f2beb8f175396a0924144ed
> git -C sky130-reram-fpga submodule update --init --recursive
> ```

- [uv](https://docs.astral.sh/uv/), then `uv sync --all-extras`. This installs everything from `pyproject.toml` and
  `uv.lock`, including `fasm` for the chip's bitstream generator.
- [OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build) with Yosys 0.70 or newer (nextpnr, Icarus,
  SymbiYosys and Verilator included). Put it on `PATH`, unpack it to `~/tools/chip/oss-cad-suite`, or set `CHIP_CAD`.

## Run the whole pipeline

```console
uv run x2c pipeline
```

```text
[4/10] Deployment engine bitstream ... PASS
      1792/2160 cells, 35/36 pins, both X2, V0 audit 47236/47236, 14.45 MHz >= 14
        saved here: chip_build/x2_engine2_4bit/x2_engine2_4bit.bit
        ...
PIPELINE PASS: 9/9 stages in 4.5 min
summary saved here: reports/pipeline/REPORT.md
```

Each stage prints PASS or FAIL and where its results are saved. `reports/pipeline/REPORT.md` collects all of it, plus
what the run does not prove. Add `--rebuild` to rebuild the engine bitstream, `--with-tests` to also run the test suite,
and `--only`/`--skip` to pick stages.

## Run single steps

`uv run x2c` lists every command, grouped. `x2c <command> --help` shows its options. (With `.venv` activated, plain
`x2c` works too.)

#### Build the engine bitstream

```console
uv run x2c build --example x2_engine2_4bit
```

The bitstream, pin map (`design.pcf`), audit and timing log go to `chip_build/x2_engine2_4bit/`.

#### Run the A2 operators on the chip

```console
uv run x2c chain --op all --target chip --runner native
```

#### Formal proofs and lint

```console
uv run x2c formal                # or --part spec,netlist,lint,engine and --depth N
```

#### Deploy a model and replay its bundle

```console
uv run x2c deploy --workload mobilenetv2 --ir tests/fixtures/ir_public/invres_net.json --sim
uv run x2c replay deploy/mobilenetv2
```

The bundle in `deploy/<model>/` holds `plan.json`, `host_program.json`, `x2_program.npz` (cell levels per tile),
`expected.npz` and `report.json`. `--weights file.npz` loads real INT8 weights.

#### Simulate the X2 path

```console
uv run x2c sim x2 --sim          # Python X2 model == the X2 developer's Verilog model
uv run x2c sim engine2 --sim     # the two-block engine, cycles vs A1's cost model
```

#### Tests

```console
uv run pytest -q
```

## Results

| | |
|---|---|
| Pipeline | 9/9 stages in about 5 minutes |
| Formal | 21/21. The proof finds a signed-multiply bug in the chip's stock `MACC_8x8_20` multiplier map (acc = −2³¹ gives +127, not −128); a corrected map in `src/x2c/chip/techmap/` is used without editing the chip repo. The latest upstream fabulous-tiles still has the original map |
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
src/x2c/          the package: cli.py (the x2c command), commands/, agents/, compiler/, chip/, verification/,
                  inputs/, bmsemi/ (bridges to the private model repository)
hardware/         designs/ (X2 recombiner, bridge, engines), operators/ (A2's verified operators),
                  formal/, sim/ (testbenches), test_designs/ (small designs for the flow tests)
specifications/   chip profile, hardware config, fabric costs, pinned repos
tests/            the test suite and its fixtures
docs/             pipeline, deploy, chip, x2, agents, verification, development
tools/            maintainer scripts: pre-push check, public export, fixtures
```

## Public mode

The development team also uses a private model repository with its own trained models. Without it, everything above
still runs. Deployment uses the public stand-in graphs in `tests/fixtures/ir_public/`, and the tests that need the
private repository are not collected; the test header and the pipeline report say "PUBLIC mode". The CI badge runs
exactly this.

## Documentation

| | |
|---|---|
| [docs/pipeline.md](docs/pipeline.md) | every pipeline stage, its output, and replaying a bundle |
| [docs/deploy.md](docs/deploy.md) | from a model to a verified bundle; how layers map onto the chip |
| [docs/chip.md](docs/chip.md) | chip profile, chip flow, bitstream audit, the signed-multiplier fix |
| [docs/x2.md](docs/x2.md) | X2 golden model, packing, recombiner, bridge, engines, real layers, X2 facts and clock |
| [docs/agents.md](docs/agents.md) | A1's schedule, A3's chip diagnosis, A4 on the chip |
| [docs/verification.md](docs/verification.md) | formal proofs and lint, operators on the chip, benchmark stages |
| [docs/development.md](docs/development.md) | setup, layout, public mode, maintainer tools, fabric costs |

## Feedback

Issues are welcome. Report anything unclear, unconvincing or wrong in the verification approach, the agents' role,
the evidence, or the code.

## License

No license has been chosen yet. The code is visible for review and feedback, but is not yet licensed for reuse.
