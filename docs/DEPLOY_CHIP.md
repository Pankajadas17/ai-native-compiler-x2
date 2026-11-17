# Deploying a model on the chip: `deploy_chip.py`

One command takes a model from its IR to a verified deployment bundle for the BM Labs chip
(sky130-reram-fpga: 2 × X2 32×32 crossbars + FABulous eFPGA). It runs at 4-bit cells and a 14 MHz clock.

```
python deploy_chip.py --workload mobilenetv2       # VS Code terminal
python deploy_chip.py --workload deepgate          # real INT8 weights + requant from BMsemi_Model
python deploy_chip.py --workload mnist
python3 deploy_chip.py --workload mobilenetv2 --sim   # Ubuntu terminal (Icarus): every layer also on the 1-bit engine RTL
```

## What the agents do

| step | who | what |
|---|---|---|
| 1 | inputs | IR from BMsemi (compiler-V2 graph / chain workload), or `--ir file.json` |
| 2 | A1 | partition (matrix ops → X2) + X2 tile schedule over both blocks; LLM proposals only if validated (`--llm none` = baseline + code schedule) |
| 3 | compiler/int8_net.py | INT8 network. Module kinds are INFERRED from structure: BN after a matrix op, ReLU6 after BN, which gives 16 + 11 on the real graph. BN is folded into the bias, ReLU6 becomes a clamp to [0,127], residual add saturates, global average pool |
| 4 | executors | every tensor is compared with an independent nested-loop golden model. 4-bit X2 Python model; `--sim`: 1-bit two-block engine in Icarus |
| 5 | A4 tools | chip status, engine bitstream, missing X2 facts |
| 6 | bundle | `deploy/<workload>/` |

How a model reaches the X2: the X2 takes only uint8 inputs. A signed tensor goes in as x+128, and the host subtracts
(128 + model zero point)·ΣW. This is exact integer arithmetic, and padding uses the same zero point.

## The bundle

| file | contents |
|---|---|
| `plan.json` | A1's partition and schedule (resident layers, policy, latency) |
| `host_program.json` | the ordered steps: X2 layers (input tensor, offsets, epilogue bias/shift/clamp or the repo's requant), adds, pool |
| `x2_program.npz` | per layer and tile: the 32×32 cell levels (0..15) and the placement |
| `expected.npz` | every intermediate tensor from the golden model (input → output) |
| `engine/` | `x2_engine2_4bit.bit` + `design.pcf` (pin map) + audit. The **same bitstream for every model**; a model is deployed by programming the cells and running the host program |
| `report.json` | checks, A1 schedule, 1-bit cycles, missing facts, evidence level |

## Results (sandbox; real BMsemi data runs only on your machine)

| workload | X2 layers | 4-bit exact | 1-bit engine RTL | A1 schedule |
|---|---|---|---|---|
| MobileNetV2 (real graph, seeded weights) | 17 (21 steps) | all 21 | all 17 layers exact, 4,295,897 cycles measured vs 4,296,056 predicted | 254 ms, 46 % below single-block |
| MNIST edge filter (stand-in data) | 1 | yes | full 28×28 exact, 461,564 cycles | 32.8 ms |
| DeepGate (stand-in data) | 10 | all | 2 small layers simulated; 8 over `--max-tiles` (640 1-bit tiles each) stay on the 4-bit path | 13.8 s: weights cannot stay resident, so reprogramming dominates |

With BMsemi_Model, the DeepGate check also requires our golden chain to equal the repo's own `fwd` chain, and MNIST requires
it to equal the repo's `edge_filter` map.

## Still assumed / missing (reported in every report.json)

- `cell_compute_4bit`: the 4-bit cell rule sum(pwm·level) is ASSUMED. The chip's X2 model is 1-bit only. Fill it in via `specifications/x2_overrides.json`.
- 4-bit cell programming commands (WRITE_LEVEL) are not in the chip's command set. `x2_program.npz` therefore stores levels, not commands.
- `silicon_variation`: unknown. Every check is exact digital simulation.
- MobileNetV2 weights are random seeded INT8 because the compiler artifact has shapes only. To use real weights, pass a model with weights.
- Evidence: Python X2 model + RTL simulation with the BM behavioral X2. NOT the configured fabric, NOT silicon.
