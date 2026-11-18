# Verification

Formal proofs and lint, the operator chain on the chip, and the BMsemi benchmark stage table.

- [Formal proofs and lint: `x2c formal`](#formal-proofs-and-lint-x2c-formal)
- [Operators on the chip: `x2c chain --target chip`](#operators-on-the-chip-x2c-chain---target-chip)
- [BMsemi benchmark stages: `x2c bmsemi benchmark`](#bmsemi-benchmark-stages-x2c-bmsemi-benchmark)

## Formal proofs and lint: `x2c formal`

Simulation checks the vectors it is given. These checks are proofs for **all** inputs, or for all X2 behaviours within
a bound. All the tools are already in the chip OSS CAD Suite: Yosys SAT, SymbiYosys with bitwuzla, and Verilator.

| part | what is proved | result (Yosys 0.70+368) |
|---|---|---|
| spec | each A2 operator (`hardware/operators/*.v`) equals an **independently written** spec (different constructions: `x < 0` vs `x[7]`; truncating division with a floor fix vs `>>>`) for every input. A broken copy of each must give a counterexample | relu8, add8_sat, requant_i32_i8 proved in < 0.2 s; all 3 broken copies caught, with the failing input |
| netlist | each operator synthesized by **the chip's own synthesis** (generated Makefile → the repo's `makefile.include`, techmaps, our signed multiplier map) equals its RTL for every input. Pad cells are removed and every LUT, carry and MACC uses the **repo's primitive models** | all 3 proved |
| netlist (repo map) | the same, with the repo's original `MACC_8x8_20_map.v` | **counterexample: acc = -2147483648 gives +127 instead of -128**. This is the signed-multiply bug that the corrected map fixes, found by a proof rather than by chance (see [chip.md](chip.md#signed-multiplies-on-the-hard-multipliers)) |
| lint | Verilator `-Wall` on the operators, recombiner, bridge, both engines and the vendor config. Fails on errors or serious classes (UNDRIVEN, MULTIDRIVEN, LATCH, BLKSEQ, CASEINCOMPLETE, …) | 8 designs clean. Remaining warnings are WIDTH* (32-bit `localparam` arithmetic assigned to narrow wires: intended) and one unused `x2_res_last` in the one-block engine |
| engine | SymbiYosys BMC of `x2_engine2` (CELL_BITS 1 and 4) against `hardware/formal/RERAM_CIM_32x32_formal.v`, an X2 whose outputs are **free** apart from the handshake rules the BM model keeps | P1-P3 hold for 10 clocks after reset (33 s / 63 s). A copy **without the forwarding path fails P3** in 3 s |

Engine properties (`hardware/formal/x2_engine2_props.vh`):
- **P1:** the arbiter never grants both blocks in one clock.
- **P2:** a pending command stays unchanged until its block takes it.
- **P3:** whenever both blocks are idle, the solver's chosen accumulator equals a shadow that adds every granted COMPUTE result exactly once. That rules out lost, doubled or misrouted results, and covers the 3-stage pipeline and its forwarding.

The host contract is assumed: start in reset, and clear a bank only while it is idle.

**Bounded** means the solver checked every behaviour up to 10 clocks after reset. That is enough for a run of results
through the 3-stage pipeline. It is not an unbounded proof. Use `--depth` for more, but run time grows quickly: depth 13
takes several minutes.

## Operators on the chip: `x2c chain --target chip`

The operator chain runs six stages and gives one verdict per operator. `--target chip` runs them on the chip in
sky130-reram-fpga; the default, `--target demo`, runs them on the earlier FABulous demo fabric.

| stage | demo fabric | chip (`src/x2c/verification/chip_chain.py`) |
|---|---|---|
| rtl | A2 Verilog, Yosys vs NumPy | the same function |
| prep | io_in/io_out/io_oeb wrapper, 28 pins | wrapper with named ports on the chip's 36 pins, checked in Yosys vs NumPy |
| env | FABulous + demo project | OSS CAD Suite (`~/tools/chip/oss-cad-suite` or `CHIP_CAD`), the project's Python with fasm, chip repo, chip profile |
| flow | FABulous `run_fab_flow.sh` | `x2c build` (repo techmaps, nextpnr, 35,304-byte bitstream, V0 audit) |
| sim | FABulous fabric simulation | the **synthesized netlist** in Icarus with the repo's primitive models, driven through the pins |
| check | demo flow check | flow report + nextpnr timing (last line per clock) + netlist simulation |

Jobs go to `reports/chip_jobs/<op>/`, builds to `chip_build/chain_<op>/`, and the report to
`reports/chip_chain_report.json`. The stage scripts set up their own `PATH`, so the terminal's setup does not matter;
`CHIP_CAD` and `CHIP_PY` override the tool locations. A run that does not pass gets a diagnosis from A3's rules
([agents.md](agents.md#a3-chip-failures)).

### Which operators fit 36 pins (one is clk)
- `relu8`: 17 pins, direct.
- `add8_sat`: 25 pins, direct.
- `requant_i32_i8`: 26 pins, two-step load. `sel=0` loads the low 16 bits and `sel=1` the high 16; this protocol is
  ours (ASSUMED), as on the demo.
- `x2_acc_step`: 84 pins, does not fit. On the chip it runs inside the X2 engine ([x2.md](x2.md)).

The direct wrapper registers the operands as well as the result, so the core sits between two registers. nextpnr only
times register-to-register paths (it printed "No Fmax available; no interior timing paths" without this).

### Results

With the corrected multiplier map (used by every build), all three operators PASS:

| op | prep | flow | netlist sim | timing (nextpnr model) | verdict |
|---|---|---|---|---|---|
| relu8 | 256 vectors | 15 LC, V0 345/345 | 256/256 | 151.5 MHz | PASS |
| add8_sat | 500 vectors | 35 LC, V0 744/744 | 500/500 | 42.2 MHz | PASS |
| requant_i32_i8 | 485 vectors | 134 LC, 4 MACC | 485/485 | no reg-to-reg path | PASS |

With the repo's original multiplier map, requant got **219 of 485 vectors wrong**, every one with a negative input. That
run found the first finding below.

### Finding 1: signed multiplies on the hard multiplier are wrong (real chip issue)
- The repo's synthesis maps multiplies onto `MACC_8x8_20` blocks (`-multiplier-map MACC_8x8_20_map.v 8 8 2 2 12`).
- The map configures every block unsigned (`SIGN_EXT=0`, no `A_SIGNED`/`B_SIGNED` branch).
- A signed multiply wider than 8 bits is split over several blocks, and the result is wrong whenever an operand is
  negative.

The same check with unsigned operands, or with the multiplier map removed, is correct:

| test (repo synthesis options, netlist simulated) | wrong |
|---|---|
| requant (signed 32 x 19) with the multiplier map | 219/485, all negative inputs |
| requant without the multiplier map (LUTs only) | 0/485 |
| 16x8 unsigned | 0 |
| 16x8 signed | 129,326/173,056, every one with a negative operand |
| 8x8 signed and unsigned (one block) | 0/65,536 |

Any operator with a signed multiply wider than 8 bits would get a wrong bitstream that passes the flow and the V0
audit. The chain's `sim` stage catches it and names the cause (`explain_mismatch`, a rule, not an LLM). The corrected
map is described in [chip.md](chip.md#signed-multiplies-on-the-hard-multipliers).

### Finding 2: the repo's IOBUF simulation model has the enable inverted (simulation only)
`primitives/IOBUF/yosys/primitives/IOBUF.v` drives the pad when `EN=0`. The techmap ties `EN=1` on outputs, and the chip
passes EN out as `fabric_io_oe_o` with `gpio_oeb = ~oe` (`sky130-reram-fpga/src/double_wide_openframe_project_wrapper.v` l.122). So on
the chip, `EN=1` drives. With the repo model every output of a synthesized design floats.
`hardware/sim/IOBUF_chip_sim.v` has the chip's polarity; the repo is not edited.

### Finding 3: `prims.v` LUTFF_SSE uses an undeclared `R` (l.173)
Icarus treats every unused module as a top and stops on it. The chain selects the testbench with `-s`, so that cell
is never elaborated.

Also: the repo Makefile runs nextpnr with `--timing-allow-fail`, so a design that misses 12 MHz still gets a bitstream
(the 4-bit engine did). The chain's check reports timing itself: if timing is not met, the verdict is UNVERIFIED,
never PASS.

### The Makefile's `python3`
The repo Makefile runs `python3 bit_gen.py` with the first `python3` on `PATH`, and `bit_gen.py` needs `fasm`. OSS CAD
Suite ships its own `python3` without `fasm`. So the stage scripts put the project's Python first
(`export PATH="$(dirname "$PY"):$CAD/bin:$PATH"`), and the preflight checks the Makefile's `python3`
(`FOUND make python3` or `MISSING fasm or numpy in the Makefile's python3 (...)`), so `env` blocks before `flow` if it
is wrong. A flow that still fails this way is named `python_module_missing` (a real make log is in
`tests/fixtures/chip_flow/python_missing.bitstream.log`).

### Evidence level
Covered:
- Yosys RTL evaluation.
- The real chip flow: repo techmaps, nextpnr, bitstream, V0 audit.
- Simulation of the synthesized netlist against NumPy.

Not covered:
- The configured fabric (tile RTL loaded with the bitstream).
- Timing-annotated simulation.
- Silicon.

## BMsemi benchmark stages: `x2c bmsemi benchmark`

`src/x2c/bmsemi/benchmark.py` STAGE_STATUS says, for each stage of the project, whether it exists. The benchmark run
itself executes S02–S04; the other stages run from their own commands. Each row has:
- `where`: the files that implement it; `tests/test_benchmark.py` checks they exist.
- `missing`: what a partial stage still lacks; required for every stage that isn't `implemented`.

The stage table is not part of the result digest.

| stage | status | still missing |
|---|---|---|
| S01 model input / selection | implemented | |
| S02 golden math / PE validation | implemented | |
| S03 typed IR / graph validation | partial | module kinds are not recorded in the artifact |
| S04 mapping and X2 package | partial | the partitioner does not estimate fabric budgets per IR node (the chip flow measures built operators) |
| S05 Verilog generation | partial | not generated from a workload's IR; the operators are a fixed list |
| S06 lint / assertions / equivalence | partial | the engine properties are bounded (10 clocks); the other X2 designs have no assertions |
| S07–S09 Yosys / nextpnr / bitstream | partial | built per operator and per X2 design, not yet for a whole workload |
| S10 V0 audit | implemented | |
| S11 configured fabric + X2 | partial | the configured fabric (tile RTL + bitstream) is not simulated; no timing-annotated simulation |
| S12 silicon | not implemented | hardware |

