# Item 4: formal checks and lint (`run_formal.py`, `python3 bm.py formal`, Ubuntu)

Before this, every check simulated chosen vectors. These checks are proofs for **all** inputs, or for all X2 behaviours
within a bound. All the tools are already in the chip OSS CAD Suite: Yosys SAT, SymbiYosys with bitwuzla, and Verilator.

| part | what is proved | result in the sandbox (Yosys 0.70+368) |
|---|---|---|
| spec | each A2 operator (`reports/rtl/*.v`) equals an **independently written** spec (different constructions: `x < 0` vs `x[7]`; truncating division with a floor fix vs `>>>`) for every input. A broken copy of each must give a counterexample | relu8, add8_sat, requant_i32_i8 proved in < 0.2 s; all 3 broken copies caught, with the failing input |
| netlist | each operator synthesized by **the chip's own synthesis** (generated Makefile → the repo's `makefile.include`, techmaps, our signed multiplier map) equals its RTL for every input. Pad cells are removed and every LUT, carry and MACC uses the **repo's primitive models** | all 3 proved |
| netlist (repo map) | the same, with the repo's original `MACC_8x8_20_map.v` | **counterexample: acc = -2147483648 gives +127 instead of -128**. The signed-multiply bug the override fixes, now found by a proof instead of by chance |
| lint | Verilator `-Wall` on the operators, recombiner, bridge, both engines and the vendor config. Fails on errors or serious classes (UNDRIVEN, MULTIDRIVEN, LATCH, BLKSEQ, CASEINCOMPLETE, …) | 8 designs clean. Remaining warnings are WIDTH* (32-bit `localparam` arithmetic assigned to narrow wires: intended) and one unused `x2_res_last` in the one-block engine |
| engine | SymbiYosys BMC of `x2_engine2` (CELL_BITS 1 and 4) against `formal/RERAM_CIM_32x32_formal.v`, an X2 whose outputs are **free** apart from the handshake rules the BM model keeps | P1-P3 hold for 10 clocks after reset (33 s / 63 s). A copy **without the forwarding path fails P3** in 3 s |

Engine properties (`formal/x2_engine2_props.vh`):
- **P1:** the arbiter never grants both blocks in one clock.
- **P2:** a pending command stays unchanged until its block takes it.
- **P3:** whenever both blocks are idle, the solver's chosen accumulator equals a shadow that adds every granted COMPUTE result exactly once. That rules out lost, doubled or misrouted results, and covers the 3-stage pipeline and its forwarding.

The host contract is assumed: start in reset, and clear a bank only while it is idle.

**Bounded** means the solver checked every behaviour up to 10 clocks after reset. That is enough for a run of results
through the 3-stage pipeline. It is not an unbounded proof. Use `--depth` for more, but run time grows quickly: depth 13
takes several minutes.

## Changes (from → to)

| file | change |
|---|---|
| `verification/formal.py` (new) | specs, broken copies, chip-synthesis of a bare operator (reuses `chip_flow.MAKEFILE`), `strip_pads()`, SAT miter, Verilator lint with waivers for the repo's primitive declaration, SymbiYosys runner |
| `run_formal.py` (new) | runs the parts, prints expected/got per check, writes `reports/formal/report.json`, checks the chip repo is unchanged |
| `formal/RERAM_CIM_32x32_formal.v` (new) | unconstrained X2 for formal only (same ports and parameters as the chip primitive) |
| `formal/x2_engine2_props.vh`, `formal/x2_engine2.sby` (new) | properties; manual `sby` entry |
| `examples/chip_designs/x2_engine2.v` | `endmodule` → `` `ifdef FORMAL `include "x2_engine2_props.vh" `endif`` then `endmodule`. The chip flow and Icarus never define FORMAL, so the built engine is unchanged (all engine simulation tests pass) |
| `bm.py` | `formal` command; also added to the Ubuntu `all` workflow |
| `.gitignore` | ignores manual SymbiYosys work folders in `formal/` |
| `tests/test_formal.py` (new) | parsing, pad stripping, hooks present everywhere; live proofs and the engine broken copy in Ubuntu |

Evidence level: proofs on RTL and on the chip-synthesized netlist with the repo's primitive models, plus a bounded
check of the engine. Not the configured fabric, not silicon.
