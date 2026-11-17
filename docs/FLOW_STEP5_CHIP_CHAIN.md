# Flow step 5: the chain on the BM Labs chip (`run_fab_chain.py --target chip`)

The chain you already had (rtl, prep, env, flow, sim, check, one verdict) ran on the FABulous demo fabric. With
`--target chip` the same six stages run on the chip in sky130-reram-fpga. Without `--target` nothing changes.

| stage | demo fabric (unchanged) | chip (new, verification/chip_chain.py) |
|---|---|---|
| rtl | A2 Verilog, Yosys vs NumPy | the same function |
| prep | io_in/io_out/io_oeb wrapper, 28 pins | wrapper with named ports on the chip's 36 pins, checked in Yosys vs NumPy |
| env | FABulous + demo project | `~/tools/chip/oss-cad-suite`, `~/chipenv` (fasm, numpy), chip repo, chip profile |
| flow | FABulous `run_fab_flow.sh` | `run_chip_flow.py` (repo techmaps, nextpnr, 35,304-byte bitstream, V0 audit) |
| sim | FABulous fabric simulation | the **synthesized netlist** in Icarus with the repo's primitive models, driven through the pins |
| check | run_flow_check | flow report + nextpnr timing (last line per clock) + netlist simulation |

Jobs go to `reports/chip_jobs/<op>/`, builds to `chip_build/chain_<op>/`, and the report to
`reports/chip_chain_report.json`. The Ubuntu scripts set their own tools, so your terminal setup doesn't matter.
`CHIP_CAD` and `CHIP_PY` override the tool locations if yours differ.

## Which operators fit 36 pins (one is clk)
- `relu8`: 17 pins, direct.
- `add8_sat`: 25 pins, direct.
- `requant_i32_i8`: 26 pins, two-step load. `sel=0` loads the low 16 bits and `sel=1` the high 16; this protocol is ours (ASSUMED), as on the demo.
- `x2_acc_step`: 84 pins, does not fit. On the chip it runs inside `x2_engine` (flow step 4).

The direct wrapper registers the operands as well as the result, so the core sits between two registers. nextpnr only
times register-to-register paths (it printed "No Fmax available; no interior timing paths" without this).

## Results (sandbox, real chip flow)

| op | prep | flow | netlist sim | timing (nextpnr model) | verdict |
|---|---|---|---|---|---|
| relu8 | 256 vectors | 15 LC, V0 345/345 | 256/256 | 151.5 MHz | PASS |
| add8_sat | 500 vectors | 35 LC, V0 744/744 | 500/500 | 42.2 MHz | PASS |
| requant_i32_i8 | 485 vectors | 134 LC, V0 3597/3597 | **219/485 wrong** | no reg-to-reg path | FAIL |

## Finding 1: signed multiplies on the hard multiplier are wrong (real chip issue)
- The repo's synthesis maps multiplies onto `MACC_8x8_20` blocks (`-multiplier-map MACC_8x8_20_map.v 8 8 2 2 12`).
- The map configures every block unsigned (`SIGN_EXT=0`, no `A_SIGNED`/`B_SIGNED` branch).
- A signed multiply wider than 8 bits is split over several blocks, and the result is wrong whenever an operand is negative.

The same check with unsigned operands, or with the multiplier map removed, is correct:

| test (repo synthesis options, netlist simulated) | wrong |
|---|---|
| requant (signed 32 x 19) with the multiplier map | 219/485, all negative inputs |
| requant without the multiplier map (LUTs only) | 0/485 |
| 16x8 unsigned | 0 |
| 16x8 signed | 129,326/173,056, every one with a negative operand |
| 8x8 signed and unsigned (one block) | 0/65,536 |

Any operator with a signed multiply wider than 8 bits would get a wrong bitstream that passes the flow and the V0 audit.
The chain's `sim` stage catches it and names the cause (`explain_mismatch`, a rule, not an LLM).

## Finding 2: the repo's IOBUF simulation model has the enable inverted (simulation only)
`primitives/IOBUF/yosys/primitives/IOBUF.v` drives the pad when `EN=0`. The techmap ties `EN=1` on outputs, and the chip
passes EN out as `fabric_io_oe_o` with `gpio_oeb = ~oe` (`src/double_wide_openframe_project_wrapper.v` l.122). So on
the chip, `EN=1` drives. With the repo model every output of a synthesized design floats.
`sim/IOBUF_chip_sim.v` has the chip's polarity; the repo is not edited.

## Finding 3: `prims.v` LUTFF_SSE uses an undeclared `R` (l.173)
Icarus treats every unused module as a top and stops on it. The chain selects the testbench with `-s`, so that cell
is never elaborated.

Also: the repo Makefile runs nextpnr with `--timing-allow-fail`, so a design that misses 12 MHz still gets a bitstream
(the 4-bit engine did). The chain's check reports timing itself: if timing is not met, the verdict is UNVERIFIED, never PASS.

## Evidence level
Covered:
- Yosys RTL evaluation.
- The real chip flow: repo techmaps, nextpnr, bitstream, V0 audit.
- Simulation of the synthesized netlist against NumPy.

Not covered:
- The configured fabric (tile RTL loaded with the bitstream; S11).
- Timing-annotated simulation.
- Silicon.

Not yet built for chip runs: the A3 / rule diagnosis (`diagnose`) only knows the demo chain's causes, so it is
skipped for `--target chip`. The multiplier rule above is the first chip rule.

## Exact changes to `run_fab_chain.py`
- Docstring: 2 lines describing `--target chip`.
- Imports: `from verification import chip_chain as cc` added after `from verification import chain_debug`.
- New option: `--target {demo,chip}`, default `demo`.
- `main`:
  - `wrappable_ops()` becomes `cc.wrappable_ops() if chip else wrappable_ops()`.
  - `cfg` gains `target=a.target`.
  - `--freq-mhz` is refused with `--target chip` (the chip flow uses the repo's 12 MHz target).
  - `--plan` uses `cc.plan_text` for chip.
  - The run loop, from:
    ```python
    removed = fc.clean_job(JOBS / op, op, cfg.from_stage)
    r = fc.run_chain(op, build_steps(op, cfg, exec_fn, preflight), cfg.from_stage, say)
    say("\n".join(fc.format_summary(r)))
    for line in diagnose(op, r):
    ```
    to:
    ```python
    removed = cc.clean_job(op, cfg.from_stage) if chip else fc.clean_job(JOBS / op, op, cfg.from_stage)
    steps = build_steps(op, cfg, exec_fn, preflight)
    if chip:
        steps = cc.build_chip_steps(op, cfg, exec_fn, preflight, steps["rtl"], say)
    r = fc.run_chain(op, steps, cfg.from_stage, say)
    say("\n".join(cc.format_summary(r) if chip else fc.format_summary(r)))
    for line in ([] if chip else diagnose(op, r)):
    ```
  - The banner says `CHIP CHAIN` for chip.
  - The last line prints the chip report path and evidence level.
- `write_report`: writes `reports/chip_chain_report.json` for chip and adds `"target"` to the report.

Your demo chain tests give identical results with the old and new file. Full diff: `docs/flow5_chip_chain.diff`.

## New files
- verification/chip_chain.py
- sim/IOBUF_chip_sim.v
- tests/test_chip_chain.py
- tests/fixtures/chip_chain/ (recorded netlist simulations, flow report, nextpnr timing lines)

## Fix 1 (2026-11-13): every operator failed at `bitstream: unknown` on the real machine
Cause: the repo Makefile runs `python3 bit_gen.py` with the first `python3` on PATH. The stage scripts put
`~/tools/chip/oss-cad-suite/bin` first, and that suite ships its own `python3` without `fasm`, so the log showed
`ModuleNotFoundError: No module named 'fasm'`. Reproduced in the sandbox with a suite that has its own `python3`.
By hand it worked, because the active `~/chipenv` came first on PATH.

`verification/chip_chain.py`, in both Ubuntu scripts, from
```bash
export PATH="$CAD/bin:$PATH"
```
to
```bash
export PATH="$(dirname "$PY"):$CAD/bin:$PATH"
```
The preflight also checks the Makefile's `python3`:
`FOUND make python3` / `MISSING fasm or numpy in the Makefile's python3 (...)`. So `env` blocks before `flow` if this
ever recurs. A flow failure now also shows the log's last error line and the log path.

`chip/chip_flow.py` gains one rule, so this cause is named instead of `unknown`:
```python
        ("bitstream", "python_module_missing", lambda: "no module named" in t),
```
plus its `RULE_EVIDENCE` entry: a real make log (`tests/fixtures/chip_flow/python_missing.bitstream.log`).
`tests/test_chip_flow.py` gains that classify case, and the new cause is added to the rule set the test pins.
