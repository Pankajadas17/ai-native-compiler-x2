# Chip flow runner (Phase 3a)

`run_chip_flow.py` builds any Verilog design for the real chip (`classic_fabric_reram`) with the chip
repo's own recipe (`user_designs/makefile.include`, unchanged), in `chip_build/<name>/`, and writes
`flow_report.json`. It is the hard check behind A1's choices and the source of real failures for A3.

## Steps
1. Copy the RTL to `chip_build/<name>/rtl/`, write a Makefile that includes the repo recipe.
2. `make synth`, then read the top module's ports from the netlist.
3. **Pin budget**: one IOBUF site per port bit; the fabric has **36**. (The 28 `all[...]` names in
   the repo's `constraints.pcf` are example assignments only, as the team confirmed.) Over budget
   -> stop with `pin_budget_exceeded`: the operator needs a narrower transfer protocol
   (serialised or multiplexed), which costs extra transfers.
4. Write `design.pcf` (port bits -> IOBUF sites in site order), `make pnr bitstream hex`.
5. Checks: device utilisation (used/available per primitive), every pin really constrained,
   FASM not empty, bitstream size = what the frame layout implies (35,304 bytes).

## Failure causes (for A3)
`failure = {stage, cause, evidence}`; `cause` comes from fixed rules and is `unknown` when no rule
matches (never guessed). `RULE_EVIDENCE` in `chip/chip_flow.py` records which rules were checked on
real tool output: all except `routing_failed` and `bitstream_size_mismatch`.

Two real tool behaviours found while calibrating:
* nextpnr places a pin that has no constraint by itself and still exits 0. The pin then sits on a
  site nobody chose, so the runner treats it as a failure (`unconstrained_io`).
* Over capacity shows up both in the utilisation table (`MACC_8x8_20: 20/18`) and as
  `no BELs remaining to implement cell type ...`.

## Examples (`examples/chip_designs/`, `--examples` checks each gives the expected result)
| Example | Expected | Why |
|---|---|---|
| repo_counter (counter.v from the repo) | ok | 11 pins, 9 logic cells |
| bitslice_acc_mux | ok | shift-and-add accumulator for bit-sliced X2 weights, result read 1 byte at a time: 30 pins |
| bitslice_acc_parallel | pin_budget_exceeded | same accumulator, 22-bit result in parallel: 42 pins |
| bitslice_acc_bad | syntax_error | deliberately broken |
| missing_mod | unknown_module | instantiates a module that does not exist |
| macc_over | over_capacity | 20 multipliers, 18 MACC blocks |

The bit-slice accumulator is an example operator that illustrates the pin trade-off; it is not a
decided design.
