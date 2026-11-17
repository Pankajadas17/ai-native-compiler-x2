# Flow step 4: the X2 engine (bridge + on-fabric recombination)

Step 3 sent every X2 COMPUTE result back to the host and the host did the shift/sign/add. Step 4 does that on the
fabric. `examples/chip_designs/x2_engine.v` takes the packed command stream (step 1) through the same 12-bit host
port as the bridge, and every COMPUTE count goes straight into an on-fabric 32-bit accumulator (`acc +/- count << shift`,
the step-2 `x2_acc_step` arithmetic). The host only reads finished partial outputs. Only row folds (layers with more
than 32 inputs per output) are still added on the host.

## How a column finds its accumulator
The packing (compiler/x2_pack.py, chip_x2.place_groups) starts every block on a multiple of CPW columns, so the
column number alone decides everything:

| | formula | 4-bit cells | 1-bit cells |
|---|---|---|---|
| CPW (cells per weight) | 2 * ceil(8 / CELL_BITS) | 4 | 16 |
| accumulators per tile | 32 / CPW | 8 | 2 |
| accumulator | col / CPW | | |
| negative bank | col % CPW >= CPW/2 | | |
| shift | (col % (CPW/2)) * CELL_BITS + slice * SPLIT | SPLIT 4 | SPLIT 8 |

`run_x2_engine_check.engine_layout` refuses any placement that breaks the alignment.

## Host pins (33 of 36)
clk, rst, cmd_data[11:0], cmd_hi, cmd_strobe, cmd_ready (as the step-3 bridge), slice, acc_clear, idle,
acc_sel[2:0], byte_sel[1:0], acc_byte[7:0]. Host loop per (position, tile): `acc_clear`, then for each input slice set
`slice` and stream the COMPUTE beats, wait for `idle`, read the accumulators byte by byte.

## Results
Simulation (1-bit cells, BM behavioral X2 wired like the chip, 2 positions, seed 2026):

| layer | tiles | reads | cycles | = engine model | output = NumPy |
|---|---|---|---|---|---|
| conv | 4 | 16 | 374493 | yes | yes |
| dw | 4 | 16 | 135202 | yes | yes |
| pw | 8 | 32 | 238518 | yes | yes |
| fc | 5 | 20 | 309281 | yes | yes |

Also PASS with seeds 11, 222, 3333 and 3 positions.

Chip builds (FABulous flow, 12 MHz target, nextpnr estimate):

| design | logic cells | RERAM blocks | Fmax | V0 audit |
|---|---|---|---|---|
| x2_engine (1-bit) | 390 | 1 of 2 | 17.5 MHz | 9592/9592 |
| x2_engine_4bit | 894 | 1 of 2 | 11.2–12.4 MHz (borderline, varies by placement seed) | 22335/22335 |

4-bit is checked for size and timing, and in Python (engine rule rebuilds NumPy exactly), but cannot be simulated
on the chip: the chip's X2 commands only SET/RESET 1-bit cells.

## Finding: clock-to-output race in simulation
The BM behavioral model updates its outputs with blocking assignments at the rising edge. Fabric logic that samples
at the same edge raced it and sometimes missed a result (the pw layer failed). The real macro has a clock-to-output
delay, so `sim/RERAM_CIM_32x32_sim.v` now passes all 32 outputs through `assign #1`. Bridge results are unchanged
(identical R lines to the step-3 fixtures).

## Exact changes to existing files

### sim/RERAM_CIM_32x32_sim.v
From (each output assigned directly, e.g.)
```verilog
    assign RESULT_DATA0 = rd[0];
    assign ERROR0 = err[0]; assign ERROR1 = err[1];
```
to (the model drives `*_m` wires, every output goes through a 1 ns delay, plus a 3-line header comment)
```verilog
    wire CMD_READY_m, BUSY_m, COMMAND_ACCEPTED_m, COMPUTE_PENDING_m, ERROR_VALID_m, OPERATION_DONE_m, PROGRAM_READBACK_ACTIVE_m, RESULT_VALID_m, RESULT_LAST_m;
    assign #1 CMD_READY = CMD_READY_m;   ... (9 status lines)
    assign #1 RESULT_DATA0 = rd[0];      ... (21 data lines)
    assign #1 ERROR0 = err[0]; assign #1 ERROR1 = err[1];
```

### run_chip_flow.py
EXAMPLES gains two lines:
```python
    "x2_engine": (EX / "x2_engine.v", "x2_engine", "ok", None),
    "x2_engine_4bit": ([EX / "x2_engine_4bit.v", EX / "x2_engine.v"], "x2_engine_4bit", "ok", None),
```
and an example may list several sources. From
```python
            path = repo.path(src[5:]) if str(src).startswith("REPO:") else Path(src)
            jobs.append((n, [path], top, st, cause))
```
to
```python
            srcs = src if isinstance(src, list) else [src]
            paths = [repo.path(x[5:]) if str(x).startswith("REPO:") else Path(x) for x in srcs]
            jobs.append((n, paths, top, st, cause))
```

### chip/chip_flow.py
After `(work / "rtl").mkdir(parents=True)` two lines are added, so a crashed run leaves a folder the next run may replace:
```python
    # mark the folder as this runner's at once: if anything below crashes, the next run may still replace it
    (work / "flow_report.json").write_text(json.dumps({"design": name, "top": top, "status": "started"}) + "\n")
```

### run_x2_bridge_check.py
Same as the file you edited: OUT is `chip_build/x2_bridge_sim`; the docstring now says so too.

## New files
examples/chip_designs/x2_engine.v, examples/chip_designs/x2_engine_4bit.v, sim/x2_engine_tb.v,
run_x2_engine_check.py, tests/test_x2_engine.py, tests/fixtures/x2_engine/engine_{conv,dw,pw,fc}.log.
Full diff: docs/flow4_engine.diff.
