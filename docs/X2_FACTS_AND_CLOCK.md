# X2 facts in one place, the project clock, and a pinned chip toolchain

## 1. Every X2 / chip number that silicon hasn't confirmed: `chip/x2_facts.py`
| fact | value | status | used by |
|---|---|---|---|
| cell_bits | 4 | decision (yours, 2026-11-13) | hardware_config, cells per weight |
| clock_mhz | 14.0 | decision (below) | chain timing check, schedule µs |
| program_cycles_per_cell | 365 | measured (engine sim fit) | schedule costs |
| mvm_cycles / beat_cycles | 511 / 3 | measured (same fit) | schedule costs |
| blocks | 2 | model (bel.txt) | hardware_config |
| cell_compute_4bit | — | **missing**: the chip's X2 model has no 4-bit cells | 4-bit packing, 4-bit engines |
| silicon_variation | — | **missing** | (a noise model would go in chip/x2_golden.py) |
| error_bits_exposed | 2 | model (primitive has ERROR0/1 only) | — |

**The path to add things later:** put the number in `specifications/x2_overrides.json`; there's no code to edit:
```json
{"program_cycles_per_cell": {"value": 410, "status": "measured", "source": "bench test, chip #3"}}
```
- Every consumer reads `x2_facts.value()`, so the override is used at once, and reports list it as `overridden_by`.
- A malformed file, an unknown name, or a missing status or source is rejected, never ignored.
- A `missing` fact raises `FactMissing`, naming the file to fill in, if anything tries to use it.
- A brand-new kind of fact is one entry in `FACTS`.

## 2. Clock: 14.0 MHz
This is the highest clock every chip design meets, with a little margin. The repo's nextpnr recipe sets no target of its
own (it only compares against nextpnr's built-in 12 MHz), so the project clock is ours to choose.

| design | nextpnr Fmax |
|---|---|
| x2_engine2_4bit | 14.45 MHz (sets the clock) |
| x2_engine2 | 16.67 |
| x2_engine | 17.51 |
| add8_sat | 42.19 |
| relu8 | 151.52 |

- requant_i32_i8 has no register-to-register path through its hard multiplier, so nextpnr gives it no Fmax.
- The chip chain now judges timing against 14.0 MHz, not nextpnr's 12 MHz. A design at 13.9 MHz is UNVERIFIED even
  though nextpnr prints PASS.
- A1's schedules report `latency_us` at this clock: tiny MobileNetV2 is about 3,557,161 cycles ≈ 254 ms per inference
  with the best schedule (an upper bound on programming).
- A faster clock needs the 4-bit two-block engine pipelined further; then change `clock_mhz` (or override it).

## 3. Pinned toolchain in `chip/chip_flow.py`
The 2026-11-16 failure (`synth: syntax_error`) came from `~/.bashrc` putting the old Yosys 0.66 first on PATH.
- `run_flow` now puts `~/chipenv/bin` and `~/tools/chip/oss-cad-suite/bin` first itself, when they exist. Override them
  with `CHIP_PY` / `CHIP_CAD`.
- It refuses a Yosys older than 0.70 with a clear message.
- You no longer need the `export PATH=...` line before `run_chip_flow.py`.

## Exact changes
- **`chip/chip_flow.py`:** new `MIN_YOSYS`, `tool_env()`, `yosys_version()`. In `run_flow`, from
  ```python
  import os
  repo = ...
  missing = [t for t in TOOLS if not shutil.which(t)]
  ...
  env = dict(env or os.environ)
  ```
  to
  ```python
  repo = ...
  env = tool_env(env)
  missing = [t for t in TOOLS if not shutil.which(t, path=env["PATH"])]
  ```
  plus the version check.
- **`verification/chip_chain.py`:** `judge()` first re-judges each clock against `x2_facts.value("clock_mhz")`.
- **`compiler/x2_schedule.py`:**
  - `PROGRAM_CMD, MVM, BEAT` come from `x2_facts.FACTS` (same values).
  - `tile_costs` reads `x2_facts.value(...)`.
  - The metrics gain `latency_us` and `clock_mhz`.
- **`agents/mapping_agent.py`:** the schedule report gains `"x2_facts": xs.x2_facts.summary()`.
- **New:** `chip/x2_facts.py` and `tests/test_x2_facts.py`. Full diff: `docs/x2_facts_and_clock.diff`.
