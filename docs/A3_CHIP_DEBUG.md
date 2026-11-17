# A3 on chip failures (`run_fab_chain.py --target chip`, `run_chain_debug.py --target chip`)

Until now the debug agent knew only the demo chain's causes, so it was skipped for chip runs. Now a chip run that does not
pass gets the same treatment as a demo run:
- the chain prints `likely cause [...]`, `fix:` and the second-opinion command, from the rules, with no LLM;
- `run_chain_debug.py --target chip` asks the LLM, and code keeps its answer only if the rules support it.

## How it decides (unchanged contract)
1. **Evidence:** `collect_chip_evidence` reads only facts. Sources:
   - the stage results;
   - `chip_build/chain_<op>/flow_report.json` (failure stage and cause) and its `Makefile` (is the corrected multiplier map in use?);
   - `reports/chip_jobs/<op>/flow_console.log` error lines;
   - the netlist simulation log (mismatches, undefined outputs, whether every wrong vector has a negative input);
   - the number of MACC blocks in the netlist;
   - the timing facts.
2. **Rules:** `chip_rule_candidates` lists the causes those facts support.
3. **LLM:** picks from the chip menu. `decide()` (unchanged) accepts its answer only if the rules support it; otherwise
   the rules answer; if the rules know nothing, the answer is `unknown`.

## Chip causes
- **Real runs:**
  - `chip_python_module_missing`: your machine, every operator at `bitstream`.
  - `chip_signed_multiplier`: your machine, requant 219/485.
  - `chip_timing_not_met`: x2_engine_4bit at 11.2 MHz.
- **Real logs from chip examples:** `chip_pins_exceeded`, `chip_over_capacity`, `chip_synth_error`.
- **Real sandbox runs:** `chip_tools_missing`, `chip_netlist_outputs_undefined`, `chip_sim_not_finished`.
- **From the chain's own messages and checks:** `chip_multiplier_setting_changed`, `chip_bitstream_audit_failed`,
  `chip_flow_crashed`, `chip_flow_tool_error`, `chip_wrapper_check_failed`, `chip_netlist_differs`, `chip_repo_modified`.
- **Shared with the demo:** `yosys_missing` (new, see below), `llm_key_missing`, `rtl_not_verified`, `ubuntu_not_answering`.

Each cause's `seen` field says which of these applies.

The tests use four REAL chip-chain failures, recorded by breaking one thing at a time in the sandbox
(`tests/fixtures/chip_debug/`):

| case | what was broken | named as |
|---|---|---|
| env_tools_missing | chipenv without fasm | chip_tools_missing |
| flow_python_missing | the Makefile's python3 without fasm | chip_python_module_missing |
| sim_signed_multiplier | the repo's unsigned multiplier map | chip_signed_multiplier |
| sim_outputs_undefined | the repo's IOBUF sim model | chip_netlist_outputs_undefined |

## A fix to the demo rules too: `yosys_missing`
A run started with a Python that has no `yowasp-yosys` fails the rtl stage with `Yosys not found`. The old rule called
that `rtl_not_verified` ("the Verilog did not pass ... regenerate it"), but the Verilog was never checked. The new cause
matches only that message and is checked first, on both chains. Your demo tests give identical results before and after.

## Exact changes
- **`verification/chain_debug.py`:**
  - `CAUSES` gains `yosys_missing`.
  - In `rule_candidates`, from
    ```python
    add("llm_key_missing" if status == "blocked" and "is not set" in low else "rtl_not_verified")
    ```
    to
    ```python
    add("yosys_missing" if "yosys not found" in low else
        "llm_key_missing" if status == "blocked" and "is not set" in low else "rtl_not_verified")
    ```
  - Added at the end: `CHIP_REPORT`, `CHIP_JOBS`, `CHIP_BUILD`, `CHIP_CAUSES`, `CHIP_SHARED`, `CAUSE_IDS` (now demo + chip + unknown),
    `cause_info()`, `chip_menu_ids()`, `collect_chip_evidence()`, `chip_rule_candidates()`. The demo functions are unchanged.
- **`agents/chain_debug_agent.py`:**
  - New `CHIP_MENU` and `CHIP_PROMPT`; the demo `PROMPT` is unchanged.
  - `run_chain_debug_agent(..., jobs_dir=None, ..., target="demo", build_dir=None)` picks the chip evidence, rules and prompt for `target="chip"`.
  - The mock and the result use `cd.cause_info()` instead of `cd.CAUSES.get()`.
  - The result gains `"target"`.
- **`run_chain_debug.py`:**
  - New `--target {demo,chip}`.
  - For chip it reads `reports/chip_chain_report.json` and writes `reports/chip_chain_debug_report.json`.
  - The banner says `CHIP CHAIN DEBUG`.
- **`run_fab_chain.py`:**
  - `diagnose(op, r, target="demo")` uses the chip evidence and rules for chip, and `cause_info()`; its second-opinion command adds `--target chip`.
  - In `main`, from `for line in ([] if chip else diagnose(op, r)):` to `for line in diagnose(op, r, a.target):`.

New: `tests/test_chip_debug.py`, `tests/fixtures/chip_debug/`. Full diff: `docs/a3_chip_debug.diff`.
