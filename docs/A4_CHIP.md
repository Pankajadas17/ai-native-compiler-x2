# A4 (verification agent) on the BM Labs chip

The verification agent can now report the chip work, the same way it reports the benchmark. The LLM chooses tools;
code does the work and computes every number.

| tool | what it does | runs anything? |
|---|---|---|
| `chip_status` | chain verdicts; each chip design's build (logic cells, X2 blocks, pins, V0 audit, nextpnr Fmax vs the 14 MHz project clock); the two-block engine check; real-layer end-to-end results; X2 facts and which are still missing | no, reads reports; a missing report is shown with the command that makes it |
| `chip_schedule` | today's vs the best X2 schedule for the real tiny MobileNetV2, in cycles and µs | Python only, about a second |
| `run_chip_layers` | the 4-bit end-to-end check of all 17 real layers in Python (assumed cell rule); `positions` 1–16 | Python only; writes `reports/chip_layer_e2e/report.json` |

When any chip tool was used, `run_agent` adds `verified_chip_facts`. That block is `chip_status()` recomputed by code at
the end, so an LLM that overclaims ("verified on silicon") can't change it.

## Exact changes
- **`verification/tools.py`:**
  - Three wrapper functions (`chip_status`, `chip_schedule`, `run_chip_layers`) before `TOOLS`; they're added to
    `TOOLS` and get three entries at the top of `TOOL_SCHEMAS`.
  - In `call_tool`, after the `seed` line:
    ```python
    if "positions" in args:
        args["positions"] = int(args["positions"])
        if not 1 <= args["positions"] <= 16:
            return {"error": "positions must be between 1 and 16"}
    ```
  - From `except (TypeError, ValueError, FileNotFoundError) as e:` to
    `except (TypeError, ValueError, FileNotFoundError, ImportError) as e:`. On a machine without BMsemi_Model,
    `chip_schedule` then returns an error instead of crashing the agent.
- **`agents/verification_agent.py`:**
  - New `_chip_facts()`.
  - A `chip_used` flag, set when a chip tool succeeds.
  - The result gains `"verified_chip_facts": _chip_facts() if chip_used else None`.
- **New:** `verification/chip_tools.py`, `tests/test_chip_tools.py`, `tests/fixtures/chip_tools/` (recorded reports).
- `docs/X2_LAYER_E2E.md` gains your real 17-layer table.

Full diff: `docs/a4_chip.diff`.
