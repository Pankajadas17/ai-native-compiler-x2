# The agents

A1 (mapping and schedule), A3 (failure diagnosis) and A4 (verification) on the chip. The LLM proposes; code decides.

- [How the agents run](#how-the-agents-run)
- [A1: the X2 schedule](#a1-the-x2-schedule)
- [A3: chip failures](#a3-chip-failures)
- [A4: the verification agent on the chip](#a4-the-verification-agent-on-the-chip)

## How the agents run

The agents run on Google's agent harness, the [Antigravity SDK](https://pypi.org/project/google-antigravity/).
`src/x2c/agents/harness.py` is the only file that imports it.
- The SDK runs the loop (model, tool calls, model ...). Each agent gives it a system prompt, its tools (plain Python
  functions) and an **end-of-turn check**.
- Only the agent's own tools are offered: the SDK's file, shell and web tools are switched off.
- **End-of-turn check:** when the agent says it is done, a hook runs code that decides whether it may stop. If not, the
  agent gets the reason and must continue. The checks:
  - A3: it read the evidence first, answered with one JSON object with every key, and chose a cause from the menu
    (at most 2 send-backs). Whether that cause is supported by the evidence is still decided by code after the run.
  - A2: the Yosys-vs-NumPy checker passed one of its submissions, or no submission is left (at most 3 send-backs). The
    Verilog is saved only after a checker pass; what the model says about its own design is never evidence.
  - A1: the validator accepted its last assignment (`validate_partition`) and, on the chip, its last schedule
    (`validate_schedule`) (at most 2 send-backs). Code still keeps the LLM's plan only if it is legal, and its schedule
    only if it is at least as fast as code's best.
  - A4 (verification agent): the full standard suite ran (default arguments, never fewer tests) and, if it failed,
    `get_failures` was read (at most 2 send-backs). The facts in the result are still recomputed from the reports.
  - A4 (test planner): `coverage_gaps` was called first, at least one spec was accepted and no coverage class is left
    open (at most 2 send-backs). The Python oracle and the replay against every bug are code, as before.
- Models: `BM_LLM_MODEL` (default `gemini-3.5-flash`), then `BM_LLM_FALLBACKS` (default `gemini-3.8-flash`,
  `gemini-3.5-flash-lite`). The next model is tried when one is busy (503), out of quota (429) or not available (404); a
  model whose daily quota is used up, or that does not exist, is skipped for the rest of the run. A wrong or missing key
  is raised, not hidden.
- **Last fallback, Groq:** when no Gemini model is left and `GROQ_API_KEY` is set, the task runs once more on Groq
  (`BM_GROQ_MODEL`, default `openai/gpt-oss-120b`) through `harness.plain_loop`, a small loop of ours with the same
  tools and the same end-of-turn check. With no model at all, the agent's code decides alone (A3 rules, A1 baseline,
  A2 "not verified", A4 facts from the reports).
- **No-model stand-ins:** `--llm mock` (and the pipeline) use fixed, rule-based behaviour per agent (`MockRTLLLM`
  submits the reference Verilog through the checker, `MockMappingLLM` proposes the baseline plan, ...). They are not
  models and are labelled as such; they also run through `plain_loop`.
- The free tier is small: 20 requests per model per day (seen 2026-11-18), reset at midnight Pacific time (13:30 in India).
  One agent task takes about 2 to 5 requests, so a full `x2c eval a3 --llm gemini --unseen --holdout` (12 bugs) needs
  both models' daily quota.
- Key: `GEMINI_API_KEY` in your environment, never in a project file. A free key: https://aistudio.google.com/apikey.
- Install: `uv sync --extra agents`. The tests use an offline stand-in model (`tests/fake_model.py`): no key needed.

On the harness so far: A3 (`x2c agent a3 --llm gemini`, `x2c eval a3 --llm gemini`), A2 (`x2c agent a2 --llm gemini`)
A1 (`x2c agent a1 --llm gemini`) and A4 (`x2c agent a4 --llm gemini`, `x2c eval a4 --llm gemini`).
`--llm mock` needs no key and is not a real model.

## A1: the X2 schedule

### The problem in one paragraph
The chip has 2 X2 blocks, and a layer needs more tiles than that (tiny MobileNetV2: 64 tiles on the chip). So tiles are
programmed, used, and replaced. Programming a tile is far slower than using it: per weight cell about 365 cycles, while an
MVM takes about 511 + 3 per row. The schedule decides which layers stay programmed, which block takes each tile, and in
what order; that decides the latency. A simple plan keeps the layer that saves the most programming resident and adds
the cycles up as if one block did everything in turn. Using both blocks, so that one programs while the other computes, can
hide much of the work. It's like prefetching the next batch while the GPU runs this one.

### Cost model, checked against real simulations
The four one-block engine simulations ([x2.md](x2.md)) fit one formula within 1.6 cycles:
`cycles = 365 x (SET + RESET commands) + 511 x MVMs + 3.1 x input beats`.
`tests/test_x2_schedule.py` re-fits it from the recorded logs on every run.
- **Programming:** every weight cell of a tile programmed once. This is an upper bound: the real count depends on the
  values, and the calibration runs used 21–29 % of it.
- **Not modelled** (listed in every result):
  - eFPGA operators between X2 layers;
  - host transfers.
- **Both blocks at once is built and measured:** `x2_engine2.v` ([x2.md](x2.md#two-block-engine-x2c-sim-engine2)) cut the latency of the 4 engine
  layers by 43–54 %, and `simulate()` predicted every run within 0.05 %. A schedule says `runs_on_built_x2_engine: true`
  only if it fits the one-block `x2_engine.v`.

### How A1 decides
1. The LLM picks a target (X2 or fabric) for each operator. Then it may call `describe_schedule`: per-tile program and compute cycles, the
   dependencies, the legal resident sets, the policies and the baseline latency. It does not see code's own best answer.
2. The LLM calls `propose_schedule` with a resident list and a policy (`earliest_free`, `round_robin`, `single_block`),
   or an explicit `[[layer, tile, block], ...]` list.
3. `validate_schedule` decides legality and computes the cycles. It rejects:
   - a tile scheduled twice, a missing tile, or a block that doesn't exist;
   - a resident block that is also used for streaming;
   - resident tiles that fill every block while layers still stream;
   - a tile issued before the layers it depends on (dependencies are traced through the graph, so residual adds count);
   - a resident set different from the plan's.
4. `best_schedule` tries every legal resident set with every policy. The system keeps the LLM's schedule only if it is
   legal and at least as fast as code's; otherwise code's, and the report says why (for example "llm schedule legal but 89.8 % slower").
5. The final plan's `chip.resident` follows the chosen schedule, and the final plan is re-validated by `validate_partition`.

### Results (graphs built with `compiler.ir`)

| graph | baseline (resident + one streaming block) | chosen | latency cut |
|---|---|---|---|
| mini (5 layers, 16 tiles, no 1-tile layer) | 4,487,870 cycles | no resident, earliest_free | 47.3 % |
| with a 1-tile first layer (like features_0_0) | 1,802,066 (c0 resident) | no resident, round_robin | 27.5 % |

In the second case code chooses MORE programming per inference (887,680 vs 730,000 cycles) for LESS latency,
because the two blocks overlap it. The team's real tiny MobileNetV2 graph is tested by `test_real_tiny_mobilenetv2`
(needs BMsemi_Model): it checks legality and that the result is no slower than the baseline, not exact numbers. On that
graph A1's schedule gives about 254 ms per inference at 14 MHz, 46 % below a single block.

## A3: chip failures

A chip run that does not pass gets the same diagnosis as a demo run:
- the chain prints `likely cause [...]`, `fix:` and the second-opinion command, from the rules, with no LLM;
- `x2c agent a3 --target chip` asks the LLM, and code keeps its answer only if the rules support it.

### How it decides
1. **Evidence:** `collect_chip_evidence` reads only facts. Sources:
   - the stage results;
   - `chip_build/chain_<op>/flow_report.json` (failure stage and cause) and its `Makefile` (is the corrected multiplier map in use?);
   - `reports/chip_jobs/<op>/flow_console.log` error lines;
   - the netlist simulation log (mismatches, undefined outputs, whether every wrong vector has a negative input);
   - the number of MACC blocks in the netlist;
   - the timing facts.
2. **Rules:** `chip_rule_candidates` lists the causes those facts support.
3. **LLM:** picks from the chip menu. `decide()` accepts its answer only if the rules support it; otherwise
   the rules answer; if the rules know nothing, the answer is `unknown`.

### Chip causes
- **Real runs:**
  - `chip_python_module_missing`: every operator failed at `bitstream`.
  - `chip_signed_multiplier`: requant, 219 of 485 vectors wrong.
  - `chip_timing_not_met`: x2_engine_4bit at 11.2 MHz.
- **Real logs from chip examples:** `chip_pins_exceeded`, `chip_over_capacity`, `chip_synth_error`.
- **Real runs in a test environment:** `chip_tools_missing`, `chip_netlist_outputs_undefined`, `chip_sim_not_finished`.
- **From the chain's own messages and checks:** `chip_multiplier_setting_changed`, `chip_bitstream_audit_failed`,
  `chip_flow_crashed`, `chip_flow_tool_error`, `chip_wrapper_check_failed`, `chip_netlist_differs`, `chip_repo_modified`.
- **Shared with the demo:** `yosys_missing`, `llm_key_missing`, `rtl_not_verified`, `ubuntu_not_answering`.

Each cause's `seen` field says which of these applies.

The tests use four REAL chip-chain failures, recorded by breaking one thing at a time in a test environment
(`tests/fixtures/chip_debug/`):

| case | what was broken | named as |
|---|---|---|
| env_tools_missing | the chip Python without fasm | chip_tools_missing |
| flow_python_missing | the Makefile's python3 without fasm | chip_python_module_missing |
| sim_signed_multiplier | the repo's unsigned multiplier map | chip_signed_multiplier |
| sim_outputs_undefined | the repo's IOBUF sim model | chip_netlist_outputs_undefined |

## A4: the verification agent on the chip

The verification agent reports the chip work, the same way it reports the benchmark. The LLM chooses tools;
code does the work and computes every number.

| tool | what it does | runs anything? |
|---|---|---|
| `chip_status` | chain verdicts; each chip design's build (logic cells, X2 blocks, pins, V0 audit, nextpnr Fmax vs the 14 MHz project clock); the two-block engine check; real-layer end-to-end results; X2 facts and which are still missing | no, reads reports; a missing report is shown with the command that makes it |
| `chip_schedule` | the baseline vs the best X2 schedule for the real tiny MobileNetV2, in cycles and µs | Python only, about a second |
| `run_chip_layers` | the 4-bit end-to-end check of all 17 real layers in Python (assumed cell rule); `positions` 1–16 | Python only; writes `reports/chip_layer_e2e/report.json` |

When any chip tool was used, `run_agent` adds `verified_chip_facts`. That block is `chip_status()` recomputed by code at
the end, so an LLM that overclaims ("verified on silicon") can't change it.
