# Benchmark stage table, updated for the chip (2026-11-13)

`orchestration/benchmark.py` STAGE_STATUS says, for each stage of the project, whether it exists. The benchmark run
itself still executes S02–S04 only; the other stages run from their own scripts. Each row now also has:
- `where`: the files that implement it; `tests/test_benchmark.py` checks they exist.
- `missing`: what a partial stage still lacks; required for every stage that isn't `implemented`.

The stage table is not part of the result digest, so digests and the 37/44 result counts are unchanged.

| stage | before | now | what is still missing |
|---|---|---|---|
| S04 mapping / X2 package | partial | partial | partitioner doesn't estimate eFPGA budgets per node |
| S05 Verilog generation | not_implemented | **partial** | not generated from a workload's IR; fixed operator list |
| S06 lint / equivalence | not_implemented | **partial** | no formal equivalence, assertions or Verilator lint |
| S07–S09 Yosys / nextpnr / bitstream | not_implemented | **partial** | built per operator / X2 design, not per whole workload |
| S10 V0 audit | partial | **implemented** | (every chip-flow bitstream is audited) |
| S11 configured fabric + X2 | not_implemented | **partial** | configured fabric (tile RTL + bitstream) not simulated |
| S12 silicon | not_implemented | not_implemented | hardware |

S01–S03 keep their status. S03's missing item is now written into its row.

## Exact changes
- `orchestration/benchmark.py`:
  - `STAGE_STATUS` is replaced (same `id`/`name`/`status`/`how` keys, plus `where` and `missing`).
  - In finding F2, from `eFPGA resource use is NOT estimated (no fabric profile).` to
    `eFPGA resource use per node is NOT estimated here; the chip flow measures it per built operator (run_fab_chain.py --target chip).`
- `tests/test_benchmark.py`:
  - `test_stage_table_is_honest` asserts the new statuses.
  - New `test_stage_table_points_at_code_that_exists_and_says_what_is_missing`.
  - `test_list_workloads_tool` now expects `S05 == "partial"`.

Full diff: `docs/benchmark_stages_2026-11-13.diff`.
