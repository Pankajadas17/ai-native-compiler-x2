# Development

Setting up, the repository layout, public mode, the maintainer tools, and measured fabric costs.

- [Setup](#setup)
- [Repository layout](#repository-layout)
- [Public mode](#public-mode)
- [Maintainer tools](#maintainer-tools)
- [Using a newer X2 model](#using-a-newer-x2-model)
- [Measured fabric costs](#measured-fabric-costs)

## Setup

Linux, or WSL on Windows.

1. Clone this repository, and inside it the chip repository at the pinned commit in
   `specifications/external_repos.json`, with its submodules. The chip repository is only read, never modified.
2. Install [uv](https://docs.astral.sh/uv/), then run `uv sync --all-extras`. This makes `.venv` from `pyproject.toml`
   and `uv.lock`:
   - `numpy` always;
   - `fasm` (extra `chip`), which the chip repo's `bit_gen.py` needs;
   - `google-antigravity` (Google's agent harness) and `groq` (the last fallback), extra `agents`, only needed for
     runs with a real model;
   - `pytest` (group `dev`).
3. Install the [OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build) with Yosys 0.70 or newer. Put it on
   `PATH`, unpack it to `~/tools/chip/oss-cad-suite`, or set `CHIP_CAD`.

Run commands with `uv run x2c …`, or activate `.venv` and run `x2c …`. `uv run x2c` lists every command, and
`uv run pytest -q` runs the tests.

The chip flow runs the chip repo's Makefile, which calls `bit_gen.py` with the first `python3` on `PATH`. x2c puts the
Python it runs with (the project's `.venv`, which has `fasm`) first. `CHIP_PY` overrides this with another Python.

## Repository layout

```
src/x2c/
  cli.py            the x2c command: every command, grouped
  commands/         one module per command (x2c sim engine2 -> commands/sim_engine2.py)
  agents/           A1-A4 and the LLM client
  compiler/         IR, partition, X2 packing, schedule, INT8 network
  chip/             chip profile, flow, bitstream audit, X2 golden model and facts, signed multiplier map (techmap/)
  verification/     checkers, operator chain, formal, failure rules
  inputs/           workload sources, schemas and normalizers
  bmsemi/           bridges to the team's private BMsemi_Model, and its benchmark
  paths.py          where the project folder is
hardware/
  designs/          the X2 designs: recombiner, bridge, one- and two-block engines
  operators/        A2's verified fabric operators (relu8, add8_sat, requant_i32_i8)
  formal/           X2 stand-in and engine properties for SymbiYosys
  sim/              testbenches and simulation models
  test_designs/     small and deliberately broken designs for the flow tests
specifications/     chip profile, hardware config, fabric costs, pinned repos
tests/              the test suite and its fixtures (recorded runs, public stand-in graphs)
tools/              maintainer scripts (below)
```

Generated output goes to `chip_build/`, `deploy/` and `reports/`, which git ignores.

## Public mode

The team also uses a private model repository, BMsemi_Model, with its own trained models. Without it, everything still
runs.

- `src/x2c/compiler/partition.py` uses BMsemi's `map_layer` when BMsemi_Model is present. Without it, it uses a stand-in based
  on the tile formula, and `MAPPER_SOURCE` records that the cross-check with the repo's mapper was skipped. A
  BMsemi_Model that is present but broken still fails loudly.
- `tests/conftest.py` collects every test file when BMsemi_Model is present. Without it, the files built on BMsemi's
  simulator and the ones that check numbers of the private graphs are not collected, and the pytest header names them.
- `tests/test_public_graphs.py` runs the same deployment checks on public stand-in graphs in every checkout. The graphs
  in `tests/fixtures/ir_public/` are hand-made by `tools/make_public_fixtures.py` and contain nothing from BMsemi_Model.
- `x2c pipeline` passes stage 1 in **PUBLIC mode**, and its deploy stage uses the public graphs. The report says which
  mode ran.

The CI in the public repository runs exactly this: the test suite on every push, and the whole pipeline on demand and
weekly.

## Maintainer tools

| tool | what it does |
|---|---|
| `tools/prepush_check.py` | run before every push. It refuses API keys, `.env` files, the cloned repos, environments and generated folders. It warns about third-party imports that `pyproject.toml` does not list, large files and documents. `--pin` records the commits of the two cloned repos in `specifications/external_repos.json` |
| `tools/export_public.py <folder>` | makes the public copy in another folder: every tracked file except the private ones, a content check (no private path, no team data, no key), then the test suite in the copy, where BMsemi_Model does not exist. Then it prints the push commands |
| `tools/make_public_fixtures.py` | writes the public stand-in graphs (deterministic) |
| `tools/extract_bitstream_fixture.py` | makes a small, self-contained bitstream-audit fixture from a compiled FABulous project |

## Using a newer X2 model

Set `BM_X2_REPO` to an X2 repository (or straight to its `.sv` file). `ChipRepo.path()` in `src/x2c/chip/chip_repo.py`
then hands that file to every reader: simulations, chip profile, hardware config and the chain. Reports record which
model was used. `x2c recheck` re-runs everything that depends on the X2 model, in order: profile, hardware config,
golden vs model, packing, engine, deploys.

## Measured fabric costs

`x2c chip costs` reads the costs per block from the chip builds (`chip_build/*/flow_report.json` and `pnr.log`) into
`specifications/fabric_costs.json`. `estimate(blocks)` sums them against the chip's budget. Missing builds are
reported, never guessed.

| block | design | logic cells | pins | MACC | fmax |
|---|---|---|---|---|---|
| x2_engine | x2_engine2_4bit | 1,792 | 35 | 0 | 14.45 MHz |
| relu | chain_relu8 | 15 | 17 | 0 | 151.5 MHz |
| add | chain_add8_sat | 35 | 25 | 0 | 42.2 MHz |
| requant | chain_requant_i32_i8 | 134 | 26 | 4 | combinational |

Engine + relu + add + requant in one design would need about **1,976 / 2,160 cells, 4 / 18 MACC, 35 pins**, so it would
fit. This is a sum of measurements, not a build. Building that combined design is the real test.
