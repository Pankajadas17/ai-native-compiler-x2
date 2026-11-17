# Public mode and the public export (branch `public-mode`)

The goal: publish the project for feedback without risking the team pipeline or the private model repository.

**Rule followed:** with BMsemi_Model present (your machine), every command runs the same code in the same order as
before. The new paths only switch on when BMsemi_Model is absent.

| file | from | to |
|---|---|---|
| `compiler/partition.py` | `from emulators.platform_ops import map_layer` at import time, so **every** module using the partitioner (deploy, engine, schedule tests) crashed without BMsemi_Model | if BMsemi_Model is present, the same import as before. If it is absent, `map_layer` is a stand-in using the tile formula, and `MAPPER_SOURCE` says the repo-mapper cross-check was skipped. A BMsemi_Model that is present but broken still fails loudly |
| `tests/conftest.py` (new) | – | with BMsemi_Model, all test files are collected (as before). Without it, the 14 files built on BMsemi's simulator and the 3 that assert numbers of the private graphs are not collected. The pytest header names them |
| `tests/test_public_graphs.py` (new) | – | the same deployment checks on public stand-in graphs; runs in **every** checkout |
| `tools/make_public_fixtures.py` (new) + `tests/fixtures/ir_public/*.json` | – | three hand-made graphs (inverted-residual CNN, MLP chain, 5×5 edge filter); a test checks they hold no BMsemi data and regenerate identically |
| `deploy_chip.py` `_deepgate()` | the synthetic chain always used sizes 640-128-…-640 | the synthetic chain follows the graph it is planned on. With the team's graph the sizes are identical, so the results are identical (checked: same expected tensors) |
| `run_pipeline.py` | stage 1 failed without BMsemi_Model | stage 1 passes in **PUBLIC mode**, and the deploy stage uses the public graphs. With BMsemi_Model nothing changes. The report says which mode ran. Also included: the earlier fix that stops at stage 1 when numpy is missing (`.venv` not active), and "did not run" in place of stale results |
| `tools/export_public.py` (new) | – | makes the public copy in another folder: no private files, content check, the test suite run in the copy, then the push commands |
| `.github/workflows/ci.yml` (new) | – | runs only in the public repo: tests on every push (the badge); the whole pipeline on demand and weekly, with its report attached |
| `README.md` | the old opening | the opening shows the pipeline output, a results table, "Try it", public mode, an invitation for feedback, publishing and license notes; the rest is kept |

**Checked in the sandbox:**
- no BMsemi_Model, team graphs present: the suite passes;
- the export copy, clean Python venv with only `requirements.txt` + yowasp-yosys: **691 passed, 22 skipped**;
- public-mode pipeline: deploy, replay and status PASS.

**Still to check on your machine:** run `python -m pytest -q` and `python3 bm.py pipeline` with BMsemi_Model present. They
must give the same results as on `main` (885 Windows / 912 Ubuntu; 9/9).
