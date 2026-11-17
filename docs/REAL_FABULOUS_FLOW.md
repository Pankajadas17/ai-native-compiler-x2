# Reproducing the real FABulous flow (what was actually run, and what was not)

Run in a Linux sandbox (Ubuntu 24.04, Python 3.12, 1 CPU, 4 GB RAM). Everything below produced the fixture in
`tests/fixtures/fabulous_demo/`. **Not tested:** native Windows (use WSL2, Docker or Codespaces; check the FABulous docs).

1. `python3 -m venv fabenv && . fabenv/bin/activate && pip install fabulous-fpga`   -> `FABulous --version` = 2.2.0
2. CAD tools: OSS CAD Suite build **2026-06-25** (the FABulous docs warn against Yosys nightlies after 29 June; this build has Yosys 0.66+154,
   nextpnr-generic 0.10-82, Icarus 14.0). Linux x64 archive (717 MB):
   `https://github.com/YosysHQ/oss-cad-suite-build/releases/download/2026-06-25/oss-cad-suite-linux-x64-20260625.tgz`, extracted (~2.4 GB);
   then `export PATH=<extract dir>/oss-cad-suite/bin:$PATH`. (`FABulous install oss-cad-suite` installs the LATEST build instead.)
3. `FABulous create-project demo`
4. **Sandbox quirk:** at start-up FABulous 2.2 tries to download a chip PDK (`FAB_PDK='ihp-sg13g2'` in `demo/.FABulous/.env`) from
   fossi-foundation.github.io. A bitstream does not need a PDK; with no access to that host the download fails (`Name or service not known`).
   Deleting the `FAB_PDK` line avoided it. On a normal internet connection FABulous simply downloads the PDK; no edit is needed.
5. `FABulous -p demo run "load_fabric; run_FABulous_fabric; gen_user_design_wrapper user_design/sequential_16bit_en.v user_design/top_wrapper.v; compile_design ./user_design/sequential_16bit_en.v; exit"`
   -> `Compile flow completed successfully.` Produces `sequential_16bit_en.bin` (12,024 bytes), `.fasm`, `.json`, nextpnr log, `.FABulous/bitStreamSpec.csv`.
   (`run FABulous.tcl` does NOT work: `run` takes commands, not a filename.)
6. `python tools/extract_bitstream_fixture.py demo sequential_16bit_en tests/fixtures/<name> --tool FABulous=2.2.0 ...` builds a small audit fixture.
7. `python run_bitstream_audit.py --demo` (or `--bin ... --fabric-csv ... [--fasm ... --spec ...]` for your own build).
