"""
Prepare an A2-verified operator for the real FABulous flow (Ubuntu), and read the result back.

    python run_fab_prep.py --op add8_sat            # needs reports/rtl/add8_sat.v (run: python run_rtl_agent.py first)
    # then in Ubuntu:  bash reports/fab_jobs/add8_sat/run_fab_flow.sh add8_sat   and   bash reports/fab_jobs/add8_sat/run_fab_sim.sh add8_sat
    python run_flow_check.py --job reports/fab_jobs/add8_sat

Step 1 here: wrap the verified core in the fabric I/O interface, check the wrapper in Yosys against the NumPy golden model through the
pin mapping, and write the job folder. Nothing here runs FABulous; that needs the Linux toolchain.
Evidence level: Yosys RTL evaluation vs NumPy golden (NOT nextpnr, NOT silicon).
"""
import argparse
import sys
from pathlib import Path

from verification.fab_sim import make_fab_testbench
from verification.fab_wrap import check_fab_wrapper, make_fab_wrapper, top_name, wrapper_layout
from verification.rtl_check import OPS

FLOW_SCRIPT = """#!/usr/bin/env bash
# Runs ONE wrapped operator through the real FABulous flow (Ubuntu/WSL). Uses the demo project from docs/REAL_FABULOUS_FLOW.md.
#   bash run_fab_flow.sh <op> [clock target in MHz]       (run from anywhere; results are copied back next to this script)
# Without a clock target nextpnr uses its default (12 MHz) and stops with an error when a design is slower than that.
# Three steps, because FABulous 2.2.0 does NOT connect the pins of a custom design: it leaves .io_in() .io_out() .io_oeb() empty and
# Yosys then removes all the logic. tools/fab_connect.py writes the connections between "generate wrapper" and "compile".
set -uo pipefail
OP="${1:?usage: run_fab_flow.sh <op> [clock target in MHz]}"
FREQ="${2:-}"
case "$FREQ" in ""|*[!0-9.]*) [ -z "$FREQ" ] || { echo "clock target must be a number of MHz, got: $FREQ"; exit 2; } ;; esac
EXTRA=""
[ -n "$FREQ" ] && EXTRA=" --nextpnr-extra-args '--freq $FREQ'"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
DEMO="${DEMO:-$HOME/work/demo}"
. "$HOME/fabenv/bin/activate"
export PATH="$HOME/tools/oss-cad-suite/bin:$PATH"
cp "$HERE/${OP}_fab.v" "$DEMO/user_design/${OP}_fab.v"
cd "$(dirname "$DEMO")"
PROJ="$(basename "$DEMO")"
rm -f "$DEMO/user_design/${OP}_fab.bin" "$DEMO/user_design/${OP}_fab.fasm" "$DEMO/user_design/${OP}_fab_npnr_log.txt"
: > "$HERE/flow_console.log"
FABulous -p "$PROJ" run "load_fabric; run_FABulous_fabric; gen_user_design_wrapper user_design/${OP}_fab.v user_design/top_wrapper.v; exit" 2>&1 | tee -a "$HERE/flow_console.log"
python3 "$ROOT/tools/fab_connect.py" "$DEMO/user_design/top_wrapper.v" "${OP}_fab" 2>&1 | tee -a "$HERE/flow_console.log"
if [ "${PIPESTATUS[0]}" != "0" ]; then echo 97 > "$HERE/exit_code.txt"; echo "could not connect the pins: see above"; exit 1; fi
FABulous -p "$PROJ" run "load_fabric; compile_design ./user_design/${OP}_fab.v${EXTRA}; exit" 2>&1 | tee -a "$HERE/flow_console.log"
echo "${PIPESTATUS[0]}" > "$HERE/exit_code.txt"
for f in bin fasm; do cp "$DEMO/user_design/${OP}_fab.$f" "$HERE/" 2>/dev/null; done
cp "$DEMO/user_design/${OP}_fab_npnr_log.txt" "$HERE/" 2>/dev/null
cp "$DEMO/user_design/top_wrapper.v" "$HERE/top_wrapper.v" 2>/dev/null
cp "$DEMO/fabric.csv" "$HERE/fabric.csv" 2>/dev/null
echo "flow exit code: $(cat "$HERE/exit_code.txt"). Now run, in the project: python run_flow_check.py --job reports/fab_jobs/${OP}"
"""


SIM_SCRIPT = """#!/usr/bin/env bash
# Simulates the configured fabric (FABulous fabric RTL + YOUR bitstream) with Icarus Verilog, through FABulous' own `run_simulation`.
#   bash run_fab_sim.sh <op>        (run run_fab_flow.sh first; results are written next to this script)
set -uo pipefail
OP="${1:?usage: run_fab_sim.sh <op>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
DEMO="${DEMO:-$HOME/work/demo}"
. "$HOME/fabenv/bin/activate"
export PATH="$HOME/tools/oss-cad-suite/bin:$PATH"
# The bitstream that is simulated must be the one in this job folder (the audited one). The copy inside Ubuntu can differ: an Ubuntu crash
# leaves files it was writing EMPTY (seen: a 0-byte .bin, which made every fabric output unknown). So a different copy is replaced first.
NOTE=""
if [ -s "$HERE/${OP}_fab.bin" ] && ! cmp -s "$HERE/${OP}_fab.bin" "$DEMO/user_design/${OP}_fab.bin"; then
  WAS="$(stat -c %s "$DEMO/user_design/${OP}_fab.bin" 2>/dev/null || echo missing)"
  cp "$HERE/${OP}_fab.bin" "$DEMO/user_design/${OP}_fab.bin" || { echo "could not copy the bitstream into $DEMO/user_design"; exit 1; }
  NOTE="FAB_SIM NOTE: the bitstream copy in Ubuntu ($WAS bytes) differed from the job folder's ($(stat -c %s "$HERE/${OP}_fab.bin") bytes); replaced it with the job folder's before simulating"
  echo "$NOTE"
fi
if [ ! -s "$DEMO/user_design/${OP}_fab.bin" ]; then echo "no bitstream (missing or empty): run run_fab_flow.sh ${OP} first"; exit 1; fi
cp "$HERE/${OP}_fab_tb.v" "$DEMO/Test/${OP}_fab_tb.v"
cd "$(dirname "$DEMO")"
FABulous -p "$(basename "$DEMO")" run "load_fabric; run_simulation vcd ./user_design/${OP}_fab.bin; exit" 2>&1 | tee "$HERE/sim_console.log"
echo "${PIPESTATUS[0]}" > "$HERE/sim_exit_code.txt"
[ -n "$NOTE" ] && echo "$NOTE" >> "$HERE/sim_console.log"
echo "simulation finished (exit $(cat "$HERE/sim_exit_code.txt")). Now run, in the project: python run_flow_check.py --job reports/fab_jobs/${OP}"
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--op", required=True)
    a = ap.parse_args(argv)
    if a.op not in OPS:
        sys.exit(f"unknown op '{a.op}'. Available: {', '.join(OPS)}")
    try:
        wrapper_layout(a.op)
    except ValueError as e:
        sys.exit(f"cannot wrap {a.op}: {e}. Only a single wide input can be loaded in two steps; several inputs that do not fit are not built yet.")
    src = Path("reports/rtl") / f"{a.op}.v"
    if not src.exists():
        sys.exit(f"{src} not found: run  python run_rtl_agent.py --op {a.op}  first (only A2-verified Verilog is wrapped)")
    text = make_fab_wrapper(a.op, src.read_text(encoding="utf-8"))
    res = check_fab_wrapper(a.op, text)
    print(f"wrapper check ({top_name(a.op)}): {res['status'].upper()} at {res['stage']}"
          + (f" | {res['vectors']} vectors through the pin mapping, {res['flop_bits']} flip-flop bits" if res["status"] == "pass" else f" | {res['errors']}"))
    if res["status"] != "pass":
        return 1
    job = Path("reports/fab_jobs") / a.op
    job.mkdir(parents=True, exist_ok=True)
    (job / f"{a.op}_fab.v").write_text(text, encoding="utf-8")
    (job / "run_fab_flow.sh").write_text(FLOW_SCRIPT, encoding="utf-8", newline="\n")
    (job / f"{a.op}_fab_tb.v").write_text(make_fab_testbench(a.op), encoding="utf-8")
    (job / "run_fab_sim.sh").write_text(SIM_SCRIPT, encoding="utf-8", newline="\n")
    lay = wrapper_layout(a.op)
    if lay["mode"] == "two_step":
        print(f"job folder: {job}  | TWO-STEP LOAD: io_in[{lay['half'] - 1}:0] = half of the input, io_in[{lay['sel_pin']}] = which half (0 low, 1 high), "
              f"io_out[{lay['out_lsb'] + lay['out_width'] - 1}:{lay['out_lsb']}] = result (ASSUMED protocol and pin use)")
    else:
        print(f"job folder: {job}  | io_in[{lay['in_bits'] - 1}:0] = inputs, io_out[{lay['out_lsb'] + lay['out_width'] - 1}:{lay['out_lsb']}] = result (ASSUMED pin use)")
    print(f"next, in Ubuntu:  bash {job.as_posix()}/run_fab_flow.sh {a.op}")
    print(f"then, in Ubuntu:  bash {job.as_posix()}/run_fab_sim.sh {a.op}      (simulates the configured fabric against NumPy)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
