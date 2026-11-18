"""
Chip chain: one A2-verified operator through every stage on the BM Labs chip (sky130-reram-fpga), one verdict.

    rtl    A2's Verilog re-checked in Yosys against NumPy (the same stage as the demo chain)
    prep   the core is wrapped for the chip's pins (36 IOBUF sites, one is clk); the wrapper is checked in Yosys vs NumPy
    env    Ubuntu has the chip toolchain (OSS CAD Suite in ~/tools/chip, fasm + numpy in the project environment (uv sync --all-extras), the chip repo)
    flow   x2c build in Ubuntu: Yosys (repo techmaps) -> nextpnr -> bitstream -> V0 audit, out of the chip repo
    sim    the SYNTHESIZED NETLIST (what nextpnr places) simulated in Icarus with the repo's FABULOUS_LC primitive models,
           driven through the chip-wrapper pins, every vector compared with NumPy
    check  one verdict from flow_report.json (status, pins, V0 audit, repo untouched), nextpnr timing and the simulation

Same rules as the demo chain (verification/fab_chain.py run_chain): a failing stage stops the chain, old results are
cleared before a run, PASS only if every stage passed in this run.

Why a netlist and not the fabric: simulating the configured fabric needs the fabric's tile RTL loaded with the bitstream;
that is not runnable here yet (S11). The netlist is one step closer to the chip than the RTL: it is Yosys' output with the
repo's own techmaps, made of the cells the bitstream configures. Together with the V0 audit (bitstream bits == FASM bits)
the remaining unchecked link is nextpnr's FASM writer and the silicon.

Two repo issues found and handled here WITHOUT editing the repo:
  * primitives/IOBUF/yosys/primitives/IOBUF.v drives the pad when EN=0, but the techmap (IOBUF_map.v) ties EN=1 on
    outputs, and the chip passes EN out as fabric_io_oe_o with gpio_oeb = ~oe (src/double_wide_openframe_project_wrapper.v
    l.122): on the chip EN=1 drives. Simulated with the repo model, every output floats. hardware/sim/IOBUF_chip_sim.v has the
    chip's polarity.
  * primitives/FABULOUS_LC/yosys/primitives/prims.v LUTFF_SSE uses an undeclared R (l.173). Icarus elaborates every
    unused module as a top, so the testbench is selected with -s and the broken cell is never elaborated.
Evidence level: Yosys RTL evaluation + real chip flow (repo techmaps, nextpnr, bitstream, V0 audit) + simulation of the
synthesized netlist vs NumPy (NOT the configured fabric, NOT timing-annotated, NOT silicon).
"""
import json
import re
import tempfile
import time
from pathlib import Path

from x2c import paths

import numpy as np

from x2c.verification import fab_chain as fc
from x2c.verification.rtl_check import BANNED, MAX_VERILOG_CHARS, OPS, _run, _signed, vectors_for

ROOT = paths.ROOT
JOBS = Path("reports/chip_jobs")
REPORT = Path("reports/chip_chain_report.json")
CHIP_BUILD = Path("chip_build")
HALF, PIN_FALLBACK = 16, 36
EVIDENCE_LEVEL = ("yosys_rtl_evaluation + real_chip_flow(repo techmaps, nextpnr, bitstream, V0 audit) + "
                  "synthesized_netlist_simulation_vs_numpy (NOT configured fabric, NOT timing-annotated, NOT silicon)")
ALWAYS_NOT_CHECKED = ("the configured fabric (tile RTL + bitstream) in simulation", "timing-annotated simulation", "silicon")
SIM_SRC = ROOT / "hardware" / "sim" / "IOBUF_chip_sim.v"
PRIMS = ("ip/fabulous-tiles/primitives/FABULOUS_LC/yosys/primitives/prims.v",          # every model the repo's synth uses,
         "ip/fabulous-tiles/primitives/GBUF/yosys/primitives/GBUF.v",                    # except IOBUF (hardware/sim/IOBUF_chip_sim.v)
         "ip/fabulous-tiles/primitives/MACC_8x8_20/yosys/primitives/MACC_8x8_20.v",
         "ip/fabulous-tiles/primitives/RAM_32x4_2R_1W/yosys/primitives/RAM_32x4_2R_1W.v")

FLOW_FILES = ("flow_console.log", "exit_code.txt")
SIM_FILES = ("netlist.v", "netlist.vvp", "sim_console.log", "sim_exit_code.txt")
CHECK_FILES = ("chip_check.json",)

PREFLIGHT_SCRIPT = r"""#!/usr/bin/env bash
# Checks that Ubuntu has what the chip chain needs. Changes nothing. Written by x2c chain --target chip.
CAD="${CHIP_CAD:-$HOME/tools/chip/oss-cad-suite}"
PY="${CHIP_PY:-$(command -v python3)}"
# the project's Python first: the repo Makefile runs bit_gen.py with the first python3 on PATH, and OSS CAD Suite ships its own (no fasm)
export PATH="$(dirname "$PY"):$CAD/bin:$PATH"
for t in yosys nextpnr-generic iverilog vvp make; do
  if command -v "$t" >/dev/null 2>&1; then echo "FOUND $t"; else echo "MISSING $t"; fi
done
if [ -x "$PY" ]; then echo "FOUND chip python"; else echo "MISSING chip python ($PY)"; fi
if "$PY" -c "import fasm, numpy" 2>/dev/null; then echo "FOUND fasm+numpy"; else echo "MISSING fasm or numpy in $PY (uv sync --all-extras)"; fi
if python3 -c "import fasm, numpy" 2>/dev/null; then echo "FOUND make python3"; else echo "MISSING fasm or numpy in the Makefile's python3 ($(command -v python3))"; fi
cd "$1" 2>/dev/null || { echo "MISSING project folder $1"; echo "PREFLIGHT DONE"; exit 0; }
if [ -f "${BM_CHIP_REPO:-sky130-reram-fpga}/user_designs/makefile.include" ]; then echo "FOUND chip repo"; else echo "MISSING chip repo (sky130-reram-fpga with its submodules)"; fi
if [ -f specifications/chip_profile.json ]; then echo "FOUND chip profile"; else echo "MISSING specifications/chip_profile.json (x2c chip profile)"; fi
echo "VERSION yosys $(yosys -V 2>/dev/null | head -1)"
echo "VERSION nextpnr-generic $(nextpnr-generic --version 2>&1 | head -1)"
echo "VERSION iverilog $(iverilog -V 2>/dev/null | head -1)"
echo "PREFLIGHT DONE"
"""

STEP_SCRIPT = r"""#!/usr/bin/env bash
# One chip-chain stage in Ubuntu: flow or sim. Written by x2c chain --target chip; reads/writes only the project folder.
CAD="${{CHIP_CAD:-$HOME/tools/chip/oss-cad-suite}}"
PY="${{CHIP_PY:-$(command -v python3)}}"
export PATH="$(dirname "$PY"):$CAD/bin:$PATH" PYTHONDONTWRITEBYTECODE=1     # the project's Python first: the Makefile's python3 needs fasm
cd "{root}" || exit 2
JOB="{job}"
if [ "$1" = flow ]; then
  PYTHONPATH="$PWD/src" "$PY" -m x2c build --name "{name}" --top "{top}" "$JOB/{wrapper}" > "$JOB/flow_console.log" 2>&1
  echo $? > "$JOB/exit_code.txt"
elif [ "$1" = sim ]; then
  PYTHONPATH="$PWD/src" "$PY" -m x2c.verification.chip_chain sim "{op}" > "$JOB/sim_console.log" 2>&1
  echo $? > "$JOB/sim_exit_code.txt"
fi
"""


# ---------------------------------------------------------------- pins and wrapper
def chip_pins(profile_path=ROOT / "specifications" / "chip_profile.json"):
    try:
        return int(json.loads(Path(profile_path).read_text())["io"]["iobuf_sites"])
    except (OSError, KeyError, ValueError):
        return PIN_FALLBACK


def chip_layout(op, pins=None):
    """direct: every input has its own pins; two_step: one wide input is loaded over 16 data pins in two clock ticks
    (sel = 0 low half, sel = 1 high half; the result is on the pins after the third tick). Pin count includes clk."""
    pins = pins or chip_pins()
    s = OPS[op]
    in_bits = sum(s["inputs"].values())
    on, ow = s["output"]
    if 1 + in_bits + ow <= pins:
        return {"mode": "direct", "pins": 1 + in_bits + ow, "inputs": dict(s["inputs"]), "out_name": on, "out_width": ow}
    if len(s["inputs"]) == 1 and HALF < in_bits <= 2 * HALF and 1 + HALF + 1 + ow <= pins:
        name = next(iter(s["inputs"]))
        return {"mode": "two_step", "pins": 1 + HALF + 1 + ow, "in_name": name, "in_width": in_bits, "half": HALF,
                "out_name": on, "out_width": ow}
    why = " (it runs inside the X2 engine, hardware/designs/x2_engine.v)" if op == "x2_acc_step" else ""
    raise ValueError(f"{op} needs 1 clk + {in_bits} input + {ow} result pins = {1 + in_bits + ow}; the chip has {pins}{why}")


def wrappable_ops():
    ok, no = [], {}
    for op in OPS:
        try:
            chip_layout(op)
            ok.append(op)
        except ValueError as e:
            no[op] = str(e)
    return ok, no


def top_name(op):
    return f"{op}_chip"


def make_chip_wrapper(op, core_verilog):
    """Wrapper top first (ports = chip pins), then <op>_chip_step (the logic of one clock tick, no registers, checkable with
    Yosys eval), then the verified core. Only the top holds registers."""
    lay, s, top = chip_layout(op), OPS[op], top_name(op)
    on, ow = lay["out_name"], lay["out_width"]
    if lay["mode"] == "direct":
        ins = lay["inputs"]
        ports = ",\n".join(f"    input  wire [{w - 1}:0] {n}" for n, w in ins.items())
        conn = ", ".join(f".{n}({n})" for n in ins)
        rconn = ", ".join(f".{n}({n}_r)" for n in ins)
        regs = "\n".join(f"    reg  [{w - 1}:0] {n}_r;" for n, w in ins.items())
        loads = "\n".join(f"        {n}_r <= {n};" for n in ins)
        text = f"""`default_nettype none
// x2c chip wrapper (direct): every operand has its own pins. Operands and result are registered on clk, so the core
// sits between two registers and nextpnr times it (pin-to-register paths have no delay model). Result: 2 ticks after the operands.
module {top} (
    input  wire clk,
{ports},
    output wire [{ow - 1}:0] {on}
);
{regs}
    reg  [{ow - 1}:0] q_r;
    wire [{ow - 1}:0] q_n;
    {top}_step step ({rconn}, .q_n(q_n));
    always @(posedge clk) begin
{loads}
        q_r <= q_n;
    end
    assign {on} = q_r;
endmodule

module {top}_step (
{",".join(chr(10) + f"    input  wire [{w - 1}:0] {n}" for n, w in ins.items())},
    output wire [{ow - 1}:0] q_n
);
    wire [{ow - 1}:0] r;
    {s['top']} core ({conn}, .{on}(r));
    assign q_n = r;
endmodule
`resetall
"""
    else:
        h, w, n = lay["half"], lay["in_width"], lay["in_name"]
        hw = w - h
        text = f"""`default_nettype none
// x2c chip wrapper (two-step load, ASSUMED protocol, ours): sel=0 stores d as the low {h} bits of {n},
// sel=1 stores d[{hw - 1}:0] as the high {hw} bits; the result of the stored value is on {on} after the next tick.
module {top} (
    input  wire clk,
    input  wire [{h - 1}:0] d,
    input  wire sel,
    output wire [{ow - 1}:0] {on}
);
    reg  [{h - 1}:0] lo_r;
    reg  [{hw - 1}:0] hi_r;
    reg  [{ow - 1}:0] q_r;
    wire [{h - 1}:0] lo_n;
    wire [{hw - 1}:0] hi_n;
    wire [{ow - 1}:0] q_n;
    {top}_step step (.d(d), .sel(sel), .lo(lo_r), .hi(hi_r), .lo_n(lo_n), .hi_n(hi_n), .q_n(q_n));
    always @(posedge clk) begin
        lo_r <= lo_n;
        hi_r <= hi_n;
        q_r  <= q_n;
    end
    assign {on} = q_r;
endmodule

module {top}_step (
    input  wire [{h - 1}:0] d,
    input  wire sel,
    input  wire [{h - 1}:0] lo,
    input  wire [{hw - 1}:0] hi,
    output wire [{h - 1}:0] lo_n,
    output wire [{hw - 1}:0] hi_n,
    output wire [{ow - 1}:0] q_n
);
    assign lo_n = sel ? lo : d;
    assign hi_n = sel ? d[{hw - 1}:0] : hi;
    wire [{w - 1}:0] v = {{hi, lo}};
    wire [{ow - 1}:0] r;
    {s['top']} core (.{n}(v), .{on}(r));
    assign q_n = r;
endmodule
`resetall
"""
    return text + "\n" + core_verilog.strip() + "\n"


def _eval_lines(op, lay, vec):
    """Yosys eval commands on <op>_chip_step: direct = one tick per vector; two-step = a harness of 3 chained ticks."""
    n = len(next(iter(vec.values())))
    if lay["mode"] == "direct":
        out = []
        for i in range(n):
            sets = " ".join(f"-set {k} {w}'b{int(vec[k][i]) & ((1 << w) - 1):0{w}b}" for k, w in lay["inputs"].items())
            out.append(f"eval {sets} -show q_n")
        return out
    h, w = lay["half"], lay["in_width"]
    out = []
    for v in vec[lay["in_name"]]:
        v = int(v) & ((1 << w) - 1)
        out.append(f"eval -set d_lo {h}'b{v & ((1 << h) - 1):0{h}b} -set d_hi {h}'b{v >> h:0{h}b} -show q3")
    return out


def _two_step_harness(op, lay):
    top, h, hw, ow = top_name(op), lay["half"], lay["in_width"] - lay["half"], lay["out_width"]
    return f"""module two_step_check (input wire [{h - 1}:0] d_lo, input wire [{h - 1}:0] d_hi, output wire [{ow - 1}:0] q3);
    wire [{h - 1}:0] lo0, lo1, lo2; wire [{hw - 1}:0] hi0, hi1, hi2; wire [{ow - 1}:0] q1, q2;
    assign lo0 = {h}'h{0x5A5A & ((1 << h) - 1):X}; assign hi0 = {hw}'h{0xC3C3 & ((1 << hw) - 1):X};
    {top}_step t1 (.d(d_lo), .sel(1'b0), .lo(lo0), .hi(hi0), .lo_n(lo1), .hi_n(hi1), .q_n(q1));
    {top}_step t2 (.d(d_hi), .sel(1'b1), .lo(lo1), .hi(hi1), .lo_n(lo2), .hi_n(hi2), .q_n(q2));
    {top}_step t3 (.d(d_hi), .sel(1'b1), .lo(lo2), .hi(hi2), .lo_n(), .hi_n(), .q_n(q3));
endmodule
"""


def check_chip_wrapper(op, text):
    """Structure of the registered top (ports = the chip pins, flip-flop bits as designed), then function: every vector
    through <op>_chip_step (and the chained two-step load) equals NumPy. Runs Yosys only (Windows yowasp works)."""
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_VERILOG_CHARS * 3 or BANNED.search(text):
        return {"status": "fail", "stage": "screen", "errors": ["wrapper text rejected"]}
    lay, top = chip_layout(op), top_name(op)
    with tempfile.TemporaryDirectory() as d:
        Path(d, "w.v").write_text(text, encoding="utf-8")
        r = _run(["-p", f"read_verilog -sv w.v; hierarchy -check -top {top}; proc; flatten; opt; check -assert; write_json f.json"], d)
        if r.returncode != 0:
            return {"status": "fail", "stage": "structure",
                    "errors": [l.strip() for l in (r.stdout + r.stderr).splitlines() if "ERROR" in l][:5] or ["yosys failed"]}
        mod = json.loads(Path(d, "f.json").read_text())["modules"][top]
        got = {n: (p["direction"], len(p["bits"])) for n, p in mod["ports"].items()}
        if lay["mode"] == "direct":
            want = {"clk": ("input", 1), **{n: ("input", w) for n, w in lay["inputs"].items()}, lay["out_name"]: ("output", lay["out_width"])}
            want_flops = lay["out_width"] + sum(lay["inputs"].values())
        else:
            want = {"clk": ("input", 1), "d": ("input", lay["half"]), "sel": ("input", 1), lay["out_name"]: ("output", lay["out_width"])}
            want_flops = lay["out_width"] + lay["in_width"]
        if got != want:
            return {"status": "fail", "stage": "structure", "errors": [f"wrapper ports must be {want}, got {got}"]}
        if sum(len(p["bits"]) for p in mod["ports"].values()) > chip_pins():
            return {"status": "fail", "stage": "structure", "errors": [f"more than {chip_pins()} pins"]}
        flops = sum(int(c["parameters"].get("WIDTH", "0"), 2) for c in mod["cells"].values()
                    if c["type"] in ("$dff", "$sdff", "$adff", "$aldff", "$dffe", "$sdffe", "$sdffce"))
        if flops != want_flops:
            return {"status": "fail", "stage": "structure", "errors": [f"expected exactly {want_flops} flip-flop bits, found {flops}"]}
        vec = vectors_for(op)
        n = len(next(iter(vec.values())))
        if lay["mode"] == "direct":
            head = ["read_verilog -sv w.v", f"hierarchy -top {top}_step", "flatten", "proc", "opt"]
            show = "q_n"
        else:
            Path(d, "h.v").write_text(_two_step_harness(op, lay), encoding="utf-8")
            head = ["read_verilog -sv w.v h.v", "hierarchy -top two_step_check", "flatten", "proc", "opt"]
            show = "q3"
        Path(d, "run.ys").write_text("\n".join(head + _eval_lines(op, lay, vec)) + "\n")
        r = _run(["-s", "run.ys"], d)
        found = re.findall(r"Eval result: \\" + show + r" = (\d+)'([01xXzZ]+)", r.stdout)
        if r.returncode != 0 or len(found) != n:
            return {"status": "fail", "stage": "function", "errors": [f"could not evaluate all vectors ({len(found)}/{n})"]}
        got_vals = [None if set(b) - {"0", "1"} else _signed(int(b, 2), lay["out_width"]) for _, b in found]
        bad = compare(op, vec, got_vals)
        if bad:
            return {"status": "fail", "stage": "function", "errors": [f"{len(bad)}/{n} vectors differ from NumPy through the chip pins"],
                    "mismatches": bad[:4], "vectors": n}
        return {"status": "pass", "stage": "function", "errors": [], "vectors": n, "flop_bits": flops, "pins": lay["pins"]}


def compare(op, vec, got):
    gold = OPS[op]["golden"](vec)
    bad = []
    for i, g in enumerate(got):
        if g is None or g != int(gold[i]):
            bad.append({"index": i, "inputs": {k: int(vec[k][i]) for k in vec}, "expected": int(gold[i]), "got": g})
    return bad


# ---------------------------------------------------------------- netlist simulation
def netlist_tb(op):
    """Testbench driving <op>_chip through its pins only. Prints 'VEC i <result bits>' per vector and 'DONE n'."""
    lay, top, vec = chip_layout(op), top_name(op), vectors_for(op)
    n = len(next(iter(vec.values())))
    on, ow = lay["out_name"], lay["out_width"]
    if lay["mode"] == "direct":
        regs = "\n".join(f"    reg [{w - 1}:0] {k} = 0; reg [{w - 1}:0] m_{k} [0:{n - 1}];" for k, w in lay["inputs"].items())
        conn = ", ".join(f".{k}({k})" for k in lay["inputs"])
        load = "\n".join(f"        m_{k}[{i}] = {w}'h{int(vec[k][i]) & ((1 << w) - 1):X};"
                         for i in range(n) for k, w in lay["inputs"].items())
        drive = " ".join(f"{k} = m_{k}[i];" for k in lay["inputs"])
        body = f"""            @(negedge clk) {drive}
            @(negedge clk) ;
            @(negedge clk) $fdisplay(f, "VEC %0d %b", i, {on});"""
    else:
        h, w = lay["half"], lay["in_width"]
        regs = f"    reg [{h - 1}:0] d = 0; reg sel = 0; reg [{w - 1}:0] m_v [0:{n - 1}];"
        conn = ".d(d), .sel(sel)"
        load = "\n".join(f"        m_v[{i}] = {w}'h{int(v) & ((1 << w) - 1):X};" for i, v in enumerate(vec[lay["in_name"]]))
        body = f"""            @(negedge clk) begin d = m_v[i][{h - 1}:0]; sel = 0; end
            @(negedge clk) begin d = m_v[i][{w - 1}:{h}]; sel = 1; end
            @(negedge clk) ;
            @(negedge clk) $fdisplay(f, "VEC %0d %b", i, {on});"""
    return f"""`timescale 1ns/1ps
// x2c chip chain: drives the synthesized netlist of {top} through its pins only (generated, do not edit).
module chip_chain_tb;
    reg clk = 0;
    always #41.667 clk = ~clk;                 // 12 MHz, the repo's nextpnr target (timing is NOT simulated: zero-delay cells)
{regs}
    wire [{ow - 1}:0] {on};
    {top} dut (.clk(clk), {conn}, .{on}({on}));
    integer i, f;
    reg [8*256-1:0] logname;
    initial begin
{load}
        if (!$value$plusargs("LOG=%s", logname)) logname = "sim.log";
        f = $fopen(logname, "w");
        repeat (2) @(negedge clk);
        for (i = 0; i < {n}; i = i + 1) begin
{body}
        end
        $fdisplay(f, "DONE %0d", {n});
        $fclose(f);
        $finish;
    end
endmodule
"""


def parse_netlist_log(op, text):
    lay, vec = chip_layout(op), vectors_for(op)
    n = len(next(iter(vec.values())))
    got = [None] * n
    for m in re.finditer(r"^VEC (\d+) ([01xXzZ]+)\s*$", text or "", re.M):
        i, bits = int(m.group(1)), m.group(2)
        if i < n:
            got[i] = None if set(bits) - {"0", "1"} else _signed(int(bits, 2), lay["out_width"])
    done = re.search(r"^DONE (\d+)\s*$", text or "", re.M)
    if not done:
        return {"status": "unverified", "reason": "the simulation did not finish (no DONE line)", "vectors": n}
    bad = compare(op, vec, got)
    if bad:
        und = sum(1 for b in bad if b["got"] is None)
        return {"status": "fail", "reason": f"{len(bad)}/{n} vectors differ from NumPy ({und} undefined)", "vectors": n, "mismatches": bad[:4],
                "all_mismatches": bad}
    return {"status": "pass", "reason": f"synthesized netlist == NumPy on all {n} vectors", "vectors": n}


MACC_SIGNED_NOTE = ("signed multiply on the hard multipliers: the repo's MACC_8x8_20_map.v is unsigned-only, so signed multiplies "
                    "wider than 8 bits are wrong for negative operands. chip_flow normally replaces it with "
                    "chip/techmap/MACC_8x8_20_map_signed.v; check the build's Makefile and flow_report.json multiplier_map")


def explain_mismatch(op, res, netlist_text):
    """A rule, not an LLM: the netlist uses hard multipliers and every wrong vector has a negative input -> MACC_SIGNED_NOTE."""
    if res.get("status") != "fail":
        return None
    n_macc = len(re.findall(r"^\s*MACC_8x8_20\s*#", netlist_text or "", re.M))
    bad = res.get("all_mismatches") or res.get("mismatches") or []
    if n_macc and bad and all(any(v < 0 for v in b["inputs"].values()) for b in bad):
        return f"netlist uses {n_macc} MACC_8x8_20 block(s) and every wrong vector has a negative input: {MACC_SIGNED_NOTE}"
    return None


def run_netlist_sim(op, repo=None):
    """In Ubuntu (called by the sim stage script): netlist JSON -> Verilog, Icarus with the repo's primitive models and
    the chip-polarity IOBUF, run, write sim.log in the job folder. Returns the parse result."""
    import subprocess
    from x2c.chip.chip_repo import ChipRepo, INPUT_FILES
    repo = repo or ChipRepo()
    before = repo.tree_fingerprint()
    job, top = JOBS / op, top_name(op)
    netjson = CHIP_BUILD / f"chain_{op}" / f"{top}.json"
    if not netjson.is_file():
        raise SystemExit(f"no synthesized netlist {netjson}: run the flow stage first")
    r = subprocess.run(["yosys", "-q", "-p", f"read_json {netjson}; write_verilog -noattr {job / 'netlist.v'}"],
                       capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("yosys could not write the netlist:\n" + r.stderr[-1500:])
    tb = job / f"{top}_tb.v"
    srcs = [repo.path(INPUT_FILES["x2_behavioral"]), ROOT / "hardware" / "sim" / "RERAM_CIM_32x32_sim.v", SIM_SRC,
            *[repo.path(p) for p in PRIMS], job / "netlist.v", tb]
    r = subprocess.run(["iverilog", "-g2012", "-s", "chip_chain_tb", "-o", str(job / "netlist.vvp"), *map(str, srcs)],
                       capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("iverilog failed:\n" + r.stderr[-1500:])
    log = job / "sim.log"
    log.unlink(missing_ok=True)
    subprocess.run(["vvp", "-n", str(job / "netlist.vvp"), f"+LOG={log}"], capture_output=True, text=True, timeout=1200)
    res = parse_netlist_log(op, log.read_text() if log.is_file() else "")
    res["chip_repo_read_only"] = before == repo.tree_fingerprint()
    print(f"NETLIST SIM {op}: {res['status'].upper()} - {res['reason']} | chip repo read-only: {res['chip_repo_read_only']}")
    return res


# ---------------------------------------------------------------- flow report and timing
_FMAX = re.compile(r"Max frequency for clock '(.+?)':\s+([\d.]+) MHz \((PASS|FAIL) at ([\d.]+) MHz\)")


def final_timing(pnr_log_text):
    """nextpnr prints Max frequency several times; the last line per clock is after routing."""
    out = {}
    for clock, f, verdict, target in _FMAX.findall(pnr_log_text or ""):
        out[clock] = {"fmax_mhz": float(f), "target_mhz": float(target), "ok": verdict == "PASS"}
    return out


def judge(report, timing, sim):
    """Timing is checked against the PROJECT clock (chip/x2_facts.py clock_mhz), not nextpnr's built-in 12 MHz target."""
    from x2c.chip import x2_facts
    clock = x2_facts.value("clock_mhz")
    timing = {c: dict(t, target_mhz=clock, ok=t["fmax_mhz"] >= clock) for c, t in (timing or {}).items()}
    """check stage: (status, detail, facts) from the flow report, timing and the netlist simulation."""
    facts = {"utilisation": report.get("utilisation"), "pins": report.get("pins"), "bitstream_audit": report.get("bitstream_audit"),
             "bitstream_bytes": report.get("bitstream_bytes"), "fasm_features": report.get("fasm_features"), "fmax": timing,
             "netlist_simulation": sim, "not_checked": list(ALWAYS_NOT_CHECKED)}
    if report.get("status") != "ok":
        cause = (report.get("failure") or {}).get("cause", report.get("status"))
        return "fail", f"chip flow status {report.get('status')} ({cause})", facts
    if not report.get("chip_repo_read_only", False):
        return "fail", "the chip repo changed during the flow", facts
    audit = report.get("bitstream_audit") or {}
    detail = (f"flow ok | V0 audit {audit.get('set_bits', '?')}/{audit.get('required_bits', '?')} | "
              f"netlist sim {str((sim or {}).get('status', 'not run')).upper()}")
    if not sim or sim.get("status") != "pass":
        return ("fail" if (sim or {}).get("status") == "fail" else "unverified"), detail, facts
    if not timing:
        facts["not_checked"].insert(0, "timing (nextpnr found no register-to-register path to time)")
        return "pass", detail + " | timing: nextpnr found no register-to-register path", facts
    late = {c: t for c, t in timing.items() if not t["ok"]}
    if late:
        c, t = next(iter(late.items()))
        return "unverified", detail + f" | timing NOT met: {t['fmax_mhz']} MHz < {t['target_mhz']} MHz (nextpnr model)", facts
    return "pass", detail + " | timing met (" + ", ".join(f"{t['fmax_mhz']} MHz" for t in timing.values()) + ", nextpnr model)", facts


# ---------------------------------------------------------------- stage functions
def downstream_files(op, from_stage):
    i = fc.STAGES.index(from_stage)
    names = list(FLOW_FILES) if i <= fc.STAGES.index("flow") else []
    names += list(SIM_FILES) + ["sim.log"] if i <= fc.STAGES.index("sim") else []
    names += list(CHECK_FILES) if i <= fc.STAGES.index("check") else []
    return names


def clean_job(op, from_stage):
    """Delete results this run regenerates. A previous flow report is overwritten with status 'cleared' (not deleted:
    chip_flow only replaces folders that hold its report), so an old 'ok' can never be read as this run's."""
    removed = []
    for name in downstream_files(op, from_stage):
        p = JOBS / op / name
        if p.exists():
            p.unlink()
            removed.append(name)
    rep = CHIP_BUILD / f"chain_{op}" / "flow_report.json"
    if fc.STAGES.index(from_stage) <= fc.STAGES.index("flow") and rep.is_file():
        rep.write_text(json.dumps({"design": f"chain_{op}", "status": "cleared"}) + "\n")
        removed.append(rep.as_posix() + " (status cleared)")
    return removed


def build_chip_steps(op, cfg, exec_fn, preflight, rtl_step, say=print):
    job = JOBS / op
    root_posix = fc.to_wsl_path(str(ROOT)) if cfg.runner == "wsl" else ROOT.as_posix()

    def ubuntu(script_name, args, timeout):
        sp = fc.script_path(cfg.runner, str((job / script_name).resolve()))
        return exec_fn(fc.build_command(cfg.runner, sp, args, cfg.distro), timeout, cfg.heartbeat_s, say)

    def prep():
        core = Path("hardware/operators") / f"{op}.v"
        if not core.is_file():
            return "fail", f"no verified core {core.as_posix()} (the rtl stage writes it)"
        job.mkdir(parents=True, exist_ok=True)
        text = make_chip_wrapper(op, core.read_text(encoding="utf-8"))
        res = check_chip_wrapper(op, text)
        if res["status"] != "pass":
            return "fail", f"wrapper check {res['stage']}: {res['errors'][0]}", {"wrapper_check": res}
        (job / f"{top_name(op)}.v").write_text(text, encoding="utf-8", newline="\n")
        (job / f"{top_name(op)}_tb.v").write_text(netlist_tb(op), encoding="utf-8", newline="\n")
        lay = chip_layout(op)
        (job / "run_chip_step.sh").write_text(STEP_SCRIPT.format(
            root=root_posix, job=job.as_posix(), name=f"chain_{op}", top=top_name(op), wrapper=f"{top_name(op)}.v", op=op),
            encoding="utf-8", newline="\n")
        return "pass", (f"wrapper check (chip pins, {lay['mode']}): {res['vectors']} vectors through {lay['pins']}/{chip_pins()} "
                        f"pins == NumPy, {res['flop_bits']} flip-flop bits"), {"wrapper_check": res}

    def env():
        if "info" not in preflight:
            JOBS.mkdir(parents=True, exist_ok=True)
            pf = JOBS / "_chip_preflight.sh"
            pf.write_text(PREFLIGHT_SCRIPT, encoding="utf-8", newline="\n")
            sp = fc.script_path(cfg.runner, str(pf.resolve()))
            run = exec_fn(fc.build_command(cfg.runner, sp, [root_posix], cfg.distro), 180, cfg.heartbeat_s, say)
            preflight["run"], preflight["info"] = run, fc.parse_preflight(run["output"])
        info = preflight["info"]
        if not info["ok"]:
            run = preflight["run"]
            if not run.get("started", True) or not info["reached_end"]:
                return "blocked", fc.explain_preflight(run, info, cfg.runner, cfg.distro)
            return "blocked", "Ubuntu is missing: " + ", ".join(info["missing"])
        short = fc.short_versions(info["versions"])
        return "pass", (f"Ubuntu has the chip toolchain (Yosys {short.get('yosys', '?')}, nextpnr {short.get('nextpnr-generic', '?')}, "
                        f"Icarus {short.get('iverilog', '?')}), the chip Python (fasm), the chip repo and profile"), {"tool_versions": info["versions"]}

    def flow():
        if not (job / "run_chip_step.sh").is_file():
            return "fail", "no run_chip_step.sh in the job folder: run from --from-stage prep"
        say(f"  ... flow: chip flow in Ubuntu (Yosys, nextpnr, bitstream, V0 audit), about a minute; log {job.as_posix()}/flow_console.log")
        t0 = time.time()
        run = ubuntu("run_chip_step.sh", ["flow"], cfg.timeout_s)
        if not run["started"]:
            return "blocked", run["output"]
        if run["timed_out"]:
            return "fail", f"timed out after {cfg.timeout_s} s"
        rep_path = CHIP_BUILD / f"chain_{op}" / "flow_report.json"
        try:
            rep = json.loads(rep_path.read_text())
        except (OSError, ValueError):
            return "fail", f"no flow report from this run ({rep_path.as_posix()}); see {job.as_posix()}/flow_console.log"
        if rep.get("status") in ("cleared", "started") or rep_path.stat().st_mtime < t0 - 5:
            return "fail", f"the chip flow did not finish this run (report status {rep.get('status')}); see flow_console.log"
        if rep["status"] != "ok":
            f = rep.get("failure") or {}
            log = CHIP_BUILD / f"chain_{op}" / f"{f.get('stage')}.log"
            tail = ""
            if log.is_file():
                lines = [l.strip() for l in log.read_text(errors="replace").splitlines() if l.strip()]
                errs = [l for l in lines if re.search(r"error|no module named", l, re.I)]
                tail = (errs or lines)[-1][:200] if lines else ""
            return "fail", (f"chip flow failed at {f.get('stage')}: {f.get('cause')}" + (f" | {tail}" if tail else "") +
                            f" | log {log.as_posix()}")
        u = (rep.get("utilisation") or {}).get("FABULOUS_LC", {})
        return "pass", (f"chip flow ok: {rep.get('bitstream_bytes')}-byte bitstream, FABULOUS_LC {u.get('used')}/{u.get('available')}, "
                        f"pins {(rep.get('pins') or {}).get('needed')}/{(rep.get('pins') or {}).get('available')}, {run['seconds']} s")

    def sim():
        say("  ... sim: synthesized netlist in Icarus (repo primitive models), usually seconds")
        run = ubuntu("run_chip_step.sh", ["sim"], cfg.timeout_s)
        if not run["started"]:
            return "blocked", run["output"]
        if run["timed_out"]:
            return "fail", f"timed out after {cfg.timeout_s} s"
        log = job / "sim.log"
        if not log.is_file():
            console = (job / "sim_console.log")
            tail = console.read_text(errors="replace")[-300:] if console.is_file() else run["output"][-300:]
            return "fail", "the netlist simulation wrote no log: " + " ".join(tail.split())
        res = parse_netlist_log(op, log.read_text(errors="replace"))
        net = job / "netlist.v"
        why = explain_mismatch(op, res, net.read_text(errors="replace") if net.is_file() else "")
        res.pop("all_mismatches", None)
        if why:
            res["likely_cause"] = why
        return res["status"], res["reason"] + (f" | likely cause: {why}" if why else ""), {"netlist_simulation": res}

    def check():
        (job / "chip_check.json").unlink(missing_ok=True)
        build = CHIP_BUILD / f"chain_{op}"
        try:
            rep = json.loads((build / "flow_report.json").read_text())
        except (OSError, ValueError):
            return "fail", "no flow report to check"
        pnr = build / "pnr.log"
        timing = final_timing(pnr.read_text(errors="replace")) if pnr.is_file() else {}
        log = job / "sim.log"
        sim_res = parse_netlist_log(op, log.read_text(errors="replace")) if log.is_file() else None
        if sim_res:
            sim_res.pop("all_mismatches", None)
        status, detail, facts = judge(rep, timing, sim_res)
        (job / "chip_check.json").write_text(json.dumps({"op": op, "status": status, "detail": detail, "facts": facts,
                                                          "evidence_level": EVIDENCE_LEVEL}, indent=2, default=str) + "\n")
        return status, detail, facts

    return {"rtl": rtl_step, "prep": prep, "env": env, "flow": flow, "sim": sim, "check": check}


def format_summary(result):
    f = result["facts"]
    lines = [f"=== CHIP CHAIN RESULT {result['op']}: {result['verdict']} ===",
             "  " + " | ".join(f"{r['stage']} {fc.LABEL[r['status']]}" for r in result["stages"])]
    if f.get("bitstream_audit"):
        a = f["bitstream_audit"]
        lines.append(f"  bitstream {f.get('bitstream_bytes')} bytes | V0 audit {a.get('set_bits')}/{a.get('required_bits')} | "
                     f"FASM features {f.get('fasm_features')}")
    for clock, t in (f.get("fmax") or {}).items():
        lines.append(f"  timing (nextpnr's model of the chip fabric, NOT measured): {t['fmax_mhz']} MHz, target {t['target_mhz']} MHz: "
                     f"{'met' if t['ok'] else 'NOT met'}")
    if result["verdict"].startswith("PASS"):
        lines.append("  shows: the netlist the chip flow placed and turned into an audited bitstream computes NumPy's result on every vector")
    lines.append("  not checked (so not claimed): " + ", ".join(f.get("not_checked") or ALWAYS_NOT_CHECKED))
    return lines


def plan_text(op, cfg):
    job = JOBS / op
    try:
        cmd = " ".join(fc.build_command(cfg.runner, fc.script_path(cfg.runner, str((job / "run_chip_step.sh").resolve())), ["<stage>"], cfg.distro))
    except ValueError as e:
        cmd = f"CANNOT RUN: {e}"
    lines = [f"chip plan for {op} (runner {cfg.runner}, starting at '{cfg.from_stage}'); nothing is run:"]
    for s in fc.STAGES:
        mark = "skip" if fc.STAGES.index(s) < fc.STAGES.index(cfg.from_stage) else "run "
        lines.append(f"  {mark} {s}" + (f"   $ {cmd.replace('<stage>', s)}" if s in ("flow", "sim") else ""))
    lines.append(f"  first cleared: {', '.join(downstream_files(op, cfg.from_stage))} in {job.as_posix()}; "
                 f"chip_build/chain_{op}/flow_report.json status set to 'cleared'")
    return lines


if __name__ == "__main__":          # Ubuntu side of the sim stage:  python -m x2c.verification.chip_chain sim <op>
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "sim":
        r = run_netlist_sim(sys.argv[2])
        sys.exit(0 if r["status"] == "pass" and r["chip_repo_read_only"] else 1)
    sys.exit("usage: python -m x2c.verification.chip_chain sim <op>")
