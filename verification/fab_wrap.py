"""
Wrap an A2-verified operator in the I/O interface the FABulous demo fabric expects, and check the wrapper deterministically.

The demo design (sequential_16bit_en.v, real) has ports  clk, io_in[27:0], io_out[27:0], io_oeb[27:0].
Our wrapper maps operator inputs onto io_in (first operand on the lowest pins), puts the result on io_out just above the inputs,
and registers the result on clk (the fabric flow is timed against a clock). io_oeb is 1 on input pins (output disabled), 0 on result pins.

ORDER MATTERS (found on a real FABulous run): FABulous' gen_user_design_wrapper reads the ports of the FIRST module in the file to decide how to
connect io_in/io_out/io_oeb. If the first module is the core (ports a, b, y) it prints "Custom design detected", connects nothing, and Yosys then
deletes all logic (0 logic cells, empty FASM, all-zero bitstream). So the wrapper module comes first and the core after it.

Test trick: the same source has `ifdef FAB_COMB`. Built with -D FAB_COMB the wrapper is purely combinational, so Yosys `eval` can check
the whole pin mapping against the NumPy golden model. Without the define it is the registered version that goes to FABulous.

ASSUMED: which pins are inputs/outputs and the io_oeb polarity (copied from the demo's style); the physical pin meaning is not confirmed.
TWO-STEP LOAD: an operator with ONE input wider than the pins (requant_i32_i8: 32 input bits + 8 result bits = 40 > 28) is loaded in two clock
ticks over the same 16 data pins: io_in[16] = 0 stores io_in[15:0] as the LOW half, io_in[16] = 1 stores it as the HIGH half; the result
appears on io_out[24:17] after the next tick. ML analogy: feeding a tensor in two batches because the pipe is too narrow.
The logic of one clock tick is a separate combinational module (<op>_fab_step), so the load sequence can be checked with Yosys `eval`
by chaining four copies of it; the top module only adds the registers. This protocol (which pin selects, the order) is ASSUMED: ours, not BM's.
Operators with several inputs that do not fit still cannot be wrapped.
Evidence level: Yosys RTL evaluation vs NumPy golden (NOT nextpnr, NOT silicon).
"""
import json
import re
import tempfile
from pathlib import Path

from verification.rtl_check import BANNED, EVIDENCE_LEVEL, MAX_VERILOG_CHARS, OPS, _run, _signed, vectors_for

IO_WIDTH = 28
FLOP_TYPES = ("$dff", "$sdff", "$adff", "$aldff", "$dffe")      # `opt` may fold a mux into a flop (e.g. $sdff), so count them all


HALF = 16                 # data pins of a two-step load: io_in[15:0]
SEL_PIN = HALF            # io_in[16]: 0 = this is the low half, 1 = this is the high half


def wrapper_layout(op):
    """mode 'direct': every input has its own pins. mode 'two_step': one wide input is loaded in two clock ticks (see the module text)."""
    s = OPS[op]
    offs, pos = {}, 0
    for name, w in s["inputs"].items():
        offs[name] = (pos, w)
        pos += w
    out_name, out_w = s["output"]
    if pos + out_w <= IO_WIDTH:
        return {"mode": "direct", "inputs": offs, "in_bits": pos, "out_lsb": pos, "out_width": out_w, "out_name": out_name}
    if len(s["inputs"]) == 1 and HALF < pos <= 2 * HALF and HALF + 1 + out_w <= IO_WIDTH:
        return {"mode": "two_step", "inputs": offs, "in_bits": HALF + 1, "out_lsb": HALF + 1, "out_width": out_w, "out_name": out_name,
                "in_name": next(iter(offs)), "in_width": pos, "half": HALF, "sel_pin": SEL_PIN}
    raise ValueError(f"{op} needs {pos} input pins + {out_w} result pins = {pos + out_w}, but the fabric has {IO_WIDTH} I/O pins")


def split_halves(value, lay):
    """The two io_in words of a two-step load for one input value: (low half with select 0, high half with select 1)."""
    v = int(value) & ((1 << lay["in_width"]) - 1)
    return v & ((1 << lay["half"]) - 1), (1 << lay["sel_pin"]) | (v >> lay["half"])


def _two_step_wrapper(op, core_verilog, lay):
    s, top, ow, on, w, h = OPS[op], top_name(op), lay["out_width"], lay["out_name"], lay["in_width"], lay["half"]
    hw = w - h
    hi_pad = IO_WIDTH - lay["out_lsb"] - ow
    parts = ([f"{hi_pad}'b0"] if hi_pad else []) + ["q"] + [f"{lay['out_lsb']}'b0"]
    oeb = "0" * (IO_WIDTH - lay["in_bits"]) + "1" * lay["in_bits"]
    return (f"""`default_nettype none
// Two-step load (ASSUMED protocol): io_in[{lay['sel_pin']}]=0 stores io_in[{h - 1}:0] as the low half, =1 stores io_in[{hw - 1}:0] as the high half.
module {top} (
    input wire clk,
    input wire [{IO_WIDTH - 1}:0] io_in,
    output wire [{IO_WIDTH - 1}:0] io_out,
    output wire [{IO_WIDTH - 1}:0] io_oeb
);
    reg [{h - 1}:0] lo_r;
    reg [{hw - 1}:0] hi_r;
    reg [{ow - 1}:0] q_r;
    wire [{h - 1}:0] lo_n;
    wire [{hw - 1}:0] hi_n;
    wire [{ow - 1}:0] q_n;
    {top}_step step (.io_in(io_in), .lo(lo_r), .hi(hi_r), .q(q_r), .lo_n(lo_n), .hi_n(hi_n), .q_n(q_n), .io_out(io_out), .io_oeb(io_oeb));
    always @(posedge clk) begin
        lo_r <= lo_n;
        hi_r <= hi_n;
        q_r <= q_n;
    end
endmodule

// Everything that happens in ONE clock tick, without registers (so it can be checked with Yosys eval).
module {top}_step (
    input wire [{IO_WIDTH - 1}:0] io_in,
    input wire [{h - 1}:0] lo,
    input wire [{hw - 1}:0] hi,
    input wire [{ow - 1}:0] q,
    output wire [{h - 1}:0] lo_n,
    output wire [{hw - 1}:0] hi_n,
    output wire [{ow - 1}:0] q_n,
    output wire [{IO_WIDTH - 1}:0] io_out,
    output wire [{IO_WIDTH - 1}:0] io_oeb
);
    assign lo_n = io_in[{lay['sel_pin']}] ? lo : io_in[{h - 1}:0];
    assign hi_n = io_in[{lay['sel_pin']}] ? io_in[{hw - 1}:0] : hi;
    wire signed [{w - 1}:0] {lay['in_name']} = {{hi, lo}};
    wire signed [{ow - 1}:0] {on};
    {s['top']} core (.{lay['in_name']}({lay['in_name']}), .{on}({on}));
    assign q_n = {on};
    assign io_out = {{{', '.join(parts)}}};
    assign io_oeb = {IO_WIDTH}'b{oeb};
endmodule
`resetall

""" + core_verilog.strip() + "\n")


def _check_two_step(op, text, lay, d, top):
    """Function check of a two-step wrapper with Yosys eval: four copies of the one-tick module are chained (tick 1 stores the low half,
    tick 2 the high half, tick 3 registers the result, tick 4 shows it on the pins), starting from a NON-zero state, so a result that
    depends on what was loaded before would be caught."""
    h, hw, ow = lay["half"], lay["in_width"] - lay["half"], lay["out_width"]
    st = lambda i: f".lo(lo{i}), .hi(hi{i}), .q(q{i})"                                    # noqa: E731
    nx = lambda i: f".lo_n(lo{i}), .hi_n(hi{i}), .q_n(q{i})"                              # noqa: E731
    wires = "\n".join(f"    wire [{h - 1}:0] lo{i}; wire [{hw - 1}:0] hi{i}; wire [{ow - 1}:0] q{i};" for i in range(4))
    harness = f"""module two_step_check (input wire [{IO_WIDTH - 1}:0] w_lo, input wire [{IO_WIDTH - 1}:0] w_hi, output wire [{IO_WIDTH - 1}:0] io_out, output wire [{IO_WIDTH - 1}:0] io_oeb);
{wires}
    assign lo0 = {h}'h{0x5A5A & ((1 << h) - 1):X}; assign hi0 = {hw}'h{0xC3C3 & ((1 << hw) - 1):X}; assign q0 = {ow}'h{0x96 & ((1 << ow) - 1):X};
    {top}_step t1 (.io_in(w_lo), {st(0)}, {nx(1)}, .io_out(), .io_oeb());
    {top}_step t2 (.io_in(w_hi), {st(1)}, {nx(2)}, .io_out(), .io_oeb());
    {top}_step t3 (.io_in(w_hi), {st(2)}, {nx(3)}, .io_out(), .io_oeb());
    {top}_step t4 (.io_in(w_hi), {st(3)}, .lo_n(), .hi_n(), .q_n(), .io_out(io_out), .io_oeb(io_oeb));
endmodule
"""
    Path(d, "h.v").write_text(harness, encoding="utf-8")
    vec = vectors_for(op)
    vals = vec[lay["in_name"]]
    lines = ["read_verilog -sv w.v h.v", "hierarchy -top two_step_check", "flatten", "proc", "opt"]
    for v in vals:
        lo, hi = split_halves(v, lay)
        lines.append(f"eval -set w_lo {IO_WIDTH}'b{lo:0{IO_WIDTH}b} -set w_hi {IO_WIDTH}'b{hi:0{IO_WIDTH}b} -show io_out -show io_oeb")
    Path(d, "run.ys").write_text("\n".join(lines) + "\n")
    r = _run(["-s", "run.ys"], d)
    outs = re.findall(r"Eval result: \\io_out = (\d+)'([01xXzZ]+)", r.stdout)
    oebs = re.findall(r"Eval result: \\io_oeb = (\d+)'([01xXzZ]+)", r.stdout)
    n = len(vals)
    if r.returncode != 0 or len(outs) != n or len(oebs) != n:
        return {"status": "fail", "stage": "function", "errors": [f"could not evaluate all vectors ({len(outs)}/{n})"]}
    want_oeb = "0" * (IO_WIDTH - lay["in_bits"]) + "1" * lay["in_bits"]
    if any(b != want_oeb for _, b in oebs):
        return {"status": "fail", "stage": "function", "errors": [f"io_oeb must be {want_oeb} (1 = input pin)"]}
    gold, bad = OPS[op]["golden"](vec), []
    for i, (_, bits) in enumerate(outs):
        if set(bits) - {"0", "1"}:
            bad.append({"expected": int(gold[i]), "got": f"undefined ({bits})"}); continue
        v = int(bits, 2)
        got = _signed((v >> lay["out_lsb"]) & ((1 << ow) - 1), ow)
        other = v & ((1 << lay["out_lsb"]) - 1) or v >> (lay["out_lsb"] + ow)
        if got != int(gold[i]) or other:
            bad.append({"inputs": {lay["in_name"]: int(vals[i])}, "expected": int(gold[i]), "got": got, "stray_bits": bool(other)})
    if bad:
        return {"status": "fail", "stage": "function", "errors": [f"{len(bad)}/{n} vectors differ from the golden model through the two-step load"],
                "mismatches": bad[:4], "vectors": n}
    return None


def top_name(op):
    return f"{op}_fab"


def make_fab_wrapper(op, core_verilog):
    """Full Verilog text for FABulous: the verified core module followed by the wrapper top module."""
    lay = wrapper_layout(op)
    if lay["mode"] == "two_step":
        return _two_step_wrapper(op, core_verilog, lay)
    s = OPS[op]
    decl, conn = [], []
    for name, (lo, w) in lay["inputs"].items():
        decl.append(f"    wire signed [{w - 1}:0] {name} = io_in[{lo + w - 1}:{lo}];")
        conn.append(f".{name}({name})")
    hi_pad = IO_WIDTH - lay["in_bits"] - lay["out_width"]
    parts = ([f"{hi_pad}'b0"] if hi_pad else []) + ["q"] + ([f"{lay['in_bits']}'b0"] if lay["in_bits"] else [])
    oeb = "0" * (IO_WIDTH - lay["in_bits"]) + "1" * lay["in_bits"]
    ow, on = lay["out_width"], lay["out_name"]
    return (f"""`default_nettype none
module {top_name(op)} (
    input wire clk,
    input wire [{IO_WIDTH - 1}:0] io_in,
    output wire [{IO_WIDTH - 1}:0] io_out,
    output wire [{IO_WIDTH - 1}:0] io_oeb
);
{chr(10).join(decl)}
    wire signed [{ow - 1}:0] {on};
    {s['top']} core ({', '.join(conn + [f'.{on}({on})'])});
    wire [{ow - 1}:0] q;
`ifdef FAB_COMB
    assign q = {on};
`else
    reg [{ow - 1}:0] q_r;
    always @(posedge clk) q_r <= {on};
    assign q = q_r;
`endif
    assign io_out = {{{', '.join(parts)}}};
    assign io_oeb = {IO_WIDTH}'b{oeb};
endmodule
`resetall

""" + core_verilog.strip() + "\n")


def check_fab_wrapper(op, text):
    """Deterministic verdict on a wrapper: structure of the registered build, then function of the combinational build."""
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_VERILOG_CHARS * 2 or BANNED.search(text):
        return {"status": "fail", "stage": "screen", "errors": ["wrapper text rejected"]}
    top = top_name(op)
    lay = wrapper_layout(op)
    first = re.search(r"^\s*module\s+(\w+)", re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S), re.M)
    if not first or first.group(1) != top:
        return {"status": "fail", "stage": "structure",
                "errors": [f"the FIRST module in the file must be '{top}' (FABulous connects pins from the first module's ports); found '{first.group(1) if first else None}'"]}
    with tempfile.TemporaryDirectory() as d:
        Path(d, "w.v").write_text(text, encoding="utf-8")
        r = _run(["-p", f"read_verilog -sv w.v; hierarchy -check -top {top}; proc; opt_clean; check -assert; write_json w.json"], d)
        if r.returncode != 0:
            return {"status": "fail", "stage": "structure", "errors": [l.strip() for l in (r.stdout + r.stderr).splitlines() if "ERROR" in l][:5] or ["yosys failed"]}
        mod = json.loads(Path(d, "w.json").read_text())["modules"][top]
        want = {"clk": ("input", 1), "io_in": ("input", IO_WIDTH), "io_out": ("output", IO_WIDTH), "io_oeb": ("output", IO_WIDTH)}
        got = {n: (p["direction"], len(p["bits"])) for n, p in mod["ports"].items()}
        if got != want:
            return {"status": "fail", "stage": "structure", "errors": [f"wrapper ports must be {want}, got {got}"]}
        # the registered build keeps the core combinational: the only state is the result register
        r2 = _run(["-p", f"read_verilog -sv w.v; hierarchy -top {top}; flatten; proc; opt; stat; write_json f.json"], d)
        fm = json.loads(Path(d, "f.json").read_text())["modules"][top]
        flops = sum(int(c["parameters"].get("WIDTH", "0"), 2) for c in fm["cells"].values() if c["type"].startswith(FLOP_TYPES))
        want_flops = lay["out_width"] + (lay["in_width"] if lay["mode"] == "two_step" else 0)       # two-step: the stored input halves + the result
        if flops != want_flops:
            return {"status": "fail", "stage": "structure", "errors": [f"expected exactly {want_flops} flip-flop bits, found {flops}"]}
        if lay["mode"] == "two_step":
            step = json.loads(Path(d, "w.json").read_text())["modules"].get(f"{top}_step")
            if step is None or any(c["type"].startswith(FLOP_TYPES) or "latch" in c["type"] for c in step["cells"].values()):
                return {"status": "fail", "stage": "structure", "errors": [f"{top}_step must exist and hold no registers (it is the logic of one clock tick)"]}
            bad = _check_two_step(op, text, lay, d, top)
            return bad or {"status": "pass", "stage": "function", "errors": [], "vectors": len(vectors_for(op)[lay["in_name"]]), "flop_bits": flops,
                           "evidence_level": EVIDENCE_LEVEL}
        vec = vectors_for(op)
        n = len(next(iter(vec.values())))
        lines = [f"read_verilog -sv -D FAB_COMB w.v", f"hierarchy -top {top}", "flatten", "proc", "opt"]
        for i in range(n):
            word = 0
            for name, (lo, w) in lay["inputs"].items():
                word |= (int(vec[name][i]) & ((1 << w) - 1)) << lo
            lines.append(f"eval -set io_in {IO_WIDTH}'b{word:0{IO_WIDTH}b} -show io_out")
        Path(d, "run.ys").write_text("\n".join(lines) + "\n")
        r3 = _run(["-s", "run.ys"], d)
        found = re.findall(r"Eval result: \\io_out = (\d+)'([01xXzZ]+)", r3.stdout)
        if r3.returncode != 0 or len(found) != n:
            return {"status": "fail", "stage": "function", "errors": [f"could not evaluate all vectors ({len(found)}/{n})"]}
        gold = OPS[op]["golden"](vec)
        bad = []
        for i, (_, bits) in enumerate(found):
            if set(bits) - {"0", "1"}:
                bad.append({"expected": int(gold[i]), "got": f"undefined ({bits})"}); continue
            v = int(bits, 2)
            got = _signed((v >> lay["out_lsb"]) & ((1 << lay["out_width"]) - 1), lay["out_width"])
            other = v & ((1 << lay["out_lsb"]) - 1) or v >> (lay["out_lsb"] + lay["out_width"])      # input pins and upper pins must read 0
            if got != int(gold[i]) or other:
                bad.append({"inputs": {k: int(vec[k][i]) for k in vec}, "expected": int(gold[i]), "got": got, "stray_bits": bool(other)})
        if bad:
            return {"status": "fail", "stage": "function", "errors": [f"{len(bad)}/{n} vectors differ from the golden model through the pin mapping"],
                    "mismatches": bad[:4], "vectors": n}
        return {"status": "pass", "stage": "function", "errors": [], "vectors": n, "flop_bits": flops, "evidence_level": EVIDENCE_LEVEL}
