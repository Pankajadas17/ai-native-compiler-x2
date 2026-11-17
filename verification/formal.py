"""
Formal checks and lint (review item 4): proofs over ALL inputs / all behaviours instead of chosen test vectors.

  spec      each A2 operator (reports/rtl/<op>.v) == an independently written Verilog spec, for EVERY input (Yosys SAT);
            a deliberately broken copy must give a counterexample (shows the check has teeth)
  netlist   the operator synthesized by the CHIP's own synthesis (repo techmaps + our signed multiplier map) is equivalent
            to its RTL for EVERY input, using the repo's primitive models (LUTs, carry, MACC). The same with the repo's
            original multiplier map must give a counterexample for requant_i32_i8 (the bug the override fixes)
  lint      Verilator -Wall on our Verilog: no errors and none of the serious warning classes
  engine    SymbiYosys bounded model check of examples/chip_designs/x2_engine2.v against an unconstrained X2
            (formal/RERAM_CIM_32x32_formal.v) with properties formal/x2_engine2_props.vh; a copy without the pipeline's
            forwarding must fail
Runs in Ubuntu (Yosys 0.70+, SymbiYosys, Verilator, a solver: all in the chip OSS CAD Suite).
"""
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORMAL_DIR = ROOT / "formal"
EVIDENCE = ("formal: Yosys SAT over all inputs (operators, RTL and chip-synthesized netlist with the repo's primitive models) "
            "+ SymbiYosys BOUNDED model check of the engine + Verilator lint; NOT the configured fabric, NOT silicon")
OPS = ("relu8", "add8_sat", "requant_i32_i8")

# Independent specs: written from the operators' definitions, NOT from the RTL (different constructions).
SPECS = {
    "relu8": """module relu8_spec(input signed [7:0] x, output signed [7:0] y);
  assign y = (x < 0) ? 8'sd0 : x;                                   // ReLU on signed INT8
endmodule""",
    "add8_sat": """module add8_sat_spec(input signed [7:0] a, input signed [7:0] b, output signed [7:0] y);
  integer s; reg signed [7:0] r;                                     // exact sum in 32 bits, then saturate
  always @(*) begin s = a + b; if (s > 127) r = 127; else if (s < -128) r = -128; else r = s; end
  assign y = r;
endmodule""",
    "requant_i32_i8": """module requant_i32_i8_spec(input signed [31:0] acc, output signed [7:0] y);
  // clamp(floor((acc*19 + 512) / 1024)): floor written as truncating division with a correction for negatives
  wire signed [63:0] n = $signed(acc) * 64'sd19 + 64'sd512;
  wire signed [63:0] q = (n >= 0) ? n / 64'sd1024 : -((-n + 64'sd1023) / 64'sd1024);
  assign y = (q > 64'sd127) ? 8'sd127 : (q < -64'sd128) ? -8'sd128 : q[7:0];
endmodule""",
}
MUTANTS = {"relu8": ("x[7]", "x[6]"), "add8_sat": ("8'sd127 :", "8'sd126 :"), "requant_i32_i8": ("48'sd19", "48'sd18")}
ENGINE_MUTANT = ("(s2_writes && s2_idx == s1_idx) ? nxt_full : acc[s1_idx]", "acc[s1_idx]")      # forwarding removed
LINT_DESIGNS = {             # top -> files (ours); the X2 primitive declaration comes from the chip repo, read-only
    "relu8": ["reports/rtl/relu8.v"], "add8_sat": ["reports/rtl/add8_sat.v"], "requant_i32_i8": ["reports/rtl/requant_i32_i8.v"],
    "x2_recombiner": ["examples/chip_designs/x2_recombiner.v"],
    "x2_bridge": ["examples/chip_designs/x2_bridge.v"],
    "x2_engine_4bit": ["examples/chip_designs/x2_engine_4bit.v", "examples/chip_designs/x2_engine.v"],
    "x2_engine2_4bit": ["examples/chip_designs/x2_engine2_4bit.v", "examples/chip_designs/x2_engine2.v"],
    "x2_vendor_cfg": ["examples/chip_designs/x2_vendor_cfg.sv"],
}
SERIOUS = {"UNDRIVEN", "MULTIDRIVEN", "LATCH", "COMBDLY", "BLKSEQ", "BLKANDNBLK", "CASEINCOMPLETE", "CASEX", "IMPLICIT",
           "SELRANGE", "ALWCOMBORDER", "INITIALDLY", "SYNCASYNCNET", "UNOPTFLAT", "MULTIDRIVEN", "PINMISSING"}
X2_PRIMITIVE = "ip/extra-tiles/primitives/RERAM_CIM_32x32/yosys/primitives/RERAM_CIM_32x32.v"
PRIM_MODELS = ("ip/fabulous-tiles/primitives/FABULOUS_LC/yosys/primitives/prims.v",
               "ip/fabulous-tiles/primitives/MACC_8x8_20/yosys/primitives/MACC_8x8_20.v")
REPO_MULTIPLIER = "ip/fabulous-tiles/primitives/MACC_8x8_20/yosys/techmap/MACC_8x8_20_map.v"


def _run(cmd, cwd, env, timeout=1800):
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout + r.stderr, round(time.time() - t0, 1)
    except subprocess.TimeoutExpired as e:
        return None, (e.stdout or "") if isinstance(e.stdout, str) else "", round(time.time() - t0, 1)


_SAT_ROW = re.compile(r"^\s+\\(in_|gold_|gate_)?(\w+)\s+(-?\d+)\s+\S+\s+([01xz]+)\s*$", re.M)


def _signed(bits):
    """Two's-complement value of a SAT bit string (the miter shows outputs unsigned)."""
    v = int(bits, 2)
    return v - (1 << len(bits)) if bits[0] == "1" else v


def parse_sat(text):
    """Yosys 'sat -prove' output -> proved / counterexample (with the input values) / error."""
    if "SUCCESS!" in text and "proof did fail" not in text:
        return {"result": "proved"}
    if "model found: FAIL!" in text:
        vals = {}
        for pre, name, dec, bits in _SAT_ROW.findall(text):
            if name == "trigger":
                continue
            v = _signed(bits) if set(bits) <= {"0", "1"} else bits
            vals[f"{pre.rstrip('_')}.{name}" if pre in ("gold_", "gate_") else name] = v
        return {"result": "counterexample", "values": vals}
    err = next((l for l in text.splitlines() if l.startswith("ERROR")), "no SAT verdict in the log")
    return {"result": "error", "error": err}


def _miter(gold_cmds, gate_cmds, gold, gate):
    return "; ".join(gate_cmds + [f"rename {gate[0]} {gate[1]}"] + gold_cmds + [f"rename {gold[0]} {gold[1]}", "proc",
                     f"miter -equiv -flatten -make_outputs {gold[1]} {gate[1]} miter", "hierarchy -top miter", "flatten",
                     "opt -fast", "sat -verify -prove trigger 0 -show-inputs -show-outputs -set-init-zero miter"])


def prove_spec(op, rtl_text, work, env):
    """RTL == spec for every input. rtl_text may be a mutated copy."""
    work.mkdir(parents=True, exist_ok=True)
    (work / f"{op}.v").write_text(rtl_text, encoding="utf-8")
    (work / f"{op}_spec.v").write_text(SPECS[op] + "\n", encoding="utf-8")
    script = _miter([f"read_verilog {op}_spec.v"], [f"read_verilog {op}.v", "proc"], (f"{op}_spec", "gold"), (op, "gate"))
    rc, out, sec = _run(["yosys", "-p", script], work, env)
    (work / "yosys.log").write_text(out, encoding="utf-8")
    return {**parse_sat(out), "seconds": sec}


def synth_core(op, rtl_file, work, env, repo, multiplier=None):
    """The chip's own synthesis (repo makefile.include via the generated Makefile) of the bare operator -> JSON netlist."""
    from chip import chip_flow as cf
    work.mkdir(parents=True, exist_ok=True)
    (work / "rtl").mkdir(exist_ok=True)
    shutil.copyfile(rtl_file, work / "rtl" / f"{op}.v")
    mm, args = multiplier or cf.multiplier_override(repo)
    (work / "Makefile").write_text(cf.MAKEFILE.format(name=f"formal_{op}", top=op, rtl=f"rtl/{op}.v", multiplier_map=mm,
                                                      multiplier_args=args), encoding="ascii", newline="\n")
    e = dict(env, REPO=str(repo.root))
    rc, out, sec = _run(["make", "synth"], work, e)
    (work / "synth.log").write_text(out, encoding="utf-8")
    if rc != 0 or not (work / f"{op}.json").is_file():
        raise RuntimeError(f"chip synthesis of {op} failed (see {work / 'synth.log'})")
    return work / f"{op}.json"


def strip_pads(netlist, top):
    """Remove the IOBUF pad cells synthesis adds at the ports (EN=0: input pin, EN=1: output pin) and the blackbox stubs,
    so the netlist's ports are the operator's ports and its cells are modelled by the repo's primitive models."""
    m = netlist["modules"][top]
    sub = {}
    for name, c in list(m["cells"].items()):
        if c["type"] != "IOBUF":
            continue
        en, pad = c["connections"]["EN"], c["connections"]["PAD"]
        if en == ["0"]:
            sub.update(zip(c["connections"]["OUT"], pad))
        elif en == ["1"]:
            sub.update(zip(pad, c["connections"]["IN"]))
        else:
            raise ValueError(f"IOBUF {name}: EN is not a constant")
        del m["cells"][name]
    f = lambda bits: [sub.get(b, b) for b in bits]      # noqa: E731
    for c in m["cells"].values():
        c["connections"] = {k: f(v) for k, v in c["connections"].items()}
    for p in m["ports"].values():
        p["bits"] = f(p["bits"])
    for n in m.get("netnames", {}).values():
        n["bits"] = f(n["bits"])
    return {**netlist, "modules": {top: m}}


def prove_netlist(op, rtl_file, netlist_json, work, env, repo):
    """Chip-synthesized netlist (with the repo's primitive models) == RTL, for every input."""
    work.mkdir(parents=True, exist_ok=True)
    g = strip_pads(json.loads(Path(netlist_json).read_text()), op)
    (work / "gate.json").write_text(json.dumps(g), encoding="utf-8")
    shutil.copyfile(rtl_file, work / f"{op}.v")
    prims = " ".join(str(repo.path(p)) for p in PRIM_MODELS)
    gate = ["read_json gate.json", f"read_verilog -sv {prims}", f"hierarchy -top {op}", "proc", "flatten"]
    script = _miter([f"read_verilog {op}.v"], gate, (op, "gold"), (op, "gate"))
    rc, out, sec = _run(["yosys", "-p", script], work, env)
    (work / "yosys.log").write_text(out, encoding="utf-8")
    return {**parse_sat(out), "seconds": sec}


_LINT = re.compile(r"^%(Warning|Error)(?:-([A-Z0-9_]+))?: ([^:]+):(\d+):", re.M)


def lint(top, files, work, env, repo):
    work.mkdir(parents=True, exist_ok=True)
    cfg = work / "waivers.vlt"
    cfg.write_text('`verilator_config\n// the chip repo\'s X2 primitive is a port-only declaration: not our code\n'
                   f'lint_off -file "*{Path(X2_PRIMITIVE).name}"\n'
                   '// outputs of the X2 primitive we deliberately leave open (status pins the engine does not use)\n'
                   'lint_off -rule PINCONNECTEMPTY\nlint_off -rule DECLFILENAME\n', encoding="utf-8")
    srcs = [str(cfg)] + [str(ROOT / f) for f in files] + [str(repo.path(X2_PRIMITIVE))]
    rc, out, sec = _run(["verilator", "--lint-only", "-Wall", "-Wno-fatal", "--top-module", top, *srcs], work, env, 600)
    (work / "verilator.log").write_text(out, encoding="utf-8")
    errors, warns = [], {}
    for kind, code, path, line in _LINT.findall(out):
        if kind == "Error":
            errors.append(f"{Path(path).name}:{line} {code}")
        else:
            warns.setdefault(code or "?", []).append(f"{Path(path).name}:{line}")
    serious = {k: v for k, v in warns.items() if k in SERIOUS}
    ok = rc is not None and not errors and not serious
    return {"result": "clean" if ok else "findings", "errors": errors, "serious": serious,
            "warnings": {k: len(v) for k, v in warns.items()}, "seconds": sec}


def _prop_name(loc):
    """'x2_engine2_props.vh:37.54-37.78' -> 'P3 (x2_engine2_props.vh:37)' using the // Pn tag on that line."""
    m = re.match(r"([^:]+):(\d+)", loc)
    if not m:
        return loc
    try:
        line = (FORMAL_DIR / m.group(1)).read_text(encoding="utf-8").splitlines()[int(m.group(2)) - 1]
    except (OSError, IndexError):
        return loc
    tag = re.search(r"//\s*(P\d)", line)
    return f"{tag.group(1) if tag else '?'} ({m.group(1)}:{m.group(2)})"


def engine_bmc(cell_bits, depth, work, env, mutate=False, solver="bitwuzla", timeout=3600):
    """SymbiYosys BMC of x2_engine2 (CELL_BITS 1 or 4) to `depth` clocks after reset."""
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    src = (ROOT / "examples" / "chip_designs" / "x2_engine2.v").read_text(encoding="utf-8")
    if mutate:
        if ENGINE_MUTANT[0] not in src:
            raise RuntimeError("the engine's forwarding expression changed: update ENGINE_MUTANT")
        src = src.replace(ENGINE_MUTANT[0], ENGINE_MUTANT[1])
    (work / "x2_engine2.v").write_text(src, encoding="utf-8")
    for f in ("RERAM_CIM_32x32_formal.v", "x2_engine2_props.vh"):
        shutil.copyfile(FORMAL_DIR / f, work / f)
    (work / "check.sby").write_text(
        f"[options]\nmode bmc\ndepth {depth}\n\n[engines]\nsmtbmc {solver}\n\n[script]\n"
        "read -formal RERAM_CIM_32x32_formal.v\nread -formal -I. x2_engine2.v\n"
        f"chparam -set CELL_BITS {cell_bits} x2_engine2\nprep -top x2_engine2\n\n[files]\n"
        "RERAM_CIM_32x32_formal.v\nx2_engine2_props.vh\nx2_engine2.v\n", encoding="utf-8")
    rc, out, sec = _run(["sby", "-f", "check.sby"], work, env, timeout)
    (work / "sby_console.log").write_text(out, encoding="utf-8")
    m = re.search(r"DONE \((PASS|FAIL|ERROR|UNKNOWN|TIMEOUT)", out)
    res = {"PASS": "pass", "FAIL": "fail"}.get(m.group(1), m.group(1).lower()) if m else ("timeout" if rc is None else "error")
    trace = next(iter(sorted(work.glob("check/engine_0/trace*.vcd"))), None)
    failed = [_prop_name(f) for f in re.findall(r"Assert failed in \S+: (\S+)", out)]
    return {"result": res, "depth": depth, "cell_bits": cell_bits, "seconds": sec,
            "trace": str(trace.relative_to(ROOT)) if trace and trace.is_relative_to(ROOT) else (str(trace) if trace else None),
            "failed_asserts": failed[:5]}
