"""
Deterministic RTL checker (no LLM). Takes Verilog text for one eFPGA-side operator and decides pass/fail:
  1. text screen   - refuse `include, $readmem, $system, etc. (untrusted LLM text must not touch the filesystem)
  2. structure     - Yosys parses it, top module exists, ports match the spec exactly, no flip-flops, `check -assert` is clean
  3. function      - Yosys `eval` runs the circuit on test vectors; outputs are compared with a NumPy golden model

ML analogy: this is unit-testing a function. The spec is the docstring, `golden_*` is the reference implementation,
the vectors are the test inputs, and Yosys is the interpreter that actually runs the candidate.

Operators here are ASSUMED eFPGA-side ops (ReLU, saturating add, requantise). Their exact numeric conventions must be confirmed
with the X2 ABI owners; change OPS below when they are.
Evidence level: Yosys RTL evaluation vs NumPy golden (NOT nextpnr, NOT bitstream, NOT silicon).
"""
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

EVIDENCE_LEVEL = "yosys_rtl_evaluation_vs_numpy_golden (ASSUMED op conventions; NOT nextpnr, NOT silicon)"
MAX_VERILOG_CHARS = 20000
YOSYS_TIMEOUT_S = 120
REQUANT_M, REQUANT_S = 19, 10          # ASSUMED fixed-point scale: y = clip((acc*19 + 512) >> 10)


def golden_relu8(x):
    return np.maximum(x, 0)


def golden_add8_sat(a, b):
    return np.clip(a.astype(np.int64) + b.astype(np.int64), -128, 127)


def golden_requant(acc):
    return np.clip((acc.astype(np.int64) * REQUANT_M + (1 << (REQUANT_S - 1))) >> REQUANT_S, -128, 127)


def golden_x2_acc_step(acc_in, count, shift, neg):
    """One X2 recombination step (compiler/x2_pack.py recombine): acc +/- (count << shift), wrapping like 32-bit hardware."""
    v = count.astype(np.int64) << shift.astype(np.int64)
    s = acc_in.astype(np.int64) + np.where(neg.astype(np.int64) == 1, -v, v)
    return ((s + (1 << 31)) % (1 << 32)) - (1 << 31)


# ports: name -> (direction, width, signed)
OPS = {
    "relu8": {"top": "relu8", "inputs": {"x": 8}, "output": ("y", 8),
              "text": "y = max(x, 0) for signed 8-bit x.", "golden": lambda v: golden_relu8(v["x"])},
    "add8_sat": {"top": "add8_sat", "inputs": {"a": 8, "b": 8}, "output": ("y", 8),
                 "text": "y = clip(a + b, -128, 127) for signed 8-bit a, b (saturating add; the true sum needs 9 bits).",
                 "golden": lambda v: golden_add8_sat(v["a"], v["b"])},
    "requant_i32_i8": {"top": "requant_i32_i8", "inputs": {"acc": 32}, "output": ("y", 8),
                       "text": f"y = clip((acc*{REQUANT_M} + {1 << (REQUANT_S - 1)}) >> {REQUANT_S}, -128, 127) for signed 32-bit acc; "
                               "arithmetic shift (round half up); the product needs more than 32 bits.",
                       "golden": lambda v: golden_requant(v["acc"])},
    "x2_acc_step": {"top": "x2_acc_step", "inputs": {"acc_in": 32, "count": 14, "shift": 4, "neg": 1}, "output": ("acc_out", 32),
                    "unsigned": ("count", "shift", "neg"),
                    "text": "acc_out = acc_in + (neg ? -(count << shift) : (count << shift)), wrapping at 32 bits; acc_in and "
                            "acc_out signed 32-bit, count unsigned 14-bit (one X2 column count), shift unsigned 4-bit (0..15), "
                            "neg 1 bit. One step of rebuilding a layer output from X2 counts (sign bank, weight digit, input slice).",
                    "golden": lambda v: golden_x2_acc_step(v["acc_in"], v["count"], v["shift"], v["neg"])},
}
BANNED = re.compile(r"`include|\$readmem|\$system|\$fopen|\$fwrite|\$fscanf|\$dumpfile|\$value\$plusargs|`define\s+\w+\s*\(", re.I)


def describe_op(op):
    s = OPS[op]
    return {"op": op, "top_module": s["top"], "behaviour": s["text"], "combinational_only": True,
            "ports": [{"name": n, "dir": "input", "width": w, "signed": n not in s.get("unsigned", ())} for n, w in s["inputs"].items()]
                     + [{"name": s["output"][0], "dir": "output", "width": s["output"][1], "signed": True}],
            "convention": "ASSUMED - to be confirmed with the X2 ABI owners"}


def vectors_for(op, seed=0):
    """Corner cases + seeded random. Exhaustive for 8-bit single-input ops; dense for the rest."""
    s = OPS[op]
    rng = np.random.default_rng(seed)
    names = list(s["inputs"])
    if op == "relu8":
        return {"x": np.arange(-128, 128, dtype=np.int64)}
    if op == "add8_sat":
        edge = np.array([-128, -127, -64, -1, 0, 1, 63, 64, 126, 127], dtype=np.int64)
        a, b = np.meshgrid(edge, edge)
        ra, rb = rng.integers(-128, 128, 400), rng.integers(-128, 128, 400)
        return {"a": np.concatenate([a.ravel(), ra]), "b": np.concatenate([b.ravel(), rb])}
    if op == "x2_acc_step":
        lo, hi = -(2 ** 31), 2 ** 31 - 1
        acc_e = np.array([lo, lo + 1, -1, 0, 1, hi - 1, hi], dtype=np.int64)
        cnt_e = np.array([0, 1, 7200, 16383], dtype=np.int64)
        sh_e = np.array([0, 1, 4, 8, 12, 15], dtype=np.int64)
        A, C, S, N = (x.ravel() for x in np.meshgrid(acc_e, cnt_e, sh_e, np.array([0, 1]), indexing="ij"))
        n = 300
        return {"acc_in": np.concatenate([A, rng.integers(lo, hi, n, dtype=np.int64)]),
                "count": np.concatenate([C, rng.integers(0, 1 << 14, n)]),
                "shift": np.concatenate([S, rng.integers(0, 16, n)]),
                "neg": np.concatenate([N, rng.integers(0, 2, n)])}
    lo, hi = -(2 ** 31), 2 ** 31 - 1
    edge = np.array([lo, lo + 1, -100000, -1000, -54, -27, -1, 0, 1, 26, 27, 53, 54, 1000, 100000, hi - 1, hi], dtype=np.int64)
    near = np.array([(k * (1 << REQUANT_S) - (1 << (REQUANT_S - 1))) // REQUANT_M + d for k in (-128, -127, 0, 126, 127, 128) for d in (-1, 0, 1)], dtype=np.int64)
    rnd = np.concatenate([rng.integers(-3000, 3000, 300), rng.integers(lo, hi, 150, dtype=np.int64)])
    return {names[0]: np.concatenate([edge, near, rnd])}


def _yosys():
    exe = shutil.which("yowasp-yosys") or shutil.which("yosys")
    if not exe:                                     # venv not "activated": look next to this Python (Scripts\\ on Windows, bin/ elsewhere)
        for d in (Path(sys.executable).parent, Path(sys.executable).parent / "Scripts"):
            exe = shutil.which("yowasp-yosys", path=str(d))
            if exe:
                break
    if not exe:
        raise RuntimeError("Yosys not found: pip install yowasp-yosys (or install OSS CAD Suite)")
    return exe


def _run(args, cwd):
    return subprocess.run([_yosys(), *args], cwd=cwd, capture_output=True, text=True, timeout=YOSYS_TIMEOUT_S)


def _signed(v, w):
    v &= (1 << w) - 1
    return v - (1 << w) if v >> (w - 1) else v


_EVAL_BITS = re.compile(r"Eval result: \\\w+ = (\d+)'([01xXzZ]+)")
_EVAL_DEC = re.compile(r"Eval result: \\\w+ = (\d+)\.\s*$")


def parse_eval_results(stdout, width):
    """Yosys `eval -show` prints a result as <width>'<bits>, or, for a fully defined 32-bit value, as a plain decimal
    ('= 2147483647.'). Both forms -> [(width, bits)] in output order."""
    out = []
    for line in stdout.splitlines():
        m = _EVAL_BITS.search(line)
        if m:
            out.append((m.group(1), m.group(2)))
            continue
        m = _EVAL_DEC.search(line)
        if m:
            out.append((str(width), format(int(m.group(1)) & ((1 << width) - 1), f"0{width}b")))
    return out


def check_rtl(op, verilog):
    """Returns {status: pass|fail, stage: screen|structure|function, errors: [...], ...}. Never raises on bad Verilog."""
    if op not in OPS:
        return {"status": "fail", "stage": "screen", "errors": [f"unknown op '{op}'; known: {sorted(OPS)}"]}
    if not isinstance(verilog, str) or not verilog.strip():
        return {"status": "fail", "stage": "screen", "errors": ["verilog must be a non-empty string"]}
    if len(verilog) > MAX_VERILOG_CHARS or BANNED.search(verilog):
        return {"status": "fail", "stage": "screen", "errors": ["verilog rejected: too long or uses include/file/system constructs"]}
    s = OPS[op]
    with tempfile.TemporaryDirectory() as d:
        Path(d, "dut.v").write_text(verilog, encoding="utf-8")
        try:
            r = _run(["-p", f"read_verilog -sv dut.v; hierarchy -check -top {s['top']}; proc; opt_clean; check -assert; write_json dut.json"], d)
        except subprocess.TimeoutExpired:
            return {"status": "fail", "stage": "structure", "errors": ["yosys timed out"]}
        if r.returncode != 0:
            errs = [l.strip() for l in (r.stdout + r.stderr).splitlines() if re.search(r"ERROR|Warning: .*(driver|loop)|syntax", l)]
            return {"status": "fail", "stage": "structure", "errors": (errs or ["yosys failed"])[:6]}
        mod = json.loads(Path(d, "dut.json").read_text())["modules"].get(s["top"])
        if mod is None:
            return {"status": "fail", "stage": "structure", "errors": [f"top module '{s['top']}' not found"]}
        want = {**{n: ("input", w) for n, w in s["inputs"].items()}, s["output"][0]: ("output", s["output"][1])}
        got = {n: (p["direction"], len(p["bits"])) for n, p in mod["ports"].items()}
        if got != want:
            return {"status": "fail", "stage": "structure", "errors": [f"ports must be exactly {want}, got {got}"]}
        seq = sorted({c["type"] for c in mod["cells"].values() if c["type"].startswith(("$dff", "$adff", "$sdff", "$dlatch", "$mem"))})
        if seq:
            return {"status": "fail", "stage": "structure", "errors": [f"combinational module expected; found state elements {seq}"]}

        vec = vectors_for(op)
        n = len(next(iter(vec.values())))
        oname, owidth = s["output"]
        lines = [f"read_verilog -sv dut.v", f"hierarchy -top {s['top']}", "proc", "opt"]
        for i in range(n):
            sets = " ".join(f"-set {k} {s['inputs'][k]}'b{(int(vec[k][i]) & ((1 << s['inputs'][k]) - 1)):0{s['inputs'][k]}b}" for k in s["inputs"])
            lines.append(f"eval {sets} -show {oname}")
        Path(d, "run.ys").write_text("\n".join(lines) + "\n")
        try:
            r = _run(["-s", "run.ys"], d)
        except subprocess.TimeoutExpired:
            return {"status": "fail", "stage": "function", "errors": ["evaluation timed out"]}
        found = parse_eval_results(r.stdout, owidth)
        if r.returncode != 0 or len(found) != n:
            return {"status": "fail", "stage": "function", "errors": [f"could not evaluate all vectors ({len(found)}/{n})"]}
        gold = s["golden"](vec)
        bad = []
        for i, (w, bits) in enumerate(found):
            if set(bits) - {"0", "1"}:
                bad.append({"inputs": {k: int(vec[k][i]) for k in vec}, "expected": int(gold[i]), "got": f"undefined ({bits})"})
            elif _signed(int(bits, 2), int(w)) != int(gold[i]):
                bad.append({"inputs": {k: int(vec[k][i]) for k in vec}, "expected": int(gold[i]), "got": _signed(int(bits, 2), int(w))})
        if bad:
            return {"status": "fail", "stage": "function", "errors": [f"{len(bad)}/{n} vectors differ from the golden model"],
                    "mismatches": bad[:4], "vectors": n}
        return {"status": "pass", "stage": "function", "errors": [], "vectors": n, "evidence_level": EVIDENCE_LEVEL}
