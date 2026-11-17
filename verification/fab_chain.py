"""
Fabric chain (step 24): the logic that runs ONE operator through every stage, in order, and gives ONE verdict.

    rtl    A2's Verilog is re-checked in Yosys against the NumPy golden model
    prep   the verified core is wrapped in the fabric I/O interface; the wrapper is checked through the pin mapping
    env    the Ubuntu side has the tools the flow needs (nothing is changed there)
    flow   the real FABulous flow in Ubuntu: Yosys, nextpnr, bitstream
    sim    RTL simulation of the FABulous fabric with that bitstream loaded (Icarus) against NumPy
    check  run_flow_check reads every output file and decides ok / suspect / failed

Rules this module enforces (all deterministic, none of it is an LLM's opinion):
  * a stage that fails stops the chain; later stages are reported as "not run", never as passed;
  * before running, every result file downstream of the first stage to run is deleted, so a failed run can never be
    mistaken for an older pass;
  * the verdict is PASS only if every stage passed in this run.
Pure logic is kept apart from the Ubuntu calls so it can be unit-tested with fakes.
Evidence level: Yosys RTL evaluation + real FABulous flow outputs + RTL simulation of the FABulous fabric vs NumPy
(NOT timing, NOT gate-level, NOT silicon).
"""
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

STAGES = ("rtl", "prep", "env", "flow", "sim", "check")
EVIDENCE_LEVEL = ("yosys_rtl_evaluation + real_FABulous_flow_outputs + rtl_simulation_of_FABulous_fabric_vs_numpy "
                  "(NOT timing, NOT gate-level, NOT silicon)")
ALWAYS_NOT_CHECKED = ("gate-level or timing-annotated simulation", "silicon")

FLOW_FILES = ("{op}_fab.bin", "{op}_fab.fasm", "{op}_fab_npnr_log.txt", "top_wrapper.v", "fabric.csv", "flow_console.log", "exit_code.txt")
SIM_FILES = ("sim_console.log", "sim_exit_code.txt")
CHECK_FILES = ("flow_check.json",)

PREFLIGHT_SCRIPT = """#!/usr/bin/env bash
# Checks that Ubuntu has what run_fab_flow.sh and run_fab_sim.sh need. Changes nothing. Written by run_fab_chain.py.
DEMO="${DEMO:-$HOME/work/demo}"
if [ -f "$HOME/fabenv/bin/activate" ]; then . "$HOME/fabenv/bin/activate"; else echo "MISSING fabenv ($HOME/fabenv)"; fi
export PATH="$HOME/tools/oss-cad-suite/bin:$PATH"
for t in FABulous yosys nextpnr-generic iverilog; do
  if command -v "$t" >/dev/null 2>&1; then echo "FOUND $t"; else echo "MISSING $t"; fi
done
if [ -d "$DEMO/user_design" ] && [ -d "$DEMO/Test" ]; then echo "FOUND demo project"; else echo "MISSING demo project ($DEMO with user_design/ and Test/)"; fi
echo "VERSION yosys $(yosys -V 2>/dev/null | head -1)"
echo "VERSION nextpnr-generic $(nextpnr-generic --version 2>&1 | head -1)"
echo "VERSION iverilog $(iverilog -V 2>/dev/null | head -1)"
echo "PREFLIGHT DONE"
"""


# ---------------------------------------------------------------- paths and commands
def to_wsl_path(path):
    """'E:\\vm\\KLOK\\bm_verif' -> '/mnt/e/vm/KLOK/bm_verif'. A path that is already POSIX is returned unchanged."""
    p = str(path)
    m = re.match(r"^([A-Za-z]):(?:[\\/](.*))?$", p)
    if m:
        return "/mnt/" + m.group(1).lower() + "/" + (m.group(2) or "").replace("\\", "/")
    if p.startswith("/"):
        return p
    raise ValueError(f"cannot turn '{p}' into a path Ubuntu can open (expected a drive path such as E:\\...)")


_WSL_SAFE = re.compile(r"^[A-Za-z0-9_./:+@%=-]+$")


def script_path(runner, abs_path):
    """Path of a script as the chosen runner will see it. For wsl the path is converted and must contain no spaces or shell characters."""
    p = to_wsl_path(abs_path) if runner == "wsl" else Path(abs_path).as_posix()
    if runner == "wsl" and not _WSL_SAFE.match(p):
        raise ValueError(f"'{p}' has spaces or special characters that wsl.exe may split; move the project to a plain path such as E:\\vm\\KLOK\\bm_verif")
    return p


def resolve_runner(choice="auto", platform=None):
    """auto -> wsl on Windows (drives Ubuntu from here), native elsewhere (already inside Linux)."""
    if choice != "auto":
        return choice
    return "wsl" if (platform or sys.platform).startswith("win") else "native"


def build_command(runner, script_posix, args=(), distro="Ubuntu"):
    base = ["bash", script_posix, *args]
    if runner == "wsl":
        return ["wsl", "-d", distro, "--", *base]
    if runner == "native":
        return base
    raise ValueError(f"unknown runner '{runner}' (use wsl or native)")


# ---------------------------------------------------------------- running a command
def decode_output(raw):
    """wsl.exe prints its OWN messages as UTF-16 (NUL bytes between letters); everything from inside Ubuntu is UTF-8, and the two can
    arrive mixed in one stream. Dropping the NULs and BOM bytes keeps both readable."""
    if not raw:
        return ""
    if b"\x00" in raw:
        raw = raw.replace(b"\x00", b"").replace(b"\xff\xfe", b"")
    return raw.decode("utf-8", errors="replace")


def run_command(cmd, timeout_s=1800, heartbeat_s=30, say=None):
    """Run a command, merge stderr into stdout, report a heartbeat while it runs, kill it at the timeout. Never raises on a bad command."""
    t0 = time.time()
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)   # no stdin: a tool that asks a question fails instead of hanging
    except (FileNotFoundError, PermissionError, OSError) as e:
        return {"started": False, "returncode": None, "output": f"cannot start '{cmd[0]}': {e}", "seconds": 0.0, "timed_out": False}
    chunks, count = [], [0]

    def reader():
        for raw in iter(proc.stdout.readline, b""):
            chunks.append(raw)
            count[0] += 1

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    timed_out = False
    try:
        while True:
            remaining = timeout_s - (time.time() - t0)
            th.join(max(0.01, min(heartbeat_s, remaining)))
            if not th.is_alive():
                break
            if time.time() - t0 >= timeout_s:
                timed_out = True
                proc.kill()
                th.join(5)
                break
            if say:
                say(f"      ... still running: {int(time.time() - t0)} s, {count[0]} lines of tool output so far")
    except KeyboardInterrupt:          # Ctrl+C: stop the child we started, then let the interrupt through
        proc.kill()
        raise
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    return {"started": True, "returncode": proc.returncode, "output": decode_output(b"".join(chunks)),
            "seconds": round(time.time() - t0, 1), "timed_out": timed_out}


# ---------------------------------------------------------------- Ubuntu preflight
def parse_preflight(output):
    out = output or ""
    missing = [l.strip()[len("MISSING "):] for l in out.splitlines() if l.startswith("MISSING ")]
    versions = {}
    for l in out.splitlines():
        m = re.match(r"^VERSION (\S+) ?(.*)$", l.strip())
        if m and m.group(2).strip():
            versions[m.group(1)] = m.group(2).strip()
    done = "PREFLIGHT DONE" in out
    return {"ok": done and not missing, "reached_end": done, "missing": missing, "versions": versions}


def short_versions(versions):
    """'Yosys 0.66+154 (git sha1 ..., Clang ...)' -> '0.66+154' etc. The full text stays in the report; the console gets the short form."""
    pats = {"yosys": [r"Yosys\s+(\S+)"],
            "nextpnr-generic": [r"Version\s+nextpnr-(\S+?)(?:-g[0-9a-f]+)?\)", r"nextpnr-(\d\S*?)(?:-g[0-9a-f]+)?(?:\s|$)"],
            "iverilog": [r"version\s+(\S+)"]}
    out = {}
    for tool, text in (versions or {}).items():
        m = next((m for m in (re.search(p, text) for p in pats.get(tool, [])) if m), None)
        out[tool] = m.group(1) if m else text[:30]
    return out


def explain_preflight(run, info, runner, distro):
    if not run.get("started", True):
        return (f"{run['output']}. This chain drives Ubuntu through wsl.exe: run it from Windows PowerShell with WSL installed, "
                f"or from a Linux shell with --runner native")
    if not info["reached_end"]:
        first = " ".join((run["output"] or "").split())[:200]
        return (f"Ubuntu did not answer the tool check ({first or 'no output'}). Check the distro name with: wsl -l -v "
                f"(now using '{distro}'; change it with --distro)")
    return ("Ubuntu is missing: " + ", ".join(info["missing"]) + ". Set up as in docs/REAL_FABULOUS_FLOW.md, then run again")


# ---------------------------------------------------------------- stale results
def downstream_files(op, from_stage):
    i = STAGES.index(from_stage)
    names = []
    if i <= STAGES.index("flow"):
        names += FLOW_FILES
    if i <= STAGES.index("sim"):
        names += SIM_FILES
    if i <= STAGES.index("check"):
        names += CHECK_FILES
    return [n.format(op=op) for n in names]


def clean_job(job, op, from_stage):
    """Delete result files that this run is about to regenerate (and everything that depends on them). Returns the names removed."""
    removed = []
    for name in downstream_files(op, from_stage):
        p = Path(job) / name
        if p.exists():
            p.unlink()
            removed.append(name)
    return removed


# ---------------------------------------------------------------- the chain
def _row(stage, status, detail, seconds=0.0):
    return {"stage": stage, "status": status, "detail": detail, "seconds": round(seconds, 1)}


def run_chain(op, steps, from_stage="rtl", say=print):
    """steps: {stage: callable() -> (status, detail[, extra dict])}; status is pass | fail | unverified | blocked.
    Returns {op, verdict, stages, facts}. The check stage still runs after a failed flow or sim, to explain what went wrong."""
    if from_stage not in STAGES:
        raise ValueError(f"unknown stage '{from_stage}'; stages: {', '.join(STAGES)}")
    start, rows, facts = STAGES.index(from_stage), [], {}
    stop_reason, ran_flow_or_sim = None, False
    for i, stage in enumerate(STAGES):
        if i < start:
            rows.append(_row(stage, "skipped", "not re-run (--from-stage); earlier results are read from the job folder"))
            continue
        if stop_reason and not (stage == "check" and ran_flow_or_sim):
            rows.append(_row(stage, "not_run", f"not run: {stop_reason}"))
            continue
        t0 = time.time()
        try:
            res = steps[stage]()
            status, detail = res[0], res[1]
            extra = res[2] if len(res) > 2 else None
        except Exception as e:  # noqa: BLE001  - a crashing stage is a failed stage, never a pass
            status, detail, extra = "fail", f"{type(e).__name__}: {e}", None
        facts.update(extra or {})
        row = _row(stage, status, detail, time.time() - t0)
        rows.append(row)
        say(format_row(row, i))
        if stage in ("flow", "sim"):
            ran_flow_or_sim = True
        if status != "pass" and stage != "check" and not stop_reason:
            stop_reason = f"{stage} {'was blocked' if status == 'blocked' else 'did not pass'}"
    return {"op": op, "verdict": verdict(rows), "stages": rows, "facts": facts}


def verdict(rows):
    st = [r["status"] for r in rows if r["status"] != "skipped"]
    skipped = [r["stage"] for r in rows if r["status"] == "skipped"]
    if "fail" in st:
        return "FAIL"
    if "unverified" in st:
        return "NOT VERIFIED"
    if "blocked" in st or "not_run" in st:
        return "NOT RUN"
    if st and all(s == "pass" for s in st):
        return "PASS" if not skipped else f"PASS (partial: {', '.join(skipped)} not re-run)"
    return "NOT RUN"


LABEL = {"pass": "PASS", "fail": "FAIL", "unverified": "UNVERIFIED", "blocked": "BLOCKED", "skipped": "skipped", "not_run": "not run"}


def format_row(row, index):
    return f"  [{index + 1}/{len(STAGES)}] {row['stage']:<5} {LABEL[row['status']]:<10} {row['detail']}   ({row['seconds']} s)"


def format_summary(result):
    f = result["facts"]
    lines = [f"=== CHAIN RESULT {result['op']}: {result['verdict']} ===",
             "  " + " | ".join(f"{r['stage']} {LABEL[r['status']]}" for r in result["stages"])]
    if f.get("bitstream_sha256"):
        lines.append(f"  bitstream sha256 {f['bitstream_sha256'][:16]}... | set bits {f.get('set_bits')} | logic cells {f.get('logic_cells')} | FASM features {f.get('fasm_features')}")
    for x in group_findings(f.get("findings")):
        lines.append(f"  finding: {x}")
    if result["verdict"].startswith("PASS"):
        lines.append("  shows: the bitstream FABulous produced makes the simulated fabric match the NumPy golden model on every vector")
    for clock, t in (f.get("fmax") or {}).items():
        lines.append(f"  timing, from nextpnr's delay model of the demo fabric (NOT measured hardware): clock {clock} can run at most "
                     f"{t.get('fmax_mhz')} MHz; target {t.get('target_mhz')} MHz: {'met' if t.get('ok') else 'NOT met'}")
    nc = list(f.get("not_checked") or [])
    if "fmax" in f and not f["fmax"]:             # only once the check stage has read the nextpnr log
        nc.append("timing (nextpnr printed no fmax for this design)")
    nc += ALWAYS_NOT_CHECKED
    lines.append("  not checked (so not claimed): " + ", ".join(nc))
    return lines


def group_findings(findings, limit=5):
    """Same kind of finding on different wires/numbers is shown once with a count: 'x5 [warning] yosys_warning: ... (first of 5)'."""
    groups = {}
    for text in findings or []:
        key = re.sub(r"\\\S+\s*\[\d+\]|\d+", "#", text)
        groups.setdefault(key, []).append(text)
    lines = [(g[0] if len(g) == 1 else f"x{len(g)} {g[0]}   (first of {len(g)} similar)") for g in groups.values()]
    return lines[:limit] + ([f"... and {len(lines) - limit} more kinds in the report"] if len(lines) > limit else [])


def utilisation_text(util):
    """FABULOUS_LC comes from the flow check as {'used': 25, 'total': 672}."""
    if isinstance(util, dict) and "used" in util and "total" in util:
        return f"{util['used']}/{util['total']}"
    return str(util)
