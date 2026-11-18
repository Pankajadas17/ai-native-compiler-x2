"""
One command for the whole chain: NumPy golden -> A2 Verilog -> fabric wrapper -> real FABulous flow -> fabric simulation -> check.

    x2c chain --op add8_sat              # run from the project folder in Windows PowerShell (drives Ubuntu itself)
    x2c chain --op all                   # every operator that fits the fabric's 28 pins (add8_sat, relu8)
    x2c chain --op add8_sat --plan       # show the stages and the exact commands, run nothing
    x2c chain --op add8_sat --from-stage sim    # redo only sim + check (keeps the bitstream from the last flow)
    x2c chain --op all --target chip     # the BM Labs chip (sky130-reram-fpga) instead of the demo fabric:
                                                       # verification/chip_chain.py, jobs in reports/chip_jobs/, builds in chip_build/chain_<op>/

Stages (stop at the first one that does not pass): rtl, prep, env, flow, sim, check. The flow and sim stages run
reports/fab_jobs/<op>/run_fab_flow.sh and run_fab_sim.sh in Ubuntu through `wsl -d Ubuntu -- bash <script>`.
Old result files are deleted first, so a failed run cannot be mistaken for an earlier pass.
With the wsl runner, Windows' free memory is checked before the env and sim stages (the Ubuntu VM stops when Windows runs out of RAM):
below 1 GB the stage is not started (NOT RUN, with what to close), below 4 GB there is a warning; --ignore-memory turns the check off.
If a simulation ends without a result, the report says how low Windows' free RAM went while it ran.
Report: reports/fab_chain_report.json. Evidence level: Yosys RTL evaluation + real FABulous flow outputs +
RTL simulation of the FABulous fabric vs NumPy (NOT timing, NOT gate-level, NOT silicon).
"""
import argparse
import contextlib
import io
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import x2c.commands.demo_prep as run_fab_prep
import x2c.commands.demo_flow_check as run_flow_check
from x2c.agents.llm_client import GeminiLLM, GroqLLM
from x2c.agents.rtl_agent import MockRTLLLM, run_rtl_agent
from x2c.verification import chain_debug
from x2c.verification import chip_chain as cc
from x2c.verification import fab_chain as fc
from x2c.verification import wsl_probe as wp
from x2c.verification.fab_sim import parse_sim_log, tb_vector_count
from x2c.verification.fab_wrap import wrapper_layout
from x2c.verification.rtl_check import OPS, check_rtl

JOBS = Path("reports/fab_jobs")
REPORT = Path("reports/fab_chain_report.json")
KEYS = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY"}


def say(msg):
    print(msg, flush=True)


def _int_in(path):
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def _tail(text, n=3):
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    return " / ".join(lines[-n:])[:300]


_TIMING_ERROR = re.compile(r"^ERROR: Max frequency for clock '(.+?)':\s+([\d.]+) MHz \(FAIL at ([\d.]+) MHz\)", re.M)


def timing_failure(npnr_log):
    """nextpnr's final timing ERROR line from the log it wrote, as {clock, fmax_mhz, target_mhz}; None when there is none (or no log)."""
    try:
        m = _TIMING_ERROR.findall(Path(npnr_log).read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return None
    return {"clock": m[-1][0], "fmax_mhz": float(m[-1][1]), "target_mhz": float(m[-1][2])} if m else None


def wrappable_ops():
    """Operators that fit the fabric's I/O, and those that do not (with the reason)."""
    ok, no = [], {}
    for op in OPS:
        try:
            wrapper_layout(op)
            ok.append(op)
        except ValueError as e:
            no[op] = str(e)
    return ok, no


def make_llm(name, op, model=None):
    if name == "mock":
        return MockRTLLLM(op)
    return {"groq": GroqLLM, "gemini": GeminiLLM}[name](model=model)


def build_steps(op, cfg, exec_fn=fc.run_command, preflight=None):
    """The real stage functions for one operator. exec_fn runs a command in Ubuntu (replaced by a fake in the tests)."""
    job = JOBS / op
    preflight = preflight if preflight is not None else {}

    def ubuntu(script, args, timeout):
        sp = fc.script_path(cfg.runner, str((job / script).resolve()))
        ubuntu_used["yes"] = True
        return exec_fn(fc.build_command(cfg.runner, sp, args, cfg.distro), timeout, cfg.heartbeat_s, say)

    ubuntu_used = {"yes": False}       # set when a stage of THIS run has already run something in Ubuntu (flow, or sim itself)

    def host_now():
        """Windows' free RAM right now; None when the memory check is off (only the wsl runner has the VM-runs-out-of-Windows-RAM problem)."""
        return cfg.host_reader() if getattr(cfg, "check_memory", False) else None

    def rtl():
        path = Path("hardware/operators") / f"{op}.v"
        note = ""
        if path.exists() and not cfg.regen_rtl:
            res = check_rtl(op, path.read_text(encoding="utf-8"))
            if res["status"] == "pass":
                return "pass", f"{path.as_posix()} re-checked in Yosys vs NumPy: {res['vectors']} vectors"
            note = f"existing file failed its re-check ({res.get('stage')}); asked A2 again. "
        if cfg.llm in KEYS and not os.environ.get(KEYS[cfg.llm]):
            return "blocked", f"{KEYS[cfg.llm]} is not set in this window (set it with Read-Host -AsSecureString) or use --llm mock"
        r = run_rtl_agent(make_llm(cfg.llm, op, cfg.model), op)
        if r["verified"]:
            return "pass", f"{note}A2 ({cfg.llm}) wrote Verilog; Yosys vs NumPy: {r['vectors']} vectors, {r['attempts']} attempt(s)"
        why = f"{r['last_stage']}: {r['last_errors'][0]}" if r["last_errors"] else (r["llm_error"] or r["stopped"])
        return "fail", f"{note}A2 produced no passing Verilog ({why})"

    def prep():
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                rc = run_fab_prep.main(["--op", op])
            except SystemExit as e:
                rc = 1
                if isinstance(e.code, str):
                    print(e.code)
                elif e.code == 0:
                    rc = 0
        out = buf.getvalue().strip()
        first = next((l for l in out.splitlines() if l.startswith("wrapper check")), out.splitlines()[0] if out else "no output")
        return ("pass" if rc == 0 and (job / f"{op}_fab.v").exists() else "fail"), first

    def env():
        if "info" not in preflight:
            gate, gate_text = wp.memory_gate(host_now())
            if gate == "block":          # nothing is asked of Ubuntu: it would only stop (and nothing is cached, so the next run checks again)
                return "blocked", gate_text + " (Ignore this check with --ignore-memory.)"
            preflight["memory"] = gate_text if gate == "warn" else ""
            pf = JOBS / "_preflight.sh"
            JOBS.mkdir(parents=True, exist_ok=True)
            pf.write_text(fc.PREFLIGHT_SCRIPT, encoding="utf-8", newline="\n")
            sp = fc.script_path(cfg.runner, str(pf.resolve()))
            run = exec_fn(fc.build_command(cfg.runner, sp, [], cfg.distro), 180, cfg.heartbeat_s, say)
            preflight["run"], preflight["info"] = run, fc.parse_preflight(run["output"])
        info, memory = preflight["info"], preflight.get("memory", "")
        if not info["ok"]:
            return "blocked", fc.explain_preflight(preflight["run"], info, cfg.runner, cfg.distro) + (f" | {memory}" if memory else "")
        v = info["versions"]
        short = fc.short_versions(v)
        return ("pass", "Ubuntu has FABulous, Yosys, nextpnr, Icarus and the demo project (Yosys " + short.get("yosys", "?") + ", nextpnr "
                + short.get("nextpnr-generic", "?") + ", Icarus " + short.get("iverilog", "?") + ")" + (f" | WARNING: {memory}" if memory else ""),
                {"tool_versions": v})

    def flow():
        say(f"  ... flow: running in Ubuntu (FABulous + Yosys + nextpnr), usually a few minutes; progress is also in {job.as_posix()}/flow_console.log")
        freq = getattr(cfg, "freq_mhz", None)
        target = f"clock target {freq:g} MHz (--freq-mhz)" if freq else "nextpnr's default clock target"
        run = ubuntu("run_fab_flow.sh", [op] + ([f"{freq:g}"] if freq else []), cfg.timeout_s)
        if not run["started"]:
            return "blocked", run["output"]
        if run["timed_out"]:
            return "fail", f"timed out after {cfg.timeout_s} s. Ubuntu may still be running it; stop it with: wsl --shutdown  (PowerShell)"
        code = _int_in(job / "exit_code.txt")
        binp = job / f"{op}_fab.bin"
        if code is None:
            return "fail", f"the flow script did not finish (no exit_code.txt). Last output: {_tail(run['output'])}"
        if code == 97:
            return "fail", "FABulous could not connect the pins of the wrapper (src/x2c/verification/fab_connect.py refused); see flow_console.log"
        timing = timing_failure(job / f"{op}_fab_npnr_log.txt")
        if code != 0 and timing:
            return "fail", (f"timing not met: nextpnr's delay model gives at most {timing['fmax_mhz']:g} MHz for clock '{timing['clock']}', the target is "
                            f"{timing['target_mhz']:g} MHz, so no bitstream was written. A lower target can be tried with --freq-mhz "
                            f"(for example {max(1, int(timing['fmax_mhz'] * 0.7))}); the number is a model of the demo fabric, not measured hardware")
        if code != 0:
            return "fail", f"FABulous exited with code {code}; see {job.as_posix()}/flow_console.log. Last output: {_tail(run['output'])}"
        if not binp.exists():
            return "fail", "FABulous exited 0 but wrote no bitstream"
        return "pass", f"FABulous exit 0, bitstream written ({binp.stat().st_size} bytes), {target}, {run['seconds']} s in Ubuntu"

    def sim():
        if not (job / f"{op}_fab.bin").exists():
            return "fail", "no bitstream to simulate"
        if (job / f"{op}_fab.bin").stat().st_size == 0:
            return "fail", f"the bitstream in the job folder is empty (0 bytes): run again with --from-stage flow"
        pre = host_now()
        gate, gate_text = wp.memory_gate(pre)
        # Only before Ubuntu's first use in this run: once an earlier stage has run there, the VM already holds its memory, and Windows'
        # headroom then looks low although nothing is missing (seen: blocked at 2.0 GB right after a passing flow stage; the simulation then passed).
        if gate == "block" and not ubuntu_used["yes"]:
            return "blocked", gate_text + " (Ignore this check with --ignore-memory.)"
        say("  ... sim: simulating the configured fabric in Ubuntu (Icarus), usually under a minute")
        sampler = wp.HostSampler(reader=cfg.host_reader, interval=getattr(cfg, "sample_s", 0.5)).start() if pre else None
        run = ubuntu("run_fab_sim.sh", [op], cfg.timeout_s)
        readings = []                                # Windows' free RAM before, during and after the run (stays empty when the check is off)
        if pre:
            readings = [{"t": 0.0, **pre}] + sampler.stop()
            post = host_now()
            if post:
                readings.append({"t": round(run.get("seconds", 0.0), 1), **post})
        memory = wp.host_summary(readings)

        def note():
            return " | " + wp.sim_memory_sentence(memory, op) if memory else ""
        if not run["started"]:
            return "blocked", run["output"]
        if run["timed_out"]:
            return "fail", f"timed out after {cfg.timeout_s} s"
        log = job / "sim_console.log"
        facts = {"windows_memory": memory} if memory else None          # kept in the report: how low Windows' free RAM went during the simulation
        if not log.exists():
            return "fail", f"no sim_console.log was written. Last output: {_tail(run['output'])}" + note(), facts
        tb = job / f"{op}_fab_tb.v"
        res = parse_sim_log(log.read_text(encoding="utf-8", errors="replace"),
                            tb_vector_count(tb.read_text(encoding="utf-8")) if tb.exists() else None)
        status = {"pass": "pass", "fail": "fail"}.get(res["status"], "unverified")
        extra = "; " + "; ".join(res["tool_errors"]) if res.get("tool_errors") else ""
        restored = next((l.strip() for l in (run["output"] or "").splitlines() if l.startswith("FAB_SIM NOTE:")), "")
        return status, res["reason"] + extra + (f" | {restored[len('FAB_SIM '):]}" if restored else "") + (note() if status == "unverified" else ""), facts

    def check():
        Path("reports/flow_check.json").unlink(missing_ok=True)      # never read a report left by an earlier run
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                run_flow_check.main(["--job", str(job)])
            except SystemExit as e:
                if isinstance(e.code, str):
                    return "fail", f"flow check could not read the job folder: {e.code}"
        rep_path = Path("reports/flow_check.json")
        if not rep_path.exists():
            return "fail", "flow check wrote no report"
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        shutil.copy(rep_path, job / "flow_check.json")
        f, simr = rep["facts"], (rep["facts"].get("fabric_simulation") or {})
        extra = {"bitstream_sha256": f.get("bitstream_sha256"), "set_bits": f.get("bitstream_set_bits"),
                 "logic_cells": fc.utilisation_text((f.get("utilisation") or {}).get("FABULOUS_LC")),
                 "fasm_features": f.get("fasm_features"), "fmax": f.get("fmax"), "not_checked": rep.get("not_checked", []),
                 "findings": [f"[{x['severity']}] {x['code']}: {x['message']}" for x in rep.get("findings", [])]}
        detail = f"FLOW STATUS {rep['status'].upper()} | fabric simulation {str(simr.get('status', 'not run')).upper()}"
        if rep["status"] == "ok" and simr.get("status") == "pass":
            return "pass", detail, extra
        return ("fail" if rep["status"] == "failed" else "unverified"), detail, extra

    return {"rtl": rtl, "prep": prep, "env": env, "flow": flow, "sim": sim, "check": check}


def plan_text(op, cfg):
    job = JOBS / op
    cmds = {}
    for stage, script in (("flow", "run_fab_flow.sh"), ("sim", "run_fab_sim.sh")):
        try:
            cmds[stage] = "$ " + " ".join(fc.build_command(cfg.runner, fc.script_path(cfg.runner, str((job / script).resolve())), [op], cfg.distro))
        except ValueError as e:          # e.g. a project path with spaces: say so now, not halfway through a run
            cmds[stage] = f"CANNOT RUN: {e}"
    lines = [f"plan for {op} (runner {cfg.runner}, starting at stage '{cfg.from_stage}'); nothing is run:"]
    for s in fc.STAGES:
        mark = "skip" if fc.STAGES.index(s) < fc.STAGES.index(cfg.from_stage) else "run "
        lines.append(f"  {mark} {s}" + (f"   {cmds[s]}" if s in cmds else ""))
    gone = fc.downstream_files(op, cfg.from_stage)
    lines.append(f"  first, old results in {job.as_posix()} are deleted: {', '.join(gone) if gone else 'none'}")
    if getattr(cfg, "check_memory", False):
        lines.append(f"  Windows' free memory is checked before env and sim: stage not started below {wp.GATE_FREE_GB:.0f} GB, warning below {wp.NEEDED_FREE_GB:.0f} GB (--ignore-memory turns it off)")
    return lines


def diagnose(op, r, target="demo"):
    """For a run that did not pass: the root cause the evidence RULES support (verification/chain_debug.py; no LLM is called here, so the chain
    stays deterministic). The result is kept in r["diagnosis"] for the report. Lines to print; nothing for a passing run."""
    if str(r.get("verdict", "")).startswith("PASS"):
        return []
    try:
        if target == "chip":
            ev = chain_debug.collect_chip_evidence(op, r)
            cands = chain_debug.chip_rule_candidates(ev)
        else:
            ev = chain_debug.collect_evidence(op, r, JOBS)
            cands = chain_debug.rule_candidates(ev)
    except Exception as e:  # noqa: BLE001  - a diagnosis is a hint; it must never break the chain's own result
        r["diagnosis"] = {"error": f"{type(e).__name__}: {e}"}
        return []
    r["diagnosis"] = {"rule_based_candidates": cands, "source": "rules (no LLM)"}
    ask = (f"x2c agent a3{' --target chip' if target == 'chip' else ''} --llm groq   "
           "(asks the debug agent; the answer is checked against the same evidence)")
    if not cands:
        return [f"  likely cause: not one of the known causes. Second opinion: {ask}"]
    c = chain_debug.cause_info(cands[0])
    return [f"  likely cause [{cands[0]}]: {c['suspect']}", f"    fix: {c['fix']}", f"    (from the evidence rules, no LLM. Second opinion: {ask})"]


def positive_mhz(text):
    """--freq-mhz value: a plain positive number (it is passed to a shell script, so nothing else is accepted)."""
    if not re.fullmatch(r"\d+(\.\d+)?", text or "") or float(text) <= 0:
        raise argparse.ArgumentTypeError(f"'{text}' is not a positive number of MHz (for example 5)")
    return float(text)


def write_report(results, cfg, versions):
    """One report for all operators: a run replaces only the operators it ran (each result keeps its own run_at), so
    running --op relu8 after --op all keeps add8_sat and requant_i32_i8. Times are local, with the UTC offset."""
    chip = getattr(cfg, "target", "demo") == "chip"
    target = getattr(cfg, "target", "demo")
    path = cc.REPORT if chip else REPORT
    path.parent.mkdir(parents=True, exist_ok=True)
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    merged = {}
    try:
        prev = json.loads(path.read_text(encoding="utf-8"))
        if prev.get("target", "demo") == target:
            merged = {r["op"]: r for r in prev.get("results", []) if isinstance(r, dict) and r.get("op")}
    except (OSError, ValueError):
        pass
    for r in results:
        r.setdefault("run_at", now)
        merged[r["op"]] = r
    if chip:
        from x2c.chip import x2_facts
        clock = {"clock_target_mhz": x2_facts.value("clock_mhz"),     # what PASS is judged against (project clock)
                 "nextpnr_target_mhz": 12.0,                          # what nextpnr itself was told (the repo's built-in target)
                 "clock_note": "timing PASS means nextpnr's fmax >= clock_target_mhz; nextpnr runs with its 12 MHz default"}
    else:
        freq = getattr(cfg, "freq_mhz", None)          # demo fabric: None = no --freq-mhz given (nextpnr's own 12 MHz)
        clock = {"clock_target_mhz": freq, "nextpnr_target_mhz": freq or 12.0}
    path.write_text(json.dumps({"generated": now, "target": target, "runner": cfg.runner, "distro": cfg.distro,
                                "tool_versions": versions, **clock,
                                "ops_this_run": [r["op"] for r in results],
                                "results": [merged[k] for k in sorted(merged)],
                                "evidence_level": cc.EVIDENCE_LEVEL if chip else fc.EVIDENCE_LEVEL},
                               indent=2, default=str), encoding="utf-8")


def main(argv=None, exec_fn=fc.run_command, host_reader=wp.read_host_memory):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--op", action="append", required=True, help="operator (repeat, or comma-separated), or 'all'")
    ap.add_argument("--from-stage", choices=fc.STAGES, default="rtl")
    ap.add_argument("--runner", choices=["auto", "wsl", "native"], default="auto")
    ap.add_argument("--distro", default="Ubuntu")
    ap.add_argument("--llm", choices=["mock", "groq", "gemini"], default="mock")
    ap.add_argument("--model", default=None)
    ap.add_argument("--regen-rtl", action="store_true", help="ask A2 for new Verilog even if hardware/operators/<op>.v already passes")
    ap.add_argument("--timeout", type=int, default=1800, help="seconds allowed for each Ubuntu stage")
    ap.add_argument("--heartbeat", type=int, default=30, help="seconds between 'still running' lines")
    ap.add_argument("--plan", action="store_true", help="show the stages and exact commands, run nothing")
    ap.add_argument("--freq-mhz", type=positive_mhz, default=None,
                    help="clock target for nextpnr in MHz (default: nextpnr's own, 12 MHz). The result is only valid for this target")
    ap.add_argument("--ignore-memory", action="store_true", help="do not check Windows' free memory before the env and sim stages")
    ap.add_argument("--target", choices=["demo", "chip"], default="demo",
                    help="demo: the FABulous demo fabric (28 pins, ~/work/demo). chip: the BM Labs chip, sky130-reram-fpga (36 pins, chip flow)")
    a = ap.parse_args(argv)

    chip = a.target == "chip"
    wrappable, unwrappable = cc.wrappable_ops() if chip else wrappable_ops()
    ops = []
    for item in a.op:
        for o in item.split(","):
            o = o.strip()
            if o == "all":
                ops += [x for x in wrappable if x not in ops]
                for x, why in unwrappable.items():
                    say(f"skipping {x}: {why}")
            elif o in unwrappable:
                sys.exit(f"cannot wrap {o}: {unwrappable[o]}." + ("" if chip else " Only a single wide input can be loaded in two steps; not built for this one."))
            elif o in OPS:
                ops += [o] if o not in ops else []
            else:
                sys.exit(f"unknown op '{o}'. Available: {', '.join(OPS)}, all")
    runner = fc.resolve_runner(a.runner)
    cfg = SimpleNamespace(runner=runner, distro=a.distro, llm=a.llm, model=a.model, regen_rtl=a.regen_rtl,
                          timeout_s=a.timeout, heartbeat_s=a.heartbeat, from_stage=a.from_stage,
                          check_memory=(runner == "wsl" and not a.ignore_memory), host_reader=host_reader, sample_s=0.5, freq_mhz=a.freq_mhz,
                          target=a.target)
    if chip and a.freq_mhz:
        sys.exit("--freq-mhz is for the demo fabric; the chip flow uses the repo's 12 MHz nextpnr target")
    if a.plan:
        for op in ops:
            print("\n".join(cc.plan_text(op, cfg) if chip else plan_text(op, cfg)))
        return 0
    if cfg.runner == "wsl" and exec_fn is fc.run_command and shutil.which("wsl") is None:
        sys.exit("wsl.exe was not found. Run this from Windows with WSL installed, or from a Linux shell with --runner native")

    preflight, results = {}, []
    for op in ops:
        say(f"\n{'CHIP' if chip else 'FAB'} CHAIN  op={op}  runner={cfg.runner}" + (f" ({cfg.distro})" if cfg.runner == "wsl" else "") + f"  from={cfg.from_stage}")
        removed = cc.clean_job(op, cfg.from_stage) if chip else fc.clean_job(JOBS / op, op, cfg.from_stage)
        if removed:
            say(f"  cleared old results: {', '.join(removed)}")
        steps = build_steps(op, cfg, exec_fn, preflight)
        if chip:                                   # same rtl stage (A2 + Yosys vs NumPy); the other five run on the chip
            steps = cc.build_chip_steps(op, cfg, exec_fn, preflight, steps["rtl"], say)
        r = fc.run_chain(op, steps, cfg.from_stage, say)
        results.append(r)
        say("\n".join(cc.format_summary(r) if chip else fc.format_summary(r)))
        for line in diagnose(op, r, a.target):                  # A3's evidence rules (no LLM): demo or chip causes
            say(line)
    versions = (preflight.get("info") or {}).get("versions", {})
    write_report(results, cfg, versions)
    if len(results) > 1:
        say("\nALL OPERATORS: " + " | ".join(f"{r['op']} {r['verdict']}" for r in results))
    say(f"report: {(cc.REPORT if chip else REPORT).as_posix()}\n[{cc.EVIDENCE_LEVEL if chip else fc.EVIDENCE_LEVEL}]")
    return 0 if all(r["verdict"].startswith("PASS") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
