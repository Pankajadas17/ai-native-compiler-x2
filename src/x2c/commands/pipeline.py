"""
x2c pipeline: the WHOLE pipeline, in order, with ONE command, and one summary report at the end.

    x2c pipeline                  # Ubuntu terminal (the chip tools live there), about 10 minutes
    x2c pipeline --rebuild        # also rebuild the engine bitstream (+10-15 min)
    x2c pipeline --with-tests     # also the full test suite (+20 min)
    x2c pipeline --only deploy,replay   /   --skip formal

Stages (all must pass for PASS; a failing stage does not stop the rest, the report shows all):
  1 environment  tools + versions, chip repo found and pinned, BMsemi_Model found
  2 operators    A2's eFPGA operators through the chip chain (Yosys vs NumPy, chip flow, bitstream, V0 audit, netlist sim)
  3 formal       proofs for all inputs (operators, chip netlist), Verilator lint, bounded engine check
  4 engine       the deployment engine bitstream (x2_engine2_4bit): reused if its source is unchanged, else rebuilt
  5 x2-model     the Python X2 golden == the X2 developer's Verilog model
  6 engine-sim   the two-block engine in simulation, exact, cycles vs the A1 cost model
  7 deploy       A1 plan + schedule, INT8 network, every tensor exact: MobileNetV2 + MNIST on the engine RTL, DeepGate
  8 replay       each bundle re-run from its files alone
  9 status       A4's chip status (missing X2 facts included)
 (tests)         optional: the full test suite
On a failure the stage line names the log, and A3's rule diagnosis is shown where the stage produces one.
Saved: reports/pipeline/REPORT.md (read this) + report.json; each stage keeps its own detailed files.
"""
from x2c import paths
import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = paths.ROOT
OUT = ROOT / "reports" / "pipeline"
PY = sys.executable
WORKLOADS = ("mobilenetv2", "mnist", "deepgate")


def _json(p):
    try:
        return json.loads((ROOT / p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _run(name, cmd, log):
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    log.write_text(r.stdout + r.stderr, encoding="utf-8")
    return r.returncode, round(time.time() - t0, 1)


def _tail(log):
    lines = [l for l in Path(log).read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()] if Path(log).is_file() else []
    return (lines[-1] if lines else "no output")[:160]


def _strip_formal(text):
    return re.sub(r"`ifdef FORMAL.*?`endif\n?", "", text, flags=re.S)


# ---------------------------------------------------------------- stages: (key, title, who, runner)
def st_environment(ctx):
    from x2c.chip import chip_flow as cf
    from x2c.chip.chip_repo import ChipRepo, x2_model_label
    env = cf.tool_env()
    tools = {}
    for t in ("yosys", "nextpnr-generic", "iverilog", "verilator", "sby"):
        p = shutil.which(t, path=env["PATH"])
        if p:
            r = subprocess.run([p, {"yosys": "-V", "iverilog": "-V"}.get(t, "--version")], capture_output=True, text=True)
            tools[t] = next((l.strip() for l in (r.stdout + r.stderr).splitlines() if l.strip()), "?")[:60]
        else:
            tools[t] = None
    repo = ChipRepo()
    ctx["fingerprint"] = repo.tree_fingerprint()
    ctx["repo"] = repo
    try:
        import BMsemi_Model  # noqa: F401
        bm = True
    except ImportError:
        bm = False
    pkgs = []
    for mod in ("numpy",) + (("pytest",) if ctx.get("need_pytest") else ()):    # pytest only for --with-tests
        try:
            __import__(mod)
        except ImportError:
            pkgs.append(mod)
    if pkgs:
        ctx["log"].write_text(f"python {sys.executable} has no {pkgs}\n", encoding="utf-8")
        return False, {"python": sys.executable, "missing_packages": pkgs}, (
            f"this python ({sys.executable}) has no {', '.join(pkgs)}: activate the project's environment first "
            f"(source .venv/bin/activate) and run again")
    missing = [t for t, v in tools.items() if v is None]
    ctx["public"] = not bm               # public clone: no private BMsemi_Model -> public stand-in graphs for the deploy stage
    pinned = _json("specifications/external_repos.json") or {}
    facts = {"tools": tools, "pinned_clones": {k: (v or {}).get("commit") for k, v in pinned.items() if isinstance(v, dict)}, "chip_repo_commit": repo.commit(), "x2_model": x2_model_label(), "bmsemi_model": bm,
             "python": platform.python_version()}
    ok = not missing
    facts["mode"] = "team (BMsemi_Model present)" if bm else "PUBLIC (no BMsemi_Model: public stand-in graphs, BMsemi-only checks skipped)"
    ctx["log"].write_text(json.dumps(facts, indent=1, default=str) + "\n", encoding="utf-8")
    return ok, facts, ("missing: " + ", ".join(missing)) if not ok else \
        (f"Yosys {tools['yosys'].split()[1]}, chip repo {str(repo.commit())[:10]}, X2 model {x2_model_label()}"
         + ("" if bm else " | PUBLIC mode: BMsemi_Model not present"))


def st_operators(ctx):
    rc, sec = _run("operators", [PY, "-m", "x2c", "chain", "--op", "all", "--target", "chip", "--runner", "native"], ctx["log"])
    rep = _json("reports/chip_chain_report.json") or {}
    res = {r["op"]: r for r in rep.get("results", [])}
    v = {op: r.get("verdict") for op, r in res.items()}
    diag = {op: (r.get("diagnosis") or {}).get("rule_based_candidates") for op, r in res.items() if r.get("diagnosis")}
    facts = {"verdicts": v, "a3_diagnosis": diag, "clock_target_mhz": rep.get("clock_target_mhz"),
             "facts": {op: {"v0_audit": "{set_bits}/{required_bits}".format(**((r.get("facts") or {}).get("bitstream_audit") or
                                                                                   {"set_bits": "?", "required_bits": "?"})),
                            "fmax_mhz": [v.get("fmax_mhz") for v in ((r.get("facts") or {}).get("fmax") or {}).values()],
                            "bitstream_bytes": (r.get("facts") or {}).get("bitstream_bytes")} for op, r in res.items()}}
    ok = rc == 0 and len(v) == 3 and all(x == "PASS" for x in v.values())
    if rc != 0 and not diag:
        return False, facts, f"the chain did not run: {_tail(ctx['log'])}"
    return ok, facts, " | ".join(f"{op} {x}" for op, x in sorted(v.items())) + (f" | A3: {diag}" if diag else "")


def st_formal(ctx):
    rc, sec = _run("formal", [PY, "-m", "x2c", "formal"], ctx["log"])
    rep = _json("reports/formal/report.json") or {}
    rows = rep.get("rows", [])
    bug = next((r for r in rows if "REPO's multiplier map" in r["check"]), {})
    facts = {"checks": f"{sum(r['ok'] for r in rows)}/{len(rows)}", "engine_depth": rep.get("engine_depth"),
             "repo_map_counterexample": bug.get("detail"), "failed": [r["check"] for r in rows if not r["ok"]]}
    if rc != 0 and not rows:
        return False, facts, f"did not run: {_tail(ctx['log'])}"
    return rc == 0 and rep.get("ok") is True, facts, \
        f"{facts['checks']} as expected (operators proved for all inputs, chip netlists equivalent, lint clean, engine bounded to {rep.get('engine_depth')} clocks)"


def st_engine(ctx):
    d = ROOT / "chip_build" / "x2_engine2_4bit"
    srcs = {"x2_engine2.v": ROOT / "hardware/designs/x2_engine2.v", "x2_engine2_4bit.v": ROOT / "hardware/designs/x2_engine2_4bit.v"}
    rep = _json("chip_build/x2_engine2_4bit/flow_report.json") or {}
    same = rep.get("status") == "ok" and all((d / "rtl" / n).is_file() and _strip_formal((d / "rtl" / n).read_text()) ==
                                             _strip_formal(p.read_text()) for n, p in srcs.items())
    action = "reused (source unchanged since the build)"
    if ctx["rebuild"] or not same:
        rc, sec = _run("engine", [PY, "-m", "x2c", "build", "--example", "x2_engine2_4bit"], ctx["log"])
        rep = _json("chip_build/x2_engine2_4bit/flow_report.json") or {}
        action = "rebuilt"
    from x2c.verification.chip_tools import chip_status
    s = (chip_status().get("designs") or {}).get("x2_engine2_4bit", {})
    facts = {"action": action, **{k: s.get(k) for k in ("logic_cells", "pins", "x2_blocks", "v0_audit", "fmax_mhz", "meets_project_clock")},
             "bitstream_bytes": rep.get("bitstream_bytes")}
    ok = rep.get("status") == "ok" and s.get("meets_project_clock") is True and s.get("v0_audit") and \
        s["v0_audit"].split("/")[0] == s["v0_audit"].split("/")[1]
    return bool(ok), facts, (f"{action}: {s.get('logic_cells')}/2160 cells, {s.get('pins')}/36 pins, both X2, "
                             f"V0 audit {s.get('v0_audit')}, {s.get('fmax_mhz')} MHz >= 14")


def st_x2_model(ctx):
    rc, sec = _run("x2-model", [PY, "-m", "x2c", "sim", "x2", "--sim"], ctx["log"])
    tail = [l for l in ctx["log"].read_text().splitlines() if l.strip()][-3:]
    return rc == 0, {"log_tail": tail}, tail[-1][:150] if tail else "no output"


def st_engine_sim(ctx):
    rc, sec = _run("engine-sim", [PY, "-m", "x2c", "sim", "engine2", "--sim"], ctx["log"])
    from x2c.verification.chip_tools import chip_status
    t = chip_status().get("two_block_engine") or {}
    if rc != 0:
        return False, t, f"simulation failed: {_tail(ctx['log'])}"
    return rc == 0 and t.get("all_pass") is True, t, (f"all exact, latency cut {t.get('latency_cut_percent')} %, "
                                                       f"cost model error <= {t.get('max_model_error_percent')} %")


PUBLIC_DEPLOY = {"mobilenetv2": ["--ir", "tests/fixtures/ir_public/invres_net.json"],          # public stand-ins
                 "mnist": ["--ir", "tests/fixtures/ir_public/edge5x5.json", "--synthetic"],
                 "deepgate": ["--ir", "tests/fixtures/ir_public/mlp_chain.json", "--synthetic"]}


def st_deploy(ctx):
    facts, ok = {}, True
    for wl in WORKLOADS:
        log = ctx["log"].with_name(f"deploy_{wl}.log")
        extra = PUBLIC_DEPLOY[wl] if ctx.get("public") else []
        rc, sec = _run(f"deploy {wl}", [PY, "-m", "x2c", "deploy", "--workload", wl, "--sim", *extra], log)
        r = _json(f"deploy/{wl}/report.json") or {}
        sim = r.get("simulation_1bit") or {}
        facts[wl] = {"ok": r.get("ok"), "model": r.get("model"), "checks": r.get("checks"), "x2_layers": r.get("x2_layers"),
                     "rtl_layers": f"{sim.get('layers_simulated')}/{r.get('x2_layers')}",
                     "cycles_measured": sim.get("total_cycles_measured"), "cycles_predicted": sim.get("total_cycles_predicted"),
                     "latency_us": (r.get("a1") or {}).get("latency_us"), "seconds": sec}
        ok &= rc == 0 and r.get("ok") is True
        if rc != 0:
            facts[wl]["error"] = ([l for l in log.read_text().splitlines() if l.strip()] or ["?"])[-1][:160]
    tag = " [public stand-in graphs]" if ctx.get("public") else ""
    return ok, facts, tag.strip() + (" " if tag else "") + " | ".join(f"{wl} {'VERIFIED' if f['ok'] else 'FAILED'}" + (f" (RTL {f['rtl_layers']} layers)" if f['ok'] else
                                 f" ({f.get('error', 'see log')})") for wl, f in facts.items())


def st_replay(ctx):
    rc, sec = _run("replay", [PY, "-m", "x2c", "replay", *[f"deploy/{wl}" for wl in WORKLOADS]], ctx["log"])
    facts = {wl: (_json(f"deploy/{wl}/replay_report.json") or {}).get("exact") for wl in WORKLOADS}
    if rc != 0 and not any(facts.values()):
        return False, facts, f"replay did not run: {_tail(ctx['log'])}"
    word = {True: "exact", False: "MISMATCH", None: "no bundle (deploy failed?)"}
    return rc == 0 and all(facts.values()), facts, " | ".join(f"{wl} {word[v]}" for wl, v in facts.items())


def st_status(ctx):
    from x2c.verification.chip_tools import chip_status
    s = chip_status()
    (OUT / "chip_status.json").write_text(json.dumps(s, indent=1, default=str), encoding="utf-8")
    if not ctx.get("repo"):
        facts = {"x2_facts_missing": s.get("x2_facts_missing"), "chip_repo_unchanged": None}
        return True, facts, (f"read-only check not compared (environment stage skipped); X2 facts still missing: "
                             f"{', '.join(s.get('x2_facts_missing') or []) or 'none'}")
    unchanged = ctx["repo"].tree_fingerprint() == ctx.get("fingerprint")
    facts = {"x2_facts_missing": s.get("x2_facts_missing"), "chip_repo_unchanged": unchanged}
    return unchanged, facts, f"chip repo unchanged: {unchanged}; X2 facts still missing: {', '.join(s.get('x2_facts_missing') or []) or 'none'}"


def st_tests(ctx):
    rc, sec = _run("tests", [PY, "-m", "pytest", "-q"], ctx["log"])
    last = [l for l in ctx["log"].read_text().splitlines() if l.strip()][-1:]
    return rc == 0, {"summary": last}, last[0] if last else "no output"


STAGES = [
    ("environment", "Environment and inputs", "tools", st_environment),
    ("operators", "eFPGA operators on the chip", "A2 + chip flow + A3 rules", st_operators),
    ("formal", "Formal proofs and lint", "Yosys SAT, SymbiYosys, Verilator", st_formal),
    ("engine", "Deployment engine bitstream", "chip flow + V0 audit", st_engine),
    ("x2-model", "X2 golden vs the X2 developer's model", "Icarus", st_x2_model),
    ("engine-sim", "Two-block engine in simulation", "Icarus + A1 cost model", st_engine_sim),
    ("deploy", "Model deployment (3 models)", "A1 + compiler + X2 path + engine RTL", st_deploy),
    ("replay", "Bundles replayed from their files", "x2c replay", st_replay),
    ("status", "Chip status and read-only check", "A4 tools", st_status),
    ("tests", "Full test suite", "pytest", st_tests),
]
OPS = ("relu8", "add8_sat", "requant_i32_i8")
STAGE_FILES = {      # stage -> the files it saves its results in (globs, relative to the project)
    "environment": ["reports/pipeline/01_environment.log", "specifications/chip_profile.json", "specifications/external_repos.json"],
    "operators": ["reports/chip_chain_report.json"] + [p for op in OPS for p in (
        f"chip_build/chain_{op}/flow_report.json", f"chip_build/chain_{op}/bitstream_audit.json", f"chip_build/chain_{op}/{op}_chip.bit",
        f"reports/chip_jobs/{op}/chip_check.json", f"reports/chip_jobs/{op}/sim.log")],
    "formal": ["reports/formal/report.json", "chip_build/formal/netlist/repo_map/prove/yosys.log",
               "chip_build/formal/engine/bit4/sby_console.log", "chip_build/formal/engine/mutant_no_forwarding/check/engine_0/trace*.vcd"],
    "engine": [f"chip_build/x2_engine2_4bit/{f}" for f in ("flow_report.json", "bitstream_audit.json", "x2_engine2_4bit.bit",
                                                          "design.pcf", "pnr.log")],
    "x2-model": ["chip_build/x2_check/report.json"],
    "engine-sim": ["chip_build/x2_engine2_sim/report.json", "chip_build/x2_engine2_sim/engine2_*_alternate.log"],
    "deploy": [f"deploy/{wl}/{f}" for wl in WORKLOADS for f in ("report.json", "plan.json", "host_program.json", "x2_program.npz",
                                                               "expected.npz")] + ["deploy/mobilenetv2/engine/x2_engine2_4bit.bit"],
    "replay": [f"deploy/{wl}/replay_report.json" for wl in WORKLOADS],
    "status": ["reports/pipeline/chip_status.json"],
    "tests": ["reports/pipeline/10_tests.log"],
}


def stage_files(key, since):
    """Where the stage's results are saved: written by this run, reused from an earlier run (with its date), or not found."""
    out = []
    for pat in STAGE_FILES.get(key, []):
        hits = sorted(ROOT.glob(pat)) or [None]
        for h in hits:
            if h is None:
                out.append({"file": pat, "state": "not found"})
                continue
            st = h.stat()
            out.append({"file": h.relative_to(ROOT).as_posix(), "bytes": st.st_size,
                        "state": "saved" if st.st_mtime >= since - 1 else "reused from " + time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))})
    return out


NOT_PROVEN = ["the configured fabric (the chip's tile RTL loaded with the bitstream) in simulation",
              "timing-annotated simulation (fmax is nextpnr's model)", "silicon",
              "the 4-bit cell rule (ASSUMED sum(pwm*level); the chip repo's X2 model is 1-bit)",
              "MobileNetV2 with trained weights (seeded INT8 weights unless --weights)",
              "the engine beyond 10 clocks after reset (bounded check)"]


def write_report(rows, ctx, total):
    ok = all(r["ok"] for r in rows if r["status"] == "ran")
    ran = [r for r in rows if r["status"] == "ran"]
    core_skipped = [r["key"] for r in rows if r["status"] != "ran" and r["key"] != "tests"]
    verdict = "FAIL" if not ok or not ran else ("PASS" if not core_skipped else f"PARTIAL (skipped: {', '.join(core_skipped)})")
    data = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "verdict": verdict,
            "stages_passed": f"{sum(r['ok'] for r in ran)}/{len(ran)}", "seconds": total, "stages": rows, "not_proven": NOT_PROVEN}
    (OUT / "report.json").write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    mode = "PUBLIC mode (no private BMsemi_Model: public stand-in graphs)" if ctx.get("public") else "team mode (BMsemi_Model present)"
    data["mode"] = mode
    (OUT / "report.json").write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    L = [f"# Pipeline report: {data['verdict']} ({data['stages_passed']} stages)", "",
         f"{data['generated']} · {total / 60:.1f} min · `x2c pipeline` · {mode}", "",
         "| # | stage | done by | result | what was found | log |", "|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        res = {"ran": "**PASS**" if r["ok"] else "**FAIL**"}.get(r["status"], r["status"])
        L.append(f"| {i} | {r['title']} | {r['who']} | {res} | {r['summary'].replace('|', '·')} | `{r['log']}` |")
    L += ["", "## Where each stage's results are saved", "",
          "saved here = written by this run · reused from <date> = kept from an earlier run · NOT FOUND = expected but not there", ""]
    for i, r in enumerate(rows, 1):
        if r["status"] != "ran":
            continue
        L.append(f"**{i}. {r['title']}: {'PASS' if r['ok'] else 'FAIL'}**")
        L += [f"- {({'saved': 'saved here', 'not found': 'NOT FOUND'}).get(e['state'], e['state'])}: `{e['file']}`"
              for e in r.get("files", [])] + [""]
    L += ["## What this does NOT prove", ""] + [f"- {n}" for n in NOT_PROVEN]
    L += ["", "## Stage facts", ""] + [f"- **{r['title']}**: `{json.dumps(r['facts'], default=str)[:600]}`" for r in rows if r["status"] == "ran"]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return data


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuild", action="store_true", help="rebuild the engine bitstream even if its source is unchanged")
    ap.add_argument("--with-tests", action="store_true", help="also run the full test suite")
    ap.add_argument("--only", help="comma list of stage keys")
    ap.add_argument("--skip", help="comma list of stage keys")
    a = ap.parse_args(argv)
    if platform.system() != "Linux":
        print("the pipeline needs the chip tools: run it in the Ubuntu terminal:  x2c pipeline")
        return 2
    keys = [k for k, *_ in STAGES]
    only = set(a.only.split(",")) if a.only else set(keys) - ({"tests"} if not a.with_tests else set())
    skip = set(a.skip.split(",")) if a.skip else set()
    bad = (only | skip) - set(keys)
    if bad:
        ap.error(f"unknown stage(s) {sorted(bad)}; stages: {', '.join(keys)}")
    OUT.mkdir(parents=True, exist_ok=True)
    ctx, rows, t0 = {"rebuild": a.rebuild, "need_pytest": "tests" in only and "tests" not in skip}, [], time.time()
    print(f"PIPELINE  {time.strftime('%Y-%m-%d %H:%M')}  ({len([k for k in keys if k in only and k not in skip])} stages)")
    for i, (key, title, who, fn) in enumerate(STAGES, 1):
        log = OUT / f"{i:02d}_{key}.log"
        row = {"key": key, "title": title, "who": who, "log": log.relative_to(ROOT).as_posix(), "status": "skipped",
               "ok": None, "summary": "skipped", "facts": {}}
        if key in only and key not in skip:
            print(f"[{i}/{len(STAGES)}] {title} ...", end="", flush=True)
            ctx["log"] = log
            t = time.time()
            try:
                ok, facts, summary = fn(ctx)
            except Exception as e:  # noqa: BLE001  (a broken stage is reported, the rest still runs)
                ok, facts, summary = False, {"error": f"{type(e).__name__}: {e}"}, f"error: {type(e).__name__}: {e}"
                if not log.exists():
                    log.write_text(summary + "\n", encoding="utf-8")
            row.update(status="ran", ok=bool(ok), summary=summary, facts=facts, seconds=round(time.time() - t, 1),
                       files=stage_files(key, t))
            print(f" {'PASS' if ok else 'FAIL'} ({row['seconds']} s)\n      {summary}")
            for e in row["files"]:
                verb = {"saved": "saved here", "not found": "NOT FOUND"}.get(e["state"], e["state"])
                print(f"        {verb}: {e['file']}")
            if not ok:
                print(f"      log: {row['log']}   (A3 second opinion: x2c agent a3 --target chip)")
            if key == "environment" and not ok:
                rows.append(row)
                print("      the environment is not ready: the other stages were not run")
                for k2, t2, w2, _ in STAGES[i:]:
                    rows.append({"key": k2, "title": t2, "who": w2, "log": "", "status": "not run (environment failed)",
                                 "ok": None, "summary": "not run", "facts": {}})
                break
        rows.append(row)
    data = write_report(rows, ctx, round(time.time() - t0, 1))
    print(f"\nPIPELINE {data['verdict']}: {data['stages_passed']} stages in {data['seconds'] / 60:.1f} min")
    print(f"summary saved here: {(OUT / 'REPORT.md').relative_to(ROOT).as_posix()}  (+ report.json, one log per stage)")
    print("not proven: " + "; ".join(NOT_PROVEN[:3]) + " ... (see REPORT.md)")
    return 0 if data["verdict"] != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
