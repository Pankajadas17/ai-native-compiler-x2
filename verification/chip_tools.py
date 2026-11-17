"""
Chip facts for the verification agent (A4): what the BM Labs chip work has verified, read from the reports by code.

chip_status      reads (never runs): chip chain verdicts, every chip design's build (logic cells, X2 blocks, pins, V0 audit,
                 nextpnr Fmax vs the project clock), the two-block engine check, the real-layer end-to-end check, the X2
                 facts (with what is still missing). A report that does not exist yet is listed with the command that makes it.
chip_schedule    today's vs the best X2 schedule for the real tiny MobileNetV2 (cycles and microseconds at the project clock).
run_chip_layers  the 4-bit end-to-end check of every real layer in Python (ASSUMED 4-bit cell rule): fast, works on Windows.
Every number comes from a file or from code; the LLM only chooses which tool to call.
"""
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAIN_REPORT = "reports/chip_chain_report.json"
DESIGNS = ("x2_engine", "x2_engine_4bit", "x2_engine2", "x2_engine2_4bit")
ENGINE2_REPORT = "chip_build/x2_engine2_sim/report.json"
LAYERS_SIM_REPORT = "chip_build/x2_layer_e2e/report.json"
LAYERS_PY_REPORT = "reports/chip_layer_e2e/report.json"
HOW = {"chain": "python run_fab_chain.py --op all --target chip   (VS Code terminal)",
       "design": "python3 run_chip_flow.py --example {name}   (Ubuntu terminal)",
       "engine2": "python3 run_x2_engine2_check.py --sim   (Ubuntu terminal)",
       "layers_sim": "python3 run_x2_layer_e2e.py --sim   (Ubuntu terminal)",
       "layers_py": "the run_chip_layers tool, or python run_x2_layer_e2e.py --out reports/chip_layer_e2e"}
EVIDENCE = ("reading of chip reports written by the chip flow, the chain and the simulations; the 4-bit results use the "
            "ASSUMED 4-bit cell rule; NOT the configured fabric, NOT silicon")


def _json(rel, root):
    p = Path(root) / rel
    try:
        return json.loads(p.read_text(encoding="utf-8")), time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))
    except (OSError, ValueError):
        return None, None


def _design(name, root, clock):
    from verification.chip_chain import final_timing
    rep, when = _json(f"chip_build/{name}/flow_report.json", root)
    if rep is None:
        return {"status": "not run", "how": HOW["design"].format(name=name)}
    u = rep.get("utilisation") or {}
    pnr = Path(root) / "chip_build" / name / "pnr.log"
    t = final_timing(pnr.read_text(errors="replace")) if pnr.is_file() else {}
    fmax = min((v["fmax_mhz"] for v in t.values()), default=None)
    audit = rep.get("bitstream_audit") or {}
    return {"status": rep.get("status"), "when": when,
            "logic_cells": (u.get("FABULOUS_LC") or {}).get("used"), "x2_blocks": (u.get("RERAM_CIM_32x32") or {}).get("used"),
            "pins": (rep.get("pins") or {}).get("needed"),
            "v0_audit": f"{audit.get('set_bits')}/{audit.get('required_bits')}" if audit else None,
            "fmax_mhz": fmax, "meets_project_clock": None if fmax is None else fmax >= clock}


def chip_status(root=ROOT):
    from chip import x2_facts
    clock = x2_facts.value("clock_mhz")
    out = {"project_clock_mhz": clock, "evidence_level": EVIDENCE}
    chain, when = _json(CHAIN_REPORT, root)
    out["chain"] = ({"when": when, "verdicts": {r["op"]: r["verdict"] for r in chain.get("results", [])},
                     "all_pass": all(str(r["verdict"]).startswith("PASS") for r in chain.get("results", []))}
                    if chain else {"status": "not run", "how": HOW["chain"]})
    out["designs"] = {n: _design(n, root, clock) for n in DESIGNS}
    e2, when = _json(ENGINE2_REPORT, root)
    if e2:
        alt = [r for r in e2["rows"] if r.get("layout") == "alternate" and r.get("measured_cut_percent") is not None]
        out["two_block_engine"] = {"when": when, "all_pass": e2["all_pass"],
                                   "latency_cut_percent": {r["layer"]: r["measured_cut_percent"] for r in alt},
                                   "max_model_error_percent": max((abs(r["model_error_percent"]) for r in e2["rows"]
                                                                   if r.get("model_error_percent") is not None), default=None)}
    else:
        out["two_block_engine"] = {"status": "not run", "how": HOW["engine2"]}
    for key, rel, how in (("layers_simulated", LAYERS_SIM_REPORT, HOW["layers_sim"]), ("layers_4bit_python", LAYERS_PY_REPORT, HOW["layers_py"])):
        rep, when = _json(rel, root)
        out[key] = _layers(rep, when) if rep else {"status": "not run", "how": how}
    f = x2_facts.facts()
    out["x2_facts"] = {k: {"value": v["value"], "status": v["status"]} for k, v in f.items()}
    out["x2_facts_missing"] = sorted(k for k, v in f.items() if v["status"] == "missing")
    return out


def _layers(rep, when):
    rows = rep.get("rows", [])
    sim = [r for r in rows if "alternate" in r.get("bits1", {})]
    cuts = [r["bits1"]["latency_cut_percent"] for r in sim if r["bits1"].get("latency_cut_percent") is not None]
    errs = [r["bits1"]["max_model_error_percent"] for r in sim if r["bits1"].get("max_model_error_percent") is not None]
    return {"when": when, "all_exact": rep.get("all_exact"), "layers": len(rows),
            "4bit_exact": sum(1 for r in rows if r.get("bits4", {}).get("exact")),
            "1bit_simulated": len(sim), "1bit_skipped": sum(1 for r in rows if "skipped" in r.get("bits1", {})),
            "latency_cut_percent_range": [min(cuts), max(cuts)] if cuts else None,
            "max_model_error_percent": max(errs) if errs else None,
            "cell_rule_4bit": "ASSUMED sum(pwm*level) (chip/x2_facts.py cell_compute_4bit is missing)"}


def chip_schedule():
    """Best X2 schedule for the real tiny MobileNetV2 (code's own search) vs today's (resident + one streaming block)."""
    from compiler import x2_schedule as xs
    from compiler.ir_builder import build_ir_from_compiler_graph
    from compiler.partition import load_hardware, propose_partition_baseline
    from inputs.bmsemi_source import ReadOnlyRepo
    from inputs.normalize import compiler_v2_workload
    r = ReadOnlyRepo()
    g, hw = build_ir_from_compiler_graph(r, compiler_v2_workload(r)), load_hardware()
    chip = propose_partition_baseline(g, hw)["chip"]
    best, bm, tried = xs.best_schedule(g, chip, hw)
    _, tm = xs.validate_schedule(g, chip, hw, xs.materialize(g, chip, hw, chip["resident"], "single_block"))
    pick = lambda m: {k: m[k] for k in ("latency_cycles", "latency_us", "program_cycles_per_inference_upper",
                                        "compute_cycles", "blocks_used")}
    return {"graph": g.name, "x2_tiles": chip["totals"]["tiles"], "candidates_tried": tried,
            "today": {"resident": chip["resident"], "policy": "single_block", **pick(tm)},
            "best": {"resident": best["resident"], "policy": best["policy"], **pick(bm)},
            "latency_reduction_percent": round(100 * (1 - bm["latency_cycles"] / tm["latency_cycles"]), 1),
            "assumptions": xs.ASSUMPTIONS, "cost_model": xs.CALIBRATION,
            "evidence_level": "calibrated behavioral-model schedule estimate (two-block overlap measured in simulation); NOT silicon"}


def run_chip_layers(positions=2, out=LAYERS_PY_REPORT):
    """4-bit end-to-end check of every real layer in Python. Writes reports/chip_layer_e2e/report.json."""
    import run_x2_layer_e2e as le
    if not 1 <= int(positions) <= 16:
        raise ValueError("positions must be between 1 and 16")
    folder = Path(out).parent
    code = le.main(["--positions", str(int(positions)), "--out", str(folder)])
    p = Path(out)
    rep = json.loads(p.read_text(encoding="utf-8"))
    when = time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))
    return {"exit_code": code, "report_path": str(out), **_layers(rep, when)}
