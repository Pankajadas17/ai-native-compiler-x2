"""
Deploy an AI model onto the BM Labs chip (sky130-reram-fpga: X2 crossbars + FABulous eFPGA) and verify the deployment.

    x2c deploy --workload mobilenetv2                 # VS Code terminal: plan + 4-bit X2 path + bundle
    x2c deploy --workload deepgate                    #   (real INT8 weights and requantization from BMsemi_Model)
    x2c deploy --workload mnist
    x2c deploy --workload mobilenetv2 --sim          # Ubuntu terminal: also every layer on the 1-bit two-block engine
    x2c deploy --workload mobilenetv2 --ir reports/ir_mobilenetv2.json --synthetic   # without BMsemi_Model

What it does (the agents' roles in brackets):
  1. IR of the model (BMsemi compiler graph / chain workload, or --ir JSON)
  2. [A1] partition (matrix ops -> X2, the rest -> eFPGA/host) + the X2 tile schedule over both blocks
  3. INT8 network: module kinds inferred, BN folded, ReLU6 -> clamp, residual adds, pool (compiler/int8_net.py)
  4. run on the X2 path, every tensor compared with an independent golden model:
       4-bit cells (the project's setting) in Python, ASSUMED cell rule sum(pwm*level);  --sim: 1-bit on the two-block
       engine in Icarus (the chip repo's X2 model), cycles measured
  5. [A4] chip status: the engine bitstream (x2_engine2_4bit: 35/36 pins, both X2, audited) and what is still missing
  6. bundle deploy/<workload>/: plan.json, host_program.json, x2_program.npz (cell levels per tile), expected.npz,
     report.json, and the engine bitstream when it has been built (Ubuntu: x2c build --example x2_engine2_4bit)
The engine bitstream is the same for every model; a model is deployed by programming the X2 cells and running the host program.
Evidence: Python X2 model (4-bit) + RTL simulation with the BM behavioral X2 (1-bit); NOT the configured fabric, NOT silicon.
"""
from x2c import paths
import argparse
import copy
import json
import shutil
from collections import Counter
import sys
import time
from pathlib import Path

import numpy as np

ROOT = paths.ROOT

from x2c.compiler import int8_net as N  # noqa: E402
from x2c.compiler import x2_pack as xp  # noqa: E402
from x2c.compiler.ir import IRGraph  # noqa: E402
from x2c.compiler.partition import load_hardware  # noqa: E402

WORKLOADS = ("mobilenetv2", "deepgate", "mnist")
ENGINE = "x2_engine2_4bit"
EVIDENCE_4BIT = "Python X2 model, 4-bit cells (ASSUMED cell rule sum(pwm*level): x2_facts cell_compute_4bit is missing)"
EVIDENCE_RTL = "RTL simulation of the two-block engine with the chip repo's behavioral X2 (1-bit cells)"


# ---------------------------------------------------------------- 1. IR
def load_ir(workload, path=None):
    if path:
        return IRGraph.from_dict(json.loads(Path(path).read_text()))
    from x2c.compiler.ir_builder import build_ir_from_chain_workload, build_ir_from_compiler_graph
    from x2c.inputs.bmsemi_source import ReadOnlyRepo
    from x2c.inputs.normalize import compiler_v2_workload, deepgate_workload, mnist_workload
    r = ReadOnlyRepo()
    if workload == "mobilenetv2":
        return build_ir_from_compiler_graph(r, compiler_v2_workload(r))
    return build_ir_from_chain_workload(deepgate_workload(r) if workload == "deepgate" else mnist_workload(r))


# ---------------------------------------------------------------- 2. A1
class _NoLLM:
    """No model: A1 falls back to its validated baseline and code's own schedule (stated in the report)."""
    def start(self, *a, **k):
        from x2c.agents.llm_client import Action
        return Action("final", text="no llm")


def plan(g, hw, llm_name, log):
    from x2c.agents.mapping_agent import run_mapping_agent
    if llm_name == "none":
        llm = _NoLLM()
    elif llm_name == "mock":
        from x2c.agents.llm_client import MockMappingLLM
        llm = MockMappingLLM()
    else:
        from x2c.agents.llm_client import GeminiLLM, GroqLLM
        llm = {"groq": GroqLLM, "gemini": GeminiLLM}[llm_name]()
    return run_mapping_agent(llm, g, hw, log_path=str(log))


# ---------------------------------------------------------------- 3. the INT8 network and its input
def build(workload, g, synthetic, weights=None):
    if weights and workload != "mobilenetv2":
        raise SystemExit(f"--weights is for mobilenetv2; {workload} uses the real weights from BMsemi_Model")
    if workload == "mobilenetv2":
        net = N.build_mobilenet(g)
        info = {"input": "random uint8 image 0..127 (seeded)"}
        if weights:
            info["weights"] = apply_weights(net, weights)
        x = np.random.default_rng(7).integers(0, 128, size=net.input_shape)
        return net, x, info
    if workload == "deepgate":
        return _deepgate(synthetic, g)
    return _mnist(synthetic)


def apply_weights(net, path):
    """Real INT8 weights for MobileNetV2: an .npz with one array per layer name, shape (K, C/groups, R, S), values in
    -128..127 (BatchNorm already folded in), and optionally '<layer>__bias' (K,) int32. Layers not in the file keep the
    seeded weights; the report says how many layers are real. Shifts are re-calibrated afterwards."""
    data = np.load(path)
    layers = {s.name: s for s in net.steps if isinstance(s, N.Layer)}
    unknown = sorted(k for k in data.files if k.split("__")[0] not in layers)
    if unknown:
        raise SystemExit(f"--weights {path}: unknown names {unknown[:5]}; layer names are {sorted(layers)}")
    real = []
    for name, L in layers.items():
        if name in data.files:
            w = np.asarray(data[name])
            if w.shape != L.W.shape:
                raise SystemExit(f"--weights: {name} has shape {w.shape}, the graph needs {L.W.shape} (K, C/groups, R, S)")
            if not np.array_equal(w, np.round(w)) or w.min() < -128 or w.max() > 127:
                raise SystemExit(f"--weights: {name} must be INT8 values -128..127 (quantize first)")
            L.W = w.astype(np.int64)
            real.append(name)
        if f"{name}__bias" in data.files:
            b = np.asarray(data[f"{name}__bias"])
            if b.shape != L.bias.shape:
                raise SystemExit(f"--weights: {name}__bias has shape {b.shape}, needs {L.bias.shape}")
            L.bias = b.astype(np.int64)
    net.notes.append(f"real weights from {Path(path).name} for {len(real)}/{len(layers)} layers"
                     + ("" if len(real) == len(layers) else "; the rest are seeded random INT8"))
    return {"file": str(path), "real_layers": len(real), "layers": len(layers)}


def _deepgate_spec(L, req, acc):
    """The repo's requantization as data, if one integer formula reproduces the repo's fwd exactly on this layer's real
    accumulators and on 64 random ones around them; otherwise the bundle says replay needs BMsemi_Model's fwd."""
    params = {k: np.asarray(L[k]).tolist() for k in ("bias", "mult", "shift", "out_zp")}
    span = int(np.abs(acc).max()) + 1
    rng = np.random.default_rng(len(acc))
    tests = [acc] + [rng.integers(-2 * span, 2 * span + 1, size=acc.shape) for _ in range(64)]
    name = N.match_formula(req, params, tests)
    if name:
        return {"kind": "fixed_point", "formula": name, **params,
                "source": f"equals BMsemi deepgate fwd on {len(tests)} accumulator vectors (1 real + 64 random)"}
    return {"kind": "bmsemi_deepgate_fwd", **params, "note": "no integer formula matched the repo's fwd: replay imports it"}


def _deepgate(synthetic, g=None):
    if not synthetic:
        from BMsemi_Model.bm_models.models import deepgate as dg
        fwd, layers, cases = dg._load()
        ds = []
        x = np.asarray(cases[0][0]["output"], dtype=np.int64).reshape(-1)
        xi = x.copy()
        for i, L in enumerate(layers):
            req = (lambda L: (lambda acc: fwd(np.asarray(acc, dtype=np.int64), np.eye(L["n_out"], dtype=np.int8), L["bias"], L["mult"],
                                              L["shift"], L["out_zp"], np.int8)))(L)
            acc = np.asarray(L["W"], dtype=np.int64) @ (xi - int(L["in_zp"]))
            ds.append({"name": f"fc{i}", "W": L["W"], "bias": np.zeros(L["n_out"]), "requant": req, "model_zp": int(L["in_zp"]),
                       "requant_spec": _deepgate_spec(L, req, acc)})
            xi = np.asarray(req(acc), dtype=np.int64)
        net = N.build_chain("bm_deepgate_ad01_int8", ds, x.shape, True,
                            notes=["real INT8 weights, zero points and requantization from BMsemi_Model deepgate._load()",
                                   "requantization runs the repo's own fwd on the X2's int32 sums"])
        # the repo's own chain, layer by layer, for a check independent of our golden model
        ref, xr = [], x.astype(np.int8)                      # exactly model_workloads.run_deepgate's repo path
        for L in layers:
            xr = fwd(xr.astype(np.int32) - L["in_zp"], L["W"], L["bias"], L["mult"], L["shift"], L["out_zp"], np.int8)
            ref.append(np.asarray(xr, dtype=np.int64))
        return net, x, {"input": "DeepGate test case 0 (repo)", "repo_chain_final": np.asarray(ref[-1]).reshape(-1)}
    rng = np.random.default_rng(1)
    dims = [640, 128, 128, 128, 128, 8, 128, 128, 128, 128, 640]
    fcs = [n for n in (g.nodes if g is not None else []) if n.op == "linear"]
    if fcs and all(fcs[i].attrs["K"] == fcs[i + 1].attrs["C"] for i in range(len(fcs) - 1)):
        dims = [fcs[0].attrs["C"]] + [n.attrs["K"] for n in fcs]       # the stand-in follows the graph it is planned on
    ds = [{"name": f"fc{i}", "W": rng.integers(-128, 128, size=(dims[i + 1], dims[i])),
           "requant": (lambda acc: np.clip(np.asarray(acc) >> 12, -128, 127)), "model_zp": -3,
           "requant_spec": {"kind": "shift_clip", "shift": 12, "round": False}} for i in range(len(dims) - 1)]
    x = rng.integers(-128, 128, size=dims[0])
    net = N.build_chain("deepgate_SYNTHETIC", ds, x.shape, True, notes=["SYNTHETIC stand-in (no BMsemi_Model here)"])
    return net, x, {"input": "SYNTHETIC"}


def _mnist(synthetic):
    if not synthetic:
        from BMsemi_Model.bm_models.models.mnist import _digit, default_kernels, edge_filter
        kp, kn = default_kernels()
        img = np.asarray(_digit(5), dtype=np.int64)
        net = N.build_mnist("bm_mnist_edge_filter", kp, kn, *img.shape)
        return net, img[None], {"input": "MNIST digit 5 (repo)", "repo_map": np.asarray(edge_filter({"digit": 5})["map"])}
    rng = np.random.default_rng(5)
    kp, kn = rng.integers(0, 2, size=(5, 5)), rng.integers(0, 2, size=(5, 5))
    net = N.build_mnist("mnist_SYNTHETIC", kp, kn)
    return net, rng.integers(0, 2, size=(1, 28, 28)), {"input": "SYNTHETIC"}


# ---------------------------------------------------------------- 4. X2 executors
def x2_4bit(hw4, programs):
    def mm(L, P):
        pkg = xp.pack_layer(L.name, {k: L.attrs[k] for k in ("K", "C", "R", "S", "groups")}, L.W, hw4)
        programs[L.name] = pkg
        y, _ = xp.run_layer(pkg, P)
        return y
    return mm


def x2_1bit_rtl(hw1, out, cycles, max_tiles, fallback):
    import x2c.commands.sim_engine2 as e2
    import x2c.commands.sim_engine1 as ec
    import x2c.commands.sim_layers as le
    from x2c.chip.chip_repo import ChipRepo
    from x2c.chip.x2_golden import X2Config
    repo = ChipRepo()
    out.mkdir(parents=True, exist_ok=True)

    def mm(L, P):
        pkg = xp.pack_layer(L.name, {k: L.attrs[k] for k in ("K", "C", "R", "S", "groups")}, L.W, hw1)
        if pkg.tiles > max_tiles:                 # too slow for Icarus: this layer stays on the 4-bit Python path
            cycles[L.name] = {"skipped": f"{pkg.tiles} one-bit tiles > --max-tiles {max_tiles}", "tiles": pkg.tiles}
            return fallback(L, P)
        text, entries, costs, nacc = le.run_sim(pkg, P, X2Config(tdc_time_out=pkg.required_tdc_time_out), "alternate", out, L.name, repo)
        got, done = e2.parse(text, nacc)
        order = ec.read_order(pkg, P)
        accs = [got.get(o) for o in order]
        if done is None or None in accs:
            raise RuntimeError(f"{L.name}: the engine simulation did not finish (see {out})")
        cycles[L.name] = {"measured": int(done[2]), "predicted": e2.predicted(entries, costs), "tiles": pkg.tiles}
        return ec.host_rebuild(pkg, P, order, accs, ec.engine_layout(pkg))
    return mm


# ---------------------------------------------------------------- 6. bundle
def write_bundle(folder, net, x, gold, programs, a1, chip, rows, extra):
    folder.mkdir(parents=True, exist_ok=True)
    host = []
    for s in net.steps:
        if isinstance(s, N.Layer):
            pkg = programs.get(s.name)
            host.append({"step": "x2_layer", "name": s.name, "reads": s.src, "writes": s.out, "attrs": s.attrs,
                         "x2_input_offset": s.in_zp, "model_zero_point": s.model_zp,
                         "epilogue": {"requant": s.requant_spec or {"kind": "unserialized"}} if s.requant is not None else
                         {"bias": s.bias.tolist(), "shift": s.shift, "clamp": [s.lo, s.hi]},
                         "x2_tiles": pkg.tiles if pkg else None,
                         "x2": None if pkg is None else {
                             "cell_bits": pkg.hw["x2"]["cell_bits"], "rows": pkg.hw["x2"]["rows"], "cols": pkg.hw["x2"]["cols"],
                             "input_bits": pkg.hw["x2"]["input_bits"], "magnitude_bits": pkg.hw["weights"]["magnitude_bits"],
                             "input_split_bits": pkg.input_split_bits, "tdc_time_out": pkg.required_tdc_time_out}})
        else:
            host.append({"step": s[0], "writes": s[1], "reads": list(s[2:])})
    (folder / "host_program.json").write_text(json.dumps({"model": net.name, "input_shape": list(net.input_shape),
                                                          "output": net.output, "steps": host, "notes": net.notes}, indent=1, default=str))
    arrays = {}
    for name, pkg in programs.items():
        for ti, cells in enumerate(pkg.cells):
            arrays[f"{name}__tile{ti}__levels"] = np.asarray(cells, dtype=np.uint8)
        arrays[f"{name}__placement"] = np.asarray(json.dumps(pkg.placement))
    np.savez_compressed(folder / "x2_program.npz", **arrays)
    np.savez_compressed(folder / "expected.npz", **{k: np.asarray(v) for k, v in gold.items()})        # includes "input"
    plan_out = {k: a1[k] for k in ("final_source", "search_space", "hardware_profile", "schedule", "fabric_estimate") if k in a1}
    plan_out["partition_totals"] = (a1.get("final_plan") or {}).get("totals")
    (folder / "plan.json").write_text(json.dumps(plan_out, indent=1, default=str))
    # the engine (same for every model): bitstream, its pin map (design.pcf: which IOBUF carries clock/reset/data) and audit
    bit, eng = None, ROOT / "chip_build" / ENGINE
    for fn in (f"{ENGINE}.bit", "design.pcf", "bitstream_audit.json", "flow_report.json"):
        if (eng / fn).is_file():
            (folder / "engine").mkdir(exist_ok=True)
            shutil.copy(eng / fn, folder / "engine" / fn)
            bit = f"engine/{fn}" if fn.endswith(".bit") else bit
    return bit


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workload", choices=WORKLOADS, required=True)
    ap.add_argument("--ir", help="IR JSON instead of reading BMsemi_Model")
    ap.add_argument("--synthetic", action="store_true", help="DeepGate/MNIST stand-in data (no BMsemi_Model)")
    ap.add_argument("--weights", help="mobilenetv2: .npz of real INT8 weights, one array per layer name (see apply_weights)")
    ap.add_argument("--llm", choices=["none", "mock", "groq", "gemini"], default="none", help="A1's model (none: baseline + code schedule)")
    ap.add_argument("--sim", action="store_true", help="also run every layer on the 1-bit two-block engine (Ubuntu, Icarus)")
    ap.add_argument("--max-tiles", type=int, default=64)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    out = Path(a.out or f"deploy/{a.workload}")
    out.mkdir(parents=True, exist_ok=True)
    hw4 = load_hardware()
    hw1 = copy.deepcopy(hw4); hw1["x2"]["cell_bits"] = 1
    g = load_ir(a.workload, a.ir)
    a1 = plan(g, hw4, a.llm, out / "a1_log.jsonl")
    net, x, info = build(a.workload, g, a.synthetic, a.weights)
    if a.workload == "mobilenetv2":
        N.calibrate(net, x)
        info["module_kinds_inferred"] = dict(Counter(N.infer_kinds(g).values()))
    gold = N.golden(net, x)
    checks = {}
    if "repo_chain_final" in info:
        checks["golden_equals_repo_deepgate_chain"] = bool(np.array_equal(gold[net.output], info["repo_chain_final"]))
    if "repo_map" in info:
        checks["golden_equals_repo_edge_map"] = bool(np.array_equal(gold[net.output], info["repo_map"]))
    programs = {}
    t4, rows4 = N.run(net, x, x2_4bit(hw4, programs), gold)
    checks["x2_4bit_all_steps_exact"] = all(r["exact"] for r in rows4)
    sim = None
    if a.sim:
        cycles = {}
        for tool in ("iverilog", "vvp"):
            if not shutil.which(tool):
                raise SystemExit(f"--sim: {tool} not on PATH (Ubuntu terminal, chip toolchain)")
        t1, rows1 = N.run(net, x, x2_1bit_rtl(hw1, out / "sim", cycles, a.max_tiles, x2_4bit(hw4, {})), gold)
        done = {k: v for k, v in cycles.items() if "measured" in v}
        checks["x2_1bit_engine_rtl_all_steps_exact"] = all(r["exact"] for r in rows1) and bool(done)
        sim = {"layers": cycles, "layers_simulated": len(done), "layers_skipped": len(cycles) - len(done),
               "total_cycles_measured": sum(v["measured"] for v in done.values()),
               "total_cycles_predicted": sum(v["predicted"] for v in done.values()),
               "note": "per simulated layer, tiles alternating over both X2 blocks; skipped layers (--max-tiles) use the 4-bit "
                       "Python path; host steps (im2col, adds, pool) not counted"}
    try:
        from x2c.verification.chip_tools import chip_status
        chip = chip_status()
    except Exception as e:  # noqa: BLE001
        chip = {"error": f"{type(e).__name__}: {e}"}
    try:
        from x2c.chip import fabric_costs as fcost
        need, host_only = fcost.blocks_for(net)
        fabric = {"engine_only": fcost.estimate(["x2_engine"]), "host_steps_on_fabric": fcost.estimate(need),
                  "host_only_steps": host_only,
                  "note": "today the epilogue (bias/shift/clamp, adds) runs on the host; this estimates moving it onto the fabric"}
    except Exception as e:  # noqa: BLE001
        fabric = {"error": f"{type(e).__name__}: {e}"}
    a1["fabric_estimate"] = fabric
    bit = write_bundle(out, net, x, gold, programs, a1, chip, rows4, info)
    try:
        import x2c.commands.replay as run_bundle
        rep = run_bundle.replay(out)
        checks["bundle_replays_exactly"] = rep["exact"]
        replay = {k: rep[k] for k in ("exact", "steps", "first_mismatch", "uses")}
    except Exception as e:  # noqa: BLE001
        checks["bundle_replays_exactly"] = False
        replay = {"error": f"{type(e).__name__}: {e}"}
    sched = a1.get("schedule") or {}
    evidence = " + ".join([EVIDENCE_4BIT] + ([EVIDENCE_RTL] if sim else [])) + "; NOT the configured fabric, NOT silicon"
    report = {"workload": a.workload, "model": net.name, "ok": all(checks.values()), "checks": checks,
              "steps": len(rows4), "x2_layers": sum(1 for s in net.steps if isinstance(s, N.Layer)),
              "output": np.asarray(gold[net.output]).tolist() if np.asarray(gold[net.output]).size <= 64 else "see expected.npz",
              "a1": {"final_source": a1.get("final_source"), "schedule_source": sched.get("source"),
                     "latency_us": (sched.get("metrics") or {}).get("latency_us"),
                     "latency_reduction_vs_today_percent": sched.get("latency_reduction_vs_today_percent")},
              "engine_bitstream": {"design": ENGINE, "file_in_bundle": bit,
                                   "build": None if bit else "Ubuntu: x2c build --example x2_engine2_4bit, then deploy again",
                                   "status": (chip.get("designs") or {}).get(ENGINE)},
              "x2_facts_missing": chip.get("x2_facts_missing"), "fabric": fabric, "weights": info.get("weights"), "bundle_replay": replay, "simulation_1bit": sim, "notes": net.notes,
              "input": info.get("input"), "module_kinds_inferred": info.get("module_kinds_inferred"), "seconds": round(time.time() - t0, 1), "evidence_level": evidence}
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"DEPLOY {a.workload} ({net.name}): {report['x2_layers']} X2 layers, {len(rows4)} steps")
    for k, v in checks.items():
        print(f"  {k}: {v}")
    if sched:
        print(f"  A1: {a1.get('final_source')} | schedule {sched.get('source')} | {sched.get('metrics', {}).get('latency_us')} us "
              f"({sched.get('latency_reduction_vs_today_percent')} % below today's)")
    if sim:
        print(f"  1-bit engine RTL: {sim['layers_simulated']} layers simulated ({sim['layers_skipped']} skipped, --max-tiles), "
              f"{sim['total_cycles_measured']} cycles measured, {sim['total_cycles_predicted']} predicted")
    print(f"  engine bitstream: {bit or 'not built yet (' + report['engine_bitstream']['build'] + ')'}")
    hs = fabric.get("host_steps_on_fabric") or {}
    if hs.get("used"):
        print(f"  eFPGA if the host steps moved onto it (ESTIMATE): {hs['used'].get('logic_cells')}/{hs['budget'].get('logic_cells')} cells, "
              f"{hs['used'].get('macc')}/{hs['budget'].get('macc')} MACC -> fits={hs['fits']}"
              + (f" (host-only: {', '.join(fabric['host_only_steps'])})" if fabric.get("host_only_steps") else ""))
    print(f"  bundle: {out.as_posix()}/  (plan.json, host_program.json, x2_program.npz, expected.npz, report.json)")
    print(f"DEPLOYMENT {'VERIFIED' if report['ok'] else 'FAILED'}   [{evidence}]")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
