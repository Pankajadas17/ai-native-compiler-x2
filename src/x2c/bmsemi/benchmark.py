"""
Benchmark orchestrator: BMsemi_Model (read-only) -> local WorkloadSpec -> existing verification pipeline.
Every run produces ONE auditable report: source commit, fingerprints of every consumed file, an explicit
stage table (what exists / what does not), per-workload results in the standard report shape (so the debug
and test agents can read them), non-failure FINDINGS, and a deterministic result digest.
"""
import hashlib
import json
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

from x2c.bmsemi.platform_ops import map_layer
from x2c.compiler.ir_builder import build_ir_from_chain_workload, build_ir_from_compiler_graph
from x2c.compiler.ir_validate import validate_ir
from x2c.compiler.partition import load_hardware, propose_partition_baseline, validate_partition
from x2c.bmsemi.x2_adapter import x2_vecmat
from x2c.inputs.bmsemi_source import ReadOnlyRepo
from x2c.inputs.graph_check import validate_compiler_graph
from x2c.inputs.normalize import all_workloads
from x2c.verification.matmul_check import compare
from x2c.verification.layer_exec import grouped_conv_golden, grouped_conv_on_x2, input_hw

LEVEL = "python_behavioral_simulation (NOT RTL, NOT silicon)"
PLANS = {}                      # workload id -> baseline partition plan (filled per run)
DEFAULT_REPORT = "reports/benchmark_report.json"

# Status of each stage across the whole project (this benchmark run itself executes S02-S04 only; the others run from
# their own scripts). "where": the code that implements it (tests/test_benchmark.py checks it exists); "missing": what a
# partial stage still lacks. Updated 2026-11-13 for the BM Labs chip (sky130-reram-fpga, 32x32 X2, 4-bit cells).
STAGE_STATUS = [
    {"id": "S01", "name": "model input / selection", "status": "implemented",
     "how": "read-only BMsemi_Model adapter + catalog + normalised WorkloadSpec",
     "where": ["src/x2c/inputs/bmsemi_source.py", "src/x2c/inputs/normalize.py"]},
    {"id": "S02", "name": "golden math / PE validation", "status": "implemented",
     "how": "independent NumPy oracles vs the X2 behavioural simulator",
     "where": ["src/x2c/verification/layer_exec.py", "src/x2c/bmsemi/x2_adapter.py"]},
    {"id": "S03", "name": "typed IR / graph validation", "status": "partial",
     "how": "typed local IR + shape inference for the compiler V2 graph and sequential models",
     "missing": "module kinds are not recorded in the artifact (pool and the shape-preserving kinds are unknown)",
     "where": ["src/x2c/compiler/ir_builder.py", "src/x2c/compiler/ir_validate.py", "src/x2c/inputs/graph_check.py"]},
    {"id": "S04", "name": "mapping and X2 package", "status": "partial",
     "how": ("compiler view (64-row tiles) + chip view on the 32x32 X2 with 4-bit cells: explicit placements checked by "
             "check_placement, cell use, resident set; layers packed into X2 command streams, exact vs NumPy (x2_pack)"),
     "missing": "eFPGA budgets per IR node are not estimated by the partitioner (the chip flow measures built operators)",
     "where": ["src/x2c/compiler/partition.py", "src/x2c/compiler/chip_x2.py", "src/x2c/compiler/x2_pack.py"]},
    {"id": "S05", "name": "Verilog generation", "status": "partial",
     "how": ("A2 (rtl_agent) writes Verilog for relu8, add8_sat, requant_i32_i8 and x2_acc_step; the X2 bridge / "
             "recombiner / engine are hand-written chip designs"),
     "missing": "Verilog is not generated from a BMsemi workload's IR; operators are a fixed list",
     "where": ["src/x2c/agents/rtl_agent.py", "hardware/designs/x2_engine.v"]},
    {"id": "S06", "name": "lint / assertions / equivalence", "status": "partial",
     "how": ("Yosys parse + check and NumPy evaluation of every operator; SAT proofs that each operator equals an "
             "independent spec and that its chip netlist equals its RTL; SymbiYosys bounded check of the two-block "
             "engine (P1-P3); Verilator lint"),
     "missing": "the engine properties are bounded (10 clocks), not unbounded; the other X2 designs have no assertions",
     "where": ["src/x2c/verification/rtl_check.py", "src/x2c/verification/chip_chain.py", "src/x2c/verification/formal.py",
               "hardware/formal/x2_engine2_props.vh"]},
    {"id": "S07-S09", "name": "Yosys / nextpnr / FASM bitstream", "status": "partial",
     "how": ("the chip repo's own flow (synth_fabulous with repo techmaps, nextpnr-generic, FASM, 35,304-byte bitstream) "
             "runs out of the repo via chip_flow; corrected signed-multiplier map; chain per operator: x2c chain --target chip"),
     "missing": "built per operator and per X2 design, not yet for a whole BMsemi workload",
     "where": ["src/x2c/chip/chip_flow.py", "src/x2c/commands/build.py", "src/x2c/chip/techmap/MACC_8x8_20_map_signed.v", "src/x2c/commands/chain.py"]},
    {"id": "S10", "name": "bitstream audit (V0)", "status": "implemented",
     "how": ("every bitstream the chip flow builds: structure + decoded bits == FASM-required bits against the chip's "
             "bitstream spec (chip examples and chain operators); also the FABulous demo audit"),
     "where": ["src/x2c/chip/chip_bitstream.py", "src/x2c/verification/bitstream_audit.py", "src/x2c/commands/audit.py"]},
    {"id": "S11", "name": "configured fabric + X2 verification", "status": "partial",
     "how": ("X2 behavioural model checked against our golden; X2 bridge and engine RTL simulated with the BM X2 model "
             "(exact vs NumPy); synthesized netlists simulated with the repo's primitive models"),
     "missing": "the configured fabric (tile RTL loaded with the bitstream) is not simulated; no timing-annotated simulation",
     "where": ["src/x2c/chip/x2_check.py", "src/x2c/commands/sim_engine1.py", "src/x2c/verification/chip_chain.py"]},
    {"id": "S12", "name": "silicon validation", "status": "not_implemented", "how": "future; no hardware",
     "missing": "hardware"},
]


def _seed(*parts):
    return zlib.crc32("/".join(map(str, parts)).encode())


def reproduce_compiler_tiles(layer, pe_rows=64):
    """Hypothesis (verified on all 17 layers of the shipped flow): tiles = ceil(K / tile_width) * ceil((C/groups)*R*S / rows)."""
    m = map_layer({"K": layer.K, "C": layer.C // layer.groups, "R": layer.R, "S": layer.S, "Y": layer.Y, "X": layer.X,
                   "pe_rows": pe_rows, "pe_cols": layer.extra["tile_width"]})
    return m["pe_tiles"]


def naive_core_mapper_tiles(layer, pe):
    return map_layer({"K": layer.K, "C": layer.C, "R": layer.R, "S": layer.S, "Y": layer.Y, "X": layer.X,
                      "pe_rows": pe["rows"], "pe_cols": pe["cols"]})["pe_tiles"]


def _ir_partition_entries(wl, ir):
    """S03 (IR validation) and S04 (baseline partition + independent validation) as standard report entries."""
    errs, warns = validate_ir(ir)
    hw = load_hardware()
    plan = propose_partition_baseline(ir, hw)
    perrs = validate_partition(ir, plan, hw)
    mk = lambda test, stage, kind, e, reason, details: {
        "test": f"{wl.id}/{test}", "stage": stage, "kind": kind, "workload": wl.id, "shape": [1, 1], "expected": [[0]],
        "actual": [[len(e)]], "passed": not e, "reason": "ok" if not e else reason, "max_abs_err": float(len(e)),
        "errors": e[:10], "details": details}
    entries = [mk("ir", "S03", "ir_validation", errs, "ir_invalid",
                  {"nodes": len(ir.nodes), "ops": dict(sorted(Counter(n.op for n in ir.nodes).items())), "warnings": warns, "ir_digest": ir.digest()}),
               mk("partition", "S04", "partition_validation", perrs, "partition_illegal", dict(plan["totals"], hardware_profile=plan["hardware_profile"]["status"]))]
    return entries, plan


def _run_compiler_v2(repo, wl, vecmat, mapping_transform, seed_base):
    res, findings = [], []
    gc = validate_compiler_graph(repo)
    res.append({"test": f"{wl.id}/graph", "stage": "S03", "kind": "graph_validation", "workload": wl.id,
                "shape": [1, 1], "expected": [[0]], "actual": [[len(gc["errors"])]], "passed": gc["ok"],
                "reason": "ok" if gc["ok"] else "graph_invalid", "max_abs_err": float(len(gc["errors"])),
                "errors": gc["errors"][:10], "details": {k: gc[k] for k in ("node_count", "ops", "layers_checked")}})
    ir_entries, plan = _ir_partition_entries(wl, build_ir_from_compiler_graph(repo, wl))
    res += ir_entries
    t = plan["totals"]
    findings.append({"id": "F2_boundary_traffic_baseline",
                     "text": (f"Baseline split: {t['x2_nodes']} matrix nodes on X2 ({t['x2_tiles']} tiles), {t['efpga_nodes']} other nodes on the eFPGA; "
                              f"every X2 layer is bracketed by non-matrix nodes, giving {t['boundary_edges']} X2<->eFPGA crossings = {t['boundary_bytes']} bytes "
                              "per inference IF activations are INT8 (assumption). eFPGA resource use per node is NOT estimated here; "
                              "the chip flow measures it per built operator (x2c chain --target chip)."),
                     "layers": []})
    PLANS[wl.id] = plan
    naive_diff = []
    for L in wl.layers:
        exp_t, got_t = L.extra["compiler_expected_tiles"], reproduce_compiler_tiles(L, wl.pe["rows"])
        naive = naive_core_mapper_tiles(L, wl.pe)
        if naive != exp_t:
            naive_diff.append(L.name)
        res.append({"test": f"{wl.id}/{L.name}/compiler_tiles", "stage": "S04", "kind": "mapping_consistency", "workload": wl.id,
                    "shape": [1, 1], "expected": [[exp_t]], "actual": [[got_t]], "passed": exp_t == got_t,
                    "reason": "ok" if exp_t == got_t else "tile_count_mismatch", "max_abs_err": float(abs(exp_t - got_t)),
                    "details": {"op": L.op, "tile_width": L.extra["tile_width"], "naive_core_mapper_tiles": naive}})
    if naive_diff:
        findings.append({"id": "F1_naive_core_mapper_vs_compiler",
                         "text": (f"The core mapper (map_layer at the default {wl.pe['rows']}x{wl.pe['cols']} geometry, no groups) disagrees "
                                  f"with the compiler's expected_tile_count on {len(naive_diff)}/{len(wl.layers)} layers; "
                                  "a groups-aware mapping with the compiler's per-layer tile_width reproduces all of them."),
                         "layers": naive_diff})
    for L in wl.layers:
        h, w = input_hw(L)
        rng = np.random.default_rng(_seed(wl.id, L.name, seed_base))
        ifm = rng.integers(-3, 4, (L.C, h, w)).astype(float)
        wt = rng.integers(-3, 4, (L.K, L.C // L.groups, L.R, L.S)).astype(float)
        expected = grouped_conv_golden(ifm, wt, L.stride, L.padding, L.groups).reshape(L.K, L.Y * L.X)
        base = {"test": f"{wl.id}/{L.name}", "stage": "S02", "kind": "layer_on_x2", "workload": wl.id,
                "data_source": wl.data_source, "layer": {"op": L.op, "K": L.K, "C": L.C, "R": L.R, "S": L.S, "Y": L.Y, "X": L.X,
                                                         "stride": L.stride, "padding": L.padding, "groups": L.groups},
                "shape": [L.K, L.Y * L.X], "expected": expected.tolist()}
        try:
            out, passes, exp_passes = grouped_conv_on_x2(ifm, wt, L.stride, L.padding, L.groups, wl.pe["rows"],
                                                         L.extra["tile_width"], vecmat, mapping_transform)
        except Exception as e:  # noqa: BLE001
            res.append({**base, "actual": None, "passed": False, "reason": "exception", "error": f"{type(e).__name__}: {e}", "max_abs_err": None})
            continue
        cmp = compare(expected, out.reshape(L.K, L.Y * L.X))
        check = {"passes_executed": passes, "mvm_ops_from_mapper": exp_passes, "ok": passes == exp_passes}
        if cmp["passed"] and not check["ok"]:
            cmp = {"passed": False, "reason": "pass_count_mismatch", "max_abs_err": cmp["max_abs_err"]}
        res.append({**base, "actual": out.reshape(L.K, L.Y * L.X).tolist(), "mapping_check": check, **cmp})
    return res, findings


def _run_model_workload(wl, vecmat, mapping_transform):
    from x2c.verification import model_workloads as mw
    if wl.loader == "deepgate_int8":
        runs = [mw.run_deepgate(0, mapping_transform=mapping_transform, vecmat=vecmat)]
    else:
        runs = [mw.run_mnist(5, mapping_transform=mapping_transform, vecmat=vecmat),
                mw.run_mnist(9, mapping_transform=mapping_transform, vecmat=vecmat)]
    ir_entries, plan = _ir_partition_entries(wl, build_ir_from_chain_workload(wl))
    PLANS[wl.id] = plan
    return [dict(r, stage="S02", workload=wl.id, data_source=wl.data_source) for r in runs] + ir_entries, []


def run_benchmark(repo=None, workload_ids=None, vecmat=x2_vecmat, mapping_transform=None,
                  report_path=DEFAULT_REPORT, seed_base=0):
    repo = repo or ReadOnlyRepo()
    wls = all_workloads(repo)
    known = {w.id: w for w in wls}
    if workload_ids:
        bad = [i for i in workload_ids if i not in known]
        if bad:
            raise ValueError(f"unknown workload ids {bad}; available: {sorted(known)}")
        wls = [known[i] for i in workload_ids]
    consumed = ["catalog.json"] + sorted({f["path"] for w in wls for f in w.provenance["files"]})
    snap, caches_before = repo.snapshot(consumed), repo.pycache_dirs()

    PLANS.clear()
    all_results, per_workload, findings = [], [], []
    for wl in wls:
        if wl.loader == "compiler_v2_layers":
            res, fnd = _run_compiler_v2(repo, wl, vecmat, mapping_transform, seed_base)
        else:
            res, fnd = _run_model_workload(wl, vecmat, mapping_transform)
        all_results += res; findings += fnd
        per_workload.append({"id": wl.id, "title": wl.title, "data_source": wl.data_source, "stages": wl.stages,
                             "layers": len(wl.layers), "tests": len(res), "passed": sum(r["passed"] for r in res),
                             "notes": wl.notes})
    repo.assert_unchanged(snap)                                   # raises if anything we consumed was touched
    unchanged_caches = repo.pycache_dirs() == caches_before
    core = {"source": {"repo_commit": repo.commit(), "consumed_files": list(snap.values())},
            "workloads": per_workload, "results": all_results, "findings": findings, "plans": dict(PLANS)}
    # The digest covers only OS-independent identity: raw on-disk sizes (LF vs CRLF checkouts) are excluded.
    canonical = dict(core, source={"repo_commit": core["source"]["repo_commit"],
                                   "consumed_files": [{k: f[k] for k in ("path", "sha256_lf", "size_lf")} for f in snap.values()]})
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, default=str).encode()).hexdigest()
    report = {"level": LEVEL, "benchmark": "bmsemi_read_only_inputs_v1", "total": len(all_results),
              "passed": sum(r["passed"] for r in all_results), "stage_status": STAGE_STATUS,
              "repo_read_only_check": {"consumed_files_unchanged": True, "no_new_pycache_in_repo": unchanged_caches},
              "result_digest_sha256": digest, **core}
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(report, indent=2, default=str))
    return report
