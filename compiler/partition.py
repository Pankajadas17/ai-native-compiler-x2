"""
S04 baseline: split an IR graph between X2 (matrix compute) and the eFPGA (everything else), and VALIDATE any plan.

validate_partition is the contract for the future mapping agent (A1): an LLM may propose any plan, but a plan
only counts if this independent validator accepts it. The validator recomputes tile counts (through the repo
mapper) and boundary traffic (through a different code path than the baseline) instead of trusting the plan.

Two views of every plan:
  * compiler view: tile counts on the BMsemi compiler's 64-row geometry (hw["compiler_v2"]), checked against the
    compiler's own compiler_expected_tiles. This proves the compiler artifact is read correctly. It is NOT the chip.
  * chip view (plan["chip"]): the real X2 from sky130-reram-fpga (32 x 32, 1-bit cells, 2 blocks): tiles per layer,
    which layers stay programmed, and behavioral-model cycle estimates. validate_partition recomputes it.

HONESTY: eFPGA budgets are known (chip profile) but per-operator resource estimates are not (they need RTL);
weights.magnitude_bits / sign_factor follow BMsemi_Model DeepGate, not the chip repo (see hardware_config.json).
"""
import json
import math
from pathlib import Path

from compiler import chip_x2
from compiler.ir import MATRIX_OPS
import importlib.util

if importlib.util.find_spec("BMsemi_Model") is not None:      # team checkout: the independent repo mapper, as before
    from emulators.platform_ops import map_layer
    MAPPER_SOURCE = "BMsemi_Model core.map_layer (independent of this code)"
else:                                                          # public clone without the private BMsemi_Model
    MAPPER_SOURCE = "tile formula ceil(K/cols)*ceil(C/groups*R*S/rows) (BMsemi_Model not present: repo-mapper cross-check skipped)"

    def map_layer(p):
        """Stand-in for BMsemi's core.map_layer when that private repo is absent (same arguments, pe_tiles only)."""
        patch = int(p["C"]) * int(p["R"]) * int(p["S"])
        return {"pe_tiles": math.ceil(int(p["K"]) / int(p["pe_cols"])) * math.ceil(patch / int(p["pe_rows"]))}

HW_PATH = Path(__file__).resolve().parents[1] / "specifications" / "hardware_config.json"


def load_hardware(path=HW_PATH):
    return json.loads(Path(path).read_text())


def _compiler_geo(hw):
    """Compiler-view geometry: hw["compiler_v2"] (v1.0 config) or hw["x2"] (the old 0.1-ASSUMED config)."""
    return hw.get("compiler_v2") or hw["x2"]


def has_chip(hw):
    return "rows" in hw.get("x2", {}) and "weights" in hw


def _tiles_formula(a, hw):
    geo = _compiler_geo(hw)
    cols = a.get("tile_width") or geo["default_tile_cols"]
    patch = (a["C"] // a["groups"]) * a["R"] * a["S"]
    return math.ceil(a["K"] / cols) * math.ceil(patch / geo["pe_rows"])


def allowed_targets(n):
    """Targets a node may legally use TODAY. With no known fabric budget a matrix op may only go to X2 and everything
    else to the eFPGA, so each node has exactly one legal target (the validator enforces the same rules)."""
    if n.op in ("input", "output"):
        return ["io"]
    return ["x2"] if n.op in MATRIX_OPS else ["efpga"]


def search_space_size(g):
    """Number of legal plans = product of each node's allowed targets. 1 means there is no placement decision to make."""
    size = 1
    for n in g.nodes:
        size *= len(allowed_targets(n))
    return size


def _target_of(assign, name):
    v = assign.get(name)
    t = v.get("target") if isinstance(v, dict) else None
    return t if isinstance(t, str) else None


def plan_from_targets(g, hw, targets, basis="proposal", resident=None):
    """Build a plan from a bare {node: target} map (what an LLM proposes). Code fills in tile counts and boundary
    traffic; the plan is NOT checked here: validate_partition alone decides legality."""
    by = g.by_name()
    assign = {}
    for name, t in (targets or {}).items():
        entry = {"target": t, "basis": basis}
        n = by.get(name)
        if t == "x2" and n is not None and n.op in MATRIX_OPS:
            try:
                entry["x2_tiles"] = _tiles_formula(n.attrs, hw)
            except (KeyError, TypeError, ZeroDivisionError):
                entry["x2_tiles"] = None
        elif t == "efpga":
            entry["resource_estimate"] = None
        assign[str(name)] = entry
    boundary = []
    for n in g.nodes:
        for i in n.inputs:
            if {_target_of(assign, i), _target_of(assign, n.name)} == {"x2", "efpga"}:
                boundary.append({"from": i, "to": n.name, "elements": by[i].out.numel()})
    return _plan(g, hw, assign, boundary, resident)


def propose_partition_baseline(g, hw):
    assign, boundary = {}, []
    for n in g.nodes:
        if n.op in ("input", "output"):
            assign[n.name] = {"target": "io"}
        elif n.op in MATRIX_OPS:
            assign[n.name] = {"target": "x2", "x2_tiles": _tiles_formula(n.attrs, hw), "basis": "matrix op -> X2 crossbar"}
        else:
            assign[n.name] = {"target": "efpga", "resource_estimate": None,
                              "basis": "non-matrix op -> eFPGA; no resource estimate (fabric profile missing)"}
    by = g.by_name()
    for n in g.nodes:
        for i in n.inputs:
            if {assign[i]["target"], assign[n.name]["target"]} == {"x2", "efpga"}:
                boundary.append({"from": i, "to": n.name, "elements": by[i].out.numel()})
    return _plan(g, hw, assign, boundary)


def chip_view(g, hw, assign, resident=None):
    """Chip numbers for the X2 nodes of a plan (closed-form path). resident=None -> best_resident_set."""
    by, layers = g.by_name(), {}
    for name, v in assign.items():
        n = by.get(name)
        if n is None or n.op not in MATRIX_OPS or _target_of(assign, name) != "x2":
            continue
        try:
            placement = chip_x2.place_groups(n.attrs, hw)
            c = chip_x2.cycles(hw, n.attrs, n.out.numel(), len(placement))
        except (KeyError, TypeError, ZeroDivisionError):
            continue
        cells = chip_x2.layer_cells(n.attrs, hw)
        layers[name] = {"tiles": len(placement), "cells": cells,
                        "cell_use": round(cells / (len(placement) * hw["x2"]["rows"] * hw["x2"]["cols"]), 4),
                        "mvms_per_tile": chip_x2.mvm_count(n.attrs, n.out.numel()),
                        "compute_cycles": c["compute"], "program_upper": c["program_upper"], "placement": placement}
    if resident is None:
        resident = chip_x2.best_resident_set(layers, hw["x2"]["blocks"])
    resident = sorted(str(r) for r in resident)
    streamed = sorted(n for n in layers if n not in resident)
    return {"layers": layers, "resident": resident, "streamed": streamed,
            "totals": {"tiles": sum(v["tiles"] for v in layers.values()),
                       "resident_tiles": sum(layers[n]["tiles"] for n in resident if n in layers),
                       "streamed_tiles": sum(layers[n]["tiles"] for n in streamed),
                       "compute_cycles_lower": sum(v["compute_cycles"] for v in layers.values()),
                       "program_cycles_per_inference_upper": sum(layers[n]["program_upper"] for n in streamed)},
            "geometry": {"rows": hw["x2"]["rows"], "cols": hw["x2"]["cols"], "blocks": hw["x2"]["blocks"],
                         "cells_per_weight": chip_x2.cells_per_weight(hw)},
            "bounds": "compute = behavioral COMPUTE_DELAY + one beat per row + one result per column (lower bound); "
                      "program = every weight cell SET once at PROGRAM_DELAY (upper bound; real count depends on values)",
            "findings": _chip_findings(g, hw, layers)}


LOW_CELL_USE = 0.25   # report layers whose placed blocks fill less than this share of their tiles' cells


def _chip_findings(g, hw, layers):
    out = []
    geo = hw.get("compiler_v2")
    if geo and (geo["pe_rows"], geo["default_tile_cols"]) != (hw["x2"]["rows"], hw["x2"]["cols"]):
        out.append(f"compiler V2 tiles assume a {geo['pe_rows']}-row PE; the chip X2 is {hw['x2']['rows']} x {hw['x2']['cols']}, "
                   "so compiler_expected_tiles do not describe the chip")
    total = sum(v["tiles"] for v in layers.values())
    low = sorted(k for k, v in layers.items() if v["cell_use"] < LOW_CELL_USE)
    if low:
        low_tiles = sum(layers[k]["tiles"] for k in low)
        out.append(f"{len(low)} layer(s) fill under {LOW_CELL_USE:.0%} of their tiles' cells and take {low_tiles} of "
                   f"{total} X2 tiles: " + ", ".join(f"{k} ({layers[k]['cell_use']:.0%})" for k in low))
    if total > hw["x2"]["blocks"]:
        out.append(f"{total} X2 tiles on {hw['x2']['blocks']} blocks: all but the resident layers are reprogrammed every inference")
    return out


def _plan(g, hw, assign, boundary, resident=None):
    nb = sum(b["elements"] for b in boundary) * hw["activation_bytes"]
    status = "assumed" if "ASSUMED" in hw.get("version", "") else "chip_profile"
    plan = {"graph": g.name, "ir_digest": g.digest(), "hardware_profile": {"version": hw["version"], "status": status},
            "assignments": assign, "boundary_edges": boundary,
            "totals": {"x2_nodes": sum(v["target"] == "x2" for v in assign.values()),
                       "efpga_nodes": sum(v["target"] == "efpga" for v in assign.values()),
                       "x2_tiles": sum((v.get("x2_tiles") or 0) for v in assign.values()),
                       "boundary_edges": len(boundary), "boundary_bytes": nb},
            "assumptions": [hw["status"], hw["activation_bytes_basis"], "eFPGA resource estimates are unavailable"]}
    if has_chip(hw):
        plan["chip"] = chip_view(g, hw, assign, resident)
    return plan


def validate_partition(g, plan, hw):
    """Independent check of ANY plan (baseline or LLM-proposed). Returns a list of errors; empty = legal.
    Never raises on a malformed plan: that is reported as an error instead."""
    if not isinstance(plan, dict):
        return ["the plan must be an object"]
    asg = plan.get("assignments")
    if not isinstance(asg, dict):
        return ["plan.assignments must be an object mapping node name to {target: ...}"]
    errs, by = [], g.by_name()
    for name, v in asg.items():
        if not isinstance(v, dict):
            errs.append(f"{name}: the assignment must be an object with a 'target'")
    for name in by:
        if name not in asg:
            errs.append(f"node '{name}' is not assigned")
    for name in asg:
        if name not in by:
            errs.append(f"plan assigns unknown node '{name}'")
    if plan.get("ir_digest") != g.digest():
        errs.append("plan was made for a different IR (digest mismatch)")
    tiles = 0
    for name, n in by.items():
        t = _target_of(asg, name) if name in asg else None
        if t not in ("x2", "efpga", "io"):
            if name in asg:
                errs.append(f"{name}: invalid target '{(asg[name] or {}).get('target') if isinstance(asg[name], dict) else asg[name]}'")
            continue
        if n.op in ("input", "output") and t != "io":
            errs.append(f"{name}: {n.op} must be 'io'")
        elif n.op not in ("input", "output") and t == "io":
            errs.append(f"{name}: only input and output nodes may be 'io', not '{n.op}'")
        elif n.op in MATRIX_OPS and t == "efpga":
            errs.append(f"{name}: matrix op on eFPGA is not allowed without a fabric budget (none is known)")
        elif n.op not in MATRIX_OPS and n.op not in ("input", "output") and t == "x2":
            errs.append(f"{name}: X2 only executes matrix ops, not '{n.op}'")
        if t == "x2" and n.op in MATRIX_OPS:                       # recompute through the REPO mapper (independent path)
            a = n.attrs
            geo = _compiler_geo(hw)
            want = map_layer({"K": a["K"], "C": a["C"] // a["groups"], "R": a["R"], "S": a["S"], "Y": 1, "X": 1,
                              "pe_rows": geo["pe_rows"], "pe_cols": a.get("tile_width") or geo["default_tile_cols"]})["pe_tiles"]
            got = asg[name].get("x2_tiles")
            if got != want:
                errs.append(f"{name}: plan says {got} X2 tiles, the mapper gives {want}")
            tiles += want
            exp = a.get("compiler_expected_tiles")
            if exp is not None and exp != want:
                errs.append(f"{name}: compiler expected {exp} tiles, mapper gives {want}")
    # boundary traffic and node counts, recomputed from the consumers map (a different code path than the baseline)
    cons, edges = g.consumers(), []
    for name, n in by.items():
        for c in cons[name]:
            if {_target_of(asg, name), _target_of(asg, c)} == {"x2", "efpga"}:
                edges.append((name, c, n.out.numel()))
    elems = sum(e[2] for e in edges)
    totals = plan.get("totals")
    if not isinstance(totals, dict):
        errs.append("plan.totals is missing")
        return errs
    counts = {"x2_nodes": sum(_target_of(asg, k) == "x2" for k in by), "efpga_nodes": sum(_target_of(asg, k) == "efpga" for k in by)}
    if totals.get("x2_tiles") != tiles:
        errs.append(f"totals.x2_tiles={totals.get('x2_tiles')} but the nodes add up to {tiles}")
    if totals.get("boundary_bytes") != elems * hw["activation_bytes"]:
        errs.append(f"totals.boundary_bytes={totals.get('boundary_bytes')} but recomputed {elems * hw['activation_bytes']}")
    for key, want in counts.items():
        if totals.get(key) != want:
            errs.append(f"totals.{key}={totals.get(key)} but the assignments give {want}")
    if totals.get("boundary_edges") != len(edges):
        errs.append(f"totals.boundary_edges={totals.get('boundary_edges')} but recomputed {len(edges)}")
    listed = plan.get("boundary_edges")
    if isinstance(listed, list):
        try:
            got_edges = sorted((b["from"], b["to"], b["elements"]) for b in listed)
        except (TypeError, KeyError):
            got_edges = None
        if got_edges != sorted(edges):
            errs.append("plan.boundary_edges does not match the edges recomputed from the assignments")
    else:
        errs.append("plan.boundary_edges is missing")
    if has_chip(hw):
        errs += _validate_chip(g, plan, hw, asg)
    return errs


def _validate_chip(g, plan, hw, asg):
    """Check the plan's placement with chip_x2.check_placement and recompute every chip total from it."""
    chip = plan.get("chip")
    if not isinstance(chip, dict):
        return ["plan.chip is missing (hardware config has the chip X2)"]
    errs, by = [], g.by_name()
    x2_layers = [k for k, n in by.items() if n.op in MATRIX_OPS and _target_of(asg, k) == "x2"]
    layers = chip.get("layers") if isinstance(chip.get("layers"), dict) else {}
    resident = chip.get("resident") if isinstance(chip.get("resident"), list) else None
    if resident is None:
        return errs + ["plan.chip.resident must be a list of layer names"]
    if set(layers) != set(x2_layers):
        errs.append("plan.chip.layers must list exactly the matrix layers placed on X2")
    tiles, cells_sum, prog, comp = {}, {}, 0, 0
    t = hw["x2"]["timing_cycles"]
    for k in x2_layers:
        a, n = by[k].attrs, by[k]
        entry = layers.get(k) if isinstance(layers.get(k), dict) else {}
        try:
            perr = chip_x2.check_placement(a, hw, entry.get("placement"))
        except (KeyError, TypeError, ZeroDivisionError) as e:
            perr = [f"cannot check ({type(e).__name__})"]
        if perr:
            errs += [f"{k}: placement: {e}" for e in perr[:5]]
            continue
        tiles[k] = len(entry["placement"])
        if entry.get("tiles") != tiles[k]:
            errs.append(f"{k}: plan says {entry.get('tiles')} chip X2 tiles, its placement has {tiles[k]}")
        g_, taps, outs = chip_x2.layer_shape(a)
        cells_sum[k] = g_ * taps * outs * hw["weights"]["sign_factor"] * math.ceil(hw["weights"]["magnitude_bits"] / hw["x2"]["cell_bits"])
        comp += tiles[k] * max(1, n.out.numel() // a["K"]) * (hw["x2"]["rows"] + t["compute"] + hw["x2"]["cols"])
    if sorted(chip.get("streamed") or []) != sorted(k for k in tiles if k not in resident):
        errs.append("plan.chip.streamed must be exactly the X2 layers that are not resident")
    unknown = [r for r in resident if r not in tiles]
    if unknown:
        errs.append(f"resident layers not on X2: {unknown}")
    res_tiles = sum(tiles.get(r, 0) for r in resident)
    streamed = [k for k in tiles if k not in resident]
    cap = hw["x2"]["blocks"] - (1 if streamed else 0)
    if res_tiles > cap:
        errs.append(f"resident layers need {res_tiles} X2 blocks but only {cap} are free "
                    f"({hw['x2']['blocks']} blocks{', one kept free for streamed layers' if streamed else ''})")
    prog = sum(cells_sum[k] for k in streamed if k in cells_sum) * t["program"]
    tot = chip.get("totals") if isinstance(chip.get("totals"), dict) else {}
    for key, want in (("tiles", sum(tiles.values())), ("resident_tiles", res_tiles),
                      ("streamed_tiles", sum(tiles[k] for k in streamed)),
                      ("compute_cycles_lower", comp), ("program_cycles_per_inference_upper", prog)):
        if tot.get(key) != want:
            errs.append(f"chip.totals.{key}={tot.get(key)} but recomputed {want}")
    return errs
