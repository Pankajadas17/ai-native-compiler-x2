"""
X2 tile schedule for one inference on the chip's 2 X2 blocks: which layers stay programmed (resident), which block each
streamed tile is programmed into, and in what order. A1 (agents/mapping_agent.py) may propose one; validate_schedule alone
decides legality and computes the cycles; best_schedule is code's own answer to compare with.

Why it matters: programming a tile costs far more than using it (per cell ~365 cycles vs ~511 + 3/row per MVM), so the
order in which the two blocks are programmed and used decides the latency:
  * resident + serial: one block keeps a layer programmed, the other programs and computes tile after tile;
  * ping-pong: no resident layer; while one block computes, the other is already programming the next tile;
  * anything in between that the validator accepts.

Cost model, CALIBRATED on the four X2 engine simulations of flow step 4 (BM behavioral X2, 1-bit cells):
    cycles = PROGRAM_CMD * program commands + MVM * MVMs + BEAT * input beats      (max error 1.6 cycles over 4 runs)
  * program: every weight cell of a tile programmed once (SET or RESET): an UPPER bound (the real count depends on the
    values; the calibration runs used 21-29 % of it);
  * compute per MVM: MVM + BEAT * rows used in the tile (every row is driven).
Not modelled (stated in every result): eFPGA operators between X2 layers (0 cycles), host transfers, reprogramming
between inferences. Using both blocks at once is BUILT and MEASURED: examples/chip_designs/x2_engine2.v (35/36 pins, both X2
sites; 16.7 MHz 1-bit, 14.5 MHz 4-bit) cut the latency of the 4 engine layers by 43-54 %, and simulate() predicted every
run within 0.05 % (run_x2_engine2_check.py). The one-block x2_engine.v runs only schedules that use a single block.
"""
from itertools import combinations

from chip import x2_facts
from compiler.ir import MATRIX_OPS

# cycles; the fitted constants of the BEHAVIORAL model (tests/test_x2_schedule.py re-fits them). tile_costs reads the current
# values from chip/x2_facts.py, so a silicon number put in specifications/x2_overrides.json is used without a code change.
PROGRAM_CMD, MVM, BEAT = (x2_facts.FACTS[k]["value"] for k in ("program_cycles_per_cell", "mvm_cycles", "beat_cycles"))
CALIBRATION = ("cycles = 365 * program commands + 511 * MVMs + 3.1 * input beats: least-squares fit to the 4 X2 engine "
               "simulations (conv, dw, pw, fc; BM behavioral X2, 1-bit cells), max error 1.6 cycles")
ASSUMPTIONS = ["both X2 blocks at once: built (x2_engine2.v) and measured on 4 layers within 0.05 % (BM behavioral X2, 1-bit cells); "
               "4-bit cells are timing/size-checked only (the chip's X2 model computes 1-bit)",
               "eFPGA operators between X2 layers and host transfers take 0 cycles (not modelled)",
               "programming = every weight cell of a tile programmed once: an upper bound (depends on the values)",
               "one inference from a cold start with resident layers already programmed"]
POLICIES = ("earliest_free", "round_robin", "single_block")
# single_block = every streamed tile through ONE block (what the built x2_engine, wired to one X2 block, can run; also what the
# old chip view's cycle sum assumed). The other policies use both blocks at once.


def tile_costs(chip, hw):
    """{layer: [{"cells", "rows", "program", "compute"}, ...]} from each layer's placement (recomputed, not trusted)."""
    out = {}
    pc, mv, bt = (x2_facts.value(k) for k in ("program_cycles_per_cell", "mvm_cycles", "beat_cycles"))
    for name, v in chip["layers"].items():
        tiles = []
        for blocks in v["placement"]:
            cells = sum(b[5] * b[6] for b in blocks)
            rows = max(b[3] + b[5] for b in blocks)
            tiles.append({"cells": cells, "rows": rows, "program": cells * pc, "compute": v["mvms_per_tile"] * (mv + bt * rows)})
        out[name] = tiles
    return out


def x2_deps(g, layers):
    """For every X2 layer, the X2 layers whose results it needs (nearest X2 ancestors, looking through non-X2 nodes)."""
    by, memo = g.by_name(), {}

    def ancestors(name):
        if name not in memo:
            found = set()
            for i in by[name].inputs:
                found |= {i} if i in layers else ancestors(i)
            memo[name] = found
        return memo[name]
    return {k: sorted(ancestors(k)) for k in layers}


def layer_order(g, layers):
    return [n.name for n in g.nodes if n.name in layers]


def validate_schedule(g, chip, hw, schedule):
    """Independent check of ANY schedule; returns (errors, metrics). metrics is None when there are errors.
    schedule = {"resident": [layer, ...], "tiles": [[layer, tile_index, block], ...] in issue order (every X2 tile once)}"""
    if not isinstance(schedule, dict):
        return ["the schedule must be an object"], None
    blocks, layers = hw["x2"]["blocks"], chip.get("layers") or {}
    resident, entries = schedule.get("resident"), schedule.get("tiles")
    if not isinstance(resident, list) or not all(isinstance(r, str) for r in resident):
        return ["schedule.resident must be a list of layer names"], None
    if not isinstance(entries, list):
        return ["schedule.tiles must be a list of [layer, tile_index, block]"], None
    costs, errs = tile_costs(chip, hw), []
    unknown = sorted(set(resident) - set(layers))
    if unknown:
        errs.append(f"resident layers not on X2: {unknown}")
    if sorted(resident) != sorted(chip.get("resident") or []):
        errs.append(f"schedule.resident {sorted(resident)} differs from the plan's chip.resident {sorted(chip.get('resident') or [])}")
    seen, res_blocks, str_blocks, clean = set(), {}, set(), []
    for i, e in enumerate(entries):
        if not (isinstance(e, list) and len(e) == 3 and isinstance(e[0], str) and all(isinstance(x, int) and not isinstance(x, bool) for x in e[1:])):
            errs.append(f"tiles[{i}]: must be [layer, tile_index, block], got {e!r}")
            continue
        L, t, b = e
        if L not in costs or not 0 <= t < len(costs[L]):
            errs.append(f"tiles[{i}]: no tile {t} in X2 layer '{L}'")
            continue
        if not 0 <= b < blocks:
            errs.append(f"tiles[{i}]: block {b} does not exist (the chip has {blocks})")
            continue
        if (L, t) in seen:
            errs.append(f"tiles[{i}]: tile {t} of '{L}' is scheduled twice")
            continue
        seen.add((L, t))
        if L in resident:
            if b in res_blocks:
                errs.append(f"tiles[{i}]: block {b} already holds resident tile {res_blocks[b]}")
            res_blocks[b] = (L, t)
        else:
            str_blocks.add(b)
        clean.append((L, t, b))
    missing = [(L, t) for L in costs for t in range(len(costs[L])) if (L, t) not in seen]
    if missing:
        errs.append(f"{len(missing)} X2 tile(s) never scheduled, e.g. {missing[:3]}")
    both = sorted(set(res_blocks) & str_blocks)
    if both:
        errs.append(f"block(s) {both} hold a resident tile and are also used for streamed tiles")
    streamed_layers = [L for L in costs if L not in resident]
    if streamed_layers and len(res_blocks) > blocks - 1:
        errs.append(f"resident tiles fill all {blocks} blocks but {len(streamed_layers)} layer(s) still stream")
    deps, done = x2_deps(g, costs), {}
    for i, (L, t, b) in enumerate(clean):
        for d in deps[L]:
            if done.get(d, 0) != len(costs[d]):
                errs.append(f"tile {t} of '{L}' is issued before every tile of '{d}' it depends on")
                break
        done[L] = done.get(L, 0) + 1
    if errs:
        return errs, None
    sim = simulate(costs, deps, clean, resident, blocks)
    one_time = sum(c["program"] for L in resident for c in costs[L])
    mhz = x2_facts.value("clock_mhz")
    metrics = dict(sim, latency_us=round(sim["latency_cycles"] / mhz, 1), clock_mhz=mhz, resident_program_once_upper=one_time, blocks_streaming=sorted(str_blocks),
                   resident_blocks={str(k): list(v) for k, v in sorted(res_blocks.items())}, tiles=len(clean),
                   blocks_used=len(set(res_blocks) | str_blocks), runs_on_built_x2_engine=len(set(res_blocks) | str_blocks) <= 1)
    return [], metrics


def simulate(costs, deps, entries, resident, blocks):
    """Latency of an issue order. costs: {layer: [{"program", "compute"}, ...]}; entries: [(layer, tile, block), ...].
    A block programs its next tile when its last compute ends; a tile computes when its weights are in and every tile of
    the layers it depends on has computed. Resident tiles are already programmed."""
    free, layer_end, finish, prog, comp = [0] * blocks, {}, 0, 0, 0
    for L, t, b in entries:
        c = costs[L][t]
        ready = max((layer_end[d] for d in deps.get(L, [])), default=0)
        if L in resident:
            loaded = free[b]
        else:
            loaded = free[b] + c["program"]
            prog += c["program"]
        start = max(loaded, ready)
        end = start + c["compute"]
        free[b], layer_end[L], finish, comp = end, max(layer_end.get(L, 0), end), max(finish, end), comp + c["compute"]
    return {"latency_cycles": finish, "program_cycles_per_inference_upper": prog, "compute_cycles": comp,
            "serial_sum_cycles": prog + comp, "hidden_by_overlap_cycles": prog + comp - finish}


def materialize(g, chip, hw, resident, policy="earliest_free"):
    """A full schedule for a resident choice: layers in graph order, tiles in index order; resident tiles on the first
    blocks; streamed tiles on the remaining blocks by `policy` (earliest_free: the block that becomes free first)."""
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    costs, blocks = tile_costs(chip, hw), hw["x2"]["blocks"]
    order = layer_order(g, costs)
    res_tiles = [(L, t) for L in order if L in resident for t in range(len(costs[L]))]
    if len(res_tiles) > blocks:
        raise ValueError(f"{len(res_tiles)} resident tiles on {blocks} blocks")
    res_block = {lt: i for i, lt in enumerate(res_tiles)}
    stream = list(range(len(res_tiles), blocks))
    if policy == "single_block":
        stream = stream[:1]
    free, k, entries = {b: 0 for b in stream}, 0, []
    for L in order:
        for t in range(len(costs[L])):
            if L in resident:
                entries.append([L, t, res_block[(L, t)]])
                continue
            if not stream:
                raise ValueError("no block left for streamed tiles")
            if policy == "round_robin":
                b = stream[k % len(stream)]
                k += 1
            else:
                b = min(stream, key=lambda x: (free[x], x))
            free[b] += costs[L][t]["program"] + costs[L][t]["compute"]          # good enough to pick the next block
            entries.append([L, t, b])
    return {"resident": sorted(resident), "tiles": entries, "policy": policy}


def resident_options(chip, hw):
    """Every resident set that can be legal: nothing, or layers whose tiles fit in blocks - 1 (all, if everything fits)."""
    costs, blocks = tile_costs(chip, hw), hw["x2"]["blocks"]
    names = sorted(costs)
    total = sum(len(costs[n]) for n in names)
    if total <= blocks:
        return [names]
    opts = [[]]
    for r in range(1, blocks):
        for combo in combinations(names, r):
            if sum(len(costs[n]) for n in combo) <= blocks - 1:
                opts.append(sorted(combo))
    return opts


def best_schedule(g, chip, hw):
    """Code's own answer: every resident option x every policy, validated; the lowest latency wins (ties: fewer
    programming cycles, then the name order). Returns (schedule, metrics, candidates_tried)."""
    best, tried = None, 0
    for res in resident_options(chip, hw):
        for pol in POLICIES:
            sched = materialize(g, chip, hw, res, pol)
            errs, m = validate_schedule(g, dict(chip, resident=res), hw, sched)
            tried += 1
            if errs:
                continue
            key = (m["latency_cycles"], m["program_cycles_per_inference_upper"], res, pol)
            if best is None or key < best[0]:
                best = (key, sched, m)
    return (best[1], best[2], tried) if best else (None, None, tried)
