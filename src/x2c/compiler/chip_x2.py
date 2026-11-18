"""Chip view of matrix layers on the real X2 (32 x 32, 1-bit cells, 2 blocks).

All numbers come from specifications/hardware_config.json v1.0 (generated from the chip profile).

A matrix layer is K output channels x (C/groups)*R*S input taps per group.
  * inputs (taps)   -> crossbar rows      (32)
  * outputs         -> crossbar columns   (32); each weight needs `cells_per_weight` columns
                       (sign banks x bit planes), combined later by shift-and-add
  * grouped / depthwise convs: each group is its own small matrix with its own inputs, so groups
    may share a tile only on disjoint rows AND columns (block-diagonal; unused cells stay RESET = 0)

The planner writes an explicit placement (place_groups) into the plan; the validator checks ANY placement
with check_placement (bounds, no shared rows/columns inside a tile, every weight covered exactly once).
"""
from math import ceil


def cells_per_weight(hw):
    w = hw["weights"]
    return w["sign_factor"] * ceil(w["magnitude_bits"] / hw["x2"]["cell_bits"])


def layer_shape(a):
    """(groups, rows needed per group, columns needed per group) for one matrix layer."""
    g = a.get("groups", 1)
    taps = (a["C"] // g) * a["R"] * a["S"]
    outs = a["K"] // g
    return g, taps, outs


def place_groups(a, hw):
    """Explicit placement (first-fit, diagonal fill) of every block of a layer.

    A block is (group, row fold, column chunk). Returns a list of tiles; each tile is a list of
    [group, row_off, col_off, r0, c0, nrows, ncols]: rows row_off..+nrows of the group's taps and
    columns col_off..+ncols of the group's output cells sit at (r0, c0) in the 32 x 32 tile."""
    rows, cols = hw["x2"]["rows"], hw["x2"]["cols"]
    g, taps, outs = layer_shape(a)
    ocols = outs * cells_per_weight(hw)
    tiles = []
    for grp in range(g):
        for ro in range(0, taps, rows):
            for co in range(0, ocols, cols):
                nr, nc = min(rows, taps - ro), min(cols, ocols - co)
                for t in tiles:
                    if t["r"] + nr <= rows and t["c"] + nc <= cols:
                        t["blocks"].append([grp, ro, co, t["r"], t["c"], nr, nc]); t["r"] += nr; t["c"] += nc
                        break
                else:
                    tiles.append({"r": nr, "c": nc, "blocks": [[grp, ro, co, 0, 0, nr, nc]]})
    return [t["blocks"] for t in tiles]


def check_placement(a, hw, placement):
    """Independent check of ANY placement (planner's or proposed). Returns a list of errors."""
    rows, cols = hw["x2"]["rows"], hw["x2"]["cols"]
    g, taps, outs = layer_shape(a)
    ocols = outs * cells_per_weight(hw)
    errs, cover = [], {}
    if not isinstance(placement, list):
        return ["placement must be a list of tiles"]
    for ti, tile in enumerate(placement):
        rr, cc = [], []
        for b in tile if isinstance(tile, list) else []:
            if not (isinstance(b, list) and len(b) == 7 and all(isinstance(v, int) for v in b)):
                errs.append(f"tile {ti}: bad block {b}"); continue
            grp, ro, co, r0, c0, nr, nc = b
            if not (0 <= grp < g and nr > 0 and nc > 0 and r0 >= 0 and c0 >= 0 and r0 + nr <= rows and c0 + nc <= cols):
                errs.append(f"tile {ti}: block {b} outside the {rows} x {cols} array or group range"); continue
            if not (0 <= ro and ro + nr <= taps and 0 <= co and co + nc <= ocols):
                errs.append(f"tile {ti}: block {b} outside its group's {taps} x {ocols} weight matrix"); continue
            rr.append((r0, r0 + nr)); cc.append((c0, c0 + nc))
            for r in range(ro, ro + nr):
                key = (grp, r)
                cover.setdefault(key, []).append((co, co + nc))
        # blocks in one tile share one MVM: they need their own input rows AND their own output columns
        for spans, what in ((rr, "rows"), (cc, "columns")):
            spans = sorted(spans)
            if any(a1[1] > a2[0] for a1, a2 in zip(spans, spans[1:])):
                errs.append(f"tile {ti}: blocks share {what}")
    for grp in range(g):
        for r in range(taps):
            got = sorted(cover.get((grp, r), []))
            pos = 0
            for lo, hi in got:
                if lo != pos:
                    break
                pos = hi
            if pos != ocols or len(got) != len(set(got)) or sum(h - l for l, h in got) != ocols:
                errs.append(f"group {grp} tap {r}: weight columns not covered exactly once")
                return errs
    return errs


def layer_tiles(a, hw):
    return len(place_groups(a, hw))


def area_lower_bound(a, hw):
    return -(-layer_cells(a, hw) // (hw["x2"]["rows"] * hw["x2"]["cols"]))


def layer_cells(a, hw):
    """Cells that hold weights (upper bound on cells to program; the exact number depends on the values)."""
    g, taps, outs = layer_shape(a)
    return g * taps * outs * cells_per_weight(hw)


def mvm_count(a, out_numel):
    """Crossbar MVMs per tile per inference = output positions (out elements / K)."""
    return max(1, out_numel // a["K"])


def cycles(hw, a, out_numel, tiles):
    t = hw["x2"]["timing_cycles"]
    rows, cols = hw["x2"]["rows"], hw["x2"]["cols"]
    per_mvm = rows + t["compute"] + cols                # beats + COMPUTE_DELAY + one result per column (lower bound)
    return {"compute": tiles * mvm_count(a, out_numel) * per_mvm,
            "program_upper": layer_cells(a, hw) * t["program"]}


def best_resident_set(layers, blocks):
    """layers: {name: {"tiles": t, "program_upper": c}}. Keep the set of layers programmed that saves the most
    programming per inference. If any layer is left streamed, one block must stay free to stream into it."""
    names = sorted(layers)
    if sum(layers[n]["tiles"] for n in names) <= blocks:
        return names
    cap = blocks - 1
    best = (0, [])
    def rec(i, used, saved, chosen):                    # tiny exhaustive search (cap is 1 on this chip)
        nonlocal best
        if saved > best[0] or (saved == best[0] and chosen < best[1]):
            best = (saved, list(chosen))
        for j in range(i, len(names)):
            t = layers[names[j]]["tiles"]
            if used + t <= cap:
                rec(j + 1, used + t, saved + layers[names[j]]["program_upper"], chosen + [names[j]])
    rec(0, 0, 0, [])
    return sorted(best[1])
