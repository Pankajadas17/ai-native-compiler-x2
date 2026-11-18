"""S04 X2 package: turn one matrix layer into what the X2 actually receives, and the recipe to rebuild the output.

For a layer with INT8 weights W (K, C/groups, R, S) and unsigned 8-bit inputs:

  1. weight cells   every signed weight is split DeepGate-style (BMsemi deep_gate/Crossbar_array.py l.29-30, 127, 165):
                    w = w_pos - w_neg, each magnitude split into digits of x2.cell_bits bits
                    -> cells_per_weight = 2 * ceil(8 / cell_bits) columns per output channel
  2. placement      the same block placement as the partition (chip_x2.place_groups), so tile counts always agree
  3. inputs         each 8-bit input is sent as `input_split_bits`-bit slices, one COMPUTE pass per slice.
                    4-bit cells: two 4-bit slices keep a 32-row sum at 32 x 15 x 15 = 7200, under the 14-bit
                    result cap (16383); a full 8-bit input would reach 32 x 255 x 15 = 122400 and saturate
  4. recombination  y = sum over passes 2^(p*split) * sum over sign (+/-) * sum over digit 2^(d*cell_bits) * count,
                    with counts of row folds (tiles holding the same columns, different rows) added together

Commands: with 1-bit cells the stream uses only the chip's own commands (SET / RESET / COMPUTE), so it can run on
the BM Labs X2 model (x2c sim pack --sim). With 4-bit cells, writing a 4-bit value is NOT in the chip's
command set (the chip repo's X2 stores 1 bit), so that stream uses an explicit WRITE_LEVEL command and is executed
by MultiLevelX2 below: a model of the 4-bit decision, checked against NumPy only.

ASSUMED: patch order c-r-s (channel, kernel row, kernel column), as in the earlier compiler checks.
"""
from dataclasses import dataclass, field
from math import ceil

import numpy as np

from x2c.compiler import chip_x2

MODE_RESET, MODE_READ, MODE_COMPUTE, MODE_SET = 0, 1, 2, 3
WRITE_LEVEL = "WRITE_LEVEL"          # 4-bit cells only: not a chip command (see module text)
RESULT_CAP = (1 << 14) - 1           # 14-bit count; ceiling = (TDC_TIME_OUT[5:0] << 8) | 0xFF, at most 16383


# ---------------------------------------------------------------- weights -> cell digits
def digits(hw):
    """[(sign, digit_index, scale)] in column order for one output channel."""
    cb = hw["x2"]["cell_bits"]
    nd = ceil(hw["weights"]["magnitude_bits"] / cb)
    return [(s, d, (1 << (d * cb))) for s in (+1, -1) for d in range(nd)]


def weight_columns(w_int8, hw):
    """w_int8: (taps, outs) int array -> (taps, outs * cells_per_weight) array of cell values in 0 .. 2^cell_bits - 1.
    Column order per output channel: (+, digit 0), (+, digit 1), ..., (-, digit 0), ..."""
    w = np.asarray(w_int8, dtype=np.int64)
    if w.min() < -128 or w.max() > 127:
        raise ValueError("weights must be INT8 (-128..127)")
    cb = hw["x2"]["cell_bits"]
    mask = (1 << cb) - 1
    cols = []
    for k in range(w.shape[1]):
        for s, d, _ in digits(hw):
            mag = np.maximum(s * w[:, k], 0)
            cols.append((mag >> (d * cb)) & mask)
    return np.stack(cols, axis=1)


# ---------------------------------------------------------------- the package
@dataclass
class LayerPackage:
    name: str
    attrs: dict
    hw: dict
    input_split_bits: int
    placement: list                      # chip_x2.place_groups(...)
    cells: list                          # per tile: 32 x 32 int array of cell values
    w_cols: list                         # per group: (taps, outs*cpw)
    required_tdc_time_out: int
    max_count: int
    notes: list = field(default_factory=list)

    @property
    def tiles(self):
        return len(self.placement)


def required_tdc(max_count):
    """Smallest TDC_TIME_OUT (0..63) whose ceiling (t << 8) | 0xFF is >= max_count; None if even 63 is too small."""
    for t in range(64):
        if ((t << 8) | 0xFF) >= max_count:
            return t
    return None


def pack_layer(name, attrs, weights, hw, input_split_bits=None):
    """weights: (K, C/groups, R, S) INT8. Returns a LayerPackage, or raises if a COMPUTE could saturate."""
    g, taps, outs = chip_x2.layer_shape(attrs)
    w = np.asarray(weights, dtype=np.int64)
    if w.shape != (attrs["K"], attrs["C"] // g, attrs["R"], attrs["S"]):
        raise ValueError(f"{name}: weights shape {w.shape} != (K, C/groups, R, S)")
    cb = hw["x2"]["cell_bits"]
    split = input_split_bits or (4 if cb > 1 else hw["x2"]["input_bits"])
    if not (1 <= split <= hw["x2"]["input_bits"]):
        raise ValueError("input_split_bits must be 1..input_bits")
    w_cols = [weight_columns(w[gi * outs:(gi + 1) * outs].reshape(outs, taps).T, hw) for gi in range(g)]
    placement = chip_x2.place_groups(attrs, hw)
    rows, cols = hw["x2"]["rows"], hw["x2"]["cols"]
    cells = []
    max_count = 0
    for tile in placement:
        t = np.zeros((rows, cols), dtype=np.int64)
        for grp, ro, co, r0, c0, nr, nc in tile:
            t[r0:r0 + nr, c0:c0 + nc] = w_cols[grp][ro:ro + nr, co:co + nc]
        cells.append(t)
        # worst case of one COMPUTE: every used row at the largest input slice
        max_count = max(max_count, int(t.sum(axis=0).max()) * ((1 << split) - 1))
    tdc = required_tdc(max_count)
    if tdc is None:
        raise ValueError(f"{name}: one COMPUTE can reach {max_count} > {RESULT_CAP} (14-bit result) with {cb}-bit cells "
                         f"and {split}-bit input slices; use smaller input slices")
    return LayerPackage(name, dict(attrs), hw, split, placement, cells, w_cols, tdc, max_count,
                        [f"inputs sent as {ceil(hw['x2']['input_bits'] / split)} slice(s) of {split} bits",
                         f"needs TDC_TIME_OUT >= {tdc} (ceiling {(tdc << 8) | 0xFF} >= worst count {max_count})"])


# ---------------------------------------------------------------- command stream
def program_commands(pkg, tile_index, current=None):
    """Commands that turn the block's current cells into this tile's cells (only cells that change).
    1-bit: SET / RESET (chip commands). 4-bit: WRITE_LEVEL (not a chip command)."""
    target = pkg.cells[tile_index]
    cur = np.zeros_like(target) if current is None else current
    cmds = []
    for r, c in zip(*np.nonzero(target != cur)):
        v = int(target[r, c])
        if pkg.hw["x2"]["cell_bits"] == 1:
            cmds.append((MODE_SET if v else MODE_RESET, int(r), int(c), 0, False, False))
        else:
            cmds.append((WRITE_LEVEL, int(r), int(c), v))
    return cmds


def input_slices(x_u8, split):
    """uint8 array -> list of (scale, slice) for each pass, least significant first."""
    x = np.asarray(x_u8, dtype=np.int64)
    if x.min() < 0 or x.max() > 255:
        raise ValueError("inputs must be unsigned 8-bit (0..255)")
    n = ceil(8 / split)
    return [(1 << (p * split), (x >> (p * split)) & ((1 << split) - 1)) for p in range(n)]


def compute_beats(pkg, tile_index, patch_by_group, slice_index):
    """COMPUTE beats for one MVM of one tile: one beat per used row, pwm = that row's input slice.
    patch_by_group: list (per group) of the taps vector (uint8) for this output position."""
    rows = {}
    for grp, ro, co, r0, c0, nr, nc in pkg.placement[tile_index]:
        for i in range(nr):
            rows[r0 + i] = int(patch_by_group[grp][ro + i])
    order = sorted(rows)
    beats = []
    for i, r in enumerate(order):
        sl = (rows[r] >> (slice_index * pkg.input_split_bits)) & ((1 << pkg.input_split_bits) - 1)
        beats.append((MODE_COMPUTE, r, 0, sl, i == len(order) - 1, True))
    return beats


def recombine(pkg, counts):
    """counts[(tile, slice)] = list of 32 column counts -> outputs per group: list of arrays (outs,)."""
    g, taps, outs = chip_x2.layer_shape(pkg.attrs)
    dg = digits(pkg.hw)
    cpw = len(dg)
    y = [np.zeros(outs, dtype=np.int64) for _ in range(g)]
    for (ti, si), col_counts in counts.items():
        scale_in = 1 << (si * pkg.input_split_bits)
        for grp, ro, co, r0, c0, nr, nc in pkg.placement[ti]:
            for j in range(nc):
                gc = co + j                         # column inside the group's expanded matrix
                k, slot = divmod(gc, cpw)
                s, _, scale_w = dg[slot]
                y[grp][k] += scale_in * s * scale_w * int(col_counts[c0 + j])
    return y


# ---------------------------------------------------------------- executing the stream
class MultiLevelX2:
    """Model of ONE X2 block with multi-level cells (the 4-bit decision). COMPUTE: count[col] = sum of pwm * cell value
    over the selected rows, clamped at the ceiling. With 1-bit cells use chip.x2_golden.X2Golden instead (chip model)."""
    def __init__(self, rows, cols, ceiling):
        self.cells = np.zeros((rows, cols), dtype=np.int64)
        self.ceiling, self._rows, self._pwm = ceiling, [], {}

    def apply(self, cmd):
        if cmd[0] == WRITE_LEVEL:
            _, r, c, v = cmd
            self.cells[r, c] = v
            return None
        mode, r, c, pwm, last, all_cols = cmd
        if mode != MODE_COMPUTE:
            raise ValueError("MultiLevelX2 only models WRITE_LEVEL and COMPUTE")
        self._pwm[r] = pwm
        if not last:
            return None
        rows, self._pwm = dict(self._pwm), {}
        acc = np.zeros(self.cells.shape[1], dtype=np.int64)
        for rr, p in rows.items():
            acc += p * self.cells[rr]
        return [int(min(v, self.ceiling)) for v in acc]


def recombine_steps(pkg, counts):
    """The same sum as recombine(), as the step list the eFPGA recombiner executes:
    per output channel (in output order), [(count, shift, neg)] with acc += (neg ? -1 : 1) * (count << shift)."""
    g, taps, outs = chip_x2.layer_shape(pkg.attrs)
    dg = digits(pkg.hw)
    cpw = len(dg)
    cb = pkg.hw["x2"]["cell_bits"]
    steps = [[] for _ in range(g * outs)]
    for (ti, si), col_counts in sorted(counts.items()):
        for grp, ro, co, r0, c0, nr, nc in pkg.placement[ti]:
            for j in range(nc):
                k, slot = divmod(co + j, cpw)
                s, d, _ = dg[slot]
                steps[grp * outs + k].append((int(col_counts[c0 + j]), si * pkg.input_split_bits + d * cb, 1 if s < 0 else 0))
    return steps


def run_layer(pkg, patches, executor="auto", return_counts=False):
    """Run every position through the package on ONE simulated X2 block, tile by tile (reprogramming in between).
    patches: (positions, groups, taps) uint8. Returns (outputs (positions, K), stats)."""
    from x2c.chip.x2_golden import X2Config, X2Golden
    cb = pkg.hw["x2"]["cell_bits"]
    ceiling = (pkg.required_tdc_time_out << 8) | 0xFF
    if cb == 1:
        blk = X2Golden(X2Config(tdc_time_out=pkg.required_tdc_time_out), rows=pkg.hw["x2"]["rows"], cols=pkg.hw["x2"]["cols"])
    else:
        blk = MultiLevelX2(pkg.hw["x2"]["rows"], pkg.hw["x2"]["cols"], ceiling)
    g, taps, outs = chip_x2.layer_shape(pkg.attrs)
    n_slices = ceil(8 / pkg.input_split_bits)
    P = patches.shape[0]
    counts = [dict() for _ in range(P)]
    state = np.zeros((pkg.hw["x2"]["rows"], pkg.hw["x2"]["cols"]), dtype=np.int64)
    stats = {"program_commands": 0, "compute_beats": 0, "computes": 0}
    for ti in range(pkg.tiles):
        prog = program_commands(pkg, ti, state)
        for cmd in prog:
            if cb == 1:
                blk.command(*cmd)
            else:
                blk.apply(cmd)
        stats["program_commands"] += len(prog)
        state = pkg.cells[ti].copy()
        for p in range(P):
            for si in range(n_slices):
                beats = compute_beats(pkg, ti, patches[p], si)
                res = None
                for b in beats:
                    out = blk.command(*b) if cb == 1 else blk.apply(b)
                    if out:
                        res = out
                stats["compute_beats"] += len(beats)
                stats["computes"] += 1
                if cb == 1:
                    col = [0] * pkg.hw["x2"]["cols"]
                    for r in res:
                        col[r.col] = r.count
                    res = col
                counts[p][(ti, si)] = res
    y = np.stack([np.concatenate(recombine(pkg, counts[p])) for p in range(P)])
    return (y, stats, counts) if return_counts else (y, stats)


def reference(weights, patches, attrs):
    """NumPy reference: per group, y = patch . W (exact integers). patches (positions, groups, taps)."""
    g, taps, outs = chip_x2.layer_shape(attrs)
    w = np.asarray(weights, dtype=np.int64).reshape(attrs["K"], taps)
    return np.stack([np.concatenate([patches[p, gi].astype(np.int64) @ w[gi * outs:(gi + 1) * outs].T
                                     for gi in range(g)]) for p in range(patches.shape[0])])
