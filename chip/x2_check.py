"""Scenarios, vector files and comparison for checking X2Golden against the BM Labs model.

Vector file (read by sim/x2_vector_tb.sv):
    CFG <8 parameters in X2Config order>
    B <mode> <row> <col> <pwm> <compute_last> <compute_all_columns>
Expected file / simulator log:
    R <mode> <col> <count> <last>
"""
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from chip.x2_golden import (MODE_COMPUTE, MODE_NAMES, MODE_READ, MODE_RESET, MODE_SET,
                            Result, X2Config, X2Golden)

Beat = Tuple[int, int, int, int, bool, bool]

# Configuration used by the vendor testbench (sky130_bmlabs_reram__cim32x32_tb.sv).
VENDOR_CFG = X2Config(target_set1=0x6472, target_set2=0x0003, target_reset1=0x000B,
                      target_reset2=0x462B, no_of_clk_cycles=5, counter_value=5,
                      tdc_time_out=62, tdc_dead_time=2)
# All parameters 0: what the chip repo's reram_cim_32x32 example design programs.
ZERO_CFG = X2Config()


@dataclass
class Scenario:
    name: str
    purpose: str
    cfg: X2Config
    beats: List[Beat]
    numeric_check: dict = None  # optional: {"W": ..., "xs": [...], "expect": [...]}

    def expected(self, golden_cls=X2Golden):
        g = golden_cls(self.cfg)
        out = []
        for b in self.beats:
            out += g.command(*b)
        return out


def _prog(w, r, c, pwm=0):
    return (MODE_SET if w else MODE_RESET, r, c, pwm, False, False)


def _read(r, c, pwm):
    return (MODE_READ, r, c, pwm, False, False)


def _stream(x, cols=None):
    return X2Golden(ZERO_CFG).matvec_beats(x, cols)


def scenario_directed():
    b = [
        _read(0, 0, 0), _read(31, 31, 200),                 # initial level 0x0E23 + pwm
        _prog(1, 0, 0, 5), _read(0, 0, 7), _prog(1, 0, 1, 9),  # SET readback uses previous READ pwm (7)
        _prog(0, 0, 1, 3), _read(0, 1, 0),                    # RESET
        _prog(1, 1, 0), _prog(1, 2, 0), _prog(1, 2, 1), _prog(1, 3, 31),
    ]
    b += _stream([10, None, None, None])                      # one row, all columns
    b += _stream([10, 20, 30, 40], cols=[0])                  # four rows, one column
    b += _stream([10, 20, 30, 40], cols=[1, 0, 31])           # three columns, given out of order
    b += _stream([None, None, 255, None], cols=[0, 1, 2, 3, 4, 5])  # more columns than rows
    # same row twice in one stream: the later pwm wins
    b += [(MODE_COMPUTE, 1, 0, 50, False, False), (MODE_COMPUTE, 1, 0, 60, True, False)]
    # an unfinished stream is cancelled by a READ; the next stream starts clean
    b += [(MODE_COMPUTE, 2, 0, 99, False, True), _read(2, 0, 1)]
    b += _stream([None, 11, None, None], cols=[0])
    return Scenario("directed", "each rule of the model once (read, set/reset readback, "
                    "stream rules, cancel)", VENDOR_CFG, b)


def scenario_binary_matmul(seed=2026, n_vectors=4):
    rnd = random.Random(seed)
    W = [[rnd.randint(0, 1) for _ in range(32)] for _ in range(32)]
    xs = [[rnd.randint(0, 255) for _ in range(32)] for _ in range(n_vectors)]
    b = [_prog(1, r, c) for r in range(32) for c in range(32) if W[r][c]]
    for x in xs:
        b += _stream(x)
    expect = [[sum(x[r] * W[r][c] for r in range(32)) for c in range(32)] for x in xs]
    return Scenario("binary_matmul_32x32", "full 32x32 array, 0/1 weights, y = x @ W for "
                    f"{n_vectors} random 8-bit input vectors", VENDOR_CFG, b,
                    {"W": W, "xs": xs, "expect": expect})


def scenario_zero_params():
    b = [_read(0, 0, 0)]
    b += [_prog(1, r, 0) for r in range(4)]
    b += _stream([100, 100, 100, 100], cols=[0])              # 400 > ceiling 255
    b += _stream([100, 100, None, None], cols=[0])            # 200 fits
    return Scenario("zero_parameters", "all eight X2 parameters 0 (as in the repo's example "
                    "design): ceiling = 255, so larger counts saturate", ZERO_CFG, b)


def scenario_random(seed=7, n_ops=120):
    rnd = random.Random(seed)
    cfg = X2Config(target_set1=rnd.randrange(1 << 16), target_set2=rnd.randrange(1 << 16),
                   target_reset1=rnd.randrange(1 << 16), target_reset2=rnd.randrange(1 << 16),
                   no_of_clk_cycles=rnd.randrange(8), counter_value=rnd.randrange(8),
                   tdc_time_out=rnd.choice([20, 40, 63]), tdc_dead_time=rnd.randrange(4))
    b = []
    for _ in range(n_ops):
        k = rnd.random()
        r, c, p = rnd.randrange(32), rnd.randrange(32), rnd.randrange(256)
        if k < 0.35:
            b.append(_prog(rnd.randint(0, 1), r, c, p))
        elif k < 0.55:
            b.append(_read(r, c, p))
        elif k < 0.65:  # unfinished stream, cancelled by the next non-compute command
            b.append((MODE_COMPUTE, r, c, p, False, rnd.random() < 0.5))
            b.append(_read(r, c, p))
        else:
            x = [rnd.randrange(256) if rnd.random() < 0.4 else None for _ in range(32)]
            if all(v is None for v in x):
                x[r] = p
            cols = None if rnd.random() < 0.3 else rnd.sample(range(32), rnd.randint(1, 6))
            b += _stream(x, cols)
    return Scenario("random_mixed", f"{n_ops} random operations, random parameters "
                    "(ceiling, targets, dead time)", cfg, b)


def all_scenarios():
    return [scenario_directed(), scenario_binary_matmul(), scenario_zero_params(), scenario_random()]


# ---------------------------------------------------------------- files
def write_vectors(path, sc):
    lines = ["CFG " + " ".join(str(v) for v in sc.cfg.as_list())]
    lines += ["B %d %d %d %d %d %d" % (m, r, c, p, int(l), int(a)) for m, r, c, p, l, a in sc.beats]
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")


def write_results(path, results):
    Path(path).write_text("".join(f"R {x.mode} {x.col} {x.count} {int(x.last)}\n" for x in results),
                          encoding="ascii", newline="\n")


def parse_log(text):
    results, failures, done = [], [], None
    for line in text.splitlines():
        f = line.split()
        if not f:
            continue
        if f[0] == "R" and len(f) == 5:
            results.append(Result(int(f[1]), int(f[2]), int(f[3]), f[4] == "1"))
        elif f[0] == "F":
            failures.append(line[2:])
        elif f[0] == "DONE" and len(f) == 4:
            done = {"beats": int(f[1]), "results": int(f[2]), "cycles": int(f[3])}
    return results, failures, done


def compare(sc, sim_text, golden_cls=X2Golden):
    expected = sc.expected(golden_cls)
    actual, failures, done = parse_log(sim_text)
    mism = []
    for i, (e, a) in enumerate(zip(expected, actual)):
        if e != a:
            mism.append({"index": i, "expected": _fmt(e), "simulated": _fmt(a)})
    if len(expected) != len(actual):
        mism.append({"index": min(len(expected), len(actual)),
                     "expected": f"{len(expected)} results", "simulated": f"{len(actual)} results"})
    problems = list(failures)
    if done is None:
        problems.append("simulation log has no DONE line (crashed or timed out?)")
    elif done["beats"] != len(sc.beats):
        problems.append(f"bench sent {done['beats']} of {len(sc.beats)} beats")
    numeric = None
    if sc.numeric_check:
        numeric = _numeric(sc, actual)
    ok = not mism and not problems and (numeric is None or numeric["ok"])
    return {"scenario": sc.name, "purpose": sc.purpose, "pass": ok,
            "beats": len(sc.beats), "expected_results": len(expected), "simulated_results": len(actual),
            "mismatches": mism[:20], "mismatch_count": len(mism), "bench_problems": problems,
            "cycles": done["cycles"] if done else None, "numeric_check": numeric}


def _numeric(sc, actual):
    """Independent check: simulated COMPUTE counts equal x @ W computed with plain Python."""
    comp = [x.count for x in actual if x.mode == MODE_COMPUTE]
    want = [v for row in sc.numeric_check["expect"] for v in row]
    return {"what": "simulated COMPUTE counts == x @ W", "ok": comp == want,
            "vectors": len(sc.numeric_check["xs"]), "values": len(want), "max_value": max(want)}


def _fmt(r):
    return f"{MODE_NAMES[r.mode]} col={r.col} count={r.count} last={int(r.last)}"
