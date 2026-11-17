"""Python golden model of one X2 block (BM Labs sky130_bmlabs_reram__cim32x32).

Transaction level: it models WHAT the block returns (mode, column, count, last flag)
for a stream of command beats, not WHEN. It is written from the behavioral model
`ip/sky130_bmlabs_reram/verilog/behavioral/sky130_bmlabs_reram__cim32x32.sv` in the
chip repo and is checked against it by running both on the same commands
(run_x2_check.py).

What the behavioral model does (and so what this model does):
  * Each cell has a binary state (SET = 1, RESET = 0) and a 14-bit level.
  * COMPUTE: count[col] = sum of pwm[row] over the selected rows whose cell
    (row, col) is SET, i.e. a dot product with 0/1 weights and 8-bit inputs.
    The cell LEVEL is not used by COMPUTE.
  * READ: count = cell level + pwm.
  * SET / RESET: stores state 1 / 0 and level = (midpoint of the two targets + pwm)
    mod 2**14, then returns a readback = level + the PWM of the previous READ.
  * Every count is clamped at ceiling = (TDC_TIME_OUT[5:0] << 8) | 0xFF.
  * COMPUTE is a stream of beats (one per row); the stream launches on the beat
    with compute_last. Any other command cancels an unfinished stream.
"""
from dataclasses import dataclass, field
from typing import List, NamedTuple

MODE_RESET, MODE_READ, MODE_COMPUTE, MODE_SET = 0, 1, 2, 3
MODE_NAMES = {MODE_RESET: "RESET", MODE_READ: "READ", MODE_COMPUTE: "COMPUTE", MODE_SET: "SET"}
LEVEL_MASK = (1 << 14) - 1


@dataclass(frozen=True)
class X2Config:
    """The eight X2 parameters. On the chip they come from the bitstream (93 config bits)."""
    target_set1: int = 0
    target_set2: int = 0
    target_reset1: int = 0
    target_reset2: int = 0
    no_of_clk_cycles: int = 0
    counter_value: int = 0
    tdc_time_out: int = 0
    tdc_dead_time: int = 0

    WIDTHS = {"target_set1": 16, "target_set2": 16, "target_reset1": 16, "target_reset2": 16,
              "no_of_clk_cycles": 10, "counter_value": 10, "tdc_time_out": 7, "tdc_dead_time": 2}

    def __post_init__(self):
        for name, width in self.WIDTHS.items():
            v = getattr(self, name)
            if not (0 <= v < (1 << width)):
                raise ValueError(f"{name}={v} does not fit in {width} bits")

    def as_list(self):
        return [getattr(self, n) for n in self.WIDTHS]

    @property
    def ceiling(self):
        return ((self.tdc_time_out & 0x3F) << 8) | 0xFF


class Result(NamedTuple):
    mode: int
    col: int
    count: int
    last: bool

    def packed(self):
        """21-bit result word {mode[1:0], column[4:0], count[13:0]}."""
        return (self.mode << 19) | (self.col << 14) | self.count


def unpack_result(word):
    return Result((word >> 19) & 0x3, (word >> 14) & 0x1F, word & 0x3FFF, False)


@dataclass
class X2Golden:
    cfg: X2Config
    rows: int = 32
    cols: int = 32
    pwm_bits: int = 8
    initial_cell_level: int = 0x0E23
    state: List[List[int]] = field(init=False)
    level: List[List[int]] = field(init=False)

    def __post_init__(self):
        self.reset_block()

    # -- block reset (rst_i) ---------------------------------------------------
    def reset_block(self):
        self.state = [[0] * self.cols for _ in range(self.rows)]
        self.level = [[self.initial_cell_level] * self.cols for _ in range(self.rows)]
        self.previous_read_pwm = 0
        self._clear_stream()

    def _clear_stream(self):
        self.s_rows, self.s_cols, self.s_pwm, self.s_all = set(), set(), {}, False

    # -- arithmetic ------------------------------------------------------------
    def clamp(self, raw):
        return min(raw, self.cfg.ceiling)

    def _programmed_level(self, set_cell, pwm):
        a, b = ((self.cfg.target_set1, self.cfg.target_set2) if set_cell
                else (self.cfg.target_reset1, self.cfg.target_reset2))
        midpoint = ((a + b) >> 1) & LEVEL_MASK
        return (midpoint + pwm) & LEVEL_MASK

    # -- one command beat ------------------------------------------------------
    def command(self, mode, row, col, pwm, last=False, all_cols=False):
        """Apply one accepted command beat. Returns the results it produces (maybe none)."""
        self._check(row, col, pwm)
        if mode == MODE_COMPUTE:
            self.s_rows.add(row)
            self.s_cols.add(col)
            self.s_pwm[row] = pwm
            self.s_all = self.s_all or bool(all_cols)
            if not last:
                return []
            rows, pwms = sorted(self.s_rows), dict(self.s_pwm)
            out_cols = list(range(self.cols)) if self.s_all else sorted(self.s_cols)
            self._clear_stream()
            res = []
            for i, c in enumerate(out_cols):
                acc = sum(pwms[r] for r in rows if self.state[r][c])
                res.append(Result(MODE_COMPUTE, c, self.clamp(acc), i == len(out_cols) - 1))
            return res

        self._clear_stream()  # a non-COMPUTE command cancels an unfinished stream
        if mode == MODE_READ:
            self.previous_read_pwm = pwm
            return [Result(MODE_READ, col, self.clamp(self.level[row][col] + pwm), True)]
        if mode in (MODE_SET, MODE_RESET):
            set_cell = mode == MODE_SET
            self.state[row][col] = 1 if set_cell else 0
            self.level[row][col] = self._programmed_level(set_cell, pwm)
            readback = self.clamp(self.level[row][col] + self.previous_read_pwm)
            return [Result(mode, col, readback, True)]
        raise ValueError(f"unknown mode {mode}")

    def _check(self, row, col, pwm):
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"row/col out of range: {row}, {col}")
        if not (0 <= pwm < (1 << self.pwm_bits)):
            raise ValueError(f"pwm {pwm} does not fit in {self.pwm_bits} bits")

    # -- convenience -------------------------------------------------------------
    def program_weights(self, weights, pwm=0):
        """weights[row][col] in {0,1}: SET where 1, RESET where 0 (only cells that differ)."""
        res = []
        for r, row in enumerate(weights):
            for c, w in enumerate(row):
                if w not in (0, 1):
                    raise ValueError("X2 cells are binary: weights must be 0 or 1")
                if w != self.state[r][c]:
                    res += self.command(MODE_SET if w else MODE_RESET, r, c, pwm)
        return res

    def matvec_beats(self, x, cols=None):
        """Command beats for one COMPUTE stream. x[row] = pwm, None = row not selected.

        cols=None -> all columns (compute_all_columns). Otherwise the stream carries one
        column per beat, so if there are more columns than rows the last row is repeated
        (same pwm) to carry the extra columns.
        """
        rows = [(r, v) for r, v in enumerate(x) if v is not None]
        if not rows:
            raise ValueError("COMPUTE needs at least one row")
        if cols is None:
            pairs = [(r, v, 0) for r, v in rows]
            all_cols = True
        else:
            cols = list(cols)
            if not cols:
                raise ValueError("cols must not be empty")
            n = max(len(rows), len(cols))
            rows = rows + [rows[-1]] * (n - len(rows))
            pairs = [(r, v, cols[min(i, len(cols) - 1)]) for i, (r, v) in enumerate(rows)]
            all_cols = False
        return [(MODE_COMPUTE, r, c, v, i == len(pairs) - 1, all_cols) for i, (r, v, c) in enumerate(pairs)]

    def matvec(self, x, cols=None):
        out = []
        for beat in self.matvec_beats(x, cols):
            out += self.command(*beat)
        return out

def from_profile(profile, cfg):
    """Build a golden model sized from specifications/chip_profile.json."""
    x2 = profile["x2"]
    init = x2["behavioral_model"]["parameters"].get("INITIAL_CELL_LEVEL", 0x0E23)
    return X2Golden(cfg, rows=x2["rows"], cols=x2["cols"], pwm_bits=x2["pwm_bits"], initial_cell_level=init)
