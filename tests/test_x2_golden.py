"""Phase 2: X2 golden model, checked against real Icarus logs of the BM Labs behavioral model.

* unit tests of the golden model's rules (no simulator);
* replay: the golden model reproduces the bundled simulator logs exactly;
* planted bugs: 9 deliberately wrong golden models, each must be caught by the logs;
* live: if iverilog is on PATH and the chip repo is present, simulate again.
"""
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chip.x2_check import all_scenarios, compare, parse_log  # noqa: E402
from chip.x2_golden import (MODE_COMPUTE, MODE_READ, MODE_RESET, MODE_SET, Result,  # noqa: E402
                            X2Config, X2Golden, unpack_result)

FIX = ROOT / "tests" / "fixtures" / "x2_check"
CFG = X2Config(target_set1=100, target_set2=200, target_reset1=10, target_reset2=30, tdc_time_out=62)


# ---------------------------------------------------------------- unit tests
def test_ceiling_formula():
    assert X2Config().ceiling == 255
    assert X2Config(tdc_time_out=32).ceiling == 0x20FF
    assert X2Config(tdc_time_out=63).ceiling == 0x3FFF
    assert X2Config(tdc_time_out=127).ceiling == 0x3FFF  # only bits [5:0] count


def test_config_rejects_values_that_do_not_fit():
    with pytest.raises(ValueError):
        X2Config(tdc_time_out=128)


def test_read_set_reset_and_readback():
    g = X2Golden(CFG)
    assert g.command(MODE_READ, 0, 0, 5) == [Result(MODE_READ, 0, 0x0E23 + 5, True)]
    # SET level = midpoint(100, 200) + pwm = 150 + 2; readback uses the previous READ pwm (5)
    assert g.command(MODE_SET, 0, 0, 2) == [Result(MODE_SET, 0, 152 + 5, True)]
    assert g.state[0][0] == 1
    assert g.command(MODE_RESET, 0, 0, 0) == [Result(MODE_RESET, 0, 20 + 5, True)]
    assert g.state[0][0] == 0


def test_compute_is_binary_dot_product_and_ignores_level():
    g = X2Golden(CFG)
    g.program_weights([[1, 0], [1, 1]])
    out = g.matvec([10, 20], cols=[0, 1])
    assert [(r.col, r.count, r.last) for r in out] == [(0, 30, False), (1, 20, True)]


def test_compute_saturates_at_ceiling():
    g = X2Golden(X2Config())  # ceiling 255
    g.program_weights([[1], [1]])
    assert g.matvec([200, 100], cols=[0])[0].count == 255


def test_stream_rules_duplicate_row_and_cancel():
    g = X2Golden(CFG)
    g.program_weights([[1]])
    assert g.command(MODE_COMPUTE, 0, 0, 7, False) == []
    assert g.command(MODE_COMPUTE, 0, 0, 9, True)[0].count == 9      # later pwm wins
    g.command(MODE_COMPUTE, 0, 0, 50, False)
    g.command(MODE_READ, 0, 0, 0)                                    # cancels the stream
    assert g.s_rows == set()


def test_more_columns_than_rows_repeats_last_row():
    beats = X2Golden(CFG).matvec_beats([None, 4], cols=[3, 1, 2])
    assert [(b[1], b[2], b[3], b[4]) for b in beats] == [(1, 3, 4, False), (1, 1, 4, False), (1, 2, 4, True)]


def test_weights_must_be_binary():
    with pytest.raises(ValueError):
        X2Golden(CFG).program_weights([[2]])


def test_result_packing_roundtrip():
    r = Result(MODE_COMPUTE, 17, 12345, False)
    assert unpack_result(r.packed()) == r
    assert r.packed() < (1 << 21)


# ---------------------------------------------------------------- replay
def _logs():
    return {sc.name: (sc, (FIX / f"{sc.name}.sim.txt").read_text()) for sc in all_scenarios()}


def test_fixture_matches_scenarios():
    man = json.loads((FIX / "MANIFEST.json").read_text())
    assert man["behavioral_model"].endswith("sky130_bmlabs_reram__cim32x32.sv")
    for name, (sc, text) in _logs().items():
        _, _, done = parse_log(text)
        assert done and done["beats"] == len(sc.beats), name


@pytest.mark.parametrize("name", [sc.name for sc in all_scenarios()])
def test_golden_reproduces_simulator(name):
    sc, text = _logs()[name]
    rep = compare(sc, text)
    assert rep["pass"], rep["mismatches"][:3] or rep["bench_problems"]


def test_binary_matmul_numeric_check():
    sc, text = _logs()["binary_matmul_32x32"]
    n = compare(sc, text)["numeric_check"]
    assert n["ok"] and n["values"] == 128


# ---------------------------------------------------------------- planted bugs
class UsesAllRows(X2Golden):            # compute ignores the cell state
    def command(self, mode, row, col, pwm, last=False, all_cols=False):
        saved = self.state
        if mode == MODE_COMPUTE and last:
            self.state = [[1] * self.cols for _ in range(self.rows)]
        try:
            return super().command(mode, row, col, pwm, last, all_cols)
        finally:
            self.state = saved


class NoClamp(X2Golden):
    def clamp(self, raw):
        return raw


class ClampOffByOne(X2Golden):
    def clamp(self, raw):
        return min(raw, self.cfg.ceiling - 1)


class NoCancel(X2Golden):
    def _clear_stream(self):
        if not hasattr(self, "s_rows"):
            super()._clear_stream()


class ReadbackUsesOwnPwm(X2Golden):
    def command(self, mode, row, col, pwm, last=False, all_cols=False):
        if mode in (MODE_SET, MODE_RESET):
            self.previous_read_pwm = pwm
        return super().command(mode, row, col, pwm, last, all_cols)


class FirstPwmWins(X2Golden):
    def command(self, mode, row, col, pwm, last=False, all_cols=False):
        if mode == MODE_COMPUTE and row in self.s_pwm:
            pwm = self.s_pwm[row]
        return super().command(mode, row, col, pwm, last, all_cols)


class MidpointNoHalving(X2Golden):
    def _programmed_level(self, set_cell, pwm):
        a, b = ((self.cfg.target_set1, self.cfg.target_set2) if set_cell
                else (self.cfg.target_reset1, self.cfg.target_reset2))
        return (a + b + pwm) & 0x3FFF


class LastFlagOnFirst(X2Golden):
    def command(self, *a, **k):
        res = super().command(*a, **k)
        return [r._replace(last=(i == 0)) for i, r in enumerate(res)]


class ColumnsInBeatOrder(X2Golden):
    def command(self, mode, row, col, pwm, last=False, all_cols=False):
        order = list(dict.fromkeys(list(getattr(self, "_order", [])) + [col]))
        self._order = order
        res = super().command(mode, row, col, pwm, last, all_cols)
        if mode == MODE_COMPUTE and last and not all_cols and len(res) > 1:
            by = {r.col: r for r in res}
            res = [by[c]._replace(last=False) for c in order if c in by]
            res[-1] = res[-1]._replace(last=True)
        if mode != MODE_COMPUTE or last:
            self._order = []
        return res


MUTANTS = [UsesAllRows, NoClamp, ClampOffByOne, NoCancel, ReadbackUsesOwnPwm, FirstPwmWins,
           MidpointNoHalving, LastFlagOnFirst, ColumnsInBeatOrder]


@pytest.mark.parametrize("mutant", MUTANTS, ids=lambda m: m.__name__)
def test_planted_bug_is_caught(mutant):
    caught_by = [name for name, (sc, text) in _logs().items() if not compare(sc, text, mutant)["pass"]]
    assert caught_by, f"{mutant.__name__} was not caught by any scenario"


# ---------------------------------------------------------------- live simulation
def test_live_simulation(tmp_path):
    if not (shutil.which("iverilog") and shutil.which("vvp")):
        pytest.skip("iverilog not on PATH (expected on Windows; run run_x2_check.py --sim in Ubuntu)")
    from chip.chip_repo import INPUT_FILES, ChipRepo, ChipRepoError
    try:
        repo = ChipRepo()
    except ChipRepoError as e:
        pytest.skip(str(e))
    if not (repo.root / INPUT_FILES["x2_behavioral"]).is_file():
        pytest.skip("X2 behavioral model missing (submodule not checked out)")
    import run_x2_check
    assert run_x2_check.main(["--sim", "--out", str(tmp_path)]) == 0
    for sc in all_scenarios():
        assert (tmp_path / f"{sc.name}.sim.txt").read_text() == (FIX / f"{sc.name}.sim.txt").read_text()


def test_fixture_was_recorded_from_this_model_version():
    """If the chip repo's X2 model changes, the bundled logs are stale: say so instead of passing."""
    from chip.chip_repo import INPUT_FILES, ChipRepo, ChipRepoError, lf_sha256
    try:
        repo = ChipRepo()
    except ChipRepoError as e:
        pytest.skip(str(e))
    if not (repo.root / INPUT_FILES["x2_behavioral"]).is_file():
        pytest.skip("X2 behavioral model missing (submodule not checked out)")
    man = json.loads((FIX / "MANIFEST.json").read_text())
    assert lf_sha256(repo.read_bytes(INPUT_FILES["x2_behavioral"])) == man["behavioral_model_lf_sha256"], \
        "chip repo has a different X2 model than the bundled logs: re-run run_x2_check.py --sim"
