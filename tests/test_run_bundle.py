"""run_bundle.py: a deployment bundle replays from its files alone; a changed cell or a changed expectation is caught."""
import json
from pathlib import Path

import numpy as np
import pytest

import deploy_chip as D
import run_bundle as RB
from compiler import int8_net as N

IR = Path(__file__).resolve().parent / "fixtures" / "ir"


@pytest.fixture(scope="module")
def bundles(tmp_path_factory):
    out = {}
    for wl, extra in (("mnist", ["--synthetic"]), ("deepgate", ["--synthetic"]), ("mobilenetv2", [])):
        d = tmp_path_factory.mktemp(wl)
        assert D.main(["--workload", wl, "--ir", str(IR / f"{wl}.json"), "--out", str(d), *extra]) == 0
        out[wl] = d
    return out


@pytest.mark.parametrize("wl", ["mnist", "deepgate", "mobilenetv2"])
def test_replays_exactly_and_deploy_checks_it(bundles, wl):
    r = RB.replay(bundles[wl])
    assert r["exact"] and r["first_mismatch"] is None and r["x2_stats"]["compute_beats"] > 0
    rep = json.loads((bundles[wl] / "report.json").read_text())
    assert rep["checks"]["bundle_replays_exactly"] and rep["bundle_replay"]["exact"]


def test_a_changed_cell_is_caught(bundles, tmp_path):
    src = bundles["mnist"]
    for f in ("host_program.json", "expected.npz"):
        (tmp_path / f).write_bytes((src / f).read_bytes())
    arrays = dict(np.load(src / "x2_program.npz"))
    key = next(k for k in arrays if k.endswith("__levels"))
    lv = arrays[key].copy()
    r, c = map(int, np.argwhere(lv > 0)[0])
    lv[r, c] -= 1                                                    # one cell programmed one level too low
    arrays[key] = lv
    np.savez_compressed(tmp_path / "x2_program.npz", **arrays)
    res = RB.replay(tmp_path)
    assert not res["exact"] and res["first_mismatch"] == "edge"


def test_a_changed_expectation_is_caught(bundles, tmp_path):
    src = bundles["deepgate"]
    for f in ("host_program.json", "x2_program.npz"):
        (tmp_path / f).write_bytes((src / f).read_bytes())
    exp = dict(np.load(src / "expected.npz"))
    exp["fc3"] = exp["fc3"].copy()
    exp["fc3"][0] += 1
    np.savez_compressed(tmp_path / "expected.npz", **exp)
    assert RB.replay(tmp_path)["first_mismatch"] == "fc3"


def test_requant_specs():
    acc = np.array([5000, -5000, 123456])
    assert np.array_equal(N.requant_from_spec({"kind": "shift_clip", "shift": 4})(acc), [127, -128, 127])
    assert np.array_equal(N.requant_from_spec({"kind": "shift_clip", "shift": 12, "round": True})(acc), [1, -1, 30])
    spec = {"bias": [0, 0, 0], "mult": 3, "shift": 14, "out_zp": 1}
    ref = lambda a: np.clip(((np.asarray(a) * 3) >> 14) + 1, -128, 127)       # noqa: E731
    assert N.match_formula(ref, spec, [acc, acc * 7]) == "floor"
    assert N.match_formula(lambda a: np.zeros_like(a) + 99, spec, [acc]) is None
    assert N.requant_from_spec({"kind": "unserialized"}) is None
    with pytest.raises(ValueError, match="not replayable"):
        RB._requant({"kind": "unserialized"}, [])


def test_cli(bundles, capsys, tmp_path):
    assert RB.main([str(bundles["mnist"])]) == 0 and "BUNDLE REPLAYS EXACTLY" in capsys.readouterr().out
    assert RB.main([str(tmp_path)]) == 1 and "CANNOT REPLAY" in capsys.readouterr().out


def test_an_old_bundle_is_refused_with_the_fix(bundles, tmp_path):
    src = bundles["mnist"]
    for f in ("x2_program.npz", "expected.npz"):
        (tmp_path / f).write_bytes((src / f).read_bytes())
    prog = json.loads((src / "host_program.json").read_text())
    prog["steps"][0]["epilogue"] = "repo requant"                     # the format before replay support
    del prog["steps"][0]["x2"]
    (tmp_path / "host_program.json").write_text(json.dumps(prog))
    with pytest.raises(ValueError, match="older deploy_chip.py"):
        RB.replay(tmp_path)
    assert RB.main([str(tmp_path)]) == 1
