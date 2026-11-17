"""The deployment path on PUBLIC stand-in graphs (tests/fixtures/ir_public/, written by tools/make_public_fixtures.py).
Runs in every checkout, team or public, so a public clone tests the same code the team's real graphs exercise."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import deploy_chip as D
import run_bundle as RB
from compiler import int8_net as N
from compiler.ir import IRGraph

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "tests" / "fixtures" / "ir_public"
CASES = {"mobilenetv2": ("invres_net", []), "mnist": ("edge5x5", ["--synthetic"]), "deepgate": ("mlp_chain", ["--synthetic"])}


def test_fixtures_are_reproducible_and_hold_no_private_data(tmp_path):
    from tools import make_public_fixtures as M
    for name, fn in M.GRAPHS.items():
        committed = (PUB / f"{name}.json").read_text(encoding="utf-8")
        assert json.loads(committed) == json.loads(json.dumps(fn()))          # regenerated == committed
        low = committed.lower()
        assert "bmsemi" not in low and "repo_commit" not in low and "projects/" not in low


def test_module_kinds_on_the_stand_in():
    g = IRGraph.from_dict(json.loads((PUB / "invres_net.json").read_text()))
    from collections import Counter
    assert Counter(N.infer_kinds(g).values()) == {"batchnorm": 7, "relu6": 5}


@pytest.mark.parametrize("wl", list(CASES))
def test_deploy_and_replay_on_public_graphs(tmp_path, wl):
    graph, extra = CASES[wl]
    out = tmp_path / wl
    assert D.main(["--workload", wl, "--ir", str(PUB / f"{graph}.json"), "--out", str(out), *extra]) == 0
    r = json.loads((out / "report.json").read_text())
    assert r["ok"] and r["checks"]["x2_4bit_all_steps_exact"] and r["checks"]["bundle_replays_exactly"]
    assert RB.replay(out)["exact"]


def test_a_wrong_x2_result_is_caught_on_the_stand_in():
    g = IRGraph.from_dict(json.loads((PUB / "invres_net.json").read_text()))
    net = N.build_mobilenet(g)
    x = np.random.default_rng(7).integers(0, 128, size=net.input_shape)
    N.calibrate(net, x)
    gold = N.golden(net, x)
    from compiler import x2_pack as xp
    bad = next(s.name for s in net.steps if isinstance(s, N.Layer) and s.name == "b1_dw")

    def mm(L, P):
        y = xp.reference(L.W, P, L.attrs)
        if L.name == bad:
            y = y.copy(); y[0, 0] += 1 << 12
        return y
    _, rows = N.run(net, x, mm, gold)
    assert next(r for r in rows if not r["exact"])["step"] == bad
