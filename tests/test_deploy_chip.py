"""x2c deploy: one command from a model's IR to a verified deployment bundle for the chip."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import x2c.commands.deploy as D

IR = Path(__file__).resolve().parent / "fixtures" / "ir"


def deploy(tmp_path, workload, *extra):
    out = tmp_path / workload
    rc = D.main(["--workload", workload, "--ir", str(IR / f"{workload}.json"), "--out", str(out), *extra])
    return rc, out, json.loads((out / "report.json").read_text())


def test_mobilenetv2_bundle(tmp_path):
    rc, out, r = deploy(tmp_path, "mobilenetv2")
    assert rc == 0 and r["ok"] and r["checks"] == {"x2_4bit_all_steps_exact": True, "bundle_replays_exactly": True}
    assert r["x2_layers"] == 17 and r["steps"] == 21 and r["module_kinds_inferred"] == {"batchnorm": 16, "relu6": 11}
    for f in ("plan.json", "host_program.json", "x2_program.npz", "expected.npz", "report.json"):
        assert (out / f).is_file(), f
    host = json.loads((out / "host_program.json").read_text())
    assert [s["step"] for s in host["steps"]].count("x2_layer") == 17 and host["output"] == "classifier"
    prog = np.load(out / "x2_program.npz")
    levels = [k for k in prog.files if k.endswith("__levels")]
    assert levels and all(prog[k].shape == (32, 32) and prog[k].max() <= 15 for k in levels)       # 4-bit cell levels
    exp = np.load(out / "expected.npz")
    assert "input" in exp.files and exp["classifier"].shape == (10,)
    assert r["a1"]["schedule_source"] and r["a1"]["latency_us"] > 0
    assert "NOT silicon" in r["evidence_level"]


@pytest.mark.parametrize("workload", ["deepgate", "mnist"])
def test_synthetic_chains(tmp_path, workload):
    rc, out, r = deploy(tmp_path, workload, "--synthetic")
    assert rc == 0 and r["ok"] and "SYNTHETIC" in r["model"]


def test_engine_bitstream_is_bundled_when_built(tmp_path):
    rc, out, r = deploy(tmp_path, "mnist", "--synthetic")
    built = (D.ROOT / "chip_build" / D.ENGINE / f"{D.ENGINE}.bit").is_file()
    assert (r["engine_bitstream"]["file_in_bundle"] is not None) == built
    if built:
        assert (out / "engine" / "design.pcf").is_file()                     # the pin map travels with the bitstream
    else:
        assert "x2c build --example x2_engine2_4bit" in r["engine_bitstream"]["build"]


@pytest.mark.skipif(not (shutil.which("iverilog") and shutil.which("vvp")), reason="Icarus not on PATH (Ubuntu terminal)")
def test_mnist_on_the_one_bit_engine_rtl(tmp_path):
    rc, out, r = deploy(tmp_path, "mnist", "--synthetic", "--sim")
    s = r["simulation_1bit"]
    assert rc == 0 and r["checks"]["x2_1bit_engine_rtl_all_steps_exact"] and s["layers_simulated"] == 1
    assert abs(s["total_cycles_predicted"] / s["total_cycles_measured"] - 1) < 0.05


def test_real_bmsemi_workloads(tmp_path):
    pytest.importorskip("BMsemi_Model.bm_models.models.deepgate")
    for wl, key in (("deepgate", "golden_equals_repo_deepgate_chain"), ("mnist", "golden_equals_repo_edge_map")):
        rc = D.main(["--workload", wl, "--out", str(tmp_path / wl)])
        r = json.loads((tmp_path / wl / "report.json").read_text())
        assert rc == 0 and r["checks"][key] and r["checks"]["x2_4bit_all_steps_exact"], (wl, r["checks"])


def test_real_weights_replace_the_seeded_ones(tmp_path):
    import x2c.compiler.int8_net as N
    g = D.load_ir("mobilenetv2", str(IR / "mobilenetv2.json"))
    net = N.build_mobilenet(g)
    first = next(s for s in net.steps if isinstance(s, N.Layer))
    w = np.random.default_rng(9).integers(-128, 128, size=first.W.shape)
    np.savez(tmp_path / "w.npz", **{first.name: w, f"{first.name}__bias": np.arange(first.W.shape[0])})
    rc, out, r = deploy(tmp_path, "mobilenetv2", "--weights", str(tmp_path / "w.npz"))
    assert rc == 0 and r["ok"] and r["weights"] == {"file": str(tmp_path / "w.npz"), "real_layers": 1, "layers": 17}
    assert any("real weights" in n for n in r["notes"])
    host = json.loads((out / "host_program.json").read_text())
    assert host["steps"][0]["epilogue"]["bias"] == list(range(first.W.shape[0]))


def test_bad_weights_are_refused(tmp_path):
    np.savez(tmp_path / "bad.npz", no_such_layer=np.zeros(3))
    with pytest.raises(SystemExit, match="unknown names"):
        D.main(["--workload", "mobilenetv2", "--ir", str(IR / "mobilenetv2.json"), "--weights", str(tmp_path / "bad.npz"),
                "--out", str(tmp_path / "o")])


def test_fabric_estimate_in_the_report(tmp_path):
    rc, out, r = deploy(tmp_path, "mobilenetv2")
    f = r["fabric"]
    assert "host_steps_on_fabric" in f or "error" in f
    if "host_steps_on_fabric" in f:
        assert f["host_only_steps"] == ["pool"]
