"""chip/fabric_costs.py: block costs are read from real build reports; a missing build is reported, never guessed."""
import json

from chip import fabric_costs as fc

PNR = "Info: Max frequency for clock 'clk': {f} MHz (PASS at 12.00 MHz)\n"


def build(root, design, used, fmax=None, status="ok"):
    d = root / "chip_build" / design
    d.mkdir(parents=True)
    avail = {"FABULOUS_LC": 2160, "IOBUF": 36, "MACC_8x8_20": 18, "RAM_32x4_2R_1W": 36, "RERAM_CIM_32x32": 2, "GBUF": 4}
    (d / "flow_report.json").write_text(json.dumps({"status": status, "utilisation": {
        k: {"used": used.get(k, 0), "available": v} for k, v in avail.items()}}))
    if fmax:
        (d / "pnr.log").write_text(PNR.format(f=fmax))


def test_measure_and_estimate(tmp_path):
    build(tmp_path, "x2_engine2_4bit", {"FABULOUS_LC": 1792, "IOBUF": 35, "RERAM_CIM_32x32": 2, "GBUF": 1}, 14.45)
    build(tmp_path, "chain_relu8", {"FABULOUS_LC": 15, "IOBUF": 17, "GBUF": 1}, 151.5)
    build(tmp_path, "chain_requant_i32_i8", {"FABULOUS_LC": 134, "IOBUF": 26, "MACC_8x8_20": 4, "GBUF": 1})
    m = fc.measure(tmp_path)
    assert m["blocks"]["x2_engine"]["used"]["logic_cells"] == 1792 and m["blocks"]["x2_engine"]["fmax_mhz"] == 14.45
    assert m["blocks"]["requant"]["timing"].startswith("combinational")
    assert m["blocks"]["add"]["status"] == "not built" and "run_fab_chain.py --op add8_sat" in m["blocks"]["add"]["how"]
    e = fc.estimate(["x2_engine", "relu", "requant"], m)
    assert e["used"]["logic_cells"] == 1941 and e["used"]["macc"] == 4
    assert e["used"]["pins"] == 35 and e["used"]["gbuf"] == 1            # pins = engine's, clock buffer shared
    assert e["fits"] and e["fmax_upper_bound_mhz"] == 14.45
    assert not fc.estimate(["x2_engine", "add"], m)["fits"]              # 'add' not built: cannot claim it fits


def test_over_budget_is_reported(tmp_path):
    build(tmp_path, "x2_engine2_4bit", {"FABULOUS_LC": 2100, "IOBUF": 35, "RERAM_CIM_32x32": 2}, 14.0)
    build(tmp_path, "chain_requant_i32_i8", {"FABULOUS_LC": 134, "MACC_8x8_20": 4})
    e = fc.estimate(["x2_engine", "requant"], fc.measure(tmp_path))
    assert not e["fits"] and "logic_cells" in e["over"]


def test_blocks_a_network_needs():
    import numpy as np
    from compiler import int8_net as N
    rng = np.random.default_rng(0)
    net = N.build_chain("t", [{"name": "a", "W": rng.integers(-8, 8, size=(4, 6)), "requant": lambda a: np.clip(a, -128, 127)}], (6,), True)
    need, host_only = fc.blocks_for(net)
    assert need == ["x2_engine", "requant"] and host_only == []
