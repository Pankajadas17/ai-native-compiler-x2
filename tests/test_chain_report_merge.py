"""run_fab_chain.write_report: a run replaces only the operators it ran; the chip report states both clocks."""
import json
from types import SimpleNamespace

import run_fab_chain as rfc
from verification import chip_chain as cc


def test_merge_and_clock_labels(tmp_path, monkeypatch):
    path = tmp_path / "chip_chain_report.json"
    monkeypatch.setattr(cc, "REPORT", path)
    cfg = SimpleNamespace(target="chip", runner="native", distro="Ubuntu")
    rfc.write_report([{"op": o, "verdict": "PASS"} for o in ("relu8", "add8_sat", "requant_i32_i8")], cfg, {})
    rfc.write_report([{"op": "relu8", "verdict": "FAIL"}], cfg, {})
    r = json.loads(path.read_text())
    assert [x["op"] for x in r["results"]] == ["add8_sat", "relu8", "requant_i32_i8"]
    assert {x["op"]: x["verdict"] for x in r["results"]}["relu8"] == "FAIL" and r["ops_this_run"] == ["relu8"]
    assert r["clock_target_mhz"] == 14.0 and r["nextpnr_target_mhz"] == 12.0
    assert r["generated"][-5] in "+-" and all("run_at" in x for x in r["results"])


def test_other_target_starts_fresh(tmp_path, monkeypatch):
    path = tmp_path / "rep.json"
    monkeypatch.setattr(cc, "REPORT", path)
    path.write_text(json.dumps({"target": "demo", "results": [{"op": "relu8", "verdict": "PASS"}]}))
    rfc.write_report([{"op": "add8_sat", "verdict": "PASS"}], SimpleNamespace(target="chip", runner="native", distro="U"), {})
    assert [x["op"] for x in json.loads(path.read_text())["results"]] == ["add8_sat"]


def test_demo_report_keeps_no_target_when_none_was_given(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rfc.write_report([{"op": "relu8", "verdict": "PASS"}], SimpleNamespace(target="demo", runner="wsl", distro="U", freq_mhz=None), {})
    r = json.loads((tmp_path / rfc.REPORT).read_text())
    assert r["clock_target_mhz"] is None and r["nextpnr_target_mhz"] == 12.0
