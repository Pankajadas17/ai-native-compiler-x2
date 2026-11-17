"""run_pipeline.py: the stage order, one summary report, partial runs labelled as partial, failures counted."""
import json

import pytest

import run_pipeline as P


def test_stage_order_is_the_pipeline_order():
    assert [k for k, *_ in P.STAGES] == ["environment", "operators", "formal", "engine", "x2-model", "engine-sim",
                                          "deploy", "replay", "status", "tests"]


def _fake(monkeypatch, tmp_path, results):
    monkeypatch.setattr(P, "OUT", tmp_path)
    monkeypatch.setattr(P, "ROOT", tmp_path.parent)
    monkeypatch.setattr(P.platform, "system", lambda: "Linux")
    stages = [(k, t, w, (lambda ok: (lambda ctx: (ok, {"k": 1}, "fine" if ok else "broken")))(results.get(k, True)))
              for k, t, w, _ in P.STAGES]
    monkeypatch.setattr(P, "STAGES", stages)


def test_all_pass_gives_one_report(monkeypatch, tmp_path):
    _fake(monkeypatch, tmp_path, {})
    assert P.main([]) == 0
    ev = json.loads((tmp_path / "report.json").read_text())
    assert ev["verdict"] == "PASS" and ev["stages_passed"] == "9/9" and ev["not_proven"]
    md = (tmp_path / "REPORT.md").read_text()
    assert md.startswith("# Pipeline report: PASS (9/9 stages)") and "What this does NOT prove" in md


def test_a_failing_stage_fails_the_pipeline_but_the_rest_still_runs(monkeypatch, tmp_path):
    _fake(monkeypatch, tmp_path, {"formal": False})
    assert P.main([]) == 1
    ev = json.loads((tmp_path / "report.json").read_text())
    assert ev["verdict"] == "FAIL" and ev["stages_passed"] == "8/9"
    assert [s["status"] for s in ev["stages"]][:9] == ["ran"] * 9


def test_partial_runs_are_labelled(monkeypatch, tmp_path):
    _fake(monkeypatch, tmp_path, {})
    assert P.main(["--only", "deploy,replay"]) == 0
    assert json.loads((tmp_path / "report.json").read_text())["verdict"].startswith("PARTIAL")
    with pytest.raises(SystemExit):
        P.main(["--only", "nonsense"])


def test_windows_is_sent_to_ubuntu(monkeypatch, capsys):
    monkeypatch.setattr(P.platform, "system", lambda: "Windows")
    assert P.main([]) == 2 and "Ubuntu terminal" in capsys.readouterr().out


def test_a_failed_environment_stops_the_run(monkeypatch, tmp_path):
    _fake(monkeypatch, tmp_path, {"environment": False})
    assert P.main([]) == 1
    ev = json.loads((tmp_path / "report.json").read_text())
    assert ev["verdict"] == "FAIL" and ev["stages"][0]["status"] == "ran"
    assert all(s["status"].startswith("not run") for s in ev["stages"][1:9])


def test_public_deploy_uses_only_public_graphs():
    for wl, args in P.PUBLIC_DEPLOY.items():
        assert args[1].startswith("tests/fixtures/ir_public/") and (P.ROOT / args[1]).is_file()
