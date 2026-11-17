"""Step 12b: the Yosys parser against REAL Yosys output (fixtures), including warnings that hide broken designs."""
from pathlib import Path

import pytest

from verification.flow_logs import check_flow, parse_yosys_log

FX = Path(__file__).parent / "fixtures" / "yosys_real"


def _codes(name):
    r = parse_yosys_log((FX / f"{name}.log").read_text(encoding="utf-8", errors="replace"))
    return [(f["severity"], f["code"]) for f in r["findings"]]


def test_real_syntax_error():
    r = parse_yosys_log((FX / "bad_syntax.log").read_text())
    f = r["findings"][0]
    assert (f["code"], f["file"], f["source_line"]) == ("yosys_syntax_error", "bad_syntax.v", 2)


def test_real_unresolved_module():
    r = parse_yosys_log((FX / "bad_unresolved.log").read_text())
    f = r["findings"][0]
    assert f["code"] == "yosys_unresolved_module" and f["module"] == "missing_mod"


def test_real_port_width_mismatch_is_suspect_not_ok():
    assert _codes("bad_width") == [("suspect", "yosys_port_width_mismatch")]
    assert check_flow(exit_code=0, yosys_log=(FX / "bad_width.log").read_text())["status"] == "suspect"


def test_real_typo_signal_is_suspect_not_ok():
    codes = _codes("undeclared")
    assert ("suspect", "yosys_implicit_wire") in codes and ("suspect", "yosys_undriven_wire") in codes
    assert check_flow(exit_code=0, yosys_log=(FX / "undeclared.log").read_text())["status"] == "suspect"


def test_promoted_warnings_are_not_double_counted():
    r = parse_yosys_log((FX / "undeclared.log").read_text())
    assert len(r["findings"]) == 2 and r["warnings_total"] == 2


def test_real_errors_make_the_flow_failed():
    assert check_flow(exit_code=1, yosys_log=(FX / "bad_syntax.log").read_text())["status"] == "failed"


def test_clean_log_has_no_findings():
    assert parse_yosys_log("Yosys 0.70\nExecuting SYNTH pass.\nEnd of script.\n")["findings"] == []
