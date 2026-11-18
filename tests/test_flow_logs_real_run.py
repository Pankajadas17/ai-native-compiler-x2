"""Step 20: the flow checker against a REAL FABulous run (fixtures/real_flow_run), plus clearly-labelled mutations of it."""
from pathlib import Path

from x2c.verification.bitstream_audit import parse_fasm
from x2c.verification.flow_logs import check_flow, parse_nextpnr_log

FX = Path(__file__).parent / "fixtures" / "real_flow_run"
LOG = (FX / "nextpnr_log.txt").read_text()
FASM = (FX / "design.fasm").read_text()
FLOW = "INFO | Compile flow completed successfully."


def test_real_nextpnr_log_is_understood():
    p = parse_nextpnr_log(LOG)
    assert p["finished_normally"] and p["findings"] == []
    assert p["utilisation"]["FABULOUS_LC"] == {"used": 32, "total": 672}
    assert p["utilisation"]["Global_Clock"] == {"used": 1, "total": 1}
    assert len(p["utilisation"]) == 11


def test_the_last_max_frequency_line_is_the_result():
    # the real log prints 16.53 MHz after placement and 30.67 MHz after routing
    f = parse_nextpnr_log(LOG)["fmax"]["clk"]
    assert f == {"fmax_mhz": 30.67, "target_mhz": 12.0, "ok": True}


def test_real_fasm_parses_with_nothing_left_over():
    feats, unparsed = parse_fasm(FASM)
    assert len(feats) == 918 and unparsed == []             # '# routing for net' comment lines are ignored, INIT ranges parse


def test_real_run_without_a_bitstream_is_inconclusive_never_ok():
    r = check_flow(exit_code=0, nextpnr_log=LOG, flow_log=FLOW, fasm_text=FASM)
    assert r["status"] == "inconclusive" and "bitstream" in r["not_checked"]
    assert r["facts"]["tool_claimed_success"] is True and r["facts"]["fasm_features"] == 918
    assert r["facts"]["utilisation"]["FABULOUS_LC"]["used"] == 32


# ---- MUTATED copies of the real log: these prove the checker catches the failure, NOT what nextpnr prints when it fails ----
def test_mutated_capacity_overflow_is_failed():
    bad = LOG.replace("FABULOUS_LC:      32/    672     4%", "FABULOUS_LC:     700/    672   104%")
    assert bad != LOG
    r = check_flow(exit_code=0, nextpnr_log=bad, flow_log=FLOW, fasm_text=FASM)
    assert r["status"] == "failed" and any(f["code"] == "capacity_exceeded" for f in r["findings"])


def test_mutated_zero_logic_is_suspect_even_though_the_tool_said_success():
    bad = LOG.replace("FABULOUS_LC:      32/    672     4%", "FABULOUS_LC:       0/    672     0%")
    r = check_flow(exit_code=0, nextpnr_log=bad, flow_log=FLOW, fasm_text=FASM)
    assert r["status"] == "suspect" and any(f["code"] == "zero_logic_cells" for f in r["findings"])


def test_mutated_timing_fail_is_failed():
    bad = LOG.replace("30.67 MHz (PASS at 12.00 MHz)", "9.50 MHz (FAIL at 12.00 MHz)")
    r = check_flow(exit_code=0, nextpnr_log=bad, flow_log=FLOW, fasm_text=FASM)
    assert r["status"] == "failed" and any(f["code"] == "timing_violation" for f in r["findings"])


def test_truncated_log_is_not_trusted():
    r = check_flow(exit_code=0, nextpnr_log=LOG.split("Info: Routing..")[0], flow_log=FLOW, fasm_text=FASM)
    assert r["status"] in ("suspect", "failed") and any(f["code"] == "nextpnr_log_incomplete" for f in r["findings"])


def test_empty_fasm_is_suspect():
    r = check_flow(exit_code=0, nextpnr_log=LOG, flow_log=FLOW, fasm_text="# routing for net 'x'\n\n")
    assert r["status"] == "suspect" and any(f["code"] == "empty_fasm" for f in r["findings"])
