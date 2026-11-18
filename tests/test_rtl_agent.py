"""Step 18: RTL checker (Yosys vs NumPy golden) and RTL agent (A2). Needs `pip install yowasp-yosys` (or Yosys on PATH)."""
import json

import pytest

from x2c.agents import llm_client as lc
from x2c.agents.llm_client import Action
from x2c.agents.rtl_agent import REFERENCE_VERILOG, MockRTLLLM, _Session, run_rtl_agent
from x2c.verification.rtl_check import OPS, check_rtl, describe_op, vectors_for

try:
    from x2c.verification.rtl_check import _yosys
    _yosys()
    HAVE = True
except RuntimeError:
    HAVE = False
pytestmark = pytest.mark.skipif(not HAVE, reason="Yosys not installed (pip install yowasp-yosys)")


@pytest.fixture
def out(tmp_path):
    return dict(out_dir=str(tmp_path / "rtl"), log_path=str(tmp_path / "log.jsonl"))


# ------------------------------------------------------------ checker
@pytest.mark.parametrize("op", sorted(OPS))
def test_reference_designs_pass(op):
    r = check_rtl(op, REFERENCE_VERILOG[op])
    assert r["status"] == "pass" and r["vectors"] >= 256 and "NOT silicon" in r["evidence_level"]


def test_relu_vectors_are_exhaustive():
    assert sorted(vectors_for("relu8")["x"].tolist()) == list(range(-128, 128))


MUTANTS = [
    ("relu8", "x[7]", "x[6]"),                                   # wrong sign bit
    ("add8_sat", "? 8'sd127", "? 8'sd126"),                        # wrong upper clip value
    ("add8_sat", "wire signed [8:0] s", "wire signed [7:0] s"),  # sum width too small: wraps
    ("requant_i32_i8", ">>> 10", ">> 10"),                       # logical instead of arithmetic shift
    ("requant_i32_i8", "48'sd512", "48'sd0"),                    # no rounding
    ("requant_i32_i8", "[47:0] p", "[31:0] p"),                  # product truncated to 32 bits
]


@pytest.mark.parametrize("op,old,new", MUTANTS)
def test_buggy_designs_are_caught(op, old, new):
    src = REFERENCE_VERILOG[op]
    assert old in src
    r = check_rtl(op, src.replace(old, new, 1))
    assert r["status"] == "fail" and r["stage"] == "function" and r["mismatches"]


def test_syntax_error_is_reported_not_raised():
    r = check_rtl("relu8", "module relu8(input signed [7:0] x, output signed [7:0] y); assign y = ; endmodule")
    assert r["status"] == "fail" and r["stage"] == "structure"


def test_wrong_ports_missing_top_and_registers_are_rejected():
    assert check_rtl("relu8", "module relu8(input clk, input signed [7:0] x, output signed [7:0] y); assign y = x; endmodule")["stage"] == "structure"
    assert "not found" in check_rtl("relu8", "module other(input [7:0] x, output [7:0] y); assign y = x; endmodule")["errors"][0]
    seq = "module relu8(input signed [7:0] x, output reg signed [7:0] y); always @* y = x; endmodule"
    assert check_rtl("relu8", seq)["status"] == "fail"           # same ports but ALSO fails function (negatives pass through)
    ff = "module relu8(input signed [7:0] x, output reg signed [7:0] y); always @(x) if (x[7]) y <= 0; endmodule"
    assert check_rtl("relu8", ff)["status"] == "fail"            # latch: state element or undefined output


def test_untrusted_text_is_screened():
    for bad in ('`include "/etc/passwd"\nmodule relu8(input [7:0] x, output [7:0] y); endmodule',
                'module relu8(input [7:0] x, output [7:0] y); initial $readmemh("f", m); endmodule', "", None, 5, "x" * 30000):
        r = check_rtl("relu8", bad)
        assert r["status"] == "fail" and r["stage"] == "screen"
    assert check_rtl("nope", "module a; endmodule")["stage"] == "screen"


# ------------------------------------------------------------ agent
def test_mock_agent_verifies_and_saves_all_ops(out, tmp_path):
    for op in OPS:
        r = run_rtl_agent(MockRTLLLM(op), op, **out)
        assert r["verified"] and r["attempts"] == 1 and r["stopped"] == "finished"
        assert (tmp_path / "rtl" / f"{op}.v").read_text() == REFERENCE_VERILOG[op]


class Scripted:
    def __init__(self, submissions):
        self.subs, self.seen = list(submissions), []

    def start(self, *a, **k):
        return Action("tool_calls", calls=[("get_spec", {})])

    def send_tool_results(self, results, force_final=False):
        self.seen.append(results[0])
        if self.subs:
            return Action("tool_calls", calls=[("submit_verilog", {"verilog": self.subs.pop(0)})])
        return Action("final", text="[scripted] done")


def test_agent_retries_after_feedback_and_then_passes(out):
    bad = REFERENCE_VERILOG["relu8"].replace("x[7]", "x[6]")
    llm = Scripted([bad, REFERENCE_VERILOG["relu8"]])
    r = run_rtl_agent(llm, "relu8", **out)
    assert r["verified"] and r["attempts"] == 2
    assert llm.seen[1][1]["status"] == "fail" and llm.seen[1][1]["mismatches"]       # the LLM saw the failure details


def test_agent_never_saves_or_claims_a_failed_design(out, tmp_path):
    bad = REFERENCE_VERILOG["relu8"].replace("x[7]", "x[6]")
    r = run_rtl_agent(Scripted([bad] * 6), "relu8", **out)
    assert not r["verified"] and r["attempts"] == 4 and r["verilog_path"] is None
    assert not (tmp_path / "rtl" / "relu8.v").exists() and r["last_stage"] == "function"


def test_llm_saying_it_passed_changes_nothing(out):
    class Liar(Scripted):
        def send_tool_results(self, results, force_final=False):
            return Action("final", text="Verified on silicon, all tests pass!")
    r = run_rtl_agent(Liar([]), "relu8", **out)
    assert not r["verified"] and r["verilog_path"] is None


def test_llm_unavailable_and_error_handling(out):
    class Err(Exception):
        def __init__(self, code):
            super().__init__("x")
            self.status_code = code

    def failing(exc):
        class L:
            def start(self, *a, **k):
                raise exc
        return L()
    assert run_rtl_agent(failing(lc.LLMUnavailable("down")), "relu8", **out)["stopped"] == "llm_unavailable"
    assert run_rtl_agent(failing(Err(429)), "relu8", **out)["stopped"] == "llm_error"
    with pytest.raises(Err):
        run_rtl_agent(failing(Err(401)), "relu8", **out)
    with pytest.raises(ZeroDivisionError):
        run_rtl_agent(failing(ZeroDivisionError()), "relu8", **out)


def test_session_rejects_unknown_tools_arguments_and_garbage():
    s = _Session("relu8")
    assert "unknown tool" in s.call("shell", {})["error"]
    assert "unknown arguments" in s.call("get_spec", {"path": "/etc/passwd"})["error"]
    assert "error" in s.call("submit_verilog", {})                                   # missing argument
    assert s.call("submit_verilog", {"verilog": 42})["status"] == "fail"
    assert describe_op("relu8")["top_module"] == "relu8"


def test_audit_log_written(out):
    run_rtl_agent(MockRTLLLM("relu8"), "relu8", **out)
    rows = [json.loads(l) for l in open(out["log_path"], encoding="utf-8")]
    assert [r["tool"] for r in rows] == ["get_spec", "submit_verilog"] and rows[1]["status"] == "pass"


def test_unknown_op_raises():
    with pytest.raises(ValueError):
        run_rtl_agent(MockRTLLLM("relu8"), "nope")
