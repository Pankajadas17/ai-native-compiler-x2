"""Proves the referee can FAIL, and that failures are classified correctly."""
import pytest
from x2c.verification import tools
from tests import faults

CASES = [("sign_flip", faults.sign_flip, "sign_inverted"),
         ("off_by_one", faults.off_by_one, "constant_offset"),
         ("doubled", faults.doubled, "scaled_output"),
         ("transposed_output", faults.transposed_output, {"transposed_output", "shape_mismatch"}),
         ("transposed_weights", faults.transposed_weights, {"interface_error", "numerical_mismatch"})]

@pytest.mark.parametrize("label,fn,expected", CASES)
def test_fault_is_caught_and_classified(label, fn, expected, tmp_path):
    rp = str(tmp_path / "r.json")
    s = tools.run_suite(report_path=rp, x2_fn=fn)
    assert s["failed"], f"{label}: harness failed to notice a broken X2!"
    cats = set(tools.summarize_report(rp)["failure_categories"])
    expected = {expected} if isinstance(expected, str) else expected
    assert cats & expected, f"{label}: got categories {cats}"

def test_fixed_2x2_diagnosis_exact():
    r = tools.run_case("fixed_2x2", x2_fn=faults.sign_flip)
    assert r["passed"] is False and r["diagnosis"]["category"] == "sign_inverted"
    r = tools.run_case("fixed_2x2", x2_fn=faults.off_by_one)
    assert r["diagnosis"]["category"] == "constant_offset"
