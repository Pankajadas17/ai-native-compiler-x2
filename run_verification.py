"""Orchestrator: generate cases -> golden -> X2 -> compare -> report.
x2_fn is injectable so tests can plug in a deliberately broken X2 (fault injection)."""
from emulators.x2_adapter import x2_matmul_report
from verification.golden_reference import golden_matmul
from verification.comparator import compare
from verification.test_generator import make_cases
from verification.report_generator import write_report


def run_one(name, A, B, x2_fn=x2_matmul_report):
    exp = golden_matmul(A, B)
    base = {"test": name, "shape": list(exp.shape), "expected": exp.tolist()}
    try:
        x2 = x2_fn(A, B)
    except Exception as e:  # interface problems become structured failures, not crashes
        return {**base, "actual": None, "passed": False, "reason": "exception",
                "error": f"{type(e).__name__}: {e}", "max_abs_err": None}
    return {**base, "actual": x2["output"].tolist(),
            "total_cycles": x2["total_cycles"],
            "energy_configured_subtotal_J": x2["energy_configured_subtotal_J"],
            "energy_status": x2["energy_status"], "simulator_status": x2["final_status"],
            "backend": x2["backend"], "physical_hardware": x2["physical_hardware"],
            "source_commit": x2["source_commit"], **compare(exp, x2["output"])}


def run_all(x2_fn=x2_matmul_report, seed=0, n_random=20, report_path="reports/report.json"):
    results = [run_one(n, A, B, x2_fn) for n, A, B in make_cases(seed=seed, n_random=n_random)]
    return write_report(results, path=report_path)


if __name__ == "__main__":
    s = run_all()
    print(f"{s['passed']}/{s['total']} passed -> reports/report.json")
