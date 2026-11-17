"""Test collection for team checkouts and public clones.

Team checkout (BMsemi_Model next to the code): every test runs, exactly as before; nothing here changes that.
Public clone (no private BMsemi_Model): the test files that are built on BMsemi's simulator or on the team's private
model graphs are not collected, and the header says which; everything else runs.
"""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
HAVE_BMSEMI = importlib.util.find_spec("BMsemi_Model") is not None
HAVE_PRIVATE_IR = (HERE / "fixtures" / "ir" / "mobilenetv2.json").is_file()

NEEDS_BMSEMI = [            # early pipeline stages: they run BMsemi's own X2/PE simulator and model code
    "test_agent.py", "test_benchmark.py", "test_debug_agent.py", "test_fault_injection.py", "test_guard.py",
    "test_inputs.py", "test_matmul.py", "test_model_workloads.py", "test_platform_ops.py", "test_test_planner.py",
    "test_tools.py", "test_workloads.py", "test_ir_partition.py", "test_mapping_agent.py",
]
NEEDS_PRIVATE_IR = [        # assert numbers of the team's real model graphs (tests/fixtures/ir/, not published);
    "test_int8_net.py", "test_deploy_chip.py", "test_run_bundle.py",       # tests/test_public_graphs.py covers the same
]                                                                          # code on public stand-in graphs everywhere

collect_ignore = ([] if HAVE_BMSEMI else NEEDS_BMSEMI) + ([] if HAVE_PRIVATE_IR else NEEDS_PRIVATE_IR)


def pytest_report_header(config):
    if not collect_ignore:
        return "bm: team checkout (BMsemi_Model and the private model graphs present): all test files collected"
    return (f"bm: PUBLIC mode: {len(collect_ignore)} test files not collected "
            f"({'BMsemi_Model absent' if not HAVE_BMSEMI else ''}{'; ' if not HAVE_BMSEMI and not HAVE_PRIVATE_IR else ''}"
            f"{'private model graphs absent' if not HAVE_PRIVATE_IR else ''}): {', '.join(collect_ignore)}")
