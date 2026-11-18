"""Step 23: fabric-simulation testbench generator, log judge, and flow-check integration.
Tests marked slow compile and run the generated testbench with Icarus (`iverilog`, as in your Ubuntu toolchain) or Verilator if either is installed,
against a STAND-IN for the fabric (not the real FABulous fabric): they prove the testbench logic, not the fabric."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from x2c.agents.rtl_agent import REFERENCE_VERILOG
from x2c.commands.demo_flow_check import main as check_main
from x2c.verification.fab_sim import make_fab_testbench, parse_sim_log, sim_vectors, tb_vector_count
from x2c.verification.fab_wrap import make_fab_wrapper, wrapper_layout

REAL = Path(__file__).parent / "fixtures" / "real_flow_run"
PASS_LINE = "FAB_SIM RESULT: PASS vectors=500 numpy_mismatches=0 rtl_mismatches=0"


def wrapped(op):
    return make_fab_wrapper(op, REFERENCE_VERILOG[op])


# ---------------- generator ----------------
def test_expected_values_are_independent_python_arithmetic():
    lay = wrapper_layout("add8_sat")
    for word, exp in sim_vectors("add8_sat"):
        a = (word & 0xFF) - 256 * ((word >> 7) & 1)
        b = ((word >> 8) & 0xFF) - 256 * ((word >> 15) & 1)
        assert exp == max(-128, min(127, a + b)) & 0xFF and word >> 16 == 0
    assert lay["out_lsb"] == 16


def test_relu_vectors_cover_every_input_once():
    v = sim_vectors("relu8")
    assert sorted(w for w, _ in v) == list(range(256)) and all(e == (w if w < 128 else 0) for w, e in v)


def test_testbench_text():
    tb = make_fab_testbench("add8_sat")
    assert "module add8_sat_fab_tb" in tb and "eFPGA_top top_i" in tb and "add8_sat_fab dut_i" in tb
    assert tb_vector_count(tb) == 500 and "NVEC = 500" in tb and "I_top[23:16]" in tb
    assert "FAB_SIM RESULT: PASS" in tb and "$readmemh" in tb and "$fatal" not in tb
    assert "I_top[15:8]" in make_fab_testbench("relu8")


def test_testbench_table_is_exactly_the_numpy_table():
    tb = make_fab_testbench("add8_sat")
    rows = re.findall(r"vec_in\[(\d+)\] = 28'h([0-9A-F]{7}); vec_exp\[\d+\] = 8'h([0-9A-F]{2});", tb)
    assert [(int(w, 16), int(e, 16)) for _, w, e in rows] == sim_vectors("add8_sat")


# ---------------- log judge ----------------
def test_pass_requires_one_result_line_and_the_expected_count():
    assert parse_sim_log(PASS_LINE, 500)["status"] == "pass"
    assert parse_sim_log(PASS_LINE, 300)["status"] == "inconclusive"                      # a truncated vector table is not a pass
    assert parse_sim_log(PASS_LINE + "\n" + PASS_LINE, 500)["status"] == "inconclusive"
    assert parse_sim_log("FAB_SIM RESULT: PASS vectors=0 numpy_mismatches=0 rtl_mismatches=0")["status"] == "inconclusive"


def test_fail_and_missing_result():
    r = parse_sim_log("FAB_SIM MISMATCH vector 3: io_in=0x1 expected=0x2 fabric=0x0\nFAB_SIM RESULT: FAIL vectors=500 numpy_mismatches=4 rtl_mismatches=4", 500)
    assert r["status"] == "fail" and r["mismatch_lines"] and "4 numpy" in r["reason"]
    # a PASS word with mismatches is still a fail
    assert parse_sim_log("FAB_SIM RESULT: PASS vectors=500 numpy_mismatches=1 rtl_mismatches=0", 500)["status"] == "fail"
    for text in ("", None, "INFO | Compile flow completed successfully.", "x.v:3: error: Unknown module type: eFPGA_top"):
        r = parse_sim_log(text, 500)
        assert r["status"] == "inconclusive" and "no FAB_SIM RESULT" in r["reason"]
    assert "Unknown module" in parse_sim_log("x.v:3: error: Unknown module type: eFPGA_top")["tool_errors"][0]


# ---------------- flow check integration ----------------
def make_job(tmp_path, sim_text):
    j = tmp_path / "job"
    j.mkdir()
    (j / "add8_sat_fab.v").write_text("module add8_sat_fab; endmodule")
    (j / "add8_sat_fab_tb.v").write_text(make_fab_testbench("add8_sat"))
    shutil.copy(REAL / "nextpnr_log.txt", j / "add8_sat_fab_npnr_log.txt")
    shutil.copy(REAL / "design.fasm", j / "add8_sat_fab.fasm")
    (j / "flow_console.log").write_text("INFO | Compile flow completed successfully.\n")
    (j / "exit_code.txt").write_text("0\n")
    if sim_text is not None:
        (j / "sim_console.log").write_text(sim_text)
    return j


def test_job_without_a_simulation_lists_it_as_not_checked(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    check_main(["--job", str(make_job(tmp_path, None))])
    assert "fabric_simulation" in capsys.readouterr().out


def test_passing_simulation_is_reported(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    check_main(["--job", str(make_job(tmp_path, "noise\n" + PASS_LINE + "\n"))])
    assert "FABRIC SIMULATION: PASS" in capsys.readouterr().out


def test_failing_simulation_makes_the_flow_failed(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    check_main(["--job", str(make_job(tmp_path, "FAB_SIM RESULT: FAIL vectors=500 numpy_mismatches=500 rtl_mismatches=500\n"))])
    out = capsys.readouterr().out
    assert "FLOW STATUS: FAILED" in out and "FABRIC SIMULATION: FAIL" in out


def test_simulation_that_compared_nothing_is_suspect(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    check_main(["--job", str(make_job(tmp_path, "INFO | Compile flow completed successfully.\n"))])
    out = capsys.readouterr().out
    assert "FLOW STATUS: SUSPECT" in out and "FABRIC SIMULATION: INCONCLUSIVE" in out


# ---------------- run the generated testbench against STAND-INS for the fabric ----------------
def _sim_tool():
    if shutil.which("iverilog") and shutil.which("vvp"):
        return "iverilog"
    if shutil.which("verilator") or shutil.which("verilator-cli"):
        return "verilator"
    return None


STUB_TEMPLATE = """module eFPGA_top (output wire [27:0] I_top, output wire [27:0] T_top, input wire [27:0] O_top,
    output wire [55:0] A_config_C, output wire [55:0] B_config_C, input wire CLK, input wire resetn,
    input wire SelfWriteStrobe, input wire [31:0] SelfWriteData, input wire Rx, output wire ComActive,
    output wire ReceiveLED, input wire s_clk, input wire s_data);
{body}
    assign A_config_C = 56'b0; assign B_config_C = 56'b0; assign ComActive = 1'b0; assign ReceiveLED = 1'b0;
endmodule
"""
IDEAL = "    wire [27:0] oeb;\n    {top} u (.clk(CLK), .io_in(O_top), .io_out(I_top), .io_oeb(oeb));\n    assign T_top = ~oeb;"
DEAD = "    assign I_top = 28'b0; assign T_top = 28'b0;"          # what the empty bitstream of the broken run would give


def simulate(op, tmp_path, stub_body, rtl_text):
    tool = _sim_tool()
    top = f"{op}_fab"
    (tmp_path / "rtl.v").write_text(rtl_text)
    (tmp_path / "stub.v").write_text(STUB_TEMPLATE.format(body=stub_body.format(top=top)))
    (tmp_path / "tb.v").write_text(make_fab_testbench(op))
    (tmp_path / "bits.hex").write_text("00\n" * 16384)
    if tool == "iverilog":
        subprocess.run(["iverilog", "-g2012", "-s", f"{top}_tb", "-o", "x.vvp", "tb.v", "stub.v", "rtl.v"], cwd=tmp_path, check=True, capture_output=True)
        r = subprocess.run(["vvp", "x.vvp", "+bitstream_hex=bits.hex"], cwd=tmp_path, capture_output=True, text=True, timeout=600)
    else:
        v = shutil.which("verilator") or shutil.which("verilator-cli")
        subprocess.run([v, "--binary", "--timing", "-Wno-fatal", "--top-module", f"{top}_tb", "--Mdir", "obj", "tb.v", "stub.v", "rtl.v"],
                       cwd=tmp_path, check=True, capture_output=True, timeout=600)
        r = subprocess.run([f"obj/V{top}_tb", "+bitstream_hex=bits.hex"], cwd=tmp_path, capture_output=True, text=True, timeout=600)
    return r.stdout + r.stderr


needs_sim = pytest.mark.skipif(_sim_tool() is None, reason="needs iverilog (OSS CAD Suite) or verilator")


@pytest.mark.slow
@needs_sim
@pytest.mark.parametrize("op", ["relu8", "add8_sat", "requant_i32_i8"])
def test_testbench_passes_when_the_fabric_computes_the_operator(op, tmp_path):
    log = simulate(op, tmp_path, IDEAL, wrapped(op))
    r = parse_sim_log(log, tb_vector_count(make_fab_testbench(op)))
    assert r["status"] == "pass", log[-600:]


@pytest.mark.slow
@needs_sim
def test_testbench_fails_when_the_fabric_outputs_nothing(tmp_path):
    log = simulate("add8_sat", tmp_path, DEAD, wrapped("add8_sat"))
    r = parse_sim_log(log, 500)
    assert r["status"] == "fail" and r["numpy_mismatches"] > 400 and r["mismatch_lines"]


@pytest.mark.slow
@needs_sim
def test_numpy_check_catches_a_wrong_operator_even_when_the_rtl_reference_agrees_with_it(tmp_path):
    bad = wrapped("add8_sat").replace("? 8'sd127", "? 8'sd126")
    assert bad != wrapped("add8_sat")
    log = simulate("add8_sat", tmp_path, IDEAL, bad)
    r = parse_sim_log(log, 500)
    assert r["status"] == "fail" and r["rtl_mismatches"] == 0 and r["numpy_mismatches"] > 0


# ---------------- prep writes the simulation files ----------------
def test_prep_also_writes_testbench_and_sim_script(tmp_path, monkeypatch):
    from x2c.commands.demo_prep import SIM_SCRIPT, main as prep_main
    if shutil.which("bash") is None:
        pytest.skip("bash not available")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "hardware" / "operators").mkdir(parents=True)
    (tmp_path / "hardware" / "operators" / "add8_sat.v").write_text(REFERENCE_VERILOG["add8_sat"])
    assert prep_main(["--op", "add8_sat"]) == 0
    job = tmp_path / "reports" / "fab_jobs" / "add8_sat"
    assert (job / "add8_sat_fab_tb.v").read_text() == make_fab_testbench("add8_sat")
    assert (job / "run_fab_sim.sh").read_text() == SIM_SCRIPT and b"\r" not in (job / "run_fab_sim.sh").read_bytes()
    # via stdin: on Windows `bash` is WSL's bash and cannot open a Windows path
    assert subprocess.run(["bash", "-n"], input=SIM_SCRIPT.encode(), capture_output=True).returncode == 0
    for part in ("run_simulation vcd", "sim_console.log", "sim_exit_code.txt", "no bitstream"):
        assert part in SIM_SCRIPT
