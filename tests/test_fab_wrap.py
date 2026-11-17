"""Step 21: wrap an A2-verified operator for the FABulous demo fabric, and read a job folder back through the flow checker."""
import shutil
from pathlib import Path

import pytest

from agents.rtl_agent import REFERENCE_VERILOG
from run_fab_prep import FLOW_SCRIPT, main as prep_main
from run_flow_check import main as check_main
from verification.fab_wrap import check_fab_wrapper, make_fab_wrapper, top_name, wrapper_layout
import verification.rtl_check as rc

try:
    rc._yosys()
    HAVE = True
except RuntimeError:
    HAVE = False
pytestmark = pytest.mark.skipif(not HAVE, reason="Yosys not installed (pip install yowasp-yosys)")
REAL = Path(__file__).parent / "fixtures" / "real_flow_run"


def wrapped(op):
    return make_fab_wrapper(op, REFERENCE_VERILOG[op])


def test_layouts():
    assert wrapper_layout("relu8")["in_bits"] == 8 and wrapper_layout("relu8")["out_lsb"] == 8
    assert wrapper_layout("add8_sat")["inputs"] == {"a": (0, 8), "b": (8, 8)} and wrapper_layout("add8_sat")["out_lsb"] == 16


WIDE2 = {"top": "wide2", "inputs": {"a": 12, "b": 12}, "output": ("y", 8), "text": "two inputs, 32 pins in all", "golden": lambda v: v["a"]}     # fits no layout


def test_requant_is_loaded_in_two_steps_and_two_wide_inputs_still_do_not_fit(monkeypatch):
    lay = wrapper_layout("requant_i32_i8")
    assert lay["mode"] == "two_step" and lay["half"] == 16 and lay["sel_pin"] == 16 and lay["in_bits"] == 17 and lay["out_lsb"] == 17 and lay["in_width"] == 32
    assert wrapper_layout("relu8")["mode"] == "direct" and wrapper_layout("add8_sat")["mode"] == "direct"
    from verification.fab_wrap import split_halves
    assert split_halves(-1, lay) == (0xFFFF, 0x1FFFF) and split_halves(0x12345678, lay) == (0x5678, 0x11234) and split_halves(-2 ** 31, lay) == (0, 0x18000)
    monkeypatch.setitem(rc.OPS, "wide2", WIDE2)
    with pytest.raises(ValueError, match="28 I/O pins"):
        wrapper_layout("wide2")


def test_two_step_wrapper_passes_and_its_mistakes_are_caught():
    """The same breaks were tried on the real FABulous flow's input: each one is caught here, before Ubuntu is needed."""
    from agents.rtl_agent import REFERENCE_VERILOG
    t = make_fab_wrapper("requant_i32_i8", REFERENCE_VERILOG["requant_i32_i8"])
    r = check_fab_wrapper("requant_i32_i8", t)
    assert r["status"] == "pass" and r["vectors"] == 485 and r["flop_bits"] == 40                       # 32 stored input bits + 8 result bits
    assert t.index("module requant_i32_i8_fab (") < t.index("module requant_i32_i8_fab_step (") < t.index("module requant_i32_i8(")
    for name, old, new, stage in [
            ("halves swapped", "{hi, lo}", "{lo, hi}", "function"),
            ("select pin inverted for the low half", "assign lo_n = io_in[16] ? lo : io_in[15:0];", "assign lo_n = io_in[16] ? io_in[15:0] : lo;", "structure"),
            ("wrong select pin", "assign hi_n = io_in[16] ?", "assign hi_n = io_in[17] ?", "function"),
            ("result on the wrong pins", "{3'b0, q, 17'b0}", "{2'b0, q, 18'b0}", "function"),
            ("result also leaks onto the input pins", "{3'b0, q, 17'b0}", "{3'b0, q, 9'b0, q}", "function"),
            ("low half mixed with what was loaded before", "assign lo_n = io_in[16] ? lo : io_in[15:0];", "assign lo_n = io_in[16] ? lo : (lo | io_in[15:0]);", "function"),
            ("io_oeb marks too few input pins", "28'b0000000000011111111111111111", "28'b0000000000001111111111111111", "function"),
            ("result register dropped", "        q_r <= q_n;\n", "", "structure"),
            ("a register hidden in the one-tick module", "    assign q_n = y;", "    reg [7:0] z; always @(posedge io_in[27]) z <= y;\n    assign q_n = z;", "structure")]:
        assert t.count(old) == 1, name
        bad = check_fab_wrapper("requant_i32_i8", t.replace(old, new))
        assert bad["status"] == "fail" and bad["stage"] == stage, (name, bad)
    moved = (t.replace("        q_r <= q_n;\n", "").replace("    reg [7:0] q_r;", "    wire [7:0] q_r = q_n;")
              .replace("    assign q_n = y;", "    reg [7:0] z; always @(posedge io_in[27]) z <= y;\n    assign q_n = z;"))
    bad = check_fab_wrapper("requant_i32_i8", moved)                    # still 40 flip-flop bits in all, but one register sits in the one-tick module
    assert bad["status"] == "fail" and "must exist and hold no registers" in bad["errors"][0]


def test_a_comment_line_does_not_hide_the_first_module():
    from agents.rtl_agent import REFERENCE_VERILOG
    t = make_fab_wrapper("relu8", REFERENCE_VERILOG["relu8"]).replace("module relu8_fab (", "// a comment before the module\nmodule relu8_fab (", 1)
    assert check_fab_wrapper("relu8", t)["status"] == "pass"


@pytest.mark.parametrize("op", ["relu8", "add8_sat"])
def test_wrapper_passes_through_the_pin_mapping(op):
    r = check_fab_wrapper(op, wrapped(op))
    assert r["status"] == "pass" and r["vectors"] >= 256 and r["flop_bits"] == 8


def test_the_wrapper_is_the_first_module_in_the_file():
    # real FABulous run: with the core first, ports were left unconnected and all logic was removed
    import re
    for op in ("relu8", "add8_sat"):
        assert re.findall(r"^module\s+(\w+)", wrapped(op), re.M) == [top_name(op), op]


def test_core_first_ordering_is_rejected():
    t = wrapped("add8_sat")
    core = REFERENCE_VERILOG["add8_sat"]
    wrapper_only = t[: t.index(core.strip())]
    bad = core + "\n" + wrapper_only
    r = check_fab_wrapper("add8_sat", bad)
    assert r["status"] == "fail" and r["stage"] == "structure" and "FIRST module" in r["errors"][0]


def test_wrapper_uses_the_fabric_port_names_and_one_result_register():
    t = wrapped("add8_sat")
    assert "module add8_sat_fab" in t and "input wire [27:0] io_in" in t and "posedge clk" in t and "`ifdef FAB_COMB" in t
    assert top_name("add8_sat") == "add8_sat_fab"


# ---- mutated wrappers must be caught (proves the check is not vacuous) ----
def test_wrong_pin_slice_is_caught():
    bad = wrapped("add8_sat").replace("io_in[15:8]", "io_in[14:7]")
    r = check_fab_wrapper("add8_sat", bad)
    assert r["status"] == "fail" and r["stage"] == "function" and r["mismatches"]


def test_result_on_the_wrong_pins_is_caught():
    bad = wrapped("relu8").replace("assign io_out = {12'b0, q, 8'b0};", "assign io_out = {12'b0, 8'b0, q};")
    assert bad != wrapped("relu8")
    assert check_fab_wrapper("relu8", bad)["status"] == "fail"


def test_missing_result_register_is_caught():
    bad = wrapped("relu8").replace("    always @(posedge clk) q_r <= y;\n    assign q = q_r;", "    assign q = y;")
    assert bad != wrapped("relu8")
    r = check_fab_wrapper("relu8", bad)
    assert r["status"] == "fail" and r["stage"] == "structure" and "flip-flop" in r["errors"][0]


def test_wrong_ports_and_garbage_are_rejected():
    bad = wrapped("relu8").replace("input wire clk,", "")
    assert check_fab_wrapper("relu8", bad)["status"] == "fail"
    assert check_fab_wrapper("relu8", "`include \"x\"")["stage"] == "screen"
    assert check_fab_wrapper("relu8", "")["status"] == "fail"


# ---- CLI + job folder ----
def test_prep_refuses_unverified_or_unwrappable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="run_rtl_agent"):
        prep_main(["--op", "add8_sat"])
    monkeypatch.setitem(rc.OPS, "wide2", WIDE2)
    with pytest.raises(SystemExit, match="cannot wrap wide2"):
        prep_main(["--op", "wide2"])
    with pytest.raises(SystemExit):
        prep_main(["--op", "nope"])


def test_prep_writes_the_job_folder(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "reports" / "rtl").mkdir(parents=True)
    (tmp_path / "reports" / "rtl" / "add8_sat.v").write_text(REFERENCE_VERILOG["add8_sat"])
    assert prep_main(["--op", "add8_sat"]) == 0
    job = tmp_path / "reports" / "fab_jobs" / "add8_sat"
    assert (job / "add8_sat_fab.v").exists() and (job / "run_fab_flow.sh").read_text() == FLOW_SCRIPT
    assert b"\r" not in (job / "run_fab_flow.sh").read_bytes()                       # bash needs LF line endings, even if written on Windows
    out = capsys.readouterr().out
    assert "PASS" in out and "run_fab_flow.sh add8_sat" in out


def test_flow_script_runs_the_documented_command():
    for part in ("load_fabric; run_FABulous_fabric; gen_user_design_wrapper", "compile_design", "PIPESTATUS", "exit_code.txt", "fabric.csv", "npnr_log"):
        assert part in FLOW_SCRIPT


def make_job(tmp_path, exit_code):
    j = tmp_path / "job"
    j.mkdir()
    (j / "x_fab.v").write_text("module x_fab; endmodule")
    shutil.copy(REAL / "nextpnr_log.txt", j / "x_fab_npnr_log.txt")
    shutil.copy(REAL / "design.fasm", j / "x_fab.fasm")
    (j / "flow_console.log").write_text("INFO | Compile flow completed successfully.\n")
    (j / "exit_code.txt").write_text(f"{exit_code}\n")
    return j


def test_job_folder_without_a_bitstream_is_inconclusive(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert check_main(["--job", str(make_job(tmp_path, 0))]) == 1
    out = capsys.readouterr().out
    assert "INCONCLUSIVE" in out and "tool claimed success: True" in out


def test_job_folder_with_failed_exit_code_is_failed(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert check_main(["--job", str(make_job(tmp_path, 1))]) == 1
    assert "FLOW STATUS: FAILED" in capsys.readouterr().out
