"""Step 22: connecting the pins of a custom design in FABulous' generated top wrapper (found on a real run: ports were left empty)."""
import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

from run_fab_prep import FLOW_SCRIPT, main as prep_main
from agents.rtl_agent import REFERENCE_VERILOG

spec = importlib.util.spec_from_file_location("fab_connect", Path(__file__).parent.parent / "tools" / "fab_connect.py")
fc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fc)

# lines copied from the REAL generated top_wrapper.v of the 2026-11-10 run
REAL_WIRES = "// bel IO_1_bidirectional_frame_config_pass output wires:\nwire [27:0]IO_1_bidirectional_frame_config_pass_O;\n"
REAL_INSTANCE = "// instantiate user_design\nadd8_sat_fab user_design_i (.clk(clk), .io_in(), .io_out(), .io_oeb());\n\n\nendmodule //top_wrapper\n"


def test_the_real_empty_instance_gets_connected():
    new, msg = fc.connect(REAL_WIRES + REAL_INSTANCE, "add8_sat_fab")
    assert ("add8_sat_fab user_design_i (.clk(clk), .io_in(IO_1_bidirectional_frame_config_pass_O), "
            ".io_out(IO_1_bidirectional_frame_config_pass_I), .io_oeb(IO_1_bidirectional_frame_config_pass_T));") in new
    assert "connected" in msg and new.count("user_design_i") == 1 and "endmodule //top_wrapper" in new


def test_connecting_twice_is_harmless():
    once, _ = fc.connect(REAL_WIRES + REAL_INSTANCE, "add8_sat_fab")
    again, msg = fc.connect(once, "add8_sat_fab")
    assert again == once and msg == "already connected"


def test_matches_what_fabulous_does_for_its_own_demo_design():
    # fabric_cad/gen_design_top_wrapper.py (FABulous 2.2.0) for sequential_16bit_en
    demo = (".clk(clk), .io_in(IO_1_bidirectional_frame_config_pass_O), "
            ".io_out(IO_1_bidirectional_frame_config_pass_I), .io_oeb(IO_1_bidirectional_frame_config_pass_T)")
    assert fc.CONNECTED == demo


@pytest.mark.parametrize("text,why", [
    (REAL_INSTANCE, "no IO_1_bidirectional"),                                   # fabric without these pins
    (REAL_WIRES + REAL_INSTANCE.replace("add8_sat_fab", "other_fab"), "found 0"),  # wrong module name
    (REAL_WIRES + REAL_INSTANCE * 2, "found 2"),
    (REAL_WIRES + "add8_sat_fab user_design_i (.clk(clk), .io_in(x));\n", "unexpected port list"),   # never overwrite hand-made wiring
])
def test_it_refuses_instead_of_guessing(text, why):
    with pytest.raises(ValueError, match=why):
        fc.connect(text, "add8_sat_fab")


def test_cli_returns_nonzero_and_leaves_the_file_alone_on_failure(tmp_path, capsys):
    f = tmp_path / "top_wrapper.v"
    f.write_text(REAL_INSTANCE)
    assert fc.main(["x", str(f), "add8_sat_fab"]) == 1
    assert f.read_text() == REAL_INSTANCE and "CANNOT CONNECT" in capsys.readouterr().err


def test_cli_edits_the_file(tmp_path):
    f = tmp_path / "top_wrapper.v"
    f.write_text(REAL_WIRES + REAL_INSTANCE)
    assert fc.main(["x", str(f), "add8_sat_fab"]) == 0 and ".io_in(IO_1_bidirectional_frame_config_pass_O)" in f.read_text()


# ---- the generated flow script ----
def test_script_is_valid_bash_with_lf_endings():
    assert "\r" not in FLOW_SCRIPT
    if shutil.which("bash") is None:
        pytest.skip("bash not available")
    # via stdin: on Windows `bash` is WSL's bash and cannot open a Windows temp-file path
    r = subprocess.run(["bash", "-n"], input=FLOW_SCRIPT.encode(), capture_output=True)
    assert r.returncode == 0, r.stderr.decode(errors="replace")


def test_script_order_generate_then_connect_then_compile():
    i_gen = FLOW_SCRIPT.index("run_FABulous_fabric; gen_user_design_wrapper")
    i_fix = FLOW_SCRIPT.index('python3 "$ROOT/tools/fab_connect.py"')
    i_cmp = FLOW_SCRIPT.index("load_fabric; compile_design")
    assert i_gen < i_fix < i_cmp
    assert "exit_code.txt" in FLOW_SCRIPT and "97" in FLOW_SCRIPT           # a failed connect is never reported as a good run
    assert "rm -f" in FLOW_SCRIPT                                            # stale outputs of an earlier run must not be picked up
