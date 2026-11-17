"""Generate specifications/hardware_config.json (v1.0) from the chip profile.

    python run_hardware_config.py          # write it (the first time, the old file is kept as
                                           # specifications/hardware_config.compiler_v2.json)
    python run_hardware_config.py --check  # exit 1 if the saved file is stale

Every chip number comes from specifications/chip_profile.json (derived from sky130-reram-fpga).
The weight scheme combines the chip's 1-bit cells with BMsemi_Model's signed-INT8 scheme; each
field says where it comes from. The 64-row compiler geometry is kept ONLY as `compiler_v2`, to check
that the BMsemi compiler artifact is read correctly; it is not the chip.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPEC = ROOT / "specifications"
PROFILE = SPEC / "chip_profile.json"
OUT = SPEC / "hardware_config.json"
OLD = SPEC / "hardware_config.compiler_v2.json"

from chip.chip_repo import x2_model_label  # noqa: E402  ($BM_X2_REPO or the chip repo's copy)

BEHAV = x2_model_label()        # line numbers below (l.NNN) refer to the chip repo's model
# Bits stored per X2 cell. DECISION (user, 2026-11-13): 4, following BMsemi DeepGate's 4-bit banks.
# The chip repo's behavioral model computes with 1-bit cells; pass --cell-bits 1 to model that instead.
CELL_BITS = 4
DEEPGATE = "BMsemi_Model/projects/1_foundation/PE/variants/deep_gate/Crossbar_array.py"


def build(profile, cell_bits=CELL_BITS):
    x2, fab, io = profile["x2"], profile["fabric"]["resources"], profile["io"]
    bm = x2["behavioral_model"]["parameters"]
    return {
        "version": "1.0-chip",
        "status": ("chip numbers derived from sky130-reram-fpga via chip_profile.json "
                   f"(input digest {profile['source']['input_digest'][:16]}); weights.magnitude_bits and "
                   "weights.sign_factor follow BMsemi_Model DeepGate and are not stated by the chip repo"),
        "x2": {
            "rows": x2["rows"], "cols": x2["cols"], "blocks": x2["count"], "sites": x2["sites"],
            "cell_bits": cell_bits, "input_bits": x2["pwm_bits"],
            "result_count_bits": next(f["width"] for f in x2["behavioral_model"]["result_format_msb_first"] if f["field"] == "count"),
            "timing_cycles": {"read": bm["READ_DELAY"], "program": bm["PROGRAM_DELAY"], "compute": bm["COMPUTE_DELAY"]},
            "basis": {
                "rows/cols": "2**width(ROW_INDEX/COL_INDEX) in RERAM_CIM_32x32.v; array_state[0:31] in " + BEHAV,
                "blocks": "two RERAM_CIM_32x32 BELs in .FABulous/bel.txt (X20Y6, X20Y14)",
                "cell_bits": (f"{cell_bits}: DECISION (user, 2026-11-13) to follow BMsemi DeepGate 4-bit banks "
                              f"({DEEPGATE} l.127, 165). NOT what the chip repo models: its behavioral X2 computes with "
                              f"1-bit cells ({BEHAV} l.802, l.847-848), so 4-bit plans cannot be checked in that "
                              "simulation. Pending team confirmation of the silicon cell precision.")
                             if cell_bits == 4 else
                             f"1: COMPUTE adds pwm only where array_state is 1 ({BEHAV} l.847-848); SET/RESET store 1/0 (l.802)",
                "input_bits": f"i_pwm_data[7:0] ({BEHAV} l.38), unsigned",
                "timing_cycles": f"READ/PROGRAM/COMPUTE_DELAY parameters ({BEHAV} l.14-16): behavioral model, not silicon",
            },
        },
        "weights": {
            "dtype": "int8_signed", "magnitude_bits": 8, "sign_factor": 2,
            "cells_per_weight_formula": "sign_factor * ceil(magnitude_bits / x2.cell_bits)",
            "basis": (f"{DEEPGATE} l.29-30: signed INT8 weights split into positive/negative banks (w = w_pos - w_neg); "
                      "l.127/165 store each magnitude as two 4-bit banks, i.e. 8 magnitude bits. "
                      f"With x2.cell_bits = {cell_bits}: 2 x ceil(8 / {cell_bits}) = {2 * -(-8 // cell_bits)} cells per weight."),
            "open_question": ("silicon cell precision: deep_gate/Crossbar_array.py uses 4-bit banks (4 cells per weight); "
                              "the X2 behavioral model computes with 1-bit cells (16 cells per weight)."),
        },
        "efpga": {
            "luts": fab["FABULOUS_LC"], "flip_flops": fab["FABULOUS_LC"], "macc_blocks": fab["MACC_8x8_20"],
            "ram_32x4_blocks": fab["RAM_32x4_2R_1W"], "global_buffers": fab["GBUF"], "io_pins": io["iobuf_sites"],
            "basis": ("BEL counts in .FABulous/bel.txt (one LUT4 + FF per FABULOUS_LC); io_pins = IOBUF sites, "
                      "the team confirmed all[0..27] in constraints.pcf are example assignments only. "
                      "Per-operator resource estimates still need RTL (run_chip_flow.py)."),
        },
        "activation_bytes": 1,
        "activation_bytes_basis": (f"X2 inputs are 8-bit ({BEHAV} l.38); DeepGate activations are unsigned INT8 "
                                   "(Crossbar_array.py l.28). compiler_graph.json itself records no dtype."),
        "compiler_v2": {
            "pe_rows": 64, "default_tile_cols": 32,
            "basis": ("BMsemi_Model bm_models/core/mapping.py PEGeometry default (64 x 32, the 'YOLO26 / FP4LLM' "
                      "geometry, mapping.py l.4-6/20-21). Used ONLY to check the compiler's own tile counts "
                      "(compiler_expected_tiles); the chip is 32 x 32."),
        },
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--cell-bits", type=int, choices=(1, 4), default=CELL_BITS,
                    help="bits per X2 cell: 4 = decision following BMsemi DeepGate (default), 1 = chip behavioral model")
    a = ap.parse_args(argv)
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    text = json.dumps(build(profile, a.cell_bits), indent=2) + "\n"
    if a.check:
        same = OUT.is_file() and OUT.read_text(encoding="utf-8") == text
        print(f"hardware_config.json up to date: {same}")
        return 0 if same else 1
    if OUT.is_file() and not OLD.is_file():
        old = json.loads(OUT.read_text(encoding="utf-8"))
        if "compiler_v2" not in old:
            shutil.copyfile(OUT, OLD)
            print(f"kept the previous file as {OLD.name}")
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    hw = json.loads(text)
    print(f"wrote {OUT}: X2 {hw['x2']['rows']}x{hw['x2']['cols']} x {hw['x2']['blocks']} blocks, "
          f"{hw['x2']['cell_bits']}-bit cells, {hw['weights']['sign_factor'] * -(-hw['weights']['magnitude_bits'] // hw['x2']['cell_bits'])} cells per weight, "
          f"eFPGA {hw['efpga']['luts']} LC / {hw['efpga']['macc_blocks']} MACC / {hw['efpga']['io_pins']} pins")
    return 0


if __name__ == "__main__":
    sys.exit(main())
