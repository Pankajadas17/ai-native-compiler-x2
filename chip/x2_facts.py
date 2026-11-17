"""
One place for every X2 / chip number the compiler uses that is NOT confirmed on silicon, and the path to replace it later.

Each fact has a value, a status and where it comes from:
  decision  -- chosen by the project (e.g. 4-bit cells per weight)
  model     -- read from the chip repo's behavioral model or tools (not silicon)
  measured  -- measured here (simulation or nextpnr), with how
  missing   -- not known at all yet; a placeholder the code refuses to use silently (value None)

To add or correct a number later (a team answer, a silicon measurement), do NOT edit code: put it in
specifications/x2_overrides.json, e.g.
    {"program_cycles_per_cell": {"value": 410, "status": "measured", "source": "bench test 2026-12-01, chip #3"}}
Every consumer reads facts through value(); tests/test_x2_facts.py checks the file and that consumers pick it up.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OVERRIDES = ROOT / "specifications" / "x2_overrides.json"
STATUSES = ("decision", "model", "measured", "missing")

FACTS = {
    "cell_bits": {
        "value": 4, "status": "decision", "unit": "bits per X2 cell",
        "source": "project decision 2026-11-13 (BMsemi DeepGate 4-bit banks); the chip repo's behavioral X2 computes 1-bit",
        "used_by": ["specifications/hardware_config.json x2.cell_bits", "compiler/chip_x2.cells_per_weight"],
        "replace_when": "the team confirms the silicon cell precision"},
    "clock_mhz": {
        "value": 14.0, "status": "decision", "unit": "MHz",
        "source": ("highest clock every chip design meets, with margin: min nextpnr Fmax = 14.45 MHz (x2_engine2_4bit, "
                   "2026-11-16); others: x2_engine2 16.67, x2_engine 17.51, relu8 151.5, add8_sat 42.2. requant_i32_i8 "
                   "has no register-to-register path through its multiplier, so nextpnr gives it no Fmax"),
        "used_by": ["verification/chip_chain.judge (timing check)", "compiler/x2_schedule (latency in microseconds)"],
        "replace_when": "a new design is slower, or the team fixes the system clock"},
    "program_cycles_per_cell": {
        "value": 365, "status": "measured", "unit": "clock cycles per SET/RESET command",
        "source": "least-squares fit to the 4 X2 engine simulations (BM behavioral X2: PROGRAM_DELAY 360 + command overhead)",
        "used_by": ["compiler/x2_schedule.tile_costs"], "replace_when": "silicon programming time is known"},
    "mvm_cycles": {
        "value": 511, "status": "measured", "unit": "clock cycles per MVM (plus beat_cycles per input row)",
        "source": "same fit (BM behavioral X2: COMPUTE_DELAY 450 + results)", "used_by": ["compiler/x2_schedule.tile_costs"],
        "replace_when": "silicon compute time is known"},
    "beat_cycles": {
        "value": 3, "status": "measured", "unit": "clock cycles per input row of an MVM", "source": "same fit (3.1)",
        "used_by": ["compiler/x2_schedule.tile_costs"], "replace_when": "silicon compute time is known"},
    "blocks": {
        "value": 2, "status": "model", "unit": "X2 blocks", "source": "two RERAM_CIM_32x32 BELs in .FABulous/bel.txt",
        "used_by": ["specifications/hardware_config.json x2.blocks"], "replace_when": "the fabric changes"},
    "cell_compute_4bit": {
        "value": None, "status": "missing", "unit": "how a 4-bit cell contributes to a column count",
        "source": ("not modelled by the chip repo (its X2 counts a binary cell as on/off). bm_verif assumes count = sum of "
                   "pwm * level, which is what compiler/x2_pack's 4-bit packing relies on; it cannot be checked in simulation"),
        "used_by": ["compiler/x2_pack (4-bit packing)", "examples/chip_designs/x2_engine*_4bit.v"],
        "replace_when": "the team gives the 4-bit cell transfer function or a 4-bit X2 model"},
    "silicon_variation": {
        "value": None, "status": "missing", "unit": "per-cell conductance spread / noise",
        "source": "unknown; every check here is exact digital simulation", "used_by": [],
        "replace_when": "silicon characterisation exists (then add a noise model to chip/x2_golden.py)"},
    "error_bits_exposed": {
        "value": 2, "status": "model", "unit": "bits of the X2's 3-bit error code visible on the fabric",
        "source": "RERAM_CIM_32x32 primitive has ERROR0/ERROR1 only", "used_by": [],
        "replace_when": "the primitive exposes the third bit"},
}


class FactMissing(RuntimeError):
    pass


def load_overrides(path=None):
    """{} when the file does not exist. Raises ValueError for anything malformed (never applied silently)."""
    p = Path(path or OVERRIDES)
    if not p.is_file():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{p}: must be an object of fact name -> {{value, status, source}}")
    for k, v in data.items():
        if k not in FACTS:
            raise ValueError(f"{p}: unknown fact '{k}' (known: {sorted(FACTS)}); add it to chip/x2_facts.py FACTS first")
        if not (isinstance(v, dict) and "value" in v and v.get("status") in STATUSES and isinstance(v.get("source"), str) and v["source"]):
            raise ValueError(f"{p}: '{k}' needs value, status (one of {STATUSES}) and a non-empty source")
    return data


def facts(path=None):
    out = {k: dict(v) for k, v in FACTS.items()}
    for k, v in load_overrides(path).items():
        out[k].update(v, overridden_by=Path(path or OVERRIDES).name)
    return out


def value(name, path=None):
    f = facts(path)[name]
    if f["value"] is None:
        raise FactMissing(f"X2 fact '{name}' is missing: {f['source']}. Add it in specifications/x2_overrides.json "
                          f"(see chip/x2_facts.py) when {f['replace_when']}.")
    return f["value"]


def summary(path=None):
    """What a report should say about the numbers it used."""
    return {k: {"value": v["value"], "status": v["status"], **({"overridden_by": v["overridden_by"]} if "overridden_by" in v else {})}
            for k, v in facts(path).items()}
