"""
Measured eFPGA cost of each building block A1 can place on the fabric, taken from real chip builds
(chip_build/<design>/flow_report.json + pnr.log), so a plan's resource use is known BEFORE a new build.

    python -m chip.fabric_costs          # either terminal: table + specifications/fabric_costs.json (from chip_build/)
    x2c chip costs                   # the same

Rules:
  * logic cells, MACC, RAM and X2 blocks add up across blocks (MEASURED per block, sum = ESTIMATE);
  * pins: an operator placed on the engine's result path uses no pins of its own (it sits between the X2 sums and the
    output bytes), so a combined design keeps the engine's pins - ASSUMED, to be confirmed by building the combination;
  * GBUF: the clock buffer is shared (max, not sum);
  * fmax of a combination is at most the slowest block's fmax (an upper bound, not a measurement).
A block without a successful build is reported as missing, never guessed.
"""
import json
import sys
import time
from pathlib import Path

from x2c import paths

ROOT = paths.ROOT
OUT = ROOT / "specifications" / "fabric_costs.json"
BLOCKS = {                       # what the eFPGA can host -> the design that implements it (built and checked)
    "x2_engine": "x2_engine2_4bit",              # both X2 blocks, 4-bit cells, 3-stage read pipeline
    "relu": "chain_relu8",                       # ReLU / ReLU6 clamp on INT8
    "add": "chain_add8_sat",                     # residual add, saturating INT8
    "requant": "chain_requant_i32_i8",           # int32 accumulator -> int8 (multiplier, shift, zero point)
}
ADDITIVE = ("FABULOUS_LC", "MACC_8x8_20", "RAM_32x4_2R_1W", "RERAM_CIM_32x32", "SYS_RESET")
NAMES = {"FABULOUS_LC": "logic_cells", "IOBUF": "pins", "MACC_8x8_20": "macc", "RAM_32x4_2R_1W": "ram32x4",
         "RERAM_CIM_32x32": "x2_blocks", "GBUF": "gbuf", "SYS_RESET": "sys_reset"}


def _fmax(design_dir):
    pnr = design_dir / "pnr.log"
    if not pnr.is_file():
        return None
    from x2c.verification.chip_chain import final_timing
    t = final_timing(pnr.read_text(errors="replace"))
    return min((v["fmax_mhz"] for v in t.values()), default=None)


def measure(root=ROOT):
    """Costs from chip_build/; None for a block whose build is missing or failed."""
    root = Path(root)
    out, budget = {}, {}
    for block, design in BLOCKS.items():
        d = root / "chip_build" / design
        rep = d / "flow_report.json"
        if not rep.is_file():
            out[block] = {"design": design, "status": "not built", "how": f"Ubuntu: x2c build --example {design}"
                          if not design.startswith("chain_") else f"Ubuntu: x2c chain --op {design[6:]} --target chip --runner native"}
            continue
        r = json.loads(rep.read_text(encoding="utf-8"))
        u = r.get("utilisation") or {}
        if r.get("status") != "ok" or not u:
            out[block] = {"design": design, "status": f"build {r.get('status')}"}
            continue
        for k, v in u.items():
            if isinstance(v, dict) and v.get("available") is not None:
                budget[NAMES.get(k, k)] = v["available"]
        out[block] = {"design": design, "status": "measured",
                      "used": {NAMES.get(k, k): v.get("used") for k, v in u.items() if isinstance(v, dict)},
                      "fmax_mhz": _fmax(d),
                      "timing": None if _fmax(d) is not None else "combinational block (no clocked path in nextpnr's report)",
                      "built": time.strftime("%Y-%m-%d", time.localtime(rep.stat().st_mtime))}
    return {"source": "chip_build/<design>/flow_report.json + pnr.log (nextpnr's model of the chip, not silicon)",
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "budget": budget, "blocks": out}


def load(root=ROOT):
    """Fresh measurement when chip_build/ exists (Ubuntu side), else the stored table (Windows side, tests)."""
    m = measure(root)
    if any(b.get("status") == "measured" for b in m["blocks"].values()):
        return m
    p = Path(root) / "specifications" / "fabric_costs.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else m


def estimate(blocks, costs=None):
    """Resource use of one design holding these blocks (each once, time-shared) and whether it fits the budget."""
    costs = costs or load()
    total, fmax, missing = {}, [], []
    for b in blocks:
        c = costs["blocks"].get(b)
        if not c or c.get("status") != "measured":
            missing.append(b)
            continue
        for k, v in c["used"].items():
            if v is None:
                continue
            raw = {n: r for r, n in NAMES.items()}.get(k, k)
            if raw in ADDITIVE:
                total[k] = total.get(k, 0) + v
            elif k == "pins":
                total[k] = c["used"][k] if b == "x2_engine" or "pins" not in total else total[k]
            else:
                total[k] = max(total.get(k, 0), v)
        if c.get("fmax_mhz"):
            fmax.append(c["fmax_mhz"])
    budget = costs.get("budget", {})
    over = {k: f"{v} > {budget[k]}" for k, v in total.items() if k in budget and v > budget[k]}
    return {"blocks": list(blocks), "used": total, "budget": budget, "fits": not over and not missing, "over": over,
            "missing_blocks": missing, "fmax_upper_bound_mhz": min(fmax) if fmax else None,
            "basis": "MEASURED per block, summed (ESTIMATE); pins assumed = engine's (operators on its result path)"}


def blocks_for(net):
    """eFPGA blocks a deployed INT8 network would need if its host steps moved onto the fabric."""
    from x2c.compiler import int8_net as N
    need, unknown = ["x2_engine"], []
    for s in net.steps:
        if isinstance(s, N.Layer):
            need += ["requant"] + (["relu"] if s.lo == 0 and s.requant is None else [])
        elif s[0] == "add":
            need.append("add")
        else:
            unknown.append(s[0])
    return list(dict.fromkeys(need)), sorted(set(unknown))


def main():
    m = measure()
    if not any(b.get("status") == "measured" for b in m["blocks"].values()):
        print("no chip builds in chip_build/ here (run this in Ubuntu after the builds); stored table:")
        m = load()
    else:
        OUT.write_text(json.dumps(m, indent=1) + "\n", encoding="utf-8")
        print(f"written: {OUT.relative_to(ROOT).as_posix()}")
    print(f"{'block':10s} {'design':24s} {'cells':>6s} {'pins':>5s} {'macc':>5s} {'x2':>3s} {'fmax MHz':>9s}  status")
    for b, c in m["blocks"].items():
        u = c.get("used", {})
        print(f"{b:10s} {c['design']:24s} {u.get('logic_cells', '-')!s:>6s} {u.get('pins', '-')!s:>5s} {u.get('macc', '-')!s:>5s} "
              f"{u.get('x2_blocks', '-')!s:>3s} {c.get('fmax_mhz') or ('comb.' if c.get('timing') else '-')!s:>9s}  {c['status']}")
    e = estimate(list(BLOCKS), m)
    print(f"\nengine + relu + add + requant in one design (ESTIMATE): {e['used']}  budget {e['budget']}  "
          f"fits={e['fits']}{' over: ' + str(e['over']) if e['over'] else ''}"
          f"{' missing: ' + str(e['missing_blocks']) if e['missing_blocks'] else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
