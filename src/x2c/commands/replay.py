"""
x2c replay: replay a deployment bundle from its FILES ALONE and check it reproduces every expected tensor.

    x2c replay deploy/mobilenetv2          # either terminal   (also: x2c replay deploy/mnist)

What it reads (nothing else about the model):
  host_program.json   the ordered steps: X2 layers (attrs, input offset, zero point, epilogue as data), adds, pool ...
  x2_program.npz      per layer and tile: the 32x32 cell levels and the placement = what gets programmed into the X2
  expected.npz        the input and every tensor the deployment must produce
What it does per X2 layer: im2col with the stored offset/padding -> the X2 model executes the PROGRAMMED CELLS tile by
tile -> recombine -> offset correction, whose weight sums are read back from the cells (an all-ones input), not from
the model -> the stored epilogue (bias/shift/clamp, or the requantization formula) -> compared with expected.npz.
It does not use the IR, BMsemi_Model, the weights or the golden model (except for a DeepGate bundle whose requantization
matched no integer formula: then the repo's fwd is imported and the report says so).
Evidence: Python X2 model (4-bit, ASSUMED cell rule), not the chip.
"""
from x2c import paths
import json
import sys
from math import ceil
from pathlib import Path

import numpy as np

ROOT = paths.ROOT

from x2c.compiler import chip_x2  # noqa: E402
from x2c.compiler import int8_net as N  # noqa: E402
from x2c.compiler import x2_pack as xp  # noqa: E402


def _package(step, arrays):
    meta, name = step["x2"], step["name"]
    hw = {"x2": {k: meta[k] for k in ("cell_bits", "rows", "cols", "input_bits")},
          "weights": {"magnitude_bits": meta["magnitude_bits"]}}
    placement = [[tuple(e) for e in tile] for tile in json.loads(str(arrays[f"{name}__placement"]))]
    cells = [np.asarray(arrays[f"{name}__tile{i}__levels"], dtype=np.int64) for i in range(len(placement))]
    return xp.LayerPackage(name, dict(step["attrs"]), hw, meta["input_split_bits"], placement, cells, None,
                           meta["tdc_time_out"], 0)


def _requant(spec, notes):
    f = N.requant_from_spec(spec)
    if f is not None:
        return f
    if spec.get("kind") == "bmsemi_deepgate_fwd":
        try:
            from BMsemi_Model.bm_models.models import deepgate as dg  # only when no integer formula matched
        except ImportError as e:
            raise RuntimeError("this bundle's requantization needs BMsemi_Model's deepgate fwd (no integer formula "
                               "matched it at deploy time), and BMsemi_Model is not importable here") from e
        fwd = dg._load()[0]
        n = "requantization: BMsemi_Model's deepgate fwd (no integer formula matched it)"
        if n not in notes:
            notes.append(n)
        return lambda acc: np.asarray(fwd(np.asarray(acc, dtype=np.int64), np.eye(len(spec["bias"]), dtype=np.int8), np.asarray(spec["bias"]),
                                          np.asarray(spec["mult"]), np.asarray(spec["shift"]), spec["out_zp"], np.int8), dtype=np.int64)
    raise ValueError(f"the bundle's requantization is not replayable: {spec}")


def _layer(step, notes):
    ep = step["epilogue"]
    a = step["attrs"]
    L = N.Layer(step["name"], a, None, np.zeros(a["K"], dtype=np.int64), step["reads"], int(step["x2_input_offset"]),
                model_zp=int(step["model_zero_point"]), out=step["writes"])
    if "requant" in ep:
        L.requant = _requant(ep["requant"], notes)
    else:
        L.bias, L.shift, (L.lo, L.hi) = np.asarray(ep["bias"], dtype=np.int64), int(ep["shift"]), ep["clamp"]
    return L


def replay(folder):
    folder = Path(folder)
    prog = json.loads((folder / "host_program.json").read_text(encoding="utf-8"))
    arrays = np.load(folder / "x2_program.npz")
    expected = np.load(folder / "expected.npz")
    old = [s["name"] for s in prog["steps"] if s["step"] == "x2_layer" and (not isinstance(s.get("x2"), dict)
                                                                          or not isinstance(s.get("epilogue"), dict))]
    if old:
        raise ValueError(f"bundle written by an older x2c deploy (no replay data for {old[0]}); deploy it again: "
                         f"x2c deploy --workload <model>")
    t = {"input": np.asarray(expected["input"], dtype=np.int64)}
    rows, notes, stats = [], [], {"program_commands": 0, "compute_beats": 0}
    for step in prog["steps"]:
        if step["step"] == "x2_layer":
            L = _layer(step, notes)
            pkg = _package(step, arrays)
            P, hwd = N.im2col(N._src(t, L), L)
            y_u, st = xp.run_layer(pkg, P)
            g, taps, _ = chip_x2.layer_shape(L.attrs)
            w_sum, _ = xp.run_layer(pkg, np.ones((1, g, taps), dtype=np.int64))     # sum of weights per output, from the cells
            acc = (y_u - (L.in_zp + L.model_zp) * w_sum[0]).T
            acc = acc.reshape(-1) if hwd is None else acc.reshape(-1, *hwd)
            t[L.out] = N.epilogue(acc, L)
            stats["program_commands"] += st["program_commands"]
            stats["compute_beats"] += st["compute_beats"]
            name, kind = L.out, "x2_layer"
        else:
            t[step["writes"]] = N._host_step((step["step"], step["writes"], *step["reads"]), t)
            name, kind = step["writes"], step["step"]
        exact = name in expected.files and np.array_equal(t[name], expected[name])
        rows.append({"step": name, "kind": kind, "exact": bool(exact)})
    bad = next((r["step"] for r in rows if not r["exact"]), None)
    return {"bundle": folder.as_posix(), "model": prog.get("model"), "exact": bad is None, "steps": len(rows),
            "first_mismatch": bad, "rows": rows, "x2_stats": stats, "notes": notes,
            "uses": "bundle files only (host_program.json, x2_program.npz, expected.npz) + the 4-bit Python X2 model",
            "output": prog.get("output")}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if argv else 2
    rc = 0
    for f in argv:
        try:
            r = replay(f)
        except (OSError, KeyError, ValueError, TypeError, RuntimeError) as e:
            print(f"CANNOT REPLAY {f}: {type(e).__name__}: {e}")
            rc = 1
            continue
        (Path(f) / "replay_report.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        print(f"REPLAY {r['bundle']} ({r['model']}): {sum(x['exact'] for x in r['rows'])}/{r['steps']} steps exact, "
              f"{r['x2_stats']['program_commands']} cell writes, {r['x2_stats']['compute_beats']} compute beats")
        for n in r["notes"]:
            print(f"  note: {n}")
        print(f"  {'BUNDLE REPLAYS EXACTLY' if r['exact'] else 'MISMATCH at ' + str(r['first_mismatch'])}   [{r['uses']}]")
        rc |= 0 if r["exact"] else 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
