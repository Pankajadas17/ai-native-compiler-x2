"""Build the chip profile from the chip repo's own files.

Every number in the profile is parsed from a repo file; the profile records which
file and how. Nothing here is a hand-typed hardware number.

  fabric resources, sites, grid ... FABulous BEL database (bel.txt, bel.v2.txt)
  I/O pins ........................ constraints.pcf
  X2 ports, buses, parameters ..... Yosys primitive RERAM_CIM_32x32.v
  X2 rows / columns ............... 2 ** width(ROW_INDEX), 2 ** width(COL_INDEX)
  X2 config bits .................. CFG entries of the X2 BEL in bel.v2.txt
  X2 modes, delays, result format . BM Labs behavioral model (.sv)
"""
import re
from collections import Counter, OrderedDict

from x2c.chip.chip_repo import INPUT_FILES, ChipRepo

SCHEMA = "bm_verif.chip_profile/1"
X2_PRIMITIVE = "RERAM_CIM_32x32"
_TILE = re.compile(r"^#Tile_X(\d+)Y(\d+)\s*$")


# ---------------------------------------------------------------- BEL database
def parse_bels(text):
    """bel.txt rows: site,X,Y,letter,TYPE,ports... ; '#Tile_XnYm' comment per tile."""
    counts, sites, xs, ys = Counter(), {}, set(), set()
    for line in text.splitlines():
        m = _TILE.match(line)
        if m:
            xs.add(int(m.group(1)))
            ys.add(int(m.group(2)))
            continue
        if not line or line.startswith("#"):
            continue
        f = line.split(",")
        if len(f) < 5 or not f[4]:
            continue
        counts[f[4]] += 1
        sites.setdefault(f[4], []).append(f"{f[0]}/{f[3]}")
    if not xs:
        raise ValueError("bel.txt: no '#Tile_XnYm' lines found")
    return {"resources": dict(sorted(counts.items())), "sites": sites,
            "grid": {"columns": max(xs) + 1, "rows": max(ys) + 1}}


def parse_bel_cfg(text, bel_type):
    """CFG entries of the FIRST BEL of bel_type in bel.v2.txt -> (n_bits, {field: width})."""
    inside, cfg = False, []
    for line in text.splitlines():
        if line.startswith("BelBegin,"):
            inside = line.split(",")[3] == bel_type
            continue
        if inside and line.startswith("BelEnd"):
            break
        if inside and line.startswith("CFG,"):
            cfg.append(line.split(",", 1)[1])
    if not cfg:
        raise ValueError(f"bel.v2.txt: no CFG entries for {bel_type}")
    widths = Counter(re.sub(r"\[\d+\]$", "", c) for c in cfg)
    return len(cfg), dict(sorted(widths.items()))


# ---------------------------------------------------------------- pin file
def parse_pcf(text):
    """set_io NAME SITE lines, grouped by the comment heading above them."""
    groups, heading = OrderedDict(), "ungrouped"
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("#"):
            heading = line.lstrip("#").strip() or heading
            continue
        parts = line.split()
        if len(parts) == 3 and parts[0] == "set_io":
            groups.setdefault(heading, OrderedDict())[parts[1]] = parts[2]
    if not groups:
        raise ValueError("constraints.pcf: no set_io lines")
    return groups


# ---------------------------------------------------------------- Verilog primitive
_PORT = re.compile(r"^\s*(?:\(\*.*?\*\)\s*)?(input|output|inout)\s+(?:wire\s+|logic\s+)?(?:\[[^\]]+\]\s*)?(\w+)", re.M)
_PARAM = re.compile(r"parameter\s+(?:\w+\s+)?\[\s*(\d+)\s*:\s*(\d+)\s*\]\s*(\w+)")


def parse_primitive(text, module):
    m = re.search(r"module\s+" + module + r"\b(.*?)\bendmodule", text, re.S)
    if not m:
        raise ValueError(f"module {module} not found")
    body = re.sub(r"//[^\n]*", "", m.group(1))
    ports = [(d, n) for d, n in _PORT.findall(body)]
    params = {n: int(hi) - int(lo) + 1 for hi, lo, n in _PARAM.findall(body)}
    if not ports:
        raise ValueError(f"module {module}: no ports parsed")
    return ports, params


def group_buses(ports):
    """MODE0, MODE1 -> MODE width 2. Single wires stay as they are. Checks bits are contiguous."""
    bits, single = {}, OrderedDict()
    for d, n in ports:
        m = re.match(r"^([A-Z_]+?)(\d+)$", n)
        if m:
            bits.setdefault(m.group(1), (d, []))[1].append(int(m.group(2)))
        else:
            single[n] = d
    buses = OrderedDict()
    for base, (d, idx) in bits.items():
        if sorted(idx) != list(range(len(idx))):
            raise ValueError(f"bus {base}: bits {sorted(idx)} are not 0..n-1")
        buses[base] = {"direction": d, "width": len(idx)}
    return buses, single


# ---------------------------------------------------------------- behavioral model
def parse_behavioral(text, module="sky130_bmlabs_reram__cim32x32"):
    m = re.search(r"module\s+" + module + r"\s*#\((.*?)\)\s*\(", text, re.S)
    params = {}
    if m:
        for name, value in re.findall(r"parameter\s+[^=]*?(\w+)\s*=\s*([^,\n]+)", m.group(1)):
            params[name] = _verilog_int(value.strip())
    modes = {name: int(v, 2) for name, v in
             re.findall(r"localparam\s+logic\s*\[1:0\]\s*MODE_(\w+)\s*=\s*2'b([01]{2})", text)}
    fmt = None
    fm = re.search(r"Data format is\s*\{([^}]*)\}", text)
    if fm:
        fmt = []
        for field in fm.group(1).split(","):
            g = re.match(r"\s*(\w+)\s*\[(\d+):(\d+)\]", field)
            if g:
                fmt.append({"field": g.group(1), "width": int(g.group(2)) - int(g.group(3)) + 1})
    return {"module": module, "parameters": params, "modes": dict(sorted(modes.items(), key=lambda kv: kv[1])),
            "result_format_msb_first": fmt}


def _verilog_int(s):
    m = re.match(r"(\d+)'([bhd])([0-9a-fA-F_]+)$", s)
    if m:
        return int(m.group(3).replace("_", ""), {"b": 2, "h": 16, "d": 10}[m.group(2)])
    try:
        return int(s)
    except ValueError:
        return s


# ---------------------------------------------------------------- the profile
def build_chip_profile(repo=None):
    repo = repo if isinstance(repo, ChipRepo) else ChipRepo(repo)
    bels = parse_bels(repo.read_text(INPUT_FILES["bels"]))
    n_cfg, cfg_fields = parse_bel_cfg(repo.read_text(INPUT_FILES["bels_v2"]), X2_PRIMITIVE)
    pins = parse_pcf(repo.read_text(INPUT_FILES["pins"]))
    ports, prim_params = parse_primitive(repo.read_text(INPUT_FILES["x2_primitive"]), X2_PRIMITIVE)
    buses, single = group_buses(ports)
    behav = parse_behavioral(repo.read_text(INPUT_FILES["x2_behavioral"]))

    general = pins.get("All I/Os", OrderedDict())
    io_sites = bels["sites"].get("IOBUF", [])
    x2_sites = bels["sites"].get(X2_PRIMITIVE, [])
    profile = {
        "schema": SCHEMA,
        "status": "derived_from_chip_repo",
        "source": repo.provenance(),
        "fabric": {
            "name": "classic_fabric_reram",
            "grid": bels["grid"],
            "resources": bels["resources"],
        },
        "io": {
            "iobuf_sites": len(io_sites),
            "iobuf_site_list": io_sites,
            "general_pins": list(general),
            "general_pin_sites": dict(general),
            "other_pin_groups": {k: dict(v) for k, v in pins.items() if k != "All I/Os"},
        },
        "x2": {
            "primitive": X2_PRIMITIVE,
            "count": bels["resources"].get(X2_PRIMITIVE, 0),
            "sites": x2_sites,
            "rows": 2 ** buses["ROW_INDEX"]["width"],
            "cols": 2 ** buses["COL_INDEX"]["width"],
            "rows_cols_basis": "2 ** width(ROW_INDEX), 2 ** width(COL_INDEX) in the Yosys primitive",
            "pwm_bits": buses["PWM_DATA"]["width"],
            "result_bits": buses["RESULT_DATA"]["width"],
            "buses": buses,
            "single_wires": single,
            "parameters": prim_params,
            "config_bits_per_block": n_cfg,
            "config_fields": cfg_fields,
            "behavioral_model": behav,
        },
        "not_in_repo": [
            "INT8 activation/weight -> PWM value and count -> output scaling (team answer needed)",
            "meaning of the eight X2 parameters (TARGET_SET1 ... TDC_DEAD_TIME) as numbers to choose",
            "analog effects: the behavioral model says they are not simulated",
        ],
    }
    _self_check(profile)
    return profile


def _self_check(p):
    """Cross-checks between independent files; raise if the repo disagrees with itself."""
    x2 = p["x2"]
    errors = []
    if x2["count"] != len(x2["sites"]):
        errors.append("X2 count != number of X2 sites")
    if set(x2["config_fields"]) != set(x2["parameters"]):
        errors.append(f"bitstream CFG fields {sorted(x2['config_fields'])} != primitive parameters {sorted(x2['parameters'])}")
    elif any(x2["config_fields"][k] != x2["parameters"][k] for k in x2["parameters"]):
        errors.append("bitstream CFG widths != primitive parameter widths")
    if x2["config_bits_per_block"] != sum(x2["parameters"].values()):
        errors.append("X2 config bits != sum of parameter widths")
    fmt = x2["behavioral_model"]["result_format_msb_first"]
    if fmt and sum(f["width"] for f in fmt) != x2["result_bits"]:
        errors.append("behavioral result format width != primitive RESULT_DATA width")
    io_sites = set(p["io"]["iobuf_site_list"])
    for group in [p["io"]["general_pin_sites"]] + list(p["io"]["other_pin_groups"].values()):
        bad = sorted(n for n, site in group.items() if site not in io_sites)
        if bad:
            errors.append(f"pins on non-IOBUF sites: {bad}")
    if errors:
        raise ValueError("chip repo is inconsistent: " + "; ".join(errors))

