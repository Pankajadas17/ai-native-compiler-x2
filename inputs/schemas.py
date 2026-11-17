"""
Local (internal) schemas. External assets are normalised into these; agents and the verification pipeline
only ever see WorkloadSpec / LayerSpec, never raw BMsemi files.
"""
from dataclasses import dataclass, field, asdict

OPS = {"conv2d", "depthwise_conv2d", "pointwise_conv2d", "linear"}
DATA_SOURCES = {"real_weights", "repo_bundled_vectors", "synthetic_values_real_geometry"}
KINDS = {"conv_net_geometry", "int8_fc_network", "crossbar_conv"}


@dataclass
class LayerSpec:
    name: str
    op: str
    K: int                      # output channels / features
    C: int                      # input channels / features
    R: int = 1
    S: int = 1
    Y: int = 1                  # OUTPUT height
    X: int = 1                  # OUTPUT width
    stride: int = 1
    padding: int = 0
    groups: int = 1
    dtype: str = "int8"
    extra: dict = field(default_factory=dict)


@dataclass
class WorkloadSpec:
    id: str
    title: str
    source_model: str           # BMsemi operation or project the workload comes from
    kind: str
    data_source: str
    loader: str                 # which local verifier consumes it
    layers: list
    pe: dict                    # {"rows": .., "cols": ..}
    provenance: dict            # {"repo_commit": .., "files": [{"path","size","sha256_lf"}]}
    stages: list = field(default_factory=list)   # stage ids this workload can exercise
    notes: str = ""

    def to_dict(self):
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d):
        errs = validate_workload(d)
        if errs:
            raise ValueError("invalid workload: " + "; ".join(errs))
        d = dict(d); d["layers"] = [LayerSpec(**l) for l in d["layers"]]
        return WorkloadSpec(**d)


def validate_workload(d):
    """Return a list of problems (empty list = valid). Hand-written: no extra dependencies."""
    errs = []
    if not isinstance(d, dict):
        return ["workload must be an object"]
    for k in ("id", "title", "source_model", "kind", "data_source", "loader", "layers", "pe", "provenance"):
        if k not in d:
            errs.append(f"missing field '{k}'")
    if errs:
        return errs
    if not str(d["id"]).replace("_", "").isalnum():
        errs.append("id must be letters, digits, underscore")
    if d["kind"] not in KINDS:
        errs.append(f"kind must be one of {sorted(KINDS)}")
    if d["data_source"] not in DATA_SOURCES:
        errs.append(f"data_source must be one of {sorted(DATA_SOURCES)}")
    pe = d["pe"]
    if not (isinstance(pe, dict) and all(isinstance(pe.get(k), int) and pe[k] > 0 for k in ("rows", "cols"))):
        errs.append("pe must be {'rows': int>0, 'cols': int>0}")
    prov = d["provenance"]
    if not (isinstance(prov, dict) and prov.get("repo_commit") and isinstance(prov.get("files"), list) and prov["files"]):
        errs.append("provenance needs repo_commit and a non-empty files list")
    if not isinstance(d["layers"], list) or not d["layers"]:
        errs.append("layers must be a non-empty list"); return errs
    names = set()
    for i, l in enumerate(d["layers"]):
        w = f"layer[{i}]"
        if not isinstance(l, dict):
            errs.append(f"{w}: must be an object"); continue
        miss = [k for k in ("name", "op", "K", "C") if k not in l]
        if miss:
            errs.append(f"{w}: missing {miss}"); continue
        if l["name"] in names:
            errs.append(f"{w}: duplicate name '{l['name']}'")
        names.add(l["name"])
        if l["op"] not in OPS:
            errs.append(f"{w}: op must be one of {sorted(OPS)}")
        for k in ("K", "C", "R", "S", "Y", "X", "stride", "groups"):
            v = l.get(k, 1)
            if not (isinstance(v, int) and v >= 1):
                errs.append(f"{w}: {k} must be an int >= 1")
        if not (isinstance(l.get("padding", 0), int) and l.get("padding", 0) >= 0):
            errs.append(f"{w}: padding must be an int >= 0")
        g = l.get("groups", 1)
        if isinstance(g, int) and g >= 1 and isinstance(l.get("K"), int) and isinstance(l.get("C"), int):
            if l["C"] % g or l["K"] % g:
                errs.append(f"{w}: groups={g} must divide both C={l['C']} and K={l['K']}")
        if l.get("op") == "depthwise_conv2d" and g != l.get("C"):
            errs.append(f"{w}: depthwise_conv2d needs groups == C")
    return errs
