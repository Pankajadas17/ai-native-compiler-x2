"""
Local typed graph IR (S03). Small, explicit and JSON-serialisable.

Types: a tensor has a shape and a dtype. The compiler artifact (compiler_graph.json) records shapes but NOT dtypes,
so dtype is "unspecified" unless a source states it; this is never silently guessed.
Node kinds (op):
  input | output | add | flatten | pool
  conv2d | depthwise_conv2d | pointwise_conv2d | linear      (matrix ops: candidates for the X2 crossbar)
  shape_preserving                                           (norm / activation / unknown module whose kind the
                                                              artifact does not record; only shape behaviour is checked)
"""
import hashlib
import json
from dataclasses import dataclass, field

MATRIX_OPS = ("conv2d", "depthwise_conv2d", "pointwise_conv2d", "linear")
OPS = MATRIX_OPS + ("input", "output", "add", "flatten", "pool", "shape_preserving")


@dataclass(frozen=True)
class TensorType:
    shape: tuple
    dtype: str = "unspecified"

    def numel(self):
        n = 1
        for d in self.shape:
            n *= d
        return n


@dataclass
class IRNode:
    name: str
    op: str
    inputs: list                       # names of producer nodes, in order
    out: TensorType
    attrs: dict = field(default_factory=dict)
    origin: dict = field(default_factory=dict)   # where this node came from (artifact, fx target, caveats)


@dataclass
class IRGraph:
    name: str
    nodes: list                        # in producer-before-consumer order
    meta: dict = field(default_factory=dict)     # provenance + explicit assumptions

    def by_name(self):
        return {n.name: n for n in self.nodes}

    def consumers(self):
        c = {n.name: [] for n in self.nodes}
        for n in self.nodes:
            for i in n.inputs:
                if i in c:
                    c[i].append(n.name)
        return c

    def to_dict(self):
        return {"name": self.name, "meta": self.meta,
                "nodes": [{"name": n.name, "op": n.op, "inputs": n.inputs, "shape": list(n.out.shape),
                           "dtype": n.out.dtype, "attrs": n.attrs, "origin": n.origin} for n in self.nodes]}

    @staticmethod
    def from_dict(d):
        return IRGraph(d["name"], [IRNode(x["name"], x["op"], list(x["inputs"]), TensorType(tuple(x["shape"]), x.get("dtype", "unspecified")),
                                          dict(x.get("attrs", {})), dict(x.get("origin", {}))) for x in d["nodes"]], dict(d.get("meta", {})))

    def digest(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, default=str).encode()).hexdigest()
