"""IR validation = shape inference + structural checks. Returns (errors, warnings); an empty errors list means valid."""
from compiler.ir import MATRIX_OPS, OPS


def _conv_out(h, w, a):
    return ((h + 2 * a["padding"] - a["R"]) // a["stride"] + 1, (w + 2 * a["padding"] - a["S"]) // a["stride"] + 1)


def validate_ir(g):
    errors, warnings = [], []
    by = {}
    for n in g.nodes:
        if n.name in by:
            errors.append(f"duplicate node name '{n.name}'")
        by[n.name] = n
        if n.op not in OPS:
            errors.append(f"{n.name}: unknown op '{n.op}'")
    seen = set()
    for n in g.nodes:                                         # producer-before-consumer order
        for i in n.inputs:
            if i not in by:
                errors.append(f"{n.name}: unknown input '{i}'")
            elif i not in seen:
                errors.append(f"{n.name}: input '{i}' appears after its consumer (not topologically ordered)")
        seen.add(n.name)
    if sum(n.op == "input" for n in g.nodes) != 1:
        errors.append("expected exactly one input node")
    if sum(n.op == "output" for n in g.nodes) < 1:
        errors.append("no output node")
    for n in g.nodes:
        if not n.out.shape or any((not isinstance(d, int)) or d <= 0 for d in n.out.shape):
            errors.append(f"{n.name}: invalid shape {n.out.shape}")
            continue
        ins = [by[i] for i in n.inputs if i in by]
        a, s = n.attrs, n.out.shape
        if n.op == "input":
            if n.inputs:
                errors.append(f"{n.name}: input node must not have inputs")
        elif n.op in ("output", "shape_preserving", "flatten", "pool") or n.op in MATRIX_OPS:
            if len(ins) != 1:
                errors.append(f"{n.name}: {n.op} needs exactly 1 input, has {len(n.inputs)}"); continue
            ish = ins[0].out.shape
            if n.op in ("output", "shape_preserving") and ish != s:
                errors.append(f"{n.name}: {n.op} must preserve shape, input {ish} -> output {s}")
            elif n.op == "flatten":
                prod = 1
                for d in ish[1:]:
                    prod *= d
                if s != (ish[0], prod):
                    errors.append(f"{n.name}: flatten of {ish} must give {(ish[0], prod)}, declared {s}")
            elif n.op == "pool":
                if len(ish) != 4 or len(s) != 4 or ish[:2] != s[:2] or s[2] > ish[2] or s[3] > ish[3]:
                    errors.append(f"{n.name}: pool must keep N,C and not grow H,W: {ish} -> {s}")
            elif n.op in MATRIX_OPS:
                if n.op == "linear":
                    if len(ish) != 2 or ish[1] != a["C"] or s != (ish[0], a["K"]):
                        errors.append(f"{n.name}: linear C={a['C']} K={a['K']} does not fit input {ish} -> output {s}")
                else:
                    if len(ish) != 4 or ish[1] != a["C"]:
                        errors.append(f"{n.name}: conv expects input channels C={a['C']}, input shape is {ish}"); continue
                    y, x = _conv_out(ish[2], ish[3], a)
                    if s != (ish[0], a["K"], y, x):
                        errors.append(f"{n.name}: shape inference gives {(ish[0], a['K'], y, x)} but the artifact declares {s}")
                    if a["C"] % a["groups"] or a["K"] % a["groups"]:
                        errors.append(f"{n.name}: groups={a['groups']} must divide C and K")
                    if n.op == "depthwise_conv2d" and not (a["groups"] == a["C"] == a["K"]):
                        errors.append(f"{n.name}: depthwise needs groups == C == K")
        elif n.op == "add":
            if len(ins) < 2:
                errors.append(f"{n.name}: add needs >= 2 inputs, has {len(ins)}")
            elif any(i.out.shape != s for i in ins):
                errors.append(f"{n.name}: add inputs {[i.out.shape for i in ins]} must all equal output {s}")
    unused = [n.name for n in g.nodes if n.op not in ("output",) and not any(n.name in m.inputs for m in g.nodes)]
    if unused:
        warnings.append(f"nodes with no consumer: {unused[:5]}")
    unknown_kinds = sum(1 for n in g.nodes if n.op == "shape_preserving" and n.origin.get("caveat"))
    if unknown_kinds:
        warnings.append(f"{unknown_kinds} node(s) classified by shape behaviour only (module kind not recorded in the artifact)")
    return errors, warnings
