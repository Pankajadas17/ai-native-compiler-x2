"""
Deterministic debugging knowledge base for the verification flow.
Each CAUSE says which failure categories it can produce, what to suspect, what to fix and what to re-check.
Used (a) to give the debug agent a menu of causes, (b) as a rule-based baseline, and (c) to VALIDATE
the LLM's answer: a cause is "supported" only if it is consistent with the categories actually observed.
"""
CAUSES = {
    "output_sign_inverted": {
        "categories": {"sign_inverted"},
        "suspect": "sign convention of X2 outputs (differential +/- column polarity) in the adapter or simulator",
        "fix": "check which column is subtracted from which; compare a single 1x1 case",
        "next_check": "run_case('fixed_2x2')"},
    "constant_offset": {
        "categories": {"constant_offset"},
        "suspect": "a constant bias/offset added to every output (bias or zero-point applied twice, or stale accumulator)",
        "fix": "inspect bias / zero-point handling and accumulator reset between passes",
        "next_check": "run_case('zeros') - a zero input should give a zero output"},
    "output_scaled": {
        "categories": {"scaled_output"},
        "suspect": "a scale factor applied to every output (double accumulation, wrong normalisation or a repeated pass)",
        "fix": "check whether each row/tile is accumulated or normalised more than once",
        "next_check": "run_case('identity')"},
    "output_transposed": {
        "categories": {"transposed_output", "shape_mismatch"},
        "suspect": "output matrix returned as (N, M) instead of (M, N): result orientation in the adapter",
        "fix": "transpose the result back / fix how rows of A are stacked",
        "next_check": "run_case('non_square')"},
    "weight_orientation_transposed": {
        "categories": {"interface_error", "transposed_output"},
        "suspect": "weights passed to the simulator as B.T instead of B (rows=K inputs, cols=N outputs expected); typical evidence: an error like 'incompatible shapes (m, k) @ (n, k)' where the inner dimension matches the SECOND dimension of the weight matrix",
        "fix": "pass B with shape (K, N) directly; remove any transpose in emulators/x2_adapter.py",
        "next_check": "run_case('non_square') - non-square shapes expose orientation bugs immediately"},
    "adapter_interface_mismatch": {
        "categories": {"interface_error"},
        "suspect": "argument/shape contract between the adapter and the simulator violated in a way NOT explained by a transposed weight matrix (e.g. wrong vector length or shapes that fit neither orientation)",
        "fix": "read the exception text; check rows/cols and input vector length against the weight matrix",
        "next_check": "run_case with a tiny custom A and B"},
    "mapper_dropped_fold": {
        "only_for": "workload reports that contain mapping_check pass counts (NEVER for plain matmul reports)",
        "categories": {"numerical_mismatch", "zero_output"},
        "suspect": "tiling plan misses a row fold (input rows never multiplied): fewer X2 passes than the mapper's mvm_ops; on a layer with a single fold the whole output is zero",
        "fix": "check that fold_sizes sum to patch_len and every fold is executed",
        "next_check": "compare passes_executed with mvm_ops_from_mapper in the failing report"},
    "mapper_dropped_output_channel": {
        "only_for": "workload reports that contain mapping_check pass counts (NEVER for plain matmul reports)",
        "categories": {"localized_mismatch"},
        "suspect": "a k-chunk boundary excludes output channel(s): only some output rows differ, pass count unchanged",
        "fix": "check k_chunks cover [0, K) without gaps; the failing rows identify the chunk",
        "next_check": "list k_chunks from map_layer and find the chunk containing the failing channels"},
}

CAUSE_IDS = sorted(CAUSES) + ["unknown"]

import re

_SHAPES_RE = re.compile(r"incompatible shapes \((\d+), (\d+)\) @ \((\d+), (\d+)\)")
_DIM_RE = re.compile(r"weights \((\d+), (\d+)\), vec \((\d+),\)")


def transposed_weight_evidence(error_text):
    """Does an interface error look like 'B was passed as B.T'?  True / False / None (cannot tell).
    (m,k) @ (r,c): the inner dimension k fits the SECOND dim c but not the first r  =>  transposed weights.
    weights (r,c) with vec (v,): v fits c but not r  =>  transposed weights."""
    m = _SHAPES_RE.search(error_text or "")
    if m:
        k, r, c = int(m.group(2)), int(m.group(3)), int(m.group(4))
        return (k == c) and (k != r)
    m = _DIM_RE.search(error_text or "")
    if m:
        r, c, v = (int(x) for x in m.groups())
        return (v == c) and (v != r)
    return None


def rule_based_candidates(failures):
    """failures: list of dicts with 'diagnosis' {'category'}, optional 'mapping_check'. Returns ranked cause ids.
    A cause is proposed only when its SPECIFIC evidence exists; with no match the result is [] (= 'unknown')."""
    votes = {}
    for f in failures:
        cat = f["diagnosis"]["category"]
        mc = f.get("mapping_check") or {}
        passes, mapper = mc.get("passes_executed"), mc.get("mvm_ops_from_mapper")
        for cause, info in CAUSES.items():
            if cat not in info["categories"]:
                continue
            if cause.startswith("mapper_") and not mc:
                continue                                   # mapper causes need a workload report with pass counts
            if cause == "mapper_dropped_fold" and not (passes is not None and mapper is not None and passes < mapper):
                continue                                   # needs the pass-count evidence
            if cause == "mapper_dropped_output_channel" and passes != mapper:
                continue                                   # a dropped chunk keeps the pass count
            if cat == "interface_error" and cause in ("weight_orientation_transposed", "adapter_interface_mismatch"):
                ev = transposed_weight_evidence(f.get("error", ""))
                if ev is True and cause == "adapter_interface_mismatch":
                    continue                                   # the specific cause explains it: do not also vote for the vague one
                if ev is False and cause == "weight_orientation_transposed":
                    continue                                   # shapes fit neither orientation: not a simple transpose
            if cause == "weight_orientation_transposed" and cat == "transposed_output":
                votes[cause] = votes.get(cause, 0) + 0.25   # weak: a lone square case can look transposed
                continue
            votes[cause] = votes.get(cause, 0) + 1
    return [c for c, _ in sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))]
