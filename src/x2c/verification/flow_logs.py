"""
Offline part of the FABulous runner: turn tool logs and output files into structured findings, and refuse to trust
a flow just because it exited with 0. Deterministic and read-only; needs no FABulous installation.

Why: a real FABulous run has "succeeded" (exit 0, "Compile flow completed successfully") with FABULOUS_LC 0/672 and
an all-zero bitstream. Status is therefore decided from the outputs, never from the exit code alone:
    failed        a tool error, non-zero exit, capacity or timing failure, or a malformed bitstream
    suspect       the flow finished but produced nothing (empty FASM, no logic cells, all-zero bitstream)
    inconclusive  nothing is wrong, but the exit code, FASM or bitstream was not available to check
    ok            exit 0, and the FASM and bitstream were both checked and are non-empty

Format evidence (see FORMAT_EVIDENCE):
    REAL     nextpnr utilisation table, "Max frequency ... PASS", "Program finished normally." (two real logs, nextpnr 0.10-82;
             tests/fixtures/real_flow_run is a fresh run from 2026-11-10 on Ubuntu/WSL2 with FABulous 2.2.0)
    NOTES    "FASM file contains no features", "Compile flow completed successfully" (recorded from real runs; logs since lost)
    ASSUMED  Yosys ERROR wording, nextpnr ERROR wording, "... FAIL at ..." timing lines. No real failing log is available yet.
             Any line starting with ERROR: is still caught; an unrecognised wording only loses its specific code.
"""
import re

from x2c.verification.bitstream_audit import audit_structure, parse_fasm

EVIDENCE_LEVEL = "tool_log_and_output_file_inspection (NOT RTL simulation, NOT silicon)"
FORMAT_EVIDENCE = {
    "real_log": ["nextpnr utilisation table", "nextpnr Max frequency PASS line", "nextpnr normal-finish line"],
    "from_notes_of_real_runs": ["FASM file contains no features", "Compile flow completed successfully"],
    "assumed_no_real_sample_yet": ["Yosys ERROR wording", "nextpnr ERROR wording", "nextpnr timing FAIL wording"],
}
LOGIC_CELL = "FABULOUS_LC"
MAX_WARNINGS = 50                # per log: a noisy tool must not bury the errors in the report

_PREFIX = r"^(?:\S+:\d+:\s*)?"
_ERROR = re.compile(_PREFIX + r"ERROR:\s*(.*)$")
_WARNING = re.compile(_PREFIX + r"Warning:\s*(.*)$")
_SOURCE = re.compile(r"(\S+\.(?:v|sv|vh|svh)):(\d+)")
_UNRESOLVED = re.compile(r"Module [`']?\\?([^'`\s]+)'? referenced")
_UTILISATION = re.compile(r"^Info:\s+(\S+):\s+(\d+)/\s*(\d+)\s+\d+%\s*$")
_FMAX = re.compile(r"^\w+: Max frequency for clock '(.+?)':\s+([\d.]+) MHz \((PASS|FAIL) at ([\d.]+) MHz\)")


def _finding(stage, severity, code, message, line_no=None, line=None, **extra):
    evidence = [{"line": line_no, "text": line.strip()[:300]}] if line is not None else []
    return {"stage": stage, "severity": severity, "code": code, "message": message[:300], "evidence": evidence, **extra}


def _scan_warnings(stage, lines):
    found = [(n, m.group(1), raw) for n, raw in lines for m in [_WARNING.match(raw)] if m]
    shown = [_finding(stage, "warning", f"{stage}_warning", msg, n, raw) for n, msg, raw in found[:MAX_WARNINGS]]
    return shown, len(found)


def _numbered(text):
    return list(enumerate((text or "").splitlines(), 1))


# Yosys exits 0 on these, but each one means the netlist does not do what the Verilog author meant. Wording is REAL
# (captured from YoWASP Yosys 0.70 on small test designs); they are reported as "suspect", not "warning".
YOSYS_SUSPECT_WARNINGS = [
    (re.compile(r"Resizing cell port\s"), "yosys_port_width_mismatch", "a port was connected to a signal of a different width"),
    (re.compile(r"Identifier .* is implicitly declared"), "yosys_implicit_wire", "a name was used without being declared (typo?)"),
    (re.compile(r"Wire \S+ is used but has no driver"), "yosys_undriven_wire", "a signal is read but nothing drives it"),
]


def _promote_yosys_warnings(lines):
    out = []
    for n, raw in lines:
        m = _WARNING.match(raw)
        if not m:
            continue
        for pattern, code, meaning in YOSYS_SUSPECT_WARNINGS:
            if pattern.search(m.group(1)):
                out.append(_finding("yosys", "suspect", code, f"{meaning}: {m.group(1)}", n, raw))
                break
    return out


def _is_promoted(finding):
    return any(p.search(finding["message"]) for p, _, _ in YOSYS_SUSPECT_WARNINGS)


def parse_yosys_log(text):
    lines, findings = _numbered(text), []
    for n, raw in lines:
        m = _ERROR.match(raw)
        if not m:
            continue
        msg = m.group(1)
        source = _SOURCE.search(raw)
        where = {"file": source.group(1), "source_line": int(source.group(2))} if source else {}
        missing = _UNRESOLVED.search(msg)
        if "syntax error" in msg.lower():
            findings.append(_finding("yosys", "error", "yosys_syntax_error", msg, n, raw, **where))
        elif missing or "is not part of the design" in msg:
            findings.append(_finding("yosys", "error", "yosys_unresolved_module", msg, n, raw,
                                     module=missing.group(1) if missing else None, **where))
        else:
            findings.append(_finding("yosys", "error", "yosys_error", msg, n, raw, **where))
    warnings, total = _scan_warnings("yosys", lines)
    return {"findings": findings + _promote_yosys_warnings(lines) + [w for w in warnings if not _is_promoted(w)],
            "warnings_total": total}


def parse_nextpnr_log(text):
    lines, findings = _numbered(text), []
    utilisation, fmax, finished = {}, {}, False
    for n, raw in lines:
        util = _UTILISATION.match(raw)
        if util:
            utilisation[util.group(1)] = {"used": int(util.group(2)), "total": int(util.group(3)), "line": n, "text": raw}
            continue
        freq = _FMAX.match(raw)
        if freq:                      # the log repeats this line: the last one is the post-routing result
            fmax[freq.group(1)] = {"fmax_mhz": float(freq.group(2)), "ok": freq.group(3) == "PASS",
                                   "target_mhz": float(freq.group(4)), "line": n, "text": raw}
            continue
        if "Program finished normally" in raw:
            finished = True
        err = _ERROR.match(raw)
        if err:
            low = err.group(1).lower()
            code = "placement_failed" if "plac" in low else "routing_failed" if "rout" in low else "nextpnr_error"
            findings.append(_finding("nextpnr", "error", code, err.group(1), n, raw))
    for bel, u in utilisation.items():
        if u["used"] > u["total"]:
            findings.append(_finding("nextpnr", "error", "capacity_exceeded",
                                     f"{bel} needs {u['used']} but the fabric has {u['total']}", u["line"], u["text"], bel=bel))
    for clock, f in fmax.items():
        if not f["ok"]:
            findings.append(_finding("nextpnr", "error", "timing_violation",
                                     f"clock {clock}: {f['fmax_mhz']} MHz is below the {f['target_mhz']} MHz target", f["line"], f["text"], clock=clock))
    warnings, total = _scan_warnings("nextpnr", lines)
    return {"findings": findings + warnings, "warnings_total": total, "finished_normally": finished,
            "utilisation": {k: {"used": v["used"], "total": v["total"]} for k, v in utilisation.items()},
            "fmax": {k: {x: v[x] for x in ("fmax_mhz", "target_mhz", "ok")} for k, v in fmax.items()}, "_util_raw": utilisation}


def _empty_design_findings(pnr):
    util, out = pnr["_util_raw"], []
    lc = util.get(LOGIC_CELL)
    if lc and lc["used"] == 0:
        out.append(_finding("nextpnr", "suspect", "zero_logic_cells",
                            f"{LOGIC_CELL} 0/{lc['total']}: no user logic was placed", lc["line"], lc["text"]))
    if util and all(u["used"] == 0 for u in util.values()):
        out.append(_finding("nextpnr", "suspect", "zero_utilisation", "nothing at all was placed on the fabric"))
    return out


def check_flow(exit_code=None, yosys_log=None, nextpnr_log=None, flow_log=None, fasm_text=None, bitstream=None, profile=None):
    """Combine everything known about one flow run into a single status. Missing inputs are listed, never assumed good."""
    findings, not_checked, facts = [], [], {"tool_claimed_success": False}
    if exit_code is None:
        not_checked.append("exit_code")
    elif exit_code != 0:
        findings.append(_finding("flow", "error", "nonzero_exit", f"the flow exited with code {exit_code}"))

    if yosys_log is None:
        not_checked.append("yosys_log")
    else:
        y = parse_yosys_log(yosys_log)
        findings += y["findings"]; facts["yosys_warnings_total"] = y["warnings_total"]

    if nextpnr_log is None:
        not_checked.append("nextpnr_log")
    else:
        p = parse_nextpnr_log(nextpnr_log)
        findings += p["findings"] + _empty_design_findings(p)
        facts.update(utilisation=p["utilisation"], fmax=p["fmax"], nextpnr_finished_normally=p["finished_normally"],
                     nextpnr_warnings_total=p["warnings_total"])
        if not p["finished_normally"] and not any(f["stage"] == "nextpnr" and f["severity"] == "error" for f in findings):
            findings.append(_finding("nextpnr", "suspect", "nextpnr_log_incomplete", "the log has no normal-finish line and no error"))

    if flow_log is not None:
        for n, raw in _numbered(flow_log):
            if "Compile flow completed successfully" in raw:
                facts["tool_claimed_success"] = True
            if "fasm file contains no features" in raw.lower():
                findings.append(_finding("bitgen", "suspect", "bitgen_no_features", "the bitstream generator reported an empty FASM", n, raw))

    if fasm_text is None:
        not_checked.append("fasm")
    else:
        features, unparsed = parse_fasm(fasm_text)
        facts["fasm_features"] = len(features)
        if not features:
            findings.append(_finding("fasm", "suspect", "empty_fasm", "the FASM contains no configuration features"))
        if unparsed:
            findings.append(_finding("fasm", "warning", "fasm_unparsed_lines", f"{len(unparsed)} FASM line(s) could not be parsed", line=unparsed[0]))

    if bitstream is None or profile is None:
        not_checked.append("bitstream")
    else:
        audit = audit_structure(bitstream, profile)
        facts["bitstream_sha256"] = audit["stats"]["sha256"]
        facts["bitstream_set_bits"] = audit["stats"].get("set_bits")
        for c in audit["checks"]:
            if not c["ok"] and c["severity"] == "error":
                findings.append(_finding("bitstream", "error", "bitstream_malformed", f"{c['id']} {c['title']}: {c['detail']}"))
        if audit["stats"].get("set_bits") == 0:
            findings.append(_finding("bitstream", "suspect", "all_zero_bitstream", "every configuration bit is 0: the fabric would be configured to do nothing"))

    severities = {f["severity"] for f in findings}
    required_missing = exit_code is None or "fasm" in not_checked or "bitstream" in not_checked
    status = ("failed" if "error" in severities else "suspect" if "suspect" in severities
              else "inconclusive" if required_missing else "ok")
    return {"stage": "S05/S10 flow check", "status": status, "findings": findings, "not_checked": not_checked, "facts": facts,
            "evidence_level": EVIDENCE_LEVEL, "format_evidence": FORMAT_EVIDENCE,
            "limits": "'ok' means the tools produced a non-empty, well-formed result; it does not prove the design behaves correctly"}
