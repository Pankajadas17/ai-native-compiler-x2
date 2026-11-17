"""
Evidence and root causes for a chain run that did not pass (run_fab_chain.py): the deterministic half of the debug agent (A3) for the FABulous chain.

Why: when the chain says FAIL / NOT VERIFIED / NOT RUN, the stage line only says WHAT happened ("500 mismatches"). Finding WHY took several manual
steps on real runs (an empty bitstream copy in Ubuntu made every fabric output unknown). This module collects the facts a person would look at
(stage results, the job folder, the first mismatch lines of the simulation log) and lists the root causes those facts support.
ML analogy: this is the feature extractor plus a rule baseline; the LLM (agents/chain_debug_agent.py) may only pick a label that these rules support.

Every cause below was either SEEN on a real run of this project or follows directly from a message the chain itself prints; `seen` says which.
Evidence level: reading of chain reports and logs (it says nothing new about the design, and nothing about timing or silicon).
"""
import json
import re
from pathlib import Path

REPORT = Path("reports/fab_chain_report.json")
JOBS = Path("reports/fab_jobs")
MISMATCH = re.compile(r"^FAB_SIM MISMATCH vector \d+: io_in=0x[0-9A-Fa-f]+ expected=0x([0-9A-Fa-fxXzZ]+) fabric=0x([0-9A-Fa-fxXzZ]+)\s*$", re.M)
RESULT = re.compile(r"^FAB_SIM RESULT: (PASS|FAIL) vectors=(\d+) numpy_mismatches=(\d+) rtl_mismatches=(\d+)\s*$", re.M)

CAUSES = {
    "llm_key_missing": {
        "suspect": "the RTL agent (A2) could not run: the API key is not set in this terminal window",
        "fix": "set the key in this window with Read-Host -AsSecureString (never paste it into chat or code), or run with --llm mock",
        "seen": "message printed by the chain"},
    "yosys_missing": {
        "suspect": "Yosys was not found by this Python, so the operator's Verilog could not be checked at all (it was NOT judged wrong)",
        "fix": "activate the project venv (it has yowasp-yosys), or pip install yowasp-yosys in it; then run the chain again",
        "seen": "real run (bm_verif sandbox, 2026-11-13): the chain was started with a Python without yowasp-yosys"},
    "rtl_not_verified": {
        "suspect": "the operator's Verilog did not pass the Yosys-vs-NumPy check, so nothing later can be trusted",
        "fix": "regenerate it: python run_fab_chain.py --op <op> --regen-rtl (with a real --llm), then read the mismatches it reports",
        "seen": "message printed by the chain"},
    "windows_memory_low": {
        "suspect": "Windows had too little commit headroom for Ubuntu's VM, so the stage was not started",
        "fix": "close browsers and other big programs, run `wsl --shutdown`, run the chain again (or --ignore-memory to try anyway)",
        "seen": "real runs: Ubuntu stopped at once with 1.5 GB of commit headroom"},
    "ubuntu_not_answering": {
        "suspect": "Ubuntu (WSL) did not answer or its tools were not found",
        "fix": "run `wsl --shutdown`, then `wsl -d Ubuntu -- echo hello`; if it still fails run python run_wsl_probe.py, which prints what wsl.exe says",
        "seen": "real runs: Ubuntu stopped answering after a crash"},
    "pins_not_connected": {
        "suspect": "the design's pins were not connected in FABulous' top wrapper, so the logic was removed or the flow stopped",
        "fix": "check that the wrapper module is the FIRST module in <op>_fab.v and that tools/fab_connect.py ran (see flow_console.log)",
        "seen": "real run: FABulous 2.2.0 left .io_in() empty and Yosys deleted all logic (0 logic cells)"},
    "timing_not_met": {
        "suspect": "the design is slower than the clock target: nextpnr's delay model gives a lower maximum frequency, so it wrote no bitstream",
        "fix": "run with a lower target (--freq-mhz N) and report the result for that target, or split the logic over more clock ticks; "
               "ask the team which clock the eFPGA side must meet",
        "seen": "real run: requant_i32_i8 reached about 7 MHz against the 12 MHz default"},
    "flow_tool_error": {
        "suspect": "FABulous, Yosys or nextpnr stopped with an error during the flow",
        "fix": "read the last lines of reports/fab_jobs/<op>/flow_console.log; fix that error, then --from-stage flow",
        "seen": "message printed by the chain"},
    "ubuntu_died_during_run": {
        "suspect": "Ubuntu stopped while the flow or the simulation was running (no result was written)",
        "fix": "close big programs, `wsl --shutdown`, then run again from the stage that stopped; python run_wsl_probe.py measures memory during a run",
        "seen": "real runs: the simulation compile needs about 2 GB inside Ubuntu"},
    "bitstream_not_loaded": {
        "suspect": "the fabric was simulated WITHOUT a valid bitstream (an empty or wrong bitstream file), so every output is unknown ('x')",
        "fix": "compare the bitstream in the job folder with the copy in Ubuntu (size, checksum); run again with --from-stage flow to rebuild both",
        "seen": "real run: a crash left a 0-byte .bin in Ubuntu; 500 of 500 vectors mismatched with fabric=0xxx"},
    "fabric_differs_from_rtl": {
        "suspect": "the configured fabric computes defined but wrong values: the flow (synthesis, place and route, pin mapping) did not preserve the design",
        "fix": "look at the first mismatch lines in sim_console.log, then at the pin mapping (io_in/io_out bits) and the nextpnr log",
        "seen": "not seen on a real run yet; follows from the testbench's two comparisons"},
    "golden_convention_mismatch": {
        "suspect": "the fabric matches the RTL but both differ from the NumPy golden model: the operator convention (ASSUMED) is not the same in both",
        "fix": "compare the op convention in verification/rtl_check.py (saturation, rounding) with the Verilog; this is a specification question for the team",
        "seen": "not seen on a real run yet; follows from the testbench's two comparisons"},
}
CAUSE_IDS = set(CAUSES) | {"unknown"}


def _read(path, limit=400_000):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError:
        return None


def load_result(op, report_path=REPORT):
    """The chain result for one operator from the report; None when the report or the operator is missing."""
    text = _read(report_path, 5_000_000)
    if text is None:
        return None
    try:
        rep = json.loads(text)
    except ValueError:
        return None
    return next((r for r in rep.get("results", []) if r.get("op") == op), None)


def collect_evidence(op, result, jobs_dir=JOBS):
    """Plain facts about one chain result. Nothing here is a conclusion: only what the report and the job folder show."""
    job = Path(jobs_dir) / op
    stages = {r["stage"]: {"status": r["status"], "detail": str(r.get("detail", ""))[:600]} for r in (result or {}).get("stages", [])}
    first_bad = next((s for s, r in stages.items() if r["status"] in ("fail", "blocked", "unverified")), None)
    detail = stages[first_bad]["detail"] if first_bad else ""
    binp = job / f"{op}_fab.bin"
    sim_log = _read(job / "sim_console.log") or ""
    res = RESULT.findall(sim_log)
    lines = MISMATCH.findall(sim_log)
    unknown = [bool(re.search(r"[xXzZ]", got)) for _, got in lines]
    exit_code = (_read(job / "exit_code.txt") or "").strip() or None
    ev = {
        "op": op, "verdict": (result or {}).get("verdict"), "stages": stages, "first_bad_stage": first_bad,
        "first_bad_status": stages[first_bad]["status"] if first_bad else None, "first_bad_detail": detail,
        "job_bitstream_bytes": binp.stat().st_size if binp.exists() else None,
        "flow_exit_code": exit_code,
        "sim_result_lines": len(res),
        "sim_vectors": int(res[0][1]) if len(res) == 1 else None,
        "sim_numpy_mismatches": int(res[0][2]) if len(res) == 1 else None,
        "sim_rtl_mismatches": int(res[0][3]) if len(res) == 1 else None,
        "sim_mismatch_lines_shown": len(lines),
        "sim_fabric_outputs_unknown": bool(lines) and all(unknown),
        "sim_fabric_outputs_defined": bool(lines) and not any(unknown),
        "sim_first_mismatches": [f"expected=0x{e} fabric=0x{g}" for e, g in lines[:3]],
        "bitstream_replaced_note": "FAB_SIM NOTE:" in sim_log or "NOTE: the bitstream copy" in detail,
    }
    return ev


def rule_candidates(ev):
    """Cause ids the evidence supports, most specific first. An empty list means: the rules do not know."""
    out, stage, status, d = [], ev.get("first_bad_stage"), ev.get("first_bad_status"), (ev.get("first_bad_detail") or "")
    low = d.lower()

    def add(c):
        if c not in out:
            out.append(c)
    if stage is None:
        return out
    if status == "blocked" and "too little for ubuntu" in low:
        add("windows_memory_low")
    if stage == "rtl":
        add("yosys_missing" if "yosys not found" in low else
            "llm_key_missing" if status == "blocked" and "is not set" in low else "rtl_not_verified")
    if stage == "env" and status == "blocked" and "too little for ubuntu" not in low:
        add("ubuntu_not_answering")
    if stage == "flow":
        if "could not connect the pins" in low or ev.get("flow_exit_code") == "97":
            add("pins_not_connected")
        elif "timing not met" in low:
            add("timing_not_met")
        elif "did not finish" in low or "timed out" in low or (status == "blocked" and "too little for ubuntu" not in low):
            add("ubuntu_died_during_run")
        elif status == "fail":
            add("flow_tool_error")
    if stage == "sim":
        n, nm, rm = ev.get("sim_vectors"), ev.get("sim_numpy_mismatches"), ev.get("sim_rtl_mismatches")
        if ev.get("job_bitstream_bytes") == 0 or (status == "fail" and nm and n and nm == n and ev.get("sim_fabric_outputs_unknown")):
            add("bitstream_not_loaded")
        elif status == "fail" and nm and rm == 0:
            add("golden_convention_mismatch")
        elif status == "fail" and nm and rm and ev.get("sim_fabric_outputs_defined"):
            add("fabric_differs_from_rtl")
        elif status in ("unverified", "fail") and ev.get("sim_result_lines") == 0 and "too little for ubuntu" not in low:
            add("ubuntu_died_during_run")
    if stage == "check" and "empty" in low and "bitstream" in low:
        add("pins_not_connected")
    return out


def decide(parsed, candidates, llm_error=None):
    """The SYSTEM's answer, decided by code: a validated LLM answer, else the rules, else unknown. Returns (cause, source, checks)."""
    checks = {"json_ok": isinstance(parsed, dict), "cause_known": False, "supported_by_evidence": False}
    said = parsed.get("cause_id") if isinstance(parsed, dict) else None
    if checks["json_ok"]:
        checks["cause_known"] = said in CAUSE_IDS
        checks["supported_by_evidence"] = (said == "unknown") or (said in candidates)
    ok = all(checks.values())
    why = "llm said unknown" if said == "unknown" else "llm claim rejected" if checks["json_ok"] else "llm unavailable" if llm_error else "no usable llm answer"
    if ok and said != "unknown":
        return said, "llm_validated", checks
    if ok and not candidates:
        return "unknown", "llm_unknown_confirmed_by_rules", checks
    if candidates:
        return candidates[0], f"rules_fallback ({why})", checks
    return "unknown", f"unknown_fallback ({why})", checks


# ====================================================================== the chain on the BM Labs chip (run_fab_chain.py --target chip)
# Same contract as above: plain facts from the report and the job/build folders, then rules; the LLM may only pick a supported cause.
CHIP_REPORT = Path("reports/chip_chain_report.json")
CHIP_JOBS = Path("reports/chip_jobs")
CHIP_BUILD = Path("chip_build")
CHIP_MULTIPLIER_MAP = "MACC_8x8_20_map_signed.v"

CHIP_CAUSES = {
    "chip_tools_missing": {
        "suspect": "Ubuntu is missing part of the chip toolchain (OSS CAD Suite tools, ~/chipenv with fasm and numpy, the chip repo or profile)",
        "fix": "install what the env stage lists as MISSING; tools elsewhere: set CHIP_CAD / CHIP_PY in Ubuntu; the profile: python3 run_chip_profile.py",
        "seen": "real preflight output (bm_verif sandbox, 2026-11-13: chipenv without fasm)"},
    "chip_python_module_missing": {
        "suspect": "the repo Makefile ran bit_gen.py with a python3 that lacks fasm (the OSS CAD Suite one came first on PATH)",
        "fix": "put ~/chipenv/bin first on PATH (the chain's stage scripts do since fix 1); by hand: activate ~/chipenv after the CAD suite export",
        "seen": "real run (your machine, 2026-11-13): every operator failed at 'bitstream'; reproduced: ModuleNotFoundError: No module named 'fasm'"},
    "chip_multiplier_setting_changed": {
        "suspect": "the repo's MULTIPLIER_MAP_PATH line changed, so chip_flow refused to override it with the corrected signed-multiplier map",
        "fix": "compare user_designs/makefile.include l.26 with chip/techmap/MACC_8x8_20_map_signed.v; re-verify (tests/test_macc_signed_map.py) "
               "before updating REPO_MULTIPLIER_LINE in chip/chip_flow.py",
        "seen": "follows from chip_flow's own check (tests/test_macc_signed_map.py); not seen on a real run"},
    "chip_pins_exceeded": {
        "suspect": "the design needs more pins than the chip's 36 IOBUF sites",
        "fix": "fewer or narrower ports: load wide operands over several ticks (as the two-step wrapper does) or keep data on the fabric",
        "seen": "real (chip example bitslice_acc_parallel, pin pre-check)"},
    "chip_over_capacity": {
        "suspect": "the design uses more of a resource than the chip has (logic cells, 18 MACC blocks, RAMs, 2 X2 blocks)",
        "fix": "read the utilisation in the flow report; share or time-multiplex the resource, or split the design",
        "seen": "real nextpnr log (chip example macc_over: 20 multipliers on 18 MACC blocks)"},
    "chip_synth_error": {
        "suspect": "Yosys stopped on the design (syntax error, missing module, or a flip-flop type the fabric does not have)",
        "fix": "read the last lines of chip_build/chain_<op>/synth.log, fix the Verilog, run again from --from-stage rtl",
        "seen": "real Yosys logs (chip examples bitslice_acc_bad, missing_mod)"},
    "chip_bitstream_audit_failed": {
        "suspect": "the bitstream is not what the FASM asks for (V0 audit) or has the wrong size",
        "fix": "read bitstream_audit.json in chip_build/chain_<op>/; rebuild from --from-stage flow; if it repeats, report it with the audit file",
        "seen": "real bitstreams with planted bit flips (tests/test_chip_bitstream_audit.py); not on a real run"},
    "chip_flow_crashed": {
        "suspect": "the chip flow did not finish (it crashed, Ubuntu stopped, or it timed out), so no result was written",
        "fix": "read reports/chip_jobs/<op>/flow_console.log (the last lines say why); run again from --from-stage flow",
        "seen": "message printed by the chain"},
    "chip_flow_tool_error": {
        "suspect": "a chip-flow tool (Yosys, nextpnr, bit_gen) stopped with an error that no rule recognises",
        "fix": "read the log the stage detail names (chip_build/chain_<op>/<stage>.log); fix that error, then --from-stage flow",
        "seen": "message printed by the chain"},
    "chip_wrapper_check_failed": {
        "suspect": "the chip-pin wrapper around the verified core does not compute the core's result (or does not fit the pins)",
        "fix": "read the wrapper_check in the chain report; the wrapper is generated by verification/chip_chain.py, so this is a bm_verif bug to report",
        "seen": "message printed by the chain; tests with planted wrapper faults (tests/test_chip_chain.py)"},
    "chip_signed_multiplier": {
        "suspect": "a signed multiply on the hard MACC blocks: the repo's MACC_8x8_20_map.v is unsigned-only, so every negative operand is wrong",
        "fix": "check that chip_build/chain_<op>/Makefile sets MULTIPLIER_MAP_PATH to chip/techmap/MACC_8x8_20_map_signed.v (chip_flow does "
               "this since the signed-multiplier fix); if it does, report the failing vectors",
        "seen": "real run (your machine, 2026-11-13): requant_i32_i8 219/485 wrong, all with negative inputs, 4 MACC blocks"},
    "chip_netlist_outputs_undefined": {
        "suspect": "the synthesized netlist drove no defined value on the result pins (x/z): a pin is undriven or a simulation model is wrong",
        "fix": "check the IOBUF model in use (sim/IOBUF_chip_sim.v, chip polarity: EN=1 drives) and that the result pins are outputs in the netlist",
        "seen": "bm_verif sandbox: with the repo's IOBUF sim model every output floated"},
    "chip_netlist_differs": {
        "suspect": "the synthesized netlist computes defined but wrong values: synthesis did not preserve the design",
        "fix": "simulate the RTL wrapper and the netlist on the first failing vector; compare the techmaps used (synth.log)",
        "seen": "not seen on a real run yet; follows from the netlist simulation's comparison"},
    "chip_sim_not_finished": {
        "suspect": "the netlist simulation did not finish (no DONE line, no log, or a time-out): it hung or crashed",
        "fix": "read reports/chip_jobs/<op>/sim_console.log; a hang usually means a signal never becomes defined (see the IOBUF note)",
        "seen": "bm_verif sandbox: the netlist simulation hung with the repo's IOBUF model"},
    "chip_timing_not_met": {
        "suspect": "nextpnr's delay model gives the design a lower maximum frequency than the 12 MHz target (the repo Makefile still writes a bitstream)",
        "fix": "add a pipeline register on the slowest path (as x2_engine does) or agree a lower clock with the team; timing is a nextpnr estimate",
        "seen": "real build: x2_engine_4bit 11.2 MHz against 12 MHz (run_chip_flow.py)"},
    "chip_repo_modified": {
        "suspect": "the chip repo (sky130-reram-fpga or a submodule) changed while the flow ran",
        "fix": "run git -C sky130-reram-fpga status --short --ignored in Ubuntu; restore the repo; build only through chip_flow (out of the repo)",
        "seen": "message printed by the chain (tree fingerprint before/after)"},
}
CHIP_SHARED = ("yosys_missing", "llm_key_missing", "rtl_not_verified", "ubuntu_not_answering")       # demo causes that mean the same on the chip
CAUSE_IDS = set(CAUSES) | set(CHIP_CAUSES) | {"unknown"}
_SYNTH = {"syntax_error", "unknown_module", "ff_type_unsupported"}
_AUDIT = {"bitstream_audit_failed", "bitstream_size_mismatch", "empty_fasm", "unconstrained_io"}


def cause_info(cause_id):
    """suspect / fix / seen for a demo or chip cause ({} for unknown)."""
    return CAUSES.get(cause_id) or CHIP_CAUSES.get(cause_id) or {}


def chip_menu_ids():
    return list(CHIP_SHARED) + list(CHIP_CAUSES)


def _json(path):
    text = _read(path, 5_000_000)
    try:
        return json.loads(text) if text else None
    except ValueError:
        return None


def collect_chip_evidence(op, result, jobs_dir=CHIP_JOBS, build_dir=CHIP_BUILD):
    """Plain facts about one chip-chain result: stage results, the chip flow report, the console and simulation logs."""
    from verification import chip_chain as cc                       # only needed for chip runs
    job, build = Path(jobs_dir) / op, Path(build_dir) / f"chain_{op}"
    stages = {r["stage"]: {"status": r["status"], "detail": str(r.get("detail", ""))[:900]} for r in (result or {}).get("stages", [])}
    first_bad = next((s for s, r in stages.items() if r["status"] in ("fail", "blocked", "unverified")), None)
    rep = _json(build / "flow_report.json") or {}
    failure = rep.get("failure") or {}
    console = _read(job / "flow_console.log") or ""
    err_lines = [l.strip() for l in console.splitlines() if re.search(r"error|no module named|refused|changed", l, re.I)]
    makefile = _read(build / "Makefile") or ""
    sim_text = _read(job / "sim.log")
    sim = None
    if sim_text is not None and op in cc.OPS:
        try:
            sim = cc.parse_netlist_log(op, sim_text)
        except ValueError:                                            # an operator the chip chain cannot wrap
            sim = None
    bad = (sim or {}).get("all_mismatches") or []
    netlist = _read(job / "netlist.v") or ""
    check = _json(job / "chip_check.json") or {}
    return {
        "op": op, "target": "chip", "verdict": (result or {}).get("verdict"), "stages": stages, "first_bad_stage": first_bad,
        "first_bad_status": stages[first_bad]["status"] if first_bad else None,
        "first_bad_detail": stages[first_bad]["detail"] if first_bad else "",
        "flow_report_status": rep.get("status"), "flow_failure_stage": failure.get("stage"), "flow_failure_cause": failure.get("cause"),
        "flow_utilisation": rep.get("utilisation"),
        "flow_console_error_lines": err_lines[-3:],
        "multiplier_line_changed": "MULTIPLIER_MAP_PATH line" in console,
        "signed_multiplier_map_in_makefile": CHIP_MULTIPLIER_MAP in makefile if makefile else None,
        "sim_status": (sim or {}).get("status"), "sim_reason": (sim or {}).get("reason"), "sim_vectors": (sim or {}).get("vectors"),
        "sim_finished": bool(sim_text) and bool(re.search(r"^DONE \d+\s*$", sim_text, re.M)),
        "sim_mismatches": len(bad), "sim_undefined_outputs": sum(1 for b in bad if b["got"] is None),
        "sim_wrong_all_have_a_negative_input": bool(bad) and all(any(v < 0 for v in b["inputs"].values()) for b in bad),
        "sim_first_mismatches": [f"inputs={b['inputs']} expected={b['expected']} got={b['got']}" for b in bad[:3]],
        "netlist_macc_blocks": len(re.findall(r"^\s*MACC_8x8_20\s*#", netlist, re.M)),
        "timing": (check.get("facts") or {}).get("fmax"),
    }


def chip_rule_candidates(ev):
    """Cause ids the chip evidence supports, most specific first. An empty list means: the rules do not know."""
    out, stage, status, d = [], ev.get("first_bad_stage"), ev.get("first_bad_status"), (ev.get("first_bad_detail") or "")
    low = d.lower()

    def add(c):
        if c not in out:
            out.append(c)
    if stage is None:
        return out
    if stage == "rtl":
        add("yosys_missing" if "yosys not found" in low else
            "llm_key_missing" if status == "blocked" and "is not set" in low else "rtl_not_verified")
    elif stage == "prep":
        add("rtl_not_verified" if "no verified core" in low else "chip_wrapper_check_failed")
    elif stage == "env":
        add("chip_tools_missing" if "ubuntu is missing" in low else "ubuntu_not_answering")
    elif stage == "flow":
        cause = ev.get("flow_failure_cause")
        if status == "blocked":
            add("ubuntu_not_answering")
        elif cause == "python_module_missing" or "no module named" in low or any("no module named" in l.lower() for l in ev.get("flow_console_error_lines") or []):
            add("chip_python_module_missing")
        elif ev.get("multiplier_line_changed"):
            add("chip_multiplier_setting_changed")
        elif cause == "pin_budget_exceeded":
            add("chip_pins_exceeded")
        elif cause == "over_capacity":
            add("chip_over_capacity")
        elif cause in _SYNTH:
            add("chip_synth_error")
        elif cause in _AUDIT:
            add("chip_bitstream_audit_failed")
        elif "timed out" in low or "did not finish" in low or "no flow report" in low:
            add("chip_flow_crashed")
        elif status == "fail":
            add("chip_flow_tool_error")
    elif stage == "sim":
        if status == "blocked":
            add("ubuntu_not_answering")
        elif not ev.get("sim_finished") or "timed out" in low or "wrote no log" in low:
            add("chip_sim_not_finished")
        elif status == "fail" and ev.get("sim_mismatches"):
            if ev.get("netlist_macc_blocks") and ev.get("sim_wrong_all_have_a_negative_input") and not ev.get("sim_undefined_outputs"):
                add("chip_signed_multiplier")
            elif ev.get("sim_undefined_outputs") == ev.get("sim_mismatches"):
                add("chip_netlist_outputs_undefined")
            else:
                add("chip_netlist_differs")
    elif stage == "check":
        if "chip repo changed" in low:
            add("chip_repo_modified")
        elif "timing not met" in low:
            add("chip_timing_not_met")
    return out
