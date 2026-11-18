"""A3 for the chain on the BM Labs chip (x2c chain --target chip, x2c agent a3 --target chip).

* four REAL chip-chain failures (recorded in a test environment by breaking one thing each: a chipenv without fasm, the Makefile's
  python3 without fasm, the repo's unsigned multiplier map, the repo's IOBUF sim model) are named correctly by the rules and the agent;
* the causes with no real run yet are named from the exact messages the chip chain prints;
* an LLM that lies, picks an unsupported or a demo-only cause, or rambles is overruled by code; unknown stays unknown;
* the CLI and the chain's own rule diagnosis work for --target chip, and the demo path is unchanged.
"""
import json
import shutil
from pathlib import Path

import pytest

import x2c.commands.agent_a3 as cli
import x2c.commands.chain as chain
from x2c.agents.chain_debug_agent import CHIP_PROMPT, PROMPT, MockChainDebugLLM, run_chain_debug_agent
from x2c.agents.llm_client import Action
from x2c.verification import chain_debug as cd

FIX = Path(__file__).resolve().parent / "fixtures" / "chip_debug"
ORDER = ["rtl", "prep", "env", "flow", "sim", "check"]
REAL = {"env_tools_missing": "chip_tools_missing", "flow_python_missing": "chip_python_module_missing",
        "sim_signed_multiplier": "chip_signed_multiplier", "sim_outputs_undefined": "chip_netlist_outputs_undefined"}


def recorded(case, root):
    """Lay a recorded case out as the chain does: reports/chip_jobs/<op>/, chip_build/chain_<op>/. Returns the chain result."""
    r = json.loads((FIX / case / "result.json").read_text())
    for src, dst in (("job", root / "reports" / "chip_jobs" / r["op"]), ("build", root / "chip_build" / f"chain_{r['op']}")):
        if (FIX / case / src).is_dir():
            shutil.copytree(FIX / case / src, dst)
        else:                                       # a run blocked early wrote nothing there (zip files keep no empty folders)
            dst.mkdir(parents=True)
    return r


def result(bad_stage, status, detail, op="relu8"):
    rows = []
    for s in ORDER:
        st = "pass" if ORDER.index(s) < ORDER.index(bad_stage) else status if s == bad_stage else "not_run"
        rows.append({"stage": s, "status": st, "detail": detail if s == bad_stage else "ok", "seconds": 1.0})
    return {"op": op, "verdict": "NOT RUN" if status == "blocked" else "FAIL", "stages": rows, "facts": {}}


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def rules(op, r):
    return cd.chip_rule_candidates(cd.collect_chip_evidence(op, r))


# ------------------------------------------------------------ real failures
@pytest.mark.parametrize("case,cause", sorted(REAL.items()))
def test_real_chip_failures_are_named(root, case, cause):
    r = recorded(case, root)
    assert rules(r["op"], r) == [cause]
    d = run_chain_debug_agent(MockChainDebugLLM(), r["op"], r, target="chip")
    assert d["final_cause"] == cause and d["final_cause_source"] == "llm_validated" and d["known_fix"] == cd.CHIP_CAUSES[cause]["fix"]


def test_the_multiplier_evidence(root):
    r = recorded("sim_signed_multiplier", root)
    ev = cd.collect_chip_evidence(r["op"], r)
    assert ev["sim_mismatches"] == 219 and ev["sim_vectors"] == 485 and ev["sim_undefined_outputs"] == 0
    assert ev["sim_wrong_all_have_a_negative_input"] is True and ev["netlist_macc_blocks"] == 4
    assert ev["sim_first_mismatches"][0].startswith("inputs={'acc': -")


def test_undefined_outputs_are_not_the_multiplier(root):
    r = recorded("sim_outputs_undefined", root)
    ev = cd.collect_chip_evidence(r["op"], r)
    assert ev["sim_undefined_outputs"] == ev["sim_mismatches"] == 500 and ev["netlist_macc_blocks"] == 0


# ------------------------------------------------------------ causes from the chain's own messages
@pytest.mark.parametrize("stage,status,detail,cause", [
    ("rtl", "fail", "Yosys vs NumPy: 3/256 vectors differ", "rtl_not_verified"),
    ("rtl", "blocked", "GROQ_API_KEY is not set in this window", "llm_key_missing"),
    ("rtl", "fail", "RuntimeError: Yosys not found: pip install yowasp-yosys (or install OSS CAD Suite)", "yosys_missing"),   # real run
    ("prep", "fail", "no verified core hardware/operators/relu8.v (the rtl stage writes it)", "rtl_not_verified"),
    ("prep", "fail", "wrapper check function: 3/256 vectors differ from NumPy through the chip pins", "chip_wrapper_check_failed"),
    ("env", "blocked", "Ubuntu (WSL) did not answer within 180 s", "ubuntu_not_answering"),
    ("flow", "blocked", "wsl.exe could not start Ubuntu", "ubuntu_not_answering"),
    ("flow", "fail", "chip flow failed at pins: pin_budget_exceeded | log chip_build/chain_relu8/pins.log", "chip_pins_exceeded"),
    ("flow", "fail", "chip flow failed at pnr: over_capacity | ERROR: no BELs remaining", "chip_over_capacity"),
    ("flow", "fail", "chip flow failed at synth: syntax_error | ERROR: syntax error", "chip_synth_error"),
    ("flow", "fail", "chip flow failed at check: bitstream_audit_failed", "chip_bitstream_audit_failed"),
    ("flow", "fail", "the chip flow did not finish this run (report status started); see flow_console.log", "chip_flow_crashed"),
    ("flow", "fail", "timed out after 1200 s", "chip_flow_crashed"),
    ("flow", "fail", "chip flow failed at pnr: routing_failed | log chip_build/chain_relu8/pnr.log", "chip_flow_tool_error"),
    ("sim", "unverified", "the simulation did not finish (no DONE line)", "chip_sim_not_finished"),
    ("sim", "fail", "the netlist simulation wrote no log: iverilog failed", "chip_sim_not_finished"),
    ("check", "unverified", "flow ok | V0 audit 345/345 | netlist sim PASS | timing NOT met: 11.2 MHz < 12.0 MHz (nextpnr model)", "chip_timing_not_met"),
    ("check", "fail", "the chip repo changed during the flow", "chip_repo_modified"),
])
def test_causes_from_chain_messages(root, stage, status, detail, cause):
    r = result(stage, status, detail)
    if "failed at" in detail:                                               # the flow report names the cause, as chip_flow writes it
        b = root / "chip_build" / "chain_relu8"
        b.mkdir(parents=True)
        (b / "flow_report.json").write_text(json.dumps({"status": "failed", "failure": {"stage": detail.split()[4].rstrip(":"),
                                                                                       "cause": detail.split(": ")[1].split(" |")[0]}}))
    assert rules("relu8", r) == [cause]


def test_a_changed_multiplier_line_is_named(root):
    job = root / "reports" / "chip_jobs" / "relu8"
    job.mkdir(parents=True)
    (job / "flow_console.log").write_text("RuntimeError: the repo's MULTIPLIER_MAP_PATH line in user_designs/makefile.include changed; "
                                          "re-check chip/techmap/MACC_8x8_20_map_signed.v against it before building\n")
    r = result("flow", "fail", "the chip flow did not finish this run (report status started); see flow_console.log")
    assert rules("relu8", r) == ["chip_multiplier_setting_changed"]


def test_defined_wrong_values_without_multipliers(root):
    job = root / "reports" / "chip_jobs" / "relu8"
    job.mkdir(parents=True)
    good = (FIX / "sim_outputs_undefined" / "job" / "sim.log").read_text()          # use the format of a real log, with relu8's vectors
    from x2c.verification import chip_chain as cc
    import re
    v = cc.vectors_for("relu8")["x"]
    lines = [f"VEC {i} {(max(int(x), 0) ^ (1 if i == 7 else 0)) & 0xFF:08b}" for i, x in enumerate(v)]
    (job / "sim.log").write_text("\n".join(lines) + f"\nDONE {len(v)}\n")
    assert re.search(r"^DONE \d+", good, re.M)
    r = result("sim", "fail", "1/256 vectors differ from NumPy (0 undefined)")
    assert rules("relu8", r) == ["chip_netlist_differs"]


def test_yosys_missing_is_not_a_wrong_design_on_either_chain(root):
    real = "RuntimeError: Yosys not found: pip install yowasp-yosys (or install OSS CAD Suite)"
    demo = cd.rule_candidates(cd.collect_evidence("relu8", result("rtl", "fail", real), root / "reports" / "fab_jobs"))
    assert demo == ["yosys_missing"] and "NOT judged wrong" in cd.CAUSES["yosys_missing"]["suspect"]


def test_nothing_to_explain_and_unknown(root):
    ok = {"op": "relu8", "verdict": "PASS", "stages": [{"stage": s, "status": "pass", "detail": "ok"} for s in ORDER]}
    assert rules("relu8", ok) == [] and rules("relu8", None) == []
    assert rules("relu8", result("check", "fail", "something new")) == []


# ------------------------------------------------------------ causes and prompts
def test_every_chip_cause_has_text_and_a_basis():
    for cid, c in cd.CHIP_CAUSES.items():
        assert cid.startswith("chip_") and c["suspect"] and c["fix"] and c["seen"], cid
    assert set(cd.CHIP_CAUSES) <= cd.CAUSE_IDS and not set(cd.CHIP_CAUSES) & set(cd.CAUSES)
    assert all(cd.cause_info(k) for k in cd.chip_menu_ids()) and cd.cause_info("unknown") == {}


def test_prompts():
    assert all(f"- {k}:" in CHIP_PROMPT for k in cd.chip_menu_ids()) and "bitstream_not_loaded" not in CHIP_PROMPT
    assert not any(k in PROMPT for k in cd.CHIP_CAUSES) and "FABulous" in PROMPT                  # the demo prompt is unchanged


# ------------------------------------------------------------ code decides
class Says:
    def __init__(self, cause):
        self.cause = cause

    def start(self, goal, tool_schemas, system_prompt=None):
        self.prompt = system_prompt
        return Action("tool_calls", calls=[("chain_evidence", {"op": goal.split("op=")[1]})])

    def send_tool_results(self, results, force_final=False):
        self.seen = results[0][1]
        return Action("final", text=json.dumps({"cause_id": self.cause, "confidence": "high", "explanation": "x",
                                                "proposed_fix": "y", "next_check": "z"}) if self.cause else "I think it is the clock")


@pytest.mark.parametrize("claim,why", [("bitstream_not_loaded", "a demo-only cause"), ("chip_timing_not_met", "an unsupported chip cause"),
                                       ("made_up", "not a cause"), (None, "no JSON")])
def test_wrong_llm_answers_are_overruled(root, claim, why):
    r = recorded("sim_signed_multiplier", root)
    llm = Says(claim)
    d = run_chain_debug_agent(llm, r["op"], r, target="chip")
    assert d["final_cause"] == "chip_signed_multiplier" and d["final_cause_source"].startswith("rules_fallback"), why
    assert llm.prompt == CHIP_PROMPT and llm.seen["target"] == "chip"


def test_blind_mode_and_unknown(root):
    r = recorded("sim_signed_multiplier", root)
    llm = Says("chip_signed_multiplier")
    d = run_chain_debug_agent(llm, r["op"], r, target="chip", blind=True)
    assert "rule_based_candidates" not in llm.seen and d["final_cause_source"] == "llm_validated"
    d = run_chain_debug_agent(Says("unknown"), "relu8", result("check", "fail", "something new"), target="chip")
    assert d["final_cause"] == "unknown" and d["final_cause_source"] == "llm_unknown_confirmed_by_rules"


# ------------------------------------------------------------ CLI and the chain
def write_chip_report(results):
    cd.CHIP_REPORT.parent.mkdir(parents=True, exist_ok=True)
    cd.CHIP_REPORT.write_text(json.dumps({"target": "chip", "results": results}))


def test_cli_target_chip(root, capsys):
    with pytest.raises(SystemExit, match="--target chip"):
        cli.main(["--target", "chip"])
    r = recorded("sim_signed_multiplier", root)
    write_chip_report([{"op": "relu8", "verdict": "PASS", "stages": []}, r])
    assert cli.main(["--target", "chip"]) == 0
    out = capsys.readouterr().out
    assert "CHIP CHAIN DEBUG  op=requant_i32_i8  chain verdict FAIL  llm=mock" in out and "relu8" not in out
    assert "ROOT CAUSE: chip_signed_multiplier  [llm_validated]" in out and "basis: real run (your machine, 2026-11-13)" in out
    rep = json.loads(cli.CHIP_OUT.read_text())
    assert rep["target"] == "chip" and rep["diagnoses"][0]["final_cause"] == "chip_signed_multiplier"
    assert not cli.OUT.exists()                                                     # the demo report is not touched


def test_the_chain_names_chip_causes(root):
    r = recorded("flow_python_missing", root)
    lines = chain.diagnose(r["op"], r, "chip")
    assert lines[0].startswith("  likely cause [chip_python_module_missing]:") and "--target chip" in lines[-1]
    assert r["diagnosis"] == {"rule_based_candidates": ["chip_python_module_missing"], "source": "rules (no LLM)"}
    assert chain.diagnose("relu8", {"op": "relu8", "verdict": "PASS", "stages": []}, "chip") == []
