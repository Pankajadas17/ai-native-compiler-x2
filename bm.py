"""
bm.py: ONE entry point for every command of the project. The individual scripts stay where they are (tests and
other scripts import them); this just gives them short names, groups them, says which terminal each needs, and runs
whole workflows with one PASS/FAIL summary.

    python bm.py                         # the list of commands
    python bm.py deploy --workload mnist # = python deploy_chip.py --workload mnist   (any arguments pass through)
    python bm.py build --help            # the script's own help
    python bm.py status                  # A4's view of the chip: builds, timing, audits, chain, missing X2 facts
    python bm.py all                     # the standard workflow for THIS terminal (Windows or Ubuntu), summarised
    python3 bm.py x2-recheck             # Ubuntu: re-check everything that depends on the X2 model (new X2 repo)
"""
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY, TOOLS = "either", "Ubuntu"            # where a command runs: "either" terminal, or only Ubuntu (chip tools)

# name: (script or [python args], terminal, what it does)
GROUPS = {
    "Checks": {
        "tests": (["-m", "pytest", "-q"], PY, "all tests (Windows skips the chip-tool tests; Ubuntu runs them)"),
        "env": ("check_environment.py", PY, "Python, packages and BMsemi_Model found"),
        "llm": ("check_llm.py", PY, "LLM keys and models answer (Groq / Gemini)"),
        "status": ("@status", PY, "A4's chip status: builds, timing, audits, chain, missing X2 facts"),
    },
    "Early pipeline (X2 simulator, BMsemi workloads)": {
        "verify": ("run_verification.py", PY, "X2 vs Python golden, fault injection"),
        "workloads": ("run_workloads.py", PY, "convolution workloads tiled onto the X2"),
        "model-workloads": ("run_model_workloads.py", PY, "MNIST edge filter + DeepGate vs the repo's own results"),
        "benchmark": ("run_benchmark.py", PY, "all BMsemi workloads, stage table and digest"),
    },
    "Agents": {
        "a1": ("run_mapping_agent.py", PY, "A1: partition + X2 schedule (LLM proposes, code validates)"),
        "a2": ("run_rtl_agent.py", PY, "A2: Verilog for an eFPGA operator, checked by Yosys vs NumPy"),
        "a3": ("run_chain_debug.py", PY, "A3: names the cause of a failed run (add --target chip)"),
        "a4": ("run_agent.py", PY, "A4: verification agent (generated tests, oracle decides)"),
        "eval-a3": ("evaluate_debug_agent.py", PY, "A3 benchmark on injected bugs"),
        "eval-a4": ("evaluate_test_generation.py", PY, "A4 generated vs fixed tests on planted bugs"),
    },
    "Chip description": {
        "chip-profile": ("run_chip_profile.py", PY, "chip resources read from sky130-reram-fpga -> specifications/chip_profile.json"),
        "hw-config": ("run_hardware_config.py", PY, "specifications/hardware_config.json for A1"),
        "costs": (["-m", "chip.fabric_costs"], PY, "measured eFPGA cost per block (engine, relu, add, requant)"),
        "wsl-probe": ("run_wsl_probe.py", PY, "Windows side: can WSL/Ubuntu run the tools, memory"),
    },
    "Chip build and audit": {
        "build": ("run_chip_flow.py", TOOLS, "Verilog -> Yosys -> nextpnr -> bitstream (e.g. --example x2_engine2_4bit)"),
        "audit": ("run_chip_bitstream_audit.py", PY, "V0: decode a chip bitstream, every required bit"),
        "chain": ("run_fab_chain.py", TOOLS, "A2 operators on the chip (--op all --target chip --runner native)"),
        "formal": ("run_formal.py", TOOLS, "proofs for all inputs + engine bounded check + Verilator lint (--part spec,netlist,lint,engine)"),
    },
    "X2 on the chip (simulation, add --sim)": {
        "x2-check": ("run_x2_check.py", TOOLS, "Python X2 golden == the X2 developer's Verilog model"),
        "x2-pack": ("run_x2_pack_check.py", TOOLS, "weight packing + command stream"),
        "x2-recombine": ("run_x2_recombine_check.py", TOOLS, "recombiner Verilog"),
        "x2-bridge": ("run_x2_bridge_check.py", TOOLS, "fabric <-> X2 bridge"),
        "x2-engine": ("run_x2_engine_check.py", TOOLS, "one-block engine"),
        "x2-engine2": ("run_x2_engine2_check.py", TOOLS, "two-block engine (the deployment engine)"),
        "layers": ("run_x2_layer_e2e.py", TOOLS, "every real MobileNetV2 layer shape"),
    },
    "Whole pipeline": {
        "pipeline": ("run_pipeline.py", TOOLS, "EVERYTHING in order, one summary saved in reports/pipeline/REPORT.md"),
    },
    "Deployment": {
        "deploy": ("deploy_chip.py", PY, "a model -> verified bundle (--workload mobilenetv2|deepgate|mnist, --sim in Ubuntu)"),
        "replay": ("run_bundle.py", PY, "replay a bundle from its files alone (e.g. replay deploy/mnist)"),
    },
    "Demo fabric (FABulous, earlier stage)": {
        "fab-prep": ("run_fab_prep.py", PY, "prepare the FABulous demo project"),
        "flow-check": ("run_flow_check.py", PY, "check a FABulous flow run's logs"),
        "audit-demo": ("run_bitstream_audit.py", PY, "V0 audit on the demo fabric"),
    },
    "Repository upkeep": {
        "inventory": ("tools/inventory.py", PY, "which files are used / generated / unused"),
        "cleanup": ("tools/cleanup_move.py", PY, "move backups out (dry run unless --apply)"),
        "prepush": ("tools/prepush_check.py", PY, "must PASS before every git push"),
    },
}
COMMANDS = {name: spec for g in GROUPS.values() for name, spec in g.items()}

# workflows: lists of (command name, args)
WINDOWS_ALL = [("tests", []), ("deploy", ["--workload", "mobilenetv2"]), ("deploy", ["--workload", "deepgate"]),
               ("deploy", ["--workload", "mnist"]), ("status", [])]
UBUNTU_ALL = [("x2-check", ["--sim"]), ("x2-engine2", ["--sim"]), ("chain", ["--op", "all", "--target", "chip", "--runner", "native"]),
              ("deploy", ["--workload", "mobilenetv2", "--sim"]), ("deploy", ["--workload", "mnist", "--sim"]),
              ("deploy", ["--workload", "deepgate", "--sim"]), ("formal", []), ("costs", []), ("status", [])]
X2_RECHECK = [("chip-profile", []), ("hw-config", []), ("x2-check", ["--sim"]), ("x2-pack", ["--sim"]), ("x2-engine2", ["--sim"]),
              ("deploy", ["--workload", "mnist", "--sim"]), ("deploy", ["--workload", "mobilenetv2", "--sim"])]
WORKFLOWS = {
    "all": ("the standard workflow for this terminal, one summary at the end", None),
    "x2-recheck": ("Ubuntu: after a new X2 model ($BM_X2_REPO): profile, config, golden vs model, engine, deploy", X2_RECHECK),
}


def on_ubuntu():
    return platform.system() == "Linux"


def tools_ready():
    return all(shutil.which(t) for t in ("yosys", "nextpnr-generic", "iverilog", "vvp"))


def show_list():
    print(__doc__.strip().split("\n\n")[0])
    print(f"\nterminal now: {'Ubuntu' if on_ubuntu() else platform.system()}"
          + ("" if not on_ubuntu() else f", chip tools {'found' if tools_ready() else 'NOT on PATH (run your toolchain export line)'}"))
    for g, cmds in GROUPS.items():
        print(f"\n{g}")
        for name, (script, where, what) in cmds.items():
            tag = "" if where == PY else "  [Ubuntu]"
            print(f"  {name:16s} {what}{tag}")
    print("\nWorkflows")
    for name, (what, _) in WORKFLOWS.items():
        print(f"  {name:16s} {what}")
    print("\npython bm.py <command> [arguments]   (Ubuntu: python3)   ·   python bm.py <command> --help")
    return 0


def status():
    from verification.chip_tools import chip_status
    s = chip_status()
    print(json.dumps(s, indent=1, default=str))
    return 0


def run(name, args):
    if name not in COMMANDS:
        print(f"unknown command '{name}'. python bm.py lists them.")
        return 2
    script, where, _ = COMMANDS[name]
    if script == "@status":
        return status()
    if where == TOOLS and not on_ubuntu() and "--help" not in args and "--plan" not in args:
        print(f"'{name}' needs the chip tools: run it in the Ubuntu terminal:  python3 bm.py {name} {' '.join(args)}".rstrip())
        return 2
    cmd = [sys.executable, *script] if isinstance(script, list) else [sys.executable, str(ROOT / script)]
    if isinstance(script, str) and not (ROOT / script).is_file():
        print(f"{script} is not in this folder")
        return 2
    return subprocess.run([*cmd, *args], cwd=ROOT).returncode


def workflow(name, extra):
    if name == "all":
        steps = UBUNTU_ALL if on_ubuntu() else WINDOWS_ALL
        if on_ubuntu() and not tools_ready():
            print("chip tools are not on PATH: run your toolchain export line first")
            return 2
    else:
        steps = WORKFLOWS[name][1]
        if name == "x2-recheck":
            from chip.chip_repo import x2_model_label
            print(f"X2 model used: {x2_model_label()}")
    rows = []
    for cmd, args in steps:
        line = f"bm.py {cmd} {' '.join(args)}".strip()
        print(f"\n===== {line}")
        t = time.time()
        rc = run(cmd, args + (extra if cmd == "deploy" else []))
        rows.append((line, rc, time.time() - t))
    print("\n===== SUMMARY")
    for line, rc, sec in rows:
        print(f"  {'PASS' if rc == 0 else 'FAIL'}  {sec:7.0f} s  {line}")
    bad = sum(1 for _, rc, _ in rows if rc)
    print(f"{len(rows) - bad}/{len(rows)} passed")
    return 1 if bad else 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "list", "help"):
        return show_list()
    name, args = argv[0], argv[1:]
    if name in WORKFLOWS:
        return workflow(name, args)
    return run(name, args)


if __name__ == "__main__":
    sys.exit(main())
