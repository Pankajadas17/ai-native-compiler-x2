"""x2c: one command for the whole project.

    x2c                          list every command
    x2c <command> --help         that command's own options
    x2c pipeline                 everything, in order, one summary (reports/pipeline/REPORT.md)

Commands marked [Linux] need the chip tools (Yosys, nextpnr, Icarus): run them in the Ubuntu terminal.
"""
import json
import os
import platform
import runpy
import shutil
import subprocess
import sys
import time

from x2c import paths

ANY, LINUX = "any", "Linux"

# group title -> {command: (module under x2c, where it runs, what it does)}
GROUPS = {
    "Run everything": {
        "pipeline": ("commands.pipeline", LINUX, "every stage in order, one summary in reports/pipeline/REPORT.md"),
        "recheck": ("@recheck", LINUX, "after a new X2 model ($BM_X2_REPO): profile, config, X2 check, engine, deploy"),
        "status": ("@status", ANY, "chip status: builds, timing, audits, chain, X2 facts still missing"),
        "test": ("@test", ANY, "the test suite (pytest; arguments pass through)"),
    },
    "Deploy a model": {
        "deploy": ("commands.deploy", ANY, "model -> verified bundle (--workload mobilenetv2|deepgate|mnist; --sim on Linux)"),
        "replay": ("commands.replay", ANY, "re-run a bundle from its files alone (e.g. x2c replay deploy/mnist)"),
    },
    "Build and prove the chip design": {
        "build": ("commands.build", LINUX, "Verilog -> Yosys -> nextpnr -> bitstream (e.g. --example x2_engine2_4bit)"),
        "audit": ("commands.audit", ANY, "V0: decode a chip bitstream, every required bit"),
        "chain": ("commands.chain", LINUX, "A2 operators on the chip (--op all --target chip --runner native)"),
        "formal": ("commands.formal", LINUX, "proofs for all inputs, engine bounded check, Verilator lint"),
    },
    "Simulate the X2 path (add --sim)": {
        "sim x2": ("commands.sim_x2", LINUX, "Python X2 golden == the X2 developer's Verilog model"),
        "sim pack": ("commands.sim_pack", LINUX, "weight packing and command stream"),
        "sim recombine": ("commands.sim_recombine", LINUX, "recombiner Verilog"),
        "sim bridge": ("commands.sim_bridge", LINUX, "fabric <-> X2 bridge"),
        "sim engine1": ("commands.sim_engine1", LINUX, "engine with one X2 block"),
        "sim engine2": ("commands.sim_engine2", LINUX, "engine with two X2 blocks (the deployment engine)"),
        "sim layers": ("commands.sim_layers", LINUX, "every real MobileNetV2 layer shape"),
    },
    "Describe the chip": {
        "chip profile": ("commands.chip_profile", ANY, "chip resources read from sky130-reram-fpga -> specifications/chip_profile.json"),
        "chip config": ("commands.chip_config", ANY, "specifications/hardware_config.json for A1"),
        "chip costs": ("chip.fabric_costs", ANY, "measured fabric cost per block (engine, relu, add, requant)"),
    },
    "Agents": {
        "agent a1": ("commands.agent_a1", ANY, "A1: partition + X2 schedule (LLM proposes, code validates)"),
        "agent a2": ("commands.agent_a2", ANY, "A2: Verilog for a fabric operator, checked by Yosys vs NumPy"),
        "agent a3": ("commands.agent_a3", ANY, "A3: names the cause of a failed run (add --target chip)"),
        "agent a4": ("commands.agent_a4", ANY, "A4: verification agent (generated tests, the oracle decides)"),
        "eval a3": ("commands.eval_a3", ANY, "A3 benchmark on injected bugs"),
        "eval a4": ("commands.eval_a4", ANY, "A4 generated vs fixed tests on planted bugs"),
    },
    "Check the setup": {
        "doctor env": ("commands.doctor_env", ANY, "Python, packages and BMsemi_Model found"),
        "doctor llm": ("commands.doctor_llm", ANY, "LLM keys and models answer (OpenRouter / Groq)"),
        "doctor wsl": ("commands.doctor_wsl", ANY, "from Windows: can WSL/Ubuntu run the tools, memory"),
    },
    "BMsemi simulator workloads (first project stage)": {
        "bmsemi verify": ("commands.bmsemi_verify", ANY, "X2 vs Python golden, fault injection"),
        "bmsemi workloads": ("commands.bmsemi_workloads", ANY, "convolution workloads tiled onto the X2"),
        "bmsemi models": ("commands.bmsemi_models", ANY, "MNIST edge filter + DeepGate vs the repo's own results"),
        "bmsemi benchmark": ("commands.bmsemi_benchmark", ANY, "all BMsemi workloads, stage table and digest"),
    },
    "FABulous demo fabric (before the chip repo)": {
        "demo prep": ("commands.demo_prep", ANY, "prepare the FABulous demo project"),
        "demo flow-check": ("commands.demo_flow_check", ANY, "check a FABulous flow run's logs"),
        "demo audit": ("commands.demo_audit", ANY, "V0 audit on the demo fabric"),
    },
}
COMMANDS = {name: spec for group in GROUPS.values() for name, spec in group.items()}
PREFIXES = sorted({name.split()[0] for name in COMMANDS if " " in name})       # sim, chip, agent, ...

# after a new X2 model: every step that depends on it, in order
RECHECK = [("chip profile", []), ("chip config", []), ("sim x2", ["--sim"]), ("sim pack", ["--sim"]),
           ("sim engine2", ["--sim"]), ("deploy", ["--workload", "mnist", "--sim"]),
           ("deploy", ["--workload", "mobilenetv2", "--sim"])]


def on_linux():
    return platform.system() == "Linux"


def tools_ready():
    return all(shutil.which(t) for t in ("yosys", "nextpnr-generic", "iverilog", "vvp"))


def show_list(only=None):
    print(__doc__.strip().split("\n\n")[0])
    if on_linux():
        print(f"\nchip tools: {'found' if tools_ready() else 'NOT on PATH (run your toolchain export line)'}")
    for title, cmds in GROUPS.items():
        names = [n for n in cmds if only is None or n.split()[0] == only]
        if not names:
            continue
        print(f"\n{title}")
        for n in names:
            print(f"  {n:18s} {cmds[n][2]}{'  [Linux]' if cmds[n][1] == LINUX else ''}")
    print("\nx2c <command> --help for its options.   Repository upkeep: python tools/prepush_check.py "
          "(before every push)")
    return 0


def _exit_code(code):
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    print(code, file=sys.stderr)
    return 1


def status():
    from x2c.verification.chip_tools import chip_status
    print(json.dumps(chip_status(), indent=1, default=str))
    return 0


def recheck():
    from x2c.chip.chip_repo import x2_model_label
    print(f"X2 model used: {x2_model_label()}")
    rows = []
    for name, args in RECHECK:
        line = f"x2c {name} {' '.join(args)}".strip()
        print(f"\n===== {line}")
        t = time.time()
        rc = subprocess.run([sys.executable, "-m", "x2c", *name.split(), *args], cwd=paths.ROOT).returncode
        rows.append((line, rc, time.time() - t))
    print("\n===== SUMMARY")
    for line, rc, sec in rows:
        print(f"  {'PASS' if rc == 0 else 'FAIL'}  {sec:7.0f} s  {line}")
    bad = sum(1 for _, rc, _ in rows if rc)
    print(f"{len(rows) - bad}/{len(rows)} passed")
    return 1 if bad else 0


def run(name, args):
    """Run one command in this process, exactly as `python <its module>` would."""
    module, where, _ = COMMANDS[name]
    if where == LINUX and not on_linux() and not {"--help", "-h", "--plan"} & set(args):
        print(f"'x2c {name}' needs the chip tools: run it in the Ubuntu terminal:  x2c {name} {' '.join(args)}".rstrip())
        return 2
    os.chdir(paths.ROOT)                       # commands read and write project folders (reports/, chip_build/, ...)
    if str(paths.ROOT) not in sys.path:       # BMsemi_Model is a folder in the project, not an installed package
        sys.path.append(str(paths.ROOT))
    if module.startswith("@") and {"-h", "--help"} & set(args):
        print(f"x2c {name}: {COMMANDS[name][2]}")
        return 0
    if module == "@status":
        return status()
    if module == "@recheck":
        return recheck()
    target = "pytest" if module == "@test" else f"x2c.{module}"
    sys.argv = [f"x2c {name}", *(args or (["-q"] if module == "@test" else []))]
    try:
        runpy.run_module(target, run_name="__main__")   # sys.argv[0] stays "x2c <command>" for --help
    except SystemExit as e:
        return _exit_code(e.code)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help", "list"):
        return show_list()
    if argv[0] in PREFIXES:
        if len(argv) == 1 or argv[1] in ("-h", "--help"):
            return show_list(only=argv[0])
        name, args = f"{argv[0]} {argv[1]}", argv[2:]
    else:
        name, args = argv[0], argv[1:]
    if name not in COMMANDS:
        print(f"unknown command 'x2c {name}'. Run x2c to list the commands.")
        return 2
    return run(name, args)


if __name__ == "__main__":
    sys.exit(main())
