"""
Find out why Ubuntu stops during the simulation compile.

    x2c doctor wsl                 # run from the project folder (the one that shows (venv)); about 1 minute
    x2c doctor wsl --op relu8

Needs a job folder made by x2c chain (reports/fab_jobs/<op>/<op>_fab_tb.v). Inside Ubuntu it runs a plain memory test, then the same
Icarus compile as the simulation step (capped, then uncapped), then the real run_fab_sim.sh on a copy, while memory is sampled on BOTH sides
(Ubuntu's log goes to reports/wsl_probe/probe.log on the Windows drive, so it survives an Ubuntu crash; Windows' free RAM is sampled by this
script). Then it says what happened.
If Windows has less than 2 GB of commit headroom the probe does not start (Ubuntu's VM would only stop again): it lists the programs using the
most memory instead. Close them, run `wsl --shutdown`, and try again; --force runs it anyway.
If Ubuntu does not answer its first question (its uptime) the probe tries again twice, shows what wsl.exe printed on each try, and adds the WSL
version, which distros are Running or Stopped, and the free disk space of C: and of this drive.
Your project changes only under reports/wsl_probe/. Evidence level: diagnosis of the Linux/WSL/Windows environment, not of the design.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from x2c.verification import fab_chain as fc
from x2c.verification import wsl_probe as wp

OUT = Path("reports/wsl_probe")


def say(msg):
    print(msg, flush=True)


def uptime_command(runner, distro):
    return ["cat", "/proc/uptime"] if runner == "native" else ["wsl", "-d", distro, "--", "cat", "/proc/uptime"]


def tail_text(text, n=3, width=300):
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    return " / ".join(lines[-n:])[:width]


UPTIME_WAITS = (5, 15)      # seconds to wait before the 2nd and 3rd try of the first question (3 tries in all)


def wait(seconds):
    time.sleep(seconds)


def ask_uptime(runner, distro, exec_fn, say, waits=None, timeout=60):
    """Ask Ubuntu for its uptime. The first command after `wsl --shutdown` has to start the whole VM and can fail once, so a failed try is repeated
    after a short wait (not when wsl.exe cannot even be started). Every failed try says what wsl.exe printed. Returns {"uptime", "result", "tries"}."""
    waits = UPTIME_WAITS if waits is None else waits
    tries = len(waits) + 1
    for i in range(tries):
        r = exec_fn(uptime_command(runner, distro), timeout, 30, say)
        up = wp.parse_uptime(r["output"])
        if r.get("started", True) and up is not None:
            return {"uptime": up, "result": r, "tries": i + 1}
        say(f"  Ubuntu did not answer (try {i + 1} of {tries}; exit code {r.get('returncode')}, {r.get('seconds') or 0:.0f} s). "
            f"wsl.exe printed: {tail_text(r.get('output')) or 'nothing'}")
        if not r.get("started", True) or i == tries - 1:
            return {"uptime": None, "result": r, "tries": i + 1}
        say(f"  Trying again in {waits[i]} s ...")
        wait(waits[i])


def main(argv=None, exec_fn=fc.run_command, clock=time.time, host_reader=wp.read_host_memory, users_reader=None, snapshot_fn=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--op", default="add8_sat")
    ap.add_argument("--runner", choices=["auto", "wsl", "native"], default="auto")
    ap.add_argument("--distro", default="Ubuntu")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--force", action="store_true", help="run even when Windows has very little free memory (Ubuntu will most likely stop)")
    a = ap.parse_args(argv)
    runner = fc.resolve_runner(a.runner)
    tb = Path("reports/fab_jobs") / a.op / f"{a.op}_fab_tb.v"
    if not tb.exists():
        sys.exit(f"{tb} not found: run  x2c chain --op {a.op}  once first (it writes the job folder)")

    first = host_reader()
    gate, gate_text = wp.memory_gate(first)
    if first:
        say(f"Windows right now: {first['free_gb']:.1f} GB free RAM of {first['total_gb']:.1f} GB"
            + (f", commit headroom {first['commit_headroom_gb']:.1f} GB" if first.get("commit_headroom_gb") is not None else "")
            + ". (Commit headroom is what Ubuntu needs: under about 2 GB it is not started; to raise it, close browsers and other big programs.)")
    if gate in ("block", "warn"):
        users = (users_reader or wp.top_memory_users)()
        if users:
            say("  Biggest memory users right now: " + wp.format_users(users))
    if gate == "block":
        if not a.force:
            say("\n=== WSL PROBE NOT STARTED ===")
            say("  " + gate_text)
            say("  Ubuntu was not touched. (To run the probe anyway: x2c doctor wsl --force)")
            return 2
        say("  --force given: running anyway; expect Ubuntu to stop.")

    OUT.mkdir(parents=True, exist_ok=True)
    script = OUT / "probe.sh"
    script.write_text(wp.PROBE_SCRIPT, encoding="utf-8", newline="\n")
    (OUT / "probe.log").unlink(missing_ok=True)
    try:
        sp = fc.script_path(runner, str(script.resolve()))
    except ValueError as e:
        sys.exit(str(e))
    say(f"WSL probe: op={a.op} runner={runner}. Reading Ubuntu's uptime (this also starts Ubuntu) ...")
    snapshot = snapshot_fn or (lambda: wp.wsl_snapshot(runner, exec_fn))
    first_try = ask_uptime(runner, a.distro, exec_fn, say)
    u0 = first_try["uptime"]
    if u0 is None:
        for l in snapshot():
            say("  " + l)
        sys.exit(f"Ubuntu did not answer `cat /proc/uptime` ({first_try['tries']} tries; what wsl.exe printed is above). Run `wsl --shutdown`, then "
                 f"`wsl -d Ubuntu -- echo hello`, and try this again; if it keeps failing, the lines above say why.")
    say(f"  Ubuntu is up (uptime {u0:.1f} s" + (f", after {first_try['tries']} tries" if first_try["tries"] > 1 else "")
        + "). Running the memory test and the compile, sampling both sides; usually under a minute ...")
    extras = wp.uptime_extras(first_try["result"]["output"])
    if extras:
        say("  (wsl.exe also printed: " + tail_text("\n".join(extras)) + ")")
    sampler = wp.HostSampler(reader=host_reader).start()
    t0 = clock()
    run = exec_fn(fc.build_command(runner, sp, [a.op], a.distro), a.timeout, 30, say)
    elapsed = clock() - t0
    samples = sampler.stop()
    r1 = exec_fn(uptime_command(runner, a.distro), 120, 30, say)
    u1 = wp.parse_uptime(r1["output"])
    restarted = wp.vm_restarted(u0, u1, elapsed)

    log = OUT / "probe.log"
    summary = wp.summarize_log(log.read_text(encoding="utf-8", errors="replace") if log.exists() else "")
    host = wp.host_summary(samples)
    lines = wp.verdict(summary, restarted, host)
    if not summary["started"]:
        lines.append(f"What wsl.exe printed (exit code {run.get('returncode')}): {tail_text(run.get('output')) or 'nothing'}")
    if u1 is None:
        lines.append(f"Asked again after the run, wsl.exe printed (exit code {r1.get('returncode')}): {tail_text(r1.get('output')) or 'nothing'}")
    facts = snapshot() if (not summary["started"] or u1 is None) else []
    lines += facts
    if run.get("timed_out"):
        lines.append(f"The probe hit the {a.timeout} s timeout.")
    say("\n=== WSL PROBE RESULT ===")
    for l in lines:
        say("  " + l)
    after = "no answer (Ubuntu was not responding)" if u1 is None else f"{u1:.1f} s"
    say(f"  uptime before {u0:.1f} s, after {after}, run took {elapsed:.0f} s | log: {log.as_posix()}")
    (OUT / "probe_summary.json").write_text(json.dumps({"summary": summary, "host": host, "host_samples": samples, "vm_restarted": restarted,
                                                        "uptime_before": u0, "uptime_after": u1, "uptime_tries": first_try["tries"], "elapsed_s": round(elapsed, 1),
                                                        "wsl_facts": facts, "conclusion": lines}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
