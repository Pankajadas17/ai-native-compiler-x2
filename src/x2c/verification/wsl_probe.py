"""
WSL probe: find out WHY Ubuntu stops during the simulation compile (the step that ended with no result line).

Three things are measured at the same time, so the result says which side ran out of what:
  * inside Ubuntu: a sampler writes available memory and the memory of the test programs to a log on the E: drive twice a second
    (the log is written outside Ubuntu's own disk, so it survives even if the whole Ubuntu VM dies);
  * on Windows: a sampler thread reads free RAM and commit headroom twice a second while Ubuntu runs;
  * Ubuntu's uptime before and after (a number that goes backwards means the VM restarted).
Phases inside Ubuntu:
  R: a plain memory test (Python allocates 128 MB at a time up to 2 GB; NO compiler), so Icarus is not needed to reproduce a crash
  A: the Icarus compile that FABulous' simulation step runs, with a 2.5 GB memory cap
  B: the same compile with no cap, exactly as FABulous runs it
  C: the real run_fab_sim.sh (FABulous itself + the compile + the simulation), on a copy, so its results do not touch your job folder
Your Windows project changes only under reports/wsl_probe/. In the Ubuntu demo project the probe makes the same copies the simulation step
makes, and removes its own temporary build/probe_*.vvp files.
Evidence level: diagnosis of the Linux/WSL/Windows environment (it says nothing about the design).
"""
import csv
import ctypes
import io
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

LOW_FREE_GB = 1.0          # rule of thumb: Windows with less free RAM than this is nearly out of memory for a VM that grows by 1.5 GB (seen: 0.6 GB free -> Ubuntu died at once)
PLENTY_FREE_GB = 1.5       # rule of thumb: if the LOWEST free RAM during a run was this much, Windows was not short of memory
GATE_FREE_GB = LOW_FREE_GB  # below this no Ubuntu run is started: it would only crash the VM
NEEDED_FREE_GB = 4.0       # comfortable: the simulation compile makes Ubuntu use about 1.5 GB more than it starts with

PROBE_SCRIPT = """#!/usr/bin/env bash
# Written by x2c doctor wsl. Records what happens to Ubuntu while a memory test and the simulation compile run.
# Everything goes to probe.log next to this script (on the Windows drive), so it survives a crash of the Ubuntu VM.
OP="${1:?usage: probe.sh <op>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
JOB="$HERE/../fab_jobs/$OP"
DEMO="${DEMO:-$HOME/work/demo}"
LOG="$HERE/probe.log"
. "$HOME/fabenv/bin/activate"
export PATH="$HOME/tools/oss-cad-suite/bin:$PATH"
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
: > "$LOG"
PARENT=$$
say "START pid=$PARENT kernel=$(uname -r) cpus=$(nproc) mem_total_kB=$(awk '/MemTotal/{print $2}' /proc/meminfo) swap_total_kB=$(awk '/SwapTotal/{print $2}' /proc/meminfo) uptime_s=$(cut -d' ' -f1 /proc/uptime)"
if [ ! -d "$DEMO/Test" ]; then say "NO_DEMO $DEMO/Test not found"; exit 1; fi
cd "$DEMO/Test" || exit 1
cp "$JOB/${OP}_fab_tb.v" "$DEMO/Test/${OP}_fab_tb.v" 2>/dev/null || say "NO_TESTBENCH $JOB/${OP}_fab_tb.v not found (run x2c chain first)"
[ -f "$DEMO/user_design/${OP}_fab.v" ] || cp "$JOB/${OP}_fab.v" "$DEMO/user_design/${OP}_fab.v" 2>/dev/null      # the flow step copies this file too
mkdir -p build/fabric_files
find ../Tile/ -name '*.v' -type f -exec cp {} build/fabric_files/ \\; 2>/dev/null
find ../Fabric/ -name '*.v' -type f -exec cp {} build/fabric_files/ \\; 2>/dev/null
say "FILES fabric_files=$(ls build/fabric_files | wc -l) build_size=$(du -sh build 2>/dev/null | cut -f1)"
alive() { [ -r "/proc/$PARENT/status" ] && ! grep -q '^State:[[:space:]]*Z' "/proc/$PARENT/status"; }     # a killed script that nobody reaped yet counts as gone
( while alive; do
    say "SAMPLE avail_kB=$(awk '/MemAvailable/{print $2}' /proc/meminfo) swap_free_kB=$(awk '/SwapFree/{print $2}' /proc/meminfo) work_rss_kB=$(ps -C iverilog,ivl,ivlpp,vvp,python3,FABulous,task -o rss= | awk '{s+=$1} END{print s+0}') load=$(cut -d' ' -f1 /proc/loadavg) top=$(ps -eo rss=,comm= --sort=-rss | head -3 | awk '{printf "%s:%dMB,", $2, $1/1024}')"
    sleep 0.5
  done ) >/dev/null 2>&1 &
SAMPLER=$!
STEPS="${RAMP_STEPS:-16}"; STEP_MB="${RAMP_STEP_MB:-128}"        # defaults: 16 steps of 128 MB = 2048 MB (the environment variables exist for the tests)
say "PHASE R_start (plain memory test: Python takes $STEP_MB MB at a time, $STEPS steps, no compiler)"
SECONDS=0
python3 - "$LOG" "$STEPS" "$STEP_MB" <<'PY' >> "$LOG" 2>&1
import sys, time
log, steps, step_mb = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
keep = []
for step in range(1, steps + 1):
    keep.append(b"a" * (step_mb * 1024 * 1024))
    with open(log, "a") as f:
        print(time.strftime("%H:%M:%S"), "RAMP", step * step_mb, "MB ok", file=f)
    time.sleep(0.25)
PY
CODE=$?
say "PHASE R_end exit=$CODE secs=$SECONDS"
compile() { iverilog -g2012 -s "${OP}_fab_tb" -o "build/probe_$1.vvp" build/fabric_files/* "../user_design/${OP}_fab.v" "${OP}_fab_tb.v"; }
say "PHASE A_start (compile capped at 2.5 GB of memory)"
SECONDS=0
( ulimit -v 2500000; compile A ) >> "$LOG" 2>&1
CODE=$?
say "PHASE A_end exit=$CODE secs=$SECONDS"
say "PHASE B_start (compile with no cap, as FABulous runs it)"
SECONDS=0
compile B >> "$LOG" 2>&1
CODE=$?
say "PHASE B_end exit=$CODE secs=$SECONDS"
rm -f build/probe_A.vvp build/probe_B.vvp
if [ -f "$DEMO/user_design/${OP}_fab.bin" ] && [ -f "$JOB/run_fab_sim.sh" ]; then
  say "PHASE C_start (the real run_fab_sim.sh, as the chain runs it; its results go to reports/wsl_probe/phaseC, not to your job folder)"
  mkdir -p "$HERE/phaseC"
  cp "$JOB/run_fab_sim.sh" "$JOB/${OP}_fab_tb.v" "$HERE/phaseC/"
  SECONDS=0
  bash "$HERE/phaseC/run_fab_sim.sh" "$OP" > "$HERE/phaseC/console.out" 2>&1
  CODE=$?
  RL=$(grep -c 'FAB_SIM RESULT' "$HERE/phaseC/sim_console.log" 2>/dev/null); RL=${RL:-0}
  say "PHASE C_end exit=$CODE secs=$SECONDS sim_exit=$(cat "$HERE/phaseC/sim_exit_code.txt" 2>/dev/null || echo none) result_line=$RL"
else
  say "PHASE C_skipped (no bitstream yet: run the chain's flow stage first)"
fi
say "DONE"
kill "$SAMPLER" 2>/dev/null
wait "$SAMPLER" 2>/dev/null
exit 0
"""


# ---------------------------------------------------------------- the Ubuntu-side log
def summarize_log(text):
    """Facts from probe.log: which phases finished, memory numbers, and where it stopped."""
    t = text or ""
    start = re.search(r"START pid=(\d+) kernel=(\S+) cpus=(\d+) mem_total_kB=(\d+) swap_total_kB=(\d+)", t)
    phases = {m.group(1): {"exit": int(m.group(2)), "secs": int(m.group(3))}
              for m in re.finditer(r"PHASE ([RABC])_end exit=(\d+) secs=(\d+)", t)}
    c_end = re.search(r"PHASE C_end exit=\d+ secs=\d+ sim_exit=(\S+) result_line=(\d+)", t)
    if c_end and "C" in phases:
        phases["C"].update({"sim_exit": c_end.group(1), "result_line": int(c_end.group(2)) > 0})
    started = re.findall(r"PHASE ([RABC])_start", t)
    samples = re.findall(r"^(\d\d:\d\d:\d\d) SAMPLE avail_kB=(\d+) swap_free_kB=(\d+) work_rss_kB=(\d+)", t, re.M)
    tops = re.findall(r"^\d\d:\d\d:\d\d SAMPLE .*? top=(\S*)", t, re.M)
    ramp = [int(x) for x in re.findall(r"RAMP (\d+) MB ok", t)]
    out = {"started": bool(start), "done": bool(re.search(r"^\d\d:\d\d:\d\d DONE$", t, re.M)), "phases": phases,
           "last_phase_started": started[-1] if started else None, "samples": len(samples), "ramp_max_mb": max(ramp) if ramp else 0,
           "ramp_step_mb": ramp[0] if ramp else None, "c_skipped": "PHASE C_skipped" in t,
           "last_top": tops[-1].rstrip(",") if tops and tops[-1] else None,
           "no_demo": "NO_DEMO" in t, "no_testbench": "NO_TESTBENCH" in t}
    if start:
        out.update({"kernel": start.group(2), "cpus": int(start.group(3)), "mem_total_mb": int(start.group(4)) // 1024,
                    "swap_total_mb": int(start.group(5)) // 1024})
    if samples:
        out.update({"last_sample_time": samples[-1][0], "last_avail_mb": int(samples[-1][1]) // 1024,
                    "min_avail_mb": min(int(s[1]) for s in samples) // 1024,
                    "peak_work_rss_mb": max(int(s[3]) for s in samples) // 1024,
                    "last_work_rss_mb": int(samples[-1][3]) // 1024})
    out["tool_messages"] = [l.strip() for l in t.splitlines()
                            if l.strip() and not re.match(r"^\d\d:\d\d:\d\d ", l) and re.search(r"error|memory|killed|cannot|fail", l, re.I)][:5]
    return out


def vm_restarted(uptime_before, uptime_after, elapsed_s):
    """True if Ubuntu's uptime counter went backwards relative to the wall clock (the VM restarted); None if the run was too short to tell."""
    if uptime_before is None or uptime_after is None:
        return None
    if elapsed_s < 6:
        return None
    return uptime_after < uptime_before + elapsed_s - 5


_UPTIME_LINE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s+\d+(?:\.\d+)?\s*$", re.M)


def parse_uptime(text):
    """Seconds up, from the line of `cat /proc/uptime` ("<seconds up> <idle seconds>"). wsl.exe sometimes prints warnings of its own above that
    line (in the same stream), so the line is searched for, not assumed to be first."""
    m = _UPTIME_LINE.search(text or "")
    return float(m.group(1)) if m else None


def uptime_extras(text):
    """The other lines of the answer: whatever wsl.exe printed besides the uptime itself (empty when it printed nothing else)."""
    out, seen = [], False
    for line in (text or "").splitlines():
        if not seen and _UPTIME_LINE.match(line):
            seen = True
            continue
        if line.strip():
            out.append(line.strip())
    return out


# ---------------------------------------------------------------- facts about WSL itself (for when Ubuntu does not answer)
LOW_DISK_GB = 5.0          # Ubuntu's virtual disk and its swap file normally live on C:; a nearly full drive makes the VM stop


def read_disk_free(paths=None, usage=shutil.disk_usage):
    """Free space in GB of C: (where Ubuntu's virtual disk and swap file normally live) and of the drive this folder is on. {} where unreadable."""
    if paths is None:
        try:
            here = Path.cwd().anchor or "/"
        except OSError:                                   # the folder was deleted under us
            here = "/"
        paths = ["C:\\", here]
    out = {}
    for p in paths:
        try:
            out[os.path.splitdrive(p)[0] or p] = round(usage(p).free / GB, 1)
        except OSError:
            continue
    return out


def _first_lines(text, n, width=170):
    lines = [" ".join(l.split()) for l in (text or "").splitlines() if l.strip()]
    return " / ".join(lines[:n])[:width]


def wsl_snapshot(runner, exec_fn, disks=None, low_disk_gb=LOW_DISK_GB):
    """Short facts about WSL itself, printed when Ubuntu does not answer: the WSL version, which distros are Running/Stopped, and free disk space.
    Only read-only wsl.exe commands that work even while the VM is down. `exec_fn(cmd, timeout_s, heartbeat_s, say)` is the same runner the probe uses."""
    lines = []
    if runner == "wsl":
        for label, cmd, n in (("WSL", ["wsl", "--version"], 2), ("Distros", ["wsl", "-l", "-v"], 3)):
            r = exec_fn(cmd, 30, 30, None)
            lines.append(f"{label}: {_first_lines(r.get('output'), n) or 'no answer'}")
    disks = read_disk_free() if disks is None else disks
    if disks:
        lines.append("Free disk space: " + ", ".join(f"{k} {v:.1f} GB" for k, v in disks.items()))
        low = [k for k, v in disks.items() if v < low_disk_gb]
        if low:
            lines.append(f"{' and '.join(low)} {'is' if len(low) == 1 else 'are'} almost full (under {low_disk_gb:g} GB): Ubuntu's virtual disk and swap file "
                         f"normally live on C:, and the VM stops when that fills up.")
    return lines


# ---------------------------------------------------------------- the Windows side
class MEMORYSTATUSEX(ctypes.Structure):
    """The Windows API structure behind GlobalMemoryStatusEx (64 bytes). DWORD is exactly 32 bits, so c_uint32 (c_ulong is 64 bits outside Windows)."""
    _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


GB = 1024 ** 3


def parse_meminfo(text):
    """Linux /proc/meminfo -> the same three numbers the Windows call gives."""
    kb = {m.group(1): int(m.group(2)) for m in re.finditer(r"^(\w+):\s+(\d+) kB", text or "", re.M)}
    if "MemAvailable" not in kb or "MemTotal" not in kb:
        return None
    limit, used = kb.get("CommitLimit"), kb.get("Committed_AS")
    return {"total_gb": kb["MemTotal"] * 1024 / GB, "free_gb": kb["MemAvailable"] * 1024 / GB,
            "commit_headroom_gb": (limit - used) * 1024 / GB if limit is not None and used is not None else None}


def read_host_memory(platform=None):
    """Free RAM and commit headroom of the machine the probe runs on (Windows: GlobalMemoryStatusEx; Linux: /proc/meminfo). None if unreadable."""
    try:
        if (platform or sys.platform).startswith("win"):
            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(st)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return None
            return {"total_gb": st.ullTotalPhys / GB, "free_gb": st.ullAvailPhys / GB, "commit_headroom_gb": st.ullAvailPageFile / GB}
        with open("/proc/meminfo", encoding="utf-8") as f:
            return parse_meminfo(f.read())
    except Exception:  # noqa: BLE001  - a failed reading must never stop the probe
        return None


class HostSampler:
    """Reads host memory twice a second on a background thread while Ubuntu runs."""
    def __init__(self, reader=read_host_memory, interval=0.5, clock=time.time):
        self.reader, self.interval, self.clock = reader, interval, clock
        self.samples, self._stop, self._thread = [], threading.Event(), None

    def _loop(self):
        t0 = self.clock()
        while not self._stop.is_set():
            r = self.reader()
            if r is None:
                break
            self.samples.append({"t": round(self.clock() - t0, 1), **r})
            self._stop.wait(self.interval)

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(3)
        return list(self.samples)


def host_summary(samples):
    if not samples:
        return None
    low = min(samples, key=lambda s: s["free_gb"])
    heads = [s["commit_headroom_gb"] for s in samples if s.get("commit_headroom_gb") is not None]
    return {"samples": len(samples), "total_gb": round(samples[0]["total_gb"], 1), "start_free_gb": round(samples[0]["free_gb"], 1),
            "min_free_gb": round(low["free_gb"], 1), "min_free_at_s": low["t"], "end_free_gb": round(samples[-1]["free_gb"], 1),
            "min_commit_headroom_gb": round(min(heads), 1) if heads else None}


def tightest_gb(host):
    """The number that decides whether Ubuntu's VM can run, and which one it is: (value, name).
    Commit headroom when Windows reports it, else free RAM. Seen on a 7.8 GB machine: Ubuntu stopped at once with 1.5 GB of commit headroom,
    and ran the whole simulation with free RAM as low as 0.4 GB while commit headroom stayed above 3 GB. So low free RAM alone does not stop it."""
    if not host:
        return None, None
    head = host.get("commit_headroom_gb")
    if head is not None:
        return head, "commit headroom"
    return host["free_gb"], "free RAM"


MEMORY_FIX = "Close browsers and other big programs, run `wsl --shutdown`"
GATE_COMMIT_GB = 2.0       # rule of thumb from the runs above (1.5 GB: stopped; 3.1 GB at the start: ran)


def memory_gate(host, gate=None, needed=NEEDED_FREE_GB):
    """Before any Ubuntu work: ('block' | 'warn' | 'ok' | 'unknown', sentence). 'block' = do not start (it would only crash the VM)."""
    value, what = tightest_gb(host)
    if value is None:
        return "unknown", ""
    if gate is None:
        gate = GATE_COMMIT_GB if what == "commit headroom" else GATE_FREE_GB
    other = f" (free RAM {host['free_gb']:.1f} GB)" if what == "commit headroom" else ""
    if value < gate:
        return "block", f"Windows has only {value:.1f} GB of {what}{other}: too little for Ubuntu's VM, which would most likely stop. {MEMORY_FIX}, then try again."
    if value < needed:
        return "warn", f"Windows has {value:.1f} GB of {what}{other}; the simulation is more reliable with about {needed:.0f} GB. {MEMORY_FIX}, then try again."
    return "ok", ""


def parse_tasklist(text):
    """`tasklist /fo csv /nh` output -> {program: [MB, count]}. The 'Mem Usage' column is '123,456 K' (digits only are used: separators differ by language)."""
    out = {}
    for row in csv.reader(io.StringIO(text or "")):
        if len(row) >= 5:
            kb = int(re.sub(r"\D", "", row[4]) or 0)
            e = out.setdefault(row[0], [0.0, 0])
            e[0] += kb / 1024
            e[1] += 1
    return out


def parse_ps(text):
    """`ps -eo rss=,comm=` output -> {program: [MB, count]}."""
    out = {}
    for m in re.finditer(r"^\s*(\d+)\s+(.+?)\s*$", text or "", re.M):
        e = out.setdefault(m.group(2), [0.0, 0])
        e[0] += int(m.group(1)) / 1024
        e[1] += 1
    return out


def _run_text(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=20).stdout


def top_memory_users(n=6, platform=None, run_text=_run_text):
    """The programs using the most memory right now: [{'name', 'mb', 'count'}], biggest first (programs with several processes are added up).
    Windows: tasklist; Linux: ps. [] if it cannot be read."""
    try:
        if (platform or sys.platform).startswith("win"):
            table = parse_tasklist(run_text(["tasklist", "/fo", "csv", "/nh"]))
        else:
            table = parse_ps(run_text(["ps", "-eo", "rss=,comm="]))
    except Exception:  # noqa: BLE001  - this is only a hint for the person; it must never stop a run
        return []
    top = sorted(table.items(), key=lambda kv: -kv[1][0])[:n]
    return [{"name": k, "mb": round(v[0]), "count": v[1]} for k, v in top]


def format_users(users):
    """'chrome.exe 1250 MB (9 processes) | Code.exe 800 MB (6 processes)'; adds what vmmem means when it is on the list."""
    text = " | ".join(f"{u['name']} {u['mb']} MB ({u['count']} process{'' if u['count'] == 1 else 'es'})" for u in users)
    if any(u["name"].lower().startswith("vmmem") and u["mb"] >= 300 for u in users):
        text += "  (vmmem/vmmemWSL is the Ubuntu VM itself: `wsl --shutdown` releases its memory)"
    return text


def sim_memory_sentence(h, op="add8_sat"):
    """One sentence for a simulation that gave no usable result, from host_summary(): was Windows short of memory while it ran?"""
    if not h:
        return ""
    again = f"then run `x2c chain --op {op} --from-stage sim`"
    v, what, low, plenty = lowest_memory(h)
    if v < low:
        return (f"Windows {what} fell to {v} GB during the run: Ubuntu was most likely stopped for lack of memory. {MEMORY_FIX}, {again}.")
    if v < plenty:
        return f"Windows {what} got low ({v} GB at its lowest): lack of memory may be the reason. {MEMORY_FIX}, {again}; `x2c doctor wsl` measures it."
    return f"Windows still had {v} GB of {what} at its lowest, so a lack of Windows memory is probably not the reason: `x2c doctor wsl` measures what happens."


PLENTY_COMMIT_GB = 3.0     # rule of thumb: the passing simulations ran with commit headroom at or above about 3 GB


def lowest_memory(h):
    """From host_summary(): (lowest value, which number, 'low' line, 'plenty' line). Commit headroom when Windows reported it, else free RAM
    (free RAM alone went as low as 0.4 GB in runs that passed, so it is only used when there is nothing better)."""
    head = h.get("min_commit_headroom_gb")
    if head is not None:
        return head, "commit headroom", GATE_COMMIT_GB, PLENTY_COMMIT_GB
    return h["min_free_gb"], "free RAM", LOW_FREE_GB, PLENTY_FREE_GB


# ---------------------------------------------------------------- the conclusion
def _host_line(host):
    return (f"Windows memory during the run: free RAM {host['start_free_gb']} GB at the start, lowest {host['min_free_gb']} GB "
            f"(of {host['total_gb']} GB total)" + (f"; commit headroom lowest {host['min_commit_headroom_gb']} GB" if host["min_commit_headroom_gb"] is not None else "")
            + f"; {host['samples']} sample{'' if host['samples'] == 1 else 's'}.")


def verdict(summary, restarted, host=None):
    """Plain-language conclusion. It only states what the logs show."""
    s = summary
    if not s["started"]:
        lines = ["The probe never started in Ubuntu (no log): Ubuntu stopped or could not run it. Check `wsl -d Ubuntu -- echo hello`."]
        if host:
            lines.append(_host_line(host))
            v, what, low, _ = lowest_memory(host)
            if v < low:
                lines.append(f"Windows was nearly out of {what} ({v} GB): Ubuntu's VM most likely stopped at once for that reason. {MEMORY_FIX}, then run the probe again.")
        return lines
    if s["no_demo"]:
        return ["The demo project was not found in Ubuntu (~/work/demo/Test). Set it up as in docs/chip.md."]
    lines = []
    r, a, b, c = (s["phases"].get(p) for p in "RABC")
    mem = ""
    if "peak_work_rss_mb" in s:
        mem = (f" Ubuntu had {s['mem_total_mb']} MB; lowest available {s['min_avail_mb']} MB; "
               f"the test programs used at most {s['peak_work_rss_mb']} MB; {s['samples']} sample{'' if s['samples'] == 1 else 's'}.")
    died = not s["done"]
    if died:
        ph = s["last_phase_started"]
        last = (f" Last sample at {s['last_sample_time']}: {s['last_avail_mb']} MB available, test programs using {s['last_work_rss_mb']} MB"
                + (f"; biggest programs then: {s['last_top']}" if s["last_top"] else "") + "." if "last_sample_time" in s else "")
        if ph == "R":
            lines.append(f"Ubuntu died during the PLAIN MEMORY TEST, after Python had taken {s['ramp_max_mb']} MB, before any compiler ran."
                         " So Icarus is not needed to reproduce the crash: Ubuntu dies when its memory use grows past "
                         f"{s['ramp_max_mb']} MB" + (f" (the next {s['ramp_step_mb']} MB step did not complete)." if s["ramp_step_mb"] else ".") + last + mem)
        else:
            where = {"A": "the capped compile (phase A)", "B": "the uncapped compile (phase B)",
                     "C": "the real simulation script (phase C)"}.get(ph, "the set-up before the memory test")
            ok_ramp = (f" The plain memory test had passed ({s['ramp_max_mb']} MB)." if r and r["exit"] == 0 else "")
            ok_compiles = (" Both bare compiles had finished, so the compile alone does not kill Ubuntu: what the real script adds on top "
                           "(FABulous itself and its task runner) matters." if ph == "C" and a and a["exit"] == 0 and b and b["exit"] == 0 else "")
            lines.append(f"Ubuntu died during {where}." + ok_ramp + ok_compiles + last + mem)
    else:
        if r and r["exit"] != 0:
            lines.append(f"The plain memory test was stopped by Ubuntu itself at {s['ramp_max_mb']} MB (exit {r['exit']}), and Ubuntu stayed up: that is normal behaviour when its limit is reached.")
        if a and a["exit"] != 0:
            lines.append(f"The CAPPED compile (2.5 GB) failed with exit {a['exit']} after {a['secs']} s: Icarus wants more memory than the cap, or it failed for another reason."
                         + (" Messages: " + " | ".join(s["tool_messages"]) if s["tool_messages"] else "") + mem)
        elif b and b["exit"] != 0:
            lines.append(f"The uncapped compile failed with exit {b['exit']} after {b['secs']} s, without Ubuntu dying."
                         + (" Messages: " + " | ".join(s["tool_messages"]) if s["tool_messages"] else "") + mem)
        else:
            if c:
                real = ("the real simulation script (it printed a result line)" if c.get("result_line")
                        else "the real simulation script (it ended WITHOUT a result line: read reports/wsl_probe/phaseC/console.out)")
                what = f"both compiles and {real} finished"
            elif s["c_skipped"]:
                what = "both compiles finished (the real simulation script was skipped: there is no bitstream yet, run the chain's flow stage first)"
            else:
                what = "both compiles finished"
            lines.append(f"The memory test reached {s['ramp_max_mb']} MB, {what} by themselves, and Ubuntu stayed up this time. "
                         "So the crash does not happen every time: run the chain's simulation step again now." + mem)
    if host:
        lines.append(_host_line(host))
    if died or restarted is True:
        if restarted is True:
            lines.append("Ubuntu's uptime counter went back to near zero: the whole Ubuntu VM restarted during the run.")
        if host:
            v, what, low, plenty = lowest_memory(host)
            if v < low:
                lines.append(f"Windows was nearly out of {what} ({v} GB) when this happened: that is the likely cause. "
                             "Close big programs (browsers) before the simulation, or lower Ubuntu's memory in .wslconfig.")
            elif v >= plenty:
                lines.append(f"Windows still had {v} GB of {what} at its lowest, so a shortage of Windows memory does not explain it: "
                             "look at WSL itself (its version, the kernel, a recent update).")
            else:
                lines.append(f"Windows {what} got low ({v} GB at its lowest) but not empty: a possible contributor, not proof.")
    elif restarted is False:
        lines.append("Ubuntu's uptime kept counting: the VM did not restart during the run.")
    return lines
