"""Run this FIRST. It proves you are talking to the real X2 simulator (not a stand-in)."""
import sys, pathlib
root = pathlib.Path(__file__).parent
print("Python:", sys.version.split()[0])
if (root / "bm_models").exists():
    print("PROBLEM: a local 'bm_models' folder exists next to this file. Rename/delete it."); sys.exit(1)
if not (root / "BMsemi_Model" / "bm_models" / "core" / "pe.py").exists():
    print("PROBLEM: BMsemi_Model folder not found inside bm_verif. Clone it (see README)."); sys.exit(1)
import numpy as np
from emulators.x2_adapter import x2_matmul_report
import BMsemi_Model.bm_models.core.pe as pe
print("Simulator file:", pe.__file__)
r = x2_matmul_report(np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]))
print("2x2 output   :", r["output"].tolist(), "(expected [[19, 22], [43, 50]])")
print("cycles       :", r["total_cycles"], "(the real simulator reports ~492 per pass; a stand-in would show 8)")
print("backend      :", r["backend"], "| physical_hardware:", r["physical_hardware"])
ok = r["output"].tolist() == [[19.0, 22.0], [43.0, 50.0]] and r["total_cycles"] > 100
print("ENVIRONMENT OK" if ok else "ENVIRONMENT PROBLEM"); sys.exit(0 if ok else 1)
