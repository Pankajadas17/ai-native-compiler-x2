"""x2c bmsemi workloads  -> runs the conv workloads and prints a table + writes reports/workload_report.json"""
from x2c.verification import tools
out = tools.run_workloads()
rep = __import__("json").load(open(out["report_path"]))
print(f"{'workload':26} {'tiles':>5} {'passes':>6} {'mapper':>6}  result")
for r in rep["results"]:
    mc = r.get("mapping_check", {})
    print(f"{r['test']:26} {r['mapper_pe_tiles']:>5} {mc.get('passes_executed','-'):>6} {mc.get('mvm_ops_from_mapper','-'):>6}  {'PASS' if r['passed'] else 'FAIL'}")
print(f"{out['passed']}/{out['total']} passed -> {out['report_path']}  [{rep['level']}]")
