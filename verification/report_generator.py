import json, time
from pathlib import Path
def write_report(results, path="reports/report.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    summary = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "level": "python_behavioral_simulation (NOT RTL, NOT silicon)",
               "total": len(results), "passed": sum(r["passed"] for r in results),
               "results": results}
    Path(path).write_text(json.dumps(summary, indent=2)); return summary
