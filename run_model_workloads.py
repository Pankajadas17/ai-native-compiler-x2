"""python run_model_workloads.py [--full]
Runs real-model workloads (MNIST edge filter, DeepGate INT8 autoencoder) on the X2 simulator.
--full computes the entire 28x28 MNIST map (784 X2 passes per digit) instead of the 10x10 crop."""
import json, sys
from verification import tools
out = tools.run_model_workloads(full="--full" in sys.argv)
if "error" in out:
    sys.exit(out["error"])
rep = json.load(open(out["report_path"]))
for r in rep["results"]:
    print(f"\n{r['test']}  <-  {r['source_model']}  ->  {'PASS' if r['passed'] else 'FAIL'}")
    if "mapping_check" in r:
        print("   ", r["mapping_check"])
    for L in r.get("layers", []):
        print(f"    layer {L['layer']}: {L['n_in']:>3}->{L['n_out']:<3} passes {L['passes']:>3}/{L['mapper_mvm_ops']:<3} "
              f"acc_exact={L['accumulators_exact']} out_matches_repo={L['output_matches_repo']} ref_err={L['ref_err_vs_test_case']:.4f}")
print(f"\n{out['passed']}/{out['total']} passed -> {out['report_path']}  [{rep['level']}]")
