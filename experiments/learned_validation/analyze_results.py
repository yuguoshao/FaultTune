"""Descriptive comparison and Monte Carlo uncertainty on the fixed case matrix."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT/"results"/"learned_validation"


def analyze(directory, bootstrap=2000):
    directory = Path(directory)
    summary = json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    spec = summary["spec"]
    cases, methods = spec["cases"], spec["methods"]
    main = [c for c in cases if c["kind"] == "main"]
    main_ids = [c["case_id"] for c in main]
    reps = spec["repetitions"]
    data = np.zeros((len(main), len(methods), reps))
    id_map = {cid:i for i,cid in enumerate(main_ids)}
    for line in (directory/"trials.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        if row["case_id"] in id_map:
            data[id_map[row["case_id"]], methods.index(row["method"]), row["rep"]] = row["shots"]
    assert np.all(data > 0)
    means = data.mean(axis=2)
    geometric = np.exp(np.mean(np.log(means), axis=0))
    random = np.random.RandomState(7391103)
    boot_log_cost = np.zeros((bootstrap, len(methods)))
    for i in range(len(main)):
        # Resample common replicate labels within each fixed case to preserve
        # shared-pilot dependence between learned methods.
        indices = random.randint(0, reps, size=(bootstrap, reps))
        boot_log_cost += np.log(data[i][:,indices].mean(axis=2).T)/len(main)
    boot_cost = np.exp(boot_log_cost)
    comparisons = {}
    for j, method in enumerate(methods):
        comparisons[method] = {}
        for k, baseline in enumerate(methods):
            ratio = boot_cost[:,j]/boot_cost[:,k]
            comparisons[method][baseline] = {
                "cost_ratio_method_over_baseline": float(geometric[j]/geometric[k]),
                "bootstrap_ci95_pointwise": [float(x) for x in np.percentile(ratio,[2.5,97.5])],
                "faster_cases": int(np.sum(means[:,j] < means[:,k])),
                "worst_case_ratio": float(np.max(means[:,j]/means[:,k])),
                "best_case_ratio": float(np.min(means[:,j]/means[:,k]))}
    factors = {}
    for factor in ("A", "rho", "B_over_R"):
        factors[factor] = {}
        for value in sorted(set(c[factor] for c in main)):
            chosen = [i for i,c in enumerate(main) if c[factor] == value]
            local = np.exp(np.mean(np.log(means[chosen]), axis=0))
            factors[factor][str(value)] = dict(zip(methods, [float(x/local[0]) for x in local]))
    return {"run": directory.name, "geomean_main_cost": dict(zip(methods,[float(x) for x in geometric])),
            "comparisons": comparisons, "cost_over_psr_mixture_by_registered_factor": factors,
            "uncertainty_scope": "2000 paired replicate bootstraps within each of 48 fixed main cases; exploratory pointwise intervals, no simultaneous or across-model population guarantee",
            "source_summary_sha256": hashlib.sha256((directory/"summary.json").read_bytes()).hexdigest(),
            "source_ledger_sha256": hashlib.sha256((directory/"trials.jsonl").read_bytes()).hexdigest()}, summary, means


def plot(runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1,len(runs),figsize=(6*len(runs),4.6),squeeze=False,sharey=True)
    for axis, (analysis,summary,means) in zip(axes[0], runs):
        methods = summary["spec"]["methods"]
        base = methods.index("psr_mixture")
        closed = methods.index("learned_closed_form")
        kl = methods.index("learned_kl_grid")
        axis.scatter(means[:,base], means[:,closed]/means[:,base], s=30,
                     alpha=.8,label="Closed-form variance design",color="#2364AA")
        axis.scatter(means[:,base], means[:,kl]/means[:,base], s=28,
                     alpha=.8,label="KL grid design",marker="x",color="#D15B28")
        axis.axhline(1,color="#333333",linewidth=1,linestyle="--")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("PSR mixture mean total shots")
        axis.set_ylabel("Total cost / PSR mixture cost")
        axis.set_title("Fixed paid pilot" if analysis["run"] == "development_v1" else "Early PSR exit before learning")
        axis.grid(True,alpha=.2,which="both")
        axis.legend(fontsize=8)
    figure.suptitle("48 fixed main cases; all learning and decision costs included",fontsize=12)
    figure.tight_layout(rect=[0,0,1,.95])
    path = RESULTS/"paid_learning_costs.png"
    figure.savefig(str(path),dpi=170)
    plt.close(figure)
    return str(path)


def main(directories):
    runs = [analyze(path) for path in directories]
    output = {"status":"descriptive development comparison",
              "runs":[entry[0] for entry in runs],
              "source_sha256":hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()}
    output["plot"] = plot(runs)
    (RESULTS/"comparison.json").write_text(json.dumps(output,indent=2),encoding="utf-8")
    print(json.dumps({"runs":[{"run":a["run"],"closed_vs_psr":a["comparisons"]["learned_closed_form"]["psr_mixture"],
                              "kl_vs_psr":a["comparisons"]["learned_kl_grid"]["psr_mixture"],
                              "closed_vs_grid":a["comparisons"]["learned_closed_form"]["learned_grid_design"]}
                             for a,unused,unused2 in runs]},indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="*",default=[str(RESULTS/"development_v1")])
    args = parser.parse_args()
    main(args.directories)
