"""Execute the frozen prospective analysis without selecting a new candidate.

The unit resampled is the independent joint replicate label within a case.
All physical costs, including errors and cap abstentions, remain in the mean.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from benchmark import ROOT


PLAN_SHA256="c54cb0776a16159b6c4f8a4ef8a2563c6d01a73e31cf40b6a3a1a572cb6a0587"
BOOTSTRAP_REPETITIONS=5000
BOOTSTRAP_SEED=910733
METHODS=("psr_beta","psr_safe","guarded_closed_beta","guarded_grid_beta")
PRIMARY="guarded_closed_beta"
COMPARATORS=("psr_beta","guarded_grid_beta","psr_safe")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(directory):
    started=time.perf_counter()
    source=Path(__file__).resolve()
    source_hash=digest(source)
    directory=Path(directory).resolve()
    plan_path=directory.parent/"holdout_plan.json"
    assert digest(plan_path)==PLAN_SHA256
    plan=json.loads(plan_path.read_text(encoding="utf-8"))
    summary=json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    spec=json.loads((directory/"spec.json").read_text(encoding="utf-8"))
    assert spec==summary["spec"] and spec["plan_sha256"]==PLAN_SHA256
    for key,value in plan.items():
        assert spec[key]==value,key
    assert plan["primary_method"]==PRIMARY and plan["primary_baseline"]=="psr_beta"
    assert plan["primary_analysis"]["bootstrap_seed"]==BOOTSTRAP_SEED
    assert tuple(plan["methods"])==METHODS and plan["repetitions"]==128
    cases=plan["cases"]
    main_cases=[case for case in cases if case["kind"]=="main"]
    assert len(main_cases)==48 and len(cases)==56
    costs={(case["case_id"],method):np.zeros(128,dtype=np.int64)
           for case in cases for method in METHODS}
    seen=set()
    counts={method:{"main_trials":0,"main_wrong":0,"main_abstained":0,
                    "null_trials":0,"null_decisions":0,"null_abstained":0}
            for method in METHODS}
    lookup={case["case_id"]:case for case in cases}
    with (directory/"trials.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row=json.loads(line)
            cid,method,rep=row["case_id"],row["method"],row["rep"]
            key=(cid,method,rep)
            assert key not in seen and 0<=rep<128
            seen.add(key)
            assert 0<row["shots"]<=plan["cap"]
            costs[cid,method][rep]=row["shots"]
            case=lookup[cid]
            if case["kind"]=="main":
                expected="accept" if case["C"]<0 else "reject"
                counts[method]["main_trials"]+=1
                counts[method]["main_wrong"]+=row["decision"] not in (expected,"abstain")
                counts[method]["main_abstained"]+=row["decision"]=="abstain"
            else:
                counts[method]["null_trials"]+=1
                counts[method]["null_decisions"]+=row["decision"]!="abstain"
                counts[method]["null_abstained"]+=row["decision"]=="abstain"
    assert len(seen)==summary["trial_count"]==28672
    assert all(np.all(values>0) for values in costs.values())
    cells=[]
    log_samples={method:np.zeros(BOOTSTRAP_REPETITIONS) for method in COMPARATORS}
    point_logs={method:[] for method in COMPARATORS}
    rng=np.random.RandomState(BOOTSTRAP_SEED)
    for case in main_cases:
        cid=case["case_id"]
        # This single index array is shared by ALL methods within this case.
        labels=rng.randint(0,128,size=(BOOTSTRAP_REPETITIONS,128))
        means={method:float(np.mean(costs[cid,method])) for method in METHODS}
        sample_means={method:costs[cid,method][labels].mean(axis=1) for method in METHODS}
        ratios={method:means[PRIMARY]/means[method] for method in COMPARATORS}
        for method in COMPARATORS:
            log_samples[method]+=np.log(sample_means[PRIMARY]/sample_means[method])/48
            point_logs[method].append(math.log(ratios[method]))
        cells.append({"case":case,"mean_total_shots":means,"primary_over_comparator":ratios})
    comparisons={}
    distributions={}
    for method in COMPARATORS:
        samples=np.exp(log_samples[method])
        ratio=math.exp(math.fsum(point_logs[method])/48)
        lower,upper=[float(x) for x in np.percentile(samples,[2.5,97.5])]
        ratios=[cell["primary_over_comparator"][method] for cell in cells]
        comparisons[method]={"geomean_cost_ratio":ratio,"ci95_percentile":[lower,upper],
                             "cost_reduction_percent":100*(1-ratio),
                             "faster_cases":sum(x<1 for x in ratios),
                             "tied_cases":sum(x==1 for x in ratios),
                             "slower_cases":sum(x>1 for x in ratios),
                             "minimum_case_ratio":min(ratios),"maximum_case_ratio":max(ratios),
                             "minimum_case_id":cells[int(np.argmin(ratios))]["case"]["case_id"],
                             "maximum_case_id":cells[int(np.argmax(ratios))]["case"]["case_id"],
                             "primary_test":method=="psr_beta",
                             "upper_endpoint_below_one":upper<1}
        distributions[method]=samples.tolist()
    primary_pass=comparisons["psr_beta"]["upper_endpoint_below_one"]
    assert digest(source)==source_hash
    result={"status":"completed","primary_outcome":"passed" if primary_pass else "failed",
            "primary_method":PRIMARY,"primary_baseline":"psr_beta",
            "prospective_scope":"New cases and independent joint replicate seeds under the previously frozen plan; not public preregistration.",
            "analysis":{"metric":"Unconditional mean physical query cost in each case; equal-case geometric mean ratio.",
                        "bootstrap_repetitions":BOOTSTRAP_REPETITIONS,"bootstrap_seed":BOOTSTRAP_SEED,
                        "resampling_unit":"Joint replicate label within each of the 48 main cases, preserving cross-method dependence.",
                        "interval":"Two-sided percentile 95% interval; primary succeeds iff upper endpoint < 1.",
                        "uncertainty_scope":"Monte Carlo uncertainty for these 48 fixed cases, not uncertainty over a population of new cases.",
                        "primary_candidate_selected_before_observations":True,
                        "secondary_intervals":"Descriptive unadjusted intervals; no multiplicity-adjusted claim."},
            "main_cases":48,"null_cases":8,"repetitions":128,"new_trajectories":len(seen),
            "comparisons":comparisons,"safety_counts":counts,"main_case_results":cells,
            "limits":["Simulation in a stationary exact single-frequency Bernoulli model.",
                      "Only the frozen named comparators are tested; no complete structured all-angle SOTA comparison.",
                      "128 repetitions do not empirically certify a 1% per-case error bound.",
                      "Secondary intervals are exploratory and unadjusted."],
            "plan_sha256":PLAN_SHA256,"analysis_source_sha256":source_hash,
            "analysis_source_unchanged_during_execution":True,
            "artifact_sha256":{name:digest(directory/name) for name in ("spec.json","summary.json","trials.jsonl")},
            "elapsed_seconds":time.perf_counter()-started}
    (directory/"bootstrap_replicates.json").write_text(json.dumps({"comparators":distributions,
             "seed":BOOTSTRAP_SEED,"resamples":BOOTSTRAP_REPETITIONS},separators=(",",":")),encoding="utf-8")
    result["bootstrap_replicates_sha256"]=digest(directory/"bootstrap_replicates.json")
    (directory/"analysis.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({key:result[key] for key in ("status","primary_outcome","comparisons","safety_counts","elapsed_seconds")},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",default=str(ROOT/"results"/"streaming_validation"/"holdout_v1"))
    args=parser.parse_args()
    main(args.directory)
