"""Shared public cases, budgets, and unconditional scoring for development."""
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import beta

ROOT=Path(__file__).resolve().parents[2]
CAP=2**22
DELTA=.01
EPSILON=.00009


def case_matrix():
    # Reuse the established PUBLIC case definitions, never their outcomes.
    path=ROOT/"results"/"learned_validation"/"development_v1"/"spec.json"
    cases=json.loads(path.read_text(encoding="utf-8"))["cases"]
    assert len(cases)==72 and sum(c["kind"]=="main" for c in cases)==48
    return cases


def budget_schedule(cap=CAP):
    values,n=[],32
    while n<cap:
        values.append(n)
        n=2*int(math.ceil(n*1.125/2))
    return sorted(set(values+[256,384,512,cap]))


def probability(case,angle):
    """Simulator-only coefficients; never a policy or certificate input."""
    return case["A"]+case["B"]*np.cos(angle)+case["C"]*np.sin(angle)


def interval(k,n,alpha=.05):
    return [float(beta.ppf(alpha/2,k,n-k+1)) if k else 0.,
            float(beta.ppf(1-alpha/2,k+1,n-k)) if k<n else 1.]


def summarize_records(records,case,method):
    expected="accept" if case["C"]<0 else "reject" if case["C"]>0 else "abstain"
    wrong=sum(r["decision"] not in ("abstain",expected) for r in records)
    correct=sum(r["decision"]==expected for r in records) if expected!="abstain" else 0
    abstain=sum(r["decision"]=="abstain" for r in records)
    shots=np.array([r["shots"] for r in records])
    correct_costs=sorted(r["shots"] for r in records if r["decision"]==expected and expected!="abstain")
    rank=int(math.ceil(.95*len(records)))
    n95=correct_costs[rank-1] if len(correct_costs)>=rank else None
    return {"case_id":case["case_id"],"kind":case["kind"],"method":method,"repetitions":len(records),
            "wrong":wrong,"correct":correct,"abstained":abstain,
            "error_ci95_pointwise":interval(wrong,len(records)),
            "power_ci95_pointwise":interval(correct,len(records)) if expected!="abstain" else None,
            "mean_shots":float(shots.mean()),"median_shots":float(np.median(shots)),
            "min_shots":int(shots.min()),"max_shots":int(shots.max()),
            "empirical_N95_correct":n95,"N95_censored":n95 is None}


def aggregate(cells,methods):
    main=[c for c in cells if c["kind"]=="main"]
    result={}
    for method in methods:
        local=[c for c in main if c["method"]==method]
        all_local=[c for c in cells if c["method"]==method]
        result[method]={"main_geomean_shots":math.exp(np.mean([math.log(c["mean_shots"]) for c in local])),
                        "main_wrong":sum(c["wrong"] for c in local),
                        "main_abstained":sum(c["abstained"] for c in local),
                        "main_trials":sum(c["repetitions"] for c in local),
                        "all_wrong":sum(c["wrong"] for c in all_local),
                        "null_decisions":sum(c["wrong"] for c in all_local if c["kind"]=="null"),
                        "weak_abstained":sum(c["abstained"] for c in all_local if c["kind"]=="weak"),
                        "main_N95_censored_cases":sum(c["N95_censored"] for c in local)}
    if "psr_beta" in result:
        base=result["psr_beta"]["main_geomean_shots"]
        for value in result.values():
            value["cost_over_psr_beta"]=value["main_geomean_shots"]/base
    return result
