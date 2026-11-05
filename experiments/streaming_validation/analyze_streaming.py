"""Equal-case total-cost comparisons, retaining every stop and abstention."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from benchmark import ROOT
RESULTS=ROOT/"results"/"streaming_validation"


def load_runs(paths):
    methods,cells,source_info=[],{},[]
    data={}
    cases=None
    repetitions=None
    for path in map(Path,paths):
        summary=json.loads((path/"summary.json").read_text(encoding="utf-8"))
        spec=summary["spec"]
        if cases is None:
            cases=spec["cases"]
            repetitions=spec["repetitions"]
        assert spec["cases"]==cases and spec["repetitions"]==repetitions
        for method in spec["methods"]:
            assert method not in methods
            methods.append(method)
            data[method]=np.zeros((len(cases),repetitions))
        for cell in summary["cells"]:
            cells[cell["case_id"],cell["method"]]=cell
        with (path/"trials.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row=json.loads(line)
                data[row["method"]][row["case_id"],row["rep"]]=row["shots"]
        source_info.append({"run":path.name,"trials":summary["trial_count"],
                            "summary_sha256":hashlib.sha256((path/"summary.json").read_bytes()).hexdigest(),
                            "ledger_sha256":hashlib.sha256((path/"trials.jsonl").read_bytes()).hexdigest()})
    assert all(np.all(values>0) for values in data.values())
    return cases,methods,cells,data,source_info


def main(paths):
    cases,methods,cells,data,source_info=load_runs(paths)
    main_ids=[c["case_id"] for c in cases if c["kind"]=="main"]
    values=np.stack([data[m][main_ids] for m in methods],axis=1)
    means=values.mean(axis=2)
    geometric=np.exp(np.log(means).mean(axis=0))
    random=np.random.RandomState(9204113)
    log_boot=np.zeros((2000,len(methods)))
    for i in range(len(main_ids)):
        indices=random.randint(0,values.shape[2],size=(2000,values.shape[2]))
        log_boot+=np.log(values[i][:,indices].mean(axis=2).T)/len(main_ids)
    compared={}
    for j,method in enumerate(methods):
        compared[method]={}
        for k,base in enumerate(methods):
            bootstrap=np.exp(log_boot[:,j]-log_boot[:,k])
            compared[method][base]={"ratio":float(geometric[j]/geometric[k]),
                                   "ci95_pointwise":[float(x) for x in np.percentile(bootstrap,[2.5,97.5])],
                                   "faster_cases":int(np.sum(means[:,j]<means[:,k])),
                                   "equal_cases":int(np.sum(means[:,j]==means[:,k])),
                                   "worst_case_ratio":float(np.max(means[:,j]/means[:,k])),
                                   "best_case_ratio":float(np.min(means[:,j]/means[:,k]))}
    by_factor={}
    for factor in ("A","rho","B_over_R"):
        by_factor[factor]={}
        for value in sorted(set(cases[i][factor] for i in main_ids)):
            selected=[j for j,i in enumerate(main_ids) if cases[i][factor]==value]
            local=np.exp(np.log(means[selected]).mean(axis=0))
            by_factor[factor][str(value)]={m:float(local[j]/local[methods.index("psr_beta")]) for j,m in enumerate(methods)}
    stats={}
    for method in methods:
        local=[cells[i,method] for i in main_ids]
        stats[method]={"main_geomean_shots":float(geometric[methods.index(method)]),
                       "main_wrong":sum(c["wrong"] for c in local),
                       "main_abstained":sum(c["abstained"] for c in local),
                       "main_repetitions":sum(c["repetitions"] for c in local),
                       "all_wrong":sum(cells[c["case_id"],method]["wrong"] for c in cases),
                       "null_decisions":sum(cells[c["case_id"],method]["wrong"] for c in cases if c["kind"]=="null"),
                       "weak_abstained":sum(cells[c["case_id"],method]["abstained"] for c in cases if c["kind"]=="weak")}
    result={"status":"development comparison; selection on this matrix is not held-out confirmation",
            "sources":source_info,"methods":methods,"stats":stats,"comparisons":compared,
            "cost_over_psr_beta_by_factor":by_factor,
            "uncertainty":"2000 paired replicate resamples within each fixed main case; shared-prefix dependence preserved; pointwise descriptive intervals, not simultaneous or across-model guarantees",
            "new_trial_count":sum(x["trials"] for x in source_info),
            "source_sha256":hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()}
    (RESULTS/"comparison.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"new_trials":result["new_trial_count"],"cost_over_psr_beta":{
        m:compared[m]["psr_beta"] for m in methods}},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("runs",nargs="+")
    args=parser.parse_args()
    main(args.runs)
