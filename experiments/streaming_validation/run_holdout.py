"""Prospective new-case check, released only after the fixed development gate.

Default operation creates the plan and performs no sampling. Execution requires
an explicit passing gate record. This is a prospectively specified validation
within an ongoing development project, not a public preregistration.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import scipy

from benchmark import ROOT,CAP,DELTA,EPSILON,budget_schedule,summarize_records,aggregate
from run_streaming import run_cell as run_psr_cell
from run_guarded import run_cell as run_guarded_cell,EXPLORE_START_SHOTS,EXPLORE_SQRT_MULTIPLIER

METHODS=("psr_beta","psr_safe","guarded_closed_beta","guarded_grid_beta")
REPETITIONS=128
REPLICATE_SEED_STRIDE=1000003
PLAN_PATH=ROOT/"results"/"streaming_validation"/"holdout_plan.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def holdout_cases():
    result=[]
    for a in (.015,.08,.3,.65):
        for rho in (.4,.97):
            radius=rho*min(a,1-a)
            for b_fraction in (0.,.8,.99):
                for sign in (-1,1):
                    result.append({"kind":"main","A":a,"B":radius*b_fraction,
                                   "C":sign*radius*math.sqrt(1-b_fraction*b_fraction),
                                   "rho":rho,"B_over_R":b_fraction})
    for a in (.015,.08,.3,.65):
        for b in (0.,.97*min(a,1-a)):
            result.append({"kind":"null","A":a,"B":b,"C":0.})
    for cid,case in enumerate(result):
        case.update({"case_id":cid,"h":.2,"epsilon":EPSILON,
                     "target_delta":2*case["C"]*math.sin(.1)})
        assert math.hypot(case["B"],case["C"])<min(case["A"],1-case["A"])
        if case["kind"]=="main":
            assert abs(case["target_delta"])>=EPSILON
    assert len(result)==56
    return result


def sources():
    directory=Path(__file__).resolve().parent
    paths=[directory/name for name in ("run_holdout.py","run_streaming.py","run_guarded.py",
                                       "benchmark.py","designs.py","betting.py","safe_two_sample.py")]
    paths.append(ROOT/"experiments"/"circuit_validation"/"direct_tests.py")
    return {str(path.relative_to(ROOT)).replace("\\","/"):sha256(path) for path in paths}


def replicate_seed(cid,rep,method):
    # Different cases/replicates have disjoint seed ranges; within a replicate
    # both guarded beta methods share the PSR beta stream until exploration.
    return 210000000+cid*1009+rep*REPLICATE_SEED_STRIDE+(71 if method=="psr_safe" else 0)


def freeze_plan(path=PLAN_PATH):
    path=Path(path)
    if path.exists():
        raise RuntimeError("Refusing to overwrite an existing prospective plan")
    assert EXPLORE_START_SHOTS==4096 and EXPLORE_SQRT_MULTIPLIER==4
    cases=holdout_cases()
    plan={"status":"prospectively specified new matrix and new seeds before guarded-v2 full outcomes; not public preregistration",
          "methodology_status":"method and guard settings developed on the previous development matrix",
          "sampling_status_at_specification":"not run; no prospective observations sampled",
          "methods":list(METHODS),"primary_method":"guarded_closed_beta","primary_baseline":"psr_beta",
          "secondary_same_information_comparator":"guarded_grid_beta","additional_control":"psr_safe",
          "cases":cases,"repetitions":REPETITIONS,"main_cases":48,"null_cases":8,
          "minimum_main_absolute_delta":min(abs(c["target_delta"]) for c in cases if c["kind"]=="main"),
          "delta":DELTA,"epsilon":EPSILON,"cap":CAP,"checkpoints":budget_schedule(),
          "guard":{"explore_start_previous_shots":4096,"sqrt_multiplier":4,
                   "target":"ceil(4*sqrt(current_total_budget/2)); deficit capped by available pairs",
                   "split":"ceil(deficit/2) atpi/3, thenfloor(deficit/2) at2pi/3; exploit remainingpairs",
                   "stopping":"unchanged common outer budget checkpoints; no subblock early looks"},
          "seed_base":210000000,"case_seed_stride":1009,"replicate_seed_stride":REPLICATE_SEED_STRIDE,
          "method_seed_offset":{"psr_beta":0,"psr_safe":71,"guarded_closed_beta":0,"guarded_grid_beta":0},
          "replicate_execution":"each repetition run separately with its registered seed; joint replicate labels remain independent across methods",
          "primary_hypothesis":"geometric mean over48new maincases of mean totalcost(guarded_closed_beta)/mean totalcost(psr_beta) <1",
          "primary_analysis":{"metric":"unconditional totalcost including everyquery,errorstop,andfullcapabstention",
                              "case_weighting":"equalweight48maincases",
                              "bootstrap":"5000joint replicate-label resamples within eachcase; recompute percase means then geometric mean",
                              "bootstrap_seed":910733,"interval":"two-sided percentile95%; successrequiresupperendpoint<1",
                              "fixed_primary":True},
          "secondary_analysis":"report identicalcostmetric and uncertainty versus guarded_grid_beta and psr_safe; no multiplicity-adjusted SOTA claim",
          "safety_analysis":"retain everyerror andabstention in main/null; mathematicalriskassumptions and certificateaudits remainrequired;128reps doesnotempiricallycertify1%percase",
          "release_gate":{"primary_method":"guarded_closed_beta","baseline":"psr_beta",
                          "development_point_ratio_below":1.,"development_upper95_bootstrap_below":1.,
                          "require_root_release_announcement":True},
          "sources":sources(),"runtime":{"python":"3.8.3","numpy":np.__version__,"scipy":scipy.__version__},
          "limits":["synthetic stationary exactsinglefrequency model","nohardwareoroptimizerloop",
                    "newcasesandseeds, butmethodologydevelopedonoldmatrix","specifiedcomparatorsarenotcompleteallanglestructuredSOTA"]}
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(plan,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"plan":str(path),"sha256":sha256(path),"main_cases":48,"null_cases":8,
                      "minimum_main_absolute_delta":plan["minimum_main_absolute_delta"],
                      "sampling":"not started"},indent=2))
    return plan


def checked_plan(path):
    path=Path(path)
    plan=json.loads(path.read_text(encoding="utf-8"))
    assert plan["cases"]==holdout_cases()
    assert plan["methods"]==list(METHODS) and plan["repetitions"]==REPETITIONS
    assert plan["checkpoints"]==budget_schedule()
    assert EXPLORE_START_SHOTS==4096 and EXPLORE_SQRT_MULTIPLIER==4
    for relative,expected in plan["sources"].items():
        assert sha256(ROOT/relative)==expected,relative
    return plan


def checked_gate(path):
    gate=json.loads(Path(path).read_text(encoding="utf-8"))
    assert gate["status"]=="passed"
    assert gate["primary_method"]=="guarded_closed_beta" and gate["baseline"]=="psr_beta"
    assert 0<gate["geomean_cost_ratio"]<1 and 0<gate["upper95_bootstrap"]<1
    assert gate["root_released"] is True
    assert sha256(gate["development_summary_path"])==gate["development_summary_sha256"]
    return gate


def run(output,gate_path,plan_path=PLAN_PATH):
    plan=checked_plan(plan_path)
    gate=checked_gate(gate_path)
    folder=Path(output)
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/"spec.json").exists():
        raise RuntimeError("Refusing to overwrite a prospective run")
    spec=dict(plan)
    spec.update({"plan_sha256":sha256(plan_path),"gate":gate,
                 "gate_record_sha256":sha256(gate_path),
                 "run_status":"released prospective new-case validation"})
    (folder/"spec.json").write_text(json.dumps(spec,indent=2,allow_nan=False),encoding="utf-8")
    started=time.time()
    cells=[]
    with (folder/"trials.jsonl").open("w",encoding="utf-8") as ledger:
        for case in plan["cases"]:
            local=[]
            for method in METHODS:
                rows=[]
                for rep in range(REPETITIONS):
                    seed=replicate_seed(case["case_id"],rep,method)
                    runner=run_guarded_cell if method.startswith("guarded_") else run_psr_cell
                    unused,record=runner(case,method,1,seed)
                    assert len(record)==1
                    row=record[0]
                    row["rep"]=rep
                    row["replicate_seed"]=seed
                    rows.append(row)
                    ledger.write(json.dumps(row,separators=(",",":"),allow_nan=False)+"\n")
                ledger.flush()
                cell=summarize_records(rows,case,method)
                if method.startswith("guarded_"):
                    cell["mean_exploration_shots"]=sum(2*r["exploration_pairs"] for r in rows)/REPETITIONS
                    cell["never_explored"]=sum(r["exploration_pairs"]==0 for r in rows)
                cells.append(cell)
                local.append(cell)
            print(json.dumps({"case":case["case_id"],"kind":case["kind"],
                              "means":[round(c["mean_shots"],1) for c in local],
                              "wrong":[c["wrong"] for c in local],
                              "abstained":[c["abstained"] for c in local]}),flush=True)
    result={"spec":spec,"cells":cells,"aggregate":aggregate(cells,METHODS),
            "trial_count":len(cells)*REPETITIONS,"new_trial_count":len(cells)*REPETITIONS,
            "reused_baseline_trajectories":0,"elapsed_seconds":time.time()-started}
    assert result["trial_count"]==28672
    (folder/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"done":True,"trials":result["trial_count"],"seconds":result["elapsed_seconds"],
                      "aggregate":result["aggregate"]}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-plan",action="store_true")
    parser.add_argument("--run",action="store_true")
    parser.add_argument("--plan",default=str(PLAN_PATH))
    parser.add_argument("--gate-record")
    parser.add_argument("--output",default=str(ROOT/"results"/"streaming_validation"/"holdout_v1"))
    args=parser.parse_args()
    if args.run:
        if not args.gate_record:
            parser.error("--run requires a passing --gate-record and prior root release")
        run(args.output,args.gate_record,args.plan)
    else:
        freeze_plan(args.plan)
