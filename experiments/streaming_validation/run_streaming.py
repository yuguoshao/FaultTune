"""Development test: all evidence retained while angles and bets adapt.

The entire next block's angle and bets use only PREVIOUS observations.
Every physical observation is paid; there are no discarded pilot outcomes.
All controls share checkpoints, risk and initial minimum sample count.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import gammaln

from benchmark import ROOT,CAP,DELTA,EPSILON,case_matrix,budget_schedule,probability,summarize_records,aggregate
from designs import empty_state,update_pairs,choose_design,_at_shift,config_dict
from betting import (predictable_bets,kelly_log_increments,hedge_log_screen,
                     certificate_from_blocks as betting_certificate,protocol_config)
from safe_two_sample import (predictable_probabilities,safe_log_increments,safe_log_screen,
                             certificate_from_blocks as safe_certificate,protocol_config as safe_protocol_config)
sys.path.insert(0,str(ROOT/"experiments"/"circuit_validation"))
from direct_tests import certificate_from_pairs

METHODS=("psr_beta","closed_beta","grid_beta","psr_bet","closed_bet","grid_bet",
         "psr_safe","closed_safe","grid_safe")
BET_DENOMINATOR=2**20


def protocol(method):
    angle,engine=method.split("_")
    policy="psr" if angle=="psr" else "paired_variance_closed" if angle=="closed" else (
        "fisher_kl_grid" if engine=="safe" else "paired_kl_grid")
    return policy,engine


def block_design(method,state,indices,previous_shots):
    policy,engine=protocol(method)
    if policy=="psr" or previous_shots<256:
        design=choose_design("psr",state,indices)
        phase="psr"
    elif previous_shots<512:
        design=choose_design("psr",state,indices)
        angle=math.pi/3 if previous_shots<384 else 2*math.pi/3
        design.update(_at_shift(design["fit"],np.full(len(indices),angle)))
        phase="explore"
    else:
        design=choose_design(policy,state,indices)
        phase="adapt"
    return design,phase,engine


def unpack_blocks(blocks,engine):
    result=[]
    for entry in blocks:
        row={"t":entry[0],"counts":entry[1:5],"denominator":BET_DENOMINATOR}
        if engine=="bet":
            row.update({"plus_numerator":entry[5],"minus_numerator":entry[6]})
        elif engine=="safe":
            row.update({"pred_minus_numerator":entry[5],"pred_plus_numerator":entry[6]})
        result.append(row)
    return result


def final_certificate(counts,blocks,engine):
    if engine=="beta":
        return certificate_from_pairs(tuple(int(x) for x in counts),delta=DELTA)
    return (betting_certificate if engine=="bet" else safe_certificate)(unpack_blocks(blocks,engine),delta=DELTA)


def run_cell(case,method,repetitions,seed):
    policy,engine=protocol(method)
    state=empty_state(repetitions)
    counts=np.zeros((repetitions,4),dtype=np.int64)
    active=np.ones(repetitions,dtype=bool)
    logplus,logminus=np.zeros((2,repetitions))
    decisions=["abstain"]*repetitions
    certificates=[None]*repetitions
    histories=[[] for unused in range(repetitions)]
    final_looks=np.zeros(repetitions,dtype=int)
    phase_shots=np.zeros((repetitions,3),dtype=np.int64)
    random=np.random.RandomState(seed)
    last=0
    for look,budget in enumerate(budget_schedule(),1):
        indices=np.flatnonzero(active)
        if not len(indices):
            break
        design,phase,unused=block_design(method,state,indices,last)
        angles=design["t"]
        # Prediction is fixed BEFORE any outcome in this block is sampled.
        if engine=="bet":
            bets=predictable_bets(counts[indices],design["estimated_p_minus"],design["estimated_p_plus"])
            num0,num1=bets["plus_numerator"],bets["minus_numerator"]
        elif engine=="safe":
            bets=predictable_probabilities(counts[indices],design["estimated_p_minus"],design["estimated_p_plus"])
            num0,num1=bets["pred_minus_numerator"],bets["pred_plus_numerator"]
        else:
            num0=num1=np.zeros(len(indices),dtype=np.int64)
        number=(budget-last)//2
        pm,pp=probability(case,-angles),probability(case,angles)
        # Sequential binomial decomposition of a multinomial joint count:
        # minus outcomes first, then independent plus outcomes in each group.
        old_ones=random.binomial(number,pm)
        both=random.binomial(old_ones.astype(np.int32),pp)
        worse=random.binomial((number-old_ones).astype(np.int32),pp)
        fresh=np.column_stack((number-old_ones-worse,worse,old_ones-both,both))
        for j,i in enumerate(indices):
            entry=[float(angles[j])]+[int(x) for x in fresh[j]]
            if engine!="beta":
                entry.extend([int(num0[j]),int(num1[j])])
            histories[i].append(entry)
        if engine!="beta":
            increments=(kelly_log_increments if engine=="bet" else safe_log_increments)(
                fresh,num0,num1,BET_DENOMINATOR)
            logplus[indices]+=increments[0]
            logminus[indices]+=increments[1]
        counts[indices]+=fresh
        update_pairs(state,angles,fresh,indices)
        phase_shots[indices,("psr","explore","adapt").index(phase)]+=budget-last
        final_looks[indices]=look
        if engine=="beta":
            b,c=counts[indices,1],counts[indices,2]
            evidence=gammaln(b+.5)+gammaln(c+.5)-gammaln(b+c+1)-2*gammaln(.5)+(b+c)*math.log(2)
            selected=indices[evidence>=math.log(1/DELTA)-1e-6]
        else:
            pos,neg,threshold=(hedge_log_screen if engine=="bet" else safe_log_screen)(
                counts[indices],logplus[indices],logminus[indices],delta=DELTA)
            selected=indices[np.maximum(pos,neg)>=threshold-1e-6]
        for i in selected:
            cert=final_certificate(counts[i],histories[i],engine)
            if cert["decision"]!="abstain":
                decisions[i],certificates[i],active[i]=cert["decision"],cert,False
        last=budget
    records=[]
    for i in range(repetitions):
        if certificates[i] is None:
            certificates[i]=final_certificate(counts[i],histories[i],engine)
            # The final registered look always receives a full certificate,
            # including a crossing a conservative float screen delayed.
            decisions[i]=certificates[i]["decision"]
        records.append({"case_id":case["case_id"],"method":method,"rep":i,"seed":seed,
                        "counts":[int(x) for x in counts[i]],"shots":int(state["shots"][i]),
                        "phase_shots":[int(x) for x in phase_shots[i]],"look_index":int(final_looks[i]),
                        "decision":decisions[i],"certificate":certificates[i],"blocks":histories[i]})
    return summarize_records(records,case,method),records


def source_hashes():
    paths=[Path(__file__).resolve().parent/n for n in (
        "benchmark.py","run_streaming.py","designs.py","betting.py","safe_two_sample.py")]
    paths += [ROOT/"experiments"/"circuit_validation"/"direct_tests.py"]
    return {str(p.relative_to(ROOT)).replace("\\","/"):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run(repetitions,output,methods=METHODS):
    folder=Path(output)
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/"spec.json").exists():
        raise RuntimeError("Refusing to overwrite existing experiment")
    cases=case_matrix()
    prior=ROOT/"results"/"learned_validation"/"evidence_manifest.json"
    spec={"status":"development experiment; settings recorded before draws, not independent confirmation",
          "methods":list(methods),"cases":cases,"repetitions":repetitions,"delta":DELTA,"epsilon":EPSILON,
          "cap":CAP,"checkpoints":budget_schedule(),"exploration":[[0,256,"pi/2"],[256,384,"pi/3"],[384,512,"2pi/3"]],
          "adaptation":"after512shots, fitallpreviouspaidpairs; chooseoneangleandpredictionspernextblock",
          "pairing":"both angles and predictions fixed before either outcome; completed pairs only",
          "all_costs":"every sample used and charged, all stoppingerrors andfullcapabstentions retained",
          "design_config":config_dict(),"betting_config":protocol_config(),"safe_config":safe_protocol_config(),
          "sources":source_hashes(),"prior_evidence_manifest_sha256":hashlib.sha256(prior.read_bytes()).hexdigest(),
          "public_case_spec_sha256":hashlib.sha256((prior.parent/"development_v1"/"spec.json").read_bytes()).hexdigest(),
          "ledger_block_schema":{"beta":["angle","n00","n01","n10","n11"],
                                "bet":["angle","n00","n01","n10","n11","positive_bet_numerator","negative_bet_numerator"],
                                "safe":["angle","n00","n01","n10","n11","predicted_minus_numerator","predicted_plus_numerator"],
                                "common_denominator":BET_DENOMINATOR},
          "limits":["stationary exactsinglefrequency commonorientation, nohardware/drift/optimizerloop",
                    "sameinformation numericaldesigncontrols; no fullallangle structuredSOTA implementation",
                    "developmentmatrix reused for modelselection; no simultaneous empirical performancequalification"]}
    (folder/"spec.json").write_text(json.dumps(spec,indent=2),encoding="utf-8")
    started=time.time()
    cells=[]
    with (folder/"trials.jsonl").open("w",encoding="utf-8") as ledger:
        for case in cases:
            local=[]
            for method in methods:
                seed=130000000+case["case_id"]*1009+METHODS.index(method)*71
                cell,records=run_cell(case,method,repetitions,seed)
                cells.append(cell)
                local.append(cell)
                for record in records:
                    ledger.write(json.dumps(record,separators=(",",":"),allow_nan=False)+"\n")
                ledger.flush()
            print(json.dumps({"case":case["case_id"],"means":[round(c["mean_shots"],1) for c in local],
                              "wrong":[c["wrong"] for c in local],"abstained":[c["abstained"] for c in local]}),flush=True)
    summary={"spec":spec,"cells":cells,"aggregate":aggregate(cells,methods),
             "trial_count":len(cells)*repetitions,"elapsed_seconds":time.time()-started}
    (folder/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps({"done":True,"aggregate":summary["aggregate"],"seconds":summary["elapsed_seconds"]}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--repetitions",type=int,default=128)
    parser.add_argument("--output",default=str(ROOT/"results"/"streaming_validation"/"development_v1"))
    parser.add_argument("--methods",default=",".join(METHODS))
    args=parser.parse_args()
    methods=tuple(args.methods.split(","))
    if args.repetitions<1 or not methods or len(set(methods))!=len(methods) or any(m not in METHODS for m in methods):
        parser.error("invalidrepetitionsormethods")
    run(args.repetitions,args.output,methods)
