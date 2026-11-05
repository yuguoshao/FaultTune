"""Development v2: delayed, persistent exploration with reusable evidence.

The settings were selected after inspecting development-v1's identifiability
and information-loss diagnostics. This is a fresh development experiment,
not held-out confirmation. Every completed pair is paid and retained.

Public policy: PSR until previous total shots>=4096. Thereafter, cumulative
exploration pairs target ceil(4*sqrt(current_total_budget/2)), limited by the
current block's available pairs. Split the deficit between pi/3 and2pi/3,
update the paid fit between subblocks, exploit with the remaining pairs.
Only the original common total-budget checkpoints may authorize stopping.
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

from benchmark import (ROOT,CAP,DELTA,EPSILON,case_matrix,budget_schedule,
                       probability,summarize_records,aggregate)
from designs import empty_state,update_pairs,choose_design,_at_shift,config_dict
from safe_two_sample import (predictable_probabilities,safe_log_increments,safe_log_screen,
                             certificate_from_blocks as safe_certificate,
                             protocol_config as safe_protocol_config)
sys.path.insert(0,str(ROOT/"experiments"/"circuit_validation"))
from direct_tests import certificate_from_pairs


METHODS=("guarded_closed_beta","guarded_grid_beta","guarded_closed_safe","guarded_grid_safe")
BET_DENOMINATOR=2**20
EXPLORE_START_SHOTS=4096
EXPLORE_SQRT_MULTIPLIER=4
PHASES=("psr","explore_plus_cosine","explore_minus_cosine","exploit")


def protocol(method):
    if method not in METHODS:
        raise ValueError("unknown guarded method")
    unused,angle,engine=method.split("_")
    policy="paired_variance_closed" if angle=="closed" else (
        "fisher_kl_grid" if engine=="safe" else "paired_kl_grid")
    return policy,engine


def seed_for(case,method):
    # Matching-engine controls use the same random stream before any guarded
    # exploration. This gives exactly matching cheap trajectories, and valid
    # coupled marginal simulations after the methods choose different angles.
    engine=protocol(method)[1]
    original_method_index=0 if engine=="beta" else 6
    return 130000000+case["case_id"]*1009+original_method_index*71


def unpack_blocks(blocks):
    return [{"outer_look":entry[0],"phase":entry[1],"t":entry[2],
             "counts":entry[3:7],"pred_minus_numerator":entry[7],
             "pred_plus_numerator":entry[8],"denominator":BET_DENOMINATOR}
            for entry in blocks]


def final_certificate(counts,blocks,engine):
    if engine=="beta":
        return certificate_from_pairs(tuple(int(x) for x in counts),delta=DELTA)
    return safe_certificate(unpack_blocks(blocks),delta=DELTA)


def next_subblock_design(policy,state,indices,phase):
    if phase in (0,1,2):
        design=choose_design("psr",state,indices)
        if phase in (1,2):
            angle=math.pi/3 if phase==1 else 2*math.pi/3
            design.update(_at_shift(design["fit"],np.full(len(indices),angle)))
        return design
    return choose_design(policy,state,indices)


def run_cell(case,method,repetitions,seed=None):
    policy,engine=protocol(method)
    seed=seed_for(case,method) if seed is None else int(seed)
    state=empty_state(repetitions)
    counts=np.zeros((repetitions,4),dtype=np.int64)
    active=np.ones(repetitions,dtype=bool)
    exploration_pairs=np.zeros(repetitions,dtype=np.int64)
    logplus,logminus=np.zeros((2,repetitions))
    decisions=["abstain"]*repetitions
    certificates=[None]*repetitions
    histories=[[] for unused in range(repetitions)]
    final_looks=np.zeros(repetitions,dtype=int)
    phase_shots=np.zeros((repetitions,4),dtype=np.int64)
    random=np.random.RandomState(seed)
    last=0
    for look,budget in enumerate(budget_schedule(),1):
        indices=np.flatnonzero(active)
        if not len(indices):
            break
        available=(budget-last)//2
        if last<EXPLORE_START_SHOTS:
            subblocks=[(0,np.full(len(indices),available,dtype=np.int64))]
        else:
            target=int(math.ceil(EXPLORE_SQRT_MULTIPLIER*math.sqrt(budget/2)))
            deficit=np.clip(target-exploration_pairs[indices],0,available)
            subblocks=[(1,(deficit+1)//2),(2,deficit//2),(3,available-deficit)]
        # Sample sizes and the order of subblocks are fixed before this
        # outer block. Forecasts/angles may use completed earlier subblocks.
        for phase,numbers in subblocks:
            present=numbers>0
            selected_indices=indices[present]
            number=numbers[present]
            if not len(selected_indices):
                continue
            design=next_subblock_design(policy,state,selected_indices,phase)
            angles=design["t"]
            if engine=="safe":
                predictions=predictable_probabilities(
                    counts[selected_indices],design["estimated_p_minus"],design["estimated_p_plus"])
                num0,num1=predictions["pred_minus_numerator"],predictions["pred_plus_numerator"]
            else:
                num0=num1=np.zeros(len(selected_indices),dtype=np.int64)
            pm,pp=probability(case,-angles),probability(case,angles)
            # Keep the scalar form when possible to reproduce the original
            # PSR control's random-number calls before the first exploration.
            first_n=int(number[0]) if np.all(number==number[0]) else number.astype(np.int32)
            old_ones=random.binomial(first_n,pm)
            both=random.binomial(old_ones.astype(np.int32),pp)
            worse=random.binomial((number-old_ones).astype(np.int32),pp)
            fresh=np.column_stack((number-old_ones-worse,worse,old_ones-both,both))
            for j,i in enumerate(selected_indices):
                entry=[look,phase,float(angles[j])]+[int(x) for x in fresh[j]]
                if engine=="safe":
                    entry.extend([int(num0[j]),int(num1[j])])
                histories[i].append(entry)
            if engine=="safe":
                increments=safe_log_increments(fresh,num0,num1,BET_DENOMINATOR)
                logplus[selected_indices]+=increments[0]
                logminus[selected_indices]+=increments[1]
            counts[selected_indices]+=fresh
            update_pairs(state,angles,fresh,selected_indices)
            phase_shots[selected_indices,phase]+=2*number
            if phase in (1,2):
                exploration_pairs[selected_indices]+=number
        final_looks[indices]=look
        assert np.all(state["shots"][indices]==budget)
        if engine=="beta":
            b,c=counts[indices,1],counts[indices,2]
            evidence=(gammaln(b+.5)+gammaln(c+.5)-gammaln(b+c+1)
                      -2*gammaln(.5)+(b+c)*math.log(2))
            selected=indices[evidence>=math.log(1/DELTA)-1e-6]
        else:
            pos,neg,threshold=safe_log_screen(counts[indices],logplus[indices],logminus[indices],delta=DELTA)
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
            decisions[i]=certificates[i]["decision"]
        records.append({"case_id":case["case_id"],"method":method,"rep":i,"seed":seed,
                        "counts":[int(x) for x in counts[i]],"shots":int(state["shots"][i]),
                        "phase_shots":[int(x) for x in phase_shots[i]],
                        "exploration_pairs":int(exploration_pairs[i]),"look_index":int(final_looks[i]),
                        "decision":decisions[i],"certificate":certificates[i],"blocks":histories[i]})
    cell=summarize_records(records,case,method)
    cell["mean_exploration_shots"]=float(np.mean(2*exploration_pairs))
    cell["never_explored"]=int(np.sum(exploration_pairs==0))
    return cell,records


def source_hashes():
    paths=[Path(__file__).resolve().parent/name for name in (
        "run_guarded.py","benchmark.py","designs.py","betting.py","safe_two_sample.py")]
    paths+=[ROOT/"experiments"/"circuit_validation"/"direct_tests.py"]
    return {str(path.relative_to(ROOT)).replace("\\","/"):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def load_controls(directory,repetitions):
    folder=Path(directory)
    summary=json.loads((folder/"summary.json").read_text(encoding="utf-8"))
    if summary["spec"]["cases"]!=case_matrix() or summary["spec"]["repetitions"]!=repetitions:
        raise ValueError("controls must have exactly the same cases and repetitions")
    controls=[cell for cell in summary["cells"] if cell["method"] in ("psr_beta","psr_safe")]
    if len(controls)!=2*len(case_matrix()):
        raise ValueError("missing matching-engine control cells")
    metadata={"source_directory":str(folder.resolve()),"methods":["psr_beta","psr_safe"],
              "reused_trajectories":len(controls)*repetitions,
              "source_sha256":{name:hashlib.sha256((folder/name).read_bytes()).hexdigest()
                               for name in ("spec.json","summary.json","trials.jsonl")}}
    return controls,metadata


def compare(cells,controls):
    base={(cell["case_id"],cell["method"]):cell for cell in controls}
    result={}
    for method in METHODS:
        local=[cell for cell in cells if cell["method"]==method and cell["kind"]=="main"]
        if not local:
            continue
        engine=protocol(method)[1]
        result[method]={}
        for comparator in ("psr_beta","psr_safe","matching_engine"):
            name="psr_"+engine if comparator=="matching_engine" else comparator
            ratios=[cell["mean_shots"]/base[cell["case_id"],name]["mean_shots"] for cell in local]
            result[method][comparator]={"geomean_cost_ratio":math.exp(np.mean(np.log(ratios))),
                                       "faster_cases":sum(value<1 for value in ratios),
                                       "tied_cases":sum(value==1 for value in ratios),
                                       "worst_case_ratio":max(ratios)}
    return result


def run(repetitions,output,control_directory,methods=METHODS):
    folder=Path(output)
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/"spec.json").exists():
        raise RuntimeError("Refusing to overwrite an existing guarded experiment")
    controls,control_metadata=load_controls(control_directory,repetitions)
    cases=case_matrix()
    spec={"status":"development v2 informed by v1 diagnostics; specified before these draws, not confirmation",
          "methods":list(methods),"cases":cases,"repetitions":repetitions,"delta":DELTA,"epsilon":EPSILON,
          "cap":CAP,"checkpoints":budget_schedule(),"explore_start_previous_shots":EXPLORE_START_SHOTS,
          "exploration_pair_target":"ceil(4*sqrt(current_total_budget/2)), deficit capped by current available pairs",
          "exploration_split":"ceil(deficit/2) atpi/3 thenfloor(deficit/2) at2pi/3",
          "exploit":"remainingpairs usefitupdatedaftercompletedexplorationsubblocks",
          "certificate_checkpoints":"only original common outer total-budget checkpoints",
          "evidence":"all completepairsretained; everyphysicalshotcharged",
          "prediction_order":"selectanglesandforecasts beforeeachsubblockoutcome; updatesonlyaftercompletedsubblocks",
          "shared_prefix_seed":"matching-engine v1PSRseed; exact same stream untilfirstexploration",
          "phase_names":list(PHASES),"design_config":config_dict(),"safe_config":safe_protocol_config(),
          "sources":source_hashes(),"controls":control_metadata,
          "ledger_block_schema":{"beta":["outer_look","phase","angle","n00","n01","n10","n11"],
                                 "safe":["outer_look","phase","angle","n00","n01","n10","n11",
                                         "predicted_minus_numerator","predicted_plus_numerator"],
                                 "common_denominator":BET_DENOMINATOR},
          "limits":["stationary exact single-frequency common-order model",
                    "no hidden parameters supplied to fitting, angle choice, forecasting or stopping",
                    "reused development matrix and controls; no simultaneous empirical qualification",
                    "oracle asymptotic headroom does not guarantee finite-budget SOTA improvement"]}
    (folder/"spec.json").write_text(json.dumps(spec,indent=2),encoding="utf-8")
    started=time.time()
    cells=[]
    with (folder/"trials.jsonl").open("w",encoding="utf-8") as ledger:
        for case in cases:
            local=[]
            for method in methods:
                cell,records=run_cell(case,method,repetitions)
                cells.append(cell)
                local.append(cell)
                for record in records:
                    ledger.write(json.dumps(record,separators=(",",":"),allow_nan=False)+"\n")
                ledger.flush()
            print(json.dumps({"case":case["case_id"],"means":[round(cell["mean_shots"],1) for cell in local],
                              "wrong":[cell["wrong"] for cell in local],
                              "abstained":[cell["abstained"] for cell in local],
                              "no_exploration":[cell["never_explored"] for cell in local]}),flush=True)
    summary={"spec":spec,"cells":cells,"aggregate":aggregate(cells,methods),
             "baseline_cells_reused":controls,"comparisons":compare(cells,controls),
             "new_trial_count":len(cells)*repetitions,"trial_count":len(cells)*repetitions,
             "reused_baseline_trajectories":control_metadata["reused_trajectories"],
             "elapsed_seconds":time.time()-started}
    (folder/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps({"done":True,"comparisons":summary["comparisons"],"seconds":summary["elapsed_seconds"]}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--repetitions",type=int,default=128)
    parser.add_argument("--output",default=str(ROOT/"results"/"streaming_validation"/"development_v2"))
    parser.add_argument("--controls",default=str(ROOT/"results"/"streaming_validation"/"development_v1"))
    parser.add_argument("--methods",default=",".join(METHODS))
    args=parser.parse_args()
    methods=tuple(args.methods.split(","))
    if args.repetitions<1 or not methods or len(set(methods))!=len(methods) or any(m not in METHODS for m in methods):
        parser.error("invalidrepetitionsormethods")
    run(args.repetitions,args.output,args.controls,methods)
