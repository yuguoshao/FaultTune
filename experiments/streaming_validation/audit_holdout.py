"""Independent prospective ledger audit with per-replicate seed replays.

Reconstructs all decisions, paid costs, causal angle/forecast choices, and
guarded exploration histories. Does not modify the frozen run or plan.
"""
import argparse
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.special import gammaln

from audit_streaming import (ROOT,digest,compare,independent_summary,
                             independent_aggregate,previous_evidence)
from designs import empty_state,update_pairs,choose_design,_at_shift
from safe_two_sample import predictable_probabilities,safe_log_increments,safe_log_screen
import run_streaming as psr
import run_guarded as guarded
import run_holdout as holdout


PLAN_SHA256="c54cb0776a16159b6c4f8a4ef8a2563c6d01a73e31cf40b6a3a1a572cb6a0587"
RUNNER_SHA256="efbc9a9164c4c55bd074703eb354f192f432ebfec9cb2f93248a86e3c0a296dd"


def header_check(rows,case,method,spec):
    assert len(rows)==spec["repetitions"]==128
    assert [row["rep"] for row in rows]==list(range(128))
    for row in rows:
        assert row["case_id"]==case["case_id"] and row["method"]==method
        # Independently restate the planned seed law, rather than call the runner.
        seed=210000000+case["case_id"]*1009+row["rep"]*1000003+(71 if method=="psr_safe" else 0)
        assert row["seed"]==row["replicate_seed"]==seed
        assert row["decision"] in ("accept","reject","abstain")
        assert type(row["look_index"]) is int and 1<=row["look_index"]<=len(spec["checkpoints"])
        assert type(row["shots"]) is int and row["shots"]==spec["checkpoints"][row["look_index"]-1]
        assert len(row["counts"])==4 and all(type(x) is int and x>=0 for x in row["counts"])
        assert 2*sum(row["counts"])==row["shots"]==sum(row["phase_shots"])
        if row["decision"]=="abstain":
            assert row["shots"]==spec["cap"]


def checkpoint_check(rows,active,counts,logplus,logminus,look,engine,cursors=None):
    """Recheck every floating candidate using the authoritative certificate.

    A stop cannot precede its evidence or be silently postponed after a
    certified crossing. At the cap every trajectory gets a final certificate.
    """
    if engine=="beta":
        b,c=counts[active,1],counts[active,2]
        score=gammaln(b+.5)+gammaln(c+.5)-gammaln(b+c+1)-2*gammaln(.5)+(b+c)*math.log(2)
        selected=active[score>=math.log(100)-1e-6]
    else:
        positive,negative,threshold=safe_log_screen(counts[active],logplus[active],logminus[active],delta=.01)
        selected=active[np.maximum(positive,negative)>=threshold-1e-6]
    for i in selected:
        blocks=rows[i]["blocks"][:look if cursors is None else cursors[i]]
        cert=psr.final_certificate(counts[i],blocks,engine)
        if cert["decision"]!="abstain":
            assert rows[i]["look_index"]==look,(rows[i]["case_id"],rows[i]["method"],i,look,"missed stop")
            assert rows[i]["decision"]==cert["decision"]
    return len(selected)


def audit_psr(rows,case,method,spec):
    header_check(rows,case,method,spec)
    engine="beta" if method=="psr_beta" else "safe"
    state=empty_state(128)
    counts=np.zeros((128,4),dtype=np.int64)
    logplus,logminus=np.zeros((2,128))
    blocks=screened=0
    previous=0
    for look,budget in enumerate(spec["checkpoints"],1):
        active=np.array([i for i,row in enumerate(rows) if row["look_index"]>=look],dtype=int)
        if not len(active):
            break
        entries=[rows[i]["blocks"][look-1] for i in active]
        for entry in entries:
            assert len(entry)==(5 if engine=="beta" else 7)
            assert entry[0]==math.pi/2
            assert all(type(x) is int and x>=0 for x in entry[1:])
            assert sum(entry[1:5])==(budget-previous)//2
        design=choose_design("psr",state,active)
        angles=np.array([entry[0] for entry in entries])
        assert np.array_equal(angles,design["t"])
        fresh=np.array([entry[1:5] for entry in entries],dtype=np.int64)
        if engine=="safe":
            predictions=predictable_probabilities(counts[active],design["estimated_p_minus"],design["estimated_p_plus"])
            nums=np.array([entry[5:7] for entry in entries],dtype=np.int64)
            assert np.array_equal(nums[:,0],predictions["pred_minus_numerator"])
            assert np.array_equal(nums[:,1],predictions["pred_plus_numerator"])
            increments=safe_log_increments(fresh,nums[:,0],nums[:,1],2**20)
            logplus[active]+=increments[0]
            logminus[active]+=increments[1]
        counts[active]+=fresh
        update_pairs(state,angles,fresh,active)
        assert np.all(state["shots"][active]==budget)
        blocks+=len(active)
        screened+=checkpoint_check(rows,active,counts,logplus,logminus,look,engine)
        previous=budget
    for i,row in enumerate(rows):
        assert len(row["blocks"])==row["look_index"]
        assert counts[i].tolist()==row["counts"] and state["shots"][i]==row["shots"]
        assert row["phase_shots"]==[row["shots"],0,0]
        cert=psr.final_certificate(counts[i],row["blocks"],engine)
        assert cert==row["certificate"] and cert["decision"]==row["decision"]
    return {"history_blocks":blocks,"causal_angles":blocks,"causal_forecasts":blocks if engine=="safe" else 0,
            "screen_candidates_certified":screened,"final_certificates":128,
            "abstention_certificates":sum(row["decision"]=="abstain" for row in rows),
            "exact_control_prefix_blocks":0,"exact_never_explored_controls":0}


def audit_guarded(rows,case,method,spec,controls):
    header_check(rows,case,method,spec)
    policy="paired_variance_closed" if method=="guarded_closed_beta" else "paired_kl_grid"
    state=empty_state(128)
    counts=np.zeros((128,4),dtype=np.int64)
    phases=np.zeros((128,4),dtype=np.int64)
    exploration=np.zeros(128,dtype=np.int64)
    cursors=np.zeros(128,dtype=int)
    blocks=prefix=screened=identical=0
    previous=0
    for look,budget in enumerate(spec["checkpoints"],1):
        active=np.array([i for i,row in enumerate(rows) if row["look_index"]>=look],dtype=int)
        if not len(active):
            break
        available=(budget-previous)//2
        if previous<4096:
            plans=[(0,np.full(len(active),available,dtype=np.int64))]
        else:
            # Integer ceiling avoids depending on the runner's floating sqrt.
            square=16*(budget//2)
            root=math.isqrt(square)
            target=root+int(root*root<square)
            deficit=np.clip(target-exploration[active],0,available)
            plans=[(1,(deficit+1)//2),(2,deficit//2),(3,available-deficit)]
        for phase,numbers in plans:
            present=numbers>0
            selected=active[present]
            number=numbers[present]
            if not len(selected):
                continue
            if phase==3:
                design=choose_design(policy,state,selected)
            else:
                design=choose_design("psr",state,selected)
                if phase in (1,2):
                    angle=math.pi/3 if phase==1 else 2*math.pi/3
                    design.update(_at_shift(design["fit"],np.full(len(selected),angle)))
            entries=[rows[i]["blocks"][cursors[i]] for i in selected]
            for i,n,entry in zip(selected,number,entries):
                assert len(entry)==7 and entry[0]==look and entry[1]==phase
                assert isinstance(entry[2],float) and math.isfinite(entry[2]) and 0<entry[2]<math.pi
                assert all(type(x) is int and x>=0 for x in entry[3:]) and sum(entry[3:])==n
                if phase==0:
                    assert entry[2:]==controls[i]["blocks"][look-1]
                    prefix+=1
            angles=np.array([entry[2] for entry in entries])
            assert np.array_equal(angles,design["t"]),(case["case_id"],method,look,phase,"angle")
            fresh=np.array([entry[3:7] for entry in entries],dtype=np.int64)
            counts[selected]+=fresh
            update_pairs(state,angles,fresh,selected)
            phases[selected,phase]+=2*number
            if phase in (1,2):
                exploration[selected]+=number
            cursors[selected]+=1
            blocks+=len(selected)
        assert np.all(state["shots"][active]==budget)
        for i in active:
            if cursors[i]<len(rows[i]["blocks"]):
                assert rows[i]["blocks"][cursors[i]][0]==look+1
            else:
                assert rows[i]["look_index"]==look
        screened+=checkpoint_check(rows,active,counts,None,None,look,"beta",cursors)
        previous=budget
    for i,row in enumerate(rows):
        assert cursors[i]==len(row["blocks"])
        assert counts[i].tolist()==row["counts"] and state["shots"][i]==row["shots"]
        assert phases[i].tolist()==row["phase_shots"]
        assert exploration[i]==row["exploration_pairs"]
        assert 2*exploration[i]==row["phase_shots"][1]+row["phase_shots"][2]
        cert=psr.final_certificate(counts[i],row["blocks"],"beta")
        assert cert==row["certificate"] and cert["decision"]==row["decision"]
        if exploration[i]==0:
            for key in ("counts","shots","decision","certificate","look_index","seed","replicate_seed"):
                assert row[key]==controls[i][key]
            identical+=1
    return {"history_blocks":blocks,"causal_angles":blocks,"causal_forecasts":0,
            "screen_candidates_certified":screened,"final_certificates":128,
            "abstention_certificates":sum(row["decision"]=="abstain" for row in rows),
            "exact_control_prefix_blocks":prefix,"exact_never_explored_controls":identical}


def main(directory,replay_cases,replay_reps):
    started=time.perf_counter()
    source=Path(__file__).resolve()
    source_hash=digest(source)
    directory=Path(directory).resolve()
    plan_path=directory.parent/"holdout_plan.json"
    gate_path=directory.parent/"holdout_gate.json"
    assert digest(plan_path)==PLAN_SHA256 and digest(source.with_name("run_holdout.py"))==RUNNER_SHA256
    plan=json.loads(plan_path.read_text(encoding="utf-8"))
    spec=json.loads((directory/"spec.json").read_text(encoding="utf-8"))
    summary=json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    assert spec==summary["spec"] and spec["plan_sha256"]==PLAN_SHA256
    for key,value in plan.items():
        assert spec[key]==value,key
    assert spec["gate_record_sha256"]==digest(gate_path)
    gate=json.loads(gate_path.read_text(encoding="utf-8"))
    assert gate==spec["gate"] and gate["status"]=="passed" and gate["root_released"] is True
    assert gate["primary_method"]=="guarded_closed_beta" and gate["baseline"]=="psr_beta"
    assert 0<gate["geomean_cost_ratio"]<1 and 0<gate["upper95_bootstrap"]<1
    for kind in ("summary","comparison"):
        assert digest(gate["development_"+kind+"_path"])==gate["development_"+kind+"_sha256"]
    assert plan["cases"]==holdout.holdout_cases() and plan["checkpoints"]==psr.budget_schedule()
    assert plan["repetitions"]==128 and plan["cap"]==2**22 and plan["delta"]==.01
    assert plan["guard"]["explore_start_previous_shots"]==4096 and plan["guard"]["sqrt_multiplier"]==4
    for name,expected in plan["sources"].items():
        assert digest(ROOT/name)==expected,name
    seeds=[210000000+case["case_id"]*1009+rep*1000003+offset
           for case in plan["cases"] for rep in range(128) for offset in (0,71)]
    assert len(seeds)==len(set(seeds))
    prior=previous_evidence()
    expected_groups=[(case["case_id"],method) for case in spec["cases"] for method in spec["methods"]]
    cases={case["case_id"]:case for case in spec["cases"]}
    cells={(cell["case_id"],cell["method"]):cell for cell in summary["cells"]}
    assert len(cells)==len(expected_groups)==224
    counters={key:0 for key in ("history_blocks","causal_angles","causal_forecasts","screen_candidates_certified",
                               "final_certificates","abstention_certificates","exact_control_prefix_blocks",
                               "exact_never_explored_controls")}
    checked=[]
    replay={}
    error_replays=set()
    controls=None
    record_count=0
    with (directory/"trials.jsonl").open(encoding="utf-8") as stream:
        groups=itertools.groupby((json.loads(line) for line in stream),key=lambda row:(row["case_id"],row["method"]))
        for index,(key,group) in enumerate(groups):
            assert key==expected_groups[index]
            rows=list(group)
            cid,method=key
            if method=="psr_beta":
                controls=rows
            local=(audit_guarded(rows,cases[cid],method,spec,controls) if method.startswith("guarded_")
                   else audit_psr(rows,cases[cid],method,spec))
            for name,value in local.items():
                counters[name]+=value
            independent=independent_summary(rows,cases[cid],method)
            if method.startswith("guarded_"):
                independent["mean_exploration_shots"]=math.fsum(2*row["exploration_pairs"] for row in rows)/128
                independent["never_explored"]=sum(row["exploration_pairs"]==0 for row in rows)
            compare(cells[key],independent,str(key))
            checked.append(independent)
            if cid in replay_cases:
                for rep in replay_reps:
                    replay[cid,method,rep]=rows[rep]
            expected="accept" if cases[cid]["C"]<0 else "reject" if cases[cid]["C"]>0 else "abstain"
            for row in rows:
                if row["decision"] not in (expected,"abstain"):
                    key=(cid,method,row["rep"])
                    replay[key]=row
                    error_replays.add(key)
            record_count+=len(rows)
            if (index+1)%4==0:
                print(json.dumps({"audited_holdout_case":cid,"records":record_count}),flush=True)
    assert len(checked)==224 and record_count==summary["trial_count"]==summary["new_trial_count"]==28672
    assert summary["reused_baseline_trajectories"]==0
    compare(summary["aggregate"],independent_aggregate(checked,spec["methods"]),"aggregate")
    replayed=0
    for cid,method,rep in sorted(replay):
        seed=210000000+cid*1009+rep*1000003+(71 if method=="psr_safe" else 0)
        runner=guarded.run_cell if method.startswith("guarded_") else psr.run_cell
        unused,rows=runner(cases[cid],method,1,seed)
        assert len(rows)==1
        rows[0]["rep"]=rep
        rows[0]["replicate_seed"]=seed
        assert rows[0]==replay[cid,method,rep],(cid,method,rep,"seed replay")
        replayed+=1
    print(json.dumps({"seed_replay_records":replayed,"all_error_records_replayed":len(error_replays)}),flush=True)
    assert digest(source)==source_hash
    result={"status":"passed","ledger_records":record_count,"new_trajectories":record_count,
            "reused_trajectories":0,"cells_and_aggregates_recomputed":len(checked),
            "independent_seed_values_verified":len(seeds),"seed_replay_cases":replay_cases,
            "seed_replay_reps":replay_reps,"exact_seed_replay_records":replayed,
            "all_error_records_replayed":len(error_replays),"error_replay_keys":[list(key) for key in sorted(error_replays)],
            **counters,**prior,"pinned_run_sources_verified":len(spec["sources"]),
            "plan_sha256":PLAN_SHA256,"gate_sha256":digest(gate_path),
            "audit_source_sha256":source_hash,"audit_source_unchanged_during_execution":True,
            "helper_source_sha256":digest(source.with_name("audit_streaming.py")),
            "review":{"costs":"All pairs and all exploration subblocks retained and charged.",
                      "predictability":"Every angle and safe forecast reconstructed before adding the corresponding observations.",
                      "exploration":"Integer target, deficits, ordering, and per-phase paid costs reconstructed independently.",
                      "stopping":"Common outer checkpoints only; each screen candidate checked for the first certified crossing.",
                      "coupling":"Before exploration, guarded beta and PSR beta histories match exactly within each independent replicate.",
                      "scope":"Implementation and reproducibility audit of the frozen synthetic holdout; no complete SOTA comparison."},
            "artifact_sha256":{name:digest(directory/name) for name in ("spec.json","summary.json","trials.jsonl")},
            "elapsed_seconds":time.perf_counter()-started}
    (directory/"audit.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",default=str(ROOT/"results"/"streaming_validation"/"holdout_v1"))
    parser.add_argument("--replay-cases",default="0,4,10,23,36,47,48,55")
    parser.add_argument("--replay-reps",default="0,63,127")
    args=parser.parse_args()
    main(args.directory,[int(x) for x in args.replay_cases.split(",")],
         [int(x) for x in args.replay_reps.split(",")])
