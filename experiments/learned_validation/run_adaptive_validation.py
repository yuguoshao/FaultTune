"""Development v2: common early PSR exit before a paid learned design.

V1 code/results are immutable inputs. V2 is a newly specified exploration,
motivated by v1 pilot overhead, and is not a held-out confirmation of v1.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.special import gammaln

from adaptive_protocol import (EARLY_RISK,HOLDOUT_RISK,POLICIES,early_schedule,
                               early_route,complete_pilot,holdout_design)
from certificates import fisher_certificate,fisher_log_tail_screen
from designs import config_dict
from run_validation import (ROOT,METHODS,CAP,EPSILON,case_matrix,budget_schedule,
                            probability,summarize_records,aggregate)


def routing_for_case(case,repetitions):
    """Simulate one common paid routing/pilot transcript for every wrapped policy."""
    seed=51000000+case["case_id"]*1009
    axis_seed=71000000+case["case_id"]*1009
    rng=np.random.RandomState(seed)
    pairs=np.zeros((repetitions,4),dtype=np.int64)
    active=np.ones(repetitions,dtype=bool)
    decisions=["abstain"]*repetitions
    certs=[None]*repetitions
    looks=np.zeros(repetitions,dtype=int)
    pm,pp=float(probability(case,-math.pi/2)),float(probability(case,math.pi/2))
    joint=np.array([(1-pm)*(1-pp),(1-pm)*pp,pm*(1-pp),pm*pp])
    joint/=joint.sum()
    for look,shots in enumerate(early_schedule(),1):
        indices=np.flatnonzero(active)
        if not len(indices):
            break
        increment=shots//2-int(pairs[indices[0]].sum())
        pairs[indices]+=rng.multinomial(increment,joint,size=len(indices))
        looks[indices]=look
        worse,better=pairs[indices,1],pairs[indices,2]
        evidence=(gammaln(worse+.5)+gammaln(better+.5)-gammaln(worse+better+1)
                  -2*gammaln(.5)+(worse+better)*math.log(2))
        selected=indices if shots==512 else indices[evidence>=math.log(1/EARLY_RISK)-1e-6]
        for i in selected:
            route=early_route(pairs[i])
            certs[i]=route["certificate"]
            if route["action"]=="stop":
                active[i]=False
                decisions[i]=route["decision"]
    counts=np.stack((np.zeros(repetitions,dtype=int),pairs[:,1]+pairs[:,3],
                     np.zeros(repetitions,dtype=int),pairs[:,2]+pairs[:,3]),axis=-1)
    n=pairs.sum(axis=-1)
    totals=np.stack((np.zeros(repetitions,dtype=int),n,np.zeros(repetitions,dtype=int),n),axis=-1)
    hard=np.flatnonzero(active)
    axes=np.zeros((repetitions,2),dtype=np.int64)
    if len(hard):
        axis_rng=np.random.RandomState(axis_seed)
        axes[hard]=axis_rng.binomial(256,probability(case,np.array([0.,math.pi])),size=(len(hard),2))
        counts[hard],totals[hard]=complete_pilot(pairs[hard],axes[hard])
    return {"pair_counts":pairs,"axis_successes":axes,"pilot_counts":counts,"pilot_totals":totals,
            "active":active,"decisions":decisions,"certificates":certs,"early_looks":looks,
            "early_seed":seed,"axis_seed":axis_seed}


def run_cell(case,method,repetitions,routing=None):
    if method not in POLICIES:
        raise ValueError("run_cell executes only wrapped policies")
    if routing is None:
        routing=routing_for_case(case,repetitions)
    active=routing["active"].copy()
    hard=np.flatnonzero(active)
    t=np.full(repetitions,math.pi/2)
    fraction=np.full(repetitions,.5)
    variance=[None]*repetitions
    shift_clipped=np.zeros(repetitions,dtype=bool)
    allocation_clipped=np.zeros(repetitions,dtype=bool)
    started=time.perf_counter()
    if len(hard):
        design=holdout_design(method,routing["pilot_counts"][hard],routing["pilot_totals"][hard])
        t[hard],fraction[hard]=design["t"],design["allocation_plus"]
        for j,i in enumerate(hard):
            variance[i]=float(design["estimated_variance"][j])
        shift_clipped[hard],allocation_clipped[hard]=design["shift_was_clipped"],design["allocation_was_clipped"]
    planning_seconds=time.perf_counter()-started
    # Simulator truth supplies draws only; it never reaches a controller or certificate.
    p_minus,p_plus=probability(case,-t),probability(case,t)
    seed=91000000+case["case_id"]*1009+METHODS.index(method)*71
    rng=np.random.RandomState(seed)
    s_minus,s_plus=np.zeros((2,repetitions),dtype=np.int64)
    n_minus,n_plus=np.zeros((2,repetitions),dtype=np.int64)
    decisions=list(routing["decisions"])
    certificates=[c if d!="abstain" else None for c,d in zip(routing["certificates"],decisions)]
    final_look=np.zeros(repetitions,dtype=int)
    schedule=budget_schedule()
    for look,budget in enumerate(schedule,1):
        if budget<=1026:
            continue
        indices=np.flatnonzero(active)
        if not len(indices):
            break
        stage2=budget-1024
        target_plus=np.clip(np.floor(fraction[indices]*stage2).astype(np.int64),1,stage2-1)
        target_minus=stage2-target_plus
        s_minus[indices]+=rng.binomial((target_minus-n_minus[indices]).astype(np.int32),p_minus[indices])
        s_plus[indices]+=rng.binomial((target_plus-n_plus[indices]).astype(np.int32),p_plus[indices])
        n_minus[indices],n_plus[indices]=target_minus,target_plus
        final_look[indices]=look
        selected=[]
        for i in indices:
            screen,threshold=fisher_log_tail_screen(int(s_minus[i]),int(n_minus[i]),int(s_plus[i]),int(n_plus[i]),
                                                   look_count=len(schedule),delta=HOLDOUT_RISK)
            if screen<=threshold+.05 or look==len(schedule):
                selected.append(i)
        for i in selected:
            cert=fisher_certificate(int(s_minus[i]),int(n_minus[i]),int(s_plus[i]),int(n_plus[i]),
                                    look,len(schedule),delta=HOLDOUT_RISK)
            if cert["decision"]!="abstain":
                decisions[i],certificates[i],active[i]=cert["decision"],cert,False
    records=[]
    for i in range(repetitions):
        pilot_shots=int(routing["pilot_totals"][i].sum())
        phase="holdout" if routing["active"][i] else "early"
        record={"case_id":case["case_id"],"method":method,"rep":i,"phase":phase,
                "validation_seed":seed,"early_seed":routing["early_seed"],"axis_seed":routing["axis_seed"],
                "pilot_counts":[int(x) for x in routing["pilot_counts"][i]],
                "pilot_totals":[int(x) for x in routing["pilot_totals"][i]],"pilot_shots":pilot_shots,
                "early_pair_counts":[int(x) for x in routing["pair_counts"][i]],
                "early_look_index":int(routing["early_looks"][i]),
                "early_certificate":routing["certificates"][i],
                "axis_successes":[int(x) for x in routing["axis_successes"][i]],
                "axis_shots":512 if phase=="holdout" else 0,
                "t":float(t[i]),"allocation_plus":float(fraction[i]),"estimated_variance":variance[i],
                "counts":[int(s_minus[i]),int(n_minus[i]),int(s_plus[i]),int(n_plus[i])],
                "pair_counts":None,"shots":pilot_shots+int(n_minus[i]+n_plus[i]),
                "look_index":int(final_look[i]) if phase=="holdout" else int(routing["early_looks"][i]),
                "decision":decisions[i],"certificate":certificates[i],
                "early_risk":EARLY_RISK,"holdout_risk":HOLDOUT_RISK,"total_risk_bound":.01}
        records.append(record)
    cell=summarize_records(records,case,method)
    cell.update({"validation_seed":seed,"early_seed":routing["early_seed"],"axis_seed":routing["axis_seed"],
                 "early_stops":int(np.sum(~routing["active"])),"entered_holdout":len(hard),
                 "planning_seconds_batch":planning_seconds,"mean_t":float(np.mean(t)),
                 "mean_allocation_plus":float(np.mean(fraction)),
                 "shift_clip_count":int(np.sum(shift_clipped)),"allocation_clip_count":int(np.sum(allocation_clipped))})
    return cell,records


def load_baselines(path,repetitions):
    path=Path(path)
    summary=json.loads((path/"summary.json").read_text(encoding="utf-8"))
    if repetitions>summary["spec"]["repetitions"]:
        raise ValueError("baseline source has fewer repetitions")
    if summary["spec"]["cases"]!=case_matrix():
        raise ValueError("baseline case matrix differs")
    rows={}
    with (path/"trials.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row=json.loads(line)
            if row["method"] in METHODS[:2] and row["rep"]<repetitions:
                row["phase"]="baseline_reuse"
                rows.setdefault((row["case_id"],row["method"]),[]).append(row)
    if any(len(rows.get((c["case_id"],m),[]))!=repetitions for c in case_matrix() for m in METHODS[:2]):
        raise ValueError("baseline records are incomplete")
    return rows,{"path":str(path.resolve()),"original_repetitions":summary["spec"]["repetitions"],
                 "sha256":{n:hashlib.sha256((path/n).read_bytes()).hexdigest() for n in ("spec.json","summary.json","trials.jsonl")}}


def run(repetitions,output,baseline):
    path=Path(output)
    path.mkdir(parents=True,exist_ok=True)
    if (path/"spec.json").exists():
        raise RuntimeError("Refusing to overwrite an existing run")
    controls,baseline_info=load_baselines(baseline,repetitions)
    cases=case_matrix()
    names=("adaptive_protocol.py","run_adaptive_validation.py","run_validation.py","designs.py","kl_designs.py","certificates.py")
    hashes={str((Path(__file__).resolve().parent/n).relative_to(ROOT)).replace("\\","/"):
            hashlib.sha256((Path(__file__).resolve().parent/n).read_bytes()).hexdigest() for n in names}
    spec={"status":"development v2 specified after v1 overhead finding, before v2 samples; not confirmation",
          "repetitions":repetitions,"cases":cases,"methods":list(METHODS),"epsilon":EPSILON,"delta":.01,
          "early_risk":EARLY_RISK,"holdout_risk":HOLDOUT_RISK,"early_checkpoints":early_schedule(),
          "early_max_shots":512,"pilot_per_angle":256,"total_completed_pilot_shots":1024,
          "total_shot_cap":CAP,"total_shot_checkpoints":budget_schedule(),"design_config":config_dict(),
          "source_sha256":hashes,"baseline_reuse":baseline_info,
          "routing":"same early PSR paired mixture for all four learned policies; survivors complete 0/pi pilot; independent holdout",
          "risk_proof":"P(wrong early)<=.005; conditional on any surviving routing/pilot history P(wrong holdout)<=.005; union bound <=.01, including equality",
          "fisher_spending":".005/(2L), L includes skipped pre-pilot total-cost checkpoints",
          "cost_scope":"all early, pilot, holdout shots charged; errors and full-cap abstentions retained",
          "limitations":["synthetic stationary exact single-frequency model","no uniform finite-cap power claim",
                         "KL grid optimizes two-angle plug-in design, not a full adaptive structured SOTA algorithm",
                         "baseline runs reused, no independent resampling of controls"]}
    (path/"spec.json").write_text(json.dumps(spec,indent=2,allow_nan=False),encoding="utf-8")
    started=time.time()
    cells=[]
    with (path/"trials.jsonl").open("w",encoding="utf-8") as stream:
        for case in cases:
            routing=routing_for_case(case,repetitions)
            local=[]
            for method in METHODS:
                if method in METHODS[:2]:
                    records=controls[case["case_id"],method]
                    cell=summarize_records(records,case,method)
                    cell["baseline_reused"]=True
                else:
                    cell,records=run_cell(case,method,repetitions,routing)
                cells.append(cell)
                local.append(cell)
                for record in records:
                    stream.write(json.dumps(record,separators=(",",":"),allow_nan=False)+"\n")
                stream.flush()
            print(json.dumps({"case":case["case_id"],"kind":case["kind"],
                              "early_stops":int(np.sum(~routing["active"])),
                              "means":[round(c["mean_shots"],1) for c in local],
                              "wrong":[c["wrong"] for c in local],"abstained":[c["abstained"] for c in local]}),flush=True)
    summary={"spec":spec,"cells":cells,"aggregate":aggregate(cells,cases),
             "trial_count":len(cells)*repetitions,"new_wrapped_trials":len(cases)*len(POLICIES)*repetitions,
             "reused_baseline_trials":len(cases)*2*repetitions,"elapsed_seconds":time.time()-started}
    (path/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"done":True,"aggregate":summary["aggregate"]["geomean_cost_ratios_baseline_over_closed"],
                      "seconds":summary["elapsed_seconds"]}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions",type=int,default=128)
    parser.add_argument("--output",default=str(ROOT/"results"/"learned_validation"/"development_v2"))
    parser.add_argument("--baseline",default=str(ROOT/"results"/"learned_validation"/"development_v1"))
    args=parser.parse_args()
    if args.repetitions<1:
        parser.error("repetitions must be positive")
    run(args.repetitions,args.output,args.baseline)
