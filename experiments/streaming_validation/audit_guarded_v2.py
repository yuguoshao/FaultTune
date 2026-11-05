"""Independent audit of guarded exploration, with an exclusive filename."""
import argparse
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np

from audit_streaming import (ROOT,digest,compare,independent_summary,
                             independent_aggregate,previous_evidence)
from designs import empty_state,update_pairs,choose_design,_at_shift,config_dict
from safe_two_sample import predictable_probabilities,certificate_from_blocks as safe_certificate
import run_guarded as runner


def audit_group(rows,case,method,spec,controls):
    repetitions=spec["repetitions"]
    assert len(rows)==repetitions
    assert [row["rep"] for row in rows]==list(range(repetitions))
    policy,engine=runner.protocol(method)
    baseline=controls["psr_"+engine]
    schedule=spec["checkpoints"]
    state=empty_state(repetitions)
    counts=np.zeros((repetitions,4),dtype=np.int64)
    phases=np.zeros((repetitions,4),dtype=np.int64)
    exploration=np.zeros(repetitions,dtype=np.int64)
    cursors=np.zeros(repetitions,dtype=int)
    blocks=prefix_blocks=identical_stops=0
    for row in rows:
        assert row["case_id"]==case["case_id"] and row["method"]==method
        assert row["seed"]==runner.seed_for(case,method)
        assert row["decision"] in ("accept","reject","abstain")
        assert type(row["look_index"]) is int and 1<=row["look_index"]<=len(schedule)
        assert row["shots"]==schedule[row["look_index"]-1]
        assert len(row["counts"])==4 and all(type(x) is int and x>=0 for x in row["counts"])
        assert 2*sum(row["counts"])==row["shots"]
        assert row["shots"]==sum(row["phase_shots"])
        if row["decision"]=="abstain":
            assert row["shots"]==spec["cap"]
    previous=0
    for look,budget in enumerate(schedule,1):
        active=np.array([i for i,row in enumerate(rows) if row["look_index"]>=look],dtype=int)
        if not len(active):
            break
        available=(budget-previous)//2
        if previous<spec["explore_start_previous_shots"]:
            plans=[(0,np.full(len(active),available,dtype=np.int64))]
        else:
            # Exact ceil(4*sqrt(budget/2)); independent of floating sqrt.
            squared=16*(budget//2)
            root=math.isqrt(squared)
            target=root+(root*root<squared)
            deficit=np.clip(target-exploration[active],0,available)
            plans=[(1,(deficit+1)//2),(2,deficit//2),(3,available-deficit)]
        for phase,numbers in plans:
            mask=numbers>0
            selected=active[mask]
            number=numbers[mask]
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
            for j,(i,entry) in enumerate(zip(selected,entries)):
                assert len(entry)==(7 if engine=="beta" else 9)
                assert entry[0]==look and entry[1]==phase
                assert all(type(x) is int and x>=0 for x in entry[3:])
                assert isinstance(entry[2],float) and 0<entry[2]<math.pi
                assert sum(entry[3:7])==number[j]
                if phase==0:
                    original=baseline[i]["blocks"][look-1]
                    assert entry[2:]==original,(case["case_id"],method,i,look,"prefix")
                    prefix_blocks+=1
            angles=np.array([entry[2] for entry in entries])
            assert np.array_equal(angles,design["t"]),(case["case_id"],method,look,phase,"angle")
            fresh=np.array([entry[3:7] for entry in entries],dtype=np.int64)
            if engine=="safe":
                predicted=predictable_probabilities(counts[selected],design["estimated_p_minus"],
                                                     design["estimated_p_plus"])
                assert np.array_equal(np.array([entry[7] for entry in entries]),
                                      predicted["pred_minus_numerator"])
                assert np.array_equal(np.array([entry[8] for entry in entries]),
                                      predicted["pred_plus_numerator"])
            counts[selected]+=fresh
            update_pairs(state,angles,fresh,selected)
            phases[selected,phase]+=2*number
            if phase in (1,2):
                exploration[selected]+=number
            cursors[selected]+=1
            blocks+=len(selected)
        assert np.all(state["shots"][active]==budget)
        # No unaccounted subblock or stopping between subblocks is possible.
        for i in active:
            if cursors[i]<len(rows[i]["blocks"]):
                assert rows[i]["blocks"][cursors[i]][0]==look+1
            else:
                assert rows[i]["look_index"]==look
        previous=budget
    certificates=abstentions=0
    for i,row in enumerate(rows):
        assert cursors[i]==len(row["blocks"])
        assert counts[i].tolist()==row["counts"]
        assert state["shots"][i]==row["shots"]
        assert phases[i].tolist()==row["phase_shots"]
        assert exploration[i]==row["exploration_pairs"]
        assert 2*exploration[i]==row["phase_shots"][1]+row["phase_shots"][2]
        if engine=="beta":
            cert=runner.certificate_from_pairs(tuple(row["counts"]),delta=spec["delta"])
        else:
            expanded=[{"counts":entry[3:7],"pred_minus_numerator":entry[7],
                       "pred_plus_numerator":entry[8],"denominator":runner.BET_DENOMINATOR}
                      for entry in row["blocks"]]
            cert=safe_certificate(expanded,delta=spec["delta"])
        assert cert==row["certificate"] and cert["decision"]==row["decision"]
        certificates+=1
        abstentions+=row["decision"]=="abstain"
        if exploration[i]==0:
            for key in ("counts","shots","decision","certificate","look_index","seed"):
                assert row[key]==baseline[i][key],(case["case_id"],method,i,key)
            identical_stops+=1
    return {"history_subblocks_reconstructed":blocks,"final_certificates_recomputed":certificates,
            "full_cap_abstention_certificates_recomputed":abstentions,
            "prefix_blocks_exactly_matched_controls":prefix_blocks,
            "no_exploration_trajectories_exactly_matched_controls":identical_stops}


def control_groups(path):
    with Path(path).open(encoding="utf-8") as stream:
        grouped=itertools.groupby((json.loads(line) for line in stream),key=lambda row:row["case_id"])
        for cid,group in grouped:
            controls={"psr_beta":[],"psr_safe":[]}
            for row in group:
                if row["method"] in controls:
                    controls[row["method"]].append(row)
            yield cid,controls


def main(directory,replay_ids):
    started=time.perf_counter()
    source=Path(__file__).resolve()
    original_hash=digest(source)
    directory=Path(directory).resolve()
    spec=json.loads((directory/"spec.json").read_text(encoding="utf-8"))
    summary=json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    assert spec==summary["spec"]
    assert spec["cases"]==runner.case_matrix()
    assert spec["checkpoints"]==runner.budget_schedule()
    assert spec["design_config"]==config_dict()
    assert spec["explore_start_previous_shots"]==4096
    assert spec["delta"]==.01 and spec["cap"]==2**22
    for name,expected in spec["sources"].items():
        assert digest(ROOT/name)==expected,name
    controls_path=Path(spec["controls"]["source_directory"])
    for name,expected in spec["controls"]["source_sha256"].items():
        assert digest(controls_path/name)==expected,name
    prior=previous_evidence()
    original_summary=json.loads((controls_path/"summary.json").read_text(encoding="utf-8"))
    original_cells=[cell for cell in original_summary["cells"] if cell["method"] in ("psr_beta","psr_safe")]
    assert original_cells==summary["baseline_cells_reused"]
    assert summary["reused_baseline_trajectories"]==len(original_cells)*spec["repetitions"]
    expected_groups=[(case["case_id"],method) for case in spec["cases"] for method in spec["methods"]]
    case_lookup={case["case_id"]:case for case in spec["cases"]}
    cells={(cell["case_id"],cell["method"]):cell for cell in summary["cells"]}
    assert len(cells)==len(expected_groups)
    controls_iterator=iter(control_groups(controls_path/"trials.jsonl"))
    current_case=None
    counters={key:0 for key in ("history_subblocks_reconstructed","final_certificates_recomputed",
                                "full_cap_abstention_certificates_recomputed","prefix_blocks_exactly_matched_controls",
                                "no_exploration_trajectories_exactly_matched_controls")}
    checked_cells=[]
    replay={}
    count=0
    with (directory/"trials.jsonl").open(encoding="utf-8") as stream:
        grouped=itertools.groupby((json.loads(line) for line in stream),key=lambda row:(row["case_id"],row["method"]))
        for index,(key,group) in enumerate(grouped):
            assert key==expected_groups[index]
            if current_case!=key[0]:
                cid,control_rows=next(controls_iterator)
                assert cid==key[0]
                current_case=cid
                for rows in control_rows.values():
                    assert [row["rep"] for row in rows]==list(range(spec["repetitions"]))
            rows=list(group)
            local=audit_group(rows,case_lookup[key[0]],key[1],spec,control_rows)
            for field,value in local.items():
                counters[field]+=value
            independent=independent_summary(rows,case_lookup[key[0]],key[1])
            independent["mean_exploration_shots"]=math.fsum(2*row["exploration_pairs"] for row in rows)/len(rows)
            independent["never_explored"]=sum(row["exploration_pairs"]==0 for row in rows)
            compare(cells[key],independent,str(key))
            checked_cells.append(independent)
            if key[0] in replay_ids:
                replay[key]=rows
            count+=len(rows)
            if (index+1)%len(spec["methods"])==0:
                print(json.dumps({"audited_guarded_case":key[0],"records":count}),flush=True)
    assert len(checked_cells)==len(expected_groups)
    assert count==summary["trial_count"]==summary["new_trial_count"]==len(expected_groups)*spec["repetitions"]
    compare(summary["aggregate"],independent_aggregate(checked_cells,spec["methods"]),"aggregate")
    # Comparisons use independently checked new cells and immutable controls.
    compare(summary["comparisons"],runner.compare(checked_cells,original_cells),"comparisons")
    replays=0
    for cid in replay_ids:
        for method in spec["methods"]:
            cell,rows=runner.run_cell(case_lookup[cid],method,spec["repetitions"])
            assert rows==replay[cid,method],(cid,method,"seed replay")
            compare(cell,cells[cid,method],str((cid,method)))
            replays+=len(rows)
        print(json.dumps({"replayed_guarded_case":cid,"records":replays}),flush=True)
    assert digest(source)==original_hash
    result={"status":"passed","ledger_records":count,"new_trajectories":count,
            "referenced_baseline_trajectories":summary["reused_baseline_trajectories"],
            "cells_and_aggregates_recomputed":len(checked_cells),"seed_replay_case_ids":replay_ids,
            "vector_seed_replay_records":replays,**counters,**prior,
            "run_source_hashes_verified":len(spec["sources"]),
            "review":{"predictability":"Every subblock angle/forecast checked before adding its observations.",
                      "exploration":"Exact integer sqrt target, deficit, split, and paid subblock costs reconstructed.",
                      "stopping":"Only common outer checkpoints; no free stopping inside exploration.",
                      "controls":"Complete pre-exploration blocks and never-explored trajectories match same-engine controls exactly.",
                      "scope":"Development implementation audit, not independent SOTA confirmation."},
            "elapsed_seconds":time.perf_counter()-started,
            "audit_source_path":str(source.relative_to(ROOT)).replace("\\","/"),
            "audit_source_sha256":original_hash,"audit_source_unchanged_during_execution":True,
            "audit_helper_source_sha256":digest(source.with_name("audit_streaming.py")),
            "artifact_sha256":{name:digest(directory/name) for name in ("spec.json","summary.json","trials.jsonl")}}
    (directory/"audit.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--directory",default=str(ROOT/"results"/"streaming_validation"/"development_v2"))
    parser.add_argument("--replay-cases",default="0,42,49")
    args=parser.parse_args()
    main(args.directory,[int(x) for x in args.replay_cases.split(",") if x])
