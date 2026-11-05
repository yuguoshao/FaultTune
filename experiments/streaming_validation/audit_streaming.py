"""Independent streaming ledger audit; previous evidence is read-only.

This file is owned by one auditor. It reconstructs angles and forecasts from
prior counts before updating each block, recomputes directed certificates,
and replays complete vector cells using their registered seeds.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
import zipfile

import numpy as np
from scipy.stats import beta

from benchmark import ROOT
from designs import empty_state,update_pairs,choose_design,_at_shift,config_dict
from betting import (predictable_bets,kelly_log_increments,hedge_log_screen,
                     certificate_from_blocks as betting_certificate,protocol_config)
from safe_two_sample import (predictable_probabilities,safe_log_increments,safe_log_screen,
                             certificate_from_blocks as safe_certificate)
import run_streaming as runner


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual,expected,tolerance=2e-12):
    assert math.isclose(float(actual),float(expected),rel_tol=tolerance,abs_tol=tolerance),(actual,expected)


def compare(actual,expected,path=""):
    if isinstance(expected,dict):
        assert actual.keys()==expected.keys(),path
        for key,value in expected.items():
            compare(actual[key],value,path+"/"+str(key))
    elif isinstance(expected,list):
        assert len(actual)==len(expected),path
        for index,value in enumerate(expected):
            compare(actual[index],value,path+"/"+str(index))
    elif isinstance(expected,float):
        close(actual,expected)
    else:
        assert actual==expected,(path,actual,expected)


def interval(k,n):
    return [float(beta.ppf(.025,k,n-k+1)) if k else 0.,
            float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]


def independent_summary(rows,case,method):
    expected="accept" if case["C"]<0 else "reject" if case["C"]>0 else "abstain"
    costs=[row["shots"] for row in rows]
    wrong=sum(row["decision"] not in ("abstain",expected) for row in rows)
    correct=sum(row["decision"]==expected for row in rows) if expected!="abstain" else 0
    abstain=sum(row["decision"]=="abstain" for row in rows)
    successful=sorted(row["shots"] for row in rows if expected!="abstain" and row["decision"]==expected)
    rank=int(math.ceil(.95*len(rows)))
    n95=successful[rank-1] if len(successful)>=rank else None
    return {"case_id":case["case_id"],"kind":case["kind"],"method":method,"repetitions":len(rows),
            "wrong":wrong,"correct":correct,"abstained":abstain,"error_ci95_pointwise":interval(wrong,len(rows)),
            "power_ci95_pointwise":interval(correct,len(rows)) if expected!="abstain" else None,
            "mean_shots":math.fsum(costs)/len(costs),"median_shots":float(np.median(costs)),
            "min_shots":min(costs),"max_shots":max(costs),
            "empirical_N95_correct":n95,"N95_censored":n95 is None}


def independent_aggregate(cells,methods):
    result={}
    for method in methods:
        local=[cell for cell in cells if cell["method"]==method]
        main=[cell for cell in local if cell["kind"]=="main"]
        result[method]={"main_geomean_shots":math.exp(math.fsum(math.log(cell["mean_shots"]) for cell in main)/len(main)),
                        "main_wrong":sum(cell["wrong"] for cell in main),
                        "main_abstained":sum(cell["abstained"] for cell in main),
                        "main_trials":sum(cell["repetitions"] for cell in main),
                        "all_wrong":sum(cell["wrong"] for cell in local),
                        "null_decisions":sum(cell["wrong"] for cell in local if cell["kind"]=="null"),
                        "weak_abstained":sum(cell["abstained"] for cell in local if cell["kind"]=="weak"),
                        "main_N95_censored_cases":sum(cell["N95_censored"] for cell in main)}
    if "psr_beta" in result:
        base=result["psr_beta"]["main_geomean_shots"]
        for value in result.values():
            value["cost_over_psr_beta"]=value["main_geomean_shots"]/base
    return result


def previous_evidence():
    path=ROOT/"results"/"learned_validation"/"evidence_manifest.json"
    manifest=json.loads(path.read_text(encoding="utf-8"))
    for name,expected in manifest["files"].items():
        assert digest(ROOT/name)==expected,name
    source_archive=ROOT/"experiments"/"circuit_validation"/"confirmation_spec_v1_sources.zip"
    sources=0
    with zipfile.ZipFile(str(source_archive)) as archive:
        for name in archive.namelist():
            if not name.endswith(".py"):
                continue
            target=ROOT/"experiments"/"circuit_validation"/name
            if not target.is_file():
                target=ROOT/name
            assert target.is_file(),name
            assert hashlib.sha256(archive.read(name)).hexdigest()==digest(target),name
            sources+=1
    return {"previous_learned_manifest_files_unchanged":len(manifest["files"]),
            "previous_learned_manifest_sha256":digest(path),
            "old_frozen_python_sources_unchanged":sources}


def audit_group(rows,case,method,spec):
    """Reconstruct a whole vector batch from compact per-trial histories."""
    repetitions=spec["repetitions"]
    assert len(rows)==repetitions
    assert [row["rep"] for row in rows]==list(range(repetitions))
    policy,engine=runner.protocol(method)
    seed=130000000+case["case_id"]*1009+runner.METHODS.index(method)*71
    schedule=spec["checkpoints"]
    state=empty_state(repetitions)
    total=np.zeros((repetitions,4),dtype=np.int64)
    phases=np.zeros((repetitions,3),dtype=np.int64)
    logplus,logminus=np.zeros((2,repetitions))
    blocks_checked,certificates,abstentions=0,0,0
    for row in rows:
        assert row["case_id"]==case["case_id"] and row["method"]==method and row["seed"]==seed
        assert row["decision"] in ("accept","reject","abstain")
        assert type(row["shots"]) is int and 0<row["shots"]<=spec["cap"]
        assert type(row["look_index"]) is int and row["look_index"]==len(row["blocks"])
        assert 1<=row["look_index"]<=len(schedule)
        assert row["shots"]==schedule[row["look_index"]-1]
        assert len(row["counts"])==4 and all(type(n) is int and n>=0 for n in row["counts"])
        assert row["shots"]==2*sum(row["counts"])
        assert len(row["phase_shots"])==3 and sum(row["phase_shots"])==row["shots"]
        if row["decision"]=="abstain":
            assert row["shots"]==spec["cap"]
    previous=0
    for look,budget in enumerate(schedule,1):
        active=np.array([i for i,row in enumerate(rows) if row["look_index"]>=look],dtype=int)
        if not len(active):
            break
        entries=[rows[i]["blocks"][look-1] for i in active]
        for entry in entries:
            assert len(entry)==(5 if engine=="beta" else 7)
            assert isinstance(entry[0],float) and math.isfinite(entry[0]) and 0<entry[0]<math.pi
            assert all(type(n) is int and n>=0 for n in entry[1:])
            assert sum(entry[1:5])==(budget-previous)//2
        # Independent selection logic: predictions see only the old state.
        if policy=="psr" or previous<256:
            prediction=choose_design("psr",state,active)
            phase=0
        elif previous<512:
            prediction=choose_design("psr",state,active)
            angle=math.pi/3 if previous<384 else 2*math.pi/3
            prediction.update(_at_shift(prediction["fit"],np.full(len(active),angle)))
            phase=1
        else:
            prediction=choose_design(policy,state,active)
            phase=2
        actual_angles=np.array([entry[0] for entry in entries])
        assert np.array_equal(actual_angles,prediction["t"]),(case["case_id"],method,look,"angle")
        fresh=np.array([entry[1:5] for entry in entries],dtype=np.int64)
        if engine!="beta":
            nums=np.array([entry[5:7] for entry in entries],dtype=np.int64)
            if engine=="bet":
                predicted=predictable_bets(total[active],prediction["estimated_p_minus"],prediction["estimated_p_plus"])
                assert np.array_equal(nums[:,0],predicted["plus_numerator"])
                assert np.array_equal(nums[:,1],predicted["minus_numerator"])
                increments=kelly_log_increments(fresh,nums[:,0],nums[:,1],runner.BET_DENOMINATOR)
            else:
                predicted=predictable_probabilities(total[active],prediction["estimated_p_minus"],prediction["estimated_p_plus"])
                assert np.array_equal(nums[:,0],predicted["pred_minus_numerator"])
                assert np.array_equal(nums[:,1],predicted["pred_plus_numerator"])
                increments=safe_log_increments(fresh,nums[:,0],nums[:,1],runner.BET_DENOMINATOR)
            assert predicted["denominator"]==runner.BET_DENOMINATOR
            logplus[active]+=increments[0]
            logminus[active]+=increments[1]
        # Only after checking causal ordering do these outcomes enter fitting.
        total[active]+=fresh
        update_pairs(state,actual_angles,fresh,active)
        phases[active,phase]+=budget-previous
        blocks_checked+=len(active)
        previous=budget
    for i,row in enumerate(rows):
        assert total[i].tolist()==row["counts"]
        assert state["shots"][i]==row["shots"]
        assert phases[i].tolist()==row["phase_shots"]
        if policy=="psr":
            assert row["phase_shots"]==[row["shots"],0,0]
        else:
            n=row["shots"]
            assert row["phase_shots"]==[min(n,256),max(0,min(n,512)-256),max(0,n-512)]
        if engine=="beta":
            expected=runner.certificate_from_pairs(tuple(row["counts"]),delta=spec["delta"])
        else:
            expanded=[]
            for entry in row["blocks"]:
                block={"counts":entry[1:5],"denominator":runner.BET_DENOMINATOR}
                if engine=="bet":
                    block.update({"plus_numerator":entry[5],"minus_numerator":entry[6]})
                else:
                    block.update({"pred_minus_numerator":entry[5],"pred_plus_numerator":entry[6]})
                expanded.append(block)
            expected=(betting_certificate if engine=="bet" else safe_certificate)(expanded,delta=spec["delta"])
        assert row["certificate"]==expected,(case["case_id"],method,i,"certificate")
        assert row["decision"]==expected["decision"]
        certificates+=1
        abstentions+=row["decision"]=="abstain"
    return {"history_blocks_checked":blocks_checked,"final_certificates_recomputed":certificates,
            "full_cap_abstention_certificates_recomputed":abstentions,
            "causal_forecasts_reconstructed":blocks_checked if engine!="beta" else 0,
            "causal_angles_reconstructed":blocks_checked}


def main(directory,replay_ids):
    started=time.perf_counter()
    source=Path(__file__).resolve()
    original_source_hash=digest(source)
    directory=Path(directory).resolve()
    spec=json.loads((directory/"spec.json").read_text(encoding="utf-8"))
    summary=json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    assert spec==summary["spec"]
    assert spec["cases"]==runner.case_matrix()
    assert spec["checkpoints"]==runner.budget_schedule()
    assert spec["design_config"]==config_dict()
    assert spec["betting_config"]==protocol_config()
    assert spec["delta"]==.01 and spec["cap"]==2**22
    for name,expected in spec["sources"].items():
        assert digest(ROOT/name)==expected,name
    preserved=previous_evidence()
    assert spec["prior_evidence_manifest_sha256"]==preserved["previous_learned_manifest_sha256"]
    assert spec["public_case_spec_sha256"]==digest(ROOT/"results"/"learned_validation"/"development_v1"/"spec.json")
    expected_groups=[(case["case_id"],method) for case in spec["cases"] for method in spec["methods"]]
    cell_lookup={(cell["case_id"],cell["method"]):cell for cell in summary["cells"]}
    assert len(cell_lookup)==len(expected_groups)
    case_lookup={case["case_id"]:case for case in spec["cases"]}
    counters={key:0 for key in ("history_blocks_checked","final_certificates_recomputed",
                                "full_cap_abstention_certificates_recomputed","causal_forecasts_reconstructed",
                                "causal_angles_reconstructed")}
    checked_cells=[]
    replay_rows={}
    count=0
    with (directory/"trials.jsonl").open(encoding="utf-8") as stream:
        grouped=itertools.groupby((json.loads(line) for line in stream),key=lambda row:(row["case_id"],row["method"]))
        for group_index,(key,records) in enumerate(grouped):
            assert key==expected_groups[group_index],key
            rows=list(records)
            result=audit_group(rows,case_lookup[key[0]],key[1],spec)
            for field,value in result.items():
                counters[field]+=value
            count+=len(rows)
            independent=independent_summary(rows,case_lookup[key[0]],key[1])
            compare(cell_lookup[key],independent,str(key))
            checked_cells.append(independent)
            if key[0] in replay_ids:
                replay_rows[key]=rows
            if (group_index+1)%len(spec["methods"])==0:
                print(json.dumps({"audited_case":key[0],"ledger_records":count}),flush=True)
    assert len(checked_cells)==len(expected_groups)
    assert count==summary["trial_count"]==len(expected_groups)*spec["repetitions"]
    compare(summary["aggregate"],independent_aggregate(checked_cells,spec["methods"]),"aggregate")
    replayed=0
    for cid in replay_ids:
        assert cid in case_lookup
        for method in spec["methods"]:
            seed=130000000+cid*1009+runner.METHODS.index(method)*71
            actual_cell,actual_rows=runner.run_cell(case_lookup[cid],method,spec["repetitions"],seed)
            assert actual_rows==replay_rows[cid,method],(cid,method,"seed replay")
            compare(actual_cell,cell_lookup[cid,method],str((cid,method)))
            replayed+=len(actual_rows)
        print(json.dumps({"replayed_case":cid,"records":replayed}),flush=True)
    assert digest(source)==original_source_hash,"audit source changed during execution"
    result={"status":"passed","ledger_records":count,"cells_and_aggregates_recomputed":len(checked_cells),
            "seed_replay_case_ids":replay_ids,"vector_seed_replay_records":replayed,
            "run_source_hashes_verified":len(spec["sources"]),"review":{
                "cost":"All physical pairs counted; exploration pairs remain in both fitted state and evidence.",
                "predictability":"Every angle and rational forecast recomputed from only previous blocks before adding outcomes.",
                "joint_sampling":"Sequential-binomial construction equals independent Bernoulli pair multinomial.",
                "risk":"Common-sign paired Beta lemma or directional predictable supermartingales; final decisions require Decimal certificates.",
                "fairness":"Same cost checkpoints, initialPSRphase, fitting and totalrisk for allanglepolicies within each engine.",
                "scope":"Development implementation audit; no SOTA or uniform finite-budget power claim."},
            **counters,**preserved,"elapsed_seconds":time.perf_counter()-started,
            "audit_source_path":str(source.relative_to(ROOT)).replace("\\","/"),
            "audit_source_sha256":original_source_hash,"audit_source_unchanged_during_execution":True,
            "artifact_sha256":{name:digest(directory/name) for name in ("spec.json","summary.json","trials.jsonl")}}
    (directory/"audit.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--directory",default=str(ROOT/"results"/"streaming_validation"/"development_v1"))
    parser.add_argument("--replay-cases",default="0,42,49")
    args=parser.parse_args()
    main(args.directory,[int(x) for x in args.replay_cases.split(",") if x])
