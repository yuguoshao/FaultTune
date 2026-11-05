"""Independent audit of the frozen fixed-PSR Track-and-Stop control."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from benchmark import ROOT,summarize_records,aggregate
from sota_baseline import TrackStopState,run_cell


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same(actual,expected,label):
    if isinstance(actual,dict):
        assert set(actual)==set(expected),label
        for key,value in actual.items():
            same(value,expected[key],label+"/"+str(key))
    elif isinstance(actual,list):
        assert len(actual)==len(expected),label
        for i,(left,right) in enumerate(zip(actual,expected)):
            same(left,right,label+"/"+str(i))
    elif isinstance(actual,float):
        assert math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-12),label
    else:
        assert actual==expected,(label,actual,expected)


def main(directory):
    started=time.perf_counter()
    source=Path(__file__).resolve()
    source_hash=digest(source)
    folder=Path(directory)
    spec=json.loads((folder/"spec.json").read_text(encoding="utf-8"))
    summary=json.loads((folder/"summary.json").read_text(encoding="utf-8"))
    assert summary["spec"]==spec
    for relative,expected in spec["sources"].items():
        assert digest(ROOT/relative)==expected,relative
    schedule=spec["checkpoints"]
    delta=spec["delta"]
    repetitions=spec["repetitions"]
    cases={c["case_id"]:c for c in spec["cases"]}
    method="psr_trackstop"
    assert spec["methods"]==[method]
    groups={cid:[] for cid in cases}
    originals={}
    seen=set()
    total_blocks=certified=abstentions=0
    replay_ids=(0,42,49)
    replay_count=min(8,repetitions)
    with (folder/"trials.jsonl").open(encoding="utf-8") as ledger:
        for line in ledger:
            row=json.loads(line)
            cid,rep=row["case_id"],row["rep"]
            key=cid,rep
            assert key not in seen
            seen.add(key)
            assert cid in cases and 0<=rep<repetitions
            assert row["method"]==method and row["pilot_shots"]==0
            assert row["cell_seed"]==170000000+cid*1009
            assert row["seed"]==(row["cell_seed"]+104729*rep)%(2**32)
            assert row["source_sha256"]==spec["sources"]["experiments/circuit_validation/kl_trackstop.py"]
            assert row["t"]==math.pi/2
            successes,trials=[0,0],[0,0]
            for look,block in enumerate(row["trajectory"],1):
                assert block["look_index"]==look
                assert block["total_shots"]==schedule[look-1]
                added_n,added_s=block["arm_trials_added"],block["arm_successes_added"]
                assert len(added_n)==len(added_s)==2
                assert sum(added_n)==block["total_shots"]-sum(trials)
                for arm in (0,1):
                    assert isinstance(added_n[arm],int) and isinstance(added_s[arm],int)
                    assert 0<=added_s[arm]<=added_n[arm]
                    trials[arm]+=added_n[arm]
                    successes[arm]+=added_s[arm]
                assert sum(trials)==block["total_shots"]
                total_blocks+=1
            assert row["counts"]==[successes[0],trials[0],successes[1],trials[1]]
            assert row["shots"]==sum(trials)
            assert row["look_index"]==len(row["trajectory"])
            assert row["shots"]==schedule[row["look_index"]-1]
            # Recompute the final certificate directly from the counted data.
            # Allocation replay is separately performed for selected seeds.
            state=TrackStopState([-1.,1.],delta=delta,prior=.5)
            for arm in (0,1):
                state.update(arm,successes[arm],trials[arm])
            certificate=state.certificate(full_audit=True)
            assert certificate==row["certificate"],(key,"certificate")
            assert certificate["decision"]==row["decision"]
            if row["decision"]=="abstain":
                assert row["shots"]==schedule[-1]
                abstentions+=1
            else:
                certified+=1
            groups[cid].append({k:row[k] for k in ("shots","decision","rep")})
            if cid in replay_ids and rep<replay_count:
                originals[key]=row
    assert len(seen)==len(cases)*repetitions==summary["trial_count"]
    cells={c["case_id"]:c for c in summary["cells"]}
    assert set(cells)==set(cases)
    for cid,case in cases.items():
        rows=groups[cid]
        assert sorted(r["rep"] for r in rows)==list(range(repetitions))
        calculated=summarize_records(rows,case,method)
        for key,value in calculated.items():
            same(value,cells[cid][key],str(cid)+"/"+key)
    same(aggregate(summary["cells"],[method]),summary["aggregate"],"aggregate")
    expected_geomean=math.exp(math.fsum(math.log(cells[cid]["mean_shots"])
                                       for cid,c in cases.items() if c["kind"]=="main")/48)
    same(expected_geomean,summary["aggregate"][method]["main_geomean_shots"],"independent geomean")
    print(json.dumps({"ledger_checked":len(seen),"certificates":certified,
                      "abstentions":abstentions,"replaying":len(originals)}),flush=True)
    replayed=0
    for cid in replay_ids:
        unused,rows=run_cell(cases[cid],replay_count,schedule,170000000+cid*1009,delta)
        for row in rows:
            assert row==originals[cid,row["rep"]],(cid,row["rep"],"full replay")
            replayed+=1
    assert digest(source)==source_hash,"Audit source changed during execution"
    result={"status":"passed","ledger_records":len(seen),"increment_blocks_checked":total_blocks,
            "decision_certificates_recomputed":certified,
            "full_cap_abstention_certificates_recomputed":abstentions,
            "cells_and_aggregate_recomputed":len(cases),
            "full_seed_and_allocation_replay_records":replayed,"replay_case_ids":list(replay_ids),
            "source_hashes_verified":len(spec["sources"]),"audit_source_path":str(source.relative_to(ROOT)).replace("\\","/"),
            "audit_source_sha256":source_hash,"audit_source_unchanged_during_execution":True,
            "artifact_sha256":{n:digest(folder/n) for n in ("spec.json","summary.json","trials.jsonl")},
            "elapsed_seconds":time.perf_counter()-started,
            "scope":"Fixed two-arm TrackStop implementation audit; not full all-angle SOTA equivalence or empirical simultaneous power proof."}
    (folder/"audit.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--directory",default=str(ROOT/"results"/"streaming_validation"/"trackstop_v1"))
    args=parser.parse_args()
    main(args.directory)
