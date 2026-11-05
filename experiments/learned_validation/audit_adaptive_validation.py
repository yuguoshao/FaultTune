"""Verify v2 routing costs, certificates, source provenance, and seed replay."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from adaptive_protocol import (EARLY_RISK,HOLDOUT_RISK,POLICIES,early_route,
                               early_schedule,complete_pilot,holdout_design)
from certificates import fisher_certificate
from audit_validation import compare_json,independently_summarize,frozen_files,close
import run_adaptive_validation as runner


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(directory):
    started=time.time()
    directory=Path(directory).resolve()
    spec=json.loads((directory/"spec.json").read_text(encoding="utf-8"))
    summary=json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    assert spec==summary["spec"]
    for name,expected in spec["source_sha256"].items():
        assert digest(runner.ROOT/name)==expected,name
    baseline=Path(spec["baseline_reuse"]["path"])
    for name,expected in spec["baseline_reuse"]["sha256"].items():
        assert digest(baseline/name)==expected,name
    baseline_rows={}
    with (baseline/"trials.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row=json.loads(line)
            if row["method"] in runner.METHODS[:2] and row["rep"]<spec["repetitions"]:
                row["phase"]="baseline_reuse"
                baseline_rows[row["case_id"],row["method"],row["rep"]]=row
    cells={}
    records=0
    baseline_verified=0
    early_checked=0
    holdout_checked=0
    decided_checked=0
    routing_cache={}
    with (directory/"trials.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row=json.loads(line)
            records+=1
            cid,method,rep=row["case_id"],row["method"],row["rep"]
            key=(cid,method)
            cells.setdefault(key,[]).append(row)
            if row["phase"]=="baseline_reuse":
                assert row==baseline_rows[cid,method,rep]
                baseline_verified+=1
                continue
            assert method in POLICIES
            counts=row["pilot_counts"]
            totals=row["pilot_totals"]
            pairs=row["early_pair_counts"]
            early_shots=2*sum(pairs)
            assert early_shots==early_schedule()[row["early_look_index"]-1]
            routing_key=(cid,rep)
            shared={name:row[name] for name in ("phase","pilot_counts","pilot_totals","early_pair_counts",
                                               "early_look_index","early_certificate","axis_successes",
                                               "axis_shots","early_seed","axis_seed")}
            if routing_key in routing_cache:
                assert routing_cache[routing_key]==shared
            else:
                routing_cache[routing_key]=shared
                route=early_route(pairs)
                compare_json(row["early_certificate"],route["certificate"])
                early_checked+=1
            assert row["pilot_shots"]==sum(totals)
            s0,n0,s1,n1=row["counts"]
            assert 0<=s0<=n0 and 0<=s1<=n1
            assert row["shots"]==sum(totals)+n0+n1
            assert row["early_risk"]==EARLY_RISK and row["holdout_risk"]==HOLDOUT_RISK
            assert row["total_risk_bound"]==EARLY_RISK+HOLDOUT_RISK==.01
            if row["phase"]=="early":
                assert row["decision"]!="abstain"
                assert row["axis_shots"]==0 and row["axis_successes"]==[0,0]
                assert row["counts"]==[0,0,0,0] and row["shots"]==early_shots<=512
                assert counts==[0,pairs[1]+pairs[3],0,pairs[2]+pairs[3]]
                assert totals==[0,sum(pairs),0,sum(pairs)]
                assert row["certificate"]==row["early_certificate"]
                assert row["certificate"]["decision"]==row["decision"]
                decided_checked+=1
            else:
                assert row["phase"]=="holdout" and early_shots==512
                assert row["early_certificate"]["decision"]=="abstain"
                expected_counts,expected_totals=complete_pilot(pairs,row["axis_successes"])
                assert counts==list(expected_counts) and totals==list(expected_totals)
                assert row["pilot_shots"]==1024 and row["axis_shots"]==512
                assert row["shots"]==spec["total_shot_checkpoints"][row["look_index"]-1]
                target=int(np.floor(row["allocation_plus"]*(row["shots"]-1024)))
                assert n1==min(max(target,1),row["shots"]-1025)
                cert=fisher_certificate(s0,n0,s1,n1,row["look_index"],len(spec["total_shot_checkpoints"]),delta=HOLDOUT_RISK)
                holdout_checked+=1
                assert cert["decision"]==row["decision"]
                if row["decision"]!="abstain":
                    compare_json(row["certificate"],cert)
                    decided_checked+=1
                else:
                    assert row["shots"]==spec["total_shot_cap"] and row["certificate"] is None
    assert records==summary["trial_count"]==72*6*spec["repetitions"]
    stored_cells={(cell["case_id"],cell["method"]):cell for cell in summary["cells"]}
    designs_checked=0
    for (cid,method),rows in cells.items():
        assert len(rows)==spec["repetitions"]
        assert sorted(row["rep"] for row in rows)==list(range(spec["repetitions"]))
        computed=independently_summarize(rows,spec["cases"][cid])
        for name,value in computed.items():
            compare_json(stored_cells[cid,method][name],value)
        if method in POLICIES:
            hard=[row for row in rows if row["phase"]=="holdout"]
            if hard:
                design=holdout_design(method,[row["pilot_counts"] for row in hard],[row["pilot_totals"] for row in hard])
                for i,row in enumerate(hard):
                    close(row["t"],float(design["t"][i]))
                    close(row["allocation_plus"],float(design["allocation_plus"][i]))
                    close(row["estimated_variance"],float(design["estimated_variance"][i]))
                    designs_checked+=1
            assert stored_cells[cid,method]["early_stops"]==len(rows)-len(hard)
            assert stored_cells[cid,method]["entered_holdout"]==len(hard)
    compare_json(summary["aggregate"],runner.aggregate(summary["cells"],spec["cases"]))
    replay_cases=[0,4,42,49]
    replay_records=0
    for cid in replay_cases:
        case=spec["cases"][cid]
        routing=runner.routing_for_case(case,spec["repetitions"])
        for method in POLICIES:
            _,fresh=runner.run_cell(case,method,spec["repetitions"],routing)
            compare_json(cells[cid,method],fresh)
            replay_records+=len(fresh)
    result={"schema":"faulttune.adaptive_validation_audit.v1","passed":True,
            "records_checked":records,"baseline_records_exactly_reused":baseline_verified,
            "unique_early_routing_certificates_recomputed":early_checked,
            "holdout_final_certificates_recomputed_including_abstentions":holdout_checked,
            "wrapped_final_decisions_checked":decided_checked,"count_only_designs_recomputed":designs_checked,
            "seed_replay_cases":replay_cases,"seed_replay_records":replay_records,
            "all_four_wrapped_policies_share_identical_routing_transcripts":True,
            "all_costs_and_cell_summaries_checked":True,"risk_bound_per_wrapped_call":.01,
            "previous_evidence":frozen_files(),"elapsed_seconds":time.time()-started,
            "scope":"implementation/provenance audit, not empirical simultaneous safety or power certification",
            "evidence_sha256":{name:digest(directory/name) for name in ("spec.json","summary.json","trials.jsonl")},
            "audit_source_sha256":digest(__file__)}
    (directory/"audit.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",default=str(runner.ROOT/"results/learned_validation/development_v2"))
    run(parser.parse_args().run)
