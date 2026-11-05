"""Select, explicitly release, then run the frozen prospective comparison.

No call to a simulator occurs in select or release mode. Existing raw data,
selection, release and result artifacts are never overwritten.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_refined as refined
import unequal_runner as unequal
import common


def execution_groups(methods):
    grouped, unequal_methods = defaultdict(list), []
    for method in sorted(methods):
        policy, engine = method.split("_", 1)
        if engine == "weighted_conditional_unequal":
            unequal_methods.append((policy, method))
        else:
            common.require(engine in refined.ENGINES and policy in refined.POLICIES,
                           "Unknown released method")
            grouped[policy].append(engine)
    calls = [(policy, "refined", tuple(sorted(engines))) for policy, engines in sorted(grouped.items())]
    calls.extend((policy, "unequal", (method,)) for policy, method in unequal_methods)
    return sorted(calls, key=lambda value: (value[0], value[1]))


def select():
    plan = common.load_plan()
    selection = common.select_from_complete_development(plan)
    result = {"status": "selected from complete audited development, not released",
              "created_utc": common.utc_now(), "plan_sha256": common.PLAN_SHA256,
              "selection": selection}
    common.write_new(common.DIRECTORY/"selection.json", result)
    print(json.dumps(result, indent=2), flush=True)


def release(reference):
    common.require(bool(reference.strip()), "Root release reference must be explicit")
    plan = common.load_plan()
    selection_path = common.DIRECTORY/"selection.json"
    selection = common.read_json(selection_path)
    common.require(selection["plan_sha256"] == common.PLAN_SHA256 and
                   selection["selection"] == common.select_from_complete_development(plan),
                   "Selection must match complete audited development")
    common.require(time.time() < plan["execution"]["cutoff_unix"], "Experiment cutoff reached")
    result = {"status": "released before prospective observations", "released_utc": common.utc_now(),
              "explicit_root_release": True, "release_reference": reference,
              "plan_sha256": common.PLAN_SHA256, "selection_sha256": common.digest(selection_path),
              "selection": selection["selection"], "wrapper_sources_sha256": common.wrapper_hashes(),
              "algorithm_sources_sha256": plan["algorithm_sources_sha256"]}
    common.write_new(common.DIRECTORY/"release.json", result)
    print(json.dumps(result, indent=2), flush=True)


def run():
    plan, selected, release_record = common.validate_release()
    directory = common.DIRECTORY
    for name in ("spec.json", "trials.jsonl", "summary.json", "progress.json"):
        common.require(not (directory/name).exists(), "Preserve existing prospective artifact: "+name)
    common.require(time.time() < plan["execution"]["cutoff_unix"], "Experiment cutoff reached")
    started = time.perf_counter()
    methods, groups = selected["methods"], execution_groups(selected["methods"])
    expected_records = plan["expected_joint_replicate_labels"]*len(methods)
    spec = {"status": "prospective frozen design", "created_utc": common.utc_now(),
            "plan_sha256": common.PLAN_SHA256, "release_sha256": common.digest(directory/"release.json"),
            "selection": selected, "cases": plan["cases"], "methods": methods,
            "repetitions_by_kind": plan["repetitions_by_kind"], "cap_by_kind": plan["cap_by_kind"],
            "delta": plan["delta"], "seed_base": plan["seed_base"], "seed_rule": plan["seed_rule"],
            "execution_groups": groups, "expected_method_trajectories": expected_records,
            "sources": plan["algorithm_sources_sha256"],
            "wrapper_sources_sha256": release_record["wrapper_sources_sha256"],
            "execution": plan["execution"], "coupling": plan["coupling"], "counting": plan["counting"]}
    common.write_new(directory/"spec.json", spec)
    completed, case_boundaries, summaries = [], {}, []
    count, timing_totals = 0, defaultdict(float)
    current_case, status, error = None, "running", None

    def progress(stream):
        stream.flush()
        value = {"status": status, "updated_utc": common.utc_now(),
                 "completed_case_ids": completed, "completed_case_ledger_bytes": case_boundaries,
                 "flushed_ledger_bytes": stream.tell(), "completed_method_trajectories": count,
                 "current_case_id": current_case, "expected_method_trajectories": expected_records,
                 "elapsed_seconds": time.perf_counter()-started, "error": error}
        temporary = directory/"progress.pending.json"
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        os.replace(str(temporary), str(directory/"progress.json"))

    with (directory/"trials.jsonl").open("xb") as stream:
        progress(stream)
        try:
            for case in plan["cases"]:
                current_case = case["case_id"]
                repetitions = plan["repetitions_by_kind"][case["kind"]]
                cap = plan["cap_by_kind"][case["kind"]]
                for rep in range(repetitions):
                    seed = plan["seed_base"]+1009*case["case_id"]+1000003*rep
                    for policy, family, engines in groups:
                        if time.time() >= plan["execution"]["cutoff_unix"]:
                            raise RuntimeError("Experiment cutoff reached; partial ledger retained")
                        if family == "refined":
                            records, timing = refined.run_trial(case, policy, rep, seed, cap, engines)
                        else:
                            record, timing = unequal.run_trial(case, policy, rep, seed, cap,
                                                               plan["execution"]["cutoff_unix"])
                            records = [record]
                        for key, value in timing.items():
                            timing_totals[key] += value
                        for record in records:
                            stream.write((json.dumps(record, separators=(",", ":"), allow_nan=False)+"\n").encode("utf-8"))
                            summaries.append({key: record[key] for key in
                                              ("case_id", "method", "rep", "seed", "shots", "decision")})
                            count += 1
                        stream.flush()
                completed.append(case["case_id"])
                case_boundaries[str(case["case_id"])] = stream.tell()
                progress(stream)
                print(json.dumps({"completed_case": case["case_id"], "kind": case["kind"],
                                  "completed_method_trajectories": count, "flushed_ledger_bytes": stream.tell(),
                                  "elapsed_seconds": time.perf_counter()-started}), flush=True)
            common.require(count == expected_records, "Prospective trajectory count mismatch")
            common.validate_release(recheck_development=False)
            status, current_case = "complete", None
        except Exception as caught:
            status = "incomplete_cutoff" if time.time() >= plan["execution"]["cutoff_unix"] else "incomplete_error"
            error = type(caught).__name__+": "+str(caught)
            progress(stream)
            raise
        progress(stream)
    cells = common.build_cells(summaries, plan["cases"], methods)
    result = {"status": "complete", "spec": spec, "cells": cells,
              "new_trajectories": count, "joint_replicate_labels": plan["expected_joint_replicate_labels"],
              "timings": dict(timing_totals), "elapsed_seconds": time.perf_counter()-started,
              "trials_sha256": common.digest(directory/"trials.jsonl")}
    common.write_new(directory/"summary.json", result)
    print(json.dumps({key: value for key, value in result.items() if key not in ("spec", "cells")}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("select", "release", "run"))
    parser.add_argument("--reference", default="")
    args = parser.parse_args()
    if args.command == "select":
        select()
    elif args.command == "release":
        release(args.reference)
    else:
        run()
