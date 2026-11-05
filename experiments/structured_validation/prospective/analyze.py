"""Prespecified complete-data paired bootstrap; no inference on partial data."""
from collections import defaultdict
import json
import math

import numpy as np

import common


def paired_bootstrap(arrays, comparisons, resamples=5000, seed=914207):
    """Arrays are sorted cases of shape (joint replicates, methods)."""
    rng = np.random.RandomState(seed)
    boot_log = np.zeros((resamples, len(comparisons)), dtype=float)
    point_log = np.zeros(len(comparisons), dtype=float)
    case_ratios = [[] for unused in comparisons]
    for array in arrays:
        array = np.asarray(array, dtype=float)
        common.require(array.ndim == 2 and len(array) > 0 and np.all(array > 0), "Invalid bootstrap costs")
        indices = rng.randint(0, len(array), size=(resamples, len(array)))
        means = array.mean(axis=0)
        resampled = array[indices].mean(axis=1)
        for j, (candidate, comparator) in enumerate(comparisons):
            ratio = means[candidate]/means[comparator]
            case_ratios[j].append(float(ratio))
            point_log[j] += math.log(ratio)
            boot_log[:, j] += np.log(resampled[:, candidate]/resampled[:, comparator])
    output = []
    for j in range(len(comparisons)):
        samples = np.exp(boot_log[:, j]/len(arrays))
        interval = np.percentile(samples, [2.5, 97.5], interpolation="linear")
        output.append({"geometric_mean_ratio": float(math.exp(point_log[j]/len(arrays))),
                       "percentile_95_ci": [float(value) for value in interval],
                       "per_case_ratios": case_ratios[j], "bootstrap_resamples": resamples,
                       "bootstrap_seed": seed})
    return output


def analyze():
    plan, selected, release = common.validate_release()
    directory = common.DIRECTORY
    common.require(not (directory/"analysis.json").exists(), "Preserve completed analysis")
    inputs = {name: common.digest(directory/name) for name in
              ("spec.json", "summary.json", "trials.jsonl", "release.json", "selection.json")}
    spec, summary, progress = (common.read_json(directory/name) for name in
                              ("spec.json", "summary.json", "progress.json"))
    common.require(summary["status"] == progress["status"] == "complete", "No inference on incomplete data")
    common.require(summary["spec"] == spec and spec["plan_sha256"] == common.PLAN_SHA256 and
                   spec["release_sha256"] == inputs["release.json"], "Prospective design binding differs")
    common.require(summary["trials_sha256"] == inputs["trials.jsonl"], "Completed raw ledger changed")
    common.require(spec["cases"] == plan["cases"] and spec["methods"] == selected["methods"] and
                   spec["repetitions_by_kind"] == plan["repetitions_by_kind"] and
                   spec["cap_by_kind"] == plan["cap_by_kind"], "Prospective matrix changed")
    methods = selected["methods"]
    cases = {case["case_id"]: case for case in plan["cases"]}
    expected = {(case_id, method, rep) for case_id, case in cases.items() for method in methods
                for rep in range(plan["repetitions_by_kind"][case["kind"]])}
    records, keyed = [], {}
    with (directory/"trials.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            raw = json.loads(line)
            record = {key: raw[key] for key in ("case_id", "method", "rep", "seed", "shots", "decision")}
            key = record["case_id"], record["method"], record["rep"]
            common.require(key in expected and key not in keyed, "Unexpected or duplicate prospective record")
            case = cases[key[0]]
            common.require(record["seed"] == plan["seed_base"]+1009*key[0]+1000003*key[2], "Seed rule differs")
            common.require(type(record["shots"]) is int and 0 < record["shots"] <= plan["cap_by_kind"][case["kind"]],
                           "Invalid total cost")
            common.require(record["decision"] in ("accept", "reject", "abstain"), "Invalid decision")
            common.require(record["decision"] != "abstain" or record["shots"] == plan["cap_by_kind"][case["kind"]],
                           "Abstention before cap")
            keyed[key] = record
            records.append(record)
    common.require(set(keyed) == expected and len(records) == summary["new_trajectories"] ==
                   plan["expected_joint_replicate_labels"]*len(methods), "Incomplete prospective record matrix")
    cells = common.build_cells(records, plan["cases"], methods)
    common.require(cells == summary["cells"], "Summary is not the unconditional ledger reconstruction")
    main_ids = sorted(case_id for case_id, case in cases.items() if case["kind"] == "main")
    arrays = [np.array([[keyed[(case_id, method, rep)]["shots"] for method in methods]
                        for rep in range(plan["repetitions_by_kind"]["main"])], dtype=float) for case_id in main_ids]
    comparisons = [(methods.index(selected["candidate"]), methods.index(selected[name]))
                   for name in ("primary_comparator", "secondary_comparator")]
    outputs = paired_bootstrap(arrays, comparisons, plan["primary_analysis"]["bootstrap_resamples"],
                               plan["primary_analysis"]["bootstrap_seed"])
    for output, name in zip(outputs, ("primary_comparator", "secondary_comparator")):
        output["candidate"], output["comparator"] = selected["candidate"], selected[name]
        output["per_case_ratios"] = [{"case_id": case_id, "ratio": ratio} for case_id, ratio in
                                     zip(main_ids, output["per_case_ratios"])]
        output["worst_case"] = max(output["per_case_ratios"], key=lambda item: (item["ratio"], -item["case_id"]))
        output["faster_cases"] = sum(item["ratio"] < 1 for item in output["per_case_ratios"])
        output["tied_cases"] = sum(item["ratio"] == 1 for item in output["per_case_ratios"])
        output["slower_cases"] = sum(item["ratio"] > 1 for item in output["per_case_ratios"])
    outcomes = []
    for kind in ("main", "null"):
        for method in methods:
            local = [cell for cell in cells if cell["kind"] == kind and cell["method"] == method]
            outcomes.append({"kind": kind, "method": method, "cases": len(local),
                             "repetitions_per_case": plan["repetitions_by_kind"][kind],
                             "cap": plan["cap_by_kind"][kind],
                             "trajectories": sum(c["repetitions"] for c in local),
                             "wrong": sum(c["wrong"] for c in local),
                             "abstained": sum(c["abstained"] for c in local)})
    common.require(inputs == {name: common.digest(directory/name) for name in inputs}, "Inputs changed during analysis")
    common.validate_release(recheck_development=False)
    result = {"status": "complete analysis; scientific acceptance requires passed independent audit",
              "created_utc": common.utc_now(), "plan_sha256": common.PLAN_SHA256,
              "input_sha256": inputs, "primary": outputs[0], "secondary": outputs[1],
              "primary_numerical_threshold_met": outputs[0]["percentile_95_ci"][1] < 1,
              "cells": cells, "outcomes": outcomes, "analysis_plan": plan["primary_analysis"],
              "claim_boundary": plan["claim_boundary"]}
    common.write_new(directory/"analysis.json", result)
    print(json.dumps({key: value for key, value in result.items() if key not in ("cells", "analysis_plan")}, indent=2), flush=True)
    return result


if __name__ == "__main__":
    analyze()
