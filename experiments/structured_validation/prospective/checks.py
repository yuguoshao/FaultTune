"""Synthetic-only checks. This script never calls an observation simulator."""
import copy
import json
import math
from pathlib import Path
import tempfile

import numpy as np

import analyze
import common
import run


def raises_assertion(call):
    try:
        call()
    except AssertionError:
        return True
    raise AssertionError("Invalid fixture was unexpectedly accepted")


def main():
    plan = common.load_plan()
    main_cases = [c for c in plan["cases"] if c["kind"] == "main"]
    null_cases = [c for c in plan["cases"] if c["kind"] == "null"]
    assert len(main_cases) == 48 and len(null_cases) == 8
    assert {c["A"] for c in main_cases} == {.012, .07, .28, .68}
    assert sum(plan["repetitions_by_kind"][c["kind"]] for c in plan["cases"]) == 3200
    labels = [plan["seed_base"]+1009*c["case_id"]+1000003*rep for c in plan["cases"]
              for rep in range(plan["repetitions_by_kind"][c["kind"]])]
    assert len(labels) == len(set(labels)) and min(labels) >= 0 and max(labels) < 2**32
    assert all(c["A"]-math.hypot(c["B"], c["C"]) > 0 and
               c["A"]+math.hypot(c["B"], c["C"]) < 1 for c in plan["cases"])
    groups = run.execution_groups(["psr_rb", "psr_beta", "grid_weighted_conditional_unequal"])
    assert groups == [("grid", "unequal", ("grid_weighted_conditional_unequal",)),
                      ("psr", "refined", ("beta", "rb"))]

    # Deliberately deterministic, non-prospective synthetic costs.
    arrays = []
    for c in range(4):
        rows = [[20+2*r+3*c, 35+(r*r)%17+c, 20+2*r+3*c] for r in range(11)]
        arrays.append(np.array(rows, dtype=float))
    output = analyze.paired_bootstrap(arrays, [(0, 1), (0, 2)], resamples=137, seed=91)
    rng = np.random.RandomState(91)
    by_case = []
    for array in arrays:
        case = []
        for unused in range(137):
            indices = [int(rng.randint(0, len(array))) for unused_r in range(len(array))]
            sums = [sum(float(array[index, method]) for index in indices) for method in range(3)]
            case.append(math.log(sums[0]/sums[1]))
        by_case.append(case)
    oracle = sorted(math.exp(sum(case[b] for case in by_case)/len(arrays)) for b in range(137))
    def quantile(p):
        position = p*(len(oracle)-1)
        lower = int(math.floor(position))
        fraction = position-lower
        return oracle[lower]*(1-fraction)+oracle[min(lower+1,len(oracle)-1)]*fraction
    assert np.max(np.abs(np.array(output[0]["percentile_95_ci"])-[quantile(.025),quantile(.975)])) < 1e-14
    point = math.exp(sum(math.log(sum(a[:,0])/sum(a[:,1])) for a in arrays)/len(arrays))
    assert abs(output[0]["geometric_mean_ratio"]-point) < 1e-14
    assert output[1]["geometric_mean_ratio"] == 1 and output[1]["percentile_95_ci"] == [1,1]
    proportional = [np.array([[20,40],[30,60],[50,100]], dtype=float)]*3
    same = analyze.paired_bootstrap(proportional, [(0,1)], resamples=71, seed=88)[0]
    assert same["geometric_mean_ratio"] == .5 and same["percentile_95_ci"] == [.5,.5]

    rows = [{"case_id":0,"method":"m","shots":n,"decision":d} for n,d in
            [(40,"reject"),(80,"accept"),(100,"abstain")]]
    cells = common.build_cells(rows,[{"case_id":0,"kind":"main","C":1}], ["m"])
    assert cells[0]["mean_shots"] == 220/3 and cells[0]["wrong"] == 1 and cells[0]["abstained"] == 1

    # Gate fixtures are clearly synthetic and never use the real experiment directory.
    with tempfile.TemporaryDirectory(prefix="prospective_gate_check_", dir=str(common.HERE)) as temp:
        fake = copy.deepcopy(plan)
        cases = [{"case_id":i,"kind":"main","C":1} for i in range(24)]
        all_methods = plan["selection"]["candidate_pool"]+plan["selection"]["comparator_pool"]
        for number, entry in enumerate(fake["selection"]["development_inputs"]):
            directory = Path(temp)/str(number)
            directory.mkdir()
            methods = [method for method in all_methods if
                       method.endswith("_weighted_conditional_unequal") == bool(number)]
            spec = {"cases":cases,"sources":{},"repetitions":16,"seed_base":810000000,
                    "cap":2**22,"delta":.01}
            common.write_new(directory/"spec.json", spec)
            (directory/"trials.jsonl").write_text("synthetic fixture only\n",encoding="utf-8")
            summary = {"spec":spec,"new_trajectories":entry["expected_records"],
                       "cells":[{"case_id":case["case_id"],"kind":"main","method":method,
                                 "repetitions":16,"mean_shots":100.,"wrong":0,"abstained":0}
                                for method in methods for case in cases]}
            common.write_new(directory/"summary.json",summary)
            hashes = {name:common.digest(directory/name) for name in ("spec.json","summary.json","trials.jsonl")}
            common.write_new(directory/"audit.json", {"status":"passed",entry["audit_hash_field"]:hashes})
            entry["directory"], entry["spec_sha256"] = str(directory), hashes["spec.json"]
        selected = common.select_from_complete_development(fake)
        assert selected["candidate"] == "closed_rb" and selected["primary_comparator"] == "closed_beta"
        directory = Path(fake["selection"]["development_inputs"][0]["directory"])
        audit_path = directory/"audit.json"
        audit = common.read_json(audit_path)
        modified = dict(audit, status="failed")
        audit_path.write_text(json.dumps(modified),encoding="utf-8")
        raises_assertion(lambda:common.select_from_complete_development(fake))
        audit_path.write_text(json.dumps(audit),encoding="utf-8")
        with (directory/"trials.jsonl").open("a",encoding="utf-8") as stream:
            stream.write("changed\n")
        raises_assertion(lambda:common.select_from_complete_development(fake))
        (directory/"summary.json").unlink()
        raises_assertion(lambda:common.select_from_complete_development(fake))
    result = {"status":"passed","scope":"Synthetic costs and gate fixtures only; no observation simulator called",
              "plan_sha256":common.PLAN_SHA256,"wrapper_sources_sha256":common.wrapper_hashes(),
              "checks":["56 physical case definitions","3200 independent unique seed labels",
                        "same-policy grouping","paired bootstrap independent scalar oracle",
                        "linear percentile interpolation","identical-method and proportional-cost invariance",
                        "unconditional error and abstention accounting","deterministic lexicographic selection",
                        "failed audit rejection","changed-ledger audit rejection","missing-summary rejection"]}
    common.write_new(common.ROOT/"results/structured_validation/prospective_wrapper_checks.json",result)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
