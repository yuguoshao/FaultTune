"""Reproduce old-data audit pathway checks; no new case/seed observations."""
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import time

import numpy as np

import analyze
import audit
import common


def main():
    results = []
    for dataset in ("refined_smoke_v1", "unequal_smoke_v1"):
        directory = common.ROOT/"results/structured_validation"/dataset
        spec = common.read_json(directory/"summary.json")["spec"]
        cases = {c["case_id"]:c for c in spec["cases"]}
        plan = {"seed_base":spec["seed_base"],"delta":spec["delta"],
                "repetitions_by_kind":{"main":16,"null":16},
                "cap_by_kind":{"main":spec["cap"],"null":spec["cap"]}}
        groups = defaultdict(list)
        for line in (directory/"trials.jsonl").read_text().splitlines():
            record = json.loads(line)
            groups[(record["case_id"],record["policy"],record["rep"])].append(record)
        for mode in ("full_interval", "independent_lower_bound"):
            current = copy.deepcopy(plan)
            if mode == "independent_lower_bound":
                current["seed_base"] -= 1000003
            totals = Counter()
            started = time.perf_counter()
            for key,originals in sorted(groups.items()):
                records = copy.deepcopy(originals)
                if mode == "independent_lower_bound":
                    for record in records:
                        record["rep"] += 1
                if dataset.startswith("refined"):
                    audit.audit_balanced(records,cases[key[0]],current,totals)
                else:
                    audit.audit_unequal(records[0],cases[key[0]],current,totals)
            results.append({"dataset":dataset,"mode":mode,"checks":dict(totals),
                            "seconds":time.perf_counter()-started})
            print(json.dumps(results[-1]),flush=True)
        # Complete-case integration on one old case, including original runner replay.
        case_id = min(cases)
        records = [r for key,local in groups.items() if key[0] == case_id for r in local]
        current = copy.deepcopy(plan)
        current["repetitions_by_kind"] = {"main":1,"null":1}
        selected = {"methods":sorted({r["method"] for r in records})}
        raw = b"".join((json.dumps(r)+"\n").encode("utf-8") for r in records)
        checked = audit.audit_case(raw,cases[case_id],current,selected)
        assert checked["checks"]["terminal_records_checked"] == len(records)
        try:
            audit.audit_case(raw+raw.splitlines(keepends=True)[0],cases[case_id],current,selected)
        except AssertionError:
            pass
        else:
            raise AssertionError("Duplicate ledger record accepted")
        results.append({"dataset":dataset,"mode":"case_integration_and_duplicate_rejection",
                        "checks":checked["checks"]})
        print(json.dumps(results[-1]),flush=True)
    # Synthetic matrices validate the independent multiplicity bootstrap path.
    plan = common.load_plan()
    selected = {"methods":["a","b","c"],"candidate":"a","primary_comparator":"b","secondary_comparator":"c"}
    audited,arrays = [],[]
    for case in plan["cases"]:
        if case["kind"] != "main":
            continue
        c = case["case_id"]
        matrix = np.array([[30+r+c,50+(r*r)%29+c,40+2*r+c] for r in range(64)],dtype=float)
        arrays.append(matrix)
        audited.append({"kind":"main","case_id":c,"cost_rows":[
            {"method":method,"rep":rep,"shots":int(matrix[rep,index])}
            for rep in range(64) for index,method in enumerate(selected["methods"])]})
    outputs = analyze.paired_bootstrap(arrays,[(0,1),(0,2)],5000,914207)
    for output in outputs:
        output["per_case_ratios"] = [{"case_id":item["case_id"],"ratio":ratio}
                                     for item,ratio in zip(audited,output["per_case_ratios"])]
    result = {"primary":outputs[0],"secondary":outputs[1],
              "primary_numerical_threshold_met":outputs[0]["percentile_95_ci"][1] < 1}
    audit.independently_check_analysis(audited,result,plan,selected)
    evidence = {"status":"passed","scope":"Existing smoke records, exact seed replays and synthetic bootstrap matrices only; no prospective observations",
                "plan_sha256":common.PLAN_SHA256,"audit_sources_sha256":audit.audit_hashes(),
                "check_source_sha256":common.digest(__file__),"checks":results,
                "independent_bootstrap_multiplicity_path":"passed"}
    common.write_new(common.ROOT/"results/structured_validation/prospective_audit_checks_v2.json",evidence)
    print(json.dumps({"status":"passed","checks":len(results),"independent_bootstrap":"passed"}),flush=True)


if __name__ == "__main__":
    main()
