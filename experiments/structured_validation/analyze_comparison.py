"""Unconditional cost comparisons, keeping errors and abstentions in the mean."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def main(directory, bootstrap=5000, baseline="psr_beta"):
    directory = Path(directory).resolve()
    summary = json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    spec = summary["spec"]
    cases = {case["case_id"]: case for case in spec["cases"]}
    data, decisions = {}, {}
    with (directory/"trials.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            cid, method, rep = row["case_id"], row["method"], row["rep"]
            key = (cid, method)
            data.setdefault(key, {})[rep] = row["shots"]
            decisions.setdefault(method, {"wrong": 0, "abstained": 0, "trials": 0})
            expected = "accept" if cases[cid]["C"] < 0 else "reject" if cases[cid]["C"] > 0 else "abstain"
            decisions[method]["wrong"] += row["decision"] not in ("abstain", expected)
            decisions[method]["abstained"] += row["decision"] == "abstain"
            decisions[method]["trials"] += 1
    methods = sorted({key[1] for key in data})
    assert baseline in methods
    main_cases = sorted(cid for cid, case in cases.items() if case["kind"] == "main")
    repetitions = spec["repetitions"]
    arrays = {}
    for key, values in data.items():
        assert set(values) == set(range(repetitions))
        arrays[key] = np.array([values[r] for r in range(repetitions)], dtype=float)
    means = {key: float(a.mean()) for key, a in arrays.items()}
    comparisons = {}
    rng = np.random.RandomState(972041)
    sums = {method: np.zeros(bootstrap) for method in methods}
    for cid in main_cases:
        labels = rng.randint(0, repetitions, size=(bootstrap, repetitions))
        resampled = {method: arrays[cid, method][labels].mean(axis=1) for method in methods}
        for method in methods:
            sums[method] += np.log(resampled[method]/resampled[baseline])/len(main_cases)
    for method in methods:
        ratios = [means[cid, method]/means[cid, baseline] for cid in main_cases]
        value = math.exp(math.fsum(math.log(r) for r in ratios)/len(ratios))
        comparisons[method] = {"geomean_cost_ratio": value,
                               "ci95_percentile": [float(x) for x in np.percentile(np.exp(sums[method]), [2.5, 97.5])],
                               "lower_cost_cases": sum(r < 1 for r in ratios),
                               "tied_cases": sum(r == 1 for r in ratios),
                               "higher_cost_cases": sum(r > 1 for r in ratios),
                               "worst_case_ratio": max(ratios), "best_case_ratio": min(ratios)}
    matched = {}
    for policy in spec["policies"]:
        if policy+"_pair_grid" in methods:
            for engine in ("fisher", "rb", "safe"):
                method = policy+"_"+engine
                if method in methods:
                    matched[method+"_over_matched_pair_grid"] = math.exp(math.fsum(
                        math.log(means[cid, method]/means[cid, policy+"_pair_grid"])
                        for cid in main_cases)/len(main_cases))
    result = {"status": "development analysis; no prospective superiority claim",
              "baseline": baseline, "new_trajectories": summary["new_trajectories"],
              "main_cases": len(main_cases), "repetitions": repetitions,
              "bootstrap": {"repetitions": bootstrap, "seed": 972041,
                            "unit": "Independent joint replicate label within fixed case",
                            "scope": "Unadjusted descriptive intervals on development data, not population generalization."},
              "comparisons": comparisons, "matched_prior_comparisons": matched,
              "all_case_safety_counts": decisions,
              "sources": {name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
                          for name in ("spec.json", "summary.json", "trials.jsonl")},
              "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (directory/"analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("comparisons", "matched_prior_comparisons", "all_case_safety_counts")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--baseline", default="psr_beta")
    args = parser.parse_args()
    main(args.directory, args.bootstrap, args.baseline)
