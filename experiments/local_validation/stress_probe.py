"""Separate off-model experiment: the hidden effect is weaker than assumed.

The policy still assumes p1=0.4. Actual p1=0.30 makes the proposal harmful by
0.05 in the additive surrogate, while invalidating the simple-hypothesis LR
guarantee. No off-model result is added to the nominal aggregate.
"""

import argparse
import json
from pathlib import Path

from audit_math import sequential_prediction
import math

from probe import MeasurementOracle, PublicConfig, run_trial, stable_seed, wilson_interval


def harm_boundary_scan(config, measure):
    """A classical composite-hypothesis baseline, NOT a new algorithm.

    A single unknown monitor change cancels the known local gain when its
    probability reaches p0 + local_gain. Using that boundary in the likelihood
    ratio controls false acceptance uniformly for every p >= that boundary.
    The power guarantee on beneficial updates covers the all-p0 null ONLY;
    intermediate beneficial changes are not covered. p0 and gain remain known.
    """
    boundary = config.p0 + config.local_gain
    one = math.log(boundary / config.p0)
    zero = math.log((1.0 - boundary) / (1.0 - config.p0))
    upper, lower = math.log(config.n / config.delta), math.log(config.delta)
    for arm in range(config.n):
        log_lr = 0.0
        while True:
            log_lr += one if measure(arm) else zero
            if log_lr >= upper:
                return "reject"
            if log_lr <= lower:
                break
    return "accept"


def run_boundary_trial(config, hidden, seed, actual_p1):
    oracle = MeasurementOracle(config, hidden, seed, actual_p1=actual_p1)
    decision = harm_boundary_scan(config, oracle.measure)
    return {"decision": decision, "shots": oracle.shots, "audit": oracle.audit()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[8, 32, 128])
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--actual-p1", type=float, default=0.30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("trials must be positive")
    groups = []
    for n in args.sizes:
        config = PublicConfig(n)
        if args.actual_p1 - config.p0 <= config.local_gain:
            parser.error("this stress experiment requires a truly harmful candidate")
        for case, hidden in (("null", None), ("first", 0)):
            for policy in ("sequential_scan", "harm_boundary_scan"):
                records = []
                for i in range(args.trials):
                    seed = stable_seed(args.seed, n, i)
                    if policy == "sequential_scan":
                        record = run_trial(policy, config, hidden, seed, actual_p1=args.actual_p1)
                    else:
                        record = run_boundary_trial(config, hidden, seed, args.actual_p1)
                    records.append(record)
                accepted = sum(r["decision"] == "accept" for r in records)
                rejected = sum(r["decision"] == "reject" for r in records)
                wrong = rejected if hidden is None else accepted
                test_p1 = config.p1 if policy == "sequential_scan" else config.p0 + config.local_gain
                prediction = sequential_prediction(n, hidden, true_p1=args.actual_p1, p1=test_p1)
                prediction_wrong = prediction["reject_probability_estimate"] if hidden is None else prediction["accept_probability_estimate"]
                groups.append({
                    "policy": policy,
                    "case": case,
                    "n": n,
                    "hidden": hidden,
                    "trials": args.trials,
                    "wrong_decision_count": wrong,
                    "wrong_decision_rate": wrong / args.trials,
                    "wrong_decision_wilson_95": list(wilson_interval(wrong, args.trials)),
                    "dp_wrong_decision_prediction": prediction_wrong,
                    "dp_mean_shots_prediction": prediction["scan_expected_shots_estimate"],
                    "dp_truncation_probability_bound": prediction["probability_truncation_bound"],
                    "mean_shots": sum(r["shots"] for r in records) / args.trials,
                    "true_delta": records[0]["audit"]["true_delta"],
                    "accept": accepted,
                    "reject": rejected,
                    "abstain": sum(r["decision"] == "abstain" for r in records),
                })
    result = {
        "status": "off-model stress test; the nominal delta guarantee does not apply",
        "policies": ["sequential_scan", "harm_boundary_scan"],
        "harm_boundary_scope": "known p0 and local gain; at most one anomalous monitor; false-accept control for p >= p0 + gain; beneficial power guaranteed only at the all-p0 null",
        "assumed_p0": 0.1,
        "assumed_p1": 0.4,
        "actual_hidden_p1": args.actual_p1,
        "local_gain": 0.15,
        "nominal_delta": 0.05,
        "seed": args.seed,
        "groups": groups,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
