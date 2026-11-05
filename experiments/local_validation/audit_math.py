"""Independent probability calculations for a fixed-candidate validation problem.

Only the Python 3.8 standard library is required. This module does not call the
Monte Carlo policies. Dynamic programming follows the distribution of a single
Bernoulli likelihood-ratio walk, absorbing it at either decision boundary.
Floating-point output is a numerical check, not a formal probability certificate.
"""

import argparse
import json
import math
from pathlib import Path


def bernoulli_kl(p, q):
    if not (0.0 < p < 1.0 and 0.0 < q < 1.0):
        raise ValueError("KL arguments must be strictly between zero and one")
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


def theoretical_bounds(n, p0=0.1, p1=0.4, delta=0.05):
    """Bounds for terminating, uniformly delta-correct tests in the stated model."""
    if n < 1 or not (0.0 < p0 < p1 < 1.0) or not (0.0 < delta < 0.5):
        raise ValueError("invalid hypothesis-test parameters")
    d01 = bernoulli_kl(p0, p1)
    return {
        "n": n,
        "kl_p0_p1": d01,
        "null_expected_distinct_lower": n * (1.0 - 2.0 * delta),
        "null_expected_shots_lower": n * bernoulli_kl(1.0 - delta, delta) / d01,
        "null_sequential_expected_shots_upper": n
        * (math.log(1.0 / delta) + math.log((1.0 - p0) / (1.0 - p1)))
        / d01,
        "fixed_shots_per_site": int(math.ceil(2.0 * math.log(n / delta) / (p1 - p0) ** 2)),
    }


def simultaneous_readout_bounds(n, p0=0.1, p1=0.4, delta=0.05):
    """A DIFFERENT oracle: every round reads all n independent monitors.

    The lower bound concerns a deterministic maximum number of full readout
    rounds, not expected samples of a sequential point-query procedure.
    A union-bound midpoint test supplies an O(log(n/delta)) achievable budget.
    """
    point = theoretical_bounds(n, p0, p1, delta)
    v = (p1 - p0) ** 2 / (p0 * (1.0 - p0))
    return {
        "maximum_rounds_lower": math.log1p(4.0 * n * (1.0 - 2.0 * delta) ** 2)
        / math.log1p(v),
        "achievable_fixed_rounds_hoeffding": point["fixed_shots_per_site"],
        "point_observations_at_that_budget": n * point["fixed_shots_per_site"],
    }


def one_site_absorption(n, true_p, p0=0.1, p1=0.4, delta=0.05,
                        tolerance=1e-14, max_steps=10000):
    """Absorption probabilities of the assumed-model log likelihood walk.

    At time t, the likelihood depends only on the number of successes s, so
    survivors can be represented by a small map s -> probability mass. The
    unresolved mass gives a bound on the missing eventual probability (up to
    floating-point error). expected_capped_samples counts samples only to the
    stopping point of this numerical calculation.
    """
    theoretical_bounds(n, p0, p1, delta)
    if not (0.0 <= true_p <= 1.0) or not (0.0 < tolerance < 1.0) or max_steps < 1:
        raise ValueError("invalid dynamic-programming parameters")
    lower, upper = math.log(delta), math.log(n / delta)
    success = math.log(p1 / p0)
    failure = math.log((1.0 - p1) / (1.0 - p0))
    alive = {0: 1.0}
    low_mass = high_mass = expected = 0.0
    steps = 0
    for t in range(1, max_steps + 1):
        expected += sum(alive.values())
        following = {}
        for s, mass in alive.items():
            for observed, weight in ((0, 1.0 - true_p), (1, true_p)):
                new_s = s + observed
                new_mass = mass * weight
                if not new_mass:
                    continue
                score = new_s * success + (t - new_s) * failure
                if score <= lower:
                    low_mass += new_mass
                elif score >= upper:
                    high_mass += new_mass
                else:
                    following[new_s] = following.get(new_s, 0.0) + new_mass
        alive = following
        steps = t
        if sum(alive.values()) <= tolerance:
            break
    unresolved = sum(alive.values())
    return {
        "true_p": true_p,
        "prob_clear": low_mass,
        "prob_alert": high_mass,
        "unresolved_mass": unresolved,
        "probability_mass_sum": low_mass + high_mass + unresolved,
        "expected_capped_samples": expected,
        "dp_steps": steps,
    }


def sequential_prediction(n, hidden=None, true_p1=None, p0=0.1, p1=0.4, delta=0.05):
    """Unlimited-budget scan prediction; decision probabilities also apply to RR.

    Permanent site clearing and independent site streams make final decisions
    order independent. Expected total samples below apply only to sequential
    scanning, which can stop before visiting later sites after an alert.
    """
    if hidden is not None and not (0 <= hidden < n):
        raise ValueError("hidden site out of range")
    if true_p1 is None:
        true_p1 = p1
    normal = one_site_absorption(n, p0, p0, p1, delta)
    abnormal = one_site_absorption(n, true_p1, p0, p1, delta)
    reach = 1.0
    expected = 0.0
    accumulated_unresolved = 0.0
    for j in range(n):
        site = abnormal if j == hidden else normal
        expected += reach * site["expected_capped_samples"]
        accumulated_unresolved += site["unresolved_mass"]
        reach *= site["prob_clear"]
    return {
        "n": n,
        "hidden": hidden,
        "true_p1": true_p1,
        "accept_probability_estimate": reach,
        "reject_probability_estimate": 1.0 - reach,
        "scan_expected_shots_estimate": expected,
        "probability_truncation_bound": accumulated_unresolved,
        "normal_site": normal,
        "hidden_site": abnormal,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = []
    for n in args.sizes:
        rows.append({
            "bounds": theoretical_bounds(n),
            "simultaneous_readout": simultaneous_readout_bounds(n),
            "nominal_null": sequential_prediction(n),
            "nominal_first": sequential_prediction(n, 0),
            "nominal_last": sequential_prediction(n, n - 1),
            "misspecified_first_p030": sequential_prediction(n, 0, true_p1=0.30),
        })
    result = {"status": "independent numerical DP under an explicitly assumed Bernoulli oracle",
              "rows": rows}
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
