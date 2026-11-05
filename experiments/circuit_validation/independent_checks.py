"""Reproducible independent statistical checks; Python 3.8 standard library.

This file checks the certificate modules without altering them. It combines
an exact finite-horizon path calculation, an independently enclosed analytic
weighted-KL optimum, and a deliberate context-scope misuse counterexample.
The finite checks supplement the mathematical guarantees; they do not prove
universal safety, power, device fidelity, or a comparative speedup.
"""

import argparse
import hashlib
import json
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from fractions import Fraction
from pathlib import Path

import direct_tests as dt
import kl_trackstop as kt


CHECK_VERSION = "faulttune.independent_statistical_checks.v1"
HERE = Path(__file__).resolve().parent
REFERENCE_PRECISION = 120


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _probability_record(value):
    return {"exact_fraction": str(value), "decimal_approximation": float(value)}


def _adaptive_arm(successes, trials, coefficients):
    """A deliberately data-dependent rule; it has no access to true means."""
    if 0 in trials:
        return trials.index(0)
    return max(range(len(trials)), key=lambda i: Fraction(
        coefficients[i] ** 2 * (successes[i] + 1) * (trials[i] - successes[i] + 1),
        (trials[i] + 2) ** 2 * (trials[i] + 1) ** 2))


def adaptive_path_check(horizon=12):
    """Enumerate all reachable count states with exact path probabilities."""
    probabilities = (Fraction(1, 4), Fraction(3, 4), Fraction(1, 3), Fraction(1, 2))
    coefficients = (-3, -1, 3, 1)
    risk = Fraction(1, 5)
    _require(sum(p * a for p, a in zip(probabilities, coefficients)) == 0,
             "the independent DP model must lie exactly on the weighted null")
    live = {((0, 0, 0, 0), (0, 0, 0, 0)): Fraction(1)}
    accepted = rejected = Fraction(0)
    checked = 0
    per_step = []
    for step in range(1, horizon + 1):
        children = {}
        for (successes, trials), mass in live.items():
            arm = _adaptive_arm(successes, trials, coefficients)
            for bit, probability in ((0, 1 - probabilities[arm]), (1, probabilities[arm])):
                next_successes, next_trials = list(successes), list(trials)
                next_successes[arm] += bit
                next_trials[arm] += 1
                state = (tuple(next_successes), tuple(next_trials))
                children[state] = children.get(state, Fraction(0)) + mass * probability
        live = {}
        for (successes, trials), mass in children.items():
            certificate = kt.joint_glr_certificate(successes, trials, coefficients,
                                                    delta=float(risk), prior=0.5)
            checked += 1
            if certificate["decision"] == "accept":
                accepted += mass
            elif certificate["decision"] == "reject":
                rejected += mass
            else:
                live[(successes, trials)] = mass
        live_mass = sum(live.values(), Fraction(0))
        _require(accepted + rejected + live_mass == 1, "exact DP mass conservation")
        _require(accepted + rejected <= risk, "finite-horizon null crossing risk")
        per_step.append({"shots": step, "states_evaluated": len(children),
                         "live_states": len(live),
                         "combined_null_error": _probability_record(accepted + rejected),
                         "live_probability": _probability_record(live_mass)})
    return {
        "passed": True, "scope": "one fixed four-arm null, one adaptive rule, finite horizon only",
        "true_probabilities_exact": [str(p) for p in probabilities],
        "coefficients_exact": list(coefficients), "true_weighted_delta_exact": "0",
        "coefficient_note": "proportional to old/new target weights (3/4,1/4)",
        "risk_exact": str(risk), "risk_decimal": float(risk), "prior": 0.5,
        "risk_caveat": "delta=0.2 is for a short exact audit, not the formal experiment's 0.01 target",
        "allocation": "first unseen arm, otherwise argmax a_i^2*(s_i+1)*(n_i-s_i+1)/((n_i+2)^2*(n_i+1)^2), first-index ties",
        "allocation_note": "counts-only adaptive audit rule; not a benchmark implementation or performance test of Track-and-Stop",
        "horizon_shots": horizon, "certificate_states_checked": checked,
        "accept_probability": _probability_record(accepted),
        "reject_probability": _probability_record(rejected),
        "combined_null_error": _probability_record(accepted + rejected),
        "exact_final_total_probability": "1", "steps": per_step,
    }


def _integer_log_bounds(value):
    """Independent enclosure; does not use either certificate's arithmetic."""
    with localcontext() as context:
        context.prec = REFERENCE_PRECISION
        estimate = Decimal(value).ln()
        return estimate.next_minus(context), estimate.next_plus(context)


def _half_boundary_glr_bounds(successes, trials):
    """Enclose sum n_i*KL(s_i/n_i,1/2) using exact integer log arguments."""
    terms = []
    for successes_i, trials_i in zip(successes, trials):
        log_n_lower, log_n_upper = _integer_log_bounds(trials_i)
        for count in (successes_i, trials_i - successes_i):
            if not count:
                continue
            log_count_lower, log_count_upper = _integer_log_bounds(2 * count)
            with localcontext() as context:
                context.prec = REFERENCE_PRECISION
                context.rounding = ROUND_FLOOR
                lower = Decimal(count) * (log_count_lower - log_n_upper)
                context.rounding = ROUND_CEILING
                upper = Decimal(count) * (log_count_upper - log_n_lower)
            terms.append((lower, upper))
    with localcontext() as context:
        context.prec = REFERENCE_PRECISION
        context.rounding = ROUND_FLOOR
        lower = sum((term[0] for term in terms), Decimal(0))
        context.rounding = ROUND_CEILING
        upper = sum((term[1] for term in terms), Decimal(0))
    return lower, upper


def weighted_kl_closed_form_check():
    """q_i=1/2 is a known exact optimum, independently of root finding."""
    coefficients = (-0.9, -0.1, 0.9, 0.1)
    exact_coefficients = tuple(Fraction(str(value)) for value in coefficients)
    rows = []
    max_gap = Decimal(0)
    max_shots = 0
    for scale in (1, 10, 1000, 100000):
        trials = [36 * scale, 20 * scale, 180 * scale, 60 * scale]
        base_successes = [0, 8 * scale, 108 * scale, 32 * scale]
        for reflected in (False, True):
            successes = ([n - s for s, n in zip(base_successes, trials)]
                         if reflected else base_successes)
            multiplier = Fraction(-80 if reflected else 80) * scale
            _require(sum(exact_coefficients) == 0, "half-boundary exact feasibility")
            for s, n, a in zip(successes, trials, exact_coefficients):
                # d/dq [n*KL(s/n,q)] at q=1/2 is 2*n-4*s.
                _require(2 * n - 4 * s + multiplier * a == 0,
                         "exact weighted-boundary KKT stationarity")
            reference_lower, reference_upper = _half_boundary_glr_bounds(successes, trials)
            for prior in (0.5, 1.0):
                certificate = kt.joint_glr_certificate(successes, trials, coefficients, prior=prior)
                certificate_lower = Decimal(certificate["evidence_decimal_lower"])
                certificate_upper = Decimal(certificate["primal_decimal_upper"])
                _require(certificate_lower <= reference_lower <= reference_upper <= certificate_upper,
                         "independent GLR enclosure must lie inside certified dual/primal bounds")
                with localcontext() as context:
                    context.prec = REFERENCE_PRECISION
                    context.rounding = ROUND_CEILING
                    gap = certificate_upper - certificate_lower
                max_gap = max(max_gap, gap)
                max_shots = max(max_shots, sum(trials))
                rows.append({"scale": scale, "reflected": reflected, "prior": prior,
                             "successes": successes, "trials": trials, "shots": sum(trials),
                             "exact_KKT_multiplier": str(multiplier),
                             "reference_decimal_lower": str(reference_lower),
                             "reference_decimal_upper": str(reference_upper),
                             "certificate_decimal_lower": str(certificate_lower),
                             "certificate_decimal_upper": str(certificate_upper),
                             "certificate_decision": certificate["decision"]})
    return {
        "passed": True, "scope": "16 constructed weighted-boundary cases, including empirical endpoint means",
        "coefficients_exact": [str(value) for value in exact_coefficients],
        "analytic_optimum": "q_i=1/2 for every arm",
        "analytic_argument": "sum a_i/2=0 and 2*n_i-4*s_i+lambda*a_i=0 exactly; convexity proves the feasible KKT point globally optimal",
        "reference_expression": "sum_i [s_i*log(2*s_i/n_i)+(n_i-s_i)*log(2*(n_i-s_i)/n_i)], with 0*log(0)=0",
        "reference_precision": REFERENCE_PRECISION,
        "numerical_scope": "independent directed Decimal arithmetic; relies on Python Decimal correctly rounded ln, not formally verified runtime code",
        "cases_checked": len(rows), "largest_total_shots": max_shots,
        "max_certified_duality_gap_decimal": str(max_gap), "cases": rows,
    }


def wrong_context_counterexample():
    weights = (Fraction(1, 10), Fraction(9, 10))
    old = (Fraction(1), Fraction(0))
    new = (Fraction(0), Fraction(1))
    global_delta = sum(w * (n - o) for w, o, n in zip(weights, old, new))
    sampled_context = 0
    local_delta = new[sampled_context] - old[sampled_context]
    counts = [0, 0, 100, 0]
    sequential = dt.certificate_from_pairs(counts)
    fixed = dt.fixed_paired_certificate(counts)
    _require(global_delta > 0 and local_delta < 0, "local and global targets differ in sign")
    _require(sequential["decision"] == fixed["decision"] == "accept",
             "the deliberate scope-misuse example must accept the locally beneficial candidate")
    _require(sequential["shots"] == fixed["shots"] == 200, "counterexample pays both branches")
    return {
        "passed": True, "status": "expected caller-scope counterexample, not a certificate defect",
        "public_target_weights_exact": [str(w) for w in weights],
        "old_context_probabilities_exact": [str(p) for p in old],
        "new_context_probabilities_exact": [str(p) for p in new],
        "global_true_delta": _probability_record(global_delta),
        "sampled_context": sampled_context, "local_true_delta": _probability_record(local_delta),
        "pair_counts_00_01_10_11": counts, "paid_shots": 200,
        "sequential_local_certificate": sequential, "fixed_local_certificate": fixed,
        "interpretation": "both accept decisions are correct for the fixed sampled context; interpreting either as a mixed-target certificate gives false acceptance with probability one",
        "required_scope": "mixed-target direct policies need fresh IID contexts from the public target distribution for each pair; a fixed context certifies only that context",
        "fixed_test_requirement": "one final look at a pair count fixed before validation",
    }


def build_checks():
    hashes = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
              for name in ("independent_checks.py", "kl_trackstop.py", "direct_tests.py")}
    return {"schema_version": CHECK_VERSION, "passed": True,
            "source_sha256": hashes,
            "scope": "deterministic independent audits; no Monte Carlo seed and no device samples",
            "not_established": ["universal safety by finite enumeration", "95 percent power",
                                "comparative cost speedup", "physical-device fidelity"],
            "adaptive_path_dp": adaptive_path_check(),
            "weighted_KL_closed_form": weighted_kl_closed_form_check(),
            "wrong_context_counterexample": wrong_context_counterexample()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results" / "independent_checks.json")
    args = parser.parse_args()
    result = build_checks()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
                           encoding="utf-8")
    print("Independent checks passed: {} adaptive DP states, {} closed-form KL cases, scope counterexample; {}".format(
        result["adaptive_path_dp"]["certificate_states_checked"],
        result["weighted_KL_closed_form"]["cases_checked"], args.output.resolve()))


if __name__ == "__main__":
    main()
