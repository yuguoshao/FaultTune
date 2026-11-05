"""Independent mathematics for sum-sign validation; Python 3.8+, standard library.

The characteristic-rate calculation is numerical, NOT a machine-certified
enclosure. The separate finite-horizon counterexamples use integer arithmetic
and exact rational probabilities; their stated rational lower bounds are exact.
Neither calculation runs a Monte Carlo policy or accesses quantum hardware.
"""

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

try:
    from .audit_math import one_site_absorption
except ImportError:
    from audit_math import one_site_absorption


def bernoulli_kl(p, q):
    """KL(Ber(p) || Ber(q)) for interior probabilities, in natural units."""
    if not (0.0 < p < 1.0 and 0.0 < q < 1.0):
        raise ValueError("KL probabilities must lie strictly between zero and one")
    if p == q:
        return 0.0
    difference = q - p
    if abs(difference) < 0.25 * min(p, 1.0 - p):
        terms = (-p * math.log1p(difference / p),
                 -(1.0 - p) * math.log1p(-difference / (1.0 - p)))
    else:
        terms = (p * (math.log(p) - math.log(q)),
                 (1.0 - p) * (math.log1p(-p) - math.log1p(-q)))
    return max(0.0, math.fsum(terms))


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("{} must be a positive integer".format(name))


def _right_kl_bracket(p0, level, iterations=80):
    """Numerical outer q bracket for kl(p0,q)=level on q >= p0.

    Monotone bisection keeps both endpoints. Floating-point evaluations of KL
    themselves are not enclosed by interval arithmetic.
    """
    if level == 0.0:
        return p0, p0
    left, right = p0, 1.0
    for _ in range(iterations):
        middle = (left + right) / 2.0
        if middle == left or middle == right:
            break
        if bernoulli_kl(p0, middle) < level:
            left = middle
        else:
            right = middle
    return left, right


def characteristic_null(group_sizes, p0=0.1, gain=0.3, delta=0.05):
    """Exact-formula rate, evaluated numerically, for arbitrary disjoint groups.

    The true vector is all p0. An action reads ONE FULL group and costs one
    round; observations within and across actions are independent Bernoulli.
    The test must distinguish every negative from every positive sum change.
    The zero-change boundary is used to compute the opposite side's infimum.

    Let d(q)=kl(p0,q). The round characteristic information I* is the unique z
    with m_g*d(q_g)=z and sum m_g*(q_g-p0)=gain. At this point the maximizing
    round proportions are w_g proportional to 1/d'(q_g). This is an analytic
    characterization, not a novel algorithm or an attainability proof for any
    implemented confidence rule.

    A feasible q gives the information upper bound max_g m_g*d(q_g). With
    w_g proportional to 1/d'(q_g), lambda=1/sum 1/d'(q_g), the Lagrange dual
    gives the lower bound sum w_g*m_g*d(q_g)+lambda*(gain-sum m_g*(q_g-p0)).
    We report this primal/dual bracket with outward numerical margins. It is
    still NOT a formal floating-point certificate. Cost lower bounds divide
    by the INFORMATION UPPER endpoint, never by a feasible allocation's rate.
    """
    sizes = tuple(group_sizes)
    if not sizes:
        raise ValueError("at least one group is required")
    for size in sizes:
        _positive_integer(size, "group size")
    n = sum(sizes)
    if not (0.0 < p0 < 1.0 and 0.0 < delta < 0.5):
        raise ValueError("require 0 < p0 < 1 and 0 < delta < 1/2")
    if not math.isfinite(gain) or not 0.0 < gain < n * (1.0 - p0):
        raise ValueError("gain must be positive and admit an opposite-sign alternative")
    uniform_q = p0 + gain / n
    if not p0 < uniform_q < 1.0:
        raise ValueError("gain is too close to a floating-point boundary")
    uniform_information = bernoulli_kl(p0, uniform_q)
    if uniform_information == 0.0:
        raise ValueError("gain is too small for this floating-point calculation")

    # Analytic starting bracket: at min(m)*d(qbar) all q_g <= qbar;
    # at max(m)*d(qbar) all q_g >= qbar.
    z_left = min(sizes) * uniform_information
    z_right = max(sizes) * uniform_information
    iterations = 0
    for iterations in range(1, 81):
        middle = (z_left + z_right) / 2.0
        if middle == z_left or middle == z_right:
            break
        intervals = [_right_kl_bracket(p0, middle / size) for size in sizes]
        effect_left = math.fsum(size * (interval[0] - p0)
                                for size, interval in zip(sizes, intervals))
        effect_right = math.fsum(size * (interval[1] - p0)
                                 for size, interval in zip(sizes, intervals))
        if effect_right < gain:
            z_left = middle
        elif effect_left > gain:
            z_right = middle
        else:
            # Do not infer a sign from an inner-bracket midpoint. The separate
            # primal/dual calculation below can remain tight in this case.
            break

    midpoint = (z_left + z_right) / 2.0
    q_values = [_right_kl_bracket(p0, midpoint / size)[1] for size in sizes]
    effect = math.fsum(size * (q - p0) for size, q in zip(sizes, q_values))

    # Make an explicit opposite-side witness. A small positive slack also
    # avoids treating a rounded, slightly infeasible q as an upper bound.
    slack = min(max(1e-13, gain * 1e-12),
                (n * (1.0 - p0) - gain) / 8.0)
    missing = gain + slack - effect
    if missing > 0.0:
        chosen = max(range(len(sizes)),
                     key=lambda g: sizes[g] * (1.0 - q_values[g]))
        q_values[chosen] += missing / sizes[chosen]
    if any(not p0 < q < 1.0 for q in q_values):
        raise ValueError("characteristic witness is too close to a numeric boundary")
    effect = math.fsum(size * (q - p0) for size, q in zip(sizes, q_values))
    if effect < gain:
        raise ArithmeticError("numerical witness failed the feasibility check")

    information = [size * bernoulli_kl(p0, q) for size, q in zip(sizes, q_values)]
    inverse_derivatives = [q * (1.0 - q) / (q - p0) for q in q_values]
    normalization = math.fsum(inverse_derivatives)
    proportions = [value / normalization for value in inverse_derivatives]
    multiplier = 1.0 / normalization
    primal_upper = max(information)
    dual_lower = math.fsum(w * value for w, value in zip(proportions, information))
    dual_lower += multiplier * (gain - effect)
    margin = max(1e-13, abs(primal_upper) * 1e-10)
    lower = max(0.0, dual_lower - margin)
    upper = primal_upper + margin
    kappa = bernoulli_kl(1.0 - delta, delta)
    kappa_lower = max(0.0, kappa - max(1e-13, abs(kappa) * 1e-12))
    object_information_upper = uniform_information + max(1e-13, uniform_information * 1e-10)
    return {
        "n": n,
        "group_sizes": list(sizes),
        "p0": p0,
        "gain": gain,
        "delta": delta,
        "round_information_estimate": (lower + upper) / 2.0,
        "round_information_bracket": [lower, upper],
        "round_information_raw_primal_upper": primal_upper,
        "round_information_raw_dual_lower": dual_lower,
        "outer_root_bracket": [z_left, z_right],
        "outer_iterations": iterations,
        "opposite_side_witness": q_values,
        "witness_effect_sum": effect,
        "witness_group_information": information,
        "round_optimal_proportions_estimate": proportions,
        "objects_per_round_at_these_proportions": math.fsum(
            w * size for w, size in zip(proportions, sizes)),
        "object_information_rate": uniform_information,
        "uniform_object_alternative": uniform_q,
        "kappa": kappa,
        "rounds_lower_using_information_upper": kappa_lower / upper,
        "objects_lower_using_information_upper": kappa_lower / object_information_upper,
        "scope": "homogeneous null; disjoint full-group actions; complete uniformly delta-correct sum-sign tests; no abstention power shortcut",
        "object_bound_scope": "homogeneous null with any legal independent-Bernoulli readout actions: the uniform boundary alternative gives information d per observed object",
        "object_rate_note": "equal per-object sampling is attainable here by complete partition sweeps; the object bound is not the star round bound multiplied by a group size",
        "numeric_status": "double-precision primal/dual bracket with outward margins, NOT a machine-certified enclosure",
        "cost_note": "round-optimal proportions need not minimize object samples; switching costs are not included",
    }


def exact_lr_state(steps, successes, n=16):
    """Exact assumed-p0=1/10, assumed-p1=2/5, delta=1/20 LR boundaries.

    LR = 2**(steps+successes) / 3**(steps-successes).
    Returns clear, alert, or unresolved. No logarithms or rounding are used.
    """
    _positive_integer(n, "n")
    if (isinstance(steps, bool) or not isinstance(steps, int) or steps < 0
            or isinstance(successes, bool) or not isinstance(successes, int)
            or not 0 <= successes <= steps):
        raise ValueError("invalid step/success counts")
    numerator = 1 << (steps + successes)
    denominator = 3 ** (steps - successes)
    if 20 * numerator <= denominator:
        return "clear"
    if numerator >= 20 * n * denominator:
        return "alert"
    return "unresolved"


def exact_site_absorption(probability_numerator, probability_denominator,
                          horizon=256, n=16):
    """Integer DP for first-exit probabilities by a fixed per-site horizon.

    At time t every integer weight has denominator probability_denominator**t.
    Already absorbed weights are multiplied by that denominator each step;
    this accounts for all unused continuations without resampling a cleared
    site. The unresolved mass is retained, never assumed eventually clear.
    """
    _positive_integer(probability_denominator, "probability denominator")
    if (isinstance(probability_numerator, bool)
            or not isinstance(probability_numerator, int)
            or not 0 <= probability_numerator <= probability_denominator):
        raise ValueError("invalid probability numerator")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 0:
        raise ValueError("horizon must be a nonnegative integer")
    _positive_integer(n, "n")
    probability = Fraction(probability_numerator, probability_denominator)
    numerator, denominator = probability.numerator, probability.denominator
    alive = {0: 1}
    clear_weight = alert_weight = 0
    scale = 1
    for step in range(1, horizon + 1):
        scale *= denominator
        clear_weight *= denominator
        alert_weight *= denominator
        following = {}
        for successes, weight in alive.items():
            for observed, factor in ((0, denominator - numerator), (1, numerator)):
                child_weight = weight * factor
                if child_weight == 0:
                    continue
                child_successes = successes + observed
                state = exact_lr_state(step, child_successes, n)
                if state == "clear":
                    clear_weight += child_weight
                elif state == "alert":
                    alert_weight += child_weight
                else:
                    following[child_successes] = following.get(child_successes, 0) + child_weight
        alive = following
        if clear_weight + alert_weight + sum(alive.values()) != scale:
            raise ArithmeticError("exact probability mass was not conserved")
    return {
        "n": n,
        "horizon": horizon,
        "probability": probability,
        "clear": Fraction(clear_weight, scale),
        "alert": Fraction(alert_weight, scale),
        "unresolved": Fraction(sum(alive.values()), scale),
    }


def rational_record(value):
    """An exact fraction plus an explicitly approximate floating-point view."""
    return {"numerator": str(value.numerator), "denominator": str(value.denominator),
            "float_approximation": float(value)}


def rational_floor_record(value, places=12):
    """A compact exact lower bound; decimal digits are obtained by integer floor."""
    scale = 10 ** places
    floor_numerator = value.numerator * scale // value.denominator
    whole, fractional = divmod(floor_numerator, scale)
    return {"numerator": str(floor_numerator), "denominator": str(scale),
            "decimal_lower": "{}.{:0{}d}".format(whole, fractional, places),
            "rounding": "exact rational floor; never rounded upward"}


def counterexample_certificate(site_results, multiplicities, gain=Fraction(3, 10)):
    """Product event: every independent site first clears within its horizon.

    This event forces global acceptance with permanent clearing and any
    unfinished-site schedule. If each horizon is T, a total budget >= N*T
    suffices on the event. It does NOT imply the same bound for smaller
    budgets, which can return abstain instead.
    """
    if not multiplicities:
        raise ValueError("a nonempty site family is required")
    n = 0
    event_probability = Fraction(1)
    remote_change = Fraction(0)
    max_object_budget = 0
    factors = []
    for name, count in multiplicities.items():
        _positive_integer(count, "multiplicity")
        site = site_results[name]
        event_probability *= site["clear"] ** count
        remote_change += count * (site["probability"] - Fraction(1, 10))
        max_object_budget += count * site["horizon"]
        n += count
        factors.append({"site_result": name, "power": count})
    if any(site_results[name]["n"] != n for name in multiplicities):
        raise ValueError("site boundary n must match the total number of monitors")
    true_delta = remote_change - gain
    delta = Fraction(1, 20)
    return {
        "n": n,
        "local_gain": rational_record(gain),
        "remote_change": rational_record(remote_change),
        "true_delta": rational_record(true_delta),
        "candidate_is_harmful_exact": true_delta > 0,
        "event_factors": factors,
        "false_acceptance_lower": rational_floor_record(event_probability),
        "event_probability_float_approximation": float(event_probability),
        "event_probability_exceeds_delta_exact": event_probability > delta,
        "displayed_lower_exceeds_delta_exact": (
            event_probability.numerator * 10 ** 12 // event_probability.denominator
        ) * 20 > 10 ** 12,
        "sufficient_total_object_budget_on_event": max_object_budget,
        "event": "all sites first hit the lower boundary by their finite per-site horizon, before any upper hit",
        "scope": "fixed independent Bernoulli streams; old single-effect guard; permanent clearing; no early abstention or total budget at least the stated sufficient budget",
        "proof_status": "exact integer boundary comparisons and rational finite-event probability; no Monte Carlo or floating-point assumption",
    }


def _serialized_site(site):
    p = float(site["probability"])
    floating = one_site_absorption(site["n"], p, tolerance=1e-300, max_steps=site["horizon"])
    return {
        "n": site["n"],
        "horizon": site["horizon"],
        "true_probability": rational_record(site["probability"]),
        "clear_by_horizon": rational_record(site["clear"]),
        "alert_by_horizon": rational_record(site["alert"]),
        "unresolved_at_horizon": rational_record(site["unresolved"]),
        "mass_conservation_exact": site["clear"] + site["alert"] + site["unresolved"] == 1,
        "floating_dp_comparison": floating,
        "floating_clear_absolute_difference": abs(floating["prob_clear"] - float(site["clear"])),
    }


def build_artifact(sizes=(8, 16, 32), horizons=(256, 512)):
    characteristics = []
    for n in sizes:
        _positive_integer(n, "size")
        if n < 2 or n % 2:
            raise ValueError("artifact sizes must be even and at least two")
        for label, groups in (("full_readout", [n]), ("balanced_two", [n // 2, n // 2]),
                              ("star_maximal_groups", [1, n - 1]), ("singletons", [1] * n)):
            row = characteristic_null(groups)
            row["label"] = label
            row["star_note"] = (
                "For round cost one, star independent-set subsets are information-dominated by its two maximal groups; this does not extend to object-cost dominance."
                if label == "star_maximal_groups" else None)
            characteristics.append(row)
    certificates = []
    for horizon in horizons:
        _positive_integer(horizon, "horizon")
        sites = {"normal_1_10": exact_site_absorption(1, 10, horizon),
                 "weak_1_5": exact_site_absorption(1, 5, horizon),
                 "dense_11_80": exact_site_absorption(11, 80, horizon)}
        certificates.append({
            "horizon": horizon,
            "sites": {name: _serialized_site(site) for name, site in sites.items()},
            "counterexamples": {
                "four_weak": counterexample_certificate(sites, {"normal_1_10": 12, "weak_1_5": 4}),
                "sixteen_dense_weak": counterexample_certificate(sites, {"dense_11_80": 16}),
            },
        })
    return {
        "schema_version": "faulttune.aggregate_analytic.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "status": "classical analytic characterization and exact finite-event counterexamples; no originality or hardware performance claim",
        "characteristic_parameters": {"p0": 0.1, "gain": 0.3, "delta": 0.05},
        "characteristic_results": characteristics,
        "guard_parameters": {"n": 16, "p0": "1/10", "assumed_boundary_probability": "2/5",
                             "local_gain": "3/10", "delta": "1/20",
                             "lower_test": "20 * 2**(t+s) <= 3**(t-s)",
                             "upper_test": "2**(t+s) >= 320 * 3**(t-s)"},
        "finite_horizon_results": certificates,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--horizons", nargs="+", type=int, default=[256, 512])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_artifact(args.sizes, args.horizons)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    for row in result["characteristic_results"]:
        if row["label"] == "star_maximal_groups":
            print("N={} star numerical round lower={:.9f}; information bracket={}".format(
                row["n"], row["rounds_lower_using_information_upper"], row["round_information_bracket"]))
    for result_row in result["finite_horizon_results"]:
        for name, certificate in result_row["counterexamples"].items():
            print("T={} {} exact false-accept lower={} > .05: {}".format(
                result_row["horizon"], name, certificate["false_acceptance_lower"]["decimal_lower"],
                certificate["displayed_lower_exceeds_delta_exact"]))


if __name__ == "__main__":
    main()
