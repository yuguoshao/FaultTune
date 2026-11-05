"""Paid holdout certificates for a pilot-frozen two-angle design.

Only validation counts enter this module. Conditional on the separate paid
pilot, angles, allocations, and the cumulative sample sizes at every look
must be fixed before drawing validation data. Future validation observations
must be independent Bernoulli observations with stationary arm means.

At look k, the Fisher lower tail for the observed direction is tested at
delta/(2*look_count). Uniform spending over a finite, predeclared look set
controls each wrong direction by delta/2, and either direction at equality
by delta. Integrating over the independent pilot preserves these bounds.
The pilot's observations cannot be inserted into these validation counts.

Numerics: fixed at most 32 (configurable in the design) relative tail terms
plus a monotone geometric remainder, with outward Decimal arithmetic. This
is an ENCLOSURE of a conditional hypergeometric tail, not its exact numeric
p-value. A sampling-without-replacement Hoeffding upper bound is also valid
and may tighten it. Floating-point screens never authorize a decision.

The frozen direct_tests module supplies its already audited interval and
log-factorial primitives. No frozen source is modified or imported as truth.
"""

import hashlib
import json
import math
import numbers
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
import sys


_NUMERIC_PATH = Path(__file__).resolve().parents[1] / "circuit_validation" / "direct_tests.py"
sys.path.insert(0, str(_NUMERIC_PATH.parent))
from direct_tests import _Bounds, _as_float_bounds, _log_factorial

NUMERIC_SOURCE_SHA256 = hashlib.sha256(_NUMERIC_PATH.read_bytes()).hexdigest()


def _integer(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(name + " must be an integer")
    value = int(value)
    if value < (1 if positive else 0):
        raise ValueError(name + " is outside its allowed range")
    return value


def _risk(delta, look_count):
    look_count = _integer(look_count, "look_count", True)
    if isinstance(delta, bool):
        raise ValueError("delta must be a probability")
    risk = delta if isinstance(delta, Fraction) else Fraction(str(delta))
    if not 0 < risk < 1:
        raise ValueError("delta must lie strictly between zero and one")
    tail = risk / (2 * look_count)
    return risk, tail, _Bounds(tail.numerator) / tail.denominator


def _inputs(old_successes, old_trials, new_successes, new_trials):
    s0 = _integer(old_successes, "old_successes")
    n0 = _integer(old_trials, "old_trials")
    s1 = _integer(new_successes, "new_successes")
    n1 = _integer(new_trials, "new_trials")
    if s0 > n0 or s1 > n1:
        raise ValueError("Success counts cannot exceed trial counts")
    return s0, n0, s1, n1


def _log_choose(n, k):
    if not 0 <= k <= n:
        raise ValueError("Invalid binomial coefficient")
    return _log_factorial(n) - _log_factorial(k) - _log_factorial(n - k)


def _selected_tail(s0, n0, s1, n1):
    """Return the left tail matching the observed difference, or no direction."""
    cross = s1 * n0 - s0 * n1
    if not n0 or not n1 or not cross:
        return None
    population, draws = n0 + n1, n1
    if cross < 0:
        return "accept", s1, population, s0 + s1, draws, abs(cross)
    return "reject", n1 - s1, population, population - s0 - s1, draws, abs(cross)


def _hypergeom_left_enclosure(k, population, successes, draws, tail_terms=32):
    """Enclose P[K<=k] for a k strictly below its hypergeometric mean.

    For j in the left tail, p(j-1)/p(j) is decreasing as j decreases.
    The uncomputed remaining mass is therefore bounded by a geometric sum.
    All combinatorial input expressions are exact integers.
    """
    tail_terms = _integer(tail_terms, "tail_terms", True)
    lower = max(0, draws - (population - successes))
    upper = min(draws, successes)
    if k < lower:
        return _Bounds(0), {"terms": 0, "complete_tail": True}
    if k >= upper:
        return _Bounds(1), {"terms": 0, "complete_tail": True}
    if k * population >= draws * successes:
        raise ValueError("This enclosure expects the selected strict lower tail")
    log_mass = (_log_choose(successes, k)
                + _log_choose(population - successes, draws - k)
                - _log_choose(population, draws))
    mass = log_mass.exp()
    term, total, index, terms = _Bounds(1), _Bounds(1), k, 1
    remainder = _Bounds(0)
    while index > lower:
        numerator = index * (population - successes - draws + index)
        denominator = (successes - index + 1) * (draws - index + 1)
        if not 0 <= numerator < denominator:
            raise ArithmeticError("Selected tail does not have decreasing mass ratios")
        ratio = _Bounds(numerator) / denominator
        remainder = term * ratio / (1 - ratio)
        if terms >= tail_terms:
            break
        term = term * ratio
        total = total + term
        index -= 1
        terms += 1
    if index == lower:
        remainder = _Bounds(0)
    enclosure = mass * _Bounds(total.lo, (total + remainder).hi)
    return _Bounds(max(Decimal(0), enclosure.lo), min(Decimal(1), enclosure.hi)), {
        "terms": terms, "complete_tail": index == lower,
        "support": [lower, upper], "observed_k": k,
        "log_observed_mass_bounds": _as_float_bounds(log_mass),
        "relative_remainder_upper_decimal": str(remainder.hi)}


def fisher_log_tail_screen(old_successes, old_trials, new_successes, new_trials,
                           look_count, delta=0.01):
    """Return (approx_log_geometric_upper, approx_log_alpha), SCREEN ONLY.

    A runner may ask for the Decimal certificate when the first value is
    <= the second plus a small screen tolerance. A screen may delay a valid
    crossing; neither its floating arithmetic nor its result is a certificate.
    """
    s0, n0, s1, n1 = _inputs(old_successes, old_trials, new_successes, new_trials)
    unused, alpha, unused_bounds = _risk(delta, look_count)
    log_alpha = math.log(float(alpha))
    selected = _selected_tail(s0, n0, s1, n1)
    if selected is None:
        return 0.0, log_alpha
    unused, k, size, success, draws, cross = selected
    def log_choose(n, j):
        return math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)
    log_mass = log_choose(success, k) + log_choose(size - success, draws - k) - log_choose(size, draws)
    lower = max(0, draws - (size - success))
    if k == lower:
        log_upper = log_mass
    else:
        ratio = k * (size - success - draws + k) / ((success - k + 1) * (draws - k + 1))
        log_upper = log_mass - math.log1p(-ratio) if ratio < 1 else 0.0
    hoeffding_log = -2.0 * (cross / size) ** 2 / min(n0, n1)
    return min(0.0, log_upper, hoeffding_log), log_alpha


def fisher_certificate(old_successes, old_trials, new_successes, new_trials,
                       look_index, look_count, delta=0.01, tail_terms=32):
    """Certified decision for new_mean-old_mean at a PREDECLARED look.

    look_index is 1-based. Counts must be validation-only. Geometric looks
    and their unequal arm sample sizes must be frozen conditional on pilot.
    Reject/accept refer to positive/negative mean differences respectively;
    the caller is responsible for the known target contrast's orientation.
    """
    s0, n0, s1, n1 = _inputs(old_successes, old_trials, new_successes, new_trials)
    risk, alpha, alpha_bounds = _risk(delta, look_count)
    look_index = _integer(look_index, "look_index", True)
    if look_index > look_count:
        raise ValueError("look_index exceeds the registered look_count")
    tail_terms = _integer(tail_terms, "tail_terms", True)
    result = {"method": "fisher_uniform_spending_certified_tail",
              "decision": "abstain", "look_index": look_index, "look_count": int(look_count),
              "delta": float(risk), "delta_exact": str(risk),
              "tail_alpha_exact": str(alpha), "tail_alpha_bounds": _as_float_bounds(alpha_bounds),
              "old_successes": s0, "old_trials": n0, "new_successes": s1, "new_trials": n1,
              "shots_stage2": n0 + n1, "tail_terms_limit": tail_terms,
              "est_mean_difference": s1 / n1 - s0 / n0 if n0 and n1 else None,
              "p_value_bounds": [1.0, 1.0], "p_value_decimal_bounds": ["1", "1"],
              "direction": None, "anytime_valid": False,
              "valid_over_registered_looks": True,
              "required_use": "independent paid holdout; pilot-frozen finite schedule and arm counts",
              "numeric_scope": "outward Decimal tail enclosure, geometric remainder, and Hoeffding bound",
              "numeric_source_sha256": NUMERIC_SOURCE_SHA256}
    selected = _selected_tail(s0, n0, s1, n1)
    if selected is None:
        return result
    direction, k, size, success, draws, cross = selected
    tail, details = _hypergeom_left_enclosure(k, size, success, draws, tail_terms)
    # Hoeffding's bound for sampling without replacement, using the smaller
    # of the complementary groups. Deviation from its conditional mean is
    # cross/size. It is independent of the unknown common Bernoulli mean.
    exponent = -(_Bounds(2 * cross * cross) / (size * size * min(n0, n1)))
    hoeffding = exponent.exp()
    upper = min(tail.hi, hoeffding.hi)
    if tail.lo > upper:
        raise ArithmeticError("Independent valid tail bounds unexpectedly disagree")
    enclosed = _Bounds(tail.lo, upper)
    result.update({"direction": direction, "p_value_bounds": _as_float_bounds(enclosed),
                   "p_value_decimal_bounds": [str(enclosed.lo), str(enclosed.hi)],
                   "tail_computation": details, "hypergeometric_tail_bounds": _as_float_bounds(tail),
                   "hoeffding_tail_upper_decimal": str(hoeffding.hi)})
    if enclosed.hi <= alpha_bounds.lo:
        result["decision"] = direction
    return result


def sufficient_final_gap(old_trials, new_trials, look_count, delta=0.01, beta=0.045):
    """Design bound, not a certificate or an assertion of actual power.

    Conditional on pilot, if the TRUE signed gap magnitude exceeds this
    value, the potential final-look correct decision occurs with probability
    >=1-beta. Earlier wrong stopping costs at most delta/2, so full-procedure
    power is >=1-beta-delta/2. At delta=.01 and beta=.045 this is .95.
    If this condition holds only on a pilot event of probability >=1-gamma,
    the unconditional lower bound is 1-gamma-beta-delta/2.
    """
    n0, n1 = _integer(old_trials, "old_trials", True), _integer(new_trials, "new_trials", True)
    unused, alpha, unused_bounds = _risk(delta, look_count)
    if not 0 < beta < 1:
        raise ValueError("beta must be between zero and one")
    effective = n0 * n1 / (n0 + n1)
    safety = math.sqrt(min(n0, n1) * math.log(1 / float(alpha)) / (2 * effective * effective))
    power = math.sqrt(math.log(1 / beta) / (2 * effective))
    return {"sufficient_true_gap": safety + power, "safety_component": safety,
            "power_component": power, "conditional_power_lower": 1 - beta - float(delta) / 2,
            "scope": "analytic sufficient design condition; floating evaluation is descriptive"}


def independent_checks():
    """Exact small-tail comparisons; optional independent SciPy large checks."""
    exact_checked, scipy_checked = 0, 0
    for size in (2, 3, 4, 8, 12, 20, 31, 40, 60):
        for draws in sorted(set((1, max(1, size // 3), size // 2, size - 1))):
            for success in sorted(set((0, 1, size // 3, size // 2, size - 1, size))):
                lo, hi = max(0, draws - (size - success)), min(draws, success)
                positions = {lo, hi, (lo + hi) // 2, max(lo, min(hi, draws * success // size - 1))}
                for k in sorted(positions):
                    if k * size >= draws * success:
                        continue
                    bound, unused = _hypergeom_left_enclosure(k, size, success, draws)
                    denominator = math.comb(size, draws)
                    value = Fraction(sum(math.comb(success, j) * math.comb(size - success, draws - j)
                                         for j in range(lo, k + 1)), denominator)
                    assert Fraction(bound.lo) <= value <= Fraction(bound.hi)
                    exact_checked += 1
    large_cases = [(13, 100, 3, 70), (550, 1000, 440, 900),
                   (501000, 1000000, 1194000, 2400000),
                   (5, 10000000, 0, 1000000), (3000000, 10000000, 200000, 1000000)]
    try:
        from scipy.stats import hypergeom
    except ImportError:
        hypergeom = None
    if hypergeom is not None:
        for counts in large_cases:
            cert = fisher_certificate(*counts, look_index=3, look_count=20)
            selected = _selected_tail(*counts)
            direction, k, size, success, draws, unused = selected
            reference = float(hypergeom.cdf(k, size, success, draws))
            lo, hi = cert["p_value_bounds"]
            # SciPy is a floating reference, not the directed-arithmetic authority.
            assert lo <= reference * (1 + 1e-7) + 1e-300
            assert hi >= reference * (1 - 1e-7) - 1e-300
            if cert["decision"] != "abstain":
                assert reference <= float(Fraction(cert["tail_alpha_exact"])) * (1 + 1e-7)
            scipy_checked += 1
    assert fisher_certificate(0, 10, 0, 10, 1, 1)["decision"] == "abstain"
    rejected_invalid = 0
    for args in ((11, 10, 0, 10, 1, 1), (0, 10, 0, 10, 0, 1), (0, 10, 0, 10, 2, 1)):
        try:
            fisher_certificate(*args)
        except ValueError:
            rejected_invalid += 1
    assert rejected_invalid == 3
    return {"passed": True, "exact_small_tail_checks": exact_checked,
            "independent_scipy_large_checks": scipy_checked,
            "invalid_inputs_rejected": rejected_invalid,
            "numeric_source_sha256": NUMERIC_SOURCE_SHA256,
            "scope": "finite numerical cross-checks, not a proof of all statistical assumptions"}


if __name__ == "__main__":
    print(json.dumps(independent_checks(), indent=2))
