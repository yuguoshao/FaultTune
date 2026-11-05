"""Deterministic analytic/arithmetic checks; no Monte Carlo or stored trial reads.

For m=1, each discordant pair contributes 2*t or 2*(1-t), so the COMPLETE
continuous mixture equals 2**D*B(s+1/2,f+1/2)/pi exactly. For integers this is
(2s)!*(2f)! / (2**D*s!*f!*D!), a rational number. A finite chord lower bound
approaches this value but is not asserted equal to it.

An independent full-integral enclosure changes variables to x=sqrt(odds)
and uses reciprocal symmetry. The integral becomes
  (2/pi)*integral_0^1 [R(x*x)+R(1/(x*x))]/(1+x*x) dx.
Its numerator and denominator are polynomials with nonnegative coefficients.
On [a,b], N(a)/D(b) <= N(x)/D(x) <= N(b)/D(a). Exact rational rectangle
endpoints plus directed accumulation enclose the ENTIRE integral, including
both infinite log-odds tails. This path does not call the Fisher CGF kernel.
"""
from collections import defaultdict
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time

import mpmath as mp

import continuous_fisher as continuous


Bounds = continuous.Bounds
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/structured_validation/continuous_fisher_checks.json"


def fraction_bounds(value):
    return Bounds(value.numerator) / value.denominator


def beta_exact(positive, negative):
    total = positive + negative
    return Fraction(math.factorial(2 * positive) * math.factorial(2 * negative),
                    2 ** total * math.factorial(positive) * math.factorial(negative)
                    * math.factorial(total))


def pair_blocks(positive, negative, concordant=True):
    result = ([{"m": 1, "minus_successes": 0, "plus_successes": 1}] * positive
              + [{"m": 1, "minus_successes": 1, "plus_successes": 0}] * negative)
    if concordant:
        result += [{"m": 1, "minus_successes": 0, "plus_successes": 0},
                   {"m": 1, "minus_successes": 1, "plus_successes": 1}]
    return result


def refined_nodes(nodes, subdivisions=4):
    result = []
    for a, b in zip(nodes[:-1], nodes[1:]):
        result.extend(a + (b - a) * Fraction(k, subdivisions) for k in range(subdivisions))
    return tuple(result + [nodes[-1]])


def full_mixture_rectangle_bounds(blocks, subdivisions=1024):
    """Independent exact-positive-polynomial full-integral enclosure."""
    polynomials, positive_power, negative_power, constant = [], 0, 0, 1
    for block in blocks:
        m, minus, plus = (block[key] for key in ("m", "minus_successes", "plus_successes"))
        total = minus + plus
        first, last = max(0, total - m), min(m, total)
        coefficients = [math.comb(m, k) * math.comb(m, total - k)
                        for k in range(first, last + 1)]
        assert coefficients[0] > 0
        polynomials.append(coefficients)
        positive_power += 2 * (plus - first)
        negative_power += 2 * (minus - first)
        constant *= math.comb(2 * m, total)

    def numerator_denominator(x):
        numerator = constant * (x ** positive_power + x ** negative_power)
        square = x * x
        denominator = 1 + square
        for coefficients in polynomials:
            value = Fraction(0)
            for coefficient in reversed(coefficients):
                value = value * square + coefficient
            denominator *= value
        return numerator, denominator

    left_num, left_den = numerator_denominator(Fraction(0))
    lower, upper = Bounds(0), Bounds(0)
    for k in range(1, subdivisions + 1):
        right_num, right_den = numerator_denominator(Fraction(k, subdivisions))
        lower += fraction_bounds(left_num / right_den)
        upper += fraction_bounds(right_num / left_den)
        left_num, left_den = right_num, right_den
    return Bounds(lower.lo, upper.hi) * 2 / subdivisions / continuous.pi_bounds()


def lower_value(certificate):
    return Bounds(Decimal(certificate["log_evidence_lower_bound"])).exp()


def finite_wrong_crossing_probability(q, delta, horizon):
    """Exact DP for EVER crossing a wrong-sign full m=1 boundary.

Earlier correct-sign crossings do not absorb the path: this larger event
also bounds a delayed numerical stopping rule's possible wrong crossings.
"""
    states = {(0, 0): Fraction(1)}
    absorbed = Fraction(0)
    for unused in range(horizon):
        updated = defaultdict(Fraction)
        for (positive, negative), mass in states.items():
            for success, probability in ((1, q), (0, 1 - q)):
                s, f = positive + success, negative + 1 - success
                wrong = (s != f if q == Fraction(1, 2)
                         else s < f if q > Fraction(1, 2) else s > f)
                if wrong and beta_exact(s, f) >= 1 / delta:
                    absorbed += mass * probability
                else:
                    updated[s, f] += mass * probability
        states = updated
    assert absorbed + sum(states.values(), Fraction(0)) == 1
    return absorbed


def run_checks():
    began = time.perf_counter()
    sources = continuous.source_hashes()
    own = Path(__file__).resolve()
    sources[own.relative_to(ROOT).as_posix()] = hashlib.sha256(own.read_bytes()).hexdigest()
    assert sources["experiments/structured_validation/conditional_fisher.py"] == "0b7840375354931a67046441fc8c96bb977657d75da7f801142d5bf5ba84147a"
    assert sources["experiments/circuit_validation/direct_tests.py"] == "2f45bd4b59536352374dafbf6805f0ea927a3d7cff82863989877fb65f17c97a"
    pi = continuous.pi_bounds()
    with mp.workdps(120):
        assert mp.mpf(str(pi.lo)) < mp.pi < mp.mpf(str(pi.hi))
        for a, b in (("-1", "-1"), ("-1e-30", "0"), ("-0.5", "0"), ("-10000", "0")):
            mean, branch = continuous._exponential_chord_mean(Decimal(a), Decimal(b), Decimal(0))
            aa, bb = mp.mpf(a), mp.mpf(b)
            exact = mp.exp(aa) if aa == bb else mp.exp(min(aa, bb)) * mp.expm1(abs(bb - aa)) / abs(bb - aa)
            assert mp.mpf(str(mean.lo)) <= exact <= mp.mpf(str(mean.hi)), branch

    coarse = continuous.geometric_nodes((2, 1), 24)
    fine = refined_nodes(coarse, 4)
    beta_cases, timings = [], []
    for positive, negative in ((0, 0), (1, 0), (1, 1), (3, 0), (3, 2), (8, 0), (8, 7), (20, 20), (32, 0)):
        blocks = pair_blocks(positive, negative)
        exact = beta_exact(positive, negative)
        expected = fraction_bounds(exact)
        start = time.perf_counter()
        original = continuous.certificate_from_blocks(blocks, coarse)
        refined = continuous.certificate_from_blocks(blocks, fine)
        seconds = time.perf_counter() - start
        timings.append(seconds)
        a, b = lower_value(original), lower_value(refined)
        assert 0 < a.lo <= a.hi < expected.lo
        assert 0 < b.lo <= b.hi < expected.lo
        assert b.lo > a.hi, "Deterministic refinement should improve these constructed cases"
        fraction = b / expected
        assert fraction.lo > Decimal("0.94")
        beta_cases.append({"positive": positive, "negative": negative,
                           "complete_mixture_exact": str(exact),
                           "coarse_lower_log": original["log_evidence_lower_bound"],
                           "refined_lower_log": refined["log_evidence_lower_bound"],
                           "refined_fraction_of_exact_bounds": [str(fraction.lo), str(fraction.hi)],
                           "seconds_for_both_certificates": seconds})
    for total in range(17):
        mean = sum((Fraction(math.comb(total, k), 2 ** total) * beta_exact(k, total - k)
                    for k in range(total + 1)), Fraction(0))
        assert mean == 1

    root6 = Bounds(6).sqrt()
    examples = [
        ("prior normalization", [], Bounds(1)),
        ("m1 one discordance", pair_blocks(1, 0, False), Bounds(1)),
        ("m1 two positive discordances", pair_blocks(2, 0, False), fraction_bounds(beta_exact(2, 0))),
        ("m1 mixed discordances", pair_blocks(3, 2, False), fraction_bounds(beta_exact(3, 2))),
        ("m2 positive extreme", [{"m": 2, "minus_successes": 0, "plus_successes": 2}], 2 * root6 - 3),
        ("m2 negative extreme", [{"m": 2, "minus_successes": 2, "plus_successes": 0}], 2 * root6 - 3),
        ("m2 central", [{"m": 2, "minus_successes": 1, "plus_successes": 1}], 3 - root6),
        ("two m2 central blocks", [{"m": 2, "minus_successes": 1, "plus_successes": 1}] * 2, 9 - Bounds(7) * root6 / 2),
        ("mixed block sizes", [{"m": 3, "minus_successes": 1, "plus_successes": 2},
                               {"m": 2, "minus_successes": 0, "plus_successes": 1}], None),
    ]
    enclosures = []
    for label, blocks, exact in examples:
        start = time.perf_counter()
        enclosed = full_mixture_rectangle_bounds(blocks, 2048)
        cert = continuous.certificate_from_blocks(blocks, fine)
        lower = lower_value(cert)
        assert lower.hi <= enclosed.hi
        if exact is not None:
            assert enclosed.lo <= exact.lo <= exact.hi <= enclosed.hi
            assert lower.hi < exact.lo
        enclosures.append({"case": label, "blocks": blocks,
                           "full_mixture_rectangle_bounds": [str(enclosed.lo), str(enclosed.hi)],
                           "analytic_bounds": [str(exact.lo), str(exact.hi)] if exact is not None else None,
                           "certified_lower_log": cert["log_evidence_lower_bound"],
                           "subdivisions": 2048, "elapsed_seconds": time.perf_counter() - start})

    decisions = []
    for positive, negative, delta, expected in ((10, 0, ".01", "reject"), (0, 10, ".01", "accept"),
                                                (10, 0, ".005", "abstain"), (5, 5, ".01", "abstain")):
        cert = continuous.certificate_from_blocks(pair_blocks(positive, negative), fine, delta)
        assert cert["decision"] == expected
        decisions.append({"positive": positive, "negative": negative, "delta": delta,
                          "decision": cert["decision"], "lower_log": cert["log_evidence_lower_bound"]})
    wrong_crossings = []
    for q in (Fraction(1, 4), Fraction(1, 2), Fraction(3, 5), Fraction(3, 4)):
        for delta in (Fraction(1, 10), Fraction(1, 100)):
            probability = finite_wrong_crossing_probability(q, delta, 24)
            bound = delta if q == Fraction(1, 2) else delta / (2 - delta)
            assert probability <= bound
            wrong_crossings.append({"q": str(q), "delta": str(delta), "horizon": 24,
                                    "probability_exact": str(probability), "bound_exact": str(bound),
                                    "event": "Ever wrong-sign complete m1 mixture crossing, including paths with earlier correct crossings"})

    invalid = 0
    for nodes in (((1, 1), (2, 2)), ((0, 1), (2, 1)), ((1, 1), (True, 2)), ((1, 1),)):
        try:
            continuous.certificate_from_blocks([], nodes)
        except ValueError:
            invalid += 1
        else:
            raise AssertionError("Invalid nodes accepted")
    for bad in (True, "0", "1", "NaN"):
        try:
            continuous.certificate_from_blocks([], delta=bad)
        except ValueError:
            invalid += 1
        else:
            raise AssertionError("Invalid risk accepted")
    for name, expected in sources.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    return {"status": "passed", "sources": sources,
            "scope": "New deterministic analytic and arithmetic checks only; no Monte Carlo, stored trial input, or empirical SOTA comparison.",
            "pi_decimal_bounds": [str(pi.lo), str(pi.hi)],
            "pi_method": "Machin identity and exact Fraction alternating-series bounds; 120-digit mpmath is corroboration only",
            "grids": {"coarse": [[q.numerator, q.denominator] for q in coarse],
                      "fine": [[q.numerator, q.denominator] for q in fine]},
            "m1_beta_cases": beta_cases, "m1_exact_central_normalizations": 17,
            "full_integral_enclosures": enclosures, "signed_decisions": decisions,
            "exact_finite_wrong_crossings": wrong_crossings,
            "exponential_chord_branches_checked": ["constant", "near_equal_positive_series", "positive_series", "large_difference_scaled"],
            "invalid_inputs_rejected": invalid,
            "sources_unchanged_during_checks": True,
            "limitations": ["Finite chord bounds are not equal to the complete continuous mixture.",
                            "Rectangles and analytic expressions independently check the integral; outward arithmetic and the Machin pi enclosure are shared primitives.",
                            "Finite-horizon exact risk enumeration supports but does not replace the infinite-horizon signed-supermartingale proof.",
                            "Uniform power is not established for fixed numerical grids or arbitrary adaptive vanishing gaps."],
            "elapsed_seconds": time.perf_counter() - began}


if __name__ == "__main__":
    if OUTPUT.exists():
        raise SystemExit("Preserve the existing checks artifact; choose a new version for future changes.")
    report = run_checks()
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({"status": report["status"], "output": str(OUTPUT),
                      "m1_cases": len(report["m1_beta_cases"]),
                      "independent_full_integral_enclosures": len(report["full_integral_enclosures"]),
                      "exact_risk_enumerations": len(report["exact_finite_wrong_crossings"]),
                      "elapsed_seconds": report["elapsed_seconds"]}, indent=2))
