"""Exact small-support, common-order and performance checks; no Monte Carlo."""
import hashlib
import json
import math
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
import time

from conditional_fisher import (ROOT, POSITIVE_ODDS_NUMERATORS, ODDS_DENOMINATOR,
                               protocol_config, centered_cgf_bounds, centered_cgf_float,
                               empty_state, update_block, screen, certificate_from_blocks)


def partition(m, total, odds):
    return sum((Fraction(math.comb(m, k)*math.comb(m, total-k))*odds**k
                for k in range(max(0, total-m), min(m, total)+1)), Fraction(0))


def exact_cgf(m, total, odds):
    z = partition(m, total, odds)
    with localcontext() as context:
        context.prec = 110
        return (Decimal(z.numerator).ln()-Decimal(z.denominator).ln()
                -Decimal(math.comb(2*m, total)).ln()
                -(Decimal(total)/2)*(Decimal(odds.numerator).ln()-Decimal(odds.denominator).ln()))


def run():
    started = time.perf_counter()
    enclosure_checks, float_error = 0, 0.
    selected = (POSITIVE_ODDS_NUMERATORS[0], POSITIVE_ODDS_NUMERATORS[7], POSITIVE_ODDS_NUMERATORS[-1])
    for m in range(1, 10):
        for total in range(2*m+1):
            for numerator in selected:
                odds = Fraction(numerator, ODDS_DENOMINATOR)
                bound, detail = centered_cgf_bounds(m, total, numerator)
                exact = exact_cgf(m, total, odds)
                # A zero CGF is known exactly; finite-precision independent
                # evaluation of cancelling logs can differ by 1e-108.
                if total in (0, 2*m):
                    exact = Decimal(0)
                assert bound.lo <= exact <= bound.hi, (m, total, numerator, bound.lo, exact, bound.hi)
                approximation = centered_cgf_float(m, total, numerator)
                float_error = max(float_error, abs(approximation-float(exact)))
                enclosure_checks += 1
    mlr_checks = 0
    for m in range(1, 7):
        for total in range(2*m+1):
            for true_odds in (Fraction(1), Fraction(3, 2), Fraction(4)):
                z_true = partition(m, total, true_odds)
                for expert in (Fraction(1, 3), Fraction(2, 3), Fraction(999, 1000)):
                    z_expert = partition(m, total, expert)
                    central_z = math.comb(2*m, total)
                    expectation = sum((Fraction(math.comb(m, k)*math.comb(m, total-k))
                                       *true_odds**k/z_true*expert**k*central_z/z_expert
                                       for k in range(max(0, total-m), min(m, total)+1)), Fraction(0))
                    assert expectation <= 1
                    mlr_checks += 1
    config = protocol_config()
    assert len(config["experts"]) == 33
    for positive, negative in zip(config["positive_odds"], config["negative_odds"]):
        assert Fraction(*positive)*Fraction(*negative) == 1
    # At m=1, conditional evidence equals a fixed-grid discordant-sign
    # mixture. Concordant observations must leave every expert unchanged.
    sequence = [(0, 0)]*3+[(0, 1)]*17+[(1, 0)]*4+[(1, 1)]*2
    state = empty_state()
    for minus, plus in sequence:
        update_block(state, 1, minus, plus)
    exact_mixture = Fraction(0)
    for expert in config["experts"]:
        odds = Fraction(expert["numerator"], expert["denominator"])
        exact_mixture += (2*odds/(1+odds))**17*(2/(1+odds))**4/33
    cert = certificate_from_blocks(state["blocks"])
    with localcontext() as context:
        context.prec = 110
        exact = Decimal(exact_mixture.numerator).ln()-Decimal(exact_mixture.denominator).ln()
        assert Decimal(cert["evidence_bounds"][0]) <= exact <= Decimal(cert["evidence_bounds"][1])
    zero = empty_state()
    update_block(zero, 100, 31, 31)
    assert screen(zero)["log_evidence"] <= 1e-10
    assert certificate_from_blocks(zero["blocks"])["decision"] == "abstain"
    # Complementing successes changes score sign but not evidence.
    original = [{"m": 15, "minus_successes": 2, "plus_successes": 7},
                {"m": 21, "minus_successes": 10, "plus_successes": 15}]
    complement = [{"m": b["m"], "minus_successes": b["m"]-b["minus_successes"],
                   "plus_successes": b["m"]-b["plus_successes"]} for b in original]
    forward, backward = certificate_from_blocks(original), certificate_from_blocks(complement)
    assert forward["evidence_bounds"] == backward["evidence_bounds"]
    fine, detail = centered_cgf_bounds(2**22, 2**23-2, selected[-1])
    assert detail["support_terms"] == detail["retained_terms"] == 3 and detail["complemented"]
    # Deterministic 64-block history using the same 1.125 checkpoint growth.
    history, budget, previous = [], 32, 0
    for unused in range(64):
        m = (budget-previous)//2
        history.append({"m": m, "minus_successes": int(.29*m), "plus_successes": int(.31*m)})
        previous, budget = budget, 2*int(math.ceil(1.125*budget/2))
    sample = empty_state()
    float_started = time.perf_counter()
    for block in history:
        update_block(sample, **block)
    float_seconds = time.perf_counter()-float_started
    cert_started = time.perf_counter()
    history_cert = certificate_from_blocks(history)
    cert_seconds = time.perf_counter()-cert_started
    large_started = time.perf_counter()
    large_cert = certificate_from_blocks([{"m": 128000, "minus_successes": 63000, "plus_successes": 65000}])
    large_seconds = time.perf_counter()-large_started
    result = {"status": "passed", "protocol": config, "exact_normalizer_enclosures": enclosure_checks,
              "maximum_small_support_float_error": float_error, "exact_conditional_mlr_expectations": mlr_checks,
              "m1_exact_matched_prior_identity": True, "zero_score_cannot_decide": True,
              "complement_evidence_identity": True, "large_m_tiny_complement_support": True,
              "history64": {"shots": sample["shots"], "float_update_seconds": float_seconds,
                            "full_certificate_seconds": cert_seconds,
                            "float_log_evidence": screen(sample)["log_evidence"],
                            "certified_log_evidence_bounds": history_cert["evidence_bounds"]},
              "large_block": {"shots": 256000, "certificate_seconds": large_seconds,
                              "retained_terms": large_cert["diagnostics"][0]["retained_terms_per_positive_odds"],
                              "support_terms": large_cert["diagnostics"][0]["support_terms"]},
              "validity_argument": [
                  "Conditional central K-T/2 is symmetric, so each centered CGF is even, convex and nonnegative.",
                  "Each fixed positive-logodds expert is at most one when the cumulative score is negative; the reverse statement holds for negative experts.",
                  "When all true block odds are at least one, every negative expert likelihood ratio to the central law has conditional expectation at most one by monotone likelihood ratio order, even conditional on each realized T.",
                  "On an erroneous negative-score crossing, the full mixture is bounded by a mean-one supermartingale formed by replacing positive experts with constant one. Apply Ville at 1/delta; the reverse case is symmetric and the tie null makes the whole mixture a martingale.",
                  "Balanced blocks may adapt their sizes and settings from the past. Each block must use a single pair of independent Bernoulli probabilities with the common ordering; do not merge blocks from different angles.",
                  "The recurrence ratios decrease away from the mode, so the omitted one-sided mass is at most next_term/(1-next_ratio); outward arithmetic encloses all retained mass and tails."],
              "limits": ["Fixed finite grid does not guarantee power for arbitrary nonzero gaps.",
                         "No finite-cost superiority or formal proof-assistant claim."],
              "monte_carlo": False, "elapsed_seconds": time.perf_counter()-started}
    sources = [Path(__file__).resolve(), Path(__file__).resolve().with_name("conditional_fisher.py")]
    result["sources"] = {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    folder = ROOT/"results"/"structured_validation"
    folder.mkdir(parents=True, exist_ok=True)
    (folder/"conditional_fisher_checks.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("protocol", "validity_argument")}, indent=2))


if __name__ == "__main__":
    run()
