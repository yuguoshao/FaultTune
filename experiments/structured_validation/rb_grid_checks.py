"""Exact finite-state and numerical checks of paired Rao-Blackwell evidence."""
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import time

import conditional_fisher as cf
from expert_config import protocol_config as prior_config
import rb_grid as rb
import weighted_fisher as weighted


ROOT = Path(__file__).resolve().parents[2]


def decimal_log(value):
    with localcontext() as context:
        context.prec = 110
        return Decimal(value.numerator).ln()-Decimal(value.denominator).ln()


def pairing_distribution(m, minus, plus):
    if not m:
        return [(0, Fraction(1))]
    return [(both, Fraction(math.comb(minus, both)*math.comb(m-minus, plus-both), math.comb(m, plus)))
            for both in range(max(0, minus+plus-m), min(minus, plus)+1)]


def exact_factor(m, minus, plus, odds):
    return sum((probability*(2*odds/(1+odds))**(plus-both)*(2/(1+odds))**(minus-both)
                for both, probability in pairing_distribution(m, minus, plus)), Fraction(0))


def exact_mixture(blocks, config):
    total = Fraction(0)
    for entry in config["experts"]:
        odds, weight = Fraction(entry["numerator"], entry["denominator"]), Fraction(*entry["weight"])
        value = Fraction(1)
        for block in blocks:
            value *= exact_factor(block["m"], block["minus_successes"], block["plus_successes"], odds)
        total += weight*value
    return total


def jensen_check(blocks, config):
    choices = [pairing_distribution(b["m"], b["minus_successes"], b["plus_successes"]) for b in blocks]
    mean, log_mean = Fraction(0), Decimal(0)
    with localcontext() as context:
        context.prec = 100
        for outcome in product(*choices):
            probability, value = Fraction(1), Fraction(0)
            for both, mass in outcome:
                probability *= mass
            for expert in config["experts"]:
                odds, capital = Fraction(expert["numerator"], expert["denominator"]), Fraction(1)
                for block, (both, unused_mass) in zip(blocks, outcome):
                    capital *= ((2*odds/(1+odds))**(block["plus_successes"]-both)
                                *(2/(1+odds))**(block["minus_successes"]-both))
                value += Fraction(*expert["weight"])*capital
            mean += probability*value
            log_mean += Decimal(probability.numerator)/Decimal(probability.denominator)*decimal_log(value)
        rb_value = exact_mixture(blocks, config)
        assert mean == rb_value
        gain = decimal_log(rb_value)-log_mean
        assert gain >= 0
    return {"conditional_expected_pair_equals_rb_exactly": True,
            "conditional_expected_log_gain": float(gain), "pairing_combinations": math.prod(map(len, choices))}


def run():
    started = time.perf_counter()
    factors, maximum_float_error = 0, 0.
    selected = [Fraction(1), Fraction(1, 3), Fraction(3, 2), Fraction(5),
                Fraction(1000100005, 1000000000), Fraction(492475503859050, 1000000000)]
    for m in range(7):
        for minus in range(m+1):
            for plus in range(m+1):
                distribution = pairing_distribution(m, minus, plus)
                assert sum((mass for both, mass in distribution), Fraction(0)) == 1
                for odds in selected:
                    exact = exact_factor(m, minus, plus, odds)
                    reference = decimal_log(exact)
                    bound, unused = rb.block_log_factor_bounds(m, minus, plus, odds.numerator, odds.denominator)
                    assert bound.lo <= reference <= bound.hi, (m, minus, plus, odds, bound.lo, reference, bound.hi)
                    pole = max(odds, 1/odds)
                    fast = .5*(plus-minus)*rb._log_ratio(odds.numerator, odds.denominator)+rb._log_h_float(
                        m, minus, plus, pole.numerator, pole.denominator)
                    maximum_float_error = max(maximum_float_error, abs(fast-float(reference)))
                    assert abs(fast-float(reference)) < 2e-10
                    if (odds >= 1 and plus <= minus) or (odds <= 1 and plus >= minus):
                        assert exact <= 1
                    factors += 1

    expectations = 0
    for m in range(1, 7):
        for pminus, pplus in ((Fraction(1, 5), Fraction(1, 5)), (Fraction(1, 2), Fraction(1, 2)),
                             (Fraction(4, 5), Fraction(1, 5)), (Fraction(1, 2), Fraction(1, 3))):
            for odds in (Fraction(3, 2), Fraction(5)):
                actual = Fraction(0)
                for minus in range(m+1):
                    for plus in range(m+1):
                        mass = (math.comb(m, minus)*pminus**minus*(1-pminus)**(m-minus)
                                *math.comb(m, plus)*pplus**plus*(1-pplus)**(m-plus))
                        actual += mass*exact_factor(m, minus, plus, odds)
                assert actual == (1+(pplus-pminus)*(odds-1)/(odds+1))**m
                assert actual <= 1
                expectations += 1

    configs = [prior_config(), cf.protocol_config(),
               {"experts": [{"numerator": 1, "denominator": 1, "weight": [1, 5]},
                            {"numerator": 2, "denominator": 1, "weight": [1, 2]},
                            {"numerator": 1, "denominator": 3, "weight": [3, 10]}]}]
    history = [dict(m=3, minus_successes=1, plus_successes=2), dict(m=4, minus_successes=2, plus_successes=3)]
    mixture_results, jensen = [], []
    for config in configs:
        state = rb.empty_state(config)
        for block in history:
            rb.update_block(state, block["m"], block["minus_successes"], block["plus_successes"])
        cert = rb.certificate_from_blocks(history, config=config)
        exact = decimal_log(exact_mixture(history, config))
        assert Decimal(cert["evidence_bounds"][0]) <= exact <= Decimal(cert["evidence_bounds"][1])
        assert cert["shots"] == 14 and cert["score"] == 2
        assert abs(rb.screen(state)["log_evidence"]-float(exact)) < 2e-10
        assert rb.protocol_config(config) == rb.protocol_config(config["experts"])
        jensen.append(jensen_check(history, config))
        mixture_results.append({"expert_count": len(config["experts"]), "exact_log_evidence": str(exact),
                                "evidence_bounds": cert["evidence_bounds"]})

    # Every m=1 block is already completely determined by its margins:
    # Rao-Blackwellization exactly preserves the matching pair prior.
    m1 = [dict(m=1, minus_successes=km, plus_successes=kp)
          for km, kp in ((0, 0), (0, 1), (0, 1), (1, 0), (1, 1))]
    m1_checks = []
    for config in configs[:2]:
        cert = rb.certificate_from_blocks(m1, config=config)
        pair = weighted.pair_grid_bounds([1, 2, 1, 1], config["experts"])
        exact = decimal_log(exact_mixture(m1, config))
        assert Decimal(cert["evidence_bounds"][0]) <= exact <= Decimal(cert["evidence_bounds"][1])
        assert pair.lo <= exact <= pair.hi
        m1_checks.append({"expert_count": len(config["experts"]), "exact_log_evidence": str(exact)})

    # Balanced Fisher and conditional averaging over arbitrary pairings differ.
    odds = Fraction(4)
    rb_counterexample = exact_factor(2, 1, 1, odds)
    fisher_counterexample = odds*math.comb(4, 2)/sum(
        (Fraction(math.comb(2, k)*math.comb(2, 2-k))*odds**k for k in range(3)), Fraction(0))
    assert rb_counterexample == Fraction(41, 50) and fisher_counterexample == Fraction(8, 11)

    rejected = 0
    invalid = [lambda: rb.empty_state({"experts": []}),
               lambda: rb.empty_state({"experts": [{"numerator": 1, "denominator": 0, "weight": [1, 1]}]}),
               lambda: rb.empty_state({"experts": [{"numerator": 1, "denominator": 1, "weight": [1, 2]}]}),
               lambda: rb.update_block(rb.empty_state(), 2, 3, 0),
               lambda: rb.update_block(rb.empty_state(), True, 0, 0),
               lambda: rb.certificate_from_blocks([], delta=True),
               lambda: rb.certificate_from_blocks([], delta=0)]
    for call in invalid:
        try:
            call()
        except ValueError:
            rejected += 1
        else:
            raise AssertionError("invalid RB input accepted")
    zero = rb.certificate_from_blocks([])
    assert zero["decision"] == "abstain" and zero["shots"] == 0

    stress = [dict(m=m, minus_successes=km, plus_successes=kp) for m, km, kp in
              ((10000, 3100, 3900), (17000, 2, 11), (22000, 21970, 21993), (100000, 49000, 51000))]
    stress_start = time.perf_counter()
    state = rb.empty_state()
    for block in stress:
        rb.update_block(state, block["m"], block["minus_successes"], block["plus_successes"])
    float_time = time.perf_counter()-stress_start
    stress_cert = rb.certificate_from_blocks(stress)
    stress_error = max(float(stress_cert["evidence_bounds"][0])-rb.screen(state)["log_evidence"],
                       rb.screen(state)["log_evidence"]-float(stress_cert["evidence_bounds"][1]), 0.)
    assert stress_error < 1e-6
    reflected = [dict(b, minus_successes=b["plus_successes"], plus_successes=b["minus_successes"]) for b in stress]
    other = rb.certificate_from_blocks(reflected)
    assert other["score"] == -stress_cert["score"]
    assert max(Decimal(other["evidence_bounds"][0]), Decimal(stress_cert["evidence_bounds"][0])) <= min(
        Decimal(other["evidence_bounds"][1]), Decimal(stress_cert["evidence_bounds"][1]))
    result = {"status": "passed", "scope": "Exact small-state and numerical arithmetic checks; no stopping-cost or SOTA claim",
              "exact_block_factors_enclosed": factors, "maximum_small_float_error": maximum_float_error,
              "exact_ordered_null_mean_checks": expectations, "exact_mixtures": mixture_results,
              "conditional_jensen_checks": jensen, "matched_m1_pair_prior_checks": m1_checks,
              "fisher_is_not_rb_counterexample": {"m": 2, "Kminus": 1, "Kplus": 1, "odds": 4,
                                                  "rb": str(rb_counterexample), "fisher": str(fisher_counterexample)},
              "invalid_inputs_rejected": rejected, "default_prior": rb.protocol_config(),
              "stress": {"blocks": stress, "float_update_seconds": float_time,
                         "elapsed_seconds": time.perf_counter()-stress_start,
                         "float_outside_certified_interval": stress_error,
                         "certificate": stress_cert, "reflection_enclosures_overlap": True},
              "source_sha256": {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                                for name in ("rb_grid.py", "rb_grid_checks.py", "conditional_fisher_unequal.py",
                                             "expert_config.py", "weighted_fisher.py")},
              "elapsed_seconds": time.perf_counter()-started}
    path = ROOT/"results/structured_validation/rb_grid_checks.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ("default_prior", "stress")}, indent=2))
    print(json.dumps({"stress_seconds": result["stress"]["elapsed_seconds"], "stress_float_error": stress_error}))
    return result


if __name__ == "__main__":
    run()
