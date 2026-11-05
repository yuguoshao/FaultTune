"""Exact unequal conditional-normalizer and signed-mixture regression checks."""
import hashlib
import json
import math
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
import time

import conditional_fisher as balanced
import conditional_fisher_unequal as unequal


def partition(nminus, nplus, total, odds):
    return sum((Fraction(math.comb(nplus, k)*math.comb(nminus, total-k))*odds**k
                for k in range(max(0, total-nminus), min(nplus, total)+1)), Fraction(0))


def exact_cgf(nminus, nplus, total, odds):
    if not nminus or not nplus or total in (0, nminus+nplus):
        return Decimal(0)
    z = partition(nminus, nplus, total, odds)
    with localcontext() as context:
        context.prec = 110
        return (Decimal(z.numerator).ln()-Decimal(z.denominator).ln()
                -Decimal(math.comb(nminus+nplus, total)).ln()
                -(Decimal(total*nplus)/Decimal(nminus+nplus))
                *(Decimal(odds.numerator).ln()-Decimal(odds.denominator).ln()))


def exact_mixture(blocks):
    result = Fraction(0)
    for expert in unequal.protocol_config()["experts"]:
        odds = Fraction(expert["numerator"], expert["denominator"])
        value = Fraction(1)
        for block in blocks:
            nm, np = block["nminus"], block["nplus"]
            total, k = block["minus_successes"]+block["plus_successes"], block["plus_successes"]
            value *= odds**k*math.comb(nm+np, total)/partition(nm, np, total, odds)
        result += value/33
    with localcontext() as context:
        context.prec = 110
        return Decimal(result.numerator).ln()-Decimal(result.denominator).ln()


def run():
    started = time.perf_counter()
    enclosure_count, max_float_error = 0, 0.
    selected = [unequal.SIGNED_ODDS[j] for j in (0, 7, 15, 16, 31)]
    for nminus in range(7):
        for nplus in range(7):
            for total in range(nminus+nplus+1):
                for p, q in selected:
                    exact = exact_cgf(nminus, nplus, total, Fraction(p, q))
                    bound, detail = unequal.centered_cgf_bounds(nminus, nplus, total, p, q)
                    assert bound.lo <= exact <= bound.hi, (nminus, nplus, total, p, q, bound.lo, exact, bound.hi)
                    fast = unequal.centered_cgf_float(nminus, nplus, total, p, q)
                    max_float_error = max(max_float_error, abs(fast-float(exact)))
                    enclosure_count += 1
    # Unequal centered CGFs are not generally even. Both directions must
    # therefore be computed, despite the reciprocal expert grid.
    assert abs(exact_cgf(3, 1, 1, Fraction(2))-exact_cgf(3, 1, 1, Fraction(1, 2))) > Decimal('.01')
    expectations = 0
    for nminus in range(1, 5):
        for nplus in range(1, 5):
            for total in range(nminus+nplus+1):
                for true_odds in (Fraction(1), Fraction(3, 2), Fraction(4)):
                    z_true = partition(nminus, nplus, total, true_odds)
                    for expert in (Fraction(1, 3), Fraction(2, 3), Fraction(999, 1000)):
                        z_expert = partition(nminus, nplus, total, expert)
                        value = sum((Fraction(math.comb(nplus, k)*math.comb(nminus, total-k))
                                     *true_odds**k/z_true*expert**k*math.comb(nminus+nplus, total)/z_expert
                                     for k in range(max(0, total-nminus), min(nplus, total)+1)), Fraction(0))
                        assert value <= 1
                        expectations += 1
    blocks = [{"nminus": 3, "nplus": 7, "minus_successes": 1, "plus_successes": 5},
              {"nminus": 11, "nplus": 4, "minus_successes": 2, "plus_successes": 3},
              {"nminus": 0, "nplus": 5, "minus_successes": 0, "plus_successes": 2}]
    exact = exact_mixture(blocks)
    cert = unequal.certificate_from_blocks(blocks)
    assert Decimal(cert["evidence_bounds"][0]) <= exact <= Decimal(cert["evidence_bounds"][1])
    score = sum((unequal._score(b) for b in blocks), Fraction(0))
    assert Fraction(cert["score_numerator"], cert["score_denominator"]) == score
    swapped = [{"nminus": b["nplus"], "nplus": b["nminus"],
                "minus_successes": b["plus_successes"], "plus_successes": b["minus_successes"]} for b in blocks]
    complement = [{"nminus": b["nminus"], "nplus": b["nplus"],
                   "minus_successes": b["nminus"]-b["minus_successes"],
                   "plus_successes": b["nplus"]-b["plus_successes"]} for b in blocks]
    for transformed in (swapped, complement):
        result = unequal.certificate_from_blocks(transformed)
        assert Decimal(result["evidence_bounds"][0]) <= exact <= Decimal(result["evidence_bounds"][1])
        assert Fraction(result["score_numerator"], result["score_denominator"]) == -score
    equal = [{"m": 15, "minus_successes": 2, "plus_successes": 7},
             {"m": 21, "minus_successes": 10, "plus_successes": 15}]
    balanced_cert = balanced.certificate_from_blocks(equal)
    adapted = [{"nminus": b["m"], "nplus": b["m"], "minus_successes": b["minus_successes"],
                "plus_successes": b["plus_successes"]} for b in equal]
    equal_cert = unequal.certificate_from_blocks(adapted)
    exact_equal = exact_mixture(adapted)
    for result in (balanced_cert, equal_cert):
        assert Decimal(result["evidence_bounds"][0]) <= exact_equal <= Decimal(result["evidence_bounds"][1])
    zero_blocks = [{"nminus": 1, "nplus": 2, "minus_successes": 0, "plus_successes": 1},
                   {"nminus": 2, "nplus": 1, "minus_successes": 1, "plus_successes": 0}]
    zero_cert = unequal.certificate_from_blocks(zero_blocks)
    assert zero_cert["score_numerator"] == 0 and zero_cert["decision"] == "abstain"
    # A negative score too small for subtracting rounded per-block floats
    # remains negative under exact Fraction accumulation.
    huge = 10**18
    tiny = [dict(nminus=huge-1, nplus=1, minus_successes=0, plus_successes=1),
            dict(nminus=1, nplus=huge, minus_successes=1, plus_successes=0)]
    exact_tiny = sum((unequal._score(b) for b in tiny), Fraction(0))
    assert exact_tiny == -Fraction(1, huge*(huge+1))
    assert sum(float(unequal._score(b)) for b in tiny) == 0.
    degenerate = unequal.certificate_from_blocks([dict(nminus=0, nplus=7, minus_successes=0, plus_successes=3)])
    assert degenerate["score_numerator"] == 0 and degenerate["decision"] == "abstain" and degenerate["shots"] == 7
    # 64 deterministic blocks using public 1.125 checkpoint spacing and a
    # 38% plus allocation; all outcomes are fixed inputs, not sampled data.
    history, budget, previous = [], 32, 0
    for unused in range(64):
        count = budget-previous
        nplus = max(1, int(.38*count))
        nminus = count-nplus
        history.append(dict(nminus=nminus, nplus=nplus, minus_successes=int(.29*nminus), plus_successes=int(.31*nplus)))
        previous, budget = budget, 2*int(math.ceil(1.125*budget/2))
    state = unequal.empty_state()
    float_start = time.perf_counter()
    for b in history:
        unequal.update_block(state, **b)
    float_seconds = time.perf_counter()-float_start
    cert_start = time.perf_counter()
    history_cert = unequal.certificate_from_blocks(history)
    certificate_seconds = time.perf_counter()-cert_start
    result = {"status": "passed", "protocol": unequal.protocol_config(),
              "exact_normalizer_enclosures": enclosure_count,
              "maximum_small_support_float_error": max_float_error,
              "exact_conditional_mlr_expectations": expectations,
              "cgf_not_even_regression": True, "whole_mixture_exact_fraction_enclosure": True,
              "arm_swap_and_complement": True, "balanced_module_exact_agreement": True,
              "fraction_cancellation_sign": str(exact_tiny), "single_arm_zero_information": True,
              "zero_score_abstains": True,
              "history64": {"shots": state["shots"], "float_update_seconds": float_seconds,
                            "certificate_seconds": certificate_seconds,
                            "float_log_evidence": unequal.screen(state)["log_evidence"],
                            "certified_log_evidence_bounds": history_cert["evidence_bounds"]},
              "theory": "Unequal central CGFs need not be even but are nonnegative by Jensen. Fixed signed experts opposing the common true ordering are conditional supermartingales by MLR; experts agreeing with that ordering are at most one on a wrong cumulative exact-score event. Replacing the latter by constant one gives a dominating mean-one supermartingale and a single 1/delta Ville bound.",
              "monte_carlo": False, "elapsed_seconds": time.perf_counter()-started}
    sources = [Path(__file__).resolve(), Path(unequal.__file__).resolve(), Path(balanced.__file__).resolve()]
    result["sources"] = {str(p.relative_to(unequal.ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    folder = unequal.ROOT/"results"/"structured_validation"
    folder.mkdir(parents=True, exist_ok=True)
    (folder/"conditional_fisher_unequal_checks.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "protocol"}, indent=2))


if __name__ == "__main__":
    run()
