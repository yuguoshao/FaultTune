"""Signed conditional Fisher mixture for unequal predictable arm counts.

This is a separate implementation; balanced conditional_fisher v1 is frozen.
Given T successes in nminus+nplus observations, K plus successes has weight
choose(nplus,K)*choose(nminus,T-K)*odds**K. Its central mean is exactly
T*nplus/(nminus+nplus). The cumulative centered score is a Fraction.

Central CGFs are not assumed even. All 32 signed noncentral experts use
their own normalizers. Jensen nonnegativity and conditional MLR order give
the same signed-mixture 1/delta guarantee under a common true ordering.
"""
import math
import numbers
from decimal import Decimal
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
from scipy.special import gammaln, logsumexp

from conditional_fisher import POSITIVE_ODDS_NUMERATORS, ODDS_DENOMINATOR

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "circuit_validation"))
from direct_tests import _Bounds, _log_factorial, _float_down, _float_up


PROTOCOL_VERSION = "conditional_fisher_unequal_fixed33_v1"
SIGNED_ODDS = (tuple((p, ODDS_DENOMINATOR) for p in POSITIVE_ODDS_NUMERATORS)
               +tuple((ODDS_DENOMINATOR, p) for p in POSITIVE_ODDS_NUMERATORS))
EXPERT_COUNT = 1+len(SIGNED_ODDS)
LOG_ODDS = np.array([math.log(p/q) for p, q in SIGNED_ODDS])
TAIL_TOLERANCE = Decimal("1e-24")
MAX_TAIL_STEPS = 200000


def protocol_config():
    return {"version": PROTOCOL_VERSION, "expert_count": EXPERT_COUNT,
            "experts": [{"numerator": p, "denominator": q, "weight": [1, EXPERT_COUNT]}
                        for p, q in ((1, 1),)+SIGNED_ODDS],
            "grid_source": "conditional_fisher_fixed33_v1 exact odds and equal weights",
            "score": "sum(K-T*nplus/(nminus+nplus)), accumulated as an exact Fraction",
            "threshold": "1/delta with exact score sign; no directional split",
            "tail_tolerance": str(TAIL_TOLERANCE), "max_tail_steps": MAX_TAIL_STEPS,
            "limits": "finite expert grid need not have positive growth at every nonzero gap"}


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError(name+" must be a nonnegative integer")
    return int(value)


def _block(nminus, nplus, minus_successes, plus_successes):
    values = [_count(value, name) for value, name in zip(
        (nminus, nplus, minus_successes, plus_successes),
        ("nminus", "nplus", "minus_successes", "plus_successes"))]
    if values[2] > values[0] or values[3] > values[1]:
        raise ValueError("success count exceeds its arm count")
    return dict(zip(("nminus", "nplus", "minus_successes", "plus_successes"), values))


def _canonical(nminus, nplus, total, numerator, denominator):
    """Reduce support while preserving the centered CGF exactly.

    Complement successes: K' = nplus-K, T'=N-T, eta'=-eta.
    Exchange arms: K'=T-K, nplus'=nminus, eta'=-eta.
    After both optional maps, support is 0..min(nplus,T).
    """
    complemented, exchanged = False, False
    if 2*total > nminus+nplus:
        total = nminus+nplus-total
        numerator, denominator = denominator, numerator
        complemented = True
    if nplus > nminus:
        nminus, nplus = nplus, nminus
        numerator, denominator = denominator, numerator
        exchanged = True
    return nminus, nplus, total, numerator, denominator, complemented, exchanged


def _ratio(nminus, nplus, total, k, numerator, denominator, direction):
    if direction == 1:
        return numerator*(nplus-k)*(total-k), denominator*(k+1)*(nminus-total+k+1)
    return denominator*k*(nminus-total+k), numerator*(nplus-k+1)*(total-k+1)


def _mode(nminus, nplus, total, numerator, denominator):
    lo, hi = 0, min(nplus, total)
    while lo < hi:
        k = (lo+hi)//2
        up, down = _ratio(nminus, nplus, total, k, numerator, denominator, 1)
        if up <= down:
            hi = k
        else:
            lo = k+1
    return lo


def _logchoose_float(n, k):
    return gammaln(n+1)-gammaln(k+1)-gammaln(n-k+1)


def _float_sum(nminus, nplus, total, mode, numerator, denominator):
    omega = numerator/denominator
    high = min(nplus, total)
    width = max(32, int(math.ceil(12*math.sqrt(high+1))))
    while True:
        left, right = max(0, mode-width), min(high, mode+width)
        vectors = {0: np.array([0.])}
        if right > mode:
            k = np.arange(mode, right, dtype=float)
            ratio = omega*((nplus-k)/(k+1))*((total-k)/(nminus-total+k+1))
            vectors[1] = np.cumsum(np.log(ratio))
        if left < mode:
            k = np.arange(mode, left, -1, dtype=float)
            ratio = ((k/(nplus-k+1))*((nminus-total+k)/(total-k+1)))/omega
            vectors[-1] = np.cumsum(np.log(ratio))
        value = float(sum(np.exp(v).sum() for v in vectors.values()))
        tail = 0.
        for end, direction in ((left, -1), (right, 1)):
            if end == (0 if direction == -1 else high):
                continue
            up, down = _ratio(nminus, nplus, total, end, numerator, denominator, direction)
            ratio = up/down
            if ratio >= 1:
                tail = math.inf
                break
            tail += math.exp(float(vectors[direction][-1]))*ratio/(1-ratio)
        if tail <= 1e-14*value or (left == 0 and right == high):
            return math.log(value)
        width *= 2


@lru_cache(maxsize=32768)
def centered_cgf_float(nminus, nplus, total, numerator, denominator):
    if not nminus or not nplus or total in (0, nminus+nplus):
        return 0.
    nminus, nplus, total, numerator, denominator, unused_c, unused_s = _canonical(
        nminus, nplus, total, numerator, denominator)
    mode = _mode(nminus, nplus, total, numerator, denominator)
    mean = total*nplus/(nminus+nplus)
    value = (_logchoose_float(nplus, mode)+_logchoose_float(nminus, total-mode)
             -_logchoose_float(nminus+nplus, total)+(mode-mean)*math.log(numerator/denominator)
             +_float_sum(nminus, nplus, total, mode, numerator, denominator))
    return max(0., float(value))


def _logchoose_bounds(n, k):
    return _log_factorial(n)-_log_factorial(k)-_log_factorial(n-k)


def _sum_bounds(nminus, nplus, total, mode, numerator, denominator):
    accumulated, omitted = _Bounds(1), _Bounds(0)
    retained, high = 1, min(nplus, total)
    tails = []
    for direction in (-1, 1):
        k, term, steps = mode, _Bounds(1), 0
        endpoint = 0 if direction == -1 else high
        tail = _Bounds(0)
        while k != endpoint:
            up, down = _ratio(nminus, nplus, total, k, numerator, denominator, direction)
            ratio = _Bounds(up)/down
            if ratio.hi < 1:
                tail = term*ratio/(1-ratio)
                target = (_Bounds(TAIL_TOLERANCE)*_Bounds(accumulated.lo)).lo
                if tail.hi <= target or steps >= MAX_TAIL_STEPS:
                    break
            term = term*ratio
            accumulated += term
            k += direction
            steps += 1
            retained += 1
            tail = _Bounds(0)
        omitted += _Bounds(0, tail.hi)
        tails.append({"direction": direction, "last_retained": k,
                      "tail_upper_decimal": str(tail.hi), "steps": steps})
    return _Bounds(accumulated.lo, (accumulated+omitted).hi), retained, tails


@lru_cache(maxsize=8192)
def centered_cgf_bounds(nminus, nplus, total, numerator, denominator):
    if not nminus or not nplus or total in (0, nminus+nplus):
        return _Bounds(0), {"retained_terms": 1, "support_terms": 1, "deterministic": True}
    nminus, nplus, total, numerator, denominator, complemented, exchanged = _canonical(
        nminus, nplus, total, numerator, denominator)
    mode = _mode(nminus, nplus, total, numerator, denominator)
    scaled, retained, tails = _sum_bounds(nminus, nplus, total, mode, numerator, denominator)
    mean = _Bounds(total*nplus)/(nminus+nplus)
    eta = (_Bounds(numerator)/denominator).log()
    value = (_logchoose_bounds(nplus, mode)+_logchoose_bounds(nminus, total-mode)
             -_logchoose_bounds(nminus+nplus, total)+(_Bounds(mode)-mean)*eta+scaled.log())
    if value.hi < 0:
        raise ArithmeticError("CGF interval contradicts Jensen nonnegativity")
    return _Bounds(max(Decimal(0), value.lo), value.hi), {
        "mode": mode, "retained_terms": retained, "support_terms": min(nplus, total)+1,
        "complemented": complemented, "exchanged_arms": exchanged, "tails": tails}


def empty_state():
    return {"score": Fraction(0), "cgf": np.zeros(len(SIGNED_ODDS)), "shots": 0, "blocks": []}


def _score(block):
    count = block["nminus"]+block["nplus"]
    if not count:
        return Fraction(0)
    total = block["minus_successes"]+block["plus_successes"]
    return Fraction(block["plus_successes"])-Fraction(total*block["nplus"], count)


def update_block(state, nminus, nplus, minus_successes, plus_successes):
    block = _block(nminus, nplus, minus_successes, plus_successes)
    total = block["minus_successes"]+block["plus_successes"]
    for j, (numerator, denominator) in enumerate(SIGNED_ODDS):
        state["cgf"][j] += centered_cgf_float(block["nminus"], block["nplus"], total, numerator, denominator)
    state["score"] += _score(block)
    state["shots"] += block["nminus"]+block["nplus"]
    state["blocks"].append(block)
    return state


def screen(state, delta=.01):
    if isinstance(delta, bool) or not math.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("delta must lie strictly between zero and one")
    experts = np.concatenate(([0.], float(state["score"])*LOG_ODDS-state["cgf"]))
    value = float(logsumexp(experts)-math.log(EXPERT_COUNT))
    threshold = math.log(1/delta)
    decision = "reject" if state["score"] > 0 else "accept" if state["score"] < 0 else "abstain"
    crossing = value >= threshold-1e-6 and decision != "abstain"
    return {"log_evidence": value, "score": float(state["score"]),
            "score_numerator": state["score"].numerator, "score_denominator": state["score"].denominator,
            "threshold": threshold, "decision_candidate": decision if crossing else "abstain",
            "crossing_candidate": crossing, "shots": state["shots"], "certified": False}


def certificate_from_blocks(blocks, delta=.01):
    risk = Decimal(str(delta))
    if not risk.is_finite() or not 0 < risk < 1:
        raise ValueError("delta must lie strictly between zero and one")
    blocks = [_block(b["nminus"], b["nplus"], b["minus_successes"], b["plus_successes"]) for b in blocks]
    score = sum((_score(b) for b in blocks), Fraction(0))
    cgfs = [_Bounds(0) for unused in SIGNED_ODDS]
    diagnostics = []
    for index, block in enumerate(blocks):
        total = block["minus_successes"]+block["plus_successes"]
        retained = []
        for j, (numerator, denominator) in enumerate(SIGNED_ODDS):
            value, detail = centered_cgf_bounds(block["nminus"], block["nplus"], total, numerator, denominator)
            cgfs[j] += value
            retained.append(detail["retained_terms"])
        diagnostics.append({"block": index, "nminus": block["nminus"], "nplus": block["nplus"],
                            "total_successes": total, "retained_terms_per_signed_odds": retained})
    exact_score = _Bounds(score.numerator)/score.denominator
    experts = [_Bounds(0)]
    for j, (numerator, denominator) in enumerate(SIGNED_ODDS):
        eta = (_Bounds(numerator)/denominator).log()
        experts.append(exact_score*eta-cgfs[j])
    offset = max(expert.hi for expert in experts)
    scaled = sum(((expert-_Bounds(offset)).exp() for expert in experts), _Bounds(0))
    evidence = _Bounds(offset)+(scaled/EXPERT_COUNT).log()
    threshold = -_Bounds(risk).log()
    crossed = evidence.lo >= threshold.hi and score != 0
    decision = ("reject" if score > 0 else "accept") if crossed else "abstain"
    return {"method": PROTOCOL_VERSION, "decision": decision, "score": float(score),
            "score_numerator": score.numerator, "score_denominator": score.denominator,
            "shots": sum(b["nminus"]+b["nplus"] for b in blocks), "blocks": len(blocks), "delta": float(risk),
            "evidence": _float_down(evidence.lo), "threshold": _float_up(threshold.hi),
            "evidence_bounds": [str(evidence.lo), str(evidence.hi)],
            "threshold_bounds": [str(threshold.lo), str(threshold.hi)], "anytime_valid": True,
            "numeric_status": "outward Decimal mode sums with geometric tails; exact Fraction score sign",
            "diagnostics": diagnostics,
            "assumptions": "predictable independent Bernoulli blocks, fixed per-arm setting within each block, common ordering"}
