"""Conditional Fisher expert mixture for predictable balanced Bernoulli blocks.

Each block has m independent observations on each side, at one common pair
of settings selected before the block. Given T total successes, K plus-side
successes has a Fisher noncentral hypergeometric law. The fixed, reciprocal
rational odds grid is never restarted or fitted to current outcomes.

The evidence is a mixture relative to central odds 1. Its sign comes from
the total plus-minus score. It controls wrong-direction crossings at 1/delta
under a common ordering, even if the true odds changes adaptively by block.
Numeric decisions use only outward Decimal certificates; float screening is
not a certificate. A finite grid need not have positive growth at every gap.
"""
import math
import numbers
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
from scipy.special import gammaln, logsumexp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "circuit_validation"))
from direct_tests import _Bounds, _log_factorial, _float_down, _float_up


PROTOCOL_VERSION = "conditional_fisher_fixed33_v1"
ODDS_DENOMINATOR = 1000000000
POSITIVE_ODDS_NUMERATORS = (
    1000100005, 1000200020, 1000400080, 1000800320,
    1001601281, 1003205125, 1006420524, 1012882271,
    1025930494, 1052533379, 1107826514, 1227279584,
    1506215179, 2268684164, 5146927836, 26490866150,
)
EXPERT_COUNT = 1+2*len(POSITIVE_ODDS_NUMERATORS)
LOG_ODDS = np.log(np.array(POSITIVE_ODDS_NUMERATORS, dtype=float)/ODDS_DENOMINATOR)
TAIL_TOLERANCE = Decimal("1e-24")
MAX_TAIL_STEPS = 200000


def protocol_config():
    positive = [[p, ODDS_DENOMINATOR] for p in POSITIVE_ODDS_NUMERATORS]
    all_odds = [[1, 1]]+positive+[[d, p] for p, d in positive]
    return {"version": PROTOCOL_VERSION, "positive_odds": positive,
            "negative_odds": [[d, p] for p, d in positive], "central_odds": [1, 1],
            "experts": [{"numerator": p, "denominator": d, "weight": [1, EXPERT_COUNT]}
                        for p, d in all_odds],
            "weight_each": [1, EXPERT_COUNT], "expert_count": EXPERT_COUNT,
            "tail_tolerance": str(TAIL_TOLERANCE), "max_tail_steps": MAX_TAIL_STEPS,
            "threshold": "1/delta with signed central mixture; no directional split",
            "blocks": "one predictable equal-size independent Bernoulli pair of settings per block",
            "limits": "fixed finite grid; weak alternatives may have nonpositive expert growth"}


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError(name+" must be a nonnegative integer")
    return int(value)


def _block(m, minus_successes, plus_successes):
    m = _count(m, "m")
    minus = _count(minus_successes, "minus_successes")
    plus = _count(plus_successes, "plus_successes")
    if minus > m or plus > m:
        raise ValueError("success count exceeds the block size per side")
    return {"m": m, "minus_successes": minus, "plus_successes": plus}


def _mode(m, total, numerator, denominator):
    """Leftmost mode of the reduced T<=m Fisher weights; exact comparisons."""
    lo, hi = 0, total
    while lo < hi:
        k = (lo+hi)//2
        up = numerator*(m-k)*(total-k)
        down = denominator*(k+1)*(m-total+k+1)
        if up <= down:
            hi = k
        else:
            lo = k+1
    return lo


def _ratio_integers(m, total, k, numerator, denominator, direction):
    if direction == 1:
        return numerator*(m-k)*(total-k), denominator*(k+1)*(m-total+k+1)
    return denominator*k*(m-total+k), numerator*(m-k+1)*(total-k+1)


def _logchoose_float(n, k):
    return gammaln(n+1)-gammaln(k+1)-gammaln(n-k+1)


def _float_scaled_sum(m, total, mode, numerator, denominator):
    """Vectorized mode-centered sum; geometric tails check the chosen window."""
    omega = numerator/denominator
    width = max(32, int(math.ceil(12*math.sqrt(total+1))))
    while True:
        left, right = max(0, mode-width), min(total, mode+width)
        logs = [np.array([0.])]
        if right > mode:
            k = np.arange(mode, right, dtype=float)
            ratios = omega*((m-k)/(k+1))*((total-k)/(m-total+k+1))
            logs.append(np.cumsum(np.log(ratios)))
        if left < mode:
            k = np.arange(mode, left, -1, dtype=float)
            ratios = ((k/(m-k+1))*((m-total+k)/(total-k+1)))/omega
            logs.append(np.cumsum(np.log(ratios)))
        combined = np.concatenate(logs)
        scale_sum = float(np.exp(combined).sum())
        tail = 0.
        for end, direction in ((left, -1), (right, 1)):
            if end == (0 if direction == -1 else total):
                continue
            a, b = _ratio_integers(m, total, end, numerator, denominator, direction)
            ratio = a/b
            if ratio >= 1:
                tail = math.inf
                break
            # The last member of this side's vector is its endpoint weight.
            index = 1 if direction == 1 and right > mode else len(logs)-1
            end_weight = math.exp(float(logs[index][-1]))
            tail += end_weight*ratio/(1-ratio)
        if tail <= 1e-14*scale_sum or (left == 0 and right == total):
            return math.log(scale_sum)
        width *= 2


@lru_cache(maxsize=32768)
def centered_cgf_float(m, total, numerator, denominator=ODDS_DENOMINATOR):
    """log E_central exp(log(odds)*(K-T/2)); even in log odds."""
    total = min(total, 2*m-total)
    if total == 0:
        return 0.
    mode = _mode(m, total, numerator, denominator)
    value = (_logchoose_float(m, mode)+_logchoose_float(m, total-mode)
             -_logchoose_float(2*m, total)+(mode-total/2)*math.log(numerator/denominator)
             +_float_scaled_sum(m, total, mode, numerator, denominator))
    return max(0., float(value))


def _logchoose_bounds(n, k):
    return _log_factorial(n)-_log_factorial(k)-_log_factorial(n-k)


def _scaled_sum_bounds(m, total, mode, numerator, denominator):
    accumulated = _Bounds(1)
    omitted = _Bounds(0)
    retained = 1
    sides = []
    for direction in (-1, 1):
        k, term, steps = mode, _Bounds(1), 0
        endpoint = 0 if direction == -1 else total
        tail = _Bounds(0)
        while k != endpoint:
            up, down = _ratio_integers(m, total, k, numerator, denominator, direction)
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
        sides.append({"direction": direction, "last_retained": k,
                      "tail_upper_decimal": str(tail.hi), "steps": steps})
    return _Bounds(accumulated.lo, (accumulated+omitted).hi), retained, sides


@lru_cache(maxsize=8192)
def centered_cgf_bounds(m, total, numerator, denominator=ODDS_DENOMINATOR):
    original_total = total
    total = min(total, 2*m-total)
    if total == 0:
        return _Bounds(0), {"mode": 0, "retained_terms": 1, "support_terms": 1,
                            "complemented": original_total != total, "tails": []}
    mode = _mode(m, total, numerator, denominator)
    scaled, retained, sides = _scaled_sum_bounds(m, total, mode, numerator, denominator)
    eta = (_Bounds(numerator)/denominator).log()
    value = (_logchoose_bounds(m, mode)+_logchoose_bounds(m, total-mode)
             -_logchoose_bounds(2*m, total)+(_Bounds(2*mode-total)/2)*eta+scaled.log())
    if value.hi < 0:
        raise ArithmeticError("central CGF interval contradicts Jensen nonnegativity")
    return _Bounds(max(Decimal(0), value.lo), value.hi), {
        "mode": mode, "retained_terms": retained, "support_terms": total+1,
        "complemented": original_total != total, "tails": sides}


def empty_state():
    return {"score": 0, "cgf": np.zeros(len(POSITIVE_ODDS_NUMERATORS)),
            "shots": 0, "blocks": []}


def update_block(state, m, minus_successes, plus_successes):
    block = _block(m, minus_successes, plus_successes)
    total = block["minus_successes"]+block["plus_successes"]
    for j, numerator in enumerate(POSITIVE_ODDS_NUMERATORS):
        state["cgf"][j] += centered_cgf_float(block["m"], total, numerator)
    state["score"] += block["plus_successes"]-block["minus_successes"]
    state["shots"] += 2*block["m"]
    state["blocks"].append(block)
    return state


def screen(state, delta=.01):
    if isinstance(delta, bool) or not math.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("delta must lie strictly between zero and one")
    linear = .5*state["score"]*LOG_ODDS
    experts = np.concatenate(([0.], linear-state["cgf"], -linear-state["cgf"]))
    value = float(logsumexp(experts)-math.log(EXPERT_COUNT))
    threshold = math.log(1/delta)
    decision = "reject" if state["score"] > 0 else "accept" if state["score"] < 0 else "abstain"
    crossing = value >= threshold-1e-6 and decision != "abstain"
    return {"log_evidence": value, "score": state["score"], "threshold": threshold,
            "decision_candidate": decision if crossing else "abstain", "crossing_candidate": crossing,
            "shots": state["shots"], "certified": False}


def certificate_from_blocks(blocks, delta=.01):
    risk = Decimal(str(delta))
    if not risk.is_finite() or not 0 < risk < 1:
        raise ValueError("delta must lie strictly between zero and one")
    blocks = [_block(b["m"], b["minus_successes"], b["plus_successes"]) for b in blocks]
    score = sum(b["plus_successes"]-b["minus_successes"] for b in blocks)
    cgfs = [_Bounds(0) for unused in POSITIVE_ODDS_NUMERATORS]
    diagnostics = []
    for block_index, block in enumerate(blocks):
        total = block["minus_successes"]+block["plus_successes"]
        counts = []
        for j, numerator in enumerate(POSITIVE_ODDS_NUMERATORS):
            value, detail = centered_cgf_bounds(block["m"], total, numerator)
            cgfs[j] += value
            counts.append(detail["retained_terms"])
        diagnostics.append({"block": block_index, "m": block["m"], "total_successes": total,
                            "retained_terms_per_positive_odds": counts,
                            "support_terms": min(total, 2*block["m"]-total)+1})
    experts = [_Bounds(0)]
    for j, numerator in enumerate(POSITIVE_ODDS_NUMERATORS):
        eta = (_Bounds(numerator)/ODDS_DENOMINATOR).log()
        linear = (_Bounds(score)/2)*eta
        experts.extend((linear-cgfs[j], -linear-cgfs[j]))
    offset = max(expert.hi for expert in experts)
    scaled = sum(((expert-_Bounds(offset)).exp() for expert in experts), _Bounds(0))
    evidence = _Bounds(offset)+(scaled/EXPERT_COUNT).log()
    threshold = -_Bounds(risk).log()
    crossed = evidence.lo >= threshold.hi and score != 0
    decision = ("reject" if score > 0 else "accept") if crossed else "abstain"
    return {"method": PROTOCOL_VERSION, "decision": decision, "score": score,
            "shots": 2*sum(b["m"] for b in blocks), "blocks": len(blocks), "delta": float(risk),
            "evidence": _float_down(evidence.lo), "threshold": _float_up(threshold.hi),
            "evidence_bounds": [str(evidence.lo), str(evidence.hi)],
            "threshold_bounds": [str(threshold.lo), str(threshold.hi)],
            "anytime_valid": True, "numeric_status": "outward Decimal mode sums plus geometric tail bounds",
            "diagnostics": diagnostics,
            "assumptions": "predictable balanced independent Bernoulli blocks with a common ordering; fixed expert grid"}
