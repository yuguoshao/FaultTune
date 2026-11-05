"""Rao-Blackwellized paired odds experts on balanced Bernoulli blocks.

For m observations per arm and margins Kplus,Kminus, the arbitrary pairing
has J double successes distributed as Hypergeom(m,Kplus,Kminus), regardless
of the two true probabilities. Averaging the paired expert over this
ancillary pairing gives

  R_omega = (2*omega/(1+omega))**Kplus * (2/(1+omega))**Kminus
            * E[((1+omega)**2/(4*omega))**J].

Equivalently log R = eta*score/2 + log H, with H=E[cosh(eta/2)**(-D)]<=1.
Every oriented expert preserves its ordered-null supermartingale bound.
On a wrong cumulative score all opposite-direction experts are <=1, so
any fixed rational mixture controls total wrong-direction risk delta at
1/delta, without splitting delta between directions.

If arm selection depends only on previous block margins, averaging the
whole paired mixture over pairings gives this mixture. Conditional Jensen
then improves expected log evidence at each fixed time for every true law.
This does not imply pointwise evidence or expected stopping-time dominance.
All decisions use outward Decimal recomputation; float values only screen.
"""
from decimal import Decimal
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
import numbers

import numpy as np
from scipy.special import logsumexp

import conditional_fisher_unequal as unequal
from expert_config import protocol_config as prior_config
from direct_tests import _Bounds, _float_down, _float_up


PROTOCOL_VERSION = "rao_blackwell_paired_grid_v1"


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(name+" must be an integer >= "+str(minimum))
    return int(value)


def _experts(config):
    result, seen = [], set()
    entries = config if isinstance(config, (list, tuple)) else config["experts"]
    for entry in entries:
        odds = Fraction(_integer(entry["numerator"], "odds numerator", 1),
                        _integer(entry["denominator"], "odds denominator", 1))
        if odds in seen:
            raise ValueError("duplicate odds expert")
        seen.add(odds)
        if len(entry["weight"]) != 2:
            raise ValueError("weights require a numerator and denominator")
        weight = Fraction(_integer(entry["weight"][0], "weight numerator", 1),
                          _integer(entry["weight"][1], "weight denominator", 1))
        try:
            finite_odds = math.isfinite(float(odds)) and math.isfinite(float(1/odds))
        except OverflowError:
            finite_odds = False
        if not finite_odds:
            raise ValueError("odds are outside the floating screening range")
        result.append((odds, weight))
    if not result or sum((weight for odds, weight in result), Fraction(0)) != 1:
        raise ValueError("positive rational weights must sum exactly to one")
    return result


def protocol_config(config=None):
    experts = _experts(prior_config() if config is None else config)
    return {"version": PROTOCOL_VERSION,
            "experts": [{"numerator": odds.numerator, "denominator": odds.denominator,
                         "weight": [weight.numerator, weight.denominator]} for odds, weight in experts],
            "expert_count": len(experts), "score": "sum(plus_successes-minus_successes)",
            "conditioning": "Average arbitrary pairing given both block margins; J is central hypergeometric",
            "normalizer": "Frozen unequal conditional Fisher centered CGF at rational kappa=(p+q)^2/(4pq)",
            "threshold": "1/delta with cumulative score sign; no directional split",
            "assumptions": "Predictable balanced independent Bernoulli blocks with common ordering",
            "log_efficiency_claim": "Fixed-time expected log evidence dominates matched paired mixture when controls use only past margins",
            "limits": "No pointwise or stopping-time dominance; finite grid may lack positive growth at weak alternatives"}


def config_digest(config=None):
    return hashlib.sha256(json.dumps(protocol_config(config), sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def _layout(config):
    experts = _experts(config)
    poles = sorted({max(odds, 1/odds) for odds, weight in experts if odds != 1})
    lookup = {odds: i for i, odds in enumerate(poles)}
    layout = [(None, 0) if odds == 1 else (lookup[max(odds, 1/odds)], 1 if odds > 1 else -1)
              for odds, weight in experts]
    return experts, poles, layout


def _block(m, minus_successes, plus_successes):
    m = _integer(m, "m")
    minus = _integer(minus_successes, "minus_successes")
    plus = _integer(plus_successes, "plus_successes")
    if minus > m or plus > m:
        raise ValueError("success count exceeds block size")
    return {"m": m, "minus_successes": minus, "plus_successes": plus}


def _log_ratio(p, q):
    return math.log1p((p-q)/q) if 2*abs(p-q) < q else math.log(p)-math.log(q)


def _kappa(p, q):
    return Fraction((p+q)*(p+q), 4*p*q)


@lru_cache(maxsize=65536)
def _log_h_float(m, minus, plus, p, q):
    if not m or p == q:
        return 0.
    kappa = _kappa(p, q)
    cgf = unequal.centered_cgf_float(m-plus, plus, minus, kappa.numerator, kappa.denominator)
    mean_discordances = (m*(plus+minus)-2*plus*minus)/m
    value = cgf-.5*mean_discordances*_log_ratio(kappa.numerator, kappa.denominator)
    # H<=1 is an exact analytical bound. Float clipping only affects screening.
    return min(0., value)


@lru_cache(maxsize=16384)
def _log_h_bounds(m, minus, plus, p, q):
    if not m or p == q:
        return _Bounds(0), {"retained_terms": 1, "support_terms": 1, "deterministic": True}
    kappa = _kappa(p, q)
    cgf, detail = unequal.centered_cgf_bounds(m-plus, plus, minus, kappa.numerator, kappa.denominator)
    mean_discordances = _Bounds(m*(plus+minus)-2*plus*minus)/m
    log_kappa = (_Bounds(kappa.numerator)/kappa.denominator).log()
    value = cgf-mean_discordances*log_kappa/2
    if value.lo > 0:
        raise ArithmeticError("RB interval contradicts H<=1")
    return _Bounds(value.lo, min(Decimal(0), value.hi)), detail


def block_log_factor_bounds(m, minus_successes, plus_successes, numerator, denominator):
    block = _block(m, minus_successes, plus_successes)
    p, q = _integer(numerator, "odds numerator", 1), _integer(denominator, "odds denominator", 1)
    pole = Fraction(max(p, q), min(p, q))
    h, detail = _log_h_bounds(block["m"], block["minus_successes"], block["plus_successes"],
                            pole.numerator, pole.denominator)
    linear = (_Bounds(block["plus_successes"]-block["minus_successes"])/2)*(_Bounds(p)/q).log()
    return linear+h, detail


def empty_state(config=None):
    config = protocol_config(config)
    experts, poles, layout = _layout(config)
    return {"config": config, "score": 0, "shots": 0, "blocks": [],
            "log_h": np.zeros(len(poles)), "_poles": poles, "_layout": layout,
            "_eta": np.array([_log_ratio(odds.numerator, odds.denominator) for odds in poles]),
            "_log_weights": np.array([_log_ratio(weight.numerator, weight.denominator)
                                       for odds, weight in experts])}


def update_block(state, m, minus_successes, plus_successes):
    block = _block(m, minus_successes, plus_successes)
    for i, pole in enumerate(state["_poles"]):
        state["log_h"][i] += _log_h_float(block["m"], block["minus_successes"], block["plus_successes"],
                                        pole.numerator, pole.denominator)
    state["score"] += block["plus_successes"]-block["minus_successes"]
    state["shots"] += 2*block["m"]
    state["blocks"].append(block)
    return state


def screen(state, delta=.01):
    if isinstance(delta, bool) or not math.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("delta must lie in (0,1)")
    values = np.array([0. if sign == 0 else
                       sign*.5*state["score"]*state["_eta"][pole]+state["log_h"][pole]
                       for pole, sign in state["_layout"]])
    value = float(logsumexp(values+state["_log_weights"]))
    threshold = math.log(1/delta)
    crossing = value >= threshold-1e-6 and state["score"] != 0
    direction = "reject" if state["score"] > 0 else "accept" if state["score"] < 0 else "abstain"
    return {"log_evidence": value, "score": state["score"], "shots": state["shots"],
            "threshold": threshold, "crossing_candidate": crossing,
            "decision_candidate": direction if crossing else "abstain", "certified": False}


def certificate_from_blocks(blocks, delta=.01, config=None):
    if isinstance(delta, bool):
        raise ValueError("delta must be a probability")
    risk = Decimal(str(delta))
    if not risk.is_finite() or not 0 < risk < 1:
        raise ValueError("delta must lie in (0,1)")
    config = protocol_config(config)
    experts, poles, layout = _layout(config)
    blocks = [_block(b["m"], b["minus_successes"], b["plus_successes"]) for b in blocks]
    score = sum(b["plus_successes"]-b["minus_successes"] for b in blocks)
    hs, diagnostics = [_Bounds(0) for unused in poles], []
    for index, block in enumerate(blocks):
        retained = []
        for j, pole in enumerate(poles):
            value, detail = _log_h_bounds(block["m"], block["minus_successes"], block["plus_successes"],
                                         pole.numerator, pole.denominator)
            hs[j] += value
            retained.append(detail["retained_terms"])
        diagnostics.append({"block": index, "m": block["m"], "retained_terms_per_odds_pole": retained})
    logs = []
    for (odds, weight), (pole, sign) in zip(experts, layout):
        value = _Bounds(0) if sign == 0 else (_Bounds(score)/2)*(_Bounds(odds.numerator)/odds.denominator).log()+hs[pole]
        logs.append(value+(_Bounds(weight.numerator)/weight.denominator).log())
    offset = max(value.hi for value in logs)
    total = sum(((value-_Bounds(offset)).exp() for value in logs), _Bounds(0))
    evidence = _Bounds(offset)+total.log()
    threshold = -_Bounds(risk).log()
    crossed = score != 0 and evidence.lo >= threshold.hi
    decision = ("reject" if score > 0 else "accept") if crossed else "abstain"
    return {"method": PROTOCOL_VERSION, "decision": decision, "score": score,
            "shots": 2*sum(b["m"] for b in blocks), "blocks": len(blocks), "delta": float(risk),
            "evidence": _float_down(evidence.lo), "threshold": _float_up(threshold.hi),
            "evidence_bounds": [str(evidence.lo), str(evidence.hi)],
            "threshold_bounds": [str(threshold.lo), str(threshold.hi)], "anytime_valid": True,
            "numeric_status": "outward Decimal hypergeometric CGF, exact rational odds and weights",
            "config_digest": config_digest(config), "diagnostics": diagnostics,
            "assumptions": config["assumptions"]}
