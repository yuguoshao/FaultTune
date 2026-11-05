"""Generic fixed weighted conditional Fisher evidence, balanced or unequal."""
from decimal import Decimal
from fractions import Fraction
import math

import numpy as np
from scipy.special import logsumexp

from expert_config import expert_config, protocol_config as prior_protocol
from conditional_fisher import centered_cgf_float as balanced_float
from conditional_fisher import centered_cgf_bounds as balanced_bounds
from conditional_fisher_unequal import centered_cgf_float as unequal_float
from conditional_fisher_unequal import centered_cgf_bounds as unequal_bounds
from direct_tests import _Bounds, _float_down, _float_up


def protocol_config():
    result = prior_protocol()
    result["evidence"] = "Conditional Fisher signed mixture; arbitrary predictable two-arm block sizes"
    result["threshold"] = "1/delta, signed-mixture common-order proof"
    return result


def _block(*args):
    if len(args) == 3:
        m, km, kp = args
        args = (m, m, km, kp)
    if len(args) != 4 or any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in args):
        raise ValueError("integer arm trials and successes required")
    nm, np_, km, kp = map(int, args)
    if not (0 <= km <= nm and 0 <= kp <= np_):
        raise ValueError("invalid block counts")
    return {"nminus": nm, "nplus": np_, "minus_successes": km, "plus_successes": kp}


def _parse(block):
    if "nminus" in block:
        return _block(block["nminus"], block["nplus"], block["minus_successes"], block["plus_successes"])
    return _block(block["m"], block["minus_successes"], block["plus_successes"])


def _score(block):
    n = block["nminus"]+block["nplus"]
    return (Fraction(block["plus_successes"], 1)
            -Fraction((block["plus_successes"]+block["minus_successes"])*block["nplus"], n)) if n else Fraction(0)


def empty_state(experts=None):
    experts = expert_config() if experts is None else experts
    weights = [Fraction(*entry["weight"]) for entry in experts]
    assert sum(weights) == 1 and min(weights) > 0
    return {"experts": experts, "score": Fraction(0), "cgf": np.zeros(len(experts)),
            "log_odds": np.array([math.log(e["numerator"]/e["denominator"]) for e in experts]),
            "log_weights": np.array([math.log(float(w)) for w in weights]),
            "shots": 0, "blocks": []}


def _cgf(block, p, q, certified=False):
    nm, np_ = block["nminus"], block["nplus"]
    t = block["minus_successes"]+block["plus_successes"]
    if p == q or not nm or not np_ or t in (0, nm+np_):
        return (_Bounds(0), {"retained_terms": 1}) if certified else 0.
    if nm == np_:
        p, q = (p, q) if p >= q else (q, p)
        return balanced_bounds(nm, t, p, q) if certified else balanced_float(nm, t, p, q)
    return unequal_bounds(nm, np_, t, p, q) if certified else unequal_float(nm, np_, t, p, q)


def update_block(state, *args):
    block = _block(*args)
    state["score"] += _score(block)
    for j, expert in enumerate(state["experts"]):
        state["cgf"][j] += _cgf(block, expert["numerator"], expert["denominator"])
    state["shots"] += block["nminus"]+block["nplus"]
    state["blocks"].append(block)
    return state


def screen(state, delta=.01):
    evidence = float(logsumexp(float(state["score"])*state["log_odds"]-state["cgf"]+state["log_weights"]))
    sign = 1 if state["score"] > 0 else -1 if state["score"] < 0 else 0
    crossing = evidence >= math.log(1/delta)-1e-6 and sign != 0
    return {"log_evidence": evidence, "score_numerator": state["score"].numerator,
            "score_denominator": state["score"].denominator,
            "decision_candidate": ("reject" if sign > 0 else "accept") if crossing else "abstain",
            "crossing_candidate": crossing, "shots": state["shots"], "certified": False}


def certificate_from_blocks(blocks, delta=.01, experts=None):
    experts = expert_config() if experts is None else experts
    blocks = [_parse(block) for block in blocks]
    assert sum(Fraction(*e["weight"]) for e in experts) == 1
    score = sum((_score(block) for block in blocks), Fraction(0))
    bounded_score = _Bounds(score.numerator)/score.denominator
    cgfs, diagnostics = [_Bounds(0) for e in experts], []
    for index, block in enumerate(blocks):
        retained = []
        for j, e in enumerate(experts):
            value, details = _cgf(block, e["numerator"], e["denominator"], True)
            cgfs[j] += value
            retained.append(details["retained_terms"])
        diagnostics.append({"block": index, "retained_terms_per_positive_odds": retained})
    logs = []
    for e, cgf in zip(experts, cgfs):
        eta = (_Bounds(e["numerator"])/e["denominator"]).log()
        weight = (_Bounds(e["weight"][0])/e["weight"][1]).log()
        logs.append(bounded_score*eta-cgf+weight)
    offset = max(x.hi for x in logs)
    evidence = _Bounds(offset)+sum(((x-_Bounds(offset)).exp() for x in logs), _Bounds(0)).log()
    threshold = -_Bounds(Decimal(str(delta))).log()
    crossing = score != 0 and evidence.lo >= threshold.hi
    return {"method": protocol_config()["version"], "decision": (
            "reject" if score > 0 else "accept") if crossing else "abstain",
            "shots": sum(b["nminus"]+b["nplus"] for b in blocks),
            "score_numerator": score.numerator, "score_denominator": score.denominator,
            "evidence_bounds": [str(evidence.lo), str(evidence.hi)],
            "threshold_bounds": [str(threshold.lo), str(threshold.hi)],
            "evidence": _float_down(evidence.lo), "threshold": _float_up(threshold.hi),
            "diagnostics": diagnostics, "anytime_valid": True,
            "scope": "Fixed prior; predictable blocks and common true arm ordering required"}


def pair_grid_log(counts, experts=None):
    experts = expert_config() if experts is None else experts
    b, c = int(counts[1]), int(counts[2])
    return float(logsumexp([b*math.log(2*e["numerator"]/(e["numerator"]+e["denominator"]))
                           +c*math.log(2*e["denominator"]/(e["numerator"]+e["denominator"]))
                           +math.log(e["weight"][0]/e["weight"][1]) for e in experts]))


def pair_grid_bounds(counts, experts=None):
    experts = expert_config() if experts is None else experts
    b, c = int(counts[1]), int(counts[2])
    logs = []
    for e in experts:
        p, q = e["numerator"], e["denominator"]
        logs.append(b*(_Bounds(2*p)/(p+q)).log()+c*(_Bounds(2*q)/(p+q)).log()
                    +(_Bounds(e["weight"][0])/e["weight"][1]).log())
    offset = max(x.hi for x in logs)
    return _Bounds(offset)+sum(((x-_Bounds(offset)).exp() for x in logs), _Bounds(0)).log()
