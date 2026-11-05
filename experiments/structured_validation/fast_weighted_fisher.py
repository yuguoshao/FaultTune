"""Lazy exact lower certificates for the same fixed weighted Fisher mixture.

Every omitted expert is nonnegative. A certified partial sum may therefore
authorize exactly the same mathematical threshold crossing. If a partial
sum does not suffice, more experts are certified; no risk budget changes.
Float values select the order only. Cap/failed crossings retain full bounds.
"""
from decimal import Decimal
from fractions import Fraction
import math
import time

import numpy as np

import weighted_fisher as base
from direct_tests import _Bounds, _float_down, _float_up

empty_state = base.empty_state
update_block = base.update_block
screen = base.screen
pair_grid_log = base.pair_grid_log
pair_grid_bounds = base.pair_grid_bounds


def protocol_config():
    config = base.protocol_config()
    config["numerical_certificate"] = "Lazy outward partial sum; omitted nonnegative experts have analytic upper bounds"
    return config


def _lse(values):
    offset = max(value.hi for value in values)
    return _Bounds(offset)+sum(((value-_Bounds(offset)).exp() for value in values), _Bounds(0)).log()


def certificate_from_blocks(blocks, delta=.01, experts=None, state=None):
    risk = Decimal(str(delta))
    if not risk.is_finite() or not 0 < risk < 1:
        raise ValueError("risk must be in (0,1)")
    parsed = [base._parse(block) for block in blocks]
    if state is None:
        state = base.empty_state(experts)
        for block in parsed:
            base.update_block(state, block["nminus"], block["nplus"],
                              block["minus_successes"], block["plus_successes"])
    experts = state["experts"] if experts is None else experts
    assert experts == state["experts"]
    score = sum((base._score(block) for block in parsed), Fraction(0))
    assert score == state["score"] and state["shots"] == sum(
        b["nminus"]+b["nplus"] for b in parsed)
    if score == 0 or not base.screen(state, delta)["crossing_candidate"]:
        return base.certificate_from_blocks(parsed, delta, experts)
    bounded_score = _Bounds(score.numerator)/score.denominator
    threshold = -_Bounds(risk).log()
    priorities = float(score)*state["log_odds"]-state["cgf"]+state["log_weights"]
    ordering = np.argsort(-priorities).tolist()
    certified, analytic = {}, {}
    for j, expert in enumerate(experts):
        eta = (_Bounds(expert["numerator"])/expert["denominator"]).log()
        weight = (_Bounds(expert["weight"][0])/expert["weight"][1]).log()
        analytic[j] = bounded_score*eta+weight  # CGF >= 0 by Jensen.
    diagnostics, lower = [], None
    for rank, j in enumerate(ordering, 1):
        expert = experts[j]
        cgf = _Bounds(0)
        terms = 0
        for block in parsed:
            value, detail = base._cgf(block, expert["numerator"], expert["denominator"], True)
            cgf += value
            terms += detail["retained_terms"]
        certified[j] = analytic[j]-cgf
        diagnostics.append({"expert": j, "retained_terms": terms})
        # First check at 3 experts, then after every 3; final always checked.
        if rank % 3 and rank != len(experts):
            continue
        lower = _lse(list(certified.values()))
        if lower.lo >= threshold.hi:
            upper = _lse([certified.get(k, analytic[k]) for k in range(len(experts))])
            return {"method": "lazy_"+base.protocol_config()["version"],
                    "decision": "reject" if score > 0 else "accept",
                    "shots": state["shots"], "score_numerator": score.numerator,
                    "score_denominator": score.denominator,
                    "evidence_bounds": [str(lower.lo), str(upper.hi)],
                    "threshold_bounds": [str(threshold.lo), str(threshold.hi)],
                    "evidence": _float_down(lower.lo), "threshold": _float_up(threshold.hi),
                    "certified_experts": rank, "total_experts": len(experts),
                    "diagnostics": diagnostics, "anytime_valid": True,
                    "scope": "Same full-mixture threshold, proved using a nonnegative partial sum"}
    return base.certificate_from_blocks(parsed, delta, experts)
