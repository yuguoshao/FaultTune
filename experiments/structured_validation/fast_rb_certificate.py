"""Independent lazy lower certificates of the unchanged RB full mixture."""
from decimal import Decimal
from fractions import Fraction
import numpy as np

import rb_grid as rb
from direct_tests import _Bounds, _float_down, _float_up


def _lse(logs):
    offset = max(log.hi for log in logs)
    return _Bounds(offset)+sum(((log-_Bounds(offset)).exp() for log in logs), _Bounds(0)).log()


def certificate_from_blocks(blocks, delta=.01, config=None, state=None):
    risk = Decimal(str(delta))
    if not risk.is_finite() or not 0 < risk < 1:
        raise ValueError("risk must lie in (0,1)")
    blocks = [rb._block(b["m"], b["minus_successes"], b["plus_successes"]) for b in blocks]
    if state is None:
        state = rb.empty_state(config)
        for b in blocks:
            rb.update_block(state, b["m"], b["minus_successes"], b["plus_successes"])
    config = state["config"] if config is None else rb.protocol_config(config)
    assert rb.config_digest(config) == rb.config_digest(state["config"])
    score = sum(b["plus_successes"]-b["minus_successes"] for b in blocks)
    assert score == state["score"] and state["shots"] == 2*sum(b["m"] for b in blocks)
    if not rb.screen(state, delta)["crossing_candidate"]:
        return rb.certificate_from_blocks(blocks, delta, config)
    experts, poles, layout = rb._layout(config)
    priorities = [state["_log_weights"][j]+(0. if sign == 0 else
                  sign*.5*score*state["_eta"][pole]+state["log_h"][pole])
                  for j, (pole, sign) in enumerate(layout)]
    order = np.argsort(-np.array(priorities)).tolist()
    threshold = -_Bounds(risk).log()
    analytic = []
    for odds, weight in experts:
        analytic.append((_Bounds(score)/2)*(_Bounds(odds.numerator)/odds.denominator).log()
                        +(_Bounds(weight.numerator)/weight.denominator).log())
    certified, verified_poles, diagnostics = {}, {}, []
    for rank, j in enumerate(order, 1):
        pole, sign = layout[j]
        if sign == 0:
            certified[j] = analytic[j]
        else:
            if pole not in verified_poles:
                h, terms = _Bounds(0), 0
                odds = poles[pole]
                for b in blocks:
                    value, detail = rb._log_h_bounds(b["m"], b["minus_successes"], b["plus_successes"],
                                                   odds.numerator, odds.denominator)
                    h += value
                    terms += detail["retained_terms"]
                verified_poles[pole] = h
                diagnostics.append({"pole": pole, "retained_terms": terms})
            certified[j] = analytic[j]+verified_poles[pole]
        if rank % 3 and rank != len(experts):
            continue
        lower = _lse(list(certified.values()))
        if lower.lo >= threshold.hi:
            upper = _lse([certified.get(k, analytic[k]) for k in range(len(experts))])
            return {"method": "independent_lazy_RB_certificate",
                    "decision": "reject" if score > 0 else "accept", "shots": state["shots"],
                    "score": score, "evidence_bounds": [str(lower.lo), str(upper.hi)],
                    "threshold_bounds": [str(threshold.lo), str(threshold.hi)],
                    "evidence": _float_down(lower.lo), "threshold": _float_up(threshold.hi),
                    "certified_experts": rank, "certified_poles": len(verified_poles),
                    "total_experts": len(experts), "diagnostics": diagnostics,
                    "scope": "Independent decision proof; does not recompute the stored tight full-mixture interval"}
    return rb.certificate_from_blocks(blocks, delta, config)
