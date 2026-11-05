"""Current-total-adaptive conditional Fisher evidence, with directed bounds.

Theory and permitted data flow
==============================
Before block b, choose a reflected arm, balanced per-arm count m_b and a
prediction rule from the past paid history F_(b-1). Bernoulli trials in the
block are independent, with probabilities p- and p+ fixed given that past.
Let S=kplus and T=kminus+kplus. Conditional on (F_(b-1),m_b,arm,T), S has
noncentral hypergeometric mass proportional to

  choose(m_b,S) choose(m_b,T-S) omega_true^S,
  omega_true = p+(1-p-) / [p-(1-p+)].

Endpoint probabilities are interpreted by their continuous limits. Select
ANY finite rational Omega_b >= 1 as a function only of (past,m_b,arm,T).
In particular q=(T+1/2)/(2m_b+1) and a clipped past-fit gap are allowed. Put

  Z_b(Omega) = sum_k choose(m_b,k) choose(m_b,T-k) Omega^k
              / choose(2m_b,T),
  F+_b = Omega_b^S / Z_b(Omega_b),
  F-_b = Omega_b^(-S) / Z_b(1/Omega_b).

At p+<=p- the true conditional distribution has odds<=1 and is MLR-smaller
than the central law. F+ is increasing in S and has central mean one, hence
its conditional mean is <=1. F- gives the symmetric result for p+>=p-.
Taking expectation over T proves the block supermartingale property. All
previously described predictable starts, past-data adaptation and FIXED
mixtures are therefore legal. At zero gap both directional processes are
supermartingales. The capital-argmax theorem in directional_safe.py gives
total wrong-direction risk delta at M=(E+ + E-)/2 >= 1/delta.

Adaptive/random query angles and m do not change the proof: condition on
their past-measurable realized values first. Both sides must be independent
within a block, and m must not be chosen by stopping on current outcomes.
The ordering must be shared across blocks; reflected positive-sine queries
in the fixed single-frequency physical model have this property.

The mathematical conditioning does not require a physical two-stage reveal
of T then S. A program may receive both counts simultaneously. Its auditor
must reconstruct Omega using only their sum T and the past, and reject any
extra dependence on their allocation S. Counterexample: m=1,T=1 under the
central null; choose Omega=9 at S=1 and Omega=1 at S=0. Then E[F+|T]=7/5>1.
Likewise a prior or expert weight fitted from the current allocation cannot
be applied retrospectively to the same block.

Arithmetic interface
====================
Each recorded block has integer keys m,total_successes,plus_successes and
odds=[positive numerator,positive denominator], with odds>=1. Distinct
experts may record distinct T-adaptive odds. log_capital_bounds(blocks)
returns (positive Bounds, negative Bounds, diagnostics). To combine a fixed
expert list, use mixture_log_capital_bounds(histories,weights). The latter
also returns a bound on log((E+ + E-)/2). These are LOG capital bounds.

Balanced symmetry gives centered K(lambda)=K(-lambda), so F+ and F- use
the same certified CGF. We reuse the frozen outward conditional_fisher
normalizer; nodes/odds are exact rational values, not rounded log-odds.
Predictability is a protocol/auditor obligation, not inferred from a list
of recorded counts. A floating screen never authorizes a decision.
"""
from fractions import Fraction
from math import comb
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/structured_validation"))
import conditional_fisher as frozen

Bounds = frozen._Bounds


def _fraction(value):
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (tuple, list)) and len(value) == 2:
        if any(isinstance(x, bool) or not isinstance(x, int) for x in value):
            raise ValueError("Fraction components must be integers")
        return Fraction(*value)
    raise ValueError("Expected Fraction or two-integer pair")


def parse_block(block):
    m, total, plus = [block[k] for k in ("m", "total_successes", "plus_successes")]
    if any(isinstance(x, bool) or not isinstance(x, int) for x in (m, total, plus)):
        raise ValueError("Block counts must be integers")
    if m < 0 or not 0 <= total <= 2*m or not max(0, total-m) <= plus <= min(m, total):
        raise ValueError("Invalid balanced total/allocation counts")
    odds = _fraction(block["odds"])
    if odds < 1:
        raise ValueError("Positive directional odds must be at least one")
    return m, total, plus, odds


def exact_block_factors(block):
    """Exhaustive finite checks only; sums can be expensive for large blocks."""
    m, total, plus, odds = parse_block(block)
    normalizer = sum(Fraction(comb(m, k)*comb(m, total-k))*odds**k
                     for k in range(max(0, total-m), min(m, total)+1)) / comb(2*m, total)
    positive = odds**plus/normalizer
    # Balanced central law is symmetric under S -> T-S.
    negative = odds**(total-plus)/normalizer
    return positive, negative


def log_capital_bounds(blocks):
    positive, negative, diagnostics = Bounds(0), Bounds(0), []
    for index, block in enumerate(blocks):
        m, total, plus, odds = parse_block(block)
        if odds == 1 or m == 0 or total in (0, 2*m):
            diagnostics.append({"block": index, "retained_terms": 1, "neutral": True})
            continue
        cgf, detail = frozen.centered_cgf_bounds(m, total, odds.numerator, odds.denominator)
        eta = (Bounds(odds.numerator)/odds.denominator).log()
        score = Bounds(2*plus-total)/2
        positive += score*eta-cgf
        negative += -score*eta-cgf
        diagnostics.append({"block": index, "retained_terms": detail["retained_terms"], "neutral": False})
    return positive, negative, diagnostics


def _log_sum(logs, weights):
    offset = max(value.hi for value in logs)
    value = sum(((x-Bounds(offset)).exp()*w.numerator/w.denominator
                 for x, w in zip(logs, weights)), Bounds(0))
    return Bounds(offset)+value.log()


def mixture_log_capital_bounds(histories, weights=None):
    histories = [list(history) for history in histories]
    if not histories:
        raise ValueError("At least one expert history is required")
    weights = [Fraction(1, len(histories))]*len(histories) if weights is None else [
        _fraction(w) for w in weights]
    if len(weights) != len(histories) or min(weights) <= 0 or sum(weights) != 1:
        raise ValueError("Fixed positive expert weights must sum to one")
    # All experts must refer to the same paid block totals and allocations.
    matrix = [[parse_block(b)[:3] for b in h] for h in histories]
    if any(x != matrix[0] for x in matrix[1:]):
        raise ValueError("Expert histories must share identical observed blocks")
    bounds = [log_capital_bounds(history) for history in histories]
    ep = _log_sum([b[0] for b in bounds], weights)
    en = _log_sum([b[1] for b in bounds], weights)
    combined = _log_sum([ep, en], [Fraction(1, 2), Fraction(1, 2)])
    return ep, en, combined, {"experts": len(histories),
                              "shots": 2*sum(x[0] for x in matrix[0]),
                              "expert_diagnostics": [b[2] for b in bounds]}


def certificate_from_histories(histories, weights=None, delta=Fraction(1, 100)):
    delta = _fraction(delta)
    if not 0 < delta < 1:
        raise ValueError("delta must be an interior exact probability")
    ep, en, total, diagnostics = mixture_log_capital_bounds(histories, weights)
    threshold = (Bounds(delta.denominator)/delta.numerator).log()
    decision = "abstain"
    if total.lo >= threshold.hi:
        if ep.lo > en.hi:
            decision = "reject"
        elif en.lo > ep.hi:
            decision = "accept"
    return {"method": "current_total_adaptive_conditional_fisher_v1",
            "decision": decision, "shots": diagnostics["shots"],
            "positive_log_capital_bounds": [str(ep.lo), str(ep.hi)],
            "negative_log_capital_bounds": [str(en.lo), str(en.hi)],
            "evidence_bounds": [str(total.lo), str(total.hi)],
            "threshold_bounds": [str(threshold.lo), str(threshold.hi)],
            "delta": [delta.numerator, delta.denominator], "diagnostics": diagnostics,
            "direction_rule": "argmax normalized directional capital; unresolved intervals abstain",
            "assumptions": "recorded odds uses past and current total only; balanced independent Bernoulli blocks with common ordering; fixed weights",
            "scope": "certified arithmetic conditional on audited data flow; no performance claim"}
