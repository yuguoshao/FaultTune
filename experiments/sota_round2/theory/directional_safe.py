"""Adaptive full-count safe evidence with CAPITAL-BASED direction selection.

Implementable theorem
=====================
For each predictable balanced block choose rational 0 < u <= v < 1 before
its observations, put r=(u+v)/2, and multiply the positive capital by

  F+ = (u/r)^k- ((1-u)/(1-r))^(m-k-)
       (v/r)^k+ ((1-v)/(1-r))^(m-k+).

F- swaps u and v. Endpoints, odds, and inactive blocks u=v may depend on all
past paid observations. Different experts may use different predictable
start times and predictors. Fixed nonnegative weights summing to one mix
their positive capitals into E+ and negative capitals into E-, separately.

Stop when M=(E+ + E-)/2 >= 1/delta, and choose the direction of the LARGER
NORMALIZED DIRECTIONAL CAPITAL. Do not use the unweighted full-history score.
Ties or numerically unresolved direction comparisons may abstain.

Proof of the block property: write d=(v-u)/2 and h=d/[r(1-r)]. For a single
pair with true probabilities a=p- and b=p+, the expectation of F+ is
  [1-h(a-r)] [1+h(b-r)].
Both factors are positive. For a>=b the first factor is maximized at a=b;
the product is then 1-h^2(b-r)^2 <= 1. Independent pairs give the m-th power.
Thus each positive capital is a nonnegative supermartingale whenever the
true ordering is nonpositive; the negative case is symmetric. Predictable
starts, endpoint changes and fixed capital mixtures preserve this property.

Proof of total wrong-direction risk, WITHOUT a factor-two threshold:
If the true ordering is positive, a negative decision implies E- >= M >=
1/delta, so Ville applied to E- bounds its probability by delta. Negative
ordering is symmetric. If every block has equal true arm probabilities,
both directional capitals, hence M, are supermartingales; any decision has
probability at most delta. The same argument works for Ville-valid
directional e-processes and any fixed convex combination M, provided the
chosen direction maximizes the normalized directional capital. Equality
is handled by the M argument rather than by a union bound on directions.

This proof permits arbitrary adaptive odds and arbitrary mixtures of
predictable starts. It does NOT permit retrospectively chosen weights,
current-outcome predictions, or an unrelated raw-score direction rule.
The caller/auditor must reconstruct predictability from its paid history.
No unknown true model parameter is needed by this arithmetic module.

Why the direction change is necessary: under the fixed physical model
A=1/2,B=0,C=1/2, first collect 64 pairs at sin(x)=0 without betting. Then
collect 8 pairs at PSR with u=1/4,v=3/4. M=21523361/65536 > 100 always.
The full-history score chooses the WRONG sign with probability
P[Binomial(128,1/2)<=55] = 0.06631240066249112. The capital rule is always
positive. All 144 shots are still charged. This is a protocol counterexample,
not a fitted instance or a new Monte Carlo result.

This module supplies certified evidence arithmetic, not a new simulation
driver or a claim of performance improvement. It reuses the already reviewed
outward-Decimal primitives; the new finite checks use exact rational factors.
"""
from decimal import Decimal
from fractions import Fraction
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/circuit_validation"))
from direct_tests import _Bounds


def _fraction(value):
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return Fraction(*value)
    if isinstance(value, int) and not isinstance(value, bool):
        return Fraction(value)
    raise ValueError("Use exact Fraction or numerator/denominator pairs")


def _parsed(block):
    m, km, kp = [block[k] for k in ("m", "minus_successes", "plus_successes")]
    if any(isinstance(x, bool) or not isinstance(x, int) for x in (m, km, kp)):
        raise ValueError("Counts must be integers")
    if not 0 <= km <= m or not 0 <= kp <= m:
        raise ValueError("Invalid balanced block")
    u, v = [_fraction(block[k]) for k in ("lower", "upper")]
    if not 0 < u <= v < 1:
        raise ValueError("Require rational 0 < lower <= upper < 1")
    return m, km, kp, u, v


def rational_factors(lower, upper):
    """Return F+ for outcomes 00, 01, 10, 11; F- swaps the middle two."""
    u, v = _fraction(lower), _fraction(upper)
    if not 0 < u <= v < 1:
        raise ValueError("Require rational interior ordered endpoints")
    r = (u + v) / 2
    return ((1-u)*(1-v)/(1-r)**2, (1-u)*v/((1-r)*r),
            u*(1-v)/((1-r)*r), u*v/r**2)


def exact_block_capitals(block):
    """Small deterministic checks only; large powers can create huge integers."""
    m, km, kp, u, v = _parsed(block)
    r = (u + v) / 2
    positive = (u/r)**km * ((1-u)/(1-r))**(m-km)
    positive *= (v/r)**kp * ((1-v)/(1-r))**(m-kp)
    negative = (v/r)**km * ((1-v)/(1-r))**(m-km)
    negative *= (u/r)**kp * ((1-u)/(1-r))**(m-kp)
    return positive, negative


def _bound(q):
    return _Bounds(q.numerator) / q.denominator


def block_log_bounds(block):
    m, km, kp, u, v = _parsed(block)
    if u == v or m == 0:
        return _Bounds(0), _Bounds(0)
    r = (u + v) / 2
    a, b = _bound(u/r).log(), _bound((1-u)/(1-r)).log()
    c, d = _bound(v/r).log(), _bound((1-v)/(1-r)).log()
    return km*a + (m-km)*b + kp*c + (m-kp)*d, \
        km*c + (m-km)*d + kp*a + (m-kp)*b


def _log_sum(logs, weights):
    offset = max(x.hi for x in logs)
    total = sum(((x-_Bounds(offset)).exp()*_bound(w)
                 for x, w in zip(logs, weights)), _Bounds(0))
    return _Bounds(offset) + total.log()


def certificate_from_histories(histories, weights=None, delta=Fraction(1, 100)):
    """Certify a FIXED mixture of recorded predictable expert histories.

    Histories must cover the same paid observations, including inactive u=v
    blocks. They can split blocks differently if the predictions were chosen
    before the corresponding observations. This function verifies equal paid
    counts; the protocol auditor must verify the actual shared observations.
    """
    histories = [list(x) for x in histories]
    if not histories:
        raise ValueError("At least one expert history is required")
    weights = [Fraction(1, len(histories))] * len(histories) if weights is None else [
        _fraction(x) for x in weights]
    if len(weights) != len(histories) or min(weights) <= 0 or sum(weights) != 1:
        raise ValueError("Fixed positive weights must sum exactly to one")
    delta = _fraction(delta)
    if not 0 < delta < 1:
        raise ValueError("delta must be an interior exact probability")
    totals, positive, negative = [], [], []
    for history in histories:
        lp, ln, shots = _Bounds(0), _Bounds(0), 0
        for block in history:
            p, n = block_log_bounds(block)
            lp, ln, shots = lp+p, ln+n, shots+2*block["m"]
        totals.append(shots)
        positive.append(lp)
        negative.append(ln)
    if len(set(totals)) != 1:
        raise ValueError("Expert histories must include equal paid shot counts")
    ep, en = _log_sum(positive, weights), _log_sum(negative, weights)
    log_m = _log_sum([ep, en], [Fraction(1, 2), Fraction(1, 2)])
    threshold = (_Bounds(delta.denominator)/delta.numerator).log()
    decision = "abstain"
    if log_m.lo >= threshold.hi:
        if ep.lo > en.hi:
            decision = "reject"
        elif en.lo > ep.hi:
            decision = "accept"
    return {"method": "adaptive_fullsafe_directional_capital_v1",
            "decision": decision, "shots": totals[0], "experts": len(histories),
            "evidence_bounds": [str(log_m.lo), str(log_m.hi)],
            "positive_log_capital_bounds": [str(ep.lo), str(ep.hi)],
            "negative_log_capital_bounds": [str(en.lo), str(en.hi)],
            "threshold_bounds": [str(threshold.lo), str(threshold.hi)],
            "delta": [delta.numerator, delta.denominator],
            "direction_rule": "argmax normalized directional capital; unresolved intervals abstain",
            "assumptions": "predictable endpoints and balanced independent Bernoulli blocks with common ordering; fixed expert weights",
            "numeric_status": "outward Decimal bounds on exact rational full-count factors",
            "scope": "validity conditional on audited predictability, no performance claim"}


def smoothed_endpoints(past_minus, past_plus, past_pairs):
    """One concrete adaptive predictor, using ONLY past paid marginal counts.

    Jeffreys-smoothed empirical rates are sorted. At n=0, or tied rates, the
    factor is one. More sophisticated structural predictions can use the same
    certificate API. This baseline predictor is not asserted to be optimal.
    """
    values = (past_minus, past_plus, past_pairs)
    if any(isinstance(x, bool) or not isinstance(x, int) for x in values):
        raise ValueError("Past counts must be integers")
    if not 0 <= past_minus <= past_pairs or not 0 <= past_plus <= past_pairs:
        raise ValueError("Invalid past counts")
    return tuple(sorted((Fraction(2*past_minus+1, 2*past_pairs+2),
                         Fraction(2*past_plus+1, 2*past_pairs+2))))
