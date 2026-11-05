"""A shared physical-parameter predictive likelihood, with a KT fallback.

Q = (Q_grid + Q_KT)/2 is a single probability distribution on observations
under predictable actions. Q_grid is a fixed rational mixture of physical
sinusoids, and Q_KT is a product of arm-wise Jeffreys marginals. Neither
depends on unknown truth. For every true physical theta, Q/L_theta is a
nonnegative likelihood-ratio martingale (supermartingale on support edges).
Consequently Q/sup_opposite L may certify either sign at 1/delta, using the
same numerator for both signs. The usual common-event argument covers C=0.

The finite grid is an engineering prior, not a continuum approximation
guarantee. Q >= Q_KT/2 bounds loss against the KT numerator by log(2), not
by a fixed multiplier on stopping cost. Shared structure or this mixture
framework is not claimed as a new statistical principle.

Only a data-selected positive partial sum of the ORIGINAL grid prior is
needed for a certified numerator lower bound. It is never renormalized.
"""
from decimal import Decimal
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.special import gammaln, logsumexp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/structured_validation"))
import certified_likelihood as likelihood
from direct_tests import _Bounds, _log_marginal

MEAN_ODDS_BASE = Fraction(8, 5)
MEAN_EXPONENTS = tuple(range(-18, 19))
RADII = (Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4),
         Fraction(9, 10), Fraction(49, 50), Fraction(999, 1000))
ANGLE_COUNT = 32
ANGLE_MAX_DENOMINATOR = 4096
DEFAULT_RETAINED_TERMS = 64


def protocol_config():
    return {"version": "physical_grid_KT_predictive_v1", "mean_odds_base": [8, 5],
            "mean_exponents": list(MEAN_EXPONENTS), "radii": [[r.numerator, r.denominator] for r in RADII],
            "angle_count": ANGLE_COUNT, "half_angle_max_denominator": ANGLE_MAX_DENOMINATOR,
            "weights": "uniform over means, uniform over radii; uniform angles for each nonzero radius",
            "outer_weights": [[1, 2], [1, 2]], "default_retained_terms": DEFAULT_RETAINED_TERMS,
            "retention": "top floating-likelihood terms, retaining their original exact weights; no renormalization",
            "sampling": "any predictable sequence on the fixed registered rational arms",
            "scope": "A valid engineering predictive mixture, not an empirical superiority or novelty claim."}


@lru_cache(maxsize=1)
def models():
    directions = []
    for j in range(ANGLE_COUNT):
        if j == ANGLE_COUNT // 2:
            arm = likelihood.rational_circle(1, 0)
        else:
            tangent = Fraction(math.tan(math.pi * j / ANGLE_COUNT)).limit_denominator(ANGLE_MAX_DENOMINATOR)
            arm = likelihood.rational_circle(tangent.numerator, tangent.denominator)
        directions.append((Fraction(arm["cos_numerator"], arm["denominator"]),
                           Fraction(arm["sin_numerator"], arm["denominator"])))
    assert len(set(directions)) == ANGLE_COUNT
    result = []
    for exponent in MEAN_EXPONENTS:
        odds = MEAN_ODDS_BASE ** exponent
        a = odds / (1 + odds)
        for relative in RADII:
            radius = relative * min(a, 1 - a)
            active_directions = directions if radius else [(Fraction(0), Fraction(0))]
            weight = Fraction(1, len(MEAN_EXPONENTS) * len(RADII) * len(active_directions))
            for co, si in active_directions:
                b, c = radius * co, radius * si
                assert b * b + c * c < min(a, 1 - a) ** 2
                result.append((a, b, c, weight))
    assert sum((r[3] for r in result), Fraction(0)) == 1
    return tuple(result)


@lru_cache(maxsize=1)
def grid_sha256():
    data = [[[v.numerator, v.denominator] for v in model] for model in models()]
    return hashlib.sha256(json.dumps(data, separators=(",", ":")).encode()).hexdigest()


def _key(arms):
    values = tuple(likelihood._coordinates(arm) for arm in arms)
    if not values or len(set(values)) != len(values):
        raise ValueError("Require distinct registered rational arms")
    return values


@lru_cache(maxsize=8)
def _tables(key):
    parameters = models()
    probabilities = []
    for co, si, denominator in key:
        values = [a + b * Fraction(co, denominator) + c * Fraction(si, denominator)
                  for a, b, c, weight in parameters]
        assert all(0 < p < 1 for p in values)
        probabilities.append(np.array([float(p) for p in values]))
    p = np.array(probabilities)
    return np.log(p), np.log1p(-p), np.array([math.log(float(m[3])) for m in parameters])


def _counts(arms, trials, successes):
    key = _key(arms)
    if len(key) != len(trials) or len(key) != len(successes):
        raise ValueError("One pair of counts per registered arm")
    for n, k in zip(trials, successes):
        if isinstance(n, (bool, np.bool_)) or isinstance(k, (bool, np.bool_)):
            raise ValueError("Boolean counts forbidden")
        if int(n) != n or int(k) != k or not 0 <= k <= n:
            raise ValueError("Require integer 0 <= successes <= trials")
    return key, np.asarray(trials, dtype=np.int64), np.asarray(successes, dtype=np.int64)


def _grid_logs(key, n, k):
    lp, lf, lw = _tables(key)
    return np.dot(k, lp) + np.dot(n - k, lf) + lw


def screen_log_numerator(arms, trials, successes):
    key, n, k = _counts(arms, trials, successes)
    grid = float(logsumexp(_grid_logs(key, n, k)))
    kt = float((gammaln(k + .5) + gammaln(n - k + .5) - gammaln(n + 1) - 2 * gammaln(.5)).sum())
    return float(np.logaddexp(grid, kt) - math.log(2))


def _bound(value):
    return _Bounds(value.numerator) / value.denominator


def numerator_lower_bound(arms, trials, successes, max_terms=DEFAULT_RETAINED_TERMS):
    key, n, k = _counts(arms, trials, successes)
    if isinstance(max_terms, bool) or int(max_terms) != max_terms or max_terms < 1:
        raise ValueError("Positive integer retained-term count required")
    parameters = models()
    scores = _grid_logs(key, n, k)
    selected = sorted(range(len(parameters)), key=lambda j: (-scores[j], j))[:int(max_terms)]
    logs, mass = [], Fraction(0)
    for j in selected:
        a, b, c, weight = parameters[j]
        value = _bound(weight).log()
        for (co, si, denominator), count, successes_ in zip(key, n, k):
            if not count:
                continue
            p = a + b * Fraction(co, denominator) + c * Fraction(si, denominator)
            if successes_:
                value += int(successes_) * _bound(p).log()
            if count - successes_:
                value += int(count - successes_) * _bound(1 - p).log()
        logs.append(value)
        mass += weight
    offset = max(value.hi for value in logs)
    grid_partial = _Bounds(offset) + sum(((v - _Bounds(offset)).exp() for v in logs), _Bounds(0)).log()
    kt = sum((_log_marginal(int(success), int(count), .5) for count, success in zip(n, k)), _Bounds(0))
    offset = max(grid_partial.hi, kt.hi)
    combined = _Bounds(offset) + (((grid_partial - _Bounds(offset)).exp() + (kt - _Bounds(offset)).exp()) / 2).log()
    return {"log_lower_decimal": str(combined.lo),
            "partial_numerator_log_bounds": [str(combined.lo), str(combined.hi)],
            "retained_model_indices": selected, "retained_prior_mass": [mass.numerator, mass.denominator],
            "grid_partial_log_bounds": [str(grid_partial.lo), str(grid_partial.hi)],
            "kt_log_bounds": [str(kt.lo), str(kt.hi)],
            "bounds_scope": "Bounds enclose the retained-grid-plus-KT partial numerator; only its lower endpoint bounds the FULL numerator."}


def certificate(arms, trials, successes, sign, delta=.01, proposal=None, max_terms=DEFAULT_RETAINED_TERMS):
    key, n, k = _counts(arms, trials, successes)
    if isinstance(sign, bool) or sign not in (-1, 1) or isinstance(delta, bool) or not 0 < delta < 1:
        raise ValueError("Invalid sign/risk")
    rows = [dict(arm, trials=int(count), successes=int(success)) for arm, count, success in zip(arms, n, k)]
    numerator = numerator_lower_bound(arms, n, k, max_terms=max_terms)
    null = likelihood.null_loglik_upper(rows, sign, candidate=proposal)
    evidence = (_Bounds(Decimal(numerator["log_lower_decimal"])) - _Bounds(Decimal(null["loglik_upper_decimal"]))).lo
    threshold = (-_Bounds(Decimal(str(delta))).log()).hi
    decision = ("reject" if sign > 0 else "accept") if evidence > threshold else "abstain"
    return {"decision": decision, "sign": int(sign), "shots": int(n.sum()), "delta": float(delta),
            "calibration": "structured_grid_safe", "evidence_lower_decimal": str(evidence),
            "threshold_upper_decimal": str(threshold), "numerator_certificate": numerator,
            "null_certificate": null, "protocol": protocol_config(), "grid_sha256": grid_sha256(),
            "anytime_valid_under_stated_model": True,
            "limits": "Fixed grid prior with KT fallback; partial-sum certification can delay stops. No fixed stopping-cost ratio guarantee."}
