"""Certified structured Bernoulli null likelihoods on exact rational queries.

Coordinates (cos_numerator, sin_numerator)/denominator define the model
exactly, and must lie on the unit circle by an integer identity. No rounded
trigonometric values are silently substituted for exact trigonometric ones.

null_fit accepts fractional pseudo-counts for experimental design ONLY.
null_loglik_upper requires integer observation counts. Its optimizer is only
a proposal: a concave tangent plus an outward support bound certifies the
global null maximum, irrespective of optimizer success or feasibility.
"""
import math
import numbers
from decimal import Decimal
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "circuit_validation"))
from direct_tests import _Bounds, _float_down, _float_up


FEATURE_KEYS = ("cos_numerator", "sin_numerator", "denominator")


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(name + " must be an integer")
    return int(value)


def rational_circle(u_num, u_den=1):
    """Unit-circle coordinates from the exact rational half-angle tangent."""
    u_num, u_den = _integer(u_num, "u_num"), _integer(u_den, "u_den")
    if u_den == 0:
        if u_num == 0:
            raise ValueError("undefined half-angle tangent")
        return dict(zip(FEATURE_KEYS, (-1, 0, 1)))
    divisor = math.gcd(u_num, u_den)
    u_num, u_den = u_num // divisor, u_den // divisor
    return dict(zip(FEATURE_KEYS, (u_den*u_den-u_num*u_num,
                                  2*u_num*u_den, u_den*u_den+u_num*u_num)))


def _coordinates(row):
    c, s, d = [_integer(row[key], key) for key in FEATURE_KEYS]
    if d <= 0 or c*c+s*s != d*d:
        raise ValueError("query coordinates must lie on the exact rational unit circle")
    divisor = math.gcd(math.gcd(abs(c), abs(s)), d)
    return c // divisor, s // divisor, d // divisor


def _rows(rows, integer_counts):
    result = []
    for row in rows:
        c, s, d = _coordinates(row)
        if integer_counts:
            k, n = (_integer(row[key], key) for key in ("successes", "trials"))
        else:
            if isinstance(row["successes"], bool) or isinstance(row["trials"], bool):
                raise ValueError("pseudo-counts must be real numbers")
            k, n = float(row["successes"]), float(row["trials"])
            if not math.isfinite(k) or not math.isfinite(n):
                raise ValueError("pseudo-counts must be finite")
        if not 0 <= k <= n:
            raise ValueError("require 0 <= successes <= trials")
        if n:
            result.append((c, s, d, k, n))
    return result


def _sign(sign):
    if isinstance(sign, bool) or sign not in (-1, 1):
        raise ValueError("sign must be +1 or -1")
    return int(sign)


def support_float(gradient, sign):
    """Numerical support of D with sign*C <= 0; not a certificate."""
    sign = _sign(sign)
    a, b, c = [float(x) for x in gradient]
    radial = math.hypot(b, max(0., -sign*c))
    return max(0., a, .5*(a+radial))


def _square(value):
    if value.lo <= 0 <= value.hi:
        upper = max((_Bounds(value.lo)*_Bounds(value.lo)).hi,
                    (_Bounds(value.hi)*_Bounds(value.hi)).hi)
        return _Bounds(0, upper)
    return value*value


def support_bounds(gradient, sign):
    """Outward interval containing the support for every enclosed gradient."""
    sign = _sign(sign)
    a, b, c = [_Bounds.as_bounds(x) for x in gradient]
    directed = -sign*c
    positive = _Bounds(max(Decimal(0), directed.lo), max(Decimal(0), directed.hi))
    radial = (_square(b)+_square(positive)).sqrt()
    middle = (a+radial)/2
    return _Bounds(max(Decimal(0), a.lo, middle.lo),
                   max(Decimal(0), a.hi, middle.hi))


def _candidate(candidate):
    if len(candidate) != 3:
        raise ValueError("candidate must contain A,B,C")
    result = []
    for value in candidate:
        if isinstance(value, bool):
            raise ValueError("candidate must be finite")
        if isinstance(value, Decimal):
            exact = value
        elif isinstance(value, numbers.Integral):
            exact = Decimal(int(value))
        else:
            exact = Decimal.from_float(float(value))
        if not exact.is_finite():
            raise ValueError("candidate must be finite")
        result.append(_Bounds(exact))
    return result


def _features(c, s, d):
    return (_Bounds(1), _Bounds(c)/d, _Bounds(s)/d)


def _likelihood_and_gradient(parsed, candidate):
    point = _candidate(candidate)
    value, gradient = _Bounds(0), [_Bounds(0) for unused in range(3)]
    for c, s, d, k, n in parsed:
        feature = _features(c, s, d)
        p = sum((feature[j]*point[j] for j in range(3)), _Bounds(0))
        if p.lo <= 0 or p.hi >= 1:
            raise ValueError("candidate must certify strictly interior predictions for observed queries")
        if k:
            value += k*p.log()
        if n-k:
            value += (n-k)*(1-p).log()
        score = _Bounds(k)/p-_Bounds(n-k)/(1-p)
        for j in range(3):
            gradient[j] += feature[j]*score
    return value, gradient, point


def loglik_bounds(rows, candidate):
    """Likelihood of ordered observations, without binomial coefficients."""
    value, unused, point = _likelihood_and_gradient(_rows(rows, True), candidate)
    return {"lower_decimal": str(value.lo), "upper_decimal": str(value.hi),
            "lower": _float_down(value.lo), "upper": _float_up(value.hi),
            "candidate_decimal": [str(x.lo) for x in point]}


def rational_prediction_loglik_bounds(rows, predictions):
    """Numerical bounds only; the caller must ensure predictions are predictable.

    Each prediction is (integer numerator, positive integer denominator).
    They must be strictly between zero and one. Rows are NOT aggregated
    across distinct predictions here.
    """
    rows = list(rows)
    predictions = list(predictions)
    if len(rows) != len(predictions):
        raise ValueError("one prediction required for every row")
    value = _Bounds(0)
    for row, prediction in zip(rows, predictions):
        parsed = _rows([row], True)
        numerator, denominator = [_integer(x, "prediction") for x in prediction]
        if not 0 < numerator < denominator:
            raise ValueError("predictions must be strictly between zero and one")
        if not parsed:
            continue
        k, n = parsed[0][3:]
        p = _Bounds(numerator)/denominator
        if k:
            value += k*p.log()
        if n-k:
            value += (n-k)*(1-p).log()
    return {"lower_decimal": str(value.lo), "upper_decimal": str(value.hi),
            "lower": _float_down(value.lo), "upper": _float_up(value.hi)}


def _saturated(parsed):
    value = _Bounds(0)
    for unused_c, unused_s, unused_d, k, n in parsed:
        if k:
            value += k*(_Bounds(k)/n).log()
        if n-k:
            value += (n-k)*(_Bounds(n-k)/n).log()
    return value


def _interior_proposal(candidate, sign, margin=1e-9):
    point = np.asarray(candidate, dtype=float).copy()
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        point = np.array([.5, 0., 0.])
    point[0] = np.clip(point[0], margin, 1-margin)
    point[2] = min(point[2], 0.) if sign == 1 else max(point[2], 0.)
    radius = math.hypot(point[1], point[2])
    maximum = max(0., min(point[0], 1-point[0])-margin)
    if radius > maximum:
        point[1:] *= maximum/radius
    return point


def null_fit(rows, sign, candidate=None):
    """Fast untrusted null MLE proposal; fractional design pseudo-counts allowed.

    The returned loglik_upper is floating point and must never authorize a
    decision. Pass candidate to null_loglik_upper for an outward certificate.
    """
    sign = _sign(sign)
    parsed = _rows(rows, False)
    if not parsed:
        return {"candidate": [.5, 0., 0.], "loglik": 0., "support_gap": 0.,
                "loglik_upper": 0., "success": True, "iterations": 0,
                "status": "empty observations", "certified": False}
    features = np.array([[1., c/d, s/d] for c, s, d, k, n in parsed])
    success = np.array([r[3] for r in parsed])
    total = np.array([r[4] for r in parsed])
    scale = float(total.sum())
    def objective(point):
        p = np.clip(features.dot(point), 1e-14, 1-1e-14)
        value = -(success*np.log(p)+(total-success)*np.log1p(-p)).sum()/scale
        gradient = -features.T.dot(success/p-(total-success)/(1-p))/scale
        return value, gradient
    def cone(point):
        a, b, c = point
        return np.array([a*a-b*b-c*c, (1-a)*(1-a)-b*b-c*c, -sign*c])
    def cone_jac(point):
        a, b, c = point
        return np.array([[2*a, -2*b, -2*c], [-2*(1-a), -2*b, -2*c], [0., 0., -sign]])
    if candidate is None:
        initial = [float((success.sum()+.5)/(scale+1.)), 0., 0.]
    else:
        initial = candidate
    initial = _interior_proposal(initial, sign)
    result = minimize(objective, initial, method="SLSQP", jac=True,
                      bounds=[(1e-10, 1-1e-10), (-.5, .5), (-.5, .5)],
                      constraints=[{"type": "ineq", "fun": cone, "jac": cone_jac}],
                      options={"ftol": 1e-12, "maxiter": 100})
    point = _interior_proposal(result.x, sign)
    value, negative_gradient = objective(point)
    loglik, gradient = -scale*value, -scale*negative_gradient
    gap = max(0., support_float(gradient, sign)-float(gradient.dot(point)))
    empirical = np.clip(success/total, 1e-300, 1-1e-16)
    saturated = float(np.where(success > 0, success*np.log(empirical), 0.).sum()
                      +np.where(total > success, (total-success)*np.log1p(-empirical), 0.).sum())
    return {"candidate": point.tolist(), "loglik": float(loglik),
            "support_gap": float(gap), "loglik_upper": min(0., saturated, float(loglik+gap)),
            "success": bool(result.success), "iterations": int(result.nit),
            "status": str(result.message), "certified": False}


def null_loglik_upper(rows, sign, candidate=None):
    """Global null log-likelihood upper bound, with exact integer counts.

    null D = {0<=A<=1, hypot(B,C)<=min(A,1-A), sign*C<=0}.
    The candidate need not belong to D. Every observed prediction must be
    certifiably inside (0,1), which is sufficient for the concave tangent.
    """
    sign = _sign(sign)
    rows = list(rows)
    parsed = _rows(rows, True)
    optimizer = None
    if candidate is None:
        optimizer = null_fit(rows, sign)
        candidate = optimizer["candidate"]
    value, gradient, point = _likelihood_and_gradient(parsed, candidate)
    support = support_bounds(gradient, sign)
    dot = sum((gradient[j]*point[j] for j in range(3)), _Bounds(0))
    tangent = value-dot+support
    saturated = _saturated(parsed)
    upper = min(Decimal(0), tangent.hi, saturated.hi)
    return {"sign": sign, "null": "sign*C<=0 in the closed physical double cone",
            "loglik_upper_decimal": str(upper), "loglik_upper": _float_up(upper),
            "tangent_upper_decimal": str(tangent.hi),
            "saturated_upper_decimal": str(saturated.hi),
            "candidate": [float(x.lo) for x in point],
            "candidate_decimal": [str(x.lo) for x in point],
            "candidate_loglik_bounds": [str(value.lo), str(value.hi)],
            "gradient_bounds": [[str(x.lo), str(x.hi)] for x in gradient],
            "support_bounds": [str(support.lo), str(support.hi)],
            "shots": sum(r[4] for r in parsed), "rows": len(parsed),
            "optimizer": optimizer, "certified": True,
            "arithmetic": "frozen direct_tests._Bounds outward Decimal arithmetic",
            "model_features": "exact rational unit-circle coordinates",
            "likelihood_convention": "ordered outcomes; no binomial coefficients"}


