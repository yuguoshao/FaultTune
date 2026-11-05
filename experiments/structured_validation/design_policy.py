"""Fit-only finite-grid Chernoff design for the physical Bernoulli sinusoid.

The fitted law is p(x)=A+B*cos(x)+C*sin(x). The simulator truth must never
be supplied to the adaptive controller. All returned optimization bounds
are floating point diagnostics, not certificates for statistical decisions.

For a physical fitted law with C!=0, a KL-minimizing wrong-sign law can be
chosen with C'=0: move any strictly wrong-sign law toward the fitted law
until C'=0, using convexity of both the physical domain and KL. The finite
arm max-min design is consequently dual to a two-dimensional convex
epigraph problem over |b|<=min(a,1-a). A small linear program recovers
primal arm weights. Actual observed-data null likelihoods still require
the full three-dimensional oracle in certified_likelihood.py.
"""
from dataclasses import asdict, dataclass
from fractions import Fraction
import math
import numbers
import time
import warnings

import numpy as np
from scipy.optimize import linprog, minimize

from certified_likelihood import rational_circle, support_float


@dataclass(frozen=True)
class DesignConfig:
    physical_margin: float = 1e-10
    tiny_signal: float = 1e-10
    epigraph_tolerance: float = 2e-12
    max_iterations: int = 150
    lp_tolerance: float = 1e-10
    acceptable_relative_gap: float = 1e-5

    def __post_init__(self):
        values = (self.physical_margin, self.tiny_signal,
                  self.epigraph_tolerance, self.lp_tolerance,
                  self.acceptable_relative_gap)
        if not all(math.isfinite(x) and x > 0 for x in values):
            raise ValueError("configuration tolerances must be finite and positive")
        if not 0 < self.physical_margin < .01:
            raise ValueError("invalid physical margin")
        if type(self.max_iterations) is not int or self.max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")


DEFAULT_CONFIG = DesignConfig()


def config_dict(config=DEFAULT_CONFIG):
    return asdict(config)


def _canonical(arm):
    values = [arm[key] for key in
              ("cos_numerator", "sin_numerator", "denominator")]
    if any(isinstance(x, bool) or not isinstance(x, numbers.Integral)
           for x in values):
        raise ValueError("arm coordinates must be integers")
    c, s, d = map(int, values)
    if d <= 0 or c*c+s*s != d*d:
        raise ValueError("arm must be on the exact rational unit circle")
    divisor = math.gcd(math.gcd(abs(c), abs(s)), d)
    return c//divisor, s//divisor, d//divisor


def rational_grid(size=65, max_denominator=64):
    """Public rational full-circle grid, paired under sin reflection.

    Includes angle zero and the exact PSR arms at +/-pi/2. Odd cardinality
    leaves pi absent. max_denominator caps the half-angle rational, not
    the denominator of the resulting circle coordinates.
    """
    if type(size) is not int or size < 5 or size % 2 != 1:
        raise ValueError("grid size must be an odd integer >=5")
    if type(max_denominator) is not int or max_denominator < 2:
        raise ValueError("max_denominator must be an integer >=2")
    angles = [2*math.pi*j/size for j in range(1, (size+1)//2)]
    angles[min(range(len(angles)), key=lambda j: abs(angles[j]-math.pi/2))] = math.pi/2
    positive = []
    for angle in angles:
        u = (Fraction(1, 1) if angle == math.pi/2 else
             Fraction(math.tan(angle/2)).limit_denominator(max_denominator))
        positive.append(_canonical(rational_circle(u.numerator, u.denominator)))
    coords = [(1, 0, 1)]+positive+[(c, -s, d) for c, s, d in positive]
    if len(set(coords)) != size or any(s <= 0 for c, s, d in positive):
        raise ValueError("rational approximation collided; increase max_denominator")
    return [{"index": index, "cos_numerator": c, "sin_numerator": s,
             "denominator": d} for index, (c, s, d) in enumerate(coords)]


GRID = rational_grid()


def _grid_arrays(grid):
    coords = [_canonical(arm) for arm in grid]
    if len(coords) < 2 or len(set(coords)) != len(coords):
        raise ValueError("grid must have distinct arms")
    if any(arm.get("index", i) != i for i, arm in enumerate(grid)):
        raise ValueError("grid index must equal its list position")
    lookup = {key: i for i, key in enumerate(coords)}
    try:
        reflection = np.array([lookup[(c, -s, d)] for c, s, d in coords], dtype=int)
    except KeyError:
        raise ValueError("grid must be closed under sin reflection")
    if (0, 1, 1) not in lookup or (0, -1, 1) not in lookup:
        raise ValueError("grid must include the exact PSR arms")
    return (np.array([c/d for c, s, d in coords]),
            np.array([s/d for c, s, d in coords]), reflection)


def _fit_array(fit):
    if isinstance(fit, dict):
        fit = fit["theta"] if "theta" in fit else [fit[k] for k in ("A", "B", "C")]
    theta = np.asarray(fit, dtype=float)
    if theta.shape != (3,) or not np.all(np.isfinite(theta)):
        raise ValueError("fit must contain finite scalar A,B,C")
    a, b, c = theta
    if not 0 < a < 1 or not math.hypot(b, c) < min(a, 1-a):
        raise ValueError("fit must be strictly inside the physical sinusoid domain")
    return theta


def _relative_entropy_remainder(x):
    x = np.asarray(x, dtype=float)
    small = np.abs(x) < .001
    safe = np.where(small, 0., x)
    regular = (1+safe)*np.log1p(safe)-safe
    polynomial = x*x*(.5+x*(-1/6+x*(1/12+x*(-1/20+x*(1/30+x*(-1/42+x/56))))))
    return np.where(small, polynomial, regular)


def bernoulli_kl(p, q):
    """Stable interior Bernoulli KL, avoiding entropy cancellation."""
    p, q = np.broadcast_arrays(np.asarray(p, dtype=float), np.asarray(q, dtype=float))
    if np.any(p <= 0) or np.any(p >= 1) or np.any(q <= 0) or np.any(q >= 1):
        raise ValueError("KL arguments must be strictly interior")
    return (q*_relative_entropy_remainder((p-q)/q)
            +(1-q)*_relative_entropy_remainder((q-p)/(1-q)))


def _diagnostics(theta, cosine, sine, weights, candidate):
    p = theta[0]+theta[1]*cosine+theta[2]*sine
    q = candidate[0]+candidate[1]*cosine+candidate[2]*sine
    rates = bernoulli_kl(p, q)
    score = weights*(p-q)/(q*(1-q))
    gradient = np.array([score.sum(), score.dot(cosine), score.dot(sine)])
    sign = 1 if theta[2] >= 0 else -1
    residual = max(0., support_float(gradient, sign)-float(gradient.dot(candidate)))
    value, upper = float(weights.dot(rates)), float(rates.max())
    lower = max(0., value-residual)
    return {"info_lower": lower, "info_upper": upper,
            "weighted_candidate_kl": value, "support_residual": residual,
            "relative_gap": max(0., upper-lower)/max(upper, 1e-300),
            "null_gradient": gradient.tolist()}


def _equal_pair(theta, cosine, sine, reflection):
    """A feasible design used as a documented numerical fallback."""
    midpoint = theta[0]+theta[1]*cosine
    p = midpoint+theta[2]*sine
    individual = bernoulli_kl(p, midpoint)
    rate = .5*(individual+individual[reflection])
    available = np.flatnonzero(sine > 0)
    index = (int(np.flatnonzero((cosine == 0) & (sine == 1))[0])
             if theta[2] == 0 else int(available[np.argmax(rate[available])]))
    weights = np.zeros(len(cosine))
    weights[index] = weights[reflection[index]] = .5
    return weights, np.array([theta[0], theta[1], 0.])


def design_from_fit(fit, grid=GRID, config=DEFAULT_CONFIG):
    """Choose arms using only an already fitted paid-data physical law.

    The caller owns exploration, sampling, shot accounting, and predictable
    replanning. No randomness or observation outcomes are consumed here.
    info_lower/info_upper refer to this finite public grid in floating point.
    Neither finite-grid global optimality nor statistical validity is certified.
    """
    started = time.perf_counter()
    theta = _fit_array(fit)
    cosine, sine, reflection = _grid_arrays(grid)
    # Canonicalizing the fitted sign makes reflection exact and reproducible.
    canonical = theta.copy()
    canonical[2] = abs(canonical[2])
    a, b, h = canonical
    fallback_weights, fallback_candidate = _equal_pair(canonical, cosine, sine, reflection)
    weights, candidate = fallback_weights, fallback_candidate
    status, solver_success, lp_success, iterations = "equal_pair_fallback", False, False, 0
    warnings_seen, epigraph_message, lp_message = [], "not run", "not run"
    physical_slack = min(a, 1-a)-math.hypot(b, h)
    margin = min(config.physical_margin, .1*physical_slack)
    p = a+b*cosine+h*sine
    scale = float(bernoulli_kl(p, a+b*cosine).max())

    if h == 0:
        status = "zero_signal_tie_equal_pair"
    elif h <= config.tiny_signal or scale <= 1e-22:
        status = "tiny_signal_equal_pair_fallback"
    else:
        def unpack(z):
            return a+h*z[0], b+h*z[1]

        def rates_gradient(z):
            aa, bb = unpack(z)
            q = np.clip(aa+bb*cosine, 1e-15, 1-1e-15)
            rate = bernoulli_kl(p, q)/scale
            derivative = h*(q-p)/(q*(1-q))/scale
            return rate, derivative

        def inequalities(z):
            rate, unused = rates_gradient(z)
            return z[2]-rate

        def inequalities_jac(z):
            unused, derivative = rates_gradient(z)
            return np.column_stack([-derivative, -derivative*cosine, np.ones(len(cosine))])

        physical_jac = np.array([[1., 1., 0.], [1., -1., 0.],
                                 [-1., 1., 0.], [-1., -1., 0.]])

        def physical(z):
            aa, bb = unpack(z)
            return np.array([aa+bb-margin, aa-bb-margin,
                             1-aa+bb-margin, 1-aa-bb-margin])/h

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = minimize(lambda z: (z[2], np.array([0., 0., 1.])),
                              np.array([0., 0., 1.]), jac=True, method="SLSQP",
                              bounds=[((margin-a)/h, (1-margin-a)/h),
                                      ((-.5+margin-b)/h, (.5-margin-b)/h), (0., None)],
                              constraints=[{"type": "ineq", "fun": inequalities,
                                            "jac": inequalities_jac},
                                           {"type": "ineq", "fun": physical,
                                            "jac": lambda z: physical_jac}],
                              options={"ftol": config.epigraph_tolerance,
                                       "maxiter": config.max_iterations})
            warnings_seen.extend(str(w.message) for w in caught)
        epigraph_message = str(result.message)
        solver_success, iterations = bool(result.success), int(result.nit)
        aa, bb = unpack(result.x)
        if math.isfinite(aa) and math.isfinite(bb):
            aa = float(np.clip(aa, margin, 1-margin))
            bb = float(np.clip(bb, -min(aa, 1-aa)+margin, min(aa, 1-aa)-margin))
            proposed = np.array([aa, bb, 0.])
            q = aa+bb*cosine
            rates = bernoulli_kl(p, q)
            derivative = (q-p)/(q*(1-q))
            equalities = np.vstack([np.ones(len(cosine)), derivative, derivative*cosine])
            row_scale = np.maximum(np.abs(equalities).max(axis=1), 1e-300)
            equalities = equalities/row_scale[:, None]
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                primal = linprog(-rates/scale, A_eq=equalities,
                                 b_eq=np.array([1., 0., 0.])/row_scale,
                                 bounds=(0., None), method="revised simplex",
                                 options={"tol": config.lp_tolerance, "maxiter": 1000})
                warnings_seen.extend(str(w.message) for w in caught)
            lp_message, lp_success = str(primal.message), bool(primal.success)
            if primal.success and np.all(np.isfinite(primal.x)) and primal.x.sum() > 0:
                proposed_weights = np.maximum(primal.x, 0.)
                proposed_weights[proposed_weights < 1e-14] = 0.
                proposed_weights /= proposed_weights.sum()
                proposal_diagnostics = _diagnostics(canonical, cosine, sine,
                                                    proposed_weights, proposed)
                fallback_diagnostics = _diagnostics(canonical, cosine, sine,
                                                    fallback_weights, fallback_candidate)
                comparison_slack = 1e-10*scale+1e-24
                if proposal_diagnostics["info_lower"] >= fallback_diagnostics["info_lower"]-comparison_slack:
                    weights, candidate = proposed_weights, proposed
                    status = ("finite_grid_primal_dual" if
                              proposal_diagnostics["relative_gap"] <= config.acceptable_relative_gap
                              else "finite_grid_primal_dual_gap_unresolved")
                else:
                    status = "equal_pair_fallback_stronger_lower_bound"
        else:
            status = "equal_pair_fallback_nonfinite_solver"
    if theta[2] < 0:
        weights = weights[reflection]
    diagnostics = _diagnostics(theta, cosine, sine, weights, candidate)
    result = {"weights": weights.tolist(),
              "support": [{"arm": int(i), "weight": float(weights[i])}
                          for i in np.flatnonzero(weights > 0)],
              "null_candidate": candidate.tolist(), "fit_theta": theta.tolist(),
              "solver_status": status, "epigraph_success": solver_success,
              "epigraph_message": epigraph_message, "epigraph_iterations": iterations,
              "lp_success": lp_success, "lp_message": lp_message,
              "warnings": sorted(set(warnings_seen)), "certified": False,
              "finite_grid_size": len(grid), "config": config_dict(config),
              "scope": "Fit-only finite-grid floating-point design; no decision certificate",
              "elapsed_seconds": time.perf_counter()-started}
    result.update(diagnostics)
    return result
