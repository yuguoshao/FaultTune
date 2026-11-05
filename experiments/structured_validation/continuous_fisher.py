"""Certified lower bound for the FIXED continuous Jeffreys Fisher mixture.

For predictable balanced Bernoulli blocks let U=sum(plus-minus)/2 and let
K_b(lambda) be the central conditional hypergeometric CGF. The complete mixture
is M = integral exp(lambda*U-sum K_b(lambda))*rho(lambda) d lambda, where
rho(lambda)=exp(lambda/2)/(pi*(1+exp(lambda))). This is a normalized, fixed prior:
under t=logistic(lambda), rho(lambda)d lambda=dt/(pi*sqrt(t*(1-t))).

g(lambda)=lambda*U-sum K_b(lambda)+log rho(lambda) is concave, since every K_b
is convex and (log rho)''=-1/(4*cosh(lambda/2)**2). If L_a,L_b are certified
lower bounds on g at two EXACT log-rational-odds nodes, concavity puts their
entire affine chord below g. Its exponential integral is
  (lambda_b-lambda_a)*(exp(L_b)-exp(L_a))/(L_b-L_a),
with the continuous constant-endpoint extension. We use a lower bound on the
positive node separation, retain only covered intervals, and discard all
outside positive mass. Thus the returned quantity is a lower bound on the
same complete M for ANY finite data-dependent node choice. Nodes are numerical
integration choices, not fitted prior weights or restarted evidence capital.

When the true ordering is positive and U<0, all positive-lambda likelihood
ratios are <=1 (K_b>=0). Consequently M<=1/2+integral_{lambda<0}rho*L. The latter
is an initial-one nonnegative supermartingale: opposite-sign experts have
conditional mean <=1 by monotone likelihood ratio and the tower property.
Ville therefore gives a wrong-sign risk <=delta at M>=1/delta, without a
directional split. The zero-ordering case uses the whole mixture. A numerical
lower bound can only delay a crossing. Validity requires predictable balanced
independent Bernoulli blocks sharing an ordering; it does not require odds to
be constant across blocks. It does not assert power under arbitrary adaptive
vanishing gaps, nor uniform power for any fixed finite integration grid.

No new simulator, stopping schedule, or experimental comparison is provided.
"""
from decimal import Decimal
from fractions import Fraction
from functools import lru_cache
import hashlib
import numbers
from pathlib import Path

import conditional_fisher as fisher


ROOT = Path(__file__).resolve().parents[2]
Bounds = fisher._Bounds
VERSION = "continuous_jeffreys_balanced_fisher_chord_v1"
SERIES_TOLERANCE = Decimal("1e-58")


def _ratio_bound(value):
    return Bounds(value.numerator) / value.denominator


def _atan_fraction_bounds(denominator, terms):
    """Alternating-series brackets for atan(1/q), computed as exact rationals."""
    total = sum((Fraction((-1) ** k, (2 * k + 1) * denominator ** (2 * k + 1))
                 for k in range(terms)), Fraction(0))
    next_term = Fraction((-1) ** terms,
                         (2 * terms + 1) * denominator ** (2 * terms + 1))
    other = total + next_term
    return min(total, other), max(total, other)


@lru_cache(maxsize=1)
def pi_bounds():
    """Machin pi=16 atan(1/5)-4 atan(1/239), with rational tail brackets.

The identity follows from tangent addition and the angles' positive quadrant;
alternating terms decrease in magnitude. Directed conversion preserves both
exact rational endpoints. The endpoint decimals are not an assumed pi table.
"""
    a, b = _atan_fraction_bounds(5, 64)
    c, d = _atan_fraction_bounds(239, 16)
    lower, upper = 16 * a - 4 * d, 16 * b - 4 * c
    return Bounds(_ratio_bound(lower).lo, _ratio_bound(upper).hi)


def rational_nodes(nodes):
    """Normalize and EXACTLY sort positive rational odds; duplicates are errors."""
    result = []
    for node in nodes:
        if isinstance(node, Fraction):
            value = node
        else:
            if isinstance(node, dict):
                pair = (node["numerator"], node["denominator"])
            else:
                pair = tuple(node)
            if len(pair) != 2 or any(isinstance(x, bool) or not isinstance(x, numbers.Integral)
                                     for x in pair):
                raise ValueError("Every node must contain two positive integers")
            if pair[0] <= 0 or pair[1] <= 0:
                raise ValueError("Odds numerator and denominator must be positive")
            value = Fraction(int(pair[0]), int(pair[1]))
        if value <= 0:
            raise ValueError("Odds must be positive")
        result.append(value)
    if len(result) < 2 or len(set(result)) != len(result):
        raise ValueError("At least two distinct rational odds nodes are required")
    return tuple(sorted(result))


def geometric_nodes(step=(5, 4), radius=64):
    """Deterministic convenience grid; its finite range is not a power claim."""
    if isinstance(radius, bool) or not isinstance(radius, numbers.Integral) or radius < 1:
        raise ValueError("radius must be a positive integer")
    ratio = rational_nodes(((1, 1), step))
    if ratio[0] != 1:
        raise ValueError("step must exceed one")
    step = ratio[1]
    return tuple(step ** k for k in range(-int(radius), int(radius) + 1))


def _exponential_chord_mean(lower_a, lower_b, offset):
    """Outward enclosure of integral_0^1 exp((1-t)*a+t*b-offset) dt."""
    if lower_a == lower_b:
        return (Bounds(lower_a) - offset).exp(), "constant"
    low, high = min(lower_a, lower_b), max(lower_a, lower_b)
    difference = Bounds(high) - low
    if difference.lo <= 0:
        raise ArithmeticError("Endpoint separation is unresolved at working precision")
    if difference.hi > 1:
        value = (Bounds(high) - offset).exp() * (1 - (-difference).exp()) / difference
        return value, "large_difference_scaled"
    # exprel(d)=sum_{k>=0}d**k/(k+1)! has positive terms, avoiding cancellation
    # near d=0. The first omitted term and decreasing term ratios give an upper
    # geometric tail; a lower sum alone would also be safe but less informative.
    term, accumulated = Bounds(1), Bounds(1)
    for k in range(96):
        next_term = term * difference / (k + 2)
        tail = next_term / (1 - difference / (k + 3))
        if tail.hi <= SERIES_TOLERANCE:
            enclosed = Bounds(accumulated.lo, (accumulated + tail).hi)
            return (Bounds(low) - offset).exp() * enclosed, "positive_exprel_series"
        accumulated += next_term
        term = next_term
    raise ArithmeticError("Exponential chord series did not reach its bound")


def _endpoint(block_counts, score, odds, log_pi):
    eta = _ratio_bound(odds).log()
    log_density = eta / 2 - log_pi - _ratio_bound(1 + odds).log()
    cumulant = Bounds(0)
    retained = 0
    for block in block_counts:
        total = block["minus_successes"] + block["plus_successes"]
        value, detail = fisher.centered_cgf_bounds(
            block["m"], total, odds.numerator, odds.denominator)
        cumulant += value
        retained += detail["retained_terms"]
    g = (Bounds(score) / 2) * eta - cumulant + log_density
    return eta, g, retained


def certificate_from_blocks(blocks, nodes=None, delta=.01, diagnostics=False):
    """Certify a signed crossing using only a lower bound on the full mixture.

The two reported quadrature log endpoints enclose the COMPUTED LOWER BOUND,
not the complete mixture. In particular their upper endpoint is not an upper
bound on M. No truncation or integration error is added to the risk budget.
"""
    if isinstance(delta, bool):
        raise ValueError("delta must lie strictly between zero and one")
    risk = Decimal(str(delta))
    if not risk.is_finite() or not 0 < risk < 1:
        raise ValueError("delta must lie strictly between zero and one")
    blocks = [fisher._block(b["m"], b["minus_successes"], b["plus_successes"])
              for b in blocks]
    odds_nodes = rational_nodes(geometric_nodes() if nodes is None else nodes)
    score = sum(b["plus_successes"] - b["minus_successes"] for b in blocks)
    log_pi = pi_bounds().log()
    endpoints = [_endpoint(blocks, score, odds, log_pi) for odds in odds_nodes]
    offset = max(endpoint[1].lo for endpoint in endpoints)
    accumulated = Bounds(0)
    segments = []
    branches = {}
    for index, ((left_eta, left_g, unused), (right_eta, right_g, unused2)) in enumerate(
            zip(endpoints[:-1], endpoints[1:])):
        separation = right_eta - left_eta
        if separation.lo <= 0:
            raise ArithmeticError("Log-odds nodes are not strictly separated at working precision")
        mean, branch = _exponential_chord_mean(left_g.lo, right_g.lo, offset)
        area = Bounds(separation.lo) * mean
        accumulated += area
        branches[branch] = branches.get(branch, 0) + 1
        if diagnostics:
            segments.append({"left_index": index, "right_index": index + 1,
                             "log_odds_separation_lower": str(separation.lo),
                             "scaled_lower_quadrature_area_bounds": [str(area.lo), str(area.hi)],
                             "evaluation": branch})
    if accumulated.lo <= 0:
        raise ArithmeticError("Quadrature lower sum lost positivity at working precision")
    evidence = Bounds(offset) + accumulated.log()
    threshold = -Bounds(risk).log()
    crossing = score != 0 and evidence.lo >= threshold.hi
    decision = ("reject" if score > 0 else "accept") if crossing else "abstain"
    result = {
        "method": VERSION, "decision": decision, "score": score,
        "score_definition": "sum(plus_successes-minus_successes)=2*U",
        "conditional_score": {"numerator": score, "denominator": 2},
        "shots": 2 * sum(b["m"] for b in blocks), "blocks": len(blocks), "delta": str(risk),
        "log_evidence_lower_bound": str(evidence.lo),
        "quadrature_lower_bound_log_bounds": [str(evidence.lo), str(evidence.hi)],
        "threshold_log_bounds": [str(threshold.lo), str(threshold.hi)],
        "fixed_prior": "rho(lambda)=exp(lambda/2)/(pi*(1+exp(lambda))); entire real line; normalized",
        "nodes": [[odds.numerator, odds.denominator] for odds in odds_nodes],
        "node_count": len(odds_nodes),
        "log_odds_coverage_bounds": [[str(endpoints[0][0].lo), str(endpoints[0][0].hi)],
                                    [str(endpoints[-1][0].lo), str(endpoints[-1][0].hi)]],
        "pi_bounds": [str(pi_bounds().lo), str(pi_bounds().hi)],
        "scaled_quadrature_sum_bounds": [str(accumulated.lo), str(accumulated.hi)],
        "log_offset": str(offset), "chord_evaluation_branches": branches,
        "retained_cgf_terms": sum(endpoint[2] for endpoint in endpoints),
        "anytime_valid_under_assumptions": True,
        "numeric_status": "outward Decimal endpoints and exponential chord integrals; exact rational Machin pi brackets",
        "assumptions": "predictable balanced independent Bernoulli blocks with a common ordering",
        "limits": ["The integral outside the finite covered interval is discarded, never renormalized.",
                   "The upper quadrature endpoint is NOT an upper bound on the complete continuous mixture.",
                   "Finite nodes and finite coverage do not establish power at every nonzero gap.",
                   "Common ordering alone does not imply power for arbitrary adaptive vanishing gaps.",
                   "No Monte Carlo, new runner, or empirical superiority claim."]}
    if diagnostics:
        result["endpoint_log_integrand_bounds"] = [[str(e[1].lo), str(e[1].hi)] for e in endpoints]
        result["segments"] = segments
    return result


def source_hashes():
    paths = [Path(__file__).resolve(), Path(fisher.__file__).resolve(),
             ROOT / "experiments/circuit_validation/direct_tests.py"]
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}
