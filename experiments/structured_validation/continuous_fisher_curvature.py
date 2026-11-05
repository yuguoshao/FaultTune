"""One-node, whole-line lower bounds for the fixed continuous Fisher mixture.

For each balanced block, T=minus+plus. The identity
 C(m,k) C(m,T-k) = m!**2/[T!(2m-T)!] C(T,k) C(2m-T,m-k)
represents the same tilted K distribution as a weighted subset of fixed size m
from T positions of weight omega and 2m-T positions of weight 1. Such subset
indicators have nonpositive pair covariance: after removing i,j it has the
sign of e_{r-2} e_r-e_{r-1}**2, nonpositive by Newton's inequalities.
Applying this to K and m-K gives Var(K)<=min(T,2m-T)/4. The original m-position
representation also gives Var(K)<=m/4.

Put S=sum min(T,2m-T), n=sum m. For the fixed Jeffreys log integrand g,
 g'' >= -(S+1)/4 >= -(n+1)/4.
The global quadratic lower support at ANY finite lambda gives
 M >= exp(g(lambda))*sqrt(8*pi/(S+1))*exp(2*g'(lambda)**2/(S+1)).
Dropping the last factor yields a computable lower bound from ONE rational
odds node. The integral is over the entire real line, with no tail cutoff.
The weaker n bound implies, for each fixed expert L(lambda),
 log M >= log L(lambda) - .5 log(n+1) - log cosh(lambda/2) - .5 log(pi/2).
This is a deterministic capital inequality, not a sample-cost or SOTA result.

Node choice and the observed S only affect evaluation of a pathwise bound on
the same fixed prior. The signed 1/delta validity assumptions and limitations
are those of continuous_fisher. The default decision uses the stronger S bound.
"""
from decimal import Decimal
from fractions import Fraction
import hashlib
import numbers
from pathlib import Path

import continuous_fisher as continuous


Bounds = continuous.Bounds
ROOT = Path(__file__).resolve().parents[2]
VERSION = "continuous_jeffreys_balanced_fisher_single_node_curvature_v1"


def _node(value):
    if isinstance(value, Fraction):
        result = value
    else:
        pair = ((value["numerator"], value["denominator"])
                if isinstance(value, dict) else tuple(value))
        if len(pair) != 2 or any(isinstance(x, bool) or not isinstance(x, numbers.Integral)
                                 or x <= 0 for x in pair):
            raise ValueError("Node must contain two positive integers")
        result = Fraction(int(pair[0]), int(pair[1]))
    if result <= 0:
        raise ValueError("Odds must be positive")
    return result


def certificate_from_blocks(blocks, node=(1, 1), delta=.01):
    if isinstance(delta, bool):
        raise ValueError("delta must lie strictly between zero and one")
    risk = Decimal(str(delta))
    if not risk.is_finite() or not 0 < risk < 1:
        raise ValueError("delta must lie strictly between zero and one")
    blocks = [continuous.fisher._block(b["m"], b["minus_successes"], b["plus_successes"])
              for b in blocks]
    odds = _node(node)
    n = sum(b["m"] for b in blocks)
    s = sum(min(b["minus_successes"] + b["plus_successes"],
                2 * b["m"] - b["minus_successes"] - b["plus_successes"])
            for b in blocks)
    assert 0 <= s <= n
    score = sum(b["plus_successes"] - b["minus_successes"] for b in blocks)
    pi = continuous.pi_bounds()
    eta, g, retained = continuous._endpoint(blocks, score, odds, pi.log())
    n_bound = g + (Bounds(8) * pi / (n + 1)).log() / 2
    s_bound = g + (Bounds(8) * pi / (s + 1)).log() / 2
    threshold = -Bounds(risk).log()
    crossing = score != 0 and s_bound.lo >= threshold.hi
    return {
        "method": VERSION,
        "decision": ("reject" if score > 0 else "accept") if crossing else "abstain",
        "delta": str(risk), "shots": 2 * n, "blocks": len(blocks),
        "score": score, "score_definition": "sum(plus-minus)=2*U",
        "node": [odds.numerator, odds.denominator],
        "node_log_odds_bounds": [str(eta.lo), str(eta.hi)],
        "log_integrand_bounds": [str(g.lo), str(g.hi)],
        "n": n, "S": s, "decision_bound": "S",
        "log_evidence_lower_bound": str(s_bound.lo),
        "single_node_S_log_bound_enclosure": [str(s_bound.lo), str(s_bound.hi)],
        "single_node_n_log_bound_enclosure": [str(n_bound.lo), str(n_bound.hi)],
        "threshold_log_bounds": [str(threshold.lo), str(threshold.hi)],
        "retained_cgf_terms": retained,
        "whole_line_integration": True,
        "fixed_prior": "rho(lambda)=1/(2*pi*cosh(lambda/2)) on the entire real line",
        "anytime_valid_under_assumptions": True,
        "assumptions": "predictable balanced independent Bernoulli blocks with a common ordering",
        "limits": ["Bound enclosures are not upper bounds on the complete mixture.",
                   "The derivative-square improvement is deliberately omitted.",
                   "A finite fixed node need not have positive growth at every gap.",
                   "This is not empirical stopping-cost or SOTA superiority."]}


def source_hashes():
    result = continuous.source_hashes()
    path = Path(__file__).resolve()
    result[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result
