"""Outward numerical operating characteristics for stationary paired tests.

This analyzes inspection after EVERY physical pair (equivalently after every
discordance). It does not analyze the older sparse physical checkpoints.
Binary simulator probabilities are treated as exact rationals. All arithmetic
is enclosed using IEEE-754 nextafter, positive interval arithmetic, and a
conservative summation-error bound for cumulative sums. No scipy CDF accuracy
assumption is used. An arbitrary frozen symmetric integer boundary is allowed.
"""
from fractions import Fraction
import math

import numpy as np

EPS = np.finfo(float).eps
TINY = float(np.nextafter(0., 1.))


def down(x):
    return np.maximum(0., np.nextafter(x, -np.inf))


def up(x):
    return np.nextafter(x, np.inf)


def rational(value):
    if isinstance(value, bool):
        raise ValueError('A probability cannot be Boolean')
    result = value if isinstance(value, Fraction) else Fraction(value)
    if not 0 <= result <= 1:
        raise ValueError('Probability outside [0,1]')
    return result


def enclose_fraction(value):
    point = float(value)
    represented = Fraction(point)
    lower = float(np.nextafter(point, -np.inf)) if represented > value else point
    upper = float(np.nextafter(point, np.inf)) if represented < value else point
    return lower, upper


def tree_sum(values, upper):
    """Enclose a sum of nonnegative binary floats using explicit additions."""
    row = np.asarray(values, dtype=float).copy()
    if row.size == 0:
        return 0.
    operation = up if upper else down
    while row.size > 1:
        paired = row.size//2
        fresh = operation(row[:2*paired:2]+row[1:2*paired:2])
        if row.size % 2:
            fresh = np.concatenate((fresh, row[-1:]))
        row = fresh
    return float(row[0])


def prefix_enclosure(values, upper):
    """Conservative rounding envelope, including gradual underflow.

    Any prefix uses at most n-1 additions of nonnegative numbers. With
    n*eps<1/16, 4*n*eps dominates the usual gamma_(n-1) relative error;
    n times the least subnormal covers absolute underflow errors. The final
    arithmetic is itself rounded outward. This also covers alternate sum
    associations with no more than n additions on a dependency path.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n*EPS >= 1/16:
        raise ValueError('Cumulative-sum rounding model horizon exceeded')
    calculated = np.cumsum(values)
    relative = 4*n*EPS
    absolute = float(up(n*TINY))
    if upper:
        return up(up(calculated/(1-relative))+absolute)
    return down(down(calculated/(1+relative))-absolute)


def binomial_intervals(n, probability):
    """Mode-centered rational-ratio construction; returns pmf/head/tail bounds."""
    if type(n) is not int or n < 0:
        raise ValueError('Nonnegative integer binomial count required')
    p = rational(probability)
    if p in (0, 1):
        masses = np.zeros(n+1)
        masses[0 if p == 0 else n] = 1.
        heads = np.cumsum(masses)
        tails = np.cumsum(masses[::-1])[::-1]
        return masses.copy(), masses.copy(), heads.copy(), heads.copy(), tails.copy(), tails.copy()
    mode = min(n, ((n+1)*p.numerator)//p.denominator)
    lo, hi = np.zeros(n+1), np.zeros(n+1)
    lo[mode] = hi[mode] = 1.
    odds = p/(1-p)
    for k in range(mode, n):
        a, b = enclose_fraction(Fraction(n-k, k+1)*odds)
        lo[k+1], hi[k+1] = down(lo[k]*a), up(hi[k]*b)
    for k in range(mode, 0, -1):
        a, b = enclose_fraction(Fraction(k, n-k+1)/odds)
        lo[k-1], hi[k-1] = down(lo[k]*a), up(hi[k]*b)
    # The mode's unnormalized weight is exactly one. The true total is
    # therefore >=1 even when downward additions to that unit weight lose
    # smaller terms (and rounding enclosure steps push the bound below 1).
    total_lo, total_hi = max(1., tree_sum(lo, False)), tree_sum(hi, True)
    if not 1 <= total_lo <= total_hi:
        raise ArithmeticError('Mode normalization lost its unit reference')
    lo, hi = down(lo/total_hi), up(hi/total_lo)
    head_lo = np.minimum(1., prefix_enclosure(lo, False))
    head_hi = np.minimum(1., prefix_enclosure(hi, True))
    tail_lo = np.minimum(1., prefix_enclosure(lo[::-1], False)[::-1])
    tail_hi = np.minimum(1., prefix_enclosure(hi[::-1], True)[::-1])
    return lo, hi, head_lo, head_hi, tail_lo, tail_hi


def validate_table(table, pairs):
    if (type(pairs) is not int or pairs < 1 or len(table) <= pairs
            or type(table[0]) is not int or table[0] != 1):
        raise ValueError('Boundary must cover the entire physical-pair cap')
    for d, cutoff in enumerate(table[1:pairs+1], 1):
        if type(cutoff) is not int or not d//2 < cutoff <= d+1:
            raise ValueError('Invalid symmetric tail boundary')


def sign_first_passage_intervals(table, probability, pairs):
    """Biased-sign recursion, keeping only the unabsorbed contiguous states."""
    validate_table(table, pairs)
    q = rational(probability)
    qlo, qhi = enclose_fraction(q)
    clo, chi = enclose_fraction(1-q)
    lo, hi, origin = np.ones(1), np.ones(1), 0
    plus_lo, plus_hi = np.zeros(pairs+1), np.zeros(pairs+1)
    minus_lo, minus_hi = np.zeros(pairs+1), np.zeros(pairs+1)
    for d in range(1, pairs+1):
        fresh_lo, fresh_hi = np.zeros(len(lo)+1), np.zeros(len(hi)+1)
        fresh_lo[:-1], fresh_hi[:-1] = down(lo*clo), up(hi*chi)
        fresh_lo[1:] = down(fresh_lo[1:]+down(lo*qlo))
        fresh_hi[1:] = up(fresh_hi[1:]+up(hi*qhi))
        cutoff = table[d]
        indexes = np.arange(origin, origin+len(fresh_lo))
        positive, negative = indexes >= cutoff, indexes <= d-cutoff
        plus_lo[d], plus_hi[d] = tree_sum(fresh_lo[positive], False), tree_sum(fresh_hi[positive], True)
        minus_lo[d], minus_hi[d] = tree_sum(fresh_lo[negative], False), tree_sum(fresh_hi[negative], True)
        surviving = ~positive & ~negative
        if not np.any(surviving):
            lo, hi = np.empty(0), np.empty(0)
            break
        origin = int(indexes[surviving][0])
        lo, hi = fresh_lo[surviving], fresh_hi[surviving]
    return {'plus': (plus_lo, plus_hi), 'minus': (minus_lo, minus_hi),
            'survival': (tree_sum(lo, False), min(1., tree_sum(hi, True)))}


def dot_interval(a, b):
    return tree_sum(down(a[0]*b[0]), False), tree_sum(up(a[1]*b[1]), True)


def operating_intervals(table, pminus, pplus, physical_pair_cap=None):
    """Enclose exact model probabilities and expected total physical shots."""
    pairs = len(table)-1 if physical_pair_cap is None else physical_pair_cap
    validate_table(table, pairs)
    pm, pp = rational(pminus), rational(pplus)
    r = pp*(1-pm)+pm*(1-pp)
    if r == 0:
        return {'plus': [0., 0.], 'minus': [0., 0.], 'abstain': [1., 1.],
                'expected_shots': [float(2*pairs), float(2*pairs)], 'pair_cap': pairs,
                'protocol': 'Every physical pair inspected; stationary independent Bernoulli arms'}
    q = pp*(1-pm)/r
    passage = sign_first_passage_intervals(table, q, pairs)
    unused, unused2, head_lo, head_hi, tail_lo, tail_hi = binomial_intervals(pairs, r)
    unused, unused2, unused3, unused4, next_tail_lo, next_tail_hi = binomial_intervals(pairs+1, r)
    d = np.arange(1, pairs+1, dtype=float)
    rlo, rhi = enclose_fraction(r)
    # Divide the tiny binomial tail before multiplying by d: d/r can
    # overflow even for legitimate binary64 probabilities. If a positive
    # rational r rounds down to zero, the known cap remains a safe upper
    # bound for the whole truncated waiting time.
    waiting_lo = down(down(down(next_tail_lo[2:]/rhi)*d)+down(pairs*head_lo[:-1]))
    if rlo == 0:
        waiting_hi = np.full(pairs, float(pairs))
    else:
        with np.errstate(over='ignore'):
            first_hi = np.minimum(float(pairs), up(up(next_tail_hi[2:]/rlo)*d))
        waiting_hi = up(first_hi+up(pairs*head_hi[:-1]))
    waiting_hi = np.minimum(float(pairs), waiting_hi)
    plus = dot_interval((passage['plus'][0][1:], passage['plus'][1][1:]), (tail_lo[1:], tail_hi[1:]))
    minus = dot_interval((passage['minus'][0][1:], passage['minus'][1][1:]), (tail_lo[1:], tail_hi[1:]))
    mass_lo = down(passage['plus'][0][1:]+passage['minus'][0][1:])
    mass_hi = up(passage['plus'][1][1:]+passage['minus'][1][1:])
    spent_lo, spent_hi = dot_interval((mass_lo, mass_hi), (waiting_lo, waiting_hi))
    spent_lo = float(down(spent_lo+down(pairs*passage['survival'][0])))
    spent_hi = min(float(pairs), float(up(spent_hi+up(pairs*passage['survival'][1]))))
    abstain = [float(down(down(1-plus[1])-minus[1])), min(1., float(up(up(1-plus[0])-minus[0])))]
    return {'plus': [max(0., plus[0]), min(1., plus[1])],
            'minus': [max(0., minus[0]), min(1., minus[1])], 'abstain': abstain,
            'expected_shots': [float(down(2*spent_lo)), float(up(2*spent_hi))],
            'pair_cap': pairs, 'exact_binary_model_probabilities': [str(pm), str(pp)],
            'discordance_rate_interval': list(enclose_fraction(r)), 'sign_probability_interval': list(enclose_fraction(q)),
            'protocol': 'Every physical pair inspected; stationary independent Bernoulli arms',
            'arithmetic': 'Outward IEEE-754 intervals; rational mode-centered binomial ratios; explicit interval sign recursion; conservative cumulative-sum error envelope'}
