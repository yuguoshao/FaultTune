"""Direct old/new validation using only paid observations.

Pairs use a fresh iid public context from the fixed target distribution, with
independent old/new measurement noise conditional on that context.  Their four
counts are ordered ``[old0_new0, old0_new1, old1_new0, old1_new1]``.  Concordant
pairs do not carry sign evidence, but BOTH shots are always charged.

``certificate_from_pairs`` is an anytime-valid beta-mixture sign test.
``fixed_paired_certificate`` is a conditional exact McNemar test and may only
be used ONCE at a sample size fixed before validation.  Repeated uncorrected
looks at that certificate are invalid.  ``direct_kl_certificate`` inverts
separate binomial mixture confidence sequences; it also permits unequal arm
counts, provided each arm continues to sample the fixed target distribution.

No function in this module takes a simulator, a true failure probability, or a
context-selection policy.  The optional power-planning functions require an
explicit registered design distribution and are not validation certificates.
Only Jeffreys (prior=.5) and uniform (prior=1) mixtures are implemented.

Numerical certificate decisions use outward Decimal intervals.  Logarithms
and exponentials rely on Python Decimal's correctly rounded implementation;
large factorials use rigorous alternating Stirling remainder bounds.  This is
numerical enclosure arithmetic, not a claim that Python itself was formally
verified.  SciPy is needed only for optional distribution-specific planning.
"""

import math
import numbers
import struct
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from fractions import Fraction
from functools import lru_cache


_PRECISION = 64
_PI_LOW = Decimal("3.141592653589793238462643383279502884197169399375105820974944592307816406286")
_PI_HIGH = Decimal("3.141592653589793238462643383279502884197169399375105820974944592307816406287")


def _decimal(value):
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        return Decimal.from_float(value)
    return Decimal(int(value))


class _Bounds:
    """Small directed-rounding interval helper; endpoints are exact Decimals."""

    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi=None):
        self.lo = _decimal(lo)
        self.hi = self.lo if hi is None else _decimal(hi)
        if self.lo > self.hi:
            raise ValueError("Reversed numerical interval")

    @staticmethod
    def as_bounds(value):
        return value if isinstance(value, _Bounds) else _Bounds(value)

    def __add__(self, other):
        other = self.as_bounds(other)
        with localcontext() as ctx:
            ctx.prec, ctx.rounding = _PRECISION, ROUND_FLOOR
            lo = self.lo + other.lo
            ctx.rounding = ROUND_CEILING
            hi = self.hi + other.hi
        return _Bounds(lo, hi)

    __radd__ = __add__

    def __neg__(self):
        # copy_negate is exact and does not use the ambient Decimal precision.
        return _Bounds(self.hi.copy_negate(), self.lo.copy_negate())

    def __sub__(self, other):
        return self + -self.as_bounds(other)

    def __rsub__(self, other):
        return self.as_bounds(other) + -self

    def __mul__(self, other):
        other = self.as_bounds(other)
        with localcontext() as ctx:
            ctx.prec, ctx.rounding = _PRECISION, ROUND_FLOOR
            lo = min(a * b for a in (self.lo, self.hi) for b in (other.lo, other.hi))
            ctx.rounding = ROUND_CEILING
            hi = max(a * b for a in (self.lo, self.hi) for b in (other.lo, other.hi))
        return _Bounds(lo, hi)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self.as_bounds(other)
        if other.lo <= 0 <= other.hi:
            raise ZeroDivisionError("Interval denominator includes zero")
        with localcontext() as ctx:
            ctx.prec, ctx.rounding = _PRECISION, ROUND_FLOOR
            lo = Decimal(1) / other.hi
            ctx.rounding = ROUND_CEILING
            hi = Decimal(1) / other.lo
        return self * _Bounds(lo, hi)

    def log(self):
        if self.lo <= 0:
            raise ValueError("Log requires a strictly positive interval")
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            # Decimal.ln is correctly rounded with ROUND_HALF_EVEN, regardless
            # of context rounding. Adjacent representable values enclose it.
            lo = self.lo.ln().next_minus()
            hi = self.hi.ln().next_plus()
        return _Bounds(lo, hi)

    def exp(self):
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            lo = max(Decimal(0), self.lo.exp().next_minus())
            hi = self.hi.exp().next_plus()
        return _Bounds(lo, hi)

    def sqrt(self):
        if self.lo < 0:
            raise ValueError("Square root requires a nonnegative interval")
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            lo = max(Decimal(0), self.lo.sqrt().next_minus())
            hi = self.hi.sqrt().next_plus()
        return _Bounds(lo, hi)


def _next_float(value, upward):
    """One binary64 step, including Python 3.8 without math.nextafter."""
    value = float(value)
    if math.isnan(value) or value == (math.inf if upward else -math.inf):
        return value
    if value == 0:
        return struct.unpack(">d", struct.pack(">Q", 1 if upward else (1 << 63) | 1))[0]
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    bits += 1 if ((value > 0) == upward) else -1
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


def _float_down(value):
    result = float(value)
    return _next_float(result, False) if Decimal.from_float(result) > value else result


def _float_up(value):
    result = float(value)
    return _next_float(result, True) if Decimal.from_float(result) < value else result


def _as_float_bounds(value):
    return [_float_down(value.lo), _float_up(value.hi)]


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError(name + " must be a nonnegative integer")
    return int(value)


def _options(delta, prior):
    if isinstance(delta, bool) or not math.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("delta must lie strictly between zero and one")
    if isinstance(prior, bool) or prior not in (0.5, 1.0):
        raise ValueError("prior must be 0.5 (Jeffreys) or 1.0 (uniform)")
    # The ordinary decimal spelling is the registered risk, e.g. .01 exactly.
    return _Bounds(Decimal(str(delta))), float(prior)


def _pairs(counts):
    try:
        counts = tuple(counts)
    except TypeError:
        raise ValueError("counts must contain four nonnegative integers")
    if len(counts) != 4:
        raise ValueError("counts must have order [00, 01, 10, 11]")
    return tuple(_count(n, "pair count") for n in counts)


@lru_cache(maxsize=8192)
def _log_factorial(n):
    if n < 32:
        return _Bounds(math.factorial(n)).log() if n > 1 else _Bounds(0)
    x = _Bounds(n)
    main = (x + _Bounds(Decimal("0.5"))) * x.log() - x
    main += (_Bounds(_PI_LOW, _PI_HIGH) * 2).log() / 2
    # For positive x the log-Gamma Stirling remainder has the sign and at
    # most the magnitude of the next term. Even truncation is a lower bound.
    lower = main + _Bounds(1) / (12 * n)
    lower -= _Bounds(1) / (360 * n ** 3)
    lower += _Bounds(1) / (1260 * n ** 5)
    lower -= _Bounds(1) / (1680 * n ** 7)
    upper = lower + _Bounds(1) / (1188 * n ** 9)
    return _Bounds(lower.lo, upper.hi)


@lru_cache(maxsize=16384)
def _log_marginal(successes, trials, prior):
    failures = trials - successes
    if trials == 0:
        return _Bounds(0)
    if prior == 0.5:
        return (_log_factorial(2 * successes) + _log_factorial(2 * failures)
                - _log_factorial(successes) - _log_factorial(failures)
                - _log_factorial(trials) - _Bounds(2).log() * (2 * trials))
    return (_log_factorial(successes) + _log_factorial(failures)
            - _log_factorial(trials + 1))


def _log_mixture_at(successes, trials, p, prior):
    result = _log_marginal(successes, trials, prior)
    if successes:
        result -= _Bounds(p).log() * successes
    if trials - successes:
        result -= (1 - _Bounds(p)).log() * (trials - successes)
    return result


def certificate_from_pairs(counts, delta=0.01, prior=0.5):
    """Anytime-valid sign certificate under iid target-context pairing.

    Conditional discordant signs have probability pi, and sign(Delta) equals
    sign(pi-.5). At the true pi the beta-mixture likelihood ratio is a
    nonnegative supermartingale (a martingale for 0<pi<1). On a wrong-side crossing its value at true pi is
    at least its value at .5; Ville therefore controls each wrong direction
    by delta, and both directions jointly by delta when Delta=0.

    Adaptive selection of contexts violates the iid assumption here.
    """
    risk, prior = _options(delta, prior)
    counts = _pairs(counts)
    pairs, worse, better = sum(counts), counts[1], counts[2]
    discordant = worse + better
    evidence = _log_marginal(worse, discordant, prior) + _Bounds(2).log() * discordant
    threshold = -risk.log()
    decision = "abstain"
    if discordant and evidence.lo >= threshold.hi:
        decision = "accept" if worse < better else "reject" if worse > better else "abstain"
    return {
        "method": "paired_beta_mixture", "decision": decision,
        "est_delta": (worse - better) / pairs if pairs else 0.0,
        "shots": 2 * pairs, "pairs": pairs, "discordant_pairs": discordant,
        "counts": list(counts), "delta": float(delta), "prior": prior,
        "evidence": _float_down(evidence.lo), "threshold": _float_up(threshold.hi),
        "evidence_bounds": _as_float_bounds(evidence),
        "threshold_bounds": _as_float_bounds(threshold), "anytime_valid": True,
        "numeric_status": "decision uses outward Decimal log bounds",
    }


def _half_binomial_cdf(k, n):
    """Outward enclosure of P[Bin(n,.5)<=k], with rigorous residual tail."""
    if k < 0:
        return _Bounds(0)
    if k >= n:
        return _Bounds(1)
    if 2 * k >= n:
        return 1 - _half_binomial_cdf(n - k - 1, n)
    log_pmf = (_log_factorial(n) - _log_factorial(k)
               - _log_factorial(n - k) - _Bounds(2).log() * n)
    term, total, index = _Bounds(1), _Bounds(1), k
    remainder = _Bounds(0)
    for unused in range(100000):
        if index == 0:
            remainder = _Bounds(0)
            break
        ratio = _Bounds(index) / (n - index + 1)
        remainder = term * ratio / (1 - ratio)
        if remainder.hi <= Decimal("1e-22") * total.lo:
            break
        term = term * ratio
        total += term
        index -= 1
    # Even a work-limit exit retains a rigorous geometric remainder bound.
    upper = total + remainder
    mass = log_pmf.exp() * _Bounds(total.lo, upper.hi)
    return _Bounds(max(Decimal(0), mass.lo), min(Decimal(1), mass.hi))


def fixed_paired_certificate(counts, delta=0.01):
    """Exact conditional McNemar certificate for ONE registered final look.

    Each tail gets delta/2. Numerical ambiguity causes abstention. For small
    samples exact integer comparison also resolves p-value boundary equality.
    The sample size may not be chosen or extended using these validation data.
    """
    risk, unused = _options(delta, 0.5)
    counts = _pairs(counts)
    pairs, worse, better = sum(counts), counts[1], counts[2]
    discordant = worse + better
    p_accept = _half_binomial_cdf(worse, discordant)
    p_reject = _half_binomial_cdf(better, discordant)
    tail_risk = risk / 2
    accept = discordant > 0 and p_accept.hi <= tail_risk.lo
    reject = discordant > 0 and p_reject.hi <= tail_risk.lo
    if 0 < discordant <= 4096:
        exact_alpha = Fraction(str(delta)) / 2
        denominator = 1 << discordant
        if not accept and p_accept.lo <= tail_risk.hi and worse < better:
            numerator = sum(math.comb(discordant, j) for j in range(worse + 1))
            accept = Fraction(numerator, denominator) <= exact_alpha
        if not reject and p_reject.lo <= tail_risk.hi and better < worse:
            numerator = sum(math.comb(discordant, j) for j in range(better + 1))
            reject = Fraction(numerator, denominator) <= exact_alpha
    p_small = p_accept if worse < better else p_reject
    if p_small.lo > 0:
        evidence = -p_small.log()
    else:
        # At extreme sample counts exp(log_p) may reach Decimal's exponent
        # floor. Its positive outward upper bound still certifies useful
        # lower log evidence; do not replace that evidence with zero.
        evidence = _Bounds((-_Bounds(p_small.hi).log()).lo, Decimal("Infinity"))
    threshold = -tail_risk.log()
    return {
        "method": "fixed_paired_mcnemar", "decision": "accept" if accept else "reject" if reject else "abstain",
        "est_delta": (worse - better) / pairs if pairs else 0.0,
        "shots": 2 * pairs, "pairs": pairs, "discordant_pairs": discordant,
        "counts": list(counts), "delta": float(delta), "tail_delta": float(delta) / 2,
        "p_accept": float((p_accept.lo + p_accept.hi) / 2),
        "p_reject": float((p_reject.lo + p_reject.hi) / 2),
        "p_accept_bounds": _as_float_bounds(p_accept), "p_reject_bounds": _as_float_bounds(p_reject),
        "evidence": _float_down(evidence.lo), "threshold": _float_up(threshold.hi),
        "anytime_valid": False, "required_use": "one predeclared fixed final pair count",
        "numeric_status": "outward Decimal tail bounds; exact integer boundary check for K<=4096",
    }


def _bernoulli_kl(p, q):
    if p == 0:
        return -math.log1p(-q)
    if p == 1:
        return -math.log(q)
    return max(0.0, p * math.log1p((p - q) / q)
               + (1 - p) * math.log1p((q - p) / (1 - q)))


def binomial_mixture_interval(successes, trials, delta=0.01, prior=0.5):
    """Time-uniform binomial interval; both endpoints round conservatively.

    Fast binary64 root estimates are only proposals. Each nontrivial returned
    endpoint is independently checked to lie OUTSIDE the mixture sublevel set
    using outward Decimal bounds. Failed checks expand toward 0 or 1.
    """
    risk, prior = _options(delta, prior)
    successes, trials = _count(successes, "successes"), _count(trials, "trials")
    if successes > trials:
        raise ValueError("successes cannot exceed trials")
    if trials == 0:
        return [0.0, 1.0]
    threshold = -risk.log()
    phat = successes / trials
    marginal = _log_marginal(successes, trials, prior)
    mle = _Bounds(0)
    if 0 < successes < trials:
        exact_phat = _Bounds(successes) / trials
        mle = exact_phat.log() * successes + (1 - exact_phat).log() * (trials - successes)
    complexity = mle - marginal
    level = (float(threshold.hi) + float(complexity.hi)) / trials
    result = [0.0, 1.0]
    for side in (0, 1):
        if (side == 0 and successes == 0) or (side == 1 and successes == trials):
            continue
        lo, hi = (0.0, phat) if side == 0 else (phat, 1.0)
        for unused in range(60):
            mid = lo + (hi - lo) / 2
            if mid == lo or mid == hi:
                break
            outside = _bernoulli_kl(phat, mid) > level
            if outside == (side == 0):
                lo = mid
            else:
                hi = mid
        proposed = lo if side == 0 else hi
        padding = max(1e-14, 1e-8 / trials)
        for unused in range(64):
            endpoint = max(0.0, proposed - padding) if side == 0 else min(1.0, proposed + padding)
            if endpoint in (0.0, 1.0):
                result[side] = endpoint
                break
            value = _log_mixture_at(successes, trials, Decimal.from_float(endpoint), prior)
            # Also enforce the correct side of the exact MLE with integer
            # arithmetic; this does not rely on rounded phat.
            numerator, denominator = endpoint.as_integer_ratio()
            correct_side = numerator * trials < successes * denominator if side == 0 else numerator * trials > successes * denominator
            if correct_side and value.lo >= threshold.hi:
                result[side] = endpoint
                break
            padding *= 2
        # On any failure the initialized endpoint 0/1 is still valid.
    return result


def direct_kl_certificate(old_failures, old_shots, new_failures, new_shots, delta=0.01, prior=0.5):
    """Separate unknown-old/new binomial mixture CS, then subtract intervals."""
    risk, unused = _options(delta, prior)
    old_failures, old_shots = _count(old_failures, "old_failures"), _count(old_shots, "old_shots")
    new_failures, new_shots = _count(new_failures, "new_failures"), _count(new_shots, "new_shots")
    # Retain the registered decimal risk under splitting; do not round each
    # arm's alpha upward through a binary64 division.
    arm_delta = (risk / 2).lo
    old_interval = binomial_mixture_interval(old_failures, old_shots, arm_delta, prior)
    new_interval = binomial_mixture_interval(new_failures, new_shots, arm_delta, prior)
    difference = _Bounds(new_interval[0], new_interval[1]) - _Bounds(old_interval[0], old_interval[1])
    decision = "accept" if difference.hi < 0 else "reject" if difference.lo > 0 else "abstain"
    return {
        "method": "direct_separate_binomial_mixture", "decision": decision,
        "est_delta": new_failures / new_shots - old_failures / old_shots if old_shots and new_shots else None,
        "shots": old_shots + new_shots, "old_shots": old_shots, "new_shots": new_shots,
        "old_failures": old_failures, "new_failures": new_failures,
        "old_interval": old_interval, "new_interval": new_interval,
        "delta_interval": _as_float_bounds(difference), "delta": float(delta), "prior": float(prior),
        "evidence": None, "threshold": None, "anytime_valid": True,
        "numeric_status": "each interval endpoint independently checked with outward Decimal bounds",
    }


def hoeffding_pair_cap(epsilon, delta=0.01, beta=0.05):
    """Conservative sufficient FIXED pair count for McNemar power >=1-beta.

    This follows from D in [-1,1] and is valid for every iid paired law with
    |Delta|>=epsilon. It is a universal upper bound, not a competitive sample
    design or a lower bound. The returned count is pairs, hence costs 2*n shots.
    """
    risk, unused = _options(delta, 0.5)
    if isinstance(epsilon, bool) or not math.isfinite(epsilon) or not 0 < epsilon <= 1:
        raise ValueError("epsilon must be in (0,1]")
    if isinstance(beta, bool) or not math.isfinite(beta) or not 0 < beta < 1:
        raise ValueError("beta must lie strictly between zero and one")
    epsilon_bound = _Bounds(Decimal(str(epsilon)))
    beta_bound = _Bounds(Decimal(str(beta)))
    root_sum = (_Bounds(2) / risk).log().sqrt() + (-beta_bound.log()).sqrt()
    value = 2 * root_sum * root_sum / (epsilon_bound * epsilon_bound)
    return int(value.hi.to_integral_value(rounding=ROUND_CEILING))


def fixed_paired_power(pairs, discordance, gap, delta=0.01, tail_tolerance=1e-12):
    """Numerical fixed-test acceptance power for Delta=-gap, discordance=d.

    K~Bin(pairs,d); conditional B~Bin(K,(d-gap)/(2*d)). Sum the exact McNemar
    rejection region over K. The omitted K tail is reported separately.
    Results use SciPy floating point and are NOT certified numerical bounds.
    Parameters are a registered design assumption, never simulator truth read
    by a validation policy. No claim that worst-case discordance is one.
    """
    pairs = _count(pairs, "pairs")
    _options(delta, 0.5)
    if not 0 <= discordance <= 1 or not 0 <= gap <= discordance:
        raise ValueError("require 0<=gap<=discordance<=1")
    if not 0 < tail_tolerance < 1:
        raise ValueError("tail_tolerance must lie in (0,1)")
    try:
        import numpy as np
        from scipy.stats import binom
    except ImportError as exc:
        raise RuntimeError("Power planning requires NumPy/SciPy; certificates use only stdlib") from exc
    if pairs == 0 or discordance == 0:
        power, omitted, first, last = 0.0, 0.0, 0, 0
    else:
        first = max(0, int(binom.ppf(tail_tolerance / 2, pairs, discordance)))
        last = min(pairs, int(binom.isf(tail_tolerance / 2, pairs, discordance)))
        k = np.arange(first, last + 1, dtype=np.int64)
        critical = binom.ppf(delta / 2, k, 0.5).astype(np.int64)
        critical -= binom.cdf(critical, k, 0.5) > delta / 2
        sign_probability = (discordance - gap) / (2 * discordance)
        power = float(np.sum(binom.pmf(k, pairs, discordance) * binom.cdf(critical, k, sign_probability)))
        omitted = float(binom.cdf(first - 1, pairs, discordance) + binom.sf(last, pairs, discordance))
    return {
        "pairs": pairs, "shots": 2 * pairs, "discordance": float(discordance),
        "gap": float(gap), "delta": float(delta), "power_estimate": power,
        "truncation_only_bounds": [power, min(1.0, power + omitted)],
        "omitted_k_probability": omitted, "k_range": [first, last],
        "numeric_status": "SciPy floating point; truncation bounds do not include rounding error; not machine-certified",
        "scope": "registered design distribution only; not uniform over unknown discordance",
    }


def plan_fixed_pairs(discordance, gap, delta=0.01, target_power=0.95, max_pairs=100000000):
    """Find and recheck a numerical fixed-sample design; not a minimal-n proof."""
    max_pairs = _count(max_pairs, "max_pairs")
    if not 0 < target_power < 1 or gap <= 0 or max_pairs == 0:
        raise ValueError("require positive gap/max_pairs and target_power in (0,1)")
    lo, hi = 0, min(32, max_pairs)
    row = fixed_paired_power(hi, discordance, gap, delta)
    while row["power_estimate"] < target_power and hi < max_pairs:
        lo, hi = hi, min(2 * hi, max_pairs)
        row = fixed_paired_power(hi, discordance, gap, delta)
    if row["power_estimate"] >= target_power:
        # Discrete power need not be monotone. This search only proposes a
        # budget whose own power is then rechecked; it does not prove minimality.
        while hi - lo > max(1, hi // 10000):
            mid = (lo + hi) // 2
            candidate = fixed_paired_power(mid, discordance, gap, delta)
            if candidate["power_estimate"] >= target_power:
                hi = mid
            else:
                lo = mid
        row = fixed_paired_power(hi, discordance, gap, delta)
    row.update(target_power=float(target_power), target_met=row["power_estimate"] >= target_power,
               design_status="numerically rechecked candidate; not proven minimal; not a universal power guarantee")
    return row
