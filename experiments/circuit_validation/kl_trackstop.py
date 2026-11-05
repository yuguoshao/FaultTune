"""Joint Bernoulli KL-GLR and a Track-and-Stop-style allocation baseline.

The estimand is sum(a[i] * p[i]); a negative value accepts the candidate.
Each update is paid, independent Bernoulli data from one fixed arm. An arm may
represent (old/new configuration, public circuit context). This module never
receives hidden probabilities, physical-model parameters, or an oracle.

Safety uses a product of Beta-mixture likelihood ratios, not a normal or
chi-squared approximation. The GLR is bounded BELOW by a Lagrange dual; its
threshold is bounded ABOVE. Decimal interval arithmetic audits both bounds.
The allocation is a numerical, batched specialization of the classical
information-allocation/Track-and-Stop framework, not a new algorithm and not
a claim of finite-sample optimality or a verbatim CombGame implementation.

References:
  Garivier--Kaufmann (2016), https://arxiv.org/abs/1602.04589
  Kaufmann--Koolen (2021), https://jmlr.org/papers/v22/18-798.html
  Jourdan et al. (2021), https://proceedings.mlr.press/v132/jourdan21a.html
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import (
    Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, localcontext,
)
from fractions import Fraction
from functools import lru_cache
import math
import struct
from typing import Iterable, Sequence


_PRECISION = 64
_INF = float("inf")


def _nextafter(value: float, toward: float) -> float:
    """One IEEE-754 binary64 step, also on the project's Python 3.8."""
    if math.isnan(value) or math.isnan(toward):
        return float("nan")
    if value == toward:
        return toward
    if value == 0:
        return math.copysign(float.fromhex("0x0.0000000000001p-1022"), toward)
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    bits += 1 if (value > 0) == (toward > value) else -1
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


@dataclass(frozen=True)
class _Interval:
    """Outward Decimal interval; ln is correctly rounded by decimal."""

    lo: Decimal
    hi: Decimal

    @staticmethod
    def exact(value: int | Decimal) -> "_Interval":
        value = Decimal(value)
        return _Interval(value, value)

    @staticmethod
    def rational(value: Fraction) -> "_Interval":
        return _Interval.exact(value.numerator) / _Interval.exact(value.denominator)

    def __add__(self, other: "_Interval") -> "_Interval":
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            ctx.rounding = ROUND_FLOOR
            lo = self.lo + other.lo
            ctx.rounding = ROUND_CEILING
            hi = self.hi + other.hi
        return _Interval(lo, hi)

    def __sub__(self, other: "_Interval") -> "_Interval":
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            ctx.rounding = ROUND_FLOOR
            lo = self.lo - other.hi
            ctx.rounding = ROUND_CEILING
            hi = self.hi - other.lo
        return _Interval(lo, hi)

    def __mul__(self, other: "_Interval") -> "_Interval":
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            ctx.rounding = ROUND_FLOOR
            lo = min(x * y for x in (self.lo, self.hi)
                     for y in (other.lo, other.hi))
            ctx.rounding = ROUND_CEILING
            hi = max(x * y for x in (self.lo, self.hi)
                     for y in (other.lo, other.hi))
        return _Interval(lo, hi)

    def __truediv__(self, other: "_Interval") -> "_Interval":
        if other.lo <= 0 <= other.hi:
            raise ZeroDivisionError("interval denominator contains zero")
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            ctx.rounding = ROUND_FLOOR
            lo = min(x / y for x in (self.lo, self.hi)
                     for y in (other.lo, other.hi))
            ctx.rounding = ROUND_CEILING
            hi = max(x / y for x in (self.lo, self.hi)
                     for y in (other.lo, other.hi))
        return _Interval(lo, hi)

    def ln(self) -> "_Interval":
        if self.lo <= 0:
            raise ValueError("log interval must be positive")
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            ctx.rounding = ROUND_HALF_EVEN
            # Decimal.ln is correctly rounded, independently of ctx.rounding.
            lo = self.lo.ln().next_minus(ctx)
            hi = self.hi.ln().next_plus(ctx)
        return _Interval(lo, hi)

    def float_lo(self) -> float:
        return _nextafter(float(self.lo), -_INF)

    def float_hi(self) -> float:
        return _nextafter(float(self.hi), _INF)


_ZERO = _Interval.exact(0)
_ONE = _Interval.exact(1)


def _sum_intervals(items: Iterable[_Interval]) -> _Interval:
    answer = _ZERO
    for item in items:
        answer = answer + item
    return answer


@lru_cache(maxsize=8192)
def _ln_integer(value: int) -> _Interval:
    if value <= 0:
        raise ValueError("positive integer required")
    return _Interval.exact(value).ln()


@lru_cache(maxsize=8192)
def _log_factorial(value: int) -> _Interval:
    if value < 0:
        raise ValueError("nonnegative factorial argument required")
    if value < 64:
        return _ln_integer(math.factorial(value))
    n = _Interval.exact(value)
    # Bounds for log(2*pi), obtained from the standard decimal expansion.
    # Keeping 61 digits is more than sufficient at the working precision.
    log_two_pi = _Interval(
        Decimal("1.8378770664093454835606594728112352797227949472755668256343030"),
        Decimal("1.8378770664093454835606594728112352797227949472755668256343031"),
    )
    half = _Interval.rational(Fraction(1, 2))
    base = (n + half) * n.ln() - n + half * log_two_pi
    base = base + _Interval.rational(Fraction(1, 12 * value))
    base = base - _Interval.rational(Fraction(1, 360 * value ** 3))
    base = base + _Interval.rational(Fraction(1, 1260 * value ** 5))
    base = base - _Interval.rational(Fraction(1, 1680 * value ** 7))
    # The positive-real Stirling remainder has the next term's sign and is
    # smaller than its magnitude. This is not an unbounded asymptotic guess.
    remainder = _Interval.rational(Fraction(1, 1188 * value ** 9))
    return _Interval(base.lo, (base + remainder).hi)


def _mle_log_likelihood(successes: int, trials: int) -> _Interval:
    if trials == 0:
        return _ZERO
    failures = trials - successes
    parts = [_Interval.exact(-trials) * _ln_integer(trials)]
    if successes:
        parts.append(_Interval.exact(successes) * _ln_integer(successes))
    if failures:
        parts.append(_Interval.exact(failures) * _ln_integer(failures))
    return _sum_intervals(parts)


def _mixture_log_likelihood(successes: int, trials: int, prior: float) -> _Interval:
    failures = trials - successes
    if prior == 0.5:
        # B(s+1/2,f+1/2)/B(1/2,1/2), without subtracting large lgammas.
        return (_log_factorial(2 * successes) + _log_factorial(2 * failures)
                - _log_factorial(successes) - _log_factorial(failures)
                - _log_factorial(trials)
                - _Interval.exact(2 * trials) * _ln_integer(2))
    if prior == 1.0:
        return (_log_factorial(successes) + _log_factorial(failures)
                - _log_factorial(trials + 1))
    raise ValueError("prior must be 0.5 (Jeffreys) or 1.0 (uniform)")


def _count_kl(successes: int, trials: int, q: Fraction) -> _Interval:
    if not 0 <= q <= 1:
        raise ValueError("probability outside [0,1]")
    failures = trials - successes
    if (q == 0 and successes) or (q == 1 and failures):
        raise ValueError("infinite KL at incompatible endpoint")
    value = _mle_log_likelihood(successes, trials)
    if successes:
        value = value - _Interval.exact(successes) * _Interval.rational(q).ln()
    if failures:
        value = value - _Interval.exact(failures) * _Interval.rational(1 - q).ln()
    return value


def bernoulli_kl(p: float, q: float) -> float:
    """Numerical KL for allocation/root finding; never alone certifies a stop."""
    if not 0 <= p <= 1 or not 0 <= q <= 1:
        raise ValueError("probabilities must be in [0,1]")
    if p == q:
        return 0.0
    if q == 0 or q == 1:
        return _INF
    if p == 0:
        return -math.log1p(-q)
    if p == 1:
        return -math.log(q)
    return max(0.0, math.fsum((
        p * math.log1p((p - q) / q),
        (1 - p) * math.log1p((q - p) / (1 - q)),
    )))


def _validated_inputs(successes, trials, coefficients):
    if not (len(successes) == len(trials) == len(coefficients)):
        raise ValueError("counts and coefficients must have equal lengths")
    counts = []
    for s, n in zip(successes, trials):
        if isinstance(s, bool) or isinstance(n, bool) or int(s) != s or int(n) != n:
            raise ValueError("counts must be integers")
        s, n = int(s), int(n)
        if not 0 <= s <= n:
            raise ValueError("require 0 <= successes <= trials")
        counts.append((s, n))
    if any(not math.isfinite(float(a)) for a in coefficients):
        raise ValueError("coefficients must be finite")
    # Decimal strings define the exact public coefficients for the audit.
    exact_a = [Fraction(str(float(a))) for a in coefficients]
    if not any(a > 0 for a in exact_a) or not any(a < 0 for a in exact_a):
        raise ValueError("a sign comparison requires positive and negative coefficients")
    return counts, exact_a


def _arm_minimizer(p: float, n: int, eta: float) -> float:
    """Argmin n*kl(p,q)+eta*q, including empirical p=0 and p=1."""
    if p == 0:
        q = 0.0 if eta >= -n else 1 + n / eta
    elif p == 1:
        q = 1.0 if eta <= n else n / eta
    else:
        k = abs(eta) / n
        p_work = p if eta >= 0 else 1 - p
        discriminant = math.hypot(1 - k, 2 * math.sqrt(k) * math.sqrt(1 - p_work))
        small = 2 * p_work / (1 + k + discriminant)
        q = small if eta >= 0 else 1 - small
    if p > 0 and q <= 0:
        q = _nextafter(0.0, 1.0)
    if p < 1 and q >= 1:
        q = _nextafter(1.0, 0.0)
    return q


def _glr_root(counts, coefficients):
    ps = [s / n for s, n in counts]

    def evaluate(multiplier):
        qs = [_arm_minimizer(p, n, multiplier * a)
              for p, (_, n), a in zip(ps, counts, coefficients)]
        return math.fsum(a * q for a, q in zip(coefficients, qs)), qs

    at_zero, q_zero = evaluate(0.0)
    if at_zero == 0:
        return 0.0, q_zero
    lo, hi = (0.0, 1.0) if at_zero > 0 else (-1.0, 0.0)
    for _ in range(1024):
        f_lo, _ = evaluate(lo)
        f_hi, _ = evaluate(hi)
        if f_lo >= 0 >= f_hi:
            break
        lo = lo * 2 if lo else lo
        hi = hi * 2 if hi else hi
        if not math.isfinite(lo) or not math.isfinite(hi):
            raise ArithmeticError("failed to bracket GLR multiplier")
    else:
        raise ArithmeticError("failed to bracket GLR multiplier")
    best = (abs(at_zero), 0.0, q_zero)
    for _ in range(100):
        mid = lo + (hi - lo) / 2
        value, qs = evaluate(mid)
        if abs(value) < best[0]:
            best = (abs(value), mid, qs)
        if mid == lo or mid == hi or value == 0:
            break
        if value > 0:
            lo = mid
        else:
            hi = mid
    return best[1], best[2]


def _feasible_primal(qs, exact_a):
    qs = [Fraction.from_float(q) for q in qs]
    residual = sum((a * q for a, q in zip(exact_a, qs)), Fraction(0))
    if residual == 0:
        return qs
    for j, a in enumerate(exact_a):
        if a:
            adjusted = qs[j] - residual / a
            if 0 < adjusted < 1:
                qs[j] = adjusted
                return qs
    # An exactly feasible, strictly interior fallback for any mixed signs.
    positive = sum((a for a in exact_a if a > 0), Fraction(0))
    negative = -sum((a for a in exact_a if a < 0), Fraction(0))
    return [negative / (positive + negative) if a > 0
            else positive / (positive + negative) if a < 0
            else Fraction(1, 2) for a in exact_a]


def joint_glr_certificate(successes: Sequence[int], trials: Sequence[int],
                          coefficients: Sequence[float], delta: float = 0.01,
                          prior: float = 0.5) -> dict:
    """Return an anytime-valid accept/reject/abstain certificate.

    Set coefficients to (-pi_0,...,-pi_C, +pi_0,...,+pi_C) for old/new
    context arms. All nonzero-coefficient arms must have at least one read.
    Outputs evidence/threshold are outward floating representations of
    Decimal-audited lower/upper bounds. Decimal strings retain audit detail.

    If P is the true arm vector, E_t(P)=L_mix/L(P) is a nonnegative
    supermartingale under predictable adaptive arm choices (a martingale
    when all true means are interior). Since
    log E_t(P)=sum n_i kl(phat_i,P_i)-R_t, a wrong sign declaration implies
    E_t(P)>1/delta. Ville controls the probability of any such declaration.
    """
    counts, exact_a = _validated_inputs(successes, trials, coefficients)
    if not 0 < delta < 1:
        raise ValueError("delta must lie in (0,1)")
    if prior not in (0.5, 1.0):
        raise ValueError("prior must be 0.5 or 1.0")
    active = [i for i, a in enumerate(exact_a) if a]
    shots = sum(n for _, n in counts)
    if any(counts[i][1] == 0 for i in active):
        return dict(decision="abstain", est_delta=None, shots=shots,
                    evidence=0.0, threshold=_INF, primal_upper=_INF,
                    duality_gap=_INF, reason="unobserved_active_arm")
    active_counts = [counts[i] for i in active]
    active_a = [exact_a[i] for i in active]
    exact_est = sum((a * Fraction(s, n)
                     for a, (s, n) in zip(active_a, active_counts)), Fraction(0))
    multiplier, numerical_q = _glr_root(active_counts, [float(a) for a in active_a])
    exact_multiplier = Fraction.from_float(multiplier)
    lambda_interval = _Interval.rational(exact_multiplier)
    dual_parts = []
    for (s, n), a, q_float in zip(active_counts, active_a, numerical_q):
        q = Fraction.from_float(q_float)
        qi = _Interval.rational(q)
        eta = lambda_interval * _Interval.rational(a)
        objective = _count_kl(s, n, q) + eta * qi
        if q == 0:
            gradient = _Interval.exact(n) + eta
        elif q == 1:
            gradient = _Interval.exact(-n) + eta
        else:
            gradient = ((_Interval.exact(n) * qi - _Interval.exact(s))
                        / (qi * (_ONE - qi)) + eta)
        # Tangent line minimizes at one of [0,1]'s endpoints. Thus this is
        # a lower bound even if the numerical q is not the exact minimizer.
        at_zero = gradient * (_ZERO - qi)
        at_one = gradient * (_ONE - qi)
        correction = _Interval(min(at_zero.lo, at_one.lo),
                               min(at_zero.hi, at_one.hi))
        dual_parts.append(objective + correction)
    dual_interval = _sum_intervals(dual_parts)
    evidence_lower = max(Decimal(0), dual_interval.lo)

    primal_q = _feasible_primal(numerical_q, active_a)
    if sum((a * q for a, q in zip(active_a, primal_q)), Fraction(0)):
        raise AssertionError("internal primal feasibility failure")
    try:
        primal = _sum_intervals(_count_kl(s, n, q)
                               for (s, n), q in zip(active_counts, primal_q))
    except ValueError:
        primal_q = _feasible_primal([0.5] * len(active_a), active_a)
        primal = _sum_intervals(_count_kl(s, n, q)
                               for (s, n), q in zip(active_counts, primal_q))
    regret = _sum_intervals(
        _mle_log_likelihood(s, n) - _mixture_log_likelihood(s, n, prior)
        for s, n in active_counts)
    log_inverse_risk = (_ONE / _Interval.rational(Fraction(str(float(delta))))).ln()
    threshold = regret + log_inverse_risk
    decision = "abstain"
    if evidence_lower > threshold.hi and exact_est:
        decision = "accept" if exact_est < 0 else "reject"
    evidence = _Interval(evidence_lower, evidence_lower).float_lo()
    if evidence_lower == 0:
        evidence = 0.0
    return dict(
        decision=decision, est_delta=float(exact_est), shots=shots,
        evidence=evidence, threshold=threshold.float_hi(),
        primal_upper=primal.float_hi(),
        duality_gap=max(0.0, float(primal.hi - evidence_lower)),
        mixture_regret_upper=regret.float_hi(),
        evidence_decimal_lower=str(evidence_lower),
        threshold_decimal_upper=str(threshold.hi),
        primal_decimal_upper=str(primal.hi),
        multiplier=multiplier,
        boundary_probabilities=[float(q) for q in primal_q],
        coefficients_exact=[str(a) for a in exact_a],
        audit="decimal_outward_dual_lower_primal_upper_beta_mixture",
        reason="certified_sign" if decision != "abstain" else "insufficient_evidence",
    )


def _kl_level_point(p: float, level: float, direction: int) -> float:
    if level <= 0:
        return p
    lo, hi = (0.0, p) if direction < 0 else (p, 1.0)
    displacement = math.sqrt(2 * level * p * (1 - p))
    q = min(hi, max(lo, p + direction * displacement))
    if not lo < q < hi:
        q = (lo + hi) / 2
    for _ in range(60):
        if q == 0.0 or q == 1.0:
            # A level point can be closer to an endpoint than binary64 can
            # represent. This affects sampling optimization only; the GLR
            # certificate never uses these allocation probabilities.
            return q
        residual = bernoulli_kl(p, q) - level
        if abs(residual) <= 1e-10 * max(level, 1e-10):
            break
        if (residual > 0) == (direction < 0):
            lo = q
        else:
            hi = q
        derivative = (q - p) / (q * (1 - q))
        candidate = q - residual / derivative if derivative else (lo + hi) / 2
        if not lo < candidate < hi:
            candidate = (lo + hi) / 2
        if candidate == q:
            break
        q = candidate
    return q


def chernoff_allocation(probabilities: Sequence[float], coefficients: Sequence[float]) -> dict:
    """Numerically solve the singleton-arm linear-sign information design.

    For an interior p and a.p != 0, move each q_i in the direction reducing
    a.p, impose kl(p_i,q_i)=z, and solve a.q=0. Then I*(p)=z and
    w_i is proportional to |a_i / d_2 kl(p_i,q_i)|. The equal-KL point gives
    the minimax upper bound; convex KKT conditions at this w give equality.

    Endpoint inputs are regularized only for sampling. The decision rule
    always uses the unregularized observed counts.
    """
    if len(probabilities) != len(coefficients):
        raise ValueError("probabilities and coefficients must have equal lengths")
    if any(not math.isfinite(float(p)) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("probabilities must be finite in [0,1]")
    _validated_inputs([0] * len(coefficients), [0] * len(coefficients), coefficients)
    ps = [min(1 - 1e-12, max(1e-12, float(p))) for p in probabilities]
    aa = [float(a) for a in coefficients]
    delta_est = math.fsum(a * p for a, p in zip(aa, ps))
    scale = math.fsum(abs(a) for a in aa)
    if abs(delta_est) <= 1e-14 * scale:
        raw = [abs(a) * math.sqrt(p * (1 - p)) for a, p in zip(aa, ps)]
        total = math.fsum(raw)
        return dict(weights=[r / total for r in raw], information_rate=0.0,
                    boundary_probabilities=ps, boundary_residual=delta_est,
                    regularized_probabilities=ps, method="local_variance_limit")
    sign = 1 if delta_est > 0 else -1

    def evaluate(level):
        qq = [_kl_level_point(p, level, -sign if a > 0 else sign)
              if a else p for p, a in zip(ps, aa)]
        residual = math.fsum(a * q for a, q in zip(aa, qq))
        derivative = math.fsum(
            a * q * (1 - q) / (q - p)
            for a, q, p in zip(aa, qq, ps) if a and q != p)
        return residual, qq, derivative

    lo, hi = 0.0, 1e-4
    for _ in range(80):
        residual, qq, derivative = evaluate(hi)
        if sign * residual <= 0:
            break
        hi *= 2
    else:
        raise ArithmeticError("failed to bracket information rate")
    level = hi / 2
    for _ in range(70):
        residual, qq, derivative = evaluate(level)
        if abs(residual) <= 1e-11 * scale:
            break
        if sign * residual > 0:
            lo = level
        else:
            hi = level
        candidate = level - residual / derivative if derivative else (lo + hi) / 2
        if not lo < candidate < hi:
            candidate = (lo + hi) / 2
        if candidate == level:
            break
        level = candidate
    raw = [max(1e-300, abs(a * q * (1 - q) / (q - p))) if a else 0.0
           for a, q, p in zip(aa, qq, ps)]
    total = math.fsum(raw)
    return dict(weights=[r / total for r in raw], information_rate=level,
                boundary_probabilities=qq, boundary_residual=residual,
                regularized_probabilities=ps,
                saturated_boundary_coordinates=[i for i, q in enumerate(qq) if q in (0.0, 1.0)],
                method="equal_KL_KKT_singleton_design")


def _screened_certificate(successes, trials, coefficients, delta, prior):
    """Cheap arithmetic may only choose abstention; it can NEVER authorize a stop."""
    counts, exact_a = _validated_inputs(successes, trials, coefficients)
    active = [i for i, a in enumerate(exact_a) if a]
    if any(counts[i][1] == 0 for i in active):
        return joint_glr_certificate(successes, trials, coefficients, delta, prior)
    ac = [counts[i] for i in active]
    aa = [float(exact_a[i]) for i in active]
    _, qs = _glr_root(ac, aa)
    approximate_glr = math.fsum(n * bernoulli_kl(s / n, q)
                                for (s, n), q in zip(ac, qs))
    approximate_regret = 0.0
    for s, n in ac:
        f = n - s
        mle = (s * math.log(s / n) if s else 0.0)
        mle += f * math.log(f / n) if f else 0.0
        log_mix = (math.lgamma(s + prior) + math.lgamma(f + prior)
                   - math.lgamma(n + 2 * prior) - 2 * math.lgamma(prior)
                   + math.lgamma(2 * prior))
        approximate_regret += mle - log_mix
    approximate_threshold = -math.log(delta) + approximate_regret
    if approximate_glr <= approximate_threshold:
        # The safe bounds here are deliberately trivial (0 and infinity).
        # The approximate fields are diagnostic screening values, not evidence
        # for a decision. Skipping an audit can only delay a valid decision.
        return dict(decision="abstain",
                    est_delta=math.fsum(a * s / n for a, (s, n) in zip(aa, ac)),
                    shots=sum(trials), evidence=0.0, threshold=_INF,
                    primal_upper=_INF, duality_gap=_INF,
                    screen_glr_approximate=approximate_glr,
                    screen_threshold_approximate=approximate_threshold,
                    reason="fast_screen_abstain", audit="no_decision_authorized")
    return joint_glr_certificate(successes, trials, coefficients, delta, prior)


def _allocate_largest_deficits(deficits: Sequence[float], budget: int) -> list[int]:
    """Integer water filling, equivalent to repeatedly reducing the largest deficit.

    Only O(K log K) work is required even for a very large batch. Numerical
    tie-breaking affects sampling only; integer accounting is exact.
    """
    if budget < 0 or int(budget) != budget:
        raise ValueError("budget must be a nonnegative integer")
    budget = int(budget)
    if not deficits:
        if budget:
            raise ValueError("cannot allocate a positive budget to no arms")
        return []
    if not budget:
        return [0] * len(deficits)
    order = sorted(range(len(deficits)), key=lambda i: (-deficits[i], i))
    prefix_total = 0.0
    level = 0.0
    for rank, i in enumerate(order, 1):
        prefix_total = math.fsum((prefix_total, deficits[i]))
        level = (prefix_total - budget) / rank
        if rank == len(order) or level >= deficits[order[rank]]:
            break
    counts = [max(0, int(math.floor(d - level))) for d in deficits]
    used = sum(counts)
    if used > budget:
        # Defensive repair for extreme floating scales; never one operation
        # per shot. It has no bearing on the statistical certificate.
        excess = used - budget
        for i in sorted(range(len(deficits)), key=lambda j: (deficits[j] - counts[j], -j)):
            removed = min(counts[i], excess)
            counts[i] -= removed
            excess -= removed
            if not excess:
                break
    elif used < budget:
        remaining = budget - used
        residual_order = sorted(range(len(deficits)),
                                key=lambda i: (-(deficits[i] - counts[i]), i))
        quotient, remainder = divmod(remaining, len(deficits))
        if quotient:
            counts = [c + quotient for c in counts]
        for i in residual_order[:remainder]:
            counts[i] += 1
    if sum(counts) != budget:
        raise AssertionError("integer allocation lost budget")
    return counts


class TrackStopState:
    """Batched numerical tracking with forced exploration and exact certificates.

    recommend_counts is based only on past paid observations. The caller
    executes the returned per-arm batches and charges every resulting shot.
    Fixed public coefficients and the Beta prior must be frozen before data.
    """

    def __init__(self, coefficients: Sequence[float], delta: float = 0.01,
                 prior: float = 0.5):
        _validated_inputs([0] * len(coefficients), [0] * len(coefficients), coefficients)
        if not 0 < delta < 1 or prior not in (0.5, 1.0):
            raise ValueError("invalid risk or prior")
        self.coefficients = list(map(float, coefficients))
        self.delta = float(delta)
        self.prior = float(prior)
        self.successes = [0] * len(coefficients)
        self.trials = [0] * len(coefficients)

    def update(self, arm: int, successes: int, trials: int = 1) -> None:
        if not 0 <= arm < len(self.trials):
            raise IndexError("unknown arm")
        if int(successes) != successes or int(trials) != trials or not 0 <= successes <= trials:
            raise ValueError("invalid binomial counts")
        self.successes[arm] += int(successes)
        self.trials[arm] += int(trials)

    def allocation(self) -> dict:
        ps = [(s + self.prior) / (n + 2 * self.prior)
              for s, n in zip(self.successes, self.trials)]
        return chernoff_allocation(ps, self.coefficients)

    def recommend_counts(self, batch_size: int = 256) -> list[int]:
        if int(batch_size) != batch_size or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        batch_size = int(batch_size)
        active = [i for i, a in enumerate(self.coefficients) if a]
        proposed = [0] * len(self.trials)
        target_total = sum(self.trials) + batch_size
        floor_count = max(1, int(math.sqrt(target_total) / len(active)))
        needs = [max(0, floor_count - self.trials[i]) for i in active]
        if sum(needs) > batch_size:
            forced = _allocate_largest_deficits([-self.trials[i] for i in active], batch_size)
        else:
            forced = needs
        for i, amount in zip(active, forced):
            proposed[i] = amount
        remaining = batch_size - sum(forced)
        if remaining:
            weights = self.allocation()["weights"]
            deficits = [target_total * weights[i] - self.trials[i] - proposed[i]
                        for i in active]
            extra = _allocate_largest_deficits(deficits, remaining)
            for i, amount in zip(active, extra):
                proposed[i] += amount
        return proposed

    def certificate(self, full_audit: bool = False) -> dict:
        if not full_audit:
            return _screened_certificate(self.successes, self.trials, self.coefficients,
                                         self.delta, self.prior)
        return joint_glr_certificate(self.successes, self.trials, self.coefficients,
                                     self.delta, self.prior)

