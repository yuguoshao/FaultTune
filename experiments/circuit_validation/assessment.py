"""Fixed-repetition statistical audit for validation benchmark records.

The runner allocates family error budgets BEFORE inspecting held-out records.
Monte Carlo repetitions within a cell must be independent draws of the same
frozen experiment. Cross-policy/common-seed pairing is allowed: union-bound
interval propagation does not require independence between cells or policies.

Clopper--Pearson endpoints invert binomial tails. SciPy, when importable, only
proposes a quantile: exact integer arithmetic checks the conservative side.
Cost intervals use Maurer--Pontil (2009), Theorem 4, applied to X/B and 1-X/B:

    r = sqrt(2 * sample_variance * log(4/alpha) / n)
        + 7 * B * log(4/alpha) / (3 * (n-1)).

The resulting two-sided interval has coverage at least 1-alpha for iid
costs in [0,B], n>=2. Means include ALL records, including wrong decisions
and noncompletion. These are fixed-repetition intervals, not confidence
sequences and not bootstrap intervals.

Primary source: https://arxiv.org/pdf/0907.3740 , Theorem 4.
"""

from __future__ import annotations

from collections import Counter
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
from functools import lru_cache
import math
import struct
from typing import Mapping, Sequence


def _nextafter(value: float, toward: float) -> float:
    if math.isnan(value) or math.isnan(toward):
        return float("nan")
    if value == toward:
        return toward
    if value == 0:
        return math.copysign(float.fromhex("0x0.0000000000001p-1022"), toward)
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    bits += 1 if (value > 0) == (toward > value) else -1
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


def _validate_alpha(alpha: float) -> Fraction:
    if not math.isfinite(float(alpha)) or not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    return Fraction(str(float(alpha)))


def _binomial_cdf_ratio(n: int, k: int, probability: Fraction):
    """Exact numerator/denominator, using the shorter tail where possible."""
    if k < 0:
        return 0, 1
    if k >= n:
        return 1, 1
    if probability == 0:
        return 1, 1
    if probability == 1:
        return 0, 1
    if not 0 < probability < 1:
        raise ValueError("invalid binomial probability")
    if k > (n - 1) // 2:
        numerator, denominator = _binomial_cdf_ratio(n, n - k - 1, 1 - probability)
        return denominator - numerator, denominator
    r, d = probability.numerator, probability.denominator
    other = d - r
    term = other ** n
    numerator = term
    for j in range(k):
        term = term * (n - j) * r // ((j + 1) * other)
        numerator += term
    return numerator, d ** n


def _cdf_at_most(n, k, probability, alpha):
    numerator, denominator = _binomial_cdf_ratio(n, k, probability)
    return numerator * alpha.denominator <= alpha.numerator * denominator


@lru_cache(maxsize=4096)
def clopper_pearson_upper(successes: int, trials: int, alpha: float) -> float:
    """One-sided 1-alpha upper limit; returned float is tail-certified."""
    if int(successes) != successes or int(trials) != trials or not 0 <= successes <= trials:
        raise ValueError("require integer 0 <= successes <= trials")
    successes, trials = int(successes), int(trials)
    risk = _validate_alpha(alpha)
    if trials == 0 or successes == trials:
        return 1.0
    proposal = None
    if successes == 0:
        proposal = -math.expm1(math.log(float(alpha)) / trials)
    else:
        try:
            from scipy.stats import beta
            proposal = float(beta.isf(float(alpha), successes + 1, trials - successes))
        except ImportError:
            pass
    if proposal is not None and math.isfinite(proposal):
        proposal = min(1.0, max(0.0, proposal))
        for _ in range(8):
            if _cdf_at_most(trials, successes, Fraction.from_float(proposal), risk):
                return proposal
            proposal = _nextafter(proposal, math.inf)
            if proposal >= 1:
                return 1.0
    # No SciPy dependency is needed for correctness. Preserve an exact
    # bracket: lo has CDF>alpha; hi has CDF<=alpha and is therefore safe.
    lo, hi = 0.0, 1.0
    for _ in range(56):
        mid = lo + (hi - lo) / 2
        if mid in (lo, hi):
            break
        if _cdf_at_most(trials, successes, Fraction.from_float(mid), risk):
            hi = mid
        else:
            lo = mid
    return hi


def clopper_pearson_lower(successes: int, trials: int, alpha: float) -> float:
    """One-sided 1-alpha lower limit, by exact success/failure symmetry."""
    if int(successes) != successes or int(trials) != trials or not 0 <= successes <= trials:
        raise ValueError("require integer 0 <= successes <= trials")
    _validate_alpha(alpha)
    if trials == 0 or successes == 0:
        return 0.0
    upper_failure = clopper_pearson_upper(trials - successes, trials, alpha)
    return max(0.0, _nextafter(1.0 - upper_failure, -math.inf))


def _fraction(value) -> Fraction:
    if isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError("finite numeric cost required")
    return Fraction(str(value))


def empirical_bernstein_interval(values: Sequence[float], bound: float,
                                  alpha: float) -> dict:
    """Two-sided fixed-n mean interval with outward Decimal arithmetic."""
    risk = _validate_alpha(alpha)
    cap = _fraction(bound)
    if cap <= 0:
        raise ValueError("bound must be positive")
    cap_float_upper = float(cap)
    if Fraction.from_float(cap_float_upper) < cap:
        cap_float_upper = _nextafter(cap_float_upper, math.inf)
    xx = [_fraction(x) for x in values]
    if any(x < 0 or x > cap for x in xx):
        raise ValueError("observed cost outside the predeclared [0,bound]")
    n = len(xx)
    total = sum(xx, Fraction(0))
    mean = total / n if n else None
    sample_variance = None
    if n >= 2:
        sample_variance = (sum((x * x for x in xx), Fraction(0))
                           - total * total / n) / (n - 1)
    base = dict(n=n, mean=float(mean) if mean is not None else None,
                bound=float(cap), alpha=float(alpha),
                sample_variance=float(sample_variance) if sample_variance is not None else None,
                method="Maurer_Pontil_2009_Theorem4_two_sided",
                independent_repetitions_required=True, valid=True)
    if n < 2:
        return dict(base, lower=0.0, upper=cap_float_upper, radius=None)
    with localcontext() as ctx:
        ctx.prec = 64
        ctx.rounding = ROUND_FLOOR
        mean_lo = Decimal(mean.numerator) / Decimal(mean.denominator)
        ctx.rounding = ROUND_CEILING
        mean_hi = Decimal(mean.numerator) / Decimal(mean.denominator)
        var_hi = Decimal(sample_variance.numerator) / Decimal(sample_variance.denominator)
        cap_hi = Decimal(cap.numerator) / Decimal(cap.denominator)
        ratio_hi = Decimal(4 * risk.denominator) / Decimal(risk.numerator)
        # ln/sqrt are correctly rounded; one Decimal successor gives an
        # upper enclosure independently of their rounding-mode convention.
        log_hi = ratio_hi.ln().next_plus(ctx)
        quadratic_hi = Decimal(2) * var_hi * log_hi / Decimal(n)
        square_root_hi = quadratic_hi.sqrt().next_plus(ctx)
        linear_hi = Decimal(7) * cap_hi * log_hi / Decimal(3 * (n - 1))
        radius_hi = square_root_hi + linear_hi
        upper = mean_hi + radius_hi
        ctx.rounding = ROUND_FLOOR
        lower = mean_lo - radius_hi
    lower_float = max(0.0, _nextafter(float(lower), -math.inf))
    upper_float = min(cap_float_upper, _nextafter(float(upper), math.inf))
    return dict(base, lower=lower_float, upper=upper_float,
                radius=_nextafter(float(radius_hi), math.inf),
                radius_decimal_upper=str(radius_hi))


def _descriptive_cost(values):
    ordered = sorted(values)
    n = len(values)
    if not n:
        return dict(median=None, p95=None, minimum=None, maximum=None)
    middle = n // 2
    median = (ordered[middle] if n % 2 else (ordered[middle - 1] + ordered[middle]) / 2)
    # A prespecified nearest-rank sample quantile, not a confidence bound.
    return dict(median=median, p95=ordered[max(0, math.ceil(.95 * n) - 1)],
                minimum=ordered[0], maximum=ordered[-1])


def summarize_cell(records: Sequence[dict], true_delta: float, epsilon: float,
                   alpha_safety: float, alpha_power: float, alpha_cost: float,
                   budget: int) -> dict:
    """Audit all records from one frozen (instance, policy, budget) cell.

    Each alpha argument is already a PER-ENDPOINT allocation from the runner.
    alpha_safety funds each of FA, FR and any-wrong; alpha_power funds each
    of good-accept and bad-reject; alpha_cost funds each of actual and penalized
    cost intervals. Inapplicable endpoint allocations are not recycled.

    Non-ok statuses never count as power successes. An accept/reject already
    present in such a record still counts for safety, avoiding hidden errors.
    Missing/invalid shot counts raise rather than silently discarding records.
    Out-of-budget observations are retained but invalidate bounded-cost CIs.
    """
    for alpha in (alpha_safety, alpha_power, alpha_cost):
        _validate_alpha(alpha)
    if not math.isfinite(true_delta) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("finite delta and positive epsilon required")
    if isinstance(budget, bool) or int(budget) != budget or budget <= 0:
        raise ValueError("budget must be a positive integer")
    budget = int(budget)
    decisions = Counter()
    statuses = Counter()
    actual, penalized = [], []
    issues = []
    valid_accepts = valid_rejects = 0
    for i, record in enumerate(records):
        if "shots" not in record:
            raise ValueError("record %d has no auditable shot count" % i)
        shots = record["shots"]
        if isinstance(shots, bool) or not math.isfinite(float(shots)) or int(shots) != shots or shots < 0:
            raise ValueError("record %d has an invalid shot count" % i)
        shots = int(shots)
        decision = record.get("decision", "missing")
        status = record.get("status", "missing")
        if not isinstance(decision, str) or not isinstance(status, str):
            raise ValueError("decision and status must be strings")
        decisions[decision] += 1
        statuses[status] += 1
        record_budget = record.get("shot_budget", budget)
        if record_budget != budget:
            issues.append(dict(record=i, issue="shot_budget_mismatch"))
        if shots > budget:
            issues.append(dict(record=i, issue="budget_exceeded", shots=shots))
        if decision not in ("accept", "reject", "abstain"):
            issues.append(dict(record=i, issue="invalid_decision", value=decision))
        if status not in ("ok", "timeout", "error"):
            issues.append(dict(record=i, issue="invalid_status", value=status))
        completed = status == "ok" and decision in ("accept", "reject")
        valid_accepts += int(completed and decision == "accept")
        valid_rejects += int(completed and decision == "reject")
        actual.append(shots)
        penalized.append(shots if completed else max(shots, budget))
    n = len(records)
    false_accepts = decisions["accept"] if true_delta >= 0 else 0
    false_rejects = decisions["reject"] if true_delta <= 0 else 0
    if true_delta > 0:
        safety_event, safety_count = "false_accept", false_accepts
    elif true_delta < 0:
        safety_event, safety_count = "false_reject", false_rejects
    else:
        safety_event = "any_decision_at_zero"
        safety_count = false_accepts + false_rejects
    safety_upper = clopper_pearson_upper(safety_count, n, alpha_safety)
    safety = dict(event=safety_event, count=safety_count, n=n,
                  observed_rate=safety_count / n if n else None,
                  upper=safety_upper, alpha=alpha_safety, target=0.01,
                  passes=n > 0 and Fraction.from_float(safety_upper) <= Fraction(1, 100))
    def safety_endpoint(event, count, applicable):
        upper = clopper_pearson_upper(count, n, alpha_safety) if applicable else None
        return dict(event=event, applicable=applicable, count=count if applicable else None,
                    n=n, observed_rate=count / n if applicable and n else None,
                    upper=upper, alpha=alpha_safety, target=0.01,
                    passes=(n > 0 and Fraction.from_float(upper) <= Fraction(1, 100)) if applicable else None)
    false_accept_endpoint = safety_endpoint("false_accept", false_accepts, true_delta >= 0)
    false_reject_endpoint = safety_endpoint("false_reject", false_rejects, true_delta <= 0)
    any_wrong_endpoint = dict(safety, applicable=True, event="any_wrong")
    applicable = abs(true_delta) >= epsilon
    power_event = ("accept_good" if true_delta <= -epsilon else
                   "reject_bad" if true_delta >= epsilon else None)
    power_count = valid_accepts if power_event == "accept_good" else valid_rejects if power_event else None
    power_lower = clopper_pearson_lower(power_count, n, alpha_power) if applicable else None
    power = dict(applicable=applicable, event=power_event, count=power_count,
                 n=n, observed_rate=power_count / n if applicable and n else None,
                 lower=power_lower, alpha=alpha_power, target=0.95,
                 passes=(n > 0 and Fraction.from_float(power_lower) >= Fraction(19, 20)) if applicable else None)
    def power_endpoint(event, count, applies):
        lower = clopper_pearson_lower(count, n, alpha_power) if applies else None
        return dict(event=event, applicable=applies, count=count if applies else None,
                    n=n, observed_rate=count / n if applies and n else None,
                    lower=lower, alpha=alpha_power, target=0.95,
                    passes=(n > 0 and Fraction.from_float(lower) >= Fraction(19, 20)) if applies else None)
    good_accept_endpoint = power_endpoint("accept_good", valid_accepts, true_delta <= -epsilon)
    bad_reject_endpoint = power_endpoint("reject_bad", valid_rejects, true_delta >= epsilon)
    cost_alpha = alpha_cost
    costs = {}
    for name, values in (("actual_cost", actual), ("penalized_cost", penalized)):
        try:
            interval = empirical_bernstein_interval(values, budget, cost_alpha)
        except ValueError:
            interval = dict(n=n, mean=sum(values) / n if n else None,
                            lower=0.0, upper=math.inf, alpha=cost_alpha,
                            bound=budget, valid=False,
                            method="invalid_predeclared_cost_bound")
        interval.update(_descriptive_cost(values))
        costs[name] = interval
    integrity = not issues
    qualified = (integrity and safety["passes"] and
                 (power["passes"] if applicable else True))
    return dict(
        n=n, true_delta=true_delta, epsilon=epsilon, budget=budget,
        eligible_for_efficiency=applicable,
        decisions=dict(decisions), statuses=dict(statuses),
        false_accept_count=false_accepts, false_reject_count=false_rejects,
        completed_count=valid_accepts + valid_rejects,
        noncompletion_count=n - valid_accepts - valid_rejects,
        abstention_or_failure_rate=(n - valid_accepts - valid_rejects) / n if n else None,
        safety=safety, power=power, qualified=bool(qualified),
        false_accept=false_accept_endpoint, false_reject=false_reject_endpoint,
        any_wrong=any_wrong_endpoint, good_accept=good_accept_endpoint,
        bad_reject=bad_reject_endpoint,
        record_integrity_pass=integrity, record_issues=issues,
        alpha_ledger=dict(false_accept_endpoint=alpha_safety, false_reject_endpoint=alpha_safety,
                          any_wrong_endpoint=alpha_safety,
                          good_accept_endpoint=alpha_power, bad_reject_endpoint=alpha_power,
                          actual_cost_interval=cost_alpha, penalized_cost_interval=cost_alpha),
        interval_scope="fixed_predeclared_repetitions_iid_within_cell",
        **costs,
    )


def _cost_ratio(candidate, baselines, metric):
    c = candidate[metric]
    bb = [(name, summary[metric]) for name, summary in baselines]
    if not bb or c["mean"] is None:
        return dict(point=None, lower=None, upper=None, fastest_point_baseline=None)
    if any(cost["mean"] is None for _, cost in bb):
        return dict(point=None, lower=0.0, upper=math.inf, fastest_point_baseline=None,
                    reason="missing_baseline_cost_mean")
    fastest = min(bb, key=lambda item: item[1]["mean"])
    denominator = c["mean"]
    point = (fastest[1]["mean"] / denominator if denominator > 0 else
             math.inf if fastest[1]["mean"] > 0 else None)
    numerator_lower = min(cost["lower"] for _, cost in bb)
    numerator_upper = min(cost["upper"] for _, cost in bb)
    lower = numerator_lower / c["upper"] if c["upper"] > 0 and math.isfinite(c["upper"]) else 0.0
    upper = numerator_upper / c["lower"] if c["lower"] > 0 else math.inf
    return dict(point=point,
                lower=max(0.0, _nextafter(lower, -math.inf)) if lower else 0.0,
                upper=_nextafter(upper, math.inf) if math.isfinite(upper) else math.inf,
                fastest_point_baseline=fastest[0],
                baseline_mean_min=fastest[1]["mean"],
                baseline_lower_min=numerator_lower, baseline_upper_min=numerator_upper,
                candidate_mean=c["mean"], candidate_lower=c["lower"], candidate_upper=c["upper"])


def compare_costs(candidate_summary: dict, baseline_summaries) -> dict:
    """Compare against the fastest qualifying preregistered baseline.

    Inputs are summarize_cell outputs; a mapping names the baselines, while a
    sequence assigns stable numeric names. Candidate failure never removes a
    predeclared eligible cell: it is reported through candidate_qualified.
    The caller aggregates the SAME frozen eligible cells, propagating zero
    lower bounds or missing evidence instead of dropping inconvenient cells.
    A baseline whose lower power bound is too small has NOT been proved
    incapable. Any unconfirmed baseline blocks qualification_pass and
    criteria_pass; subset comparisons remain visible only as diagnostics.

    criteria_pass is a per-cell convenience flag requiring qualification
    plus BOTH actual and penalized point speedup >=3 and lower bound >=2.
    It is not a replacement for the runner's frozen all-cell aggregation.
    """
    if isinstance(baseline_summaries, Mapping):
        all_baselines = list(baseline_summaries.items())
    else:
        all_baselines = [(str(i), b) for i, b in enumerate(baseline_summaries)]
    for name, baseline in all_baselines:
        for key in ("true_delta", "epsilon", "budget"):
            if baseline[key] != candidate_summary[key]:
                raise ValueError("baseline %s has a different %s" % (name, key))
    qualifying = [(name, b) for name, b in all_baselines if b.get("qualified", False)]
    unconfirmed = [name for name, b in all_baselines if not b.get("qualified", False)]
    baseline_complete = bool(all_baselines) and not unconfirmed
    status = ("no_qualifying_baseline" if not qualifying else
              "baseline_qualification_incomplete" if not baseline_complete else "ok")
    if not candidate_summary.get("eligible_for_efficiency", False):
        status = "outside_predeclared_gap_domain"
    actual_ratio = _cost_ratio(candidate_summary, qualifying, "actual_cost")
    penalized_ratio = _cost_ratio(candidate_summary, qualifying, "penalized_cost")
    qualification_pass = status == "ok" and candidate_summary.get("qualified", False)
    def reaches_targets(ratio):
        return (ratio["point"] is not None and ratio["lower"] is not None
                and ratio["point"] >= 3.0 and ratio["lower"] >= 2.0)
    actual_target_pass = reaches_targets(actual_ratio)
    penalized_target_pass = reaches_targets(penalized_ratio)
    return dict(
        status=status,
        eligible_for_efficiency=candidate_summary.get("eligible_for_efficiency", False),
        candidate_qualified=candidate_summary.get("qualified", False),
        baseline_qualification_complete=baseline_complete,
        qualification_pass=bool(qualification_pass),
        criteria_pass=bool(qualification_pass and actual_target_pass and penalized_target_pass),
        criteria_targets=dict(point_speedup=3.0, lower_speedup=2.0,
                              metrics=["actual_cost", "penalized_cost"]),
        criteria_scope="this_cell_only_not_the_frozen_aggregate_test",
        actual_target_pass=actual_target_pass,
        penalized_target_pass=penalized_target_pass,
        qualifying_baselines=[name for name, _ in qualifying],
        unconfirmed_baselines=unconfirmed,
        excluded_baselines=unconfirmed,
        ratio_scope="all_preregistered_baselines" if baseline_complete else "qualified_subset_only_no_pass_claim",
        actual_cost=actual_ratio,
        penalized_cost=penalized_ratio,
        descriptive_all_baselines=dict(
            actual_cost=_cost_ratio(candidate_summary, all_baselines, "actual_cost"),
            penalized_cost=_cost_ratio(candidate_summary, all_baselines, "penalized_cost")),
        confidence_scope="inherits_runner_simultaneous_endpoint_coverage_no_extra_alpha",
    )
