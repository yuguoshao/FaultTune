"""Frozen-candidate validation policies using only paid observations.

The strongest comparison is joint_trackstop, a specialization of existing
linear pure exploration. neyman_glr and balanced_glr are transparent candidate
designs sharing exactly the same valid stopping certificate. No policy here
is presented as a new statistical algorithm.
"""

import math

from direct_tests import certificate_from_pairs, direct_kl_certificate, fixed_paired_certificate
from kl_trackstop import TrackStopState
from combgame import run_combgame


POLICIES = ("direct_fixed", "direct_mixture", "direct_intervals", "joint_trackstop", "combgame_glr",
            "neyman_glr", "balanced_glr", "calibration_only")
BASELINES = ("direct_fixed", "direct_mixture", "joint_trackstop", "combgame_glr")
CANDIDATES = ("neyman_glr", "balanced_glr")


def _next_total(current, limit, initial, growth, max_extra):
    target = initial if current == 0 else max(current + initial, int(math.ceil(current * growth)))
    return min(limit, current + max_extra, target)


def _pair_screen(counts, delta, prior=0.5):
    worse, better = counts[1], counts[2]
    discordant = worse + better
    if not discordant or worse == better:
        return False
    log_mix = (math.lgamma(worse + prior) + math.lgamma(better + prior)
               - math.lgamma(discordant + 2 * prior) - 2 * math.lgamma(prior)
               + math.lgamma(2 * prior))
    return log_mix + discordant * math.log(2.0) >= math.log(1.0 / delta)


def _weighted_batch(state, batch, weights):
    """Batch allocation with sublinear forced exploration; all safety is in GLR."""
    count = len(weights)
    target_total = sum(state.trials) + batch
    floor = max(1, int(math.sqrt(target_total) / count))
    proposed = [max(0, floor - n) for n in state.trials]
    if sum(proposed) > batch:
        # This case occurs only for tiny initial budgets; bounded O(K*batch)
        # is at most the initialization/exploration deficit, not a long run.
        proposed = [0] * count
        for unused in range(batch):
            arm = min(range(count), key=lambda i: (state.trials[i] + proposed[i], i))
            proposed[arm] += 1
        return proposed
    remaining = batch - sum(proposed)
    deficits = [max(0.0, target_total * w - n - add)
                for w, n, add in zip(weights, state.trials, proposed)]
    total = math.fsum(deficits)
    if not total:
        deficits, total = list(weights), math.fsum(weights)
    real = [remaining * d / total for d in deficits]
    extra = [int(x) for x in real]
    for arm in sorted(range(count), key=lambda i: (real[i] - extra[i], -i), reverse=True)[:remaining - sum(extra)]:
        extra[arm] += 1
    return [a + b for a, b in zip(proposed, extra)]


def run_policy(name, config, sample_arm, sample_pairs, fixed_pairs=None,
               growth=1.3, initial=32, max_extra=262144, combgame_macro_budget=512):
    if name not in POLICIES or not 1 < growth <= 2:
        raise ValueError("invalid policy or checkpoint growth")
    if (isinstance(initial, bool) or not isinstance(initial, int) or initial < 1
            or isinstance(max_extra, bool) or not isinstance(max_extra, int) or max_extra < 2):
        raise ValueError("initial must be a positive integer and max_extra an integer >=2")
    if name == "combgame_glr":
        return run_combgame(config, sample_arm, macro_budget=combgame_macro_budget)
    if name == "direct_fixed":
        if (isinstance(fixed_pairs, bool) or not isinstance(fixed_pairs, int)
                or fixed_pairs < 1 or 2 * fixed_pairs > config.shot_budget):
            raise ValueError("fixed sample count must be preregistered and fit the budget")
        joint = sample_pairs(fixed_pairs)["joint"]
        cert = fixed_paired_certificate(joint, config.delta)
        return {"decision": cert["decision"], "certificate": cert, "checks": 1,
                "pair_counts": joint, "checkpoint_growth": None,
                "guarantee_in_scope": True}

    if name in ("direct_mixture", "direct_intervals", "calibration_only"):
        counts, pairs, checks = [0] * 4, 0, 0
        limit = config.shot_budget // 2
        cert = None
        while pairs < limit:
            target = _next_total(pairs, limit, initial, growth, max_extra // 2)
            result = sample_pairs(target - pairs, context=0 if name == "calibration_only" else None)
            counts = [a + b for a, b in zip(counts, result["joint"])]
            pairs = target
            checks += 1
            if name == "direct_intervals":
                cert = direct_kl_certificate(counts[2] + counts[3], pairs,
                                             counts[1] + counts[3], pairs, config.delta)
            elif pairs == limit or _pair_screen(counts, config.delta):
                # A binary64 prefilter can only delay a stop. Every actual
                # decision passes the separate outward-Decimal certificate.
                cert = certificate_from_pairs(counts, config.delta)
            else:
                continue
            if cert["decision"] != "abstain":
                break
        if cert is None:
            cert = certificate_from_pairs(counts, config.delta)
        return {"decision": cert["decision"], "certificate": cert, "checks": checks,
                "pair_counts": counts, "checkpoint_growth": growth,
                "guarantee_in_scope": name != "calibration_only"}

    state = TrackStopState(config.coefficients, config.delta)
    total, checks = 0, 0
    cert = None
    while total < config.shot_budget:
        target = _next_total(total, config.shot_budget, initial * config.arms, growth, max_extra)
        batch = target - total
        if name == "joint_trackstop":
            proposed = state.recommend_counts(batch)
        elif name == "balanced_glr":
            proposed = _weighted_batch(state, batch, [1.0 / config.arms] * config.arms)
        else:
            means = [(s + .5) / (n + 1) for s, n in zip(state.successes, state.trials)]
            weights = [abs(a) * math.sqrt(p * (1 - p)) for a, p in zip(config.coefficients, means)]
            mass = math.fsum(weights)
            proposed = _weighted_batch(state, batch, [w / mass for w in weights])
        if sum(proposed) != batch:
            raise ArithmeticError("allocation failed to conserve the paid batch")
        for arm, count in enumerate(proposed):
            if count:
                successes = sample_arm(arm, count)
                state.update(arm, successes, count)
        total = target
        checks += 1
        cert = state.certificate(full_audit=total == config.shot_budget)
        if cert["decision"] != "abstain":
            break
    if cert is None:
        cert = state.certificate(full_audit=True)
    return {"decision": cert["decision"], "certificate": cert, "checks": checks,
            "checkpoint_growth": growth, "guarantee_in_scope": True,
            "arm_successes": state.successes, "arm_trials": state.trials}
