"""CombGame-AdaHedge sampling for fixed independent Bernoulli blocks.

Algorithm 1, sections 3.2--3.3 of Jourdan et al. (ALT 2021), specialized
to singleton actions and two weighted-sum answers. The two answer-specific
AdaHedge learners, KL best response, optimistic rewards and ratio C-tracking
are implemented explicitly. One action observes a FIXED block of b Bernoulli
shots: its KL is b*kl(p,q). This is an exponential-family arm in its own right.

Declared changes: a sampling-only compact alternative box prevents infinite
endpoint KL rewards; the stopping threshold is our shared audited Beta-GLR;
one last incomplete block is used only for the final certificate. Consequently
we claim the independently proved safety guarantee, not automatic transfer of
the original paper's asymptotic efficiency theorem to this implementation.
"""

import math
from functools import lru_cache

from kl_trackstop import TrackStopState, _arm_minimizer, _kl_level_point, bernoulli_kl


COMBGAME_SOURCE = "https://proceedings.mlr.press/v132/jourdan21a/jourdan21a.pdf"
ADAHEDGE_SOURCE = "https://jmlr.org/papers/volume15/rooij14a/rooij14a.pdf"
DEFAULT_MACRO_BUDGET = 512
ALTERNATIVE_CLIP = 1e-9


def _mix(losses, eta):
    """Stable uniform-prior Hedge weights and cumulative mix loss."""
    least = min(losses)
    if math.isinf(eta):
        weights = [1.0 if loss == least else 0.0 for loss in losses]
        total = math.fsum(weights)
        return [w / total for w in weights], least
    weights = [math.exp(-eta * (loss - least)) for loss in losses]
    total = math.fsum(weights)
    return [w / total for w in weights], least - math.log(total / len(losses)) / eta


class AdaHedge:
    """de Rooij et al. (2014), equation (8) and Figure 1.

    Rewards become losses by subtracting from their per-round maximum, an
    invariant common shift. No learning-rate tuning or forced weight floor.
    Binary64 exponential underflow may produce zero weights; the best-response
    and tracker below support them. Cumulative losses are shifted for stability.
    """

    def __init__(self, arms):
        if isinstance(arms, bool) or not isinstance(arms, int) or arms < 2:
            raise ValueError("at least two arms are required")
        self.losses = [0.0] * arms
        self.mixability_gap = 0.0
        self.rounds = 0

    @property
    def eta(self):
        return (math.log(len(self.losses)) / self.mixability_gap
                if self.mixability_gap > 0 else math.inf)

    def weights(self):
        return _mix(self.losses, self.eta)[0]

    def update(self, rewards):
        if len(rewards) != len(self.losses) or any(not math.isfinite(r) for r in rewards):
            raise ValueError("finite reward vector with one value per arm required")
        maximum = max(rewards)
        losses = [maximum - reward for reward in rewards]
        eta = self.eta
        weights, before = _mix(self.losses, eta)
        expected = math.fsum(w * loss for w, loss in zip(weights, losses))
        updated = [old + new for old, new in zip(self.losses, losses)]
        unused, after = _mix(updated, eta)
        # The published implementation also clips numerical Jensen violations.
        self.mixability_gap += max(0.0, expected - (after - before))
        least = min(updated)
        self.losses = [loss - least for loss in updated]
        self.rounds += 1


@lru_cache(maxsize=4096)
def exploration_bonus(completed_blocks):
    """Appendix G.1: Wbar(4 log t), with its two positive constants both 1."""
    if completed_blocks < 1:
        raise ValueError("at least one completed block required")
    value = max(1.0, 4.0 * math.log(completed_blocks))
    if value == 1:
        return 1.0
    # Wbar(x) is the root y>=1 of y-log(y)=x. Newton from above.
    y = value + math.log(value) + 2.0
    for unused in range(16):
        updated = y - (y - math.log(y) - value) / (1.0 - 1.0 / y)
        if abs(updated - y) <= 2e-15 * y:
            return updated
        y = updated
    return y


def best_response(probabilities, weights, coefficients, clip=ALTERNATIVE_CLIP,
                  answer_sign=None):
    """Minimize weighted KL in the compact alternative (or its boundary).

    With answer_sign in {-1,+1}, constrain answer_sign*a.q<=0; without it,
    constrain a.q=0. The latter is normally the former's active boundary.

    Convex KKT multiplier solve, including zero learner weights. The box is a
    declared numerical regularization of sampling only; this value is NEVER
    used to authorize a stop. The public old/new coefficients must sum to zero.
    """
    k = len(probabilities)
    if not k or not (len(weights) == len(coefficients) == k):
        raise ValueError("matching nonempty vectors required")
    if not 0 < clip < .5:
        raise ValueError("clip must be in (0,.5)")
    if answer_sign is not None and answer_sign not in (-1, 1):
        raise ValueError("answer_sign must be -1, +1 or None")
    if (any(not 0 <= p <= 1 for p in probabilities)
            or any(not math.isfinite(w) or w < 0 for w in weights)
            or not math.isclose(math.fsum(weights), 1.0, abs_tol=1e-12)
            or any(not math.isfinite(a) for a in coefficients)
            or not any(a > 0 for a in coefficients)
            or not any(a < 0 for a in coefficients)
            or abs(math.fsum(coefficients)) > 1e-12):
        raise ValueError("probabilities, simplex weights and balanced coefficients required")
    upper = 1.0 - clip

    def clamp(q):
        return min(upper, max(clip, q))

    base = [clamp(p) for p in probabilities]
    base_residual = math.fsum(a * q for a, q in zip(coefficients, base))
    # The empirical vector can lie outside the compact alternative box.
    # Its unconstrained box projection may already be in the alternative
    # halfspace, in which case forcing the boundary would be incorrect.
    if answer_sign is not None and answer_sign * base_residual <= 0:
        return base
    zeros = [i for i, w in enumerate(weights) if w == 0]
    # At multiplier zero, zero-weight coordinates can repair the equality
    # freely. This handles a set-valued best response without a weight floor.
    low_q = list(base)
    for i in zeros:
        low_q[i] = clip if coefficients[i] >= 0 else upper
    low_residual = math.fsum(a * q for a, q in zip(coefficients, low_q))
    capacity = math.fsum(abs(coefficients[i]) * (upper - clip) for i in zeros)
    if low_residual <= 0 <= low_residual + capacity:
        remaining = -low_residual
        for i in zeros:
            amount = min(remaining, abs(coefficients[i]) * (upper - clip))
            if coefficients[i]:
                low_q[i] += amount / coefficients[i]
            remaining -= amount
        return low_q
    if base_residual == 0:
        return base

    def evaluate(multiplier):
        qs = []
        for p, w, a in zip(probabilities, weights, coefficients):
            eta = multiplier * a
            if w:
                q = clamp(_arm_minimizer(p, w, eta))
            else:
                q = clip if eta > 0 else upper if eta < 0 else clamp(p)
            qs.append(q)
        return math.fsum(a * q for a, q in zip(coefficients, qs)), qs

    positive = low_residual > 0
    lo, hi = (0.0, 1.0) if positive else (-1.0, 0.0)
    for unused in range(1024):
        flo, qlo = evaluate(lo)
        fhi, qhi = evaluate(hi)
        if flo >= 0 >= fhi:
            break
        lo = 2.0 * lo
        hi = 2.0 * hi
    else:
        raise ArithmeticError("compact best response could not be bracketed")
    best = (abs(base_residual), base)
    for unused in range(1100):
        mid = lo + (hi - lo) / 2.0
        residual, qs = evaluate(mid)
        if abs(residual) < best[0]:
            best = (abs(residual), qs)
        if abs(residual) <= 1e-13 or mid == lo or mid == hi:
            break
        if residual > 0:
            lo = mid
        else:
            hi = mid
    if best[0] > 1e-8:
        raise ArithmeticError("compact best-response equality residual too large")
    return best[1]


def optimistic_rewards(probabilities, trials, alternatives, block_size, bonus):
    """Original endpoint optimism, in fixed-block KL units.

    Confidence endpoints solve n_shots*kl(phat,endpoint)=bonus. Rewards
    max{bonus/n_blocks, block_size*max_endpoint kl(endpoint,alternative)}.
    Raw empirical means and full [0,1] confidence endpoints are retained.
    """
    rewards = []
    for p, n, q in zip(probabilities, trials, alternatives):
        if n < block_size or n % block_size:
            raise ValueError("optimism requires initialized, complete fixed blocks")
        level = bonus / n
        left = _kl_level_point(p, level, -1)
        right = _kl_level_point(p, level, 1)
        rewards.append(block_size * max(level, bernoulli_kl(left, q), bernoulli_kl(right, q)))
    return rewards


class CombGameState:
    def __init__(self, coefficients, block_size, delta=.01, clip=ALTERNATIVE_CLIP):
        if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
            raise ValueError("block_size must be a positive integer")
        if not 0 < clip < .5:
            raise ValueError("clip must be in (0,.5)")
        self.raw = TrackStopState(coefficients, delta)
        self.coefficients = list(coefficients)
        self.block_size = block_size
        self.clip = clip
        self.learners = [AdaHedge(len(coefficients)), AdaHedge(len(coefficients))]
        self.blocks = [0] * len(coefficients)
        self.cumulative_weights = [0.0] * len(coefficients)
        self.pending_arm = None
        self.terminal = False
        self.last_round = None

    def recommend_arm(self):
        if self.terminal or self.pending_arm is not None:
            raise RuntimeError("terminal state or unconsumed recommendation")
        for arm, count in enumerate(self.blocks):
            if count == 0:
                self.cumulative_weights[arm] += 1.0
                self.pending_arm = arm
                self.last_round = {"initialization": True}
                return arm
        ps = [s / n for s, n in zip(self.raw.successes, self.raw.trials)]
        answer = int(math.fsum(a * p for a, p in zip(self.coefficients, ps)) >= 0)
        learner = self.learners[answer]
        weights = learner.weights()
        self.cumulative_weights = [s + w for s, w in zip(self.cumulative_weights, weights)]
        arm = min((i for i, w in enumerate(self.cumulative_weights) if w > 0),
                  key=lambda i: (self.blocks[i] / self.cumulative_weights[i], i))
        alternative = best_response(ps, weights, self.coefficients, self.clip,
                                    answer_sign=1 if answer else -1)
        bonus = exploration_bonus(sum(self.blocks))
        rewards = optimistic_rewards(ps, self.raw.trials, alternative, self.block_size, bonus)
        learner.update(rewards)
        self.pending_arm = arm
        self.last_round = {"initialization": False, "answer": answer,
                           "weights": weights, "alternative": alternative,
                           "rewards": rewards, "bonus": bonus}
        return arm

    def update(self, arm, successes, trials, final_partial=False):
        if self.terminal or arm != self.pending_arm:
            raise RuntimeError("update must consume the pending recommendation")
        if final_partial:
            if not 0 < trials < self.block_size:
                raise ValueError("final partial block must be shorter than a full block")
        elif trials != self.block_size:
            raise ValueError("regular rounds require exactly block_size shots")
        self.raw.update(arm, successes, trials)
        if final_partial:
            self.terminal = True
        else:
            self.blocks[arm] += 1
        self.pending_arm = None

    def certificate(self, full_audit=False):
        return self.raw.certificate(full_audit=full_audit)


def run_combgame(config, sample_arm, macro_budget=DEFAULT_MACRO_BUDGET):
    """Drop-in policy runner; config exposes only coefficients/delta/shot_budget.

    macro_budget=512 was fixed before confirmation. b=ceil(B/512) depends
    only on the public common shot cap. Initialization is included in cost.
    """
    if isinstance(macro_budget, bool) or not isinstance(macro_budget, int) or macro_budget < 1:
        raise ValueError("macro_budget must be a positive integer")
    budget = config.shot_budget
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError("shot_budget must be a positive integer")
    block_size = (budget + macro_budget - 1) // macro_budget
    state = CombGameState(config.coefficients, block_size, config.delta)
    total, checks, partial_shots = 0, 0, 0
    certificate = None
    while total < budget:
        arm = state.recommend_arm()
        count = min(block_size, budget - total)
        partial = count < block_size
        successes = sample_arm(arm, count)
        state.update(arm, successes, count, final_partial=partial)
        total += count
        checks += 1
        if partial:
            partial_shots = count
        certificate = state.certificate(full_audit=total == budget)
        if certificate["decision"] != "abstain":
            break
    return {"decision": certificate["decision"], "certificate": certificate,
            "checks": checks, "checkpoint_growth": None, "guarantee_in_scope": True,
            "arm_successes": state.raw.successes, "arm_trials": state.raw.trials,
            "sampling_metadata": {
                "algorithm": "CombGame-AdaHedge singleton two-answer specialization",
                "algorithm_source": COMBGAME_SOURCE,
                "algorithm_parts": "Algorithm 1; sections 3.2-3.3; Appendix G.1/G.5.1",
                "learner_source": ADAHEDGE_SOURCE,
                "learner_parts": "equation (8), Figure 1; one learner per answer",
                "tracking": "argmin N_previous[i]/cumulative_weight[i]",
                "macro_budget": macro_budget, "block_size": block_size,
                "block_rule": "ceil(public_shot_budget / macro_budget)",
                "block_kl": "block_size * bernoulli_kl",
                "initialization": "one paid full block per arm, budget permitting",
                "complete_blocks": sum(state.blocks), "final_partial_shots": partial_shots,
                "partial_rule": "last partial block updates raw counts for final certificate only",
                "exploration_bonus": "Wbar(4*log(completed_blocks)); Wbar(x)-log(Wbar(x))=x",
                "alternative_probability_clip": state.clip,
                "empirical_mean_smoothing": None, "learner_weight_floor": None,
                "numerical_details": "binary64 KKT root and softmax; underflow zeros allowed; clipped Jensen gap",
                "stopping": "shared raw-count outward-Decimal joint Beta-mixture GLR",
                "stopping_threshold_substituted": True,
                "original_asymptotic_theorem_claimed": False,
                "answer_learner_rounds": [learner.rounds for learner in state.learners],
            }}
