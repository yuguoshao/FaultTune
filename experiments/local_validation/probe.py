"""A measurement-only probe of validating a locally improved proposal.

Python 3.8+, standard library only. The nominal family has either no anomalous
arm or exactly one arm with Bernoulli probability p1; other arms have p0.
The hidden audit objective change is -local_gain + (p1 - p0) for an anomaly,
and -local_gain otherwise. Lower objective values are better.

Policies receive ONLY PublicConfig and a measure(index) callable. Hidden state
and the true objective change belong to the runner, never to a policy. This is
an API separation for reproducible experiments, not a hostile-code sandbox.

Example (write results outside the source file):
  python probe.py --sizes 8 32 128 --trials 200 --output aggregate.json
  python probe.py --sizes 8 --trials 20 --budget 100 --output budget.json

run_trial also supports fixed per-arm transcripts for deterministic tests and
an actual_p1 override for SEPARATE model-misspecification experiments. The CLI
always samples the nominal family; off-model data are not pooled into it.
"""

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple


PROBE_VERSION = "0.1.0"
SCHEMA_VERSION = "faulttune.local_validation.v1"
DECISIONS = ("accept", "reject", "abstain")
CASES = ("null", "first", "middle", "last")
Measure = Callable[[int], int]


@dataclass(frozen=True)
class PublicConfig:
    n: int
    p0: float = 0.1
    p1: float = 0.4
    local_gain: float = 0.15
    delta: float = 0.05

    def __post_init__(self) -> None:
        if isinstance(self.n, bool) or not isinstance(self.n, int) or self.n < 1:
            raise ValueError("n must be a positive integer")
        if not (0.0 < self.p0 < self.p1 < 1.0):
            raise ValueError("require 0 < p0 < p1 < 1")
        if not (0.0 < self.local_gain < self.p1 - self.p0):
            raise ValueError("require 0 < local_gain < p1 - p0")
        if not (0.0 < self.delta < 0.5):
            raise ValueError("require 0 < delta < 0.5")


class BudgetExhausted(RuntimeError):
    """The next measurement would exceed the allowed number of shots."""


class TranscriptExhausted(RuntimeError):
    """A deterministic test requested a missing transcript observation."""


def stable_seed(*parts: Any) -> int:
    """Stable integer seed; does not depend on Python's randomized hash()."""
    encoded = json.dumps(parts, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    digest = hashlib.blake2b(encoded, digest_size=16, person=b"ft-local-valid").digest()
    return int.from_bytes(digest, "big")


class MeasurementOracle:
    """Private environment. Pass only its measure callable to policies.

    Each arm owns an independent Random stream seeded by (instance seed, arm).
    Thus arm j's kth draw is shared by all policies, whatever their schedules.
    With transcripts supplied, EVERY requested observation must be present;
    missing values raise TranscriptExhausted rather than falling back to RNG.
    """

    def __init__(
        self,
        config: PublicConfig,
        hidden: Optional[int],
        seed: int,
        budget: Optional[int] = None,
        transcripts: Optional[Mapping[int, Sequence[int]]] = None,
        actual_p1: Optional[float] = None,
    ) -> None:
        if hidden is not None and (
            isinstance(hidden, bool) or not isinstance(hidden, int) or not 0 <= hidden < config.n
        ):
            raise ValueError("hidden must be None or an arm index")
        if budget is not None and (
            isinstance(budget, bool) or not isinstance(budget, int) or budget < 0
        ):
            raise ValueError("budget must be None or a nonnegative integer")
        actual = config.p1 if actual_p1 is None else actual_p1
        if not (0.0 < actual < 1.0):
            raise ValueError("actual_p1 must lie strictly between 0 and 1")
        self._config = config
        self._hidden = hidden
        self._actual_p1 = actual
        self._seed = seed
        self._budget = budget
        self._counts = [0] * config.n
        self._shots = 0
        self._covered = 0
        self._streams = [None] * config.n  # type: List[Optional[random.Random]]
        self._transcripts = None  # type: Optional[Dict[int, Tuple[int, ...]]]
        if transcripts is not None:
            copied = {}  # type: Dict[int, Tuple[int, ...]]
            for arm, values in transcripts.items():
                if isinstance(arm, bool) or not isinstance(arm, int) or not 0 <= arm < config.n:
                    raise ValueError("transcript key must be an arm index")
                row = tuple(values)
                if any(value not in (0, 1) for value in row):
                    raise ValueError("transcript observations must be zero or one")
                copied[arm] = row
            self._transcripts = copied

    def measure(self, index: int) -> int:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < self._config.n:
            raise ValueError("measurement index is outside the arm set")
        if self._budget is not None and self._shots >= self._budget:
            raise BudgetExhausted("measurement budget exhausted")
        count = self._counts[index]
        if self._transcripts is not None:
            row = self._transcripts.get(index, ())
            if count >= len(row):
                raise TranscriptExhausted("arm {} has no observation {}".format(index, count))
            observation = int(row[count])
        else:
            stream = self._streams[index]
            if stream is None:
                stream = random.Random(stable_seed(self._seed, "arm", index))
                self._streams[index] = stream
            probability = self._actual_p1 if index == self._hidden else self._config.p0
            observation = int(stream.random() < probability)
        self._shots += 1
        self._counts[index] = count + 1
        if count == 0:
            self._covered += 1
        return observation

    @property
    def shots(self) -> int:
        return self._shots

    @property
    def covered(self) -> int:
        return self._covered

    @property
    def counts(self) -> Tuple[int, ...]:
        return tuple(self._counts)

    @property
    def max_samples(self) -> int:
        return max(self._counts)

    def audit(self) -> Dict[str, Any]:
        change = -self._config.local_gain
        if self._hidden is not None:
            change += self._actual_p1 - self._config.p0
        return {
            "hidden": self._hidden,
            "actual_p1": self._actual_p1,
            "true_delta": change,
            "expected_decision": "accept" if change <= 0.0 else "reject",
            "nominal_model": self._actual_p1 == self._config.p1,
            "observation_source": "random" if self._transcripts is None else "fixed_transcripts",
        }


def fixed_shots_per_arm(config: PublicConfig) -> int:
    """Hoeffding midpoint test: each arm's relevant error is at most delta/n."""
    gap = config.p1 - config.p0
    return int(math.ceil(2.0 / (gap * gap) * math.log(config.n / config.delta)))


def full_fixed(config: PublicConfig, measure: Measure) -> str:
    """Always measure all arms at fixed depth before deciding."""
    depth = fixed_shots_per_arm(config)
    threshold = depth * (config.p0 + config.p1) / 2.0
    any_anomaly = False
    for arm in range(config.n):
        successes = 0
        for _ in range(depth):
            successes += measure(arm)
        any_anomaly = any_anomaly or successes >= threshold
    return "reject" if any_anomaly else "accept"


def fixed_scan(config: PublicConfig, measure: Measure) -> str:
    """Fixed depth for each visited arm; stop when its midpoint test rejects."""
    depth = fixed_shots_per_arm(config)
    threshold = depth * (config.p0 + config.p1) / 2.0
    for arm in range(config.n):
        successes = 0
        for _ in range(depth):
            successes += measure(arm)
        if successes >= threshold:
            return "reject"
    return "accept"


def binomial_probabilities(n: int, p: float) -> List[float]:
    """Binomial(n,p) PMF, evaluated in floating point without simulation.

    Start at a mode with relative weight one, recurse toward both tails, then
    normalize. This avoids starting with (1-p)**n, which can underflow even
    when probabilities near the mode are substantial.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError("binomial n must be a nonnegative integer")
    if not 0.0 < p < 1.0:
        raise ValueError("binomial p must lie strictly between zero and one")
    mode = min(n, int(math.floor((n + 1) * p)))
    weights = [0.0] * (n + 1)
    weights[mode] = 1.0
    odds = p / (1.0 - p)
    for k in range(mode, 0, -1):
        weights[k - 1] = weights[k] * k / (n - k + 1) / odds
    for k in range(mode, n):
        weights[k + 1] = weights[k] * (n - k) / (k + 1) * odds
    total = math.fsum(weights)
    return [weight / total for weight in weights]


@lru_cache(maxsize=128)
def _fixed_exact_design(n_arms: int, p0: float, p1: float, delta: float) -> Tuple[int, int, float, float, float, float]:
    target = delta - min(1e-12, delta * 1e-6)
    depth = 0
    while True:
        depth += 1
        pmf0 = binomial_probabilities(depth, p0)
        pmf1 = binomial_probabilities(depth, p1)
        tail0 = [0.0] * (depth + 2)
        lower1 = [0.0] * (depth + 2)
        for k in range(depth, -1, -1):
            tail0[k] = min(1.0, math.fsum((tail0[k + 1], pmf0[k])))
        for k in range(depth + 1):
            lower1[k + 1] = min(1.0, math.fsum((lower1[k], pmf1[k])))
        for threshold in range(1, depth + 1):
            r0 = tail0[threshold]
            r1 = lower1[threshold]
            null_error = 1.0 if r0 >= 1.0 else -math.expm1(n_arms * math.log1p(-r0))
            if null_error > target:
                continue
            alternative_error = r1 * (1.0 - r0) ** (n_arms - 1)
            if alternative_error <= target:
                return depth, threshold, r0, r1, null_error, alternative_error
            # This is the smallest threshold meeting the null requirement.
            # Larger thresholds increase both r1 and (1-r0), so cannot help.
            break


def fixed_exact_design(config: PublicConfig) -> Dict[str, Any]:
    """Smallest fixed depth meeting both exact family error inequalities.

    'Exact' refers to the binomial model rather than a concentration bound;
    numerical evaluation uses double precision with a conservative 1e-12
    probability margin (scaled down when delta is below 1e-6). With rejection when ANY arm
    has S >= threshold, null error is 1-(1-r0)**n and one-anomaly acceptance
    error is r1*(1-r0)**(n-1). This accounts for independent observations.
    """
    depth, threshold, r0, r1, null_error, alternative_error = _fixed_exact_design(
        config.n, config.p0, config.p1, config.delta
    )
    return {
        "shots_per_arm": depth,
        "threshold": threshold,
        "arm_null_reject_probability": r0,
        "arm_anomaly_accept_probability": r1,
        "null_reject_probability": null_error,
        "alternative_accept_probability": alternative_error,
        "probability_margin": min(1e-12, config.delta * 1e-6),
    }


def fixed_exact(config: PublicConfig, measure: Measure) -> str:
    """Full fixed sampling with a binomial-designed integer threshold."""
    design = fixed_exact_design(config)
    any_anomaly = False
    for arm in range(config.n):
        successes = 0
        for _ in range(design["shots_per_arm"]):
            successes += measure(arm)
        any_anomaly = any_anomaly or successes >= design["threshold"]
    return "reject" if any_anomaly else "accept"


def fixed_exact_scan(config: PublicConfig, measure: Measure) -> str:
    """Binomial-designed fixed sampling, rejecting at the first failing arm."""
    design = fixed_exact_design(config)
    for arm in range(config.n):
        successes = 0
        for _ in range(design["shots_per_arm"]):
            successes += measure(arm)
        if successes >= design["threshold"]:
            return "reject"
    return "accept"


def _lr_constants(config: PublicConfig) -> Tuple[float, float, float, float]:
    return (
        math.log(config.p1 / config.p0),
        math.log((1.0 - config.p1) / (1.0 - config.p0)),
        math.log(config.n / config.delta),
        math.log(config.delta),
    )


def sequential_scan(config: PublicConfig, measure: Measure) -> str:
    """Finish one arm's likelihood-ratio test before visiting the next arm."""
    one, zero, upper, lower = _lr_constants(config)
    for arm in range(config.n):
        log_lr = 0.0
        while True:
            log_lr += one if measure(arm) else zero
            if log_lr >= upper:
                return "reject"
            if log_lr <= lower:
                break
    return "accept"


def sequential_round_robin(config: PublicConfig, measure: Measure) -> str:
    """Round-robin among uncleared arms; cleared arms remain frozen."""
    one, zero, upper, lower = _lr_constants(config)
    log_lrs = [0.0] * config.n
    pending = list(range(config.n))
    while pending:
        next_pending = []  # type: List[int]
        for arm in pending:
            log_lrs[arm] += one if measure(arm) else zero
            if log_lrs[arm] >= upper:
                return "reject"
            if log_lrs[arm] > lower:
                next_pending.append(arm)
        pending = next_pending
    return "accept"


def local_only(config: PublicConfig, measure: Measure) -> str:
    """INVALID baseline: a local improvement is accepted without remote checks."""
    return "accept"


POLICIES = {
    "full_fixed": full_fixed,
    "fixed_scan": fixed_scan,
    "fixed_exact": fixed_exact,
    "fixed_exact_scan": fixed_exact_scan,
    "sequential_scan": sequential_scan,
    "sequential_round_robin": sequential_round_robin,
    "local_only": local_only,
}  # type: Dict[str, Callable[[PublicConfig, Measure], str]]


def run_trial(
    policy: str,
    config: PublicConfig,
    hidden: Optional[int],
    seed: int,
    budget: Optional[int] = None,
    transcripts: Optional[Mapping[int, Sequence[int]]] = None,
    actual_p1: Optional[float] = None,
) -> Dict[str, Any]:
    """Run one policy. An exhausted budget produces abstain, never success.

    A decision made on the final available shot is retained: budget exhaustion
    occurs only if the policy requests another observation. Errors in injected
    transcripts propagate, so a short transcript cannot silently mean abstain.
    """
    if policy not in POLICIES:
        raise ValueError("unknown policy: {}".format(policy))
    oracle = MeasurementOracle(config, hidden, seed, budget, transcripts, actual_p1)
    exhausted = False
    try:
        decision = POLICIES[policy](config, oracle.measure)
    except BudgetExhausted:
        decision = "abstain"
        exhausted = True
    if decision not in DECISIONS:
        raise ValueError("policy returned an unknown decision")
    audit = oracle.audit()
    audit["decision_correct"] = decision == audit["expected_decision"]
    audit["policy_valid"] = policy != "local_only"
    return {
        "policy": policy,
        "n": config.n,
        "seed": str(seed),
        "decision": decision,
        "shots": oracle.shots,
        "covered": oracle.covered,
        "max_samples": oracle.max_samples,
        "budget": budget,
        "budget_exhausted": exhausted,
        "audit": audit,
    }


def bernoulli_kl(p: float, q: float) -> float:
    if not (0.0 < p < 1.0 and 0.0 < q < 1.0):
        raise ValueError("KL helper requires interior Bernoulli probabilities")
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


def kl_null_lower_bound(n: int, p0: float = 0.1, p1: float = 0.4, delta: float = 0.05) -> float:
    """Null E[T] lower bound for uniformly delta-correct validation.

    Apply data processing to the accept event for null versus each one-arm
    alternative, then sum the n change-of-measure inequalities. In particular,
    correctness is unconditional: a high null abstention rate invalidates the
    premise; it must not be discarded by conditioning on decided runs.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if not (0.0 < p0 < p1 < 1.0 and 0.0 < delta < 0.5):
        raise ValueError("require 0 < p0 < p1 < 1 and 0 < delta < 0.5")
    return n * bernoulli_kl(1.0 - delta, delta) / bernoulli_kl(p0, p1)


def analytic_bounds(config: PublicConfig) -> Dict[str, float]:
    per_arm_kl = bernoulli_kl(config.p0, config.p1)
    exact = fixed_exact_design(config)
    return {
        "kl_p0_p1": per_arm_kl,
        "kl_decision": bernoulli_kl(1.0 - config.delta, config.delta),
        "null_expected_shots_lower": kl_null_lower_bound(config.n, config.p0, config.p1, config.delta),
        "sequential_null_expected_shots_upper": config.n
        * (math.log(1.0 / config.delta) + math.log((1.0 - config.p0) / (1.0 - config.p1)))
        / per_arm_kl,
        "fixed_shots_per_arm": fixed_shots_per_arm(config),
        "full_fixed_shots": config.n * fixed_shots_per_arm(config),
        "fixed_exact_shots_per_arm": exact["shots_per_arm"],
        "fixed_exact_threshold": exact["threshold"],
        "fixed_exact_full_shots": config.n * exact["shots_per_arm"],
        "fixed_exact_null_error": exact["null_reject_probability"],
        "fixed_exact_alternative_error": exact["alternative_accept_probability"],
    }


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if trials < 1 or successes < 0 or successes > trials:
        raise ValueError("require trials >= 1 and 0 <= successes <= trials")
    rate = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    centre = (rate + z2 / (2.0 * trials)) / denominator
    radius = z * math.sqrt(rate * (1.0 - rate) / trials + z2 / (4.0 * trials * trials)) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def _rate(count: int, total: int) -> Dict[str, Any]:
    return {"count": count, "rate": count / total, "wilson_95": list(wilson_interval(count, total))}


def _number_summary(values: Sequence[int]) -> Dict[str, Any]:
    ordered = sorted(values)

    def quantile(q: float) -> float:
        index = q * (len(ordered) - 1)
        left = int(math.floor(index))
        right = int(math.ceil(index))
        return ordered[left] + (ordered[right] - ordered[left]) * (index - left)

    return {
        "min": ordered[0],
        "median": quantile(0.5),
        "p95": quantile(0.95),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
        "total": sum(ordered),
    }


def aggregate_trials(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not records:
        raise ValueError("cannot aggregate an empty group")
    first = records[0]
    key = (first["policy"], first["n"], first.get("case"), first["budget"])
    for record in records:
        if (record["policy"], record["n"], record.get("case"), record["budget"]) != key:
            raise ValueError("aggregate group mixes policies, sizes, cases or budgets")
        if not record["audit"]["nominal_model"] or record["audit"]["observation_source"] != "random":
            raise ValueError("nominal aggregate excludes off-model runs and injected transcripts")
    total = len(records)
    decisions = {decision: _rate(sum(r["decision"] == decision for r in records), total) for decision in DECISIONS}
    correct = sum(bool(record["audit"]["decision_correct"]) for record in records)
    expected = first["audit"]["expected_decision"]
    if any(record["audit"]["expected_decision"] != expected for record in records):
        raise ValueError("aggregate group mixes expected decisions")
    return {
        "policy": key[0],
        "n": key[1],
        "case": key[2],
        "budget": key[3],
        "trials": total,
        "policy_valid": first["audit"]["policy_valid"],
        "expected_decision": expected,
        "decisions": decisions,
        "correct_unconditional": _rate(correct, total),
        "wrong_decision": _rate(sum(r["decision"] not in (expected, "abstain") for r in records), total),
        "shots": _number_summary([record["shots"] for record in records]),
        "covered": _number_summary([record["covered"] for record in records]),
        "max_samples": _number_summary([record["max_samples"] for record in records]),
    }


def hidden_for_case(case: str, n: int) -> Optional[int]:
    if case == "null":
        return None
    if case == "first":
        return 0
    if case == "middle":
        return n // 2
    if case == "last":
        return n - 1
    raise ValueError("unknown case: {}".format(case))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--policies", nargs="+", choices=list(POLICIES), default=list(POLICIES))
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--budget", type=int, default=None, help="per-run shot budget; default unlimited")
    parser.add_argument("--p0", type=float, default=0.1)
    parser.add_argument("--p1", type=float, default=0.4)
    parser.add_argument("--local-gain", type=float, default=0.15)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=None, help="aggregate JSON path; default stdout")
    parser.add_argument("--raw-output", type=Path, default=None, help="optional per-trial JSONL; no individual shots")
    parser.add_argument("--source-sha", default="unspecified", help="provenance commit supplied by the caller")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.trials < 1:
        parser.error("--trials must be positive")
    if args.budget is not None and args.budget < 0:
        parser.error("--budget must be nonnegative")
    if args.output is not None and args.raw_output is not None and args.output.resolve() == args.raw_output.resolve():
        parser.error("aggregate and raw output paths must differ")
    for name in ("sizes", "cases", "policies"):
        if len(set(getattr(args, name))) != len(getattr(args, name)):
            parser.error("--{} must not contain duplicate entries".format(name))
    try:
        configs = [PublicConfig(n, args.p0, args.p1, args.local_gain, args.delta) for n in args.sizes]
    except ValueError as error:
        parser.error(str(error))
    groups = []  # type: List[Dict[str, Any]]
    raw = None
    try:
        if args.raw_output is not None:
            args.raw_output.parent.mkdir(parents=True, exist_ok=True)
            raw = args.raw_output.open("w", encoding="utf-8", newline="\n")
        for config in configs:
            for case in args.cases:
                hidden = hidden_for_case(case, config.n)
                by_policy = {policy: [] for policy in args.policies}  # type: Dict[str, List[Dict[str, Any]]]
                for trial in range(args.trials):
                    # Cases share background uniforms as well. Changing the hidden arm
                    # changes only that arm's probability threshold, not its RNG draws.
                    seed = stable_seed(args.seed, "instance", config.n, trial)
                    for policy in args.policies:
                        record = run_trial(policy, config, hidden, seed, args.budget)
                        record["case"] = case
                        record["trial"] = trial
                        by_policy[policy].append(record)
                        if raw is not None:
                            raw.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                groups.extend(aggregate_trials(by_policy[policy]) for policy in args.policies)
    finally:
        if raw is not None:
            raw.close()
    result = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "probe_version": PROBE_VERSION,
            "source_sha": args.source_sha,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version.split()[0],
            "base_seed": args.seed,
            "stream_scheme": "BLAKE2b-128 namespace ft-local-valid; independent Random(instance, arm); same kth uniform across policies",
            "model": "null or exactly one anomalous Bernoulli arm; independent observations conditional on hypothesis",
            "p0": args.p0,
            "p1": args.p1,
            "local_gain": args.local_gain,
            "delta": args.delta,
            "sizes": args.sizes,
            "cases": args.cases,
            "policies": args.policies,
            "trials_per_group": args.trials,
            "budget_per_run": args.budget,
            "intervals": "marginal 95% Wilson intervals for each rate; not simultaneous confidence bounds",
            "accuracy": "abstention is never correct; local_only is an invalid unvalidated baseline",
            "lower_bound_premise": "uniform unconditional delta-correctness across null and every one-arm alternative; high abstention can violate this premise",
            "sequential_upper_bound_premise": "nominal model; unlimited-budget likelihood-ratio policy",
            "fixed_exact_design": "minimal depth and threshold from binomial probabilities evaluated in double precision; both family null rejection and one-anomaly acceptance <= delta - min(1e-12, delta*1e-6); floating numerical calibration, not interval arithmetic",
            "quantiles": "linear interpolation at q*(trials-1)",
        },
        "analytic_bounds": {str(config.n): analytic_bounds(config) for config in configs},
        "groups": groups,
    }
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        sys.stdout.write(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
