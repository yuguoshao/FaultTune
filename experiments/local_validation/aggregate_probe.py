"""Validate the SUM of multiple monitor changes with legal parallel actions.

Python 3.8+, standard library. These are classical Chernoff/peeling baselines,
not claimed novel algorithms. All probability statements require independent,
stationary Bernoulli streams, a fixed proposal, and known p0 and local gain.
"""

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from group_oracle import (GroupMeasurementOracle, PublicGroupConfig,
                          clique_graph, empty_graph, path_graph)
from probe import BudgetExhausted, stable_seed, wilson_interval
from stress_probe import harm_boundary_scan


def kl(p, q):
    """Bernoulli KL including empirical endpoints, with natural logarithms."""
    if p == q:
        return 0.0
    if q <= 0.0 or q >= 1.0:
        return float("inf")
    if p == 0.0:
        return -math.log1p(-q)
    if p == 1.0:
        return -math.log(q)
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


def kl_interval(successes, observations, alpha):
    """Chernoff interval for the average mean of independent Bernoulli bits.

    The bits need not have equal probabilities. Bisection returns outer
    bracket endpoints and adds a small floating-point margin. It is not a
    formal interval-arithmetic implementation.
    """
    if observations < 1 or not 0 <= successes <= observations or not 0.0 < alpha < 1.0:
        raise ValueError("invalid count or error budget")
    empirical = successes / observations
    threshold = math.log(2.0 / alpha) / observations
    if empirical == 0.0:
        lower = 0.0
    else:
        left, right = 0.0, empirical
        for _ in range(50):
            middle = (left + right) / 2.0
            if kl(empirical, middle) > threshold:
                left = middle
            else:
                right = middle
        lower = left
    if empirical == 1.0:
        upper = 1.0
    else:
        left, right = empirical, 1.0
        for _ in range(50):
            middle = (left + right) / 2.0
            if kl(empirical, middle) > threshold:
                right = middle
            else:
                left = middle
        upper = right
    return max(0.0, lower - 1e-12), min(1.0, upper + 1e-12)


def checkpoint_alpha(delta, index, groups=1):
    return delta / (groups * (index + 1) * (index + 2))


def interval_decision(interval):
    lower, upper = interval
    if upper < 0.0:
        return "accept"
    if lower > 0.0:
        return "reject"
    return None


def sweep_kl(config, measure):
    """Pool all objects after balanced sweeps, checking only at 1,2,4,... ."""
    sweeps = successes = checkpoint = 0
    next_check = 1
    latest = [-config.n * config.p0 - config.local_gain,
              config.n * (1.0 - config.p0) - config.local_gain]
    try:
        while True:
            # Every completed sweep purchases exactly one observation per arm.
            # A budget interruption may consume part of a sweep; that partial
            # data are charged but not used in a balanced-checkpoint certificate.
            for group in config.partition:
                successes += sum(measure(group))
            sweeps += 1
            if sweeps != next_check:
                continue
            lower, upper = kl_interval(successes, config.n * sweeps,
                                       checkpoint_alpha(config.delta, checkpoint))
            offset = config.n * config.p0 + config.local_gain
            latest = [config.n * lower - offset, config.n * upper - offset]
            decision = interval_decision(latest)
            if decision:
                return {"decision": decision, "interval": latest,
                        "checkpoint_counts": [sweeps] * len(config.partition)}
            checkpoint += 1
            next_check *= 2
    except BudgetExhausted:
        return {"decision": "abstain", "interval": latest,
                "checkpoint_counts": [0 if sweeps == 0 else 2 ** (sweeps.bit_length() - 1)] * len(config.partition)}


def group_kl(config, measure, adaptive):
    """Sum simultaneously valid group intervals; optionally adapt next group.

    Selection maximizes current contribution width / extra rounds to double
    that group's sample count. This is a heuristic for this rectangular
    certificate, not an information-optimal sampling rule.
    """
    groups = config.partition
    counts = [0] * len(groups)
    successes = [0] * len(groups)
    intervals = [[0.0, float(len(group))] for group in groups]
    offset = config.n * config.p0 + config.local_gain

    def current_interval():
        return [math.fsum(value[0] for value in intervals) - offset,
                math.fsum(value[1] for value in intervals) - offset]

    try:
        while True:
            if 0 in counts:
                chosen = counts.index(0)
            elif adaptive:
                chosen = max(range(len(groups)),
                             key=lambda g: ((intervals[g][1] - intervals[g][0]) / counts[g], -g))
            else:
                chosen = min(range(len(groups)), key=lambda g: (counts[g], g))
            previous = counts[chosen]
            target = 1 if previous == 0 else 2 * previous
            # counts records completed checkpoints only. Incomplete blocks
            # still incur oracle cost, without reusing an unallocated test.
            for _ in range(target - previous):
                successes[chosen] += sum(measure(groups[chosen]))
            counts[chosen] = target
            index = target.bit_length() - 1
            alpha = checkpoint_alpha(config.delta, index, len(groups))
            lower, upper = kl_interval(successes[chosen], target * len(groups[chosen]), alpha)
            intervals[chosen] = [len(groups[chosen]) * lower, len(groups[chosen]) * upper]
            latest = current_interval()
            decision = interval_decision(latest)
            if decision:
                return {"decision": decision, "interval": latest,
                        "checkpoint_counts": counts[:]}
    except BudgetExhausted:
        return {"decision": "abstain", "interval": current_interval(),
                "checkpoint_counts": counts[:]}


def group_kl_balanced(config, measure):
    return group_kl(config, measure, False)


def group_kl_adaptive(config, measure):
    return group_kl(config, measure, True)


def legacy_guard(config, measure):
    """An intentionally out-of-scope single-effect test on a multi-effect family."""
    if config.p0 + config.local_gain >= 1.0:
        raise ValueError("legacy guard requires p0 + local_gain < 1")
    try:
        decision = harm_boundary_scan(config, lambda index: measure((index,))[0])
    except BudgetExhausted:
        decision = "abstain"
    return {"decision": decision, "interval": None, "checkpoint_counts": None}


POLICIES = {"sweep_kl": sweep_kl, "group_kl_balanced": group_kl_balanced,
            "group_kl_adaptive": group_kl_adaptive, "legacy_guard": legacy_guard}
def star_graph(n):
    return tuple((0, index) for index in range(1, n))


GRAPHS = {"empty": empty_graph, "path": path_graph, "clique": clique_graph,
          "star": star_graph}
CASES = ("null", "four_cluster", "four_spread", "many_weak", "beneficial_weak")


def case_probabilities(config, case):
    if config.n < 8:
        raise ValueError("the benchmark cases require n >= 8")
    probabilities = [config.p0] * config.n
    if case == "null":
        pass
    elif case in ("four_cluster", "four_spread", "beneficial_weak"):
        # For a path with even n, the cluster belongs to one color class.
        positions = [0, 2, 4, 6] if case == "four_cluster" else [0, 1, config.n - 2, config.n - 1]
        bump = config.local_gain / (8.0 if case == "beneficial_weak" else 3.0)
        for position in positions:
            probabilities[position] += bump
    elif case == "many_weak":
        probabilities = [config.p0 + 2.0 * config.local_gain / config.n] * config.n
    else:
        raise ValueError("unknown case")
    if any(p >= 1.0 for p in probabilities):
        raise ValueError("case probabilities must be below one")
    return probabilities


def run_trial(policy, config, probabilities, seed, round_budget=None,
              object_budget=None, transcripts=None):
    if policy not in POLICIES:
        raise ValueError("unknown policy")
    oracle = GroupMeasurementOracle(config, probabilities, seed, round_budget,
                                    object_budget, transcripts)
    start = time.perf_counter()
    result = POLICIES[policy](config, oracle.measure)
    elapsed = time.perf_counter() - start
    audit = oracle.audit()
    result.update({"policy": policy, "n": config.n, "seed": str(seed),
                   "rounds": oracle.rounds, "objects": oracle.objects,
                   "switches": oracle.switches, "counts": oracle.counts,
                   "elapsed_seconds": elapsed, "audit": audit,
                   "guarantee_in_scope": policy != "legacy_guard"})
    return result


def summarize(records):
    first = records[0]
    total = len(records)
    expected = first["audit"]["expected_decision"]
    if any(r["audit"]["expected_decision"] != expected for r in records):
        raise ValueError("cannot pool opposite target signs")

    def rate(count):
        return {"count": count, "rate": count / total,
                "wilson_95": list(wilson_interval(count, total))}

    def numbers(key):
        values = sorted(r[key] for r in records)
        location = 0.95 * (len(values) - 1)
        low = int(location)
        p95 = values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (location - low)
        return {"mean": statistics.mean(values), "median": statistics.median(values),
                "p95": p95, "max": max(values)}

    return {"policy": first["policy"], "n": first["n"], "graph": first["graph"],
            "case": first["case"], "trials": total, "expected_decision": expected,
            "true_delta": first["audit"]["true_delta"],
            "guarantee_in_scope": first["guarantee_in_scope"],
            "decisions": {decision: rate(sum(r["decision"] == decision for r in records))
                          for decision in ("accept", "reject", "abstain")},
            "wrong": rate(sum(r["decision"] not in (expected, "abstain") for r in records)),
            "rounds": numbers("rounds"), "objects": numbers("objects"),
            "switches": numbers("switches"), "elapsed_seconds": numbers("elapsed_seconds")}


def null_bound(n, p0, gain, delta, group_count):
    """Analytic anchor for equal disjoint groups, homogeneous null.

    A boundary alternative p_j=p0+gain/n attains the inner KL infimum.
    This value is not a generic-graph or a non-null numerical optimum.
    """
    if n % group_count or not 0.0 < p0 < p0 + gain / n < 1.0:
        raise ValueError("anchor requires equal groups and a feasible boundary")
    divergence = kl(p0, p0 + gain / n)
    kappa = kl(1.0 - delta, delta)
    return {"rounds_lower": kappa / ((n / group_count) * divergence),
            "objects_lower": kappa / divergence,
            "scope": "equal-group action family; complete uniform delta-correct tests on all nonzero sum changes"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--p0", type=float, default=0.1)
    parser.add_argument("--gain", type=float, default=0.3)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--graphs", nargs="+", choices=list(GRAPHS), default=["path"])
    parser.add_argument("--cases", nargs="+", choices=CASES,
                        default=["null", "four_cluster", "four_spread", "many_weak"])
    parser.add_argument("--policies", nargs="+", choices=list(POLICIES), default=list(POLICIES))
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--round-budget", type=int, default=262144)
    parser.add_argument("--object-budget", type=int, default=262144)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path)
    args = parser.parse_args()
    if args.trials < 1 or args.n < 8:
        parser.error("require trials >= 1 and n >= 8")
    groups, configs = [], []
    raw = None
    try:
        if args.raw_output:
            args.raw_output.parent.mkdir(parents=True, exist_ok=True)
            raw = args.raw_output.open("w", encoding="utf-8")
        for graph in args.graphs:
            config = PublicGroupConfig(args.n, args.p0, args.gain, args.delta, GRAPHS[graph](args.n))
            configs.append({"graph": graph, "edges": config.conflict_edges, "partition": config.partition,
                            "null_bound": null_bound(args.n, args.p0, args.gain, args.delta, len(config.partition))
                            if args.n % len(config.partition) == 0 and len({len(g) for g in config.partition}) == 1 else None})
            for case in args.cases:
                probabilities = case_probabilities(config, case)
                rows = {policy: [] for policy in args.policies}
                for trial in range(args.trials):
                    seed = stable_seed(args.seed, args.n, trial)
                    for policy in args.policies:
                        row = run_trial(policy, config, probabilities, seed,
                                        args.round_budget, args.object_budget)
                        row.update({"graph": graph, "case": case, "trial": trial})
                        rows[policy].append(row)
                        if raw:
                            raw.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                groups.extend(summarize(rows[policy]) for policy in args.policies)
                print("completed {} {} ({} trials per policy)".format(graph, case, args.trials), file=sys.stderr)
    finally:
        if raw:
            raw.close()
    result = {"schema_version": "faulttune.aggregate_validation.v1",
              "metadata": {"n": args.n, "p0": args.p0, "gain": args.gain, "delta": args.delta,
                           "trials_per_group": args.trials, "seed": args.seed,
                           "round_budget": args.round_budget, "object_budget": args.object_budget,
                           "graphs": args.graphs, "cases": args.cases, "policies": args.policies,
                           "python": sys.version.split()[0], "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "confidence": "fixed geometric checkpoints; summable error budgets; independent stationary Bernoulli; double-precision outer bracket with numerical margin",
                           "cost": "rounds and scalar objects charged separately; switches exclude first action; elapsed is Python runtime, not device time",
                           "oracle": "conflict graph controls simultaneous actions; each monitor distribution is action-independent; known p0 and gain",
                           "pairing": "same per-arm kth random uniform across policies, graphs and cases"},
              "configurations": configs, "groups": groups}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
