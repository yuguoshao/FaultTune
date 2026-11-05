"""Implementation audit of a completed unequal-allocation development run.

Every RNG stream, paid row, predictable planning call, allocation, checkpoint
screen and terminal Decimal certificate is rebuilt. Frozen planner and
normalizer functions are reused; this is an implementation/data-flow audit,
not an independent theoretical proof or independent optimization algorithm.
"""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import groupby
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

import unequal_runner as runner
import weighted_fisher as evidence


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def equal_values(actual, expected, path="value"):
    if isinstance(expected, dict):
        require(set(actual) == set(expected), path+" keys differ")
        for key in expected:
            if key != "elapsed_seconds":
                equal_values(actual[key], expected[key], path+"."+key)
    elif isinstance(expected, list):
        require(len(actual) == len(expected), path+" length differs")
        for index, (a, e) in enumerate(zip(actual, expected)):
            equal_values(a, e, path+"["+str(index)+"]")
    elif isinstance(expected, float):
        require(math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12),
                path+" numeric value differs")
    else:
        require(actual == expected, path+" differs")


def add_row(rows, arm, n, k):
    if not n:
        return
    coordinate = tuple(arm[name] for name in runner.FEATURE_KEYS)
    for row in rows:
        if tuple(row[name] for name in runner.FEATURE_KEYS) == coordinate:
            row["trials"] += n
            row["successes"] += k
            return
    rows.append(dict(arm, trials=n, successes=k))


def phase_schedule(policy, previous, budget, explored, spec):
    shots = budget-previous
    if policy == "psr" or previous < spec["explore_after_previous_shots"]:
        return [("psr", shots)]
    target = int(math.ceil(spec["exploration_sqrt_multiplier"]*math.sqrt(budget/2)))
    needed = min(shots//2, max(0, target-explored))
    return [(phase, count) for phase, count in (
        ("explore_a", 2*((needed+1)//2)), ("explore_b", 2*(needed//2)),
        ("exploit", shots-2*needed)) if count]


def audit_record(record, case, spec):
    policy, rep = record["policy"], record["rep"]
    seed = spec["seed_base"]+1009*record["case_id"]+1000003*rep
    require(record["seed"] == seed, "seed rule mismatch")
    require(record["method"] == policy+"_weighted_conditional_unequal", "method label mismatch")
    rng = np.random.RandomState(seed)
    rows, actual_plans = [], record["plans"]
    current_plan, regime, next_replan, planned = None, None, 0, 0
    state = evidence.empty_state(spec["evidence_protocol"]["experts"])
    previous, explored, exact_score, look_count = 0, 0, Fraction(0), 0
    blocks_seen, checked_screens, cert_calls = 0, 0, 0
    prefix, final_certificate = [], None
    for look, grouped in groupby(record["history"], key=lambda block: block["look"]):
        look_count += 1
        require(look == look_count, "nonconsecutive look")
        budget = spec["checkpoints"][look-1]
        blocks = list(grouped)
        phases = phase_schedule(policy, previous, budget, explored, spec)
        require([(b["phase"], b["nminus"]+b["nplus"]) for b in blocks] == phases,
                "phase budget or exploration accounting differs")
        for block in blocks:
            require(block["budget"] == budget, "block budget mismatch")
            phase = block["phase"]
            count = block["nminus"]+block["nplus"]
            if phase.startswith("explore"):
                expected_arm = spec["exploration_coordinates"][0 if phase == "explore_a" else 1]
                weight, expected_plan_id = .5, None
            else:
                effective_policy = "psr" if phase == "psr" else policy
                paid_shots = sum(row["trials"] for row in rows)
                if current_plan is None or regime != effective_policy or paid_shots >= next_replan:
                    expected_plan = runner.plan_from_rows(effective_policy, rows)
                    expected_plan["plan_id"], expected_plan["look"] = planned, look
                    require(planned < len(actual_plans), "missing predictable plan")
                    equal_values(actual_plans[planned], expected_plan, "plan")
                    current_plan = actual_plans[planned]
                    planned += 1
                    regime, next_replan = effective_policy, max(32, 2*paid_shots)
                expected_arm = current_plan["positive_arm"]
                weight, expected_plan_id = current_plan["weight_plus"], current_plan["plan_id"]
            require(block["positive_arm"] == expected_arm and block["plan_id"] == expected_plan_id,
                    "action does not follow previous paid-data plan")
            require(block["planned_weight_plus"] == weight, "planned weight changed")
            plus_count = int(math.floor(count*weight+.5))
            if count >= 2:
                plus_count = min(count-1, max(1, plus_count))
            minus_count = count-plus_count
            require((block["nminus"], block["nplus"]) == (minus_count, plus_count), "rounded allocation differs")
            c, s, d = (expected_arm[name] for name in runner.FEATURE_KEYS)
            require(all(type(v) is int for v in (c, s, d)) and d > 0 and s > 0 and c*c+s*s == d*d,
                    "query is not an exact positively oriented rational circle pair")
            pm, pp = case["A"]+case["B"]*(c/d)-case["C"]*(s/d), case["A"]+case["B"]*(c/d)+case["C"]*(s/d)
            replay_minus = int(rng.binomial(minus_count, pm))
            replay_plus = int(rng.binomial(plus_count, pp))
            require((replay_minus, replay_plus) == (block["minus_successes"], block["plus_successes"]),
                    "independent replicate RNG replay differs")
            evidence.update_block(state, minus_count, plus_count, replay_minus, replay_plus)
            exact_score += Fraction(replay_plus)-Fraction((replay_minus+replay_plus)*plus_count, count)
            add_row(rows, expected_arm, plus_count, replay_plus)
            negative = dict(expected_arm, sin_numerator=-s)
            add_row(rows, negative, minus_count, replay_minus)
            if phase.startswith("explore"):
                explored += count//2
            if previous < spec["explore_after_previous_shots"]:
                prefix.append(block)
            blocks_seen += 1
        require(state["shots"] == budget == sum(r["trials"] for r in rows), "unpaid or duplicated shots")
        require(state["score"] == exact_score, "exact Fraction score differs")
        screened = evidence.screen(state, spec["delta"])
        checked_screens += 1
        if screened["crossing_candidate"] or budget == spec["cap"]:
            cert = evidence.certificate_from_blocks(state["blocks"], spec["delta"], state["experts"])
            cert_calls += 1
            if cert["decision"] != "abstain" or budget == spec["cap"]:
                require(budget == record["shots"], "missed earlier certified stopping checkpoint")
                require(cert == record["certificate"], "terminal outward certificate differs")
                require(cert["decision"] == record["decision"], "terminal decision mismatch")
                final_certificate = cert
        require(final_certificate is None or block is record["history"][-1], "extra data recorded after stopping")
        previous = budget
    require(final_certificate is not None and previous == record["shots"], "missing valid terminal event")
    require(planned == len(actual_plans), "unexecuted plans recorded")
    require(rows == record["final_rows"] and explored == record["exploration_pairs"], "final paid state mismatch")
    require(record["full_grid_replans"] == sum(p["policy"] == "grid" for p in actual_plans), "grid plan count mismatch")
    require(record["pair_fallback_replans"] == sum(p["pair_fallback"] for p in actual_plans), "fallback count mismatch")
    return {"blocks": blocks_seen, "screens": checked_screens, "certificates": cert_calls,
            "plans": planned, "prefix": prefix}


def audit(directory, output):
    started = time.perf_counter()
    directory, output = Path(directory).resolve(), Path(output).resolve()
    require(not output.exists(), "preserve completed audit output")
    paths = {name: directory/name for name in ("spec.json", "summary.json", "trials.jsonl")}
    inputs = {name: digest(path) for name, path in paths.items()}
    own_hash = digest(__file__)
    spec, summary = [json.loads(paths[name].read_text(encoding="utf-8")) for name in ("spec.json", "summary.json")]
    require(summary["spec"] == spec, "spec/summary mismatch")
    require(spec["sources"] == runner.source_hashes(), "run source has changed")
    require(spec["evidence_protocol"] == evidence.protocol_config(), "weighted evidence prior changed")
    require(spec["checkpoints"] == runner.schedule(spec["cap"]), "checkpoint schedule changed")
    cases = {c["case_id"]: c for c in spec["cases"]}
    expected_keys = {(case_id, policy, rep) for case_id in cases for policy in spec["policies"]
                     for rep in range(spec["repetitions"])}
    seen, cells, prefixes = set(), defaultdict(list), {}
    totals = Counter()
    with paths["trials.jsonl"].open(encoding="utf-8") as ledger:
        for line in ledger:
            record = json.loads(line)
            key = record["case_id"], record["policy"], record["rep"]
            require(key in expected_keys and key not in seen, "duplicate or unexpected trajectory")
            seen.add(key)
            checked = audit_record(record, cases[key[0]], spec)
            for metric in ("blocks", "screens", "certificates", "plans"):
                totals[metric] += checked[metric]
            pair_key = key[0], key[2]
            prefix = checked["prefix"]
            if pair_key in prefixes:
                previous_prefix = prefixes[pair_key]
                common = min(len(prefix), len(previous_prefix))
                require(prefix[:common] == previous_prefix[:common], "policy streams differ before planned divergence")
                totals["shared_prefix_checks"] += 1
                if len(prefix) > len(previous_prefix):
                    prefixes[pair_key] = prefix
            else:
                prefixes[pair_key] = prefix
            cells[(key[0], record["method"])].append(record)
            totals["seed_replayed_trajectories"] += 1
            if key[2] == spec["repetitions"]-1:
                print(json.dumps({"audited_case": key[0], "policy": key[1], "trajectories": len(seen),
                                  "elapsed_seconds": time.perf_counter()-started}), flush=True)
    require(seen == expected_keys and len(seen) == summary["new_trajectories"], "incomplete record collection")
    reported = {(c["case_id"], c["method"]): c for c in summary["cells"]}
    require(set(reported) == set(cells) and len(reported) == len(summary["cells"]), "summary cell identities differ")
    outcomes = Counter()
    for (case_id, method), records in cells.items():
        case = cases[case_id]
        expected = "accept" if case["C"] < 0 else "reject" if case["C"] > 0 else "abstain"
        rebuilt = {"case_id": case_id, "kind": case["kind"], "method": method,
                   "repetitions": len(records), "mean_shots": float(np.mean([r["shots"] for r in records])),
                   "wrong": sum(r["decision"] not in ("abstain", expected) for r in records),
                   "abstained": sum(r["decision"] == "abstain" for r in records),
                   "full_grid_replans": sum(r["full_grid_replans"] for r in records),
                   "pair_fallback_replans": sum(r["pair_fallback_replans"] for r in records)}
        require(reported[(case_id, method)] == rebuilt, "summary metric differs from unconditional ledger costs")
        outcomes["wrong"] += rebuilt["wrong"]
        outcomes["abstained"] += rebuilt["abstained"]
        outcomes["paid_shots"] += sum(r["shots"] for r in records)
    require(inputs == {name: digest(path) for name, path in paths.items()}, "input files changed during audit")
    require(own_hash == digest(__file__) and spec["sources"] == runner.source_hashes(), "audit/run source changed during audit")
    result = {"status": "passed", "audit_type": "implementation and data-flow audit, not independent theoretical verification",
              "directory": str(directory), "trajectories": len(seen), "checks": dict(totals),
              "outcomes": dict(outcomes), "input_sha256": inputs, "audit_source_sha256": own_hash,
              "all_rng_streams_replayed": True, "all_predictable_actions_rebuilt_from_paid_counts": True,
              "all_rounded_allocations_and_costs_reconstructed": True,
              "all_first_certified_stops_recomputed": True, "summary_cells_rebuilt": True,
              "limits": "Reuses frozen planner and certified normalizer routines; their mathematical correctness is reviewed separately. No power or SOTA claim.",
              "elapsed_seconds": time.perf_counter()-started}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audit(args.directory, args.output or args.directory/"audit.json")
