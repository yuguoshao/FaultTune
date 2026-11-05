"""Development runner for paid pair designs with unequal arm allocation.

PSR, closed-angle, and finite-grid Chernoff policies use the same weighted
conditional evidence. All controls optimize the two-arm sampling allocation.
The grid policy executes its returned weights only when its complete support
is one exact reflected pair; other supports fall back explicitly to the best
grid pair. Planning only receives paid-data fits, never simulator parameters.
"""
import argparse
from datetime import datetime
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from predictive import ROOT, fit_rows, arm_key
from certified_likelihood import rational_circle
from design_policy import GRID, bernoulli_kl, design_from_fit
from probe_fixed import schedule
import weighted_fisher as evidence


POLICIES = ("psr", "closed", "grid")
DELTA = .01
EXPLORE_AFTER = 4096
EXPLORE_MULTIPLIER = 4
FIRST_REPLAN_SHOTS = 32
HALF_ANGLE_DENOMINATOR_CAP = 65536
FEATURE_KEYS = ("cos_numerator", "sin_numerator", "denominator")


def _arm(arm):
    return {key: int(arm[key]) for key in FEATURE_KEYS}


def merge_row(rows, arm, trials, successes):
    if not trials:
        return
    key = arm_key(arm)
    for row in rows:
        if arm_key(row) == key:
            row["trials"] += int(trials)
            row["successes"] += int(successes)
            return
    row = _arm(arm)
    row.update(trials=int(trials), successes=int(successes))
    rows.append(row)


def two_arm_capacity(pminus, pplus):
    """Numerical Chernoff capacity and plus weight, for one or many pairs.

    The derivative of I(w) is KL(pplus||q)-KL(pminus||q), with
    q=(1-w)*pminus+w*pplus. It is decreasing, so bisection is global.
    This is a fit-based design calculation, never a decision certificate.
    """
    pm, pp = np.broadcast_arrays(np.asarray(pminus, dtype=float), np.asarray(pplus, dtype=float))
    if np.any(pm <= 0) or np.any(pm >= 1) or np.any(pp <= 0) or np.any(pp >= 1):
        raise ValueError("capacity inputs must be strictly interior")
    lower, upper = np.zeros_like(pm), np.ones_like(pm)
    tied = np.abs(pp-pm) <= 1e-12
    for unused in range(48):
        weight = .5*(lower+upper)
        q = (1-weight)*pm+weight*pp
        derivative = bernoulli_kl(pp, q)-bernoulli_kl(pm, q)
        lower = np.where(derivative > 0, weight, lower)
        upper = np.where(derivative > 0, upper, weight)
    weight = np.where(tied, .5, .5*(lower+upper))
    q = (1-weight)*pm+weight*pp
    rate = (1-weight)*bernoulli_kl(pm, q)+weight*bernoulli_kl(pp, q)
    return weight, rate


def _pair_predictions(theta, arms):
    cosine = np.array([a["cos_numerator"]/a["denominator"] for a in arms])
    sine = np.array([a["sin_numerator"]/a["denominator"] for a in arms])
    center, difference = theta[0]+theta[1]*cosine, theta[2]*sine
    return center-difference, center+difference


def _closed_arm(theta):
    a, b, unused_c = theta
    d, linear = a*(1-a)-b*b, (1-2*a)*b
    z = -linear/(d+math.sqrt(max(0., (d-linear)*(d+linear))))
    tangent = math.sqrt((1-z)/(1+z))
    half_angle = Fraction(tangent).limit_denominator(HALF_ANGLE_DENOMINATOR_CAP)
    return rational_circle(half_angle.numerator, half_angle.denominator)


def _best_grid_pair(theta):
    positive = [_arm(a) for a in GRID if a["sin_numerator"] > 0]
    pm, pp = _pair_predictions(theta, positive)
    weights, rates = two_arm_capacity(pm, pp)
    index = int(np.argmax(rates))
    return positive[index], float(weights[index]), float(rates[index])


def _exact_pair_from_support(design):
    active = design["support"]
    if len(active) != 2:
        return None
    first, second = (GRID[item["arm"]] for item in active)
    c1, s1, d1 = (first[key] for key in FEATURE_KEYS)
    c2, s2, d2 = (second[key] for key in FEATURE_KEYS)
    if c1*d2 != c2*d1 or s1*d2 != -s2*d1 or not s1 or not s2:
        return None
    index = 0 if s1 > 0 else 1
    positive = first if index == 0 else second
    denominator = active[0]["weight"]+active[1]["weight"]
    return _arm(positive), float(active[index]["weight"]/denominator)


def plan_from_rows(policy, rows):
    """Public planning API: receives only observed row counts and a policy."""
    if policy not in POLICIES:
        raise ValueError("unknown policy")
    fit = fit_rows(rows)
    theta = fit["theta"]
    detail, fallback, reason = None, False, None
    if policy == "grid":
        detail = design_from_fit(theta)
        pair = _exact_pair_from_support(detail)
        if pair is None:
            arm, weight, unused_rate = _best_grid_pair(theta)
            fallback, reason = True, "full returned support is not one exact reflected pair"
        else:
            arm, weight = pair
    else:
        arm = rational_circle(1, 1) if policy == "psr" else _closed_arm(theta)
        pm, pp = _pair_predictions(theta, [arm])
        weights, unused_rates = two_arm_capacity(pm, pp)
        weight = float(weights[0])
    pm, pp = _pair_predictions(theta, [arm])
    capacity_weight, capacity_rate = two_arm_capacity(pm, pp)
    q = (1-weight)*pm+weight*pp
    executed_rate = float(((1-weight)*bernoulli_kl(pm, q)+weight*bernoulli_kl(pp, q))[0])
    return {"policy": policy, "fit_theta": theta.tolist(), "fit_shots": int(fit["shots"]),
            "positive_arm": _arm(arm), "weight_plus": weight,
            "predicted_pminus": float(pm[0]), "predicted_pplus": float(pp[0]),
            "fit_information_rate": executed_rate,
            "pair_capacity_rate": float(capacity_rate[0]),
            "pair_capacity_plus_weight": float(capacity_weight[0]),
            "pair_fallback": fallback, "fallback_reason": reason,
            "full_grid_design": detail, "certified": False,
            "scope": "Paid-fit numerical design; full-grid support is executed only when it is one reflected pair"}


def allocation(total_shots, weight_plus):
    """Deterministic nearest-integer allocation, retaining both arms if possible."""
    if total_shots < 0 or not 0 <= weight_plus <= 1:
        raise ValueError("invalid allocation")
    plus = int(math.floor(total_shots*weight_plus+.5))
    if total_shots >= 2:
        plus = min(total_shots-1, max(1, plus))
    return int(total_shots-plus), plus


def _cached_plan(policy, rows, state, look):
    shots = sum(row["trials"] for row in rows)
    if state["plan"] is None or state["regime"] != policy or shots >= state["next_replan"]:
        began = time.perf_counter()
        plan = plan_from_rows(policy, rows)
        state["planning_seconds"] += time.perf_counter()-began
        plan["plan_id"], plan["look"] = len(state["plans"]), look
        state["plans"].append(plan)
        state["plan"], state["regime"] = plan, policy
        state["next_replan"] = max(FIRST_REPLAN_SHOTS, 2*shots)
    return state["plan"]


def _deadline(control):
    return datetime.fromisoformat(control["stop_new_experiments_by_utc"].replace("Z", "+00:00")).timestamp()


def run_trial(case, policy, rep, seed, cap, cutoff=None):
    """Independent replicate; only this simulator boundary reads case truth."""
    rng = np.random.RandomState(seed)
    rows, history = [], []
    state = evidence.empty_state()
    planner = {"plan": None, "regime": None, "next_replan": 0, "plans": [], "planning_seconds": 0.}
    last, explored = 0, 0
    screening_time, certification_time = 0., 0.
    for look, budget in enumerate(schedule(cap), 1):
        if cutoff is not None and time.time() >= cutoff:
            raise RuntimeError("Experiment cutoff reached; completed records remain in the ledger.")
        available = budget-last
        if policy == "psr" or last < EXPLORE_AFTER:
            phases = [("psr", available)]
        else:
            target = int(math.ceil(EXPLORE_MULTIPLIER*math.sqrt(budget/2)))
            deficit = min(available//2, max(0, target-explored))
            phases = [("explore_a", 2*((deficit+1)//2)), ("explore_b", 2*(deficit//2)),
                      ("exploit", available-2*deficit)]
        for phase, count in phases:
            if not count:
                continue
            if phase.startswith("explore"):
                arm = rational_circle(4, 7) if phase == "explore_a" else rational_circle(7, 4)
                weight, plan_id = .5, None
            else:
                effective_policy = "psr" if phase == "psr" else policy
                plan = _cached_plan(effective_policy, rows, planner, look)
                arm, weight, plan_id = plan["positive_arm"], plan["weight_plus"], plan["plan_id"]
            nminus, nplus = allocation(count, weight)
            co, si = arm["cos_numerator"]/arm["denominator"], arm["sin_numerator"]/arm["denominator"]
            pm = case["A"]+case["B"]*co-case["C"]*si
            pp = case["A"]+case["B"]*co+case["C"]*si
            km, kp = int(rng.binomial(nminus, pm)), int(rng.binomial(nplus, pp))
            block = {"look": look, "budget": budget, "phase": phase, "positive_arm": _arm(arm),
                     "plan_id": plan_id, "planned_weight_plus": weight,
                     "nminus": nminus, "nplus": nplus, "minus_successes": km, "plus_successes": kp}
            history.append(block)
            began = time.perf_counter()
            evidence.update_block(state, nminus, nplus, km, kp)
            screening_time += time.perf_counter()-began
            merge_row(rows, arm, nplus, kp)
            negative = dict(arm)
            negative["sin_numerator"] *= -1
            merge_row(rows, negative, nminus, km)
            if phase.startswith("explore"):
                explored += count//2
        assert state["shots"] == sum(row["trials"] for row in rows) == budget
        screened = evidence.screen(state, DELTA)
        if screened["crossing_candidate"] or budget == cap:
            began = time.perf_counter()
            cert = evidence.certificate_from_blocks(state["blocks"], delta=DELTA, experts=state["experts"])
            certification_time += time.perf_counter()-began
            if cert["decision"] != "abstain" or budget == cap:
                result = {"case_id": case["case_id"], "rep": rep, "seed": seed,
                          "policy": policy, "engine": "weighted_conditional_unequal",
                          "method": policy+"_weighted_conditional_unequal", "shots": budget,
                          "decision": cert["decision"], "certificate": cert,
                          "history": history, "plans": planner["plans"], "final_rows": rows,
                          "exploration_pairs": explored,
                          "full_grid_replans": sum(p["policy"] == "grid" for p in planner["plans"]),
                          "pair_fallback_replans": sum(p["pair_fallback"] for p in planner["plans"])}
                timing = {"planning_seconds": planner["planning_seconds"], "screening_seconds": screening_time,
                          "certification_seconds": certification_time}
                return result, timing
        last = budget
    raise AssertionError("cap checkpoint must return a terminal record")


def source_hashes():
    directory = Path(__file__).resolve().parent
    paths = [directory/name for name in ("unequal_runner.py", "weighted_fisher.py", "expert_config.py",
             "conditional_fisher.py", "conditional_fisher_unequal.py", "design_policy.py",
             "certified_likelihood.py", "predictive.py", "probe_fixed.py")]
    paths.append(ROOT/"experiments/circuit_validation/direct_tests.py")
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main(args):
    started = time.perf_counter()
    if args.repetitions < 1 or args.cap < 32 or args.cap % 2:
        raise ValueError("positive repetitions and even cap >=32 required")
    if Path(args.output).name != args.output or args.output in (".", ".."):
        raise ValueError("output must be a new result directory name")
    policies = args.policies.split(",")
    if not policies or len(set(policies)) != len(policies) or not set(policies) <= set(POLICIES):
        raise ValueError("invalid policy set")
    control = json.loads((ROOT/"results/structured_validation/run_control.json").read_text())
    cutoff = _deadline(control)
    all_cases = json.loads((ROOT/"results/learned_validation/development_v1/spec.json").read_text())["cases"]
    if args.case_ids:
        ids = {int(x) for x in args.case_ids.split(",")}
        cases = [c for c in all_cases if c["case_id"] in ids]
        if len(cases) != len(ids):
            raise ValueError("unknown case ID")
    else:
        cases = [c for c in all_cases if c["kind"] == "main" and c["C"] > 0]
    directory = ROOT/"results/structured_validation"/args.output
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/"spec.json").exists():
        raise RuntimeError("preserve existing runs; select a fresh version")
    sources = source_hashes()
    spec = {"status": "development only", "cases": cases, "repetitions": args.repetitions,
            "policies": policies, "cap": args.cap, "delta": DELTA, "checkpoints": schedule(args.cap),
            "seed_base": args.seed_base, "seed_rule": "base+1009*case_id+1000003*rep, separate RNG per policy/rep",
            "coupling": "All policies share an independent replicate seed; identical prefixes agree until allocations or angles diverge.",
            "explore_after_previous_shots": EXPLORE_AFTER, "exploration_sqrt_multiplier": EXPLORE_MULTIPLIER,
            "exploration_coordinates": [rational_circle(4, 7), rational_circle(7, 4)],
            "replanning": "First plan at zero; then at paid shots>=32 and subsequent doubling of the last planning count; regime change forces replanning.",
            "planning_order": "At most once per outer look, immediately before its exploitation subblock, after completed exploration subblocks.",
            "allocation": "Numerical two-arm Chernoff capacity for PSR/closed; raw full-grid pair weights when eligible. Round plus count to nearest integer, retain both arms when block size>=2.",
            "full_grid_limit": "Only exact two-arm reflected supports are executed; any other support is explicitly replaced by the strongest optimized grid pair and logged.",
            "grid": GRID, "half_angle_denominator_cap": HALF_ANGLE_DENOMINATOR_CAP,
            "evidence_protocol": evidence.protocol_config(), "sources": sources, "run_control": control,
            "cost": "All physical shots including learning, errors and cap abstentions."}
    (directory/"spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    cells, timings, total_records = [], [], 0
    with (directory/"trials.jsonl").open("w", encoding="utf-8") as ledger:
        for case in cases:
            for policy in policies:
                records = []
                for rep in range(args.repetitions):
                    seed = args.seed_base+1009*case["case_id"]+1000003*rep
                    if not 0 <= seed < 2**32:
                        raise ValueError("seed exceeds RandomState range")
                    record, timing = run_trial(case, policy, rep, seed, args.cap, cutoff)
                    records.append(record)
                    timings.append(timing)
                    ledger.write(json.dumps(record, separators=(",", ":"), allow_nan=False)+"\n")
                    ledger.flush()
                    total_records += 1
                expected = "accept" if case["C"] < 0 else "reject" if case["C"] > 0 else "abstain"
                cell = {"case_id": case["case_id"], "kind": case["kind"], "method": policy+"_weighted_conditional_unequal",
                        "repetitions": len(records), "mean_shots": float(np.mean([r["shots"] for r in records])),
                        "wrong": sum(r["decision"] not in ("abstain", expected) for r in records),
                        "abstained": sum(r["decision"] == "abstain" for r in records),
                        "full_grid_replans": sum(r["full_grid_replans"] for r in records),
                        "pair_fallback_replans": sum(r["pair_fallback_replans"] for r in records)}
                cells.append(cell)
                print(json.dumps({"cell": cell, "elapsed_seconds": time.perf_counter()-started}), flush=True)
    if source_hashes() != sources:
        raise RuntimeError("sources changed during the run; ledger retained but run not frozen")
    summary = {"spec": spec, "cells": cells, "new_trajectories": total_records,
               "timings": {key: sum(t[key] for t in timings) for key in timings[0]},
               "elapsed_seconds": time.perf_counter()-started}
    (directory/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("spec", "cells")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case-ids", default="")
    parser.add_argument("--policies", default="psr,closed,grid")
    parser.add_argument("--repetitions", type=int, default=16)
    parser.add_argument("--cap", type=int, default=2**22)
    parser.add_argument("--seed-base", type=int, default=710000000)
    main(parser.parse_args())
