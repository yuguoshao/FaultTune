"""Paid pilot -> frozen design -> independent sign validation.

This is a development benchmark, not a preregistered SOTA confirmation.
The public case matrix and all settings are recorded before any trial draws.
Truth is used only by the simulator and post-hoc descriptive diagnostics.
No fitting or stopping API receives simulator coefficients.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import gammaln
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "circuit_validation"))
from direct_tests import certificate_from_pairs
from designs import choose_design, config_dict, PILOT_ANGLES
from certificates import fisher_certificate, fisher_log_tail_screen
from kl_designs import choose_kl_design

METHODS = ("psr_mixture", "psr_fisher", "learned_psr_neyman",
           "learned_closed_form", "learned_grid_design", "learned_kl_grid")
PILOT_PER_ANGLE = 256
CAP = 2**22
EPSILON = .00009
DELTA = .01


def case_matrix():
    result = []
    for a in (.01, .05, .2, .5):
        for rho in (.5, .98):
            radius = rho * min(a, 1-a)
            for fraction in (0., .9, .995):
                for sign in (-1, 1):
                    c = sign * radius * math.sqrt(1-fraction*fraction)
                    result.append({"kind": "main", "A": a, "B": radius*fraction,
                                   "C": c, "rho": rho, "B_over_R": fraction})
    for a in (.01, .05, .2, .5):
        for b in (0., .98*min(a, 1-a)):
            for sign in (-1, 0, 1):
                c = sign * EPSILON / (8*math.sin(.1))
                result.append({"kind": "null" if sign == 0 else "weak",
                               "A": a, "B": b, "C": c})
    for i, case in enumerate(result):
        case.update({"case_id": i, "h": .2, "epsilon": EPSILON})
        case["target_delta"] = 2*case["C"]*math.sin(.1)
        assert math.hypot(case["B"], case["C"]) < min(case["A"], 1-case["A"])
        if case["kind"] == "main":
            assert abs(case["target_delta"]) >= EPSILON
    return result


def budget_schedule(cap=CAP):
    values, n = [], 32
    while n < cap:
        values.append(n)
        n = 2*int(math.ceil(n*1.125/2))
    values.append(cap)
    return values


def source_hashes():
    paths = [Path(__file__).resolve(), Path(__file__).resolve().with_name("designs.py"),
             Path(__file__).resolve().with_name("kl_designs.py"),
             Path(__file__).resolve().with_name("certificates.py"),
             ROOT / "experiments" / "circuit_validation" / "direct_tests.py"]
    return {str(p.relative_to(ROOT)).replace("\\", "/"):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def probability(case, angle):
    return case["A"]+case["B"]*np.cos(angle)+case["C"]*np.sin(angle)


def pilot_for_case(case, repetitions):
    seed = 41000000 + case["case_id"]*1009
    rng = np.random.RandomState(seed)
    counts = rng.binomial(PILOT_PER_ANGLE, probability(case, PILOT_ANGLES),
                          size=(repetitions, 4))
    return counts, np.full_like(counts, PILOT_PER_ANGLE), seed


def cp_interval(successes, trials, alpha=.05):
    return [0. if not successes else float(beta.ppf(alpha/2, successes, trials-successes+1)),
            1. if successes == trials else float(beta.ppf(1-alpha/2, successes+1, trials-successes))]


def summarize_records(records, case, method):
    shots = np.array([r["shots"] for r in records])
    expected = "accept" if case["C"] < 0 else "reject" if case["C"] > 0 else "abstain"
    wrong = sum(r["decision"] not in ("abstain", expected) for r in records)
    correct = sum(r["decision"] == expected for r in records) if expected != "abstain" else 0
    abstained = sum(r["decision"] == "abstain" for r in records)
    # Failures/abstentions cannot count toward a 95%-correct stopping budget.
    correct_costs = sorted(r["shots"] for r in records if r["decision"] == expected
                          and expected != "abstain")
    rank = int(math.ceil(.95*len(records)))
    n95 = correct_costs[rank-1] if len(correct_costs) >= rank else None
    return {"case_id": case["case_id"], "kind": case["kind"], "method": method,
            "repetitions": len(records), "wrong": wrong, "correct": correct,
            "abstained": abstained, "error_ci95_pointwise": cp_interval(wrong, len(records)),
            "power_ci95_pointwise": cp_interval(correct, len(records)) if expected != "abstain" else None,
            "mean_shots": float(shots.mean()), "median_shots": float(np.median(shots)),
            "min_shots": int(shots.min()), "max_shots": int(shots.max()),
            "empirical_N95_correct": n95, "N95_censored": n95 is None,
            "N95_scope": "empirical order statistic; no simultaneous guarantee"}


def run_cell(case, method, repetitions, pilot_counts=None, pilot_totals=None):
    if pilot_counts is None:
        pilot_counts, pilot_totals, unused = pilot_for_case(case, repetitions)
    learned = method.startswith("learned_")
    start = time.perf_counter()
    if method == "learned_kl_grid":
        design = choose_kl_design(pilot_counts, pilot_totals)
    else:
        design = choose_design(method if learned else "psr_uniform",
                               pilot_counts if learned else np.zeros((repetitions, 4), dtype=int),
                               pilot_totals if learned else np.zeros((repetitions, 4), dtype=int))
    planning_seconds = time.perf_counter()-start
    t, fraction = design["t"], design["allocation_plus"]
    pilot_cost = 4*PILOT_PER_ANGLE if learned else 0
    # Independent observations after a pilot-frozen design; truth ends here.
    p_minus, p_plus = probability(case, -t), probability(case, t)
    seed = 81000000 + case["case_id"]*1009 + METHODS.index(method)*71
    rng = np.random.RandomState(seed)
    active = np.ones(repetitions, dtype=bool)
    s_minus, s_plus = np.zeros((2, repetitions), dtype=np.int64)
    n_minus, n_plus = np.zeros((2, repetitions), dtype=np.int64)
    pairs = np.zeros((repetitions, 4), dtype=np.int64)
    decisions = ["abstain"]*repetitions
    certs = [None]*repetitions
    final_look = np.zeros(repetitions, dtype=int)
    schedule = budget_schedule()
    for look_index, budget in enumerate(schedule, 1):
        if budget <= pilot_cost+2:
            continue
        indices = np.flatnonzero(active)
        if not len(indices):
            break
        stage2 = budget-pilot_cost
        targets_plus = np.clip(np.floor(fraction[indices]*stage2).astype(np.int64), 1, stage2-1)
        targets_minus = stage2-targets_plus
        if method == "psr_mixture":
            assert np.all(targets_plus == targets_minus)
            new_pairs = int(targets_plus[0]-n_plus[indices[0]])
            pm, pp = float(p_minus[0]), float(p_plus[0])
            joint = np.array([(1-pm)*(1-pp), (1-pm)*pp, pm*(1-pp), pm*pp])
            joint /= joint.sum()
            pairs[indices] += rng.multinomial(new_pairs, joint, size=len(indices))
            s_minus[indices] = pairs[indices, 2]+pairs[indices, 3]
            s_plus[indices] = pairs[indices, 1]+pairs[indices, 3]
        else:
            # NumPy 1.18 Windows binomial expects C long (int32); the public
            # cap is below 2**31, while persistent ledgers remain int64.
            s_minus[indices] += rng.binomial((targets_minus-n_minus[indices]).astype(np.int32), p_minus[indices])
            s_plus[indices] += rng.binomial((targets_plus-n_plus[indices]).astype(np.int32), p_plus[indices])
        n_minus[indices], n_plus[indices] = targets_minus, targets_plus
        final_look[indices] = look_index
        if method == "psr_mixture":
            worse, better = pairs[indices, 1], pairs[indices, 2]
            evidence = (gammaln(worse+.5)+gammaln(better+.5)-gammaln(worse+better+1)
                        -2*gammaln(.5)+(worse+better)*math.log(2))
            selected = indices[evidence >= math.log(1/DELTA)-1e-6]
        else:
            selected = []
            for i in indices:
                screen, threshold = fisher_log_tail_screen(
                    int(s_minus[i]), int(n_minus[i]), int(s_plus[i]), int(n_plus[i]),
                    look_count=len(schedule), delta=DELTA)
                if screen <= threshold+.05 or look_index == len(schedule):
                    selected.append(i)
        for i in selected:
            if method == "psr_mixture":
                cert = certificate_from_pairs(tuple(int(x) for x in pairs[i]), delta=DELTA)
            else:
                cert = fisher_certificate(int(s_minus[i]), int(n_minus[i]),
                                          int(s_plus[i]), int(n_plus[i]), look_index,
                                          len(schedule), delta=DELTA)
            # Floating-point screens never authorize a decision.
            if cert["decision"] != "abstain":
                decisions[i], certs[i], active[i] = cert["decision"], cert, False
    records = []
    for i in range(repetitions):
        record = {"case_id": case["case_id"], "method": method, "rep": i, "validation_seed": seed,
                  "pilot_counts": [int(x) for x in pilot_counts[i]] if learned else [0]*4,
                  "pilot_totals": [int(x) for x in pilot_totals[i]] if learned else [0]*4,
                  "pilot_shots": pilot_cost, "t": float(t[i]), "allocation_plus": float(fraction[i]),
                  "estimated_variance": float(design["estimated_variance"][i]),
                  "counts": [int(s_minus[i]), int(n_minus[i]), int(s_plus[i]), int(n_plus[i])],
                  "pair_counts": [int(x) for x in pairs[i]] if method == "psr_mixture" else None,
                  "shots": pilot_cost+int(n_minus[i]+n_plus[i]), "look_index": int(final_look[i]),
                  "decision": decisions[i], "certificate": certs[i]}
        records.append(record)
    cell = summarize_records(records, case, method)
    cell.update({"validation_seed": seed, "pilot_seed": 41000000+case["case_id"]*1009,
                 "planning_seconds_batch": planning_seconds,
                 "mean_t": float(np.mean(t)), "mean_allocation_plus": float(np.mean(fraction)),
                 "shift_clip_count": int(np.sum(design["shift_was_clipped"])),
                 "allocation_clip_count": int(np.sum(design["allocation_was_clipped"]))})
    return cell, records


def aggregate(cells, cases):
    table = {(c["case_id"], c["method"]): c for c in cells}
    main_ids = [c["case_id"] for c in cases if c["kind"] == "main"]
    methods = {}
    for method in METHODS:
        local = [table[i, method] for i in main_ids]
        methods[method] = {
            "main_trials": sum(x["repetitions"] for x in local),
            "main_wrong": sum(x["wrong"] for x in local),
            "main_abstained": sum(x["abstained"] for x in local),
            "main_correct": sum(x["correct"] for x in local),
            "all_wrong": sum(x["wrong"] for x in cells if x["method"] == method),
            "weak_abstained": sum(x["abstained"] for x in cells if x["method"] == method and x["kind"] == "weak"),
            "null_decisions": sum(x["repetitions"]-x["abstained"] for x in cells if x["method"] == method and x["kind"] == "null"),
            "main_geomean_shots": math.exp(np.mean([math.log(x["mean_shots"]) for x in local])),
            "main_N95_censored_cases": sum(x["N95_censored"] for x in local)}
    ratios = []
    for cid in main_ids:
        row = {"case_id": cid}
        for baseline in METHODS:
            row[baseline+"_over_closed"] = table[cid, baseline]["mean_shots"]/table[cid, "learned_closed_form"]["mean_shots"]
        row["fastest_mean_method"] = min(METHODS, key=lambda m: table[cid,m]["mean_shots"])
        ratios.append(row)
    return {"methods": methods, "ratios": ratios,
            "geomean_cost_ratios_baseline_over_closed": {
                b: math.exp(np.mean([math.log(r[b+"_over_closed"]) for r in ratios])) for b in METHODS},
            "closed_faster_cases": {b: sum(r[b+"_over_closed"] > 1 for r in ratios) for b in METHODS},
            "metric_scope": "unconditional mean cost, including pilot, wrong stops, and full-cap abstentions; all 48 main cases equally weighted"}


def run(repetitions, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    spec_path = output/"spec.json"
    if spec_path.exists():
        raise RuntimeError("Refusing to overwrite an existing run")
    cases = case_matrix()
    spec = {"status": "development specification saved before sampling; not confirmatory preregistration",
            "repetitions": repetitions, "cases": cases, "methods": list(METHODS),
            "delta": DELTA, "epsilon": EPSILON, "pilot_per_angle": PILOT_PER_ANGLE,
            "pilot_angles": list(PILOT_ANGLES), "total_shot_cap": CAP,
            "total_shot_checkpoints": budget_schedule(), "look_ratio": 1.125,
            "design_config": config_dict(), "source_sha256": source_hashes(),
            "risk_scope": "per validation call; not global across all simulations or optimizer updates",
            "fisher_spending": "delta/(2*L) per direction; L includes skipped pre-pilot looks",
            "fisher_screen": "one-term geometric upper screen with log tolerance .05; may delay a32-term certified crossing; final look always certified",
            "cost_scope": "equal cost independent Bernoulli queries; pilot charged to every learned method",
            "assumptions": ["stationary exact single-frequency response",
                            "angle-independent pre/post channels", "independent holdout after fixed pilot"],
            "primary_metric": "geometric mean of main-case unconditional mean cost ratios",
            "limitations": ["A,B sign restrictions and hand-designed synthetic cases",
                            "Grid KL comparator is a fixed two-angle plug-in design, not a full adaptive structured-KL algorithm",
                            "No hardware, drift, generic ansatz, optimizer-loop or uniform finite-cap power claim"]}
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    started = time.time()
    cells = []
    with (output/"trials.jsonl").open("w", encoding="utf-8") as ledger:
        for case in cases:
            pilot_counts, pilot_totals, unused = pilot_for_case(case, repetitions)
            local = []
            for method in METHODS:
                cell, records = run_cell(case, method, repetitions, pilot_counts, pilot_totals)
                cells.append(cell)
                local.append(cell)
                for record in records:
                    ledger.write(json.dumps(record, separators=(",", ":"))+"\n")
                ledger.flush()
            print(json.dumps({"case": case["case_id"], "kind": case["kind"],
                              "means": [round(c["mean_shots"], 1) for c in local],
                              "wrong": [c["wrong"] for c in local],
                              "abstain": [c["abstained"] for c in local]}), flush=True)
    summary = {"spec": spec, "cells": cells, "aggregate": aggregate(cells, cases),
               "trial_count": len(cells)*repetitions, "elapsed_seconds": time.time()-started}
    (output/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"done": True, "trials": summary["trial_count"],
                      "aggregate": summary["aggregate"]["geomean_cost_ratios_baseline_over_closed"],
                      "seconds": summary["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=128)
    parser.add_argument("--output", default=str(ROOT/"results"/"learned_validation"/"development_v1"))
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    run(args.repetitions, args.output)
