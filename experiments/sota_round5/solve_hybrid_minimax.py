"""Solve the frozen hybrid minimax design, then audit the actual policy.

Numerical LP t is a solver reference, not an exact optimality certificate.
The joint fair-null certificate uses exact integers across the full horizon.
"""
from fractions import Fraction
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.special import logsumexp

import hybrid_terminal_coefficients as suffix
import optimal_sign_stopping as original
import physical_cost_minimax as previous
import evaluate_optimal_sign_physical as physical

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = suffix.DIRECTORY
PLAN = suffix.PLAN
COEFFICIENTS = DIRECTORY / "terminal_coefficients_float.json"
CONTROLS = suffix.CONTROL_GRID
BETA = suffix.BETA


def orientation_ratios(d, j, positive):
    k = d-j
    hypotheses = [.5]+[float(v) for q in positive for v in (q, 1-q)]
    reference = float(logsumexp([k*math.log(q)+j*math.log1p(-q) for q in hypotheses]))-math.log(len(hypotheses))
    return [(math.exp(k*math.log(float(q))+j*math.log1p(-float(q))-reference),
             math.exp(j*math.log(float(q))+k*math.log1p(-float(q))-reference)) for q in positive]


def solve(plan, coefficients, envelope):
    h, cap = plan["prefix_sign_horizon"], plan["physical_pair_cap"]
    positive = tuple(Fraction(*value) for value in plan["q_exact"])
    grid = [{"grid_id": 3*qi+ri, "q": q, "r": Fraction(*rv), "qi": qi}
            for qi, q in enumerate(positive) for ri, rv in enumerate(plan["r_exact"])]
    states = [(d, j) for d in range(h+1) for j in range(d//2+1)]
    si = {state: i for i, state in enumerate(states)}
    variables, vi = [], {}
    for d, j in states:
        actions = ["continue"] if d < h else ["beta_suffix"]
        if 2*j != d:
            actions.append("report")
        for action in actions:
            vi[d, j, action] = len(variables); variables.append((d, j, action))
    t_column = len(variables); nvar = t_column+1
    objective = np.zeros(nvar); objective[t_column] = 1.
    inequality = np.zeros((19, nvar)); inequality[10:, t_column] = -1.
    er, ec, ev = [], [], []
    rewards = {point["grid_id"]: suffix.waiting_rewards(cap, h, point["r"]) for point in grid}
    future = {row["grid_id"]: row for row in coefficients["rows"]}
    fair_future = coefficients["fair_null_any_report_by_plus_count"]
    for column, (d, j, action) in enumerate(variables):
        er.append(si[d, j]); ec.append(column); ev.append(1.)
        majority, null_lr, cost_lr, correct_lr = original.likelihood_data(d, j, positive)
        if action == "continue":
            branches = [(j, 1.)] if 2*j == d else [(j, majority), (j+1, 1-majority)]
            for fresh_j, probability in branches:
                er.append(si[d+1, fresh_j]); ec.append(column); ev.append(-probability)
        elif action == "report":
            inequality[0, column] = null_lr
            for point in grid:
                g, qi = point["grid_id"], point["qi"]
                before, cost = rewards[g]
                inequality[1+g, column] = -correct_lr[qi]*before[d]
                inequality[10+g, column] = cost_lr[qi]*cost[d]/envelope[g]["expected_pairs_float"]
        else:
            k = h-j
            ratios = orientation_ratios(h, j, positive)
            inequality[0, column] = null_lr*(fair_future[k]+fair_future[j])/2
            for point in grid:
                g, qi = point["grid_id"], point["qi"]
                upper, lower = ratios[qi]
                values = future[g]
                inequality[1+g, column] = -(upper*values["positive"][k]+lower*values["positive"][j])/2
                inequality[10+g, column] = (upper*values["paid_pairs"][k]+lower*values["paid_pairs"][j])/(2*envelope[g]["expected_pairs_float"])
    equality = coo_matrix((ev, (er, ec)), shape=(len(states), nvar)).tocsr()
    rhs = np.zeros(len(states)); rhs[0] = 1.
    guard = Fraction(*plan["fixed_solver_guard_exact"])
    alpha = Fraction(*plan["null_constraint_exact"])-guard
    power = Fraction(*plan["each_power_constraint_exact"])+guard
    upper = np.asarray([float(alpha)]+[-float(power)]*9+[0.]*9)
    started = time.perf_counter()
    result = linprog(objective, A_ub=coo_matrix(inequality).tocsr(), b_ub=upper,
                     A_eq=equality, b_eq=rhs, bounds=(0, None), method="interior-point",
                     options={"tol": 1e-9, "maxiter": 200, "sparse": True, "rr": False})
    diagnostic = {"success": bool(result.success), "status": int(result.status), "message": result.message,
                  "states": len(states), "variables_including_t": nvar, "iterations": int(result.nit),
                  "solver_elapsed_seconds": time.perf_counter()-started,
                  "claim": "Numerical solver reference, not an exact minimax optimality certificate"}
    if result.x is not None and np.all(np.isfinite(result.x)):
        diagnostic.update(numerical_t_before_rounding=float(result.x[t_column]),
                          maximum_flow_residual=float(np.max(np.abs(equality@result.x-rhs))),
                          maximum_inequality_violation=float(max(0., np.max(inequality@result.x-upper))),
                          predicted_joint_null=float(inequality[0]@result.x),
                          predicted_power_by_grid=(-inequality[1:10]@result.x).tolist(),
                          predicted_cost_ratio_by_grid=(inequality[10:]@result.x+result.x[t_column]).tolist())
    suffix.write_new(DIRECTORY/"solver_status.json", diagnostic)
    if not result.success:
        return None, grid, diagnostic
    denominator = 10**12
    rows = []
    for d in range(h+1):
        row = []
        for j in range(d//2+1):
            if (d, j, "report") not in vi:
                row.append(0); continue
            report = result.x[vi[d, j, "report"]]
            other = result.x[vi[d, j, "continue" if d < h else "beta_suffix"]]
            probability = max(0., min(1., report/(report+other))) if report+other > 0 else 0.
            if probability <= 1e-9:
                probability = 0.
            elif probability >= 1-1e-9:
                probability = 1.
            row.append(int(round(probability*denominator)))
        rows.append(row)
    return rows, grid, diagnostic


def exact_joint_null(prefix, denominator, boundaries):
    """Fraction prefix plus integer Pascal suffix; beta begins atH+1."""
    h = len(prefix)-1
    _, _, row = suffix.prefix_distribution(prefix, denominator, Fraction(1, 2))
    common = (2*denominator)**h
    counts = {}
    for k, probability in enumerate(row):
        numerator = probability*common
        assert numerator.denominator == 1
        if numerator:
            counts[k] = numerator.numerator
    for d in range(h+1, len(boundaries)):
        b = boundaries[d]
        fresh = {}
        for k, count in counts.items():
            if d-b < k < b:
                fresh[k] = fresh.get(k, 0)+count
            if d-b < k+1 < b:
                fresh[k+1] = fresh.get(k+1, 0)+count
        counts = fresh
    total_denominator = common*(1 << (len(boundaries)-1-h))
    risk = Fraction(total_denominator-sum(counts.values()), total_denominator)
    return risk


def float_full_first_passage(prefix, denominator, boundaries, q):
    full, h = len(boundaries)-1, len(prefix)-1
    positive, negative = np.zeros(full+1), np.zeros(full+1)
    row = np.ones(1)
    for d in range(1, full+1):
        fresh = np.zeros(d+1)
        fresh[:-1] += (1-q)*row; fresh[1:] += q*row
        if d <= h:
            k = np.arange(d+1)
            j = np.minimum(k, d-k)
            stopped = fresh*np.asarray(prefix[d], dtype=float)[j]/denominator
            positive[d] = math.fsum(stopped[k*2 > d].tolist())
            negative[d] = math.fsum(stopped[k*2 < d].tolist())
            fresh -= stopped
        else:
            b = boundaries[d]
            positive[d] = math.fsum(fresh[b:].tolist())
            negative[d] = math.fsum(fresh[:d-b+1].tolist())
            fresh[b:] = 0.; fresh[:d-b+1] = 0.
        row = fresh
        if not np.any(row):
            break
    survival = math.fsum(row.tolist())
    assert abs(math.fsum(positive.tolist())+math.fsum(negative.tolist())+survival-1) < 1e-11
    return positive, negative, survival, None


def main():
    started = time.perf_counter()
    paths = (PLAN, COEFFICIENTS, CONTROLS, BETA, Path(__file__).resolve(), Path(original.__file__),
             Path(suffix.__file__), Path(physical.__file__), Path(previous.__file__))
    hashes = {p.relative_to(ROOT).as_posix(): suffix.sha(p) for p in paths}
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    coefficients = json.loads(COEFFICIENTS.read_text(encoding="utf-8"))
    controls = json.loads(CONTROLS.read_text(encoding="utf-8"))
    checks = json.loads((DIRECTORY/"small_coefficient_checks.json").read_text(encoding="utf-8"))
    assert checks["status"] == "passed_exact_small_hybrid_splice_checks"
    assert coefficients["plan_sha256"] == suffix.sha(PLAN)
    assert all(suffix.sha(ROOT/path) == value for path, value in plan["input_sha256"].items())
    envelope = {row["grid_id"]: row for row in controls["hindsight_envelope"]}
    prefix, grid, solver = solve(plan, coefficients, envelope)
    if prefix is None:
        print(json.dumps({"status": "saved_solver_failure_without_design_changes"}), flush=True)
        return
    denominator = 10**12
    boundaries = json.loads(BETA.read_text(encoding="utf-8"))["boundaries"]
    policy = {"status": "frozen_hybrid_candidate_pending_independent_acceptance",
              "plan_sha256": suffix.sha(PLAN),
              "protocol": {"policy_numerators": prefix, "probability_denominator": denominator,
                           "prefix_sign_horizon": 128, "full_sign_horizon": 8192,
                           "physical_pair_cap": 8192, "continuation_start_discordance": 129,
                           "continuation_boundary_path": BETA.relative_to(ROOT).as_posix(),
                           "continuation_boundary_sha256": suffix.sha(BETA),
                           "continuation_boundaries": boundaries,
                           "truth_inputs": [],
                           "monitoring": "Fresh coin only after a discordant pair through128; afterward apply unchanged beta at each new discordance starting129. Concordant pairs are paid but never trigger a coin redraw."},
              "solver": solver, "input_sha256": hashes, "new_observations": 0}
    suffix.write_new(DIRECTORY/"candidate_policy.json", policy)
    print(json.dumps({"candidate_frozen": str(DIRECTORY/"candidate_policy.json"), "solver_t_reference": solver["numerical_t_before_rounding"]}), flush=True)
    risk = exact_joint_null(prefix, denominator, boundaries)
    null = {"status": "exact_integer_full_horizon_joint_null",
            "risk": previous.enclosure(risk), "null_at_most_00999_exact": risk <= Fraction(*plan["null_constraint_exact"]),
            "prefix_horizon": 128, "full_horizon": 8192,
            "risk_exact_numerator": str(risk.numerator), "risk_exact_denominator": str(risk.denominator),
            "candidate_sha256": suffix.sha(DIRECTORY/"candidate_policy.json"),
            "method": "Exact Fraction prefix; surviving prefix probabilities put over a common integer denominator, then fair-sign Pascal recurrence through8192 with beta absorption beginning129."}
    suffix.write_new(DIRECTORY/"joint_null_exact.json", null)
    print(json.dumps({"joint_null_complete": float(risk), "accepted": null["null_at_most_00999_exact"]}), flush=True)
    fair = float_full_first_passage(prefix, denominator, boundaries, .5)
    assert abs(float(risk)-math.fsum(fair[0].tolist())-math.fsum(fair[1].tolist())) < 1e-10
    cache, rows = {}, []
    for point in grid:
        q, r = point["q"], point["r"]
        if q not in cache:
            cache[q] = float_full_first_passage(prefix, denominator, boundaries, float(q))
        oc = physical.waiting_mixture(cache[q], float(r), plan["physical_pair_cap"])
        rows.append({"grid_id": point["grid_id"], "q_exact": [q.numerator, q.denominator],
                     "r_exact": [r.numerator, r.denominator], **oc,
                     "expected_shots": 2*oc["expected_pairs"], "correct_probability": oc["positive_probability"],
                     "error_probability": oc["negative_probability"],
                     "cost_over_envelope_float": oc["expected_pairs"]/envelope[point["grid_id"]]["expected_pairs_float"]})
    costs = {row["grid_id"]: row["expected_shots"] for row in rows}
    ratios = [row["cost_over_envelope_float"] for row in rows]
    comparisons = []
    for name in dict.fromkeys(row["control"] for row in controls["rows"]):
        baseline = [row for row in controls["rows"] if row["control"] == name]
        comparisons.append({"control": name, "globally_qualified_float": name in controls["globally_qualified_pool"],
                            "candidate_over_control_geometric_cost_float": math.exp(math.fsum(math.log(costs[row["grid_id"]]/row["expected_shots_float"]) for row in baseline)/9),
                            "candidate_over_control_arithmetic_cost_float": math.fsum(costs.values())/math.fsum(row["expected_shots_float"] for row in baseline),
                            "worst_grid_candidate_over_control_cost_float": max(costs[row["grid_id"]]/row["expected_shots_float"] for row in baseline)})
    result = {"status": "joint_null_exactly_audited_hybrid_grid_float_evaluation",
              "candidate_policy_sha256": suffix.sha(DIRECTORY/"candidate_policy.json"),
              "protocol": policy["protocol"], "solver": solver, "joint_null": null, "rows": rows,
              "aggregate": {"minimum_grid_correct_probability_float": min(row["correct_probability"] for row in rows),
                            "every_grid_power_at_least_95001_float": all(row["correct_probability"] >= .95001 for row in rows),
                            "maximum_envelope_ratio_actual_policy_float": max(ratios),
                            "geometric_envelope_ratio_float": math.exp(math.fsum(math.log(value) for value in ratios)/9),
                            "arithmetic_envelope_ratio_float": math.fsum(costs.values())/math.fsum(row["expected_shots_float"] for row in envelope.values()),
                            "equal_grid_mean_expected_shots_float": math.fsum(costs.values())/9},
              "solver_vs_forward_after_rounding": {
                  "maximum_power_difference": max(abs(rows[i]["correct_probability"]-solver["predicted_power_by_grid"][i]) for i in range(9)),
                  "maximum_cost_ratio_difference": max(abs(rows[i]["cost_over_envelope_float"]-solver["predicted_cost_ratio_by_grid"][i]) for i in range(9))},
              "control_comparisons": comparisons,
              "input_sha256": hashes, "elapsed_seconds": time.perf_counter()-started, "new_observations": 0,
              "claim_boundary": "Only the frozen policy joint fair-null risk is exact here. Physical power/cost and comparison numbers are independently forward-computed floating diagnostics pending interval certification. Classical development evidence; solver success does not prove exact minimax optimality, SOTA, novelty,3x, or benchmark superiority."}
    assert all(suffix.sha(ROOT/path) == value for path, value in hashes.items())
    suffix.write_new(DIRECTORY/"candidate.json", result)
    print(json.dumps({"output": str(DIRECTORY/"candidate.json"), "aggregate": result["aggregate"],
                      "comparisons": comparisons, "solver_forward_difference": result["solver_vs_forward_after_rounding"],
                      "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
