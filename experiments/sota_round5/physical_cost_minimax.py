"""One preregistered classical physical-cost minimax stopping LP.

The target envelope is used offline only. Runtime never chooses a control using
truth: it executes one frozen state/coin policy for every scenario. A failure is
saved without changing the grid, horizon, objective, or constraints.
"""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/sota_round4"))
import analytic_sign_cost as analytic
import optimal_sign_stopping as original
import optimal_sign_policy as runtime
import evaluate_optimal_sign_physical as evaluation

DIRECTORY = ROOT / "results/sota_round5/physical_minimax"
PLAN = DIRECTORY / "development_plan.json"
OUT = DIRECTORY / "candidate.json"
MANIFEST = ROOT / "results/sota_round4/lr_mixture_boundaries/calibration_manifest.json"
BETA = ROOT / "results/sota_round3/cap16384_preparation/beta_exact.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False); handle.write("\n")


def enclosure(value, digits=12):
    value = Fraction(value)
    scale = 10**digits
    low = value.numerator*scale//value.denominator
    high = -((-value.numerator*scale)//value.denominator)
    def render(integer):
        return str(integer//scale) + "." + str(integer % scale).zfill(digits)
    return {"lower": render(low), "upper": render(high), "float_diagnostic": float(value),
            "method": "Exact rational value rounded outward to a fixed decimal grid"}


def preregistration():
    paths = (Path(__file__).resolve(), Path(original.__file__), Path(analytic.__file__),
             Path(evaluation.__file__), Path(runtime.__file__), MANIFEST, BETA)
    return {
        "version": "physical_minimax_classical_stopping_v1",
        "status": "fixed_before_design_grid_control_evaluation_and_optimization",
        "positive_q_exact": [[7, 10], [9, 10], [99, 100]],
        "discordance_rate_exact": [[1, 50], [1, 10], [2, 5]],
        "sign_horizon": 128, "physical_pair_cap": 8192, "physical_shot_cap": 16384,
        "scientific_null_constraint_exact": [999, 100000],
        "scientific_each_point_power_constraint_exact": [95001, 100000],
        "fixed_numerical_guard_exact": [1, 10000000],
        "guard_rule": "Before optimization, reserve this fixed guard on both constraints for numerical solver error and policy rounding: solver null<=.0099899 and each power>=.9500101. Exact final acceptance uses the scientific .00999/.95001 constraints; no result-dependent guard changes.",
        "classical_controls": "Frozen H8192 beta_exact and all11 LR-mixture registry artifacts; no recalibration or removal based on candidate results.",
        "pool_qualification": "A fixed control must reach .95 correct-report probability at ALL9 design points. Initial pool membership is a float analytic design calculation; margins are reported and it is not called an interval certificate.",
        "offline_objective": "Minimize t such that candidate all-path expected physical cost at each of9 points is <= t times the minimum cost among the globally qualified fixed-control pool at that point.",
        "oracle_boundary": "This hindsight envelope defines an offline optimization target only; online policy does not know q,r,case ID or which control is best.",
        "abstention_accounting": "All paths without a report through H128 pay the full8192 physical pairs; subsequent reports are forbidden. Earlier physical-cap expiry also abstains and pays the full cap.",
        "allowed_policy": "Symmetric randomized majority report after each discordance, depending only on d and minority-count j; at ties no report. A fresh independent integer coin is consumed only on discordant pairs.",
        "rounding": {"denominator": 10**12, "snap_tolerance": 1e-9, "rule": "Snap only within the predeclared tolerance of0 or1, otherwise round to nearest integer numerator."},
        "solver": {"method": "scipy1.5 sparse interior-point", "tolerance": 1e-9, "maximum_iterations": 200, "redundancy_removal": False},
        "failure_rule": "Save numerical solver failure or exact-audit failure without changing q,r,H,cap,targets or objective. Do not claim an infeasibility certificate unless the solver supplied one.",
        "reporting": "Keep t>=1 outcomes and regressions; report arithmetic and equal-point geometric costs on all9 grid points and all fixed controls. Development only, no SOTA, novelty or threefold claim.",
        "prior_art": "Classical constrained optimal stopping/alpha spending; Silva, Kulldorff and Yih (2020), doi:10.1111/rssb.12379, is prior art rather than an equivalence claim for this implementation.",
        "new_observations": 0, "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
    }


def controls_on_grid(plan):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete" and manifest["max_discordances"] == 8192
    specs = [{"shape": "beta_exact", "path": BETA.relative_to(ROOT).as_posix(), "sha256": sha(BETA)}] + manifest["records"]
    assert len(specs) == 12
    q_values = [Fraction(*v) for v in plan["positive_q_exact"]]
    rates = [Fraction(*v) for v in plan["discordance_rate_exact"]]
    grid = [{"grid_id": i*len(rates)+j, "q": q, "r": r}
            for i, q in enumerate(q_values) for j, r in enumerate(rates)]
    rows, inputs = [], {}
    for spec in specs:
        path = ROOT/spec["path"]
        assert sha(path) == spec["sha256"]
        artifact = json.loads(path.read_text(encoding="utf-8"))
        assert artifact["max_discordances"] == 8192
        inputs[spec["path"]] = spec["sha256"]
        for q in q_values:
            distribution = analytic.first_crossing_distribution(artifact["boundaries"], float(q))
            for point in (point for point in grid if point["q"] == q):
                oc = analytic.oc_from_distribution(distribution, float(point["r"]), plan["physical_pair_cap"])
                rows.append({"control": spec["shape"], "grid_id": point["grid_id"],
                             "correct_probability_float": oc["positive_probability"],
                             "error_probability_float": oc["negative_probability"],
                             "abstain_probability_float": oc["abstain_probability"],
                             "expected_pairs_float": oc["expected_pairs"], "expected_shots_float": oc["expected_shots"]})
        print(json.dumps({"grid_control_complete": spec["shape"]}), flush=True)
    pool = [spec["shape"] for spec in specs if min(r["correct_probability_float"] for r in rows if r["control"] == spec["shape"]) >= .95]
    if not pool:
        raise ArithmeticError("No fixed control qualifies on all9 points; preregistered objective undefined")
    envelope = {point["grid_id"]: min((r for r in rows if r["grid_id"] == point["grid_id"] and r["control"] in pool), key=lambda r: r["expected_pairs_float"]) for point in grid}
    artifact = {"status": "frozen_grid_control_float_characteristics", "rows": rows, "globally_qualified_pool": pool,
                "minimum_power_by_control": {spec["shape"]: min(r["correct_probability_float"] for r in rows if r["control"] == spec["shape"]) for spec in specs},
                "hindsight_envelope": [envelope[i] for i in sorted(envelope)], "input_sha256": inputs,
                "qualification_status": "Floating design-grid qualification; margins reported. No tiny numerical gap is asserted to be an interval certificate.", "new_observations": 0}
    write_new(DIRECTORY/"control_grid_float.json", artifact)
    return grid, rows, pool, envelope, inputs


def solve(plan, grid, envelope):
    horizon, cap = plan["sign_horizon"], plan["physical_pair_cap"]
    positive = tuple(Fraction(*value) for value in plan["positive_q_exact"])
    q_index = {q: i for i, q in enumerate(positive)}
    states = [(d, j) for d in range(horizon+1) for j in range(d//2+1)]
    state_index = {state: i for i, state in enumerate(states)}
    variables, index = [], {}
    for d, j in states:
        actions = ["continue"] if d < horizon else ["abstain"]
        if 2*j != d:
            actions.append("report")
        for action in actions:
            index[d, j, action] = len(variables); variables.append((d, j, action))
    t_column, nvar = len(variables), len(variables)+1
    objective = np.zeros(nvar); objective[t_column] = 1.
    inequality = np.zeros((1+2*len(grid), nvar))
    inequality[1+len(grid):, t_column] = -1.
    eq_rows, eq_cols, eq_values = [], [], []
    d_array = np.arange(1, horizon+1, dtype=float)
    kernels = {}
    for point in grid:
        r = float(point["r"])
        before = binom.sf(d_array-1, cap, r)
        truncated = d_array*(binom.sf(d_array, cap+1, r)/r)+cap*binom.cdf(d_array-1, cap, r)
        kernels[point["grid_id"]] = (before, truncated)
    for column, (d, j, action) in enumerate(variables):
        eq_rows.append(state_index[d, j]); eq_cols.append(column); eq_values.append(1.)
        majority, null_lr, cost_lr, correct_lr = original.likelihood_data(d, j, positive)
        if action == "continue":
            branches = [(j, 1.)] if 2*j == d else [(j, majority), (j+1, 1-majority)]
            for fresh_j, probability in branches:
                eq_rows.append(state_index[d+1, fresh_j]); eq_cols.append(column); eq_values.append(-probability)
        else:
            if action == "report":
                inequality[0, column] = null_lr
            for point in grid:
                g, qi = point["grid_id"], q_index[point["q"]]
                before, truncated = kernels[g]
                if action == "report":
                    inequality[1+g, column] = -correct_lr[qi]*before[d-1]
                    cost = cost_lr[qi]*truncated[d-1]
                else:
                    cost = cost_lr[qi]*cap
                inequality[1+len(grid)+g, column] = cost/envelope[g]["expected_pairs_float"]
    equality = coo_matrix((eq_values, (eq_rows, eq_cols)), shape=(len(states), nvar)).tocsr()
    rhs = np.zeros(len(states)); rhs[0] = 1.
    guard = Fraction(*plan["fixed_numerical_guard_exact"])
    alpha = Fraction(*plan["scientific_null_constraint_exact"])-guard
    power = Fraction(*plan["scientific_each_point_power_constraint_exact"])+guard
    upper = np.asarray([float(alpha)]+[-float(power)]*len(grid)+[0.]*len(grid))
    result = linprog(objective, A_ub=coo_matrix(inequality).tocsr(), b_ub=upper,
                     A_eq=equality, b_eq=rhs, bounds=(0, None), method="interior-point",
                     options={"tol": 1e-9, "maxiter": 200, "sparse": True, "rr": False})
    diagnostics = {"success": bool(result.success), "status": int(result.status), "message": result.message,
                   "states": len(states), "variables_including_t": nvar, "iterations": int(result.nit),
                   "scientific_constraints": {"null": ".00999", "each_power": ".95001"},
                   "fixed_guard_applied_before_solving": "0.0000001"}
    if result.x is not None and np.all(np.isfinite(result.x)):
        diagnostics.update(numerical_t_before_policy_rounding=float(result.x[t_column]),
                           maximum_equality_residual=float(np.max(np.abs(equality@result.x-rhs))),
                           maximum_inequality_violation=float(max(0., np.max(inequality@result.x-upper))))
    write_new(DIRECTORY/"solver_status.json", diagnostics)
    if not result.success:
        return None, diagnostics
    denominator = plan["rounding"]["denominator"]
    policy_rows = []
    for d in range(horizon+1):
        row = []
        for j in range(d//2+1):
            if (d, j, "report") not in index:
                row.append(0); continue
            stop = result.x[index[d, j, "report"]]
            other = result.x[index[d, j, "continue" if d < horizon else "abstain"]]
            probability = max(0., min(1., stop/(stop+other))) if stop+other > 0 else 0.
            if probability <= 1e-9:
                probability = 0.
            elif probability >= 1-1e-9:
                probability = 1.
            row.append(int(round(probability*denominator)))
        policy_rows.append(row)
    return policy_rows, diagnostics


def exact_binomial_tails(n, rate, maximum):
    """Exact integer low-tail recurrence, stable even at thousands of pairs."""
    a, b = rate.numerator, rate.denominator
    denominator = b**n
    term = (b-a)**n
    cumulative = 0
    tails = [Fraction(1)]
    for k in range(maximum):
        cumulative += term
        tails.append(Fraction(denominator-cumulative, denominator))
        term = term*(n-k)*a//((k+1)*(b-a))
    return tails


def exact_grid_audit(policy, denominator, plan, grid):
    horizon, cap = plan["sign_horizon"], plan["physical_pair_cap"]
    null_dist = evaluation.first_passage(policy, denominator, Fraction(1, 2), exact=True)
    null_risk = sum(null_dist[0])+sum(null_dist[1])
    alpha = Fraction(*plan["scientific_null_constraint_exact"])
    target = Fraction(*plan["scientific_each_point_power_constraint_exact"])
    distributions, kernels, rows = {}, {}, []
    for point in grid:
        q, r = point["q"], point["r"]
        if q not in distributions:
            distributions[q] = evaluation.first_passage(policy, denominator, q, exact=True)
        if r not in kernels:
            tail = exact_binomial_tails(cap, r, horizon)
            future = exact_binomial_tails(cap+1, r, horizon+1)
            truncated = [Fraction(0)]+[Fraction(d, 1)/r*future[d+1]+cap*(1-tail[d]) for d in range(1, horizon+1)]
            kernels[r] = tail, truncated
        positive, negative, survival, _ = distributions[q]
        before, truncated = kernels[r]
        correct = sum((positive[d]*before[d] for d in range(1, horizon+1)), Fraction(0))
        wrong = sum((negative[d]*before[d] for d in range(1, horizon+1)), Fraction(0))
        expected = cap*survival+sum(((positive[d]+negative[d])*truncated[d] for d in range(1, horizon+1)), Fraction(0))
        assert correct+wrong <= 1 and 0 <= expected <= cap
        rows.append({"grid_id": point["grid_id"], "q_exact": [q.numerator, q.denominator],
                     "r_exact": [r.numerator, r.denominator], "correct_probability": enclosure(correct),
                     "error_probability": enclosure(wrong), "abstain_probability": enclosure(1-correct-wrong),
                     "expected_shots": enclosure(2*expected), "power_at_least_95001_exact": correct >= target,
                     "power_at_least_95_exact": correct >= Fraction(19, 20)})
        print(json.dumps({"exact_grid_point_complete": point["grid_id"], "cost": float(2*expected), "power": float(correct)}), flush=True)
    return {"null_any_report": enclosure(null_risk), "null_at_most_00999_exact": null_risk <= alpha,
            "all9_power_at_least_95001_exact": all(row["power_at_least_95001_exact"] for row in rows),
            "rows": rows, "numerical_method": "Exact rational sign first passage, integer binomial low-tail sums, exact truncated negative-binomial expectations. Displayed endpoints are exact rational floor/ceil on a10^-12 decimal grid."}


def main():
    started = time.perf_counter()
    plan = preregistration()
    write_new(PLAN, plan)
    grid, controls, pool, envelope, control_hashes = controls_on_grid(plan)
    policy, solver = solve(plan, grid, envelope)
    if policy is None:
        print(json.dumps({"status": "saved_solver_failure_without_design_changes", "solver": solver}), flush=True)
        return
    denominator = plan["rounding"]["denominator"]
    write_new(DIRECTORY/"rounded_policy_before_exact_audit.json", {
        "plan_sha256": sha(PLAN), "protocol": {"policy_numerators": policy, "probability_denominator": denominator}, "solver": solver})
    audit = exact_grid_audit(policy, denominator, plan, grid)
    write_new(DIRECTORY/"exact_grid_audit.json", audit)
    if not audit["null_at_most_00999_exact"] or not audit["all9_power_at_least_95001_exact"]:
        print(json.dumps({"status": "saved_exact_audit_failure_without_design_changes"}), flush=True)
        return
    costs = {row["grid_id"]: row["expected_shots"]["float_diagnostic"] for row in audit["rows"]}
    ratios = [costs[i]/envelope[i]["expected_shots_float"] for i in sorted(costs)]
    comparisons = []
    for name in dict.fromkeys(row["control"] for row in controls):
        rows = [row for row in controls if row["control"] == name]
        comparison = {"control": name, "globally_qualified_float": name in pool,
                      "candidate_over_control_geometric_cost_float": math.exp(math.fsum(math.log(costs[row["grid_id"]]/row["expected_shots_float"]) for row in rows)/len(rows)),
                      "candidate_over_control_arithmetic_cost_float": math.fsum(costs.values())/math.fsum(row["expected_shots_float"] for row in rows),
                      "worst_grid_candidate_over_control_cost_float": max(costs[row["grid_id"]]/row["expected_shots_float"] for row in rows)}
        comparisons.append(comparison)
    frozen_policy = runtime.Policy(tuple(tuple(row) for row in policy), denominator, "pending_artifact_hash")
    independent_null, _ = runtime.folded_null(frozen_policy)
    assert enclosure(independent_null) == audit["null_any_report"]
    result = {
        "status": "exact_audited_classical_physical_minimax_development_policy",
        "plan_sha256": sha(PLAN), "development_plan": plan,
        "protocol": {"policy_numerators": policy, "probability_denominator": denominator,
                     "sign_horizon": plan["sign_horizon"], "physical_pair_cap": plan["physical_pair_cap"],
                     "truth_inputs": [], "action": "After a discordant pair draw a fresh uniform integer in[0,10^12). Report the current majority iff it is below policy_numerators[d][min(k,d-k)]. AtH128 forbid all later reports; unrejected paths still pay to the physical cap."},
        "solver": solver, "exact_grid_audit": audit,
        "aggregate": {"actual_rounded_policy_maximum_envelope_ratio_float": max(ratios),
                      "candidate_over_envelope_geometric_cost_float": math.exp(math.fsum(math.log(value) for value in ratios)/len(ratios)),
                      "candidate_over_envelope_arithmetic_cost_float": math.fsum(costs.values())/math.fsum(row["expected_shots_float"] for row in envelope.values()),
                      "candidate_equal_point_mean_expected_shots_float": math.fsum(costs.values())/len(costs),
                      "envelope_equal_point_mean_expected_shots_float": math.fsum(row["expected_shots_float"] for row in envelope.values())/len(envelope)},
        "control_comparisons": comparisons, "globally_qualified_control_pool": pool,
        "envelope_rows": [envelope[i] for i in sorted(envelope)],
        "independent_folded_null_matches_exact_full_state_null": True,
        "input_sha256": {**plan["source_sha256"], **control_hashes},
        "elapsed_seconds": time.perf_counter()-started, "new_observations": 0,
        "claim_boundary": "Classical minimax constrained stopping development on the predefined9-point grid. Policy error/power/cost are rationally audited; fixed-control costs and pool qualification used float analytic design values. No independent benchmark, SOTA, novelty, universal optimality or3x claim. Numerical LP t is not an exact optimality certificate.",
    }
    assert all(sha(ROOT/path) == expected for path, expected in result["input_sha256"].items())
    write_new(OUT, result)
    print(json.dumps({"output": str(OUT), "aggregate": result["aggregate"], "comparisons": comparisons,
                      "qualified_pool": pool, "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
