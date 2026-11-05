"""Classical finite-horizon sequential-testing occupancy LP, with exact audit.

The decision at each discordance uses (d, min(plus, d-plus)) and an independent
random coin. Rejection reports the current majority sign. No case parameters or
estimated effect enter the online rule. This is an implementation of classical
constrained optimal stopping, not a claim of a new statistical framework.
"""
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/sota_round3"))
sys.path.insert(0, str(ROOT / "experiments/sota_round4"))
import exact_sign_boundary as beta
import generic_sign_boundary as generic
import lr_mixture_boundaries as mixtures

DIRECTORY = ROOT / "results/sota_round5/optimal_sign_stopping"
PLAN = DIRECTORY / "development_plan.json"
OUT = DIRECTORY / "prototype.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fraction_record(value):
    value = Fraction(value)
    return {"numerator": str(value.numerator), "denominator": str(value.denominator),
            "float_diagnostic": float(value)}


def write_new(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2, allow_nan=False)
        handle.write("\n")


def design_plan():
    return {
        "version": "classical_sign_stopping_occupancy_lp_development_v1",
        "status": "design_fixed_before_optimizer_execution",
        "discordance_horizon": 128,
        "positive_alternatives_exact": [[7, 10], [9, 10], [99, 100]],
        "alternative_weighting": "Equal prior weights; mirrored alternatives have identical cost by symmetry.",
        "external_parameter_cases_read": False,
        "selection_policy": "Run one LP with the fixed grid, horizon, cost and constraints. Save all fixed classical controls without choosing a favorable subset. Abort on exact feasibility failure; do not retune constraints after seeing results.",
        "null_report_bound_exact": [1, 100],
        "minimum_each_alternative_correct_report_exact": [19, 20],
        "optimizer_null_interior_exact": [999, 100000],
        "optimizer_power_interior_exact": [95001, 100000],
        "objective": "Minimize equal-alternative mean expected discordances over all paths, including wrong reports and terminal abstentions.",
        "allowed_actions": "At each non-tie state: report majority sign with a frozen state-dependent probability, otherwise continue; at H continue becomes terminal abstention. At ties continue or terminal abstention. No early abstention, no truth input, no data-dependent prior.",
        "numerical_policy": {"solver": "scipy.optimize.linprog(method=interior-point, sparse=True)",
                             "solver_tolerance": 1e-9,
                             "probability_denominator": 10**12,
                             "snap_to_0_or_1_tolerance": 1e-9,
                             "acceptance": "Round policy probabilities once under this fixed rule, then require exact rational null<=.01 and each design q correct>=.95; optimization feasibility is not certification."},
        "comparators": ["beta_exact"] + list(mixtures.fixed_specs()),
        "fairness": "All 12 classical controls receive exact null calibration at the same H and .01 budget; the LP is an additional classical optimized control, not an exclusive FaultTune method. LP design-grid performance is in-sample development, not independent confirmation.",
        "physical_scope": "This prototype minimizes discordances, not physical shots. A physical cap and discordance waiting process need explicit integration before a physical-cost advantage claim.",
        "prior_art": [{"title": "Optimal Alpha Spending for Sequential Analysis with Binomial Data",
                       "authors": "Silva, Kulldorff and Yih", "year": 2020,
                       "doi": "10.1111/rssb.12379",
                       "relation": "Classical finite-horizon optimization prior art; this local occupancy formulation is not asserted to reproduce the paper's exact algorithm."}],
        "source_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in (
            Path(__file__).resolve(), Path(beta.__file__), Path(generic.__file__), Path(mixtures.__file__))},
    }


def likelihood_data(d, j, positive):
    """Bounded ratios to a symmetric reference mixture including the null."""
    k = d-j
    hypotheses = [.5] + [float(x) for q in positive for x in (q, 1-q)]
    logs = np.asarray([k*math.log(q)+(d-k)*math.log1p(-q) for q in hypotheses])
    norm = float(logsumexp(logs))
    reference_log = norm-math.log(len(hypotheses))
    transition_majority = float(np.dot(np.exp(logs-norm), hypotheses))
    null_ratio = math.exp(d*math.log(.5)-reference_log)
    alternative_costs, correct_coefficients = [], []
    for q in positive:
        q = float(q)
        upper = math.exp(k*math.log(q)+(d-k)*math.log1p(-q)-reference_log)
        lower = math.exp((d-k)*math.log(q)+k*math.log1p(-q)-reference_log)
        alternative_costs.append((upper+lower)/2)
        correct_coefficients.append(upper/2 if k != j else 0.)
    return transition_majority, null_ratio, alternative_costs, correct_coefficients


def solve_occupancy(horizon, positive, alpha, power):
    """LP over folded symmetric states. Solution is numerical until audited."""
    states = [(d, j) for d in range(horizon+1) for j in range(d//2+1)]
    state_index = {state: i for i, state in enumerate(states)}
    variables, index = [], {}
    for d, j in states:
        allowed = ["continue"] if d < horizon else ["abstain"]
        if j != d-j:
            allowed.append("report")
        for action in allowed:
            index[d, j, action] = len(variables)
            variables.append((d, j, action))
    nvar = len(variables)
    objective = np.zeros(nvar)
    inequality = np.zeros((len(positive)+1, nvar))
    eq_rows, eq_cols, eq_values = [], [], []
    for column, (d, j, action) in enumerate(variables):
        eq_rows.append(state_index[d, j]); eq_cols.append(column); eq_values.append(1.)
        majority, null_lr, cost_lr, correct_lr = likelihood_data(d, j, positive)
        if action == "report":
            inequality[0, column] = null_lr
            inequality[1:, column] = -np.asarray(correct_lr)
        elif action == "continue":
            objective[column] = math.fsum(cost_lr)/len(positive)
            branches = [(j, 1.)] if 2*j == d else [(j, majority), (j+1, 1-majority)]
            for fresh_j, probability in branches:
                eq_rows.append(state_index[d+1, fresh_j]); eq_cols.append(column); eq_values.append(-probability)
    equality = coo_matrix((eq_values, (eq_rows, eq_cols)), shape=(len(states), nvar)).tocsr()
    rhs = np.zeros(len(states)); rhs[0] = 1.
    upper = np.asarray([float(alpha)] + [-float(power)]*len(positive))
    result = linprog(objective, A_ub=coo_matrix(inequality).tocsr(), b_ub=upper, A_eq=equality, b_eq=rhs,
                     bounds=(0, None), method="interior-point",
                     options={"tol": 1e-9, "maxiter": 200, "sparse": True, "rr": False})
    if not result.success:
        raise ArithmeticError("Predeclared LP failed: " + result.message)
    policy = {}
    for d, j in states:
        if (d, j, "report") not in index:
            policy[d, j] = Fraction(0)
            continue
        stop = result.x[index[d, j, "report"]]
        other = result.x[index[d, j, "continue" if d < horizon else "abstain"]]
        total = stop+other
        probability = max(0., min(1., stop/total)) if total > 0 else 0.
        if probability <= 1e-9:
            probability = 0.
        elif probability >= 1-1e-9:
            probability = 1.
        policy[d, j] = Fraction(int(round(probability*10**12)), 10**12)
    return policy, {
        "solver_status": int(result.status), "solver_message": result.message,
        "folded_states": len(states), "variables": nvar,
        "numerical_objective_before_rounding": float(result.fun),
        "maximum_equality_residual": float(np.max(np.abs(equality@result.x-rhs))),
        "maximum_inequality_violation": float(max(0., np.max(inequality@result.x-upper))),
        "iterations": int(result.nit),
        "randomized_state_count": sum(0 < value < 1 for value in policy.values()),
        "claim": "A numerical optimum of the specified LP before rounding, not an exact optimality certificate and not a universal optimum.",
    }


def exact_operating_characteristics(policy, horizon, q):
    q = Fraction(q)
    row = [Fraction(1)]
    plus, minus, expected = Fraction(0), Fraction(0), Fraction(0)
    for d in range(1, horizon+1):
        expected += sum(row, Fraction(0))
        fresh = [Fraction(0)]*(d+1)
        for k, mass in enumerate(row):
            fresh[k] += mass*(1-q)
            fresh[k+1] += mass*q
        for k, mass in enumerate(fresh):
            stop = mass*policy[d, min(k, d-k)]
            if 2*k > d:
                plus += stop
            elif 2*k < d:
                minus += stop
            else:
                assert stop == 0
            fresh[k] -= stop
        row = fresh
    abstain = sum(row, Fraction(0))
    assert plus+minus+abstain == 1
    return {"positive": plus, "negative": minus, "abstain": abstain, "expected_discordances": expected}


def exhaustive_paths(policy, horizon, q):
    """Independent enumeration: integrate policy coins on each complete string."""
    positive = negative = abstain = expected = Fraction(0)
    for bits in itertools.product((0, 1), repeat=horizon):
        path_probability = q**sum(bits)*(1-q)**(horizon-sum(bits))
        survival = Fraction(1)
        k = 0
        for d, bit in enumerate(bits, 1):
            expected += path_probability*survival
            k += bit
            stop = survival*policy[d, min(k, d-k)]
            if 2*k > d:
                positive += path_probability*stop
            elif 2*k < d:
                negative += path_probability*stop
            else:
                assert stop == 0
            survival -= stop
        abstain += path_probability*survival
    return {"positive": positive, "negative": negative, "abstain": abstain,
            "expected_discordances": expected}


def checks():
    small_horizon = 8
    policy = {(d, j): Fraction((d+j) % 4, 4) if 2*j != d else Fraction(0)
              for d in range(small_horizon+1) for j in range(d//2+1)}
    for q in (Fraction(1, 2), Fraction(7, 10), Fraction(1, 10)):
        assert exact_operating_characteristics(policy, small_horizon, q) == exhaustive_paths(policy, small_horizon, q)
    # An independent convex mixture over ALL deterministic H4 policies must
    # agree with the folded occupancy LP. Full path enumeration supplies data.
    horizon = 4
    choices = [(d, j) for d in range(1, horizon+1) for j in range(d//2+1) if 2*j != d]
    q, alpha, power = Fraction(3, 4), Fraction(1, 4), Fraction(2, 5)
    costs, risks, powers = [], [], []
    for bits in itertools.product((0, 1), repeat=len(choices)):
        candidate = {(d, j): Fraction(0) for d in range(horizon+1) for j in range(d//2+1)}
        candidate.update(zip(choices, map(Fraction, bits)))
        alt = exhaustive_paths(candidate, horizon, q)
        null = exhaustive_paths(candidate, horizon, Fraction(1, 2))
        costs.append(float(alt["expected_discordances"]))
        risks.append(float(null["positive"]+null["negative"]))
        powers.append(float(alt["positive"]))
    convex = linprog(costs, A_ub=[risks, [-p for p in powers]], b_ub=[float(alpha), -float(power)],
                     A_eq=np.ones((1, len(costs))), b_eq=[1.], bounds=(0, None), method="interior-point",
                     options={"tol": 1e-10})
    assert convex.success
    lp_policy, diagnostics = solve_occupancy(horizon, (q,), alpha, power)
    assert abs(float(convex.fun)-diagnostics["numerical_objective_before_rounding"]) < 1e-8
    assert abs(float(exact_operating_characteristics(lp_policy, horizon, q)["expected_discordances"])-convex.fun) < 1e-7
    return {"independent_exact_path_enumeration": {"horizon": small_horizon, "q_values": ["1/2", "7/10", "1/10"], "passed": True},
            "all_deterministic_policy_convex_hull_check": {"horizon": horizon,
                "enumerated_policies": len(costs), "path_statistics": "exact rational enumeration",
                "convex_hull_optimum_float": float(convex.fun),
                "occupancy_optimum_float": diagnostics["numerical_objective_before_rounding"], "passed": True}}


def table_policy(table):
    return {(d, j): Fraction(int(d-j >= table[d]))
            for d in range(len(table)) for j in range(d//2+1)}


def summarize_method(name, policy, horizon, positive):
    null = exact_operating_characteristics(policy, horizon, Fraction(1, 2))
    rows = []
    for q in positive:
        oc = exact_operating_characteristics(policy, horizon, q)
        rows.append({"q_exact": [q.numerator, q.denominator],
                     **{key: fraction_record(value) for key, value in oc.items()}})
    null_risk = null["positive"]+null["negative"]
    mean = sum((Fraction(int(row["expected_discordances"]["numerator"]), int(row["expected_discordances"]["denominator"]))
                for row in rows), Fraction(0))/len(rows)
    return {"method": name, "exact_null_any_report": fraction_record(null_risk), "rows": rows,
            "each_grid_q_power_at_least_95_percent_exact": all(
                Fraction(int(row["positive"]["numerator"]), int(row["positive"]["denominator"])) >= Fraction(19, 20) for row in rows),
            "null_at_most_1_percent_exact": null_risk <= Fraction(1, 100),
            "mean_expected_discordances_exact": fraction_record(mean)}


def main():
    started = time.perf_counter()
    plan = design_plan()
    if PLAN.exists():
        original = json.loads(PLAN.read_text(encoding="utf-8"))
        assert {k: v for k, v in original.items() if k != "source_sha256"} == {
            k: v for k, v in plan.items() if k != "source_sha256"}
        amendment = DIRECTORY / ("implementation_revision_" + digest(Path(__file__).resolve())[:12] + ".json")
        if not amendment.exists():
            write_new(amendment, {"original_plan_sha256": digest(PLAN),
                                  "scientific_design_changed": False,
                                  "reason": "Resume after a programming/runtime compatibility correction; original plan is preserved and every non-source field is checked identical before execution.",
                                  "original_source_sha256": original["source_sha256"],
                                  "current_source_sha256": plan["source_sha256"]})
    else:
        write_new(PLAN, plan)
    audit = checks()
    horizon = plan["discordance_horizon"]
    positive = tuple(Fraction(*q) for q in plan["positive_alternatives_exact"])
    policy, solver = solve_occupancy(horizon, positive, Fraction(*plan["optimizer_null_interior_exact"]),
                                    Fraction(*plan["optimizer_power_interior_exact"]))
    candidate = summarize_method("classical_optimized_randomized_stopping", policy, horizon, positive)
    if not (candidate["null_at_most_1_percent_exact"] and candidate["each_grid_q_power_at_least_95_percent_exact"]):
        write_new(DIRECTORY / "failed_exact_audit.json", {"candidate": candidate, "solver": solver})
        raise ArithmeticError("Fixed design failed exact acceptance; saved failure without retuning")
    print(json.dumps({"candidate_complete": candidate["mean_expected_discordances_exact"]["float_diagnostic"],
                      "randomized_states": solver["randomized_state_count"]}), flush=True)
    baseline_artifacts, results = {}, [candidate]
    artifact = beta.calibrate(horizon)
    beta.verify_artifact(artifact)
    baseline_artifacts["beta_exact"] = artifact
    for name in mixtures.fixed_specs():
        artifact = mixtures.calibrate(horizon, name)
        mixtures.verify_artifact(artifact)
        baseline_artifacts[name] = artifact
    for name, artifact in baseline_artifacts.items():
        results.append(summarize_method(name, table_policy(artifact["boundaries"]), horizon, positive))
        print(json.dumps({"baseline_complete": name}), flush=True)
    denominator = plan["numerical_policy"]["probability_denominator"]
    policy_rows = [[int(policy[d, j]*denominator) for j in range(d//2+1)] for d in range(horizon+1)]
    all_qualified = [r for r in results[1:] if r["each_grid_q_power_at_least_95_percent_exact"]]
    mean_candidate = candidate["mean_expected_discordances_exact"]["float_diagnostic"]
    comparison = [{"control": r["method"],
                   "candidate_over_control_mean_cost_float": mean_candidate/r["mean_expected_discordances_exact"]["float_diagnostic"],
                   "control_qualified_each_q": r["each_grid_q_power_at_least_95_percent_exact"]} for r in results[1:]]
    for path, expected in plan["source_sha256"].items():
        assert digest(ROOT/path) == expected
    result = {
        "status": "exact_audited_classical_development_prototype",
        "plan_sha256": digest(PLAN), "development_plan": plan,
        "new_observations": 0, "monte_carlo_trials": 0,
        "protocol": {"state": "(d, j=min(plus,d-plus)) after each discordance",
                     "action": "Report the current majority sign iff a fresh independent uniform integer U in [0,10^12) is below policy_numerators[d][j]; otherwise continue. At H any non-report is terminal abstention.",
                     "probability_denominator": denominator, "policy_numerators": policy_rows,
                     "risk_scope": "Uniform fair conditional discordant-sign null through this H; the symmetric policy's exact iid-fair calculation applies to conditionally fair signs with independent policy coins. The H cap must never be exceeded.",
                     "truth_inputs": []},
        "solver": solver, "checks": audit, "results": results,
        "comparisons": comparison,
        "qualified_fixed_controls": [r["method"] for r in all_qualified],
        "baseline_artifacts": baseline_artifacts,
        "elapsed_seconds": time.perf_counter()-started,
        "claim_boundary": "The rounded implementable randomized rule has exact rational operating characteristics on the preregistered sign-probability design grid. Its optimizer is classical, and design-grid improvement is development evidence only. No physical-shot, independent test-grid, SOTA, novel-method or exact-global-optimality claim follows.",
    }
    write_new(OUT, result)
    print(json.dumps({"output": str(OUT), "elapsed_seconds": result["elapsed_seconds"],
                      "qualified_controls": result["qualified_fixed_controls"], "comparisons": comparison}), flush=True)


if __name__ == "__main__":
    main()
