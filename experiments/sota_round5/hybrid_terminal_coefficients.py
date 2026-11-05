"""Frozen-beta continuation coefficients for a classical prefix stopping LP.

Each returned cost is the TOTAL physical cost from time zero, integrated over
the independent waiting clock. It is not a suffix increment. Beta is first
checked at prefix_H+1, never again at prefix_H itself.
"""
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.stats import binom

import evaluate_optimal_sign_physical as physical

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "results/sota_round5/hybrid_minimax"
PLAN = DIRECTORY / "development_plan.json"
BETA = ROOT / "results/sota_round3/cap16384_preparation/beta_exact.json"
CONTROL_GRID = ROOT / "results/sota_round5/physical_minimax/control_grid_float.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2, allow_nan=False); handle.write("\n")


def frozen_plan():
    return {
        "version": "hybrid_prefix_minimax_frozen_beta_continuation_v1",
        "status": "scientific_design_fixed_before_coefficient_validation_and_optimization",
        "q_exact": [[7, 10], [9, 10], [99, 100]], "r_exact": [[1, 50], [1, 10], [2, 5]],
        "prefix_sign_horizon": 128, "full_sign_horizon": 8192, "physical_pair_cap": 8192,
        "physical_shot_cap": 16384,
        "null_constraint_exact": [999, 100000], "each_power_constraint_exact": [95001, 100000],
        "fixed_solver_guard_exact": [1, 10000000],
        "objective": "Minimize worst9-point physical expected-cost ratio to the pre-existing globally-qualified fixed-control cost envelope. Preserve the same grid, minimax objective, qualification pool and targets as the previous physical_minimax design.",
        "prefix_rule": "One symmetric randomized majority-sign rule indexed by(d,min(k,d-k)), with a fresh independent integer coin only on discordant pairs, for1<=d<=128. Probability denominator10^12; fixed snap tolerance1e-9.",
        "continuation_rule": "If prefix rule has not reported at128, continue with the unchanged frozen H8192 beta boundary at d=129,130,...8192. Do not evaluate beta at d=128. All physical observations including concordances and non-reporting paths are charged; physical cap expiry abstains.",
        "null_accounting": "Prefix report probability plus the conditional fair-sign beta-continuation report probability is constrained through the entire8192-sign horizon, even when a physical cap would reduce crossing risk.",
        "terminal_cost_semantics": "For each possible prefix state k, backward-DP terminal values integrate the TOTAL NB(d,r) waiting time from the start to a later report, truncated at physical cap. Terminal no-report paths cost the full physical cap. Prefix-suffix costs are never added twice.",
        "folding": "For a folded state j and k=128-j, multiply each suffix value by its own orientation likelihood ratio and average the two. A currently negative-majority state may later correctly report plus, so its positive-suffix contribution is retained.",
        "solver": {"method": "scipy1.5 sparse interior-point", "tol": 1e-9, "maxiter": 200, "rr": False},
        "validation_order": "First exact small-horizon physical DP, full sign-string enumeration, and folded/unfolded coefficient checks; then production coefficients; only afterward solve the LP.",
        "failure_rule": "Save failures without automatically changing grid, horizons, targets, continuation or objective.",
        "claim_boundary": "Classical finite-horizon planning with a fixed classical continuation, not an originality, SOTA or3x claim; development and benchmark evidence remain separate.",
        "new_observations": 0,
        "input_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in (BETA, CONTROL_GRID, Path(__file__).resolve())},
    }


def waiting_rewards(pair_cap, full_horizon, rate, exact=False):
    if exact:
        rate = Fraction(rate)
        before = [Fraction(0)]+[physical.exact_binomial_tail(pair_cap, rate, d) for d in range(1, full_horizon+1)]
        cost = [Fraction(0)]+[sum((1-physical.exact_binomial_tail(n, rate, d) for n in range(pair_cap)), Fraction(0))
                                for d in range(1, full_horizon+1)]
        return before, cost
    d = np.arange(1, full_horizon+1, dtype=float)
    rate = float(rate)
    before = binom.sf(d-1, pair_cap, rate)
    cost = d*(binom.sf(d, pair_cap+1, rate)/rate)+pair_cap*binom.cdf(d-1, pair_cap, rate)
    return np.concatenate(([0.], before)), np.concatenate(([0.], cost))


def beta_suffix_values(boundaries, prefix_horizon, pair_cap, q, rate, exact=False, rewards=None):
    """Return positive[k], negative[k], paid_pairs[k] for k=0,...prefix_H."""
    full_horizon = len(boundaries)-1
    assert 0 <= prefix_horizon <= full_horizon <= pair_cap
    before, cost = waiting_rewards(pair_cap, full_horizon, rate, exact) if rewards is None else rewards
    if exact:
        q = Fraction(q)
        vp = [Fraction(0)]*(full_horizon+2)
        vn = [Fraction(0)]*(full_horizon+2)
        vc = [Fraction(pair_cap)]*(full_horizon+2)
        for d in range(full_horizon, prefix_horizon, -1):
            fresh_p, fresh_n, fresh_c = [], [], []
            for k in range(d+1):
                if k >= boundaries[d]:
                    p, n, c = before[d], Fraction(0), cost[d]
                elif k <= d-boundaries[d]:
                    p, n, c = Fraction(0), before[d], cost[d]
                else:
                    p, n, c = ((1-q)*vp[k]+q*vp[k+1], (1-q)*vn[k]+q*vn[k+1],
                               (1-q)*vc[k]+q*vc[k+1])
                fresh_p.append(p); fresh_n.append(n); fresh_c.append(c)
            vp, vn, vc = fresh_p, fresh_n, fresh_c
        return {"positive": [(1-q)*vp[k]+q*vp[k+1] for k in range(prefix_horizon+1)],
                "negative": [(1-q)*vn[k]+q*vn[k+1] for k in range(prefix_horizon+1)],
                "paid_pairs": [(1-q)*vc[k]+q*vc[k+1] for k in range(prefix_horizon+1)]}
    q = float(q)
    vp = np.zeros(full_horizon+2); vn = vp.copy(); vc = np.full(full_horizon+2, float(pair_cap))
    for d in range(full_horizon, prefix_horizon, -1):
        vp = (1-q)*vp[:-1]+q*vp[1:]
        vn = (1-q)*vn[:-1]+q*vn[1:]
        vc = (1-q)*vc[:-1]+q*vc[1:]
        b = boundaries[d]
        if b <= d:
            vp[b:] = before[d]; vn[b:] = 0.; vc[b:] = cost[d]
            end = d-b+1
            vp[:end] = 0.; vn[:end] = before[d]; vc[:end] = cost[d]
    return {"positive": (1-q)*vp[:-1]+q*vp[1:],
            "negative": (1-q)*vn[:-1]+q*vn[1:],
            "paid_pairs": (1-q)*vc[:-1]+q*vc[1:]}


def fair_suffix_any_report(boundaries, prefix_horizon, exact=False):
    full_horizon = len(boundaries)-1
    if exact:
        values = [Fraction(0)]*(full_horizon+2)
        for d in range(full_horizon, prefix_horizon, -1):
            values = [Fraction(1) if k >= boundaries[d] or k <= d-boundaries[d] else (values[k]+values[k+1])/2
                      for k in range(d+1)]
        return [(values[k]+values[k+1])/2 for k in range(prefix_horizon+1)]
    values = np.zeros(full_horizon+2)
    for d in range(full_horizon, prefix_horizon, -1):
        values = .5*(values[:-1]+values[1:])
        b = boundaries[d]
        if b <= d:
            values[b:] = 1.; values[:d-b+1] = 1.
    return .5*(values[:-1]+values[1:])


def prefix_distribution(prefix, denominator, q):
    row = [Fraction(1)]
    plus, minus = [Fraction(0)], [Fraction(0)]
    for d in range(1, len(prefix)):
        fresh = [Fraction(0)]*(d+1)
        for k, mass in enumerate(row):
            fresh[k] += (1-q)*mass; fresh[k+1] += q*mass
        p = n = Fraction(0)
        for k, mass in enumerate(fresh):
            stop = mass*Fraction(prefix[d][min(k, d-k)], denominator)
            if 2*k > d:
                p += stop
            elif 2*k < d:
                n += stop
            else:
                assert stop == 0
            fresh[k] -= stop
        plus.append(p); minus.append(n); row = fresh
    return plus, minus, row


def hybrid_table(prefix, denominator, boundaries):
    return prefix+[[(denominator if d-j >= boundaries[d] else 0) for j in range(d//2+1)]
                   for d in range(len(prefix), len(boundaries))]


def enumerated_sign_first_passage(policy, denominator, q):
    full_horizon = len(policy)-1
    positive, negative = [Fraction(0)]*(full_horizon+1), [Fraction(0)]*(full_horizon+1)
    surviving = Fraction(0)
    for bits in itertools.product((0, 1), repeat=full_horizon):
        path = q**sum(bits)*(1-q)**(full_horizon-sum(bits))
        live = Fraction(1); k = 0
        for d, bit in enumerate(bits, 1):
            k += bit
            stop = live*Fraction(policy[d][min(k, d-k)], denominator)
            if 2*k > d:
                positive[d] += path*stop
            elif 2*k < d:
                negative[d] += path*stop
            else:
                assert stop == 0
            live -= stop
        surviving += path*live
    return positive, negative, surviving


def exact_small_checks():
    prefix, denominator = [[0], [1], [2, 0]], 6
    boundaries = [1, 2, 3, 3, 4, 4, 5]
    h, cap = 2, 8
    policy = hybrid_table(prefix, denominator, boundaries)
    checks = []
    for q, rate in ((Fraction(7, 10), Fraction(1, 5)), (Fraction(1, 2), Fraction(1, 2)),
                    (Fraction(1, 10), Fraction(1))):
        full = physical.first_passage(policy, denominator, q, exact=True)
        enumerated = enumerated_sign_first_passage(policy, denominator, q)
        assert full[:3] == enumerated
        mixture = physical.exact_waiting_mixture(full, rate, cap)
        direct = physical.exact_physical_dp(policy, denominator, rate, q, cap)
        assert mixture == direct
        before, cost = waiting_rewards(cap, len(boundaries)-1, rate, exact=True)
        suffix = beta_suffix_values(boundaries, h, cap, q, rate, exact=True, rewards=(before, cost))
        p, n, live = prefix_distribution(prefix, denominator, q)
        plus = sum((p[d]*before[d] for d in range(1, h+1)), Fraction(0))+sum((live[k]*suffix["positive"][k] for k in range(h+1)), Fraction(0))
        minus = sum((n[d]*before[d] for d in range(1, h+1)), Fraction(0))+sum((live[k]*suffix["negative"][k] for k in range(h+1)), Fraction(0))
        expected = sum(((p[d]+n[d])*cost[d] for d in range(1, h+1)), Fraction(0))+sum((live[k]*suffix["paid_pairs"][k] for k in range(h+1)), Fraction(0))
        assert (plus, minus, 1-plus-minus, expected) == direct
        floating = beta_suffix_values(boundaries, h, cap, float(q), float(rate))
        assert max(abs(float(suffix[key][k])-floating[key][k]) for key in suffix for k in range(h+1)) < 1e-12
        checks.append({"q": str(q), "r": str(rate), "full_sign_strings": 2**(len(boundaries)-1),
                       "forward_sign_dp_equals_full_enumeration": True,
                       "terminal_splice_equals_direct_physical_dp_exactly": True,
                       "float_suffix_agrees_with_exact": True})
    fair = Fraction(1, 2)
    p, n, live = prefix_distribution(prefix, denominator, fair)
    suffix = fair_suffix_any_report(boundaries, h, exact=True)
    risk = sum(p)+sum(n)+sum((mass*value for mass, value in zip(live, suffix)), Fraction(0))
    full = physical.first_passage(policy, denominator, fair, exact=True)
    assert risk == sum(full[0])+sum(full[1])
    # Verify exact orientation folding using a symmetric reference mixture.
    q = Fraction(7, 10); rate = Fraction(1, 5)
    hypotheses = (fair, q, 1-q)
    ref_rows = [prefix_distribution(prefix, denominator, value)[2] for value in hypotheses]
    ref_live = [sum((row[k] for row in ref_rows), Fraction(0))/len(hypotheses) for k in range(h+1)]
    actual_live = prefix_distribution(prefix, denominator, q)[2]
    suffix = beta_suffix_values(boundaries, h, cap, q, rate, exact=True)
    for key in suffix:
        folded = Fraction(0)
        for j in range(h//2+1):
            k = h-j
            reference_path = sum((v**k*(1-v)**j for v in hypotheses), Fraction(0))/len(hypotheses)
            upper_lr, lower_lr = q**k*(1-q)**j/reference_path, q**j*(1-q)**k/reference_path
            arrival = ref_live[k]+ref_live[j] if k != j else ref_live[k]
            coefficient = (upper_lr*suffix[key][k]+lower_lr*suffix[key][j])/2
            folded += arrival*coefficient
        assert folded == sum((mass*value for mass, value in zip(actual_live, suffix[key])), Fraction(0))
    return {"status": "passed_exact_small_hybrid_splice_checks", "cases": checks,
            "fair_null_prefix_plus_continuation_equals_full_exact": True,
            "folded_orientation_coefficients_equal_unfolded_exact": True,
            "beta_not_rechecked_at_prefix_horizon": True, "new_observations": 0}


def main():
    started = time.perf_counter()
    plan = frozen_plan()
    write_new(PLAN, plan)
    checks = exact_small_checks()
    checks["input_sha256"] = {"source": sha(Path(__file__).resolve()), "plan": sha(PLAN)}
    write_new(DIRECTORY/"small_coefficient_checks.json", checks)
    print(json.dumps({"small_checks": "passed", "elapsed_seconds": time.perf_counter()-started}), flush=True)
    artifact = json.loads(BETA.read_text(encoding="utf-8"))
    boundaries = artifact["boundaries"]
    assert len(boundaries)-1 == plan["full_sign_horizon"]
    h, cap = plan["prefix_sign_horizon"], plan["physical_pair_cap"]
    fair = fair_suffix_any_report(boundaries, h)
    rewards = {tuple(value): waiting_rewards(cap, len(boundaries)-1, Fraction(*value)) for value in plan["r_exact"]}
    rows = []
    for qi, q_value in enumerate(plan["q_exact"]):
        for ri, r_value in enumerate(plan["r_exact"]):
            values = beta_suffix_values(boundaries, h, cap, Fraction(*q_value), Fraction(*r_value), rewards=rewards[tuple(r_value)])
            rows.append({"grid_id": 3*qi+ri, "q_exact": q_value, "r_exact": r_value,
                         **{key: list(map(float, array)) for key, array in values.items()}})
            print(json.dumps({"production_suffix_point_complete": 3*qi+ri}), flush=True)
    result = {"status": "float_beta_continuation_coefficients_with_exact_small_splice_checks",
              "prefix_horizon": h, "full_horizon": len(boundaries)-1, "physical_pair_cap": cap,
              "fair_null_any_report_by_plus_count": list(map(float, fair)), "rows": rows,
              "schema": "Each array is indexed by actual plus count k=0..128, without folding. paid_pairs is total paid cost from time zero, not a suffix increment. First beta check is at129.",
              "input_sha256": plan["input_sha256"], "plan_sha256": sha(PLAN),
              "elapsed_seconds": time.perf_counter()-started, "new_observations": 0,
              "claim_boundary": "These are floating optimization coefficients, not outward certificates. The final frozen hybrid needs independent full-horizon null and physical-cost verification after policy rounding."}
    assert all(sha(ROOT/path) == value for path, value in plan["input_sha256"].items())
    write_new(DIRECTORY/"terminal_coefficients_float.json", result)
    print(json.dumps({"output": str(DIRECTORY/"terminal_coefficients_float.json"), "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
