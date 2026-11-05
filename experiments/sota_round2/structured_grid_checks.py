"""Deterministic arithmetic, normalization and adaptive-action checks."""
from decimal import Decimal, localcontext
from fractions import Fraction
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.special import gammaln, logsumexp

import structured_grid_predictive as grid
from adaptive_safe_replay import digest, write_new


def exact_kt(k, n):
    return Fraction(math.factorial(2*k) * math.factorial(2*(n-k)),
                    4**n * math.factorial(k) * math.factorial(n-k) * math.factorial(n))


def exact_partial(arms, n, k, selected):
    total = Fraction(0)
    for index in selected:
        a, b, c, weight = grid.models()[index]
        value = weight
        for arm, count, success in zip(arms, n, k):
            p = a + b * Fraction(arm["cos_numerator"], arm["denominator"]) + c * Fraction(arm["sin_numerator"], arm["denominator"])
            value *= p**success * (1-p)**(count-success)
        total += value
    kt = math.prod(exact_kt(success, count) for count, success in zip(n, k))
    return (total + kt) / 2


def main():
    started = time.perf_counter()
    arms = [grid.likelihood.rational_circle(1, 1), grid.likelihood.rational_circle(-1, 1),
            grid.likelihood.rational_circle(1, 2)]
    models = grid.models()
    assert len(models) == 7141
    assert sum((m[3] for m in models), Fraction(0)) == 1
    assert all(0 < a < 1 and b*b+c*c < min(a, 1-a)**2 for a,b,c,w in models)
    partial_checks = []
    patterns = (([0,0,0],[0,0,0]), ([1,1,1],[0,1,0]), ([3,2,1],[2,0,1]),
                ([2,3,2],[0,3,1]), ([12,12,1],[12,0,1]))
    for n, k in patterns:
        for terms in (1, 4, 16):
            result = grid.numerator_lower_bound(arms, n, k, terms)
            exact = exact_partial(arms, n, k, result["retained_model_indices"])
            with localcontext() as context:
                context.prec = 110
                expected = (Decimal(exact.numerator)/Decimal(exact.denominator)).ln()
            lo, hi = map(Decimal, result["partial_numerator_log_bounds"])
            assert lo <= expected <= hi
            full_float = grid.screen_log_numerator(arms, n, k)
            assert float(lo) <= full_float + 1e-11
            kt = sum(math.log(float(exact_kt(s,t))) for t,s in zip(n,k))
            assert full_float >= kt - math.log(2) - 1e-11
            partial_checks.append({"trials":n,"successes":k,"terms":terms,
                                   "partial_log_interval_width":str(hi-lo),
                                   "full_minus_certified_partial_log":full_float-float(lo)})
    # Sum ordered path probabilities under a genuinely adaptive action rule.
    # This is a separate tree computation using posterior predictive ratios;
    # normalized Q must remain a probability law even when arms depend on data.
    maximum_error = 0.
    frontier = [(np.zeros(3,dtype=int), np.zeros(3,dtype=int), 0., 1.)]
    horizon_mass, crossing_probability = [], 0.
    # Under the true all-arm p=.5 model each path has probability 2^-depth.
    for depth in range(1, 9):
        following = []
        for n, k, old_q, true_probability in frontier:
            arm = (int(k.sum()) + depth) % 3
            children = []
            for outcome in (0,1):
                nn, kk = n.copy(), k.copy()
                nn[arm] += 1
                kk[arm] += outcome
                logq = grid.screen_log_numerator(arms,nn,kk)
                children.append(math.exp(logq-old_q))
                following.append((nn,kk,logq,true_probability/2))
            maximum_error = max(maximum_error,abs(sum(children)-1))
        frontier = following
        mass = sum(math.exp(row[2]) for row in frontier)
        horizon_mass.append(mass)
        assert abs(mass-1) < 2e-12
    assert maximum_error < 2e-12
    reject_checks = 0
    for n,k in (([True,0,0],[0,0,0]),([-1,0,0],[0,0,0]),([1,0,0],[2,0,0]),([1.5,0,0],[1,0,0])):
        try:
            grid.screen_log_numerator(arms,n,k)
        except ValueError:
            reject_checks += 1
        else:
            raise AssertionError("Invalid counts accepted")
    certificates = []
    for sign in (-1,1):
        certificate = grid.certificate(arms,[24,24,1],[24,0,1],sign,max_terms=64)
        assert certificate["decision"] in ("abstain","reject")
        if sign == -1:
            assert certificate["decision"] == "abstain"
        certificates.append(certificate)
    sources = {str(p.relative_to(grid.ROOT)).replace("\\","/"):digest(p) for p in
        (Path(__file__).resolve(), Path(grid.__file__).resolve(),
         grid.ROOT/"experiments/structured_validation/certified_likelihood.py",
         grid.ROOT/"experiments/circuit_validation/direct_tests.py")}
    result = {"status":"passed", "sources":sources, "grid_sha256":grid.grid_sha256(),
              "physical_models":len(models), "exact_weights_sum":"1", "partial_sum_checks":partial_checks,
              "adaptive_tree_terminal_paths":len(frontier), "adaptive_tree_horizon_mass":horizon_mass,
              "maximum_predictive_normalization_error":maximum_error,"invalid_input_rejections":reject_checks,
              "certificate_examples":certificates,"elapsed_seconds":time.perf_counter()-started,
              "limits":"Deterministic finite checks support arithmetic and normalized prediction. Anytime validity follows the documented LR proof, not these finite examples; no performance claim."}
    directory = grid.ROOT/"results/sota_round2"
    directory.mkdir(parents=True,exist_ok=True)
    write_new(directory/"structured_grid_checks.json",result)
    print(json.dumps({k:result[k] for k in ("status","physical_models","grid_sha256","maximum_predictive_normalization_error","elapsed_seconds")},indent=2))


if __name__ == "__main__":
    main()
