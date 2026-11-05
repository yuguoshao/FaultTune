"""Deterministic checks of the fit-only finite-grid structured design.

This is an optimization/API check on synthetic fitted states, not a
Monte Carlo experiment, a statistical certificate, or a hardware result.
"""
from collections import Counter
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from certified_likelihood import null_fit
from design_policy import (DEFAULT_CONFIG, GRID, bernoulli_kl, config_dict,
                           design_from_fit, rational_grid)

ROOT = Path(__file__).resolve().parents[2]


def coordinates(grid):
    return np.array([[1., arm["cos_numerator"]/arm["denominator"],
                      arm["sin_numerator"]/arm["denominator"]] for arm in grid])


def grid_checks(grid):
    keys = [(arm["cos_numerator"], arm["sin_numerator"], arm["denominator"])
            for arm in grid]
    assert len(set(keys)) == len(keys)
    assert all(c*c+s*s == d*d and d > 0 for c, s, d in keys)
    assert all((c, -s, d) in keys for c, s, d in keys)
    assert all(arm["index"] == i for i, arm in enumerate(grid))
    assert (1, 0, 1) in keys and (0, 1, 1) in keys and (0, -1, 1) in keys
    angles = sorted(math.atan2(s, c) % (2*math.pi) for c, s, d in keys)
    gaps = np.diff(angles+[angles[0]+2*math.pi])
    assert float(gaps.max()) < 2*(2*math.pi/len(grid))
    return {"size": len(grid), "exact_unit_circle": True,
            "sin_reflection_closed": True, "psr_included": True,
            "maximum_angular_gap_radians": float(gaps.max()),
            "maximum_coordinate_denominator": max(d for c, s, d in keys)}


def stable_kl_checks():
    pairs = [(p, q) for p, q in ((.5, .5), (.500000000001, .5), (.51, .49),
                                 (.0001, .00010000001), (1e-8, 1e-8+1e-12),
                                 (.99999999, .99999998), (.001, .9), (.9, .001))]
    maximum = 0.
    with localcontext() as ctx:
        ctx.prec = 80
        for p, q in pairs:
            pp, qq = Decimal.from_float(p), Decimal.from_float(q)
            reference = float(pp*(pp/qq).ln()+(1-pp)*((1-pp)/(1-qq)).ln())
            actual = float(bernoulli_kl(p, q))
            error = abs(actual-reference)/max(abs(reference), 1e-300)
            maximum = max(maximum, error)
            assert error < 3e-10
    return {"pairs": len(pairs), "reference_decimal_precision": 80,
            "maximum_relative_error": maximum}


def invalid_input_checks():
    cases = [lambda: rational_grid(64), lambda: rational_grid(3),
             lambda: rational_grid(65, 2),
             lambda: design_from_fit([.5, .5, 0.]),
             lambda: design_from_fit([0., 0., 0.]),
             lambda: design_from_fit([.5, 0., float("nan")]),
             lambda: design_from_fit([.5, 0.]),
             lambda: design_from_fit([.5, 0., .1], grid=GRID[:-1])]
    for call in cases:
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("invalid design input was accepted")
    return {"rejected_cases": len(cases)}


def fitted_states():
    states = []
    for a in (.0001, .001, .01, .1, .3, .5, .7, .9, .99, .999, .9999):
        for radial_fraction in (.1, .6, .98):
            for phase in (.02, .35, .9, 1.5, 2.8):
                radius = radial_fraction*min(a, 1-a)
                states.append([a, radius*math.cos(phase), radius*math.sin(phase)])
    rng = np.random.RandomState(2026110301)
    for unused in range(48):
        a = float(1/(1+math.exp(-rng.uniform(-8, 8))))
        radius = float(rng.uniform(.05, .98))*min(a, 1-a)
        phase = float(rng.uniform(.01, math.pi-.01))
        states.append([a, radius*math.cos(phase), radius*math.sin(phase)])
    return states


def validate_result(result, grid):
    weights = np.array(result["weights"])
    assert weights.shape == (len(grid),) and np.all(weights >= 0)
    assert abs(float(weights.sum())-1) < 2e-14
    assert result["info_lower"] >= 0 and result["info_upper"] >= 0
    assert result["info_lower"] <= result["info_upper"]+1e-12*max(result["info_upper"], 1e-14)
    assert result["certified"] is False
    assert len(result["support"]) <= 3
    for entry in result["support"]:
        assert 0 <= entry["arm"] < len(grid)
        assert entry["weight"] == result["weights"][entry["arm"]]
    a, b, c = result["null_candidate"]
    assert c == 0 and abs(b) <= min(a, 1-a)


def union_grid(first, second):
    seen, result = set(), []
    for arm in first+second:
        key = (arm["cos_numerator"], arm["sin_numerator"], arm["denominator"])
        if key not in seen:
            seen.add(key)
            result.append(dict(arm, index=len(result)))
    return result


def main():
    started = time.perf_counter()
    grid129 = rational_grid(129)
    union = union_grid(GRID, grid129)
    grids = [grid_checks(grid) for grid in (GRID, grid129, union)]
    reference_kl = stable_kl_checks()
    invalid = invalid_input_checks()
    features = coordinates(GRID)
    keys = [(arm["cos_numerator"], arm["sin_numerator"], arm["denominator"])
            for arm in GRID]
    reflection = [keys.index((c, -s, d)) for c, s, d in keys]
    states = fitted_states()
    records, results = [], []
    sign_error = 0.
    for index, theta in enumerate(states):
        fit = ({"A": theta[0], "B": theta[1], "C": theta[2]} if index % 3 == 0
               else {"theta": np.array(theta)} if index % 3 == 1 else theta)
        result = design_from_fit(fit)
        validate_result(result, GRID)
        reflected = design_from_fit([theta[0], theta[1], -theta[2]])
        validate_result(reflected, GRID)
        sign_error = max(sign_error, float(np.max(np.abs(np.array(result["weights"])[reflection]
                                                        -np.array(reflected["weights"])))))
        assert abs(result["info_lower"]-reflected["info_lower"]) < 2e-13*max(result["info_upper"], 1e-14)
        assert result["relative_gap"] < DEFAULT_CONFIG.acceptable_relative_gap
        p, q = features.dot(theta), features.dot(result["null_candidate"])
        pair_q = theta[0]+theta[1]*features[:, 1]
        equal_rates = bernoulli_kl(p, pair_q)
        best_equal_pair = float(np.max(.5*(equal_rates+equal_rates[reflection])))
        psr = keys.index((0, 1, 1))
        psr_rate = float(.5*(equal_rates[psr]+equal_rates[reflection[psr]]))
        assert result["info_lower"] >= best_equal_pair-1e-8*max(best_equal_pair, 1e-20)
        record = {key: result[key] for key in
                  ("fit_theta", "info_lower", "info_upper", "relative_gap", "support",
                   "solver_status", "epigraph_success", "lp_success", "warnings", "elapsed_seconds")}
        record.update({"case_id": index, "equal_psr_information": psr_rate,
                       "best_equal_pair_information": best_equal_pair,
                       "information_ratio_to_equal_psr": result["info_lower"]/psr_rate,
                       "information_ratio_to_best_equal_pair": result["info_lower"]/best_equal_pair})
        records.append(record)
        results.append(result)
    assert sign_error == 0.

    oracle_records = []
    for index in range(0, len(states), 7):
        result, theta = results[index], np.array(states[index])
        weights = np.array(result["weights"])
        probabilities = features.dot(theta)
        rows = [dict(GRID[i], successes=float(weights[i]*probabilities[i]), trials=float(weights[i]))
                for i in np.flatnonzero(weights > 0)]
        reference = null_fit(rows, 1, candidate=result["null_candidate"])
        q = features.dot(reference["candidate"])
        inner_upper = float(weights.dot(bernoulli_kl(probabilities, q)))
        inner_lower = inner_upper-reference["support_gap"]
        scale = max(result["info_upper"], 1e-20)
        assert result["info_lower"] <= inner_upper+2e-8*scale+1e-16
        assert inner_lower <= result["weighted_candidate_kl"]+2e-8*scale+1e-16
        oracle_records.append({"case_id": index, "inner_value": inner_upper,
                               "inner_lower": inner_lower, "oracle_support_gap": reference["support_gap"],
                               "oracle_success": reference["success"],
                               "C_null": reference["candidate"][2],
                               "relative_value_difference": abs(inner_upper-result["weighted_candidate_kl"])/scale})

    refinements = []
    for index in range(0, len(states), 6):
        fine = design_from_fit(states[index], grid129)
        both = design_from_fit(states[index], union)
        validate_result(fine, grid129)
        validate_result(both, union)
        assert fine["relative_gap"] < DEFAULT_CONFIG.acceptable_relative_gap
        assert both["relative_gap"] < DEFAULT_CONFIG.acceptable_relative_gap
        coarse = results[index]
        assert coarse["info_lower"] <= both["info_upper"]+1e-9*both["info_upper"]
        assert fine["info_lower"] <= both["info_upper"]+1e-9*both["info_upper"]
        refinements.append({"case_id": index, "info65": coarse["info_lower"],
                            "info129": fine["info_lower"], "union_lower": both["info_lower"],
                            "union_upper": both["info_upper"],
                            "ratio129_to65": fine["info_lower"]/coarse["info_lower"],
                            "relative_65_loss_to_union_upper": max(0., 1-coarse["info_lower"]/both["info_upper"])})

    special = []
    for theta in ([.5, 0., 0.], [.02, .01, 0.], [.5, .2, 1e-12], [.1, .08, 1e-7]):
        result = design_from_fit(theta)
        validate_result(result, GRID)
        if theta[2] == 0:
            assert result["info_lower"] == result["info_upper"] == 0
        special.append({key: result[key] for key in
                        ("fit_theta", "solver_status", "info_lower", "info_upper", "relative_gap",
                         "epigraph_success", "lp_success", "elapsed_seconds")})

    timings = [record["elapsed_seconds"] for record in records]
    output = {"status": "passed", "scope": "Synthetic fitted-state optimization checks only; no Monte Carlo or SOTA claim",
              "config": config_dict(), "grids": grids, "stable_kl": reference_kl,
              "invalid_inputs": invalid, "main_fitted_states": len(states),
              "sign_reflection_checks": len(states), "maximum_weight_reflection_error": sign_error,
              "maximum_relative_primal_dual_gap": max(record["relative_gap"] for record in records),
              "solver_status_counts": dict(Counter(record["solver_status"] for record in records)),
              "epigraph_success_count": sum(record["epigraph_success"] for record in records),
              "lp_success_count": sum(record["lp_success"] for record in records),
              "warning_cases": sum(bool(record["warnings"]) for record in records),
              "runtime_seconds": {"median": float(np.median(timings)),
                                  "p95": float(np.percentile(timings, 95)), "max": max(timings)},
              "maximum_65_loss_to_union_upper": max(record["relative_65_loss_to_union_upper"] for record in refinements),
              "minimum_129_to65_ratio": min(record["ratio129_to65"] for record in refinements),
              "maximum_129_to65_ratio": max(record["ratio129_to65"] for record in refinements),
              "refinement_note": "65 and 129 are not nested; union is a finite-grid comparison, not a continuum bound",
              "oracle_crosschecks": oracle_records, "refinements": refinements,
              "special_cases": special, "records": records,
              "elapsed_seconds": time.perf_counter()-started,
              "sources": {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                          for name in ("design_policy.py", "design_checks.py", "certified_likelihood.py")}}
    target = ROOT / "results/structured_validation/design_checks.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items()
                      if key not in ("records", "oracle_crosschecks", "refinements")}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
