"""Recover the preregistered20 finite-grid results after a Python3.8 typo.

The original scan, failed rows and sources stay unchanged. This repeats only
the same preselected calls and audits their fixed weights at100-digit precision.
"""
from fractions import Fraction
import json
from pathlib import Path
import time

import mpmath as mp

import reflection_counterexamples as scan

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = scan.DIRECTORY
OUT = DIRECTORY / "three_support_technical_retry.json"


def as_mp(value):
    value = Fraction(value)
    return mp.mpf(value.numerator)/value.denominator


def main():
    started = time.perf_counter()
    rows = [json.loads(line) for line in scan.ROWS.read_text(encoding="utf-8").splitlines()]
    selected = [row for row in rows if row.get("finite_grid_failure") == "unsupported operand type(s) for |: 'dict' and 'dict'"]
    assert len(selected) == 20
    sources = (Path(__file__).resolve(), Path(scan.__file__).resolve(), scan.PLAN, scan.ROWS,
               ROOT/"experiments/structured_validation/design_policy.py")
    hashes = {path.relative_to(ROOT).as_posix(): scan.sha(path) for path in sources}
    scan.write_new(DIRECTORY/"three_support_retry_plan.json", {
        "status": "fixed_technical_retry_before_reexecution", "parameter_ids": [row["parameter"]["parameter_id"] for row in selected],
        "scientific_registry_changed": False, "reason": "Python3.8 cannot merge dicts with|. Re-execute the same20 oracle calls and serialize with ordinary dictionary update.",
        "high_precision_check": "Interpret the saved arm weights and candidate null coefficients as exact binary rationals; normalize weights exactly, rebuild the true model from rational/algebraic registry expressions, and evaluate the convex first-order lower bound over the four vertices of the physical affine-null diamond.",
        "source_sha256": hashes, "new_simulation_observations": 0})
    design = scan.refined.initial.design
    grid = design.rational_grid(129, max_denominator=1024)
    output = []
    for previous in selected:
        row = {"parameter": previous["parameter"]}
        try:
            result = design.design_from_fit(previous["float_screen_theta"], grid=grid)
            row["finite_grid"] = {key: result[key] for key in ("support", "weights", "null_candidate", "solver_status", "epigraph_success", "lp_success", "info_lower", "info_upper", "relative_gap")}
            mp.mp.dps = 100
            A, B, C = scan.model(previous["parameter"])
            weights = [Fraction.from_float(float(value)) for value in result["weights"]]
            total = sum(weights, Fraction(0)); weights = [value/total for value in weights]
            aa, bb = map(as_mp, result["null_candidate"][:2])
            value, ga, gb = mp.mpf(0), mp.mpf(0), mp.mpf(0)
            for arm, weight in zip(grid, weights):
                if not weight:
                    continue
                z = as_mp(Fraction(arm["cos_numerator"], arm["denominator"]))
                sine = as_mp(Fraction(arm["sin_numerator"], arm["denominator"]))
                p, q = A+B*z+C*sine, aa+bb*z
                assert 0 < p < 1 and 0 < q < 1
                w = as_mp(weight)
                value += w*(scan.entropy(p)-p*mp.log(q)-(1-p)*mp.log1p(-q))
                derivative = w*(q-p)/(q*(1-q))
                ga += derivative; gb += derivative*z
            residual = min(ga*(a-aa)+gb*(b-bb) for a, b in ((0, 0), (mp.mpf('.5'), mp.mpf('.5')), (1, 0), (mp.mpf('.5'), -mp.mpf('.5'))))
            lower = max(mp.mpf(0), value+residual)
            pair = previous.get("high_precision") or scan.high_precision(previous["parameter"], previous["float_screen"])
            radius = mp.mpf(pair["radius"])
            row["high_precision_fixed_primal_diagnostic"] = {
                "precision_digits": 100, "weighted_value": mp.nstr(value, 75),
                "gradient_lower_correction": mp.nstr(residual, 75), "convex_lower_bound": mp.nstr(lower, 75),
                "lower_over_best_numeric_pair": mp.nstr(lower/radius, 75),
                "bound_formula": "F(a,b)+min_{v in{(0,0),(.5,.5),(1,0),(.5,-.5)}} gradF(a,b).(v-(a,b))",
                "status": "High-precision evaluation of a valid convex bound formula; not an outward numerical certificate and pair maximum still not globally certified."}
        except Exception as exc:
            row["failure"] = str(exc)
        output.append(row)
    assert all(scan.sha(ROOT/path) == expected for path, expected in hashes.items())
    result = {"status": "completed_same20_point_technical_retry", "rows": output,
              "rational_arm_grid": grid, "original_failures_retained": True,
              "maximum_high_precision_primal_lower_over_numeric_pair": max(float(row["high_precision_fixed_primal_diagnostic"]["lower_over_best_numeric_pair"]) for row in output if "high_precision_fixed_primal_diagnostic" in row),
              "support_size_counts": {str(n): sum(len(row.get("finite_grid", {}).get("support", [])) == n for row in output) for n in sorted({len(row.get("finite_grid", {}).get("support", [])) for row in output})},
              "failures": sum("failure" in row for row in output), "source_sha256": hashes,
              "actual_three_support_counterexample_certified": False,
              "elapsed_seconds": time.perf_counter()-started, "new_simulation_observations": 0}
    scan.write_new(OUT, result)
    print(json.dumps({key: result[key] for key in ("status", "maximum_high_precision_primal_lower_over_numeric_pair", "support_size_counts", "failures", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
