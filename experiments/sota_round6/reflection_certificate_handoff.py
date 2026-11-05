"""Exact model specifications and rational seeds for root's interval checker.

This selects already evaluated extreme rows. It runs no further search and
does not label high-precision stationarity brackets as interval certificates.
"""
from fractions import Fraction
import json
import math
from pathlib import Path

import mpmath as mp

import reflection_counterexamples as scan

ROOT = Path(__file__).resolve().parents[2]
OUT = scan.DIRECTORY / "interval_certificate_handoff.json"


def rational(value):
    value = Fraction(value)
    return {"numerator": str(value.numerator), "denominator": str(value.denominator), "exact_expression": str(value)}


def sqrt_record(squared):
    squared = Fraction(squared)
    a, b = math.isqrt(squared.numerator), math.isqrt(squared.denominator)
    if a*a == squared.numerator and b*b == squared.denominator:
        return {"kind": "rational", **rational(Fraction(a, b))}
    return {"kind": "positive_algebraic_square_root", "squared_exact": rational(squared),
            "exact_expression": "sqrt("+str(squared)+")"}


def main():
    mp.mp.dps = 100
    rows = {row["parameter"]["parameter_id"]: row for row in map(json.loads, scan.ROWS.read_text(encoding="utf-8").splitlines())}
    records = []
    for identifier in (35, 327, 335, 343, 351):
        row = rows[identifier]; point = row["parameter"]; hp = row["high_precision"]
        A, rho = Fraction(point["A"]), Fraction(point["rho"])
        R = min(A, 1-A)*rho
        if "B_over_R" in point:
            u = Fraction(point["B_over_R"])
            B, C = {"kind": "rational", **rational(R*u)}, sqrt_record(R*R*(1-u*u))
        else:
            v = Fraction(point["C_over_R"])
            B, C = sqrt_record(R*R*(1-v*v)), {"kind": "rational", **rational(R*v)}
        left, right = map(mp.mpf, hp["local_angle_bracket"])
        records.append({
            "parameter_id": identifier, "registered_parameter": point,
            "exact_model": {"A": {"kind": "rational", **rational(A)}, "B": B, "C": C},
            "exact_physical_slack_squared": rational(min(A, 1-A)**2-R*R),
            "high_precision_local_angle_bracket_seed": hp["local_angle_bracket"],
            "cosine_bracket_seed": [mp.nstr(mp.cos(right), 75), mp.nstr(mp.cos(left), 75)],
            "candidate_rational_affine_null": {
                "a": rational(Fraction(hp["null_intercept"])),
                "b": rational(Fraction(hp["null_slope"])),
                "a_decimal": hp["null_intercept"], "b_decimal": hp["null_slope"]},
            "pair_radius_diagnostic": hp["radius"],
            "tangent_physical_margin_diagnostic": hp["null_physical_margin"],
            "strongest_curvature_witnesses": {key: value for key, value in hp["witnesses"].items() if "curvature" in key},
            "status": "Exact rational/algebraic model; angle bracket and tangent are high-precision seeds requiring independent interval verification. Rounded rational tangent need not touch exactly, so a relative-gap U/L certificate is the natural goal.",
        })
    sources = (Path(__file__).resolve(), scan.PLAN, scan.ROWS, scan.DIRECTORY/"summary.json", scan.DIRECTORY/"three_support_technical_retry.json")
    result = {"status": "handoff_without_additional_parameter_scan", "records": records,
              "suggested_priority": [35, 343, 327, 335, 351],
              "reason": "35 is extreme lowA with rational B,C;343 is the closest retained normalized-curvature margin to0;327/335 stress nonuniform weak-signal limits;351 is an exactly physical-boundary rational model.",
              "seed_warning": "The brackets were obtained with high-precision derivatives, not outward signs. Recheck them before using them as rigorous root brackets. The certificate need not assume pair globality: any fixed reflected pair supplies a lower bound, and one fixed rational tangent supplies an all-angle upper bound if independently interval-certified.",
              "new_simulation_observations": 0, "source_sha256": {p.relative_to(ROOT).as_posix(): scan.sha(p) for p in sources}}
    scan.write_new(OUT, result)
    print(json.dumps({"output": str(OUT), "sha256": scan.sha(OUT), "parameter_ids": list(rows[identifier]["parameter"]["parameter_id"] for identifier in (35, 327, 335, 343, 351))}))


if __name__ == "__main__":
    main()
