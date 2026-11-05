"""Deterministic checks of structured-null certificates; no Monte Carlo."""
import hashlib
import itertools
import json
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
import time

from certified_likelihood import (ROOT, _Bounds, rational_circle, support_bounds,
                                 null_fit, null_loglik_upper, loglik_bounds,
                                 rational_prediction_loglik_bounds)


def row(c, s, d, k, n):
    return {"cos_numerator": c, "sin_numerator": s, "denominator": d,
            "successes": k, "trials": n}


def exact_loglik(rows, point):
    with localcontext() as context:
        context.prec = 100
        value = Decimal(0)
        for record in rows:
            c = Decimal(record["cos_numerator"])/record["denominator"]
            s = Decimal(record["sin_numerator"])/record["denominator"]
            p = Decimal(point[0])+Decimal(point[1])*c+Decimal(point[2])*s
            k, n = record["successes"], record["trials"]
            if (p == 0 and k) or (p == 1 and n-k):
                return Decimal("-Infinity")
            if k:
                value += k*p.ln()
            if n-k:
                value += (n-k)*(1-p).ln()
        return value


def run():
    started = time.perf_counter()
    support_checks = 0
    circles = [rational_circle(j, 4) for j in range(-8, 9)]
    for circle in circles:
        assert circle["cos_numerator"]**2+circle["sin_numerator"]**2 == circle["denominator"]**2
    for sign in (-1, 1):
        for gradient in itertools.product((-4, -1, 0, 2, 5), repeat=3):
            bound = support_bounds(gradient, sign)
            upper = Fraction(bound.hi)
            for a in (Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(1)):
                radius = min(a, 1-a)
                for circle in circles:
                    b = radius*Fraction(circle["cos_numerator"], circle["denominator"])
                    c = radius*Fraction(circle["sin_numerator"], circle["denominator"])
                    if sign*c > 0:
                        continue
                    assert gradient[0]*a+gradient[1]*b+gradient[2]*c <= upper
                    support_checks += 1
            with localcontext() as context:
                context.prec = 100
                ga, gb, gc = [Decimal(x) for x in gradient]
                radius = (gb*gb+max(Decimal(0), -sign*gc)**2).sqrt()
                exact = max(Decimal(0), ga, (ga+radius)/2)
                assert bound.lo <= exact <= bound.hi
    # Interval gradients, especially intervals straddling zero, must not
    # produce a negative squared lower endpoint or invalid sqrt.
    interval = [_Bounds(-2, 3), _Bounds(-4, 1), _Bounds(-3, 5)]
    for sign in (-1, 1):
        bound = support_bounds(interval, sign)
        for values in itertools.product((-2, 3), (-4, 1), (-3, 5)):
            assert support_bounds(values, sign).hi <= bound.hi
    cases = []
    analytic = [row(0, -1, 1, 10, 100), row(0, 1, 1, 80, 100)]
    for sign, optimum in ((1, (Decimal('.45'), 0, 0)),
                          (-1, (Decimal('.45'), 0, Decimal('.35')))):
        fit = null_fit(analytic, sign)
        cert = null_loglik_upper(analytic, sign, fit["candidate"])
        exact = exact_loglik(analytic, optimum)
        gap = Decimal(cert["loglik_upper_decimal"])-exact
        assert 0 <= gap < Decimal('0.001'), (sign, gap, cert)
        cases.append({"sign": sign, "case": "analytic two-angle null",
                      "certified_upper_minus_exact": str(gap), "fit": fit})
    boundary_cases = 0
    for successes in (0, 37):
        data = [row(3, 4, 5, successes, 37), row(-5, 12, 13, successes, 37)]
        for sign in (-1, 1):
            cert = null_loglik_upper(data, sign)
            assert Decimal(cert["loglik_upper_decimal"]) == 0
            boundary_cases += 1
    # An infeasible null candidate is permitted: tangent validity only
    # requires all observed probabilities be strictly interior.
    data = [row(3, 4, 5, 12, 41), row(-3, -4, 5, 29, 43), row(0, 1, 1, 15, 31)]
    certificates = [null_loglik_upper(data, sign, [.5, .11, .23]) for sign in (-1, 1)]
    point_checks = 0
    for sign, cert in zip((-1, 1), certificates):
        upper = Decimal(cert["loglik_upper_decimal"])
        for a, b, c in itertools.product((Decimal('.1'), Decimal('.3'), Decimal('.5'), Decimal('.7'), Decimal('.9')),
                                          (Decimal('-.08'), Decimal(0), Decimal('.08')),
                                          (Decimal('-.05'), Decimal(0), Decimal('.05'))):
            if sign*c <= 0 and b*b+c*c <= min(a, 1-a)**2:
                assert exact_loglik(data, (a, b, c)) <= upper
                point_checks += 1
    # Integer count validation is separate from numerical pseudo-count fits.
    weighted = [row(3, 4, 5, .13, .5), row(-3, 4, 5, .19, .5)]
    assert null_fit(weighted, 1)["certified"] is False
    rejected = 0
    for invalid in (weighted, [row(3, 4, 6, 1, 2)], [row(0, 1, 1, 3, 2)]):
        try:
            null_loglik_upper(invalid, 1)
        except ValueError:
            rejected += 1
    assert rejected == 3
    # Extremely fine rational circles retain their exact mathematical
    # feature, even when a binary64 cosine proposal rounds to one.
    fine = rational_circle(1, 10**30)
    fine.update(successes=19, trials=40)
    fine_cert = null_loglik_upper([fine], 1, [.5, .2, .1])
    assert fine_cert["model_features"] == "exact rational unit-circle coordinates"
    predicted = rational_prediction_loglik_bounds(analytic, [(1, 3), (2, 3)])
    with localcontext() as context:
        context.prec = 100
        exact = 30*(Decimal(1)/3).ln()+170*(Decimal(2)/3).ln()
        assert Decimal(predicted["lower_decimal"]) <= exact <= Decimal(predicted["upper_decimal"])
    candidate_likelihood = loglik_bounds(data, [.5, .11, .23])
    exact = exact_loglik(data, tuple(Decimal.from_float(x) for x in [.5, .11, .23]))
    assert Decimal(candidate_likelihood["lower_decimal"]) <= exact <= Decimal(candidate_likelihood["upper_decimal"])
    folder = ROOT / "results" / "structured_validation"
    folder.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__).resolve(), Path(__file__).resolve().with_name("certified_likelihood.py"),
               ROOT / "experiments" / "circuit_validation" / "direct_tests.py"]
    result = {"status": "passed", "support_point_inequalities": support_checks,
              "support_formula_high_precision_checks": 250, "gradient_interval_corner_checks": 16,
              "analytic_cases": cases, "exact_boundary_cases": boundary_cases,
              "infeasible_candidate_null_grid_checks": point_checks,
              "invalid_input_rejections": rejected, "fine_rational_feature_check": True,
              "predictive_loglik_enclosure": True, "candidate_loglik_enclosure": True,
              "monte_carlo": False, "formal_verification": False,
              "theory": {
                  "tangent": "For every null theta, l(theta)<=l(theta0)+g(theta0) dot (theta-theta0). The exact closed physical support upper bounds the last term.",
                  "support": "h(g)=max(0,gA,(gA+sqrt(gB^2+max(0,-sign*gC)^2))/2).",
                  "numerator": "A product of normalized predictable predictive probabilities defines Q_n. For each true null theta, Q_n/L_theta,n is a nonnegative martingale (or supermartingale at zero-probability boundaries).",
                  "denominator": "exp(U_n)>=sup_null L_theta,n implies Q_n/exp(U_n)<=Q_n/L_true,n at every time.",
                  "two_directions": "If both directions share the SAME Q_n, every wrong-direction evidence is dominated by the SAME true-model likelihood-ratio process, so threshold 1/delta controls their union. Distinct direction-specific Q_n instead require an explicit allocation such as delta/2 each.",
                  "count_aggregation": "Denominator may aggregate identical queries; numerator must preserve the predictive probabilities used before each observation/block. No binomial coefficients in either ordered-outcome likelihood.",
                  "limits": "Numerical enclosure prototype, not formal proof-assistant verification; no SOTA or finite-cost claim."},
              "sources": {str(path.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
              "elapsed_seconds": time.perf_counter()-started}
    (folder / "certified_likelihood_checks.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
