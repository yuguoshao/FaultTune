"""Closed-form c-optimal design for a scalar Bernoulli sinusoid.

Assumptions: p(x)=A+B*cos(x)+C*sin(x), R^2=B^2+C^2<min(A^2,(1-A)^2),
0<A<1. A,B,C are all unknown; linear unbiasedness is required throughout
their open parameter model. Each independent angle query has equal unit
cost and returns one Bernoulli observation. The estimand is 2*C.

This is a local variance / approximate-design statement, NOT a lower bound
on arbitrary finite-budget hypothesis tests, stopping times, or hardware
cost. Extra output statistics, known coefficient constraints, unequal query
costs, or coherent queries are outside this statement.

Write D=A*(1-A), L=1-2*A, R2=B*B+C*C, and
  S=sqrt((A*A-R2)*((1-A)*(1-A)-R2)),
  u=(D+B*B-C*C+S)/2, w=u-B*B, z=-L*B/(2*w).
Then the exact optimum normalized variance is V*=4*u.

PROOF. The local Elfving dual asks for d such that
  (d0+d1*cos(x)+d2*sin(x))^2 <= p(x)*(1-p(x))  for every x.
Choose d=(L*C/(2*sqrt(u)), -B*C/sqrt(u), sqrt(u)). A direct identity is

 p(x)*(1-p(x)) - (d0+d1*cos(x)+d2*sin(x))^2
       = ((u+C*C)*w/u)*(cos(x)-z)^2 >= 0.

For the constant coefficient this identity reduces to
  4*(u-B*B)*(D-C*C-u)-L*L*R2=0,
which is the quadratic equation defining u. The sin and cos*sin terms
vanish directly; the cos and cos^2 terms match the displayed square.

Let H=D-R2. Strict feasibility gives H>|L|*sqrt(R2), and therefore
  w=(H+S)/2 > |L|*sqrt(R2)/2 >= |L*B|/2.
So |z|<1. At x_plus=acos(z), x_minus=-acos(z), put y=sqrt(1-z*z).
The dual contacts have opposite signs, because
  d0+d1*z = sqrt(u)*L*C/(2*w)
and L^2*(B^2+C^2)/(4*w^2)<1. Thus the two standard deviations sigma_plus,
sigma_minus satisfy sigma_plus+sigma_minus=2*sqrt(u)*y.

Assign angle proportions sigma_plus/(sigma_plus+sigma_minus) and its
complement. The estimator (mean_plus-mean_minus)/y is unbiased for 2*C.
Its normalized variance is (sigma_plus+sigma_minus)^2/y^2=4*u.
Conversely, for any linear unbiased design with coefficients a(x),
  (c.d)^2 = (integral a(x)*(phi(x).d) dw)^2
          <= [integral a(x)^2*v(x) dw]*[integral (phi(x).d)^2/v(x) dw]
          <= V,
where c=(0,0,2), phi=(1,cos,sin). The dual therefore proves V>=4*u.

Uniform sampling at +/-pi/2 has V_pi=4*(D-C*C). Consequently
  V_pi/V*=(D-C*C)/u <= 2,
strictly less than 2 in the stated open model. The supremum 2 is approached
with B=0, A fixed away from 1/2, and |C| approaching min(A,1-A). At A=1/2,
uniform +/-pi/2 is already optimal. Rational identities and high precision
numerical contacts below are independent checks of this analytic proof.
"""

import json
from decimal import Decimal, localcontext
from fractions import Fraction


def dec(value):
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    return Decimal(value)


def design(a, b, c):
    a, b, c = dec(a), dec(b), dec(c)
    one = Decimal(1)
    r2, d, ell = b * b + c * c, a * (one - a), one - 2 * a
    if not (0 < a < 1 and r2 < min(a * a, (one - a) ** 2)):
        raise ValueError("The full three-parameter model must be strictly interior")
    determinant = ((a * a - r2) * ((one - a) ** 2 - r2)).sqrt()
    u = (d + b * b - c * c + determinant) / 2
    w, root_u = u - b * b, u.sqrt()
    z = -ell * b / (2 * w)
    y = (one - z * z).sqrt()
    dual = (ell * c / (2 * root_u), -b * c / root_u, root_u)
    p_plus, p_minus = a + b * z + c * y, a + b * z - c * y
    sigma_plus = (p_plus * (one - p_plus)).sqrt()
    sigma_minus = (p_minus * (one - p_minus)).sqrt()
    v_primal = (sigma_plus + sigma_minus) ** 2 / (y * y)
    v_dual, v_uniform = 4 * u, 4 * (d - c * c)
    return {"A": a, "B": b, "C": c, "u": u, "w": w, "z": z,
            "dual": dual, "square_coefficient": (u + c * c) * w / u,
            "plus_proportion": sigma_plus / (sigma_plus + sigma_minus),
            "V_primal": v_primal, "V_dual": v_dual,
            "V_uniform": v_uniform, "uniform_over_optimum": v_uniform / v_dual,
            "quadratic_residual": 4 * w * (d - c * c - u) - ell * ell * r2}


def main():
    with localcontext() as context:
        context.prec = 80
        directions = [(Fraction(0), Fraction(1)),
                      (Fraction(3, 5), Fraction(4, 5)),
                      (Fraction(1), Fraction(0)),
                      (Fraction(-4, 5), Fraction(3, 5))]
        circle_points = directions + [(x, -y) for x, y in directions]
        count, max_gap, max_square_residual, max_quadratic = 0, Decimal(0), Decimal(0), Decimal(0)
        for a in map(Fraction, (".05", ".1", ".25", ".5", ".75", ".9", ".95")):
            for fraction in map(Fraction, (".01", ".3", ".8", ".99", ".999999")):
                radius = min(a, 1 - a) * fraction
                for bx, cy in directions:
                    b, c = radius * bx, radius * cy
                    r2, d, ell = b * b + c * c, a * (1 - a), 1 - 2 * a
                    # Exactly validate the discriminant identity over rationals.
                    assert (d - r2) ** 2 - ell ** 2 * r2 == (a * a - r2) * ((1 - a) ** 2 - r2)
                    result = design(a, b, c)
                    assert abs(result["z"]) < 1 and 0 < result["plus_proportion"] < 1
                    assert result["uniform_over_optimum"] <= 2
                    max_gap = max(max_gap, abs(result["V_primal"] - result["V_dual"]))
                    max_quadratic = max(max_quadratic, abs(result["quadratic_residual"]))
                    for cx, sx in circle_points:
                        x, y = dec(cx), dec(sx)
                        probability = dec(a) + dec(b) * x + dec(c) * y
                        dual_value = result["dual"][0] + result["dual"][1] * x + result["dual"][2] * y
                        lhs = probability * (1 - probability) - dual_value ** 2
                        rhs = result["square_coefficient"] * (x - result["z"]) ** 2
                        max_square_residual = max(max_square_residual, abs(lhs - rhs))
                    count += 1
        assert max(max_gap, max_quadratic, max_square_residual) < Decimal("1e-65")
        tight_cases = []
        for exponent in (1, 3, 6, 9, 12):
            a, b = Fraction(1, 4), Fraction(0)
            c = a * (1 - Fraction(1, 10 ** exponent))
            result = design(a, b, c)
            tight_cases.append({"A": str(a), "B": str(b), "C": str(c),
                                "uniform_over_optimum": str(result["uniform_over_optimum"])})
        print(json.dumps({"schema": "faulttune.single_frequency_design_bound.v1",
                          "status": "algebra_checks_passed", "precision": context.prec,
                          "rational_parameter_cases": count,
                          "max_primal_dual_gap": str(max_gap),
                          "max_square_identity_residual": str(max_square_residual),
                          "max_quadratic_residual": str(max_quadratic),
                          "sharp_ratio_supremum": 2,
                          "tightness_sequence": tight_cases,
                          "scope": "Local scalar-Bernoulli equal-cost linear-unbiased variance; not finite-test cost"}, indent=2))


if __name__ == "__main__":
    main()
