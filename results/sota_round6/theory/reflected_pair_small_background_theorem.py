"""Strong-signal open-region theorem around B=0, plus algebra diagnostics.

This is an analytic oracle-design result. No Monte Carlo, numeric epsilon,
general all-B theorem, finite-cost claim, or SOTA claim is made.
"""
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import mpmath as mp


ROOT = Path(__file__).resolve().parents[3]
mp.mp.dps = 110


def mm(value):
    value = Fraction(value)
    return mp.mpf(value.numerator) / value.denominator


def text(value):
    return mp.nstr(value, 80)


def entropy(p):
    return p * mp.log(p) + (1 - p) * mp.log(1 - p)


def logit(p):
    return mp.log(p / (1 - p))


def kl(p, q):
    return entropy(p) - p * mp.log(q) - (1 - p) * mp.log(1 - q)


def center(a, c):
    secant = (entropy(a + c) - entropy(a - c)) / (2 * c)
    return 1 / (1 + mp.exp(-secant))


def capacity(a, c):
    return kl(a + c, center(a, c))


def run_checks():
    checks = []
    for aa, cc in [('1/100', '9/1000'), ('1/20', '1/25'),
                   ('1/5', '9/50'), ('49/100', '9/20'),
                   ('1/2', '9/20'), ('7/10', '1/4')]:
        a, c = mm(aa), mm(cc)
        q = center(a, c)
        wp = (q - a + c) / (2 * c)
        gp, gm = logit(a + c) - logit(q), logit(a - c) - logit(q)
        ra = wp * gp + (1 - wp) * gm
        rc = wp * gp - (1 - wp) * gm
        qa = q * (1 - q) * (gp - gm) / (2 * c)
        qc = q * (1 - q) * (gp + gm) / (2 * c)
        assert rc > 0 and gp > 0 > gm
        assert abs(ra - mp.diff(lambda y: capacity(y, c), a)) < mp.mpf('1e-95')
        assert abs(rc - mp.diff(lambda y: capacity(a, y), c)) < mp.mpf('1e-95')
        assert abs(qa - mp.diff(lambda y: center(y, c), a)) < mp.mpf('1e-95')
        assert abs(qc - mp.diff(lambda y: center(a, y), c)) < mp.mpf('1e-95')
        z1 = ra / (c * rc)
        slope1 = qa - ra * qc / rc
        slope1_simple = -q * (1 - q) * gp * gm / (c * rc)
        assert abs(slope1 - slope1_simple) < mp.mpf('1e-95')
        assert slope1 > 0
        value2 = ra * ra / (2 * c * rc)
        b_test = mp.sqrt(min(a, 1 - a)**2 - c*c) / 1000

        def center_z(z):
            return center(a + b_test * z, c * mp.sqrt(1 - z*z))

        def radius_z(z):
            return capacity(a + b_test * z, c * mp.sqrt(1 - z*z))

        zstar = mp.findroot(lambda z: mp.diff(radius_z, z),
                           (z1 * b_test - mp.mpf('.001'),
                            z1 * b_test + mp.mpf('.001')))
        assert -1 < zstar < 1
        b_null = mp.diff(center_z, zstar)
        a_null = center_z(zstar) - b_null * zstar
        assert min(a_null, 1 - a_null) > abs(b_null)
        plus = lambda z: kl(a + b_test*z + c*mp.sqrt(1-z*z), a_null+b_null*z)
        minus = lambda z: kl(a + b_test*z - c*mp.sqrt(1-z*z), a_null+b_null*z)
        assert abs(plus(zstar) - minus(zstar)) < mp.mpf('1e-90')
        assert abs(mp.diff(plus, zstar)) < mp.mpf('1e-90')
        assert abs(mp.diff(minus, zstar)) < mp.mpf('1e-90')
        assert mp.diff(plus, zstar, 2) < 0
        assert mp.diff(minus, zstar, 2) < 0
        checks.append({
            'A_exact': aa, 'C_exact': cc,
            'C_over_physical_maximum': text(c/min(a, 1-a)),
            'base_capacity': text(capacity(a, c)),
            'base_capacity_center': text(q),
            'capacity_A_derivative': text(ra),
            'capacity_C_derivative_positive': text(rc),
            'cosine_linear_in_B_coefficient': text(z1),
            'null_slope_linear_in_B_coefficient': text(slope1),
            'value_quadratic_in_B_coefficient': text(value2),
            'diagnostic_B': text(b_test),
            'diagnostic_cosine': text(zstar),
            'diagnostic_null_intercept': text(a_null),
            'diagnostic_null_slope': text(b_null),
            'cosine_remainder_divided_by_B_cubed': text((zstar-z1*b_test)/b_test**3),
            'value_remainder_divided_by_B_fourth': text((radius_z(zstar)-capacity(a,c)-value2*b_test**2)/b_test**4),
            'two_equal_values_and_two_stationary_contacts_checked_to_90_digits': True,
            'scope': 'Pointwise algebra checks do not certify a numerical neighborhood or global optimality at the particular nonzero diagnostic B.'
        })
    return checks


def main():
    checks = run_checks()
    result = {
        'status': 'complete_analytic_small_background_argument_pending_independent_review',
        'new_simulation_observations': 0,
        'distinction_from_round5': 'This fixes arbitrary nonzero C strictly inside its physical range and perturbs B around zero. C is not assumed small. The older theorem instead fixed interior (A,B) and required sufficiently small C.',
        'definitions': {
            'model': 'p(x)=A+B cos(x)+C sin(x), B^2+C^2<=min(A,1-A)^2; take C>0 by reflection.',
            'oracle': 'V(A,B,C)=sup_w inf_{physical(a,b)} integral kl(p(x)||a+b cos(x)) w(dx), with natural logarithms.',
            'center_function': 'Q(a,c)=logistic([f(a+c)-f(a-c)]/(2c)), f(p)=p log(p)+(1-p)log(1-p).',
            'capacity_function': 'K(a,c)=kl(a+c||Q(a,c))=kl(a-c||Q(a,c)).',
            'reflection_objective': 'R_B(z)=K(A+Bz,C sqrt(1-z^2)), z in[-1,1], with the continuous value0 at endpoints.',
            'reflection_center': 'q_B(z)=Q(A+Bz,C sqrt(1-z^2)), continuously extended by q_B(+/-1)=A+/-B.'
        },
        'theorem': {
            'assumptions': 'Fix0<A<1 and0<C<min(A,1-A). There exists epsilon(A,C)>0 such that all |B|<epsilon satisfy physicality and the statements below.',
            'unique_optimum': 'R_B has a unique interior maximizer z*(B), real analytic in B near0, with z*(0)=0.',
            'exact_design': 'V(A,B,C)=R_B(z*(B)). The reflection angles +/-arccos(z*) with plus weight omega=[q_B(z*)-(A+Bz*-C sqrt(1-z*^2))]/[2C sqrt(1-z*^2)] attain it.',
            'global_null_certificate': 'b*=partial_z q_B(z*), a*=q_B(z*)-b*z*. This physical affine null satisfies kl(p(x)||a*+b*cos(x))<=R_B(z*) for every real x.',
            'one_sided_null': 'The same design/value hold when the null includes all physical sine coefficients C_null<=0; negative C follows by reflection.',
            'scope': 'The neighborhood depends on the fixed(A,C); no uniform statement as C approaches0 or min(A,1-A), and no numeric epsilon is supplied.'
        },
        'proof': [
            {'step': 1, 'claim': 'The B=0 certificate is a strict global two-contact saddle.',
             'argument': 'Let q0=Q(A,C), K0=K(A,C)>0. The two rates A+/-C lie strictly in(0,1) and q0 is strictly between them. For every z!=0, both A+/-C sqrt(1-z^2) lie strictly between A-C and A+C. Strict convexity of p->kl(p||q0) and equality of its two endpoint values therefore give F_plus(z),F_minus(z)<K0; both reach K0 atz=0. Their second derivatives there are -C[logit(A+C)-logit(q0)]<0 and C[logit(A-C)-logit(q0)]<0, respectively.'},
            {'step': 2, 'claim': 'The reflection capacity maximizer is nondegenerate.',
             'argument': 'With omega=(q0-A+C)/(2C), g_plus=logit(A+C)-logit(q0)>0 and g_minus=logit(A-C)-logit(q0)<0, the envelope derivative is K_C=omega*g_plus-(1-omega)*g_minus>0. AtB=0,z=0, partial_z R_0=0 and partial_zz R_0=-C*K_C<0. Since the constant null in Step1 has strictly smaller maximal KL away fromz=0, the capacity R_0 also has its unique global maximum atz=0.'},
            {'step': 3, 'claim': 'The capacity maximum persists uniquely for small B.',
             'argument': 'Fixed C<min(A,1-A) supplies a uniform positive probability margin for sufficiently smallB. The symmetric entropy secant and capacity are analytic in the squared half-width c^2, includingc=0, so R_B and q_B are analytic in(B,z) around the compact strip B=0,z in[-1,1]. The implicit-function theorem at the nondegenerate maximum gives analytic z*(B). Choose a compact interval J around0 where R_0 has uniformly negative second derivative. LocalC2 convergence preserves strict concavity onJ; uniformC0 convergence and the strict gap on the compact complement of interior(J) exclude maximizers outsideJ. Thus the global maximizer is unique.'},
            {'step': 4, 'claim': 'The tangent null remains physical and has two equal stationary contacts.',
             'argument': 'AtB=0 the center is q0 atz=0 and its z derivative is0. Hence(a*,b*) tends to(q0,0), strictly inside the physical affine-null region, so physicality persists. Atz* the tangent line matches q_B in value and first z derivative. Each branch F_plus/minus=kl(A+Bz+/-C sqrt(1-z^2)||a*+b*z) therefore equals R_B(z*) and has derivative partial_z R_B(z*)=0.'},
            {'step': 5, 'claim': 'The two stationary contacts globally dominate the full circle.',
             'argument': 'Choose a smaller compact J around0 where the two B=0 branch second derivatives are uniformly negative; this is possible by Step1. OnJ, the perturbed branches converge in the C2 function norm, so they remain strictly concave and attain their unique maxima atz*. On K=[-1,1] minus interior(J), Step1 and compactness give a strictly positive gap belowK0 for both branches. Their uniformC0 convergence on[-1,1] and R_B(z*)->K0 preserve the gap for smallB. Thus F_plus/minus(z)<=R_B(z*) for everyz. This is a global domination proof, not merely a local stationary-point argument.'},
            {'step': 6, 'claim': 'The design lower bound matches the global null upper bound.',
             'argument': 'At a reflected pair every C_null=0 affine null has one common rate. The chosen capacity weight makes that rate minimized at q_B(z*), with valueR_B(z*), and a constant physical null attains the minimum. Thus the design gives V>=R_B(z*). The physical tangent null from Step5 gives V<=R_B(z*) for every design, regardless of support size; it also rules out a better boundary null design indirectly through this global upper bound. No minimax exchange is assumed.'},
            {'step': 7, 'claim': 'Opposite-sign null extension.',
             'argument': 'At the reflected pair, an opposite-sign null has q_plus<=q_minus while the true endpoints are strictly ordered the other way. Weighted Bernoulli-KL projection onto this order cone pools at the weighted mean q_B(z*), which the constant physical null attains. The same C_null=0 tangent line gives the upper certificate, completing the sandwich.'}
        ],
        'closed_local_expansions': {
            'notation': 'All Q,K and their A,C derivatives below are evaluated at the fixed(A,C), and all O constants depend on(A,C).',
            'symmetry': 'R_{-B}(-z)=R_B(z), so uniqueness implies z*(-B)=-z*(B), V(-B)=V(B), a*(-B)=a*(B), and b*(-B)=-b*(B).',
            'optimal_cosine': 'z*(B)=B K_A/(C K_C)+O(B^3).',
            'value': 'V(A,B,C)=K(A,C)+B^2 K_A^2/(2 C K_C)+O(B^4).',
            'null_intercept': 'a*(B)=Q(A,C)+O(B^2).',
            'null_slope': 'b*(B)=B[Q_A-(K_A/K_C)Q_C]+O(B^3). The coefficient equals -Q(1-Q)g_plus*g_minus/(C K_C)>0.',
            'derivative_formulas': 'K_A=omega*g_plus+(1-omega)*g_minus; K_C=omega*g_plus-(1-omega)*g_minus>0; Q_A=Q(1-Q)(g_plus-g_minus)/(2C); Q_C=Q(1-Q)(g_plus+g_minus)/(2C).',
            'interpretation': 'The quadratic improvement relative to the B-independent PSR capacity K(A,C) is nonnegative. This is oracle information only; it is not a finite stopping-cost gain.'
        },
        'diagnostics': {
            'mpmath_digits': mp.mp.dps,
            'cases': checks,
            'scope': 'Deterministic high-precision algebra and contact checks supplement, but do not replace, the analytic compactness and implicit-function proof.'
        },
        'unresolved': [
            'Arbitrary physical B at a fixed nonzero C.',
            'Uniform or explicit admissible epsilon(A,C).',
            'Boundary models with |C|=min(A,1-A); this boundary forces B=0 and has an elementary exact solution, but no open physical B-neighborhood.',
            'Finite-risk finite-cost benefit, paid parameter learning, and independent novelty review.'
        ],
        'source_sha256': {
            Path(__file__).resolve().relative_to(ROOT).as_posix(): hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'results/sota_round5/theory/reflected_pair_weak_signal_theorem.json': hashlib.sha256((ROOT/'results/sota_round5/theory/reflected_pair_weak_signal_theorem.json').read_bytes()).hexdigest()
        },
        'claim_boundary': 'This proves a different local parameter neighborhood from the earlier weak-C theorem. The general all-C/all-B reflected-pair conjecture remains unresolved. No frozen Round5 source is changed.'
    }
    out = Path(__file__).with_suffix('.json')
    with out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'artifact': str(out), 'sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
                      'status': result['status'], 'checks': len(checks)}, indent=2))


if __name__ == '__main__':
    main()
