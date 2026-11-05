"""Open-neighborhood extension of the reviewed A=1/2 operator certificate.

Only a deterministic analytic theorem and algebra/contact diagnostics are made.
The general all-A/B/C reflected-pair conjecture remains unresolved.
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
    return mp.mpf(value.numerator)/value.denominator


def show(value):
    return mp.nstr(value, 80)


def f(p):
    return p*mp.log(p)+(1-p)*mp.log(1-p)


def kl(p, q):
    return f(p)-p*mp.log(q)-(1-p)*mp.log(1-q)


def center(a, c):
    secant = (f(a+c)-f(a-c))/(2*c)
    return 1/(1+mp.exp(-secant))


def capacity(a, c):
    return kl(a+c, center(a, c))


def checks():
    rows = []
    for bb, cc in [('49/100', '1/20'), ('1/5', '2/5'),
                   ('3/10', '3/10'), ('1/10', '12/25'),
                   ('9/20', '1/10'), ('0', '49/100')]:
        b, c = mm(bb), mm(cc)
        t = 2*c
        lam = mp.atanh(t)/t
        d = 1/(1-t*t)-2*lam+lam*lam
        h = t*t*lam-4*b*b*d
        assert b*b+c*c < mp.mpf('.25')
        assert lam > 1 and (1-t*t)*lam < 1
        assert d > 0 and h > 0
        factored_bound = (lam-1)*((1-t*t)*lam-1)
        assert -h < factored_bound < 0
        baseline_plus = lambda z: kl(mp.mpf('.5')+b*z+c*mp.sqrt(1-z*z),
                                     mp.mpf('.5')+b*lam*z)
        baseline_minus = lambda z: kl(mp.mpf('.5')+b*z-c*mp.sqrt(1-z*z),
                                      mp.mpf('.5')+b*lam*z)
        baseline_capacity = lambda z: capacity(mp.mpf('.5')+b*z,
                                                c*mp.sqrt(1-z*z))
        assert abs(mp.diff(baseline_plus, 0, 2)+h) < mp.mpf('1e-94')
        assert abs(mp.diff(baseline_minus, 0, 2)+h) < mp.mpf('1e-94')
        assert abs(mp.diff(baseline_capacity, 0, 2)+h) < mp.mpf('1e-94')
        assert abs(mp.diff(lambda a: capacity(a, c), mp.mpf('.5'), 2)-4*d) < mp.mpf('1e-94')
        z1 = 4*b*d/h
        value2 = 2*d*t*t*lam/h
        alpha = (mp.mpf('.5')-mp.sqrt(b*b+c*c))/1000
        a = mp.mpf('.5')+alpha

        def qz(z):
            return center(a+b*z, c*mp.sqrt(1-z*z))

        def rz(z):
            return capacity(a+b*z, c*mp.sqrt(1-z*z))

        zstar = mp.findroot(lambda z: mp.diff(rz, z),
                           (z1*alpha-mp.mpf('.001'), z1*alpha+mp.mpf('.001')))
        bn = mp.diff(qz, zstar)
        an = qz(zstar)-bn*zstar
        assert min(an, 1-an) > abs(bn)
        plus = lambda z: kl(a+b*z+c*mp.sqrt(1-z*z), an+bn*z)
        minus = lambda z: kl(a+b*z-c*mp.sqrt(1-z*z), an+bn*z)
        assert abs(plus(zstar)-minus(zstar)) < mp.mpf('1e-90')
        assert abs(mp.diff(plus, zstar)) < mp.mpf('1e-90')
        assert abs(mp.diff(minus, zstar)) < mp.mpf('1e-90')
        assert mp.diff(plus, zstar, 2) < 0
        assert mp.diff(minus, zstar, 2) < 0
        rows.append({
            'B_exact': bb, 'C_exact': cc,
            'lambda': show(lam), 'D_positive': show(d), 'H_positive': show(h),
            'both_baseline_KL_second_derivatives': show(-h),
            'strict_negative_factorized_upper_bound': show(factored_bound),
            'optimal_cosine_linear_alpha_coefficient': show(z1),
            'value_quadratic_alpha_coefficient': show(value2),
            'diagnostic_alpha': show(alpha), 'diagnostic_cosine': show(zstar),
            'diagnostic_null_intercept': show(an), 'diagnostic_null_slope': show(bn),
            'cosine_remainder_divided_by_alpha_cubed': show((zstar-z1*alpha)/alpha**3),
            'value_remainder_divided_by_alpha_fourth': show((rz(zstar)-capacity(mp.mpf('.5'), c)-value2*alpha**2)/alpha**4),
            'null_intercept_remainder_divided_by_alpha_cubed': show((an-mp.mpf('.5')-lam*alpha)/alpha**3),
            'two_equal_KLs_and_two_zero_derivatives_checked_to_90_digits': True,
            'scope': 'These algebra/contact checks do not supply a numerical epsilon or certify this nonzero diagnostic alpha by themselves.'
        })
    return rows


def main():
    rows = checks()
    result = {
        'status': 'complete_analytic_near_symmetric_argument_pending_independent_review',
        'new_simulation_observations': 0,
        'previous_theorem_used': 'The reviewed A=1/2 all-angle operator certificate in docs/CONTINUOUS_SYMMETRIC_DESIGN.md, with its exact source hash recorded below.',
        'theorem': {
            'assumptions': 'Fix realB and0<C with B^2+C^2<1/4. There exists epsilon(B,C)>0 such that for A=1/2+alpha and |alpha|<epsilon, the physical model p(x)=A+B cos(x)+C sin(x) has the exact reflected-pair optimal design described below. C<0 follows by reflection.',
            'design': 'Let Q(a,c)=logistic([f(a+c)-f(a-c)]/(2c)), K(a,c)=kl(a+c||Q(a,c)), and R_alpha(z)=K(1/2+alpha+Bz,C sqrt(1-z^2)). Its unique interior maximizer z*(alpha) gives the two reflected angles, with plus weight omega=[Q(p0,c0)-(p0-c0)]/(2c0), p0=A+Bz*, c0=C sqrt(1-z*^2).',
            'global_certificate': 'The tangent to q_alpha(z)=Q(A+Bz,C sqrt(1-z^2)) atz* is a physical affine null q_t(z)=a*+b*z and satisfies kl(p(x)||q_t(cos(x)))<=R_alpha(z*) for EVERY realx.',
            'oracle_value': 'V(A,B,C)=R_alpha(z*), even when arbitrary design supports and physical boundary nulls are allowed. The same equality holds for the larger opposite-sign physical null C_null<=0.',
            'scope': 'This is a neighborhood ofA=1/2 for each fixed strictly physical(B,C), not a uniform neighborhood over all(B,C) or the full global conjecture. No explicit epsilon is supplied.'
        },
        'baseline_strictness_proof': [
            {'step': 1, 'claim': 'The old operator certificate is strict away from the two PSR angles in a full-rank model.',
             'argument': 'Write t=2C, lambda=atanh(t)/t, rho=(I+2B Z+t X)/2, sigma=(I+2B lambda Z)/2, H*=I+tX. Strict B^2+C^2<1/4 makes rho positive definite, and the earlier proof makes sigma positive definite. For Hermitian E!=0, the Hessian of G(H)=Tr(rho log H) is -2 integral_0^infty Tr[rho R E R E R] ds<0, R=(H+sI)^(-1): RERER is positive semidefinite and nonzero, and rho is positive definite. Thus G is strictly concave and its tangent inequality atH* is strict unlessH=H*. For each angle, the likelihood-ratio matrix H_x=(p/q)P_x+((1-p)/(1-q))(I-P_x) is positive definite and Tr(sigma H_x)=1. The earlier proof therefore gives KL(p(x)||q*(x))<Phi(t) unlessH_x=H*. Since C!=0, H* has two distinct eigenvalues, so this equality forces P_x to be an X eigenprojection, precisely x=+/-pi/2 modulo2pi. Both PSR angles attain equality.'},
            {'step': 2, 'claim': 'Both baseline contacts are strictly nondegenerate.',
             'argument': 'For z=cos(x) near0, p_plus/minus=1/2+Bz+/-C sqrt(1-z^2), q*=1/2+B lambda z. Direct differentiation gives d2F_plus/dz2(0)=d2F_minus/dz2(0)=4B^2 D-t^2 lambda, D=1/(1-t^2)-2lambda+lambda^2=(lambda-1)^2+t^2/(1-t^2)>0. Since4B^2<1-t^2, this is strictly less than (1-t^2)D-t^2 lambda=(lambda-1)((1-t^2)lambda-1)<0. Here lambda>1 and (1-t^2)lambda<1 follow by integrating the elementary inequalities for atanh(t),0<t<1.'},
            {'step': 3, 'claim': 'The baseline capacity also has a unique nondegenerate global maximum.',
             'argument': 'The constant-at-a-pair interpretation of the affine null gives R_0(z)<=max(F_plus(z),F_minus(z)). Step1 makes this strictly less thanPhi(t) for everyz!=0, while R_0(0)=Phi(t). Moreover q_0(-z)=1-q_0(z), so d2q_0/dz2(0)=0, and q_0(0)=1/2, dq_0/dz(0)=B lambda. At0 it agrees with the old affine null through second derivative. Therefore d2R_0/dz2(0)=d2F_plus/dz2(0)=d2F_minus/dz2(0)=-H<0, H=t^2 lambda-4B^2D.'}
        ],
        'perturbation_proof': [
            {'step': 4, 'claim': 'The capacity maximizer and tangent null persist for small alpha.',
             'argument': 'Choose |alpha|<1/2-sqrt(B^2+C^2) to retain a strict physical margin. The entropy secant and capacity are analytic in the squared half-width, including its zero value atz=+/-1, so R_alpha and q_alpha are analytic around the compact strip alpha=0,z in[-1,1]. The nondegenerate capacity maximum gives analytic z*(alpha) by the implicit-function theorem. LocalC2 convergence keeps strict concavity in a compact neighborhoodJ of0, and uniformC0 convergence preserves the baseline strict gap on the compact complement of interior(J). Thus the global maximizer is unique and remains interior. Its tangent null coefficients converge to(1/2,B lambda), which has a strict physical margin by the old certificate, so the new tangent null remains physical.'},
            {'step': 5, 'claim': 'Global KL domination persists, not just stationarity.',
             'argument': 'At the newz*, q_t and q_alpha match in value and first derivative. Both KL branches have exactly equal valuesR_alpha(z*) and exactly zero first derivatives there. ChooseJ sufficiently small that both baseline branches have negative second derivatives throughoutJ, using Step2. LocalC2 convergence makes each perturbed branch strictly concave onJ and maximized atz*. On the compact complement of interior(J), Step1 supplies a strictly positive baseline gap; uniformC0 convergence of both branches preserves that gap relative toR_alpha(z*) for smallalpha. This proves the all-angle upper certificate.'},
            {'step': 6, 'claim': 'The global upper certificate matches the reflected design lower bound.',
             'argument': 'At the chosen pair the capacity weight makes the meanq_alpha(z*). Every sine-zero null has one common rate at the pair, so its weighted KL is minimized there with valueR_alpha(z*), attained by a physical constant null. The single physical affine tangent from Step5 gives the same upper bound for every design distribution. Thus arbitrary three-point or larger-support designs cannot do better in this neighborhood. If C_null<=0 is allowed, the two null rates have reverse order, whose weighted KL projection pools at the same mean; the upper certificate still has C_null=0.'}
        ],
        'closed_expansions': {
            'notation': 't=2C, lambda=atanh(t)/t, D=1/(1-t^2)-2lambda+lambda^2, H=t^2 lambda-4B^2D>0; all O constants depend on fixed(B,C).',
            'symmetry': 'Probability complementation with z reflection gives z*(-alpha)=-z*(alpha), V(-alpha)=V(alpha), a*(-alpha)=1-a*(alpha), b*(-alpha)=b*(alpha).',
            'capacity_derivatives_at_half': 'K_A(1/2,C)=0, K_AA(1/2,C)=4D, C*K_C(1/2,C)=t^2 lambda.',
            'optimal_cosine': 'z*(alpha)=4B D alpha/H+O(alpha^3).',
            'value': 'V(1/2+alpha,B,C)=Phi(t)+2D t^2 lambda alpha^2/H+O(alpha^4).',
            'tangent_null': 'a*(alpha)=1/2+lambda alpha+O(alpha^3), b*(alpha)=B lambda+O(alpha^2).',
            'derivation': 'R_zalpha at(alpha,z)=(0,0) equals B*K_AA=4BD, while R_zz=-H, so z* prime=4BD/H. Differentiating the optimized value gives V second=K_AA-(B*K_AA)^2/R_zz=4D*t^2 lambda/H. Center_Q_A at(1/2,C) islambda and center_Q_C is0; the linear z* contributions cancel from the tangent intercept.'
        },
        'diagnostics': {'mpmath_digits': mp.mp.dps, 'cases': rows,
                        'scope': 'High-precision algebra and stationary-contact checks. The analytic proof, not these finitely many checks, supplies the existence statement; no numeric epsilon is certified.'},
        'source_sha256': {
            Path(__file__).resolve().relative_to(ROOT).as_posix(): hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'docs/CONTINUOUS_SYMMETRIC_DESIGN.md': hashlib.sha256((ROOT/'docs/CONTINUOUS_SYMMETRIC_DESIGN.md').read_bytes()).hexdigest(),
            'results/sota_round5/theory/reflected_pair_weak_signal_peer_review.json': hashlib.sha256((ROOT/'results/sota_round5/theory/reflected_pair_weak_signal_peer_review.json').read_bytes()).hexdigest()
        },
        'claim_boundary': 'Full-rank baseline is used for strictness. The unrestricted reflected-pair conjecture, finite-cost superiority, numeric neighborhood size, and novelty remain unresolved. This is an analytic open-region extension of an existing reviewed certificate, with no frozen-source edits.'
    }
    out = Path(__file__).with_suffix('.json')
    with out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'artifact': str(out), 'sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
                      'status': result['status'], 'checks': len(rows)}, indent=2))


if __name__ == '__main__':
    main()
