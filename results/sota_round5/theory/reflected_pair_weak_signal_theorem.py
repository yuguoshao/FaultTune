"""Analytic weak-signal theorem and high-precision checks of its formulas.

The unrestricted all-C reflected-pair conjecture remains open. No sampling or
claim of exact finite-cost/SOTA superiority is made here.
"""
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import mpmath as mp

ROOT=Path(__file__).resolve().parents[3]
mp.mp.dps=120


def mm(x):
    x=Fraction(x)
    return mp.mpf(x.numerator)/x.denominator


def f(p):
    return p*mp.log(p)+(1-p)*mp.log(1-p)


def kl(p,q):
    return p*mp.log(p/q)+(1-p)*mp.log((1-p)/(1-q))


def show(x):
    return mp.nstr(x,90)


def main():
    checks=[]
    for aa,bb in [(Fraction(1,100),Fraction(0)),(Fraction(1,100),Fraction(9,1000)),
                  (Fraction(1,5),Fraction(1,6)),(Fraction(49,100),Fraction(1,10)),
                  (Fraction(1,2),Fraction(49,100)),(Fraction(7,10),Fraction(1,5))]:
        A,B=mm(aa),mm(bb)
        m=min(A,1-A)
        C=(m-abs(B))/100
        v=A*(1-A)
        d=v-B*B
        e=B*(1-2*A)
        root=mp.sqrt(d*d-e*e)
        z0=-e/(d+root)
        h=lambda z:(1-z*z)/(2*(A+B*z)*(1-A-B*z))
        hmax=1/(v+B*B+mp.sqrt((A*A-B*B)*((1-A)**2-B*B)))
        assert abs(h(z0)-hmax)<mp.mpf('1e-100')
        assert abs(mp.diff(h,z0))<mp.mpf('1e-100')
        assert mp.diff(h,z0,2)<0
        def ps(z):
            delta=C*mp.sqrt(1-z*z)
            return A+B*z-delta,A+B*z+delta
        def center(z):
            pm,pp=ps(z)
            sec=(f(pp)-f(pm))/(pp-pm)
            return 1/(1+mp.exp(-sec))
        def radius(z):
            pm,pp=ps(z)
            return kl(pp,center(z))
        zstar=mp.findroot(lambda z:mp.diff(radius,z),(z0-mp.mpf('.01'),z0+mp.mpf('.01')))
        assert -1<zstar<1
        qc=center(zstar)
        b=mp.diff(center,zstar)
        a=qc-b*zstar
        assert min(a,1-a)>abs(b)
        plus=lambda z:kl(ps(z)[1],a+b*z)
        minus=lambda z:kl(ps(z)[0],a+b*z)
        assert abs(plus(zstar)-minus(zstar))<mp.mpf('1e-95')
        assert abs(mp.diff(plus,zstar))<mp.mpf('1e-95')
        assert abs(mp.diff(minus,zstar))<mp.mpf('1e-95')
        assert mp.diff(plus,zstar,2)<0 and mp.diff(minus,zstar,2)<0
        checks.append({'A_exact':str(aa),'B_exact':str(bb),'C_diagnostic':show(C),
                       'z0_closed_form':show(z0),'hmax_closed_form':show(hmax),
                       'capacity_stationary_cosine':show(zstar),
                       'z_shift_divided_by_C_squared':show((zstar-z0)/(C*C)),
                       'null_intercept':show(a),'null_slope':show(b),
                       'null_physical_margin':show(min(a,1-a)-abs(b)),
                       'radius':show(radius(zstar)),
                       'radius_over_C_squared_minus_hmax':show(radius(zstar)/(C*C)-hmax),
                       'two_equal_values_and_two_zero_derivatives_checked_to_95_digits':True,
                       'diagnostic_only':'These point checks do not supply the existential epsilon or prove a global optimum for the selected nonzero C. The mathematical theorem below supplies an asymptotic neighborhood, without a numeric epsilon.'})
    result={
        'status':'complete_analytic_weak_signal_argument_pending_independent_review',
        'new_simulation_observations':0,
        'global_all_C_conjecture_status':'Unresolved; no counterexample found in the retained finite diagnostics. The theorem below is local in C for each fixed interior(A,B).',
        'definitions':{
            'model':'p(x)=A+B cos x+C sin x, B^2+C^2<=min(A,1-A)^2.',
            'oracle':'V(A,B,C)=sup_w inf_{physical(a,b)} integral kl(p(x)||a+b cos x) w(dx), with natural logarithms.',
            'reflection':'Write z=cos x in[-1,1], s=sqrt(1-z^2), and p_minus=A+Bz-C s,p_plus=A+Bz+C s, taking C>0 without loss of generality.',
            'entropy_secant':'Let f(p)=p log p+(1-p)log(1-p), l(z,C)=[f(p_plus)-f(p_minus)]/(p_plus-p_minus), q_c(z,C)=logistic(l(z,C)).',
            'capacity':'R(z,C)=kl(p_minus||q_c)=kl(p_plus||q_c). Its achieving plus-arm weight is omega=(q_c-p_minus)/(p_plus-p_minus), strictly between0 and1 for distinct interior endpoints.',
            'zero_extension':'At C=0 or z=+/-1 define q_c=A+Bz and R=0 by continuity. For fixed interior(A,B), q_c and R are analytic in z and C^2 on a neighborhood of[-1,1] times{0}.'},
        'theorem':{
            'assumptions':'Fix0<A<1 and |B|<min(A,1-A). There exists epsilon(A,B)>0 such that for every0<|C|<epsilon(A,B), the model is physical and the following statements hold.',
            'one_dimensional_exact_value':'V(A,B,C)=max_{-1<=z<=1} R(z,C). The maximizer z*(C) is unique and interior in this weak-signal neighborhood.',
            'optimal_design':'Use x*=arccos z*, its reflection -x*, and weights omega(z*,C),1-omega(z*,C). For negative C swap the arm labels/weights as needed.',
            'explicit_tangent_null':'Let b*=partial_z q_c(z*,C), a*=q_c(z*,C)-b*z*. Then q*(x)=a*+b* cos x is physically feasible and kl(p(x)||q*(x))<=R(z*,C) for EVERY real x.',
            'one_sided_extension':'The same value/design/certificate hold if the null is enlarged to the opposite physical sign C_null<=0 for C>0, or C_null>=0 for C<0.',
            'claim_boundary':'This is an oracle information-design theorem. It does not give a computable uniform epsilon, paid-learning policy, finite95%/1% cost gain, or novelty/SOTA conclusion.'},
        'proof_steps':[
            {'step':1,'claim':'Every reflection pair yields a capacity lower bound.',
             'argument':'The logit secant lies strictly between logit(p_minus) and logit(p_plus), so p_minus<q_c<p_plus. Its definition makes the two endpoint KLs equal. With omega=(q_c-p_minus)/(p_plus-p_minus), the weighted Bernoulli KL is minimized at the mixture mean q_c. A constant null(a,b)=(q_c,0) is physical, so this two-arm design has exact null-information value R. Consequently V>=max_z R.'},
            {'step':2,'claim':'The limiting one-dimensional objective has a unique nondegenerate interior maximum.',
             'argument':'Let p0=A+Bz, v=A(1-A), d=v-B^2, e=B(1-2A). Because(A+/-B)(1-A-/+B)>0, d>|e|. The limiting objective is h(z)=(1-z^2)/(2 p0(1-p0)). Its reciprocal apart from2 is w(z)=p0(1-p0)/(1-z^2)=B^2+(d+ez)/(1-z^2). Its derivative has numerator e(1+z^2)+2dz, with exactly one root inside(-1,1): z0=-e/(d+sqrt(d^2-e^2)); this formula also gives0 when e=0. At the root d2w/dz2=2 sqrt(d^2-e^2)/(1-z0^2)^2>0. Thus h has a unique strict nondegenerate global maximum z0; h(+/-1)=0.'},
            {'step':3,'claim':'The capacity maximizer persists analytically for small C.',
             'argument':'The strict interior margin of p0 on[-1,1] makes f analytic uniformly around every p0. The entropy secant has an even power series in C sqrt(1-z^2); hence q_c=p0+C^2 g(z)+O(C^4) and R=C^2 h(z)+O(C^4), including derivatives in z on[-1,1]. Here g(z)=(1-z^2)(2p0-1)/(6p0(1-p0)). Therefore R/C^2 extends analytically in(z,C^2). The implicit-function theorem at the nondegenerate maximum gives z*(C)=z0+O(C^2), analytic in C^2. Uniform convergence and the nonzero derivative of h off any neighborhood of z0 make this the unique global maximum for all sufficiently small C.'},
            {'step':4,'claim':'The tangent null is physical and matches two stationary equal-height contacts.',
             'argument':'Since q_c=p0+O(C^2) with its first z derivative, the tangent coefficients satisfy(a*,b*)=(A,B)+O(C^2). The fixed strict null margin |B|<min(A,1-A) guarantees physical feasibility for small C. At z*, the line q_t(z)=a*+b*z has both the same value and same first derivative as q_c. Thus F_plus(z)=kl(p_plus(z)||q_t(z)) and F_minus(z)=kl(p_minus(z)||q_t(z)) satisfy F_plus(z*)=F_minus(z*)=R(z*,C) and partial_z F_plus(z*)=partial_z F_minus(z*)=partial_z R(z*,C)=0.'},
            {'step':5,'claim':'These two contacts are GLOBAL KL maxima, not just tangencies.',
             'argument':'Choose a closed neighborhood J of z0 strictly inside(-1,1), small enough that d2h/dz2<=-lambda<0 on J. On J, F_plus/C^2 and F_minus/C^2 converge to h in the C^2 norm as C->0; this means convergence of functions and two z derivatives, NOT an O(C^2) convergence rate. Hence each F branch is strictly concave on J for sufficiently small C, and its stationary point z* is its unique maximum there. Globally on[-1,1], both normalized branches converge uniformly to h: Taylor expansion of KL uses p_plus/minus-q_t=+/-C sqrt(1-z^2)+O(C^2), uniformly including endpoints. The strict global gap h(z0)-max_{z outside J}h(z)>0 therefore implies F_plus(z),F_minus(z)<R(z*,C) outside J once C is small enough. This proves the uniform all-angle bound.'},
            {'step':6,'claim':'The upper certificate matches the design lower bound.',
             'argument':'For every distribution w over angles, the same physical null q* gives inf_null integral KL<=integral KL(p||q*)<=R(z*,C). Combined with Step1 this proves equality and the optimal design.'},
            {'step':7,'claim':'The opposite-sign extension has the same value.',
             'argument':'At the reflected angles, any opposite-sign null has ordered rates q_plus<=q_minus. Weighted KL to the true ordered p_plus>p_minus is minimized over this order cone by pooling at their weighted mean q_c. The constant physical null attains it. The upper bound still uses the same C_null=0 tangent null, so the sandwich remains exact.'}
        ],
        'closed_asymptotics':{
            'discriminant_identity':'d^2-e^2=(A^2-B^2)((1-A)^2-B^2)>0.',
            'value':'V(A,B,C)=C^2/[A(1-A)+B^2+sqrt((A^2-B^2)((1-A)^2-B^2))]+O_{A,B}(C^4).',
            'angle':'cos x*(C)=z0+O_{A,B}(C^2).',
            'weight':'omega=1/2+C sqrt(1-z0^2)(2(A+Bz0)-1)/[12(A+Bz0)(1-A-Bz0)]+O_{A,B}(C^3), for the plus endpoint when C>0.',
            'null':'b*=B+C^2 partial_z g(z0)+O(C^4), a*=A+C^2[g(z0)-z0 partial_z g(z0)]+O(C^4).',
            'nonuniformity':'All O constants and the existence neighborhood depend on fixed(A,B). The proof is not uniform when |B| approaches min(A,1-A) while C also approaches0; the limiting maximizer may approach z=+/-1.'},
        'other_exact_cases':{
            'B_zero':'For B=0 and any physical C, p(x) lies in[A-|C|,A+|C|]. Choose the Bernoulli capacity center of these endpoints as a constant null. KL is convex in p, so every angle is dominated by the two endpoint KLs. The reflected PSR pair+/-pi/2 with capacity weights is globally optimal, including boundary endpoints by continuity.',
            'A_half':'The previously proved operator-concavity theorem gives global exact optimality of equally weighted PSR for A=1/2 and every physical B,C.',
            'C_zero':'The model is a null; the oracle value is0.'},
        'global_conjecture_reduction':{
            'two_support_kkt':'If a positive-information two-support design has an INTERIOR minimizing affine null, KKT gives sum_i g_i=0 and sum_i z_i g_i=0, g_i=w_i(q_i-p_i)/[q_i(1-q_i)]. Distinct z_i force both g_i=0 and zero information. Thus such a two-support solution must have the same cosine at both angles, hence be a reflection pair. This does not exclude three or more supports or a boundary null.',
            'sufficient_curvature_lemma':'Let R*=max_z R(z,C). Define ell_R(p) and u_R(p) as the lower and upper endpoints of{q:kl(p||q)<=R}. If L(z)=ell_R*(p_plus(z)) is concave and U(z)=u_R*(p_minus(z)) is convex globally, then L<=U and at a capacity maximum they meet. Their common tangent line lies above L and below U, giving the full conjecture immediately.',
            'exact_lower_inverse_derivatives':'For p>q=ell_R(p), put K=q(1-q)[logit(p)-logit(q)]/(p-q). Then d ell_R/dp=K and d2 ell_R/dp2={1/[p(1-p)]-2K/[q(1-q)]+[p/q^2+(1-p)/(1-q)^2]K^2} q(1-q)/(p-q). Consequently d2L/dz2=(d2 ell_R/dp2)(B-Cz/sqrt(1-z^2))^2-(d ell_R/dp) C/(1-z^2)^(3/2). The upper branch follows by probability complementation.',
            'remaining_gap':'The required global curvature sign has not been proved from physicality at R*. Joint convexity of KL alone is insufficient, and the corresponding claim fails for unrestricted large radii. The prior operator proof uses a normalized trace-one state with A=1/2 and does not directly establish the general tangent-null claim.'},
        'deterministic_checks':{'mpmath_precision_digits':120,'formula_checks':checks,
                                'scope':'Independent high-precision algebra/stationarity diagnostics. They do not replace the compactness/implicit-function proof or certify a numerical epsilon.'},
        'source_sha256':{Path(__file__).resolve().relative_to(ROOT).as_posix():hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         'docs/CONTINUOUS_SYMMETRIC_DESIGN.md':hashlib.sha256((ROOT/'docs/CONTINUOUS_SYMMETRIC_DESIGN.md').read_bytes()).hexdigest(),
                         'results/sota_round5/continuous_chernoff_refinement.json':hashlib.sha256((ROOT/'results/sota_round5/continuous_chernoff_refinement.json').read_bytes()).hexdigest()}}
    path=Path(__file__).with_suffix('.json')
    with path.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({'artifact':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                      'status':result['status'],'high_precision_cases':len(checks),
                      'global_conjecture_status':result['global_all_C_conjecture_status']},indent=2))


if __name__=='__main__':
    main()
