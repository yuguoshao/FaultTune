"""Exact inverse-KL curvature identity and a certified nonmonotonicity example."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.circuit_validation.direct_tests import _Bounds as I


def limits(value):
    return [str(value.lo), str(value.hi)]


def main():
    p = I(999)/1000
    variance = p*(1-p)
    rows = []
    for v_int in (3, 5):
        v = I(v_int)
        q = p/(p+(1-p)*v.exp())
        r = p-q
        delta = variance*v-r
        j = v+delta*delta/(variance*v*r*r)
        radius = p*(p/q).log()+(1-p)*((1-p)/(1-q)).log()
        rows.append({'p_exact': '999/1000', 'logit_difference_exact': v_int,
                     'q_interval': limits(q), 'radius_interval': limits(radius),
                     'curvature_ratio_interval': limits(j)})
    r3lo, r3hi = rows[0]['radius_interval']
    r5lo, r5hi = rows[1]['radius_interval']
    j3lo, j3hi = rows[0]['curvature_ratio_interval']
    j5lo, j5hi = rows[1]['curvature_ratio_interval']
    from decimal import Decimal
    assert 0 < Decimal(r3lo) <= Decimal(r3hi) < Decimal(r5lo)
    assert Decimal(r5hi) < I(2).log().lo
    assert Decimal(j3lo) > Decimal(j5hi) > 0
    result = {
        'status': 'exact_curvature_identity_and_certified_radius_nonmonotonicity',
        'new_simulation_observations': 0,
        'definition': 'ell_R(p) is the lower endpoint of {q:kl(p||q)<=R}, with0<p<1 andR>0. Thus0<q=ell_R(p)<p. Let v=logit(p)-logit(q)>0, r=p-q>0 andP=p(1-p)>0.',
        'exact_identity': {
            'first_derivative': 'K=d ell_R(p)/dp=q(1-q)*v/r>0.',
            'logarithmic_curvature': 'J=(d2 ell_R/dp2)/(d ell_R/dp)=v+(P*v-r)^2/(P*v*r^2)>0.',
            'hyperbolic_form': 'For k=1/2-p and T(v)=coth(v/2)/2-1/v, J=(v/P)[1/4+2k*T(v)+T(v)^2].',
            'proof': 'Implicit differentiation ofD(p,q)=R gives ell_prime=K and ell_second=(q(1-q)/r)[1/P-2K/(q(1-q))+(p/q^2+(1-p)/(1-q)^2)K^2]. Dividing byK and using p(1-q)^2+(1-p)q^2=P+r^2 gives J=1/(Pv)-2/r+(P+r^2)v/r^2=v+(Pv-r)^2/(Pv r^2). Also q=p/[p+(1-p)e^v], hence Pv/r=v[(1-p)+1/(e^v-1)], which yields the hyperbolic form.'
        },
        'ellipse_reduction': {
            'model': 'p_plus(z)=A+Bz+C sqrt(1-z^2), C>0 and-1<z<1.',
            'curvature_formula': 'Let s=sqrt(1-z^2), evaluateK,J atp=p_plus(z), q=ell_R(p). Then L(z)=ell_R(p_plus(z)) satisfies L_second=(K/s^3)[J*s*(B*s-C*z)^2-C].',
            'exact_remaining_inequality': 'At the model-dependent radiusR*=max_z BernoulliCapacity(A+Bz-C*s,A+Bz+C*s), the desired concavity is exactly J*s*(B*s-C*z)^2<=C for everyz. IfB*s-C*z=0, strict concavity follows immediately.',
            'upper_branch': 'U(z)=u_R(p_minus(z))=1-ell_R(1-p_minus(z)). Thus U convexity is precisely the same lower-concavity condition for the complemented model(Abar,Bbar,Cbar)=(1-A,-B,C).',
            'status': 'The reduction is proved. The remaining inequality has not been proved for all physical models, including atR*. This file does not establish the unrestricted reflected-pair conjecture.'
        },
        'radius_nonmonotonicity': {
            'claim': 'For fixedp, J need not increase withR, even when both radii are belowlog(2).',
            'exact_construction': 'Usep=999/1000 andv=3 or5 exactly, q_v=p/[p+(1-p)exp(v)], R_v=kl(p||q_v). The retained64-digit outward intervals prove0<R_3<R_5<log(2), whileJ_3>J_5>0.',
            'certified_intervals': rows,
            'implication': 'Verifying curvature at a larger radius upper boundU does not by itself verify it atR*, unless an additional justified monotonicity or domination property is available. This counterexample concerns the inverse-KL ratio only, not the physical-ellipse conjecture.'
        },
        'source_sha256': {
            Path(__file__).resolve().relative_to(ROOT).as_posix(): hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'experiments/circuit_validation/direct_tests.py': hashlib.sha256((ROOT/'experiments/circuit_validation/direct_tests.py').read_bytes()).hexdigest()
        },
        'claim_boundary': 'Analytic simplification and a certified caution about a proof shortcut; no new data, no full all-angle theorem, no finite-cost claim and no SOTA claim.'
    }
    out = Path(__file__).with_suffix('.json')
    with out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'path': str(out), 'sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
                      'status': result['status'], 'nonmonotonicity': rows}, indent=2))


if __name__ == '__main__':
    main()
