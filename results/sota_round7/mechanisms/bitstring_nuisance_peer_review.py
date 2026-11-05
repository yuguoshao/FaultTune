"""Independent exact/rational and high-precision review of the bitstring diagnostic."""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import mpmath as mp

ROOT=Path(__file__).resolve().parents[3]
mp.mp.dps=150


def m(value):
    value=F(value)
    return mp.mpf(value.numerator)/value.denominator


def kl(p,q):
    p,q=m(p),m(q)
    return p*mp.log(p/q)+(1-p)*mp.log((1-p)/(1-q))


def contains(row,value):
    assert m(row['lower_decimal'])<=value<=m(row['upper_decimal'])


def main():
    source=ROOT/'results/sota_round7/mechanisms/bitstring_nuisance_diagnostic.json'
    data=json.loads(source.read_text(encoding='utf-8'))
    for name,digest in data['source_files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    example=data['example']
    truth=example['private_scorer_truth']
    xo,xn,eta=map(F,(truth['observed_q0_old'],truth['observed_q0_new'],truth['readout_flip']))
    t=F(truth['ideal_phase_sin_squared'])
    ideal=[1-3*t/4,t/4,t/4,t/4]
    spectator=[]
    for observed in range(4):
        total=F(0)
        for original,p in enumerate(ideal):
            mismatches=((observed^original)&1)+(((observed^original)>>1)&1)
            total+=p*eta**mismatches*(1-eta)**(2-mismatches)
        spectator.append(total)
    assert spectator==list(map(F,truth['common_spectator_pmf_exact']))
    assert sum(spectator)==1
    s=spectator[0]
    assert s==F(1,2)>0
    pmfs=[]
    for rate,key in ((xo,'old_full_pmf_exact'),(xn,'new_full_pmf_exact')):
        probabilities=[(rate if i&1 else 1-rate)*spectator[i>>1] for i in range(8)]
        assert probabilities==list(map(F,truth[key]))
        assert sum(probabilities)==1
        pmfs.append(probabilities)
    yo,yn=1-pmfs[0][0],1-pmfs[1][0]
    assert yn-yo==s*(xn-xo)==F(-1,100)
    jx=(kl(xo,(xo+xn)/2)+kl(xn,(xo+xn)/2))/2
    jy=(kl(yo,(yo+yn)/2)+kl(yn,(yo+yn)/2))/2
    dx=(kl(xo,xn)+kl(xn,xo))/2
    dy=(kl(yo,yn)+kl(yn,yo))/2
    full_dx=(sum(m(p)*mp.log(m(p)/m(q)) for p,q in zip(pmfs[0],pmfs[1]))+
             sum(m(p)*mp.log(m(p)/m(q)) for p,q in zip(pmfs[1],pmfs[0])))/2
    assert abs(full_dx-dx)<mp.mpf('1e-135')
    kappa=kl(F(19,20),F(1,100))
    enlarged=example['enlarged_factorized_model_information']
    reflected=example['physical_binary_reflection_information']
    values={
        'balanced_JS_full_per_shot':jx,'balanced_JS_OR_per_shot':jy,
        'balanced_JS_ratio_full_over_OR':jx/jy,
        'necessary_balanced_total_shots_full':kappa/jx,
        'necessary_balanced_total_shots_OR':kappa/jy,
        'event_kl_95_percent_power_1_percent_risk':kappa}
    for key,value in values.items():contains(enlarged[key],value)
    values2={'balanced_KL_full_per_shot':dx,'balanced_KL_OR_per_shot':dy,
             'balanced_KL_ratio_full_over_OR':dx/dy,
             'necessary_balanced_total_shots_full':kappa/dx,
             'necessary_balanced_total_shots_OR':kappa/dy}
    for key,value in values2.items():contains(reflected[key],value)
    variance=lambda p:p*(1-p)
    vf=2*s*s*(variance(xo)+variance(xn))+(xn-xo)**2*s*(1-s)
    vo=2*(variance(yo)+variance(yn))
    assert vf==F(enlarged['delta_variance_full_coefficient_exact'])
    assert vo==F(enlarged['delta_variance_OR_coefficient_exact'])
    assert vo/vf==F(enlarged['delta_variance_ratio_exact'])
    result={
        'status':'passed_declared_information_diagnostic_with_two_scope_clarifications',
        'new_simulation_observations':0,
        'blocking_findings_for_current_information_diagnostic':[],
        'checks':{'exact_spectator_convolution':True,'exact_full_pmfs_and_Delta':True,
                  'independent_mpmath_digits':mp.mp.dps,'all_reported_information_intervals_contain_independent_values':True,
                  'full_8_outcome_reflection_KL_equals_q0_Bernoulli_KL':True,
                  'exact_asymptotic_variance_coefficients':True,
                  'JS_ratio':mp.nstr(jx/jy,60),'physical_reflection_KL_ratio':mp.nstr(dx/dy,60)},
        'analytic_review':[
            'The Hamiltonian, initial state and local noise factor across q0 versus q1q2. Equal duration with only a q0 amplitude change leaves the spectator lawQ invariant and givesDelta=Q(00)(x_new-x_old).',
            'For exact symbolic controls at common duration1, omega_prime=-u_old-u_new-omega negates and swaps the two q0 rotation phases, so sin^2 symmetry swaps the q0 laws and leavesQ unchanged. This supplies a genuine physical opposite-sign alternative.',
            'The JS common-mean null is legitimate in the enlarged common-Q family. Its complete-pair transcript information per paid shot is JS. The reflected physical alternative instead has symmetrized Bernoulli KL per paid shot. Balanced-stop scope is necessary and is stated.',
            'The displayed information and variance ratios do not establish ratios of attainable stopping costs. The report correctly gives them as diagnostics and allows every strong comparator access to complete bitstrings and the same structural model.',
            'The delta estimator variance coefficient usesN/2 samples per arm andN pooled spectators. Independence ofq0 and spectators makes the first-order covariance vanish, yielding the reported coefficient.'
        ],
        'scope_clarifications':[
            {'item':'Exact symbolic instance versus binary64 gate diagnostic',
             'detail':'The very narrow directed KL intervals certify the exact rational rates and Q generated by the symbolic controls u_old=2asin(sqrt(5/198)),u_new=2asin(sqrt(1/198)) and symbolic gamma. They do not certify the same intervals for the JSON binary64 amplitudes/gamma interpreted as exact model parameters. The gate/Kraus comparisons at about1e-16 are floating consistency checks. A future executable finite-risk result must pick one parameter semantics and recompute bounds accordingly.'},
            {'item':'Common spectator positivity for target-sign sufficiency',
             'detail':'The reduction sign(Delta)=sign(x_new-x_old) requires s=Q(00)>0. An unrestricted arbitrary-Q enlarged family also contains s=0, whereDelta=0 regardless oftheq0 contrast; anX-only decision rule would then fail the target-sign risk guarantee. State s>0 or a physical positive lower bound explicitly. Alternatively, require at least one observed spectator00 before either directional report: this event is impossible whens=0, and gating cannot increase a wrong-report probability. This does not alter the designated s=1/2 diagnostic or its necessary-cost bounds.'}
        ],
        'claim_boundary':'Current information identities and physical symbolic example pass. This review does not certify an implemented validator, finite-budget power, measured cost advantage, arbitrary arm-dependent spectator perturbations, or SOTA superiority.',
        'source_sha256':{source.relative_to(ROOT).as_posix():hashlib.sha256(source.read_bytes()).hexdigest(),
                         **data['source_files'],
                         Path(__file__).resolve().relative_to(ROOT).as_posix():hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    }
    out=Path(__file__).with_suffix('.json')
    with out.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({'path':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'status':result['status'],'checks':result['checks']},indent=2))


if __name__=='__main__':main()
