"""Matched exact-parameter adaptive-angle cost obstruction, without sampling.

Parameters A,B,C are the exact rational values of their parsed binary64
numbers. The baseline uses the exact rational A-C and A+C, NOT another
binary64 rounding of these probabilities. Candidate lower bounds use U(A,C)
for this same continuous-angle Bernoulli model.
"""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiments/sota_round4'))
import interval_sign_cost as operating
sys.path.insert(0,str(ROOT/'results/sota_round4/theory'))
import finite_budget_information_bounds as information

PLAN=ROOT/'results/sota_round3/paired_benchmark/preliminary_plan.json'
BETA=ROOT/'results/sota_round3/cap16384_preparation/beta_exact.json'
INFO_AUDIT=ROOT/'results/sota_round4/theory/finite_budget_information_bounds.json'
INTERVAL_AUDIT=ROOT/'results/sota_round4/theory/audit_interval_sign_cost.json'
BETA_AUDIT=ROOT/'results/sota_round2/theory/cap16384_preparation_independent_audit.json'
OUT=Path(__file__).with_suffix('.json')
CASE_IDS=(9,7,19,21,31,33,43,45)
EXPECTED_INTERVAL_SOURCE='14a01510987617c05e37d991c995698e99d8f6cc28d5e7e7c5afe51512b48c40'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact(x):
    return {'numerator':str(x.numerator),'denominator':str(x.denominator)}


def bracket_root(product,degree,scale=10**12):
    low,high=0,4*scale
    while Fraction(high,scale)**degree<=product:
        high*=2
    while high-low>1:
        mid=(low+high)//2
        if Fraction(mid,scale)**degree<=product:
            low=mid
        else:
            high=mid
    assert Fraction(low,scale)**degree<=product<Fraction(high,scale)**degree
    return {'scale':scale,'lower_integer_tick':low,'upper_integer_tick':high,
            'lower_exact':exact(Fraction(low,scale)),
            'strict_upper_exact':exact(Fraction(high,scale))}


def main():
    started=time.perf_counter()
    assert sha(operating.__file__)==EXPECTED_INTERVAL_SOURCE
    interval_audit=json.loads(INTERVAL_AUDIT.read_text(encoding='utf-8'))
    assert interval_audit['status']=='passed' and interval_audit['target_source_sha256']==EXPECTED_INTERVAL_SOURCE
    information_audit=json.loads(INFO_AUDIT.read_text(encoding='utf-8'))
    for path,expected in information_audit['source_sha256'].items():
        assert sha(ROOT/path)==expected
    beta=json.loads(BETA.read_text(encoding='utf-8'))
    beta_audit=json.loads(BETA_AUDIT.read_text(encoding='utf-8'))
    assert beta_audit['status']=='passed'
    beta_record=next(r for r in beta_audit['artifacts'] if r['path']==BETA.relative_to(ROOT).as_posix())
    assert beta_record['file_sha256']==sha(BETA) and beta_record['status']=='passed'
    assert beta['max_discordances']==8192
    assert 100*int(beta['fair_null_risk']['crossing_numerator'])<=int(beta['fair_null_risk']['fair_denominator'])
    plan=json.loads(PLAN.read_text(encoding='utf-8'))
    cases={c['case_id']:c for c in plan['cases']}
    sources=[Path(__file__).resolve(),Path(operating.__file__),Path(information.__file__),PLAN,BETA,
             INFO_AUDIT,INTERVAL_AUDIT,BETA_AUDIT,
             ROOT/'experiments/circuit_validation/direct_tests.py',
             ROOT/'docs/CONTINUOUS_PSR_INFORMATION_BOUND.md',
             ROOT/'docs/CONTINUOUS_SYMMETRIC_DESIGN.md']
    source_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sources}
    kappa=information.kl(Fraction(19,20),Fraction(1,100))
    kappam=information.mkl(Fraction(19,20),Fraction(1,100))
    rows=[]
    product=Fraction(1)
    for case_id in CASE_IDS:
        start=time.perf_counter()
        case=cases[case_id]
        assert case['kind']=='main' and case['C']>0
        A,B,C=[Fraction.from_float(case[k]) for k in ('A','B','C')]
        pm,pp=A-C,A+C
        assert 0<pm<pp<1
        m=min(A,1-A)
        assert B*B+C*C<=m*m
        # Exact Fractions are passed through, with no float(A +/- C).
        oc=operating.operating_intervals(beta['boundaries'],pm,pp,8192)
        assert oc['exact_binary_model_probabilities']==[str(pm),str(pp)]
        assert Fraction(oc['plus'][0])>=Fraction(19,20)
        info=information.calculate(A,B,C,kappa,kappam)
        lower=Fraction(info['expected_physical_shots_lower_bound_arbitrary_adaptive_angles']['lower'])
        high=Fraction(oc['expected_shots'][1])
        ratio=high/lower
        product*=ratio
        mirror=cases[case_id+1000]
        Am,Bm,Cm=[Fraction.from_float(mirror[k]) for k in ('A','B','C')]
        assert (Am,Bm,Cm)==(A,B,-C)
        assert (Am-Cm,Am+Cm)==(pp,pm)
        rows.append({'case_id':case_id,'mirror_case_id':case_id+1000,
                     'exact_parameter_model':{'A':exact(A),'B':exact(B),'C':exact(C)},
                     'parameter_binary64_hex':{k:case[k].hex() for k in ('A','B','C')},
                     'exact_psr_probabilities':{'minus':exact(pm),'plus':exact(pp)},
                     'probabilities_were_not_rounded_again':True,
                     'beta_operating_intervals_exact_parameter_model':oc,
                     'beta_power_lower_endpoint_at_least_95_percent':True,
                     'information_bounds_exact_parameter_model':info,
                     'candidate95percent_expected_shots_lower_endpoint_exact':exact(lower),
                     'beta_expected_shots_upper_endpoint_exact':exact(high),
                     'candidate_speedup_upper_exact':exact(ratio),
                     'individual_threefold_cost_gain_excluded':ratio<3,
                     'exact_threefold_cross_multiplication_slack':str(3*ratio.denominator-ratio.numerator),
                     'elapsed_seconds':time.perf_counter()-start})
        print(json.dumps({'case_id':case_id,'baseline_cost':oc['expected_shots'],
                          'baseline_power':oc['plus'],
                          'any_angle95percent_cost_lower':str(lower),
                          'maximum_speedup_float_diagnostic':float(ratio),
                          'individual_threefold_excluded':ratio<3,
                          'seconds':rows[-1]['elapsed_seconds']}),flush=True)
    assert all(sha(ROOT/path)==digest for path,digest in source_hashes.items())
    degree=len(rows)
    root=bracket_root(product,degree)
    # Upward thousandth bound, checked by an exact integer-power comparison.
    thousandth=(root['upper_integer_tick']*1000+root['scale']-1)//root['scale']
    rounded=Fraction(thousandth,1000)
    assert product<rounded**degree
    below_three=product<3**degree
    result={
        'status':'certified_exact_parameter_adaptive_angle_cost_obstruction',
        'new_simulation_observations':0,
        'goal_achieved':False,
        'exact_parameter_semantics':'Each input A,B,C is the exact rational represented by its parsed binary64 value. The continuous-angle model is p(x)=A+B cos(x)+C sin(x) in real arithmetic. Baseline PSR rates are exact rational A-C and A+C, with NO second rounding to binary64. Candidate U(A,C) and baseline operating characteristics refer to this same model.',
        'baseline':{'name':'beta_exact','pair_cap':8192,'physical_shot_cap':16384,
                    'monitoring':'Every complete PSR pair',
                    'every_positive_case_power_certified_at_least_95_percent':True,
                    'exact_null_crossing_probability_at_most_one_percent':True},
        'candidate_scope':{
            'model':'Stationary single-harmonic Bernoulli p(x)=A+B cos(x)+C sin(x), physical B^2+C^2<=min(A,1-A)^2.',
            'design':'Arbitrary angles and allocation chosen predictably from all previous paid outcomes and independent internal randomness; stopping may be after any paid shot, with any finite cap.',
            'error_and_power':'Probability of a sign report<=.01 at EVERY physical C=0 null, and correct-sign report probability>=.95 separately at each compared nonnull case.',
            'cost':'Unconditional expected number of single Bernoulli queries. Every query costs one, including data used for calibration/exploration and all wrong/abstaining paths.',
            'no_truth_access':'The policy and output kernels are identical under the true parameter and its counterfactual null; parameters are not supplied to the candidate.'},
        'theorem':{
            'explicit_null':'For m=min(A,1-A) and0<|C|<m, q*(x)=A+B lambda(C/m) cos(x), lambda(v)=atanh(v)/v. This is a single feasible C=0 null for all angles, with a positive physical margin certified per row.',
            'pointwise_information':'The cited analytic proof gives kl(p(x)||q*(x))<=U(A,C)=2m Phi(C/m) for every real angle x. No finite angle grid or optimized sampling distribution is substituted for this uniform statement.',
            'adaptive_change_of_measure':'Finite transcript chain rule gives KL(P_theta transcript||P_q* transcript)<=U E_theta[N] even for adaptive angles, unbalanced counts, and mid-pair stopping. For the correct terminal-sign event, binary data processing and the.95/.01 requirements give KL>=kappa=kl(.95,.01). Hence candidate expected cost>=kappa/U.',
            'matched_cost_product':'For each case i, certified beta mean B_i<=b_i and candidate mean C_i>=l_i, where b_i is the outward binary64 cost upper endpoint and l_i is the outward Decimal lower endpoint of kappa/U. Exact Fraction arithmetic yields geometric speedup G<=(product b_i/l_i)^(1/8).',
            'mirrors':'C->-C leaves Phi and lambda unchanged, hence U unchanged; exact PSR rates swap, so the symmetric beta baseline has the identical expected cost and correct power. Applying the same lower bound to each mirror yields the same16-case geometric bound, without requiring a mirror-symmetric candidate.',
            'status':'Analytic proof plus audited numerical enclosure primitives and source-bound exact rational comparisons. Independent round5 review is pending; no proof-assistant certification.'},
        'kappa':information.record(kappa,kappam),
        'rows':rows,
        'exact_product_of_eight_upper_speedup_ratios':exact(product),
        'geometric_upper_root_bracket_by_integer_bisection':root,
        'upward_thousandth_geometric_speedup_bound':exact(rounded),
        'exact_geometric_threefold_excluded':below_three,
        'exact_geometric_threefold_integer_slack':str(3**degree*product.denominator-product.numerator),
        'exact_rounded_bound_integer_slack':str(rounded.numerator**degree*product.denominator-product.numerator*rounded.denominator**degree),
        'every_individual_case_threefold_excluded':all(r['individual_threefold_cost_gain_excluded'] for r in rows),
        'mechanistic_implications':[
            'If the recorded threefold exclusions hold, changing only angle choice, allocation, stopped-data use, prior, boundary or paid exploration within this Bernoulli resource model cannot reach the stated3x target on these fixed cases.',
            'The proof already uses the full Bernoulli transcript and the physical parameter constraint. Recovering discarded counts or enforcing the same physical constraint more carefully cannot evade its lower bound.',
            'At A=.5 the explicit upper bound equals the full-information PSR value, so angle choice has no oracle information-rate advantage. For the other cases U/E<2. Their combined information room and baseline inefficiency determine this cost ceiling.',
            'This does not make the whole project impossible. A valid new mechanism would have to change an assumption explicitly: e.g. justified external calibration that truly restricts the null family, an additional observable/resource outside the single-harmonic Bernoulli query family, a richer physical task, or a different predeclared scientific endpoint. Its full information and resource costs must then be charged and compared with strong applicable methods.',
            'Changing benchmarks or conditioning away abstentions solely to make the3x number larger would not resolve this obstruction for the original task.',
            'The certified upper limit is necessary, not an achievable gain or proof of SOTA. The original research objective remains active.'
        ],
        'source_sha256':source_hashes,'elapsed_seconds':time.perf_counter()-started}
    with OUT.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({'artifact':str(OUT),'sha256':sha(OUT),
                      'threefold_geometric_excluded':below_three,
                      'upward_thousandth_bound':str(rounded),
                      'root_bracket':root,
                      'all_individual_threefold_excluded':result['every_individual_case_threefold_excluded'],
                      'elapsed_seconds':result['elapsed_seconds']},indent=2),flush=True)


if __name__=='__main__':
    main()
