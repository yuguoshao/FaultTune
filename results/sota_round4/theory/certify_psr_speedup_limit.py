"""Exact rational proof that a threefold PSR cost gain is impossible here.

Uses only certified baseline cost upper endpoints and certified95%-power
cost lower endpoints for the SAME rounded binary64 arm probabilities.
"""
from fractions import Fraction
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).with_suffix('.json')
INPUTS = {
    'operating':'results/sota_round4/certified_operating_comparison.json',
    'bounds':'results/sota_round4/theory/finite_budget_information_bounds.json',
    'beta':'results/sota_round3/cap16384_preparation/beta_exact.json',
    'beta_audit':'results/sota_round2/theory/cap16384_preparation_independent_audit.json'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frac(record):
    return Fraction(int(record['numerator']),int(record['denominator']))


def exact(value):
    return {'numerator':str(value.numerator),'denominator':str(value.denominator)}


def root_bracket(product,degree,scale=10**12):
    low,high=0,3*scale
    assert Fraction(high,scale)**degree>product
    while high-low>1:
        middle=(low+high)//2
        if Fraction(middle,scale)**degree<=product:
            low=middle
        else:
            high=middle
    assert Fraction(low,scale)**degree<=product<Fraction(high,scale)**degree
    return {'lower':str(Fraction(low,scale)),'strict_upper':str(Fraction(high,scale)),
            'scale':scale,'lower_integer_tick':low,'upper_integer_tick':high}


def main():
    data={name:json.loads((ROOT/path).read_text(encoding='utf-8')) for name,path in INPUTS.items()}
    assert data['operating']['status']=='certified_deterministic_operating_comparison'
    assert data['bounds']['status']=='proved_finite_cap_theorem_with_outward_formula_evaluation'
    assert data['beta_audit']['status']=='passed'
    audited_beta=next(r for r in data['beta_audit']['artifacts'] if r['path']==INPUTS['beta'])
    assert audited_beta['status']=='passed' and audited_beta['file_sha256']==sha(ROOT/INPUTS['beta'])
    beta=data['beta']
    assert beta['max_discordances']==8192
    fair=beta['fair_null_risk']
    assert 100*int(fair['crossing_numerator'])<=int(fair['fair_denominator'])
    costs={r['case_id']:r for r in data['operating']['rows'] if r['method']=='beta_exact'}
    lower={r['case_id']:r for r in data['bounds']['rows']}
    assert sorted(costs)==sorted(lower)==[7,9,19,21,31,33,43,45]
    rows=[]
    product=Fraction(1)
    for case_id in sorted(costs):
        cost=costs[case_id]
        bound=lower[case_id]['actual_round3_binary64_psr_probabilities']
        pm=Fraction(cost['source_binary64_probabilities']['pminus_rational'])
        pp=Fraction(cost['source_binary64_probabilities']['pplus_rational'])
        assert pm==frac(bound['minus']) and pp==frac(bound['plus'])
        assert Fraction(cost['operating_intervals']['plus'][0])>=Fraction(19,20)
        high=Fraction(cost['operating_intervals']['expected_shots'][1])
        low=Fraction(bound['expected_shots_lower_bound']['lower'])
        assert low>0
        ratio=high/low
        product*=ratio
        rows.append({'case_id':case_id,'exact_arm_probability_match':True,
                     'beta_correct_probability_lower_endpoint_binary64_rational':exact(Fraction(cost['operating_intervals']['plus'][0])),
                     'beta_mean_cost_upper_endpoint_exact':exact(high),
                     'candidate95percent_mean_cost_lower_endpoint_exact':exact(low),
                     'maximum_cost_speedup_upper_exact':exact(ratio)})
    degree=len(rows)
    threshold=Fraction(1641,1000)
    assert product<threshold**degree<Fraction(3)**degree
    slack=threshold.numerator**degree*product.denominator-product.numerator*threshold.denominator**degree
    assert slack>0
    ratio9=frac(next(r['maximum_cost_speedup_upper_exact'] for r in rows if r['case_id']==9))
    assert ratio9<Fraction(743,500)
    slack9=743*ratio9.denominator-500*ratio9.numerator
    assert slack9>0
    result={
        'status':'passed_exact_rational_impossibility_certificate',
        'new_simulation_observations':0,
        'baseline':'Frozen beta_exact H8192, common physical-shot cap16384, inspection after every completed PSR pair.',
        'candidate_requirements':{
            'observations':'Only independent Bernoulli observations at the same fixed +/-pi/2 PSR arms, balanced in complete paid pairs; stopping at complete-pair boundaries; no uncharged side information.',
            'correct_probability':'At least19/20 separately on every one of the eight positive registered cases (and each mirror if claiming the16-case version).',
            'null_report_probability':'At most1/100 for every physical null C=0.',
            'cost':'Unconditional expected physical shots, including wrong reports and abstentions. Any finite cap is allowed; a larger cap does not remove the information lower bound.'},
        'baseline_qualification':{'every_case95percent_power_checked_from_certified_lower_endpoint':True,
                                  'null_report_budget_checked_by_exact_integer_crossing_count':True,
                                  'previous_full_boundary_audit_bound_to_current_file':True},
        'proof':[
            'For each i, the finite-transcript KL theorem gives candidate mean C_i>=L_i, where L_i is the stored downward-enclosing endpoint of kl(.95,.01)/E_i for the exact rounded binary64 PSR arm probabilities.',
            'The certified interval computation gives the actual baseline mean B_i<=U_i. Every baseline cell separately has correct probability>=.95 and the exact frozen boundary has null crossing probability<=.01.',
            'Therefore the equal-case geometric speedup G=(product_i B_i/C_i)^(1/8) is at most (product_i U_i/L_i)^(1/8).',
            'Each U_i is converted from binary64 to its exact rational. Each L_i is parsed from its outward decimal endpoint as an exact rational. Their ratio product P is computed with Fraction arithmetic.',
            'Integer cross multiplication proves P<(1641/1000)^8<3^8. Hence G<1.641 and a geometric threefold gain is impossible within the stated PSR/resource/power scope.',
            'For case9 alone exact cross multiplication gives B_9/C_9<=U_9/L_9<743/500=1.486.',
            'For all16 sign-mirrored cases, beta costs and the information lower bounds repeat exactly by arm swap. Applying the bound to each candidate mirror separately yields product<=P^2, so the same G<1.641 conclusion holds even if the candidate itself is not mirror symmetric.'
        ],
        'rows':rows,'product_of_eight_speedup_upper_ratios_exact':exact(product),
        'geometric_speedup_strict_upper_exact':exact(threshold),
        'geometric_speedup_upper_root_bracket_by_integer_bisection':root_bracket(product,degree),
        'integer_cross_multiplication_witness':{
            'formula':'1641^8 * P.denominator - 1000^8 * P.numerator >0',
            'positive_integer_slack':str(slack),
            'product_strictly_below_3_power_8':product<3**8},
        'case9':{'strict_upper_exact':exact(Fraction(743,500)),
                 'ratio_upper_exact':exact(ratio9),
                 'positive_integer_slack_743den_minus_500num':str(slack9)},
        'claim_boundary':[
            'This proves an impossibility for the specified balanced-PSR statistical cost comparison, not a claim of algorithmic optimality or SOTA superiority.',
            'The bound is necessary and can be unattainable. Less than1.641 remaining multiplicative room does not imply that this gain is achievable.',
            'It does not cover arbitrary adaptive angles, unequal or mid-pair stopping, other observation resources, other comparator algorithms, different power/error requirements, or CPU/runtime speedups.',
            'No empirical cost or successful-trajectory mean is substituted for a true expected cost; all inputs are deterministic certified bounds with exactly matched sampler probabilities.',
            'No floating logarithm or root approximation is used in either the1.641 or3-fold impossibility certificate.'
        ],
        'source_sha256':{**{path:sha(ROOT/path) for path in INPUTS.values()},
                         Path(__file__).resolve().relative_to(ROOT).as_posix():sha(__file__)}}
    with OUT.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({'artifact':str(OUT),'sha256':sha(OUT),
                      'maximum_geometric_speedup_strict_upper':'1.641',
                      'root_bracket':result['geometric_speedup_upper_root_bracket_by_integer_bisection'],
                      'case9_maximum_speedup_strict_upper':'1.486','new_observations':0},indent=2))


if __name__=='__main__':
    main()
