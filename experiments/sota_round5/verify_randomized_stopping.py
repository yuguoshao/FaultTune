"""Exact small physical-word checks and original-case interval evaluation."""
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
from fractions import Fraction as F
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.sota_round5.interval_randomized_stopping import canonical_policy, operating_intervals


def contains(interval, exact):
    assert all(math.isfinite(float(x)) for x in interval)
    assert F(float(interval[0]))<=exact<=F(float(interval[1])), (interval,exact)


def exact_physical_words(policy,pm,pp,cap):
    den,rows=policy
    rates=((1-pm)*(1-pp),(1-pm)*pp,pm*(1-pp),pm*pp)
    result=dict(plus=F(0),minus=F(0),abstain=F(0),expected_shots=F(0))
    for word in itertools.product(range(4),repeat=cap):
        probability=F(1)
        for outcome in word: probability*=rates[outcome]
        if not probability: continue
        remaining=probability
        d,k=0,0
        for t,outcome in enumerate(word,1):
            if outcome not in (1,2): continue
            d+=1; k+=int(outcome==1)
            if d>=len(rows): continue
            chance=F(rows[d][min(k,d-k)],den)
            stopped=remaining*chance
            if stopped:
                result['plus' if 2*k>d else 'minus']+=stopped
                result['expected_shots']+=2*t*stopped
                remaining-=stopped
        result['abstain']+=remaining
        result['expected_shots']+=2*cap*remaining
    return result


def small_checks():
    count=0
    models=[(F(0),F(0)),(F(1),F(1)),(F(0),F(1)),(F(1),F(0)),
            (F(1,3),F(2,3)),(F(.1),F(.3)),(F(3,5),F(1,5)),
            (F(0),F(1,2**1075))]
    for h in (1,2,4):
        for recipe in ('never','immediate','mixed'):
            rows=[]
            for d in range(h+1):
                row=[]
                for j in range(d//2+1):
                    row.append(0 if 2*j==d or recipe=='never' else
                               12 if recipe=='immediate' else (7*d+3*j)%13)
                rows.append(row)
            policy=canonical_policy(12,rows)
            for cap in (1,3,5):
                for pm,pp in models:
                    exact=exact_physical_words(policy,pm,pp,cap)
                    interval=operating_intervals(policy,pm,pp,cap)
                    for name,value in exact.items(): contains(interval[name],value)
                    count+=1
    return count


def decimal_endpoint(value,upper):
    with localcontext() as ctx:
        ctx.prec=50
        ctx.rounding=ROUND_CEILING if upper else ROUND_FLOOR
        return str(Decimal(value.numerator)/Decimal(value.denominator))


def geometric_root_bounds(product_lo,product_hi,degree):
    scale=10**12
    lower,upper=0,scale
    while F(upper,scale)**degree<=product_hi: upper*=2
    while upper-lower>1:
        mid=(lower+upper)//2
        if F(mid,scale)**degree<=product_hi: lower=mid
        else: upper=mid
    high=upper
    lower,upper=0,high
    while upper-lower>1:
        mid=(lower+upper)//2
        if F(mid,scale)**degree<=product_lo: lower=mid
        else: upper=mid
    return [decimal_endpoint(F(lower,scale),False),decimal_endpoint(F(high,scale),True)]


def main():
    out=ROOT/'results/sota_round5/optimal_sign_stopping/physical_cap16384_intervals.json'
    if out.exists(): raise FileExistsError('Do not overwrite prior certification')
    checks=small_checks()
    proto_path=ROOT/'results/sota_round5/optimal_sign_stopping/prototype.json'
    proto=json.loads(proto_path.read_text(encoding='utf-8'))['protocol']
    policy=canonical_policy(proto['probability_denominator'],proto['policy_numerators'])
    plan_path=ROOT/'results/sota_round3/paired_benchmark/preliminary_plan.json'
    cases=json.loads(plan_path.read_text(encoding='utf-8'))['cases']
    floatpath=out.with_name('physical_cap16384_float.json')
    approximate={row['case_id']:row for row in json.loads(floatpath.read_text(encoding='utf-8'))['rows']}
    rows=[]; outside=[]
    for case in cases:
        pm,pp=case['A']-case['C'],case['A']+case['C']
        interval=operating_intervals(policy,pm,pp,8192)
        correct=interval['plus'] if case['C']>0 else interval['minus'] if case['C']<0 else None
        rows.append({'case_id':case['case_id'],'kind':case['kind'],'C_sign':int(np.sign(case['C'])),
                     'operating_intervals':interval,'correct_probability':correct})
        for name,old in [('plus','positive_probability'),('minus','negative_probability'),
                          ('abstain','abstain_probability'),('expected_shots','expected_shots')]:
            point=approximate[case['case_id']][old]
            if not interval[name][0]<=point<=interval[name][1]:
                outside.append({'case_id':case['case_id'],'quantity':name,'float':point,'interval':interval[name]})
    mainrows=[r for r in rows if r['kind']=='main']
    assert all(F(r['correct_probability'][0])>=F(19,20) for r in mainrows)
    positive=[r for r in mainrows if r['C_sign']>0]
    comparison_path=ROOT/'results/sota_round4/certified_operating_comparison.json'
    bases=json.loads(comparison_path.read_text(encoding='utf-8'))
    comparisons=[]
    for method in bases['methods']:
        base={r['case_id']:r['operating_intervals'] for r in bases['rows'] if r['method']==method}
        productlo,producthi=F(1),F(1)
        for row in positive:
            a=row['operating_intervals']['expected_shots']; b=base[row['case_id']]['expected_shots']
            productlo*=F(a[0])/F(b[1]); producthi*=F(a[1])/F(b[0])
        comparisons.append({'method':method,'geometric_cost_ratio':geometric_root_bounds(productlo,producthi,len(positive))})
    sources=[Path(__file__).resolve(),ROOT/'experiments/sota_round5/interval_randomized_stopping.py',
             ROOT/'experiments/sota_round4/interval_sign_cost.py',proto_path,plan_path,comparison_path,floatpath]
    result={'status':'certified_randomized_policy_physical_operating_characteristics',
            'new_simulation_observations':0,'exact_small_physical_word_models_checked':checks,
            'physical_shot_cap':16384,'policy_discordance_horizon':len(policy[1])-1,
            'all_16_main_cases_power_certified_at_least_95_percent':True,
            'minimum_correct_probability_interval':[min(r['correct_probability'][i] for r in mainrows) for i in (0,1)],
            'mean_expected_shots_interval':[decimal_endpoint(sum((F(r['operating_intervals']['expected_shots'][i]) for r in mainrows),F(0))/len(mainrows),bool(i)) for i in (0,1)],
            'comparisons':comparisons,'rows':rows,'float_outside_certified_interval':outside,
            'main_cost_and_correct_probability_crosscheck':all(
                item['quantity'] not in ('expected_shots','plus' if row['C_sign']>0 else 'minus')
                for item in outside for row in mainrows if item['case_id']==row['case_id']),
            'source_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            'scope':'Old development model points and frozen classical LP stopping policy. Every undecided path pays the full physical cap, including paths surviving its discordance horizon. No held-out or SOTA superiority claim. Directed sign risk is separately certified for stationary iid pairs; this evaluator does not authorize adaptive-angle sampling.'}
    with out.open('x',encoding='utf-8') as f: json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('status','exact_small_physical_word_models_checked','minimum_correct_probability_interval','mean_expected_shots_interval','comparisons','float_outside_certified_interval')},indent=2))


if __name__=='__main__': main()
