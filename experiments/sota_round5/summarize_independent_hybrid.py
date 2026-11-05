"""Bound the complete predeclared independent comparison without new sampling."""
from datetime import datetime,timezone
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.sota_round5.verify_randomized_stopping import decimal_endpoint,geometric_root_bounds

DIRECTORY=ROOT/'results/sota_round5/hybrid_minimax'


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def cost_comparison(candidate,control,ids):
    plo,phi=F(1),F(1); alo,ahi,blo,bhi=F(0),F(0),F(0),F(0)
    lower_count,upper_count,overlap=0,0,0
    percase=[]
    for i in ids:
        a=list(map(F,candidate[i]['expected_shots'])); b=list(map(F,control[i]['expected_shots']))
        low,high=a[0]/b[1],a[1]/b[0]
        plo*=low;phi*=high; alo+=a[0];ahi+=a[1];blo+=b[0];bhi+=b[1]
        lower_count+=high<1;upper_count+=low>1;overlap+=not(high<1 or low>1)
        percase.append({'case_id':i,'cost_ratio':[decimal_endpoint(low,False),decimal_endpoint(high,True)]})
    return {'geometric_cost_ratio':geometric_root_bounds(plo,phi,len(ids)),
            'arithmetic_cost_ratio':[decimal_endpoint(alo/bhi,False),decimal_endpoint(ahi/blo,True)],
            'candidate_proved_cheaper_cases':lower_count,'candidate_proved_more_expensive_cases':upper_count,
            'cost_interval_overlaps':overlap,'per_case':percase}


def main():
    output=DIRECTORY/'independent_comparison.json'
    if output.exists():raise FileExistsError('Preserve first complete comparison')
    names=['independent_grid_plan.json','independent_candidate_intervals.json',
           'independent_controls_a.json','independent_controls_b.json',
           'independent_joint_null_review.json','candidate_policy.json',
           'independent_candidate_float.json']
    paths=[DIRECTORY/name for name in names]
    plan,candidate,parta,partb,review,policy,approx=map(read,paths)
    assert sha(paths[5])=='a2690100803c140b166068f5203b441eb24e01298485721cb11ffbb68b3b7e72'
    assert sha(paths[0])=='64e43a84da3a1265e3ef125b3e458f5a694fe19851703aff5aa1200d4925df16'
    expected={row['case_id']:row for row in plan['rows']}
    ids=[i for i,row in expected.items() if row['kind']=='main']
    cand={row['case_id']:row for row in candidate['rows']}
    controls={method:{} for method in plan['comparison_methods']}
    for row in parta['rows']+partb['rows']:
        method=row.get('method',row.get('control'))
        assert method in controls
        assert row['case_id'] not in controls[method]
        controls[method][row['case_id']]=row
    for values in [cand]+list(controls.values()):
        assert set(values)==set(expected)
        for i,row in values.items():
            assert F(row['r_exact'])==F(expected[i]['r_exact'])
            assert F(row['q_exact'])==F(expected[i]['q_exact'])
            assert row['kind']==expected[i]['kind']
    qualified=[method for method,values in controls.items()
               if all(F(values[i]['correct_probability'][0])>=F(19,20) for i in ids)]
    assert candidate['uniform_null_risk_at_most_one_percent']
    assert candidate['all_main_correct_probability_at_least_95_percent']
    comparisons={method:cost_comparison(cand,values,ids) for method,values in controls.items()}
    envelope={i:{'expected_shots':[min(F(controls[m][i]['expected_shots'][j]) for m in qualified) for j in (0,1)]} for i in ids}
    envelope_comparison=cost_comparison(cand,envelope,ids)
    outsiders=[]
    for row in approx['rows']:
        i=row['case_id']
        mappings=[('expected_shots','expected_shots')]
        if i in ids:mappings.append(('correct_probability','correct_probability'))
        for field,old in mappings:
            point=F(row[old]); lo,hi=map(F,cand[i][field])
            if not lo<=point<=hi:outsiders.append({'case_id':i,'quantity':field,'float':row[old],'interval':cand[i][field]})
    candidate_mean=[decimal_endpoint(sum((F(cand[i]['expected_shots'][j]) for i in ids),F(0))/len(ids),bool(j)) for j in (0,1)]
    result={'status':'complete_certified_independent_parameter_comparison','created_utc':datetime.now(timezone.utc).isoformat(),
            'candidate_sha256':sha(paths[5]),'main_cases':len(ids),'null_cases':len(expected)-len(ids),
            'qualified_fixed_controls':qualified,'excluded_controls':[m for m in controls if m not in qualified],
            'candidate_minimum_correct_probability_interval':candidate['minimum_main_correct_probability_interval'],
            'candidate_mean_expected_shots_interval':candidate_mean,
            'candidate_user_one_percent_risk_pass':True,'candidate_stricter_internal_00999_pass':False,
            'all_fixed_comparisons':comparisons,'qualified_casewise_envelope_comparison':envelope_comparison,
            'candidate_cost_and_power_float_crosscheck_outside_intervals':outsiders,
            'sota_goal_complete':False,'new_simulation_observations':0,
            'source_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in paths+[Path(__file__).resolve(),ROOT/'experiments/sota_round5/verify_randomized_stopping.py']},
            'scope':['The same first frozen policy was evaluated on the complete previously sealed grid. No candidate replacement, point removal, or online truth access.',
                     'All comparisons use the same fixed-pair resources and physical cap. A method qualifies only if every main point has certified correct probability>=.95 and its uniform1 percent risk certificate is valid.',
                     'The casewise minimum of qualified controls is a hindsight cost comparator, not an executable policy.',
                     'These are exact-model numerical interval results on fixed parameters, not a hardware or new-circuit experiment, a proof of novelty, or full-field SOTA superiority.',
                     'The original stricter internal development-risk target failed by about4.06e-8. Independent testing does not erase that failure.']}
    with output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({'qualified':qualified,'mean_cost':candidate_mean,'minimum_power':result['candidate_minimum_correct_probability_interval'],
                      'geometric_ratios':{m:v['geometric_cost_ratio'] for m,v in comparisons.items()},
                      'envelope_geometric_ratio':envelope_comparison['geometric_cost_ratio'],'cost_power_crosscheck_outliers':outsiders},indent=2))


if __name__=='__main__':main()
