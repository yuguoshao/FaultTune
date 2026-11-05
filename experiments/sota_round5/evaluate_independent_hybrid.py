"""Evaluate the first frozen hybrid on the independently sealed parameter grid."""
from datetime import datetime,timezone
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.sota_round5.interval_randomized_stopping import canonical_policy
from experiments.sota_round5.interval_hybrid_stopping import operating_rq

DIRECTORY=ROOT/'results/sota_round5/hybrid_minimax'


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    target=DIRECTORY/'independent_candidate_intervals.json'
    if target.exists(): raise FileExistsError('Do not overwrite independent evaluation')
    started=time.perf_counter()
    planpath=DIRECTORY/'independent_grid_plan.json'
    candidatepath=DIRECTORY/'candidate_policy.json'
    developmentpath=DIRECTORY/'development_plan.json'
    riskpath=DIRECTORY/'joint_null_exact.json'
    baselinepath=ROOT/'results/sota_round3/cap16384_preparation/beta_exact.json'
    plan,candidate,risk=read(planpath),read(candidatepath),read(riskpath)
    assert sha(planpath)=='64e43a84da3a1265e3ef125b3e458f5a694fe19851703aff5aa1200d4925df16'
    assert sha(candidatepath)=='a2690100803c140b166068f5203b441eb24e01298485721cb11ffbb68b3b7e72'
    assert sha(developmentpath)==candidate['plan_sha256']
    assert risk['candidate_sha256']==sha(candidatepath)
    actual_risk=F(int(risk['risk_exact_numerator']),int(risk['risk_exact_denominator']))
    assert actual_risk<=F(1,100)
    protocol=candidate['protocol']
    policy=canonical_policy(protocol['probability_denominator'],protocol['policy_numerators'])
    boundary=read(baselinepath)['boundaries']
    assert len(policy[1])-1==plan['candidate_prefix_horizon']
    assert len(boundary)-1==plan['physical_pair_cap']
    sources=[planpath,candidatepath,developmentpath,riskpath,baselinepath,Path(__file__).resolve(),
             ROOT/'experiments/sota_round5/interval_hybrid_stopping.py',
             ROOT/'experiments/sota_round5/interval_randomized_stopping.py',
             ROOT/'experiments/sota_round4/interval_sign_cost.py']
    binding={'status':'bound_without_candidate_revision','created_utc':datetime.now(timezone.utc).isoformat(),
             'independent_plan_was_written_before_candidate_existed':True,
             'candidate_strict_internal_null_00999_pass':actual_risk<=F(999,100000),
             'candidate_original_one_percent_null_pass':True,
             'source_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sources}}
    with (DIRECTORY/'independent_grid_binding.json').open('x',encoding='utf-8') as f:
        json.dump(binding,f,indent=2)
    direct={}; rows=[]
    cases=sorted(plan['rows'],key=lambda row:(row['correct_sign']<0,row['case_id']))
    for case in cases:
        q,r=F(case['q_exact']),F(case['r_exact'])
        if case['correct_sign']<0:
            original=direct[(1-q,r)]
            value=dict(original)
            value['plus'],value['minus']=original['minus'],original['plus']
            value['exact_q']=str(q)
            derived=True
        else:
            value=operating_rq(policy,boundary,r,q,plan['physical_pair_cap'])
            direct[(q,r)]=value
            derived=False
            if len(direct)%4==0:
                print(json.dumps({'direct_cells':len(direct),'target':28,'seconds':time.perf_counter()-started}),flush=True)
        correct=(value['plus'] if case['correct_sign']>0 else value['minus']) if case['kind']=='main' else None
        rows.append(dict(case,**{k:value[k] for k in ('plus','minus','abstain','expected_shots')},
                         correct_probability=correct,derived_by_exact_sign_complement=derived))
    mainrows=[r for r in rows if r['kind']=='main']
    failed=[r['case_id'] for r in mainrows if F(r['correct_probability'][0])<F(19,20)]
    result={'status':'certified_independent_parameter_grid_operating_characteristics',
            'candidate_sha256':sha(candidatepath),'new_simulation_observations':0,
            'directly_evaluated_cells':len(direct),'mirrors_proved_by_sign_complement':24,
            'main_cases':len(mainrows),'main_cases_below_95_percent_certified_power':failed,
            'all_main_correct_probability_at_least_95_percent':not failed,
            'minimum_main_correct_probability_interval':[min(r['correct_probability'][i] for r in mainrows) for i in (0,1)],
            'uniform_null_risk_at_most_one_percent':True,
            'stricter_internal_preregistration_null_target_pass':actual_risk<=F(999,100000),
            'rows':rows,'source_sha256':binding['source_sha256'],'elapsed_seconds':time.perf_counter()-started,
            'mirror_proof':'Prefix stopping depends on min(k,d-k) and reports the current majority. The fallback is symmetric. Complementing every sign and leaving all uniform coins/waiting times unchanged swaps report directions without changing any cost, abstention, or correct probability. Registered q->1-q and unchanged r are exact Fractions.',
            'scope':'First frozen candidate evaluated without revision on the predeclared separate parameter grid. This is a deterministic validation of these exact models, not a new physical circuit/hardware experiment, not broad SOTA validation, and not a reversal of the stricter internal preregistration failure.'}
    with target.open('x',encoding='utf-8') as f: json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('status','minimum_main_correct_probability_interval','main_cases_below_95_percent_certified_power','stricter_internal_preregistration_null_target_pass','elapsed_seconds')},indent=2))


if __name__=='__main__': main()
