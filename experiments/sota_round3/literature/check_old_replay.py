"""Independent deterministic review of old-path censoring and paired costs."""
from collections import defaultdict, Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OLD=ROOT/'results/sota_round2/benchmark'
DIR=ROOT/'results/sota_round3/replay_v1'
OUT=ROOT/'results/sota_round3/literature/old_replay_review.json'

def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def decide(counts,table):
    n,k=counts[1]+counts[2],counts[1]
    if table is None:
        numerator=math.comb(2*k,k)*math.comb(2*(n-k),n-k)
        denominator=2**n*math.comb(n,k)
        hit=2*k!=n and numerator>=100*denominator
        return ('reject' if 2*k>n else 'accept') if hit else 'abstain'
    b=table[n]
    return 'reject' if k>=b else 'accept' if k<=n-b else 'abstain'

def main():
    inputs={name:sha(ROOT/name) for name in ['experiments/sota_round3/replay_boundaries.py',
        'experiments/sota_round3/literature/check_old_replay.py','results/sota_round3/replay_v1/summary.json',
        'results/sota_round3/replay_v1/records.jsonl','results/sota_round2/benchmark/paths.jsonl',
        'results/sota_round2/benchmark/preliminary_plan.json','results/sota_round2/benchmark_audit/audit.json']}
    prior_audit=read(ROOT/'results/sota_round2/benchmark_audit/audit.json')
    assert prior_audit['status']=='passed' and prior_audit['inputs']['paths.jsonl']==sha(OLD/'paths.jsonl')
    plan=read(OLD/'preliminary_plan.json');cases={c['case_id']:c for c in plan['cases']}
    tables={'beta_exact':read(ROOT/'results/sota_round3/beta_boundary_N2048.json')['boundaries'],'beta_ville':None}
    for path in sorted((ROOT/'results/sota_round3/boundaries').glob('*.json')):
        tables[path.stem]=read(path)['boundaries']
    records=[json.loads(line) for line in (DIR/'records.jsonl').open()]
    indexed={(r['case_id'],r['rep'],r['path'],r['method']):r for r in records}
    expected_keys={(case_id,rep,path,method) for case_id in cases for rep in range(4)
        for path in ('closed_hedge_original','closed_hedge_fine_initial') for method in tables}
    assert len(records)==len(indexed)==720 and set(indexed)==expected_keys
    paths=[json.loads(line) for line in (OLD/'paths.jsonl').open() if '"path":"closed_hedge_' in line]
    assert len(paths)==80
    statuses=Counter()
    for path in paths:
        counts=[0]*4;first={}
        schedule=plan['hedge_original_checkpoints' if path['path']=='closed_hedge_original' else 'hedge_fine_checkpoints']
        for j,block in enumerate(path['history']):
            assert block['budget']==schedule[j]
            counts=[a+b for a,b in zip(counts,block['pair_counts'])]
            for name,table in tables.items():
                if name in first:continue
                decision=decide(counts,table)
                if decision!='abstain' or block['budget']==4096:
                    first[name]=(block['budget'],decision,j+1)
        for name in tables:
            row=indexed[path['case_id'],path['rep'],path['path'],name]
            assert row['reference_shots']==path['outcomes']['hedge']['shots']
            assert row['reference_decision']==path['outcomes']['hedge']['decision']
            if name in first:
                budget,decision,prefix=first[name]
                assert row['decision']==decision and row['history_prefix_length']==prefix
                assert row['shots_lower']==row['shots_upper']==budget
                assert row['status']==('observed_cap_abstention' if decision=='abstain' else 'observed_stop')
                assert row['certificate']['decision']==decision and row['certificate']['shots']==budget
            else:
                assert path['shots_observed']<4096 and row['decision'] is None
                assert row['status']=='right_censored_at_reference_stop'
                assert row['shots_lower']==next(n for n in schedule if n>path['shots_observed'])
                assert row['shots_upper']==4096 and row['last_observed_shots']==path['shots_observed']
            statuses[row['status']]+=1
    summary=read(DIR/'summary.json')
    for name,h in summary['input_sha256'].items():assert sha(ROOT/name)==h
    for outcome in summary['outcomes']:
        local=[r for r in records if (r['path'],r['method'],r['kind'])==(outcome['path'],outcome['method'],outcome['kind'])]
        assert outcome['trajectories']==len(local)
        counts=Counter()
        groups=defaultdict(list)
        for r in local:
            groups[r['case_id']].append(r)
            d=r['decision'];truth='reject' if cases[r['case_id']]['C']>0 else 'accept'
            counts['right_censored' if d is None else 'observed_cap_abstentions' if d=='abstain'
                else 'observed_correct_stops' if r['kind']=='main' and d==truth else 'observed_wrong_stops']+=1
        for key in ('right_censored','observed_cap_abstentions','observed_correct_stops','observed_wrong_stops'):
            assert outcome[key]==counts[key]
        for endpoint in ('lower','upper'):
            value=math.exp(sum(math.log(sum(r['shots_'+endpoint] for r in group)/sum(r['reference_shots'] for r in group))
                for group in groups.values())/len(groups))
            assert abs(outcome['geometric_cost_ratio_'+endpoint]-value)<1e-14
    selected=[r for r in records if r['path']=='closed_hedge_fine_initial' and r['kind']=='main' and r['method']=='beta_exact']
    for peer in ('pair_grid','pairwise_hedge'):
        assert all((r['shots_lower'],r['shots_upper'],r['decision'],r['status'])==
            tuple(indexed[r['case_id'],r['rep'],r['path'],peer][key] for key in ('shots_lower','shots_upper','decision','status'))
            for r in selected)
    focus=next(r for r in summary['outcomes'] if (r['path'],r['method'],r['kind'])==('closed_hedge_fine_initial','beta_exact','main'))
    assert focus['geometric_cost_ratio_lower']==focus['geometric_cost_ratio_upper']==.9371254597117815
    assert focus['observed_correct_stops']==29 and focus['observed_cap_abstentions']==3 and focus['right_censored']==0
    result={'status':'passed','inputs':inputs,'path_records':80,'method_records':720,'statuses':dict(statuses),
        'focus':focus,'pair_grid_and_pairwise_hedge_same_observed_costs_and_decisions_on_32_fine_main':True,
        'new_simulation_observations':0,'independent_reconstruction':'All available checkpoint counts, rational Ville-Beta decisions, frozen integer table decisions, first crossings, censor intervals and summary arithmetic; original RNG ledger trusted through passed round2 full audit.',
        'limits':['Only the old reference schedules were inspected; this is not every-discordance stopping or an unobserved future continuation.',
                  'Right-censoring cost intervals are logical bounds, not observed cap abstentions or confidence intervals.',
                  'The 6.287% geometric cost reduction is a development comparison; beta_exact ties pair_grid and pairwise_hedge on these32 observations.',
                  'Same-path comparisons use each reference path separately. Fine and original reference paths have different old RNG batch histories.',
                  'The fine-path null error remains an observed wrong stop and is not excluded.'],
        'frozen_boundary_sources':{name:h for name,h in summary['input_sha256'].items() if 'boundar' in name}}
    assert not OUT.exists();OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'passed','path':str(OUT),'sha256':sha(OUT),'focus_ratio':focus['geometric_cost_ratio_lower'],'statuses':dict(statuses)}))

if __name__=='__main__':main()
