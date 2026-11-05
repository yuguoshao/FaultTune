"""Independent ledger audit of the released shared paired development pilot.

Integer stopping tables, all first crossing prefixes, conditional-Fisher state,
RNG calls and outcome summaries are reconstructed here. Reviewed interval and
conditional-CGF primitives are reused and named. Every case's reps0/31 receive
a separately counted ORIGINAL-RUNNER replay; this is not another independent
controller implementation. No fresh seeds or new observations are generated.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import gammaln, logsumexp

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/structured_validation'))
import paired_benchmark as runner
import audit_refined as old_audit
import weighted_fisher as fisher
from direct_tests import _Bounds, _log_marginal, _float_down, _float_up, _as_float_bounds

DIRECTORY=ROOT/'results/sota_round3/paired_benchmark'
OUTPUT=ROOT/'results/sota_round3/paired_benchmark_audit'
GENERIC=['maxsprt','pair_grid','pairwise_hedge','sprt_0.6','sprt_0.75','sprt_0.9','sprt_0.99']
METHODS=['beta_ville100','beta_exact']+GENERIC+['closed_hedge_original','closed_hedge_fine_initial']
COUNTS=Counter()
ERRORS={}


def require(condition,message):
    if not condition:raise AssertionError(message)


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):result.update(part)
    return result.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def close(a,b,name,tolerance=2e-9):
    require(math.isfinite(a) and math.isfinite(b),'Nonfinite '+name)
    error=abs(a-b);ERRORS[name]=max(ERRORS.get(name,0.),error)
    require(error<=tolerance,name+' differs: '+str(error))


def check_binding(binding):
    path=ROOT/binding['path']
    require(path.resolve().relative_to(ROOT).as_posix()==binding['path'],'Noncanonical file binding')
    require(digest(path)==binding['sha256'],'Changed file: '+binding['path'])
    return path


def schedules():
    coarse=[];n=32
    while n<4096:
        coarse.append(n);n=2*int(math.ceil(n*1.125/2))
    coarse=sorted(set(coarse+[4096]))
    return coarse,sorted(set(range(2,33,2))|{n for n in coarse if n>32}|{4096})


def inspect_plan():
    plan=read(DIRECTORY/'preliminary_plan.json')
    require(plan['version']=='paired_shared_development_v1' and plan['status']=='preliminary_requires_explicit_root_release','Plan type changed')
    require((plan['cap'],plan['delta'],plan['repetitions'],plan['seed_base'])==(4096,.01,32,330411000),'Fixed budget changed')
    require(plan['methods']==METHODS and plan['fixed_positive_arm']=={'cos_numerator':0,'sin_numerator':2,'denominator':2},'Method/arm registry differs')
    coarse,fine=schedules()
    require(plan['fine_checkpoints']==fine and plan['coarse_checkpoints']==coarse,'Checkpoint schedule changed')
    require((plan['expected_path_records'],plan['expected_method_records'],plan['max_physically_generated_shots'])==(576,6336,2359296),'Matrix size changed')
    prior=read(check_binding(plan['previous_plan']));lookup={c['case_id']:c for c in prior['cases']}
    expected=[]
    for mirror in (False,True):
        for identifier in (7,9,19,21,31,33,43,45):
            source=lookup[identifier]
            expected.append({'case_id':identifier+(1000 if mirror else 0),'kind':'main','A':source['A'],'B':source['B'],
                'C':-source['C'] if mirror else source['C'],'source_case_id':identifier,'mirror':mirror,
                'rho':source['rho'],'B_over_R':source['B_over_R']})
    expected.extend({name:lookup[i][name] for name in ('case_id','kind','A','B','C')} for i in (100,101))
    require(plan['cases']==expected,'Cases/mirrors changed')
    review=read(check_binding(plan['boundary_review']))
    require(review['status']=='passed','Boundary source review did not pass')
    for name,h in review['sources'].items():require(digest(ROOT/name)==h,'Reviewed boundary source changed: '+name)
    return plan


def independent_table_risk(table):
    require(len(table)==2049 and table[0]==1 and type(table[0]) is int,'Wrong table horizon/zero state')
    survivors=[1];absorbed=0;first_crossing=[]
    for n,bound in enumerate(table[1:],1):
        require(type(bound) is int and n//2<bound<=n+1,'Invalid monotone upper/lower tails')
        fresh=[0]*(n+1)
        for k,value in enumerate(survivors):
            fresh[k]+=value;fresh[k+1]+=value
        killed=0
        for k in range(n+1):
            if k>=bound or k<=n-bound:
                killed+=fresh[k];fresh[k]=0
        first_crossing.append(killed)
        absorbed=2*absorbed+killed
        survivors=fresh
        require(absorbed+sum(survivors)==1<<n,'Exact probability mass mismatch')
    return absorbed,1<<2048,survivors,first_crossing


def jeffreys_ge(n,k,threshold):
    numerator=math.comb(2*k,k)*math.comb(2*(n-k),n-k)
    denominator=(1<<n)*math.comb(n,k)
    return numerator*threshold.denominator>=denominator*threshold.numerator


def inspect_boundaries(plan):
    require(set(plan['boundaries'])=={'beta_exact',*GENERIC},'Missing/extra boundary artifacts')
    artifacts={};reports={}
    for name,binding in plan['boundaries'].items():
        artifact=read(check_binding(binding))
        require(artifact['artifact_sha256']==binding['artifact_sha256']==canonical_hash({k:v for k,v in artifact.items() if k!='artifact_sha256'}),'Boundary payload differs')
        require(artifact['max_discordances']==2048 and artifact['delta_exact']=={'numerator':'1','denominator':'100'},'Boundary risk/horizon differs')
        for source,h in artifact['source_hashes'].items():require(digest(ROOT/source)==h,'Boundary source changed')
        table=artifact['boundaries']
        numerator,denominator,survivors,crossings=independent_table_risk(table)
        saved=artifact['fair_null_risk']
        require(numerator==int(saved['crossing_numerator']) and denominator==int(saved['fair_denominator']) and numerator*100<=denominator,'Exact risk certificate differs/exceeds .01')
        if name=='beta_exact':
            require(artifact['version']=='exact_finite_discordance_jeffreys_v1' and artifact['status']=='exact_finite_horizon_boundary_certified','Wrong exact-beta artifact')
            core={key:artifact[key] for key in ('version','max_discordances','delta_exact','threshold_exact','boundaries')}
            require(canonical_hash(core)==artifact['boundary_sha256']==binding['boundary_sha256'],'Beta table binding differs')
            threshold=Fraction(int(artifact['threshold_exact']['numerator']),int(artifact['threshold_exact']['denominator']))
            require(threshold>0,'Nonpositive Beta threshold')
            # Closed-form rational capitals at both sides of every selected
            # integer tail establish the actual shape independently.
            for n,bound in enumerate(table[1:],1):
                if bound<=n:require(jeffreys_ge(n,bound,threshold),'Beta boundary fails exact rational crossing')
                lower=bound-1
                if 2*lower>n:require(not jeffreys_ge(n,lower,threshold),'Beta boundary missed earlier exact crossing')
            require(int(saved['surviving_total'])==sum(survivors),'Beta surviving mass differs')
            require(list(map(int,saved['first_crossing_counts']))==crossings,'Beta all first-crossing counts differ')
            low=saved['survivor_lower_plus'];kept=list(map(int,saved['survivor_counts']))
            require(survivors[low:low+len(kept)]==kept and not any(survivors[:low]+survivors[low+len(kept):]),'Beta final surviving states differ')
        else:
            require(artifact['version']=='finite_integer_tail_boundary_v1' and artifact['status']=='integer_boundary_exact_risk_certified' and artifact['shape']==name,'Generic boundary identity differs')
        artifacts[name]=artifact
        reports[name]={'path':binding['path'],'sha256':binding['sha256'],'artifact_sha256':artifact['artifact_sha256'],
            'exact_fair_crossing_numerator':str(numerator),'exact_fair_denominator':str(denominator),
            'probability_float_diagnostic':numerator/denominator,'independent_integer_path_count_passed':True}
    return artifacts,reports


def inspect_release(plan,complete):
    spec=read(DIRECTORY/'spec.json');release=read(DIRECTORY/'release.json')
    require(spec['plan']==plan and check_binding(spec['plan_reference'])==DIRECTORY/'preliminary_plan.json','Spec/plan mismatch')
    require(check_binding(spec['release_reference'])==DIRECTORY/'release.json','Spec/release mismatch')
    require(release['status']=='released_by_root' and release['explicit_root_release'] is True and release['reference'].strip(),'Root release absent')
    require(release['plan_sha256']==digest(DIRECTORY/'preliminary_plan.json') and release['boundaries']==plan['boundaries'],'Released plan/boundaries differ')
    require(spec['sources']==release['sources'],'Spec source map differs')
    for name,h in release['sources'].items():require(digest(ROOT/name)==h,'Released source changed: '+name)
    review=read(check_binding(release['review']))
    require(review['status']=='passed' and review['plan_sha256']==release['plan_sha256'] and review['sources']==release['sources'] and review['boundaries']==plan['boundaries'],'Runner review does not cover released protocol')
    require(all(type(release[key]) in (int,float) and math.isfinite(release[key]) for key in ('sampling_cutoff_unix','hard_deadline_unix')) and release['sampling_cutoff_unix']<=release['hard_deadline_unix'],'Invalid release deadlines')
    if complete:
        summary=read(DIRECTORY/'summary.json')
        require(summary['status']=='complete_descriptive_development_comparison','Producer incomplete')
        require(summary['input_sha256']=={name:digest(DIRECTORY/name) for name in ('spec.json','paths.jsonl','method_results.jsonl')},'Summary input hashes differ')
    return spec,release


def table_certificate(counts,name,artifact):
    n,k=counts[1]+counts[2],counts[1]
    b=artifact['boundaries'][n]
    decision='reject' if k>=b else 'accept' if k<=n-b else 'abstain'
    if name=='beta_exact':
        return {'method':'exact_finite_discordance_jeffreys','decision':decision,'shots':2*sum(counts),'pairs':sum(counts),
            'discordant_pairs':n,'discordances':n,'score':counts[1]-counts[2],'counts':counts.copy(),
            'delta_exact':artifact['delta_exact'],'threshold_exact':artifact['threshold_exact'],
            'integer_boundary':{'plus_count':k,'upper':b,'lower':n-b},
            'boundary_artifact_sha256':artifact['artifact_sha256'],'boundary_sha256':artifact['boundary_sha256'],
            'max_discordances':2048,'valid_up_to_registered_discordance_horizon':True,
            'numeric_status':'Decision uses integer counts and the exact certified boundary table',
            'scope':'Predictable paired queries with common sign ordering; finite discordance horizon only'}
    return {'decision':decision,'shots':2*sum(counts),'discordances':n,'score':counts[1]-counts[2],
            'integer_boundary':b,'boundary_sha256':artifact['artifact_sha256'],'shape':name,'finite_horizon_only':True}


def beta_certificate(counts):
    pairs=sum(counts);b,c=counts[1:3];discordant=b+c
    evidence=_log_marginal(b,discordant,.5)+_Bounds(2).log()*discordant
    threshold=-_Bounds(Decimal('.01')).log()
    decision=('reject' if b>c else 'accept') if b!=c and evidence.lo>=threshold.hi else 'abstain'
    return {'method':'paired_beta_mixture','decision':decision,'est_delta':(b-c)/pairs if pairs else 0.,
        'shots':2*pairs,'pairs':pairs,'discordant_pairs':discordant,'counts':counts.copy(),'delta':.01,'prior':.5,
        'evidence':_float_down(evidence.lo),'threshold':_float_up(threshold.hi),'evidence_bounds':_as_float_bounds(evidence),
        'threshold_bounds':_as_float_bounds(threshold),'anytime_valid':True,'numeric_status':'decision uses outward Decimal log bounds'}


class FisherReconstruction:
    def __init__(self):
        self.experts=fisher.protocol_config()['experts']
        require(sum((Fraction(*e['weight']) for e in self.experts),Fraction(0))==1,'Fisher prior not normalized')
        self.cgf=np.zeros(len(self.experts));self.score=Fraction(0);self.blocks=[];self.shots=0
        self.log_odds=np.array([math.log(e['numerator']/e['denominator']) for e in self.experts])
        self.log_weights=np.array([math.log(float(Fraction(*e['weight']))) for e in self.experts])
    def update(self,counts):
        n=sum(counts);minus=counts[2]+counts[3];plus=counts[1]+counts[3]
        block={'nminus':n,'nplus':n,'minus_successes':minus,'plus_successes':plus}
        self.blocks.append(block);self.shots+=2*n;self.score+=Fraction(plus-minus,2)
        for j,e in enumerate(self.experts):self.cgf[j]+=fisher._cgf(block,e['numerator'],e['denominator'])
    def screen(self):return float(logsumexp(float(self.score)*self.log_odds-self.cgf+self.log_weights))
    def certificate(self,counts):
        return old_audit.audit_certificate('hedge',np.array(counts,dtype=np.int64),{'blocks':self.blocks},None,None,
            {'delta':.01,'conditional_protocol':fisher.protocol_config()})


def inspect_path(record,case,rep,plan,artifacts):
    require((record['case_id'],record['kind'],record['rep'])==(case['case_id'],case['kind'],rep),'Path identity/order differs')
    seed=330411000+1009*case['case_id']+1000003*rep
    require(record['seed']==seed and record['cap']==4096 and record['delta']==.01 and record['truth_access']=='simulator_only','Path seed/budget differs')
    require(record['status']=='complete' and record['error'] is None,'Incomplete producer path')
    require(all(math.isfinite(v) and v>=0 for v in record['timings'].values()),'Invalid timings')
    require(len(record['history'])==len(record['checkpoint_history'])>0,'History/checkpoint count mismatch')
    rng=np.random.RandomState(seed);pm,pp=case['A']-case['C'],case['A']+case['C']
    fine,coarse=FisherReconstruction(),FisherReconstruction()
    counts=[0]*4;initial=[0]*4;stopped={};last=0
    for index,(block,checkpoint) in enumerate(zip(record['history'],record['checkpoint_history'])):
        budget=plan['fine_checkpoints'][index];number=(budget-last)//2
        require(all(type(block[key]) is int for key in ('look','budget','m','minus_successes','plus_successes')) and
            block['look']==index+1 and block['budget']==budget and block['m']==number and block['arm']==plan['fixed_positive_arm'],'Invalid paid block/action')
        fresh=block['pair_counts']
        require(len(fresh)==4 and all(type(v) is int and v>=0 for v in fresh) and sum(fresh)==number,'Invalid observed categories')
        minus=int(rng.binomial(number,pm));both=int(rng.binomial(minus,pp));plus_only=int(rng.binomial(number-minus,pp))
        require(fresh==[number-minus-plus_only,plus_only,minus-both,both] and block['minus_successes']==minus and block['plus_successes']==plus_only+both,'Same-seed three-binomial observation differs')
        COUNTS['binomial_rng_calls_reconstructed']+=3
        counts=[a+b for a,b in zip(counts,fresh)];fine.update(fresh)
        if budget<=32:
            initial=[a+b for a,b in zip(initial,fresh)]
            if budget==32:coarse.update(initial)
        else:coarse.update(fresh)
        require(fine.shots==budget and coarse.shots==(budget if budget>=32 else 0),'Independent coarse/fine paid-state mismatch')
        require(type(checkpoint['shots']) is int and all(type(v) is int for v in checkpoint['counts']) and
            checkpoint['shots']==budget==2*sum(counts) and checkpoint['counts']==counts,'Checkpoint counts differ')
        require(checkpoint['fine_blocks']==len(fine.blocks) and checkpoint['original_blocks']==len(coarse.blocks),'Fisher block aggregation differs')
        active=[method for method in METHODS if method not in stopped]
        require(set(checkpoint['methods'])==set(active),'Missing/extra active method')
        b,c=counts[1:3]
        beta=float(gammaln(b+.5)+gammaln(c+.5)-gammaln(b+c+1)-2*gammaln(.5)+(b+c)*math.log(2))
        for method in active:
            detail=checkpoint['methods'][method];cert=None
            if method in artifacts:
                cert=table_certificate(counts,method,artifacts[method])
                require(detail=={'certificate':cert},'Independent integer-table certificate differs: '+method)
                COUNTS['integer_boundary_prefix_certificates_recomputed']+=1
            elif method=='closed_hedge_original' and budget<32:
                require(detail=={'eligible':False,'reason':'First original physical checkpoint is 32'},'Original hedge early eligibility differs')
                COUNTS['coarse_initial_ineligible_checks']+=1
            else:
                screen=beta if method=='beta_ville100' else float(np.logaddexp(beta,(coarse if method=='closed_hedge_original' else fine).screen())-math.log(2))
                require(detail['eligible'] is True,'Unexpected ineligible continuous evidence')
                close(detail['log_screen_float'],screen,method+'_screen')
                eligible=screen>=math.log(100)-1e-6 or budget==4096
                require(('certificate' in detail)==eligible,'Evidence certificate eligibility differs')
                if eligible:
                    cert=beta_certificate(counts) if method=='beta_ville100' else (coarse if method=='closed_hedge_original' else fine).certificate(counts)
                    require(cert==detail['certificate'],'Outward terminal/checkpoint certificate differs: '+method)
                    COUNTS['outward_certificates_recomputed']+=1
                COUNTS['continuous_evidence_screens_checked']+=1
            if cert is not None and (cert['decision']!='abstain' or budget==4096):
                stopped[method]={'decision':cert['decision'],'shots':budget,'counts':counts.copy(),
                    'history_prefix_length':index+1,'certificate':cert}
        require(len(stopped)<11 or index==len(record['history'])-1,'Path continued after all methods stopped')
        last=budget;COUNTS['paid_blocks_reconstructed']+=1
    require(set(stopped)==set(METHODS) and record['outcomes']==stopped,'First-stop/cap outcomes differ')
    require(type(record['shots_observed']) is int and record['shots_observed']==last==2*sum(counts) and record['counts']==counts and
        all(type(v) is int for v in record['counts']),'Terminal physical counts differ')
    rows=[]
    for method,outcome in record['outcomes'].items():
        require(type(outcome['shots']) is int and type(outcome['history_prefix_length']) is int and
            (outcome['decision']!='abstain' or outcome['shots']==4096),'Invalid stopped prefix/cap')
        rows.append(dict(case_id=case['case_id'],kind=case['kind'],rep=rep,seed=seed,method=method,**outcome))
    COUNTS['path_records_checked']+=1;COUNTS['method_records_checked']+=len(rows);COUNTS['terminal_certificates_checked']+=len(rows)
    COUNTS['physically_generated_shots_accounted']+=record['shots_observed']
    return rows


def inspect_summary(rows,plan):
    summary=read(DIRECTORY/'summary.json')
    keyed={(r['case_id'],r['rep'],r['method']):r for r in rows}
    expected={(c['case_id'],rep,m) for c in plan['cases'] for rep in range(32) for m in METHODS}
    require(len(rows)==len(keyed)==6336 and set(keyed)==expected,'Incomplete/duplicate method matrix')
    cells=[];means={}
    for case in plan['cases']:
        truth='reject' if case['C']>0 else 'accept' if case['C']<0 else 'abstain'
        for method in METHODS:
            local=[keyed[case['case_id'],rep,method] for rep in range(32)]
            mean=sum(r['shots'] for r in local)/32;means[case['case_id'],method]=mean
            cells.append({'case_id':case['case_id'],'kind':case['kind'],'sign':1 if case['C']>0 else -1 if case['C']<0 else 0,
                'method':method,'trajectories':32,'mean_paid_shots':mean,'wrong':sum(r['decision'] not in ('abstain',truth) for r in local),
                'correct_stops':sum(r['decision']==truth and truth!='abstain' for r in local),'abstained':sum(r['decision']=='abstain' for r in local)})
    require(cells==summary['cells'],'Per-case unconditional cost/error/abstention cells differ')
    outcomes=[]
    for group in ('main','positive','negative','null'):
        for method in METHODS:
            local=[c for c in cells if c['method']==method and (c['kind']==group or group=='positive' and c['sign']==1 or group=='negative' and c['sign']==-1)]
            outcomes.append({'group':group,'method':method,'trajectories':sum(c['trajectories'] for c in local),
                'mean_paid_shots':sum(c['mean_paid_shots'] for c in local)/len(local),
                **{name:sum(c[name] for c in local) for name in ('wrong','correct_stops','abstained')}})
    require(outcomes==summary['outcomes'],'Outcome groups differ')
    main=[c['case_id'] for c in plan['cases'] if c['kind']=='main']
    ratios={m:{ref:math.exp(sum(math.log(means[i,m]/means[i,ref]) for i in main)/16) for ref in METHODS} for m in METHODS}
    require(ratios==summary['equal_case_geometric_cost_ratios'],'Equal-case geometric ratios differ')
    require(summary['path_records']==COUNTS['path_records_checked']==576 and summary['method_records']==COUNTS['method_records_checked']==6336 and
        summary['physically_generated_shots']==COUNTS['physically_generated_shots_accounted'],'Summary ledger totals differ')
    return {'outcomes':outcomes,'equal_case_geometric_cost_ratios':ratios,'all_errors_and_cap_abstentions_in_cost':True,
        'confidence_intervals_computed':False}


def audit(boundaries_only=False):
    started=time.perf_counter();plan=inspect_plan();artifacts,boundary_report=inspect_boundaries(plan)
    if boundaries_only:
        print(json.dumps({'status':'independent_boundary_tables_passed','boundaries':len(boundary_report),'elapsed_seconds':time.perf_counter()-started}),flush=True)
        return
    spec,release=inspect_release(plan,complete=True)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    for name in ('path_audit.jsonl','replays.jsonl','audit.json','failed.json'):
        require(not (OUTPUT/name).exists(),'Preserve existing auditor output: '+name)
    inputs={name:digest(DIRECTORY/name) for name in ('preliminary_plan.json','release.json','spec.json','paths.jsonl','method_results.jsonl','summary.json')}
    own_sources={name:digest(ROOT/name) for name in ('experiments/sota_round3/audit_paired_benchmark.py',
        'experiments/structured_validation/audit_refined.py','experiments/structured_validation/audit_conditional.py')}
    def deadline():require(time.time()<release['hard_deadline_unix'],'Released replay hard deadline reached')
    # Same frozen providers are needed only for the separately declared
    # original-runner replay. Independent path checks use raw integer tables.
    runtime_boundaries=runner.load_boundaries(plan)
    all_rows=[];replays=0;current=None
    try:
        with (DIRECTORY/'paths.jsonl').open(encoding='utf-8') as paths, (DIRECTORY/'method_results.jsonl').open(encoding='utf-8') as methods, \
             (OUTPUT/'path_audit.jsonl').open('x',encoding='utf-8') as path_log, (OUTPUT/'replays.jsonl').open('x',encoding='utf-8') as replay_log:
            for case in plan['cases']:
                for rep in range(32):
                    current=(case['case_id'],rep);deadline();record=json.loads(next(paths));before=Counter(COUNTS)
                    rows=inspect_path(record,case,rep,plan,artifacts)
                    for row in rows:require(json.loads(next(methods))==row,'Method ledger differs from independently reconstructed prefix')
                    all_rows.extend(rows)
                    path_log.write(json.dumps({'status':'passed','case_id':case['case_id'],'rep':rep,'checks':dict(COUNTS-before)},separators=(',',':'))+'\n');path_log.flush()
                    if rep in (0,31):
                        replayed=runner.run_case(case,rep,plan,runtime_boundaries,deadline)
                        original={k:v for k,v in record.items() if k!='timings'};repeated={k:v for k,v in replayed.items() if k!='timings'}
                        require(original==repeated,'Original-runner fixed-seed replay differs: '+str(current))
                        replays+=1;replay_log.write(json.dumps({'status':'exact_original_runner_seed_replay_passed','case_id':case['case_id'],'rep':rep,
                            'nontiming_protocol_sha256':canonical_hash(original),'new_simulation_observations':0},separators=(',',':'))+'\n');replay_log.flush()
                    if COUNTS['path_records_checked']%16==0:
                        print(json.dumps({'status':'auditing','paths':COUNTS['path_records_checked'],'methods':COUNTS['method_records_checked'],
                            'original_runner_replays':replays,'current':current,'elapsed_seconds':time.perf_counter()-started}),flush=True)
            require(next(paths,None) is None and next(methods,None) is None,'Extra ledger lines')
        summary_report=inspect_summary(all_rows,plan)
        require(replays==36 and COUNTS['terminal_certificates_checked']==6336,'Incomplete required replay/terminal coverage')
        require(inputs=={name:digest(DIRECTORY/name) for name in inputs},'Producer inputs changed during audit')
        require(own_sources=={name:digest(ROOT/name) for name in own_sources},'Auditor source changed during audit')
        inspect_release(plan,complete=True)
        for binding in plan['boundaries'].values():check_binding(binding)
        result={'status':'passed','created_utc':datetime.now(timezone.utc).isoformat(),'inputs':inputs,'sources':spec['sources'],
            'audit_sources':own_sources,'counts':dict(COUNTS),'maximum_screen_float_differences':ERRORS,'independent_boundary_risk_checks':boundary_report,
            'original_runner_seed_replay':{'selection':'Every case rep0 and rep31, mechanically specified before audit','paths':replays,
                'all_nontiming_fields_exact':True,'independent_implementation':False,'new_simulation_observations':0},
            'summary_reconstruction':summary_report,'elapsed_seconds':time.perf_counter()-started,
            'limitations':['Descriptive development pilot only; audit passing does not prove SOTA superiority or empirically certify1% risk.',
                'Frozen integer tables and fair path counts are exact. Generic floating score shapes are not certified real-arithmetic GLR level sets.',
                'Independent audit reconstructs conditional-Fisher state and all first eligible stops, but reuses reviewed conditional-CGF and outward interval primitives.',
                'Original-runner replay is separately counted reproducibility evidence, not independent implementation.',
                'Fine and coarse hedges share actual fine-block observations; coarse combines initial32 shots and has the original distribution but not the old runner RNG call history.',
                'Finite-discordance guarantees end at2048; the physical4096-shot cap is enforced separately.',
                'Wrong outcomes and cap abstentions are included in every reported unconditional cost.']}
        write_new(OUTPUT/'audit.json',result)
        print(json.dumps({'status':'passed','audit':str(OUTPUT/'audit.json'),'sha256':digest(OUTPUT/'audit.json'),'elapsed_seconds':result['elapsed_seconds']}),flush=True)
    except BaseException as caught:
        write_new(OUTPUT/'failed.json',{'status':'failed','current_key':current,'exception':type(caught).__name__+': '+str(caught),
            'inputs':inputs,'audit_sources':own_sources,'counts':dict(COUNTS),'original_runner_replays':replays,'elapsed_seconds':time.perf_counter()-started})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--boundaries-only',action='store_true',help='Pure deterministic table/probability checks; no data replay')
    audit(boundaries_only=parser.parse_args().boundaries_only)
