"""Independent deterministic review of the paid oracle and frozen sign policy."""
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import mpmath as mp

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiments/sota_round5'))
import dkm_oracle_certificate as oracle
import optimal_sign_policy as runtime
mp.mp.dps=120


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fraction(x):
    return Fraction(int(x['numerator']),int(x['denominator']))


def ratio(x):
    return Fraction(int(x[0]),int(x[1]))


def mm(x):
    x=Fraction(x)
    return mp.mpf(x.numerator)/x.denominator


def check_interval(pair,value):
    assert mp.mpf(pair[0])<=value<=mp.mpf(pair[1])


def write_new(name,record):
    path=Path(__file__).with_name(name)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(record,stream,indent=2,allow_nan=False)
        stream.write('\n')
    return {'path':str(path),'sha256':sha(path)}


def oracle_review():
    checks=ROOT/'results/sota_round5/literature/dkm_oracle_component_checks.json'
    saved=json.loads(checks.read_text(encoding='utf-8'))
    assert saved['status']=='passed'
    for path,digest in saved['source_sha256'].items():
        assert sha(ROOT/path)==digest
    tested=[]
    eta=Fraction(1,1000000)
    for successes in ((0,7),(5,0),(1,2),(5,7),(3,4),(2,3)):
        for sign in (-1,1):
            for weights in ((1,1),(0,1),(3,1)):
                cert=oracle.exact_psr_best_response_from_paid_counts([5,7],list(successes),list(weights),sign)
                mu=[max(eta,min(1-eta,Fraction(k,n))) for n,k in zip((5,7),successes)]
                w=[Fraction(x,sum(weights)) for x in weights]
                q=mu if sign*(mu[1]-mu[0])<=0 else [sum(x*y for x,y in zip(w,mu))]*2
                assert [ratio(x) for x in cert['predictions_exact']]==q
                reference=sum(mm(ww)*(mm(p)*mp.log(mm(p)/mm(qq))+(1-mm(p))*mp.log((1-mm(p))/(1-mm(qq)))) for ww,p,qq in zip(w,mu,q))
                check_interval(cert['objective_bounds'],reference)
                check_interval(cert['optimum_bounds'],reference)
                assert cert['status']=='certified_approximate_best_response'
                assert Decimal(cert['convexity_gap_bounds'][1])<Decimal('1e-40')
                tested.append({'successes':list(successes),'weights':list(weights),'sign':sign})
    # Check the generic cone certificate at rational feasible, generally
    # nonoptimal points. Reconstruct the rational gradient without its helper.
    arms=oracle.old.default_arms()
    generic_cases=[]
    for sign,point in ((1,[Fraction(1,3),Fraction(1,10),Fraction(-1,20)]),
                       (-1,[Fraction(2,3),Fraction(-1,9),Fraction(1,25)])):
        cert=oracle.certificate_from_paid_counts(arms,[13,17,19,23,29,31],[2,4,3,8,7,12],
                                                 [1,2,3,4,5,6],sign,point)
        gradient=[Fraction(0)]*3
        value=mp.mpf(0)
        for arm,n,k,w in zip(arms,[13,17,19,23,29,31],[2,4,3,8,7,12],range(1,7)):
            features=[Fraction(1),Fraction(arm['cos_numerator'],arm['denominator']),
                      Fraction(arm['sin_numerator'],arm['denominator'])]
            p=max(eta,min(1-eta,Fraction(k,n)))
            q=eta+(1-2*eta)*sum(x*y for x,y in zip(features,point))
            ww=Fraction(w,21)
            value+=mm(ww)*(mm(p)*mp.log(mm(p)/mm(q))+(1-mm(p))*mp.log((1-mm(p))/(1-mm(q))))
            scalar=ww*(1-2*eta)*(q-p)/(q*(1-q))
            gradient=[g+scalar*x for g,x in zip(gradient,features)]
        assert [ratio(x) for x in cert['rational_gradient_exact']]==gradient
        h=[-mm(g) for g in gradient]
        radial=mp.sqrt(h[1]**2+max(mp.mpf(0),-sign*h[2])**2)
        support=max(mp.mpf(0),h[0],(h[0]+radial)/2)
        gap=sum(mm(g)*mm(x) for g,x in zip(gradient,point))+support
        check_interval(cert['objective_bounds'],value)
        check_interval(cert['convexity_gap_bounds'],gap)
        generic_cases.append({'sign':sign,'point_exact':[str(x) for x in point],
                              'gradient_exact_match':True,'mpmath120_objective_and_gap_inside':True})
    weak=saved['weak_signal_old_absolute_gate_counterexample']
    assert weak['status']=='unresolved_best_response_accuracy'
    assert Decimal(weak['convexity_gap_bounds'][1])<Decimal('1e-5')
    result={'status':'passed_independent_paid_oracle_component_review','new_simulation_observations':0,
            'independent_exact_psr_cases':len(tested),'generic_rational_point_checks':generic_cases,
            'mathematical_review':{
                'problem':'For exact clipped paid empirical means mu_i and normalized nonnegative allocation weights w_i, minimize F(u)=sum w_i kl(mu_i,eta+(1-2eta)X_i u) over the closed physical cone with sign*C<=0. Every registered arm requires paid initialization.',
                'gradient':'Each rational derivative is(1-2eta)w_i(q_i-mu_i)/(q_i(1-q_i)) times the exact rational feature row. This is the gradient with respect to unit-cone u, matching the support domain.',
                'support':'For h and sign s, sup_{u in opposite cone} h dot u=max(0,h_A,(h_A+sqrt(h_B^2+max(0,-s h_C)^2))/2). First maximize the half-disk at fixed A, then the resulting piecewise linear expression over A has candidates0,1/2,1. The frozen support_bounds encloses this formula.',
                'gap':'Convexity gives0<=F(u)-F*<=g dot u+support(-g). The stored F* interval uses lower max(0,F_lower-gap_upper) and upper F_upper. Both are conservative for an exactly feasible u.',
                'acceptance':'gap_upper<=lower(atol+rtol*F*_lower) implies the requested absolute-plus-relative accuracy against the TRUE optimum. A floating optimizer success flag alone is not used as certification.',
                'psr_pooling':'Two contracted PSR probabilities fill the whole square[eta,1-eta]^2 using B=0. If the clipped means violate the opposite ordering, the exact minimizer pools them at the weighted arithmetic mean; otherwise keep them. Zero weights may give nonunique minimizers but the construction remains optimal.',
                'repair':'Numerical proposals are converted to exact rationals and rechecked for unit-cone feasibility. Subsequent certification does not rely on the old floating feasibility/optimality diagnostic.',
                'limits':'This is an optimizer component on the declared contracted/clipped design problem. It does not establish a full DKM reproduction, stopping validity, an exact-oracle sample-complexity theorem, or a speedup.'},
            'source_sha256':{Path(__file__).resolve().relative_to(ROOT).as_posix():sha(__file__),
                             checks.relative_to(ROOT).as_posix():sha(checks),**saved['source_sha256']}}
    return write_new('dkm_oracle_component_review.json',result)


def integer_full_state(numerators,denominator,q):
    """No producer/runtime DP is called: common-denominator integer recurrence."""
    q=Fraction(q)
    a,b=q.numerator,q.denominator
    base=b*denominator
    live=[1]
    total_den=1
    positive=negative=cost=0
    for d in range(1,len(numerators)):
        cost=(cost+sum(live))*base
        positive*=base
        negative*=base
        incoming=[0]*(d+1)
        for k,mass in enumerate(live):
            incoming[k]+=(b-a)*mass
            incoming[k+1]+=a*mass
        fresh=[]
        for k,mass in enumerate(incoming):
            cut=numerators[d][min(k,d-k)]
            stopped=mass*cut
            if 2*k>d:
                positive+=stopped
            elif 2*k<d:
                negative+=stopped
            else:
                assert stopped==0
            fresh.append(mass*(denominator-cut))
        live=fresh
        total_den*=base
        assert positive+negative+sum(live)==total_den
    return {'positive':Fraction(positive,total_den),'negative':Fraction(negative,total_den),
            'abstain':Fraction(sum(live),total_den),'expected_discordances':Fraction(cost,total_den)}


def policy_review():
    artifact_path=runtime.ARTIFACT
    audit_path=runtime.AUDIT
    physical_plan=artifact_path.with_name('physical_evaluation_plan.json')
    value=json.loads(artifact_path.read_text(encoding='utf-8'))
    old_audit=json.loads(audit_path.read_text(encoding='utf-8'))
    assert value['status']=='exact_audited_classical_development_prototype'
    assert old_audit['artifact_sha256']==sha(artifact_path)
    assert old_audit['runtime_source_sha256']==sha(runtime.__file__)
    rows=value['protocol']['policy_numerators']
    den=value['protocol']['probability_denominator']
    assert den==10**12 and len(rows)==129
    for d,row in enumerate(rows):
        assert len(row)==d//2+1 and all(type(x) is int and 0<=x<=den for x in row)
        if d%2==0:
            assert row[d//2]==0
    fair=integer_full_state(rows,den,Fraction(1,2))
    risk=fair['positive']+fair['negative']
    assert fair['positive']==fair['negative']
    assert risk==fraction(value['results'][0]['exact_null_any_report'])
    assert risk==Fraction(int(old_audit['null_risk_numerator']),int(old_audit['null_risk_denominator']))
    assert risk<=Fraction(1,100)
    # Exact named design alternatives plus several adverse sign probabilities.
    q_records=[]
    for q in (Fraction(0),Fraction(1,10),Fraction(1,4),Fraction(49,100),Fraction(1,2),
              Fraction(7,10),Fraction(9,10),Fraction(99,100),Fraction(1)):
        got=fair if q==Fraction(1,2) else integer_full_state(rows,den,q)
        wrong=got['positive'] if q<Fraction(1,2) else got['negative'] if q>Fraction(1,2) else risk
        assert wrong<=(risk if q==Fraction(1,2) else risk/2)
        matching=[row for row in value['results'][0]['rows'] if Fraction(*row['q_exact'])==q]
        for row in matching:
            for name,answer in got.items():
                assert answer==fraction(row[name])
        q_records.append({'q_exact':str(q),'wrong_probability_float_diagnostic':float(wrong),
                          'prefix_lemma_bound_checked_exactly':True,
                          'saved_design_row_compared':bool(matching)})
    policy=runtime.load_policy()
    calls=0
    for d in range(1,129):
        for k in range(d+1):
            cut=rows[d][min(k,d-k)]
            # There are exactly cut accepted integers in[0,den); endpoint
            # checks establish matching runtime strict-inequality semantics.
            for coin in sorted({0,den-1,max(0,cut-1),min(den-1,cut)}):
                answer=runtime.decide(policy,d,k,coin)
                expected=('plus' if 2*k>d else 'minus') if coin<cut else ('abstain' if d==128 else 'continue')
                assert answer==expected
                assert runtime.decide(policy,d,d-k,coin)=={'plus':'minus','minus':'plus'}.get(answer,answer)
                calls+=1
    invalid=[(0,0,0),(129,0,0),(1,2,0),(1,0,-1),(1,0,den),(1,0,True)]
    for args in invalid:
        try:
            runtime.decide(policy,*args)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid runtime call accepted')
    result={'status':'passed_independent_uniform_iid_sign_policy_review','new_simulation_observations':0,
            'horizon':128,'probability_denominator':den,
            'independent_integer_full_state_risk_recomputed':True,
            'exact_null_any_report':{'numerator':str(risk.numerator),'denominator':str(risk.denominator)},
            'exact_null_any_report_float_diagnostic':float(risk),
            'uniform_strict_wrong_direction_upper_float_diagnostic':float(risk/2),
            'exact_null_within_one_percent':True,
            'exact_fixed_q_checks':q_records,'runtime_endpoint_calls':calls,
            'proof':{
                'protocol':'At each NEW discordance d<=128, use the frozen probability depending only on(d,min(k,d-k)), draw one independent uniform integer U in[0,10^12), and report the strict current majority iff U<threshold. On a report stop; on a nonreport atH128 permanently abstain from further reports.',
                'path_weight':'For a particular sign prefix z of length d with k plus signs, its first-report weight equals q^k(1-q)^(d-k) times the product of earlier continue probabilities and the final report probability. These policy-coin factors are fixed and do not depend on q.',
                'uniform_positive_error':'A plus-report prefix has k>d/2. On0<q<=1/2, the derivative of log[q^k(1-q)^(d-k)] is(k-dq)/(q(1-q))>0. Therefore every such prefix likelihood is<=2^(-d), with the q=0 boundary by continuity. Multiplying by its same policy-coin weight and summing disjoint first-report prefixes gives P_q(report plus)<=P_.5(report plus).',
                'negative_error':'Reflect every sign word and use the same folded-state probability. This bijection shows equal plus/minus fair risks and gives P_q(report minus)<=P_.5(report minus) for fixed q>=1/2.',
                'null_union':'At q=.5 both report directions are errors and their disjoint union has the independently recomputed exact risk alpha<=.01. At fixed strict alternatives the wrong direction is bounded by alpha/2<=.005.',
                'physical_cap':'For fixed iid physical PSR pairs, conditional discordant signs have one fixed q. Couple the runtime to the full latent sign stream and the same independent policy coins. A physical cap only prevents some latent reports from arriving. Concordant pairs must NOT redraw a policy coin. AfterH128 no reports are permitted; charging remaining pairs to the physical cap changes cost, not risk.',
                'randomness_semantics':'The strict integer comparison accepts exactly threshold of the denominator possible integer values, so the report probability is exactly the stored rational. This assumes genuinely independent uniform integer input; the audit does not claim that an arbitrary caller/RNG supplies this contract.',
                'scope_limit':'The prefix proof uses a single fixed iid q and does not establish control for arbitrary history-dependent changing q, policy retuning from outcomes, repeated coins at the same state, or continuing after a terminal report/horizon.'},
            'runtime_contract_review':{
                'hash_and_exact_risk_before_use':'load_policy registers an immutable policy only after structural validation and exact risk checking. Unverified policy objects are rejected.',
                'majority_and_tie':'Every tie threshold is zero. The runtime strict comparison thus cannot report a tie; reflected counts report the reflected sign.',
                'horizon':'Inputs outside1<=d<=128 are rejected. At128 a nonreport returnsabstain.',
                'caller_statefulness':'decide is intentionally a stateless decision kernel. The caller must enforce exactly one call/coin per new discordance and absorbing stop/abstention. These protocol obligations are necessary for the theorem; the physical evaluation plan and direct DP enforce them.',
                'input_types':'Counts and uniform_integer must be Python ints, excluding bool; an external numpy integer draw needs explicit int conversion.',
                'no_truth_input':'Only verified table, paid counts and a fresh independent coin enter decide. No true or estimated q is accepted.'},
            'not_claimed':['No exact global optimality of the floating occupancy LP.',
                           'No power guarantee away from explicitly checked design points.',
                           'No certified physical-cost gain or SOTA novelty from this risk review.'],
            'source_sha256':{Path(__file__).resolve().relative_to(ROOT).as_posix():sha(__file__),
                             Path(runtime.__file__).resolve().relative_to(ROOT).as_posix():sha(runtime.__file__),
                             artifact_path.relative_to(ROOT).as_posix():sha(artifact_path),
                             audit_path.relative_to(ROOT).as_posix():sha(audit_path),
                             physical_plan.relative_to(ROOT).as_posix():sha(physical_plan),
                             'experiments/sota_round5/evaluate_optimal_sign_physical.py':sha(ROOT/'experiments/sota_round5/evaluate_optimal_sign_physical.py')}}
    return write_new('optimal_sign_policy_review.json',result)


if __name__=='__main__':
    print(json.dumps({'oracle':oracle_review()}),flush=True)
    print(json.dumps({'policy':policy_review()}),flush=True)
