"""Independent numerical checks for paid streaming design choices.

These are design correctness checks, not Monte Carlo validation results.
"""
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import xlogy

from designs import (DEFAULT_CONFIG,DesignConfig,POLICIES,choose_design,config_dict,
                     empty_state,fit_state,fisher_rate,half_coin_kl_from_bias,pair_scores,
                     uniform_variance_shift,update_pairs)

ROOT=Path(__file__).resolve().parents[2]


def main():
    started=time.perf_counter()
    rng=np.random.RandomState(480131)
    size,blocks=48,8
    a=rng.uniform(.005,.995,size)
    radius=rng.uniform(.01,.98,size)*np.minimum(a,1-a)
    phase=rng.uniform(-math.pi,math.pi,size)
    b,c=radius*np.cos(phase),radius*np.sin(phase)
    state=empty_state(size)
    scalar_states=[empty_state(1) for unused in range(size)]
    direct_gram=np.zeros((size,3,3))
    direct_rhs=np.zeros((size,3))
    direct_shots=np.zeros(size,dtype=int)
    count_only_records=[]
    for block in range(blocks):
        if block<3:
            t=np.full(size,(math.pi/2,math.pi/3,2*math.pi/3)[block])
        else:
            t=choose_design("paired_variance_closed",state)["t"]
        counts=[]
        for i in range(size):
            pp=a[i]+b[i]*math.cos(t[i])+c[i]*math.sin(t[i])
            pm=a[i]+b[i]*math.cos(t[i])-c[i]*math.sin(t[i])
            number=int(rng.randint(1,33))
            joint=np.array([(1-pm)*(1-pp),(1-pm)*pp,pm*(1-pp),pm*pp])
            joint/=joint.sum()
            counts.append(rng.multinomial(number,joint))
        counts=np.array(counts,dtype=np.int64)
        count_only_records.append((t.copy(),counts.copy()))
        update_pairs(state,t,counts)
        for i in range(size):
            update_pairs(scalar_states[i],float(t[i]),counts[i:i+1])
            n=int(counts[i].sum())
            sminus=int(counts[i,2]+counts[i,3])
            splus=int(counts[i,1]+counts[i,3])
            # Independently expand a weighted design matrix. The ordering of
            # responses within one angle does not affect normal equations.
            features=np.array([[1,math.cos(t[i]),-math.sin(t[i])],
                               [1,math.cos(t[i]), math.sin(t[i])]])
            direct_gram[i]+=n*features.T.dot(features)
            direct_rhs[i]+=features.T.dot(np.array([sminus,splus]))
            direct_shots[i]+=2*n
    assert np.max(np.abs(state["gram"]-direct_gram))<2e-12
    assert np.max(np.abs(state["rhs"]-direct_rhs))<2e-12
    assert np.array_equal(state["shots"],direct_shots)
    assert np.array_equal(state["shots"],2*state["pairs"])
    assert np.array_equal(state["blocks"],np.full(size,blocks))
    fit=fit_state(state)
    reference=np.array([np.linalg.solve(direct_gram[i]+np.eye(3),
                        direct_rhs[i]+np.array([.5,0,0])) for i in range(size)])
    assert np.max(np.abs(np.column_stack([fit["raw_A"],fit["raw_B"],fit["raw_C"]])-reference))<1e-12
    max_scalar=0.
    for policy in POLICIES:
        result=choose_design(policy,state)
        for i in range(size):
            single=choose_design(policy,scalar_states[i])
            for field in ("t","estimated_variance","estimated_pair_rate"):
                max_scalar=max(max_scalar,abs(float(result[field][i])-float(single[field][0])))
        assert np.all(result["fit"]["fitted_probability_min"]>0)
        assert np.all(result["fit"]["fitted_probability_max"]<1)
    assert max_scalar<2e-12
    # Update a selected subset and verify untouched states remain identical.
    subset=empty_state(size)
    chosen=np.array([0,3,11,35,47],dtype=int)
    update_pairs(subset,count_only_records[0][0][chosen],count_only_records[0][1][chosen],indices=chosen)
    untouched=np.ones(size,dtype=bool)
    untouched[chosen]=False
    assert not np.any(subset["shots"][untouched])
    assert not np.any(subset["gram"][untouched])
    assert not np.any(subset["rhs"][untouched])
    full=choose_design("paired_kl_grid",state)
    partial=choose_design("paired_kl_grid",state,indices=chosen)
    assert np.array_equal(partial["t"],full["t"][chosen])
    # Before exploration, B is unidentified and stays at its ridge prior.
    psr_only=empty_state(2)
    update_pairs(psr_only,math.pi/2,np.array([[50,20,10,20],[0,0,0,100]]))
    assert np.max(np.abs(fit_state(psr_only)["raw_B"]))<1e-12
    assert np.allclose(choose_design("paired_variance_closed",psr_only)["t"],math.pi/2)
    assert np.array_equal(choose_design("paired_variance_closed",empty_state(3))["t"],
                          np.full(3,math.pi/2))

    # Independent minimization of the true variance objective (not the
    # derivative used in the implementation), plus 80-digit root checks.
    points=[]
    for aa in (.00011,.01,.1,.3,.5,.8,.99,.99989):
        for rho in (.2,.8,.999):
            for fraction in (-.999,-.4,0.,.4,.999):
                radius=rho*min(aa,1-aa)
                points.append((aa,radius*fraction,radius*math.sqrt(1-fraction*fraction)))
    oracle_fit={key:np.array([p[j] for p in points]) for j,key in enumerate(("A","B","C"))}
    selected_t,unused=uniform_variance_shift(oracle_fit)
    max_var_loss,max_root_error,max_grid_loss=0.,0.,0.
    for i,(aa,bb,cc) in enumerate(points):
        def independent_variance(t):
            pp=aa+bb*math.cos(t)+cc*math.sin(t)
            pm=aa+bb*math.cos(t)-cc*math.sin(t)
            return 2*(pp*(1-pp)+pm*(1-pm))/(math.sin(t)**2)
        answer=minimize_scalar(independent_variance,method="bounded",
                               bounds=(DEFAULT_CONFIG.min_shift,math.pi-DEFAULT_CONFIG.min_shift),
                               options={"xatol":1e-13})
        reference=min(answer.fun,independent_variance(DEFAULT_CONFIG.min_shift),
                      independent_variance(math.pi-DEFAULT_CONFIG.min_shift))
        max_var_loss=max(max_var_loss,independent_variance(selected_t[i])/reference-1)
        with localcontext() as context:
            context.prec=80
            da,db=Decimal.from_float(aa),Decimal.from_float(bb)
            dd=da*(1-da)-db*db
            dl=(1-2*da)*db
            zz=-dl/(dd+(dd*dd-dl*dl).sqrt())
            expected=np.clip(math.acos(float(zz)),DEFAULT_CONFIG.min_shift,math.pi-DEFAULT_CONFIG.min_shift)
        max_root_error=max(max_root_error,abs(float(selected_t[i])-expected))
    assert max_var_loss<5e-10 and max_root_error<2e-12

    # Two independent KL computations: direct four-outcome rate and a
    # high-precision expansion including tiny differences.
    max_kl_relative=0.
    max_fisher_relative=0.
    kl_cases=0
    for center in (.00001,.01,.2,.5,.9,.99999):
        for fraction in (0.,1e-8,1e-4,.01,.5,.999):
            delta=fraction*min(center,1-center)
            pp,pm=center+delta,center-delta
            unused,rate,unused=pair_scores(pp,pm,1.)
            with localcontext() as context:
                context.prec=80
                dp,dm=Decimal.from_float(pp),Decimal.from_float(pm)
                ww=(1-dm)*dp
                better=dm*(1-dp)
                dd=ww+better
                reference=(ww*(2*ww/dd).ln()+better*(2*better/dd).ln())/2
                center_decimal=(dp+dm)/2
                fisher_reference=sum(p*(p/center_decimal).ln()+(1-p)*((1-p)/(1-center_decimal)).ln()
                                     for p in (dp,dm))/2
            ref=float(reference)
            fisher_ref=float(fisher_reference)
            if ref>1e-30:
                max_kl_relative=max(max_kl_relative,abs(float(rate)/ref-1))
                max_fisher_relative=max(max_fisher_relative,abs(float(fisher_rate(pp,pm))/fisher_ref-1))
            else:
                assert float(rate)==0.
            kl_cases+=1
    assert max_kl_relative<1e-8 and max_fisher_relative<1e-8
    assert half_coin_kl_from_bias(0.)==0
    assert abs(float(half_coin_kl_from_bias(1.))-math.log(2))<1e-15
    assert abs(float(half_coin_kl_from_bias(-1.))-math.log(2))<1e-15
    dense_config=DesignConfig(grid_points=4097)
    coarse=choose_design("paired_kl_grid",state)
    dense=choose_design("paired_kl_grid",state,config=dense_config)
    nonzero=dense["estimated_pair_rate"]>1e-20
    max_grid_loss=float(np.max(1-coarse["estimated_pair_rate"][nonzero]/dense["estimated_pair_rate"][nonzero]))
    assert max_grid_loss<1e-4
    fisher_coarse=choose_design("fisher_kl_grid",state)
    fisher_dense=choose_design("fisher_kl_grid",state,config=dense_config)
    nonzero=fisher_dense["estimated_fisher_rate"]>1e-20
    max_fisher_grid_loss=float(np.max(1-fisher_coarse["estimated_fisher_rate"][nonzero]/
                                      fisher_dense["estimated_fisher_rate"][nonzero]))
    assert max_fisher_grid_loss<1e-4
    var_grid=choose_design("paired_variance_grid",state)
    var_closed=choose_design("paired_variance_closed",state)
    assert np.all(var_closed["estimated_variance"]<=var_grid["estimated_variance"]+1e-12)
    # Reflect all paid outcomes around the target midpoint; choices are even
    # in C, while p+ and p- and fitted C exchange.
    reflected=empty_state(size)
    for t,counts in count_only_records:
        swapped=counts[:,[0,2,1,3]]
        update_pairs(reflected,t,swapped)
    for policy in POLICIES:
        original=choose_design(policy,state)
        reverse=choose_design(policy,reflected)
        assert np.max(np.abs(original["t"]-reverse["t"]))<1e-12
        assert np.max(np.abs(original["estimated_p_plus"]-reverse["estimated_p_minus"]))<1e-12
    invalid=0
    for action in (
        lambda:empty_state(0),
        lambda:empty_state(True),
        lambda:update_pairs(empty_state(1),0.,[[1,0,0,0]]),
        lambda:update_pairs(empty_state(1),math.pi,[[1,0,0,0]]),
        lambda:update_pairs(empty_state(1),1.,[[1,-1,0,0]]),
        lambda:update_pairs(empty_state(1),1.,[[.5,0,0,0]]),
        lambda:update_pairs(empty_state(2),1.,[[1,0,0,0],[1,0,0,0]],indices=[0,0]),
        lambda:choose_design("bad",empty_state(1)),
    ):
        try:
            action()
        except ValueError:
            invalid+=1
        else:
            raise AssertionError("invalid input accepted")
    result={"status":"passed","config":config_dict(),"adaptive_paid_state_records":size,
            "blocks_per_state":blocks,"policy_scalar_batch_max_difference":max_scalar,
            "uniform_variance_oracle_checks":len(points),
            "max_closed_variance_relative_loss":max_var_loss,"max_80_digit_root_error":max_root_error,
            "independent_80_digit_pair_rate_checks":kl_cases,"max_pair_rate_relative_error":max_kl_relative,
            "independent_80_digit_fisher_rate_checks":kl_cases,"max_fisher_rate_relative_error":max_fisher_relative,
            "max_513_grid_pair_rate_loss_against4097":max_grid_loss,
            "max_513_grid_fisher_rate_loss_against4097":max_fisher_grid_loss,
            "reflection_policy_checks":len(POLICIES)*size,"invalid_inputs_rejected":invalid,
            "scope":"design correctness and numerical checks; no sequential performance or SOTA claim",
            "elapsed_seconds":time.perf_counter()-started,
            "source_sha256":{name:hashlib.sha256((Path(__file__).resolve().parent/name).read_bytes()).hexdigest()
                              for name in ("designs.py","design_checks.py")}}
    output=ROOT/"results"/"streaming_validation"/"design_checks.json"
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
