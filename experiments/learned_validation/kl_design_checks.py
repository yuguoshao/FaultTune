"""Independent checks of the same-pilot Bernoulli KL design."""

import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize_scalar

from designs import DEFAULT_CONFIG, choose_design
from kl_designs import bernoulli_kl, choose_kl_design, optimal_kl_allocation


def decimal_reference(pp,pm):
    with localcontext() as ctx:
        ctx.prec=80
        a,b=Decimal.from_float(float(pp)),Decimal.from_float(float(pm))
        one=Decimal(1)
        if a==b:
            return .5,0.
        def h(p):
            return -p*p.ln()-(one-p)*(one-p).ln()
        slope=(h(a)-h(b))/(a-b)
        r=one/(one+slope.exp())
        q=(r-b)/(a-b)
        def kl(p):
            return p*(p/r).ln()+(one-p)*((one-p)/(one-r)).ln()
        return float(q),float(q*kl(a)+(one-q)*kl(b))


def run():
    started=time.time()
    rng=np.random.RandomState(742311)
    pp=rng.uniform(.00002,.99998,128)
    pm=rng.uniform(.00002,.99998,128)
    special=[]
    for center in (.00002,.001,.01,.1,.5,.9,.99,.99998):
        for scale in (0.,1e-10,1e-7,1e-4,.000999,.001001,.01,.9):
            d=scale*min(center,1-center)
            special.append((center+d,center-d))
            special.append((center-d,center+d))
    pp=np.concatenate((pp,np.array(special)[:,0]))
    pm=np.concatenate((pm,np.array(special)[:,1]))
    q=optimal_kl_allocation(pp,pm)
    r=q*pp+(1-q)*pm
    rate=q*bernoulli_kl(pp,r)+(1-q)*bernoulli_kl(pm,r)
    max_allocation_error=0.
    max_rate_relative_error=0.
    max_independent_optimization_loss=0.
    for i in range(len(pp)):
        ref_q,ref_rate=decimal_reference(pp[i],pm[i])
        max_allocation_error=max(max_allocation_error,abs(float(q[i])-ref_q))
        if ref_rate>1e-24:
            max_rate_relative_error=max(max_rate_relative_error,abs(float(rate[i])/ref_rate-1))
        def objective(fraction):
            null=fraction*pp[i]+(1-fraction)*pm[i]
            return -float(fraction*bernoulli_kl(pp[i],null)+(1-fraction)*bernoulli_kl(pm[i],null))
        opt=minimize_scalar(objective,bounds=(.1,.9),method="bounded",options={"xatol":1e-12})
        qc=float(np.clip(q[i],.1,.9))
        loss=objective(qc)-opt.fun
        max_independent_optimization_loss=max(max_independent_optimization_loss,loss)
    assert max_allocation_error<2e-9
    # Near-identical float probabilities lose relative precision in their
    # reconstructed null mean; absolute evidence remains tiny and harmless.
    assert max_rate_relative_error<1e-5
    assert max_independent_optimization_loss<1e-12
    assert np.max(np.abs(q+optimal_kl_allocation(pm,pp)-1))<1e-12

    totals=np.full((96,4),256,dtype=int)
    counts=rng.binomial(totals,rng.uniform(.001,.999,(96,4)))
    counts[:4]=np.array([[0,0,0,0],[256,256,256,256],[128,128,128,128],[256,128,0,128]])
    designs=choose_kl_design(counts,totals)
    variance_design=choose_design("learned_grid_design",counts,totals)
    max_scalar_difference=0.
    max_grid_loss=0.
    max_sign_symmetry_difference=0.
    for i in range(len(counts)):
        scalar=choose_kl_design(counts[i],totals[i])
        for field in ("t","allocation_plus","estimated_variance","estimated_kl_rate"):
            max_scalar_difference=max(max_scalar_difference,abs(float(scalar[field])-float(designs[field][i])))
        reflection=counts[i].copy()
        reflection[1],reflection[3]=counts[i,3],counts[i,1]
        reflected=choose_kl_design(reflection,totals[i])
        max_sign_symmetry_difference=max(max_sign_symmetry_difference,
            abs(float(reflected["t"])-float(scalar["t"])),
            abs(float(reflected["allocation_plus"])+float(scalar["allocation_plus"])-1))
        fit=scalar["fit"]
        a,b,c=(float(fit[k]) for k in ("A","B","C"))
        grid=np.linspace(DEFAULT_CONFIG.min_shift,math.pi-DEFAULT_CONFIG.min_shift,4097)
        center=a+b*np.cos(grid)
        pplus,pminus=center+c*np.sin(grid),center-c*np.sin(grid)
        qs=np.clip(optimal_kl_allocation(pplus,pminus),.1,.9)
        null=qs*pplus+(1-qs)*pminus
        evidence=qs*bernoulli_kl(pplus,null)+(1-qs)*bernoulli_kl(pminus,null)
        if np.max(evidence)>1e-18:
            max_grid_loss=max(max_grid_loss,1-float(scalar["estimated_kl_rate"])/float(np.max(evidence)))
        else:
            assert abs(float(scalar["t"])-float(variance_design["t"][i]))<1e-12
    assert max_scalar_difference==0
    assert max_sign_symmetry_difference<1e-10
    assert max_grid_loss<.001
    assert np.all(np.isfinite(designs["estimated_variance"]))
    assert np.all(designs["estimated_kl_rate"]>=0)
    assert np.all(designs["pilot_shots"]==1024)
    # At a vanishing gap both allocations converge to one half, although
    # their first-order corrections have opposite signs in skewed models.
    center=.1
    gaps=np.array([1e-4,1e-6,1e-8])
    kl_q=optimal_kl_allocation(center+gaps,center-gaps)
    vplus=(center+gaps)*(1-center-gaps)
    vminus=(center-gaps)*(1-center+gaps)
    neyman_q=np.sqrt(vplus)/(np.sqrt(vplus)+np.sqrt(vminus))
    assert np.all(kl_q<.5) and np.all(neyman_q>.5)
    assert abs(float(kl_q[-1])-.5)<1e-8
    assert abs(float(neyman_q[-1])-.5)<3e-8
    # Public matrix feasibility is checked independently of runner results.
    gaps_main=[]
    for a in (.01,.05,.2,.5):
        for rho in (.5,.98):
            radius=rho*min(a,1-a)
            for ratio in (0.,.9,.995):
                for sign in (-1,1):
                    b=radius*ratio
                    c=sign*radius*math.sqrt(1-ratio*ratio)
                    assert b*b+c*c<min(a*a,(1-a)*(1-a))
                    gaps_main.append(abs(2*c*math.sin(.1)))
    assert min(gaps_main)>.00009
    return {"schema":"faulttune.learned_kl_design_checks.v1","passed":True,
            "allocation_reference_cases":len(pp),"pilot_design_cases":len(counts),
            "maximum_allocation_error_against_80_digit_reference":max_allocation_error,
            "maximum_rate_relative_error_above_1e_minus_24":max_rate_relative_error,
            "maximum_rate_loss_against_independent_scalar_optimizer":max_independent_optimization_loss,
            "maximum_scalar_batch_difference":max_scalar_difference,
            "maximum_sign_reflection_difference":max_sign_symmetry_difference,
            "maximum_513_grid_rate_loss_against_4097_grid":max_grid_loss,
            "zero_gap_local_limit_checked":True,
            "zero_gap_allocation_limit":.5,
            "first_order_neyman_equality_claim":False,
            "main_matrix_cases":len(gaps_main),"main_matrix_minimum_absolute_delta":min(gaps_main),
            "common_epsilon":.00009,
            "scope":"same-pilot symmetric two-angle KL objective only; no full structured-identification SOTA claim",
            "elapsed_seconds":time.time()-started,
            "source_sha256":{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                for name in ("kl_designs.py","kl_design_checks.py")}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="results/learned_validation/kl_design_checks.json")
    args=parser.parse_args()
    result=run()
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
