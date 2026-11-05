"""Independent finite checks for paid-pilot design code, not acceptance results.

Truth is used only in this checker to generate synthetic pilot records and to
check the exact conditional shift identity after a possibly inaccurate fit.
No truth is passed to a design policy.  Numerical optimization below is an
independent check, not part of the deployable design path.
"""

import argparse
from decimal import localcontext
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy.optimize import minimize_scalar

from designs import DEFAULT_CONFIG,DesignConfig,PILOT_ANGLES,POLICIES,all_designs,choose_design,config_dict,fit_pilot


def run():
    started=time.time()
    rng=np.random.RandomState(314159)
    nworld=96
    a=rng.uniform(.025,.975,nworld)
    radius=rng.uniform(0,.99,nworld)*np.minimum(a,1-a)
    phase=rng.uniform(-math.pi,math.pi,nworld)
    b,c=radius*np.cos(phase),radius*np.sin(phase)
    true_p=a[:,None]+b[:,None]*np.cos(PILOT_ANGLES)+c[:,None]*np.sin(PILOT_ANGLES)
    totals=np.broadcast_to(np.array([8,32,128,512])[rng.randint(0,4,nworld)][:,None],(nworld,4)).copy()
    counts=rng.binomial(totals,true_p)
    # Empty, all-zero, all-one and extremely inconsistent four-angle records.
    edge_counts=np.array([[0,0,0,0],[0,0,0,0],[100,100,100,100],
                          [100,100,0,0],[100,0,100,0],[0,100,0,0],
                          [0,0,0,0],[10000,10000,10000,10000],
                          [500,1000,500,0],[500,0,500,1000]],dtype=int)
    edge_totals=np.array([[0,0,0,0]]+[[100,100,100,100]]*5+[[10000]*4]*4)
    counts=np.concatenate((counts,edge_counts))
    totals=np.concatenate((totals,edge_totals))
    fit=fit_pilot(counts,totals)
    decisions=all_designs(counts,totals)
    assert np.all(fit["fitted_probability_min"]>0)
    assert np.all(fit["fitted_probability_max"]<1)
    assert np.all(fit["pilot_shots"]==np.sum(totals,axis=-1))
    assert fit["pilot_shots"][nworld]==0
    assert np.sum(fit["mean_was_clipped"])>=2
    max_batch_difference=0.
    max_identity_error=0.
    for policy,result in decisions.items():
        assert np.all(np.isfinite(result["estimated_variance"]))
        assert np.all(result["t"]>=DEFAULT_CONFIG.min_shift)
        assert np.all(result["t"]<=math.pi-DEFAULT_CONFIG.min_shift)
        assert np.all(result["allocation_plus"]>=DEFAULT_CONFIG.min_allocation)
        assert np.all(result["allocation_plus"]<=1-DEFAULT_CONFIG.min_allocation)
        assert result["fit"] is fit or np.array_equal(result["fit"]["A"],fit["A"])
        # Scalar and vector paths, including the final irregular pilot, agree.
        for index in (0,7,31,len(counts)-1):
            scalar=choose_design(policy,counts[index],totals[index])
            for field in ("t","allocation_plus","estimated_variance","pilot_shots"):
                max_batch_difference=max(max_batch_difference,abs(float(scalar[field])-float(result[field][index])))
        t=result["t"][:nworld]
        # Exact finite-difference identity for the TRUE law, even though the
        # selected t was fitted from noisy and often very small pilot records.
        h=.2
        fp=a+b*np.cos(t)+c*np.sin(t)
        fm=a+b*np.cos(t)-c*np.sin(t)
        reconstructed=math.sin(h/2)/np.sin(t)*(fp-fm)
        exact=2*c*math.sin(h/2)
        max_identity_error=max(max_identity_error,float(np.max(np.abs(reconstructed-exact))))
    assert max_batch_difference<1e-12 and max_identity_error<2e-15
    assert float(decisions["learned_closed_form"]["t"][nworld])==math.pi/2
    assert float(decisions["learned_closed_form"]["allocation_plus"][nworld])==.5
    assert np.sum(decisions["learned_closed_form"]["allocation_was_clipped"])>=2
    restrictive=DesignConfig(min_shift=1.1,min_allocation=.3)
    capped=choose_design("learned_closed_form",[1000,500,0,500],[10000]*4,restrictive)
    assert bool(capped["shift_was_clipped"])
    assert float(capped["t"])==math.pi-restrictive.min_shift
    # Generic grid scores the actual clipped-allocation variance.  Its public
    # grid contains pi/2, so it must not be worse than pilot-fitted PSR there.
    grid_scores=decisions["learned_grid_design"]["estimated_variance"]
    psr_scores=decisions["learned_psr_neyman"]["estimated_variance"]
    assert np.all(grid_scores<=psr_scores+2e-13)
    max_closed_unclipped_error=0.
    max_high_precision_error=0.
    high_precision_checks=0
    reference_path=Path(__file__).resolve().parents[1]/"phase_validation"/"design_bound_check.py"
    reference_spec=importlib.util.spec_from_file_location("independent_closed_design_reference",str(reference_path))
    reference_module=importlib.util.module_from_spec(reference_spec)
    reference_spec.loader.exec_module(reference_module)
    max_grid_over_dense=0.
    closed_checks=0
    dense_checks=0
    for index in range(len(counts)):
        aa,bb,cc=(float(fit[key][index]) for key in ("A","B","C"))
        def score(t,clip):
            pp=aa+bb*np.cos(t)+cc*np.sin(t)
            pm=aa+bb*np.cos(t)-cc*np.sin(t)
            vp,vm=pp*(1-pp),pm*(1-pm)
            q=np.sqrt(vp)/(np.sqrt(vp)+np.sqrt(vm))
            if clip:
                q=np.clip(q,DEFAULT_CONFIG.min_allocation,1-DEFAULT_CONFIG.min_allocation)
            return (vp/q+vm/(1-q))/np.sin(t)**2
        dense=np.linspace(DEFAULT_CONFIG.min_shift,math.pi-DEFAULT_CONFIG.min_shift,4097)
        values=score(dense,True)
        dense_best=float(np.min(values))
        max_grid_over_dense=max(max_grid_over_dense,float(grid_scores[index]/dense_best-1))
        dense_checks+=1
        closed=decisions["learned_closed_form"]
        if not closed["shift_was_clipped"][index] and not closed["allocation_was_clipped"][index]:
            mesh=np.linspace(.0001,math.pi-.0001,2001)
            k=int(np.argmin(score(mesh,False)))
            optimum=minimize_scalar(lambda t:float(score(t,False)),method="bounded",
                 bounds=(float(mesh[max(0,k-1)]),float(mesh[min(2000,k+1)])),options={"xatol":1e-12})
            relative=abs(float(closed["estimated_variance"][index])/optimum.fun-1)
            max_closed_unclipped_error=max(max_closed_unclipped_error,relative)
            closed_checks+=1
            with localcontext() as context:
                context.prec=80
                reference=reference_module.design(str(aa),str(bb),str(cc))
                reference_variance=float(reference["V_primal"])
                reference_fraction=float(reference["plus_proportion"])
                reference_shift=math.acos(float(reference["z"]))
                max_high_precision_error=max(max_high_precision_error,
                    abs(float(closed["estimated_variance"][index])/reference_variance-1),
                    abs(float(closed["allocation_plus"][index])-reference_fraction),
                    abs(float(closed["t"][index])-reference_shift))
                high_precision_checks+=1
    assert max_closed_unclipped_error<1e-9
    assert max_high_precision_error<1e-10
    assert max_grid_over_dense<.001
    rejected=0
    for bad_counts,bad_totals in (([1,0,0,0],[0,0,0,0]),([-1,0,0,0],[1]*4),
                                  ([.5,0,0,0],[1]*4),([float("nan"),0,0,0],[1]*4),
                                  ([0,0,0],[1]*3)):
        try:
            choose_design("learned_closed_form",bad_counts,bad_totals)
        except ValueError:
            rejected+=1
        else:
            raise AssertionError("invalid pilot accepted")
    return {"schema":"faulttune.paid_pilot_design_checks.v1","passed":True,
            "config":config_dict(),"policies":list(POLICIES),
            "policy_signature":str(inspect.signature(choose_design)),
            "paid_pilot_records_checked":len(counts),"edge_case_records":len(edge_counts),
            "invalid_inputs_rejected":rejected,"independent_dense_grid_checks":dense_checks,
            "independent_unclipped_scalar_minimization_checks":closed_checks,
            "independent_80_digit_no_cap_reference_checks":high_precision_checks,
            "maximum_80_digit_reference_error":max_high_precision_error,
            "high_precision_reference_sha256":hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            "maximum_scalar_vector_difference":max_batch_difference,
            "maximum_true_conditional_identity_error":max_identity_error,
            "maximum_closed_form_relative_error_against_independent_minimization":max_closed_unclipped_error,
            "maximum_public_grid_variance_over_dense_grid_relative_excess":max_grid_over_dense,
            "mean_fit_clip_count":int(np.sum(fit["mean_was_clipped"])),
            "radial_fit_clip_count":int(np.sum(fit["radius_was_clipped"])),
            "default_allocation_clip_count":int(np.sum(decisions["learned_closed_form"]["allocation_was_clipped"])),
            "restrictive_public_shift_cap_checked":True,
            "scope":"finite correctness checks only; synthetic truth appears in this checker, never in the paid-pilot design API; no acceptance certification or cost-performance claim",
            "elapsed_seconds":time.time()-started,
            "versions":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__},
            "source_sha256":{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                             for name in ("designs.py","design_checks.py")}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="results/learned_design_checks.json")
    args=parser.parse_args()
    result=run()
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
