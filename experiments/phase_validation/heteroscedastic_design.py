"""Local heteroscedastic design for f(x)=A+B*cos(x)+C*sin(x).

All A,B,C are unknown to inference.  Probability laws are physical on the full
circle: A +/- sqrt(B**2+C**2) are in [0,1].  This is an oracle information
diagnostic, not an implementable adaptive strategy or a finite-sample result.
The finite-angle c-optimal design is solved numerically by the Elfving linear
program, with separate primal/dual residuals.  Every prerequest grid point is
retained, including zero target differences.  No source from previous studies
is modified or used to supply data.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy.optimize import linprog, minimize_scalar


def probability(a,b,c,x):
    return a+b*np.cos(x)+c*np.sin(x)


def symmetric_variance(a,b,c,t):
    plus=probability(a,b,c,t)
    minus=probability(a,b,c,-t)
    return (np.sqrt(plus*(1-plus))+np.sqrt(minus*(1-minus)))**2/np.sin(t)**2


def symmetric_optimum(a,b,c):
    # Deterministic dense mesh plus local bounded refinement.  This is a
    # numerical search, not a proof of continuous global optimality.
    mesh=np.linspace(1e-5,math.pi-1e-5,1001)
    values=symmetric_variance(a,b,c,mesh)
    index=int(np.argmin(values))
    low=mesh[max(0,index-1)]
    high=mesh[min(len(mesh)-1,index+1)]
    refined=minimize_scalar(lambda t:float(symmetric_variance(a,b,c,t)),
                            bounds=(low,high),method="bounded",options={"xatol":1e-12})
    if refined.fun > values[index]:
        return float(values[index]),float(mesh[index])
    return float(refined.fun),float(refined.x)


def finite_angle_c_optimum(a,b,c,angles):
    """Minimize (sum |q_i|sqrt(v_i))^2 subject to sum q_i X_i=(0,0,2).

    These are coefficients estimating 2C, so the target old/new difference
    variance is this value times sin(h/2)^2.  Given coefficients, the optimum
    independent shot proportions are |q_i|sqrt(v_i)/sum|q_i|sqrt(v_i).
    """
    probabilities=probability(a,b,c,angles)
    if np.any(probabilities<=0) or np.any(probabilities>=1):
        raise ValueError("interior probabilities required for regular local design")
    sd=np.sqrt(probabilities*(1-probabilities))
    features=np.array([np.ones(len(angles)),np.cos(angles),np.sin(angles)])
    target=np.array([0.,0.,2.])
    equality=np.concatenate((features,-features),axis=1)
    costs=np.concatenate((sd,sd))
    primal=linprog(costs,A_eq=equality,b_eq=target,bounds=(0,None),
                   method="simplex",options={"tol":1e-10,"maxiter":3000})
    if not primal.success:
        raise ArithmeticError("c-optimal primal LP failed: "+primal.message)
    dual=linprog(-target,A_ub=equality.T,b_ub=costs,bounds=[(None,None)]*3,
                 method="simplex",options={"tol":1e-10,"maxiter":3000})
    if not dual.success:
        raise ArithmeticError("c-optimal dual LP failed: "+dual.message)
    q=primal.x[:len(angles)]-primal.x[len(angles):]
    normalizer=float(np.sum(np.abs(q)*sd))
    weights=np.abs(q)*sd/normalizer
    residual=float(np.max(np.abs(features.dot(q)-target)))
    dual_violation=float(max(0,np.max(equality.T.dot(dual.x)-costs)))
    gap=float(primal.fun+dual.fun)
    if residual>1e-8 or dual_violation>1e-8 or abs(gap)>1e-8:
        raise ArithmeticError("LP residual exceeds diagnostic tolerance")
    support=[{"angle":float(angles[j]),"probability":float(probabilities[j]),
              "coefficient_for_2C":float(q[j]),"oracle_shot_fraction":float(weights[j])}
             for j in range(len(angles)) if weights[j]>1e-9]
    # At any full-rank regular point these same coefficients are locally
    # unbiased while retaining both nuisance parameters A and B.
    return {"variance":normalizer**2,"support":support,
            "primal_equality_residual":residual,"dual_feasibility_violation":dual_violation,
            "primal_dual_objective_gap":gap,"dual_vector":dual.x.tolist(),
            "scope":"double precision numerical optimum over this finite public angle menu, not certified continuous optimum or an actual allocation chosen without truth"}


def evaluate(a,saturation,phase,angles):
    radius=saturation*min(a,1-a)
    b,c=radius*math.cos(phase),radius*math.sin(phase)
    pp,pm=a+c,a-c
    uniform=2*(pp*(1-pp)+pm*(1-pm))
    pi_opt=(math.sqrt(pp*(1-pp))+math.sqrt(pm*(1-pm)))**2
    symmetric,shift=symmetric_optimum(a,b,c)
    full=finite_angle_c_optimum(a,b,c,angles)
    return {"A":a,"B":b,"C":c,"radius":radius,"radial_saturation":saturation,
            "phase":phase,"global_probability_min":a-radius,"global_probability_max":a+radius,
            "public_old_new_separation":.2,"target_delta":2*c*math.sin(.1),
            "pi_over_two_uniform_variance":uniform,"pi_over_two_oracle_allocation_variance":pi_opt,
            "symmetric_free_shift_variance":symmetric,"symmetric_oracle_shift":shift,
            "finite_angle_c_opt":full,
            "ratios":{"pi_uniform_over_pi_optimal_allocation":uniform/pi_opt,
                "pi_uniform_over_symmetric_optimum":uniform/symmetric,
                "pi_optimal_allocation_over_symmetric_optimum":pi_opt/symmetric,
                "pi_uniform_over_finite_c_opt":uniform/full["variance"],
                "pi_optimal_allocation_over_finite_c_opt":pi_opt/full["variance"],
                "symmetric_optimum_over_finite_c_opt":symmetric/full["variance"]}}


def run(angle_count):
    started=time.time()
    if angle_count%4:
        raise ValueError("angle count must include +/-pi/2")
    angles=2*math.pi*np.arange(angle_count)/angle_count
    # Public complete grid, fixed before any probability or ratio is computed.
    means=(.001,.01,.1,.5,.9,.99,.999)
    saturations=(.25,.9,.9999)
    phases=tuple(k*math.pi/12 for k in range(7))
    rows=[]
    for a in means:
        for saturation in saturations:
            for phase in phases:
                rows.append(evaluate(a,saturation,phase,angles))
        print("Completed A={} ({}/{} points, {:.1f}s)".format(a,len(rows),len(means)*len(saturations)*len(phases),time.time()-started),flush=True)
    ratio_keys=list(rows[0]["ratios"])
    extrema={key:{"minimum":min(r["ratios"][key] for r in rows),
                  "maximum":max(r["ratios"][key] for r in rows),
                  "count_over_2":sum(r["ratios"][key]>2+1e-8 for r in rows),
                  "count_over_3":sum(r["ratios"][key]>3+1e-8 for r in rows)} for key in ratio_keys}
    key="pi_optimal_allocation_over_finite_c_opt"
    nonzero=[r for r in rows if abs(r["target_delta"])>1e-12]
    best=max(rows,key=lambda r:r["ratios"][key])
    best_nonzero=max(nonzero,key=lambda r:r["ratios"][key])
    return {"schema":"faulttune.sinusoid_heteroscedastic_design.v1",
            "scope":"finite-grid local information exploration; not a theorem, acceptance experiment, SOTA speedup, or deployable truth-free design",
            "model":"f(x)=A+B*cos(x)+C*sin(x), all three coefficients unknown",
            "variance_normalization":"per total shot for estimating 2C; multiply all variances by sin(h/2)^2 for Delta at public old/new +/-h/2",
            "modern_baseline_note":"standard +/-pi/2 with optimal independent allocation is stronger than its uniform-allocation version; compare both",
            "grid":{"A":list(means),"radial_saturations":list(saturations),"phases":list(phases),
                    "angle_count":angle_count,"angles":angles.tolist(),"all_cells_retained":True},
            "extrema":extrema,"best_vs_pi_oracle_allocation":best,"best_nonzero_target_vs_pi_oracle_allocation":best_nonzero,
            "points":rows,"elapsed_seconds":time.time()-started,
            "versions":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__},
            "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--angles",type=int,default=128)
    parser.add_argument("--output",default="results/heteroscedastic_design.json")
    args=parser.parse_args()
    result=run(args.angles)
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf8")
    print(json.dumps({"output":str(output.resolve()),"extrema":result["extrema"],
                     "best_nonzero_target_vs_pi_oracle_allocation":result["best_nonzero_target_vs_pi_oracle_allocation"],
                     "elapsed_seconds":result["elapsed_seconds"]},indent=2))


if __name__=="__main__":
    main()
