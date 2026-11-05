"""Small quantum-realizable multi-frequency heteroscedastic design probe.

Private forward worlds are pure two-qubit psi/phi pairs.  U(x) has diagonal
(1,exp(ix),exp(2ix),exp(3ix)); a projection onto phi is followed by a fixed
binary readout flip r.  Thus r <= p(x) <= 1-r for all real x, not just the grid.

Inference deliberately retains all seven unknown Fourier coefficients.  The
state vectors document physical realizability; supplying their exact forward
means to a validator would make this a different, trivial inference problem.
Each datum is one paid binary projection outcome.  LP design uses private true
variances only as an oracle diagnostic: no deployed or finite-sample algorithm
is claimed.  Linear experimental design itself is classical.
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
from scipy.optimize import linprog


def features(angles):
    return np.array([np.ones(len(angles))]+[f(k*angles) for k in (1,2,3) for f in (np.cos,np.sin)])


def probabilities(psi,phi,r,angles):
    coefficients=np.conj(phi)*psi
    amplitudes=np.exp(1j*np.outer(angles,np.arange(4))).dot(coefficients)
    return r+(1-2*r)*np.abs(amplitudes)**2


def fourier_coefficients(psi,phi,r):
    coefficients=np.conj(phi)*psi
    output=[r+(1-2*r)*float(np.sum(np.abs(coefficients)**2))]
    for k in (1,2,3):
        value=np.dot(coefficients[k:],np.conj(coefficients[:-k]))
        output.extend([(1-2*r)*2*float(value.real),-(1-2*r)*2*float(value.imag)])
    return np.array(output)


def lp_witness(feature_matrix,target,costs,l1_upper=None,check_dual=False):
    size=feature_matrix.shape[1]
    equality=np.concatenate((feature_matrix,-feature_matrix),axis=1)
    objective=np.concatenate((costs,costs))
    options={"tol":1e-10,"maxiter":5000}
    kwargs={"A_ub":np.ones((1,2*size)),"b_ub":[l1_upper]} if l1_upper is not None else {}
    solution=linprog(objective,A_eq=equality,b_eq=target,bounds=(0,None),
                     method="simplex",options=options,**kwargs)
    if not solution.success:
        raise ArithmeticError("primal witness LP failed: "+solution.message)
    q=solution.x[:size]-solution.x[size:]
    equality_residual=float(np.max(np.abs(feature_matrix.dot(q)-target)))
    l1=float(np.sum(np.abs(q)))
    if equality_residual>1e-8 or l1_upper is not None and l1>l1_upper+1e-8:
        raise ArithmeticError("witness feasibility check failed")
    certificate={"primal_equality_residual":equality_residual,"l1":l1,
                 "linear_objective":float(np.sum(np.abs(q)*costs))}
    if check_dual:
        dual=linprog(-target,A_ub=equality.T,b_ub=objective,
                     bounds=[(None,None)]*len(target),method="simplex",options=options)
        if not dual.success:
            raise ArithmeticError("dual LP failed: "+dual.message)
        violation=float(max(0,np.max(equality.T.dot(dual.x)-objective)))
        gap=float(solution.fun+dual.fun)
        if violation>1e-8 or abs(gap)>1e-8:
            raise ArithmeticError("LP dual residual check failed")
        certificate.update({"dual_feasibility_violation":violation,"primal_dual_gap":gap,
                            "dual_vector":dual.x.tolist()})
    return q,certificate


def allocation_record(q,p,angles):
    sd=np.sqrt(p*(1-p))
    normalization=float(np.sum(np.abs(q)*sd))
    weights=np.abs(q)*sd/normalization
    return {"variance_per_total_shot":normalization**2,
            "support":[{"angle":float(angles[j]),"probability":float(p[j]),
                         "coefficient":float(q[j]),"oracle_shot_fraction":float(weights[j])}
                        for j in range(len(q)) if abs(q[j])>1e-8]}


def vector_record(vector):
    return [[float(z.real),float(z.imag)] for z in vector]


def pure_state(rng):
    state=rng.normal(size=4)+1j*rng.normal(size=4)
    return state/np.linalg.norm(state)


def state_pairs(seed):
    """16 predeclared worlds; neither signs nor variances control their selection."""
    rng=np.random.RandomState(seed)
    pairs=[]
    for index in range(16):
        psi=pure_state(rng)
        if index<4:
            phi=psi.copy()
            family="return_projection_zero_difference_control"
            offset=0.
        elif index<8:
            offset=(index-3)*math.pi/12
            phi=psi*np.exp(1j*offset*np.arange(4))
            family="public_offset_return_projection"
        else:
            phi=pure_state(rng)
            family="independent_fixed_projection"
            offset=None
        pairs.append({"id":index,"family":family,"public_projection_offset":offset,"psi":psi,"phi":phi})
    return pairs


def run(seed,angle_count):
    started=time.time()
    angles=2*math.pi*np.arange(angle_count)/angle_count
    feature_matrix=features(angles)
    h=.2
    target=np.array([0.,0.,2*math.sin(h/2),0.,2*math.sin(h),0.,2*math.sin(3*h/2)])
    standard,standard_certificate=lp_witness(feature_matrix,target,np.ones(angle_count),check_dual=True)
    # An additional stronger baseline resolves all numerical ties between
    # minimum-l1 PSR rules in favor of the best true-variance allocation.
    standard_l1=standard_certificate["l1"]
    l1_slack=1e-9*max(1.,standard_l1)
    rows=[]
    for pair in state_pairs(seed):
        psi,phi=pair["psi"],pair["phi"]
        for r in (.01,.05):
            p=probabilities(psi,phi,r,angles)
            coefficients=fourier_coefficients(psi,phi,r)
            reconstruction_error=float(np.max(np.abs(feature_matrix.T.dot(coefficients)-p)))
            direct_difference=float(np.diff(probabilities(psi,phi,r,np.array([-h/2,h/2])))[0])
            delta=float(target.dot(coefficients))
            if reconstruction_error>1e-12 or abs(delta-direct_difference)>1e-12:
                raise ArithmeticError("quantum/Fourier forward identity check failed")
            if np.min(p)<r-1e-12 or np.max(p)>1-r+1e-12:
                raise ArithmeticError("global physical probability bound contradicted")
            sd=np.sqrt(p*(1-p))
            optimized,certificate=lp_witness(feature_matrix,target,sd,check_dual=True)
            best_standard,best_standard_certificate=lp_witness(feature_matrix,target,sd,
                                                               l1_upper=standard_l1+l1_slack)
            baseline=allocation_record(standard,p,angles)
            strong_baseline=allocation_record(best_standard,p,angles)
            full=allocation_record(optimized,p,angles)
            fixed_variance=float(standard_l1*np.sum(np.abs(standard)*p*(1-p)))
            rows.append({"pair_id":pair["id"],"family":pair["family"],
                         "public_projection_offset":pair["public_projection_offset"],
                         "psi":vector_record(psi),"phi":vector_record(phi),"binary_readout_flip":r,
                         "global_probability_bounds":[r,1-r],"observed_grid_probability_range":[float(np.min(p)),float(np.max(p))],
                         "fourier_coefficients":coefficients.tolist(),"target_delta":delta,
                         "fourier_reconstruction_error":reconstruction_error,
                         "standard_fixed_homoscedastic_allocation_variance":fixed_variance,
                         "standard_minimum_l1_with_oracle_allocation":baseline,
                         "best_variance_among_minimum_l1_rules":strong_baseline,
                         "full_heteroscedastic_c_opt":full,
                         "c_opt_lp_certificate":certificate,"best_minimum_l1_certificate":best_standard_certificate,
                         "ratios":{"fixed_standard_over_c_opt":fixed_variance/full["variance_per_total_shot"],
                             "standard_oracle_allocation_over_c_opt":baseline["variance_per_total_shot"]/full["variance_per_total_shot"],
                             "best_minimum_l1_rule_over_c_opt":strong_baseline["variance_per_total_shot"]/full["variance_per_total_shot"]}})
        print("Completed pair {}/16 ({} points, {:.1f}s)".format(pair["id"]+1,len(rows),time.time()-started),flush=True)
    keys=list(rows[0]["ratios"])
    summary={key:{"min":min(row["ratios"][key] for row in rows),"max":max(row["ratios"][key] for row in rows),
                  "count_over_2":sum(row["ratios"][key]>2+1e-8 for row in rows),
                  "count_over_3":sum(row["ratios"][key]>3+1e-8 for row in rows)} for key in keys}
    nonzero=[row for row in rows if abs(row["target_delta"])>1e-10]
    strongest="best_minimum_l1_rule_over_c_opt"
    return {"schema":"faulttune.quantum_multifrequency_design.v1",
            "scope":"32 fixed quantum-realizable worlds; oracle local variance diagnostics only; no Monte Carlo acceptance trials, paid pilot, confidence guarantee, or SOTA novelty claim",
            "novelty_boundary":"LP/c-optimal design and generalized parameter shifts are classical existing methods. This probe seeks a physically realizable regime where paid online design might be worthwhile; it does not establish a new algorithm.",
            "public_design":{"seed":seed,"pure_state_pairs":16,"readout_flips":[.01,.05],"angles":angles.tolist(),
                "h":h,"old_x":-h/2,"new_x":h/2,"all_cells_retained":True,
                "generation":"first 4 return projections, next 4 fixed offset projections, final 8 independent projections; normalized complex Gaussian states; no truth filtering"},
            "feature_order":["1","cos(x)","sin(x)","cos(2x)","sin(2x)","cos(3x)","sin(3x)"],
            "estimand_vector":target.tolist(),"standard_l1_witness":standard.tolist(),
            "standard_l1_lp_certificate":standard_certificate,"standard_l1_tie_slack":l1_slack,
            "summary":summary,"best_nonzero_target_against_strongest_baseline":max(nonzero,key=lambda row:row["ratios"][strongest]),
            "points":rows,"elapsed_seconds":time.time()-started,
            "versions":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__},
            "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,default=271828)
    parser.add_argument("--angles",type=int,default=128)
    parser.add_argument("--output",default="results/multifrequency_design.json")
    args=parser.parse_args()
    result=run(args.seed,args.angles)
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf8")
    best=result["best_nonzero_target_against_strongest_baseline"]
    print(json.dumps({"output":str(path.resolve()),"summary":result["summary"],
                     "best_nonzero":{"pair_id":best["pair_id"],"family":best["family"],
                       "r":best["binary_readout_flip"],"delta":best["target_delta"],"ratios":best["ratios"]},
                     "elapsed_seconds":result["elapsed_seconds"]},indent=2))


if __name__=="__main__":
    main()
