"""Small reproducible Fisher-information diagnostic for phase-sensitive menus.

No simulation trials or acceptance policies are run.  All actions and design
weights are fixed publicly.  Reported variances are regular local information
quantities under the common four-parameter physical model, not finite-sample
stopping costs, certified speedups, or a comparison to published hardware.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from phase_model import (PARAMETER_NAMES,PhaseParameters,PublicControls,fisher_information,
                         phase_menu,probabilities,reference_phase_menu,self_check,
                         target_delta_and_gradient,task_menu)


def local_variance(information,gradient,indices=(0,1,2,3)):
    """Target CR quadratic form, retaining nuisance parameters in indices."""
    matrix=information[np.ix_(indices,indices)]
    g=gradient[list(indices)]
    scale=np.sqrt(np.maximum(np.diag(matrix),1e-300))
    normalized=matrix/scale[:,None]/scale[None,:]
    values,vectors=np.linalg.eigh((normalized+normalized.T)/2)
    transformed=vectors.T.dot(g/scale)
    keep=values>1e-10
    null_residual=float(np.linalg.norm(transformed[~keep]))
    identifiable=null_residual <= 1e-8*max(1.,float(np.linalg.norm(g/scale)))
    return {"variance_per_shot":float(np.sum(transformed[keep]**2/values[keep])) if identifiable else None,
            "target_locally_identifiable":identifiable,"information_rank":int(np.sum(keep)),
            "parameter_count":len(indices),"normalized_eigenvalues":values.tolist(),
            "gradient_nullspace_residual":null_residual}


def design_result(parameters,controls,menu,weights,gradient,observation):
    information=np.array([fisher_information(parameters,controls,e,observation) for e in menu])
    mean=np.einsum("e,eij->ij",weights,information)
    return {"menu_size":len(menu),"observation":observation,"weights":list(map(float,weights)),
            "mean_information":mean.tolist(),
            "all_four_unknown":local_variance(mean,gradient),
            "kappa_known_only":local_variance(mean,gradient,(0,1,2)),
            "both_noise_parameters_known":local_variance(mean,gradient,(0,1))}


def point(parameters,controls,weight_plus):
    delta,gradient=target_delta_and_gradient(parameters,controls,weight_plus)
    direct=task_menu()
    full=phase_menu()
    reference=reference_phase_menu()
    direct_weights=np.array([1-weight_plus,weight_plus,1-weight_plus,weight_plus])/2
    designs={"direct_task_loss_fixed":design_result(parameters,controls,direct,direct_weights,gradient,"loss"),
             "direct_task_full_fixed":design_result(parameters,controls,direct,direct_weights,gradient,"full"),
             "extended_24_uniform":design_result(parameters,controls,full,np.ones(len(full))/len(full),gradient,"full"),
             "reference_phase_8_uniform":design_result(parameters,controls,reference,np.ones(len(reference))/len(reference),gradient,"full")}
    max_kappa_information=max(fisher_information(parameters,controls,e)[3,3] for e in full)
    nuisance_floor=float(gradient[3]**2/max_kappa_information)
    variances={key:value["all_four_unknown"]["variance_per_shot"] for key,value in designs.items()}
    direct_loss=variances["direct_task_loss_fixed"]
    ratios={key:direct_loss/value if value is not None and value>0 else None for key,value in variances.items()}
    p=np.array([1-probabilities(parameters,controls,e)[0] for e in direct])
    coefficients=np.array([-(1-weight_plus),-weight_plus,1-weight_plus,weight_plus])
    nonparam_fixed=float(np.sum(coefficients**2*p*(1-p)/direct_weights))
    nonparam_optimal=float(np.sum(np.abs(coefficients)*np.sqrt(p*(1-p)))**2)
    return {"parameters":dict(zip(PARAMETER_NAMES,map(float,parameters.as_array()))),
            "public_controls":asdict(controls),"target_delta":delta,"target_gradient":dict(zip(PARAMETER_NAMES,map(float,gradient))),
            "designs":designs,"direct_loss_variance_over_design_variance":ratios,
            "nonparametric_direct_loss_variance":{"fixed_public_allocation":nonparam_fixed,
                "pointwise_oracle_allocation":nonparam_optimal,
                "scope":"unstructured four-Bernoulli target estimator; oracle allocation is not a deployed policy"},
            "any_extended_menu_allocation_local_nuisance_floor":{"value":nuisance_floor,
                "maximum_single_shot_kappa_fisher":float(max_kappa_information),
                "formula":"g^T M^{-1} g >= g_kappa^2 / M_kappa,kappa >= g_kappa^2 / max_e I_e,kappa,kappa",
                "scope":"regular local Fisher quadratic form only; includes unknown kappa and any convex shot allocation over the 24 fixed actions"}}


def verify_frozen_task_agreement():
    """Read-only regression to the frozen original at a=omega, where controls agree."""
    import importlib.util
    import sys
    path=Path(__file__).resolve().parents[1]/"circuit_validation"/"quantum_model.py"
    module_spec=importlib.util.spec_from_file_location("faulttune_original_forward_check",str(path))
    module=importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name]=module
    module_spec.loader.exec_module(module)
    maximum=0.
    count=0
    for omega,gamma,r,kappa in ((.3,.17,.01,.003),(.03,.03,.07,.01),(1.0471975511965976,1.0471975511965976,0.,0.)):
        theta=PhaseParameters(omega,gamma,r,kappa)
        controls=PublicControls(omega)
        old=module.QuantumModel(omega,gamma,r,kappa)
        for e in task_menu():
            context="spectator_plus" if e.q2_preparation=="+" else "spectator_0"
            actual=probabilities(theta,controls,e)
            expected=np.array(module.circuit_probabilities(old,e.variant,context))
            maximum=max(maximum,float(np.max(np.abs(actual-expected))))
            count+=1
    return {"passed":maximum<2e-13,"distributions_checked":count,"maximum_absolute_error":maximum,
            "frozen_source_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "scope":"frozen module is imported only for this forward regression, never to choose controls or calculate information"}


def run():
    start=time.time()
    check=self_check()
    frozen=verify_frozen_task_agreement()
    if not check["passed"] or not frozen["passed"]:
        raise ArithmeticError("physical forward checks failed")
    # Public amplitudes and all world factors are written before evaluating any
    # truth.  No case is selected or removed by Delta, and no measurement basis
    # changes across these cases.  Positive kappa keeps this local probe interior.
    amplitudes=(.3,.1,.03,.01,.003,.001)
    gamma_factors=(.2,1.)
    kappas=(1e-6,.003)
    rows=[]
    for amplitude in amplitudes:
        controls=PublicControls(amplitude)
        for gamma_factor in gamma_factors:
            for kappa in kappas:
                theta=PhaseParameters(amplitude,gamma_factor*amplitude,.01,kappa)
                rows.append(point(theta,controls,.8))
    return {"schema":"faulttune.phase_information_diagnostic.v1","scope":"exploratory finite parameter grid; no trials, no acceptance certificate, no stopping-time or hardware speedup claim",
            "sampling_model":"eight outcomes per paid execution; independent per-qubit readout flip r shared across experiments; q1/q2 dephasing rate kappa shared, attenuation exp(-kappa*tau); ideal known reset and preparation/analysis gates",
            "parameter_order":list(PARAMETER_NAMES),"all_four_parameters_unknown_in_primary_comparison":True,
            "fixed_public_design":{"amplitudes":list(amplitudes),"omega_factors":[1.],"gamma_factors":list(gamma_factors),
                "readout_flip_values":[.01],"kappa_values":list(kappas),"weight_plus":.8,
                "old_duration":1.,"new_duration":2.,"grid_selection":"all Cartesian cells retained regardless of target sign; omega equals public amplitude only at these chosen evaluation points, not in the four-parameter inference model"},
            "menu_descriptions":[e.description(PublicControls(.1)) for e in phase_menu()],
            "menu_description_amplitude_note":"example public amplitude .1; each row states its own public amplitude",
            "self_check":check,"frozen_task_regression":frozen,"points":rows,
            "elapsed_seconds":time.time()-start,"versions":{"python":platform.python_version(),"numpy":np.__version__},
            "sources":{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                       for name in ("phase_model.py","information_probe.py")}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="results/phase_information_probe.json")
    args=parser.parse_args()
    result=run()
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf8")
    print(json.dumps({"output":str(output.resolve()),"points":len(result["points"]),
                      "self_check":result["self_check"],"frozen_task_regression":result["frozen_task_regression"],
                      "elapsed_seconds":result["elapsed_seconds"]},indent=2))
    print("a gamma/a kappa delta direct_loss direct_full extended24 reference8 reference8_known_kappa nuisance_floor")
    for row in result["points"]:
        theta=row["parameters"]
        a=row["public_controls"]["candidate_amplitude"]
        d=row["designs"]
        print(" ".join("{:.7g}".format(v) for v in (a,theta["gamma"]/a,theta["dephasing_rate"],row["target_delta"],
              d["direct_task_loss_fixed"]["all_four_unknown"]["variance_per_shot"],
              d["direct_task_full_fixed"]["all_four_unknown"]["variance_per_shot"],
              d["extended_24_uniform"]["all_four_unknown"]["variance_per_shot"],
              d["reference_phase_8_uniform"]["all_four_unknown"]["variance_per_shot"],
              d["reference_phase_8_uniform"]["kappa_known_only"]["variance_per_shot"],
              row["any_extended_menu_allocation_local_nuisance_floor"]["value"])))


if __name__=="__main__":
    main()
