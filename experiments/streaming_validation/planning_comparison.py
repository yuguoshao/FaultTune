"""Planning time on existing paid histories, without new simulation.

Both scalar methods optimize the same uniform-pair variance with the same
angle cap. The numerical comparator uses the cheaper cosine-coordinate
objective, checks endpoints, and avoids unnecessary trigonometric calls.
The vector comparison uses a tuned variance-only grid, not a KL calculation.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np
from scipy.optimize import minimize_scalar

from designs import DEFAULT_CONFIG,empty_state,update_pairs,fit_state,uniform_variance_shift

ROOT=Path(__file__).resolve().parents[2]
DIRECTORY=ROOT/"results"/"streaming_validation"
ROUND_COUNT=7
ITERATIONS=10
NUMERIC_XATOL=1e-5
CAP=math.cos(DEFAULT_CONFIG.min_shift)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_paid_fit():
    directory=DIRECTORY/"development_v1"
    spec=json.loads((directory/"spec.json").read_text(encoding="utf-8"))
    main_ids={case["case_id"] for case in spec["cases"] if case["kind"]=="main"}
    cid_expression=re.compile(r'"case_id":(\d+)')
    rep_expression=re.compile(r'"rep":(\d+)')
    selected=[]
    with (directory/"trials.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            header=line[:160]
            if '"method":"closed_beta"' not in header:
                continue
            cid=int(cid_expression.search(header).group(1))
            rep=int(rep_expression.search(header).group(1))
            if cid in main_ids and rep<8:
                selected.append(json.loads(line))
    assert len(selected)==384
    assert {(row["case_id"],row["rep"]) for row in selected}=={
        (cid,rep) for cid in main_ids for rep in range(8)}
    state=empty_state(len(selected))
    for block_index in range(max(len(row["blocks"]) for row in selected)):
        indices=np.array([i for i,row in enumerate(selected) if len(row["blocks"])>block_index],dtype=int)
        entries=[selected[i]["blocks"][block_index] for i in indices]
        update_pairs(state,np.array([entry[0] for entry in entries]),
                     np.array([entry[1:5] for entry in entries],dtype=np.int64),indices)
    assert state["shots"].tolist()==[row["shots"] for row in selected]
    return fit_state(state),selected,{"spec.json":digest(directory/"spec.json"),
                                    "trials.jsonl":digest(directory/"trials.jsonl")}


def objective_z(z,d,linear,b2):
    return (d+linear*z-b2*z*z)/(1-z*z)


def closed_scalar_one(a,b):
    d=a*(1-a)-b*b
    linear=(1-2*a)*b
    z=-linear/(d+math.sqrt(max(0.,(d-linear)*(d+linear))))
    return math.acos(max(-CAP,min(CAP,z)))


def numeric_scalar_one(a,b):
    d=a*(1-a)
    linear=(1-2*a)*b
    b2=b*b
    answer=minimize_scalar(objective_z,args=(d,linear,b2),method="bounded",
                           bounds=(-CAP,CAP),options={"xatol":NUMERIC_XATOL,"maxiter":200})
    if not answer.success:
        raise ArithmeticError("bounded optimizer failed")
    candidates=(-CAP,float(answer.x),CAP)
    z=min(candidates,key=lambda value:objective_z(value,d,linear,b2))
    return math.acos(z),int(answer.nfev)+2


def variance(fit,t):
    z=np.cos(t)
    center=fit["A"]+fit["B"]*z
    return 4*(center*(1-center)/(1-z*z)-fit["C"]**2)


def main():
    fit,records,input_hashes=load_paid_fit()
    coefficients=list(zip(fit["A"].tolist(),fit["B"].tolist()))
    angles=np.linspace(DEFAULT_CONFIG.min_shift,math.pi-DEFAULT_CONFIG.min_shift,513)
    angles[256]=math.pi/2
    cosine=np.cos(angles)[None,:]
    d=(fit["A"]*(1-fit["A"]))[:,None]
    linear=((1-2*fit["A"])*fit["B"])[:,None]
    b2=(fit["B"]**2)[:,None]
    def closed_scalar():
        return np.array([closed_scalar_one(a,b) for a,b in coefficients])
    def numeric_scalar():
        return np.array([numeric_scalar_one(a,b)[0] for a,b in coefficients])
    def closed_vector():
        return uniform_variance_shift(fit)[0]
    def tuned_grid_vector():
        score=(d+linear*cosine-b2*cosine*cosine)/(1-cosine*cosine)
        return angles[np.argmin(score,axis=1)]
    functions={"closed_scalar":closed_scalar,"scipy_bounded_scalar":numeric_scalar,
               "closed_vector":closed_vector,"tuned_513_grid_vector":tuned_grid_vector}
    outputs={name:function() for name,function in functions.items()}
    closed_variance=variance(fit,outputs["closed_scalar"])
    scalar_angle_error=float(np.max(np.abs(outputs["closed_scalar"]-outputs["scipy_bounded_scalar"])))
    scalar_variance_error=float(np.max(np.abs(variance(fit,outputs["scipy_bounded_scalar"])/closed_variance-1)))
    vector_angle_error=float(np.max(np.abs(outputs["closed_scalar"]-outputs["closed_vector"])))
    grid_variance_excess=float(np.max(variance(fit,outputs["tuned_513_grid_vector"])/closed_variance-1))
    assert scalar_variance_error<1e-8
    assert vector_angle_error<2e-12
    assert grid_variance_excess<1e-4
    evaluation_counts=[numeric_scalar_one(a,b)[1] for a,b in coefficients]
    observations={name:[] for name in functions}
    names=list(functions)
    sink=0.
    for repeat in range(ROUND_COUNT):
        order=names[repeat%len(names):]+names[:repeat%len(names)]
        for name in order:
            start=time.perf_counter()
            for unused in range(ITERATIONS):
                result=functions[name]()
                sink+=float(result[0])
            observations[name].append((time.perf_counter()-start)/ITERATIONS)
    medians={name:float(np.median(values)) for name,values in observations.items()}
    output={"status":"local planning microbenchmark on existing paid states",
            "states":len(records),"selection":"first8closed_beta finalhistories from each48maincases",
            "new_simulated_trajectories":0,"rounds":ROUND_COUNT,"iterations_per_round":ITERATIONS,
            "fitting":"same fit precomputed once and excluded from every planning timing",
            "scalar_comparison":"pure-Python scalar closed formula versus SciPy bounded scalar optimization in cosine coordinate, with endpoint checks",
            "vector_comparison":"batched closed formula versus tuned 513-point variance-only grid",
            "bounded_optimizer_options":{"xatol":NUMERIC_XATOL,"maxiter":200,"coordinate":"cos(t)",
                                         "interpretation":"practical SciPy default x tolerance; same objective and caps"},
            "median_seconds_per384states":medians,"all_seconds_per384states":observations,
            "scalar_numeric_over_closed":medians["scipy_bounded_scalar"]/medians["closed_scalar"],
            "vector_grid_over_closed":medians["tuned_513_grid_vector"]/medians["closed_vector"],
            "numeric_objective_evaluations":{"minimum":min(evaluation_counts),
                                              "median":float(np.median(evaluation_counts)),"maximum":max(evaluation_counts)},
            "agreement":{"scalar_numeric_max_angle_difference_radians":scalar_angle_error,
                         "scalar_numeric_max_relative_variance_difference":scalar_variance_error,
                         "scalar_vector_closed_max_angle_difference_radians":vector_angle_error,
                         "grid_max_relative_variance_excess":grid_variance_excess},
            "limits":["one Windows machine under background load","finite selected observed states",
                      "specific numerical implementations, not all optimizers","no shot-saving or SOTA claim"],
            "state_keys":[[row["case_id"],row["rep"]] for row in records],
            "input_sha256":input_hashes,
            "source_sha256":{name:digest(Path(__file__).resolve().parent/name)
                             for name in ("planning_comparison.py","designs.py")},
            "timing_sink":sink}
    path=DIRECTORY/"planning_comparison.json"
    path.write_text(json.dumps(output,indent=2),encoding="utf-8")
    print(json.dumps({key:value for key,value in output.items()
                      if key not in ("state_keys","timing_sink")},indent=2))


if __name__=="__main__":
    main()
