"""Independent development probe: all-angle null minimax versus two angles.

Numerical results are heuristic bounds on a finite grid, not certified proofs.
No pilot validation experiment is run here.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from kl_designs import bernoulli_kl, optimal_kl_allocation


def pair_rate(t, a, b, c):
    plus = a+b*np.cos(t)+c*np.sin(t)
    minus = a+b*np.cos(t)-c*np.sin(t)
    fraction = optimal_kl_allocation(plus, minus)
    null = fraction*plus+(1-fraction)*minus
    rate = fraction*bernoulli_kl(plus, null)+(1-fraction)*bernoulli_kl(minus, null)
    return rate, plus, minus, fraction, null


def probe(a, b, c, points=2048):
    grid = np.linspace(1e-5, math.pi-1e-5, points+1)
    scores = pair_rate(grid, a, b, c)[0]
    k = int(np.argmax(scores))
    result = minimize_scalar(lambda t: -float(pair_rate(t, a, b, c)[0]),
                             bounds=(grid[max(0,k-1)], grid[min(points,k+1)]),
                             method="bounded", options={"xatol": 1e-14})
    t = float(result.x)
    rate, pp, pm, q, null = (float(x) for x in pair_rate(t,a,b,c))
    z, y = math.cos(t), math.sin(t)
    log_ratio = math.log(pp/(1-pp))-math.log(null/(1-null))
    slope = (b-c*z/y)*log_ratio*null*(1-null)/(pp-null)
    intercept = null-slope*z

    angles = np.linspace(-math.pi, math.pi, 2*points+1)
    cosine = np.cos(angles)
    probability = a+b*cosine+c*np.sin(angles)
    null_line = intercept+slope*cosine
    tangent_physical = 0 < intercept-abs(slope) and intercept+abs(slope) < 1
    if tangent_physical:
        tangent_max = float(np.max(bernoulli_kl(probability,null_line)))
    else:
        tangent_max = None
    scale = max(rate,1e-14)
    start = np.array([a,b,float(np.max(bernoulli_kl(probability,a+b*cosine)))/scale])
    def constraints(v):
        new_p = v[0]+v[1]*cosine
        if np.min(new_p) <= 0 or np.max(new_p) >= 1:
            new_p = np.clip(new_p,1e-14,1-1e-14)
        return v[2]-bernoulli_kl(probability,new_p)/scale
    def jac(v):
        new_p = np.clip(v[0]+v[1]*cosine,1e-14,1-1e-14)
        dq = (new_p-probability)/(new_p*(1-new_p))
        return np.column_stack([-dq/scale,-dq*cosine/scale,np.ones(len(cosine))])
    physical = {"type": "ineq",
                "fun": lambda v: np.array([v[0]-v[1]-1e-10,v[0]+v[1]-1e-10,
                                            1-1e-10-v[0]+v[1],1-1e-10-v[0]-v[1]]),
                "jac": lambda v: np.array([[1.,-1.,0.],[1.,1.,0.],[-1.,1.,0.],[-1.,-1.,0.]])}
    solved = minimize(lambda v:v[2],start,jac=lambda v:np.array([0.,0.,1.]),
                      method="SLSQP",constraints=[{"type":"ineq","fun":constraints,"jac":jac},physical],
                      options={"ftol":1e-10,"maxiter":500})
    optimized = float(np.max(bernoulli_kl(probability,solved.x[0]+solved.x[1]*cosine)))
    return {"A":a,"B":b,"C":c,"two_angle":t,"plus_allocation":q,"pair_rate":rate,
            "tangent_null":[intercept,slope],"tangent_physical":tangent_physical,
            "tangent_max_over_pair":tangent_max/rate if tangent_max is not None else None,
            "grid_minimax_over_pair":optimized/rate,
            "grid_minimax_null":list(solved.x[:2]),"optimizer_success":bool(solved.success),
            "optimizer_message":str(solved.message)}


def main():
    rows=[]
    for a in (.01,.1,.3,.5,.8):
        for rho in (.5,.98,.9999):
            for fraction in (0.,.5,.9,.995):
                radius=rho*min(a,1-a)
                rows.append(probe(a,radius*fraction,radius*math.sqrt(1-fraction*fraction)))
    result={"status":"heuristic finite-grid development probe",
            "cases":len(rows),"rows":rows,
            "max_grid_minimax_over_pair":max(x["grid_minimax_over_pair"] for x in rows),
            "max_tangent_over_pair":max(x["tangent_max_over_pair"] for x in rows if x["tangent_physical"]),
            "nonphysical_tangents":sum(not x["tangent_physical"] for x in rows),
            "failed_optimizers":sum(not x["optimizer_success"] for x in rows)}
    path=Path(__file__).resolve().parents[2]/"results"/"learned_validation"/"global_kl_probe.json"
    path.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))


if __name__=="__main__":
    main()
