"""Reproduce the 1,000-point heuristic affine-null stress check.

No result in this file certifies a continuum inequality. Existing output is
compared byte for byte rather than overwritten.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from global_kl_probe import pair_rate
from kl_designs import bernoulli_kl


def main():
    rng=np.random.RandomState(319417)
    rows=[]
    for i in range(1000):
        a=1/(1+math.exp(rng.uniform(-9,9)))
        radius=min(a,1-a)*(1-10**rng.uniform(-8,-.01))
        phi=rng.uniform(.001,math.pi-.001)
        b,c=radius*math.cos(phi),radius*math.sin(phi)
        grid=np.linspace(1e-6,math.pi-1e-6,1025)
        values=pair_rate(grid,a,b,c)[0]
        k=int(np.argmax(values))
        opt=minimize_scalar(lambda t:-float(pair_rate(t,a,b,c)[0]),
                            bounds=(grid[max(0,k-1)],grid[min(1024,k+1)]),
                            method="bounded",options={"xatol":1e-14})
        t=opt.x
        rate,pp,pm,q,null=(float(v) for v in pair_rate(t,a,b,c))
        z,y=math.cos(t),math.sin(t)
        logratio=math.log(pp/(1-pp))-math.log(null/(1-null))
        slope=(b-c*z/y)*logratio*null*(1-null)/(pp-null)
        intercept=null-slope*z
        ang=np.linspace(-math.pi,math.pi,16385)
        prob=a+b*np.cos(ang)+c*np.sin(ang)
        null_line=intercept+slope*np.cos(ang)
        physical=min(null_line)>0 and max(null_line)<1
        ratio=float(np.max(bernoulli_kl(prob,null_line)))/rate if physical else None
        rows.append({"A":a,"B":b,"C":c,"rate":rate,"t":float(t),
                     "tangent":[intercept,slope],"physical":bool(physical),"max_ratio":ratio})
    result={"status":"heuristic random continuum-grid tangent check","seed":319417,
            "cases":1000,"grid_points":16385,"nonphysical":sum(not r["physical"] for r in rows),
            "max_ratio":max(r["max_ratio"] for r in rows if r["physical"]),"rows":rows}
    path=Path(__file__).resolve().parents[2]/"results"/"learned_validation"/"global_kl_random_probe.json"
    data=json.dumps(result,indent=2)
    if path.exists():
        assert path.read_text(encoding="utf8")==data, "Existing run differs; preserve it for investigation"
    else:
        path.write_text(data,encoding="utf8")
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))


if __name__=="__main__":
    main()
