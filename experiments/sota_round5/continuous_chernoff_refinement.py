"""Resolve inaccurate entropy quadrature in the initial floating probe.

Preserves the initial source/results and uses a closed entropy secant except
where cancellation favors a well-conditioned short-interval quadrature.
No continuous optimum or statistical stopping guarantee is asserted.
"""
import hashlib
import json
import math
from pathlib import Path
import sys

import mpmath as mp
import numpy as np
from scipy.special import expit,logit

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.sota_round5 import continuous_chernoff_probe as initial


def center(pm,pp):
    if not 0<pm<pp<1: raise ValueError('Interior ordered laws required')
    width=pp-pm
    if width < .01*min(pm,1-pp):
        secant=float(initial.GL_W.dot(logit((pm+pp)/2+width*initial.GL_X/2))/2)
    else:
        secant=math.fsum([pp*math.log(pp),(1-pp)*math.log1p(-pp),
                          -pm*math.log(pm),-(1-pm)*math.log1p(-pm)])/width
    return float(expit(secant))


def radius(pm,pp):
    q=center(pm,pp)
    weight=(q-pm)/(pp-pm)
    left,right=map(float,initial.design.bernoulli_kl(np.array([pm,pp]),q))
    return {'radius':weight*right+(1-weight)*left,'center':q,'weight_plus':weight,
            'equal_kl_residual':abs(left-right)}


def main():
    path=ROOT/'results/sota_round5/continuous_chernoff_refinement.json'
    if path.exists(): raise FileExistsError('Preserve prior refinement')
    oldpath=ROOT/'results/sota_round5/continuous_chernoff_probe.json'
    old=json.loads(oldpath.read_text(encoding='utf-8'))
    initial.radius_and_center=radius
    mp.mp.dps=80
    results=[]; discrepancy=0.
    for row in old['rows']:
        theta=[row['model'][key] for key in ('A','B','C')]
        pair=initial.reflected_pair(theta)
        a,b,c=theta; x=pair['angle']
        pm_,pp_=a+b*math.cos(x)-c*math.sin(x),a+b*math.cos(x)+c*math.sin(x)
        pm,pp=mp.mpf(pm_),mp.mpf(pp_)
        f=lambda p:p*mp.log(p)+(1-p)*mp.log1p(-p)
        reference=1/(1+mp.exp(-(f(pp)-f(pm))/(pp-pm)))
        error=float(abs(mp.mpf(pair['center'])-reference))
        discrepancy=max(discrepancy,error)
        assert error < 2e-13
        results.append({'model':row['model'],'reflected_pair':pair,
                        'finite_grid_lower_over_refined_pair':row['general_finite_grid']['info_lower']/pair['radius'],
                        'center_high_precision_absolute_error':error})
    result={'status':'complete_refined_float_diagnostic','new_simulation_observations':0,
            'initial_numerical_issue':'A fixed40-point quadrature of logit over an interval reaching near zero introduced center/equal-KL error. It caused the initial dense residual around2.6e-4; this was a numerical artifact, not a design counterexample.',
            'independent_center_checks':{'precision_digits':80,'models':len(results),'maximum_absolute_error':discrepancy},
            'maximum_grid_lower_over_pair':max(r['finite_grid_lower_over_refined_pair'] for r in results),
            'maximum_dense_upper_over_pair':max(r['reflected_pair']['dense_upper_over_pair_radius'] for r in results),
            'rows':results,
            'source_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                (Path(__file__).resolve(),ROOT/'experiments/sota_round5/continuous_chernoff_probe.py',oldpath)},
            'claim_boundary':'No robust numerical counterexample in the retained48-point registry would not prove continuous two-point optimality. All grid, optimization and cost implications remain exploratory. This conjecture does not evade the current threefold information obstruction.'}
    with path.open('x',encoding='utf-8') as f: json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('status','independent_center_checks','maximum_grid_lower_over_pair','maximum_dense_upper_over_pair')},indent=2))


if __name__=='__main__': main()
