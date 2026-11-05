"""Deterministic test of a two-reflected-arm information-design conjecture.

All outputs are floating diagnostics, not continuous certificates. True model
parameters are used only by this offline information oracle, never by a paid
data policy. This does not claim SOTA, threefold savings, or a novel framework.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/structured_validation'))
import design_policy as design

GL_X, GL_W = np.polynomial.legendre.leggauss(40)


def radius_and_center(pm, pp):
    """Capacity of two Bernoulli laws: equal KL to their optimal mixture."""
    if not 0 < pm < pp < 1:
        raise ValueError('Strictly ordered interior Bernoulli laws required')
    # Integral of logit is the entropy secant. Quadrature avoids subtracting
    # nearly equal entropies at a weak signal; it is NOT an interval bound.
    middle = (pm+pp)/2
    secant = float(GL_W.dot(logit(middle+(pp-pm)*GL_X/2))/2)
    q = float(expit(secant))
    weight_plus = (q-pm)/(pp-pm)
    dminus, dplus = map(float, design.bernoulli_kl(np.array([pm, pp]), q))
    return {'radius': weight_plus*dplus+(1-weight_plus)*dminus,
            'center': q, 'weight_plus': weight_plus,
            'equal_kl_residual': abs(dplus-dminus)}


def reflected_pair(theta):
    a,b,c = map(float, theta)
    c = abs(c)
    def at(angle):
        middle = a+b*math.cos(angle)
        gap = c*math.sin(angle)
        return radius_and_center(middle-gap, middle+gap)
    mesh = np.linspace(1e-5, math.pi-1e-5, 513)
    radii = np.array([at(float(x))['radius'] for x in mesh])
    maxima = [j for j in range(1, len(mesh)-1)
              if radii[j]>=radii[j-1] and radii[j]>=radii[j+1]]
    solutions = []
    for j in maxima:
        result = minimize_scalar(lambda x:-at(float(x))['radius'],
                                 bounds=(mesh[j-1],mesh[j+1]),method='bounded',
                                 options={'xatol':1e-13,'maxiter':200})
        solutions.append((float(result.x), at(float(result.x))))
    if not solutions:
        raise ArithmeticError('No interior radius maximum found')
    angle, record = max(solutions, key=lambda x:x[1]['radius'])
    co,si = math.cos(angle),math.sin(angle)
    pm,pp = a+b*co-c*si,a+b*co+c*si
    q = record['center']
    slopes=[]
    for p,derivative in ((pm,-b*si-c*co),(pp,-b*si+c*co)):
        dp = float(logit(p)-logit(q))
        dq = (q-p)/(q*(1-q))
        slopes.append(dp*derivative/(dq*si))
    bb = sum(slopes)/2
    aa = q-bb*co
    margin = min(aa,1-aa)-abs(bb)
    record.update({'angle':angle,'cosine':co,'null_candidate':[aa,bb,0.],
                   'null_physical_margin':margin,'slope_disagreement':abs(slopes[0]-slopes[1]),
                   'stationary_candidates_checked':len(solutions)})
    if margin > 0:
        full = np.linspace(-math.pi, math.pi, 8193)
        p = a+b*np.cos(full)+c*np.sin(full)
        q = aa+bb*np.cos(full)
        divergences = design.bernoulli_kl(p,q)
        maximum = float(divergences.max())
        record.update({'dense_circle_null_kl_max':maximum,
                       'dense_upper_over_pair_radius':maximum/record['radius'],
                       'maximizing_grid_angle':float(full[int(divergences.argmax())]),
                       'continuous_saddle_proved':False})
    return record


def main():
    target=ROOT/'results/sota_round5/continuous_chernoff_probe.json'
    if target.exists():
        raise FileExistsError('Preserve existing diagnostic; choose a new version')
    started=time.perf_counter()
    rows=[]
    # All points registered before any value is evaluated; no sign filtering.
    registry=[{'A':a,'rho':rho,'B_over_R':cosine}
              for a in (.01,.05,.2,.5) for rho in (.5,.98,.9999)
              for cosine in (0.,.5,.9,.995)]
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'registry':registry,
          'purpose':'Counterexample search for reflected-two-arm continuous KL minimax design.',
          'new_simulation_observations':0,'scope':'Floating deterministic information diagnostics only.'}
    target.parent.mkdir(parents=True,exist_ok=True)
    planpath=target.with_name('continuous_chernoff_probe_plan.json')
    with planpath.open('x',encoding='utf-8') as f: json.dump(plan,f,indent=2)
    grid=design.rational_grid(129,max_denominator=1024)
    for i,point in enumerate(registry):
        a,rho,u=point['A'],point['rho'],point['B_over_R']
        r=rho*min(a,1-a)
        theta=[a,r*u,r*math.sqrt(1-u*u)]
        pair=reflected_pair(theta)
        general=design.design_from_fit(theta,grid=grid)
        rows.append({'model':dict(A=theta[0],B=theta[1],C=theta[2]),
                     'registry_point':point,'reflected_pair':pair,
                     'general_finite_grid':{k:general[k] for k in
                      ('info_lower','info_upper','relative_gap','support','null_candidate','solver_status')},
                     'finite_grid_lower_over_pair':general['info_lower']/pair['radius']})
        if (i+1)%12==0:
            print(json.dumps({'completed':i+1,'total':len(registry),'seconds':time.perf_counter()-started}),flush=True)
    result={'status':'complete_float_counterexample_search','rows':rows,
            'new_simulation_observations':0,'elapsed_seconds':time.perf_counter()-started,
            'diagnostics':{'maximum_grid_lower_over_reflected_pair':max(r['finite_grid_lower_over_pair'] for r in rows),
                'maximum_dense_upper_over_reflected_pair':max(r['reflected_pair'].get('dense_upper_over_pair_radius',float('inf')) for r in rows),
                'infeasible_tangent_nulls':sum(r['reflected_pair']['null_physical_margin']<=0 for r in rows),
                'finite_grid_detected_counterexamples_at_1e_minus5_relative':sum(r['finite_grid_lower_over_pair']>1.00001 for r in rows)},
            'source_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (Path(__file__).resolve(),ROOT/'experiments/structured_validation/design_policy.py',planpath)},
            'claim_boundary':'A dense grid cannot prove a continuous saddle. Entropy quadrature, optimization and convex residuals here are floating diagnostics. The proposed two-arm information design is a conjecture, not an algorithm with paid learning or certified finite-cost superiority.'}
    with target.open('x',encoding='utf-8') as f: json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result['diagnostics'],indent=2))


if __name__=='__main__':
    main()
