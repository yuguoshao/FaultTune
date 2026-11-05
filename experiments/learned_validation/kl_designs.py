"""Paid-pilot symmetric two-angle designs maximizing Bernoulli null KL.

This is a finite-menu plug-in specialization of established structured
identification design, not a claim of globally optimal three-parameter
sequential identification. The independent confirmation certificate must
still authorize every decision. Pilot counts are the only data inputs.
"""

import math

import numpy as np

from designs import DEFAULT_CONFIG, fit_pilot


def _entropy(p):
    return -p*np.log(p)-(1-p)*np.log1p(-p)


def _remainder(x):
    """Stable (1+x) log(1+x)-x, for x > -1."""
    small=np.abs(x)<.001
    # Terms through order eight; the relative remainder is O(x**7).
    series=x*x*(.5+x*(-1/6+x*(1/12+x*(-1/20+x*(1/30+x*(-1/42+x/56))))))
    return np.where(small,series,(1+x)*np.log1p(x)-x)


def bernoulli_kl(p,q):
    """Stable KL for strictly interior Bernoulli probabilities."""
    p,q=np.broadcast_arrays(np.asarray(p,dtype=float),np.asarray(q,dtype=float))
    e=p-q
    return q*_remainder(e/q)+(1-q)*_remainder(-e/(1-q))


def optimal_kl_allocation(p_plus,p_minus):
    """Maximize q KL(p+||r)+(1-q) KL(p-||r), r=q p++(1-q)p-.

    At equality use the continuous allocation limit 1/2. A cubic series
    avoids subtracting two close entropy values and two close probabilities.
    Unlike Neyman allocation, this maximizes a fixed-gap testing rate.
    """
    pp,pm=np.broadcast_arrays(np.asarray(p_plus,dtype=float),np.asarray(p_minus,dtype=float))
    if np.any(pp<=0) or np.any(pp>=1) or np.any(pm<=0) or np.any(pm>=1):
        raise ValueError("KL design requires strictly interior probabilities")
    center=(pp+pm)/2
    d=(pp-pm)/2
    v=center*(1-center)
    ell=1-2*center
    small=np.abs(d)<=.001*np.minimum(center,1-center)
    q=np.empty_like(center)
    q[small]=(.5-ell[small]*d[small]/(12*v[small])
              +ell[small]*(16*v[small]-13)*d[small]**3/(720*v[small]**3))
    if np.any(~small):
        slope=(_entropy(pp[~small])-_entropy(pm[~small]))/(2*d[~small])
        r=1/(1+np.exp(slope))
        q[~small]=(r-pm[~small])/(2*d[~small])
    # Numerical roundoff is bounded; clipping also protects future callers.
    return np.clip(q,0.,1.)


def _at_grid(a,b,c,grid,config):
    sine,cosine=np.sin(grid)[None,:],np.cos(grid)[None,:]
    center=a[:,None]+b[:,None]*cosine
    difference=c[:,None]*sine
    pp,pm=center+difference,center-difference
    raw_q=optimal_kl_allocation(pp,pm)
    q=np.clip(raw_q,config.min_allocation,1-config.min_allocation)
    null=q*pp+(1-q)*pm
    rate=q*bernoulli_kl(pp,null)+(1-q)*bernoulli_kl(pm,null)
    # Exactly zero fitted C gives a zero objective at every angle. Its
    # continuous local limit selects the informative rather than endpoint
    # design; this uses only the fitted model and is part of the policy.
    zero=c==0
    score=rate.copy()
    score[zero]=sine*sine/(2*center[zero]*(1-center[zero]))
    return pp,pm,q,raw_q,rate,score


def choose_kl_design(pilot_counts,pilot_totals,config=DEFAULT_CONFIG):
    """Choose angle and allocation from the same paid fit and public caps."""
    fit=fit_pilot(pilot_counts,pilot_totals,config)
    shape=np.shape(fit["A"])
    a,b,c=(np.asarray(fit[key]).reshape(-1) for key in ("A","B","C"))
    grid=np.linspace(config.min_shift,math.pi-config.min_shift,config.grid_points)
    grid[config.grid_points//2]=math.pi/2
    t=np.empty_like(a)
    plus=np.empty_like(a)
    minus=np.empty_like(a)
    fraction=np.empty_like(a)
    raw_fraction=np.empty_like(a)
    kl_rate=np.empty_like(a)
    for begin in range(0,len(a),config.grid_batch_size):
        end=min(len(a),begin+config.grid_batch_size)
        pp,pm,q,raw_q,rate,score=_at_grid(a[begin:end],b[begin:end],c[begin:end],grid,config)
        k=np.argmax(score,axis=-1)
        j=np.arange(end-begin)
        t[begin:end]=grid[k]
        plus[begin:end],minus[begin:end]=pp[j,k],pm[j,k]
        fraction[begin:end],raw_fraction[begin:end]=q[j,k],raw_q[j,k]
        kl_rate[begin:end]=rate[j,k]
    variance=(plus*(1-plus)/fraction+minus*(1-minus)/(1-fraction))/np.sin(t)**2
    def shaped(x):
        return x.reshape(shape)
    return {"policy":"learned_kl_grid","fit":fit,"pilot_shots":fit["pilot_shots"],
            "t":shaped(t),"raw_t":shaped(t.copy()),
            "allocation_plus":shaped(fraction),"unclipped_kl_plus":shaped(raw_fraction),
            "estimated_p_plus":shaped(plus),"estimated_p_minus":shaped(minus),
            "estimated_variance":shaped(variance),"estimated_kl_rate":shaped(kl_rate),
            "shift_was_clipped":np.zeros(shape,dtype=bool),
            "allocation_was_clipped":shaped(fraction!=raw_fraction),
            "zero_gap_local_limit_used":shaped(c==0),
            "estimated_variance_normalization":"per confirmation shot for 2*C; multiply by sin(h/2)^2 for task Delta; pilot cost is separate and must also be paid",
            "kl_scope":"two symmetric angles, equality-null KL, fixed fitted coefficients, public finite grid; not full structured all-angle identification optimum"}
