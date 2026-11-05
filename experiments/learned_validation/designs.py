"""Vectorized measurement designs using ONLY paid four-angle pilot counts.

Pilot axes, relative to the fixed task midpoint, are 0, pi/2, pi, 3*pi/2.
All four policies share the same smoothing, sinusoid fit, physical clipping,
shift caps, and allocation caps.  Coefficients are estimates, not certificates.
The runner must pay for every pilot shot and use independent confirmation
observations after choosing the design.  No true probability is an input here.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np


PILOT_ANGLES = np.array([0., math.pi/2, math.pi, 3*math.pi/2])
POLICIES = ("psr_uniform", "learned_psr_neyman", "learned_closed_form", "learned_grid_design")


@dataclass(frozen=True)
class DesignConfig:
    pseudocount: float = .5
    mean_clip: float = .001
    radial_fraction: float = .98
    min_shift: float = .2
    min_allocation: float = .1
    grid_points: int = 513
    grid_batch_size: int = 256

    def __post_init__(self):
        scalars = (self.pseudocount,self.mean_clip,self.radial_fraction,self.min_shift,self.min_allocation)
        if not all(math.isfinite(v) for v in scalars):
            raise ValueError("configuration scalars must be finite")
        if self.pseudocount <= 0 or not 0 < self.mean_clip < .5 or not 0 < self.radial_fraction < 1:
            raise ValueError("invalid smoothing or physical-fit regularization")
        if not 0 < self.min_shift < math.pi/2 or not 0 < self.min_allocation <= .5:
            raise ValueError("invalid shift or allocation cap")
        if type(self.grid_points) is not int or self.grid_points < 3 or self.grid_points % 2 != 1:
            raise ValueError("grid_points must be an odd integer >=3")
        if type(self.grid_batch_size) is not int or self.grid_batch_size < 1:
            raise ValueError("grid_batch_size must be a positive integer")


DEFAULT_CONFIG = DesignConfig()


def config_dict(config=DEFAULT_CONFIG):
    return asdict(config)


def _pilot_arrays(counts, totals):
    counts,totals=np.broadcast_arrays(np.asarray(counts,dtype=float),np.asarray(totals,dtype=float))
    if counts.ndim < 1 or counts.shape[-1] != 4:
        raise ValueError("pilot counts/totals must broadcast to (...,4)")
    if not np.all(np.isfinite(counts)) or not np.all(np.isfinite(totals)):
        raise ValueError("pilot counts/totals must be finite")
    if np.any(counts<0) or np.any(totals<0) or np.any(counts>totals):
        raise ValueError("require 0 <= counts <= totals")
    if np.any(counts!=np.floor(counts)) or np.any(totals!=np.floor(totals)):
        raise ValueError("pilot counts and totals must be integers")
    return counts,totals


def fit_pilot(pilot_counts, pilot_totals, config=DEFAULT_CONFIG):
    """Fit A,B,C and radially contract to a strictly physical sinusoid.

    This deterministic least-squares fit uses the four smoothed empirical
    probabilities with equal angular weights, even when per-angle totals
    differ.  Pseudocounts are regularization and are NOT counted as paid shots.
    Empty or very small pilots are legal; caps keep all resulting designs safe
    to execute, while their statistical usefulness remains for the runner to
    assess with independent confirmation data.
    """
    counts,totals=_pilot_arrays(pilot_counts,pilot_totals)
    p=(counts+config.pseudocount)/(totals+2*config.pseudocount)
    raw_a=np.mean(p,axis=-1)
    raw_b=(p[...,0]-p[...,2])/2
    raw_c=(p[...,1]-p[...,3])/2
    a=np.clip(raw_a,config.mean_clip,1-config.mean_clip)
    radius=np.hypot(raw_b,raw_c)
    allowed=config.radial_fraction*np.minimum(a,1-a)
    radial_scale=np.minimum(1.,allowed/np.maximum(radius,np.finfo(float).tiny))
    b,c=raw_b*radial_scale,raw_c*radial_scale
    fitted_radius=np.hypot(b,c)
    return {"A":a,"B":b,"C":c,"raw_A":raw_a,"raw_B":raw_b,"raw_C":raw_c,
            "pilot_probabilities":p,"pilot_shots":np.sum(totals,axis=-1),
            "mean_was_clipped":a!=raw_a,"radius_was_clipped":radial_scale<1,
            "radial_scale":radial_scale,"fitted_probability_min":a-fitted_radius,
            "fitted_probability_max":a+fitted_radius}


def _probabilities(fit,t):
    midpoint=fit["A"]+fit["B"]*np.cos(t)
    difference=fit["C"]*np.sin(t)
    return midpoint+difference,midpoint-difference


def _at_shift(fit,t,config,uniform=False):
    plus,minus=_probabilities(fit,t)
    var_plus=plus*(1-plus)
    var_minus=minus*(1-minus)
    sigma_plus,sigma_minus=np.sqrt(var_plus),np.sqrt(var_minus)
    neyman=sigma_plus/(sigma_plus+sigma_minus)
    fraction=np.full_like(neyman,.5) if uniform else np.clip(neyman,config.min_allocation,1-config.min_allocation)
    variance=(var_plus/fraction+var_minus/(1-fraction))/np.sin(t)**2
    return {"t":np.asarray(t),"allocation_plus":fraction,"estimated_p_plus":plus,
            "estimated_p_minus":minus,"estimated_variance":variance,
            "unclipped_neyman_plus":neyman,"allocation_was_clipped":np.zeros_like(neyman,dtype=bool)
                if uniform else fraction!=neyman}


def _closed_shift(fit,config):
    """Unconstrained closed form followed by public caps.

    Clipping the shift and then the Neyman fraction does not claim to solve
    the constrained optimization problem.  The grid baseline explicitly
    minimizes its capped objective and may therefore improve on this rule.
    """
    a,b,c=fit["A"],fit["B"],fit["C"]
    radius2=b*b+c*c
    determinant=np.sqrt(np.maximum(0.,(a*a-radius2)*((1-a)**2-radius2)))
    # w=(A(1-A)-R^2+S)/2 avoids subtracting B^2 from u afterward.
    w=(a*(1-a)-radius2+determinant)/2
    cosine=-(1-2*a)*b/(2*w)
    raw_shift=np.arccos(np.clip(cosine,-1.,1.))
    shift=np.clip(raw_shift,config.min_shift,math.pi-config.min_shift)
    return shift,raw_shift


def _grid_shift(fit,config):
    shape=np.shape(fit["A"])
    a,b,c=(np.asarray(fit[key]).reshape(-1) for key in ("A","B","C"))
    grid=np.linspace(config.min_shift,math.pi-config.min_shift,config.grid_points)
    grid[config.grid_points//2]=math.pi/2
    cosine,sine=np.cos(grid)[None,:],np.sin(grid)[None,:]
    result=np.empty(len(a),dtype=float)
    for begin in range(0,len(a),config.grid_batch_size):
        end=min(len(a),begin+config.grid_batch_size)
        center=a[begin:end,None]+b[begin:end,None]*cosine
        difference=c[begin:end,None]*sine
        plus,minus=center+difference,center-difference
        var_plus,var_minus=plus*(1-plus),minus*(1-minus)
        splus,sminus=np.sqrt(var_plus),np.sqrt(var_minus)
        fraction=np.clip(splus/(splus+sminus),config.min_allocation,1-config.min_allocation)
        score=(var_plus/fraction+var_minus/(1-fraction))/sine**2
        # Exact ties choose the public grid order, not the hidden outcome sign.
        index=np.argmin(score,axis=-1)
        result[begin:end]=grid[index]
    return result.reshape(shape)


def _choose_from_fit(policy,fit,config):
    if policy not in POLICIES:
        raise ValueError("unknown policy: "+str(policy))
    shape=np.shape(fit["A"])
    if policy in ("psr_uniform","learned_psr_neyman"):
        t=np.full(shape,math.pi/2)
        raw=t
    elif policy=="learned_closed_form":
        t,raw=_closed_shift(fit,config)
    else:
        t=_grid_shift(fit,config)
        raw=t
    result=_at_shift(fit,t,config,uniform=policy=="psr_uniform")
    result.update({"policy":policy,"pilot_shots":fit["pilot_shots"],"fit":fit,
                   "raw_t":raw,"shift_was_clipped":t!=raw,
                   "estimated_variance_normalization":"per confirmation shot for 2*C; multiply by sin(h/2)^2 for task Delta; pilot cost is separate and must also be paid"})
    return result


def choose_design(policy,pilot_counts,pilot_totals,config=DEFAULT_CONFIG):
    """Return a design using only paid pilot counts; every array has batch shape.

    A zero-pilot PSR control supplies zero counts and totals and pays zero
    pilot shots. Learned methods using the same randomized pilot realization
    must each be charged its full pilot cost in their separate runs.
    """
    return _choose_from_fit(policy,fit_pilot(pilot_counts,pilot_totals,config),config)


def all_designs(pilot_counts,pilot_totals,config=DEFAULT_CONFIG):
    """Evaluate all four policies with one identical fit and identical pilot cost."""
    fit=fit_pilot(pilot_counts,pilot_totals,config)
    return {policy:_choose_from_fit(policy,fit,config) for policy in POLICIES}
