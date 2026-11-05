"""Adaptive symmetric designs from paid observations only.

The state stores least-squares sufficient statistics of all completed pairs.
Different angles need not have the same mean. This is an estimated design,
not a confidence certificate; adaptive reuse is justified separately for the
paired evidence process. Every policy receives the same fitted state.
"""
from dataclasses import asdict, dataclass
import math
import numbers

import numpy as np


@dataclass(frozen=True)
class DesignConfig:
    ridge: float = 1.
    mean_clip: float = .0001
    radial_fraction: float = .98
    min_shift: float = .2
    grid_points: int = 513
    grid_batch_size: int = 128

    def __post_init__(self):
        values=(self.ridge,self.mean_clip,self.radial_fraction,self.min_shift)
        if not all(math.isfinite(x) for x in values):
            raise ValueError("configuration must be finite")
        if self.ridge<=0 or not 0<self.mean_clip<.5 or not 0<self.radial_fraction<1:
            raise ValueError("invalid regularization")
        if not 0<self.min_shift<math.pi/2:
            raise ValueError("invalid shift cap")
        if type(self.grid_points) is not int or self.grid_points<3 or self.grid_points%2!=1:
            raise ValueError("grid_points must be odd and at least three")
        if type(self.grid_batch_size) is not int or self.grid_batch_size<1:
            raise ValueError("grid_batch_size must be positive")


DEFAULT_CONFIG=DesignConfig()
POLICIES=("psr","paired_variance_closed","paired_variance_grid","paired_kl_grid","fisher_kl_grid")


def config_dict(config=DEFAULT_CONFIG):
    return asdict(config)


def empty_state(batch_size):
    """Create independent batched paid-count sufficient statistics."""
    if isinstance(batch_size,bool) or not isinstance(batch_size,numbers.Integral) or batch_size<1:
        raise ValueError("batch_size must be a positive integer")
    return {"gram":np.zeros((int(batch_size),3,3),dtype=float),
            "rhs":np.zeros((int(batch_size),3),dtype=float),
            "shots":np.zeros(int(batch_size),dtype=np.int64),
            "pairs":np.zeros(int(batch_size),dtype=np.int64),
            "blocks":np.zeros(int(batch_size),dtype=np.int64)}


def _indices(state,indices):
    size=len(state["shots"])
    if indices is None:
        return np.arange(size,dtype=int)
    result=np.asarray(indices)
    if result.ndim!=1 or not np.issubdtype(result.dtype,np.integer):
        raise ValueError("indices must be a one-dimensional integer array")
    if np.any(result<0) or np.any(result>=size) or len(np.unique(result))!=len(result):
        raise ValueError("indices must be unique and inside the state")
    return result.astype(int,copy=False)


def update_pairs(state,t,delta_pair_counts,indices=None):
    """Accumulate completed pairs; counts ordered 00,01,10,11.

    The angle t may be scalar or one value per selected row. It must be
    selected before both outcomes in its block. No observed pair can be
    reassigned to a different angle after sampling.
    """
    selected=_indices(state,indices)
    counts=np.asarray(delta_pair_counts)
    if counts.shape!=(len(selected),4):
        raise ValueError("pair counts must have shape (selected rows,4)")
    if not np.all(np.isfinite(counts)) or np.any(counts<0) or np.any(counts!=np.floor(counts)):
        raise ValueError("pair counts must be finite nonnegative integers")
    counts=counts.astype(np.int64)
    angles=np.broadcast_to(np.asarray(t,dtype=float),(len(selected),))
    if not np.all(np.isfinite(angles)) or np.any(angles<=0) or np.any(angles>=math.pi):
        raise ValueError("angles must lie strictly between zero and pi")
    number=counts.sum(axis=1)
    plus=counts[:,1]+counts[:,3]
    minus=counts[:,2]+counts[:,3]
    cosine,sine=np.cos(angles),np.sin(angles)
    gram=np.zeros((len(selected),3,3))
    gram[:,0,0]=2*number
    gram[:,0,1]=gram[:,1,0]=2*number*cosine
    gram[:,1,1]=2*number*cosine*cosine
    gram[:,2,2]=2*number*sine*sine
    rhs=np.column_stack((plus+minus,(plus+minus)*cosine,(plus-minus)*sine))
    state["gram"][selected]+=gram
    state["rhs"][selected]+=rhs
    state["shots"][selected]+=2*number
    state["pairs"][selected]+=number
    state["blocks"][selected]+=(number>0).astype(np.int64)
    return state


def fit_state(state,indices=None,config=DEFAULT_CONFIG):
    """Fixed-ridge least squares, then strict physical projection.

    One ridge unit is a public regularizer toward (.5,0,0), not a paid
    observation. A singular design, such as PSR alone before exploration,
    therefore keeps the unidentifiable cosine coefficient at its prior.
    """
    selected=_indices(state,indices)
    gram=state["gram"][selected]
    rhs=state["rhs"][selected].copy()
    rhs[:,0]+=.5*config.ridge
    raw=np.linalg.solve(gram+config.ridge*np.eye(3)[None,:,:],rhs)
    a=np.clip(raw[:,0],config.mean_clip,1-config.mean_clip)
    radius=np.hypot(raw[:,1],raw[:,2])
    maximum=config.radial_fraction*np.minimum(a,1-a)
    scale=np.minimum(1.,maximum/np.maximum(radius,np.finfo(float).tiny))
    b,c=raw[:,1]*scale,raw[:,2]*scale
    return {"A":a,"B":b,"C":c,"raw_A":raw[:,0],"raw_B":raw[:,1],"raw_C":raw[:,2],
            "radial_scale":scale,"mean_was_clipped":a!=raw[:,0],
            "radius_was_clipped":scale<1,"shots":state["shots"][selected].copy(),
            "pairs":state["pairs"][selected].copy(),
            "fitted_probability_min":a-np.hypot(b,c),
            "fitted_probability_max":a+np.hypot(b,c)}


def uniform_variance_shift(fit,config=DEFAULT_CONFIG):
    """Exact uniform-pair variance optimum, then the public shift cap.

    For z=cos(t), the t-dependent part is
      [D+(1-2A)B*z-B^2*z^2]/(1-z^2).
    Its derivative numerator is (1-2A)B*(1+z^2)+2(D-B^2)z.
    The physical interior implies D-B^2>|(1-2A)B|, and hence a unique
    interior minimum. The subtraction of C^2 is constant in t.
    """
    a,b=fit["A"],fit["B"]
    d=a*(1-a)-b*b
    linear=(1-2*a)*b
    discriminant=np.sqrt(np.maximum(0.,(d-linear)*(d+linear)))
    z=-linear/(d+discriminant)
    raw=np.arccos(np.clip(z,-1.,1.))
    return np.clip(raw,config.min_shift,math.pi-config.min_shift),raw


def half_coin_kl_from_bias(bias):
    """Stable kl((1+bias)/2 || 1/2), including the zero-bias limit."""
    x=np.asarray(bias,dtype=float)
    if np.any(np.abs(x)>1):
        raise ValueError("Bernoulli bias lies outside [-1,1]")
    x2=x*x
    series=x2*(.5+x2*(1/12+x2*(1/30+x2*(1/56+x2/90))))
    up=np.maximum(1+x,np.finfo(float).tiny)
    down=np.maximum(1-x,np.finfo(float).tiny)
    exact=.5*((1+x)*np.log(up)+(1-x)*np.log(down))
    return np.where(np.abs(x)<.01,series,exact)


def pair_scores(p_plus,p_minus,sine):
    """Uniform-pair variance and fixed-angle Beta-mixture KL rate.

    The rate is per physical Bernoulli query, not per pair. Changing angles
    and pooling a common mixture do not automatically attain the weighted
    average of the individual fixed-angle rates.
    """
    pp,pm,sn=np.broadcast_arrays(np.asarray(p_plus,dtype=float),np.asarray(p_minus,dtype=float),
                                 np.asarray(sine,dtype=float))
    variance=2*(pp*(1-pp)+pm*(1-pm))/(sn*sn)
    worse=(1-pm)*pp
    better=pm*(1-pp)
    discordance=worse+better
    bias=(pp-pm)/discordance
    rate=.5*discordance*half_coin_kl_from_bias(np.clip(bias,-1.,1.))
    return variance,rate,discordance


def fisher_rate(p_plus,p_minus):
    """Uniform two-sample KL rate per physical query, with stable cancellation.

    Let m=(p+ + p-)/2 and d=(p+ - p-)/2. The symmetric Bernoulli KL average
    equals m*h(d/m)+(1-m)*h(d/(1-m)), with h(x)=kl((1+x)/2 || 1/2).
    Unlike the paired-discordance rate, this uses the two complete marginals.
    """
    pp,pm=np.broadcast_arrays(np.asarray(p_plus,dtype=float),np.asarray(p_minus,dtype=float))
    center=(pp+pm)/2
    difference=(pp-pm)/2
    return (center*half_coin_kl_from_bias(np.clip(difference/center,-1.,1.))
            +(1-center)*half_coin_kl_from_bias(np.clip(difference/(1-center),-1.,1.)))


def _at_shift(fit,t):
    center=fit["A"]+fit["B"]*np.cos(t)
    difference=fit["C"]*np.sin(t)
    pp,pm=center+difference,center-difference
    variance,rate,discordance=pair_scores(pp,pm,np.sin(t))
    return {"t":np.asarray(t),"estimated_p_plus":pp,"estimated_p_minus":pm,
            "estimated_variance":variance,"estimated_pair_rate":rate,
            "estimated_fisher_rate":fisher_rate(pp,pm),
            "paired_difference":pp-pm,"discordance_probability":discordance,
            "allocation_plus":np.full_like(pp,.5)}


def _grid_shift(fit,policy,config):
    grid=np.linspace(config.min_shift,math.pi-config.min_shift,config.grid_points)
    grid[config.grid_points//2]=math.pi/2
    sine,cosine=np.sin(grid)[None,:],np.cos(grid)[None,:]
    count=len(fit["A"])
    result=np.empty(count)
    for begin in range(0,count,config.grid_batch_size):
        end=min(count,begin+config.grid_batch_size)
        a,b,c=(fit[key][begin:end,None] for key in ("A","B","C"))
        center=a+b*cosine
        difference=c*sine
        variance,rate,unused=pair_scores(center+difference,center-difference,sine)
        if policy=="paired_variance_grid":
            k=np.argmin(variance,axis=1)
        else:
            score=fisher_rate(center+difference,center-difference) if policy=="fisher_kl_grid" else rate.copy()
            zero=(c[:,0]==0)
            score[zero]=sine*sine/(2*center[zero]*(1-center[zero]))
            k=np.argmax(score,axis=1)
        result[begin:end]=grid[k]
    return result


def choose_design(policy,state,indices=None,config=DEFAULT_CONFIG):
    """Select the next angle from history available before the next block."""
    if policy not in POLICIES:
        raise ValueError("unknown streaming design policy")
    fit=fit_state(state,indices,config)
    if policy=="psr":
        t=np.full_like(fit["A"],math.pi/2)
        raw=t
    elif policy=="paired_variance_closed":
        t,raw=uniform_variance_shift(fit,config)
    else:
        t=_grid_shift(fit,policy,config)
        raw=t
    result=_at_shift(fit,t)
    result.update({"policy":policy,"fit":fit,"raw_t":raw,"shift_was_clipped":t!=raw,
                   "shots":fit["shots"],"paired_only":True,
                   "variance_scope":"per physical query variance for estimating 2*C at a fixed selected angle",
                   "rate_scope":"fixed-angle discordant-sign mixture asymptotic rate per physical query"})
    return result
