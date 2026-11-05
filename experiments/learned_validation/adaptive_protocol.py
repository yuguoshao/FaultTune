"""Count-only routing for a paid early-exit/learned-design validator.

Risk proof: early PSR uses an anytime-valid paired mixture at risk .005.
Conditional on the entire routing and pilot history, new holdout observations
are independent at pilot-frozen angles/allocation, so finite-look Fisher has
risk at most .005. Union bound gives at most .01 for a wrong decision,
including either direction at equality. Selecting hard pilots by survival
does not invalidate conditional holdout safety. No power guarantee follows.

The early PSR counts may be reused for DESIGN, never as holdout evidence.
All four wrapped policies receive precisely the same routing privilege.
"""

import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"circuit_validation"))
from direct_tests import certificate_from_pairs
from designs import DEFAULT_CONFIG, choose_design
from kl_designs import choose_kl_design

EARLY_RISK=.005
HOLDOUT_RISK=.005
EARLY_MAX_SHOTS=512
PILOT_PER_ANGLE=256
POLICIES=("learned_psr_neyman","learned_closed_form","learned_grid_design","learned_kl_grid")


def early_schedule():
    result=[]
    shots=32
    while shots<EARLY_MAX_SHOTS:
        result.append(shots)
        shots=2*int(math.ceil(shots*1.125/2))
    result.append(EARLY_MAX_SHOTS)
    return result


def _counts(values, width, name):
    result=np.asarray(values)
    if result.ndim<1 or result.shape[-1]!=width:
        raise ValueError(name+" must have last dimension "+str(width))
    if not np.all(np.isfinite(result)) or np.any(result<0) or np.any(result!=np.floor(result)):
        raise ValueError(name+" must contain finite nonnegative integers")
    return result.astype(np.int64)


def early_route(pair_counts):
    """One public look; pairs ordered old0/new0, old0/new1, old1/new0, old1/new1."""
    pairs=_counts(pair_counts,4,"pair counts")
    if pairs.ndim!=1:
        raise ValueError("early_route accepts a single transcript")
    shots=2*int(pairs.sum())
    if shots not in early_schedule():
        raise ValueError("early route must occur at a public early look")
    cert=certificate_from_pairs(tuple(int(x) for x in pairs),delta=EARLY_RISK)
    if cert["decision"]!="abstain":
        action="stop"
    else:
        action="complete_pilot" if shots==EARLY_MAX_SHOTS else "continue_early"
    return {"action":action,"decision":cert["decision"],"certificate":cert,
            "paid_shots":shots,"early_risk":EARLY_RISK,"holdout_risk":HOLDOUT_RISK,
            "total_risk_bound":EARLY_RISK+HOLDOUT_RISK}


def complete_pilot(pair_counts,axis_successes):
    """Combine full early pairs with newly paid 0/pi successes (256 each).

    Output ordering is 0, pi/2, pi, 3pi/2. The original paired transcript is
    preserved by the runner; this function supplies sufficient design counts.
    """
    pairs=_counts(pair_counts,4,"pair counts")
    axes=_counts(axis_successes,2,"axis successes")
    if axes.shape[:-1]!=pairs.shape[:-1]:
        raise ValueError("pilot batch dimensions must agree")
    if np.any(pairs.sum(axis=-1)!=PILOT_PER_ANGLE):
        raise ValueError("a hard pilot requires exactly 256 completed early pairs")
    if np.any(axes>PILOT_PER_ANGLE):
        raise ValueError("axis successes exceed paid trials")
    counts=np.stack((axes[...,0],pairs[...,1]+pairs[...,3],axes[...,1],pairs[...,2]+pairs[...,3]),axis=-1)
    return counts,np.full_like(counts,PILOT_PER_ANGLE)


def holdout_design(policy,pilot_counts,pilot_totals,config=DEFAULT_CONFIG):
    if policy not in POLICIES:
        raise ValueError("unknown wrapped policy")
    counts=_counts(pilot_counts,4,"pilot counts")
    totals=_counts(pilot_totals,4,"pilot totals")
    if counts.shape!=totals.shape or np.any(totals!=PILOT_PER_ANGLE) or np.any(counts>totals):
        raise ValueError("holdout design requires the complete paid four-angle pilot")
    if policy=="learned_kl_grid":
        return choose_kl_design(counts,totals,config)
    return choose_design(policy,counts,totals,config)
