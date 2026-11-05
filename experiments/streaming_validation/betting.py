"""Predictable paired betting with a fixed Kelly/fixed-bet hedge.

Both angles and both bets must be chosen before observing a block.  A block
contains complete symmetric pairs with a common target-sign orientation.
For positive-direction lambda in [0,1), the factor is 1+lambda on (0,1),
1-lambda on (1,0), and 1 on concordance. Its conditional expectation is
1+lambda*(P(0,1)-P(1,0)) <= 1 under the positive-direction null.
The negative direction reverses the factors. Predictable products and
fixed convex mixtures therefore retain the nonnegative-supermartingale
property. Threshold 2/delta in each direction controls their union by delta.

This is established betting/e-process machinery specialized to the paired
common-sign protocol, not a claim of a new universal statistical method.
Truth is never an input. The optional fitted probabilities are past-data
predictions; a caller must not pass hidden simulation probabilities.

All floating calculations below are prediction or screening only.
certificate_from_blocks recomputes every capital with outward Decimal
interval arithmetic before authorizing a decision.
"""
from decimal import Decimal
from fractions import Fraction
from functools import lru_cache
import hashlib
import math
import numbers
from pathlib import Path
import sys

import numpy as np

_NUMERIC_PATH = Path(__file__).resolve().parents[1]/"circuit_validation"/"direct_tests.py"
sys.path.insert(0,str(_NUMERIC_PATH.parent))
from direct_tests import _Bounds, _as_float_bounds

NUMERIC_SOURCE_SHA256=hashlib.sha256(_NUMERIC_PATH.read_bytes()).hexdigest()
BET_DENOMINATOR=2**20
SHRINKAGE=4
BET_CAP=Fraction(1023,1024)
FIXED_BETS=tuple(sorted({Fraction(1,2**k) for k in range(1,13)}
                        | {1-Fraction(1,2**k) for k in range(2,11)}))
_GRID_FLOAT=np.array([float(x) for x in FIXED_BETS])
_GRID_LOG_PLUS=np.log1p(_GRID_FLOAT)
_GRID_LOG_MINUS=np.log1p(-_GRID_FLOAT)


def protocol_config():
    return {"kelly_weight_exact":"1/2","fixed_grid_weight_exact":"1/2",
            "fixed_bets_exact":[str(x) for x in FIXED_BETS],
            "fixed_expert_count":len(FIXED_BETS),"fixed_weights":"uniform",
            "predictor_shrinkage":SHRINKAGE,"bet_cap_exact":str(BET_CAP),
            "default_bet_denominator":BET_DENOMINATOR,
            "directional_threshold":"2/delta",
            "required_use":"bets and angles frozen before each block; only complete common-sign pairs"}


def _count_array(counts):
    values=np.asarray(counts)
    if values.ndim<1 or values.shape[-1]!=4:
        raise ValueError("counts must have last dimension four, ordered00,01,10,11")
    if not np.all(np.isfinite(values)) or np.any(values<0) or np.any(values!=np.floor(values)):
        raise ValueError("counts must be finite nonnegative integers")
    return values.astype(np.int64)


def _integer(value,name,positive=False):
    if isinstance(value,bool) or not isinstance(value,numbers.Integral):
        raise ValueError(name+" must be an integer")
    value=int(value)
    if value<(1 if positive else 0):
        raise ValueError(name+" outside allowed range")
    return value


def _risk(delta):
    if isinstance(delta,bool):
        raise ValueError("delta must be a probability")
    value=delta if isinstance(delta,Fraction) else Fraction(str(delta))
    if not 0<value<1:
        raise ValueError("delta must lie in(0,1)")
    return value


def predictable_bets(past_counts,estimated_p_minus=None,estimated_p_plus=None,
                     denominator=BET_DENOMINATOR):
    """Use only PRIOR observations/predictions to select the next block's bets.

    With no model predictions, use discordant-sign bias with four symmetric
    pseudocounts. With predictions, shrink their conditional sign bias by the
    observed discordant_count/(discordant_count+4). Quantization makes the
    actual executed bet a public exact rational, including in certificates.
    """
    values=_count_array(past_counts)
    denominator=_integer(denominator,"denominator",True)
    worse,better=values[...,1],values[...,2]
    discordant=worse+better
    if (estimated_p_minus is None)!=(estimated_p_plus is None):
        raise ValueError("supply either both fitted probabilities or neither")
    if estimated_p_minus is None:
        bias=(worse-better)/(discordant+SHRINKAGE)
    else:
        pm,pp,n=np.broadcast_arrays(np.asarray(estimated_p_minus,dtype=float),
                                   np.asarray(estimated_p_plus,dtype=float),discordant)
        if (not np.all(np.isfinite(pm)) or not np.all(np.isfinite(pp))
                or np.any(pm<0) or np.any(pm>1) or np.any(pp<0) or np.any(pp>1)):
            raise ValueError("fitted probabilities must lie in[0,1]")
        worse_probability=(1-pm)*pp
        better_probability=pm*(1-pp)
        d=worse_probability+better_probability
        raw=np.divide(worse_probability-better_probability,d,
                      out=np.zeros_like(d),where=d>0)
        bias=raw*n/(n+SHRINKAGE)
    cap_numerator=int(BET_CAP*denominator)
    signed=np.rint(np.clip(bias,-float(BET_CAP),float(BET_CAP))*denominator).astype(np.int64)
    signed=np.clip(signed,-cap_numerator,cap_numerator)
    return {"plus_numerator":np.maximum(signed,0),
            "minus_numerator":np.maximum(-signed,0),"denominator":denominator}


def _bet_arrays(plus_numerator,minus_numerator,denominator):
    denominator=_integer(denominator,"denominator",True)
    plus,minus=np.broadcast_arrays(np.asarray(plus_numerator),np.asarray(minus_numerator))
    for value in (plus,minus):
        if (not np.all(np.isfinite(value)) or np.any(value<0) or np.any(value>=denominator)
                or np.any(value!=np.floor(value))):
            raise ValueError("bet numerators must be integers in[0,denominator)")
    return plus/denominator,minus/denominator


def kelly_log_increments(block_counts,plus_numerator,minus_numerator,
                         denominator=BET_DENOMINATOR):
    """Float-only incremental log capital; not a decision certificate."""
    counts=_count_array(block_counts)
    plus,minus=_bet_arrays(plus_numerator,minus_numerator,denominator)
    b,c=counts[...,1],counts[...,2]
    return (b*np.log1p(plus)+c*np.log1p(-plus),
            c*np.log1p(minus)+b*np.log1p(-minus))


def _logsumexp(values,axis=-1):
    maximum=np.max(values,axis=axis,keepdims=True)
    return np.squeeze(maximum,axis=axis)+np.log(np.sum(np.exp(values-maximum),axis=axis))


def hedge_log_screen(cumulative_counts,kelly_log_plus,kelly_log_minus,delta=.01):
    """Return(float_log_positive_hedge,float_log_negative_hedge,log_threshold).

    A runner may trigger Decimal certification when either log hedge exceeds
    threshold minus a small screening tolerance. Neither the floating scores
    nor their comparisons authorize any decision.
    """
    counts=_count_array(cumulative_counts)
    b,c=counts[...,1,None],counts[...,2,None]
    fixed_plus=_logsumexp(b*_GRID_LOG_PLUS+c*_GRID_LOG_MINUS)-math.log(len(FIXED_BETS))
    fixed_minus=_logsumexp(c*_GRID_LOG_PLUS+b*_GRID_LOG_MINUS)-math.log(len(FIXED_BETS))
    lp=np.logaddexp(np.asarray(kelly_log_plus),fixed_plus)-math.log(2)
    lm=np.logaddexp(np.asarray(kelly_log_minus),fixed_minus)-math.log(2)
    return lp,lm,math.log(2/float(_risk(delta)))


@lru_cache(maxsize=32768)
def _log_factors(numerator,denominator):
    numerator=_integer(numerator,"numerator")
    denominator=_integer(denominator,"denominator",True)
    if numerator>=denominator:
        raise ValueError("bet numerator must be smaller than denominator")
    return ((_Bounds(denominator+numerator)/denominator).log(),
            (_Bounds(denominator-numerator)/denominator).log())


def _interval_logsumexp(values):
    # Center on an exact Decimal endpoint, avoiding both overflow and a
    # floating-point dependency in certificate arithmetic.
    center=_Bounds(max(x.lo for x in values))
    total=_Bounds(0)
    for value in values:
        total=total+(value-center).exp()
    return center+total.log()


def _fixed_log_capital(b,c,positive=True):
    logs=[]
    for bet in FIXED_BETS:
        lp,lm=_log_factors(bet.numerator,bet.denominator)
        logs.append(lp*(b if positive else c)+lm*(c if positive else b))
    return _interval_logsumexp(logs)-_Bounds(len(FIXED_BETS)).log()


def certificate_from_blocks(blocks,delta=.01):
    """Outward Decimal certificate for the fixed Kelly/fixed-grid hedge.

    Every block has counts[00,01,10,11], plus_numerator, minus_numerator,
    denominator. Other provenance keys are ignored. The certificate can
    verify arithmetic, not whether a caller chose bets before seeing counts;
    that temporal condition must be checked by transcript replay.
    """
    risk=_risk(delta)
    capital_plus,capital_minus=_Bounds(0),_Bounds(0)
    total=[0,0,0,0]
    block_count=0
    for block in blocks:
        counts=list(block["counts"])
        if len(counts)!=4:
            raise ValueError("each block requires four pair counts")
        counts=[_integer(x,"pair count") for x in counts]
        pnum=_integer(block["plus_numerator"],"plus_numerator")
        mnum=_integer(block["minus_numerator"],"minus_numerator")
        den=_integer(block["denominator"],"denominator",True)
        pp,pm=_log_factors(pnum,den)
        mp,mm=_log_factors(mnum,den)
        b,c=counts[1:3]
        capital_plus=capital_plus+pp*b+pm*c
        capital_minus=capital_minus+mp*c+mm*b
        total=[left+right for left,right in zip(total,counts)]
        block_count+=1
    fixed_plus=_fixed_log_capital(total[1],total[2],True)
    fixed_minus=_fixed_log_capital(total[1],total[2],False)
    log2=_Bounds(2).log()
    hedge_plus=_interval_logsumexp([capital_plus,fixed_plus])-log2
    hedge_minus=_interval_logsumexp([capital_minus,fixed_minus])-log2
    threshold=(_Bounds(2*risk.denominator)/risk.numerator).log()
    reject=hedge_plus.lo>=threshold.hi
    accept=hedge_minus.lo>=threshold.hi
    decision="reject" if reject and not accept else "accept" if accept and not reject else "abstain"
    return {"method":"predictable_kelly_fixed_grid_hedge","decision":decision,
            "counts":total,"shots":2*sum(total),"blocks":block_count,
            "delta":float(risk),"delta_exact":str(risk),"threshold_exact":str(2/risk),
            "log_hedge_plus_bounds":_as_float_bounds(hedge_plus),
            "log_hedge_minus_bounds":_as_float_bounds(hedge_minus),
            "log_hedge_plus_decimal_bounds":[str(hedge_plus.lo),str(hedge_plus.hi)],
            "log_hedge_minus_decimal_bounds":[str(hedge_minus.lo),str(hedge_minus.hi)],
            "log_kelly_plus_bounds":_as_float_bounds(capital_plus),
            "log_kelly_minus_bounds":_as_float_bounds(capital_minus),
            "log_fixed_plus_bounds":_as_float_bounds(fixed_plus),
            "log_fixed_minus_bounds":_as_float_bounds(fixed_minus),
            "threshold_log_bounds":_as_float_bounds(threshold),
            "both_directions_crossed":bool(reject and accept),
            "anytime_valid":True,
            "numeric_scope":"outward Decimal log capitals; exact rational executed bets",
            "required_use":"complete common-sign pairs; bets and both angles predictable before each block",
            "numeric_source_sha256":NUMERIC_SOURCE_SHA256}
