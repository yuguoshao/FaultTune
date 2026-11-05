"""Predictable full-pair safe two-sample likelihood e-process.

This implements the established safe Bernoulli two-sample construction
(Turner, Ly and Gruenwald; https://arxiv.org/abs/2106.02693, and the AISTATS
exposition https://safestatistics.com/wp-content/uploads/2023/10/TurnerAIStats23.pdf
Section2, equations3-6), with a direct common-order null argument below.
It is a stronger comparator, not a new SOTA claim for FaultTune.

At each paired block, past data predict probabilities a<=b and r=(a+b)/2.
The full-pair factor is f_a(Yminus)f_b(Yplus)/
                            [f_r(Yminus)f_r(Yplus)].
Under true pminus=u>=v=pplus, its expectation is A(u)B(v), where A decreases
and B increases. Thus its maximum over u>=v occurs on u=v=q, and equals
1-[(q-r)(b-a)/(2r(1-r))]^2 <= 1. Predictable products remain valid.
The opposite direction reverses the predicted ordering. Predictions pointing
in the wrong direction are replaced by a neutral factor exactly equal to1.

Each direction is hedged1/2 with a fixed21-expert directional paired bet
mixture. Threshold2/delta in each direction gives total anytime error<=delta.
Both forecast probabilities and both physical angles must be fixed before
either outcome in a pair/block. Additional independent model assumptions
are needed to turn physical pair ordering into the target update's sign.

Floating prediction and screening never authorize decisions. All executed
forecast probabilities are exact rationals and certificate_from_blocks
recomputes all four outcome factors with outward Decimal arithmetic.
"""
from fractions import Fraction
from functools import lru_cache
import hashlib
import math
from pathlib import Path

import numpy as np

from betting import (_Bounds,_as_float_bounds,_count_array,_integer,_risk,
                     _fixed_log_capital,_interval_logsumexp,hedge_log_screen,
                     protocol_config as betting_protocol_config,
                     BET_DENOMINATOR,NUMERIC_SOURCE_SHA256)

BETTING_SOURCE_SHA256=hashlib.sha256(Path(__file__).resolve().with_name("betting.py").read_bytes()).hexdigest()


def protocol_config():
    result=betting_protocol_config()
    result.pop("predictor_shrinkage")
    result.pop("bet_cap_exact")
    result["strategy_component"]="predictable full-pair safe two-sample likelihood"
    result["marginal_pseudocount_exact"]="1/2"
    result["probability_clip_numerators"]="1 through denominator-1"
    result["full_pair_formula"]="f_pm(yminus)*f_pp(yplus)/(f_r(yminus)*f_r(yplus)), r=(pm+pp)/2"
    result["source"]="Turner-Ly-Gruenwald safe two-sample Bernoulli e-variable; predictable extension"
    result["ordered_null_extension"]="direct expectation bound for pminus>=pplus or reversed ordering"
    return result


def predictable_probabilities(past_counts,estimated_p_minus=None,estimated_p_plus=None,
                              denominator=BET_DENOMINATOR):
    """Predict the next block using past paired counts or past-data model fits.

    The default marginal Jeffreys estimates are appropriate for a fixed PSR
    angle; under changing angles a caller can supply fitted predictions at
    the next selected angle. Predictions only affect efficiency, while their
    predictability and strict interior rationalization ensure safe execution.
    """
    counts=_count_array(past_counts)
    denominator=_integer(denominator,"denominator",True)
    if denominator<2:
        raise ValueError("probability denominator must be at least2")
    if (estimated_p_minus is None)!=(estimated_p_plus is None):
        raise ValueError("supply both predicted probabilities or neither")
    if estimated_p_minus is None:
        n=np.sum(counts,axis=-1)
        pm=(counts[...,2]+counts[...,3]+.5)/(n+1)
        pp=(counts[...,1]+counts[...,3]+.5)/(n+1)
    else:
        pm,pp,unused=np.broadcast_arrays(np.asarray(estimated_p_minus,dtype=float),
                                        np.asarray(estimated_p_plus,dtype=float),counts[...,0])
        if (not np.all(np.isfinite(pm)) or not np.all(np.isfinite(pp))
                or np.any(pm<0) or np.any(pm>1) or np.any(pp<0) or np.any(pp>1)):
            raise ValueError("predicted probabilities must lie in[0,1]")
    return {"pred_minus_numerator":np.clip(np.rint(pm*denominator),1,denominator-1).astype(np.int64),
            "pred_plus_numerator":np.clip(np.rint(pp*denominator),1,denominator-1).astype(np.int64),
            "denominator":denominator}


def _prediction_arrays(minus_numerator,plus_numerator,denominator):
    denominator=_integer(denominator,"denominator",True)
    nm,np_=np.broadcast_arrays(np.asarray(minus_numerator),np.asarray(plus_numerator))
    for value in (nm,np_):
        if (not np.all(np.isfinite(value)) or np.any(value<=0) or np.any(value>=denominator)
                or np.any(value!=np.floor(value))):
            raise ValueError("prediction numerators must be integers strictly between0anddenominator")
    return nm/denominator,np_/denominator


def safe_log_increments(block_counts,pred_minus_numerator,pred_plus_numerator,
                        denominator=BET_DENOMINATOR):
    """Float-only log increments for positive and negative ordered forecasts."""
    counts=_count_array(block_counts)
    pm,pp=_prediction_arrays(pred_minus_numerator,pred_plus_numerator,denominator)
    r=(pm+pp)/2
    half_gap=(pp-pm)/2
    # Stable concordance factors avoid subtracting close likelihood logs.
    log00=np.log1p(-(half_gap/(1-r))**2)
    log11=np.log1p(-(half_gap/r)**2)
    log01=np.log1p(half_gap/(1-r))+np.log1p(half_gap/r)
    log10=np.log1p(-half_gap/r)+np.log1p(-half_gap/(1-r))
    increment=(counts[...,0]*log00+counts[...,1]*log01
               +counts[...,2]*log10+counts[...,3]*log11)
    return np.where(pp>=pm,increment,0.),np.where(pm>=pp,increment,0.)


def safe_log_screen(cumulative_counts,safe_log_plus,safe_log_minus,delta=.01):
    """Same fixed hedge and directional threshold as the Kelly comparator."""
    return hedge_log_screen(cumulative_counts,safe_log_plus,safe_log_minus,delta)


@lru_cache(maxsize=32768)
def _full_log_factors(nminus,nplus,denominator):
    nminus=_integer(nminus,"pred_minus_numerator",True)
    nplus=_integer(nplus,"pred_plus_numerator",True)
    denominator=_integer(denominator,"denominator",True)
    if nminus>=denominator or nplus>=denominator:
        raise ValueError("prediction numerators must be smaller than denominator")
    twice_center=nminus+nplus
    twice_complement=2*denominator-twice_center
    factors=((4*(denominator-nminus)*(denominator-nplus),twice_complement**2),
             (4*(denominator-nminus)*nplus,twice_complement*twice_center),
             (4*nminus*(denominator-nplus),twice_center*twice_complement),
             (4*nminus*nplus,twice_center**2))
    return tuple((_Bounds(num)/den).log() for num,den in factors)


def certificate_from_blocks(blocks,delta=.01):
    """Recompute a full-pair safe likelihood hedge from rational forecasts.

    Block schema: counts[00,01,10,11], pred_minus_numerator,
    pred_plus_numerator, denominator. Extra provenance keys are ignored.
    A transcript audit must establish each prediction was chosen before its
    observations; this numeric function cannot infer causal ordering.
    """
    risk=_risk(delta)
    capital_plus,capital_minus=_Bounds(0),_Bounds(0)
    total=[0,0,0,0]
    block_count=0
    for block in blocks:
        counts=list(block["counts"])
        if len(counts)!=4:
            raise ValueError("each block requires four paired counts")
        counts=[_integer(x,"pair count") for x in counts]
        nm=_integer(block["pred_minus_numerator"],"pred_minus_numerator",True)
        np_=_integer(block["pred_plus_numerator"],"pred_plus_numerator",True)
        den=_integer(block["denominator"],"denominator",True)
        factors=_full_log_factors(nm,np_,den)
        increment=_Bounds(0)
        for count,factor in zip(counts,factors):
            increment=increment+factor*count
        if np_>=nm:
            capital_plus=capital_plus+increment
        if nm>=np_:
            capital_minus=capital_minus+increment
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
    return {"method":"predictable_safe_two_sample_fixed_grid_hedge","decision":decision,
            "counts":total,"shots":2*sum(total),"blocks":block_count,
            "delta":float(risk),"delta_exact":str(risk),"threshold_exact":str(2/risk),
            "log_hedge_plus_bounds":_as_float_bounds(hedge_plus),
            "log_hedge_minus_bounds":_as_float_bounds(hedge_minus),
            "log_hedge_plus_decimal_bounds":[str(hedge_plus.lo),str(hedge_plus.hi)],
            "log_hedge_minus_decimal_bounds":[str(hedge_minus.lo),str(hedge_minus.hi)],
            "log_safe_plus_bounds":_as_float_bounds(capital_plus),
            "log_safe_minus_bounds":_as_float_bounds(capital_minus),
            "log_fixed_plus_bounds":_as_float_bounds(fixed_plus),
            "log_fixed_minus_bounds":_as_float_bounds(fixed_minus),
            "threshold_log_bounds":_as_float_bounds(threshold),
            "both_directions_crossed":bool(reject and accept),
            "anytime_valid":True,
            "numeric_scope":"outward Decimal full-pair likelihoods; exact rational predictions",
            "required_use":"both pair probabilities and angles chosen predictably before each block",
            "numeric_source_sha256":NUMERIC_SOURCE_SHA256,
            "fixed_hedge_source_sha256":BETTING_SOURCE_SHA256}
