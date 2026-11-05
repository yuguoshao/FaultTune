"""Balanced full-count safe tests with fixed odds experts and a shared threshold.

Each expert keeps its rational odds ratio fixed for the entire run. Its
nuisance center may be predicted using previous paid observations. Mirror
experts swap the same rational endpoints exactly. This preserves the signed
capital bound needed for total wrong-direction risk delta at threshold 1/delta.

The supplied experts configuration may be shared with conditional Fisher or
paired-grid controls. Floating calculations only screen; Decimal intervals
recompute all likelihood factors before any decision is authorized.
"""
import hashlib
import json
import math
import numbers
from decimal import Decimal
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
from scipy.special import logsumexp

import conditional_fisher

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"experiments"/"circuit_validation"))
from direct_tests import _Bounds,_float_down,_float_up


PROTOCOL_VERSION="safe_fixed_odds_grid_v1"
ENDPOINT_DENOMINATOR=2**40


def _integer(value,name,minimum=0):
    if isinstance(value,bool) or not isinstance(value,numbers.Integral) or value<minimum:
        raise ValueError(name+" must be an integer at least "+str(minimum))
    return int(value)


def _experts(config):
    rows=config["experts"]
    if not rows:
        raise ValueError("at least one expert is required")
    result=[]
    seen=set()
    for row in rows:
        numerator=_integer(row["numerator"],"odds numerator",1)
        denominator=_integer(row["denominator"],"odds denominator",1)
        odds=Fraction(numerator,denominator)
        if odds in seen:
            raise ValueError("duplicate odds expert")
        seen.add(odds)
        weight=Fraction(_integer(row["weight"][0],"weight numerator",1),
                        _integer(row["weight"][1],"weight denominator",1))
        result.append((odds,weight))
    if sum((weight for odds,weight in result),Fraction(0))!=1:
        raise ValueError("expert weights must sum exactly to one")
    return result


def protocol_config(config=None,endpoint_denominator=None):
    base=conditional_fisher.protocol_config() if config is None else config
    experts=_experts(base)
    denominator=_integer(endpoint_denominator if endpoint_denominator is not None else
                         base.get("endpoint_denominator",ENDPOINT_DENOMINATOR),"endpoint denominator",2)
    return {"version":PROTOCOL_VERSION,"endpoint_denominator":denominator,
            "experts":[{"numerator":odds.numerator,"denominator":odds.denominator,
                        "weight":[weight.numerator,weight.denominator]} for odds,weight in experts],
            "predictor":"past-data nuisance center; stable quadratic then rational lower endpoint",
            "upper_endpoint":"v=omega*u/(1-u+omega*u) exactly; negative expert swaps endpoints",
            "threshold":"1/delta, common cumulative plus-minus score, no directional split",
            "assumptions":"predictable balanced independent Bernoulli blocks with common ordering",
            "limits":"fixed finite odds grid; predictor affects efficiency; no universal positive-growth guarantee"}


def config_digest(config):
    return hashlib.sha256(json.dumps(protocol_config(config),sort_keys=True,separators=(",",":")).encode()).hexdigest()


def _layout(config):
    experts=_experts(config)
    poles=sorted({odds if odds>1 else 1/odds for odds,weight in experts if odds!=1})
    lookup={odds:i for i,odds in enumerate(poles)}
    layout=[(None,0) if odds==1 else (lookup[odds if odds>1 else 1/odds],1 if odds>1 else -1)
            for odds,weight in experts]
    return experts,poles,layout


def predict_endpoints(center,config=None):
    """Call before outcomes; only the center and fixed config select endpoints."""
    config=protocol_config(config)
    if isinstance(center,bool) or not math.isfinite(center) or not 0<=center<=1:
        raise ValueError("center must be finite and in [0,1]")
    denominator=config["endpoint_denominator"]
    q=min(1-1/denominator,max(1/denominator,float(center)))
    unused,poles,unused_layout=_layout(config)
    numerators=[]
    for odds in poles:
        omega=float(odds)
        if not math.isfinite(omega):
            raise ValueError("odds exceed the floating predictor range")
        difference=omega-1
        # Solve fixed odds with requested midpoint q using the stable root.
        discriminant=math.sqrt(max(0.,(omega+1)**2-4*difference*difference*q*(1-q)))
        half_gap=2*difference*q*(1-q)/(omega+1+discriminant)
        lower=q-half_gap
        numerators.append(min(denominator-1,max(1,int(round(lower*denominator)))))
    return {"center_requested":float(center),"center_clipped":q,
            "odds_poles":[[odds.numerator,odds.denominator] for odds in poles],
            "lower_numerators":numerators,"lower_denominator":denominator}


def _checked_block(block,config,poles):
    m=_integer(block["m"],"m")
    minus=_integer(block["minus_successes"],"minus successes")
    plus=_integer(block["plus_successes"],"plus successes")
    if minus>m or plus>m:
        raise ValueError("success count exceeds block size")
    prediction=block["prediction"]
    denominator=_integer(prediction["lower_denominator"],"lower denominator",2)
    if denominator!=config["endpoint_denominator"]:
        raise ValueError("endpoint denominator differs from fixed configuration")
    if prediction["odds_poles"]!=[[odds.numerator,odds.denominator] for odds in poles]:
        raise ValueError("odds experts changed")
    numerators=[_integer(value,"lower numerator",1) for value in prediction["lower_numerators"]]
    if len(numerators)!=len(poles) or any(value>=denominator for value in numerators):
        raise ValueError("invalid endpoint list")
    return m,minus,plus,numerators,denominator


def rational_endpoints(lower_numerator,lower_denominator,odds_numerator,odds_denominator):
    u=Fraction(_integer(lower_numerator,"lower numerator",1),
               _integer(lower_denominator,"lower denominator",2))
    odds=Fraction(_integer(odds_numerator,"odds numerator",1),
                  _integer(odds_denominator,"odds denominator",1))
    if not 0<u<1 or odds<=1:
        raise ValueError("need an interior lower endpoint and positive odds pole")
    v=odds*u/(1-u+odds*u)
    return u,v


def rational_factors(lower_numerator,lower_denominator,odds_numerator,odds_denominator):
    u,v=rational_endpoints(lower_numerator,lower_denominator,odds_numerator,odds_denominator)
    r=(u+v)/2
    return ((1-u)*(1-v)/(1-r)**2,(1-u)*v/((1-r)*r),
            u*(1-v)/((1-r)*r),u*v/r**2)


@lru_cache(maxsize=65536)
def _log_factors(lower_numerator,lower_denominator,odds_numerator,odds_denominator):
    return tuple((_Bounds(factor.numerator)/factor.denominator).log()
                 for factor in rational_factors(lower_numerator,lower_denominator,odds_numerator,odds_denominator))


@lru_cache(maxsize=65536)
def _float_log_factors(lower_numerator,lower_denominator,odds_numerator,odds_denominator):
    result=[]
    for factor in rational_factors(lower_numerator,lower_denominator,odds_numerator,odds_denominator):
        difference=factor.numerator-factor.denominator
        if 2*abs(difference)<factor.denominator:
            result.append(math.log1p(difference/factor.denominator))
        else:
            result.append(math.log(factor.numerator)-math.log(factor.denominator))
    return np.array(result)


def _pair_counts(m,minus,plus):
    # Any pairing has the same full-count likelihood; use a canonical one.
    both=max(0,minus+plus-m)
    return [m-minus-plus+both,plus-both,minus-both,both]


def empty_state(config=None):
    config=protocol_config(config)
    experts,poles,layout=_layout(config)
    return {"config":config,"log_capital":np.zeros(len(experts)),"score":0,"shots":0,"blocks":[],
            "_poles":poles,"_layout":layout,"_log_weights":np.array([math.log(float(w)) for odds,w in experts])}


def update_block(state,m,minus_successes,plus_successes,prediction):
    """Add outcomes only after the recorded prediction has been selected."""
    block={"m":m,"minus_successes":minus_successes,"plus_successes":plus_successes,
           "prediction":json.loads(json.dumps(prediction))}
    m,minus,plus,numerators,denominator=_checked_block(block,state["config"],state["_poles"])
    block.update({"m":m,"minus_successes":minus,"plus_successes":plus})
    counts=np.array(_pair_counts(m,minus,plus),dtype=float)
    positive=[]
    negative=[]
    for numerator,odds in zip(numerators,state["_poles"]):
        factors=_float_log_factors(numerator,denominator,odds.numerator,odds.denominator)
        positive.append(float(np.dot(counts,factors)))
        negative.append(float(np.dot(counts,factors[[0,2,1,3]])))
    for j,(pole,sign) in enumerate(state["_layout"]):
        if sign:
            state["log_capital"][j]+=(positive if sign>0 else negative)[pole]
    state["score"]+=plus-minus
    state["shots"]+=2*m
    state["blocks"].append(block)
    return state


def screen(state,delta=.01):
    if isinstance(delta,bool) or not math.isfinite(delta) or not 0<delta<1:
        raise ValueError("delta must be in (0,1)")
    value=float(logsumexp(state["log_capital"]+state["_log_weights"]))
    threshold=math.log(1/delta)
    crossed=value>=threshold-1e-6 and state["score"]!=0
    direction="reject" if state["score"]>0 else "accept" if state["score"]<0 else "abstain"
    return {"log_evidence":value,"threshold":threshold,"score":state["score"],"shots":state["shots"],
            "decision_candidate":direction if crossed else "abstain","crossing_candidate":crossed,"certified":False}


def certificate_from_blocks(blocks,delta=.01,config=None):
    config=protocol_config(config)
    blocks=list(blocks)
    experts,poles,layout=_layout(config)
    if isinstance(delta,bool):
        raise ValueError("delta must be a probability")
    risk=Fraction(str(delta))
    if not 0<risk<1:
        raise ValueError("delta must be in (0,1)")
    capitals=[_Bounds(0) for unused in experts]
    score=shots=0
    for block in blocks:
        m,minus,plus,numerators,denominator=_checked_block(block,config,poles)
        counts=_pair_counts(m,minus,plus)
        positive=[]
        negative=[]
        for numerator,odds in zip(numerators,poles):
            factors=_log_factors(numerator,denominator,odds.numerator,odds.denominator)
            positive.append(sum((factor*count for factor,count in zip(factors,counts)),_Bounds(0)))
            negative.append(sum((factors[k]*counts[j] for j,k in enumerate((0,2,1,3))),_Bounds(0)))
        for j,(pole,sign) in enumerate(layout):
            if sign:
                capitals[j]+=(positive if sign>0 else negative)[pole]
        score+=plus-minus
        shots+=2*m
    offset=max(capital.hi for capital in capitals)
    scaled=sum((((capital-_Bounds(offset)).exp()*weight.numerator/weight.denominator)
                for capital,(odds,weight) in zip(capitals,experts)),_Bounds(0))
    evidence=_Bounds(offset)+scaled.log()
    threshold=(_Bounds(risk.denominator)/risk.numerator).log()
    crossed=evidence.lo>=threshold.hi and score!=0
    decision=("reject" if score>0 else "accept") if crossed else "abstain"
    return {"method":PROTOCOL_VERSION,"protocol_sha256":config_digest(config),"decision":decision,
            "score":score,"shots":shots,"blocks":len(blocks),"delta":float(risk),
            "evidence":_float_down(evidence.lo),"threshold":_float_up(threshold.hi),
            "evidence_bounds":[str(evidence.lo),str(evidence.hi)],
            "threshold_bounds":[str(threshold.lo),str(threshold.hi)],"anytime_valid":True,
            "numeric_status":"outward Decimal logs of exact rational mirrored endpoint factors",
            "assumptions":"balanced predictable independent Bernoulli blocks, common ordering, fixed odds and weights"}
