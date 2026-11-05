"""Independent rational checks for the full-pair safe likelihood comparator."""
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

import safe_two_sample as safe
from betting import FIXED_BETS


ROOT=Path(__file__).resolve().parents[2]


def factors(a,b):
    r=(a+b)/2
    return ((1-a)*(1-b)/(1-r)**2,(1-a)*b/((1-r)*r),
            a*(1-b)/(r*(1-r)),a*b/r**2)


def exact_hedges(blocks):
    plus,minus=Fraction(1),Fraction(1)
    b,c=0,0
    for block in blocks:
        a=Fraction(block["pred_minus_numerator"],block["denominator"])
        p=Fraction(block["pred_plus_numerator"],block["denominator"])
        update=Fraction(1)
        for count,factor in zip(block["counts"],factors(a,p)):
            update*=factor**count
        if p>=a:
            plus*=update
        if a>=p:
            minus*=update
        b+=block["counts"][1]
        c+=block["counts"][2]
    fixed_plus=sum((1+l)**b*(1-l)**c for l in FIXED_BETS)/len(FIXED_BETS)
    fixed_minus=sum((1+l)**c*(1-l)**b for l in FIXED_BETS)/len(FIXED_BETS)
    return (plus+fixed_plus)/2,(minus+fixed_minus)/2


def reference_log(value):
    with localcontext() as context:
        context.prec=110
        return (Decimal(value.numerator)/Decimal(value.denominator)).ln()


def exact_null_checks():
    count,equality=0,0
    probabilities=[Fraction(k,16) for k in range(17)]
    for numerator_a in (1,2,4,8,12,14,15):
        for numerator_b in (1,2,4,8,12,14,15):
            if numerator_b<numerator_a:
                continue
            a,b=Fraction(numerator_a,16),Fraction(numerator_b,16)
            r=(a+b)/2
            outcome_factors=factors(a,b)
            for u in probabilities:
                for v in probabilities:
                    if u<v:
                        continue
                    distribution=((1-u)*(1-v),(1-u)*v,u*(1-v),u*v)
                    expectation=sum(p*f for p,f in zip(distribution,outcome_factors))
                    assert expectation<=1
                    if u==v:
                        reference=1-((u-r)*(b-a)/(2*r*(1-r)))**2
                        assert expectation==reference
                        equality+=1
                    count+=1
    return count,equality


def adaptive_null_expectation(steps=14):
    """Exact DP over past counts; predictions and true common means adapt.

    Capital can differ between histories with equal counts. The DP carries
    probability-weighted capital, whose update is linear because the next
    predictable forecast and data distribution depend only on those counts.
    All four outcomes matter for the forecast and its next conditional mean.
    """
    # State=(00,01,10,11), value=(probability,weighted_positive_safe_capital).
    states={(0,0,0,0):(Fraction(1),Fraction(1))}
    worst=Fraction(0)
    for step in range(steps):
        new={}
        for counts,(probability,capital) in states.items():
            n=sum(counts)
            a=Fraction(2*(counts[2]+counts[3])+1,2*(n+1))
            b=Fraction(2*(counts[1]+counts[3])+1,2*(n+1))
            predicted=factors(a,b) if b>=a else (Fraction(1),)*4
            # A predictable angle-dependent common Bernoulli mean under C=0.
            q=Fraction(1,4) if counts[1]>counts[2] else Fraction(3,5)
            probabilities=((1-q)**2,(1-q)*q,q*(1-q),q*q)
            for index,(p,factor) in enumerate(zip(probabilities,predicted)):
                after=list(counts)
                after[index]+=1
                key=tuple(after)
                oldprob,oldcap=new.get(key,(Fraction(0),Fraction(0)))
                new[key]=(oldprob+probability*p,oldcap+capital*p*factor)
        states=new
        assert sum(value[0] for value in states.values())==1
        expectation=sum(value[1] for value in states.values())
        assert expectation<=1
        worst=max(worst,expectation)
    return {"steps":steps,"final_count_states":len(states),
            "maximum_positive_capital_expectation_exact":str(worst),
            "final_positive_capital_expectation":float(expectation)}


def certificate_checks():
    rng=np.random.RandomState(809137)
    checked=0
    max_screen_error=0.
    for trial in range(96):
        blocks=[]
        cumulative=np.zeros(4,dtype=np.int64)
        lp=lm=0.
        for unused in range(1+trial%7):
            # These arbitrary predictable records test arithmetic. Separate
            # exact-null checks above verify the statistical inequality.
            den=64
            nm=int(rng.randint(1,den))
            np_=int(rng.randint(1,den))
            counts=[int(x) for x in rng.randint(0,8,size=4)]
            block={"counts":counts,"pred_minus_numerator":nm,
                   "pred_plus_numerator":np_,"denominator":den}
            blocks.append(block)
            increment=safe.safe_log_increments(counts,nm,np_,den)
            lp+=float(increment[0])
            lm+=float(increment[1])
            cumulative+=counts
        exact=exact_hedges(blocks)
        cert=safe.certificate_from_blocks(blocks,delta=.01)
        for direction,value in zip(("plus","minus"),exact):
            lo,hi=(Decimal(x) for x in cert["log_hedge_"+direction+"_decimal_bounds"])
            assert lo<=reference_log(value)<=hi
        if cert["decision"]=="reject":
            assert exact[0]>=200
        if cert["decision"]=="accept":
            assert exact[1]>=200
        screen=safe.safe_log_screen(cumulative,lp,lm)
        max_screen_error=max(max_screen_error,*(abs(float(reference_log(value))-float(observed))
                                               for value,observed in zip(exact,screen[:2])))
        reflected=[]
        for block in blocks:
            reflected.append({"counts":[block["counts"][0],block["counts"][2],block["counts"][1],block["counts"][3]],
                              "pred_minus_numerator":block["pred_plus_numerator"],
                              "pred_plus_numerator":block["pred_minus_numerator"],"denominator":block["denominator"]})
        reflection=safe.certificate_from_blocks(reflected)
        # Swapping01and10 permutes directed additions; interval endpoints
        # need not be bit-identical even though the exact capitals coincide.
        for direction,other in (("plus","minus"),("minus","plus")):
            lo,hi=map(Decimal,cert["log_hedge_"+direction+"_decimal_bounds"])
            rlo,rhi=map(Decimal,reflection["log_hedge_"+other+"_decimal_bounds"])
            assert max(lo,rlo)<=min(hi,rhi)
        assert exact_hedges(reflected)==(exact[1],exact[0])
        checked+=1
    return checked,max_screen_error


def vector_and_boundary_checks():
    rng=np.random.RandomState(12837)
    counts=rng.randint(0,100,size=(256,4))
    predictions=safe.predictable_probabilities(counts)
    increments=safe.safe_log_increments(counts,predictions["pred_minus_numerator"],
                                        predictions["pred_plus_numerator"],predictions["denominator"])
    for i in range(len(counts)):
        scalar=safe.predictable_probabilities(counts[i])
        assert int(scalar["pred_minus_numerator"])==int(predictions["pred_minus_numerator"][i])
        assert int(scalar["pred_plus_numerator"])==int(predictions["pred_plus_numerator"][i])
        item=safe.safe_log_increments(counts[i],**scalar)
        assert float(item[0])==float(increments[0][i])
        assert float(item[1])==float(increments[1][i])
    boundaries=[]
    for nm,np_ in ((1,1048575),(1048575,1),(1,1),(1048575,1048575),(524288,524288)):
        block={"counts":[1000000,700000,300000,1000000],"pred_minus_numerator":nm,
               "pred_plus_numerator":np_,"denominator":1048576}
        cert=safe.certificate_from_blocks([block])
        inc=safe.safe_log_increments(block["counts"],nm,np_)
        assert all(math.isfinite(float(x)) for x in inc)
        assert all(math.isfinite(x) for key in ("log_hedge_plus_bounds","log_hedge_minus_bounds")
                   for x in cert[key])
        boundaries.append(cert["decision"])
    neutral=safe.certificate_from_blocks([])
    assert neutral["decision"]=="abstain" and neutral["shots"]==0
    bad=[{"counts":[0,0,0],"pred_minus_numerator":1,"pred_plus_numerator":2,"denominator":4},
         {"counts":[0,-1,0,0],"pred_minus_numerator":1,"pred_plus_numerator":2,"denominator":4},
         {"counts":[0,1,0,0],"pred_minus_numerator":0,"pred_plus_numerator":2,"denominator":4},
         {"counts":[0,1,0,0],"pred_minus_numerator":1,"pred_plus_numerator":4,"denominator":4},
         {"counts":[0,1,0,0],"pred_minus_numerator":1.5,"pred_plus_numerator":2,"denominator":4}]
    rejected=0
    for block in bad:
        try:
            safe.certificate_from_blocks([block])
        except ValueError:
            rejected+=1
    assert rejected==len(bad)
    for value in (0,1,-.1,True):
        try:
            safe.certificate_from_blocks([],delta=value)
        except ValueError:
            rejected+=1
    assert rejected==len(bad)+4
    return {"vector_scalar_predictions_and_increments":len(counts),
            "large_count_extreme_forecasts":len(boundaries),"invalid_inputs_rejected":rejected}


def main():
    started=time.perf_counter()
    order,equality=exact_null_checks()
    certificates,screen_error=certificate_checks()
    result={"passed":True,"exact_ordered_null_expectation_checks":order,
            "exact_common_null_identity_checks":equality,
            "rational_hedge_capital_certificates":certificates,
            "exact_capital_and_interval_sign_reflection_checks":certificates,
            "maximum_float_screen_log_error":screen_error,
            "adaptive_null_DP":adaptive_null_expectation(),
            "vector_boundaries":vector_and_boundary_checks(),
            "statistical_scope":"Predictable full-pair likelihood products are supermartingales under the ordered null; fixed hedge and two-direction thresholds preserve anytime risk. Finite checks supplement the algebra, not a universal empirical safety qualification.",
            "source_sha256":{name:hashlib.sha256((Path(__file__).resolve().parent/name).read_bytes()).hexdigest()
                             for name in ("safe_two_sample.py","safe_two_sample_checks.py","betting.py")},
            "numeric_source_sha256":safe.NUMERIC_SOURCE_SHA256,
            "elapsed_seconds":time.perf_counter()-started}
    output=ROOT/"results"/"streaming_validation"
    output.mkdir(parents=True,exist_ok=True)
    (output/"safe_two_sample_checks.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
