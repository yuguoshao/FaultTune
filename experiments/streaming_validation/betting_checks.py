"""Independent rational and adaptive-null checks for predictable paired bets."""
from decimal import Decimal,localcontext
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

import betting as b

ROOT=Path(__file__).resolve().parents[2]


def exact_capitals(blocks):
    plus,minus=Fraction(1),Fraction(1)
    worse=better=0
    for block in blocks:
        c=block["counts"]
        lp=Fraction(block["plus_numerator"],block["denominator"])
        lm=Fraction(block["minus_numerator"],block["denominator"])
        plus*=(1+lp)**c[1]*(1-lp)**c[2]
        minus*=(1+lm)**c[2]*(1-lm)**c[1]
        worse+=c[1]
        better+=c[2]
    fixed_plus=sum(((1+x)**worse*(1-x)**better for x in b.FIXED_BETS),Fraction(0))/len(b.FIXED_BETS)
    fixed_minus=sum(((1+x)**better*(1-x)**worse for x in b.FIXED_BETS),Fraction(0))/len(b.FIXED_BETS)
    return (plus+fixed_plus)/2,(minus+fixed_minus)/2


def check_exact_logs():
    rng=np.random.RandomState(340901)
    checked=decisions=reflections=0
    for count in range(96):
        blocks=[]
        for unused in range(1+count%5):
            den=int(rng.choice([2,8,64,1024,b.BET_DENOMINATOR]))
            blocks.append({"counts":[int(x) for x in rng.randint(0,7,size=4)],
                           "plus_numerator":int(rng.randint(0,den)),
                           "minus_numerator":int(rng.randint(0,den)),
                           "denominator":den})
        cap_plus,cap_minus=exact_capitals(blocks)
        cert=b.certificate_from_blocks(blocks,delta=Fraction(1,20))
        for cap,field in ((cap_plus,"plus"),(cap_minus,"minus")):
            with localcontext() as ctx:
                ctx.prec=160
                log_exact=(Decimal(cap.numerator)/Decimal(cap.denominator)).ln()
            lo,hi=map(Decimal,cert["log_hedge_"+field+"_decimal_bounds"])
            assert lo<=log_exact<=hi,(lo,log_exact,hi)
            checked+=1
        if cert["decision"]!="abstain":
            assert (cap_plus if cert["decision"]=="reject" else cap_minus)>=40
            decisions+=1
        reverse=[{"counts":[x["counts"][0],x["counts"][2],x["counts"][1],x["counts"][3]],
                  "plus_numerator":x["minus_numerator"],"minus_numerator":x["plus_numerator"],
                  "denominator":x["denominator"]} for x in blocks]
        other=b.certificate_from_blocks(reverse,delta=Fraction(1,20))
        assert cert["log_hedge_plus_decimal_bounds"]==other["log_hedge_minus_decimal_bounds"]
        assert cert["log_hedge_minus_decimal_bounds"]==other["log_hedge_plus_decimal_bounds"]
        assert other["decision"]=={"accept":"reject","reject":"accept","abstain":"abstain"}[cert["decision"]]
        reflections+=1
    return {"rational_capital_log_enclosures":checked,"certified_decisions_checked":decisions,
            "exact_direction_reflections":reflections}


def check_vectors_and_large():
    rng=np.random.RandomState(31083)
    counts=rng.randint(0,100,size=(64,4))
    pm=rng.uniform(.001,.999,size=64)
    pp=rng.uniform(.001,.999,size=64)
    bets=b.predictable_bets(counts,pm,pp)
    inc=b.kelly_log_increments(counts,bets["plus_numerator"],bets["minus_numerator"])
    vector=b.hedge_log_screen(counts,*inc)
    for i in range(64):
        scalar=b.predictable_bets(counts[i],pm[i],pp[i])
        assert int(scalar["plus_numerator"])==int(bets["plus_numerator"][i])
        assert int(scalar["minus_numerator"])==int(bets["minus_numerator"][i])
        one=b.kelly_log_increments(counts[i],scalar["plus_numerator"],scalar["minus_numerator"])
        screen=b.hedge_log_screen(counts[i],*one)
        assert float(screen[0])==float(vector[0][i])
        assert float(screen[1])==float(vector[1][i])
    # Logs far outside floating exp's range must still have finite certificates.
    cases=[[0,4_000_000,0,0],[0,0,4_000_000,0],[0,2_000_000,2_000_000,0]]
    for counts in cases:
        block={"counts":counts,"plus_numerator":1023,"minus_numerator":1023,"denominator":1024}
        cert=b.certificate_from_blocks([block])
        assert all(math.isfinite(x) for field in ("log_hedge_plus_bounds","log_hedge_minus_bounds")
                   for x in cert[field])
    assert b.certificate_from_blocks([])["decision"]=="abstain"
    return {"scalar_batch_matches":64,"large_log_capital_cases":len(cases)}


def check_null_expectations():
    checked=0
    # Exactly rational, independently varying physical angles/true null means.
    for denominator in (8,16):
        for u_num in range(denominator+1):
            for v_num in range(u_num+1):
                u,v=Fraction(u_num,denominator),Fraction(v_num,denominator)
                worse=(1-u)*v
                better=u*(1-v)
                concordant=1-worse-better
                for lam in (Fraction(0),Fraction(1,4),Fraction(3,4),Fraction(1023,1024)):
                    expectation=concordant+worse*(1+lam)+better*(1-lam)
                    assert expectation==1+lam*(v-u)
                    assert expectation<=1
                    checked+=1
    # A finite adaptive policy: outcome-dependent next angle means and bet.
    # State carries exact mass and positive directional capital. Testing one
    # expert suffices to check predictability across changing contexts; fixed
    # convex hedges follow by linearity, as in the separate certificate test.
    states={(0,0,Fraction(1)):Fraction(1)}
    martingale_steps=[]
    for step in range(7):
        following={}
        for (worse,better,capital),mass in states.items():
            # Choose one of two ordered null pairs using only prior counts.
            u,v=(Fraction(3,4),Fraction(1,4)) if worse>=better else (Fraction(1,2),Fraction(1,2))
            lam=Fraction(max(0,worse-better),worse+better+4)
            probabilities=((1-u)*(1-v),(1-u)*v,u*(1-v),u*v)
            for outcome,p in enumerate(probabilities):
                factor=1+lam if outcome==1 else 1-lam if outcome==2 else Fraction(1)
                key=(worse+(outcome==1),better+(outcome==2),capital*factor)
                following[key]=following.get(key,Fraction(0))+mass*p
        states=following
        total_mass=sum(states.values(),Fraction(0))
        mean=sum((mass*state[2] for state,mass in states.items()),Fraction(0))
        assert total_mass==1
        assert mean<=1
        martingale_steps.append(str(mean))
    return {"exact_ordered_null_one_step_checks":checked,"adaptive_null_steps":len(martingale_steps),
            "adaptive_directional_mean_capital_exact":martingale_steps,
            "scope":"finite rational checks of the independently derived conditional-supermartingale argument"}


def check_invalid():
    invalid=[
        lambda:b.certificate_from_blocks([{"counts":[0,1,0,0],"plus_numerator":2,"minus_numerator":0,"denominator":2}]),
        lambda:b.certificate_from_blocks([{"counts":[0,-1,0,0],"plus_numerator":0,"minus_numerator":0,"denominator":2}]),
        lambda:b.certificate_from_blocks([],delta=0),
        lambda:b.certificate_from_blocks([],delta=1),
        lambda:b.certificate_from_blocks([],delta=True),
        lambda:b.predictable_bets([0,1,2]),
        lambda:b.predictable_bets([0,1,2,0],.5,None),
        lambda:b.predictable_bets([0,1,2,0],-.1,.5),
        lambda:b.predictable_bets([0,1,2,0],denominator=0),
        lambda:b.kelly_log_increments([0,1,0,0],1.5,0,2),
    ]
    for call in invalid:
        try:call()
        except (ValueError,TypeError):pass
        else:raise AssertionError("invalid input accepted")
    return len(invalid)


def main():
    started=time.perf_counter()
    result={"passed":True,"exact_arithmetic":check_exact_logs(),"vector_and_extreme":check_vectors_and_large(),
            "adaptive_null":check_null_expectations(),"invalid_inputs_rejected":check_invalid()}
    result["elapsed_seconds"]=time.perf_counter()-started
    result["source_sha256"]={name:hashlib.sha256((Path(__file__).resolve().parent/name).read_bytes()).hexdigest()
                             for name in ("betting.py","betting_checks.py")}
    result["numeric_source_sha256"]=b.NUMERIC_SOURCE_SHA256
    folder=ROOT/"results"/"streaming_validation"
    folder.mkdir(parents=True,exist_ok=True)
    (folder/"betting_checks.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
