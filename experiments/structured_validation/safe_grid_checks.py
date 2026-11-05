"""Deterministic exact checks for the full-count fixed-odds safe grid."""
import hashlib
import json
import math
from decimal import Decimal,localcontext
from fractions import Fraction
from pathlib import Path
import time

import safe_grid as module
import expert_config


ROOT=Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fraction(value):
    return Fraction(value[0],value[1])


def exact_capital(blocks,config):
    """Independent marginal likelihood calculation; no reconstructed pairing."""
    result=Fraction(0)
    for row in config["experts"]:
        odds=Fraction(row["numerator"],row["denominator"])
        capital=Fraction(1)
        for block in blocks:
            if odds==1:
                continue
            pole=odds if odds>1 else 1/odds
            prediction=block["prediction"]
            index=[fraction(pair) for pair in prediction["odds_poles"]].index(pole)
            u=Fraction(prediction["lower_numerators"][index],prediction["lower_denominator"])
            v=pole*u/(1-u+pole*u)
            pm,pp=(u,v) if odds>1 else (v,u)
            r=(pm+pp)/2
            m,sm,sp=block["m"],block["minus_successes"],block["plus_successes"]
            capital*=((pm/r)**sm*((1-pm)/(1-r))**(m-sm)
                      *(pp/r)**sp*((1-pp)/(1-r))**(m-sp))
        result+=fraction(row["weight"])*capital
    return result


def check_certificate(blocks,config):
    exact=exact_capital(blocks,config)
    certificate=module.certificate_from_blocks(blocks,config=config)
    with localcontext() as context:
        context.prec=120
        reference=(Decimal(exact.numerator)/Decimal(exact.denominator)).ln()
    lo,hi=map(Decimal,certificate["evidence_bounds"])
    assert lo<=reference<=hi,(lo,reference,hi)
    score=sum(block["plus_successes"]-block["minus_successes"] for block in blocks)
    assert certificate["score"]==score
    assert certificate["shots"]==2*sum(block["m"] for block in blocks)
    if certificate["decision"]!="abstain":
        assert exact>=100 and score!=0
        assert certificate["decision"]==("reject" if score>0 else "accept")
    return certificate


def main():
    started=time.perf_counter()
    source=Path(module.__file__).resolve()
    source_hash=digest(source)
    checks_source=Path(__file__).resolve()
    checks_hash=digest(checks_source)
    default=module.protocol_config()
    jeffreys71=module.protocol_config(expert_config.protocol_config())
    small=module.protocol_config({"experts":[
        {"numerator":1,"denominator":1,"weight":[1,5]},
        {"numerator":3,"denominator":1,"weight":[2,5]},
        {"numerator":1,"denominator":3,"weight":[2,5]}]})
    asymmetric=module.protocol_config({"experts":[
        {"numerator":1,"denominator":1,"weight":[1,10]},
        {"numerator":2,"denominator":1,"weight":[1,5]},
        {"numerator":1,"denominator":2,"weight":[3,10]},
        {"numerator":5,"denominator":1,"weight":[1,10]},
        {"numerator":1,"denominator":5,"weight":[3,10]}]})
    assert module.protocol_config(default)==default
    assert len(default["experts"])==33
    assert len(jeffreys71["experts"])==71
    identity_checks=expectation_checks=pairing_checks=0
    probabilities=[Fraction(1,10),Fraction(1,3),Fraction(1,2),Fraction(2,3),Fraction(9,10)]
    for center in (0.,1e-8,.001,.015,.2,.5,.8,.985,.999,1-1e-8,1.):
        prediction=module.predict_endpoints(center,default)
        denominator=prediction["lower_denominator"]
        for pole_index in (0,7,15):
            numerator=prediction["lower_numerators"][pole_index]
            p,d=prediction["odds_poles"][pole_index]
            odds=Fraction(p,d)
            u,v=module.rational_endpoints(numerator,denominator,p,d)
            factors=module.rational_factors(numerator,denominator,p,d)
            assert 0<u<v<1
            assert v*(1-u)/(u*(1-v))==odds
            assert factors[1]/factors[2]==odds
            assert factors[0]*factors[3]==factors[1]*factors[2]
            assert all(factors[j]*factors[k]<=1 for j,k in enumerate((0,2,1,3)))
            identity_checks+=5
            # Independent pairing choices give exactly the same marginal evidence.
            m,sm,sp=6,3,4
            capitals=[]
            for both in range(max(0,sm+sp-m),min(sm,sp)+1):
                counts=(m-sm-sp+both,sp-both,sm-both,both)
                capital=Fraction(1)
                for factor,count in zip(factors,counts):capital*=factor**count
                capitals.append(capital)
            assert len(set(capitals))==1
            pairing_checks+=1
            for pm in probabilities:
                for pp in probabilities:
                    distribution=((1-pm)*(1-pp),(1-pm)*pp,pm*(1-pp),pm*pp)
                    if pm>=pp:
                        assert sum((f*w for f,w in zip(factors,distribution)),Fraction(0))<=1
                        expectation_checks+=1
                    if pp>=pm:
                        assert sum((factors[k]*distribution[j] for j,k in enumerate((0,2,1,3))),Fraction(0))<=1
                        expectation_checks+=1
    certificate_checks=screen_checks=reflection_checks=0
    for config in (default,small,asymmetric,jeffreys71):
        state=module.empty_state(config)
        blocks=[]
        for m,sm,sp,q in ((3,1,2,.2),(5,1,4,.7),(7,2,6,.4),(2,1,0,.9)):
            prediction=module.predict_endpoints(q,config)
            module.update_block(state,m,sm,sp,prediction)
            blocks.append({"m":m,"minus_successes":sm,"plus_successes":sp,"prediction":prediction})
            cert=check_certificate(blocks,config)
            certificate_checks+=1
            screen=module.screen(state)
            assert screen["shots"]==cert["shots"] and screen["score"]==cert["score"]
            assert abs(screen["log_evidence"]-cert["evidence"])<2e-11
            screen_checks+=1
        if config is not asymmetric:
            reflected=[dict(block,minus_successes=block["plus_successes"],plus_successes=block["minus_successes"])
                       for block in blocks]
            assert exact_capital(blocks,config)==exact_capital(reflected,config)
            reflected_cert=check_certificate(reflected,config)
            assert reflected_cert["score"]==-cert["score"]
            reflection_checks+=1
    # A genuine crossing and its opposite, using exact independent marginal likelihoods.
    for sm,sp in ((0,100),(100,0)):
        prediction=module.predict_endpoints(.5,small)
        cert=check_certificate([{"m":100,"minus_successes":sm,"plus_successes":sp,"prediction":prediction}],small)
        assert cert["decision"]==("reject" if sp>sm else "accept")
        certificate_checks+=1
    zero=module.empty_state(default)
    module.update_block(zero,0,0,0,module.predict_endpoints(.5,default))
    assert module.certificate_from_blocks(zero["blocks"],config=default)["decision"]=="abstain"
    extreme_screen_checks=0
    for config in (default,jeffreys71):
        for q in (0.,1.):
            state=module.empty_state(config)
            module.update_block(state,8,1,7,module.predict_endpoints(q,config))
            cert=check_certificate(state["blocks"],config)
            assert math.isfinite(module.screen(state)["log_evidence"])
            extreme_screen_checks+=1
    # Exhaustively enumerate adaptive-nuisance paths under three common-order laws.
    horizon=4
    exhaustive={}
    for scenario in ("null","positive","negative"):
        terminal_mass=terminal_mixture=wrong_mass=Fraction(0)
        def visit(depth,minus_total,plus_total,cap_plus,cap_minus,mass,ever_wrong):
            nonlocal terminal_mass,terminal_mixture,wrong_mass
            mixture=Fraction(1,5)+Fraction(2,5)*(cap_plus+cap_minus)
            score=plus_total-minus_total
            wrong=(score!=0) if scenario=="null" else (score<0 if scenario=="positive" else score>0)
            crossed=mixture>=2 and wrong
            ever_wrong=ever_wrong or crossed
            if depth==horizon:
                terminal_mass+=mass
                terminal_mixture+=mass*mixture
                if ever_wrong:wrong_mass+=mass
                return
            center=(minus_total+plus_total+.5)/(2*depth+1)
            prediction=module.predict_endpoints(center,small)
            factors=module.rational_factors(prediction["lower_numerators"][0],prediction["lower_denominator"],3,1)
            base=Fraction(1,5) if (minus_total+plus_total)%2==0 else Fraction(2,5)
            pm,pp=(base,base) if scenario=="null" else (base,base+Fraction(1,5)) if scenario=="positive" else (base+Fraction(1,5),base)
            probabilities=((1-pm)*(1-pp),(1-pm)*pp,pm*(1-pp),pm*pp)
            for outcome,(ym,yp) in enumerate(((0,0),(0,1),(1,0),(1,1))):
                visit(depth+1,minus_total+ym,plus_total+yp,cap_plus*factors[outcome],
                      cap_minus*factors[(0,2,1,3)[outcome]],mass*probabilities[outcome],ever_wrong)
        visit(0,0,0,Fraction(1),Fraction(1),Fraction(1),False)
        assert terminal_mass==1 and wrong_mass<=Fraction(1,2)
        if scenario=="null":assert terminal_mixture<=1
        exhaustive[scenario]={"paths":4**horizon,"wrong_crossing_probability_exact":str(wrong_mass),
                              "expected_terminal_mixture_exact":str(terminal_mixture)}
    invalid_checks=0
    invalid_calls=[lambda:module.predict_endpoints(float("nan")),lambda:module.predict_endpoints(-.1),
                   lambda:module.predict_endpoints(True),lambda:module.protocol_config({"experts":[]}),
                   lambda:module.protocol_config({"experts":[{"numerator":1,"denominator":1,"weight":[1,2]}]}),
                   lambda:module.certificate_from_blocks([],delta=0),lambda:module.certificate_from_blocks([],delta=1),
                   lambda:module.certificate_from_blocks([],delta=True),lambda:module.rational_endpoints(0,10,3,1),
                   lambda:module.update_block(module.empty_state(),1,2,0,module.predict_endpoints(.5))]
    for call in invalid_calls:
        try:call()
        except (ValueError,TypeError,ArithmeticError):invalid_checks+=1
        else:raise AssertionError("invalid input accepted")
    # Deterministic computational timing, not a sampled experiment.
    state=module.empty_state(default)
    update_started=time.perf_counter()
    for k in range(64):
        m=32+16*(k%8)
        prediction=module.predict_endpoints(.2+.005*(k%7),default)
        module.update_block(state,m,m//5,m//4,prediction)
    update_seconds=time.perf_counter()-update_started
    cert_started=time.perf_counter()
    cert=module.certificate_from_blocks(state["blocks"],config=default)
    cert_seconds=time.perf_counter()-cert_started
    assert digest(source)==source_hash and digest(checks_source)==checks_hash
    result={"status":"passed","protocol":default,"source_sha256":source_hash,
            "checks_source_sha256":checks_hash,"exact_factor_identities":identity_checks,
            "jeffreys71_config_sha256":module.config_digest(jeffreys71),
            "jeffreys71_source_sha256":digest(Path(expert_config.__file__)),
            "exact_ordered_null_expectation_checks":expectation_checks,"pairing_invariance_checks":pairing_checks,
            "rational_capital_certificate_enclosures":certificate_checks+extreme_screen_checks+reflection_checks,
            "float_screen_agreements":screen_checks,"reflection_checks":reflection_checks,
            "extreme_prediction_checks":extreme_screen_checks,"invalid_inputs_rejected":invalid_checks,
            "adaptive_nuisance_exhaustive_checks":exhaustive,
            "timing":{"type":"deterministic constructed counts, not a sampling benchmark",
                      "blocks":64,"shots":state["shots"],"update_seconds":update_seconds,"certificate_seconds":cert_seconds},
            "limits":["Exhaustive checks support the implementation but do not replace the mathematical common-order proof.",
                      "No claim of finite-budget superiority is implied by correctness or computational timing."],
            "elapsed_seconds":time.perf_counter()-started}
    destination=ROOT/"results"/"structured_validation"/"safe_grid_checks.json"
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({key:value for key,value in result.items() if key not in ("protocol","adaptive_nuisance_exhaustive_checks")},indent=2))


if __name__=="__main__":main()
