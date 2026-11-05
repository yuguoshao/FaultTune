"""Independent complete audit of the fixed-PSR UI development probe."""
import hashlib
import json
import math
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
import time

import numpy as np
from scipy.special import betaln,gammaln,xlogy

import probe_fixed as runner
from direct_tests import _Bounds,_log_factorial,certificate_from_pairs

ROOT=Path(__file__).resolve().parents[2]
METHODS=("psr_beta","psr_full_ui")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def budgets(cap):
    result=[]
    value=32
    while value<cap:
        result.append(value)
        value=2*((9*value+15)//16)
    return sorted(set(result+[cap]))


def exact_marginal(successes,trials):
    failures=trials-successes
    return Fraction(math.factorial(2*successes)*math.factorial(2*failures),
                    4**trials*math.factorial(successes)*math.factorial(failures)*math.factorial(trials))


def marginal_log_bounds(successes,trials):
    failures=trials-successes
    return (_log_factorial(2*successes)+_log_factorial(2*failures)
            -2*trials*_Bounds(2).log()-_log_factorial(successes)
            -_log_factorial(failures)-_log_factorial(trials))


def independent_ui_bounds(counts):
    a,b,c,d=counts
    n=sum(counts)
    plus,minus=b+d,c+d
    numerator=marginal_log_bounds(plus,n)+marginal_log_bounds(minus,n)
    total=plus+minus
    denominator=_Bounds(0)
    if total:denominator+=total*(_Bounds(total)/(2*n)).log()
    if total<2*n:denominator+=(2*n-total)*(_Bounds(2*n-total)/(2*n)).log()
    return numerator-denominator


def density_checks():
    normalized=0
    for n in list(range(17))+[32]:
        assert sum((math.comb(n,k)*exact_marginal(k,n) for k in range(n+1)),Fraction(0))==1
        normalized+=1
    for past_n,past_s in ((0,0),(4,0),(4,2),(4,4),(9,3)):
        for n in (1,2,5):
            assert sum((math.comb(n,k)*exact_marginal(past_s+k,past_n+n)/exact_marginal(past_s,past_n)
                        for k in range(n+1)),Fraction(0))==1
            normalized+=1
    multinomial=0
    for pm,pp in ((Fraction(1,10),Fraction(1,5)),(Fraction(1,3),Fraction(2,3)),(Fraction(4,5),Fraction(9,10))):
        for m in range(6):
            total_probability=Fraction(0)
            for a in range(m+1):
                for b in range(m-a+1):
                    for c in range(m-a-b+1):
                        d=m-a-b-c
                        decomposition=(math.comb(m,c+d)*pm**(c+d)*(1-pm)**(a+b)
                                       *math.comb(c+d,d)*pp**d*(1-pp)**c
                                       *math.comb(a+b,b)*pp**b*(1-pp)**a)
                        coefficient=math.factorial(m)//(math.factorial(a)*math.factorial(b)*math.factorial(c)*math.factorial(d))
                        direct=(coefficient*((1-pm)*(1-pp))**a*((1-pm)*pp)**b
                                *(pm*(1-pp))**c*(pm*pp)**d)
                        assert decomposition==direct
                        total_probability+=direct
                        multinomial+=1
            assert total_probability==1
    return normalized,multinomial


def main():
    started=time.perf_counter()
    source=Path(__file__).resolve()
    source_hash=digest(source)
    folder=ROOT/"results"/"structured_validation"/"fixed_psr_probe"
    spec=json.loads((folder/"spec.json").read_text(encoding="utf-8"))
    summary=json.loads((folder/"summary.json").read_text(encoding="utf-8"))
    assert summary["spec"]==spec
    assert spec["status"]=="development only" and spec["paired_stream"] is True
    assert spec["repetitions"]==32 and spec["cap"]==2**22 and spec["delta"]==.01
    assert spec["seed_base"]==410000000
    for name,expected in spec["sources"].items():assert digest(ROOT/name)==expected,name
    public_path=ROOT/"results"/"learned_validation"/"development_v1"/"spec.json"
    public=json.loads(public_path.read_text(encoding="utf-8"))["cases"]
    expected_cases=[case for case in public if case["kind"]=="main" and case["C"]>0]
    assert spec["cases"]==expected_cases and len(expected_cases)==24
    schedule=budgets(spec["cap"])
    assert schedule==runner.schedule(spec["cap"])
    normalized,multinomial=density_checks()
    rows={}
    history_blocks=certificates=independent_ui=abstentions=0
    with (folder/"trials.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row=json.loads(line)
            key=(row["case_id"],row["rep"],row["method"])
            assert key not in rows and row["method"] in METHODS and 0<=row["rep"]<32
            assert row["seed"]==410000000+row["case_id"]*1009+row["rep"]*1000003
            cert=row["certificate"]
            assert cert["decision"] in ("accept","reject","abstain")
            assert row["history"] and len(row["history"])<=len(schedule)
            total=[0]*4
            previous=0
            for index,block in enumerate(row["history"]):
                assert len(block)==5 and block[0]==schedule[index]
                assert all(type(value) is int and value>=0 for value in block)
                assert 2*sum(block[1:])==block[0]-previous
                total=[left+right for left,right in zip(total,block[1:])]
                previous=block[0]
                history_blocks+=1
            assert previous==cert["shots"]==2*sum(total) and total==cert["counts"]
            if cert["decision"]=="abstain":
                assert cert["shots"]==spec["cap"]
                abstentions+=1
            expected=(certificate_from_pairs(total,delta=spec["delta"]) if row["method"]=="psr_beta"
                      else runner.full_ui_certificate(total,delta=spec["delta"]))
            assert expected==cert,key
            certificates+=1
            if row["method"]=="psr_full_ui":
                interval=independent_ui_bounds(total)
                lo,hi=map(Decimal.from_float,cert["evidence_bounds"])
                assert lo<=interval.hi and hi>=interval.lo,(key,"independent factorial enclosure")
                independent_ui+=1
            rows[key]=row
    expected_keys={(case["case_id"],rep,method) for case in expected_cases for rep in range(32) for method in METHODS}
    assert set(rows)==expected_keys and len(rows)==summary["new_trajectories"]==1536
    replay_blocks=replay_records=shared_prefix=0
    seeds=set()
    for case in expected_cases:
        cid=case["case_id"]
        pm,pp=case["A"]-case["C"],case["A"]+case["C"]
        assert 0<pm<pp<1
        for rep in range(32):
            local={method:rows[cid,rep,method] for method in METHODS}
            seed=410000000+cid*1009+rep*1000003
            assert seed not in seeds
            seeds.add(seed)
            smaller=min(len(row["history"]) for row in local.values())
            assert local[METHODS[0]]["history"][:smaller]==local[METHODS[1]]["history"][:smaller]
            shared_prefix+=smaller
            longest=max((row["history"] for row in local.values()),key=len)
            random=np.random.RandomState(seed)
            counts=[0]*4
            stopped={}
            previous=0
            for block in longest:
                budget=block[0]
                n=(budget-previous)//2
                old=int(random.binomial(n,pm))
                both=int(random.binomial(old,pp))
                worse=int(random.binomial(n-old,pp))
                fresh=[n-old-worse,worse,old-both,both]
                assert fresh==block[1:],(cid,rep,budget,"random decomposition")
                counts=[left+right for left,right in zip(counts,fresh)]
                a,b,c,d=counts
                pairs=sum(counts)
                beta_score=gammaln(b+.5)+gammaln(c+.5)-gammaln(b+c+1)-2*gammaln(.5)+(b+c)*math.log(2)
                kp,km=b+d,c+d
                q=(kp+km)/(2*pairs)
                ui_score=(betaln(kp+.5,pairs-kp+.5)+betaln(km+.5,pairs-km+.5)
                          -2*betaln(.5,.5)-xlogy(kp+km,q)-xlogy(2*pairs-kp-km,1-q))
                for method,value in zip(METHODS,(beta_score,ui_score)):
                    if method in stopped:continue
                    if value>=math.log(100)-1e-7 or budget==spec["cap"]:
                        cert=(certificate_from_pairs(counts) if method=="psr_beta" else runner.full_ui_certificate(counts))
                        if cert["decision"]!="abstain" or budget==spec["cap"]:
                            assert cert==local[method]["certificate"],(cid,rep,method,"first certified stop")
                            stopped[method]=budget
                previous=budget
                replay_blocks+=1
            assert set(stopped)==set(METHODS)
            replay_records+=2
        print(json.dumps({"audited_case":cid,"seed_replay_records":replay_records}),flush=True)
    independent_cells=[]
    ratios=[]
    totals={method:{"trials":0,"wrong":0,"abstained":0} for method in METHODS}
    for case in expected_cases:
        local_means={}
        for method in METHODS:
            certificates_local=[rows[case["case_id"],rep,method]["certificate"] for rep in range(32)]
            cell={"case_id":case["case_id"],"method":method,"kind":"main","repetitions":32,
                  "mean_shots":math.fsum(cert["shots"] for cert in certificates_local)/32,
                  "wrong":sum(cert["decision"]=="accept" for cert in certificates_local),
                  "abstained":sum(cert["decision"]=="abstain" for cert in certificates_local)}
            independent_cells.append(cell)
            local_means[method]=cell["mean_shots"]
            totals[method]["trials"]+=32
            totals[method]["wrong"]+=cell["wrong"]
            totals[method]["abstained"]+=cell["abstained"]
        ratios.append(local_means["psr_full_ui"]/local_means["psr_beta"])
    assert independent_cells==summary["cells"]
    ratio=math.exp(math.fsum(math.log(value) for value in ratios)/len(ratios))
    assert math.isclose(ratio,summary["full_ui_over_beta"],rel_tol=2e-14)
    assert digest(source)==source_hash
    result={"status":"passed","method_records":len(rows),"independent_joint_replicate_seeds":len(seeds),
            "records_exactly_replayed":replay_records,"unique_random_history_blocks_replayed":replay_blocks,
            "method_history_blocks_checked":history_blocks,"shared_prefix_blocks_checked":shared_prefix,
            "final_certificates_recomputed":certificates,"full_cap_abstention_certificates":abstentions,
            "ui_certificates_independently_checked_by_factorial_formula":independent_ui,
            "cells_recomputed":len(independent_cells),"exact_predictive_normalizations":normalized,
            "exact_multinomial_decomposition_checks":multinomial,"pinned_sources_verified":len(spec["sources"]),
            "aggregate":{"full_ui_over_beta":ratio,"minimum_case_ratio":min(ratios),"maximum_case_ratio":max(ratios),
                         "faster_cases":sum(value<1 for value in ratios),"tied_cases":sum(value==1 for value in ratios),
                         "slower_cases":sum(value>1 for value in ratios),"safety_counts":totals},
            "review":{"predictive_density":"Independent Jeffreys arm marginals; exact posterior block normalization checked.",
                      "sampling":"Every independent seed and every shared-stream binomial decomposition replayed.",
                      "costs":"Every physical pair charged; both methods stop on prefixes of the same jointly coupled stream.",
                      "stopping":"Original common checkpoints and first certified crossing reconstructed for both methods.",
                      "scope":"Old public positive-sign development cases; no independent confirmation or SOTA claim."},
            "audit_source_sha256":source_hash,"audit_source_unchanged_during_execution":True,
            "public_case_source_sha256":digest(public_path),
            "artifact_sha256":{name:digest(folder/name) for name in ("spec.json","summary.json","trials.jsonl")},
            "elapsed_seconds":time.perf_counter()-started}
    (folder/"audit.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
