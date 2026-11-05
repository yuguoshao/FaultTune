"""Read-only information diagnostics for the frozen streaming development run.

True coefficients are used ONLY here for post-hoc diagnostics, never by a
policy. Oracle rate ratios are known-parameter/asymptotic quantities, not
finite-cost guarantees. Expected information along a realized stopped design
path is descriptive; it is not an independent performance experiment.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from benchmark import ROOT,case_matrix,probability
from designs import pair_scores,fisher_rate,uniform_variance_shift,half_coin_kl_from_bias


def information(case,angles):
    pm=probability(case,-angles)
    pp=probability(case,angles)
    variance,pair,discordance=pair_scores(pp,pm,np.sin(angles))
    return variance,pair,fisher_rate(pp,pm),pm,pp,discordance


def oracle(case):
    grid=np.linspace(.2,math.pi-.2,4097)
    rates=information(case,grid)
    opt={}
    for name,scores in (("paired",rates[1]),("fisher",rates[2])):
        index=int(np.argmax(scores))
        result=minimize_scalar(lambda x:-float(information(case,x)[1 if name=="paired" else 2]),
                               bounds=(grid[max(0,index-1)],grid[min(len(grid)-1,index+1)]),
                               method="bounded",options={"xatol":1e-12})
        opt[name+"_t"]=float(result.x)
        opt[name+"_rate"]=-float(result.fun)
    fit={name:np.array([case[name]]) for name in ("A","B","C")}
    closed=float(uniform_variance_shift(fit)[0][0])
    baseline=information(case,math.pi/2)
    closed_info=information(case,closed)
    opt.update({"case_id":case["case_id"],"A":case["A"],"rho":case.get("rho"),
                "B_over_R":case.get("B_over_R"),"kind":case["kind"],
                "psr_paired_rate":float(baseline[1]),"psr_fisher_rate":float(baseline[2]),
                "closed_t":closed,"closed_variance":float(closed_info[0]),
                "variance_gain":float(baseline[0]/closed_info[0]),
                "closed_paired_rate":float(closed_info[1]),
                "closed_fisher_rate":float(closed_info[2])})
    for engine in ("paired","fisher"):
        opt[engine+"_rate_gain"]=opt[engine+"_rate"]/opt["psr_"+engine+"_rate"] if case["C"] else None
    return opt


def realized_diagnostic(row,case,reference):
    data=np.array(row["blocks"],dtype=float)
    angles=data[:,0]
    n=data[:,1:5].sum(axis=1)
    shots=2*n
    previous=2*np.cumsum(n)-shots
    variance,pair_rate,fisher,pm,pp,discordance=information(case,angles)
    a=(1-pm)*pp
    d=pm*(1-pp)
    paired_info=float(np.dot(shots,pair_rate))
    fisher_info=float(np.dot(shots,fisher))
    total_a,total_d=float(np.dot(n,a)),float(np.dot(n,d))
    total_disc=total_a+total_d
    pooled_info=float(total_disc*half_coin_kl_from_bias((total_a-total_d)/total_disc)) if total_disc else 0.
    prefix,engine=row["method"].split("_")
    forced=(previous>=256)&(previous<512) if prefix!="psr" else np.zeros(len(n),dtype=bool)
    adapting=previous>=512
    target_rate=fisher if engine=="safe" else pair_rate
    oracle_rate=reference["fisher_rate" if engine=="safe" else "paired_rate"]
    baseline_rate=reference["psr_fisher_rate" if engine=="safe" else "psr_paired_rate"]
    result={"shots":row["shots"],"explore_shots":int(shots[forced].sum()),
            "angle_information":float(np.dot(shots,target_rate)),
            "oracle_constant_angle_information":float(row["shots"]*oracle_rate),
            "forced_explore_information_loss_vs_psr":float(np.dot(shots[forced],baseline_rate-target_rate[forced])),
            "adaptation_information_loss_vs_oracle":float(np.dot(shots[adapting],oracle_rate-target_rate[adapting])),
            "paired_additive_information":paired_info,"paired_pooled_information":pooled_info,
            "pooled_beta_jensen_loss":max(0.,paired_info-pooled_info),
            "fisher_additive_information":fisher_info}
    if engine!="beta":
        positive=case["C"]>0
        if engine=="bet":
            selected=data[:,5 if positive else 6]/2**20
            good,bad=(a,d) if positive else (d,a)
            expected_log=good*np.log1p(selected)+bad*np.log1p(-selected)
            predictor_info=float(np.dot(n,expected_log))
            optimal_info=paired_info
            capital_key="log_kelly_"
        else:
            predm,predp=data[:,5]/2**20,data[:,6]/2**20
            center=(predm+predp)/2
            log00=np.log((1-predm)/(1-center))+np.log((1-predp)/(1-center))
            log01=np.log((1-predm)/(1-center))+np.log(predp/center)
            log10=np.log(predm/center)+np.log((1-predp)/(1-center))
            log11=np.log(predm/center)+np.log(predp/center)
            expected_log=((1-pm)*(1-pp)*log00+a*log01+d*log10+pm*pp*log11)
            enabled=predp>=predm if positive else predm>=predp
            predictor_info=float(np.dot(n,expected_log*enabled))
            optimal_info=fisher_info
            capital_key="log_safe_"
        result["predictor_expected_log_capital"]=predictor_info
        result["prediction_information_loss"]=max(0.,optimal_info-predictor_info)
        direction="plus" if positive else "minus"
        cert=row["certificate"]
        main=float(np.mean(cert[capital_key+direction+"_bounds"]))
        fixed=float(np.mean(cert["log_fixed_"+direction+"_bounds"]))
        hedge=float(np.mean(cert["log_hedge_"+direction+"_bounds"]))
        result["observed_hedge_log_penalty_vs_best_component"]=max(main,fixed)-hedge
        result["fixed_component_won"]=int(fixed>main)
    return result


def summarize(rows):
    count=len(rows)
    sums={key:float(math.fsum(row.get(key,0.) for row in rows)) for key in rows[0]}
    info=sums["angle_information"]
    result={"trajectories":count,
            "mean_shots":sums["shots"]/count,
            "mean_explore_shots":sums["explore_shots"]/count,
            "angle_information_over_oracle":info/sums["oracle_constant_angle_information"],
            "forced_explore_loss_fraction_of_angle_information":sums["forced_explore_information_loss_vs_psr"]/info,
            "adaptive_misdesign_loss_fraction_of_angle_information":sums["adaptation_information_loss_vs_oracle"]/info,
            "pooled_beta_jensen_loss_fraction":sums["pooled_beta_jensen_loss"]/sums["paired_additive_information"]}
    if "prediction_information_loss" in sums:
        result["mean_prediction_information_loss"]=sums["prediction_information_loss"]/count
        result["prediction_loss_fraction_of_angle_information"]=sums["prediction_information_loss"]/info
        result["predictor_expected_log_capital_per_trial"]=sums["predictor_expected_log_capital"]/count
        result["fixed_component_win_fraction"]=sums["fixed_component_won"]/count
        result["mean_observed_hedge_log_penalty"]=sums["observed_hedge_log_penalty_vs_best_component"]/count
    return result


def main(directory):
    cases=case_matrix()
    references=[oracle(case) for case in cases if case["kind"]=="main"]
    reference_by_id={x["case_id"]:x for x in references}
    case_by_id={c["case_id"]:c for c in cases}
    oracle_summary={}
    for key in ("variance_gain","paired_rate_gain","fisher_rate_gain"):
        values=[row[key] for row in references]
        oracle_summary[key]={"geomean":math.exp(np.mean(np.log(values))),
                             "maximum":max(values),"exactly_no_gain_cases":sum(abs(v-1)<1e-9 for v in values)}
    groups={}
    path=Path(directory)/"trials.jsonl"
    if path.is_file():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    row=json.loads(line)
                except json.JSONDecodeError:
                    break
                if row["case_id"] not in reference_by_id:
                    continue
                item=realized_diagnostic(row,case_by_id[row["case_id"]],reference_by_id[row["case_id"]])
                groups.setdefault(row["method"],[]).append(item)
    result={"status":"post-hoc known-parameter information diagnostics; no new simulated trajectories",
            "source_run":str(Path(directory).resolve()),"source_run_complete":(Path(directory)/"summary.json").is_file(),
            "oracle_summary":oracle_summary,"oracle_cases":references,
            "realized_path_diagnostics":{method:summarize(rows) for method,rows in groups.items()},
            "scope":["Oracle rates describe known coefficients and leading asymptotic information, not paid finite-cost gains.",
                     "Trajectory diagnostics condition on the realized stopped angle path and use true coefficients only retrospectively.",
                     "Pooled-Beta Jensen loss separates averaging differently informative discordance streams from design loss.",
                     "Prediction loss is the oracle fixed-angle information minus the predictable strategy component expected log growth.",
                     "A half-half hedge loses at most log2 relative to its better component on each path."],
            "source_sha256":hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()}
    if result["source_run_complete"]:
        completed=json.loads((Path(directory)/"summary.json").read_text(encoding="utf-8"))
        expected=len(references)*completed["spec"]["repetitions"]
        assert all(len(groups[method])==expected for method in completed["spec"]["methods"])
        result["main_records_diagnosed"]=sum(len(rows) for rows in groups.values())
        result["source_ledger_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
    output=ROOT/"results"/"streaming_validation"/"diagnostics.json"
    output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({key:value for key,value in result.items() if key!="oracle_cases"},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--directory",default=str(ROOT/"results"/"streaming_validation"/"development_v1"))
    args=parser.parse_args()
    main(args.directory)
