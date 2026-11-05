"""Independent formulas, causal ledgers, and timing checks for the new baseline."""
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import time

import mpmath as mp
import numpy as np
import dkm_baseline as dkm

DESTINATION = dkm.ROOT/"results/sota_round2/literature"


def reference_calibration(delta, dimension):
    def inverse(x):
        return -mp.lambertw(-mp.exp(-x), -1).real
    z = mp.mpf(3)/2
    x = -mp.log(mp.mpf(str(delta)))/dimension
    y = (inverse(1+x)+mp.log(2*mp.zeta(2)))/2
    if y >= 1/mp.log(z)-mp.log(1/mp.log(z)):
        root = inverse(y)
        ht = mp.exp(1/root)*root
    else:
        ht = z*(y-mp.log(mp.log(z)))
    return 2*dimension*ht


def check_thresholds():
    rows = []
    with mp.workdps(100):
        for delta in (.5,.1,.01,1e-6):
            for dimension in (1,2,6,14):
                counts = [1+17*j*j for j in range(dimension)]
                reference = reference_calibration(delta,dimension)+sum(3*mp.log(1+mp.log(n)) for n in counts)
                actual = mp.mpf(str(dkm.kk_threshold_upper(counts,delta)))
                gap = actual-reference
                assert gap >= 0 and gap < mp.mpf('1e-30'), (delta,dimension,gap)
                rows.append({"delta":delta,"arms":dimension,"counts":counts,
                             "outward_excess":mp.nstr(gap,12),"threshold":mp.nstr(actual,22)})
    return rows


def check_adahedge():
    dimension, cumulative, gap = 4, [mp.mpf(0)]*4, mp.mpf(0)
    learner = dkm.AdaHedge(dimension)
    largest_weight_error = largest_gap_error = 0.
    sequence = [[0,0,0,0],[1,1,1,1],[-2,1,0,3],[4,-3,1,0],
                [1,2,-4,3],[7,-2,0,-1],[-5,1,3,0]]*7
    with mp.workdps(90):
        for losses in sequence:
            actual, actual_rate = learner.distribution()
            if gap == 0:
                best = min(cumulative)
                mask = [x == best for x in cumulative]
                expected = [mp.mpf(int(x))/sum(mask) for x in mask]
                mix = min(mp.mpf(x) for x in losses)
            else:
                rate = mp.log(dimension)/gap
                weights = [mp.exp(-rate*(x-min(cumulative))) for x in cumulative]
                expected = [x/sum(weights) for x in weights]
                mix = -mp.log(sum(p*mp.exp(-rate*x) for p,x in zip(expected,losses)))/rate
            average = sum(p*x for p,x in zip(expected,losses))
            gap += average-mix
            cumulative = [a+b for a,b in zip(cumulative,losses)]
            learner.update(losses,actual,actual_rate)
            largest_weight_error = max(largest_weight_error,max(abs(float(e)-a) for e,a in zip(expected,actual)))
            largest_gap_error = max(largest_gap_error,abs(float(gap)-learner.gap))
        assert largest_weight_error < 1e-12 and largest_gap_error < 1e-11
    return {"rounds":len(sequence),"independent_method":"90-digit direct exponential-weight and mix-loss formulas",
            "maximum_weight_error":largest_weight_error,"maximum_gap_error":largest_gap_error}


def check_ledger(record, case):
    count, success = np.zeros(len(record["arms"]),dtype=int), np.zeros(len(record["arms"]),dtype=int)
    features = dkm.features(record["arms"])
    probabilities = features.dot([case["A"],case["B"],case["C"]])
    rng = np.random.RandomState(record["seed"])
    target = np.ones(len(count))
    maximum_interval_excess = maximum_oracle_gap = 0.
    failures = 0
    for step in record["history"]:
        assert step["observed_shots"] == int(count.sum())
        if step["stage"] == "dkm_game":
            weights = np.array(step["allocation"])
            assert np.all(weights >= 0) and abs(weights.sum()-1) < 1e-12
            target += step["batch_size"]*weights
            assert np.max(np.abs(target-step["tracking_target"])) < 1e-10
            assert step["arm"] == int(np.argmin(count/target))
            argument = mp.mpf('1.1')**2*mp.log(int(count.sum()))
            expected_bonus = float(argument+mp.log(argument)+mp.mpf('.5'))
            assert abs(expected_bonus-step["exploration_bonus"]) < 1e-12
            estimates = np.clip(success/count,step["design_margin"],1-step["design_margin"])
            q = features.dot(step["best_response_theta"])
            for arm,((lower,upper),estimate,n) in enumerate(zip(step["confidence_intervals"],estimates,count)):
                assert lower <= estimate <= upper
                divergences = []
                for endpoint in (lower,upper):
                    p, u = mp.mpf(float(estimate)), mp.mpf(float(endpoint))
                    value = p*mp.log(p/u)+(1-p)*mp.log((1-p)/(1-u))
                    maximum_interval_excess = max(maximum_interval_excess,float(n*value)-step["exploration_bonus"])
                    u, v = mp.mpf(float(endpoint)), mp.mpf(float(q[arm]))
                    divergences.append(float(u*mp.log(u/v)+(1-u)*mp.log((1-u)/(1-v))))
                expected = max(step["exploration_bonus"]/n,*divergences)
                assert abs(expected-step["optimistic_payoffs"][arm]) < 1e-8
            maximum_oracle_gap = max(maximum_oracle_gap,step["best_response_gap_float_diagnostic"])
            failures += int(not step["best_response_success"])
        else:
            assert step["arm"] == int(np.flatnonzero(count == 0)[0]) and step["batch_size"] == 1
        arm, length = step["arm"],step["batch_size"]
        assert step["successes"] == int(rng.binomial(length,probabilities[arm]))
        count[arm] += length
        success[arm] += step["successes"]
        if step["stage"] == "dkm_game" and length == 1:
            assert np.all(count-target <= 1+1e-10) and np.all(count-target >= -(len(count)-1)-1e-10)
    assert count.tolist() == record["trials"] and success.tolist() == record["successes"]
    assert int(count.sum()) == record["shots"] == record["terminal_plan"]["observed_shots"]
    assert record["terminal_plan_executed"] is False
    assert maximum_interval_excess < 1e-5
    assert maximum_oracle_gap <= 1e-5 and failures == 0
    # A feasible true null likelihood must not exceed the certified null upper.
    if record["certificate"]["sign"]*case["C"] <= 0:
        observed = dkm.observations(record["arms"],count,success)
        true_bounds = dkm.likelihood.loglik_bounds(observed,[case["A"],case["B"],case["C"]])
        assert Decimal(true_bounds["lower_decimal"]) <= Decimal(record["certificate"]["null_certificate"]["loglik_upper_decimal"])
    return {"rng_blocks_replayed":len(record["history"]),"maximum_interval_KL_excess":maximum_interval_excess,
            "maximum_oracle_gap_float_diagnostic":maximum_oracle_gap,"unsuccessful_best_responses":failures,
            "current_count_terminal_label_verified":True}


def main():
    started = time.perf_counter()
    thresholds = check_thresholds()
    hedge = check_adahedge()
    probes = []
    configurations = [
        ({"A":.5,"B":0.,"C":.45},512,"kk21",1),
        ({"A":.5,"B":0.,"C":.45},512,"kt_safe",1),
        ({"A":.07,"B":.06,"C":.01},512,"kt_safe",1),
        ({"A":.012,"B":.0114,"C":.0015},512,"kt_safe",1),
        ({"A":.28,"B":.2,"C":0.},4096,"kt_safe",1),
        ({"A":.28,"B":.2,"C":0.},256,"kt_safe",8)]
    for index,(case,cap,calibration,batch) in enumerate(configurations):
        began = time.perf_counter()
        result = dkm.run_trial(case,2200101+index,cap=cap,calibration=calibration,batch_size=batch,check_every=1)
        seconds = time.perf_counter()-began
        ledger = check_ledger(result,case)
        summary = {"case":case,"seed":result["seed"],"cap":cap,"calibration":calibration,
                   "batch_size":batch,"check_every":1,"shots":result["shots"],"decision":result["decision"],
                   "runner_seconds":seconds,"milliseconds_per_paid_shot":1000*seconds/result["shots"],
                   "linear_extrapolation_4096_seconds":seconds/result["shots"]*4096,
                   "checks":ledger,"timing_limit":"Engineering measurement on this machine, not a performance comparison or precise runtime prediction."}
        probes.append(summary)
        print(json.dumps(summary),flush=True)
    report = {"status":"passed","checked_utc":datetime.now(timezone.utc).isoformat(),
              "source_sha256":hashlib.sha256(Path(dkm.__file__).read_bytes()).hexdigest(),
              "checker_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "thresholds":thresholds,"adahedge":hedge,"engineering_probes":probes,
              "elapsed_seconds":time.perf_counter()-started,
              "scope":"Independent formula and mechanical checks plus runtime probes; not proof of SOTA superiority or a new held-out comparison."}
    (DESTINATION/"dkm_checks.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"passed","elapsed_seconds":report["elapsed_seconds"],"source_sha256":report["source_sha256"]}),flush=True)


if __name__ == "__main__":
    main()
