"""Fixed-PSR two-arm Track-and-Stop-style control.

This reuses the frozen audited TrackStopState without changing its algorithm.
It models the two fixed arm means as independent Bernoulli parameters; it is
NOT the full structured all-angle algorithm of Degenne et al. (2019).
Unknown simulator truth is used only to generate independent observations.
"""
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/"experiments"/"circuit_validation"/"kl_trackstop.py"
_spec = importlib.util.spec_from_file_location("_faulttune_frozen_kl_trackstop",str(SOURCE))
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
TrackStopState = _module.TrackStopState
METHOD = "psr_trackstop"
SOURCE_SHA256 = hashlib.sha256(SOURCE.read_bytes()).hexdigest()


def _cp(count,total):
    return [0. if count==0 else float(beta.ppf(.025,count,total-count+1)),
            1. if count==total else float(beta.ppf(.975,count+1,total-count))]


def _summary(case,records):
    expected="accept" if case["C"]<0 else "reject" if case["C"]>0 else "abstain"
    shots=np.array([row["shots"] for row in records],dtype=np.int64)
    wrong=sum(row["decision"] not in ("abstain",expected) for row in records)
    correct=sum(row["decision"]==expected for row in records) if expected!="abstain" else 0
    correct_cost=sorted(row["shots"] for row in records if row["decision"]==expected and expected!="abstain")
    n95_rank=int(math.ceil(.95*len(records)))
    n95=correct_cost[n95_rank-1] if len(correct_cost)>=n95_rank else None
    return {"case_id":case["case_id"],"kind":case["kind"],"method":METHOD,
            "repetitions":len(records),"wrong":wrong,"correct":correct,
            "abstained":sum(row["decision"]=="abstain" for row in records),
            "error_ci95_pointwise":_cp(wrong,len(records)),
            "power_ci95_pointwise":_cp(correct,len(records)) if expected!="abstain" else None,
            "mean_shots":float(shots.mean()),"median_shots":float(np.median(shots)),
            "min_shots":int(shots.min()),"max_shots":int(shots.max()),
            "empirical_N95_correct":n95,"N95_censored":n95 is None,
            "N95_scope":"empirical order statistic; no simultaneous guarantee"}


def run_cell(case,repetitions,schedule,seed,delta=.01):
    """Return (summary, raw records), with a full increment ledger per trial.

    schedule contains total physical-shot budgets, shared with other methods.
    The original state chooses the next arm-count allocation predictably.
    Each trial has an independent registered seed, so individual replay works.
    """
    schedule=list(schedule)
    if type(repetitions) is not int or repetitions<1:
        raise ValueError("repetitions must be a positive integer")
    if not schedule or any(type(n) is not int or n<2 for n in schedule) or any(a>=b for a,b in zip(schedule,schedule[1:])):
        raise ValueError("schedule must contain strictly increasing integer budgets >=2")
    started=time.perf_counter()
    # These are simulator values; no statistical controller receives them.
    p_minus=case["A"]-case["C"]
    p_plus=case["A"]+case["C"]
    probabilities=(p_minus,p_plus)
    if not all(0<=p<=1 for p in probabilities):
        raise ValueError("nonphysical simulator probabilities")
    records=[]
    for rep in range(repetitions):
        trial_seed=(int(seed)+104729*rep)%(2**32)
        rng=np.random.RandomState(trial_seed)
        state=TrackStopState([-1.,1.],delta=delta,prior=.5)
        transcript=[]
        certificate=None
        decision="abstain"
        for look,budget in enumerate(schedule,1):
            increment=budget-sum(state.trials)
            plan=state.recommend_counts(increment)
            observed=[int(rng.binomial(int(n),p)) for n,p in zip(plan,probabilities)]
            for arm,(s,n) in enumerate(zip(observed,plan)):
                state.update(arm,s,n)
            transcript.append({"look_index":look,"total_shots":budget,
                               "arm_trials_added":plan,"arm_successes_added":observed})
            candidate=state.certificate(full_audit=look==len(schedule))
            if candidate["decision"]!="abstain" or look==len(schedule):
                certificate=candidate
            if candidate["decision"]!="abstain":
                decision=candidate["decision"]
                break
        records.append({"case_id":case["case_id"],"method":METHOD,"rep":rep,
                        "seed":trial_seed,"cell_seed":int(seed),"phase":"fixed_psr_trackstop",
                        "t":math.pi/2,"pilot_shots":0,
                        "counts":[state.successes[0],state.trials[0],state.successes[1],state.trials[1]],
                        "shots":sum(state.trials),"look_index":len(transcript),
                        "decision":decision,"certificate":certificate,"trajectory":transcript,
                        "source_sha256":SOURCE_SHA256})
    summary=_summary(case,records)
    summary.update({"elapsed_seconds":time.perf_counter()-started,
                    "source_sha256":SOURCE_SHA256,
                    "baseline_scope":"fixed two-arm adaptive allocation; not full all-angle structured identification"})
    return summary,records


def verify_record(record,schedule,delta=.01):
    """Independent count/certificate reconstruction without simulator truth."""
    state=TrackStopState([-1.,1.],delta=delta,prior=.5)
    for block in record["trajectory"]:
        look=block["look_index"]
        assert block["total_shots"]==schedule[look-1]
        plan=state.recommend_counts(block["total_shots"]-sum(state.trials))
        assert plan==block["arm_trials_added"]
        for arm,(s,n) in enumerate(zip(block["arm_successes_added"],plan)):
            state.update(arm,s,n)
    assert record["counts"]==[state.successes[0],state.trials[0],state.successes[1],state.trials[1]]
    assert record["shots"]==sum(state.trials)
    certificate=state.certificate(full_audit=True)
    assert certificate==record["certificate"]
    assert certificate["decision"]==record["decision"]
    assert record["decision"]!="abstain" or record["shots"]==schedule[-1]
    return True
