"""Development comparison of conditional Fisher and matched paired evidence.

Every repetition has its own random stream. Policies share the stream until
their paid observations diverge; engines within a policy share observations.
Each engine has its own stopping cost. No true parameter enters a policy.
"""
import argparse
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.special import gammaln, logsumexp, xlogy

import conditional_fisher as cf
from predictive import ROOT, _Bounds, _log_marginal, fit_rows, arm_key
from certified_likelihood import rational_circle
from design_policy import GRID
from direct_tests import certificate_from_pairs
from probe_fixed import schedule


ENGINES = ("beta", "pair_grid", "fisher", "hedge")
POLICIES = ("psr", "closed", "grid")
EXPLORE_AFTER = 4096
SQRT_MULTIPLIER = 4
DELTA = .01


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    paths = [Path(__file__).resolve().with_name(name) for name in
             ("run_conditional.py", "conditional_fisher.py", "predictive.py",
              "certified_likelihood.py", "design_policy.py", "probe_fixed.py")]
    paths.append(ROOT / "experiments/circuit_validation/direct_tests.py")
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}


def merge_row(rows, arm, n, k):
    key = arm_key(arm)
    for row in rows:
        if arm_key(row) == key:
            row["trials"] += int(n)
            row["successes"] += int(k)
            return
    rows.append({name: int(arm[name]) for name in
                 ("cos_numerator", "sin_numerator", "denominator")})
    rows[-1].update(trials=int(n), successes=int(k))


def positive_arm(policy, rows):
    if policy == "psr":
        return rational_circle(1, 1)
    theta = fit_rows(rows)["theta"]
    a, b, c = theta
    if policy == "closed":
        d, linear = a*(1-a)-b*b, (1-2*a)*b
        z = -linear/(d+math.sqrt(max(0., (d-linear)*(d+linear))))
        tangent = math.sqrt((1-z)/(1+z))
        rational = Fraction(tangent).limit_denominator(65536)
        return rational_circle(rational.numerator, rational.denominator)
    if policy != "grid":
        raise ValueError("unknown policy")
    best, best_value = None, -math.inf
    for arm in GRID:
        if arm["sin_numerator"] <= 0:
            continue
        co, si = arm["cos_numerator"]/arm["denominator"], arm["sin_numerator"]/arm["denominator"]
        q, u = a+b*co, c*si
        pm, pp = q-u, q+u
        value = .5*(xlogy(pm, pm/q)+xlogy(1-pm, (1-pm)/(1-q))
                      +xlogy(pp, pp/q)+xlogy(1-pp, (1-pp)/(1-q)))
        if value > best_value:
            best, best_value = arm, value
    return {key: best[key] for key in ("cos_numerator", "sin_numerator", "denominator")}


def pair_grid_log(counts):
    b, c = int(counts[1]), int(counts[2])
    experts = cf.protocol_config()["experts"]
    logs = [b*math.log(2*e["numerator"]/(e["numerator"]+e["denominator"]))
            +c*math.log(2*e["denominator"]/(e["numerator"]+e["denominator"]))
            for e in experts]
    return float(logsumexp(logs)-math.log(len(experts)))


def pair_grid_bounds(counts):
    b, c = int(counts[1]), int(counts[2])
    logs = []
    for expert in cf.protocol_config()["experts"]:
        p, q = expert["numerator"], expert["denominator"]
        logs.append(b*(_Bounds(2*p)/(p+q)).log()+c*(_Bounds(2*q)/(p+q)).log())
    offset = max(x.hi for x in logs)
    summed = sum(((x-_Bounds(offset)).exp() for x in logs), _Bounds(0))
    return _Bounds(offset)+(summed/len(logs)).log()


def beta_bounds(counts):
    b, c = int(counts[1]), int(counts[2])
    return _log_marginal(b, b+c, .5)+(b+c)*_Bounds(2).log()


def bounds_from_certificate(cert):
    return _Bounds(Decimal(str(cert["evidence_bounds"][0])),
                   Decimal(str(cert["evidence_bounds"][1])))


def evidence_certificate(engine, counts, state, cached_fisher=None):
    if engine == "beta":
        return certificate_from_pairs(tuple(map(int, counts)), delta=DELTA)
    if engine == "pair_grid":
        evidence = pair_grid_bounds(counts)
        detail = None
    else:
        detail = cached_fisher or cf.certificate_from_blocks(state["blocks"], delta=DELTA)
        evidence = bounds_from_certificate(detail)
        if engine == "hedge":
            beta = beta_bounds(counts)
            offset = max(beta.hi, evidence.hi)
            evidence = _Bounds(offset)+(((beta-_Bounds(offset)).exp()
                                         +(evidence-_Bounds(offset)).exp())/2).log()
        elif engine != "fisher":
            raise ValueError("unknown engine")
    threshold = -_Bounds(Decimal(str(DELTA))).log()
    score = int(counts[1]-counts[2])
    decision = ("reject" if score > 0 else "accept") if (
        score != 0 and evidence.lo >= threshold.hi) else "abstain"
    return {"decision": decision, "method": engine, "shots": int(2*sum(counts)),
            "score": score, "evidence_bounds": [str(evidence.lo), str(evidence.hi)],
            "threshold_bounds": [str(threshold.lo), str(threshold.hi)],
            "fisher_retained_terms": (sum(sum(x["retained_terms_per_positive_odds"])
                                         for x in detail["diagnostics"]) if detail else None)}


def run_trial(case, policy, rep, seed, cap, engines=ENGINES):
    rng = np.random.RandomState(seed)
    rows, history, stopped = [], [], {}
    state, counts = cf.empty_state(), np.zeros(4, dtype=np.int64)
    explored, last = 0, 0
    planning_time, screening_time, certification_time = 0., 0., 0.
    for look, budget in enumerate(schedule(cap), 1):
        available = (budget-last)//2
        if policy == "psr" or last < EXPLORE_AFTER:
            phases = [("psr", available)]
        else:
            target = int(math.ceil(SQRT_MULTIPLIER*math.sqrt(budget/2)))
            deficit = min(available, max(0, target-explored))
            phases = [("explore_a", (deficit+1)//2),
                      ("explore_b", deficit//2), ("exploit", available-deficit)]
        for phase, number in phases:
            if not number:
                continue
            begin = time.perf_counter()
            if phase == "psr":
                arm = positive_arm("psr", rows)
            elif phase == "explore_a":
                arm = rational_circle(4, 7)
            elif phase == "explore_b":
                arm = rational_circle(7, 4)
            else:
                arm = positive_arm(policy, rows)
            planning_time += time.perf_counter()-begin
            co, si = arm["cos_numerator"]/arm["denominator"], arm["sin_numerator"]/arm["denominator"]
            pm, pp = case["A"]+case["B"]*co-case["C"]*si, case["A"]+case["B"]*co+case["C"]*si
            old = int(rng.binomial(number, pm))
            both = int(rng.binomial(old, pp))
            worse = int(rng.binomial(number-old, pp))
            fresh = [number-old-worse, worse, old-both, both]
            plus, minus = worse+both, old
            history.append({"look": look, "budget": budget, "phase": phase,
                            "arm": arm, "m": number, "pair_counts": fresh,
                            "minus_successes": minus, "plus_successes": plus})
            counts += fresh
            begin = time.perf_counter()
            cf.update_block(state, number, minus, plus)
            screening_time += time.perf_counter()-begin
            merge_row(rows, arm, number, plus)
            negative = dict(arm)
            negative["sin_numerator"] = -negative["sin_numerator"]
            merge_row(rows, negative, number, minus)
            if phase.startswith("explore"):
                explored += number
        assert int(sum(counts)*2) == state["shots"] == budget
        b, c = counts[1:3]
        beta_log = float(gammaln(b+.5)+gammaln(c+.5)-gammaln(b+c+1)
                         -2*gammaln(.5)+(b+c)*math.log(2))
        fisher_log = cf.screen(state, DELTA)["log_evidence"]
        screens = {"beta": beta_log, "pair_grid": pair_grid_log(counts),
                   "fisher": fisher_log, "hedge": float(np.logaddexp(beta_log, fisher_log)-math.log(2))}
        cached = None
        for engine in engines:
            if engine in stopped:
                continue
            if screens[engine] >= math.log(1/DELTA)-1e-6 or budget == cap:
                begin = time.perf_counter()
                if engine in ("fisher", "hedge") and cached is None:
                    cached = cf.certificate_from_blocks(state["blocks"], delta=DELTA)
                cert = evidence_certificate(engine, counts, state, cached)
                certification_time += time.perf_counter()-begin
                if cert["decision"] != "abstain" or budget == cap:
                    stopped[engine] = {"case_id": case["case_id"], "rep": rep,
                                       "seed": seed, "policy": policy, "engine": engine,
                                       "method": policy+"_"+engine, "shots": budget,
                                       "decision": cert["decision"], "certificate": cert,
                                       "counts": list(map(int, counts)),
                                       "history": list(history), "exploration_pairs": explored}
        last = budget
        if len(stopped) == len(engines):
            break
    return list(stopped.values()), {"planning_seconds": planning_time,
                                    "screening_seconds": screening_time,
                                    "certification_seconds": certification_time}


def main(args):
    started = time.perf_counter()
    control = json.loads((ROOT / "results/structured_validation/run_control.json").read_text())
    public = ROOT / "results/learned_validation/development_v1/spec.json"
    all_cases = json.loads(public.read_text(encoding="utf-8"))["cases"]
    if args.case_ids:
        ids = {int(x) for x in args.case_ids.split(",")}
        cases = [case for case in all_cases if case["case_id"] in ids]
        assert len(cases) == len(ids)
    else:
        cases = [case for case in all_cases if case["kind"] == "main" and case["C"] > 0]
    policies, engines = args.policies.split(","), args.engines.split(",")
    assert set(policies) <= set(POLICIES) and set(engines) <= set(ENGINES)
    directory = ROOT / "results/structured_validation" / args.output
    directory.mkdir(parents=True, exist_ok=True)
    assert not (directory / "spec.json").exists(), "Preserve an existing experiment; use a new version."
    sources = source_hashes()
    spec = {"status": "development only", "cases": cases, "repetitions": args.repetitions,
            "policies": policies, "engines": engines, "cap": args.cap,
            "seed_base": args.seed_base, "seed_rule": "base+1009*case_id+1000003*rep",
            "delta": DELTA, "checkpoints": schedule(args.cap),
            "explore_after_previous_shots": EXPLORE_AFTER, "sqrt_multiplier": SQRT_MULTIPLIER,
            "exploration_coordinates": [rational_circle(4, 7), rational_circle(7, 4)],
            "grid": GRID, "conditional_protocol": cf.protocol_config(),
            "sources": sources, "run_control": control,
            "counting": "All shots including errors, exploration and cap abstentions; each engine is a separate protocol trajectory.",
            "coupling": "Independent joint replicate streams; policies use same seed, engines share a policy trajectory."}
    (directory / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    all_cells, timings, records_count = [], [], 0
    with (directory / "trials.jsonl").open("w", encoding="utf-8") as stream:
        for case in cases:
            for policy in policies:
                records = []
                for rep in range(args.repetitions):
                    # Deadline is checked between independent repetitions.
                    if time.time() >= 1793731034:  # 2026-11-03 18:37:14 UTC
                        raise RuntimeError("Experiment cutoff reached; partial ledger retained.")
                    seed = args.seed_base+1009*case["case_id"]+1000003*rep
                    local, timing = run_trial(case, policy, rep, seed, args.cap, engines)
                    records.extend(local)
                    timings.append(timing)
                    for row in local:
                        stream.write(json.dumps(row, separators=(",", ":"))+"\n")
                        records_count += 1
                    stream.flush()
                expected = "accept" if case["C"] < 0 else "reject" if case["C"] > 0 else "abstain"
                cells = []
                for engine in engines:
                    local = [r for r in records if r["engine"] == engine]
                    cells.append({"case_id": case["case_id"], "kind": case["kind"],
                                  "method": policy+"_"+engine, "repetitions": len(local),
                                  "mean_shots": float(np.mean([r["shots"] for r in local])),
                                  "wrong": sum(r["decision"] not in ("abstain", expected) for r in local),
                                  "abstained": sum(r["decision"] == "abstain" for r in local)})
                all_cells.extend(cells)
                print(json.dumps({"case": case["case_id"], "policy": policy, "cells": cells,
                                  "elapsed_seconds": time.perf_counter()-started}), flush=True)
    assert source_hashes() == sources, "Run sources changed during execution."
    result = {"spec": spec, "cells": all_cells, "new_trajectories": records_count,
              "timings": {key: sum(t[key] for t in timings) for key in timings[0]},
              "elapsed_seconds": time.perf_counter()-started}
    (directory / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("spec", "cells")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case-ids", default="")
    parser.add_argument("--policies", default="psr,closed,grid")
    parser.add_argument("--engines", default="beta,pair_grid,fisher,hedge")
    parser.add_argument("--repetitions", type=int, default=16)
    parser.add_argument("--cap", type=int, default=2**22)
    parser.add_argument("--seed-base", type=int, default=510000000)
    main(parser.parse_args())
