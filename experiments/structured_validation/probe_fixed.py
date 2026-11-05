"""Transparent PSR paired-Beta versus full-count universal-inference probe."""
import argparse
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import betaln, gammaln, xlogy

from predictive import _Bounds, _log_marginal, ROOT
from direct_tests import certificate_from_pairs, _as_float_bounds


def schedule(cap):
    budgets, n = [], 32
    while n < cap:
        budgets.append(n)
        n = 2*int(math.ceil(n*1.125/2))
    return sorted(set(budgets+[cap]))


def full_ui_certificate(counts, delta=.01):
    a, b, c, d = map(int, counts)
    pairs = a+b+c+d
    plus, minus = b+d, c+d
    total, successes = 2*pairs, b+c+2*d
    numerator = _log_marginal(plus, pairs, .5)+_log_marginal(minus, pairs, .5)
    denominator = _Bounds(0)
    if successes:
        denominator += successes*(_Bounds(successes)/total).log()
    if total-successes:
        denominator += (total-successes)*(_Bounds(total-successes)/total).log()
    evidence = numerator-denominator
    threshold = -_Bounds(Decimal(str(delta))).log()
    decision = ("reject" if plus > minus else "accept") if (
        plus != minus and evidence.lo >= threshold.hi) else "abstain"
    return {"decision": decision, "shots": total, "counts": list(map(int, counts)),
            "evidence_bounds": _as_float_bounds(evidence),
            "threshold_bounds": _as_float_bounds(threshold),
            "shared_numerator_total_risk": delta}


def run_case(case, repetitions, seed_base, cap):
    costs = {name: [] for name in ("psr_beta", "psr_full_ui")}
    records = []
    for rep in range(repetitions):
        seed = seed_base+case["case_id"]*1009+rep*1000003
        rng = np.random.RandomState(seed)
        counts = np.zeros(4, dtype=np.int64)
        stopped, history, last = {}, [], 0
        pm, pp = case["A"]-case["C"], case["A"]+case["C"]
        for budget in schedule(cap):
            n = (budget-last)//2
            old = int(rng.binomial(n, pm))
            both = int(rng.binomial(old, pp))
            worse = int(rng.binomial(n-old, pp))
            fresh = [n-old-worse, worse, old-both, both]
            counts += fresh
            history.append([budget]+fresh)
            a, b, c, d = counts
            pairs = counts.sum()
            beta_screen = (gammaln(b+.5)+gammaln(c+.5)-gammaln(b+c+1)
                           -2*gammaln(.5)+(b+c)*math.log(2))
            kp, km = b+d, c+d
            numerator = (betaln(kp+.5, pairs-kp+.5)+betaln(km+.5, pairs-km+.5)
                         -2*betaln(.5, .5))
            q = (kp+km)/(2*pairs)
            null = xlogy(kp+km, q)+xlogy(2*pairs-kp-km, 1-q)
            for method, screen in (("psr_beta", beta_screen), ("psr_full_ui", numerator-null)):
                if method in stopped:
                    continue
                if screen >= math.log(100)-1e-7 or budget == cap:
                    cert = (certificate_from_pairs(counts) if method == "psr_beta"
                            else full_ui_certificate(counts))
                    if cert["decision"] != "abstain" or budget == cap:
                        stopped[method] = cert
            last = budget
            if len(stopped) == 2:
                break
        for method, cert in stopped.items():
            costs[method].append(cert["shots"])
            records.append({"case_id": case["case_id"], "rep": rep, "seed": seed,
                            "method": method, "certificate": cert,
                            "history": [h for h in history if h[0] <= cert["shots"]]})
    cells = []
    expected = "accept" if case["C"] < 0 else "reject" if case["C"] > 0 else "abstain"
    for method in costs:
        local = [r["certificate"] for r in records if r["method"] == method]
        cells.append({"case_id": case["case_id"], "method": method, "kind": case["kind"],
                      "repetitions": repetitions, "mean_shots": float(np.mean(costs[method])),
                      "wrong": sum(r["decision"] not in ("abstain", expected) for r in local),
                      "abstained": sum(r["decision"] == "abstain" for r in local)})
    return cells, records


def main(repetitions, cap):
    started = time.perf_counter()
    public = ROOT / "results/learned_validation/development_v1/spec.json"
    all_cases = json.loads(public.read_text(encoding="utf-8"))["cases"]
    # All old public cases are development data, with wholly new replicate seeds.
    cases = [case for case in all_cases if case["kind"] == "main" and case["C"] > 0]
    directory = ROOT / "results/structured_validation/fixed_psr_probe"
    directory.mkdir(parents=True, exist_ok=True)
    assert not (directory / "summary.json").exists(), "Preserve completed probes; use a new version."
    seed_base = 410000000
    spec = {"status": "development only", "cases": cases, "repetitions": repetitions,
            "cap": cap, "delta": .01, "seed_base": seed_base,
            "seed_rule": "base+1009*case_id+1000003*rep", "paired_stream": True,
            "numerator": "Independent Jeffreys predictive densities for the two PSR arms",
            "threshold": "Shared numerator, both directions use 1/delta",
            "sources": {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (Path(__file__).resolve(), Path(__file__).resolve().with_name("predictive.py"),
                                  ROOT / "experiments/circuit_validation/direct_tests.py")}}
    (directory / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    cells = []
    with (directory / "trials.jsonl").open("w", encoding="utf-8") as stream:
        for case in cases:
            local, records = run_case(case, repetitions, seed_base, cap)
            cells.extend(local)
            for row in records:
                stream.write(json.dumps(row, separators=(",", ":"))+"\n")
            stream.flush()
            print(json.dumps({"case": case["case_id"], "ratio": local[1]["mean_shots"]/local[0]["mean_shots"]}), flush=True)
    ratio = math.exp(np.mean([math.log(cells[i+1]["mean_shots"]/cells[i]["mean_shots"])
                              for i in range(0, len(cells), 2)]))
    result = {"spec": spec, "cells": cells, "full_ui_over_beta": ratio,
              "new_trajectories": len(cases)*repetitions*2,
              "elapsed_seconds": time.perf_counter()-started}
    (directory / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ("spec", "cells")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=32)
    parser.add_argument("--cap", type=int, default=2**22)
    args = parser.parse_args()
    main(args.repetitions, args.cap)
