"""Exploratory finite-update validation using an EXISTING parameter-shift rule.

This is a new, equal-duration model, not the frozen duration-1/2 experiment.
For a single Pauli rotation surrounded by parameter-independent channels,
f(x)=A+B*cos(x)+C*sin(x).  At old/new m-h/2,m+h/2,
Delta=sin(h/2)*(f(m+pi/2)-f(m-pi/2)).

The identity and its variance benefit are standard parameter-shift results.
Both protocols use the SAME audited conditional Beta-mixture sign test.
Only the selected experiments differ. A model-aware PSR baseline is therefore
identical to shifted validation; beating endpoint A/B is NOT beating SOTA.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "circuit_validation"))
from direct_tests import certificate_from_pairs


def loss_probability(angle, phase, visibility, readout):
    """Rx error with fixed phase, dephasing visibility and symmetric readout.

    Parameters are simulator truth. The stopping rule only receives counts.
    All four angles have the same duration and parameter-independent noise.
    """
    return readout + (1.0 - 2.0 * readout) * (
        1.0 - visibility * math.cos(angle - phase)) / 2.0


def cases():
    result = []
    for h in (0.4, 0.2, 0.1):
        for r, kappa in ((0.01, 0.0), (0.01, 0.08), (0.05, 0.0), (0.05, 0.08)):
            for sign in (-1, 1):
                result.append({"h": h, "phase": sign * h / 4.0,
                               "readout": r, "visibility": math.exp(-kappa),
                               "kind": "main"})
    # Null cases are not included in the efficiency score.
    for h in (0.4, 0.2, 0.1):
        result.append({"h": h, "phase": 0.0, "readout": 0.05,
                       "visibility": math.exp(-0.08), "kind": "null"})
    return result


def schedule(public_h):
    """Same public cap for both methods; no hidden per-case parameters.

    This is an exploration schedule, not a pre-registered confirmation.
    The cap scales as h^-4, consistent with endpoint A/B for this family.
    """
    cap = int(math.ceil(1024.0 / public_h ** 4))
    values, n = [], 32
    while n < cap:
        values.append(n)
        n = max(n + 1, int(math.ceil(n * 1.6)))
    values.append(cap)
    return values


def run_cell(case, method, repetitions, seed, delta=0.01):
    h = case["h"]
    shift = h / 2.0 if method == "endpoint_ab" else math.pi / 2.0
    p0, p1 = [loss_probability(x, case["phase"], case["visibility"],
                               case["readout"]) for x in (-shift, shift)]
    # Two independent physical shots per pair, no shared measurement noise.
    joint = np.array([(1-p0)*(1-p1), (1-p0)*p1, p0*(1-p1), p0*p1])
    joint /= joint.sum()
    rng = np.random.RandomState(seed)
    counts = np.zeros((repetitions, 4), dtype=np.int64)
    active = np.ones(repetitions, dtype=bool)
    shots = np.zeros(repetitions, dtype=np.int64)
    decisions = ["abstain"] * repetitions
    certificates = {}
    last = 0
    for n in schedule(h):
        indices = np.flatnonzero(active)
        if not len(indices):
            break
        counts[indices] += rng.multinomial(n-last, joint, size=len(indices))
        shots[indices] = 2*n
        b, c = counts[indices, 1], counts[indices, 2]
        k = b + c
        evidence = (gammaln(b + .5) + gammaln(c + .5) - gammaln(k + 1)
                    - 2*gammaln(.5) + k*math.log(2.0))
        # Fast screening cannot make a decision. Every decision uses the
        # existing outward-rounded Decimal certificate and its raw counts.
        selected = indices[evidence >= math.log(1.0/delta) - 1e-6]
        for index in selected:
            cert = certificate_from_pairs(tuple(int(x) for x in counts[index]), delta)
            if cert["decision"] != "abstain":
                decisions[index] = cert["decision"]
                certificates[int(index)] = cert
                active[index] = False
        last = n
    target_old, target_new = [loss_probability(x, case["phase"],
                                              case["visibility"], case["readout"])
                              for x in (-h/2, h/2)]
    effect = target_new - target_old
    expected = ("accept" if case["phase"] > 0 else "reject"
                if case["phase"] < 0 else "abstain")
    wrong = sum(d not in (expected, "abstain") for d in decisions)
    if expected == "abstain":
        wrong = sum(d != "abstain" for d in decisions)
    trials = [{"rep": i, "counts": [int(x) for x in counts[i]],
               "shots": int(shots[i]), "decision": decisions[i],
               "certificate": certificates.get(i)} for i in range(repetitions)]
    return {"method": method, "case": case, "seed": seed,
            "target_delta": effect, "observed_gap": p1-p0,
            "identity_residual": effect - math.sin(h/2)*(p1-p0)
              if method == "parameter_shift" else None,
            "repetitions": repetitions, "wrong": wrong,
            "decided": sum(d != "abstain" for d in decisions),
            "abstained": sum(d == "abstain" for d in decisions),
            "mean_shots": float(shots.mean()),
            "median_shots": float(np.median(shots)),
            "min_shots": int(shots.min()), "max_shots": int(shots.max()),
            "trials": trials}


def physical_identity_audit(seed=7619, count=1024):
    """Independent Bloch-channel checks including nonunital asymmetric noise.

    Arbitrary mixed input -> Rx -> dephasing -> amplitude damping -> arbitrary
    rank-one binary measurement -> asymmetric classical readout. The channels
    are fixed for all angles in each independent identity check.
    """
    rng = np.random.RandomState(seed)
    residuals = []
    for _ in range(count):
        state = rng.normal(size=3)
        state *= rng.uniform(0, 1) / np.linalg.norm(state)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        dephase, damping = rng.uniform(0, .6, size=2)
        r01, r10 = rng.uniform(0, .2, size=2)
        midpoint, h = rng.uniform(-math.pi, math.pi), rng.uniform(.001, 2.0)

        def probability(angle):
            x, y, z = state
            cy, sy = math.cos(angle), math.sin(angle)
            rotated = np.array([x, cy*y-sy*z, sy*y+cy*z])
            rotated[:2] *= 1-dephase
            rotated[:2] *= math.sqrt(1-damping)
            rotated[2] = (1-damping)*rotated[2] + damping
            measured_one = (1-float(np.dot(axis, rotated)))/2
            return r01 + (1-r01-r10)*measured_one

        effect = probability(midpoint+h/2)-probability(midpoint-h/2)
        contrast = probability(midpoint+math.pi/2)-probability(midpoint-math.pi/2)
        residuals.append(abs(effect-math.sin(h/2)*contrast))
    return {"checks": count, "max_absolute_residual": max(residuals),
            "meaning": "Floating-point physical checks; proof uses linearity of channels."}


def run(repetitions, output):
    start = time.time()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    spec = {"status": "exploratory, not frozen confirmation", "delta": .01,
            "repetitions": repetitions, "cases": cases(),
            "schedules_pairs": {str(h): schedule(h) for h in (.4, .2, .1)},
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "methods": ["endpoint_ab", "parameter_shift"],
            "same_information_baseline": "Standard model-aware PSR is exactly the parameter_shift method."}
    (output / "shift_pilot_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    rows, ratios, audit_count = [], [], 0
    with (output / "shift_pilot_trials.jsonl").open("w", encoding="utf-8") as ledger:
        for cid, case in enumerate(cases()):
            local = []
            for mid, method in enumerate(spec["methods"]):
                cell = run_cell(case, method, repetitions, 900001 + cid*17 + mid)
                trials = cell.pop("trials")
                for trial in trials:
                    audit_count += trial["certificate"] is not None
                    ledger.write(json.dumps({"case_id": cid, "method": method, **trial})+"\n")
                rows.append(cell)
                local.append(cell)
            if case["kind"] == "main":
                ratios.append({"case_id": cid, "h": case["h"],
                               "endpoint_over_shift": local[0]["mean_shots"]/local[1]["mean_shots"]})
            print(json.dumps({"case": cid, "h": case["h"], "kind": case["kind"],
                              "means": [round(c["mean_shots"], 1) for c in local],
                              "wrong": [c["wrong"] for c in local],
                              "abstained": [c["abstained"] for c in local]}), flush=True)
    summary = {"spec": spec, "physical_audit": physical_identity_audit(), "cells": rows,
               "ratios": ratios, "geomean_endpoint_over_shift": math.exp(
                   sum(math.log(x["endpoint_over_shift"]) for x in ratios)/len(ratios)),
               "by_h": {str(h): math.exp(np.mean([math.log(x["endpoint_over_shift"])
                           for x in ratios if x["h"] == h])) for h in (.4, .2, .1)},
               "trials": len(rows)*repetitions,
               "audited_decision_certificates": int(audit_count),
               "elapsed_seconds": time.time()-start,
               "limits": ["Exploratory means; no confirmatory simultaneous performance intervals.",
                          "Common cap includes every abstention at full cost.",
                          "Unknown noise must be angle-independent; equal duration alone is insufficient.",
                          "Target is observed noisy task loss, not ideal gate infidelity.",
                          "This is an existing PSR method, not evidence of outperforming model-aware SOTA."]}
    (output / "shift_pilot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"done": True, "trials": summary["trials"],
                      "ratio": summary["geomean_endpoint_over_shift"],
                      "by_h": summary["by_h"], "elapsed": time.time()-start}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=128)
    parser.add_argument("--output", default=str(ROOT / "results" / "phase_validation"))
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    run(args.repetitions, args.output)
