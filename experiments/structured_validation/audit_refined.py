"""Audit frozen refined runs: paid predictions, weighted prior, safe and RB.

All final certificates are recomputed. For each case and policy the first
and last repetition are replayed completely, without outcome selection.
The immutable conditional audit supplies bookkeeping helpers only.
"""
import argparse
from collections import Counter, defaultdict
from decimal import Decimal
from fractions import Fraction
from itertools import groupby
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.special import gammaln, logsumexp

import audit_conditional as base
import run_refined as runner
import weighted_fisher as fisher
import safe_grid as safe
import rb_grid as rb
from predictive import ROOT, _Bounds, _log_marginal
from direct_tests import certificate_from_pairs

require = base.require


def weighted_pair_log(counts, experts):
    b, c = int(counts[1]), int(counts[2])
    return float(logsumexp([b*math.log(2*e["numerator"]/(e["numerator"]+e["denominator"]))
                           +c*math.log(2*e["denominator"]/(e["numerator"]+e["denominator"]))
                           +math.log(e["weight"][0]/e["weight"][1]) for e in experts]))


def weighted_pair_bounds(counts, experts):
    b, c = int(counts[1]), int(counts[2])
    logs = []
    for e in experts:
        p, q = e["numerator"], e["denominator"]
        logs.append(b*(_Bounds(2*p)/(p+q)).log()+c*(_Bounds(2*q)/(p+q)).log()
                    +(_Bounds(e["weight"][0])/e["weight"][1]).log())
    offset = max(value.hi for value in logs)
    return _Bounds(offset)+sum(((value-_Bounds(offset)).exp() for value in logs), _Bounds(0)).log()


def expert_values(config):
    return [(Fraction(e["numerator"], e["denominator"]), Fraction(*e["weight"])) for e in config["experts"]]


def prior_checks(spec):
    config = spec["conditional_protocol"]
    require(config == fisher.protocol_config(), "frozen weighted Fisher prior changed")
    values = expert_values(config)
    require(len(values) == len(set(odds for odds, weight in values)), "duplicate odds")
    require(all(odds > 0 and weight > 0 for odds, weight in values), "nonpositive odds or weight")
    require(sum((weight for odds, weight in values), Fraction(0)) == 1, "weights do not sum exactly to one")
    require(all((1/odds, weight) in values for odds, weight in values), "reciprocal weights differ")
    require(expert_values(safe.protocol_config(config)) == values, "safe uses a different prior")
    require(expert_values(rb.protocol_config(config)) == values, "RB uses a different prior")
    error = 0.
    for counts in ([1, 2, 1, 1], [20, 80, 25, 11], [0, 900, 1100, 0]):
        score, discordant = counts[1]-counts[2], counts[1]+counts[2]
        logs = []
        for odds, weight in values:
            eta = math.log(float(odds))
            cgf = float(np.logaddexp(eta/2, -eta/2)-math.log(2))
            logs.append(score*eta/2-discordant*cgf+math.log(float(weight)))
        error = max(error, abs(float(logsumexp(logs))-weighted_pair_log(counts, config["experts"])))
    require(error < 1e-9, "m=1 weighted conditional/pair identity differs")
    return {"expert_count": len(values), "weights_exactly_sum_to_one": True,
            "reciprocal_odds_and_weights": True, "fisher_safe_rb_pair_share_exact_prior": True,
            "m1_log_identity_max_error": error}


def safe_prediction_from_past(rows, arm, config):
    # Reconstruct the proposal before incorporating this block's outcomes.
    a, b, unused_c = base.paid_fit(rows)
    cosine = arm["cos_numerator"]/arm["denominator"]
    center = float(a+b*cosine)
    denominator = config["endpoint_denominator"]
    q = min(1-1/denominator, max(1/denominator, center))
    poles = sorted({max(odds, 1/odds) for odds, weight in expert_values(config) if odds != 1})
    numerators = []
    for odds in poles:
        omega = float(odds)
        difference = omega-1
        discriminant = math.sqrt(max(0., (omega+1)**2-4*difference*difference*q*(1-q)))
        half_gap = 2*difference*q*(1-q)/(omega+1+discriminant)
        numerators.append(min(denominator-1, max(1, int(round((q-half_gap)*denominator)))))
    return {"center_requested": center, "center_clipped": q,
            "odds_poles": [[odds.numerator, odds.denominator] for odds in poles],
            "lower_numerators": numerators, "lower_denominator": denominator}


def audit_certificate(engine, counts, fstate, sstate, rstate, spec, cached=None):
    delta, config = spec["delta"], spec["conditional_protocol"]
    if engine == "safe":
        return safe.certificate_from_blocks(sstate["blocks"], delta=delta, config=sstate["config"])
    if engine == "rb":
        return rb.certificate_from_blocks(rstate["blocks"], delta=delta, config=config)
    if engine == "beta":
        return certificate_from_pairs(tuple(map(int, counts)), delta=delta)
    detail = None
    if engine == "pair_grid":
        evidence = weighted_pair_bounds(counts, config["experts"])
    else:
        detail = cached or fisher.certificate_from_blocks(fstate["blocks"], delta=delta, experts=config["experts"])
        evidence = _Bounds(Decimal(detail["evidence_bounds"][0]), Decimal(detail["evidence_bounds"][1]))
        if engine == "hedge":
            b, c = int(counts[1]), int(counts[2])
            beta = _log_marginal(b, b+c, .5)+(b+c)*_Bounds(2).log()
            evidence = base.mixture_bounds([beta, evidence])
        else:
            require(engine == "fisher", "unknown engine")
    score = int(counts[1]-counts[2])
    threshold = -_Bounds(Decimal(str(delta))).log()
    decision = ("reject" if score > 0 else "accept") if (
        score and evidence.lo >= threshold.hi) else "abstain"
    return {"decision": decision, "method": engine, "shots": int(2*sum(counts)),
            "score": score, "evidence_bounds": [str(evidence.lo), str(evidence.hi)],
            "threshold_bounds": [str(threshold.lo), str(threshold.hi)],
            "fisher_retained_terms": (sum(sum(x["retained_terms_per_positive_odds"])
                                         for x in detail["diagnostics"]) if detail else None)}


def audit_trial(key, records, spec):
    case_id, policy, rep = key
    config, delta = spec["conditional_protocol"], spec["delta"]
    require(len(records) == len(spec["engines"]), "incomplete trial engines")
    by_engine = {record["engine"]: record for record in records}
    require(set(by_engine) == set(spec["engines"]), "duplicate or missing engine")
    seed = spec["seed_base"]+1009*case_id+1000003*rep
    longest = max(records, key=lambda record: record["shots"])
    history = longest["history"]
    require(history and history[-1]["budget"] == longest["shots"], "invalid longest history")
    for record in records:
        require((record["case_id"], record["policy"], record["rep"]) == key and record["seed"] == seed,
                "identity or seed mismatch")
        require(record["method"] == policy+"_"+record["engine"], "method mismatch")
        require(record["history"] == [block for block in history if block["budget"] <= record["shots"]],
                "engine histories did not share exactly the same observations")
    fstate = fisher.empty_state(config["experts"])
    sstate, rstate = safe.empty_state(safe.protocol_config(config)), rb.empty_state(config)
    rows, counts = [], np.zeros(4, dtype=np.int64)
    previous = explored = look_count = cert_calls = screen_checks = 0
    first_screen, stopped, stop_looks, max_errors, initial_prefix = {}, set(), {}, {}, []
    threshold = math.log(1/delta)-1e-6
    for look, grouped in groupby(history, key=lambda block: block["look"]):
        blocks = list(grouped)
        look_count += 1
        require(look == look_count and look <= len(spec["checkpoints"]), "nonconsecutive checkpoints")
        budget = spec["checkpoints"][look-1]
        expected_phases = base.expected_phases(policy, previous, budget, explored, spec)
        require([(block["phase"], block["m"]) for block in blocks] == expected_phases, "phase cost mismatch")
        for block in blocks:
            require(block["budget"] == budget, "incorrect checkpoint budget")
            arm = base.predictable_arm(block["phase"], policy, rows, spec)
            require(block["arm"] == arm, "measurement action differs from past-data reconstruction")
            co, si, denominator = base.arm_key(arm)
            require(all(type(x) is int for x in (co, si, denominator)) and denominator > 0
                    and co*co+si*si == denominator*denominator and si > 0, "invalid rational query")
            prediction = safe_prediction_from_past(rows, arm, sstate["config"])
            require(block["safe_prediction"] == prediction, "safe nuisance prediction is not the recorded past-data proposal")
            n, fresh = block["m"], block["pair_counts"]
            require(type(n) is int and n > 0 and len(fresh) == 4 and all(type(x) is int and x >= 0 for x in fresh)
                    and sum(fresh) == n, "invalid joint counts")
            minus, plus = fresh[2]+fresh[3], fresh[1]+fresh[3]
            require(block["minus_successes"] == minus and block["plus_successes"] == plus, "invalid margins")
            counts += fresh
            fisher.update_block(fstate, n, minus, plus)
            safe.update_block(sstate, n, minus, plus, prediction)
            rb.update_block(rstate, n, minus, plus)
            require(rstate["blocks"][-1] == {"m": n, "minus_successes": minus, "plus_successes": plus},
                    "RB state uses more than marginal block observations")
            base.add_paid_row(rows, arm, n, plus)
            base.add_paid_row(rows, dict(arm, sin_numerator=-si), n, minus)
            if block["phase"].startswith("explore"):
                explored += n
            if previous < spec["explore_after_previous_shots"]:
                initial_prefix.append(block)
        require(int(counts.sum())*2 == budget == fstate["shots"] == sstate["shots"] == rstate["shots"]
                == sum(row["trials"] for row in rows), "paid shot mismatch")
        require(2*fstate["score"] == sstate["score"] == rstate["score"] == int(counts[1]-counts[2]),
                "state score mismatch")
        b, c = counts[1:3]
        beta = float(gammaln(b+.5)+gammaln(c+.5)-gammaln(b+c+1)-2*gammaln(.5)+(b+c)*math.log(2))
        fl = fisher.screen(fstate, delta)["log_evidence"]
        screens = {"beta": beta, "pair_grid": weighted_pair_log(counts, config["experts"]), "fisher": fl,
                   "hedge": float(np.logaddexp(beta, fl)-math.log(2)),
                   "safe": safe.screen(sstate, delta)["log_evidence"], "rb": rb.screen(rstate, delta)["log_evidence"]}
        cached = None
        for engine in spec["engines"]:
            if engine in stopped:
                continue
            screen_checks += 1
            record = by_engine[engine]
            require(budget <= record["shots"], "recorded stop had no valid first crossing")
            if screens[engine] < threshold and budget != spec["cap"]:
                require(budget != record["shots"], "stopped without an eligible floating screen")
                continue
            if screens[engine] >= threshold and engine not in first_screen:
                first_screen[engine] = budget
            if engine in ("fisher", "hedge") and cached is None:
                cached = fisher.certificate_from_blocks(fstate["blocks"], delta=delta, experts=config["experts"])
            cert = audit_certificate(engine, counts, fstate, sstate, rstate, spec, cached)
            cert_calls += 1
            lo, hi = map(float, cert["evidence_bounds"])
            max_errors[engine] = max(max_errors.get(engine, 0.), lo-screens[engine], screens[engine]-hi, 0.)
            if cert["decision"] != "abstain" or budget == spec["cap"]:
                require(budget == record["shots"], "missed an earlier certified crossing")
                require(record["certificate"] == cert, "final Decimal certificate differs")
                require(record["decision"] == cert["decision"] and record["counts"] == counts.tolist(),
                        "final decision or counts mismatch")
                require(record["exploration_pairs"] == explored and cert["shots"] == budget, "final cost mismatch")
                stopped.add(engine)
                stop_looks[engine] = look
            else:
                require(budget < record["shots"], "stopped on an abstention before cap")
        previous = budget
    require(stopped == set(spec["engines"]), "missing final engine decision")
    return {"history_blocks": len(history), "past_safe_predictions": len(history), "rb_marginal_blocks": len(history),
            "screen_checks": screen_checks, "certificates_recomputed": cert_calls, "final_certificates": len(records),
            "maximum_screen_errors": max_errors, "first_float_crossing": first_screen,
            "stop_looks": stop_looks, "initial_prefix": initial_prefix}


def audit(directory, output):
    started = time.perf_counter()
    directory = Path(directory).resolve()
    require((directory/"summary.json").exists(), "run incomplete: summary.json required")
    require(not output.exists(), "preserve existing audits; select another output path")
    inputs = {name: base.sha256(directory/name) for name in ("spec.json", "summary.json", "trials.jsonl")}
    audit_sources = {name: base.sha256(Path(__file__).with_name(name)) for name in ("audit_refined.py", "audit_conditional.py")}
    spec = json.loads((directory/"spec.json").read_text(encoding="utf-8"))
    summary = json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    require(summary["spec"] == spec, "summary/spec mismatch")
    require(spec["sources"] == runner.source_hashes(), "run source changed since freezing")
    require(spec["checkpoints"] == base.expected_checkpoints(spec["cap"]), "checkpoint schedule differs")
    require(spec["seed_rule"] == "base+1009*case_id+1000003*rep", "unsupported seed rule")
    require(spec["delta"] == runner.DELTA and spec["explore_after_previous_shots"] == runner.EXPLORE_AFTER
            and spec["sqrt_multiplier"] == runner.SQRT_MULTIPLIER, "runtime constants differ from specification")
    frozen = base.frozen_evidence_checks()
    priors = prior_checks(spec)
    cases = {case["case_id"]: case for case in spec["cases"]}
    require(len(cases) == len(spec["cases"]), "duplicate case IDs")
    require(len(set(spec["policies"])) == len(spec["policies"]) and set(spec["policies"]) <= set(runner.POLICIES),
            "invalid policies")
    require(len(set(spec["engines"])) == len(spec["engines"]) and set(spec["engines"]) <= set(runner.ENGINES),
            "invalid engines")
    repetitions = spec["repetitions"]
    require(type(repetitions) is int and repetitions > 0, "invalid repetitions")
    selected = sorted(set((0, repetitions-1)))
    expected = {(case_id, policy, rep) for case_id in cases for policy in spec["policies"] for rep in range(repetitions)}
    seen, prefixes, cells = set(), {}, defaultdict(list)
    totals, replay_hashes, stop_records, maximum_errors = Counter(), [], [], {}
    for key, records in base.records_by_trial(directory/"trials.jsonl"):
        require(key in expected and key not in seen, "unexpected or duplicated trial")
        seen.add(key)
        case_id, policy, rep = key
        checked = audit_trial(key, records, spec)
        for name in ("history_blocks", "past_safe_predictions", "rb_marginal_blocks", "screen_checks",
                     "certificates_recomputed", "final_certificates"):
            totals[name] += checked[name]
        for engine, error in checked["maximum_screen_errors"].items():
            maximum_errors[engine] = max(maximum_errors.get(engine, 0.), error)
        stop_records.append({"case_id": case_id, "policy": policy, "rep": rep,
                             "first_float_crossing": checked["first_float_crossing"], "stop_looks": checked["stop_looks"]})
        prefix_key, prefix = (case_id, rep), checked["initial_prefix"]
        if prefix_key in prefixes:
            old = prefixes[prefix_key]
            common = min(len(old), len(prefix))
            require(old[:common] == prefix[:common], "initial same-seed policy coupling differs")
            totals["policy_prefix_comparisons"] += 1
            if len(prefix) > len(old):
                prefixes[prefix_key] = prefix
        else:
            prefixes[prefix_key] = prefix
        for record in records:
            cells[(case_id, record["method"])].append({"shots": record["shots"], "decision": record["decision"]})
        if rep in selected:
            seed = spec["seed_base"]+1009*case_id+1000003*rep
            replayed, unused_timing = runner.run_trial(cases[case_id], policy, rep, seed, spec["cap"], spec["engines"])
            actual, replay = sorted(records, key=lambda r: r["engine"]), sorted(replayed, key=lambda r: r["engine"])
            require(actual == replay, "full seeded replay differs")
            replay_hashes.append({"case_id": case_id, "policy": policy, "rep": rep, "seed": seed,
                                  "records_sha256": base.stable_json_hash(actual)})
            totals["replayed_trials"] += 1
            totals["replayed_records"] += len(records)
        if rep == repetitions-1:
            print(json.dumps({"audited_case": case_id, "policy": policy, "trials": len(seen),
                              "records": totals["final_certificates"], "elapsed_seconds": time.perf_counter()-started}), flush=True)
    require(seen == expected, "ledger lacks required trials")
    require(summary["new_trajectories"] == totals["final_certificates"], "summary trajectory count differs")
    reported = {(cell["case_id"], cell["method"]): cell for cell in summary["cells"]}
    require(len(reported) == len(summary["cells"]) and set(reported) == set(cells), "summary cell identities differ")
    outcomes = Counter()
    for key, records in cells.items():
        case_id, method = key
        case = cases[case_id]
        direction = "accept" if case["C"] < 0 else "reject" if case["C"] > 0 else "abstain"
        cell = {"case_id": case_id, "kind": case["kind"], "method": method, "repetitions": len(records),
                "mean_shots": float(np.mean([r["shots"] for r in records])),
                "wrong": sum(r["decision"] not in ("abstain", direction) for r in records),
                "abstained": sum(r["decision"] == "abstain" for r in records)}
        require(reported[key] == cell, "summary costs or decisions differ")
        outcomes["wrong"] += cell["wrong"]
        outcomes["abstained"] += cell["abstained"]
        outcomes["paid_shots_across_protocol_records"] += sum(r["shots"] for r in records)
    require(inputs == {name: base.sha256(directory/name) for name in inputs}, "audit inputs changed")
    require(audit_sources == {name: base.sha256(Path(__file__).with_name(name)) for name in audit_sources}, "audit source changed")
    require(spec["sources"] == runner.source_hashes(), "run sources changed during audit")
    require(base.frozen_evidence_checks() == frozen, "frozen old evidence changed")
    result = {"status": "passed", "directory": str(directory), "run_input_sha256": inputs,
              "audit_sources_sha256": audit_sources, "run_sources_match_spec": True,
              "frozen_evidence_before_and_after": frozen, "spec_summary_and_all_cells_match": True,
              "trial_groups": len(seen), "replay_rule": "Reps 0 and R-1 for every case and policy, deduplicated, no outcome selection",
              "replayed_rep_indices": selected, "checks": dict(totals), "outcomes": dict(outcomes),
              "matched_prior": priors, "maximum_screen_outside_certificate_by_engine": maximum_errors,
              "decision_rule": "First eligible floating screen with outward Decimal crossing; all cap abstentions retained",
              "limits": "Complete protocol and certificate audit with mechanical seed replays; no power/SOTA conclusion",
              "replays": replay_hashes, "stopping_checks": stop_records, "elapsed_seconds": time.perf_counter()-started}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ("replays", "stopping_checks")}, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    audit(args.directory, args.output or args.directory/"audit.json")
