"""Audit a completed conditional benchmark without changing run sources.

Rebuilds paid counts, predictable actions, checkpoint screens and every
stopping certificate. Replays reps 0 and R-1 for every case and policy.
The audit accepts a result directory and does not inspect outcomes to select
replays. Floating screens are routing checks, never decision certificates.
"""
import argparse
from collections import Counter, defaultdict
from decimal import Decimal
from fractions import Fraction
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.special import gammaln, logsumexp, xlogy

import conditional_fisher as cf
import run_conditional as runner
from predictive import ROOT, _Bounds, _log_marginal
from direct_tests import certificate_from_pairs


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(1024*1024)
            if not chunk:
                return result.hexdigest()
            result.update(chunk)


def stable_json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def frozen_evidence_checks():
    manifest_paths = ("results/streaming_validation/evidence_manifest.json",
                      "results/learned_validation/evidence_manifest.json")
    expected_streaming = "3af4a45b0a3a7aa9d139f0efb0752651da1286635b54e7166f7e529f982343ed"
    require(sha256(ROOT/manifest_paths[0]) == expected_streaming, "frozen 107-file manifest changed")
    checked, result = {}, []
    for relative in manifest_paths:
        path = ROOT/relative
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for name, expected in manifest["files"].items():
            if name not in checked:
                checked[name] = sha256(ROOT/name)
            require(checked[name] == expected, "frozen evidence changed: "+name)
        result.append({"manifest": relative, "sha256": sha256(path),
                       "verified_files": len(manifest["files"])})
    return {"manifests": result, "unique_verified_files": len(checked)}


def expected_checkpoints(cap):
    result, budget = [], 32
    while budget < cap:
        result.append(budget)
        budget = 2*int(math.ceil(budget*1.125/2))
    return sorted(set(result+[cap]))


def rational_arm(numerator, denominator):
    divisor = math.gcd(numerator, denominator)
    p, q = numerator//divisor, denominator//divisor
    return {"cos_numerator": q*q-p*p, "sin_numerator": 2*p*q,
            "denominator": q*q+p*p}


def arm_key(arm):
    return tuple(arm[key] for key in ("cos_numerator", "sin_numerator", "denominator"))


def add_paid_row(rows, arm, trials, successes):
    for row in rows:
        if arm_key(row) == arm_key(arm):
            row["trials"] += trials
            row["successes"] += successes
            return
    rows.append(dict(arm, trials=trials, successes=successes))


def paid_fit(rows):
    gram, rhs = np.eye(3), np.array([.5, 0., 0.])
    for row in rows:
        c, s, d = arm_key(row)
        feature = np.array([1., c/d, s/d])
        gram += row["trials"]*np.outer(feature, feature)
        rhs += row["successes"]*feature
    raw = np.linalg.solve(gram, rhs)
    a = float(np.clip(raw[0], .0001, .9999))
    radius = float(np.hypot(raw[1], raw[2]))
    contraction = min(1., .98*min(a, 1-a)/max(radius, 1e-300))
    return a, raw[1]*contraction, raw[2]*contraction


def predictable_arm(phase, policy, rows, spec):
    if phase == "psr":
        return rational_arm(1, 1)
    if phase in ("explore_a", "explore_b"):
        return spec["exploration_coordinates"][0 if phase == "explore_a" else 1]
    require(phase == "exploit", "unknown phase")
    a, b, c = paid_fit(rows)
    if policy == "closed":
        d, linear = a*(1-a)-b*b, (1-2*a)*b
        cosine = -linear/(d+math.sqrt(max(0., (d-linear)*(d+linear))))
        tangent = math.sqrt((1-cosine)/(1+cosine))
        fraction = Fraction(tangent).limit_denominator(65536)
        return rational_arm(fraction.numerator, fraction.denominator)
    require(policy == "grid", "unexpected exploit policy")
    best, largest = None, -math.inf
    for arm in spec["grid"]:
        if arm["sin_numerator"] <= 0:
            continue
        co, si = arm["cos_numerator"]/arm["denominator"], arm["sin_numerator"]/arm["denominator"]
        midpoint, shift = a+b*co, c*si
        minus, plus = midpoint-shift, midpoint+shift
        value = .5*(xlogy(minus, minus/midpoint)+xlogy(1-minus, (1-minus)/(1-midpoint))
                     +xlogy(plus, plus/midpoint)+xlogy(1-plus, (1-plus)/(1-midpoint)))
        if value > largest:
            best, largest = arm, value
    return {key: best[key] for key in ("cos_numerator", "sin_numerator", "denominator")}


def expected_phases(policy, previous, budget, explored, spec):
    available = (budget-previous)//2
    if policy == "psr" or previous < spec["explore_after_previous_shots"]:
        return [("psr", available)]
    target = int(math.ceil(spec["sqrt_multiplier"]*math.sqrt(budget/2)))
    deficit = min(available, max(0, target-explored))
    return [(phase, n) for phase, n in
            (("explore_a", (deficit+1)//2), ("explore_b", deficit//2),
             ("exploit", available-deficit)) if n]


def pair_log(counts, experts):
    b, c = int(counts[1]), int(counts[2])
    logs = [b*math.log(2*e["numerator"]/(e["numerator"]+e["denominator"]))
            +c*math.log(2*e["denominator"]/(e["numerator"]+e["denominator"])) for e in experts]
    return float(logsumexp(logs)-math.log(len(experts)))


def mixture_bounds(logs):
    offset = max(value.hi for value in logs)
    scaled = sum(((value-_Bounds(offset)).exp() for value in logs), _Bounds(0))
    return _Bounds(offset)+(scaled/len(logs)).log()


def audit_certificate(engine, counts, state, delta, experts, fisher=None):
    if engine == "beta":
        return certificate_from_pairs(tuple(map(int, counts)), delta=delta)
    detail = None
    if engine == "pair_grid":
        b, c = int(counts[1]), int(counts[2])
        evidence = mixture_bounds([
            b*(_Bounds(2*e["numerator"])/(e["numerator"]+e["denominator"])).log()
            +c*(_Bounds(2*e["denominator"])/(e["numerator"]+e["denominator"])).log()
            for e in experts])
    else:
        detail = fisher or cf.certificate_from_blocks(state["blocks"], delta=delta)
        evidence = _Bounds(Decimal(detail["evidence_bounds"][0]), Decimal(detail["evidence_bounds"][1]))
        if engine == "hedge":
            b, c = int(counts[1]), int(counts[2])
            beta = _log_marginal(b, b+c, .5)+(b+c)*_Bounds(2).log()
            evidence = mixture_bounds([beta, evidence])
        else:
            require(engine == "fisher", "unknown evidence engine")
    threshold = -_Bounds(Decimal(str(delta))).log()
    score = int(counts[1]-counts[2])
    decision = ("reject" if score > 0 else "accept") if (
        score != 0 and evidence.lo >= threshold.hi) else "abstain"
    return {"decision": decision, "method": engine, "shots": int(2*sum(counts)),
            "score": score, "evidence_bounds": [str(evidence.lo), str(evidence.hi)],
            "threshold_bounds": [str(threshold.lo), str(threshold.hi)],
            "fisher_retained_terms": (sum(sum(x["retained_terms_per_positive_odds"])
                                         for x in detail["diagnostics"]) if detail else None)}


def prior_checks(spec):
    protocol = spec["conditional_protocol"]
    require(protocol == cf.protocol_config(), "conditional prior/config changed")
    experts = protocol["experts"]
    require(all(e["weight"] == [1, len(experts)] for e in experts), "unequal expert weights")
    odds = [(e["numerator"], e["denominator"]) for e in experts]
    require(all((q, p) in odds for p, q in odds), "odds grid not reciprocal")
    max_error = 0.
    for counts in ([12, 0, 0, 7], [1, 2, 3, 0], [100, 83, 22, 15], [0, 900, 1100, 0]):
        score, discordant = counts[1]-counts[2], counts[1]+counts[2]
        logs = []
        for e in experts:
            eta = math.log(e["numerator"]/e["denominator"])
            cgf = float(np.logaddexp(eta/2, -eta/2)-math.log(2))
            logs.append(score*eta/2-discordant*cgf)
        expected = float(logsumexp(logs)-math.log(len(experts)))
        actual = pair_log(counts, experts)
        max_error = max(max_error, abs(expected-actual))
        require(abs(expected-actual) < 1e-9, "matched prior m=1 identity failed")
    return {"same_rational_odds_and_weights": True, "expert_count": len(experts),
            "m1_conditional_pair_log_identity_max_error": max_error}


def records_by_trial(path):
    current_key, current = None, []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            require(bool(line.strip()), "blank ledger line {}".format(line_number))
            record = json.loads(line)
            key = record["case_id"], record["policy"], record["rep"]
            if current_key is not None and key != current_key:
                yield current_key, current
                current = []
            current_key = key
            current.append(record)
    if current_key is not None:
        yield current_key, current


def audit_trial(key, records, spec, case):
    case_id, policy, rep = key
    delta, experts = spec["delta"], spec["conditional_protocol"]["experts"]
    require(len(records) == len(spec["engines"]), "incomplete engines in trial {}".format(key))
    by_engine = {record["engine"]: record for record in records}
    require(set(by_engine) == set(spec["engines"]), "duplicate/missing engine")
    seed = spec["seed_base"]+1009*case_id+1000003*rep
    longest = max(records, key=lambda record: record["shots"])
    history = longest["history"]
    require(history and history[-1]["budget"] == longest["shots"], "invalid trajectory end")
    for record in records:
        require(record["case_id"] == case_id and record["rep"] == rep and
                record["policy"] == policy and record["seed"] == seed, "identity/seed mismatch")
        require(record["method"] == policy+"_"+record["engine"], "method mismatch")
        require(record["history"] == [block for block in history if block["budget"] <= record["shots"]],
                "engines did not share an identical trajectory prefix")
    state, rows = cf.empty_state(), []
    counts = np.zeros(4, dtype=np.int64)
    previous, explored, look_count, cert_calls, screen_checks = 0, 0, 0, 0, 0
    first_screen, stopped, checkpoint_counts = {}, set(), {}
    maximum_screen_error = 0.
    threshold = math.log(1/delta)-1e-6
    initial_prefix = []
    for look, grouped in groupby(history, key=lambda block: block["look"]):
        blocks = list(grouped)
        look_count += 1
        require(look == look_count and look <= len(spec["checkpoints"]), "nonconsecutive looks")
        budget = spec["checkpoints"][look-1]
        phases = expected_phases(policy, previous, budget, explored, spec)
        require([(b["phase"], b["m"]) for b in blocks] == phases, "phase schedule/cost mismatch")
        for block in blocks:
            require(block["budget"] == budget, "wrong checkpoint budget")
            arm = predictable_arm(block["phase"], policy, rows, spec)
            require(block["arm"] == arm, "action was not reconstructed from previous paid data")
            co, si, denominator = arm_key(arm)
            require(all(type(x) is int for x in (co, si, denominator)) and denominator > 0
                    and co*co+si*si == denominator*denominator and si > 0, "invalid rational query")
            n, fresh = block["m"], block["pair_counts"]
            require(type(n) is int and n > 0 and len(fresh) == 4 and
                    all(type(x) is int and x >= 0 for x in fresh) and sum(fresh) == n,
                    "invalid joint counts")
            minus, plus = fresh[2]+fresh[3], fresh[1]+fresh[3]
            require(block["minus_successes"] == minus and block["plus_successes"] == plus,
                    "marginal/joint count mismatch")
            counts += fresh
            cf.update_block(state, n, minus, plus)
            add_paid_row(rows, arm, n, plus)
            negative = dict(arm, sin_numerator=-si)
            add_paid_row(rows, negative, n, minus)
            if block["phase"].startswith("explore"):
                explored += n
            if previous < spec["explore_after_previous_shots"]:
                initial_prefix.append(block)
        require(int(counts.sum())*2 == budget == state["shots"] == sum(row["trials"] for row in rows),
                "paid shots mismatch")
        require(state["score"] == int(counts[1]-counts[2]), "score mismatch")
        b, c = counts[1:3]
        beta = float(gammaln(b+.5)+gammaln(c+.5)-gammaln(b+c+1)-2*gammaln(.5)+(b+c)*math.log(2))
        fisher = cf.screen(state, delta)["log_evidence"]
        screens = {"beta": beta, "pair_grid": pair_log(counts, experts), "fisher": fisher,
                   "hedge": float(np.logaddexp(beta, fisher)-math.log(2))}
        cached_fisher = None
        for engine in spec["engines"]:
            if engine in stopped:
                continue
            screen_checks += 1
            record = by_engine[engine]
            require(budget <= record["shots"], "recorded stop had no preceding valid crossing")
            eligible = screens[engine] >= threshold or budget == spec["cap"]
            if not eligible:
                require(budget != record["shots"], "stopped without eligible floating screen")
                continue
            if screens[engine] >= threshold and engine not in first_screen:
                first_screen[engine] = budget
            if engine in ("fisher", "hedge") and cached_fisher is None:
                cached_fisher = cf.certificate_from_blocks(state["blocks"], delta=delta)
            cert = audit_certificate(engine, counts, state, delta, experts, cached_fisher)
            cert_calls += 1
            lo, hi = map(float, cert["evidence_bounds"])
            maximum_screen_error = max(maximum_screen_error, lo-screens[engine], screens[engine]-hi, 0.)
            if cert["decision"] != "abstain" or budget == spec["cap"]:
                require(budget == record["shots"], "missed earlier certified crossing")
                require(record["certificate"] == cert, "final Decimal certificate mismatch")
                require(record["decision"] == cert["decision"] and record["counts"] == counts.tolist(),
                        "final decision/count mismatch")
                require(record["exploration_pairs"] == explored, "exploration cost mismatch")
                require(cert["shots"] == budget, "certificate shot mismatch")
                stopped.add(engine)
                checkpoint_counts[engine] = look
            else:
                require(budget < record["shots"], "abstention ended before cap")
        previous = budget
    require(stopped == set(spec["engines"]), "unstopped engines after final history")
    return {"history_blocks": len(history), "screen_checks": screen_checks,
            "certificates_recomputed": cert_calls, "final_certificates": len(records),
            "maximum_screen_outside_certificate": maximum_screen_error,
            "first_float_crossing": first_screen, "stop_looks": checkpoint_counts,
            "initial_prefix": initial_prefix}


def audit(directory, output):
    started = time.perf_counter()
    directory = Path(directory).resolve()
    require(directory.is_dir(), "result directory missing")
    spec_path, summary_path, ledger_path = [directory/name for name in ("spec.json", "summary.json", "trials.jsonl")]
    require(summary_path.exists(), "run incomplete: summary.json is required before audit")
    require(not output.exists(), "preserve completed audits; choose another output path")
    inputs = {name: sha256(directory/name) for name in ("spec.json", "summary.json", "trials.jsonl")}
    own_hash = sha256(__file__)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require(summary["spec"] == spec, "summary/spec mismatch")
    require(spec["checkpoints"] == expected_checkpoints(spec["cap"]), "checkpoint specification mismatch")
    require(spec["seed_rule"] == "base+1009*case_id+1000003*rep", "unsupported seed rule")
    require(spec["delta"] == runner.DELTA and spec["explore_after_previous_shots"] == runner.EXPLORE_AFTER
            and spec["sqrt_multiplier"] == runner.SQRT_MULTIPLIER, "runner/spec protocol mismatch")
    require(spec["sources"] == runner.source_hashes(), "run source changed since freezing")
    frozen_before = frozen_evidence_checks()
    prior = prior_checks(spec)
    cases = {case["case_id"]: case for case in spec["cases"]}
    require(len(cases) == len(spec["cases"]), "duplicate case IDs")
    require(len(set(spec["policies"])) == len(spec["policies"]) and
            set(spec["policies"]) <= set(runner.POLICIES), "invalid policies")
    require(len(set(spec["engines"])) == len(spec["engines"]) and
            set(spec["engines"]) <= set(runner.ENGINES), "invalid engines")
    repetitions = spec["repetitions"]
    require(type(repetitions) is int and repetitions > 0, "invalid repetition count")
    selected = sorted(set((0, repetitions-1)))
    expected_keys = {(case_id, policy, rep) for case_id in cases for policy in spec["policies"]
                     for rep in range(repetitions)}
    seen, prefixes, cells = set(), {}, defaultdict(list)
    totals, replay_hashes, stop_records = Counter(), [], []
    maximum_screen_error = 0.
    for key, records in records_by_trial(ledger_path):
        require(key in expected_keys and key not in seen, "unexpected or duplicated trial")
        seen.add(key)
        case_id, policy, rep = key
        checked = audit_trial(key, records, spec, cases[case_id])
        for name in ("history_blocks", "screen_checks", "certificates_recomputed", "final_certificates"):
            totals[name] += checked[name]
        maximum_screen_error = max(maximum_screen_error, checked["maximum_screen_outside_certificate"])
        stop_records.append({"case_id": case_id, "policy": policy, "rep": rep,
                             "first_float_crossing": checked["first_float_crossing"],
                             "stop_looks": checked["stop_looks"]})
        prefix_key = case_id, rep
        prefix = checked["initial_prefix"]
        if prefix_key in prefixes:
            previous_prefix = prefixes[prefix_key]
            common = min(len(prefix), len(previous_prefix))
            require(prefix[:common] == previous_prefix[:common], "same seed initial policy coupling mismatch")
            totals["policy_prefix_comparisons"] += 1
            if len(prefix) > len(previous_prefix):
                prefixes[prefix_key] = prefix
        else:
            prefixes[prefix_key] = prefix
        for record in records:
            cells[(case_id, record["method"])].append({"shots": record["shots"], "decision": record["decision"]})
        if rep in selected:
            seed = spec["seed_base"]+1009*case_id+1000003*rep
            replayed, unused_timing = runner.run_trial(cases[case_id], policy, rep, seed,
                                                      spec["cap"], spec["engines"])
            actual = sorted(records, key=lambda r: r["engine"])
            expected = sorted(replayed, key=lambda r: r["engine"])
            require(actual == expected, "full seeded replay differs")
            replay_hashes.append({"case_id": case_id, "policy": policy, "rep": rep, "seed": seed,
                                  "records_sha256": stable_json_hash(actual)})
            totals["replayed_trials"] += 1
            totals["replayed_records"] += len(records)
        if rep == repetitions-1:
            print(json.dumps({"audited_case": case_id, "policy": policy, "trials": len(seen),
                              "records": totals["final_certificates"],
                              "elapsed_seconds": time.perf_counter()-started}), flush=True)
    require(seen == expected_keys, "ledger missing trials")
    require(summary["new_trajectories"] == totals["final_certificates"], "summary record count mismatch")
    reported_cells = {(cell["case_id"], cell["method"]): cell for cell in summary["cells"]}
    require(len(reported_cells) == len(summary["cells"]) and set(reported_cells) == set(cells),
            "summary cell identity mismatch")
    outcomes = Counter()
    for key, records in cells.items():
        case_id, method = key
        case = cases[case_id]
        expected_decision = "accept" if case["C"] < 0 else "reject" if case["C"] > 0 else "abstain"
        expected = {"case_id": case_id, "kind": case["kind"], "method": method,
                    "repetitions": len(records), "mean_shots": float(np.mean([r["shots"] for r in records])),
                    "wrong": sum(r["decision"] not in ("abstain", expected_decision) for r in records),
                    "abstained": sum(r["decision"] == "abstain" for r in records)}
        require(reported_cells[key] == expected, "summary cost/outcome cell mismatch")
        outcomes["wrong"] += expected["wrong"]
        outcomes["abstained"] += expected["abstained"]
        outcomes["paid_shots_across_protocol_records"] += sum(r["shots"] for r in records)
    require(inputs == {name: sha256(directory/name) for name in inputs}, "input changed during audit")
    require(sha256(__file__) == own_hash and spec["sources"] == runner.source_hashes(),
            "audit/run source changed during audit")
    require(frozen_evidence_checks() == frozen_before, "frozen evidence changed during audit")
    result = {"status": "passed", "directory": str(directory), "run_input_sha256": inputs,
              "audit_source_sha256": own_hash, "run_sources_match_spec": True,
              "frozen_evidence_before_and_after": frozen_before,
              "spec_summary_and_all_cells_match": True, "trial_groups": len(seen),
              "replay_rule": "For every case and policy choose reps 0 and R-1, deduplicated; no outcome selection",
              "replayed_rep_indices": selected, "checks": dict(totals), "outcomes": dict(outcomes),
              "matched_prior": prior, "maximum_screen_outside_certificate": maximum_screen_error,
              "decision_rule": "Recomputed first eligible float screen followed by outward Decimal crossing; cap abstentions retained",
              "limits": "Full ledger/strategy/certificate audit with selected complete seed replays; not a proof of power or SOTA",
              "replays": replay_hashes, "stopping_checks": stop_records,
              "elapsed_seconds": time.perf_counter()-started}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("replays", "stopping_checks")}, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Completed result directory")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    audit(args.directory, args.output or args.directory/"audit.json")
