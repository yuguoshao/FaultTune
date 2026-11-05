"""Full refined audit while an append-only experiment ledger is still written.

Only complete trial groups are checked. Every observed original ledger byte
is hashed, then compared with the completed file before final acceptance.
No runner or existing audit source is modified and no sample is selected by
outcome. The same final certificates and mechanical seed replays are checked
as by audit_refined.py, overlapping audit work with remaining experiments.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import audit_refined as checked

base, runner, require = checked.base, checked.runner, checked.require


def source_hashes():
    return {name: base.sha256(Path(__file__).with_name(name)) for name in
            ("audit_refined_incremental.py", "audit_refined.py", "audit_conditional.py")}


def audit(directory, output):
    started = time.perf_counter()
    started_utc = datetime.utcnow().isoformat()+"Z"
    directory = Path(directory).resolve()
    require(not output.exists(), "preserve completed audits; choose another output path")
    spec_path, summary_path, ledger_path = [directory/name for name in ("spec.json", "summary.json", "trials.jsonl")]
    require(spec_path.exists() and ledger_path.exists(), "specification and ledger required")
    began_before_summary = not summary_path.exists()
    spec_digest, own_sources = base.sha256(spec_path), source_hashes()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    require(spec["sources"] == runner.source_hashes(), "run source changed since freezing")
    require(spec["checkpoints"] == base.expected_checkpoints(spec["cap"]), "checkpoint schedule differs")
    require(spec["seed_rule"] == "base+1009*case_id+1000003*rep", "unsupported seed rule")
    require(spec["delta"] == runner.DELTA and spec["explore_after_previous_shots"] == runner.EXPLORE_AFTER
            and spec["sqrt_multiplier"] == runner.SQRT_MULTIPLIER, "runtime constants differ from spec")
    frozen, priors = base.frozen_evidence_checks(), checked.prior_checks(spec)
    cases = {case["case_id"]: case for case in spec["cases"]}
    require(len(cases) == len(spec["cases"]), "duplicate case IDs")
    require(len(set(spec["policies"])) == len(spec["policies"]) and set(spec["policies"]) <= set(runner.POLICIES),
            "invalid policies")
    require(len(set(spec["engines"])) == len(spec["engines"]) and set(spec["engines"]) <= set(runner.ENGINES),
            "invalid engines")
    repetitions = spec["repetitions"]
    require(type(repetitions) is int and repetitions > 0, "invalid repetition count")
    selected = sorted(set((0, repetitions-1)))
    expected_keys = {(case_id, policy, rep) for case_id in cases for policy in spec["policies"]
                     for rep in range(repetitions)}
    seen, cells, prefixes = set(), defaultdict(list), {}
    totals, replays, stops, trial_hashes, maximum_errors = Counter(), [], [], [], {}
    ledger_hash, lines, wait_seconds = hashlib.sha256(), 0, 0.
    final_deadline = datetime.fromisoformat(spec["run_control"]["hard_deadline_utc"].replace("Z", "+00:00")).timestamp()
    # A completed historical run may be audited later. Only waiting for a
    # still-running writer observes the original run's hard deadline.
    enforce_wait_deadline = began_before_summary
    current_key, group, group_hash = None, [], hashlib.sha256()
    with ledger_path.open("rb") as stream:
        while True:
            position = stream.tell()
            line = stream.readline()
            if not line or not line.endswith(b"\n"):
                stream.seek(position)
                if summary_path.exists():
                    require(not line and not group, "completed ledger ended with a partial record or trial")
                    break
                require(not enforce_wait_deadline or time.time() < final_deadline,
                        "run hard deadline reached without complete summary; no final audit acceptance")
                before = time.perf_counter()
                time.sleep(.25)
                wait_seconds += time.perf_counter()-before
                continue
            require(bool(line.strip()), "blank ledger line")
            ledger_hash.update(line)
            group_hash.update(line)
            lines += 1
            record = json.loads(line)
            key = record["case_id"], record["policy"], record["rep"]
            if current_key is None:
                current_key = key
            require(key == current_key, "new trial started before all prior engines were written")
            group.append(record)
            require(len(group) <= len(spec["engines"]), "too many engines in a trial")
            if len(group) < len(spec["engines"]):
                continue
            require(key in expected_keys and key not in seen, "unexpected or duplicated trial")
            seen.add(key)
            case_id, policy, rep = key
            result = checked.audit_trial(key, group, spec)
            for name in ("history_blocks", "past_safe_predictions", "rb_marginal_blocks", "screen_checks",
                         "certificates_recomputed", "final_certificates"):
                totals[name] += result[name]
            for engine, error in result["maximum_screen_errors"].items():
                maximum_errors[engine] = max(maximum_errors.get(engine, 0.), error)
            stops.append({"case_id": case_id, "policy": policy, "rep": rep,
                          "first_float_crossing": result["first_float_crossing"], "stop_looks": result["stop_looks"]})
            trial_hashes.append({"case_id": case_id, "policy": policy, "rep": rep,
                                 "original_ledger_bytes_sha256": group_hash.hexdigest()})
            prefix_key, prefix = (case_id, rep), result["initial_prefix"]
            if prefix_key in prefixes:
                old = prefixes[prefix_key]
                common = min(len(old), len(prefix))
                require(old[:common] == prefix[:common], "same-seed initial policy coupling differs")
                totals["policy_prefix_comparisons"] += 1
                if len(prefix) > len(old):
                    prefixes[prefix_key] = prefix
            else:
                prefixes[prefix_key] = prefix
            for entry in group:
                cells[(case_id, entry["method"])].append({"shots": entry["shots"], "decision": entry["decision"]})
            if rep in selected:
                seed = spec["seed_base"]+1009*case_id+1000003*rep
                replayed, unused_timing = runner.run_trial(cases[case_id], policy, rep, seed, spec["cap"], spec["engines"])
                actual, replay = sorted(group, key=lambda r: r["engine"]), sorted(replayed, key=lambda r: r["engine"])
                require(actual == replay, "full seeded replay differs")
                replays.append({"case_id": case_id, "policy": policy, "rep": rep, "seed": seed,
                                "records_sha256": base.stable_json_hash(actual)})
                totals["replayed_trials"] += 1
                totals["replayed_records"] += len(group)
            if rep == repetitions-1:
                print(json.dumps({"audited_case": case_id, "policy": policy, "trials": len(seen),
                                  "records": totals["final_certificates"], "summary_exists": summary_path.exists(),
                                  "elapsed_seconds": time.perf_counter()-started}), flush=True)
            current_key, group, group_hash = None, [], hashlib.sha256()
    require(seen == expected_keys, "completed ledger lacks required trials")
    require(lines == totals["final_certificates"], "parsed line count differs from audited records")
    inputs = {"spec.json": base.sha256(spec_path), "summary.json": base.sha256(summary_path),
              "trials.jsonl": base.sha256(ledger_path)}
    require(spec_digest == inputs["spec.json"], "specification changed during incremental audit")
    require(ledger_hash.hexdigest() == inputs["trials.jsonl"], "completed ledger differs from original bytes audited")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require(summary["spec"] == spec, "summary specification differs")
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
        require(reported[key] == cell, "summary cost or outcome cell differs")
        outcomes["wrong"] += cell["wrong"]
        outcomes["abstained"] += cell["abstained"]
        outcomes["paid_shots_across_protocol_records"] += sum(r["shots"] for r in records)
    require(own_sources == source_hashes(), "audit source changed during verification")
    require(spec["sources"] == runner.source_hashes(), "run source changed during verification")
    require(base.frozen_evidence_checks() == frozen, "old frozen evidence changed")
    require(inputs == {name: base.sha256(directory/name) for name in inputs}, "completed input changed during final checks")
    output_data = {"status": "passed", "directory": str(directory), "run_input_sha256": inputs,
                   "audit_sources_sha256": own_sources, "run_sources_match_spec": True,
                   "frozen_evidence_before_and_after": frozen, "spec_summary_and_all_cells_match": True,
                   "audit_mode": "Incremental full verification of complete trials; final whole-file byte hash equality",
                   "began_before_summary": began_before_summary, "started_utc": started_utc,
                   "completed_utc": datetime.utcnow().isoformat()+"Z", "wait_seconds": wait_seconds,
                   "audited_ledger_bytes_sha256": ledger_hash.hexdigest(), "audited_ledger_lines": lines,
                   "trial_groups": len(seen), "checks": dict(totals), "outcomes": dict(outcomes),
                   "replay_rule": "Reps 0 and R-1 for every case and policy, deduplicated, no outcome selection",
                   "replayed_rep_indices": selected, "matched_prior": priors,
                   "maximum_screen_outside_certificate_by_engine": maximum_errors,
                   "decision_rule": "First eligible float screen with outward Decimal crossing; all cap abstentions retained",
                   "limits": "Complete protocol and certificate audit with mechanical seed replays; no power/SOTA conclusion",
                   "trial_ledger_hashes": trial_hashes, "replays": replays, "stopping_checks": stops,
                   "elapsed_seconds": time.perf_counter()-started}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_data, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in output_data.items()
                      if key not in ("trial_ledger_hashes", "replays", "stopping_checks")}, indent=2), flush=True)
    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    audit(args.directory, args.output or args.directory/"audit.json")
