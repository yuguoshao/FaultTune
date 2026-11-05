"""Audit each flushed complete unequal record with the original full auditor.

No experiment, stopping rule, certificate or audit coverage is changed. The
final audit is issued only after all expected records and the final summary
exist, and every byte audited while the ledger grew matches its final hash.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

import audit_unequal as original


def audit(directory, output, watch=False):
    started = time.perf_counter()
    directory, output = Path(directory).resolve(), Path(output).resolve()
    original.require(not output.exists(), "Preserve completed audit output")
    spec_path, summary_path, ledger_path = [directory/name for name in
                                          ("spec.json", "summary.json", "trials.jsonl")]
    spec_bytes = spec_path.read_bytes()
    spec_hash = hashlib.sha256(spec_bytes).hexdigest()
    spec = json.loads(spec_bytes)
    source_paths = [Path(__file__), Path(original.__file__)]
    sources = {p.name: original.digest(p) for p in source_paths}
    original.require(spec["sources"] == original.runner.source_hashes(), "Run source changed")
    original.require(spec["evidence_protocol"] == original.evidence.protocol_config(), "Evidence prior changed")
    original.require(spec["checkpoints"] == original.runner.schedule(spec["cap"]), "Checkpoint schedule changed")
    deadline = datetime.fromisoformat(spec["run_control"]["hard_deadline_utc"].replace("Z", "+00:00")).timestamp()
    cases = {c["case_id"]: c for c in spec["cases"]}
    expected = {(cid, policy, rep) for cid in cases for policy in spec["policies"]
                for rep in range(spec["repetitions"])}
    seen, cells, prefixes, totals = set(), defaultdict(list), {}, Counter()
    ledger_hash, offset = hashlib.sha256(), 0
    progress_path = output.with_name(output.stem+"_progress.json")
    original.require(not progress_path.exists(), "Use one incremental auditor with a new progress path")

    def check_sources():
        original.require(original.digest(spec_path) == spec_hash, "Spec changed while auditing")
        original.require(sources == {p.name: original.digest(p) for p in source_paths}, "Audit source changed")
        original.require(spec["sources"] == original.runner.source_hashes(), "Run source changed during audit")

    def save_progress(status):
        value = {"status": status, "trajectories": len(seen), "expected_trajectories": len(expected),
                 "audited_ledger_bytes": offset, "audited_prefix_sha256": ledger_hash.hexdigest(),
                 "checks": dict(totals), "audit_sources_sha256": sources,
                 "elapsed_seconds": time.perf_counter()-started,
                 "scope": "Progress only; no completed audit or inferential claim"}
        temporary = progress_path.with_name(progress_path.name+".pending")
        temporary.write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")
        os.replace(str(temporary), str(progress_path))

    with ledger_path.open("rb") as ledger:
        while seen != expected:
            if time.time() >= deadline:
                save_progress("incomplete_hard_deadline")
                raise RuntimeError("Hard deadline reached; partial audit progress retained")
            ledger.seek(offset)
            raw = ledger.readline()
            if not raw.endswith(b"\n"):
                if not watch:
                    save_progress("incomplete_no_complete_record")
                    return
                time.sleep(5)
                continue
            record = json.loads(raw)
            key = record["case_id"], record["policy"], record["rep"]
            original.require(key in expected and key not in seen, "Duplicate or unexpected trajectory")
            checked = original.audit_record(record, cases[key[0]], spec)
            for metric in ("blocks", "screens", "certificates", "plans"):
                totals[metric] += checked[metric]
            pair_key, prefix = (key[0], key[2]), checked["prefix"]
            if pair_key in prefixes:
                previous = prefixes[pair_key]
                common = min(len(prefix), len(previous))
                original.require(prefix[:common] == previous[:common], "Policy prefixes differ")
                totals["shared_prefix_checks"] += 1
                if len(prefix) > len(previous):
                    prefixes[pair_key] = prefix
            else:
                prefixes[pair_key] = prefix
            fields = ("shots", "decision", "full_grid_replans", "pair_fallback_replans")
            cells[(key[0], record["method"])].append({k: record[k] for k in fields})
            seen.add(key)
            ledger_hash.update(raw)
            offset += len(raw)
            totals["seed_replayed_trajectories"] += 1
            if key[2] == spec["repetitions"]-1:
                check_sources()
                save_progress("running")
                print(json.dumps({"audited_case": key[0], "policy": key[1], "trajectories": len(seen),
                                  "elapsed_seconds": time.perf_counter()-started}), flush=True)

    while True:
        if time.time() >= deadline:
            save_progress("incomplete_hard_deadline_before_summary")
            raise RuntimeError("Hard deadline before completed summary")
        try:
            summary_bytes = summary_path.read_bytes()
            summary = json.loads(summary_bytes)
            break
        except (FileNotFoundError, json.JSONDecodeError):
            if not watch:
                save_progress("incomplete_summary_unavailable")
                return
            time.sleep(5)
    paths = {name: directory/name for name in ("spec.json", "summary.json", "trials.jsonl")}
    inputs = {name: original.digest(path) for name, path in paths.items()}
    original.require(inputs["spec.json"] == spec_hash and summary["spec"] == spec, "Final spec mismatch")
    original.require(inputs["summary.json"] == hashlib.sha256(summary_bytes).hexdigest(), "Parsed summary bytes changed")
    original.require(ledger_path.stat().st_size == offset and inputs["trials.jsonl"] == ledger_hash.hexdigest(),
                     "Final ledger differs from exactly the complete bytes audited")
    original.require(len(seen) == summary["new_trajectories"], "Final trajectory count differs")
    reported = {(c["case_id"], c["method"]): c for c in summary["cells"]}
    original.require(set(reported) == set(cells) and len(reported) == len(summary["cells"]), "Summary cells differ")
    outcomes = Counter()
    for (case_id, method), records in cells.items():
        case = cases[case_id]
        expected_decision = "accept" if case["C"] < 0 else "reject" if case["C"] > 0 else "abstain"
        rebuilt = {"case_id": case_id, "kind": case["kind"], "method": method,
                   "repetitions": len(records), "mean_shots": float(np.mean([r["shots"] for r in records])),
                   "wrong": sum(r["decision"] not in ("abstain", expected_decision) for r in records),
                   "abstained": sum(r["decision"] == "abstain" for r in records),
                   "full_grid_replans": sum(r["full_grid_replans"] for r in records),
                   "pair_fallback_replans": sum(r["pair_fallback_replans"] for r in records)}
        original.require(reported[(case_id, method)] == rebuilt, "Summary unconditional metric differs")
        outcomes["wrong"] += rebuilt["wrong"]
        outcomes["abstained"] += rebuilt["abstained"]
        outcomes["paid_shots"] += sum(r["shots"] for r in records)
    check_sources()
    original.require(inputs == {name: original.digest(path) for name, path in paths.items()}, "Final inputs changed")
    result = {"status": "passed", "audit_type": "implementation and data-flow audit, not independent theoretical verification",
              "directory": str(directory), "trajectories": len(seen), "checks": dict(totals),
              "outcomes": dict(outcomes), "input_sha256": inputs, "audit_source_sha256": sources[Path(__file__).name],
              "audit_sources_sha256": sources, "all_rng_streams_replayed": True,
              "all_predictable_actions_rebuilt_from_paid_counts": True,
              "all_rounded_allocations_and_costs_reconstructed": True,
              "all_first_certified_stops_recomputed": True, "summary_cells_rebuilt": True,
              "all_original_full_interval_certificates_recomputed": True,
              "incremental_prefix_equals_final_ledger": True,
              "limits": "Same original audit_record and full Decimal normalizers for every trajectory; no lazy certificates, power or SOTA claim.",
              "elapsed_seconds": time.perf_counter()-started}
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2)+"\n")
    save_progress("complete")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    audit(args.directory, args.output or args.directory/"audit.json", args.watch)
