"""Replay predetermined frozen trials through the SAME registered implementation.

For every case/policy cell, select trial indices sorted({0,repetitions-1}),
independently of observed outcomes. Read the raw JSONL/gzip once, retain only
these records, then rerun run_experiments.run_trial with the recorded seed rule.
All record fields except started_epoch and elapsed_seconds must match exactly.
This is deterministic same-implementation reproduction, not an independent
implementation or an additional statistical confirmation sample.
"""

import argparse
import gzip
import hashlib
import json
import multiprocessing
from pathlib import Path
import platform
import sys

from audit_records import strict_json_loads


SCHEMA = "faulttune.registered_replay_audit.v1"
IGNORED_WALLTIME_FIELDS = frozenset(("started_epoch", "elapsed_seconds"))
_SPEC = None
_SPEC_HASH = None
_CASES = None
_RUN_TRIAL = None


def selected_indices(repetitions):
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    return sorted({0, repetitions - 1})


def registered_keys(spec):
    indices = selected_indices(spec["repetitions"])
    return [(case["case_id"], policy, index)
            for case in spec["cases"] for policy in spec["policies"] for index in indices]


def read_selected_records(spec, spec_hash, trials_path):
    """One sequential raw-file pass; selection never depends on result fields."""
    keys = registered_keys(spec)
    wanted = set(keys)
    selected, issues = {}, []
    lines = 0
    issue_count = 0

    def issue(code, detail, line=None):
        nonlocal issue_count
        issue_count += 1
        if len(issues) < 1000:
            issues.append({"code": code, "detail": detail, "line": line})

    path = Path(trials_path)
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                lines += 1
                try:
                    record = strict_json_loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("record must be a JSON object")
                    if record.get("spec_sha256") != spec_hash or record.get("phase") != spec["phase"]:
                        issue("record_spec", "record does not identify the selected frozen spec/phase", line_number)
                    key = (record.get("case_id"), record.get("policy"), record.get("trial_index"))
                    if isinstance(key[2], bool) or not isinstance(key[2], int):
                        raise ValueError("trial index must be an integer")
                    if key in wanted:
                        if key in selected:
                            issue("duplicate_selected_key", str(key), line_number)
                        else:
                            selected[key] = record
                except (ValueError, TypeError) as error:
                    issue("record_json_or_identity", str(error), line_number)
    except (OSError, EOFError, UnicodeDecodeError) as error:
        issue("trials_read_error", type(error).__name__ + ": " + str(error))
    missing = [key for key in keys if key not in selected]
    return {"ordered_keys": keys, "selected": selected, "missing_keys": missing,
            "raw_records_scanned": lines, "issues": issues, "issue_count": issue_count}


def comparison_payload(record):
    return {key: value for key, value in record.items() if key not in IGNORED_WALLTIME_FIELDS}


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def compare_trial_records(expected, replayed):
    """Strict typed JSON comparison, including complete result, cost and events."""
    first, second = comparison_payload(expected), comparison_payload(replayed)
    encoded_first, encoded_second = _canonical(first), _canonical(second)
    fields = sorted(set(first) | set(second))
    missing = object()
    differing = []
    for field in fields:
        left, right = first.get(field, missing), second.get(field, missing)
        if left is missing or right is missing or _canonical(left) != _canonical(right):
            differing.append(field)
    return {"match": encoded_first == encoded_second, "mismatched_fields": differing,
            "expected_comparison_sha256": hashlib.sha256(encoded_first).hexdigest(),
            "replayed_comparison_sha256": hashlib.sha256(encoded_second).hexdigest(),
            "compared_fields": fields}


def _initialize_replay(spec, spec_hash):
    global _SPEC, _SPEC_HASH, _CASES, _RUN_TRIAL
    from run_experiments import run_trial
    _SPEC, _SPEC_HASH = spec, spec_hash
    _CASES = {case["case_id"]: case for case in spec["cases"]}
    _RUN_TRIAL = run_trial


def _replay_one(task):
    order, expected = task
    key = (expected["case_id"], expected["policy"], expected["trial_index"])
    row = {"order": order, "key": list(key), "expected_seed": expected.get("seed"),
           "expected_status": expected.get("status"), "expected_decision": expected.get("decision")}
    try:
        # A goal deadline is a wall-clock condition, not a seeded observation.
        # Original timeout/error records are retained and compared, not filtered.
        replayed = _RUN_TRIAL(_SPEC, _SPEC_HASH, _CASES[key[0]], key[1], key[2],
                              deadline=None, keep_events=True)
        row.update(compare_trial_records(expected, replayed))
        row.update({"replayed_seed": replayed.get("seed"), "replayed_status": replayed.get("status"),
                    "replayed_decision": replayed.get("decision"), "replayed_shots": replayed.get("shots"),
                    "error": None})
    except Exception as error:
        row.update({"match": False, "mismatched_fields": [],
                    "error": type(error).__name__ + ": " + str(error)})
    return row


def _runtime_information():
    import numpy
    import scipy
    return {"python": sys.version, "platform": platform.platform(),
            "numpy": numpy.__version__, "scipy": scipy.__version__}


def _save(path, result):
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def replay_registered(spec_path, trials_path, output_path=None, workers=4):
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    from run_experiments import verify_spec
    try:
        spec, digest = verify_spec(spec_path, check_sources=True)
    except Exception as error:
        result = {"schema": SCHEMA, "passed": False, "stage": "frozen_source_verification",
                  "error": type(error).__name__ + ": " + str(error),
                  "scope": "replay requires the frozen current implementation; archived code is not executed"}
        _save(output_path, result)
        return result
    selection = read_selected_records(spec, digest, trials_path)
    tasks = [(order, selection["selected"][key])
             for order, key in enumerate(selection["ordered_keys"]) if key in selection["selected"]]
    results = []
    if workers == 1 or len(tasks) < 2:
        _initialize_replay(spec, digest)
        results = [_replay_one(task) for task in tasks]
    else:
        context = multiprocessing.get_context("spawn")
        with context.Pool(min(workers, len(tasks)), initializer=_initialize_replay,
                          initargs=(spec, digest)) as pool:
            results = list(pool.imap_unordered(_replay_one, tasks, chunksize=1))
    results.sort(key=lambda row: row["order"])
    postcheck_error = None
    try:
        unused, after_digest = verify_spec(spec_path, check_sources=True)
        if after_digest != digest:
            raise ValueError("frozen specification changed during replay")
    except Exception as error:
        postcheck_error = type(error).__name__ + ": " + str(error)
    environment = _runtime_information()
    environment_checks = {key: {"expected": value, "actual": environment.get(key),
                                "matches": environment.get(key) == value}
                          for key, value in spec.get("environment", {}).items() if key in environment}
    mismatches = sum(not row["match"] for row in results)
    passed = (not selection["issue_count"] and not selection["missing_keys"] and not mismatches
              and postcheck_error is None and all(row["matches"] for row in environment_checks.values()))
    result = {
        "schema": SCHEMA, "passed": passed, "phase": spec["phase"], "spec_sha256": digest,
        "spec_path": str(Path(spec_path).resolve()), "trials_path": str(Path(trials_path).resolve()),
        "scope": "deterministic reproduction by the same frozen run_trial implementation, NumPy sampling primitives and policy code; not an independent algorithm implementation",
        "selection_rule": "for every preregistered case/policy cell, sorted({0,repetitions-1}); no outcome-dependent replacement",
        "selected_trial_indices": selected_indices(spec["repetitions"]),
        "planned_selected_records": len(selection["ordered_keys"]), "raw_records_scanned": selection["raw_records_scanned"],
        "records_replayed": len(results), "records_matched": len(results) - mismatches,
        "records_mismatched": mismatches, "missing_selected_keys": [list(key) for key in selection["missing_keys"]],
        "selection_issue_count": selection["issue_count"], "selection_issues": selection["issues"],
        "ignored_fields": sorted(IGNORED_WALLTIME_FIELDS),
        "deadline_scope": "replays use deadline=None; original timeout/error records are retained and may fail equality because wall-clock interruption is not a seeded event",
        "replay_data_usage": "audit only; replay outcomes do not add statistical repetitions or alter the frozen trial results",
        "workers": workers, "environment_checks": environment_checks,
        "source_hashes_verified_before_and_after": postcheck_error is None,
        "post_replay_source_error": postcheck_error,
        "replay_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "records": results,
    }
    _save(output_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--trials", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    result = replay_registered(args.spec, args.trials, args.output, args.workers)
    print(json.dumps({key: result.get(key) for key in ("passed", "planned_selected_records", "records_replayed",
                                                       "records_matched", "records_mismatched", "stage", "error")}, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
