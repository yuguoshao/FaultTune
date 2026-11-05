"""Supplementary delivery integrity audit, outside the frozen experiment sources.

This read-only checker does not change or rerun any statistical experiment.
It additionally checks the formal spec's top-level source_archive_sha256,
which the frozen audit_records.py archive reader does not recognize when
source_archive is a string. Per-entry hashes remain independently checked.

Omit --trials while a gzip stream is still open. After the runner closes it,
--trials checks every started_epoch and the complete expected record count.
"""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import zipfile


SCHEMA = "faulttune.delivery_integrity.v1"
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
LEGACY_MANIFESTS = (
    "experiments/local_validation/results/manifest.json",
    "experiments/local_validation/results/aggregate_manifest.json",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def contained_path(base, relative):
    path = (base / relative).resolve()
    path.relative_to(base)
    return path


def check_legacy_manifest(repo, relative):
    path = contained_path(repo, relative)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for item in manifest["files"]:
        try:
            current = contained_path(repo, item["path"])
            actual_hash, actual_bytes = sha256_file(current), current.stat().st_size
        except (OSError, ValueError):
            actual_hash, actual_bytes = None, None
        rows.append({"path": item["path"], "expected_sha256": item["sha256"],
                     "actual_sha256": actual_hash, "expected_bytes": item["bytes"],
                     "actual_bytes": actual_bytes,
                     "matches": actual_hash == item["sha256"] and actual_bytes == item["bytes"]})
    return {"manifest_path": relative, "manifest_sha256": sha256_file(path),
            "files_checked": len(rows), "passed": all(row["matches"] for row in rows), "files": rows}


def check_frozen_sources(spec, spec_path, repo):
    archive_declaration = spec.get("source_archive")
    if isinstance(archive_declaration, dict):
        archive_name = archive_declaration.get("name")
        expected_archive_hash = archive_declaration.get("sha256")
    else:
        archive_name = archive_declaration
        expected_archive_hash = spec.get("source_archive_sha256")
    archive_path = contained_path(spec_path.parent, archive_name)
    actual_archive_hash = sha256_file(archive_path)
    entries, duplicate_entries = {}, []
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            if member.filename in entries:
                duplicate_entries.append(member.filename)
            entries[member.filename] = hashlib.sha256(archive.read(member)).hexdigest()
    expected_sources = spec["source_hashes"]
    extras = sorted(set(entries) - set(expected_sources))
    rows = []
    for name, expected in expected_sources.items():
        try:
            actual_current = sha256_file(contained_path(repo, name))
        except (OSError, ValueError):
            actual_current = None
        rows.append({"path": name, "expected_sha256": expected,
                     "current_sha256": actual_current, "archived_sha256": entries.get(name),
                     "current_matches": actual_current == expected,
                     "archived_matches": entries.get(name) == expected})
    current_matches = all(row["current_matches"] for row in rows)
    archive_matches = all(row["archived_matches"] for row in rows) and not extras and not duplicate_entries
    container_matches = expected_archive_hash is not None and actual_archive_hash == expected_archive_hash
    return {"passed": current_matches and archive_matches and container_matches,
            "registered_source_files": len(rows), "archive_entry_count": len(entries),
            "archive_path": str(archive_path), "expected_container_sha256": expected_archive_hash,
            "actual_container_sha256": actual_archive_hash, "container_hash_matches": container_matches,
            "current_source_matches": current_matches, "archive_entry_matches": archive_matches,
            "extra_archive_entries": extras, "duplicate_archive_entries": duplicate_entries, "files": rows}


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def check_raw_start_times(path, spec, spec_hash):
    """Metadata-only validation; does not evaluate outcomes or tune a method."""
    expected_count = len(spec["cases"]) * len(spec["policies"]) * spec["repetitions"]
    rows = checked = bad = 0
    minimum, maximum = None, None
    issues = []
    completed_scan = False
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                rows += 1
                try:
                    record = json.loads(line)
                    started = record["started_epoch"]
                    valid = (record.get("spec_sha256") == spec_hash
                             and record.get("phase") == spec["phase"]
                             and _finite_number(started) and started > spec["created_epoch"])
                    if _finite_number(started):
                        checked += 1
                        minimum = started if minimum is None else min(minimum, started)
                        maximum = started if maximum is None else max(maximum, started)
                    if not valid:
                        raise ValueError("started_epoch/spec/phase does not establish freeze before this trial")
                except (KeyError, TypeError, ValueError) as error:
                    bad += 1
                    if len(issues) < 1000:
                        issues.append({"line": line_number, "error": str(error)})
        completed_scan = True
    except (OSError, EOFError, UnicodeDecodeError) as error:
        issues.append({"line": None, "error": type(error).__name__ + ": " + str(error)})
    return {"performed": True, "passed": completed_scan and rows == expected_count and not bad,
            "complete_scan": completed_scan, "records_read": rows, "expected_records": expected_count,
            "finite_start_times_checked": checked, "bad_records": bad,
            "minimum_started_epoch": minimum, "maximum_started_epoch": maximum,
            "freeze_created_epoch": spec["created_epoch"], "issues": issues,
            "scope": "every retained trial start follows frozen specification creation; trial uniqueness and paid ledgers are checked by the separate full record audit"}


def run_integrity(spec_path, output_path, trials_path=None, status_path=None,
                  expected_spec_sha256=None, repo_root=REPO):
    spec_path, repo = Path(spec_path).resolve(), Path(repo_root).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    actual_spec_hash = sha256_file(spec_path)
    sidecar = spec_path.with_suffix(spec_path.suffix + ".sha256")
    sidecar_tokens = sidecar.read_text(encoding="ascii").split() if sidecar.is_file() else []
    sidecar_hash = sidecar_tokens[0] if sidecar_tokens else None
    spec_check = {"actual_sha256": actual_spec_hash, "sidecar_sha256": sidecar_hash,
                  "independently_supplied_sha256": expected_spec_sha256,
                  "passed": actual_spec_hash == sidecar_hash and
                      (expected_spec_sha256 is None or actual_spec_hash == expected_spec_sha256),
                  "created_epoch": spec["created_epoch"], "created_utc": spec.get("created_utc")}
    frozen = check_frozen_sources(spec, spec_path, repo)
    legacy = [check_legacy_manifest(repo, name) for name in LEGACY_MANIFESTS]
    runner_start = {"performed": False, "scope": "a runner-start check is additional metadata, not a replacement for every raw trial timestamp"}
    if status_path is not None:
        status = json.loads(Path(status_path).read_text(encoding="utf-8"))
        started = status.get("started_epoch")
        runner_start.update({"performed": True, "started_epoch": started,
                             "passed": _finite_number(started) and spec["created_epoch"] < started})
    static_passed = spec_check["passed"] and frozen["passed"] and all(row["passed"] for row in legacy)
    if runner_start["performed"]:
        static_passed = static_passed and runner_start["passed"]
    raw = (check_raw_start_times(Path(trials_path), spec, actual_spec_hash) if trials_path is not None
           else {"performed": False, "passed": None,
                 "reason": "deferred until the runner closes the full gzip; do not scan an active raw stream"})
    result = {"schema": SCHEMA, "static_passed": static_passed,
              "passed": static_passed and raw["passed"] if raw["performed"] else None,
              "status": "complete" if raw["performed"] else "static_checked_raw_timestamps_pending",
              "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "spec_path": str(spec_path), "specification": spec_check,
              "frozen_sources": frozen, "legacy_manifests": legacy,
              "runner_start": runner_start, "raw_timestamp_check": raw,
              "audit_supplement": "the frozen audit_records.py recognizes nested source_archive.sha256, but the formal spec stores source_archive as a string and source_archive_sha256 at top level; this separate check authenticates the top-level container hash in addition to all entry hashes",
              "scope": "delivery integrity only; no frozen source edits, no statistical policy changes, no new experimental observations or changed scientific conclusions",
              "checker_is_part_of_frozen_experiment": False,
              "checker_sha256": sha256_file(Path(__file__).resolve())}
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=Path)
    parser.add_argument("--status", type=Path)
    parser.add_argument("--expected-spec-sha256")
    args = parser.parse_args()
    result = run_integrity(args.spec, args.output, args.trials, args.status, args.expected_spec_sha256)
    print(json.dumps({"static_passed": result["static_passed"], "passed": result["passed"],
                      "status": result["status"], "frozen_source_files": result["frozen_sources"]["registered_source_files"],
                      "legacy_files": [row["files_checked"] for row in result["legacy_manifests"]]}, sort_keys=True))
    raise SystemExit(0 if result["static_passed"] and result["passed"] is not False else 1)


if __name__ == "__main__":
    main()
