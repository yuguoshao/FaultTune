"""Package the current structured study after its experiment/audit work stops."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import zipfile

from check_recorded_bindings import check as check_recorded_bindings


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT/"results/structured_validation"
MANIFEST = RESULTS/"evidence_manifest.json"
PREVIOUS = ROOT/"results/streaming_validation/evidence_manifest.json"
PREVIOUS_SHA = "3af4a45b0a3a7aa9d139f0efb0752651da1286635b54e7166f7e529f982343ed"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def checked(mapping, base=None):
    base = ROOT if base is None else base
    for name, expected in mapping.items():
        assert digest(base/name) == expected, name


def ledger_metadata(path):
    """Read an already stopped ledger, requiring complete newline records."""
    value, records, size = hashlib.sha256(), 0, 0
    with path.open("rb") as stream:
        for line in stream:
            assert line.endswith(b"\n") and line.strip(), "Incomplete/empty ledger line: " + str(path)
            value.update(line)
            records += 1
            size += len(line)
    assert size == path.stat().st_size, "Ledger changed during inspection: " + str(path)
    return {"records": records, "bytes": size, "sha256": value.hexdigest()}


def unchanged_prefix(original, combined, size):
    """Verify actual prefix bytes, independently of the completion wrapper."""
    assert original.stat().st_size == size
    with original.open("rb") as old_stream, combined.open("rb") as new_stream:
        remaining = size
        while remaining:
            block = old_stream.read(min(1024 * 1024, remaining))
            assert block and new_stream.read(len(block)) == block, "Original raw prefix differs"
            remaining -= len(block)
        assert old_stream.read(1) == b"", "Original raw prefix grew"


def utc_timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None, "Administrative times must be timezone-aware"
    return parsed.timestamp()


def completion_status():
    """Validate the separately named administrative completion, if present.

This runs only during the final freeze, after both producer and auditor stop.
It preserves the original interrupted study as a separate manifest entry.
"""
    directory = RESULTS / "prospective_completion_v1"
    status = {"present": directory.exists(), "complete": False,
              "audit_status": "missing", "primary_accepted": False}
    if not directory.exists():
        return status
    amendment_path = directory / "amendment.json"
    amendment = read(amendment_path)
    original = RESULTS / "prospective_v1"
    plan_path = RESULTS / "prospective_plan.json"
    plan = read(plan_path)
    assert digest(plan_path) == amendment["original_plan_sha256"]
    old_files = amendment["original_files_sha256"]
    for name in ("spec.json", "selection.json", "release.json", "progress.json", "trials.jsonl"):
        assert relative(original / name) in old_files, "Missing original binding: " + name
    checked(old_files)
    checked(amendment["algorithm_sources_sha256"])
    checked(amendment["original_wrapper_sources_sha256"])
    checked(amendment["completion_sources_sha256"])
    expected_wrappers = {"experiments/structured_validation/" + name for name in
                         ("completion_common.py", "complete_prospective.py", "audit_prospective_completion.py")}
    assert set(amendment["completion_sources_sha256"]) == expected_wrappers
    original_spec = read(original / "spec.json")
    selection_record = read(original / "selection.json")
    original_selection = selection_record["selection"]
    original_release = read(original / "release.json")
    original_progress = read(original / "progress.json")
    assert amendment["algorithm_sources_sha256"] == original_release["algorithm_sources_sha256"]
    assert amendment["original_wrapper_sources_sha256"] == original_release["wrapper_sources_sha256"]
    for name in ("spec.json", "selection.json", "release.json"):
        assert (directory / name).read_bytes() == (original / name).read_bytes(), "Changed copied " + name
    assert original_spec["plan_sha256"] == amendment["original_plan_sha256"]
    assert selection_record["plan_sha256"] == amendment["original_plan_sha256"]
    assert original_spec["selection"] == original_selection
    assert original_release["selection"] == original_selection
    assert original_spec["cases"] == plan["cases"]
    assert original_spec["release_sha256"] == digest(directory / "release.json")
    expected_records = plan["expected_joint_replicate_labels"] * len(original_selection["methods"])
    assert (amendment["original_records"], amendment["remaining_records"],
            amendment["expected_records"]) == (7937, 1663, 9600)
    assert amendment["original_records"] + amendment["remaining_records"] == expected_records == 9600
    assert original_progress["completed_method_trajectories"] == amendment["original_records"]
    assert original_progress["expected_method_trajectories"] == expected_records
    assert original_progress["flushed_ledger_bytes"] == amendment["original_ledger_bytes"]
    old_case_ids = amendment["original_complete_case_ids"]
    assert old_case_ids == original_progress["completed_case_ids"]
    assert old_case_ids == [case["case_id"] for case in plan["cases"][:len(old_case_ids)]]
    for case_id in old_case_ids:
        assert relative(original / "audit_cases" / ("case_%03d.json" % case_id)) in old_files
    execution = amendment["execution"]
    started = utc_timestamp(execution["started_utc"])
    cutoff = utc_timestamp(execution["cutoff_utc"])
    deadline = utc_timestamp(execution["hard_deadline_utc"])
    assert execution["max_minutes"] == 90
    assert started < cutoff <= deadline <= started + 90 * 60
    assert started == utc_timestamp("2026-11-04T08:22:02Z")
    assert deadline == utc_timestamp("2026-11-04T09:52:02Z")
    assert execution["cutoff_unix"] == cutoff and execution["hard_deadline_unix"] == deadline
    status.update(amendment_sha256=digest(amendment_path), execution=execution,
                  original_directory=relative(original), completion_directory=relative(directory),
                  original_method_trajectories=7937, authorized_remaining_method_trajectories=1663,
                  expected_method_trajectories=9600, original_complete_case_ids=old_case_ids,
                  copied_statistical_design_unchanged=True,
                  administrative_extension_only=True)
    run_path = directory / "completion_run.json"
    status["released"] = run_path.exists()
    if run_path.exists():
        run = read(run_path)
        assert run["amendment_sha256"] == digest(amendment_path)
        assert started <= utc_timestamp(run["created_utc"]) <= cutoff
        assert run["original_records"] == 7937 and run["remaining_records"] == 1663
        review_ref = run["wrapper_review"]
        review_path = ROOT / review_ref["path"]
        assert digest(review_path) == review_ref["sha256"]
        wrapper_review = read(review_path)
        assert wrapper_review["status"] == "passed"
        checked(wrapper_review["sources"])
        for name, expected in amendment["completion_sources_sha256"].items():
            assert wrapper_review["sources"].get(name) == expected
        assert wrapper_review["sources"].get(relative(amendment_path)) == digest(amendment_path)
        status["completion_run_sha256"] = digest(run_path)
        status["wrapper_review"] = review_ref
    progress_path = directory / "progress.json"
    progress = read(progress_path) if progress_path.exists() else {}
    assert progress.get("status") != "running", "Completion producer is still running"
    audit_progress_path = directory / "audit_progress.json"
    audit_progress = read(audit_progress_path) if audit_progress_path.exists() else {}
    assert audit_progress.get("status") != "running", "Completion auditor is still running"
    status.update(runner_status=progress.get("status", "missing"),
                  runner_error=progress.get("error"),
                  audit_progress_status=audit_progress.get("status", "missing"))
    ledger = directory / "trials.jsonl"
    metadata = None
    if ledger.exists():
        old_metadata = ledger_metadata(original / "trials.jsonl")
        assert old_metadata["records"] == 7937 and old_metadata["bytes"] == amendment["original_ledger_bytes"]
        assert old_metadata["sha256"] == old_files[relative(original / "trials.jsonl")]
        unchanged_prefix(original / "trials.jsonl", ledger, amendment["original_ledger_bytes"])
        metadata = ledger_metadata(ledger)
        assert 7937 <= metadata["records"] <= 9600
        if progress:
            assert progress["flushed_ledger_bytes"] == metadata["bytes"]
            assert progress["completed_method_trajectories"] == metadata["records"]
            assert progress["expected_method_trajectories"] == 9600
        status.update(completed_method_trajectories=metadata["records"],
                      newly_appended_method_trajectories=metadata["records"] - 7937,
                      ledger_bytes=metadata["bytes"], ledger_sha256=metadata["sha256"],
                      original_prefix_byte_identical=True)
        if audit_progress:
            audited_bytes = audit_progress["audited_ledger_bytes"]
            assert 0 <= audited_bytes <= metadata["bytes"]
            status.update(case_audits_completed=len(audit_progress["audited_case_ids"]),
                          audited_case_prefix_bytes=audited_bytes,
                          bytes_outside_completed_case_audits=metadata["bytes"] - audited_bytes)
    status["completed_case_ids"] = progress.get("completed_case_ids", [])
    summary_path, analysis_path, audit_path = (directory / name for name in
                                             ("summary.json", "analysis.json", "audit.json"))
    summary = read(summary_path) if summary_path.exists() else {}
    if summary:
        assert metadata is not None
        assert summary["spec"] == original_spec and summary["trials_sha256"] == metadata["sha256"]
        if summary.get("status") == "complete":
            assert progress.get("status") == "complete" and metadata["records"] == 9600
            assert summary["new_trajectories"] == 9600
            assert progress["completed_case_ids"] == [case["case_id"] for case in plan["cases"]]
            status["complete"] = True
    audit = read(audit_path) if audit_path.exists() else {}
    if audit:
        required_inputs = {"spec.json", "summary.json", "trials.jsonl", "analysis.json",
                           "selection.json", "release.json", "amendment.json", "completion_run.json",
                           "analysis_provenance.json"}
        assert required_inputs <= set(audit["input_sha256"])
        checked(audit["input_sha256"], directory)
        checked(audit["audit_sources_sha256"])
        for name, expected in amendment["completion_sources_sha256"].items():
            assert audit["audit_sources_sha256"].get(name) == expected
        if old_case_ids:
            old_audit_sources = read(original / "audit_cases" / ("case_%03d.json" % old_case_ids[0]))["audit_sources_sha256"]
            checked(old_audit_sources)
            for name, expected in old_audit_sources.items():
                assert audit["audit_sources_sha256"].get(name) == expected
        status["audit_status"] = audit["status"]
        if audit["status"] == "passed":
            assert status["complete"] and status["released"]
            assert audit["checks"]["terminal_records_checked"] == 9600
            assert audit["audited_cases"] == len(plan["cases"])
            assert audit["original_records"] == 7937 and audit["newly_generated_method_trajectories"] == 1663
            assert audit["reused_original_case_count"] == len(old_case_ids)
            assert audit["freshly_audited_case_count"] == len(plan["cases"]) - len(old_case_ids)
            status.update(audited_cases=audit["audited_cases"],
                          reused_original_case_count=audit["reused_original_case_count"],
                          freshly_audited_case_count=audit["freshly_audited_case_count"],
                          all_9600_method_trajectories_audited=True)
    if analysis_path.exists():
        analysis = read(analysis_path)
        assert {"spec.json", "summary.json", "trials.jsonl", "selection.json", "release.json"} <= set(analysis["input_sha256"])
        checked(analysis["input_sha256"], directory)
        provenance_path = directory / "analysis_provenance.json"
        provenance = read(provenance_path)
        analyzer = "experiments/structured_validation/prospective/analyze.py"
        assert provenance["sources"] == {analyzer: amendment["original_wrapper_sources_sha256"][analyzer]}
        checked(provenance["sources"])
        assert set(provenance["input_sha256"]) == {"amendment.json", "completion_run.json", "analysis.json"}
        checked(provenance["input_sha256"], directory)
        assert provenance["claim_boundary"] == analysis["claim_boundary"]
        assert status["complete"], "Analysis on an incomplete completion is not accepted"
        bounds = analysis["primary"]["percentile_95_ci"]
        assert len(bounds) == 2 and all(math.isfinite(value) for value in bounds)
        assert bounds[0] <= bounds[1]
        threshold_met = bounds[1] < 1
        assert threshold_met == analysis["primary_numerical_threshold_met"]
        if audit:
            assert audit["primary_pass"] == threshold_met
        status.update(primary_numerical_threshold_met=threshold_met,
                      primary_accepted=bool(status["released"] and status["complete"] and
                                            status["audit_status"] == "passed" and threshold_met),
                      claim_boundary=analysis["claim_boundary"])
    status["unfinished_primary_analysis"] = not analysis_path.exists()
    # Recheck immutable inputs after the potentially large prefix and ledger reads.
    checked(old_files)
    checked(amendment["algorithm_sources_sha256"])
    checked(amendment["original_wrapper_sources_sha256"])
    checked(amendment["completion_sources_sha256"])
    assert digest(amendment_path) == status["amendment_sha256"]
    if run_path.exists():
        assert digest(run_path) == status["completion_run_sha256"]
    return status


def verify():
    result = read(MANIFEST)
    checked(result["files"])
    assert digest(PREVIOUS) == PREVIOUS_SHA
    checked(read(PREVIOUS)["files"])
    learned = ROOT/"results/learned_validation/evidence_manifest.json"
    assert digest(learned) == result["previous_learned_manifest_sha256"]
    checked(read(learned)["files"])
    return {"status": "passed", "files": len(result["files"]),
            "manifest_sha256": digest(MANIFEST), "manifest": str(MANIFEST)}


def main():
    archive_path = RESULTS/"structured_validation_sources.zip"
    assert not MANIFEST.exists() and not archive_path.exists(), "Preserve existing frozen evidence; use --verify-only."
    recorded_bindings_audit = check_recorded_bindings()
    assert digest(PREVIOUS) == PREVIOUS_SHA
    checked(read(PREVIOUS)["files"])
    old = ROOT/"results/learned_validation/evidence_manifest.json"
    checked(read(old)["files"])
    studies = {}
    for spec_path in sorted(RESULTS.rglob("spec.json")):
        folder = spec_path.parent
        spec = read(spec_path)
        source_map = spec.get("sources", spec.get("algorithm_sources_sha256", {}))
        checked(source_map)
        summary_path, audit_path = folder/"summary.json", folder/"audit.json"
        summary = read(summary_path) if summary_path.exists() else {}
        audit = read(audit_path) if audit_path.exists() else {}
        for field in ("run_input_sha256", "artifact_sha256", "input_sha256"):
            if field in audit:
                checked(audit[field], folder)
        if spec.get("wrapper_sources_sha256"):
            checked(spec["wrapper_sources_sha256"])
        if audit.get("audit_sources_sha256"):
            for name, expected in audit["audit_sources_sha256"].items():
                path = ROOT/name if "/" in name else ROOT/"experiments/structured_validation"/name
                assert digest(path) == expected, str(path)
        studies[relative(folder)] = {
            "has_summary": bool(summary), "audit_status": audit.get("status", "missing"),
            "new_trajectories": summary.get("new_trajectories", summary.get("trial_count")),
            "spec_sha256": digest(spec_path),
            "summary_sha256": digest(summary_path) if summary_path.exists() else None,
            "audit_sha256": digest(audit_path) if audit_path.exists() else None}
    for review_name in ("theory_review.json", "rb_theory_review.json", "continuous_symmetric_design_root_review.json",
                        "continuous_psr_bound_root_review.json"):
        review = read(RESULTS/review_name)
        assert review["status"] == "passed"
        assert digest(ROOT/review["source"]["path"]) == review["source"]["sha256"]
    symmetric_review = read(RESULTS/"continuous_symmetric_design_review.json")
    assert symmetric_review["status"] == "passed"
    assert digest(ROOT/"docs/CONTINUOUS_SYMMETRIC_DESIGN.md") == symmetric_review["document_sha256"]
    psr_review = read(RESULTS/"continuous_psr_bound_sota_review.json")
    assert psr_review["status"] == "passed"
    checked(psr_review["sources"])
    checked(read(RESULTS/"psr_information_bound_table.json")["sources"])
    for name in ("continuous_fisher_checks.json", "continuous_fisher_independent_review.json",
                 "continuous_fisher_replay_independent_review.json",
                 "continuous_fisher_curvature_checks.json", "continuous_fisher_curvature_review.json",
                 "continuous_fisher_curvature_implementation_review.json"):
        review = read(RESULTS/name)
        assert review["status"] == "passed", name
        checked(review["sources"])
    prospective = RESULTS/"prospective_v1"
    prospective_summary = read(prospective/"summary.json") if (prospective/"summary.json").exists() else {}
    prospective_status = {"released": (prospective/"release.json").exists(),
                          "complete": prospective_summary.get("status") == "complete",
                          "audit_status": "missing", "primary_accepted": False}
    if (prospective/"progress.json").exists():
        progress = read(prospective/"progress.json")
        assert progress["status"] != "running", "Do not freeze an experiment that is still running."
        ledger_bytes = (prospective/"trials.jsonl").stat().st_size
        assert progress["flushed_ledger_bytes"] == ledger_bytes
        case_audits = [read(path) for path in sorted((prospective/"audit_cases").glob("case_*.json"))]
        audited_bytes = case_audits[-1]["end_byte"] if case_audits else 0
        assert 0 <= audited_bytes <= ledger_bytes
        prospective_status.update(
            runner_status=progress["status"], runner_error=progress.get("error"),
            completed_method_trajectories=progress["completed_method_trajectories"],
            expected_method_trajectories=progress["expected_method_trajectories"],
            completed_case_ids=progress["completed_case_ids"],
            case_audits_completed=len(case_audits),
            ledger_bytes=ledger_bytes, audited_case_prefix_bytes=audited_bytes,
            bytes_outside_completed_case_audits=ledger_bytes-audited_bytes,
            unfinished_primary_analysis=not (prospective/"analysis.json").exists())
    if prospective_status["released"]:
        release = read(prospective/"release.json")
        assert digest(RESULTS/"prospective_plan.json") == release["plan_sha256"]
        assert digest(prospective/"selection.json") == release["selection_sha256"]
        checked(release["wrapper_sources_sha256"])
        checked(release["algorithm_sources_sha256"])
    if (prospective/"audit.json").exists():
        prospective_audit = read(prospective/"audit.json")
        checked(prospective_audit["input_sha256"], prospective)
        prospective_status["audit_status"] = prospective_audit["status"]
    if (prospective/"analysis.json").exists():
        analysis = read(prospective/"analysis.json")
        checked(analysis["input_sha256"], prospective)
        threshold_met = analysis["primary"]["percentile_95_ci"][1] < 1
        assert threshold_met == analysis["primary_numerical_threshold_met"]
        if (prospective/"audit.json").exists():
            assert prospective_audit["primary_pass"] == threshold_met
        prospective_status["primary_numerical_threshold_met"] = threshold_met
        prospective_status["primary_accepted"] = bool(prospective_status["released"] and prospective_status["complete"] and
            prospective_status["audit_status"] == "passed" and analysis["primary_numerical_threshold_met"])
        prospective_status["claim_boundary"] = analysis["claim_boundary"]
    prospective_completion_status = completion_status()
    current_prospective_status = (prospective_completion_status
                                  if prospective_completion_status["present"] else prospective_status)
    report = ROOT/"docs/STRUCTURED_VALIDATION_RESEARCH.md"
    assert report.exists(), "Write the factual final report before freezing."
    sources = sorted((ROOT/"experiments/structured_validation").rglob("*.py"))
    dependencies = sorted((ROOT/"experiments/circuit_validation").glob("*.py"))
    documents = [report, ROOT/"docs/STRUCTURED_EVIDENCE_THEORY.md",
                 ROOT/"docs/RAO_BLACKWELL_BLOCK_EVIDENCE.md",
                 ROOT/"docs/CONTINUOUS_SYMMETRIC_DESIGN.md",
                 ROOT/"docs/CONTINUOUS_PSR_INFORMATION_BOUND.md",
                 ROOT/"docs/CONTINUOUS_FISHER_EXPLORATION.md",
                 ROOT/"docs/CONTINUOUS_FISHER_CURVATURE_BOUND.md"]
    bundle_files = sorted(set(sources+dependencies+documents+[
        ROOT/"results/learned_validation/development_v1/spec.json"]))
    with zipfile.ZipFile(str(archive_path), "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in bundle_files:
            archive.write(str(path), relative(path))
    with zipfile.ZipFile(str(archive_path)) as archive:
        for path in bundle_files:
            assert hashlib.sha256(archive.read(relative(path))).hexdigest() == digest(path)
    assert recorded_bindings_audit == check_recorded_bindings(), "Recorded provenance changed while packaging"
    artifacts = [p for p in RESULTS.rglob("*") if p.is_file() and p != MANIFEST
                 and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    files = sorted(set(bundle_files+artifacts))
    result = {
        "status": "Evidence snapshot; consult report and individual audits for completion and superiority",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_control": read(RESULTS/"run_control.json"),
        "previous_streaming_manifest_sha256": PREVIOUS_SHA,
        "previous_streaming_files_verified": len(read(PREVIOUS)["files"]),
        "previous_learned_manifest_sha256": digest(old),
        "previous_learned_files_verified": len(read(old)["files"]),
        "previous_distinct_files_verified": len(set(read(PREVIOUS)["files"]) | set(read(old)["files"])),
        "prospective_status": prospective_status,
        "prospective_completion_status": prospective_completion_status,
        "current_prospective_result": ("results/structured_validation/prospective_completion_v1"
                                       if prospective_completion_status["present"]
                                       else "results/structured_validation/prospective_v1"),
        "primary_accepted": current_prospective_status["primary_accepted"],
        "recorded_bindings_audit": recorded_bindings_audit,
        "studies": studies, "source_archive_files": len(bundle_files),
        "files": {relative(path): digest(path) for path in files}}
    with MANIFEST.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2))
    print(json.dumps(verify(), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify(), indent=2))
    else:
        main()
