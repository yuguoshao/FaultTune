"""Reproducible administrative fixtures for the final evidence freezer.

All apparent ledgers are tiny temporary sequences of empty JSON objects; they
contain no simulator observations. The real original/completion ledgers are
never opened. The only call to main() is deliberately stopped by its existing-
manifest guard before any freezing operation. Temporary cleanup is restricted
to the explicitly checked temporary child directory.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import time


ROOT = Path(__file__).resolve().parents[2]
FREEZER = ROOT / "experiments/structured_validation/freeze_evidence.py"
OUTPUT = ROOT / "results/structured_validation/freeze_completion_checks.json"
PINNED_FREEZER = "a5d6b8823282731e01d90f52a74b066773f44719b7aae2a3e566ac70b85339a4"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_fixture(freezer, root, mode):
    result = root / "results/structured_validation"
    old, new = result / "prospective_v1", result / "prospective_completion_v1"
    old.mkdir(parents=True)
    new.mkdir()

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")

    def relative(path):
        return path.relative_to(root).as_posix()

    def code(name):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("administrative test fixture only", encoding="utf-8")
        return name, digest(path)

    algorithm = dict([code("experiments/structured_validation/fixture_algorithm.py")])
    wrappers = dict(code("experiments/structured_validation/prospective/" + name)
                    for name in ("run.py", "common.py", "analyze.py"))
    auditors = dict([code("experiments/structured_validation/prospective/audit.py")])
    completion = dict(code("experiments/structured_validation/" + name)
                      for name in ("completion_common.py", "complete_prospective.py", "audit_prospective_completion.py"))
    cases, complete_ids = [{"case_id": i} for i in range(56)], list(range(41))
    selected = {"methods": ["closed_hedge", "closed_rb", "psr_beta"]}
    write(result / "prospective_plan.json", {"cases": cases, "expected_joint_replicate_labels": 3200})
    plan_hash = digest(result / "prospective_plan.json")
    write(old / "selection.json", {"plan_sha256": plan_hash, "selection": selected})
    write(old / "release.json", {"plan_sha256": plan_hash, "selection": selected,
                                 "algorithm_sources_sha256": algorithm, "wrapper_sources_sha256": wrappers})
    spec = {"plan_sha256": plan_hash, "selection": selected, "cases": cases,
            "release_sha256": digest(old / "release.json")}
    write(old / "spec.json", spec)
    raw = b"{}\n" * 7937
    (old / "trials.jsonl").write_bytes(raw)
    write(old / "progress.json", {"status": "incomplete_cutoff", "completed_method_trajectories": 7937,
                                  "expected_method_trajectories": 9600, "flushed_ledger_bytes": len(raw),
                                  "completed_case_ids": complete_ids})
    for case_id in complete_ids:
        write(old / "audit_cases" / ("case_%03d.json" % case_id), {"audit_sources_sha256": auditors})
    old_files = {relative(path): digest(path) for path in old.rglob("*") if path.is_file()}
    execution = {"started_utc": "2026-11-04T08:22:02Z", "cutoff_utc": "2026-11-04T09:32:02Z",
                 "cutoff_unix": freezer.utc_timestamp("2026-11-04T09:32:02Z"),
                 "hard_deadline_utc": "2026-11-04T09:52:02Z",
                 "hard_deadline_unix": freezer.utc_timestamp("2026-11-04T09:52:02Z"), "max_minutes": 90}
    amendment = {"original_plan_sha256": plan_hash, "original_files_sha256": old_files,
                 "algorithm_sources_sha256": algorithm, "original_wrapper_sources_sha256": wrappers,
                 "completion_sources_sha256": completion, "original_records": 7937,
                 "remaining_records": 1663, "expected_records": 9600, "original_ledger_bytes": len(raw),
                 "original_complete_case_ids": complete_ids, "execution": execution}
    if mode == "changed_sample_size":
        amendment["expected_records"] = 9601
    if mode == "window_overrun":
        execution["hard_deadline_utc"] = "2026-11-04T09:53:02Z"
        execution["hard_deadline_unix"] = freezer.utc_timestamp(execution["hard_deadline_utc"])
    write(new / "amendment.json", amendment)
    reviewed = dict(completion)
    reviewed[relative(new / "amendment.json")] = digest(new / "amendment.json")
    if mode == "review_missing_wrapper":
        reviewed.pop(next(iter(completion)))
    review_path = result / "completion_wrapper_review.json"
    write(review_path, {"status": "passed", "sources": reviewed})
    write(new / "completion_run.json", {
        "amendment_sha256": digest(new / "amendment.json"), "created_utc": "2026-11-04T08:42:00Z",
        "original_records": 7937, "remaining_records": 1663,
        "wrapper_review": {"path": relative(review_path), "sha256": digest(review_path)}})
    for name in ("spec.json", "selection.json", "release.json"):
        (new / name).write_bytes((old / name).read_bytes())
    if mode == "changed_copy":
        write(new / "selection.json", {"changed": True})
    count = 7940 if mode == "partial" else 9600
    combined = b"{}\n" * count
    if mode == "changed_prefix":
        combined = b"[]\n" + combined[3:]
    (new / "trials.jsonl").write_bytes(combined)
    progress = {"status": "incomplete_cutoff" if mode == "partial" else "running" if mode == "running" else "complete",
                "flushed_ledger_bytes": len(combined), "completed_method_trajectories": count,
                "expected_method_trajectories": 9600,
                "completed_case_ids": complete_ids if mode == "partial" else list(range(56))}
    if mode == "wrong_progress_count":
        progress["completed_method_trajectories"] -= 1
    write(new / "progress.json", progress)
    write(new / "audit_progress.json", {
        "status": "incomplete_deadline" if mode == "partial" else "complete",
        "audited_ledger_bytes": len(raw) if mode == "partial" else len(combined),
        "audited_case_ids": complete_ids if mode == "partial" else list(range(56))})
    if mode == "partial":
        return
    write(new / "summary.json", {"status": "complete", "spec": spec,
                                 "trials_sha256": digest(new / "trials.jsonl"), "new_trajectories": 9600})
    met = mode != "numerical_not_met"
    write(new / "analysis.json", {
        "input_sha256": {name: digest(new / name) for name in
                         ("spec.json", "summary.json", "trials.jsonl", "selection.json", "release.json")},
        "primary": {"percentile_95_ci": [.9, .99] if met else [.9, 1.01]},
        "primary_numerical_threshold_met": met, "claim_boundary": "fixture only"})
    if mode != "missing_provenance":
        analyzer = "experiments/structured_validation/prospective/analyze.py"
        write(new / "analysis_provenance.json", {
            "sources": {analyzer: wrappers[analyzer]},
            "input_sha256": {name: digest(new / name) for name in
                             ("amendment.json", "completion_run.json", "analysis.json")},
            "claim_boundary": "fixture only"})
    if mode in ("missing_audit", "missing_provenance"):
        return
    inputs = {name: digest(new / name) for name in
              ("spec.json", "summary.json", "trials.jsonl", "analysis.json", "selection.json", "release.json",
               "amendment.json", "completion_run.json", "analysis_provenance.json")}
    if mode == "audit_missing_input":
        inputs.pop("amendment.json")
    audit = {"status": "passed", "input_sha256": inputs,
             "audit_sources_sha256": dict(auditors, **completion),
             "checks": {"terminal_records_checked": 9600}, "audited_cases": 56,
             "original_records": 7937, "newly_generated_method_trajectories": 1663,
             "reused_original_case_count": 41, "freshly_audited_case_count": 15, "primary_pass": met}
    if mode == "wrong_audit_count":
        audit["checks"]["terminal_records_checked"] = 9599
    write(new / "audit.json", audit)


def run_checks():
    started = time.perf_counter()
    assert digest(FREEZER) == PINNED_FREEZER, "Use a new checks version if the reviewed freezer changes"
    source_files = (FREEZER, Path(__file__).resolve(), FREEZER.with_name("check_recorded_bindings.py"))
    sources = {path.relative_to(ROOT).as_posix(): digest(path) for path in source_files}
    spec = importlib.util.spec_from_file_location("freezer_completion_fixture_only", str(FREEZER))
    freezer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(freezer)
    real_globals = freezer.ROOT, freezer.RESULTS, freezer.MANIFEST
    checks = []
    try:
        with tempfile.TemporaryDirectory(prefix="faulttune_freezer_fixture_") as temporary:
            temporary = Path(temporary).resolve()
            assert temporary.parent == Path(tempfile.gettempdir()).resolve()
            assert temporary.name.startswith("faulttune_freezer_fixture_")
            freezer.ROOT = temporary / "missing"
            freezer.RESULTS = freezer.ROOT / "results/structured_validation"
            assert freezer.completion_status() == {"present": False, "complete": False,
                                                   "audit_status": "missing", "primary_accepted": False}
            checks.append("absent completion unaccepted")
            valid = ("complete", "numerical_not_met", "missing_audit", "partial")
            invalid = ("changed_prefix", "changed_copy", "changed_sample_size", "window_overrun", "running",
                       "wrong_progress_count", "audit_missing_input", "wrong_audit_count", "missing_provenance",
                       "review_missing_wrapper")
            for mode in valid + invalid:
                root = temporary / mode
                build_fixture(freezer, root, mode)
                freezer.ROOT, freezer.RESULTS = root, root / "results/structured_validation"
                try:
                    status = freezer.completion_status()
                except (AssertionError, FileNotFoundError):
                    assert mode in invalid, mode
                    checks.append(mode + " rejected")
                else:
                    assert mode in valid and status["primary_accepted"] == (mode == "complete"), mode
                    if mode == "partial":
                        assert status["completed_method_trajectories"] == 7940 and not status["complete"]
                    if mode == "missing_audit":
                        assert status["complete"] and status["audit_status"] == "missing"
                    checks.append(mode + " classified correctly")
            freezer.ROOT = temporary / "exclusive"
            freezer.RESULTS = freezer.ROOT / "results/structured_validation"
            freezer.RESULTS.mkdir(parents=True)
            freezer.MANIFEST = freezer.RESULTS / "evidence_manifest.json"
            freezer.MANIFEST.write_text("preserve fixture", encoding="utf-8")
            try:
                freezer.main()
            except AssertionError:
                pass
            else:
                raise AssertionError("Existing manifest was not protected")
            assert freezer.MANIFEST.read_text(encoding="utf-8") == "preserve fixture"
            checks.append("existing manifest preserved")
            assert temporary.parent == Path(tempfile.gettempdir()).resolve()
            assert temporary.name.startswith("faulttune_freezer_fixture_")
    finally:
        freezer.ROOT, freezer.RESULTS, freezer.MANIFEST = real_globals
    for name, expected in sources.items():
        assert digest(ROOT / name) == expected, name
    assert len(checks) == 16
    return {"status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(), "sources": sources,
            "scope": "Isolated temporary administrative fixtures, not study observations or numerical evidence tests",
            "checks": checks, "checks_count": len(checks),
            "real_study_files_modified": False, "real_ledgers_read": False,
            "freeze_executed": False, "sources_unchanged": True,
            "elapsed_seconds": time.perf_counter() - started}


if __name__ == "__main__":
    assert not OUTPUT.exists(), "Preserve the existing checks artifact"
    report = run_checks()
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({"status": report["status"], "checks_count": report["checks_count"],
                      "elapsed_seconds": report["elapsed_seconds"], "output": str(OUTPUT)}, indent=2))
