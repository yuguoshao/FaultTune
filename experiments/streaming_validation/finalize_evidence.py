"""Freeze this study after audits, retaining previous evidence byte for byte."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "streaming_validation"
MANIFEST = RESULTS / "evidence_manifest.json"
PLAN_SHA256 = "c54cb0776a16159b6c4f8a4ef8a2563c6d01a73e31cf40b6a3a1a572cb6a0587"
PREVIOUS_SHA256 = "5e0633642dfbd38be219a9a71df780d3bd8cd354557320d734da44c95b0f3abd"
RUNS = {"development_v1": 82944, "development_v2": 36864,
        "trackstop_v1": 9216, "holdout_v1": 28672}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def check_hashes(mapping, base=ROOT):
    for name, expected in mapping.items():
        assert digest(base / name) == expected, name
    return len(mapping)


def verify_manifest():
    manifest = read(MANIFEST)
    checked = check_hashes(manifest["files"])
    assert digest(ROOT / manifest["previous_evidence"]["manifest"]) == PREVIOUS_SHA256
    check_hashes(read(ROOT / manifest["previous_evidence"]["manifest"])["files"])
    assert digest(ROOT / manifest["previous_evidence"]["frozen_archive"]) == (
        manifest["previous_evidence"]["frozen_archive_sha256"])
    assert manifest["new_simulated_protocol_trajectories"] == sum(RUNS.values()) == 157696
    return {"status": "passed", "manifest_files_verified": checked,
            "new_trajectories": sum(RUNS.values()), "manifest": str(MANIFEST),
            "manifest_sha256": digest(MANIFEST)}


def main():
    prior_path = ROOT / "results" / "learned_validation" / "evidence_manifest.json"
    assert digest(prior_path) == PREVIOUS_SHA256
    prior = read(prior_path)
    assert check_hashes(prior["files"]) == 39
    old_archive = ROOT / "experiments" / "circuit_validation" / "confirmation_spec_v1_sources.zip"
    old_sources = []
    with zipfile.ZipFile(str(old_archive)) as archive:
        for name in archive.namelist():
            if not name.endswith(".py"):
                continue
            path = old_archive.parent / name
            if not path.is_file():
                path = ROOT / name
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest(path), name
            old_sources.append(path)
    assert len(old_sources) == 31
    plan_path = RESULTS / "holdout_plan.json"
    assert digest(plan_path) == PLAN_SHA256
    plan = read(plan_path)
    check_hashes(plan["sources"])
    gate = read(RESULTS / "holdout_gate.json")
    assert gate["status"] == "passed" and gate["root_released"] is True
    for kind in ("summary", "comparison"):
        assert digest(gate["development_" + kind + "_path"]) == gate["development_" + kind + "_sha256"]

    audits = {}
    for name, expected_count in RUNS.items():
        directory = RESULTS / name
        spec, summary, audit = [read(directory / (key + ".json"))
                                for key in ("spec", "summary", "audit")]
        assert audit["status"] == "passed" and audit["ledger_records"] == expected_count
        check_hashes(spec["sources"])
        check_hashes(audit["artifact_sha256"], directory)
        audit_source = audit.get("audit_source_path",
                                 "experiments/streaming_validation/audit_holdout.py")
        assert digest(ROOT / audit_source) == audit["audit_source_sha256"]
        for helper_key in ("audit_helper_source_sha256", "helper_source_sha256"):
            if helper_key in audit:
                assert digest(ROOT / "experiments/streaming_validation/audit_streaming.py") == audit[helper_key]
        if name != "development_v2":
            assert summary["trial_count"] == expected_count
        else:
            assert audit["referenced_baseline_trajectories"] == 18432
        audits[name] = {"new_records": expected_count, "audit_sha256": digest(directory / "audit.json"),
                        "audit_source": audit_source, "audit_source_sha256": audit["audit_source_sha256"]}

    analysis = read(RESULTS / "holdout_v1" / "analysis.json")
    assert analysis["status"] == "completed" and analysis["plan_sha256"] == PLAN_SHA256
    assert analysis["new_trajectories"] == 28672
    assert analysis["analysis"]["bootstrap_repetitions"] == 5000
    assert analysis["analysis"]["bootstrap_seed"] == 910733
    assert digest(ROOT / "experiments/streaming_validation/analyze_holdout.py") == analysis["analysis_source_sha256"]
    check_hashes(analysis["artifact_sha256"], RESULTS / "holdout_v1")
    assert digest(RESULTS / "holdout_v1/bootstrap_replicates.json") == analysis["bootstrap_replicates_sha256"]
    for review_name in ("theory_review.json", "learned_asymptotic_review.json"):
        review = read(RESULTS / review_name)
        assert review["status"] == "passed"
        assert digest(ROOT / review["source"]["path"]) == review["source"]["sha256"]
    planning = read(RESULTS / "planning_comparison.json")
    check_hashes(planning["source_sha256"], ROOT / "experiments/streaming_validation")
    check_hashes(planning["input_sha256"], RESULTS / "development_v1")
    plot = read(RESULTS / "holdout_plot_manifest.json")
    assert plot["analysis_sha256"] == digest(RESULTS / "holdout_v1/analysis.json")
    assert plot["source_sha256"] == digest(ROOT / "experiments/streaming_validation/plot_holdout.py")
    assert plot["image_sha256"] == digest(RESULTS / "holdout_comparison.png")

    documents = sorted((ROOT / "docs").glob("STREAMING_*.md"))
    assert (ROOT / "docs/STREAMING_VALIDATION_RESEARCH.md") in documents
    sources = sorted((ROOT / "experiments/streaming_validation").glob("*.py"))
    # Include all local study dependencies and the historical public case definition.
    sources += sorted((ROOT / "experiments/learned_validation").glob("*.py"))
    sources += sorted((ROOT / "experiments/circuit_validation").glob("*.py"))
    source_files = sorted(set(sources + documents + [
        ROOT / "results/learned_validation/development_v1/spec.json"]))
    archive_path = RESULTS / "streaming_validation_sources.zip"
    with zipfile.ZipFile(str(archive_path), "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in source_files:
            archive.write(str(path), relative(path))
    with zipfile.ZipFile(str(archive_path)) as archive:
        for path in source_files:
            assert hashlib.sha256(archive.read(relative(path))).hexdigest() == digest(path)

    artifacts = sorted(path for path in RESULTS.rglob("*")
                       if path.is_file() and path != MANIFEST
                       and "__pycache__" not in path.parts and path.suffix != ".pyc")
    files = sorted(set(source_files + artifacts))
    manifest = {
        "status": "audited prospective simulation evidence; not complete SOTA validation",
        "client_date": "2026-11-03",
        "new_simulated_protocol_trajectories": sum(RUNS.values()),
        "development_and_trackstop_new_trajectories": 129024,
        "prospective_holdout_new_trajectories": 28672,
        "referenced_v1_controls_in_guarded_v2_not_counted_as_new": 18432,
        "holdout_primary_outcome": analysis["primary_outcome"],
        "holdout_primary_comparison": analysis["comparisons"]["psr_beta"],
        "audits": audits,
        "previous_evidence": {
            "manifest": relative(prior_path), "manifest_sha256": PREVIOUS_SHA256,
            "unchanged_manifest_files": 39, "unchanged_frozen_python_sources": 31,
            "frozen_archive": relative(old_archive), "frozen_archive_sha256": digest(old_archive)},
        "prospective_plan_sha256": PLAN_SHA256,
        "source_archive_files": len(source_files),
        "files": {relative(path): digest(path) for path in files}}
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(verify_manifest(), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_manifest(), indent=2))
    else:
        main()
