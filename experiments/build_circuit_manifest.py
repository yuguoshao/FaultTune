"""Hash and verify the first circuit study's deliverables after closeout.

This is delivery tooling outside the frozen scientific experiment. Mutable
run_state.json and this manifest's own hash sidecar are explicitly excluded.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
BASE = REPO / "experiments" / "circuit_validation"
RESULTS = BASE / "results"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            value.update(chunk)
    return value.hexdigest()


def check(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    issues = []
    for row in manifest["files"]:
        file = REPO / row["path"]
        if not file.is_file() or file.stat().st_size != row["bytes"] or digest(file) != row["sha256"]:
            issues.append(row["path"])
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii").split()[0] != digest(path):
        issues.append("manifest sidecar")
    return {"passed": not issues, "files_checked": len(manifest["files"]), "issues": issues}


def build(path):
    summary_path = RESULTS / "confirmation_v1_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary["integrity"]["pass"] or not summary["integrity"]["complete_grid_pass"]:
        raise ValueError("formal raw summary is not complete and valid")
    required_audits = ["confirmation_v1_records_audit.json", "confirmation_v1_replay_audit.json",
                       "delivery_integrity.json"]
    audit_results = {}
    for name in required_audits:
        value = json.loads((RESULTS / name).read_text(encoding="utf-8"))
        if value.get("passed") is not True:
            raise ValueError("required audit has not passed: " + name)
        audit_results[name] = True
    independent_path = REPO / "results" / "confirmation_summary_verification.json"
    independent = json.loads(independent_path.read_text(encoding="utf-8"))
    if independent.get("passed") is not True or independent.get("raw_records_independently_consumed") != 704000:
        raise ValueError("independent reconstruction of all raw records has not passed")
    audit_results["results/confirmation_summary_verification.json"] = True
    for name in ("CIRCUIT_VALIDATION_RESEARCH.md", "CIRCUIT_COMPLETION_AUDIT.md"):
        if not (REPO / "docs" / name).is_file():
            raise ValueError("required final document is missing: " + name)
    files = set((REPO / "docs").glob("CIRCUIT_*.md"))
    files.add(REPO / "docs" / "VALIDATION_ACCEPTANCE_CRITERIA.md")
    files.add(REPO / "experiments" / "local_validation" / "results" / "manifest.json")
    files.add(REPO / "experiments" / "local_validation" / "results" / "aggregate_manifest.json")
    for folder in (BASE, RESULTS):
        files.update(p for p in folder.iterdir() if p.is_file())
    extras = [REPO / "experiments" / "variance_diagnostic.py",
              REPO / "results" / "variance_diagnostic.json",
              REPO / "experiments" / "verify_raw_summary.py",
              REPO / "experiments" / "verify_zero_mcnemar.py",
              REPO / "results" / "confirmation_raw_aggregate.json",
              REPO / "results" / "zero_mcnemar_diagnostic.json",
              independent_path,
              REPO / "experiments" / "closeout_run.py", Path(__file__).resolve()]
    files.update(p for p in extras if p.is_file())
    # Optional diagnostic files are kept in the delivery even though they
    # cannot change the frozen confirmatory decision or error allocation.
    files.update((REPO / "experiments").glob("*null*diagnostic*.py"))
    files.update((REPO / "results").glob("*null*diagnostic*.json"))
    excluded = {path.resolve(), path.with_suffix(path.suffix + ".sha256").resolve(),
                (BASE / "run_state.json").resolve()}
    files = sorted(p for p in files if p.resolve() not in excluded and not p.name.endswith(".tmp"))
    rows = [{"path": p.relative_to(REPO).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)}
            for p in files]
    manifest = {"schema": "faulttune.circuit_delivery_manifest.v1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "first three-qubit circuit study; negative performance outcomes do not invalidate a completed research delivery",
                "confirmation_spec_sha256": summary["provenance"]["spec_sha256"],
                "planned_records": summary["integrity"]["planned_records"],
                "actual_records": summary["integrity"]["raw_records"],
                "challenge_passed": summary["overall"]["criteria_pass"],
                "required_audits": audit_results,
                "excluded_mutable_coordination_file": "experiments/circuit_validation/run_state.json",
                "self_hash": "separate SHA256 sidecar; neither file recursively hashes itself",
                "files": rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    path.with_suffix(path.suffix + ".sha256").write_text(digest(path) + "  " + path.name + "\n", encoding="ascii")
    return check(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS / "delivery_manifest.json")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = check(args.output) if args.verify else build(args.output)
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
