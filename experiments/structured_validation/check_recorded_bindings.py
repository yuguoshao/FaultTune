"""Read-only verification of source hashes already recorded by this study.

This supplements freeze_evidence's dataset input checks. It never opens any
prospective_v1 file or any trial ledger. A known superseded audit check remains
historical evidence; it is explicitly not validation of the current auditor.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/structured_validation"
SOURCE_DIR = ROOT / "experiments/structured_validation"
SHA256 = re.compile(r"^[0-9a-f]{64}$")

# Each entry names the recorded field, not a fresh hash of the current source.
REQUIRED = {
    "certified_likelihood_checks.json": ("sources",),
    "conditional_fisher_checks.json": ("sources",),
    "conditional_fisher_unequal_checks.json": ("sources",),
    "continuous_psr_bound_root_review.json": ("source",),
    "continuous_psr_bound_sota_review.json": ("sources",),
    "continuous_symmetric_design_literature_review.json": ("local_proof", "primary_sources"),
    "continuous_symmetric_design_peer_review.json": ("document_sha256",),
    "continuous_symmetric_design_review.json": ("document_sha256",),
    "continuous_symmetric_design_root_review.json": ("source",),
    "design_checks.json": ("sources",),
    "fast_rb_checks.json": ("source_sha256",),
    "incremental_audit_checks.json": ("audit_source_sha256", "test_source_sha256"),
    "incremental_unequal_audit_review.json": ("source", "boundary_checks", "smoke_equivalence"),
    "predictive_checks.json": ("source_sha256",),
    "prospective_audit_checks.json": ("plan_sha256", "audit_sources_sha256"),
    "prospective_audit_checks_v2.json": ("plan_sha256", "audit_sources_sha256", "check_source_sha256"),
    "prospective_audit_review.json": ("source_sha256", "smoke_checks_sha256"),
    "prospective_case_integrity_review.json": ("plan_sha256", "prior_specs"),
    "prospective_plan.json": ("algorithm_sources_sha256", "run_control_sha256", "selection"),
    "prospective_wrapper_checks.json": ("plan_sha256", "wrapper_sources_sha256"),
    "prospective_wrapper_review.json": ("plan_sha256", "wrapper_sources_sha256"),
    "psr_information_bound_table.json": ("sources",),
    "rb_grid_checks.json": ("source_sha256",),
    "rb_theory_review.json": ("source",),
    "safe_grid_checks.json": ("source_sha256", "checks_source_sha256", "jeffreys71_source_sha256"),
    "theory_review.json": ("source", "implementation_checks"),
    "unequal_runner_checks.json": ("sources",),
    "weighted_fisher_checks.json": ("sources",),
}

SCALAR_TARGETS = {
    ("fast_rb_checks.json", "source_sha256"): "experiments/structured_validation/fast_rb_certificate.py",
    ("predictive_checks.json", "source_sha256"): "experiments/structured_validation/predictive.py",
    ("safe_grid_checks.json", "source_sha256"): "experiments/structured_validation/safe_grid.py",
    ("safe_grid_checks.json", "checks_source_sha256"): "experiments/structured_validation/safe_grid_checks.py",
    ("safe_grid_checks.json", "jeffreys71_source_sha256"): "experiments/structured_validation/expert_config.py",
    ("incremental_audit_checks.json", "audit_source_sha256"): "experiments/structured_validation/audit_unequal_incremental.py",
    ("incremental_audit_checks.json", "test_source_sha256"): "experiments/structured_validation/incremental_audit_checks.py",
    ("prospective_audit_checks_v2.json", "check_source_sha256"): "experiments/structured_validation/prospective/audit_checks.py",
    ("prospective_audit_review.json", "source_sha256"): "experiments/structured_validation/prospective/audit.py",
    ("theory_review.json", "implementation_checks.safe_grid_sha256"): "experiments/structured_validation/safe_grid.py",
    ("theory_review.json", "implementation_checks.safe_grid_checks_source_sha256"): "experiments/structured_validation/safe_grid_checks.py",
    ("theory_review.json", "implementation_checks.conditional_fisher_v1_source_sha256"): "experiments/structured_validation/conditional_fisher.py",
    ("incremental_unequal_audit_review.json", "smoke_equivalence.original_audit_sha256"): "results/structured_validation/unequal_smoke_v1/audit.json",
    ("incremental_unequal_audit_review.json", "smoke_equivalence.incremental_audit_sha256"): "results/structured_validation/unequal_smoke_v1/incremental_audit.json",
}

AUDIT_TARGETS = {
    "fixed_psr_probe/audit.json": "audit_fixed_psr_probe.py",
    "smoke_v1/audit.json": "audit_conditional.py",
    "development_v1/audit.json": "audit_conditional.py",
    "unequal_smoke_v1/audit.json": "audit_unequal.py",
    "unequal_smoke_v1/incremental_audit.json": "audit_unequal_incremental.py",
    "unequal_development_v1/audit.json": "audit_unequal_incremental.py",
}

SOURCE_MAPS = {"sources", "source_sha256", "audit_sources_sha256",
               "wrapper_sources_sha256", "algorithm_sources_sha256",
               "source_artifacts_sha256"}
DATA_HASH_FIELDS = {"input_sha256", "run_input_sha256", "artifact_sha256",
                    "records_sha256", "audited_ledger_bytes_sha256",
                    "original_ledger_bytes_sha256", "jeffreys71_config_sha256"}
HISTORICAL_ARTIFACT = "prospective_audit_checks.json"
HISTORICAL_SOURCE = "experiments/structured_validation/prospective/audit.py"
HISTORICAL_SHA = "dcd084140936f5d2ca129000f56a0887926128ff10e990d015ba269a29b2c523"
CURRENT_REVIEW_SHA = "da2be2884828d81919530ce3997482ba387ea535ded66c3aa1cc461664dcba10"


class BindingError(RuntimeError):
    def __init__(self, report):
        self.report = report
        super().__init__("Recorded provenance check failed: " + "; ".join(report["errors"]))


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def relative(path):
    return path.relative_to(ROOT).as_posix()


def resolve_target(name):
    """A source basename belongs to SOURCE_DIR; paths belong to ROOT."""
    if not isinstance(name, str) or not name:
        raise ValueError("Binding target must be a nonempty path")
    normalized = name.replace("\\", "/")
    raw = Path(normalized)
    if raw.is_absolute() or ":" in normalized or ".." in raw.parts:
        raise ValueError("Binding target must stay inside the repository: " + name)
    mode = "root_relative" if "/" in normalized else "source_basename"
    path = ((ROOT if mode == "root_relative" else SOURCE_DIR) / raw).resolve()
    path.relative_to(ROOT)
    if RESULTS in path.parents and "prospective_v1" in [part.casefold() for part in path.relative_to(RESULTS).parts]:
        raise ValueError("Prospective run artifacts are outside this check: " + name)
    if path.suffix.casefold() == ".jsonl":
        raise ValueError("Trial ledgers are outside this check: " + name)
    return path, mode


def check():
    """Return recorded-to-current bindings, or raise BindingError on any failure."""
    entries, artifacts, before, errors = [], {}, {}, []
    checker_path = Path(__file__).resolve()
    before[checker_path] = digest(checker_path)
    excluded = Counter()

    def bind(artifact, field, name, expected):
        label = relative(artifact) + ":" + field
        try:
            if not isinstance(expected, str) or not SHA256.fullmatch(expected):
                raise ValueError("Invalid recorded SHA256")
            target, resolution = resolve_target(name)
            actual = digest(target)
            old = before.setdefault(target, actual)
            if old != actual:
                raise ValueError("Target changed while checking")
            entry = {"artifact": relative(artifact), "field": field,
                     "target": relative(target), "resolution": resolution,
                     "recorded_sha256": expected, "observed_sha256": actual,
                     "status": "matched"}
            if (artifact.parent == RESULTS and artifact.name == HISTORICAL_ARTIFACT
                    and relative(target) == HISTORICAL_SOURCE
                    and field == "audit_sources_sha256[" + HISTORICAL_SOURCE + "]"):
                if expected != HISTORICAL_SHA or actual != CURRENT_REVIEW_SHA:
                    raise ValueError("Unexpected historical or current auditor version")
                entry.update(status="superseded_historical", current_source_verified=False,
                             superseded_by="results/structured_validation/prospective_audit_checks_v2.json",
                             reason="Original pre-revision smoke check retained unchanged; it does not validate the current auditor. The current auditor must match v2 checks and the independent review.")
            elif actual != expected:
                entry["status"] = "mismatch"
                errors.append(label + " differs from " + relative(target))
            entries.append(entry)
        except (OSError, ValueError) as exc:
            errors.append(label + ": " + str(exc))

    def walk(artifact, node, prefix=""):
        if isinstance(node, list):
            for index, child in enumerate(node):
                walk(artifact, child, prefix + "[" + str(index) + "]")
            return
        if not isinstance(node, dict):
            return
        if "path" in node and "sha256" in node:
            bind(artifact, prefix + ".sha256", node["path"], node["sha256"])
        for key, value in node.items():
            field = prefix + "." + key if prefix else key
            if key == "sha256" and "path" in node:
                continue
            if key in ("source", "local_proof") and (
                    not isinstance(value, dict) or not {"path", "sha256"} <= set(value)):
                errors.append(relative(artifact) + ":" + field + " lacks its path/SHA256 pair")
                continue
            if key in DATA_HASH_FIELDS:
                excluded[key] += len(value) if isinstance(value, dict) else 1
                continue
            if key in SOURCE_MAPS and (key != "source_sha256" or isinstance(value, dict)):
                if (artifact.name == "theory_review.json" and field == "literature_review.sources"
                        and isinstance(value, list) and value):
                    walk(artifact, value, field)
                    continue
                if not isinstance(value, dict) or not value:
                    errors.append(relative(artifact) + ":" + field + " must be a nonempty path/SHA256 mapping")
                    continue
                for name, expected in value.items():
                    bind(artifact, field + "[" + name + "]", name, expected)
                continue
            target = SCALAR_TARGETS.get((artifact.name, field))
            if target is None and key == "plan_sha256":
                target = "results/structured_validation/prospective_plan.json"
            elif target is None and key == "sha256" and "manifest" in node:
                target = node["manifest"]
            elif target is None and key == "run_control_sha256":
                target = "results/structured_validation/run_control.json"
            elif target is None and key == "spec_sha256" and "directory" in node:
                target = node["directory"].rstrip("/") + "/spec.json"
            elif target is None and key == "smoke_checks_sha256" and "smoke_checks_file" in node:
                target = node["smoke_checks_file"]
            elif target is None and key == "document_sha256" and artifact.name in (
                    "continuous_symmetric_design_review.json", "continuous_symmetric_design_peer_review.json"):
                target = "docs/CONTINUOUS_SYMMETRIC_DESIGN.md"
            elif target is None and key == "audit_source_sha256":
                local = artifact.relative_to(RESULTS).as_posix()
                if local in AUDIT_TARGETS:
                    target = AUDIT_TARGETS[local]
            elif target is None and key == "public_case_source_sha256" and artifact.parent.name == "fixed_psr_probe":
                target = "results/learned_validation/development_v1/spec.json"
            if target is not None:
                bind(artifact, field, target, value)
            elif "sha256" in key.casefold():
                errors.append(relative(artifact) + ":" + field + " has no explicit binding rule")
            else:
                walk(artifact, value, field)

    paths = set(RESULTS / name for name in REQUIRED)
    paths.update(RESULTS.glob("*checks*.json"))
    paths.update(RESULTS.glob("*review*.json"))
    for folder in RESULTS.iterdir():
        if not folder.is_dir() or folder.name == "prospective_v1":
            continue
        for name in ("audit.json", "audit_incremental.json", "incremental_audit.json"):
            path = folder / name
            if path.exists():
                paths.add(path)
    for artifact in sorted(paths):
        try:
            resolved_artifact, unused = resolve_target(relative(artifact))
            raw = resolved_artifact.read_bytes()
            actual_hash = hashlib.sha256(raw).hexdigest()
            previous_hash = before.setdefault(resolved_artifact, actual_hash)
            if previous_hash != actual_hash:
                raise ValueError("Artifact changed after it was checked as a binding target")
            node = json.loads(raw.decode("utf-8"))
            if not isinstance(node, dict):
                raise ValueError("Top-level artifact must be an object")
            for field in REQUIRED.get(artifact.name, ()):
                if field not in node:
                    raise ValueError("Missing required recorded field " + field)
            is_audit = artifact.parent != RESULTS
            if is_audit and not ({"audit_source_sha256", "audit_sources_sha256"} & set(node)):
                raise ValueError("Dataset audit contains no recorded auditor source hash")
            start = len(entries)
            walk(artifact, node)
            required_counts = {}
            for field in REQUIRED.get(artifact.name, ()):
                required_counts[field] = sum(
                    entry["field"] == field or entry["field"].startswith(field + ".")
                    or entry["field"].startswith(field + "[") for entry in entries[start:])
                if not required_counts[field]:
                    errors.append(relative(artifact) + ":" + field
                                  + " provides no checkable recorded binding")
            artifacts[relative(artifact)] = {
                "artifact_sha256": actual_hash,
                "recorded_status": node.get("status"),
                "bindings_examined": len(entries) - start,
                "required_field_binding_counts": required_counts,
                "scope": "auditor source bindings; dataset inputs separately checked by freeze_evidence" if is_audit else "recorded source and review artifact bindings"}
        except (OSError, ValueError) as exc:
            errors.append(relative(artifact) + ": " + str(exc))
    for path, expected in sorted(before.items()):
        try:
            if digest(path) != expected:
                errors.append(relative(path) + " changed during provenance verification")
        except OSError as exc:
            errors.append(relative(path) + ": " + str(exc))
    historical = [entry for entry in entries if entry["status"] == "superseded_historical"]
    if len(historical) != 1:
        errors.append("Expected exactly one explicitly identified historical source binding")
    for name in ("prospective_audit_checks_v2.json", "prospective_audit_review.json"):
        current = [entry for entry in entries
                   if entry["artifact"] == "results/structured_validation/" + name
                   and entry["target"] == HISTORICAL_SOURCE]
        if len(current) != 1 or current[0]["status"] != "matched" or current[0]["recorded_sha256"] != CURRENT_REVIEW_SHA:
            errors.append(name + " must bind the current reviewed auditor exactly once")
    report = {
        "status": "failed" if errors else "passed",
        "scope": "Verifies recorded source/review hashes against files, not merely hashes of current files; not a statistical or mathematical audit.",
        "checker_source_sha256": before[checker_path],
        "artifacts_examined": len(artifacts),
        "matched_bindings": sum(entry["status"] == "matched" for entry in entries),
        "superseded_historical_bindings": len(historical),
        "files_rechecked_unchanged": len(before),
        "artifacts": artifacts,
        "bindings": entries,
        "historical": historical,
        "out_of_scope_non_source_hash_fields": dict(sorted(excluded.items())),
        "limits": ["No prospective_v1 file or trial ledger is read or hashed.",
                   "Dataset input hashes, numerical certificates and seed replays remain the responsibility of the individual audits and freeze_evidence.",
                   "Derived configuration/record hashes are not file provenance and are not recalculated here.",
                   "Recorded artifact status is retained verbatim; provenance passing does not turn an incomplete audit into a completed one."],
        "errors": errors}
    if errors:
        raise BindingError(report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Print the read-only provenance report (the only supported operation).")
    parser.parse_args()
    try:
        print(json.dumps(check(), indent=2))
    except BindingError as error:
        print(json.dumps(error.report, indent=2))
        raise SystemExit(1)
