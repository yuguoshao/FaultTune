"""Audit the saved aggregate experiment and optionally record a new manifest.

Use --write-manifest --tests only after intentionally generating a new snapshot.
Hashes include timestamps and Python runtimes: seeded decisions/costs reproduce,
but the byte hashes of a fresh run need not match the historical snapshot.
"""

import argparse
import collections
import hashlib
import json
import platform
import re
import subprocess
import sys
import unittest
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from urllib.parse import unquote

import aggregate_probe


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
EXPERIMENTS = {"aggregate_path": 1600, "aggregate_star": 1200,
               "aggregate_topology": 180}
NEW_FILES = [
    "docs/AGGREGATE_VALIDATION_RESEARCH.md",
    "experiments/local_validation/AGGREGATE_README.md",
    "experiments/local_validation/group_oracle.py",
    "experiments/local_validation/aggregate_probe.py",
    "experiments/local_validation/test_aggregate.py",
    "experiments/local_validation/aggregate_math.py",
    "experiments/local_validation/test_aggregate_math.py",
    "experiments/local_validation/batched_sweep.py",
    "experiments/local_validation/test_batched_sweep.py",
    "experiments/local_validation/verify_aggregate.py",
    "experiments/local_validation/results/aggregate_analytic.json",
]
REUSED_FILES = [
    "experiments/local_validation/probe.py",
    "experiments/local_validation/audit_math.py",
    "experiments/local_validation/stress_probe.py",
    "experiments/local_validation/test_probe.py",
    "experiments/local_validation/.gitignore",
    "experiments/local_validation/results/manifest.json",
]
COMMANDS = [
    "python -B experiments/local_validation/aggregate_probe.py --trials 100 --output experiments/local_validation/results/aggregate_path.json --raw-output experiments/local_validation/results/aggregate_path_trials.jsonl",
    "python -B experiments/local_validation/aggregate_probe.py --graphs star --policies sweep_kl group_kl_balanced group_kl_adaptive --trials 100 --output experiments/local_validation/results/aggregate_star.json --raw-output experiments/local_validation/results/aggregate_star_trials.jsonl",
    "python -B experiments/local_validation/aggregate_probe.py --graphs empty path clique --policies sweep_kl --cases null four_cluster --trials 30 --output experiments/local_validation/results/aggregate_topology.json --raw-output experiments/local_validation/results/aggregate_topology_trials.jsonl",
    "python -B experiments/local_validation/aggregate_math.py --output experiments/local_validation/results/aggregate_analytic.json",
    "python -B experiments/local_validation/verify_aggregate.py --write-manifest --tests",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_manifest(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    for entry in data["files"]:
        actual = ROOT / entry["path"]
        require(actual.is_file(), "missing snapshot file: " + entry["path"])
        require(actual.stat().st_size == entry["bytes"], "size mismatch: " + entry["path"])
        require(digest(actual) == entry["sha256"], "hash mismatch: " + entry["path"])
    return len(data["files"])


def audit_experiments():
    audits, raw_by_name = [], {}
    source_hash = digest(HERE / "aggregate_probe.py")
    for stem, expected_count in EXPERIMENTS.items():
        saved = json.loads((RESULTS / (stem + ".json")).read_text(encoding="utf-8"))
        require(saved["metadata"]["script_sha256"] == source_hash, "stale source hash: " + stem)
        rows = [json.loads(line) for line in
                (RESULTS / (stem + "_trials.jsonl")).read_text(encoding="utf-8").splitlines()]
        require(len(rows) == expected_count, "unexpected raw trial count: " + stem)
        buckets = collections.defaultdict(list)
        identities = set()
        for row in rows:
            identity = (row["graph"], row["case"], row["policy"], row["trial"])
            require(identity not in identities, "duplicate trial: " + str(identity))
            identities.add(identity)
            require(sum(row["counts"]) == row["objects"], "inconsistent object accounting")
            require(0 <= row["switches"] <= max(0, row["rounds"] - 1), "invalid switch count")
            require(row["rounds"] <= saved["metadata"]["round_budget"], "round budget exceeded")
            require(row["objects"] <= saved["metadata"]["object_budget"], "object budget exceeded")
            buckets[identity[:3]].append(row)
        require(len(buckets) == len(saved["groups"]), "unexpected number of cells")
        for cell in saved["groups"]:
            key = (cell["graph"], cell["case"], cell["policy"])
            require(len(buckets[key]) == saved["metadata"]["trials_per_group"], "missing cell trials")
            require(aggregate_probe.summarize(buckets[key]) == cell, "summary mismatch: " + str(key))
        audits.append({"experiment": stem, "raw_trials": len(rows), "groups": len(buckets),
                       "aggregation_source_and_cost_checks": "passed"})
        raw_by_name[stem] = rows
    return audits, raw_by_name


def audit_pairing(raw):
    keys = ("decision", "objects", "counts", "interval", "seed", "audit", "checkpoint_counts")
    topology = collections.defaultdict(dict)
    for row in raw["aggregate_topology"]:
        topology[(row["case"], row["trial"])][row["graph"]] = row
    for identity, by_graph in topology.items():
        reference = by_graph["empty"]
        for graph, factor in (("path", 2), ("clique", 16)):
            row = by_graph[graph]
            for key in keys:
                # This vector has one entry per color class, so compare its
                # common checkpoint count rather than its graph-dependent size.
                if key == "checkpoint_counts":
                    require(set(row[key]) == set(reference[key]), "checkpoint pairing mismatch")
                else:
                    require(row[key] == reference[key], "topology pairing mismatch: " + key)
            require(row["rounds"] == factor * reference["rounds"], "topology round ratio mismatch")
    path = {(r["case"], r["trial"]): r for r in raw["aggregate_path"] if r["policy"] == "sweep_kl"}
    star = {(r["case"], r["trial"]): r for r in raw["aggregate_star"] if r["policy"] == "sweep_kl"}
    require(path.keys() == star.keys(), "path/star identity mismatch")
    for identity, row in path.items():
        require(all(row[key] == star[identity][key] for key in keys + ("rounds",)),
                "path/star sweep mismatch")
    return {"topology_triplets": len(topology), "round_ratio": [1, 2, 16],
            "path_star_sweep_pairs": len(path), "status": "passed"}


def audit_analytic():
    saved = json.loads((RESULTS / "aggregate_analytic.json").read_text(encoding="utf-8"))
    require(saved["script_sha256"] == digest(HERE / "aggregate_math.py"), "stale analytic source hash")

    def fraction(record):
        return Fraction(int(record["numerator"]), int(record["denominator"]))

    certificates = 0
    for row in saved["finite_horizon_results"]:
        for site in row["sites"].values():
            total = sum(fraction(site[key]) for key in
                        ("clear_by_horizon", "alert_by_horizon", "unresolved_at_horizon"))
            require(total == 1, "analytic probability mass is not exactly one")
        for certificate in row["counterexamples"].values():
            event = Fraction(1)
            for factor in certificate["event_factors"]:
                event *= fraction(row["sites"][factor["site_result"]]["clear_by_horizon"]) ** factor["power"]
            lower = fraction(certificate["false_acceptance_lower"])
            require(Fraction(1, 20) < lower <= event, "invalid exact counterexample lower bound")
            require(event - lower < Fraction(1, 10 ** 12), "incorrect rational floor")
            require(fraction(certificate["true_delta"]) > 0, "counterexample is not harmful")
            certificates += 1
    return {"finite_event_certificates_checked": certificates,
            "characteristic_rate_cases": len(saved["characteristic_results"]),
            "exact_fractions_and_source_hash": "passed"}


def audit_links(allow_pending_manifest=False):
    count = 0
    for relative in NEW_FILES[:2]:
        path = ROOT / relative
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if target.startswith(("https://", "http://", "#")):
                continue
            resolved = path.parent / unquote(target.split("#", 1)[0].strip("<>"))
            pending = allow_pending_manifest and resolved.resolve() == (RESULTS / "aggregate_manifest.json").resolve()
            require(resolved.is_file() or pending, "broken document link: " + relative + " -> " + target)
            count += 1
    return count


def run_tests():
    suite = unittest.defaultTestLoader.discover(str(HERE), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    require(result.wasSuccessful(), "unit tests failed")
    return result.testsRun


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests", action="store_true")
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    if args.write_manifest and not args.tests:
        parser.error("writing a manifest requires --tests")
    old_files = validate_manifest(RESULTS / "manifest.json")
    audits, raw = audit_experiments()
    pairing = audit_pairing(raw)
    analytic = audit_analytic()
    links = audit_links(allow_pending_manifest=args.write_manifest)
    tests = run_tests() if args.tests else None
    output = RESULTS / "aggregate_manifest.json"
    if args.write_manifest:
        files = NEW_FILES + REUSED_FILES + [
            "experiments/local_validation/results/" + stem + suffix
            for stem in EXPERIMENTS for suffix in (".json", "_trials.jsonl")]
        revisions = {name: subprocess.check_output(
            ["git", "rev-parse", name], cwd=str(ROOT), text=True).strip()
            for name in ("HEAD", "origin/dev", "origin/dev_hub")}
        manifest = {
            "schema_version": "faulttune.aggregate_manifest.v1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "base_revisions": revisions,
            "scope": "standalone fixed-candidate Bernoulli research; new uncommitted files; no FaultTune engine or quantum circuit integration",
            "python": platform.python_version(), "platform": platform.platform(),
            "dependencies": "Python standard library only",
            "commands": COMMANDS,
            "monte_carlo_trials": dict(EXPERIMENTS, total=sum(EXPERIMENTS.values())),
            "verification": {"unit_tests_passed": tests, "raw_checks": audits,
                             "pairing": pairing, "analytic": analytic,
                             "first_round_snapshot_files_unchanged": old_files,
                             "local_document_links_checked": links},
            "files": [{"path": relative, "bytes": (ROOT / relative).stat().st_size,
                       "sha256": digest(ROOT / relative)} for relative in files],
        }
        output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    current_files = validate_manifest(output)
    require(audit_links() == links, "document links changed during audit")
    print(json.dumps({"manifest": str(output), "snapshot_files_checked": current_files,
                      "first_round_files_unchanged": old_files, "raw_checks": audits,
                      "pairing": pairing, "analytic": analytic, "local_links_checked": links,
                      "unit_tests_passed_this_run": tests}, indent=2))


if __name__ == "__main__":
    main()
