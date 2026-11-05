"""Recompute the NEW exploratory ledger and preserve the old frozen sources."""
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "circuit_validation"))
from direct_tests import certificate_from_pairs
from theory_check import (exact_obstruction_check, conditional_shift_check,
                          mismatch_obstruction_check)
from shift_validation import run_cell


def main():
    result_dir = ROOT / "results" / "phase_validation"
    summary = json.loads((result_dir / "shift_pilot_summary.json").read_text(encoding="utf-8"))
    source = ROOT / "experiments" / "phase_validation" / "shift_validation.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == summary["spec"]["source_sha256"]
    cells = summary["cells"]
    methods = summary["spec"]["methods"]
    groups = {(i, method): [] for i in range(len(summary["spec"]["cases"])) for method in methods}
    unique = set()
    certificates = 0
    scipy_checks = 0
    for line in (result_dir / "shift_pilot_trials.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = (row["case_id"], row["method"], row["rep"])
        assert key not in unique
        unique.add(key)
        assert len(row["counts"]) == 4 and all(isinstance(n, int) and n >= 0 for n in row["counts"])
        assert 2*sum(row["counts"]) == row["shots"]
        groups[key[:2]].append(row)
        if row["certificate"] is not None:
            recomputed = certificate_from_pairs(row["counts"], delta=.01)
            assert recomputed == row["certificate"]
            assert recomputed["decision"] == row["decision"]
            certificates += 1
            b, c = row["counts"][1:3]
            evidence = gammaln(b+.5)+gammaln(c+.5)-gammaln(b+c+1)-2*gammaln(.5)+(b+c)*math.log(2)
            # This is a separate floating-point cross-check, not the numerical
            # certificate. Large-count log-Gamma subtraction has cancellation.
            assert evidence > math.log(100)-1e-5
            assert ("accept" if b < c else "reject") == row["decision"]
            scipy_checks += 1
    assert len(unique) == summary["trials"]
    for cid, case in enumerate(summary["spec"]["cases"]):
        for mid, method in enumerate(methods):
            rows = groups[(cid, method)]
            expected_summary = cells[cid*len(methods)+mid]
            assert len(rows) == summary["spec"]["repetitions"]
            assert {row["rep"] for row in rows} == set(range(len(rows)))
            assert abs(sum(row["shots"] for row in rows)/len(rows)-expected_summary["mean_shots"]) < 1e-9
            expected = "accept" if case["phase"] > 0 else "reject" if case["phase"] < 0 else "abstain"
            wrong = sum(row["decision"] not in ("abstain", expected) for row in rows)
            if expected == "abstain":
                wrong = sum(row["decision"] != "abstain" for row in rows)
            assert wrong == expected_summary["wrong"]
            assert sum(row["decision"] == "abstain" for row in rows) == expected_summary["abstained"]
    # Replay both methods for one small-update, one large-update and one null
    # case. Entire vector batches are replayed to preserve RNG call shapes.
    replayed = 0
    for cid in (0, 23, 26):
        for mid, method in enumerate(methods):
            old = cells[cid*len(methods)+mid]
            new = run_cell(old["case"], method, old["repetitions"], old["seed"])
            original = sorted(groups[(cid, method)], key=lambda r:r["rep"])
            for actual, expected in zip(new.pop("trials"), original):
                assert actual == {k:v for k,v in expected.items() if k not in ("case_id", "method")}
                replayed += 1
            assert new == old
    unchanged = []
    archive_path = ROOT / "experiments" / "circuit_validation" / "confirmation_spec_v1_sources.zip"
    with zipfile.ZipFile(str(archive_path)) as archive:
        for name in archive.namelist():
            if not name.endswith(".py"):
                continue
            target = ROOT / "experiments" / "circuit_validation" / name
            if not target.is_file():
                target = ROOT / name
            assert target.is_file(), name
            assert hashlib.sha256(archive.read(name)).digest() == hashlib.sha256(target.read_bytes()).digest(), name
            unchanged.append(name)
    theory = {"original_model": exact_obstruction_check(),
              "common_channel_shift": conditional_shift_check(),
              "mismatch_threshold": mismatch_obstruction_check()}
    (result_dir / "theory_checks.json").write_text(json.dumps(theory, indent=2), encoding="utf-8")
    audit = {"status": "passed", "ledger_records": len(unique),
             "decision_certificates_recomputed": certificates,
             "independent_scipy_decision_checks": scipy_checks,
             "vector_seed_replay_records": replayed,
             "old_frozen_python_sources_unchanged": len(unchanged),
             "old_source_paths": unchanged,
             "method_error_totals": {method: sum(c["wrong"] for c in cells if c["method"] == method)
                                     for method in methods},
             "physical_identity_max_residual": summary["physical_audit"]["max_absolute_residual"],
             "scope": "Implementation/algebra audit; not a new simultaneous performance qualification."}
    (result_dir / "exploration_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
