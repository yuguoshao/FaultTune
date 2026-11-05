"""Corruption and round-trip tests for the independent raw-record auditor."""

import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import warnings
import zipfile

import audit_records as ar
from direct_tests import certificate_from_pairs, direct_kl_certificate, fixed_paired_certificate
from kl_trackstop import joint_glr_certificate


POLICIES = ["direct_fixed", "direct_mixture", "direct_intervals", "joint_trackstop",
            "neyman_glr", "balanced_glr", "combgame_glr", "calibration_only"]


def specification(policies=None, repetitions=1):
    return {"schema": "faulttune.circuit_spec.v1", "phase": "development",
            "master_seed": 481, "repetitions": repetitions,
            "policies": POLICIES[:] if policies is None else list(policies),
            "policy_parameters": {"growth": 1.3, "initial": 32, "max_extra": 262144,
                                  "combgame_macro_budget": 16},
            "delta": .01,
            "cases": [{"case_id": "synthetic", "contexts": ["spectator_0", "spectator_plus"],
                       "weights": [.1, .9], "epsilon": .01, "shot_budget": 512, "fixed_pairs": 32}],
            "observation_contract": {"cost": {"measured_qubits": 3, "old_duration": 3.2,
                                                "new_duration": 4.2, "switch_duration": .1}},
            "source_hashes": {}}


def encoded_spec(spec):
    return (json.dumps(spec, sort_keys=True, indent=2) + "\n").encode("utf-8")


def manual_cost(events):
    counts, failures, digest = [0] * 4, [0] * 4, hashlib.sha256()
    switches, duration, last = 0, 0.0, None
    for event in events:
        charges = []
        if event["kind"] == "arm":
            charges.append((event["arm"], event["shots"], event["failures"]))
        else:
            for piece in event["pieces"]:
                c, n, joint = piece["context"], piece["pairs"], piece["joint"]
                charges.extend(((c, n, joint[2] + joint[3]), (2 + c, n, joint[1] + joint[3])))
        for arm, shots, failed in charges:
            if last is not None and last != arm:
                switches += 1
            last = arm
            counts[arm] += shots
            failures[arm] += failed
            duration += shots * (3.2 if arm < 2 else 4.2)
        digest.update((json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii"))
    return {"shots": sum(counts), "counts": counts, "failures": failures,
            "readout_objects": 3 * sum(counts), "switches": switches, "batches": len(events),
            "execution_duration": duration, "modeled_device_duration": duration + .1 * switches,
            "event_sha256": digest.hexdigest()}


def make_record(spec, policy, index=0):
    case = spec["cases"][0]
    opaque = hashlib.sha256(b"synthetic::public").hexdigest()
    config = {"contexts": ["spectator_0", "spectator_plus"], "weights": [.1, .9],
              "delta": .01, "epsilon": .01, "shot_budget": 512, "measured_qubits": 3,
              "old_duration": 3.2, "new_duration": 4.2, "switch_duration": .1,
              "candidate_id": opaque[:32], "workload_id": opaque[32:], "phase": "development"}
    seed_text = json.dumps((481, "development", "synthetic", policy, index), separators=(",", ":"))
    seed = int.from_bytes(hashlib.blake2b(seed_text.encode("ascii"), digest_size=16,
                                        person=b"ft-circuit-v1").digest(), "big")
    if policy in ar.PAIR_POLICIES:
        pieces = [{"context": 0, "pairs": 32, "old_start": 0, "new_start": 0, "joint": [0, 0, 32, 0]}]
        if policy != "calibration_only":
            pieces = [{"context": c, "pairs": 16, "old_start": 0, "new_start": 0,
                       "joint": [0, 0, 16, 0]} for c in (0, 1)]
        events = [{"kind": "pairs", "mode": "fixed" if policy == "calibration_only" else "iid",
                   "pairs": 32, "pieces": pieces, "joint": [0, 0, 32, 0]}]
        if policy == "direct_fixed":
            certificate = fixed_paired_certificate([0, 0, 32, 0])
        elif policy == "direct_intervals":
            certificate = direct_kl_certificate(32, 32, 0, 32)
        else:
            certificate = certificate_from_pairs([0, 0, 32, 0])
        result = {"decision": "accept", "certificate": certificate, "checks": 1,
                  "pair_counts": [0, 0, 32, 0], "guarantee_in_scope": policy != "calibration_only"}
    else:
        events = [{"kind": "arm", "arm": arm, "start": 0, "shots": 32,
                   "failures": 32 if arm < 2 else 0} for arm in range(4)]
        certificate = joint_glr_certificate([32, 32, 0, 0], [32] * 4, [-.1, -.9, .1, .9])
        result = {"decision": "accept", "certificate": certificate, "checks": 1,
                  "arm_trials": [32] * 4, "arm_successes": [32, 32, 0, 0], "guarantee_in_scope": True}
        if policy == "combgame_glr":
            result["checks"] = 4
            result["sampling_metadata"] = {"macro_budget": 16, "block_size": 32,
                                            "complete_blocks": 4, "final_partial_shots": 0}
    cost = manual_cost(events)
    return {"schema": "faulttune.circuit_trial.v1", "phase": "development",
            "spec_sha256": hashlib.sha256(encoded_spec(spec)).hexdigest(), "case_id": case["case_id"],
            "policy": policy, "trial_index": index, "seed": str(seed), "status": "ok",
            "decision": "accept", "shots": cost["shots"], "shot_budget": 512,
            "cost": cost, "result": result, "error": None, "public_config": config,
            "events": events}


def issue_codes(record, spec):
    result = ar.audit_record(record, spec, hashlib.sha256(encoded_spec(spec)).hexdigest())
    return {issue["code"] for issue in result["issues"]}


class RecordChecks(unittest.TestCase):
    def test_all_registered_methods_including_new_arm_name_reproduce(self):
        spec = specification(POLICIES + ["future_allocation"])
        for policy in spec["policies"]:
            with self.subTest(policy=policy):
                record = make_record(spec, policy)
                self.assertEqual(record["result"]["certificate"]["decision"], "accept")
                self.assertEqual(issue_codes(record, spec), set())

    def test_failures_and_timeouts_keep_partial_paid_ledger(self):
        spec = specification(["joint_trackstop"])
        for status in ("error", "timeout"):
            record = make_record(spec, "joint_trackstop")
            record["status"], record["decision"], record["result"] = status, "abstain", None
            record["events"] = record["events"][:1]
            record["cost"] = manual_cost(record["events"])
            record["shots"] = record["cost"]["shots"]
            self.assertEqual(issue_codes(record, spec), set())
            record["decision"] = "accept"
            self.assertIn("failed_trial_result", issue_codes(record, spec))

    def test_cost_corruption_is_detected_per_dimension(self):
        spec = specification(["direct_fixed"])
        for field in ("readout_objects", "switches", "batches", "execution_duration", "modeled_device_duration"):
            record = make_record(spec, "direct_fixed")
            record["cost"][field] += 1
            with self.subTest(field=field):
                self.assertIn("ledger_" + field, issue_codes(record, spec))
        record = make_record(spec, "direct_fixed")
        record["cost"]["counts"][0] += 1
        self.assertIn("ledger_counts", issue_codes(record, spec))
        record = make_record(spec, "direct_fixed")
        record["cost"]["failures"][0] -= 1
        self.assertIn("ledger_failures", issue_codes(record, spec))

    def test_boolean_counts_and_offsets_are_not_integers(self):
        spec = specification(["direct_fixed"])
        record = make_record(spec, "direct_fixed")
        record["cost"]["failures"][2] = False
        self.assertIn("cost_count_types", issue_codes(record, spec))
        record = make_record(spec, "direct_fixed")
        record["events"][0]["pieces"][0]["old_start"] = False
        record["cost"] = manual_cost(record["events"])
        self.assertIn("event_offset", issue_codes(record, spec))

    def test_offset_and_margin_corruption_survive_rehashed_ledger(self):
        spec = specification(["direct_mixture"])
        record = make_record(spec, "direct_mixture")
        record["events"][0]["pieces"][0]["old_start"] = 1
        record["cost"] = manual_cost(record["events"])
        self.assertIn("event_offset", issue_codes(record, spec))
        record = make_record(spec, "direct_mixture")
        record["events"][0]["joint"] = [0, 1, 31, 0]
        record["cost"] = manual_cost(record["events"])
        self.assertIn("pair_margins", issue_codes(record, spec))

    def test_context_scope_and_diagnostic_flag_are_enforced(self):
        spec = specification(["direct_mixture", "calibration_only"])
        record = make_record(spec, "direct_mixture")
        record["events"][0]["mode"] = "fixed"
        record["cost"] = manual_cost(record["events"])
        self.assertIn("context_scope", issue_codes(record, spec))
        record = make_record(spec, "calibration_only")
        record["result"]["guarantee_in_scope"] = True
        self.assertIn("guarantee_scope", issue_codes(record, spec))

    def test_fixed_design_must_match_registered_n_and_single_look(self):
        spec = specification(["direct_fixed"])
        spec["cases"][0]["fixed_pairs"] = 64
        record = make_record(spec, "direct_fixed")
        self.assertIn("fixed_sample_size", issue_codes(record, spec))

    def test_recomputed_decision_overrides_forged_stored_certificate(self):
        spec = specification(["direct_mixture"])
        record = make_record(spec, "direct_mixture")
        event = record["events"][0]
        event["joint"] = [0, 32, 0, 0]
        for piece in event["pieces"]:
            piece["joint"] = [0, 16, 0, 0]
        record["cost"] = manual_cost(record["events"])
        record["result"]["pair_counts"] = event["joint"][:]
        codes = issue_codes(record, spec)
        self.assertIn("recomputed_decision", codes)
        self.assertIn("certificate_reproduction", codes)

    def test_certificate_evidence_and_sufficient_statistics_are_checked(self):
        spec = specification(["joint_trackstop"])
        record = make_record(spec, "joint_trackstop")
        record["result"]["certificate"]["evidence"] += 1
        self.assertIn("certificate_reproduction", issue_codes(record, spec))
        record = make_record(spec, "joint_trackstop")
        record["result"]["arm_trials"][0] += 1
        self.assertIn("result_arm_counts", issue_codes(record, spec))

    def test_frozen_config_hash_and_seed_are_checked(self):
        spec = specification(["direct_fixed"])
        for key, value, expected in (("spec_sha256", "0" * 64, "spec_hash"),
                                      ("seed", "1", "seed"), ("phase", "confirmation", "phase"),
                                      ("shot_budget", 1024, "registered_budget")):
            record = make_record(spec, "direct_fixed")
            record[key] = value
            self.assertIn(expected, issue_codes(record, spec))
        record = make_record(spec, "direct_fixed")
        record["public_config"]["weights"] = [.9, .1]
        self.assertIn("public_config", issue_codes(record, spec))

    def test_event_digest_and_missing_ledger_are_checked(self):
        spec = specification(["direct_fixed"])
        record = make_record(spec, "direct_fixed")
        record["cost"]["event_sha256"] = "0" * 64
        self.assertIn("ledger_event_sha256", issue_codes(record, spec))
        del record["events"]
        self.assertIn("ledger_missing", issue_codes(record, spec))

    def test_incomplete_successful_arm_checkpoint_is_rejected(self):
        spec = specification(["joint_trackstop"])
        record = make_record(spec, "joint_trackstop")
        record["events"] = record["events"][:-1]
        record["cost"] = manual_cost(record["events"])
        record["shots"] = record["cost"]["shots"]
        self.assertIn("incomplete_checkpoint", issue_codes(record, spec))

    def test_intermediate_crossing_is_explicitly_diagnostic(self):
        spec = specification(["direct_mixture"])
        record = make_record(spec, "direct_mixture")
        second = copy.deepcopy(record["events"][0])
        for piece in second["pieces"]:
            piece["old_start"] = piece["new_start"] = 16
        record["events"].append(second)
        record["cost"] = manual_cost(record["events"])
        record["shots"] = record["cost"]["shots"]
        record["result"]["pair_counts"] = [0, 0, 64, 0]
        record["result"]["checks"] = 2
        record["result"]["certificate"] = certificate_from_pairs([0, 0, 64, 0])
        audited = ar.audit_record(record, spec, hashlib.sha256(encoded_spec(spec)).hexdigest(), intermediate=True)
        self.assertEqual(audited["issues"], [])
        self.assertEqual(audited["diagnostics"]["intermediate_checkpoints_recomputed"], 1)
        self.assertEqual(audited["diagnostics"]["earlier_exact_crossings"][0]["shots"], 64)

    def test_macro_blocks_initialization_and_final_partial_are_independent_of_geometric_schedule(self):
        spec = specification(["combgame_glr"])
        spec["cases"][0]["shot_budget"] = 129
        spec["policy_parameters"]["combgame_macro_budget"] = 4
        record = make_record(spec, "combgame_glr")
        record["shot_budget"] = record["public_config"]["shot_budget"] = 129
        for event, size in zip(record["events"], (33, 33, 33, 30)):
            event["shots"] = size
            event["failures"] = size if event["arm"] < 2 else 0
        record["cost"] = manual_cost(record["events"])
        record["shots"] = record["cost"]["shots"]
        record["result"]["arm_trials"] = [33, 33, 33, 30]
        record["result"]["arm_successes"] = [33, 33, 0, 0]
        record["result"]["certificate"] = joint_glr_certificate([33, 33, 0, 0], [33, 33, 33, 30], [-.1, -.9, .1, .9])
        record["result"]["sampling_metadata"] = {"macro_budget": 4, "block_size": 33,
                                                   "complete_blocks": 3, "final_partial_shots": 30}
        self.assertEqual(issue_codes(record, spec), set())
        reordered = copy.deepcopy(record)
        reordered["events"][:2] = reversed(reordered["events"][:2])
        reordered["cost"] = manual_cost(reordered["events"])
        self.assertIn("macro_initialization_order", issue_codes(reordered, spec))
        record["events"][-1]["shots"] = 29
        record["cost"] = manual_cost(record["events"])
        record["shots"] = record["cost"]["shots"]
        self.assertIn("macro_block_size", issue_codes(record, spec))


class CollectionAndFileChecks(unittest.TestCase):
    def test_full_grid_duplicates_missing_keys_and_seed_reuse(self):
        spec = specification()
        digest = hashlib.sha256(encoded_spec(spec)).hexdigest()
        auditor = ar.RecordAuditor(spec, digest)
        rows = [make_record(spec, policy) for policy in POLICIES]
        for row in rows:
            auditor.consume(row)
        complete = auditor.finish()
        self.assertTrue(complete["passed"])
        self.assertEqual(complete["final_decision_certificates_recomputed"], len(POLICIES))
        auditor.consume(copy.deepcopy(rows[0]))
        repeated = auditor.finish()
        self.assertIn("duplicate_key", repeated["issue_counts"])
        self.assertIn("duplicate_seed", repeated["issue_counts"])
        missing = ar.RecordAuditor(spec, digest).finish()
        self.assertFalse(missing["complete"])
        self.assertEqual(missing["missing_records"], len(POLICIES))

    def test_strict_json_rejects_duplicate_keys_and_nonstandard_numbers(self):
        for payload in ('{"shots":1,"shots":2}', '{"evidence":Infinity}', '{"x":NaN}'):
            with self.assertRaises(ValueError):
                ar.strict_json_loads(payload)
        self.assertEqual(ar.strict_json_loads('{"threshold":"Infinity"}'), {"threshold": "Infinity"})

    def test_plain_and_gzip_files_round_trip(self):
        spec = specification(["direct_fixed", "combgame_glr"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path = root / "spec.json"
            spec_path.write_bytes(encoded_spec(spec))
            digest = hashlib.sha256(encoded_spec(spec)).hexdigest()
            spec_path.with_suffix(".json.sha256").write_text(digest + "  spec.json\n", encoding="ascii")
            rows = [make_record(spec, policy) for policy in spec["policies"]]
            for extension in (".jsonl", ".jsonl.gz"):
                trials = root / ("trials" + extension)
                opener = gzip.open if extension.endswith(".gz") else open
                with opener(trials, "wt", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row) + "\n")
                summaries = []
                for workers in (1, 2):
                    result = ar.audit_files(spec_path, trials, repo_root=root, workers=workers, chunk_records=1)
                    self.assertTrue(result["passed"], result["issues"])
                    self.assertTrue(result["complete"])
                    self.assertFalse(result["confirmation_engineering_eligible"])
                    summaries.append({key: result[key] for key in ("records_read", "planned_records", "invalid_records",
                                                                   "unique_trial_keys", "unique_seeds", "status_counts",
                                                                   "final_decision_certificates_recomputed", "issue_counts")})
                self.assertEqual(summaries[0], summaries[1])

    def test_malformed_json_line_and_source_change_cannot_pass(self):
        spec = specification(["direct_fixed"])
        spec["source_hashes"] = {"model.py": "0" * 64}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path, trials_path = root / "spec.json", root / "trials.jsonl"
            spec_path.write_bytes(encoded_spec(spec))
            digest = hashlib.sha256(encoded_spec(spec)).hexdigest()
            spec_path.with_suffix(".json.sha256").write_text(digest, encoding="ascii")
            (root / "model.py").write_text("pass\n", encoding="utf-8")
            trials_path.write_text('{"shots":1,"shots":2}\n', encoding="utf-8")
            result = ar.audit_files(spec_path, trials_path, repo_root=root)
            self.assertFalse(result["passed"])
            self.assertIn("source_hash", result["issue_counts"])
            self.assertIn("record_json", result["issue_counts"])
            self.assertEqual(result["invalid_records"], 1)

    def test_truncated_gzip_and_empty_spec_sidecar_fail_explicitly(self):
        spec = specification(["direct_fixed"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path, trials_path = root / "spec.json", root / "trials.jsonl.gz"
            spec_path.write_bytes(encoded_spec(spec))
            spec_path.with_suffix(".json.sha256").write_text("", encoding="ascii")
            with gzip.open(trials_path, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(make_record(spec, "direct_fixed")) + "\n")
            trials_path.write_bytes(trials_path.read_bytes()[:-4])
            result = ar.audit_files(spec_path, trials_path, repo_root=root)
            self.assertFalse(result["passed"])
            self.assertIn("spec_sidecar_hash", result["issue_counts"])
            self.assertIn("trials_read_error", result["issue_counts"])

    def test_explicit_archive_authenticates_frozen_sources_after_current_code_changes(self):
        spec = specification(["direct_fixed"])
        original = b"original source\n"
        spec["source_hashes"] = {"model.py": hashlib.sha256(original).hexdigest()}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "sources.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("model.py", original)
            spec["source_archive"] = {"name": archive.name,
                                       "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
            (root / "model.py").write_bytes(b"updated source\n")
            spec_path, trials_path = root / "spec.json", root / "trials.jsonl"
            spec_path.write_bytes(encoded_spec(spec))
            digest = hashlib.sha256(encoded_spec(spec)).hexdigest()
            spec_path.with_suffix(".json.sha256").write_text(digest, encoding="ascii")
            trials_path.write_text(json.dumps(make_record(spec, "direct_fixed")) + "\n", encoding="utf-8")
            current = ar.audit_files(spec_path, trials_path, repo_root=root)
            self.assertFalse(current["passed"])
            archived = ar.audit_files(spec_path, trials_path, repo_root=root, source_archive=archive)
            self.assertTrue(archived["passed"], archived["issues"])
            self.assertFalse(archived["current_source_matches"])
            self.assertTrue(archived["archived_source_matches"])
            self.assertEqual(archived["source_verification_mode"], "archive_explicit")
            self.assertTrue(archived["source_archive"]["container_hash_matches"])
            self.assertIn("current_certificate_module_provenance", archived)

    def test_archive_missing_extra_duplicate_and_changed_entries_are_detected(self):
        original = b"original\n"
        spec = specification(["direct_fixed"])
        spec["source_hashes"] = {"model.py": hashlib.sha256(original).hexdigest()}
        mutations = (([], "archived_source_hash"),
                     ([("model.py", original), ("extra.py", b"x")], "archive_extra_entry"),
                     ([("model.py", original), ("model.py", original)], "archive_duplicate_entry"),
                     ([("model.py", b"changed")], "archived_source_hash"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.py").write_bytes(original)
            for entries, expected_code in mutations:
                archive = root / "sources.zip"
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    with zipfile.ZipFile(archive, "w") as handle:
                        for name, data in entries:
                            handle.writestr(name, data)
                result = ar._source_evidence(spec, root, archive)
                self.assertIn(expected_code, {issue["code"] for issue in result["issues"]})
                self.assertFalse(result["archived_source_matches"])
                self.assertTrue(result["current_source_matches"])

    def test_declared_archive_container_hash_cannot_be_bypassed(self):
        spec = specification(["direct_fixed"])
        original = b"original\n"
        spec["source_hashes"] = {"model.py": hashlib.sha256(original).hexdigest()}
        spec["source_archive"] = {"name": "sources.zip", "sha256": "0" * 64}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.py").write_bytes(original)
            archive = root / "sources.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("model.py", original)
            result = ar._source_evidence(spec, root, archive)
            self.assertIn("archive_container_hash", {issue["code"] for issue in result["issues"]})
            self.assertFalse(result["archived_source_matches"])


if __name__ == "__main__":
    unittest.main()
