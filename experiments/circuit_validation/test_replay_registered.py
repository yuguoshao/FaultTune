"""Small mechanics and actual-runner tests for predetermined seed replay."""

import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import replay_registered as replay


class ReplaySelectionTests(unittest.TestCase):
    def test_indices_are_fixed_before_outcomes_and_deduplicate_one_repetition(self):
        self.assertEqual(replay.selected_indices(2000), [0, 1999])
        self.assertEqual(replay.selected_indices(1), [0])
        for invalid in (0, -1, True, 2.5):
            with self.assertRaises(ValueError):
                replay.selected_indices(invalid)

    def test_only_wall_clock_fields_are_ignored(self):
        first = {"seed": "7", "decision": "accept", "status": "ok", "shots": 2,
                 "cost": {"event_sha256": "abc"}, "events": [{"shots": 2}],
                 "result": {"certificate": {"threshold": "Infinity"}},
                 "started_epoch": 1, "elapsed_seconds": .5}
        second = copy.deepcopy(first)
        second["started_epoch"], second["elapsed_seconds"] = 20, 9
        self.assertTrue(replay.compare_trial_records(first, second)["match"])
        for field in ("seed", "decision", "status", "shots", "cost", "result", "events"):
            changed = copy.deepcopy(second)
            changed[field] = None
            comparison = replay.compare_trial_records(first, changed)
            self.assertFalse(comparison["match"])
            self.assertIn(field, comparison["mismatched_fields"])
        changed = copy.deepcopy(second)
        changed["shots"] = 2.0
        self.assertFalse(replay.compare_trial_records(first, changed)["match"])

    def test_single_pass_selection_retains_failures_and_detects_missing_duplicate_keys(self):
        spec = {"phase": "test", "repetitions": 3, "cases": [{"case_id": "a"}], "policies": ["p", "q"]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.jsonl.gz"
            rows = [{"phase": "test", "spec_sha256": "digest", "case_id": "a", "policy": policy,
                     "trial_index": index, "status": "timeout" if index == 0 else "ok", "decision": "abstain"}
                    for policy in ("p", "q") for index in range(3)]
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                for row in reversed(rows):
                    handle.write(json.dumps(row) + "\n")
            selected = replay.read_selected_records(spec, "digest", path)
            self.assertEqual(selected["raw_records_scanned"], 6)
            self.assertEqual(len(selected["selected"]), 4)
            self.assertEqual(selected["selected"][("a", "p", 0)]["status"], "timeout")
            self.assertEqual(selected["missing_keys"], [])
            with gzip.open(path, "at", encoding="utf-8") as handle:
                handle.write(json.dumps(rows[0]) + "\n")
            duplicated = replay.read_selected_records(spec, "digest", path)
            self.assertTrue(any(issue["code"] == "duplicate_selected_key" for issue in duplicated["issues"]))
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                for row in rows[:-1]:
                    handle.write(json.dumps(row) + "\n")
            missing = replay.read_selected_records(spec, "digest", path)
            self.assertEqual(missing["missing_keys"], [("a", "q", 2)])


class ActualRunnerReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import run_experiments
        except ImportError as error:
            raise unittest.SkipTest("actual runner requires the project's NumPy environment: " + str(error))
        cls.runner = run_experiments

    def test_tiny_seeded_runner_replays_selected_trials_in_serial_and_parallel(self):
        runner = self.runner
        case = {"case_id": "replay_unit_case", "contexts": ["spectator_0", "spectator_plus"],
                "weights": [.5, .5], "epsilon": .01, "shot_budget": 256, "fixed_pairs": 32,
                "probabilities": [.8, .65, .15, .2], "feasible": True, "infeasible_reason": None}
        spec = {"schema": "faulttune.circuit_spec.v1", "phase": "development", "master_seed": 931,
                "repetitions": 3, "policies": ["direct_mixture", "combgame_glr"], "cases": [case],
                "policy_parameters": {"growth": 1.3, "initial": 32, "max_extra": 262144, "combgame_macro_budget": 16},
                "delta": .01, "observation_contract": {"cost": {"measured_qubits": 3, "old_duration": 3.2,
                                                                   "new_duration": 4.2, "switch_duration": .1}},
                "source_hashes": runner.source_hashes(), "environment": replay._runtime_information()}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path, trials = root / "spec.json", root / "raw.jsonl.gz"
            runner.save_json(spec_path, spec)
            digest = hashlib.sha256(spec_path.read_bytes()).hexdigest()
            spec_path.with_suffix(".json.sha256").write_text(digest, encoding="ascii")
            with gzip.open(trials, "wt", encoding="utf-8") as handle:
                for policy in spec["policies"]:
                    for index in range(spec["repetitions"]):
                        handle.write(runner.encoded(runner.run_trial(spec, digest, case, policy, index, keep_events=True)))
            outputs = []
            for workers in (1, 2):
                result = replay.replay_registered(spec_path, trials, root / ("replay_{}.json".format(workers)), workers)
                self.assertTrue(result["passed"], result)
                self.assertEqual(result["raw_records_scanned"], 6)
                self.assertEqual(result["records_replayed"], 4)
                self.assertEqual(result["records_matched"], 4)
                outputs.append(result["records"])
            self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
