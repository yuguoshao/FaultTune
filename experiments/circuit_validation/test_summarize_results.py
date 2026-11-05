"""Summary information-flow, missing-grid, corruption, and aggregation checks."""

import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from case_factory import build_cases
import summarize_results as sr


class SummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = build_cases("development")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def fixture(self, mutate=None, phase="development"):
        cases = copy.deepcopy(self.cases)
        for case in cases:
            case["shot_budget"] = 1000
        policies = ["b1", "b2", "b3", "combgame_glr", "c1", "c2", "control", "unsafe"]
        ncell = len(cases) * len(policies)
        # New specs deliberately round each allocation downward. Tests must
        # not accidentally depend on an epsilon tolerance for alpha spending.
        safety_alpha = sr._next_float(.02 / (3*ncell), False)
        power_alpha = sr._next_float(.02 / (2*ncell), False)
        cost_alpha = sr._next_float(.01 / (2*ncell), False)
        spec = dict(schema="faulttune.circuit_spec.v1", phase=phase, repetitions=2,
                    policies=policies, baselines=policies[:4], candidates=["c1", "c2"], diagnostic_policies=["unsafe"],
                    cases=cases, master_seed=123, delta=.01, source_hashes={}, confirmation_freeze=phase == "confirmation",
                    observation_contract={"cost": {"measured_qubits": 3}},
                    analysis=dict(alpha_safety=safety_alpha, alpha_power=power_alpha, alpha_cost=cost_alpha,
                                  safety_family_alpha=.02, power_family_alpha=.02, cost_family_alpha=.01,
                                  family_alpha=.05, unused_alpha_recycled=False, speedup_point_target=3, speedup_lower_target=2))
        spec_path = self.directory / "spec.json"
        spec_path.write_text(json.dumps(spec), encoding="utf8")
        digest = hashlib.sha256(spec_path.read_bytes()).hexdigest()
        spec_path.with_suffix(".json.sha256").write_text(digest, encoding="ascii")
        rows = []
        for case in cases:
            for policy in policies:
                for index in range(2):
                    decision = ("accept" if case["true_delta"] < -case["epsilon"] else "reject"
                                if case["true_delta"] > case["epsilon"] else "abstain")
                    if policy == "unsafe":
                        decision = "accept"
                    shots = 80 if policy in ("c1", "c2") else 400
                    rows.append(dict(schema="faulttune.circuit_trial.v1", phase=phase, spec_sha256=digest,
                                     case_id=case["case_id"], policy=policy, trial_index=index,
                                     seed=sr._seed(spec, case["case_id"], policy, index), status="ok", decision=decision,
                                     shots=shots, shot_budget=1000, elapsed_seconds=.1,
                                     cost=dict(shots=shots, counts=[shots,0,0,0], failures=[0,0,0,0],
                                               readout_objects=3*shots, switches=0, execution_duration=shots*3.2,
                                               modeled_device_duration=shots*3.2),
                                     public_config=dict(shot_budget=1000, epsilon=case["epsilon"], weights=case["weights"],
                                                        contexts=case["contexts"], delta=.01, phase=phase),
                                     result=dict(decision=decision, certificate={"decision":decision}, guarantee_in_scope=policy!="unsafe")))
        if mutate:
            mutate(rows)
        raw_path = self.directory / "trials.jsonl.gz"
        with gzip.open(str(raw_path), "wt", encoding="utf8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return spec_path, raw_path

    def test_dynamic_eight_policy_grid_and_all_main_cases_preserved(self):
        result = sr.summarize_results(*self._args(self.fixture()))
        self.assertTrue(result["integrity"]["pass"])
        self.assertEqual(len(result["per_policy"]), 8)
        self.assertEqual(len(result["baselines"]), 4)
        self.assertEqual(result["overall"]["power_cases"], 38)
        self.assertEqual(result["overall"]["safety_cases"], 44)
        for row in result["per_candidate"].values():
            self.assertEqual(row["main"]["case_count"], 36)
            self.assertAlmostEqual(row["main"]["actual_cost"]["point"], 5)
            self.assertFalse(row["criteria_pass"])
            self.assertFalse(row["main"]["qualification_pass"])
            self.assertEqual({axis:len(groups) for axis,groups in row["strata"].items()},
                             {"epsilon":3,"noise":3,"weight":2,"sign":2})
        self.assertTrue(result["per_policy"]["unsafe"]["diagnostic_only"])
        self.assertFalse(result["per_policy"]["unsafe"]["safety_all_cases_pass"])
        outputs = sr.write_outputs(result, self.directory / "export")
        self.assertTrue(all(Path(path).is_file() for path in outputs.values()))
        json.loads(Path(outputs["json"]).read_text(encoding="utf8"))

    @staticmethod
    def _args(paths):
        return paths[0], [paths[1]]

    def test_missing_duplicate_wrong_spec_and_error_each_block_pass(self):
        mutations = {
            "missing": lambda rows: rows.pop(),
            "duplicate_trial_key": lambda rows: rows.append(copy.deepcopy(rows[0])),
            "record_spec_sha256_mismatch": lambda rows: rows[0].update(spec_sha256="wrong"),
            "non_ok_trial_status": lambda rows: rows[0].update(status="error"),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                result = sr.summarize_results(*self._args(self.fixture(mutation, phase="confirmation")))
                self.assertFalse(result["integrity"]["pass"])
                self.assertFalse(result["overall"]["criteria_pass"])
                if name == "missing":
                    self.assertFalse(result["integrity"]["complete_grid_pass"])
                    self.assertEqual(result["integrity"]["missing_cells"][0]["missing"], 1)
                else:
                    self.assertGreater(result["integrity"]["issue_counts"][name], 0)
                self.assertEqual(result["per_candidate"]["c1"]["main"]["case_count"], 36)

    def test_unknown_actual_cost_has_no_fabricated_reported_mean(self):
        def mutate(rows):
            rows[0].pop("shots")
        result = sr.summarize_results(*self._args(self.fixture(mutate)))
        first = result["per_cell"][0]
        self.assertTrue(first["actual_cost_unknown"])
        self.assertIsNone(first["actual_cost"]["mean"])
        self.assertIsNone(result["per_policy"]["b1"]["actual_shots_descriptive"]["mean"])
        self.assertFalse(result["integrity"]["pass"])

    def test_spec_hash_tampering_is_rejected_before_reading_results(self):
        spec_path, raw_path = self.fixture()
        spec_path.write_text(spec_path.read_text(encoding="utf8") + " ", encoding="utf8")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            sr.summarize_results(spec_path, [raw_path])

    def test_formal_alpha_ledger_rejects_even_tiny_overspend(self):
        spec_path, unused = self.fixture(phase="confirmation")
        spec = json.loads(spec_path.read_text(encoding="utf8"))
        self.assertTrue(sr._alpha_ledger(spec)["valid"])
        ncell = len(spec["cases"]) * len(spec["policies"])
        spec["analysis"]["alpha_safety"] = .02 / (3*ncell) + 1e-18
        ledger = sr._alpha_ledger(spec)
        self.assertFalse(ledger["valid"])
        self.assertFalse(ledger["formal_simultaneous_confidence_claim_allowed"])
        self.assertNotEqual(ledger["budget_excesses_exact"]["safety"], "0")
        spec["phase"] = "development"
        legacy = sr._alpha_ledger(spec)
        self.assertFalse(legacy["valid"])
        self.assertTrue(legacy["legacy_development_rounding_only"])

    def test_geometric_bounds_keep_zero_missing_and_infinite_cells(self):
        self.assertEqual(sr.geometric_mean([0, 4], "lower"), 0)
        self.assertIsNone(sr.geometric_mean([None, 4]))
        self.assertEqual(sr.geometric_mean([None, 4], "lower"), 0)
        self.assertEqual(sr.geometric_mean([4, math.inf], "upper"), math.inf)
        self.assertAlmostEqual(sr.geometric_mean([1, 4, 16]), 4)
        self.assertLessEqual(sr.geometric_mean([1, 4, 16], "lower"), 4)
        self.assertGreaterEqual(sr.geometric_mean([1, 4, 16], "upper"), 4)


if __name__ == "__main__":
    unittest.main()
