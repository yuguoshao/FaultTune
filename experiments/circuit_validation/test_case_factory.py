"""Coverage, physical crosschecks, phase separation, and failed-cell retention."""

import itertools
import json
import math
import unittest
from collections import Counter

import case_factory as cf
from quantum_model import QuantumModel, circuit_probabilities, failure_probability


class CaseFactoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.development = cf.build_cases("development")
        cls.confirmation = cf.build_cases("confirmation")

    def test_complete_cartesian_grid_and_diagnostics_are_retained(self):
        expected = set(itertools.product(cf.EPSILONS, cf.PLUS_WEIGHTS,
                                          (noise[0] for noise in cf.NOISE_SETTINGS),
                                          ("beneficial", "harmful")))
        for rows in (self.development, self.confirmation):
            self.assertEqual(len(rows), 44)
            self.assertEqual(len({row["case_id"] for row in rows}), 44)
            self.assertEqual(Counter(row["benchmark_role"] for row in rows),
                             {"main": 36, "gray_diagnostic": 6, "blind_diagnostic": 2})
            actual = {(row["epsilon"], row["weights"][1], row["stratum"], row["category"])
                      for row in rows if row["benchmark_role"] == "main"}
            self.assertEqual(actual, expected)
            self.assertTrue(all(row["feasible"] for row in rows))
            self.assertEqual(sum(row["efficiency_eligible"] for row in rows), 36)
            json.dumps(rows, allow_nan=False)

    def test_main_physical_effects_match_phase_targets_with_margin(self):
        for rows, multiplier in ((self.development, 1.6), (self.confirmation, 1.25)):
            for row in rows:
                if row["benchmark_role"] != "main":
                    continue
                sign = -1 if row["category"] == "beneficial" else 1
                self.assertAlmostEqual(row["true_delta"], sign * multiplier * row["epsilon"], places=12)
                self.assertGreater(abs(row["true_delta"]), row["epsilon"])
                self.assertLess(abs(row["target_residual"]), 2e-12)
                self.assertGreater(row["construction"]["gamma_derivative_factor_lower_numeric"], 0)
                for parameter in ("omega", "gamma"):
                    self.assertGreaterEqual(row["model"][parameter], 0)
                    self.assertLessEqual(row["model"][parameter], math.pi / 3)

    def test_base_omega_really_has_negative_two_epsilon_effect(self):
        for row in self.confirmation:
            if row["benchmark_role"] != "main":
                continue
            snapshot = row["model"]
            model = QuantumModel(snapshot["omega"], 0, snapshot["readout_flip"], snapshot["dephasing_rate"])
            old = sum(w * failure_probability(model, "old", context) for w, context in zip(row["weights"], row["contexts"]))
            new = sum(w * failure_probability(model, "new", context) for w, context in zip(row["weights"], row["contexts"]))
            self.assertAlmostEqual(new - old, -2 * row["epsilon"], places=12)

    def test_probabilities_are_full_physical_evolution_in_registered_arm_order(self):
        for row in self.confirmation:
            snapshot = row["model"]
            model = QuantumModel(snapshot["omega"], snapshot["gamma"], snapshot["readout_flip"], snapshot["dephasing_rate"])
            expected = []
            for variant in ("old", "new"):
                for context in row["contexts"]:
                    distribution = circuit_probabilities(model, variant, context)
                    expected.append(math.fsum(distribution[1:]))
            self.assertEqual(row["probabilities"], expected)
            self.assertLess(row["physics_crosscheck_max_error"], 1e-12)
            self.assertIn("private scorer", row["truth_access"])
            self.assertFalse(row["selection_uses_algorithm_data"])

    def test_main_phase_parameters_differ_before_algorithm_data(self):
        for development, confirmation in zip(self.development[:36], self.confirmation[:36]):
            self.assertEqual(development["strata"], confirmation["strata"])
            self.assertEqual(development["model"]["omega"], confirmation["model"]["omega"])
            self.assertNotEqual(development["model"]["gamma"], confirmation["model"]["gamma"])
            self.assertNotEqual(development["model_hash"], confirmation["model_hash"])
            self.assertNotEqual(development["case_id"], confirmation["case_id"])
            # The candidate remains fixed across the worlds; only the hidden
            # physical gamma root changes between these phase cases.
            self.assertEqual(development["candidate_id"], confirmation["candidate_id"])

    def test_gray_cases_stay_out_of_efficiency_scoring(self):
        for rows in (self.development, self.confirmation):
            gray = [row for row in rows if row["benchmark_role"] == "gray_diagnostic"]
            self.assertEqual(Counter(row["category"] for row in gray), {"zero": 3, "near_boundary": 3})
            for row in gray:
                self.assertFalse(row["efficiency_eligible"])
                self.assertTrue(row["construction"]["shared_across_phases"])
                expected = 0 if row["category"] == "zero" else row["epsilon"] / 10
                self.assertAlmostEqual(row["true_delta"], expected, places=12)
                self.assertLess(abs(row["true_delta"]), row["epsilon"])

    def test_blind_worlds_match_calibration_but_reverse_mixed_target(self):
        old_world, coupled_world = self.confirmation[-2:]
        self.assertLess(old_world["true_delta"], -old_world["epsilon"])
        self.assertGreater(coupled_world["true_delta"], coupled_world["epsilon"])
        for index in (0, 2):
            self.assertEqual(old_world["probabilities"][index], coupled_world["probabilities"][index])
        self.assertEqual(old_world["candidate_id"], coupled_world["candidate_id"])
        self.assertAlmostEqual(coupled_world["true_delta"], .9 * .165993625 - .1 * .2401245, places=12)
        self.assertFalse(old_world["efficiency_eligible"])
        self.assertFalse(coupled_world["efficiency_eligible"])

    def test_infeasible_prescription_returns_a_marked_case_instead_of_dropping_it(self):
        row = cf._target_case("infeasible-test", "confirmation", .01, .2,
                              cf.NOISE_SETTINGS[1], .9, "harmful")
        self.assertFalse(row["feasible"])
        self.assertEqual(row["status"], "infeasible")
        self.assertIn("gamma_root", row["infeasible_reason"])
        self.assertTrue(row["intended_efficiency_eligible"])
        self.assertFalse(row["efficiency_eligible"])
        self.assertEqual(len(row["probabilities"]), 4)
        self.assertEqual(row["requested_delta"], .9)
        self.assertNotEqual(row["true_delta"], .9)

    def test_build_is_deterministic_and_unknown_phase_is_rejected(self):
        self.assertEqual(cf.build_cases("confirmation"), self.confirmation)
        self.assertEqual(cf.build_cases("formal"), self.confirmation)
        with self.assertRaises(ValueError):
            cf.build_cases("choose-after-results")


if __name__ == "__main__":
    unittest.main()
