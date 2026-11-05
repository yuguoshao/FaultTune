"""Independent limit cases, exact path enumeration, and certificate checks."""

import itertools
import json
import math
import unittest
from fractions import Fraction

import aggregate_math as am


class CharacteristicRateTests(unittest.TestCase):
    def test_equal_groups_reduce_to_known_analytic_rate(self):
        p0, gain, sizes = 0.1, 0.3, [8, 8]
        row = am.characteristic_null(sizes, p0, gain)
        exact_formula = 8 * am.bernoulli_kl(p0, p0 + gain / 16)
        lower, upper = row["round_information_bracket"]
        self.assertLessEqual(lower, exact_formula)
        self.assertGreaterEqual(upper, exact_formula)
        for weight in row["round_optimal_proportions_estimate"]:
            self.assertAlmostEqual(weight, 0.5, places=9)

    def test_star_witness_satisfies_feasibility_equal_information_and_kkt(self):
        row = am.characteristic_null([1, 15])
        self.assertGreaterEqual(row["witness_effect_sum"], 0.3)
        information = row["witness_group_information"]
        self.assertLess(max(information) - min(information), 1e-10)
        multipliers = [w * (q - 0.1) / (q * (1.0 - q)) for w, q in zip(
            row["round_optimal_proportions_estimate"], row["opposite_side_witness"])]
        self.assertAlmostEqual(multipliers[0], multipliers[1], places=12)
        self.assertAlmostEqual(row["round_information_estimate"], 0.0184248688103613, places=10)
        self.assertLess(row["round_optimal_proportions_estimate"][0], 0.5)

    def test_group_permutation_does_not_change_information(self):
        first = am.characteristic_null([3, 5, 8])
        second = am.characteristic_null([8, 3, 5])
        self.assertAlmostEqual(first["round_information_estimate"], second["round_information_estimate"], places=11)
        for left, right in zip(first["round_optimal_proportions_estimate"],
                               second["round_optimal_proportions_estimate"][1:] + second["round_optimal_proportions_estimate"][:1]):
            self.assertAlmostEqual(left, right, places=9)

    def test_object_information_is_partition_invariant_but_round_information_is_not(self):
        rows = [am.characteristic_null(sizes) for sizes in ([16], [8, 8], [1, 15], [1] * 16)]
        for row in rows[1:]:
            self.assertEqual(row["object_information_rate"], rows[0]["object_information_rate"])
            self.assertEqual(row["objects_lower_using_information_upper"], rows[0]["objects_lower_using_information_upper"])
        self.assertGreater(rows[0]["round_information_estimate"], rows[2]["round_information_estimate"])
        self.assertGreater(rows[2]["round_information_estimate"], rows[1]["round_information_estimate"])
        self.assertGreater(rows[1]["round_information_estimate"], rows[3]["round_information_estimate"])

    def test_small_gain_limit_recovers_square_root_allocation(self):
        gain = 0.001
        row = am.characteristic_null([1, 15], gain=gain)
        rate_limit = gain * gain / (2 * 0.1 * 0.9 * (1 + math.sqrt(15)) ** 2)
        self.assertLess(abs(row["round_information_estimate"] / rate_limit - 1.0), 0.01)
        weight_limit = 1.0 / (1.0 + math.sqrt(15))
        self.assertLess(abs(row["round_optimal_proportions_estimate"][0] - weight_limit), 0.001)

    def test_cost_lower_uses_upper_information_endpoint(self):
        row = am.characteristic_null([1, 15])
        self.assertLessEqual(row["rounds_lower_using_information_upper"],
                             row["kappa"] / row["round_information_bracket"][1])
        self.assertIn("NOT a machine-certified", row["numeric_status"])

    def test_invalid_characteristic_models_are_rejected(self):
        for sizes, options in (([], {}), ([0, 2], {}), ([True, 2], {}),
                               ([2], {"gain": 0}), ([2], {"gain": 1.8}),
                               ([2], {"delta": 0.5}), ([2], {"p0": 0})):
            with self.subTest(sizes=sizes, options=options), self.assertRaises(ValueError):
                am.characteristic_null(sizes, **options)


class ExactProbabilityTests(unittest.TestCase):
    def test_integer_lr_matches_direct_fraction_at_all_small_states(self):
        for steps in range(21):
            for successes in range(steps + 1):
                likelihood = Fraction(4) ** successes * Fraction(2, 3) ** (steps - successes)
                expected = ("clear" if likelihood <= Fraction(1, 20) else
                            "alert" if likelihood >= 320 else "unresolved")
                self.assertEqual(am.exact_lr_state(steps, successes), expected)
        self.assertEqual(am.exact_lr_state(7, 0), "unresolved")
        self.assertEqual(am.exact_lr_state(8, 0), "clear")
        self.assertEqual(am.exact_lr_state(4, 4), "unresolved")
        self.assertEqual(am.exact_lr_state(5, 5), "alert")

    def test_exact_dp_matches_exhaustive_bernoulli_paths(self):
        horizon, p = 10, Fraction(1, 5)
        masses = {"clear": Fraction(0), "alert": Fraction(0), "unresolved": Fraction(0)}
        for path in itertools.product((0, 1), repeat=horizon):
            state, successes = "unresolved", 0
            for step, outcome in enumerate(path, 1):
                successes += outcome
                # Independently compute the LR rather than using the DP classifier.
                likelihood = Fraction(4) ** successes * Fraction(2, 3) ** (step - successes)
                if likelihood <= Fraction(1, 20):
                    state = "clear"
                    break
                if likelihood >= 320:
                    state = "alert"
                    break
            total_successes = sum(path)
            masses[state] += p ** total_successes * (1 - p) ** (horizon - total_successes)
        exact = am.exact_site_absorption(1, 5, horizon)
        for name in masses:
            self.assertEqual(exact[name], masses[name])

    def test_mass_conservation_and_monotone_finite_events(self):
        for numerator, denominator in ((0, 1), (1, 1), (1, 10), (1, 5), (11, 80)):
            previous = None
            for horizon in (0, 8, 32, 256, 512):
                row = am.exact_site_absorption(numerator, denominator, horizon)
                self.assertEqual(row["clear"] + row["alert"] + row["unresolved"], 1)
                if previous:
                    self.assertGreaterEqual(row["clear"], previous["clear"])
                    self.assertGreaterEqual(row["alert"], previous["alert"])
                    self.assertLessEqual(row["unresolved"], previous["unresolved"])
                previous = row

    def test_exact_and_floating_absorption_agree_without_using_float_as_proof(self):
        for numerator, denominator in ((1, 10), (1, 5), (11, 80)):
            exact = am.exact_site_absorption(numerator, denominator, 256)
            floating = am.one_site_absorption(16, numerator / denominator,
                                              tolerance=1e-300, max_steps=256)
            for name, other in (("clear", "prob_clear"), ("alert", "prob_alert"),
                                ("unresolved", "unresolved_mass")):
                self.assertAlmostEqual(float(exact[name]), floating[other], places=12)

    def test_two_multiple_weak_effects_have_exact_finite_counterexamples(self):
        sites = {"normal": am.exact_site_absorption(1, 10, 256),
                 "weak": am.exact_site_absorption(1, 5, 256),
                 "dense": am.exact_site_absorption(11, 80, 256)}
        for composition in ({"normal": 12, "weak": 4}, {"dense": 16}):
            certificate = am.counterexample_certificate(sites, composition)
            self.assertTrue(certificate["candidate_is_harmful_exact"])
            self.assertTrue(certificate["event_probability_exceeds_delta_exact"])
            self.assertTrue(certificate["displayed_lower_exceeds_delta_exact"])
            self.assertEqual(certificate["sufficient_total_object_budget_on_event"], 4096)

    def test_printed_floor_is_an_exact_lower_bound(self):
        value = Fraction(1, 7)
        printed = am.rational_floor_record(value)
        parsed = Fraction(int(printed["numerator"]), int(printed["denominator"]))
        self.assertLessEqual(parsed, value)
        self.assertLess(value - parsed, Fraction(1, 10 ** 12))
        self.assertEqual(printed["decimal_lower"], "0.142857142857")

    def test_artifact_serializes_exact_factors_without_huge_product_integers(self):
        result = am.build_artifact(sizes=(16,), horizons=(512,))
        encoded = json.dumps(result, allow_nan=False)
        self.assertIn("exact rational floor", encoded)
        self.assertIn("NOT a machine-certified", encoded)
        self.assertEqual(len(result["finite_horizon_results"][0]["counterexamples"]), 2)

    def test_invalid_exact_dp_parameters_and_inconsistent_site_count(self):
        for args in ((1, 0, 8), (2, 1, 8), (-1, 5, 8), (1, 5, -1), (True, 5, 8)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                am.exact_site_absorption(*args)
        site = am.exact_site_absorption(1, 5, 8, n=16)
        with self.assertRaises(ValueError):
            am.counterexample_certificate({"weak": site}, {"weak": 4})


if __name__ == "__main__":
    unittest.main()
