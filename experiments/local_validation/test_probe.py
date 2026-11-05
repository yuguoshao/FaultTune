"""Behavioral and independent mathematical checks; unittest, Python 3.8+."""

import math
import unittest

import audit_math
import probe
import stress_probe


class OracleTests(unittest.TestCase):
    def test_configuration_rejects_invalid_models(self):
        for kwargs in ({"n": 0}, {"n": True}, {"n": 2, "p0": 0.4},
                       {"n": 2, "local_gain": 0.4}, {"n": 2, "delta": 0.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                probe.PublicConfig(**kwargs)

    def test_oracle_costs_and_strict_budget(self):
        oracle = probe.MeasurementOracle(probe.PublicConfig(3), None, 7, budget=2)
        oracle.measure(2)
        oracle.measure(2)
        self.assertEqual((oracle.shots, oracle.covered, oracle.counts), (2, 1, (0, 0, 2)))
        with self.assertRaises(probe.BudgetExhausted):
            oracle.measure(1)
        self.assertEqual(oracle.shots, 2)

    def test_common_random_streams_ignore_query_order(self):
        config = probe.PublicConfig(3)
        first = probe.MeasurementOracle(config, 1, 183)
        second = probe.MeasurementOracle(config, 1, 183)
        grouped = [[first.measure(j) for _ in range(20)] for j in range(3)]
        interleaved = [[], [], []]
        for _ in range(20):
            for j in (2, 0, 1):
                interleaved[j].append(second.measure(j))
        self.assertEqual(grouped, interleaved)

    def test_missing_transcript_is_not_a_budget_abstention(self):
        with self.assertRaises(probe.TranscriptExhausted):
            probe.run_trial("sequential_scan", probe.PublicConfig(2), None, 0,
                            transcripts={0: [0]})

    def test_hidden_state_does_not_change_policy_with_same_observations(self):
        transcript = {j: [0] * 20 for j in range(4)}
        normal = probe.run_trial("sequential_scan", probe.PublicConfig(4), None, 0,
                                 transcripts=transcript)
        hidden = probe.run_trial("sequential_scan", probe.PublicConfig(4), 2, 0,
                                 transcripts=transcript)
        for key in ("decision", "shots", "covered", "max_samples"):
            self.assertEqual(normal[key], hidden[key])
        self.assertNotEqual(normal["audit"]["true_delta"], hidden["audit"]["true_delta"])


class PolicyTests(unittest.TestCase):
    def test_sequential_decision_on_exact_budget_boundary(self):
        config = probe.PublicConfig(4)
        transcripts = {j: [0] * 20 for j in range(4)}
        for policy in ("sequential_scan", "sequential_round_robin"):
            with self.subTest(policy=policy):
                exact = probe.run_trial(policy, config, None, 0, budget=32, transcripts=transcripts)
                short = probe.run_trial(policy, config, None, 0, budget=31, transcripts=transcripts)
                self.assertEqual((exact["decision"], exact["shots"]), ("accept", 32))
                self.assertEqual((short["decision"], short["shots"]), ("abstain", 31))

    def test_anomaly_in_last_position_requires_coverage(self):
        config = probe.PublicConfig(4)
        transcripts = {0: [0] * 20, 1: [0] * 20, 2: [0] * 20, 3: [1] * 20}
        result = probe.run_trial("sequential_scan", config, 3, 0, transcripts=transcripts)
        self.assertEqual((result["decision"], result["covered"], result["shots"]), ("reject", 4, 28))

    def test_fixed_baseline_early_stop_is_separate_from_full_batch(self):
        config = probe.PublicConfig(4)
        depth = probe.fixed_shots_per_arm(config)
        transcripts = {j: [1] * depth for j in range(4)}
        full = probe.run_trial("full_fixed", config, 0, 0, transcripts=transcripts)
        scan = probe.run_trial("fixed_scan", config, 0, 0, transcripts=transcripts)
        self.assertEqual(full["decision"], "reject")
        self.assertEqual(scan["decision"], "reject")
        self.assertEqual(full["shots"], 4 * depth)
        self.assertEqual(scan["shots"], depth)

    def test_zero_budget_and_invalid_local_baseline(self):
        config = probe.PublicConfig(4)
        for policy in probe.POLICIES:
            result = probe.run_trial(policy, config, 3, 0, budget=0)
            self.assertEqual(result["shots"], 0)
            if policy == "local_only":
                self.assertEqual(result["decision"], "accept")
                self.assertFalse(result["audit"]["policy_valid"])
                self.assertFalse(result["audit"]["decision_correct"])
            else:
                self.assertEqual(result["decision"], "abstain")

    def test_scan_and_round_robin_have_same_unlimited_verdict(self):
        config = probe.PublicConfig(4)
        for hidden in (None, 0, 3):
            for seed in range(20):
                scan = probe.run_trial("sequential_scan", config, hidden, seed)
                rr = probe.run_trial("sequential_round_robin", config, hidden, seed)
                self.assertEqual(scan["decision"], rr["decision"])
                if scan["decision"] == "accept":
                    self.assertEqual(scan["shots"], rr["shots"])

    def test_abstention_is_neither_success_nor_wrong_decision(self):
        record = probe.run_trial("sequential_scan", probe.PublicConfig(4), None, 0, budget=0)
        aggregate = probe.aggregate_trials([record])
        self.assertEqual(aggregate["correct_unconditional"]["count"], 0)
        self.assertEqual(aggregate["wrong_decision"]["count"], 0)
        self.assertEqual(aggregate["decisions"]["abstain"]["count"], 1)

    def test_off_model_records_cannot_enter_nominal_aggregate(self):
        record = probe.run_trial("sequential_scan", probe.PublicConfig(4), 0, 1, actual_p1=0.3)
        self.assertAlmostEqual(record["audit"]["true_delta"], 0.05)
        with self.assertRaises(ValueError):
            probe.aggregate_trials([record])


class IndependentProbabilityTests(unittest.TestCase):
    def test_binomial_baseline_probabilities_against_direct_formula(self):
        for n in (8, 32, 128):
            design = probe.fixed_exact_design(probe.PublicConfig(n))
            depth, threshold = design["shots_per_arm"], design["threshold"]
            r0 = math.fsum(math.comb(depth, k) * 0.1 ** k * 0.9 ** (depth - k)
                           for k in range(threshold, depth + 1))
            r1 = math.fsum(math.comb(depth, k) * 0.4 ** k * 0.6 ** (depth - k)
                           for k in range(threshold))
            self.assertAlmostEqual(design["null_reject_probability"], 1.0 - (1.0 - r0) ** n, places=12)
            self.assertAlmostEqual(design["alternative_accept_probability"], r1 * (1.0 - r0) ** (n - 1), places=12)
            self.assertLess(design["null_reject_probability"], 0.05)
            self.assertLess(design["alternative_accept_probability"], 0.05)
            for shallower in range(1, depth):
                for k in range(1, shallower + 1):
                    q0 = math.fsum(math.comb(shallower, s) * 0.1 ** s * 0.9 ** (shallower - s)
                                   for s in range(k, shallower + 1))
                    q1 = math.fsum(math.comb(shallower, s) * 0.4 ** s * 0.6 ** (shallower - s)
                                   for s in range(k))
                    self.assertTrue(1.0 - (1.0 - q0) ** n > 0.05
                                    or q1 * (1.0 - q0) ** (n - 1) > 0.05)

    def test_absorption_reduces_to_deterministic_walks(self):
        zero = audit_math.one_site_absorption(4, true_p=0.0)
        one = audit_math.one_site_absorption(4, true_p=1.0)
        self.assertEqual((zero["prob_clear"], zero["expected_capped_samples"]), (1.0, 8.0))
        self.assertEqual((one["prob_alert"], one["expected_capped_samples"]), (1.0, 4.0))

    def test_dp_mass_error_bounds_and_expected_cost(self):
        for n in (1, 8, 32, 128):
            null = audit_math.sequential_prediction(n)
            alternative = audit_math.sequential_prediction(n, hidden=n - 1)
            bounds = audit_math.theoretical_bounds(n)
            for site in (null["normal_site"], alternative["hidden_site"]):
                self.assertAlmostEqual(site["probability_mass_sum"], 1.0, places=12)
                self.assertLess(site["unresolved_mass"], 1e-13)
            self.assertLessEqual(null["reject_probability_estimate"], 0.05)
            self.assertLessEqual(alternative["accept_probability_estimate"], 0.05)
            self.assertGreaterEqual(null["scan_expected_shots_estimate"], bounds["null_expected_shots_lower"])
            self.assertLessEqual(null["scan_expected_shots_estimate"], bounds["null_sequential_expected_shots_upper"])

    def test_assumed_distribution_matters_for_error_guarantee(self):
        missed = audit_math.sequential_prediction(32, hidden=0, true_p1=0.3)
        self.assertGreater(missed["accept_probability_estimate"], 0.2)

    def test_harm_boundary_controls_composite_alternatives(self):
        for actual in (0.25, 0.30, 0.40, 0.60):
            predicted = audit_math.sequential_prediction(32, hidden=0, true_p1=actual, p1=0.25)
            self.assertLessEqual(predicted["accept_probability_estimate"], 0.05)
        config = probe.PublicConfig(4)
        oracle = probe.MeasurementOracle(config, None, 0, transcripts={j: [0] * 30 for j in range(4)})
        self.assertEqual(stress_probe.harm_boundary_scan(config, oracle.measure), "accept")
        self.assertEqual(oracle.shots, 4 * 17)

    def test_parallel_round_bound_is_not_point_sample_bound(self):
        bounds = audit_math.simultaneous_readout_bounds(128)
        self.assertLess(bounds["maximum_rounds_lower"], 9)
        self.assertGreater(audit_math.theoretical_bounds(128)["null_expected_shots_lower"], 1400)


if __name__ == "__main__":
    unittest.main()
