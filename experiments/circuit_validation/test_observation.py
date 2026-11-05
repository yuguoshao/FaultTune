import time
import unittest
from types import SimpleNamespace

import numpy as np

from observation import (BudgetExhausted, CircuitObservationOracle, DeadlineExceeded,
                         PublicExperiment, TranscriptExhausted)


class ObservationTests(unittest.TestCase):
    def test_atomic_budget_and_invalid_action_leave_streams_unchanged(self):
        config = PublicExperiment(shot_budget=12)
        a = CircuitObservationOracle(config, [.1, .2, .3, .4], 16)
        b = CircuitObservationOracle(config, [.1, .2, .3, .4], 16)
        with self.assertRaises(BudgetExhausted):
            a.sample_pairs(7)
        for arm in (-1, 4, True):
            with self.assertRaises(ValueError):
                a.sample_arm(arm, 1)
        self.assertEqual(a.cost_snapshot(), b.cost_snapshot())
        self.assertEqual(a.sample_pairs(6), b.sample_pairs(6))
        self.assertEqual(a.shots, 12)

    def test_kth_arm_sample_is_invariant_to_batch_partition(self):
        config = PublicExperiment(shot_budget=10000)
        a = CircuitObservationOracle(config, [.2, .3, .4, .5], 31)
        b = CircuitObservationOracle(config, [.2, .3, .4, .5], 31)
        for arm in (3, 0, 2, 1):
            whole = a.sample_arm(arm, 1200)
            pieces = sum(b.sample_arm(arm, n) for n in (100, 300, 1, 799))
            self.assertEqual(whole, pieces)
        self.assertEqual(a.counts, b.counts)
        self.assertEqual(a.failures, b.failures)

    def test_pairs_use_independent_branch_streams_and_pay_both(self):
        config = PublicExperiment(weights=(.25, .75), shot_budget=100000)
        oracle = CircuitObservationOracle(config, [.5] * 4, 17)
        joint = oracle.sample_pairs(40000)["joint"]
        self.assertTrue(all(abs(x - 10000) < 500 for x in joint), joint)
        self.assertEqual(oracle.shots, 80000)
        self.assertEqual(oracle.cost_snapshot()["readout_objects"], 240000)
        self.assertEqual(oracle.counts[:2], oracle.counts[2:])
        self.assertEqual(oracle.switches, 3)
        self.assertAlmostEqual(oracle.execution_duration, 40000 * (3.2 + 4.2))

    def test_exact_joint_counts_and_public_context(self):
        config = PublicExperiment(shot_budget=8)
        fixed = {0: [0, 0, 1, 1], 2: [0, 1, 0, 1]}
        oracle = CircuitObservationOracle(config, [.9] * 4, 4, transcripts=fixed)
        self.assertEqual(oracle.sample_pairs(4, context=0)["joint"], [1, 1, 1, 1])
        self.assertEqual(oracle.counts, [4, 0, 4, 0])
        self.assertEqual(oracle.failures, [2, 0, 2, 0])
        self.assertEqual(oracle.switches, 1)

    def test_missing_transcript_preserves_atomicity_including_context_rng(self):
        config = PublicExperiment(shot_budget=100)
        oracle = CircuitObservationOracle(config, [.2] * 4, 12,
                                           transcripts={i: [] for i in range(4)})
        before = oracle.cost_snapshot()
        state = oracle._context_stream.bit_generator.state
        with self.assertRaises(TranscriptExhausted):
            oracle.sample_pairs(20)
        self.assertEqual(before, oracle.cost_snapshot())
        self.assertEqual(state, oracle._context_stream.bit_generator.state)

    def test_fixed_data_are_independent_of_hidden_truth(self):
        config = PublicExperiment(shot_budget=20)
        transcripts = {i: [i % 2] * 10 for i in range(4)}
        a = CircuitObservationOracle(config, [.1] * 4, 5, transcripts=transcripts)
        b = CircuitObservationOracle(config, [.9, .8, .1, .2], 5, transcripts=transcripts)
        self.assertEqual(a.sample_pairs(10), b.sample_pairs(10))
        self.assertEqual(a.cost_snapshot(), b.cost_snapshot())
        self.assertNotEqual(a.audit_truth()["true_delta"], b.audit_truth()["true_delta"])

    def test_expired_deadline_does_not_spend_shots(self):
        oracle = CircuitObservationOracle(PublicExperiment(), [.1] * 4, 1,
                                           deadline_epoch=time.time() - 1)
        with self.assertRaises(DeadlineExceeded):
            oracle.sample_arm(0, 1)
        self.assertEqual(oracle.shots, 0)

    def test_configuration_is_deeply_frozen_and_coefficients_match_estimand(self):
        weights = [.1, .9]
        config = PublicExperiment(weights=weights)
        weights[0] = .8
        self.assertEqual(config.coefficients, (-.1, -.9, .1, .9))
        for bad in ((0, 1), (.2, .2), (-1, 2)):
            with self.assertRaises(ValueError):
                PublicExperiment(weights=bad)

    def test_last_context_interval_includes_all_valid_uniform_draws(self):
        config = PublicExperiment(weights=(.5, .5 - 5e-13), shot_budget=2)
        oracle = CircuitObservationOracle(config, [.2] * 4, 8,
                                           transcripts={1: [0], 3: [1]})
        oracle._context_stream = SimpleNamespace(
            bit_generator=SimpleNamespace(state={"fixed": True}),
            random=lambda count: np.full(count, np.nextafter(1.0, 0.0)))
        result = oracle.sample_pairs(1)
        self.assertEqual(result["context_counts"], [0, 1])
        self.assertEqual(result["joint"], [0, 1, 0, 0])
        self.assertEqual(oracle.counts, [0, 1, 0, 1])
        self.assertEqual(oracle.shots, 2)


if __name__ == "__main__":
    unittest.main()
