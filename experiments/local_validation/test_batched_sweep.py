"""Scheduling-equivalence and partial-checkpoint tests, Python 3.8+."""

import math
import unittest

import aggregate_probe as aggregate
from batched_sweep import sweep_kl_batched
from group_oracle import (GroupMeasurementOracle, PublicGroupConfig,
                          TranscriptExhausted, clique_graph, empty_graph, path_graph)


def run(policy, config, probabilities, seed=135, round_budget=None,
        object_budget=None, transcripts=None):
    oracle = GroupMeasurementOracle(config, probabilities, seed, round_budget,
                                    object_budget, transcripts)
    actions = []

    def measure(action):
        values = oracle.measure(action)
        actions.append(tuple(sorted(action)))
        return values

    return policy(config, measure), oracle, actions


def zero_checkpoint_interval(config, sweeps):
    """Independent closed-form all-zero Chernoff interval (without rounding margin)."""
    k = sweeps.bit_length() - 1
    alpha = config.delta / ((k + 1) * (k + 2))
    upper_mean = -math.expm1(-math.log(2.0 / alpha) / (config.n * sweeps))
    offset = config.n * config.p0 + config.local_gain
    return -offset, config.n * upper_mean - offset


class BatchedSweepTests(unittest.TestCase):
    def test_complete_checkpoints_match_original_on_common_random_streams(self):
        n = 8
        graphs = (empty_graph(n), path_graph(n), clique_graph(n),
                  tuple((0, j) for j in range(1, n)))
        for edges in graphs:
            config = PublicGroupConfig(n, local_gain=0.3, conflict_edges=edges)
            for probabilities in ([0.1] * n, [0.1, 0.13, 0.15, 0.17, 0.2, 0.22, 0.25, 0.3]):
                with self.subTest(groups=len(config.partition), probabilities=probabilities):
                    old, before, _ = run(aggregate.sweep_kl, config, probabilities)
                    new, after, _ = run(sweep_kl_batched, config, probabilities)
                    self.assertEqual(old, new)
                    self.assertIn(new["decision"], ("accept", "reject"))
                    self.assertEqual(before.counts, after.counts)
                    self.assertEqual(before.objects, after.objects)
                    self.assertEqual(before.rounds, after.rounds)
                    self.assertEqual(before.audit(), after.audit())

    def test_switch_count_is_logarithmic_in_the_per_arm_depth(self):
        for n, edges in ((4, path_graph(4)), (4, clique_graph(4)),
                         (7, tuple((0, j) for j in range(1, 7)))):
            config = PublicGroupConfig(n, local_gain=0.3, conflict_edges=edges)
            transcripts = {j: [0] * 2048 for j in range(n)}
            result, oracle, actions = run(sweep_kl_batched, config, [0.0] * n,
                                          transcripts=transcripts)
            depth = result["checkpoint_counts"][0]
            groups = len(config.partition)
            self.assertGreater(groups, 1)
            self.assertGreater(depth, 2)
            self.assertEqual(depth & (depth - 1), 0)
            self.assertEqual(oracle.switches, groups * depth.bit_length() - 1)
            self.assertEqual(oracle.switches, sum(a != b for a, b in zip(actions, actions[1:])))
            self.assertEqual(oracle.rounds, groups * depth)
            self.assertEqual(oracle.objects, n * depth)
            self.assertLess(oracle.switches, oracle.rounds - 1)
            self.assertTrue(all(action in config.partition for action in actions))

    def test_one_group_never_switches(self):
        config = PublicGroupConfig(4, local_gain=0.3)
        result, oracle, _ = run(sweep_kl_batched, config, [0.0] * 4,
                                transcripts={j: [0] * 256 for j in range(4)})
        self.assertEqual(result["decision"], "accept")
        self.assertGreater(oracle.rounds, 1)
        self.assertEqual(oracle.switches, 0)

    def test_budget_before_first_complete_checkpoint_retains_initial_interval(self):
        config = PublicGroupConfig(4, local_gain=0.3, conflict_edges=path_graph(4))
        transcripts = {j: [0] * 32 for j in range(4)}
        for budget in (0, 1):
            result, oracle, _ = run(sweep_kl_batched, config, [0.0] * 4,
                                    round_budget=budget, transcripts=transcripts)
            self.assertEqual(result["decision"], "abstain")
            self.assertEqual(result["checkpoint_counts"], [0, 0])
            for got, wanted in zip(result["interval"], (-0.7, 3.3)):
                self.assertAlmostEqual(got, wanted, places=14)
            self.assertEqual((oracle.rounds, oracle.objects), (budget, 2 * budget))

    def test_partial_later_block_retains_last_balanced_certificate(self):
        config = PublicGroupConfig(4, local_gain=0.3, conflict_edges=path_graph(4))
        result, oracle, _ = run(sweep_kl_batched, config, [0.0] * 4,
                                round_budget=6, transcripts={j: [0] * 64 for j in range(4)})
        self.assertEqual(result["decision"], "abstain")
        self.assertEqual(result["checkpoint_counts"], [2, 2])
        self.assertEqual(oracle.counts, (4, 2, 4, 2))
        self.assertEqual((oracle.rounds, oracle.objects), (6, 12))
        expected = zero_checkpoint_interval(config, 2)
        for got, wanted in zip(result["interval"], expected):
            self.assertAlmostEqual(got, wanted, places=9)
        self.assertLess(result["interval"][0], 0)
        self.assertGreater(result["interval"][1], 0)

    def test_unequal_groups_may_charge_different_partial_costs_but_same_certificate(self):
        config = PublicGroupConfig(7, local_gain=0.5,
                                   conflict_edges=tuple((0, j) for j in range(1, 7)))
        transcripts = {j: [0] * 64 for j in range(7)}
        old, before, _ = run(aggregate.sweep_kl, config, [0.0] * 7,
                             object_budget=30, transcripts=transcripts)
        new, after, _ = run(sweep_kl_batched, config, [0.0] * 7,
                            object_budget=30, transcripts=transcripts)
        self.assertEqual(old, new)
        self.assertEqual(new["decision"], "abstain")
        self.assertEqual(new["checkpoint_counts"], [4, 4])
        self.assertEqual((before.objects, after.objects), (29, 30))
        self.assertEqual(before.counts, (5, 4, 4, 4, 4, 4, 4))
        self.assertEqual(after.counts, (6, 4, 4, 4, 4, 4, 4))

    def test_exact_final_budget_keeps_decision_and_one_less_cannot_certify(self):
        config = PublicGroupConfig(4, local_gain=0.3, conflict_edges=path_graph(4))
        transcripts = {j: [0] * 256 for j in range(4)}
        full, oracle, _ = run(sweep_kl_batched, config, [0.0] * 4, transcripts=transcripts)
        exact, exact_oracle, _ = run(sweep_kl_batched, config, [0.0] * 4,
                                     round_budget=oracle.rounds, object_budget=oracle.objects,
                                     transcripts=transcripts)
        self.assertEqual(full, exact)
        self.assertEqual(exact["decision"], "accept")
        self.assertEqual((exact_oracle.rounds, exact_oracle.objects), (oracle.rounds, oracle.objects))
        for budgets in ({"round_budget": oracle.rounds - 1}, {"object_budget": oracle.objects - 1}):
            result, _, _ = run(sweep_kl_batched, config, [0.0] * 4,
                                transcripts=transcripts, **budgets)
            self.assertEqual(result["decision"], "abstain")
            self.assertEqual(result["checkpoint_counts"], [oracle.counts[0] // 2] * 2)

    def test_fixed_observations_do_not_reveal_the_hidden_probabilities(self):
        config = PublicGroupConfig(4, local_gain=0.3, conflict_edges=path_graph(4))
        transcripts = {j: [0] * 256 for j in range(4)}
        normal, before, actions_before = run(sweep_kl_batched, config, [0.1] * 4,
                                             transcripts=transcripts)
        harmful, after, actions_after = run(sweep_kl_batched, config, [0.3] * 4,
                                            transcripts=transcripts)
        self.assertEqual(normal, harmful)
        self.assertEqual(actions_before, actions_after)
        self.assertEqual(before.counts, after.counts)
        self.assertNotEqual(before.audit()["expected_decision"], after.audit()["expected_decision"])

    def test_missing_transcript_is_not_reported_as_budget_abstention(self):
        config = PublicGroupConfig(4, local_gain=0.3, conflict_edges=path_graph(4))
        oracle = GroupMeasurementOracle(config, [0.1] * 4, 0,
                                        transcripts={0: [0], 1: [], 2: [0], 3: []})
        with self.assertRaises(TranscriptExhausted):
            sweep_kl_batched(config, oracle.measure)
        self.assertEqual((oracle.rounds, oracle.objects, oracle.counts), (1, 2, (1, 0, 1, 0)))


if __name__ == "__main__":
    unittest.main()
