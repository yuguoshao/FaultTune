"""Independent behavioral/math checks for aggregate validation; Python 3.8+."""

import itertools
import math
import unittest
from dataclasses import FrozenInstanceError

import aggregate_probe as aggregate
import group_oracle as group


def enumerate_poisson_binomial(probabilities):
    """Exact finite enumeration of heterogeneous independent Bernoulli outcomes."""
    masses = [[] for _ in range(len(probabilities) + 1)]
    for bits in itertools.product((0, 1), repeat=len(probabilities)):
        mass = math.prod(p if bit else 1.0 - p for bit, p in zip(bits, probabilities))
        masses[sum(bits)].append(mass)
    return [math.fsum(row) for row in masses]


def direct_chernoff(count, observations, mean):
    """Closed Chernoff tail expression, independent of aggregate.kl()."""
    if count == 0:
        return (1.0 - mean) ** observations
    if count == observations:
        return mean ** observations
    x = count / observations
    divergence = x * math.log(x / mean) + (1.0 - x) * math.log((1.0 - x) / (1.0 - mean))
    return math.exp(-observations * divergence)


def snapshot(oracle):
    return oracle.rounds, oracle.objects, oracle.switches, oracle.counts


class GroupOracleTests(unittest.TestCase):
    def test_configuration_is_deeply_frozen_and_rejects_invalid_partitions(self):
        edges, partition = [[0, 1]], [[0, 2], [1]]
        config = group.PublicGroupConfig(3, conflict_edges=edges, partition=partition)
        edges[0][0] = 2
        partition[0].append(1)
        self.assertEqual(config.conflict_edges, ((0, 1),))
        self.assertEqual(config.partition, ((0, 2), (1,)))
        with self.assertRaises(FrozenInstanceError):
            config.n = 4
        for bad in (((0, 1), (2,)), ((0,), (1,)), ((0, 2), (1, 2)), ((), (0, 1, 2))):
            with self.subTest(partition=bad), self.assertRaises(ValueError):
                group.PublicGroupConfig(3, conflict_edges=((0, 1),), partition=bad)

    def test_graph_and_partition_builders_have_expected_legal_covers(self):
        self.assertEqual(group.greedy_partition(5, group.empty_graph(5)), (tuple(range(5)),))
        self.assertEqual(group.greedy_partition(5, group.path_graph(5)), ((0, 2, 4), (1, 3)))
        self.assertEqual(group.greedy_partition(5, group.clique_graph(5)), group.singleton_partition(5))
        edges = group.grid_graph(3, 4)
        self.assertEqual(len(edges), 3 * 3 + 2 * 4)
        partition = group.checkerboard_partition(3, 4)
        self.assertEqual(sorted(sum(partition, ())), list(range(12)))
        self.assertEqual(len(partition), 2)
        config = group.PublicGroupConfig(12, conflict_edges=edges, partition=partition)
        for action in partition:
            self.assertEqual(config.validate_group(action), action)
        with self.assertRaises(ValueError):
            group.full_partition(2, group.path_graph(2))

    def test_illegal_actions_do_not_consume_cost_or_randomness(self):
        config = group.PublicGroupConfig(4, conflict_edges=group.path_graph(4))
        oracle = group.GroupMeasurementOracle(config, [0.1, 0.2, 0.3, 0.4], 16)
        control = group.GroupMeasurementOracle(config, [0.1, 0.2, 0.3, 0.4], 16)
        for action in ((), (0, 0), (0, 1), (-1,), (4,), (True,)):
            with self.subTest(action=action), self.assertRaises(ValueError):
                oracle.measure(action)
            self.assertEqual(snapshot(oracle), (0, 0, 0, (0, 0, 0, 0)))
        for _ in range(20):
            self.assertEqual(oracle.measure((0, 2)), control.measure((0, 2)))

    def test_both_budgets_preflight_atomically_and_allow_exact_final_action(self):
        config = group.PublicGroupConfig(3)
        oracle = group.GroupMeasurementOracle(config, [0.2, 0.4, 0.6], 13,
                                              round_budget=1, object_budget=2)
        control = group.GroupMeasurementOracle(config, [0.2, 0.4, 0.6], 13)
        with self.assertRaises(group.BudgetExhausted):
            oracle.measure((0, 1, 2))
        self.assertEqual(snapshot(oracle), (0, 0, 0, (0, 0, 0)))
        self.assertEqual(oracle.measure((0, 2)), control.measure((0, 2)))
        self.assertEqual(snapshot(oracle), (1, 2, 0, (1, 0, 1)))
        with self.assertRaises(group.BudgetExhausted):
            oracle.measure((1,))
        self.assertEqual(snapshot(oracle), (1, 2, 0, (1, 0, 1)))
        rounds_only = group.GroupMeasurementOracle(config, [0.2] * 3, 13,
                                                   round_budget=0, object_budget=100)
        with self.assertRaises(group.BudgetExhausted):
            rounds_only.measure((0,))
        self.assertEqual(snapshot(rounds_only), (0, 0, 0, (0, 0, 0)))

    def test_missing_late_transcript_does_not_partially_consume_an_action(self):
        oracle = group.GroupMeasurementOracle(group.PublicGroupConfig(3), [0.1] * 3, 0,
                                              transcripts={0: [1, 0], 1: [0], 2: []})
        with self.assertRaises(group.TranscriptExhausted):
            oracle.measure((0, 1, 2))
        self.assertEqual(snapshot(oracle), (0, 0, 0, (0, 0, 0)))
        self.assertEqual(oracle.measure((1, 0)), (1, 0))
        with self.assertRaises(group.TranscriptExhausted):
            oracle.measure((0, 1))
        self.assertEqual(snapshot(oracle), (1, 2, 0, (1, 1, 0)))
        self.assertEqual(oracle.measure((0,)), (0,))

    def test_common_random_streams_are_invariant_under_grouping_and_order(self):
        config = group.PublicGroupConfig(4)
        probabilities = [0.05, 0.2, 0.55, 0.9]
        batch = group.GroupMeasurementOracle(config, probabilities, 117)
        mixed = group.GroupMeasurementOracle(config, probabilities, 117)
        whole = [[] for _ in probabilities]
        split = [[] for _ in probabilities]
        for _ in range(40):
            for arm, value in enumerate(batch.measure((3, 1, 0, 2))):
                whole[arm].append(value)
            split[3].append(mixed.measure((3,))[0])
            for arm, value in zip((0, 2), mixed.measure((2, 0))):
                split[arm].append(value)
            split[1].append(mixed.measure((1,))[0])
        self.assertEqual(whole, split)
        self.assertEqual(batch.objects, mixed.objects)
        self.assertEqual((batch.rounds, mixed.rounds), (40, 120))

    def test_canonical_actions_define_switches_and_output_order(self):
        config = group.PublicGroupConfig(3)
        oracle = group.GroupMeasurementOracle(config, [0.1] * 3, 0,
                                              transcripts={0: [0, 0], 1: [1], 2: [1, 1]})
        self.assertEqual(oracle.measure((2, 0)), (0, 1))
        self.assertEqual(oracle.switches, 0)
        self.assertEqual(oracle.measure((0, 2)), (0, 1))
        self.assertEqual(oracle.switches, 0)
        self.assertEqual(oracle.measure((1,)), (1,))
        self.assertEqual((oracle.switches, oracle.objects), (1, 5))

    def test_audit_uses_the_sum_of_weak_changes(self):
        config = group.PublicGroupConfig(8, local_gain=0.15)
        oracle = group.GroupMeasurementOracle(config, [0.125] * 8, 0)
        self.assertAlmostEqual(oracle.audit()["remote_change"], 0.2)
        self.assertAlmostEqual(oracle.audit()["true_delta"], 0.05)
        self.assertEqual(oracle.audit()["expected_decision"], "reject")
        self.assertLess(0.125, config.p0 + config.local_gain)
        self.assertEqual(oracle.objects, 0)


class ChernoffIntervalTests(unittest.TestCase):
    def test_endpoint_intervals_match_independent_closed_forms(self):
        for observations in (1, 7, 32):
            for alpha in (0.05, 0.2):
                radius = math.log(2.0 / alpha) / observations
                all_zero = aggregate.kl_interval(0, observations, alpha)
                all_one = aggregate.kl_interval(observations, observations, alpha)
                self.assertEqual(all_zero[0], 0.0)
                self.assertEqual(all_one[1], 1.0)
                self.assertGreaterEqual(all_zero[1], -math.expm1(-radius))
                self.assertLessEqual(all_one[0], math.exp(-radius))
                self.assertAlmostEqual(all_zero[1], -math.expm1(-radius), places=9)
                self.assertAlmostEqual(all_one[0], math.exp(-radius), places=9)
        lower, upper = aggregate.kl_interval(3, 7, 0.1)
        for endpoint in (lower, upper):
            p = 3.0 / 7.0
            direct = 7 * (p * math.log(p / endpoint) + (1 - p) * math.log((1 - p) / (1 - endpoint)))
            self.assertGreaterEqual(direct, math.log(20))
            self.assertAlmostEqual(direct, math.log(20), places=8)

    def test_invalid_counts_and_error_budgets_are_rejected(self):
        for args in ((0, 0, 0.1), (-1, 4, 0.1), (5, 4, 0.1), (1, 4, 0), (1, 4, 1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                aggregate.kl_interval(*args)

    def test_heterogeneous_poisson_binomial_tails_obey_direct_chernoff(self):
        for probabilities in ((0.02, 0.08, 0.17, 0.39, 0.64, 0.91),
                              (0.01, 0.03, 0.09, 0.12, 0.21, 0.45, 0.73, 0.95)):
            masses = enumerate_poisson_binomial(probabilities)
            self.assertAlmostEqual(math.fsum(masses), 1.0, places=13)
            mean = math.fsum(probabilities) / len(probabilities)
            n = len(probabilities)
            for count in range(n + 1):
                if count / n < mean:
                    tail = math.fsum(masses[:count + 1])
                elif count / n > mean:
                    tail = math.fsum(masses[count:])
                else:
                    continue
                with self.subTest(probabilities=probabilities, count=count):
                    self.assertLessEqual(tail, direct_chernoff(count, n, mean) + 1e-13)

    def test_heterogeneous_exact_enumeration_checks_interval_coverage(self):
        saw_nonzero_failure = False
        for probabilities in ((0.02, 0.08, 0.17, 0.39, 0.64, 0.91),
                              (0.05, 0.10, 0.15, 0.20, 0.30, 0.35, 0.40, 0.60)):
            masses = enumerate_poisson_binomial(probabilities)
            mean = math.fsum(probabilities) / len(probabilities)
            for alpha in (0.05, 0.2, 0.4):
                errors = []
                for count, mass in enumerate(masses):
                    lower, upper = aggregate.kl_interval(count, len(probabilities), alpha)
                    if not lower <= mean <= upper:
                        errors.append(mass)
                failure = math.fsum(errors)
                saw_nonzero_failure |= failure > 0
                self.assertLessEqual(failure, alpha + 1e-13)
        self.assertTrue(saw_nonzero_failure, "coverage check must exercise excluded outcomes")

    def test_checkpoint_budgets_telescope_across_groups(self):
        for groups in (1, 2, 7):
            for checkpoints in (1, 4, 20):
                allocation = groups * math.fsum(aggregate.checkpoint_alpha(0.05, k, groups)
                                                for k in range(checkpoints))
                self.assertAlmostEqual(allocation, 0.05 * (1.0 - 1.0 / (checkpoints + 1)), places=14)
                self.assertLess(allocation, 0.05)


class AggregatePolicyTests(unittest.TestCase):
    def test_same_observations_hide_the_true_probability_vector_from_policies(self):
        config = group.PublicGroupConfig(4, local_gain=0.3, conflict_edges=group.path_graph(4))
        transcripts = {j: [0] * 256 for j in range(4)}
        for policy in aggregate.POLICIES:
            with self.subTest(policy=policy):
                beneficial = aggregate.run_trial(policy, config, [0.1] * 4, 1,
                                                 object_budget=1024, transcripts=transcripts)
                harmful = aggregate.run_trial(policy, config, [0.3] * 4, 1,
                                              object_budget=1024, transcripts=transcripts)
                for key in ("decision", "interval", "checkpoint_counts", "rounds", "objects", "switches", "counts"):
                    self.assertEqual(beneficial[key], harmful[key])
                self.assertNotEqual(beneficial["audit"]["expected_decision"], harmful["audit"]["expected_decision"])

    def test_complete_sweeps_match_across_empty_path_and_clique(self):
        for probabilities in ([0.03, 0.05, 0.08, 0.1], [0.05, 0.2, 0.4, 0.8]):
            records = []
            for graph in (group.empty_graph, group.path_graph, group.clique_graph):
                config = group.PublicGroupConfig(4, local_gain=0.3, conflict_edges=graph(4))
                result = aggregate.run_trial("sweep_kl", config, probabilities, 135,
                                             round_budget=131072, object_budget=131072)
                self.assertNotEqual(result["decision"], "abstain")
                sweeps = result["counts"][0]
                self.assertEqual(result["counts"], (sweeps,) * 4)
                self.assertEqual(result["objects"], 4 * sweeps)
                self.assertEqual(result["rounds"], len(config.partition) * sweeps)
                self.assertEqual(result["checkpoint_counts"], [sweeps] * len(config.partition))
                records.append(result)
            for result in records[1:]:
                for key in ("decision", "objects", "counts", "interval"):
                    self.assertEqual(result[key], records[0][key])

    def test_partial_sweep_budget_keeps_previous_checkpoint_and_abstains(self):
        config = group.PublicGroupConfig(4, local_gain=0.3, conflict_edges=group.path_graph(4))
        transcripts = {j: [0] * 256 for j in range(4)}
        complete = aggregate.run_trial("sweep_kl", config, [0.0] * 4, 0, transcripts=transcripts)
        self.assertEqual(complete["decision"], "accept")
        result = aggregate.run_trial("sweep_kl", config, [0.0] * 4, 0,
                                     round_budget=complete["rounds"] - 1, transcripts=transcripts)
        self.assertEqual(result["decision"], "abstain")
        prior = complete["counts"][0] // 2
        self.assertGreater(prior, 0)
        self.assertEqual(result["checkpoint_counts"], [prior] * 2)
        self.assertEqual(result["objects"], complete["objects"] - len(config.partition[-1]))
        self.assertNotEqual(result["counts"][0], result["counts"][1])
        index = int(math.log2(prior))
        alpha = config.delta / ((index + 1) * (index + 2))
        upper_mean = -math.expm1(-math.log(2 / alpha) / (config.n * prior))
        expected_upper = config.n * upper_mean - config.n * config.p0 - config.local_gain
        self.assertAlmostEqual(result["interval"][1], expected_upper, places=9)
        self.assertLessEqual(result["interval"][0], 0)
        self.assertGreaterEqual(result["interval"][1], 0)

    def test_adaptive_groups_return_legal_actions_and_consistent_checkpoint_bounds(self):
        edges = tuple((0, j) for j in range(1, 7))
        config = group.PublicGroupConfig(7, local_gain=0.5, conflict_edges=edges)
        transcripts = {0: [1, 0] * 1024}
        transcripts.update({j: [0] * 2048 for j in range(1, 7)})
        oracle = group.GroupMeasurementOracle(config, [0.5] + [0.0] * 6, 0,
                                              object_budget=14000, transcripts=transcripts)
        actions = []

        def measure(action):
            normalized = config.validate_group(action)
            self.assertIn(normalized, config.partition)
            values = oracle.measure(action)
            actions.append(normalized)
            return values

        result = aggregate.group_kl_adaptive(config, measure)
        self.assertEqual(result["decision"], "accept")
        self.assertLess(result["interval"][1], 0.0)
        self.assertLessEqual(result["interval"][0], oracle.audit()["true_delta"])
        self.assertGreaterEqual(result["interval"][1], oracle.audit()["true_delta"])
        checkpoints = result["checkpoint_counts"]
        self.assertEqual(oracle.rounds, sum(checkpoints))
        self.assertEqual(oracle.objects, sum(len(action) * count for action, count in zip(config.partition, checkpoints)))
        self.assertEqual(oracle.switches, sum(left != right for left, right in zip(actions, actions[1:])))
        for action, count in zip(config.partition, checkpoints):
            self.assertGreater(count, 0)
            self.assertEqual(count & (count - 1), 0)
            for arm in action:
                self.assertEqual(oracle.counts[arm], count)

    def test_partial_group_block_does_not_certify_unallocated_partial_data(self):
        config = group.PublicGroupConfig(4, local_gain=0.3, conflict_edges=group.path_graph(4))
        transcripts = {j: [0] * 16 for j in range(4)}
        for policy in ("group_kl_balanced", "group_kl_adaptive"):
            with self.subTest(policy=policy):
                result = aggregate.run_trial(policy, config, [0.0] * 4, 0,
                                             round_budget=5, transcripts=transcripts)
                self.assertEqual(result["decision"], "abstain")
                self.assertEqual(result["checkpoint_counts"], [2, 2])
                self.assertEqual(result["counts"], (3, 2, 3, 2))
                self.assertEqual((result["rounds"], result["objects"]), (5, 10))
                alpha = config.delta / (2 * 2 * 3)
                upper_mean = -math.expm1(-math.log(2 / alpha) / 4)
                expected_upper = 4 * upper_mean - 4 * config.p0 - config.local_gain
                self.assertAlmostEqual(result["interval"][1], expected_upper, places=9)
                self.assertLessEqual(result["interval"][0], 0)
                self.assertGreaterEqual(result["interval"][1], 0)


if __name__ == "__main__":
    unittest.main()
