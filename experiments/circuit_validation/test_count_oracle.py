"""Exact small-law enumeration, moment checks, and ledger/atomicity tests."""

import copy
import math
import time
import unittest
from fractions import Fraction
from types import SimpleNamespace

import numpy as np

from count_oracle import CountObservationOracle
from observation import BudgetExhausted, CircuitObservationOracle, DeadlineExceeded, PublicExperiment


def multinomial_mass(cells, probabilities):
    numerator = math.factorial(sum(cells))
    for count in cells:
        numerator //= math.factorial(count)
    result = Fraction(numerator)
    for count, probability in zip(cells, probabilities):
        result *= probability ** count
    return result


def cells_of_size(n):
    for a in range(n + 1):
        for b in range(n - a + 1):
            for c in range(n - a - b + 1):
                yield (a, b, c, n - a - b - c)


def exact_aggregate_law(n, p_old, p_new):
    """Independent rational enumeration of the binomial/hypergeometric law."""
    result = {}
    for old in range(n + 1):
        mass_old = math.comb(n, old) * p_old ** old * (1 - p_old) ** (n - old)
        for new in range(n + 1):
            mass_new = math.comb(n, new) * p_new ** new * (1 - p_new) ** (n - new)
            for overlap in range(max(0, old + new - n), min(old, new) + 1):
                conditional = Fraction(math.comb(old, overlap) * math.comb(n - old, new - overlap),
                                       math.comb(n, new))
                cells = (n - old - new + overlap, new - overlap, old - overlap, overlap)
                result[cells] = result.get(cells, Fraction(0)) + mass_old * mass_new * conditional
    return result


class ExactCountLawTests(unittest.TestCase):
    def test_small_conditional_law_matches_independent_pairs_exactly(self):
        for n in range(6):
            for old, new in ((Fraction(1, 3), Fraction(2, 5)),
                             (Fraction(0), Fraction(3, 4)), (Fraction(1), Fraction(1))):
                probabilities = ((1 - old) * (1 - new), (1 - old) * new, old * (1 - new), old * new)
                aggregate = exact_aggregate_law(n, old, new)
                self.assertEqual(sum(aggregate.values()), 1)
                for cells in cells_of_size(n):
                    self.assertEqual(aggregate.get(cells, 0), multinomial_mass(cells, probabilities))

    def test_iid_context_mixture_law_matches_full_enumeration(self):
        n, weight = 3, Fraction(2, 7)
        branches = ((Fraction(1, 4), Fraction(2, 3)), (Fraction(3, 5), Fraction(1, 6)))
        conditional = [((1 - p) * (1 - q), (1 - p) * q, p * (1 - q), p * q) for p, q in branches]
        target = tuple(weight * left + (1 - weight) * right for left, right in zip(*conditional))
        result = {}
        for first_n in range(n + 1):
            context_mass = math.comb(n, first_n) * weight ** first_n * (1 - weight) ** (n - first_n)
            first = exact_aggregate_law(first_n, *branches[0])
            second = exact_aggregate_law(n - first_n, *branches[1])
            for left, left_mass in first.items():
                for right, right_mass in second.items():
                    cells = tuple(a + b for a, b in zip(left, right))
                    result[cells] = result.get(cells, Fraction(0)) + context_mass * left_mass * right_mass
        for cells in cells_of_size(n):
            self.assertEqual(result[cells], multinomial_mass(cells, target))
        self.assertEqual(sum(result.values()), 1)

    def test_actual_oracle_reconstructs_every_feasible_small_margin(self):
        config = PublicExperiment(shot_budget=100)
        for n in range(1, 5):
            for old in range(n + 1):
                for new in range(n + 1):
                    for both in range(max(0, old + new - n), min(old, new) + 1):
                        oracle = CountObservationOracle(config, [.25, .2, .6, .7], 1)
                        calls = []
                        def binomial_old(count, probability):
                            calls.append(("old", count, probability))
                            return old
                        def binomial_new(count, probability):
                            calls.append(("new", count, probability))
                            return new
                        def hypergeometric(good, bad, sample):
                            calls.append(("pair", good, bad, sample))
                            return both
                        oracle._streams[0] = SimpleNamespace(binomial=binomial_old)
                        oracle._streams[2] = SimpleNamespace(binomial=binomial_new)
                        oracle._pairing_streams[0] = SimpleNamespace(hypergeometric=hypergeometric)
                        result = oracle.sample_pairs(n, context=0)
                        self.assertEqual(calls, [("old", n, .25), ("new", n, .6), ("pair", old, n - old, new)])
                        self.assertEqual(result["joint"], [n - old - new + both, new - both, old - both, both])
                        self.assertEqual(oracle.failures, [old, 0, new, 0])


class SamplingAndLedgerTests(unittest.TestCase):
    def test_fixed_context_old_new_moments_and_independence(self):
        repetitions, pairs, p_old, p_new = 4000, 128, .3, .65
        oracle = CountObservationOracle(PublicExperiment(shot_budget=2 * repetitions * pairs),
                                        [p_old, .1, p_new, .9], 9123)
        cells = np.asarray([oracle.sample_pairs(pairs, context=0)["joint"] for unused in range(repetitions)])
        old, new = cells[:, 2] + cells[:, 3], cells[:, 1] + cells[:, 3]
        variance_old, variance_new = pairs * p_old * (1 - p_old), pairs * p_new * (1 - p_new)
        self.assertLess(abs(float(old.mean()) - pairs * p_old), 6 * math.sqrt(variance_old / repetitions))
        self.assertLess(abs(float(new.mean()) - pairs * p_new), 6 * math.sqrt(variance_new / repetitions))
        self.assertLess(abs(float(np.var(old, ddof=1)) / variance_old - 1), .1)
        self.assertLess(abs(float(np.var(new, ddof=1)) / variance_new - 1), .1)
        covariance = float(np.cov(old, new, ddof=1)[0, 1])
        self.assertLess(abs(covariance), 6 * math.sqrt(variance_old * variance_new / repetitions))
        probabilities = ((1 - p_old) * (1 - p_new), (1 - p_old) * p_new,
                         p_old * (1 - p_new), p_old * p_new)
        for observed, probability in zip(cells.mean(axis=0), probabilities):
            self.assertLess(abs(float(observed) - pairs * probability),
                            6 * math.sqrt(pairs * probability * (1 - probability) / repetitions))
        self.assertEqual(oracle.shots, 2 * repetitions * pairs)

    def test_iid_context_frequencies_and_paid_pieces(self):
        oracle = CountObservationOracle(PublicExperiment(weights=(.2, .8), shot_budget=200000),
                                        [0, 1, 1, 0], 918, keep_events=True)
        result = oracle.sample_pairs(100000)
        first, second = result["context_counts"]
        self.assertLess(abs(first - 20000), 6 * math.sqrt(100000 * .2 * .8))
        self.assertEqual(result["joint"], [0, first, second, 0])
        self.assertEqual(oracle.counts, [first, second, first, second])
        self.assertEqual(oracle.failures, [0, second, first, 0])
        self.assertEqual(oracle.shots, 200000)
        self.assertEqual(oracle.switches, 3)
        self.assertAlmostEqual(oracle.execution_duration, 100000 * (3.2 + 4.2))
        self.assertEqual(oracle.cost_snapshot()["readout_objects"], 600000)
        self.assertEqual(oracle.events[0]["joint"], result["joint"])

    def test_uniform_oracle_and_count_oracle_share_exact_deterministic_ledger(self):
        config = PublicExperiment(shot_budget=100)
        aggregate = CountObservationOracle(config, [0, 1, 1, 0], 13, keep_events=True)
        explicit = CircuitObservationOracle(config, [0, 1, 1, 0], 13, keep_events=True)
        for oracle in (aggregate, explicit):
            oracle.sample_arm(0, 5)
            oracle.sample_pairs(11, context=1)
            oracle.sample_pairs(7, context=0)
            oracle.sample_arm(3, 9)
        self.assertEqual(aggregate.cost_snapshot(), explicit.cost_snapshot())
        self.assertEqual(aggregate.events, explicit.events)

    def test_budget_invalid_action_and_deadline_are_atomic(self):
        config = PublicExperiment(shot_budget=20)
        oracle = CountObservationOracle(config, [.1, .2, .3, .4], 52)
        def state():
            return (oracle.cost_snapshot(), copy.deepcopy(oracle._context_stream.bit_generator.state),
                    [copy.deepcopy(rng.bit_generator.state) for rng in oracle._streams],
                    [copy.deepcopy(rng.bit_generator.state) for rng in oracle._pairing_streams])
        before = state()
        invalid_calls = ((BudgetExhausted, lambda: oracle.sample_pairs(11)),
                         (BudgetExhausted, lambda: oracle.sample_arm(0, 21)),
                         (ValueError, lambda: oracle.sample_arm(True, 1)),
                         (ValueError, lambda: oracle.sample_arm(0, 0)),
                         (ValueError, lambda: oracle.sample_pairs(2, context=2)),
                         (ValueError, lambda: oracle.sample_pairs(2, context=True)))
        for exception, call in invalid_calls:
            with self.assertRaises(exception):
                call()
            self.assertEqual(state(), before)
        oracle._deadline = time.time() - 1
        with self.assertRaises(DeadlineExceeded):
            oracle.sample_pairs(10)
        with self.assertRaises(DeadlineExceeded):
            oracle.sample_arm(0, 1)
        self.assertEqual(state(), before)

    def test_same_seed_action_trace_is_reproducible_without_crn_claim(self):
        config = PublicExperiment(shot_budget=100000)
        first = CountObservationOracle(config, [.1, .2, .3, .4], 212)
        second = CountObservationOracle(config, [.1, .2, .3, .4], 212)
        for size in (1, 37, 800, 4100):
            self.assertEqual(first.sample_pairs(size), second.sample_pairs(size))
            self.assertEqual(first.sample_arm(3, size), second.sample_arm(3, size))
        self.assertEqual(first.cost_snapshot(), second.cost_snapshot())
        self.assertFalse(first.audit_truth()["batch_partition_invariant"])
        self.assertIn("independent per-method seeds", first.audit_truth()["comparison_seed_policy"])

    def test_huge_batch_uses_scalar_count_draws_and_full_accounting(self):
        pairs = 50000000
        oracle = CountObservationOracle(PublicExperiment(shot_budget=2 * pairs), [.2, .3, .4, .5], 842)
        result = oracle.sample_pairs(pairs)
        self.assertEqual(sum(result["joint"]), pairs)
        self.assertEqual(sum(result["context_counts"]), pairs)
        self.assertEqual(oracle.shots, 2 * pairs)
        self.assertEqual(oracle.remaining, 0)
        self.assertTrue(all(isinstance(value, int) for value in result["joint"]))
        self.assertEqual(oracle.batches, 1)

    def test_transcripts_are_explicitly_rejected(self):
        for transcripts in ({}, {0: [0, 1]}):
            with self.assertRaisesRegex(ValueError, "use CircuitObservationOracle"):
                CountObservationOracle(PublicExperiment(), [.1] * 4, 1, transcripts=transcripts)


if __name__ == "__main__":
    unittest.main()
