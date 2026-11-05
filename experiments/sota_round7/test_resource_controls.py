"""Deterministic protocol tests, not sampled performance evidence."""
from fractions import Fraction as F
import hashlib
import itertools
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.sota_round7.resource_controls import (
    I, InvariantBitstringControl, RamseyFixedDepthControl, TwoSidedBet)


class ControlChecks(unittest.TestCase):
    def test_exact_crossings_every_small_count(self):
        for b in (F(1, 2), F(3, 250), F(7, 500)):
            for positive, negative in itertools.product(range(26), range(26)):
                test = TwoSidedBet(b)
                test.positive, test.negative = positive, negative
                exact = (1+b)**positive*(1-b)**negative >= 200
                got = test.certificate()
                self.assertEqual(got['crossing_sign'] == 1, exact)
                fallback, method = test._crossed(positive, negative, I(-100, 100))
                self.assertEqual(fallback, exact)
                self.assertEqual(method, 'exact_integer')

    def test_quadrature_debiasing_symbolic_rational_cases(self):
        for bias, mu in ((F(2, 5), F(1, 100)), (F(3, 5), F(-1, 50)), (F(1, 2), F(0))):
            p_plus = bias+mu/2
            p_minus = bias-mu/2
            prob_zplus = (p_plus+(1-p_minus))/2
            self.assertEqual(prob_zplus, (1+mu)/2)
            for b in (F(3, 250), F(7, 500)):
                self.assertEqual(prob_zplus*(1+b)+(1-prob_zplus)*(1-b), 1+b*mu)

    def test_ramsey_stopping_matches_exact_threshold_and_mirrors(self):
        for depth, b in ((8, F(3, 250)), (16, F(7, 500))):
            exact_n = 0
            value = F(1)
            while value < 200:
                value *= 1+b
                exact_n += 1
            summaries = []
            for z in (-1, 1):
                policy = RamseyFixedDepthControl(depth, seed=7, cap=1024)
                while policy.status == 'running':
                    action = policy.next_action()
                    outcome = (1+action['phase_sign']*z)//2
                    policy.observe(action['action_id'], outcome)
                summary = policy.summary()
                self.assertEqual(summary['actual_paid_shots'], exact_n)
                self.assertEqual(summary['decision'], 'reject' if z > 0 else 'accept')
                self.assertEqual(summary['sensing_pulse_uses'], exact_n*depth)
                self.assertEqual(F(summary['device_time_exact']), F(22*exact_n+10*exact_n*depth+summary['phase_switches'], 10))
                self.assertTrue(all(row['status'] == 'delivered' for row in policy.ledger))
                summaries.append(summary)
            self.assertEqual(summaries[0]['phase_switches'], summaries[1]['phase_switches'])

    def test_bitstring_full_counts_and_pair_stopping(self):
        policy = InvariantBitstringControl('same_duration_disjoint_invariant_spectators', cap=100)
        while policy.status == 'running':
            action = policy.next_action()
            # Synthetically prescribed transcript; no physical law is sampled.
            value = 7 if action['config'] == 'old' else 6
            summary = policy.observe(action['action_id'], value)
        self.assertEqual(summary['decision'], 'accept')
        self.assertEqual(summary['actual_paid_shots'], 28)
        self.assertEqual(summary['complete_three_bit_counts']['old'][7], 14)
        self.assertEqual(summary['complete_three_bit_counts']['new'][6], 14)
        self.assertEqual(summary['readout_bits_charged'], 84)
        self.assertEqual(summary['control_window_uses'], 28)
        self.assertEqual(F(summary['device_time_exact']), F(462, 5))
        self.assertEqual(summary['evidence']['positive'], 14)
        self.assertEqual(sum(sum(v) for v in summary['complete_three_bit_counts'].values()), 28)

    def test_cap_and_failed_execution_accounting(self):
        policy = RamseyFixedDepthControl(8, seed=0, cap=10)
        for n in range(10):
            action = policy.next_action()
            z = -1 if n % 2 else 1
            policy.observe(action['action_id'], (1+action['phase_sign']*z)//2)
        self.assertEqual(policy.summary()['status'], 'cap_abstention')
        self.assertEqual(policy.summary()['full_cap_penalized_shots'], 10)
        with self.assertRaises(RuntimeError):
            policy.next_action()
        failed = InvariantBitstringControl('same_duration_disjoint_invariant_spectators', cap=100)
        action = failed.next_action()
        self.assertEqual(failed.summary()['actual_paid_shots'], 1)
        failed.fail_pending(action['action_id'])
        self.assertEqual(failed.summary()['actual_paid_shots'], 1)
        self.assertEqual(failed.summary()['full_cap_penalized_shots'], 100)
        self.assertEqual(failed.summary()['status'], 'execution_error')
        self.assertEqual(failed.summary()['readout_bits_charged'], 3)
        self.assertEqual(failed.ledger[0]['status'], 'execution_failed')

    def test_delivery_order_and_contracts_fail_closed(self):
        with self.assertRaises(ValueError):
            InvariantBitstringControl('unknown_amplitude_crosstalk')
        with self.assertRaises(ValueError):
            RamseyFixedDepthControl(32, seed=0)
        with self.assertRaises(ValueError):
            RamseyFixedDepthControl(8.0, seed=0)
        policy = RamseyFixedDepthControl(8, seed=0, cap=2)
        action = policy.next_action()
        with self.assertRaises(RuntimeError):
            policy.next_action()
        with self.assertRaises(RuntimeError):
            policy.observe(action['action_id']+1, 0)
        with self.assertRaises(ValueError):
            policy.observe(action['action_id'], True)
        self.assertEqual(policy.summary()['actual_paid_shots'], 1)
        self.assertEqual(len(policy.ledger), 1)
        policy.observe(action['action_id'], 0)
        with self.assertRaises(RuntimeError):
            policy.observe(action['action_id'], 0)
        self.assertEqual(policy.summary()['actual_paid_shots'], 1)

    def test_unfinished_snapshots_and_explicit_interruption(self):
        policy = RamseyFixedDepthControl(8, seed=0, cap=20)
        self.assertFalse(policy.summary()['terminal'])
        self.assertEqual(policy.summary()['full_cap_penalized_shots'], 20)
        action = policy.next_action()
        self.assertEqual(policy.summary()['actual_paid_shots'], 1)
        self.assertEqual(policy.summary()['full_cap_penalized_shots'], 20)
        self.assertIn('no inherited probabilistic', policy.summary()['power_scope'])
        policy.finalize_incomplete('deterministic executor interruption test')
        self.assertEqual(policy.summary()['status'], 'incomplete')
        self.assertTrue(policy.summary()['terminal'])
        self.assertEqual(policy.summary()['actual_paid_shots'], 1)
        self.assertEqual(policy.summary()['full_cap_penalized_shots'], 20)
        self.assertEqual(policy.ledger[-1]['status'], 'interrupted')
        self.assertEqual(policy.summary()['sensing_pulse_uses'], 8)
        with self.assertRaises(RuntimeError):
            policy.observe(action['action_id'], 0)
        with self.assertRaises(RuntimeError):
            policy.next_action()
        self.assertEqual(len(policy.ledger), 1)
        self.assertEqual(policy.summary()['actual_paid_shots'], 1)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ControlChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        payload = {'status': 'passed' if result.wasSuccessful() else 'failed',
                   'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
                   'new_simulation_observations': 0,
                   'scope': 'Deterministic arithmetic, replay, action-order and accounting tests only; no empirical power/cost claim.',
                   'source_sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in (Path(__file__).resolve(), ROOT/'experiments/sota_round7/resource_controls.py',
                                               ROOT/'experiments/circuit_validation/direct_tests.py')}}
        with path.open('x', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2)
    sys.exit(0 if result.wasSuccessful() else 1)
