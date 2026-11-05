"""Independent analytic, unitary, channel and context checks; Python 3.8+."""

import cmath
import json
import math
import unittest
from dataclasses import FrozenInstanceError

import quantum_model as qm


def dagger_product_sum(operators):
    """Compute sum K-dagger K directly, independently of the channel helper."""
    return tuple(tuple(sum(operator[k][row].conjugate() * operator[k][column]
                           for operator in operators for k in range(2))
                       for column in range(2)) for row in range(2))


class QuantumGateAndChannelTests(unittest.TestCase):
    def assert_identity(self, matrix):
        for row in range(2):
            for column in range(2):
                self.assertAlmostEqual(abs(matrix[row][column] - (1 if row == column else 0)), 0.0, places=13)

    def test_h_and_rx_are_unitary(self):
        for operator in (qm.H2,) + tuple(qm.rx_matrix(angle) for angle in (0.0, 0.37, math.pi, -2.0)):
            self.assert_identity(dagger_product_sum((operator,)))

    def test_controlled_phase_equals_fixed_zz_plus_local_z_hamiltonian(self):
        angle = 0.713
        for basis in range(8):
            state = tuple(1.0 + 0j if index == basis else 0j for index in range(8))
            output = qm.apply_controlled_phase(state, angle)
            z1 = 1.0 if not basis & 2 else -1.0
            z2 = 1.0 if not basis & 4 else -1.0
            energy = (1.0 - z1 - z2 + z1 * z2) / 4.0
            expected_phase = cmath.exp(-1j * angle * energy)
            self.assertAlmostEqual(abs(output[basis] - expected_phase), 0.0, places=14)
            self.assertAlmostEqual(qm.state_norm_squared(output), 1.0, places=14)
            self.assertTrue(all(output[index] == 0j for index in range(8) if index != basis))

    def test_controlled_phase_is_identity_on_entire_idle_spectator_subspace(self):
        # This checks all four basis states of the q2=0 subspace, not only
        # equality of one marginal failure rate for one prepared state.
        for gamma in (0.1, math.pi / 3.0, 2.7, -0.3):
            for basis in range(4):
                state = tuple(1.0 + 0j if index == basis else 0j for index in range(8))
                self.assertEqual(qm.apply_controlled_phase(state, gamma), state)

    def test_kraus_operators_are_complete_including_endpoints(self):
        for probability in (0.0, 0.01, 0.25, 0.5, 1.0):
            for constructor in (qm.bit_flip_kraus, qm.dephasing_kraus):
                self.assert_identity(dagger_product_sum(constructor(probability)))

    def test_kraus_ensemble_preserves_trace_without_normalizing_branches(self):
        raw = tuple(complex(index + 1, 2 - index) for index in range(8))
        norm = math.sqrt(sum(abs(value) ** 2 for value in raw))
        state = tuple(value / norm for value in raw)
        for constructor in (qm.bit_flip_kraus, qm.dephasing_kraus):
            for qubit in range(3):
                branches = qm.apply_kraus_channel((state,), constructor(0.3), qubit)
                self.assertEqual(len(branches), 2)
                norms = [sum(abs(value) ** 2 for value in branch) for branch in branches]
                self.assertAlmostEqual(norms[0], 0.7, places=13)
                self.assertAlmostEqual(norms[1], 0.3, places=13)
                self.assertAlmostEqual(sum(norms), 1.0, places=13)

    def test_dephasing_semigroup_uses_actual_window_duration(self):
        model = qm.QuantumModel(dephasing_rate=0.17)
        old = qm.dephasing_probability(model, "old")
        new = qm.dephasing_probability(model, "new")
        self.assertAlmostEqual(1.0 - 2.0 * old, math.exp(-0.17), places=14)
        self.assertAlmostEqual(1.0 - 2.0 * new, math.exp(-0.34), places=14)
        self.assertAlmostEqual(1.0 - 2.0 * new, (1.0 - 2.0 * old) ** 2, places=14)
        self.assertGreater(new, old)

    def test_dephasing_kraus_maps_commute_with_fixed_controlled_phase(self):
        state = tuple(complex(index + 1, 1 - index) for index in range(8))
        for qubit in (1, 2):
            for operator in qm.dephasing_kraus(0.23):
                first = qm.apply_controlled_phase(qm.apply_single_qubit(state, operator, qubit), 0.51)
                second = qm.apply_single_qubit(qm.apply_controlled_phase(state, 0.51), operator, qubit)
                for a, b in zip(first, second):
                    self.assertAlmostEqual(abs(a - b), 0.0, places=13)


class ContextBlindnessTests(unittest.TestCase):
    def test_fixed_candidate_is_identical_between_hidden_worlds_and_costs_more_time(self):
        first, second = qm.make_blind_pair()
        self.assertEqual(qm.control_for(first, "old"), qm.PulseControl(0.0, 1.0))
        self.assertEqual(qm.control_for(first, "new"), qm.PulseControl(-first.omega, 2.0))
        for variant in qm.VARIANTS:
            self.assertEqual(qm.control_snapshot(first, variant), qm.control_snapshot(second, variant))
            for context in qm.CONTEXTS:
                self.assertEqual(qm.circuit_hash(first, variant, context), qm.circuit_hash(second, variant, context))
        self.assertNotEqual(qm.model_hash(first), qm.model_hash(second))

    def test_calibration_full_distributions_coincide_with_both_channels(self):
        for omega in (0.063, math.pi / 3.0, 1.7):
            for flip, rate in ((0.0, 0.0), (0.01, 0.03), (0.2, 0.4)):
                first, second = qm.make_blind_pair(omega, flip, rate)
                for variant in qm.VARIANTS:
                    self.assertEqual(qm.circuit_probabilities(first, variant, qm.CALIBRATION_CONTEXT),
                                     qm.circuit_probabilities(second, variant, qm.CALIBRATION_CONTEXT))

    def test_noiseless_reference_has_independent_rational_probabilities(self):
        first, second = qm.make_blind_pair(readout_flip=0.0)
        expected_old = (39 / 64, 13 / 64, 3 / 64, 1 / 64, 3 / 64, 1 / 64, 3 / 64, 1 / 64)
        expected_new = (7 / 16, 0.0, 3 / 16, 0.0, 3 / 16, 0.0, 3 / 16, 0.0)
        for actual, expected in ((qm.circuit_probabilities(second, "old", qm.WORKLOAD_CONTEXT), expected_old),
                                 (qm.circuit_probabilities(second, "new", qm.WORKLOAD_CONTEXT), expected_new)):
            for value, target in zip(actual, expected):
                self.assertAlmostEqual(value, target, places=13)
        first_truth = qm.truth_summary(first)
        second_truth = qm.truth_summary(second)
        self.assertAlmostEqual(first_truth["p_fail_old"], 1 / 4, places=13)
        self.assertAlmostEqual(first_truth["p_fail_new"], 0.0, places=13)
        self.assertAlmostEqual(first_truth["delta"], -1 / 4, places=13)
        self.assertAlmostEqual(second_truth["p_fail_old"], 25 / 64, places=13)
        self.assertAlmostEqual(second_truth["p_fail_new"], 9 / 16, places=13)
        self.assertAlmostEqual(second_truth["delta"], 11 / 64, places=13)

    def test_failure_reads_all_three_bits_not_only_the_locally_improved_qubit(self):
        _, harmful = qm.make_blind_pair(readout_flip=0.0)
        new = qm.circuit_probabilities(harmful, "new", qm.WORKLOAD_CONTEXT)
        self.assertAlmostEqual(sum(new[index] for index in range(8) if index & 1), 0.0, places=13)
        self.assertAlmostEqual(qm.failure_probability(harmful, "new", qm.WORKLOAD_CONTEXT), 9 / 16, places=13)
        self.assertFalse(qm.is_failure(0))
        self.assertTrue(all(qm.is_failure(index) for index in range(1, 8)))

    def test_noiseless_small_effect_family_has_opposite_signs(self):
        for s in (0.001, 0.003, 0.01, 0.1, 0.25, 0.49):
            omega = 2.0 * math.asin(math.sqrt(s))
            first, second = qm.make_blind_pair(omega, readout_flip=0.0)
            good = qm.truth_summary(first)["delta"]
            bad = qm.truth_summary(second)["delta"]
            self.assertAlmostEqual(good, -s, places=12)
            self.assertAlmostEqual(bad, s * (5.0 - 9.0 * s) / 4.0, places=12)
            self.assertLess(good, 0)
            self.assertGreater(bad, 0)

    def test_readout_noise_keeps_nonzero_baselines_and_matches_delta_polynomial(self):
        for r in (0.01, 0.1, 0.3, 0.49):
            for s in (0.001, 0.1, 0.25):
                first, second = qm.make_blind_pair(2 * math.asin(math.sqrt(s)), r)
                q = 1.0 - r
                good = qm.truth_summary(first)
                bad = qm.truth_summary(second)
                expected_good = -(1.0 - 2.0 * r) * s * q * q
                expected_bad = (q - 0.5) * s * (q * (q + 1.5) - 2 * (q + 0.5) ** 2 * s)
                self.assertAlmostEqual(good["delta"], expected_good, places=12)
                self.assertAlmostEqual(bad["delta"], expected_bad, places=12)
                self.assertLess(good["delta"], 0)
                self.assertGreater(bad["delta"], 0)
                self.assertGreater(good["p_fail_new"], 0)
                self.assertAlmostEqual(good["p_fail_new"], 1.0 - q ** 3, places=12)


class ForwardModelAndProvenanceTests(unittest.TestCase):
    def test_statevector_kraus_and_independent_closed_forms_agree(self):
        for omega, gamma, flip, rate in ((0.12, 0.0, 0.01, 0.0),
                                         (0.73, 0.31, 0.08, 0.12),
                                         (math.pi / 3, math.pi / 3, 0.01, 0.05),
                                         (1.2, -0.83, 0.23, 0.7),
                                         (-0.4, 1.1, 0.5, 0.3),
                                         (0.1, 1.3, 1.0, 0.0)):
            model = qm.QuantumModel(omega, gamma, flip, rate)
            for variant in qm.VARIANTS:
                for context in qm.CONTEXTS:
                    with self.subTest(model=model, variant=variant, context=context):
                        probabilities = qm.circuit_probabilities(model, variant, context)
                        independent = qm.analytic_probabilities(model, variant, context)
                        self.assertEqual(len(probabilities), 8)
                        self.assertTrue(all(0 <= value <= 1 for value in probabilities))
                        self.assertAlmostEqual(math.fsum(probabilities), 1.0, places=13)
                        for actual, expected in zip(probabilities, independent):
                            self.assertAlmostEqual(actual, expected, places=12)
                        self.assertAlmostEqual(qm.failure_probability(model, variant, context),
                                               qm.analytic_failure_probability(model, variant, context), places=12)
                        self.assertAlmostEqual(qm.state_norm_squared(qm.unitary_statevector(model, variant, context)),
                                               1.0, places=13)

    def test_full_half_flip_readout_erases_state_to_uniform_measurement(self):
        model = qm.QuantumModel(gamma=0.77, readout_flip=0.5, dephasing_rate=0.23)
        for variant in qm.VARIANTS:
            for context in qm.CONTEXTS:
                for probability in qm.circuit_probabilities(model, variant, context):
                    self.assertAlmostEqual(probability, 1.0 / 8.0, places=13)

    def test_invalid_models_controls_contexts_and_outcomes_are_rejected(self):
        for kwargs in ({"omega": float("nan")}, {"gamma": float("inf")},
                       {"readout_flip": -0.01}, {"readout_flip": 1.01},
                       {"readout_flip": float("nan")}, {"dephasing_rate": -0.1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                qm.QuantumModel(**kwargs)
        for amplitude, duration in ((0.0, 0.0), (0.0, -1.0), (float("inf"), 1.0)):
            with self.assertRaises(ValueError):
                qm.PulseControl(amplitude, duration)
        with self.assertRaises(ValueError):
            qm.circuit_probabilities(qm.QuantumModel(), "modified", qm.CALIBRATION_CONTEXT)
        with self.assertRaises(ValueError):
            qm.circuit_probabilities(qm.QuantumModel(), "old", "illegal_context")
        for outcome in (-1, 8, True):
            with self.assertRaises(ValueError):
                qm.is_failure(outcome)

    def test_snapshots_are_stable_serializable_and_record_both_durations(self):
        model = qm.QuantumModel()
        with self.assertRaises(FrozenInstanceError):
            model.gamma = 1.0
        snapshot = qm.model_snapshot(model)
        self.assertEqual(json.loads(json.dumps(snapshot, allow_nan=False)), snapshot)
        self.assertEqual(qm.model_hash(model), qm.model_hash(qm.QuantumModel()))
        self.assertEqual(len(qm.model_hash(model)), 64)
        for variant, duration in (("old", 1.0), ("new", 2.0)):
            for context in qm.CONTEXTS:
                program = qm.circuit_snapshot(model, variant, context)
                self.assertEqual(program["window_duration"], duration)
                self.assertEqual(program["readout_objects"], 3)
                self.assertEqual(program["reset_objects"], 3)
                self.assertEqual(program["measured_z_qubits"], [0, 1, 2])
                self.assertEqual(program["failure_predicate"], "outcome != 000")
                self.assertEqual(program["prepare_h_qubits"], program["undo_h_qubits"])


if __name__ == "__main__":
    unittest.main()
