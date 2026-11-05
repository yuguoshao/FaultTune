import unittest

from observation import CircuitObservationOracle, PublicExperiment
from policies import POLICIES, run_policy


class CircuitPolicyTests(unittest.TestCase):
    def test_all_methods_decide_strong_improvement_using_only_paid_data(self):
        config = PublicExperiment(shot_budget=4096)
        data = {0: [1] * 2048, 1: [1] * 2048, 2: [0] * 2048, 3: [0] * 2048}
        for policy in POLICIES:
            with self.subTest(policy=policy):
                oracle = CircuitObservationOracle(config, [.5] * 4, 7, transcripts=data)
                result = run_policy(policy, config, oracle.sample_arm, oracle.sample_pairs, fixed_pairs=32)
                self.assertEqual(result["decision"], "accept")
                self.assertEqual(result["certificate"]["shots"], oracle.shots)
                self.assertEqual(sum(oracle.counts), oracle.shots)
                self.assertLessEqual(oracle.shots, config.shot_budget)

    def test_unchanged_deterministic_data_exhaust_budget_and_abstain(self):
        config = PublicExperiment(shot_budget=512)
        for policy in POLICIES:
            with self.subTest(policy=policy):
                oracle = CircuitObservationOracle(config, [.5] * 4, 7,
                                                   transcripts={i: [0] * 512 for i in range(4)})
                result = run_policy(policy, config, oracle.sample_arm, oracle.sample_pairs, fixed_pairs=256)
                self.assertEqual(result["decision"], "abstain")
                self.assertEqual(oracle.shots, 512)

    def test_same_data_and_public_configuration_hide_world_truth(self):
        config = PublicExperiment(shot_budget=1024)
        data = {i: [i < 2] * 1024 for i in range(4)}
        for policy in POLICIES:
            outputs = []
            for probabilities in ([.1] * 4, [.1, .2, .8, .9]):
                oracle = CircuitObservationOracle(config, probabilities, 91, transcripts=data)
                result = run_policy(policy, config, oracle.sample_arm, oracle.sample_pairs, fixed_pairs=64)
                outputs.append((result, oracle.cost_snapshot()))
            self.assertEqual(outputs[0], outputs[1])

    def test_fixed_sample_count_is_required_and_does_not_early_stop(self):
        config = PublicExperiment(shot_budget=256)
        oracle = CircuitObservationOracle(config, [1, 1, 0, 0], 3)
        with self.assertRaises(ValueError):
            run_policy("direct_fixed", config, oracle.sample_arm, oracle.sample_pairs)
        result = run_policy("direct_fixed", config, oracle.sample_arm, oracle.sample_pairs, fixed_pairs=128)
        self.assertEqual(result["decision"], "accept")
        self.assertEqual(oracle.shots, 256)
        self.assertEqual(result["checks"], 1)


if __name__ == "__main__":
    unittest.main()
