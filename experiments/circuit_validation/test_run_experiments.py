"""Integration checks for information separation and complete trial accounting."""
import copy
from fractions import Fraction
import time
import unittest

from run_experiments import endpoint_alpha, public_config, run_trial


def tiny_fixture():
    case = {"case_id": "private_harmful_label", "contexts": ["spectator_0", "spectator_plus"],
            "weights": [.2, .8], "epsilon": .01, "shot_budget": 256, "fixed_pairs": 128,
            "probabilities": [1, 1, 0, 0], "feasible": True}
    spec = {"phase": "development", "master_seed": 7123, "delta": .01,
            "policy_parameters": {"growth": 1.3, "initial": 32, "max_extra": 262144},
            "observation_contract": {"cost": {"measured_qubits": 3, "old_duration": 3.2,
                                               "new_duration": 4.2, "switch_duration": .1}}}
    return spec, case


class RunnerIntegrationTests(unittest.TestCase):
    def test_json_endpoint_risks_do_not_overspend_exact_family_budgets(self):
        cells = 44 * 8
        safety = Fraction(str(endpoint_alpha(1, 50 * 3 * cells))) * 3 * cells
        power = Fraction(str(endpoint_alpha(1, 50 * 2 * cells))) * 2 * cells
        cost = Fraction(str(endpoint_alpha(1, 100 * 2 * cells))) * 2 * cells
        self.assertLessEqual(safety, Fraction(1, 50))
        self.assertLessEqual(power, Fraction(1, 50))
        self.assertLessEqual(cost, Fraction(1, 100))
        self.assertLessEqual(safety + power + cost, Fraction(1, 20))

    def test_truth_and_sign_label_never_reach_public_configuration(self):
        spec, case = tiny_fixture()
        config = public_config(spec, case)
        second = copy.deepcopy(case)
        second["probabilities"] = [0, 0, 1, 1]
        second["true_delta"] = 1.0
        self.assertEqual(config, public_config(spec, second))
        self.assertNotIn("harmful", config.candidate_id + config.workload_id)
        self.assertFalse(hasattr(config, "probabilities"))

    def test_full_paid_trace_repeats_with_seed_and_policies_use_distinct_seeds(self):
        spec, case = tiny_fixture()
        a = run_trial(spec, "hash", case, "direct_fixed", 0)
        b = run_trial(spec, "hash", case, "direct_fixed", 0)
        c = run_trial(spec, "hash", case, "direct_mixture", 0)
        self.assertEqual(a["events"], b["events"])
        self.assertEqual(a["cost"], b["cost"])
        self.assertEqual(a["decision"], "accept")
        self.assertEqual(a["shots"], 256)
        self.assertEqual(a["cost"]["readout_objects"], 768)
        self.assertNotEqual(a["seed"], c["seed"])
        self.assertNotEqual(a["seed"], run_trial(spec, "hash", case, "direct_fixed", 1)["seed"])

    def test_timeout_is_a_preserved_trial_not_a_success(self):
        spec, case = tiny_fixture()
        record = run_trial(spec, "hash", case, "direct_fixed", 0, deadline=time.time() - 1)
        self.assertEqual(record["status"], "timeout")
        self.assertEqual(record["decision"], "abstain")
        self.assertEqual(record["shots"], 0)
        self.assertEqual(record["events"], [])

    def test_invalid_registered_case_is_preserved_as_error(self):
        spec, case = tiny_fixture()
        case.update(feasible=False, infeasible_reason="deliberate test")
        record = run_trial(spec, "hash", case, "direct_fixed", 0)
        self.assertEqual(record["status"], "error")
        self.assertEqual(record["decision"], "abstain")
        self.assertEqual(record["shots"], 0)
        self.assertIn("deliberate test", record["error"])


if __name__ == "__main__":
    unittest.main()
