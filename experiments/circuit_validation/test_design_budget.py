"""Development-only provenance, paired-law planning, and shared-budget checks."""

import copy
import json
import math
import unittest
from unittest.mock import patch

import design_budget as db
from case_factory import build_cases


class PlanningPrimitiveTests(unittest.TestCase):
    def test_discordance_uses_shared_context_and_independent_branches(self):
        # Contexts each have deterministic but opposite old/new outcomes.
        # Drawing old/new contexts separately would instead give discordance .5.
        self.assertEqual(db._paired_discordance([0, 1, 1, 0], [.5, .5]), 1)
        self.assertEqual(db._paired_discordance([.5] * 4, [.2, .8]), .5)
        self.assertEqual(db._paired_discordance([0] * 4, [.2, .8]), 0)

    def test_insufficient_numeric_power_increases_pairs_before_freezing(self):
        rows = build_cases("development")[:2]
        answers = [{"power_estimate": value, "omitted_k_probability": 0.0} for value in (.95, .979, .987)]
        with patch.object(db, "fixed_paired_power", side_effect=answers) as planner:
            result = db._plan_stratum(rows)
        self.assertEqual(planner.call_count, 3)
        attempted = [entry["pairs"] for entry in result["planning_attempts"]]
        self.assertGreaterEqual(attempted[1], math.ceil(1.1 * attempted[0]))
        self.assertGreaterEqual(attempted[2], math.ceil(1.1 * attempted[1]))
        self.assertEqual(result["fixed_pairs"], attempted[-1])
        self.assertEqual(result["shot_budget"], 8 * result["fixed_pairs"])
        self.assertEqual(result["power_plan"]["power_estimate"], .987)

    def test_confirmation_or_missing_sign_cannot_enter_planning(self):
        rows = build_cases("development")[:2]
        with self.assertRaises(ValueError):
            db._plan_stratum(rows[:1])
        altered = copy.deepcopy(rows)
        altered[0]["phase"] = "confirmation"
        with self.assertRaisesRegex(ValueError, "restricted to development"):
            db._plan_stratum(altered)
        altered = copy.deepcopy(rows)
        altered[0]["feasible"] = False
        with self.assertRaisesRegex(ValueError, "infeasible development cell"):
            db._plan_stratum(altered)


class RegisteredBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import scipy.stats
        except ImportError:
            raise unittest.SkipTest("numerical budget planning requires optional SciPy")
        cls.cases = build_cases("development")
        # This spy fails immediately if confirmation is loaded while planning.
        def development_only(phase):
            if phase != "development":
                raise AssertionError("confirmation truth was accessed")
            return copy.deepcopy(cls.cases)
        with patch.object(db, "build_cases", side_effect=development_only) as factory:
            cls.budgets = db.design_budgets()
        cls.factory_calls = factory.call_args_list

    def test_only_development_factory_called_and_complete_matrix_mapped(self):
        self.assertEqual(len(self.factory_calls), 1)
        self.assertEqual(self.factory_calls[0][0], ("development",))
        self.assertFalse(self.budgets["metadata"]["uses_confirmation_truth"])
        self.assertFalse(self.budgets["metadata"]["uses_algorithm_observations"])
        self.assertEqual(len(self.budgets["strata"]), 18)
        self.assertEqual(len(self.budgets["by_case"]), 88)
        self.assertEqual(len(self.budgets["diagnostic_budgets"]), 3)
        json.dumps(self.budgets, allow_nan=False)

    def test_every_design_has_verified_power_and_full_even_shot_budget(self):
        for row in self.budgets["strata"].values():
            self.assertGreaterEqual(row["power_plan"]["power_estimate"], .98)
            self.assertEqual(row["power_plan"]["gap"], row["epsilon"])
            self.assertEqual(row["power_plan"]["discordance"], row["design_discordance"])
            self.assertGreaterEqual(row["fixed_pairs"], row["normal_initial_pairs"])
            self.assertEqual(row["fixed_shots"], 2 * row["fixed_pairs"])
            self.assertEqual(row["shot_budget"], 4 * row["fixed_shots"])
            self.assertEqual(row["shot_budget"] % 2, 0)
            self.assertIn("not uniform", row["power_scope"])

    def test_design_point_is_inflated_maximum_of_both_development_signs(self):
        by_id = {row["case_id"]: row for row in self.cases}
        for row in self.budgets["strata"].values():
            self.assertEqual(len(row["source_case_ids"]), 2)
            sources = [by_id[case_id] for case_id in row["source_case_ids"]]
            self.assertEqual({source["category"] for source in sources}, {"beneficial", "harmful"})
            exact_formula = []
            for source in sources:
                old0, old1, new0, new1 = source["probabilities"]
                w0, w1 = source["weights"]
                exact_formula.append(w0 * (old0 + new0 - 2 * old0 * new0)
                                     + w1 * (old1 + new1 - 2 * old1 * new1))
            self.assertAlmostEqual(row["design_discordance"], min(1, 1.1 * max(exact_formula)), places=14)
            left, right = (self.budgets["by_case"][source["case_id"]] for source in sources)
            self.assertEqual(left["fixed_pairs"], right["fixed_pairs"])
            self.assertEqual(left["shot_budget"], right["shot_budget"])

    def test_confirmation_ids_share_development_budgets_without_truth_lookup(self):
        for case in self.cases:
            development = self.budgets["by_case"][case["case_id"]]
            confirmation_id = "confirm_" + case["case_id"][4:]
            confirmation = self.budgets["by_case"][confirmation_id]
            for field in ("fixed_pairs", "fixed_shots", "shot_budget", "design_group_key", "mapping_scheme"):
                self.assertEqual(development[field], confirmation[field])

    def test_gray_and_blind_cases_use_same_epsilon_maximum_main_budget(self):
        for case in self.cases:
            if case["benchmark_role"] == "main":
                continue
            actual = self.budgets["by_case"][case["case_id"]]
            candidates = [row for row in self.budgets["strata"].values() if row["epsilon"] == case["epsilon"]]
            self.assertEqual(actual["fixed_pairs"], max(row["fixed_pairs"] for row in candidates))
            self.assertEqual(actual["shot_budget"], max(row["shot_budget"] for row in candidates))
            self.assertIn("diagnostic truth unused", actual["mapping_scheme"])

    def test_diagnostic_truth_cannot_change_any_budget(self):
        poisoned = copy.deepcopy(self.cases)
        for row in poisoned:
            row["true_delta"] = "FORBIDDEN_UNUSED_TRUTH"
            if row["benchmark_role"] != "main":
                row["probabilities"] = ["FORBIDDEN_DIAGNOSTIC_TRUTH"] * 4
                row["model"] = {"must_not_be_read": True}
        # Reuse independently verified per-group plans to focus this test on
        # information flow and mapping, rather than repeating SciPy arithmetic.
        def frozen_plan(rows):
            first = rows[0]
            key = db._group_key(first["epsilon"], first["weights"][1], first["stratum"])
            return copy.deepcopy(self.budgets["strata"][key])
        with patch.object(db, "build_cases", return_value=poisoned), patch.object(db, "_plan_stratum", side_effect=frozen_plan):
            result = db.design_budgets()
        self.assertEqual(result, self.budgets)


if __name__ == "__main__":
    unittest.main()
