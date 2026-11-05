"""Finite-sample interval and all-record accounting tests."""

import itertools
import math
import unittest
from decimal import Decimal, localcontext
from fractions import Fraction

import assessment as a


def record(decision="accept", shots=10, status="ok", budget=100):
    return dict(decision=decision, shots=shots, status=status, shot_budget=budget,
                cost={"unused": "must not replace primary shots"})


class ClopperPearsonTests(unittest.TestCase):
    def test_integer_tail_matches_direct_small_binomial_sum(self):
        for n in range(1, 13):
            for k in range(n + 1):
                for p in (Fraction(1, 7), Fraction(1, 2), Fraction(5, 7)):
                    numerator, denominator = a._binomial_cdf_ratio(n, k, p)
                    expected = sum((Fraction(math.comb(n, j)) * p ** j * (1 - p) ** (n - j)
                                    for j in range(k + 1)), Fraction(0))
                    self.assertEqual(Fraction(numerator, denominator), expected)

    def test_endpoints_are_on_conservative_exact_binomial_side(self):
        alpha = Fraction(1, 20)
        for n in range(1, 16):
            for k in range(n + 1):
                upper = a.clopper_pearson_upper(k, n, float(alpha))
                lower = a.clopper_pearson_lower(k, n, float(alpha))
                self.assertLessEqual(lower, k / n)
                self.assertGreaterEqual(upper, k / n)
                if k < n:
                    num, den = a._binomial_cdf_ratio(n, k, Fraction.from_float(upper))
                    self.assertLessEqual(Fraction(num, den), alpha)
                if k:
                    # P_lower[X>=k] <= alpha is the lower-endpoint condition.
                    num, den = a._binomial_cdf_ratio(n, k - 1, Fraction.from_float(lower))
                    self.assertLessEqual(1 - Fraction(num, den), alpha)

    def test_exact_small_n_one_sided_coverage(self):
        alpha = Fraction(1, 10)
        for n in (3, 7, 12):
            upper = [a.clopper_pearson_upper(k, n, float(alpha)) for k in range(n + 1)]
            lower = [a.clopper_pearson_lower(k, n, float(alpha)) for k in range(n + 1)]
            for p in (Fraction(1, 10), Fraction(1, 3), Fraction(2, 3), Fraction(9, 10)):
                miss_upper = miss_lower = Fraction(0)
                for k in range(n + 1):
                    mass = Fraction(math.comb(n, k)) * p ** k * (1 - p) ** (n - k)
                    if Fraction.from_float(upper[k]) < p:
                        miss_upper += mass
                    if Fraction.from_float(lower[k]) > p:
                        miss_lower += mass
                self.assertLessEqual(miss_upper, alpha)
                self.assertLessEqual(miss_lower, alpha)

    def test_zero_error_sample_size_boundary(self):
        self.assertGreater(a.clopper_pearson_upper(0, 298, .05), .01)
        self.assertLess(a.clopper_pearson_upper(0, 299, .05), .01)
        self.assertEqual(a.clopper_pearson_upper(0, 0, .05), 1)
        self.assertEqual(a.clopper_pearson_lower(0, 0, .05), 0)


class CostIntervalTests(unittest.TestCase):
    def test_formula_and_unbiased_variance(self):
        values, bound, alpha = [0, 1, 2, 3, 4], 4, .1
        result = a.empirical_bernstein_interval(values, bound, alpha)
        self.assertEqual(result["mean"], 2)
        self.assertEqual(result["sample_variance"], 2.5)
        with localcontext() as ctx:
            ctx.prec = 100
            log_term = Decimal(40).ln()
            expected = (Decimal(2) * Decimal("2.5") * log_term / Decimal(5)).sqrt()
            expected += Decimal(7 * bound) * log_term / Decimal(3 * 4)
        self.assertGreaterEqual(Decimal(result["radius_decimal_upper"]), expected)
        self.assertEqual(result["lower"], 0)
        self.assertEqual(result["upper"], 4)

    def test_constant_cost_still_has_nonzero_uncertainty(self):
        result = a.empirical_bernstein_interval([20] * 1000, 100, .01)
        self.assertEqual(result["sample_variance"], 0)
        self.assertLess(result["lower"], 20)
        self.assertGreater(result["upper"], 20)
        self.assertGreater(result["radius"], 0)

    def test_empty_singleton_and_invalid_bounds(self):
        for values in ([], [3]):
            result = a.empirical_bernstein_interval(values, 100, .01)
            self.assertEqual((result["lower"], result["upper"]), (0, 100))
        for values in ([101], [-1], [float("nan")]):
            with self.assertRaises(ValueError):
                a.empirical_bernstein_interval(values, 100, .01)
        noninteger = a.empirical_bernstein_interval([], .3, .01)
        self.assertGreaterEqual(Fraction.from_float(noninteger["upper"]), Fraction(3, 10))

    def test_exact_two_point_coverage_with_known_mean(self):
        # Independent Bernoulli costs are a bounded-cost special case.
        # Enumeration evaluates coverage rather than comparing duplicate code.
        n, p, alpha = 128, Fraction(1, 4), Fraction(1, 10)
        missed = Fraction(0)
        for successes in range(n + 1):
            result = a.empirical_bernstein_interval([1] * successes + [0] * (n - successes), 1, float(alpha))
            if not result["lower"] <= float(p) <= result["upper"]:
                missed += Fraction(math.comb(n, successes)) * p ** successes * (1 - p) ** (n - successes)
        self.assertLessEqual(missed, alpha)
        self.assertGreater(missed, 0)  # The test must not only exercise [0,1].


class RecordAccountingTests(unittest.TestCase):
    def summarize(self, records, delta=-.02, epsilon=.01):
        return a.summarize_cell(records, delta, epsilon, .05, .05, .02, 100)

    def test_all_records_and_failure_penalty_are_retained(self):
        records = [record(shots=10), record("reject", 20),
                   record("abstain", 30), record("accept", 40, "error"),
                   record("abstain", 5, "timeout")]
        result = self.summarize(records)
        self.assertEqual(result["n"], 5)
        self.assertEqual(result["actual_cost"]["mean"], 21)
        self.assertEqual(result["penalized_cost"]["mean"], 66)
        self.assertEqual(result["false_reject_count"], 1)
        self.assertEqual(result["power"]["count"], 1)
        self.assertEqual(result["completed_count"], 2)
        self.assertEqual(result["noncompletion_count"], 3)
        self.assertEqual(result["actual_cost"]["alpha"], .02)
        self.assertEqual(result["penalized_cost"]["alpha"], .02)

    def test_error_status_does_not_hide_a_bad_acceptance(self):
        result = self.summarize([record("accept", 1, "error")], delta=.02)
        self.assertEqual(result["false_accept_count"], 1)
        self.assertEqual(result["safety"]["count"], 1)
        self.assertEqual(result["power"]["count"], 0)
        self.assertEqual(result["penalized_cost"]["mean"], 100)

    def test_at_zero_any_decision_is_wrong_and_no_power_is_claimed(self):
        result = self.summarize([record("accept"), record("reject"), record("abstain")], delta=0)
        self.assertEqual(result["safety"]["count"], 2)
        self.assertEqual(result["false_accept"]["count"], 1)
        self.assertEqual(result["false_reject"]["count"], 1)
        self.assertEqual(result["any_wrong"]["count"], 2)
        self.assertFalse(result["power"]["applicable"])
        self.assertFalse(result["eligible_for_efficiency"])
        self.assertEqual(result["alpha_ledger"]["good_accept_endpoint"], .05)

    def test_out_of_budget_kept_but_does_not_get_a_false_cost_certificate(self):
        result = self.summarize([record(shots=110), record(shots=10)])
        self.assertEqual(result["actual_cost"]["mean"], 60)
        self.assertFalse(result["actual_cost"]["valid"])
        self.assertFalse(result["record_integrity_pass"])
        self.assertFalse(result["qualified"])
        self.assertTrue(result["eligible_for_efficiency"])

    def test_missing_cost_cannot_be_silently_dropped(self):
        with self.assertRaises(ValueError):
            self.summarize([record(), {"decision": "abstain", "status": "error"}])


class ComparisonTests(unittest.TestCase):
    @staticmethod
    def summary(mean, lower, upper, qualified=True):
        cost = dict(mean=mean, lower=lower, upper=upper, valid=True)
        return dict(true_delta=-.02, epsilon=.01, eligible_for_efficiency=True,
                    budget=100, qualified=qualified, actual_cost=dict(cost), penalized_cost=dict(cost))

    def test_fastest_baseline_interval_propagation(self):
        candidate = self.summary(10, 8, 12)
        first = self.summary(30, 25, 35)
        second = self.summary(20, 15, 25)
        result = a.compare_costs(candidate, {"first": first, "second": second})
        ratio = result["actual_cost"]
        self.assertEqual(ratio["point"], 2)
        self.assertLessEqual(ratio["lower"], 15 / 12)
        self.assertGreaterEqual(ratio["upper"], 25 / 8)
        self.assertEqual(ratio["fastest_point_baseline"], "second")
        self.assertTrue(result["qualification_pass"])
        self.assertFalse(result["criteria_pass"])

    def test_complete_criteria_include_both_cost_metrics_and_both_targets(self):
        candidate = self.summary(10, 9, 11)
        baseline = self.summary(40, 35, 45)
        result = a.compare_costs(candidate, {"strong": baseline})
        self.assertTrue(result["qualification_pass"])
        self.assertTrue(result["criteria_pass"])
        candidate["penalized_cost"] = dict(mean=20, lower=18, upper=22, valid=True)
        result = a.compare_costs(candidate, {"strong": baseline})
        self.assertTrue(result["actual_target_pass"])
        self.assertFalse(result["penalized_target_pass"])
        self.assertFalse(result["criteria_pass"])
        candidate["penalized_cost"] = dict(candidate["actual_cost"])
        baseline["actual_cost"]["lower"] = 20
        result = a.compare_costs(candidate, {"strong": baseline})
        self.assertFalse(result["criteria_pass"])

    def test_unconfirmed_fast_baseline_blocks_a_subset_victory(self):
        candidate = self.summary(5, 4, 6)
        slow = self.summary(40, 35, 45)
        fast_unconfirmed = self.summary(6, 4, 8, qualified=False)
        result = a.compare_costs(candidate, {"slow": slow, "fast": fast_unconfirmed})
        self.assertEqual(result["status"], "baseline_qualification_incomplete")
        self.assertEqual(result["unconfirmed_baselines"], ["fast"])
        self.assertGreater(result["actual_cost"]["point"], 3)
        self.assertFalse(result["qualification_pass"])
        self.assertFalse(result["criteria_pass"])
        self.assertTrue(result["eligible_for_efficiency"])

    def test_failed_candidate_stays_in_the_predeclared_domain(self):
        candidate = self.summary(1, 0, 2, qualified=False)
        result = a.compare_costs(candidate, {"baseline": self.summary(20, 15, 25)})
        self.assertTrue(result["eligible_for_efficiency"])
        self.assertFalse(result["candidate_qualified"])
        self.assertFalse(result["criteria_pass"])
        self.assertEqual(result["actual_cost"]["upper"], math.inf)

    def test_no_qualifying_baseline_is_missing_evidence_not_a_win(self):
        result = a.compare_costs(self.summary(1, 0, 2), {"baseline": self.summary(20, 15, 25, False)})
        self.assertEqual(result["status"], "no_qualifying_baseline")
        self.assertIsNone(result["actual_cost"]["point"])
        self.assertFalse(result["criteria_pass"])
        self.assertEqual(result["descriptive_all_baselines"]["actual_cost"]["point"], 20)

    def test_empty_baseline_cell_cannot_disappear_from_descriptive_comparison(self):
        empty = a.summarize_cell([], -.02, .01, .05, .05, .02, 100)
        result = a.compare_costs(self.summary(1, 0, 2), {"empty": empty})
        self.assertEqual(result["status"], "no_qualifying_baseline")
        self.assertIsNone(result["descriptive_all_baselines"]["actual_cost"]["point"])

    def test_budget_mismatch_is_not_a_fair_same_budget_comparison(self):
        baseline = self.summary(20, 15, 25)
        baseline["budget"] = 200
        with self.assertRaises(ValueError):
            a.compare_costs(self.summary(10, 5, 15), {"different_budget": baseline})


if __name__ == "__main__":
    unittest.main()
