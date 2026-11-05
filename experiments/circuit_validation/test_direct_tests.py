"""Independent exact-small-state and numerical-enclosure checks for direct tests."""

import math
import unittest
from decimal import Decimal, localcontext
from fractions import Fraction

import direct_tests as dt


def exact_marginal(successes, trials, prior):
    failures = trials - successes
    if prior == 0.5:
        return Fraction(math.factorial(2 * successes) * math.factorial(2 * failures),
                        4 ** trials * math.factorial(successes)
                        * math.factorial(failures) * math.factorial(trials))
    return Fraction(math.factorial(successes) * math.factorial(failures),
                    math.factorial(trials + 1))


def exact_half_cdf(k, n):
    return Fraction(sum(math.comb(n, j) for j in range(k + 1)), 2 ** n)


class MixtureSignTests(unittest.TestCase):
    def test_small_states_match_exact_beta_mixture(self):
        for prior in (0.5, 1.0):
            for trials in range(25):
                for successes in range(trials + 1):
                    evidence = exact_marginal(successes, trials, prior) * 2 ** trials
                    row = dt.certificate_from_pairs([0, successes, trials - successes, 0], delta=0.03, prior=prior)
                    crossing = evidence >= Fraction(100, 3)
                    expected = ("accept" if successes * 2 < trials else "reject") if crossing else "abstain"
                    self.assertEqual(row["decision"], expected)
                    self.assertLessEqual(row["evidence_bounds"][0], math.log(float(evidence)) + 1e-14)
                    self.assertGreaterEqual(row["evidence_bounds"][1], math.log(float(evidence)) - 1e-14)

    def test_concordant_pairs_cost_shots_without_changing_sign_evidence(self):
        first = dt.certificate_from_pairs([0, 2, 42, 0])
        second = dt.certificate_from_pairs([10000, 2, 42, 3000])
        self.assertEqual(first["evidence_bounds"], second["evidence_bounds"])
        self.assertEqual(first["decision"], "accept")
        self.assertEqual(second["shots"], first["shots"] + 26000)
        self.assertEqual(second["pairs"], sum(second["counts"]))
        self.assertEqual(second["est_delta"], -40 / 13044)
        swapped = dt.certificate_from_pairs([10000, 42, 2, 3000])
        self.assertEqual(swapped["decision"], "reject")

    def test_no_discordances_or_tie_cannot_decide(self):
        for counts in ([0, 0, 0, 0], [9999, 0, 0, 9999], [1, 100, 100, 0]):
            self.assertEqual(dt.certificate_from_pairs(counts)["decision"], "abstain")

    def test_true_parameter_likelihood_ratio_is_exact_martingale(self):
        for prior in (0.5, 1.0):
            for pi in (Fraction(1, 7), Fraction(1, 2), Fraction(5, 6)):
                for n in range(8):
                    for s in range(n + 1):
                        current = exact_marginal(s, n, prior) / (pi ** s * (1 - pi) ** (n - s))
                        up = exact_marginal(s + 1, n + 1, prior) / (pi ** (s + 1) * (1 - pi) ** (n - s))
                        down = exact_marginal(s, n + 1, prior) / (pi ** s * (1 - pi) ** (n + 1 - s))
                        self.assertEqual(pi * up + (1 - pi) * down, current)

    def test_anytime_exact_path_dp_controls_wrong_decisions(self):
        # Finite path probability is evaluated independently using rational
        # transition probabilities and exact beta-mixture evidence.
        delta = Fraction(1, 5)
        for pi in (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)):
            live, accepted, rejected = {0: Fraction(1)}, Fraction(0), Fraction(0)
            for n in range(1, 41):
                following = {}
                for previous_s, mass in live.items():
                    for s, probability in ((previous_s, 1 - pi), (previous_s + 1, pi)):
                        child = mass * probability
                        crossing = exact_marginal(s, n, 0.5) * 2 ** n >= 1 / delta
                        if crossing and 2 * s < n:
                            accepted += child
                        elif crossing and 2 * s > n:
                            rejected += child
                        else:
                            following[s] = following.get(s, Fraction(0)) + child
                live = following
            if pi >= Fraction(1, 2):
                self.assertLessEqual(accepted, delta)
            if pi <= Fraction(1, 2):
                self.assertLessEqual(rejected, delta)
            if pi == Fraction(1, 2):
                self.assertLessEqual(accepted + rejected, delta)
            self.assertEqual(accepted + rejected + sum(live.values()), 1)


class ExactFixedTests(unittest.TestCase):
    def test_small_fixed_decisions_match_independent_exact_tails(self):
        for n in range(31):
            for b in range(n + 1):
                row = dt.fixed_paired_certificate([1, b, n - b, 7], delta=0.125)
                accept, reject = exact_half_cdf(b, n), exact_half_cdf(n - b, n)
                expected = "accept" if accept <= Fraction(1, 16) else "reject" if reject <= Fraction(1, 16) else "abstain"
                self.assertEqual(row["decision"], expected)
                self.assertFalse(row["anytime_valid"])
                for key, exact in (("p_accept_bounds", accept), ("p_reject_bounds", reject)):
                    self.assertLessEqual(Fraction.from_float(row[key][0]), exact)
                    self.assertGreaterEqual(Fraction.from_float(row[key][1]), exact)

    def test_boundary_equality_and_zero_discordance(self):
        # Four same-sign discordances have conditional p exactly 1/16.
        self.assertEqual(dt.fixed_paired_certificate([0, 0, 4, 0], delta=0.125)["decision"], "accept")
        self.assertEqual(dt.fixed_paired_certificate([0, 4, 0, 0], delta=0.125)["decision"], "reject")
        self.assertEqual(dt.fixed_paired_certificate([1000, 0, 0, 50])["decision"], "abstain")

    def test_large_tail_encloses_exact_integer_probability(self):
        # Cross the Stirling branch and exercise a long recurrence. Construct
        # the exact binomial sum by an independent integer recurrence.
        n, k = 10000, 4500
        term, numerator = 1, 1
        for j in range(1, k + 1):
            term = term * (n - j + 1) // j
            numerator += term
        exact = Fraction(numerator, 1 << n)
        interval = dt._half_binomial_cdf(k, n)
        self.assertLessEqual(Fraction(interval.lo), exact)
        self.assertGreaterEqual(Fraction(interval.hi), exact)
        self.assertLess((interval.hi - interval.lo) / interval.hi, Decimal("1e-18"))

    def test_fixed_null_combined_error_is_at_most_delta(self):
        for n in (8, 20, 40):
            error = Fraction(0)
            for b in range(n + 1):
                row = dt.fixed_paired_certificate([0, b, n - b, 0], delta=0.1)
                if row["decision"] != "abstain":
                    error += Fraction(math.comb(n, b), 2 ** n)
            self.assertLessEqual(error, Fraction(1, 10))

    def test_extreme_tail_underflow_keeps_conservative_nonzero_upper_bound(self):
        row = dt.fixed_paired_certificate([0, 0, 4000000, 0])
        self.assertEqual(row["decision"], "accept")
        self.assertGreater(row["p_accept_bounds"][1], 0)
        self.assertGreater(row["evidence"], row["threshold"])


class BinomialIntervalTests(unittest.TestCase):
    def test_unknown_old_is_not_treated_as_known(self):
        row = dt.direct_kl_certificate(0, 0, 0, 100000)
        self.assertEqual(row["decision"], "abstain")
        self.assertEqual(row["old_interval"], [0, 1])
        self.assertIsNone(row["est_delta"])
        self.assertEqual(row["shots"], 100000)

    def test_large_difference_decides_and_charges_unequal_arm_counts(self):
        row = dt.direct_kl_certificate(300, 1000, 100, 2000)
        self.assertEqual(row["decision"], "accept")
        self.assertEqual(row["shots"], 3000)
        self.assertLess(row["delta_interval"][1], 0)
        reversed_row = dt.direct_kl_certificate(100, 2000, 300, 1000)
        self.assertEqual(reversed_row["decision"], "reject")
        self.assertGreater(reversed_row["delta_interval"][0], 0)

    def test_endpoint_cases_have_finite_valid_intervals(self):
        self.assertEqual(dt.binomial_mixture_interval(0, 0), [0, 1])
        lower = dt.binomial_mixture_interval(0, 100)
        upper = dt.binomial_mixture_interval(100, 100)
        self.assertEqual(lower[0], 0)
        self.assertEqual(upper[1], 1)
        self.assertAlmostEqual(lower[1], 1 - upper[0], places=13)

    def test_returned_endpoints_are_outside_exact_mixture_set(self):
        # Using Fractions gives an independent check of the certified endpoint
        # directions, including small, sparse, and very unbalanced samples.
        for prior in (0.5, 1.0):
            for n, s in ((1, 0), (1, 1), (10, 2), (32, 16), (100, 1), (100, 99), (100, 50)):
                interval = dt.binomial_mixture_interval(s, n, delta=0.05, prior=prior)
                marginal = exact_marginal(s, n, prior)
                for side, endpoint in enumerate(interval):
                    if endpoint in (0, 1):
                        continue
                    p = Fraction.from_float(endpoint)
                    likelihood = p ** s * (1 - p) ** (n - s)
                    self.assertGreaterEqual(marginal / likelihood, 20)
                    if side == 0:
                        self.assertLess(p, Fraction(s, n))
                    else:
                        self.assertGreater(p, Fraction(s, n))

    def test_finite_horizon_cs_miscoverage(self):
        delta, p, live, excluded = 0.2, Fraction(1, 3), {0: Fraction(1)}, Fraction(0)
        for n in range(1, 19):
            following = {}
            for previous_s, mass in live.items():
                for s, probability in ((previous_s, 1 - p), (previous_s + 1, p)):
                    interval = dt.binomial_mixture_interval(s, n, delta)
                    if p < Fraction.from_float(interval[0]) or p > Fraction.from_float(interval[1]):
                        excluded += mass * probability
                    else:
                        following[s] = following.get(s, Fraction(0)) + mass * probability
            live = following
        self.assertLessEqual(excluded, Fraction(1, 5))
        self.assertEqual(excluded + sum(live.values()), 1)

    def test_log_factorial_bounds_enclose_high_precision_direct_log(self):
        for n in (0, 1, 2, 31, 32, 33, 100, 1000):
            bounds = dt._log_factorial(n)
            with localcontext() as ctx:
                ctx.prec = 100
                reference = Decimal(math.factorial(n)).ln()
            self.assertLessEqual(bounds.lo, reference)
            self.assertGreaterEqual(bounds.hi, reference)

    def test_invalid_inputs_fail_explicitly(self):
        calls = (
            lambda: dt.certificate_from_pairs([0, 1, 2]),
            lambda: dt.certificate_from_pairs([0, True, 2, 0]),
            lambda: dt.certificate_from_pairs([0, -1, 2, 0]),
            lambda: dt.certificate_from_pairs([0, 1.5, 2, 0]),
            lambda: dt.certificate_from_pairs([0, 1, 2, 0], prior=0.2),
            lambda: dt.certificate_from_pairs([0, 1, 2, 0], delta=0),
            lambda: dt.certificate_from_pairs([0, 1, 2, 0], delta=float("nan")),
            lambda: dt.binomial_mixture_interval(11, 10),
            lambda: dt.direct_kl_certificate(1, 0, 0, 5),
        )
        for call in calls:
            with self.assertRaises(ValueError):
                call()


class OptionalPlanningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import scipy.stats
        except ImportError:
            raise unittest.SkipTest("SciPy optional; core certificates already tested")

    def test_power_matches_independent_small_conditional_sum(self):
        n, d, gap, delta = 30, Fraction(1, 2), Fraction(1, 5), Fraction(1, 5)
        pi, total = (d - gap) / (2 * d), Fraction(0)
        for k in range(n + 1):
            mass_k = math.comb(n, k) * d ** k * (1 - d) ** (n - k)
            conditional = sum(math.comb(k, b) * pi ** b * (1 - pi) ** (k - b)
                              for b in range(k + 1) if exact_half_cdf(b, k) <= delta / 2)
            total += mass_k * conditional
        row = dt.fixed_paired_power(n, float(d), float(gap), float(delta))
        self.assertAlmostEqual(row["power_estimate"], float(total), places=12)
        self.assertIn("not machine-certified", row["numeric_status"])

    def test_deterministic_sign_and_empty_design_limits(self):
        self.assertEqual(dt.fixed_paired_power(0, 0.5, 0.1)["power_estimate"], 0)
        self.assertEqual(dt.fixed_paired_power(10, 0, 0)["power_estimate"], 0)
        self.assertAlmostEqual(dt.fixed_paired_power(20, 1, 1)["power_estimate"], 1)

    def test_plan_rechecks_its_own_power_and_marks_scope(self):
        row = dt.plan_fixed_pairs(0.5, 0.1, max_pairs=10000)
        self.assertTrue(row["target_met"])
        self.assertGreaterEqual(row["power_estimate"], 0.95)
        self.assertEqual(row["shots"], 2 * row["pairs"])
        self.assertIn("not proven minimal", row["design_status"])
        self.assertLess(row["pairs"], dt.hoeffding_pair_cap(0.1))


if __name__ == "__main__":
    unittest.main()
