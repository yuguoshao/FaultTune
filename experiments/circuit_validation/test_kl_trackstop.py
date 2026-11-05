"""Independent identities and adversarial endpoint checks for KL TrackStop."""

import itertools
import math
import random
import unittest
from decimal import Decimal, localcontext
from fractions import Fraction

import kl_trackstop as kt


class MixtureArithmeticTests(unittest.TestCase):
    def test_small_mixtures_match_exact_rational_integral(self):
        for n in range(15):
            for s in range(n + 1):
                f = n - s
                for prior in (0.5, 1.0):
                    if prior == 0.5:
                        exact = Fraction(math.factorial(2 * s) * math.factorial(2 * f),
                                         4 ** n * math.factorial(s)
                                         * math.factorial(f) * math.factorial(n))
                    else:
                        exact = Fraction(math.factorial(s) * math.factorial(f),
                                         math.factorial(n + 1))
                    with localcontext() as ctx:
                        ctx.prec = 100
                        expected = (Decimal(exact.numerator) / Decimal(exact.denominator)).ln()
                    interval = kt._mixture_log_likelihood(s, n, prior)
                    self.assertLessEqual(interval.lo, expected)
                    self.assertGreaterEqual(interval.hi, expected)

    def test_stirling_encloses_exact_factorials_across_cutoff(self):
        for n in (0, 1, 32, 63, 64, 65, 100, 511, 1000, 10000):
            with localcontext() as ctx:
                ctx.prec = 100
                exact = Decimal(math.factorial(n)).ln()
            interval = kt._log_factorial(n)
            self.assertLessEqual(interval.lo, exact)
            self.assertGreaterEqual(interval.hi, exact)
            self.assertLess(float(interval.hi - interval.lo), 1e-15)

    def test_product_mixture_expectation_under_adaptive_sampling(self):
        # Exhaustively enumerate a genuinely data-dependent two-arm rule.
        # This checks the martingale identity itself, not Monte Carlo errors.
        p = [Fraction(1, 5), Fraction(3, 5)]
        expectation = Fraction(0)
        for outcomes in itertools.product((0, 1), repeat=8):
            s, n = [0, 0], [0, 0]
            path_probability = Fraction(1)
            for t, bit in enumerate(outcomes):
                arm = 0 if t == 0 else outcomes[t - 1]
                path_probability *= p[arm] if bit else 1 - p[arm]
                s[arm] += bit
                n[arm] += 1
            mixture = Fraction(1)
            likelihood = Fraction(1)
            for j in (0, 1):
                f = n[j] - s[j]
                mixture *= Fraction(math.factorial(2 * s[j]) * math.factorial(2 * f),
                                    4 ** n[j] * math.factorial(s[j])
                                    * math.factorial(f) * math.factorial(n[j]))
                likelihood *= p[j] ** s[j] * (1 - p[j]) ** f
            expectation += path_probability * mixture / likelihood
        self.assertEqual(expectation, 1)

    def test_nextafter_compatibility(self):
        self.assertGreater(kt._nextafter(1.0, math.inf), 1)
        self.assertLess(kt._nextafter(1.0, -math.inf), 1)
        self.assertGreater(kt._nextafter(-1.0, math.inf), -1)
        self.assertLess(kt._nextafter(-1.0, -math.inf), -1)
        self.assertGreater(kt._nextafter(0.0, 1.0), 0)
        self.assertLess(kt._nextafter(0.0, -1.0), 0)
        self.assertTrue(math.isfinite(kt._nextafter(math.inf, 0.0)))

    def test_endpoint_true_means_give_supermartingale_not_equality(self):
        # At p=0 the only possible transcript is all zeros. Its mixture LR
        # decreases; alternative mass on impossible outcomes explains why
        # expectation is <=1 rather than =1 at a boundary parameter.
        previous = Fraction(1)
        for n in range(1, 20):
            current = Fraction(math.factorial(2 * n), 4 ** n * math.factorial(n) ** 2)
            self.assertLessEqual(current, previous)
            previous = current


class JointCertificateTests(unittest.TestCase):
    def test_two_arm_glr_has_independent_pooled_solution(self):
        for successes, trials in (([20, 5], [50, 50]),
                                   ([30, 150], [100, 1000]),
                                   ([0, 10], [100, 200]),
                                   ([100, 0], [100, 100])):
            certificate = kt.joint_glr_certificate(successes, trials, [-1, 1])
            pooled = Fraction(sum(successes), sum(trials))
            exact = kt._sum_intervals(kt._count_kl(s, n, pooled)
                                     for s, n in zip(successes, trials))
            lower = Decimal(certificate["evidence_decimal_lower"])
            upper = Decimal(certificate["primal_decimal_upper"])
            self.assertLessEqual(lower, exact.hi)
            self.assertGreaterEqual(upper, exact.lo)
            self.assertLess(float(upper - lower), 1e-8)

    def test_sign_boundary_and_unobserved_arms(self):
        good = kt.joint_glr_certificate([200, 50], [500, 500], [-1, 1])
        bad = kt.joint_glr_certificate([200, 50], [500, 500], [1, -1])
        self.assertEqual(good["decision"], "accept")
        self.assertEqual(bad["decision"], "reject")
        self.assertEqual(good["evidence"], bad["evidence"])
        for counts in (([0, 0], [100, 100]), ([100, 100], [100, 100]),
                       ([1, 0], [2, 0])):
            self.assertEqual(kt.joint_glr_certificate(*counts, [-1, 1])["decision"],
                             "abstain")

    def test_zero_and_one_empirical_means(self):
        for successes, expected in (([0, 100], "reject"), ([100, 0], "accept")):
            certificate = kt.joint_glr_certificate(successes, [100, 100], [-1, 1])
            self.assertEqual(certificate["decision"], expected)
            self.assertLess(certificate["duality_gap"], 1e-9)
        certificate = kt.joint_glr_certificate([0, 1, 100, 99], [100] * 4,
                                               [-0.999, -0.001, 0.999, 0.001])
        self.assertEqual(certificate["decision"], "reject")
        self.assertLess(certificate["duality_gap"], 1e-8)

    def test_primal_feasibility_is_exact_not_a_float_tolerance(self):
        aa = [Fraction("-.73"), Fraction("-.27"), Fraction(".73"), Fraction(".27")]
        counts = [(0, 13), (10, 31), (21, 91), (2, 3)]
        multiplier, q = kt._glr_root(counts, list(map(float, aa)))
        primal = kt._feasible_primal(q, aa)
        self.assertEqual(sum(a * x for a, x in zip(aa, primal)), 0)
        self.assertTrue(all(0 <= x <= 1 for x in primal))
        certificate = kt.joint_glr_certificate([s for s, n in counts],
                                               [n for s, n in counts],
                                               list(map(float, aa)))
        self.assertLess(certificate["duality_gap"], 1e-8)

    def test_off_optimum_tangent_dual_stays_below_true_glr(self):
        # A deliberately poor multiplier still yields a LOWER bound. This
        # is the crucial direction that a merely feasible primal cannot give.
        s, n, a, multiplier = [20, 5], [50, 50], [-1, 1], 3.0
        parts = []
        for si, ni, ai in zip(s, n, a):
            q = Fraction(1, 2)
            qi = kt._Interval.rational(q)
            eta = kt._Interval.exact(int(multiplier * ai))
            g = (kt._Interval.exact(ni) * qi - kt._Interval.exact(si)) / (qi * (kt._ONE - qi)) + eta
            f = kt._count_kl(si, ni, q) + eta * qi
            c0, c1 = g * (kt._ZERO - qi), g * (kt._ONE - qi)
            parts.append(f + kt._Interval(min(c0.lo, c1.lo), min(c0.hi, c1.hi)))
        lower = kt._sum_intervals(parts).lo
        true_glr = kt._sum_intervals(kt._count_kl(si, ni, Fraction(1, 4))
                                    for si, ni in zip(s, n))
        self.assertLessEqual(lower, true_glr.lo)

    def test_input_validation(self):
        for s, n, a in (([2, 1], [1, 1], [-1, 1]),
                        ([0, 0], [1, 1], [1, 1]),
                        ([0, 0], [1, 1], [math.nan, 1])):
            with self.assertRaises(ValueError):
                kt.joint_glr_certificate(s, n, a)


class AllocationTests(unittest.TestCase):
    def test_waterfill_agrees_with_naive_integer_deficit_tracking(self):
        rng = random.Random(181)
        for _ in range(300):
            deficits = [rng.randrange(-100, 100) / 4 for _ in range(5)]
            budget = rng.randrange(50)
            expected = [0] * len(deficits)
            for _ in range(budget):
                selected = max(range(len(deficits)), key=lambda i: (deficits[i] - expected[i], -i))
                expected[selected] += 1
            self.assertEqual(kt._allocate_largest_deficits(deficits, budget), expected)

    def test_large_batch_has_exact_accounting(self):
        state = kt.TrackStopState([-.75, -.25, .75, .25])
        proposed = state.recommend_counts(10 ** 8)
        self.assertEqual(sum(proposed), 10 ** 8)
        self.assertTrue(all(n >= 2500 for n in proposed))

    def test_two_arm_allocation_equalizes_KL_and_satisfies_KKT(self):
        p, a = [.4, .1], [-1, 1]
        solution = kt.chernoff_allocation(p, a)
        q, w, z = (solution[k] for k in ("boundary_probabilities", "weights", "information_rate"))
        self.assertAlmostEqual(q[0], q[1], places=9)
        self.assertAlmostEqual(sum(w), 1, places=12)
        for pi, qi in zip(p, q):
            self.assertAlmostEqual(kt.bernoulli_kl(pi, qi), z, places=9)
        derivatives = [wi * (qi - pi) / (qi * (1 - qi)) / ai
                       for pi, qi, wi, ai in zip(p, q, w, a)]
        self.assertAlmostEqual(derivatives[0], derivatives[1], places=9)

    def test_weighted_context_design_and_endpoint_regularization(self):
        for p in ([.1, .3, .12, .35], [0, 1, .1, .9], [.001, .5, .02, .7]):
            a = [-.8, -.2, .8, .2]
            solution = kt.chernoff_allocation(p, a)
            self.assertAlmostEqual(sum(solution["weights"]), 1, places=12)
            self.assertTrue(all(w > 0 for w in solution["weights"]))
            self.assertLess(abs(solution["boundary_residual"]), 1e-8)
            for pi, qi in zip(solution["regularized_probabilities"],
                              solution["boundary_probabilities"]):
                if qi in (0.0, 1.0):
                    # Sampling optimization records binary64 endpoint
                    # saturation; it does not claim a KL certificate there.
                    self.assertTrue(solution["saturated_boundary_coordinates"])
                    continue
                self.assertAlmostEqual(kt.bernoulli_kl(pi, qi),
                                       solution["information_rate"], places=7)

    def test_batched_accounting_and_forced_exploration(self):
        state = kt.TrackStopState([-.75, -.25, .75, .25])
        self.assertEqual(state.certificate()["decision"], "abstain")
        rng = random.Random(7)
        for _ in range(5):
            recommendation = state.recommend_counts(256)
            self.assertEqual(sum(recommendation), 256)
            self.assertTrue(all(isinstance(n, int) and n >= 0 for n in recommendation))
            for arm, n in enumerate(recommendation):
                probability = [.1, .2, .4, .5][arm]
                s = sum(rng.random() < probability for _ in range(n))
                state.update(arm, s, n)
        self.assertEqual(sum(state.trials), 1280)
        self.assertTrue(all(n > 0 for n in state.trials))
        self.assertEqual(state.certificate()["decision"], "reject")

    def test_fast_screen_cannot_authorize_an_unaudited_decision(self):
        state = kt.TrackStopState([-1, 1])
        state.update(0, 1, 10)
        state.update(1, 2, 10)
        screened = state.certificate()
        self.assertEqual(screened["decision"], "abstain")
        self.assertEqual(screened["audit"], "no_decision_authorized")
        state.update(0, 10, 1000)
        state.update(1, 200, 1000)
        screened = state.certificate()
        full = state.certificate(full_audit=True)
        self.assertEqual(screened, full)
        self.assertEqual(screened["decision"], "reject")


if __name__ == "__main__":
    unittest.main()
