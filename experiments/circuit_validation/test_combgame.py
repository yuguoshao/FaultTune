"""Independent identities and accounting checks for the CombGame baseline."""

import math
import random
import unittest
from types import SimpleNamespace

import combgame as cg
from kl_trackstop import bernoulli_kl, joint_glr_certificate


class AdaHedgeTests(unittest.TestCase):
    def test_first_update_matches_hand_calculation(self):
        learner = cg.AdaHedge(2)
        self.assertEqual(learner.weights(), [.5, .5])
        learner.update([1.0, 0.0])
        self.assertEqual(learner.mixability_gap, .5)
        self.assertAlmostEqual(learner.eta, 2 * math.log(2))
        self.assertAlmostEqual(learner.weights()[0], .8)
        self.assertAlmostEqual(learner.weights()[1], .2)
        learner.update([0.0, 1.0])
        expected_mix_loss = -math.log(.8 * .25 + .2) / (2 * math.log(2))
        self.assertAlmostEqual(learner.mixability_gap, .5 + .8 - expected_mix_loss)
        self.assertEqual(learner.weights(), [.5, .5])

    def test_common_shifts_and_global_positive_scale_leave_weights_invariant(self):
        original, transformed = cg.AdaHedge(4), cg.AdaHedge(4)
        rng = random.Random(199)
        for step in range(80):
            rewards = [rng.uniform(-3, 5) for unused in range(4)]
            for first, second in zip(original.weights(), transformed.weights()):
                self.assertAlmostEqual(first, second, places=12)
            original.update(rewards)
            transformed.update([7.0 * r + 3.0 * step for r in rewards])

    def test_constant_rewards_preserve_infinite_rate_and_uniform_weights(self):
        learner = cg.AdaHedge(4)
        for value in range(5):
            learner.update([float(value)] * 4)
        self.assertTrue(math.isinf(learner.eta))
        self.assertEqual(learner.weights(), [.25] * 4)

    def test_zero_weight_limit_uses_cumulative_mix_loss(self):
        learner = cg.AdaHedge(2)
        learner.losses = [0.0, 2.0]
        learner.update([0.0, 3.0])
        self.assertEqual(learner.mixability_gap, 1.0)
        self.assertEqual(learner.losses, [1.0, 0.0])


class BestResponseTests(unittest.TestCase):
    def test_two_arm_solution_is_weighted_pool(self):
        for p, w in (([.2, .7], [.3, .7]), ([0., 1.], [.8, .2]),
                     ([.8, .1], [0., 1.]), ([.8, .1], [1., 0.])):
            q = cg.best_response(p, w, [-1., 1.])
            pooled = sum(pi * wi for pi, wi in zip(p, w))
            pooled = max(cg.ALTERNATIVE_CLIP, min(1 - cg.ALTERNATIVE_CLIP, pooled))
            for qi in q:
                self.assertAlmostEqual(qi, pooled, places=10)

    def test_zero_weights_absorb_equality_without_cost(self):
        p = [.1, .9, .9, .1]
        w = [.5, .5, 0., 0.]
        a = [-.3, -.7, .3, .7]
        q = cg.best_response(p, w, a)
        self.assertAlmostEqual(q[0], p[0])
        self.assertAlmostEqual(q[1], p[1])
        self.assertAlmostEqual(sum(ai * qi for ai, qi in zip(a, q)), 0., places=13)

    def test_kkt_stationarity_on_random_interior_problems(self):
        rng = random.Random(33)
        a = [-.2, -.8, .2, .8]
        for unused in range(40):
            p = [rng.uniform(.05, .95) for j in a]
            w = [rng.uniform(.01, 1) for j in a]
            total = sum(w)
            w = [x / total for x in w]
            q = cg.best_response(p, w, a)
            multipliers = [-wi * (qi - pi) / (qi * (1 - qi) * ai)
                           for pi, qi, wi, ai in zip(p, q, w, a)]
            self.assertLess(max(multipliers) - min(multipliers), 1e-9)
            self.assertAlmostEqual(sum(ai * qi for ai, qi in zip(a, q)), 0., places=11)

    def test_tiny_and_zero_weights_have_finite_feasible_responses(self):
        for w in ([1., 1e-200, 1e-200, 0.], [0., 1., 0., 0.]):
            q = cg.best_response([0., .3, .9, 1.], w, [-.2, -.8, .2, .8])
            self.assertTrue(all(0 < qi < 1 for qi in q))
            self.assertAlmostEqual(sum(a * qi for a, qi in zip([-.2, -.8, .2, .8], q)), 0., places=8)

    def test_compact_constraint_solves_endpoint_problem(self):
        q = cg.best_response([0., 0.], [.3, .7], [-1., 1.], clip=.01)
        self.assertEqual(q, [.01, .01])

    def test_compact_projection_already_in_alternative_is_returned(self):
        a = [-.2, -.8, .2, .8]
        p = [8e-9, 0., 0., 2.025e-9]
        self.assertGreater(sum(ai * pi for ai, pi in zip(a, p)), 0.)
        expected = [max(1e-9, pi) for pi in p]
        self.assertLess(sum(ai * qi for ai, qi in zip(a, expected)), 0.)
        actual = cg.best_response(p, [.25] * 4, a, answer_sign=1)
        self.assertEqual(actual, expected)


class OptimismTests(unittest.TestCase):
    def test_exploration_bonus_solves_lambert_equation(self):
        for rounds in (1, 2, 4, 10, 512, 100000):
            bonus = cg.exploration_bonus(rounds)
            self.assertAlmostEqual(bonus - math.log(bonus), max(1., 4 * math.log(rounds)), places=12)

    def test_optimism_dominates_grid_of_confidence_points(self):
        p, n, q, block, bonus = [.2, .7], [500, 1000], [.3, .3], 10, 3.
        rewards = cg.optimistic_rewards(p, n, q, block, bonus)
        for pi, ni, qi, ri in zip(p, n, q, rewards):
            self.assertGreaterEqual(ri, bonus / (ni / block))
            for grid in range(1001):
                value = grid / 1000.
                if bernoulli_kl(pi, value) <= bonus / ni:
                    self.assertLessEqual(block * bernoulli_kl(value, qi), ri + 1e-9)

    def test_endpoint_optimism_is_finite(self):
        rewards = cg.optimistic_rewards([0., 1.], [10, 10], [1e-9, 1 - 1e-9], 10, 4.)
        self.assertTrue(all(math.isfinite(value) and value >= 4 for value in rewards))


class StateAndAccountingTests(unittest.TestCase):
    def test_c_tracking_discrepancy_and_separate_learners(self):
        state = cg.CombGameState([-.25, -.75, .25, .75], 20)
        rng = random.Random(2)
        for step in range(400):
            arm = state.recommend_arm()
            state.update(arm, sum(rng.random() < (.3 + .1 * (arm >= 2)) for j in range(20)), 20)
            for n, target in zip(state.blocks, state.cumulative_weights):
                self.assertLessEqual(n - target, 1. + 1e-10)
                self.assertGreaterEqual(n - target, -3. - 1e-10)
        self.assertEqual(sum(l.rounds for l in state.learners), 396)
        self.assertIsNot(state.learners[0], state.learners[1])

    def test_raw_certificate_is_exactly_the_shared_certificate(self):
        state = cg.CombGameState([-1., 1.], 1000)
        state.update(state.recommend_arm(), 400, 1000)
        state.update(state.recommend_arm(), 100, 1000)
        observed = state.certificate(full_audit=True)
        expected = joint_glr_certificate([400, 100], [1000, 1000], [-1., 1.])
        self.assertEqual(observed, expected)
        self.assertEqual(observed['decision'], 'accept')

    def test_run_conserves_full_budget_and_accounts_for_final_partial(self):
        calls = []
        config = SimpleNamespace(coefficients=[-.5, -.5, .5, .5], delta=.01, shot_budget=1001)
        def sample(arm, count):
            calls.append((arm, count))
            return 0
        result = cg.run_combgame(config, sample)
        self.assertEqual(sum(n for arm, n in calls), 1001)
        self.assertEqual(result['checks'], 501)
        self.assertEqual([arm for arm, n in calls[:4]], [0, 1, 2, 3])
        self.assertTrue(all(n == 2 for arm, n in calls[:-1]))
        self.assertEqual(calls[-1][1], 1)
        self.assertEqual(result['sampling_metadata']['complete_blocks'], 500)
        self.assertEqual(result['sampling_metadata']['final_partial_shots'], 1)
        self.assertEqual(result['certificate']['shots'], 1001)
        self.assertEqual(result['decision'], 'abstain')

    def test_partial_block_is_terminal_and_never_enters_macro_estimates(self):
        state = cg.CombGameState([-1., 1.], 4)
        arm = state.recommend_arm()
        state.update(arm, 1, 2, final_partial=True)
        self.assertEqual(state.blocks, [0, 0])
        self.assertEqual(state.raw.trials, [2, 0])
        with self.assertRaises(RuntimeError):
            state.recommend_arm()

    def test_public_config_needs_no_hidden_truth_or_pair_oracle(self):
        config = SimpleNamespace(coefficients=[-1., 1.], delta=.01, shot_budget=20000)
        calls = []
        def sample(arm, count):
            calls.append((arm, count))
            return count if arm == 0 else 0
        result = cg.run_combgame(config, sample)
        self.assertEqual(result['decision'], 'accept')
        self.assertLess(sum(n for arm, n in calls), config.shot_budget)
        self.assertEqual(result['checks'], len(calls))
        self.assertEqual(result['certificate']['shots'], sum(n for arm, n in calls))


if __name__ == '__main__':
    unittest.main()
