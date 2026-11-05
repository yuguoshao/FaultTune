"""Independent differential and exact-special-case checks for the certifier."""
from decimal import Decimal
from fractions import Fraction as F
from pathlib import Path
import sys
import unittest

import mpmath as mp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.sota_round6.continuum_null_certificate import (
    I, Model, certify, enclosure, square)


def point(value):
    value = F(value)
    return mp.mpf(value.numerator)/value.denominator


def contained(value, bounds):
    return mp.mpf(str(bounds.lo)) <= value <= mp.mpf(str(bounds.hi))


class ContinuumChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mp.mp.dps = 100

    def test_square_crossing_zero(self):
        got = square(I(-2, 1))
        self.assertEqual((got.lo, got.hi), (Decimal(0), Decimal(4)))
        got = square(I(-3, -2))
        self.assertEqual((got.lo, got.hi), (Decimal(4), Decimal(9)))

    def test_derivatives_against_independent_differentiation(self):
        models = [((F(1, 3), F(1, 7), F(1, 9)), (F(2, 5), F(1, 6))),
                  ((F(1, 100), F(1, 200), F(-1, 200)),
                   (F(9, 1000), F(1, 250)))]
        for theta, null in models:
            model = Model(theta, null)
            A, B, C = map(point, theta)
            a, b = map(point, null)
            for sign in (-1, 1):
                def function(t):
                    cosine, sine = sign*(1-t*t)/(1+t*t), 2*t/(1+t*t)
                    p, q = A+B*cosine+C*sine, a+b*cosine
                    return p*mp.log(p/q)+(1-p)*mp.log((1-p)/(1-q))
                for t in (F(-1), F(-2, 5), F(0), F(3, 7), F(1)):
                    got = model.derivatives(enclosure(t), sign)
                    for order in (0, 1, 2):
                        self.assertTrue(contained(mp.diff(function, point(t), order), got[order]))
                box = I(enclosure(F(-2, 5)).lo, enclosure(F(3, 7)).hi)
                second = model.derivatives(box, sign)[2]
                for t in (F(-2, 5), F(0), F(3, 7)):
                    self.assertTrue(contained(mp.diff(function, point(t), 2), second))

    def test_exact_symmetric_capacity_and_leaf_partition(self):
        theta, null = (F(1, 2), F(0), F(1, 4)), (F(1, 2), F(0))
        result = certify(theta, null, F(0), F(1, 10000))
        self.assertEqual(result['status'], 'certified')
        exact = mp.mpf('0.75')*mp.log(mp.mpf('1.5'))+mp.mpf('0.25')*mp.log(mp.mpf('0.5'))
        lower = list(map(mp.mpf, result['pair_information_interval']))
        upper = mp.mpf(result['all_angle_information_upper'])
        self.assertLessEqual(lower[0], exact)
        self.assertLessEqual(exact, lower[1])
        self.assertLessEqual(exact, upper)
        self.assertLessEqual(upper/lower[0], point(F(10001, 10000)))
        for sign in (-1, 1):
            leaves = sorted((F(x['left']), F(x['right'])) for x in result['accepted_leaves']
                            if x['chart'] == sign)
            self.assertEqual(leaves[0][0], -1)
            self.assertEqual(leaves[-1][1], 1)
            self.assertTrue(all(left < right for left, right in leaves))
            self.assertTrue(all(x[1] == y[0] for x, y in zip(leaves, leaves[1:])))

    def test_fail_closed_invalid_models_and_exhausted_budget(self):
        with self.assertRaises(ValueError):
            Model((F(1, 2), 0, F(1, 2)), (F(1, 2), 0))
        with self.assertRaises(ValueError):
            Model((F(1, 2), 0, F(1, 4)), (F(1, 2), F(3, 4)))
        result = certify((F(1, 2), 0, F(1, 4)), (F(1, 2), 0),
                         0, max_nodes=1)
        self.assertEqual(result['status'], 'unresolved')
        self.assertNotIn('all_angle_information_upper', result)
        self.assertTrue(result['unresolved_leaves'])


if __name__ == '__main__':
    unittest.main()
