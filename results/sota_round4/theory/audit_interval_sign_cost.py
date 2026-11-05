"""Independent exact physical-time checks and outward-arithmetic source review."""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'experiments/sota_round4'))
import interval_sign_cost as target


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encloses(pair, exact):
    assert all(math.isfinite(float(x)) for x in pair), 'Nonfinite interval'
    assert Fraction(float(pair[0])) <= exact <= Fraction(float(pair[1])), (pair, str(exact))


def physical_dp(table, pm, pp, cap):
    """A physical-time recursion, independent of sign/negative-binomial mixing."""
    probability = (pm*(1-pp), (1-pm)*pp, (1-pm)*(1-pp)+pm*pp)
    alive = {(0, 0): Fraction(1)}
    plus = minus = cost = Fraction(0)
    for _ in range(cap):
        cost += sum(alive.values(), Fraction(0))
        fresh = {}
        for (d, k), mass in alive.items():
            for increment_d, increment_k, rate in ((1, 0, probability[0]), (1, 1, probability[1]), (0, 0, probability[2])):
                newd, newk = d+increment_d, k+increment_k
                term = mass*rate
                if newk >= table[newd]:
                    plus += term
                elif newk <= newd-table[newd]:
                    minus += term
                else:
                    fresh[newd, newk] = fresh.get((newd, newk), Fraction(0))+term
        alive = fresh
    abstain = sum(alive.values(), Fraction(0))
    assert plus+minus+abstain == 1
    return {'plus': plus, 'minus': minus, 'abstain': abstain, 'expected_shots': 2*cost}


def main():
    source_hash = digest(target.__file__)
    # Small exact probabilities include subnormal and below-subnormal rates.
    small = Fraction(1, 2**1074)
    params = [(Fraction(0), Fraction(0)), (Fraction(1), Fraction(1)),
              (Fraction(0), Fraction(1)), (Fraction(1), Fraction(0)),
              (Fraction(1, 4), Fraction(3, 4)), (Fraction(1, 2), Fraction(1, 2)),
              (Fraction(0), small), (1-small, Fraction(1)),
              (Fraction(0), Fraction(1, 10**400)), (small, 1-small)]
    tables = [[1, 2, 2, 4, 4, 6, 6], [1, 2, 3, 4, 5, 6, 7],
              [1, 1, 2, 2, 3, 3, 4], [1, 2, 3, 3, 5, 4, 6]]
    comparisons = 0
    for table in tables:
        for cap in (1, 3, 6):
            for pm, pp in params:
                reference = physical_dp(table, pm, pp, cap)
                got = target.operating_intervals(table, pm, pp, cap)
                for key, value in reference.items():
                    encloses(got[key], value)
                    comparisons += 1
    # Direct exact binomial pmfs, heads and tails stress mode ties and underflow.
    binomial_comparisons = 0
    for n, p in [(0, Fraction(1, 3)), (12, Fraction(1, 2)), (13, Fraction(1, 2)),
                 (64, Fraction(1, 2**30)), (64, 1-Fraction(1, 2**30)),
                 (12, small), (12, 1-small), (12, Fraction(1, 10**400))]:
        got = target.binomial_intervals(n, p)
        masses = [math.comb(n,k)*p**k*(1-p)**(n-k) for k in range(n+1)]
        head = Fraction(0)
        for k, mass in enumerate(masses):
            head += mass
            tail = sum(masses[k:], Fraction(0))
            for lo, hi, value in ((got[0][k], got[1][k], mass),
                                  (got[2][k], got[3][k], head), (got[4][k], got[5][k], tail)):
                encloses((lo, hi), value)
                binomial_comparisons += 1
    # Summation envelope over mixed magnitudes and underflow-adjacent values.
    arrays = [np.array([0., target.TINY, 1., 2.**-52, 2.**-500]*1024),
              np.array([target.TINY]*8193), np.array([1., 0.]*4096)]
    prefix_comparisons = 0
    for array in arrays:
        low, high = target.prefix_enclosure(array, False), target.prefix_enclosure(array, True)
        value = Fraction(0)
        for x, a, b in zip(array, low, high):
            value += Fraction(float(x))
            encloses((a,b), value)
            prefix_comparisons += 1
        encloses((target.tree_sum(array, False), target.tree_sum(array, True)), value)
    assert digest(target.__file__) == source_hash, 'Target source changed during audit'
    result = {'status': 'passed', 'new_simulation_observations': 0,
              'target_source_sha256': source_hash,
              'checks': {'exact_physical_time_comparisons': comparisons,
                         'exact_binomial_pmf_head_tail_comparisons': binomial_comparisons,
                         'exact_prefix_sum_comparisons': prefix_comparisons,
                         'covered': ['r=0 and1', 'q=0 and1', 'minimum binary64 subnormal r',
                                     'positive rational r below binary64 range', 'near-one probabilities',
                                     'cap1 and shortened horizons', 'mode ties', 'never-stop and immediate-stop integer tables',
                                     'physical-time recursion independent of negative-binomial mixture']},
              'source_review': {
                  'probability_semantics': 'pminus and pplus are converted to exact Fractions; r and q are computed exactly before binary64 enclosure. Enclose_fraction compares the exact rational to Fraction(float(value)) and moves only a needed endpoint.',
                  'binomial_normalization': 'A mode has relative unnormalized mass1. Exact adjacent ratios are<=1 away from a mode, and positive interval multiplication encloses every relative mass. Their total is at least1, so max(1,total_lo) is a valid strengthened lower total. Dividing lower mass by total_hi and upper mass by total_lo gives valid pmf intervals.',
                  'point_operations': 'Under correctly rounded IEEE-754 binary64 basic arithmetic and gradual underflow, one nextafter step outward bounds each operation; clipping a negative lower endpoint to0 is valid for nonnegative quantities. Subnormal positive upper endpoints intentionally persist after numerical zero.',
                  'prefix_sum': 'For n*eps<1/16, the standard cumulative-addition relative error gamma_(n-1), with unit roundoff eps/2, is strictly below4*n*eps. The n*TINY absolute allowance covers gradual-underflow errors. Outward division by1+relative or1-relative and outward endpoint correction conservatively enclose prefixes. This assumes numpy.cumsum uses no more than n additions on a dependency path.',
                  'sign_dp': 'The contiguous unabsorbed interval is indexed by origin. Bernoulli transitions reach origin through origin+len(row), then symmetric upper/lower tails absorb first-crossing mass. Sign-only stopping has no direct dependence on physical waiting times.',
                  'waiting_identity': 'For d>=1 and r>0, E[min(NB(d,r),T)]=(d/r)P(Bin(T+1,r)>=d+1)+T*P(Bin(T,r)<d). The next-tail[2:] and head[:-1] slices correctly match d+1 and d-1 for d=1,...,T.',
                  'waiting_overflow_fix': 'Divide next-tail by r before multiplying by d. The upper first term may overflow but is safely capped atT; the whole truncated wait is<=T. If a positive exact r has binary64 lower endpoint0, use the rigorous upper waitT directly. The lower tail/rhi cannot overflow for these valid probabilities because its exact value is<= (T+1)/(d+1).',
                  'total_cost': 'Multiply first-crossing probabilities by truncated waits and add T times sign-survival probability. Multiply pair cost by2. This includes wrong reports and abstentions and agrees with an independent physical-time exact DP.',
                  'abstention': 'It equals1-P(plus)-P(minus); outward subtractions and probability clipping enclose it. Small floating widths are not empirical uncertainty intervals.',
                  'limitations': 'Analysis requires stationary independent Bernoulli arms, sign-only frozen boundary and inspection after every physical pair. It does not analyze old sparse checkpoints, adaptive arm selection, or physical-prefix gates. Source-level arithmetic review and finite exact checks are not formal verification of Python/numpy/IEEE execution.'},
              'source_sha256': {Path(__file__).resolve().relative_to(ROOT).as_posix(): digest(__file__),
                                'experiments/sota_round4/interval_sign_cost.py': source_hash},
              'runtime': {'python': sys.version, 'numpy': np.__version__}}
    path = Path(__file__).with_suffix('.json')
    with path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'artifact': str(path), 'sha256': digest(path), 'checks': result['checks']}))


if __name__ == '__main__':
    main()
