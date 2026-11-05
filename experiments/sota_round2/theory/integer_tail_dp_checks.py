"""Exhaustive small finite-boundary checks; no random observations."""
from fractions import Fraction
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'experiments/sota_round3'))
import exact_sign_boundary as exact
import generic_sign_boundary as generic
from verify_integer_tail_dp import full_vector_dp, inspect_jeffreys_table, jeffreys_coefficients


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exhaustive_path_counts(table):
    horizon = len(table)-1
    survivors, crossing = [0]*(horizon+1), 0
    for symbols in product((0, 1), repeat=horizon):
        plus, hit = 0, False
        for d, symbol in enumerate(symbols, 1):
            plus += symbol
            if plus >= table[d] or plus <= d-table[d]:
                hit = True
        if hit:
            crossing += 1
        else:
            survivors[plus] += 1
    return survivors, crossing


def checks():
    tables_checked, paths_checked, contour_states = 0, 0, 0
    for horizon in range(1, 7):
        choices = [range(d//2+1, d+2) for d in range(1, horizon+1)]
        for tail in product(*choices):
            table = [1]+list(tail)
            enumerated, crossed = exhaustive_path_counts(table)
            vector, first, independent_crossed, denominator = full_vector_dp(table)
            require(vector == enumerated and independent_crossed == crossed,
                    'Independent full-vector DP differs from path enumeration')
            require(generic.exact_risk(table) == (crossed, 1 << horizon),
                    'Generic DP differs from path enumeration')
            compact = exact.count_surviving_paths(table)
            require(compact['crossing_numerator'] == crossed and compact['fair_denominator'] == denominator,
                    'Compact DP differs from path enumeration')
            packed = [0]*(horizon+1)
            lower = compact['survivor_lower_plus']
            packed[lower:lower+len(compact['survivor_counts'])] = compact['survivor_counts']
            require(packed == vector, 'Compact survivor state differs from enumeration')
            require(list(compact['first_crossing_counts']) == first,
                    'Compact first-crossing decomposition differs')
            tables_checked += 1
            paths_checked += 1 << horizon
    for threshold in map(Fraction, ('1/3', '1', '1.000001', '2', '3.25', '10', '100')):
        table = list(exact.constant_boundary(48, threshold))
        inspect_jeffreys_table(table, threshold, jeffreys_coefficients(48))
        for d in range(1, 49):
            for b in range(d+1):
                c = d-b
                capital = Fraction(math.factorial(2*b)*math.factorial(2*c),
                                   (1 << d)*math.factorial(b)*math.factorial(c)*math.factorial(d))
                desired = 2*b != d and capital >= threshold
                encoded = b >= table[d] or b <= d-table[d]
                require(desired == encoded, 'Exact Jeffreys table differs from factorial contour')
                contour_states += 1
    shapes = ('maxsprt', 'beta', 'pair_grid', 'pairwise_hedge',
              'sprt_0.6', 'sprt_0.75', 'sprt_0.9', 'sprt_0.99')
    shape_tables = 0
    for shape in shapes:
        rows = generic.score_rows(12, shape)
        for level in (0., .1, 1., 2., 5., 20.):
            table = generic.level_table(rows, level)
            vector, first, crossed, denominator = full_vector_dp(table)
            require(generic.exact_risk(table) == (crossed, denominator), 'Score-shape risk differs')
            shape_tables += 1
    source_names = ['experiments/sota_round3/exact_sign_boundary.py',
                    'experiments/sota_round3/generic_sign_boundary.py',
                    'experiments/sota_round2/theory/verify_integer_tail_dp.py',
                    'experiments/sota_round2/theory/integer_tail_dp_checks.py']
    return {'status': 'passed', 'sources': {name: sha(ROOT/name) for name in source_names},
            'all_possible_symmetric_tables_through_horizon': 6,
            'symmetric_tables_exhaustively_checked': tables_checked,
            'full_sign_strings_enumerated': paths_checked,
            'jeffreys_factorial_contour_states': contour_states,
            'floating_shape_integer_tables_checked': shape_tables,
            'new_simulation_observations': 0,
            'limits': 'Exhaustive small-horizon implementation checks plus exact-contour checks; full-horizon artifacts require separate independent DP verification.'}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    report = checks()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(report))
