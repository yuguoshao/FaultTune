"""Deterministic feasibility probe for fixed-prefix unconditional GLR gates.

This is not a released stopping method or a threshold-tuning implementation.
It computes exact rational Bernstein coefficients and certified upper bounds
for fixed examples. No simulator observations or Monte Carlo are used.
"""
from fractions import Fraction
from itertools import product
from pathlib import Path
import argparse
import json
import math
import time

import exact_sign_boundary as beta

ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def glr_ratio(kminus, kplus, pairs):
    """Unrestricted two-binomial MLE likelihood / common-p null MLE likelihood."""
    a, b, m = kminus, kplus, pairs
    total = a+b
    # Python's 0**0 == 1 supplies the zero-count likelihood convention.
    numerator = 2**(2*m)*a**a*(m-a)**(m-a)*b**b*(m-b)**(m-b)
    denominator = total**total*(2*m-total)**(2*m-total)
    return Fraction(numerator, denominator)


def glr_gate(pairs, likelihood_ratio_threshold=Fraction(100)):
    require(type(pairs) is int and pairs >= 1 and likelihood_ratio_threshold > 1, 'Invalid fixed gate specification')
    gate = [[0]*(pairs+1) for unused in range(pairs+1)]
    for minus in range(pairs+1):
        for plus in range(pairs+1):
            if minus != plus and glr_ratio(minus, plus, pairs) >= likelihood_ratio_threshold:
                gate[minus][plus] = 1 if plus > minus else -1
    validate_gate(gate)
    return gate


def validate_gate(gate):
    m = len(gate)-1
    require(m >= 1 and all(len(row) == m+1 for row in gate), 'Gate must be a square count table')
    for a in range(m+1):
        for b in range(m+1):
            decision = gate[a][b]
            require(type(decision) is int and decision in (-1, 0, 1), 'Gate direction must be an integer')
            require(decision == -gate[b][a] and (a != b or decision == 0), 'Gate lacks swap symmetry or excludes no ties')
            if decision > 0:
                require(b > a and (a == 0 or gate[a-1][b] == 1) and (b == m or gate[a][b+1] == 1),
                        'Positive gate is not decreasing in minus and increasing in plus')
            if decision < 0:
                require(a > b and (a == m or gate[a+1][b] == -1) and (b == 0 or gate[a][b-1] == -1),
                        'Negative gate is not increasing in minus and decreasing in plus')


def backward_survival(table, retain_through):
    """w_d(b) counts future fair suffixes; only d<=retain_through are retained."""
    beta.validate_table(table)
    h = len(table)-1
    require(0 <= retain_through <= h, 'Invalid retained prefix horizon')
    row = [int(h-table[h] < b < table[h]) for b in range(h+1)]
    retained = {h: tuple(row)} if h <= retain_through else {}
    for d in range(h-1, -1, -1):
        row = [row[b]+row[b+1] if d-table[d] < b < table[d] else 0 for b in range(d+1)]
        if d <= retain_through:
            retained[d] = tuple(row)
    return retained


def safe_physical_prefix_counts(table, pairs):
    """Ordered physical category-word counts indexed by (D, plus-only, both-one)."""
    states = {(0, 0, 0): 1}
    visited, transitions = 0, 0
    for t in range(pairs):
        visited += len(states)
        fresh = {}
        for (d, b, both), count in states.items():
            for key in ((d, b, both), (d+1, b+1, both), (d+1, b, both), (d, b, both+1)):
                transitions += 1
                nd, nb, nj = key
                if nd-table[nd] < nb < table[nd]:
                    fresh[key] = fresh.get(key, 0)+count
        states = fresh
    return states, {'prefix_states_visited': visited, 'category_transitions': transitions,
                    'terminal_states': len(states), 'terminal_state_ceiling': math.comb(pairs+3, 3)}


def bernstein_coefficients(gate, table, future):
    validate_gate(gate)
    m, h = len(gate)-1, len(table)-1
    require(m <= h and all(d in future for d in range(m+1)), 'Future suffix rows missing')
    ordinary = [0]*(2*m+1)
    for minus in range(m+1):
        for plus in range(m+1):
            if gate[minus][plus]:
                ordinary[minus+plus] += math.comb(m, minus)*math.comb(m, plus)
    states, work = safe_physical_prefix_counts(table, m)
    joint = [0]*(2*m+1)
    for (d, plus_only, both), count in states.items():
        minus, plus = d-plus_only+both, plus_only+both
        if gate[minus][plus]:
            total = d+2*both
            joint[total] += count*future[d][plus_only]*2**d
    ordinary_coefficients = [Fraction(value, math.comb(2*m, total)) for total, value in enumerate(ordinary)]
    joint_coefficients = [Fraction(value, 2**h*math.comb(2*m, total)) for total, value in enumerate(joint)]
    require(all(0 <= joint <= ordinary <= 1 for joint, ordinary in zip(joint_coefficients, ordinary_coefficients)),
            'Overlap Bernstein coefficients exceed the containing gate probability')
    return ordinary_coefficients, joint_coefficients, work


def split_bernstein(coefficients):
    """Exact rational de Casteljau subdivision at the interval midpoint."""
    row = tuple(coefficients)
    left, right = [row[0]], [row[-1]]
    while len(row) > 1:
        row = tuple((a+b)/2 for a, b in zip(row[:-1], row[1:]))
        left.append(row[0])
        right.append(row[-1])
    return tuple(left), tuple(reversed(right))


def bernstein_value(coefficients, p):
    n = len(coefficients)-1
    return sum((coefficient*math.comb(n, t)*p**t*(1-p)**(n-t)
                for t, coefficient in enumerate(coefficients)), Fraction(0))


def supremum_bound(coefficients, tolerance=Fraction(1, 10**8), max_boxes=2048, max_depth=32):
    """A rigorous rational upper bound, not an asserted exact supremum."""
    boxes = [('', tuple(coefficients))]
    lower = max(coefficients[0], coefficients[-1])
    reason = 'tolerance_met'
    while True:
        index = max(range(len(boxes)), key=lambda j: max(boxes[j][1]))
        path, row = boxes[index]
        upper = max(row)
        if upper-lower <= tolerance:
            break
        if len(boxes) >= max_boxes or len(path) >= max_depth:
            reason = 'finite_work_limit_upper_bound_retained'
            break
        left, right = split_bernstein(row)
        lower = max(lower, left[-1])
        boxes[index] = (path+'0', left)
        boxes.append((path+'1', right))
    upper = max(max(row) for path, row in boxes)
    return {'lower': lower, 'upper': upper, 'leaf_paths': [path for path, row in boxes],
            'leaf_count': len(boxes), 'max_depth': max(len(path) for path, row in boxes),
            'termination': reason, 'tolerance': tolerance}


def verify_supremum_bound(coefficients, certificate):
    paths = certificate['leaf_paths']
    require(len(paths) == len(set(paths)) and
            sum((Fraction(1, 2**len(path)) for path in paths), Fraction(0)) == 1,
            'Subdivision leaves do not cover unit mass')
    require(all(not a.startswith(b) for a in paths for b in paths if a != b), 'Subdivision leaves overlap')
    upper, endpoints = Fraction(0), []
    for path in paths:
        require(all(bit in '01' for bit in path), 'Invalid dyadic subdivision path')
        row = tuple(coefficients)
        for bit in path:
            row = split_bernstein(row)[int(bit)]
        upper = max(upper, max(row))
        endpoints.extend((row[0], row[-1]))
    require(upper == certificate['upper'] and max(endpoints) == certificate['lower'], 'Subdivision bound does not reproduce')
    return True


def small_exact_checks():
    for coefficients, known in (((Fraction(0), Fraction(1), Fraction(0)), Fraction(1, 2)),
                                ((Fraction(1, 3),)*5, Fraction(1, 3)),
                                ((Fraction(0), Fraction(1)), Fraction(1))):
        bound = supremum_bound(coefficients)
        verify_supremum_bound(coefficients, bound)
        require(bound['lower'] == bound['upper'] == known, 'Known Bernstein maximum differs')
    cases = 0
    for h, m in ((3, 2), (5, 3)):
        table = beta.constant_boundary(h, Fraction(3))
        future = backward_survival(table, m)
        gate = glr_gate(m, Fraction(3))
        ordinary, joint, work = bernstein_coefficients(gate, table, future)
        for p in (Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(1)):
            direct_gate, direct_joint = Fraction(0), Fraction(0)
            for word in product(range(4), repeat=m):
                d, b, both, crossed = 0, 0, 0, False
                total_successes = 0
                for category in word:
                    d += int(category in (1, 2))
                    b += int(category == 1)
                    both += int(category == 3)
                    total_successes += 2 if category == 3 else int(category in (1, 2))
                    crossed |= b >= table[d] or b <= d-table[d]
                probability = p**total_successes*(1-p)**(2*m-total_successes)
                if not gate[d-b+both][b+both]:
                    continue
                direct_gate += probability
                if crossed:
                    continue
                surviving_suffixes = 0
                for suffix in product((0, 1), repeat=h-d):
                    nd, nb, failed = d, b, False
                    for bit in suffix:
                        nd += 1
                        nb += bit
                        failed |= nb >= table[nd] or nb <= nd-table[nd]
                    surviving_suffixes += not failed
                direct_joint += probability*Fraction(surviving_suffixes, 2**(h-d))
            require(direct_gate == bernstein_value(ordinary, p) and direct_joint == bernstein_value(joint, p),
                    'Physical-word/fair-suffix enumeration differs from Bernstein coefficient formula')
            cases += 1
    return {'status': 'passed', 'exact_physical_prefix_suffix_evaluations': cases,
            'known_bernstein_extrema': 3, 'random_observations': 0}


def printable_bound(bound):
    return {**{key: value for key, value in bound.items() if key not in ('lower', 'upper', 'tolerance')},
            **{key: beta.fraction_record(bound[key]) for key in ('lower', 'upper', 'tolerance')},
            'lower_float_diagnostic': float(bound['lower']), 'upper_float_diagnostic': float(bound['upper'])}


def run_probe():
    checks = small_exact_checks()
    h, beta_threshold, gate_threshold = 2048, Fraction(100), Fraction(100)
    began = time.perf_counter()
    table = beta.constant_boundary(h, beta_threshold)
    future = backward_survival(table, 16)
    backward_seconds = time.perf_counter()-began
    p_beta = Fraction(2**h-future[0][0], 2**h)
    require(beta.count_surviving_paths(table)['crossing_numerator'] == p_beta.numerator*(2**h//p_beta.denominator),
            'Backward and forward beta probabilities differ')
    examples = []
    for m in (8, 12, 16):
        start = time.perf_counter()
        gate = glr_gate(m, gate_threshold)
        ordinary, joint, work = bernstein_coefficients(gate, table, future)
        gate_bound, joint_bound = supremum_bound(ordinary), supremum_bound(joint)
        verify_supremum_bound(ordinary, gate_bound)
        verify_supremum_bound(joint, joint_bound)
        null_upper, strict_upper = p_beta+joint_bound['upper'], p_beta+gate_bound['upper']/2
        examples.append({'prefix_pairs': m, 'physical_gate_checkpoint': 2*m, 'gate': gate,
                         'gate_glr_ratio_threshold_exact': beta.fraction_record(gate_threshold),
                         'gate_only_bernstein_coefficients': [beta.fraction_record(value) for value in ordinary],
                         'gate_beta_nonoverlap_bernstein_coefficients': [beta.fraction_record(value) for value in joint],
                         'gate_only_supremum_bounds': printable_bound(gate_bound),
                         'nonoverlap_supremum_bounds': printable_bound(joint_bound),
                         'null_union_upper_exact': beta.fraction_record(null_upper),
                         'strict_wrong_union_upper_exact': beta.fraction_record(strict_upper),
                         'null_union_upper_float_diagnostic': float(null_upper),
                         'strict_wrong_union_upper_float_diagnostic': float(strict_upper),
                         'both_bounds_le_001_in_this_fixed_example': null_upper <= Fraction(1, 100) and strict_upper <= Fraction(1, 100),
                         'work': work, 'elapsed_seconds': time.perf_counter()-start})
    return {'status': 'deterministic_feasibility_probe_not_released_protocol', 'checks': checks,
            'fixed_example_design': {'H': h, 'beta_threshold_exact': beta.fraction_record(beta_threshold),
                                     'gate_glr_ratio_threshold_exact': beta.fraction_record(gate_threshold),
                                     'm_values': [8, 12, 16], 'threshold_search_performed': False},
            'beta_latent_crossing_probability_exact': beta.fraction_record(p_beta),
            'beta_latent_crossing_probability_float_diagnostic': float(p_beta),
            'backward_dp_elapsed_seconds': backward_seconds, 'examples': examples,
            'proof_status': 'Coefficient identity and strict gate-alone coupling discussed with independent theory reviewer; no general joint least-favorable-null claim.',
            'assumptions': 'The first m pairs use fixed PSR and independent iid Bernoulli minus/plus samples. Under the null they have one common constant p. This polynomial calibration does not cover drifting/adaptive p_i in the first m pairs.',
            'limits': 'Rational Bernstein subdivision certifies upper bounds on suprema, not exact suprema. This probe is not an optimized gate, stopping-cost comparison, novelty result, or authorized fresh-sampling release.',
            'sources': {**beta.source_hashes(), Path(__file__).resolve().relative_to(ROOT).as_posix(): beta.file_sha256(__file__)}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/sota_round3/unconditional_gate_feasibility.json')
    args = parser.parse_args()
    result = run_probe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'output': str(args.output),
                      'backward_dp_seconds': result['backward_dp_elapsed_seconds'],
                      'examples': [{key: example[key] for key in ('prefix_pairs', 'null_union_upper_float_diagnostic',
                                   'strict_wrong_union_upper_float_diagnostic', 'elapsed_seconds')} for example in result['examples']]}))
