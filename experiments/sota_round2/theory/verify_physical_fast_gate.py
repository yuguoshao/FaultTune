"""Independent full-vector physical-prefix risk-bound verification.

No production boundary module is imported and no simulation is performed.
"""
from fractions import Fraction
from itertools import product
import argparse
import hashlib
import json
from pathlib import Path

from verify_integer_tail_dp import (full_vector_dp, validate_table, jeffreys_coefficients,
                                   inspect_jeffreys_table, independent_jeffreys_table,
                                   fraction, exact_text)

ROOT = Path(__file__).resolve().parents[3]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def restricted_full_vector(table, m):
    """Propagate from d=0, retaining only homogeneous prefixes through m."""
    validate_table(table)
    require(type(m) is int and 1 <= m < len(table), 'Invalid gate length')
    vector = [1]
    for d, upper in enumerate(table[1:], 1):
        updated = [0]*(d+1)
        for b, count in enumerate(vector):
            updated[b] += count
            updated[b+1] += count
        for b in range(d+1):
            if b <= d-upper or b >= upper or (d <= m and b not in (0, d)):
                updated[b] = 0
        vector = updated
    return vector


def independently_computed_bounds(table, m):
    vector, first, crossing, denominator = full_vector_dp(table)
    restricted = sum(restricted_full_vector(table, m))
    horizon = len(table)-1
    require(0 <= restricted <= sum(vector) and restricted <= 1 << (horizon-m+1),
            'Restricted paths exceed containing sample space')
    beta_risk = Fraction(crossing, denominator)
    restricted_probability = Fraction(restricted, denominator)
    return {
        'beta_crossing_numerator': crossing, 'beta_denominator': denominator,
        'restricted_survivor_numerator': restricted, 'restricted_denominator': denominator,
        'null_union_numerator': (crossing << m)+restricted,
        'null_union_denominator': denominator << m,
        'strict_wrong_numerator': (crossing << (2*m))+denominator,
        'strict_wrong_denominator': denominator << (2*m),
        'null': beta_risk+restricted_probability/(1 << m),
        'strict': beta_risk+Fraction(1, 1 << (2*m))
    }


def compare_report(saved, calculated):
    for name, value in calculated.items():
        if name not in ('null', 'strict'):
            require(exact_text(saved[name]) == value, 'Risk-bound integer differs: '+name)


def small_enumeration():
    tables, prefix_tests, paths = 0, 0, 0
    for horizon in range(1, 6):
        for tail in product(*[range(d//2+1, d+2) for d in range(1, horizon+1)]):
            table = [1]+list(tail)
            tables += 1
            for m in range(1, horizon+1):
                expected = [0]*(horizon+1)
                for symbols in product((0, 1), repeat=horizon):
                    plus, hit = 0, False
                    for d, symbol in enumerate(symbols, 1):
                        plus += symbol
                        hit = hit or plus >= table[d] or plus <= d-table[d]
                    if not hit and len(set(symbols[:m])) == 1:
                        expected[plus] += 1
                    paths += 1
                require(restricted_full_vector(table, m) == expected,
                        'Restricted full-vector DP differs from complete sign enumeration')
                prefix_tests += 1
    return {'all_symmetric_tables_through_H': 5, 'tables': tables,
            'table_and_prefix_choices': prefix_tests, 'sign_strings_enumerated': paths}


def verify(path):
    initial_hash = sha(path)
    artifact = json.loads(Path(path).read_text(encoding='utf-8'))
    require(artifact['version'] == 'fixed_physical_prefix_plus_beta_v1' and
            artifact['status'] == 'fixed_prefix_joint_risk_bounds_certified', 'Wrong gate artifact type')
    horizon, m = artifact['max_discordances'], artifact['prefix_pairs']
    require(type(horizon) is int and horizon == 2048 and type(m) is int and m in (5, 6, 7),
            'Unregistered horizon or gate')
    require(type(artifact['max_physical_pairs']) is int and artifact['max_physical_pairs'] == horizon,
            'Physical pair budget differs')
    table = artifact['beta_boundaries']
    require(len(table) == horizon+1, 'Wrong beta table length')
    for name, expected in artifact['source_hashes'].items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Nonlocal source binding')
        require(sha(ROOT/relative) == expected, 'Frozen gate source changed: '+name)
    content = {key: value for key, value in artifact.items() if key != 'artifact_sha256'}
    require(hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest() == artifact['artifact_sha256'],
            'Gate content hash differs')
    core = {key: artifact[key] for key in ('version', 'max_discordances', 'prefix_pairs',
                                         'delta_exact', 'beta_threshold_exact', 'beta_boundaries')}
    require(hashlib.sha256(json.dumps(core, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            == artifact['boundary_sha256'], 'Gate decision hash differs')
    threshold, delta = fraction(artifact['beta_threshold_exact']), fraction(artifact['delta_exact'])
    require(delta == Fraction(1, 100), 'Risk is not exactly 1/100')
    coefficients = jeffreys_coefficients(horizon)
    inspect_jeffreys_table(table, threshold, coefficients)
    bound = independently_computed_bounds(table, m)
    compare_report(artifact['risk_bounds_exact'], bound)
    require(bound['null'] <= delta and bound['strict'] <= delta, 'Gate risk upper bound exceeds delta')
    selection = artifact['selection']
    require(selection['kind'] == 'least_jointly_certified_positive_threshold_on_fixed_rational_lattice',
            'Unrecognized gate threshold search')
    scale, index = selection['threshold_scale'], selection['chosen_lattice_index']
    require(type(scale) is int and scale > 0 and type(index) is int and index > 1 and
            threshold == Fraction(index, scale), 'Gate threshold is not on declared lattice')
    previous = selection['lower_neighbor']
    previous_threshold = fraction(previous['threshold_exact'])
    require(previous_threshold == Fraction(index-1, scale), 'Wrong gate lower-neighbor threshold')
    previous_table = independent_jeffreys_table(horizon, previous_threshold, coefficients)
    previous_bound = independently_computed_bounds(previous_table, m)
    compare_report(previous['risk_bounds_exact'], previous_bound)
    require(previous_bound['null'] > delta or previous_bound['strict'] > delta,
            'Lower lattice point meets both constraints')
    require(initial_hash == sha(path), 'Gate artifact changed during independent verification')
    return {'status': 'passed', 'artifact': Path(path).resolve().relative_to(ROOT).as_posix(),
            'artifact_file_sha256': initial_hash, 'prefix_pairs': m, 'threshold_exact': str(threshold),
            'null_union_upper_float': float(bound['null']), 'strict_wrong_upper_float': float(bound['strict']),
            'all_integer_bounds_recomputed': True, 'lower_lattice_neighbor_independently_verified': True,
            'producer_imported_for_risk_verification': False,
            'scope': 'Two sufficient risk upper bounds, not an estimate of actual physical stopping risk.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifacts', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    exhaustive = small_enumeration()
    reports = [verify(path) for path in args.artifacts]
    result = {'status': 'passed', 'artifacts': reports, 'restricted_dp_exhaustive_checks': exhaustive,
              'sources': {'experiments/sota_round2/theory/verify_physical_fast_gate.py': sha(__file__),
                          'experiments/sota_round2/theory/verify_integer_tail_dp.py': sha(ROOT/'experiments/sota_round2/theory/verify_integer_tail_dp.py'),
                          'results/sota_round2/theory/physical_fast_gate_lemma.json': sha(ROOT/'results/sota_round2/theory/physical_fast_gate_lemma.json')},
              'new_simulation_observations': 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'output': str(args.output), 'sha256': sha(args.output),
                      'artifacts': reports, 'exhaustive_checks': exhaustive}))
