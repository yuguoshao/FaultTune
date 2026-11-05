"""Independent piecewise-prefix-gate artifact verification, without sampling."""
from fractions import Fraction
import argparse
import hashlib
import json
from pathlib import Path

from verify_integer_tail_dp import fraction, jeffreys_coefficients, independent_jeffreys_table
from verify_physical_fast_gate import independently_computed_bounds, compare_report

ROOT = Path(__file__).resolve().parents[3]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def verify(path):
    initial = sha(path)
    artifact = json.loads(path.read_text(encoding='utf-8'))
    require(artifact['version'] == 'fixed_early_beta_piecewise_physical_gate_v1' and
            artifact['status'] == 'piecewise_table_and_single_gate_risk_bounds_certified', 'Wrong piecewise artifact type')
    for name, expected in artifact['source_hashes'].items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Nonlocal source binding')
        require(sha(ROOT/relative) == expected, 'Piecewise source changed: '+name)
    require(content_hash({k: v for k, v in artifact.items() if k != 'artifact_sha256'}) == artifact['artifact_sha256'],
            'Piecewise payload hash differs')
    core = {key: artifact[key] for key in ('version', 'max_discordances', 'prefix_pairs', 'switch_discordances',
                                         'delta_exact', 'early_threshold_exact', 'late_threshold_exact', 'boundaries')}
    require(content_hash(core) == artifact['boundary_sha256'], 'Piecewise table hash differs')
    for key, expected in [('max_discordances', 2048), ('max_physical_pairs', 2048),
                          ('prefix_pairs', 6), ('switch_discordances', 512)]:
        require(type(artifact[key]) is int and artifact[key] == expected, 'Unregistered design value: '+key)
    delta, early_h, late_h = map(fraction, (artifact['delta_exact'], artifact['early_threshold_exact'], artifact['late_threshold_exact']))
    require(delta == Fraction(1, 100) and early_h == Fraction(40587151, 500000) and late_h >= early_h,
            'Risk or early/late threshold differs')
    binding = artifact['early_boundary_reference']
    require(binding['path'] == 'results/sota_round3/beta_boundary_N2048.json' and
            sha(ROOT/binding['path']) == binding['file_sha256'], 'Early-reference file changed')
    early = json.loads((ROOT/binding['path']).read_text(encoding='utf-8'))
    require(early['artifact_sha256'] == binding['artifact_sha256'], 'Early-reference payload differs')
    coefficients = jeffreys_coefficients(2048)
    independent_early = independent_jeffreys_table(2048, early_h, coefficients)
    require(independent_early == early['boundaries'], 'Independent early Jeffreys contour differs')
    independent_late = independent_jeffreys_table(2048, late_h, coefficients)
    joined = independent_early[:513]+independent_late[513:]
    require(joined == artifact['boundaries'] and joined[:513] == early['boundaries'][:513] and
            all(new >= old for new, old in zip(joined, independent_early)), 'Piecewise splice changed protected early decisions')
    bounds = independently_computed_bounds(joined, 6)
    compare_report(artifact['risk_bounds_exact'], bounds)
    require(bounds['null'] <= delta and bounds['strict'] <= delta, 'Piecewise risk exceeds delta')
    never = independent_early[:513]+[d+1 for d in range(513, 2049)]
    never_bounds = independently_computed_bounds(never, 6)
    same_bounds = independently_computed_bounds(independent_early, 6)
    compare_report(artifact['never_late_stop_feasibility_bounds'], never_bounds)
    compare_report(artifact['same_threshold_plus_gate_risk_bounds'], same_bounds)
    require(never_bounds['null'] <= delta and never_bounds['strict'] <= delta, 'No-late limiting table is infeasible')
    selection = artifact['selection']
    require(selection['kind'] == 'least_jointly_certified_late_threshold_on_fixed_rational_lattice', 'Wrong late-threshold search')
    scale, index = selection['threshold_scale'], selection['chosen_lattice_index']
    require(type(scale) is int and scale > 0 and type(index) is int and index > 0 and late_h == Fraction(index, scale),
            'Late threshold is not on declared lattice')
    minimum = (early_h.numerator*scale+early_h.denominator-1)//early_h.denominator
    require(selection['minimum_lattice_index'] == minimum and index >= minimum, 'Wrong lower search constraint')
    previous = selection['lower_neighbor']
    previous_h = fraction(previous['threshold_exact'])
    require(previous_h == Fraction(index-1, scale), 'Wrong lower late-threshold neighbor')
    previous_bounds = None
    if index == minimum:
        require(previous.get('excluded_below_early_threshold') is True and previous_h < early_h,
                'Predecessor is not excluded by early-threshold constraint')
    else:
        lower_late = independent_jeffreys_table(2048, previous_h, coefficients)
        lower_joined = independent_early[:513]+lower_late[513:]
        previous_bounds = independently_computed_bounds(lower_joined, 6)
        compare_report(previous['risk_bounds_exact'], previous_bounds)
        require(previous_bounds['null'] > delta or previous_bounds['strict'] > delta,
                'Lower late-threshold neighbor satisfies both constraints')
    require(initial == sha(path), 'Piecewise artifact changed during verification')
    return {'status': 'passed', 'artifact': path.resolve().relative_to(ROOT).as_posix(), 'artifact_file_sha256': initial,
            'early_threshold_exact': str(early_h), 'late_threshold_exact': str(late_h),
            'protected_early_indices': 513, 'changed_protected_entries': 0,
            'null_union_upper_float': float(bounds['null']), 'strict_wrong_upper_float': float(bounds['strict']),
            'never_late_null_upper_float': float(never_bounds['null']),
            'same_threshold_null_upper_float': float(same_bounds['null']),
            'lower_neighbor_null_upper_float': float(previous_bounds['null']) if previous_bounds else None,
            'all_table_entries_and_integer_risk_bounds_recomputed_without_producer_import': True,
            'late_lattice_lower_neighbor_independently_verified': True, 'new_simulation_observations': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.artifact)
    result['sources'] = {p.relative_to(ROOT).as_posix(): sha(p) for p in
                        (Path(__file__).resolve(), ROOT/'experiments/sota_round2/theory/verify_integer_tail_dp.py',
                         ROOT/'experiments/sota_round2/theory/verify_physical_fast_gate.py')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result))
