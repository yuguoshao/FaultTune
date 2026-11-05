"""Safety audit of prepared H=8192 boundaries, without new observations.

Rechecks complete risk counts and every selected contour edge. It deliberately
does not repeat lower-neighbor lattice optimization; that is not needed for
the risk guarantee and remains separately labelled producer evidence.
"""
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from verify_integer_tail_dp import audit_artifact, fraction, jeffreys_coefficients, inspect_jeffreys_table
from verify_physical_fast_gate import independently_computed_bounds, compare_report

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'results/sota_round3/cap16384_preparation'


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    files = [BASE/'beta_exact.json', BASE/'physical_gate_m6.json', BASE/'physical_gate_m7.json']
    before = {path.name: sha(path) for path in files}
    coefficients = jeffreys_coefficients(8192)
    reports = []
    for path in files:
        artifact = json.loads(path.read_text(encoding='utf-8'))
        require(type(artifact['max_discordances']) is int and artifact['max_discordances'] == 8192,
                'Preparation horizon differs')
        require(fraction(artifact['delta_exact']) == Fraction(1, 100), 'Preparation risk differs')
        for name, expected in artifact['source_hashes'].items():
            require(sha(ROOT/name) == expected, 'Preparation source changed: '+name)
        payload = {k: v for k, v in artifact.items() if k != 'artifact_sha256'}
        require(hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
                == artifact['artifact_sha256'], 'Preparation payload hash differs')
        if path.name == 'beta_exact.json':
            require(artifact['version'] == 'exact_finite_discordance_jeffreys_v1' and
                    artifact['status'] == 'exact_finite_horizon_boundary_certified', 'Wrong beta artifact type')
            threshold = fraction(artifact['threshold_exact'])
            inspect_jeffreys_table(artifact['boundaries'], threshold, coefficients)
            detail = audit_artifact(artifact, require_jeffreys=False)
            risk = {'fair_beta_crossing_probability': detail['float_probability_diagnostic']}
        else:
            m = 6 if path.name == 'physical_gate_m6.json' else 7
            require(artifact['version'] == 'fixed_physical_prefix_plus_beta_v1' and
                    artifact['status'] == 'fixed_prefix_joint_risk_bounds_certified' and
                    type(artifact['prefix_pairs']) is int and artifact['prefix_pairs'] == m and
                    type(artifact['max_physical_pairs']) is int and artifact['max_physical_pairs'] == 8192,
                    'Wrong gate preparation protocol')
            threshold = fraction(artifact['beta_threshold_exact'])
            inspect_jeffreys_table(artifact['beta_boundaries'], threshold, coefficients)
            calculated = independently_computed_bounds(artifact['beta_boundaries'], m)
            compare_report(artifact['risk_bounds_exact'], calculated)
            require(calculated['null'] <= Fraction(1, 100) and calculated['strict'] <= Fraction(1, 100),
                    'Prepared gate risk bound exceeds delta')
            risk = {'null_union_upper': float(calculated['null']), 'strict_wrong_upper': float(calculated['strict'])}
        reports.append({'status': 'passed', 'path': path.relative_to(ROOT).as_posix(),
                        'file_sha256': before[path.name], 'threshold_exact': str(threshold),
                        'all_8192_contour_edges_independently_checked': True,
                        'full_integer_risk_recomputed_without_producer_import': True,
                        'lattice_minimality_independently_recomputed': False, 'risk': risk})
        print(json.dumps(reports[-1]), flush=True)
    require(before == {path.name: sha(path) for path in files}, 'Prepared artifacts changed during audit')
    result = {'status': 'passed', 'physical_shot_cap': 16384, 'discordance_horizon': 8192,
              'artifacts': reports, 'new_simulation_observations': 0,
              'verifier_source_sha256': sha(__file__),
              'shared_independent_helpers': {p.relative_to(ROOT).as_posix(): sha(p) for p in
                   (ROOT/'experiments/sota_round2/theory/verify_integer_tail_dp.py',
                    ROOT/'experiments/sota_round2/theory/verify_physical_fast_gate.py')},
              'limits': ['Only deterministic boundary safety is established; no larger-budget observations were generated.',
                         'Lower-neighbor lattice optimality is producer evidence and was not independently repeated in this audit.',
                         'Each prepared method is separately certified; their union and switching caps after observing outcomes are not covered.']}
    output = ROOT/'results/sota_round2/theory/cap16384_preparation_independent_audit.json'
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': 'passed', 'path': output.as_posix(), 'sha256': sha(output)}), flush=True)


if __name__ == '__main__':
    main()
