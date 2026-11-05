"""Independent generic-table DP audit plus separately labelled API checks."""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'experiments/sota_round3'))
import generic_sign_boundary as producer
from verify_integer_tail_dp import audit_artifact, fraction, full_vector_dp
from integer_tail_dp_checks import checks as small_table_checks


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def api_checks(artifact):
    runtime = producer.verify_artifact(artifact)
    table = artifact['boundaries']
    horizon = len(table)-1
    decisions = 0
    for d in (0, 1, 2, 7, 8, 9, 10, 31, 128, horizon):
        if d > horizon:
            continue
        for b in sorted({0, d, d//2, table[d]-1, table[d]}):
            if not 0 <= b <= d:
                continue
            result = producer.certificate_from_pairs([3, b, d-b, 5], runtime)
            expected = 'reject' if b >= table[d] else 'accept' if b <= d-table[d] else 'abstain'
            require(result['decision'] == expected and result['shots'] == 2*(d+8),
                    'Runtime differs from integer table or paid-pair count')
            decisions += 1
    rejected = []
    def must_fail(name, function):
        try:
            function()
        except (ValueError, TypeError, KeyError):
            rejected.append(name)
        else:
            raise AssertionError('Invalid runtime input accepted: '+name)
    for name, counts in [('negative_counts', [0, -1, 1, 0]),
                         ('boolean_counts', [True, 0, 0, 0]),
                         ('fractional_counts', [0, 0.5, 0, 0]),
                         ('beyond_horizon', [0, horizon+1, 0, 0])]:
        must_fail(name, lambda counts=counts: producer.certificate_from_pairs(counts, runtime))
    return {'runtime_decisions_checked': decisions, 'runtime_inputs_rejected': rejected}


def malformed_artifact_checks(base):
    rejected = []
    def mutation(name, transform):
        altered = deepcopy(base)
        transform(altered)
        altered['artifact_sha256'] = producer.payload_hash(altered)
        try:
            producer.verify_artifact(altered)
        except (ValueError, TypeError, KeyError):
            rejected.append(name)
        else:
            raise AssertionError('Malformed artifact accepted: '+name)
    mutation('zero_delta_denominator', lambda a: a['delta_exact'].__setitem__('denominator', '0'))
    mutation('zero_delta_numerator', lambda a: a['delta_exact'].__setitem__('numerator', '0'))
    mutation('negative_delta_denominator', lambda a: a['delta_exact'].__setitem__('denominator', '-100'))
    mutation('unknown_version', lambda a: a.__setitem__('version', 'unknown'))
    mutation('unpassed_status', lambda a: a.__setitem__('status', 'not_certified'))
    mutation('missing_source_bindings', lambda a: a.__setitem__('source_hashes', {}))
    mutation('boolean_horizon', lambda a: a.__setitem__('max_discordances', True))
    mutation('boolean_zero_sentinel', lambda a: a['boundaries'].__setitem__(0, True))
    mutation('zero_score_crossing', lambda a: a['boundaries'].__setitem__(2, 1))
    mutation('incorrect_risk_count', lambda a: a['fair_null_risk'].__setitem__('crossing_numerator', '0'))
    def overspend(a):
        a['boundaries'] = [1]+[d//2+1 for d in range(1, a['max_discordances']+1)]
        denominator = 1 << a['max_discordances']
        a['fair_null_risk']['crossing_numerator'] = str(denominator)
        a['fair_null_risk']['fair_denominator'] = str(denominator)
    mutation('exact_risk_over_budget', overspend)
    for scale in (0, -1, True):
        try:
            producer.calibrate(2, 'maxsprt', log_threshold_scale=scale)
        except ValueError:
            rejected.append('invalid_scale_'+str(scale))
        else:
            raise AssertionError('Invalid search scale accepted')
    return rejected


def main(output):
    shapes = ['maxsprt', 'pair_grid', 'pairwise_hedge', 'sprt_0.6', 'sprt_0.75', 'sprt_0.9', 'sprt_0.99']
    inputs = {shape: ROOT/('results/sota_round3/boundaries/'+shape+'.json') for shape in shapes}
    before = {shape: sha(path) for shape, path in inputs.items()}
    reports, first = [], None
    for shape, path in inputs.items():
        artifact = json.loads(path.read_text(encoding='utf-8'))
        require(artifact['shape'] == shape and artifact['max_discordances'] == 2048 and
                fraction(artifact['delta_exact']) == Fraction(1, 100), 'Registered shape/horizon/risk differs')
        independent = audit_artifact(artifact)
        independent.update(shape=shape, artifact_path=path.relative_to(ROOT).as_posix(),
                           artifact_file_sha256=before[shape],
                           first_homogeneous_discordance=next((d for d in range(1, 2049)
                                                               if artifact['boundaries'][d] <= d), None),
                           runtime_api_checks=api_checks(artifact))
        reports.append(independent)
        first = artifact if first is None else first
    rejections = malformed_artifact_checks(first)
    exhaustive = small_table_checks()
    require(before == {shape: sha(path) for shape, path in inputs.items()}, 'Input changed during audit')
    source_names = ['experiments/sota_round3/generic_sign_boundary.py',
                    'experiments/sota_round2/theory/audit_generic_boundaries.py',
                    'experiments/sota_round2/theory/verify_integer_tail_dp.py',
                    'experiments/sota_round2/theory/integer_tail_dp_checks.py',
                    'results/sota_round2/theory/generic_symmetric_tail_dp_lemma.json']
    result = {'status': 'passed', 'sources': {name: sha(ROOT/name) for name in source_names},
              'boundaries': reports, 'malformed_artifact_rejections': rejections,
              'small_horizon_exhaustive_checks': exhaustive,
              'new_simulation_observations': 0,
              'independence': 'Risk reconstruction imports no producer code; runtime and rejection checks intentionally call the production API and are separately labelled.',
              'limits': ['Integer table risk is exact; real GLR or other floating score level sets are not certified.',
                         'Each table separately controls 1% through D=2048 under the stated common-ordering model. Their union is not certified.',
                         'No novel-method or performance-superiority claim follows from deterministic boundary calibration.']}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'output': str(output), 'sha256': sha(output),
                      'boundaries': [{'shape': row['shape'], 'risk': row['float_probability_diagnostic'],
                                      'first_homogeneous_d': row['first_homogeneous_discordance']} for row in reports]}))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
