"""Same every-physical-pair operating-characteristic calculation for fixed priors.

No simulation, fitted priors, sparse-monitoring comparison, or interval claim.
"""
from pathlib import Path
import json
import math
import time

import analytic_sign_cost as analytic
import lr_mixture_boundaries as mixture

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT/'results/sota_round4/lr_mixture_boundaries/calibration_manifest.json'
BETA = ROOT/'results/sota_round3/cap16384_preparation/beta_exact.json'
CASE9_OUTPUT = ROOT/'results/sota_round4/analytic_case9_comparison.json'
FULL_OUTPUT = ROOT/'results/sota_round4/analytic_prior_comparison.json'


def main():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    analytic.require(manifest['status'] == 'complete' and manifest['max_discordances'] == 8192 and
                     manifest['source_hashes'] == mixture.source_hashes(), 'Fixed-prior manifest/source mismatch')
    plan = json.loads(analytic.CASE_PATH.read_text(encoding='utf-8'))
    cases = plan['cases']
    example = next(case for case in cases if case['case_id'] == 9)
    specs = [{'shape': 'beta_exact', 'path': BETA.relative_to(ROOT).as_posix(), 'sha256': analytic.digest(BETA)}]+manifest['records']
    analytic.require(len(specs) == 12 and len({item['shape'] for item in specs}) == 12, 'Expected beta plus 11 fixed classical priors')
    methods, example_rows = {}, []
    inputs = {path.relative_to(ROOT).as_posix(): analytic.digest(path)
              for path in (MANIFEST, analytic.CASE_PATH, Path(__file__).resolve(), Path(analytic.__file__).resolve())}
    started = time.perf_counter()
    for item in specs:
        path = ROOT/item['path']
        analytic.require(analytic.digest(path) == item['sha256'], 'Frozen boundary file changed')
        artifact = json.loads(path.read_text(encoding='utf-8'))
        if item['shape'] == 'beta_exact':
            analytic.exact.verify_artifact(artifact)
        else:
            mixture.verify_artifact(artifact)
            analytic.require(artifact['shape'] == item['shape'], 'Shape name mismatch')
        analytic.require(artifact['max_discordances'] == 8192, 'Wrong calibrated horizon')
        inputs[item['path']] = item['sha256']
        params = analytic.psr_parameters(example)
        q = params['conditional_plus_probability']
        distribution = analytic.first_crossing_distribution(artifact['boundaries'], q)
        row = analytic.analyze_case(artifact['boundaries'], example, 8192, distribution)
        row['method'] = item['shape']
        example_rows.append(row)
        methods[item['shape']] = {'artifact': artifact, 'cache': {q: distribution}}
    case9 = {'status': 'complete_fixed_case_analytic_float_comparison', 'case': example, 'pair_cap': 8192,
             'physical_shot_cap': 16384, 'results': example_rows, 'input_sha256': inputs,
             'new_observations': 0, 'scope': 'One fixed case; all-pair monitoring. Floating estimates require independent interval checks; no global superiority conclusion.'}
    analytic.write_new(CASE9_OUTPUT, case9)
    print(json.dumps({'case9': [{key: row[key] for key in ('method', 'expected_shots', 'correct_probability',
                                                        'error_probability', 'abstain_probability')} for row in example_rows]}), flush=True)
    rows, aggregates = [], []
    for name, item in methods.items():
        table, cache = item['artifact']['boundaries'], item['cache']
        for case in cases:
            params = analytic.psr_parameters(case)
            q = params['conditional_plus_probability']
            key = .5 if q is None else q
            if key not in cache:
                cache[key] = analytic.first_crossing_distribution(table, key)
            row = analytic.analyze_case(table, case, 8192, cache[key])
            row.update(method=name, kind=case['kind'])
            rows.append(row)
        print(json.dumps({'method_complete': name, 'analytic_case_count': len(cases)}), flush=True)
    base = {row['case_id']: row for row in rows if row['method'] == 'beta_exact'}
    for name in methods:
        main = [row for row in rows if row['method'] == name and row['kind'] == 'main']
        null = [row for row in rows if row['method'] == name and row['kind'] == 'null']
        aggregates.append({'method': name, 'main_cases': len(main), 'null_cases': len(null),
                           'equal_case_geometric_cost_ratio_to_beta': math.exp(math.fsum(math.log(row['expected_shots']/base[row['case_id']]['expected_shots']) for row in main)/len(main)),
                           'equal_case_mean_expected_shots': math.fsum(row['expected_shots'] for row in main)/len(main),
                           'equal_case_correct_probability': math.fsum(row['correct_probability'] for row in main)/len(main),
                           'minimum_case_correct_probability': min(row['correct_probability'] for row in main),
                           'maximum_main_error_probability': max(row['error_probability'] for row in main),
                           'maximum_null_false_stop_probability': max(row['error_probability'] for row in null)})
    for path, expected in inputs.items():
        analytic.require(analytic.digest(ROOT/path) == expected, 'Input changed during analytic comparison')
    result = {'status': 'complete_fixed_prior_float_analytic_comparison', 'pair_cap': 8192, 'physical_shot_cap': 16384,
              'method_count': len(methods), 'case_count': len(cases), 'analytic_cells': len(rows), 'rows': rows,
              'aggregates': aggregates, 'input_sha256': inputs, 'source_hashes': mixture.source_hashes(),
              'elapsed_seconds': time.perf_counter()-started, 'new_observations': 0,
              'claim_boundary': 'All 12 frozen sign-only methods receive identical every-physical-pair monitoring and cap. Priors were fixed without case observations. Float operating characteristics, not validated numerical intervals, hardware results, sparse-protocol results or field-wide SOTA claims.'}
    analytic.write_new(FULL_OUTPUT, result)
    print(json.dumps({'status': result['status'], 'elapsed_seconds': result['elapsed_seconds'], 'aggregates': aggregates}), flush=True)


if __name__ == '__main__':
    main()
