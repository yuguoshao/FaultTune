"""Deterministic bookkeeping checks and replay of already observed old data.

This entrypoint does not invoke DKMSampler.choose or draw random observations.
It does not release or run the proposed development benchmark.
"""
from copy import deepcopy
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import benchmark_runner as runner


def synthetic_certificate(shots, decision, sign=1, evidence='10', threshold='5'):
    return {'shots': shots, 'decision': decision, 'sign': sign,
            'evidence_lower_decimal': evidence, 'threshold_upper_decimal': threshold,
            'scope': 'synthetic bookkeeping fixture, not statistical evidence'}


def cyclic_chooser(n, k):
    total = int(n.sum())
    return {'arm': total % len(n), 'batch_size': 1, 'observed_shots': total}


def controller_checks():
    arms = runner.dkm.default_arms()
    active_history = []
    calibrations = [SimpleNamespace(name=name) for name in ('early8', 'early10', 'capped')]

    def evaluator(arms_, n, k, active, delta, force, timings):
        total = int(n.sum())
        active_history.append([total, [c.name for c in active], force])
        runner.require(all(value >= 1 for value in n), 'Checked before all fixed arms initialized')
        results = {}
        for item in active:
            stop = 8 if item.name == 'early8' else 10 if item.name == 'early10' else 12
            if total == stop:
                results[item.name] = synthetic_certificate(total, 'abstain' if item.name == 'capped' else 'reject')
        return results, {'fixture': 'deterministic'}

    path = runner.counts_path(arms, cyclic_chooser, lambda arm, length: arm % 2,
                             calibrations, set(range(2, 13, 2)), 12, .01, lambda: None, evaluator)
    runner.require(path['status'] == 'complete' and path['shots_observed'] == len(path['history']) == 12,
                   'Shared path failed its complete physical cost')
    runner.require(active_history == [[6, ['early8', 'early10', 'capped'], False],
                                     [8, ['early8', 'early10', 'capped'], False],
                                     [10, ['early10', 'capped'], False], [12, ['capped'], True]],
                   'Initialization/checkpoint/active-calibration sequence differs')
    for name, expected in (('early8', 8), ('early10', 10), ('capped', 12)):
        result = path['outcomes'][name]
        runner.require(result['shots'] == result['history_prefix_length'] == sum(result['trials']) == expected,
                       'Early stopping prefix includes later shared observations')
        observed = path['history'][:result['history_prefix_length']]
        rebuilt_n, rebuilt_k = [0]*6, [0]*6
        for row in observed:
            rebuilt_n[row['arm']] += row['trials']
            rebuilt_k[row['arm']] += row['successes']
        runner.require(rebuilt_n == result['trials'] and rebuilt_k == result['successes'], 'Prefix counts differ')
    path.update(case_id=7, kind='main', rep=0, seed=0, path='dkm_one_shot')
    rows = runner.method_rows(path)
    runner.require([row['shots'] for row in rows] == [8, 10, 12] and
                   [row['history_prefix_length'] for row in rows] == [8, 10, 12], 'Method rows miscount shared prefixes')

    calls = 0
    def cutoff():
        nonlocal calls
        calls += 1
        if calls == 10:
            raise RuntimeError('synthetic deadline after nine paid observations')
    partial = runner.counts_path(arms, cyclic_chooser, lambda arm, length: 0,
                                calibrations, set(range(2, 13, 2)), 12, .01, cutoff, evaluator)
    runner.require(partial['status'] == 'incomplete' and partial['shots_observed'] == len(partial['history']) == 9 and
                   partial['outcomes']['early8']['shots'] == 8 and len(partial['outcomes']) == 1,
                   'Deadline discarded observed data or completed methods')
    failures = []
    invalid = [('batch_two', lambda n, k: {'arm': 0, 'batch_size': 2, 'observed_shots': 0}, lambda a, n: 0),
               ('fractional_arm', lambda n, k: {'arm': .9, 'batch_size': 1, 'observed_shots': 0}, lambda a, n: 0),
               ('fractional_batch', lambda n, k: {'arm': 0, 'batch_size': 1.1, 'observed_shots': 0}, lambda a, n: 0),
               ('wrong_prefix', lambda n, k: {'arm': 0, 'batch_size': 1, 'observed_shots': 1}, lambda a, n: 0),
               ('too_many_successes', cyclic_chooser, lambda a, n: 2),
               ('fractional_successes', cyclic_chooser, lambda a, n: .5),
               ('boolean_successes', cyclic_chooser, lambda a, n: True)]
    for name, chooser, observer in invalid:
        failed = runner.counts_path(arms, chooser, observer, calibrations, {6, 12}, 12, .01, lambda: None, evaluator)
        runner.require(failed['status'] == 'incomplete' and failed['shots_observed'] == 0 and not failed['history'],
                       'Invalid input accepted: '+name)
        failures.append({'fixture': name, 'rejection': failed['error']})
    return {'status': 'passed', 'first_check': 6, 'shared_physical_shots': 12,
            'stopped_prefix_shots': [8, 10, 12], 'deadline_retained_shots': 9,
            'invalid_input_rejections': failures}


def direction_checks():
    class FakeCalibration:
        def __init__(self, name, evidence):
            self.name, self.evidence = name, evidence
        def numerator(self, arms, n, k):
            return 20.
        def threshold(self, n, delta):
            return 5.
        def certify(self, arms, n, k, sign, delta, proposal):
            return synthetic_certificate(int(n.sum()), 'reject' if sign > 0 else 'accept',
                                         sign, self.evidence[sign])
    arms, n, k = runner.dkm.default_arms(), runner.np.ones(6, dtype=int), runner.np.zeros(6, dtype=int)
    calibrations = [FakeCalibration('tie', {-1: '10', 1: '10'}),
                    FakeCalibration('larger_negative', {-1: '10.0000000000000000001', 1: '10'})]
    with patch.object(runner.dkm.likelihood, 'null_fit', return_value={'loglik': 0., 'candidate': None}):
        results, checks = runner.evaluate_calibrations(arms, n, k, calibrations, .01, False,
                                                       runner.defaultdict(float))
    runner.require(results['tie']['sign'] == 1 and results['larger_negative']['sign'] == -1,
                   'Both-sign largest-margin rule or positive tie break changed')
    runner.require(all(c['eligible_signs'] == [-1, 1] for c in checks.values()), 'Both signs were not checked')
    return {'status': 'passed', 'positive_tie_break': True, 'decimal_margin_ordering': True}


def matrix_checks():
    plan = {'cases': [{'case_id': 7, 'kind': 'main', 'C': .1}, {'case_id': 100, 'kind': 'null', 'C': 0.}],
            'cap': 4096, 'calibrations': [{'name': name} for name in ('kk21', 'kt_safe', 'structured_grid_safe')],
            'claim_boundary': 'Synthetic bookkeeping fixture'}
    methods = [path+'__'+item['name'] for path in runner.PATH_NAMES[:2] for item in plan['calibrations']]+list(runner.PATH_NAMES[2:])
    rows = []
    for case in plan['cases']:
        for method in methods:
            decisions = ('reject', 'accept', 'abstain', 'reject') if case['kind'] == 'main' else ('abstain', 'reject', 'accept', 'abstain')
            costs = (8, 10, 4096, 14) if case['kind'] == 'main' else (4096, 10, 12, 4096)
            rows.extend({'case_id': case['case_id'], 'rep': rep, 'method': method,
                         'decision': decision, 'shots': shots} for rep, (decision, shots) in enumerate(zip(decisions, costs)))
    summary = runner.summarize(rows, plan)
    for cell in summary['cells']:
        if cell['kind'] == 'main':
            runner.require(cell['mean_shots'] == 1032 and cell['wrong'] == 1 and cell['correct_stops'] == 2 and cell['abstained'] == 1,
                           'Main wrong/cap costs were dropped')
        else:
            runner.require(cell['mean_shots'] == 2053.5 and cell['wrong'] == 2 and cell['correct_stops'] == 0 and cell['abstained'] == 2,
                           'Null false/cap costs were dropped')
    runner.require(all(value == 1 for ratios in summary['equal_case_geometric_cost_ratios'].values() for value in ratios.values()),
                   'Identical synthetic methods must have cost ratio one')
    missing, duplicated, invalid = rows[:-1], rows+[deepcopy(rows[0])], deepcopy(rows)
    invalid[0]['decision'] = 'invalid'
    failures = []
    for label, matrix in [('missing', missing), ('duplicate', duplicated), ('invalid_decision', invalid)]:
        try:
            runner.summarize(matrix, plan)
        except AssertionError as caught:
            failures.append({'fixture': label, 'rejection': str(caught)})
        else:
            raise AssertionError('Invalid matrix accepted: '+label)
    return {'status': 'passed', 'method_records': len(rows), 'main_mean_all_costs': 1032,
            'null_mean_all_costs': 2053.5, 'invalid_matrices_rejected': failures}


def old_hedge_replay_check():
    """Supply old paid counts; no RNG draw or new observation is performed."""
    path = runner.ROOT/'results/structured_validation/refined_development_v1/trials.jsonl'
    fixture = None
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['method'] == 'closed_hedge' and row['shots'] <= 4096 and row['decision'] != 'abstain':
                fixture = row
                break
    runner.require(fixture is not None, 'No already observed hedge fixture within benchmark cap')
    specification = runner.read_json(runner.ROOT/'results/structured_validation/refined_development_v1/spec.json')
    case = next(c for c in specification['cases'] if c['case_id'] == fixture['case_id'])
    calls = []
    for block in fixture['history']:
        number, minus, both, worse = block['m'], block['minus_successes'], block['pair_counts'][3], block['pair_counts'][1]
        arm = block['arm']
        co, si = arm['cos_numerator']/arm['denominator'], arm['sin_numerator']/arm['denominator']
        pm, pp = case['A']+case['B']*co-case['C']*si, case['A']+case['B']*co+case['C']*si
        calls.extend([(number, pm, minus), (minus, pp, both), (number-minus, pp, worse)])
    cursor = 0
    def observed_binomial(number, probability):
        nonlocal cursor
        expected_number, expected_probability, observed = calls[cursor]
        cursor += 1
        runner.require(number == expected_number and probability == expected_probability, 'Old hedge paid query differs')
        return observed
    with patch.object(runner.np.random, 'RandomState', side_effect=AssertionError('No RNG allowed in old-count replay')):
        replayed = runner.hedge_path(case, fixture['seed'], 4096, .01, runner.original.schedule(4096),
                                    lambda: None, binomial=observed_binomial)
    expected_history = [{key: value for key, value in block.items() if key != 'safe_prediction'} for block in fixture['history']]
    runner.require(replayed['status'] == 'complete' and cursor == len(calls) and replayed['history'] == expected_history,
                   'Copied hedge path differs from old observed protocol history: '+str(replayed['error']))
    result = replayed['outcomes']['hedge']
    for field in ('shots', 'decision', 'certificate', 'counts', 'exploration_pairs'):
        runner.require(result[field] == fixture[field], 'Copied hedge differs from frozen old '+field)
    return {'status': 'passed', 'fixture': {'path': runner.relative(path), 'sha256': runner.digest(path),
            'case_id': fixture['case_id'], 'rep': fixture['rep'], 'seed': fixture['seed'], 'shots': fixture['shots']},
            'exact_fields': ['paid_queries', 'history_except_unused_safe_prediction', 'shots', 'decision', 'certificate', 'counts', 'exploration_pairs'],
            'new_observations': 0, 'random_draws': 0}


def main(output):
    result = {'status': 'running', 'scope': 'Synthetic bookkeeping plus frozen old-count replay; no new Monte Carlo or release',
              'created_utc': runner.utc_now(), 'checks': {}}
    try:
        # The same guard covers all bookkeeping checks, including future edits.
        with patch.object(runner.dkm.DKMSampler, 'choose', side_effect=AssertionError('No DKM allocation allowed in bookkeeping checks')):
            with patch.object(runner.np.random, 'RandomState', side_effect=AssertionError('No RNG allowed in bookkeeping checks')):
                for name, check in [('controller', controller_checks), ('direction_selection', direction_checks),
                                    ('complete_matrix', matrix_checks), ('old_hedge_replay', old_hedge_replay_check)]:
                    result['checks'][name] = check()
        result['status'] = 'passed'
    except BaseException as caught:
        result.update(status='failed', error=type(caught).__name__+': '+str(caught))
        raise
    finally:
        result['sources'] = runner.original.source_hashes()
        for path in (Path(__file__).resolve(), Path(runner.__file__).resolve(),
                     runner.ROOT/'experiments/sota_round2/literature/dkm_baseline.py'):
            result['sources'][runner.relative(path)] = runner.digest(path)
        runner.write_new(output, result)
        print(json.dumps({'status': result['status'], 'output': str(output), 'sha256': runner.digest(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=runner.DIRECTORY/'synthetic_checks.json')
    main(parser.parse_args().output)
