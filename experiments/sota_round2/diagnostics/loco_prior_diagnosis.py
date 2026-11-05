"""LOCO prior calibration on old development histories, equally offered to controls.

No observations are generated. Rep 0 of every old development case is replayed
through the longest existing closed-policy history. All calibrated crossings
are floating diagnostics and censored trajectories retain explicit cost bounds.
"""
from collections import defaultdict
from fractions import Fraction
from itertools import groupby
from pathlib import Path
import hashlib
import json
import math
import sys
import time

import numpy as np
from scipy.special import betaln, gammaln, logsumexp

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'experiments/structured_validation'))
import run_refined as runner
import weighted_fisher as fisher
import rb_grid as rb
import safe_grid as safe

BASE = ROOT/'results/structured_validation/refined_development_v1'
OUTPUT = ROOT/'results/sota_round2/diagnostics/loco_prior_diagnosis.json'
BETA_ALPHAS = (.125, .25, .5, 1., 2., 4., 8.)
LEARNED_FRACTION = .75


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def symmetric_layout(experts):
    groups = defaultdict(list)
    for index, expert in enumerate(experts):
        odds = Fraction(expert['numerator'], expert['denominator'])
        groups[max(odds, 1/odds)].append(index)
    rows, weights, poles = [], [], []
    for pole, indices in sorted(groups.items()):
        assert len(indices) == (1 if pole == 1 else 2)
        if len(indices) == 2:
            assert experts[indices[0]]['weight'] == experts[indices[1]]['weight']
        rows.append(indices)
        weights.append(sum(float(Fraction(*experts[i]['weight'])) for i in indices))
        poles.append([pole.numerator, pole.denominator])
    assert abs(sum(weights)-1) < 1e-14
    return rows, np.array(weights), poles


def symmetrize(logs, groups):
    return np.array([float(logsumexp(logs[indices])-math.log(len(indices))) for indices in groups])


def fit_prior(logs, original):
    """Same terminal-evidence objective, EM budget and regularization for each engine."""
    shifted = np.exp(logs-np.max(logs, axis=1)[:, None])
    weights = np.ones(logs.shape[1])/logs.shape[1]
    for iteration in range(400):
        responsibility = shifted*weights
        responsibility /= responsibility.sum(axis=1)[:, None]
        updated = np.maximum(responsibility.mean(axis=0), 1e-12)
        updated /= updated.sum()
        change = float(np.max(np.abs(updated-weights)))
        weights = updated
        if change < 1e-11:
            break
    mixed = LEARNED_FRACTION*weights+(1-LEARNED_FRACTION)*original
    mixed /= mixed.sum()
    return mixed, iteration+1


def log_capital(logs, weights):
    indices = weights > 0
    return float(logsumexp(logs[indices]+np.log(weights[indices])))


def summary_bounds(rows, method, reference='old_hedge'):
    lo, hi = [], []
    for row in rows:
        item = row['methods'][method]
        denominator = (row['old_hedge_shots'], row['old_hedge_shots']) if reference == 'old_hedge' else (
            row['methods'][reference]['cost_lower_bound'], row['methods'][reference]['cost_upper_bound'])
        lo.append(item['cost_lower_bound']/denominator[1])
        hi.append(item['cost_upper_bound']/denominator[0])
    return {'cost_ratio_geomean_lower_bound': math.exp(float(np.mean(np.log(lo)))),
            'cost_ratio_geomean_upper_bound': math.exp(float(np.mean(np.log(hi)))),
            'censored_cases': sum(row['methods'][method]['censored'] for row in rows),
            'point_cost_ratio_available': all(not row['methods'][method]['censored'] and
                (reference == 'old_hedge' or not row['methods'][reference]['censored']) for row in rows)}


def main():
    started = time.perf_counter()
    assert not OUTPUT.exists()
    spec = json.loads((BASE/'spec.json').read_text())
    audit = json.loads((BASE/'audit.json').read_text())
    assert audit['status'] == 'passed' and len(spec['cases']) == 24
    inputs = {name: digest(BASE/name) for name in ('spec.json', 'summary.json', 'trials.jsonl', 'audit.json')}
    assert {name: inputs[name] for name in audit['run_input_sha256']} == audit['run_input_sha256']
    sources = dict(spec['sources'])
    for name, expected in sources.items():
        assert digest(ROOT/name) == expected
    sources[Path(__file__).resolve().relative_to(ROOT).as_posix()] = digest(Path(__file__).resolve())
    records = defaultdict(dict)
    with (BASE/'trials.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['policy'] == 'closed' and row['rep'] == 0:
                assert row['engine'] not in records[row['case_id']]
                records[row['case_id']][row['engine']] = row
    assert set(records) == {case['case_id'] for case in spec['cases']}
    experts = spec['conditional_protocol']['experts']
    groups, original_grid, poles = symmetric_layout(experts)
    alpha_original = np.zeros(len(BETA_ALPHAS))
    alpha_original[BETA_ALPHAS.index(.5)] = 1.
    original = {name: original_grid.copy() for name in ('paired_grid', 'fisher', 'rb', 'safe')}
    original['beta_family'] = alpha_original
    original['hedge'] = np.r_[.5*original_grid, .5*alpha_original]
    frames, reference_frames, default_costs = {}, {}, {}
    threshold = math.log(100)
    for case_id, methods in sorted(records.items()):
        assert set(methods) == set(runner.ENGINES)
        longest = max(methods.values(), key=lambda row: row['shots'])
        for row in methods.values():
            assert row['history'] == [b for b in longest['history'] if b['budget'] <= row['shots']]
        fs, rs, ss = fisher.empty_state(experts), rb.empty_state(spec['conditional_protocol']), safe.empty_state(safe.protocol_config(spec['conditional_protocol']))
        counts = np.zeros(4, dtype=np.int64)
        local = []
        for look, blocks in groupby(longest['history'], key=lambda b: b['look']):
            for block in blocks:
                counts += block['pair_counts']
                args = block['m'], block['minus_successes'], block['plus_successes']
                fisher.update_block(fs, *args)
                rb.update_block(rs, *args)
                safe.update_block(ss, *args, block['safe_prediction'])
            b, c = map(int, counts[1:3])
            pair_logs = np.array([b*math.log(2*e['numerator']/(e['numerator']+e['denominator']))+
                c*math.log(2*e['denominator']/(e['numerator']+e['denominator'])) for e in experts])
            rb_logs = np.array([0. if sign == 0 else sign*.5*rs['score']*rs['_eta'][pole]+rs['log_h'][pole]
                                for pole, sign in rs['_layout']])
            beta_logs = np.array([betaln(b+a, c+a)-betaln(a, a)+(b+c)*math.log(2) for a in BETA_ALPHAS])
            values = {'paired_grid': symmetrize(pair_logs, groups),
                      'fisher': symmetrize(float(fs['score'])*fs['log_odds']-fs['cgf'], groups),
                      'rb': symmetrize(rb_logs, groups),
                      'safe': symmetrize(ss['log_capital'], groups), 'beta_family': beta_logs}
            values['hedge'] = np.r_[values['fisher'], beta_logs]
            local.append({'shots': fs['shots'], 'score': b-c, 'logs': values})
        frames[case_id] = local
        reference_frames[case_id] = next(frame for frame in local if frame['shots'] == methods['hedge']['shots'])
        default_costs[case_id] = {}
        for method, weights in original.items():
            first = next((frame['shots'] for frame in local if frame['score'] and
                          log_capital(frame['logs'][method], weights) >= threshold-1e-6), None)
            engine = {'paired_grid': 'pair_grid', 'beta_family': 'beta'}.get(method, method)
            assert first == methods[engine]['shots'], (case_id, method, first, methods[engine]['shots'])
            default_costs[case_id][method] = first
        print(json.dumps({'replayed_case': case_id, 'cases': len(frames)}), flush=True)
    results = []
    for held_out in sorted(frames):
        training_ids = [case_id for case_id in sorted(frames) if case_id != held_out]
        item = {'case_id': held_out, 'rep': 0, 'training_case_ids': training_ids,
                'old_hedge_shots': records[held_out]['hedge']['shots'],
                'available_history_shots': frames[held_out][-1]['shots'], 'methods': {}}
        for method, prior in original.items():
            logs = np.array([reference_frames[case_id]['logs'][method] for case_id in training_ids])
            learned, iterations = fit_prior(logs, prior)
            before = log_capital(reference_frames[held_out]['logs'][method], prior)
            after = log_capital(reference_frames[held_out]['logs'][method], learned)
            first = next((frame['shots'] for frame in frames[held_out] if frame['score'] and
                          log_capital(frame['logs'][method], learned) >= threshold-1e-6), None)
            last = frames[held_out][-1]['shots']
            censored = first is None and last < spec['cap']
            lower = first if first is not None else min(spec['cap'], last+2)
            upper = first if first is not None else spec['cap']
            item['methods'][method] = {'learned_weights': list(map(float, learned)), 'em_iterations': iterations,
                'held_out_log_capital_gain_at_old_hedge_stop': after-before,
                'old_default_engine_shots': default_costs[held_out][method],
                'floating_first_crossing_shots': first, 'censored': censored,
                'cost_lower_bound': lower, 'cost_upper_bound': upper}
        results.append(item)
    report = {'status': 'completed_floating_development_diagnostic',
        'selection': {'cases': 24, 'rep': 0, 'histories': 'Longest old closed-policy history in each case; all six default engine stops reproduced.',
                      'cross_validation': 'Leave one complete case out; all fit data come from the other 23 old development cases.'},
        'sources': sources,
        'input_sha256': {(BASE/name).relative_to(ROOT).as_posix(): value for name, value in inputs.items()},
        'calibration_rule': {'objective': 'Mean log capital at each training case original closed_hedge stopping checkpoint.',
            'common_optimizer': '400-step mixture EM; stop at max weight change <1e-11; floor fitted weights at1e-12.',
            'learned_fraction': LEARNED_FRACTION,
            'original_prior_fraction': 1-LEARNED_FRACTION,
            'all_engines_receive_same_optimizer_and_regularization': True,
            'sign_symmetry': 'Each nonzero odds component is the equal mixture of reciprocal oriented experts.',
            'odds_poles': poles, 'beta_family_alphas': BETA_ALPHAS,
            'hedge_components': 'All symmetric Fisher components plus all seven symmetric Beta-prior kernels; original half/half mixture is included.'},
        'method_summaries': {}, 'held_out_cases': results,
        'limits': ['No new observations, Monte Carlo trajectories, prospective files or changed frozen sources.',
            'LOCO is a development diagnostic on fixed cases, not confirmatory inference or population generalization.',
            'All new crossings use floating arithmetic only and require independent outward certification before a performance claim.',
            'Uncrossed histories retain lower/upper cost bounds; the last observed cost is never imputed as a successful stop.',
            'Grid weights are floating calibration outputs, not deployable rational priors; exact rationalization and safety checks remain.',
            'The same calibration opportunity is given to Fisher, RB, paired-grid, safe, Beta-family and hedge controls. Tuned RB must be compared with tuned strong controls.',
            'The common training objective concerns terminal log evidence, not directly mean stopping cost; optimization may not transfer.',
            'Any final prior or hyperparameter selection still needs new held-out experimental validation.'],
        'elapsed_seconds': time.perf_counter()-started}
    for method in original:
        gains = [row['methods'][method]['held_out_log_capital_gain_at_old_hedge_stop'] for row in results]
        report['method_summaries'][method] = {'log_capital_gain_mean': float(np.mean(gains)),
            'log_capital_gain_median': float(np.median(gains)), 'positive_gain_cases': sum(g > 0 for g in gains),
            'versus_original_hedge': summary_bounds(results, method),
            'versus_equally_calibrated_hedge': summary_bounds(results, method, 'hedge')}
    for name, expected in sources.items():
        assert digest(ROOT/name) == expected
    for name, expected in inputs.items():
        assert digest(BASE/name) == expected
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(OUTPUT), 'methods': report['method_summaries'], 'elapsed_seconds': report['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
