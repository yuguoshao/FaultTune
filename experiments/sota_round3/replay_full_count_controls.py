"""Replay four frozen full-count controls on every audited round3 pilot path.

This file never samples observations or fills in an unobserved continuation.
Each method stops at its own first screened and outward-certified crossing.
Safe endpoint predictions use the frozen fit of earlier paid rows only.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/structured_validation'))
import run_refined as old
import safe_grid as safe
import rb_grid as rb
import weighted_fisher as fisher

BASE = ROOT/'results/sota_round3/paired_benchmark'
AUDIT = ROOT/'results/sota_round3/paired_benchmark_audit/audit.json'
OUT = ROOT/'results/sota_round3/full_count_replay'
METHODS = ('full_safe_fine', 'full_safe_coarse', 'rb_fine', 'rb_coarse')
ARM = {'cos_numerator': 0, 'sin_numerator': 2, 'denominator': 2}
CAP, DELTA = 4096, .01


def require(value, message):
    if not value:
        raise AssertionError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def object_digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def source_hashes():
    result = old.source_hashes()
    result[Path(__file__).resolve().relative_to(ROOT).as_posix()] = digest(__file__)
    return result


def configs():
    base = fisher.protocol_config()
    safe_config, rb_config = safe.protocol_config(base), rb.protocol_config(base)
    def normalized(config):
        return [(Fraction(entry['numerator'], entry['denominator']), Fraction(*entry['weight']))
                for entry in config['experts']]
    require(normalized(safe_config) == normalized(rb_config) == normalized(base), 'Expert lists differ')
    require(len(base['experts']) == 71, 'Not the frozen 71-expert protocol')
    require(sum((Fraction(*entry['weight']) for entry in base['experts']), Fraction(0)) == 1,
            'Expert weights must sum exactly to one')
    require(safe_config['endpoint_denominator'] == 2**40, 'Safe endpoint precision changed')
    return safe_config, rb_config


def schedules():
    coarse = old.schedule(CAP)
    fine = sorted(set(range(2, 33, 2)) | {n for n in coarse if n > 32})
    return {'fine': fine, 'coarse': coarse}


def prediction_from_previous_paid_rows(rows, config):
    """The predictor's sole data argument is past paid rows, never a case/block."""
    fitted = old.fit_rows(rows)
    return {'paid_rows_before': deepcopy(rows),
            'paid_rows_before_sha256': object_digest(rows),
            'previous_paid_shots': int(fitted['shots']),
            'theta_before': [float(x) for x in fitted['theta']],
            'prediction': safe.predict_endpoints(float(fitted['theta'][0]), config)}


def schedule_blocks(raw_history, schedule):
    """Aggregate complete eligible blocks, retaining a partial initial32 tail."""
    output, counts, previous, raw_start = [], [0]*4, 0, 0
    for j, raw in enumerate(raw_history):
        require(raw['arm'] == ARM, 'Only the fixed PSR arm is authorized')
        require(raw['budget']-previous == 2*raw['m'], 'Raw physical shot accounting failed')
        require(sum(raw['pair_counts']) == raw['m'], 'Raw pair total failed')
        require(raw['plus_successes'] == raw['pair_counts'][1]+raw['pair_counts'][3] and
                raw['minus_successes'] == raw['pair_counts'][2]+raw['pair_counts'][3],
                'Raw margins failed')
        require(all(isinstance(x, int) and not isinstance(x, bool) and x >= 0
                    for x in raw['pair_counts']), 'Invalid raw counts')
        previous = raw['budget']
        counts = [a+b for a, b in zip(counts, raw['pair_counts'])]
        if raw['budget'] in schedule:
            output.append({'budget': raw['budget'], 'arm': dict(ARM), 'm': sum(counts),
                           'pair_counts': counts, 'minus_successes': counts[2]+counts[3],
                           'plus_successes': counts[1]+counts[3],
                           'source_raw_start_index': raw_start,
                           'source_raw_end_index_exclusive': j+1})
            raw_start, counts = j+1, [0]*4
    require([block['budget'] for block in output] == [n for n in schedule if n <= previous],
            'A scheduled checkpoint is missing from the observed path')
    tail = {'source_raw_start_index': raw_start,
            'source_raw_end_index_exclusive': len(raw_history), 'pair_counts': counts,
            'shots': 2*sum(counts)}
    return output, tail


def replay_method(blocks, method, last_observed_shots, schedule, safe_config, rb_config):
    """No case metadata/truth enters this method or its prediction function."""
    engine = safe if method.startswith('full_safe_') else rb
    config = safe_config if engine is safe else rb_config
    state, rows, counts, history, checkpoints = engine.empty_state(config), [], [0]*4, [], []
    stopped = None
    for block in blocks:
        # Stage the prediction before incorporating any outcome of this block.
        predictable = prediction_from_previous_paid_rows(rows, config) if engine is safe else None
        if predictable is not None:
            require(predictable['previous_paid_shots'] == state['shots'], 'Predictor prefix mismatch')
        current = deepcopy(block)
        m, minus, plus = current['m'], current['minus_successes'], current['plus_successes']
        if engine is safe:
            current['predictable'] = predictable
            safe.update_block(state, m, minus, plus, predictable['prediction'])
        else:
            rb.update_block(state, m, minus, plus)
        # Paid rows become available only after the current prediction/update.
        old.merge_row(rows, ARM, m, plus)
        old.merge_row(rows, dict(ARM, sin_numerator=-2), m, minus)
        counts = [a+b for a, b in zip(counts, current['pair_counts'])]
        history.append(current)
        require(state['shots'] == current['budget'] == 2*sum(counts), 'Method budget mismatch')
        screened = engine.screen(state, delta=DELTA)
        checkpoint = {'shots': current['budget'], 'counts': list(counts),
                      'history_prefix_length': len(history), 'screen': screened,
                      'certificate': None}
        # This is exactly run_refined.py's conservative floating screen gate.
        if screened['log_evidence'] >= math.log(1/DELTA)-1e-6 or current['budget'] == CAP:
            cert = engine.certificate_from_blocks(state['blocks'], delta=DELTA, config=config)
            checkpoint['certificate'] = cert
            require(cert['shots'] == state['shots'] and cert['score'] == counts[1]-counts[2],
                    'Certificate prefix mismatch')
            if cert['decision'] != 'abstain' or current['budget'] == CAP:
                stopped = {'status': 'observed_cap_abstention' if cert['decision'] == 'abstain'
                           else 'observed_stop', 'decision': cert['decision'],
                           'shots_lower': current['budget'], 'shots_upper': current['budget'],
                           'certificate': cert}
        checkpoints.append(checkpoint)
        if stopped is not None:
            break
    if stopped is None:
        require(last_observed_shots < CAP, 'A cap-reaching method must have an observed terminal')
        stopped = {'status': 'right_censored_at_observed_path_end', 'decision': None,
                   'shots_lower': next(n for n in schedule if n > last_observed_shots),
                   'shots_upper': CAP, 'certificate': None}
    stopped.update({'method': method, 'counts_at_last_eligible_prefix': list(counts),
                    'history_prefix_length': len(history),
                    'source_raw_history_prefix_length': history[-1]['source_raw_end_index_exclusive'] if history else 0,
                    'last_eligible_shots': state['shots'], 'last_observed_shots': last_observed_shots,
                    'history': history, 'checkpoint_history': checkpoints})
    return stopped


def replay_history(raw_history, last_observed_shots):
    """Accept only recorded observations, with no truth or random generator."""
    require(raw_history and raw_history[-1]['budget'] == last_observed_shots, 'Bad path end')
    safe_config, rb_config = configs()
    grids = schedules()
    streams = {name: schedule_blocks(raw_history, grid) for name, grid in grids.items()}
    results = []
    for method in METHODS:
        name = method.rsplit('_', 1)[1]
        result = replay_method(streams[name][0], method, last_observed_shots,
                               grids[name], safe_config, rb_config)
        result['unconsumed_partial_schedule_tail'] = streams[name][1]
        results.append(result)
    return results


def toy_history(allocation, fine_budgets):
    history, last = [], 0
    for j, budget in enumerate(fine_budgets):
        m = (budget-last)//2
        counts = allocation(j, m)
        history.append({'budget': budget, 'arm': dict(ARM), 'm': m, 'pair_counts': counts,
                        'plus_successes': counts[1]+counts[3], 'minus_successes': counts[2]+counts[3]})
        last = budget
    return history


def interface_checks():
    safe_config, rb_config = configs()
    grid = schedules()
    first32 = toy_history(lambda j, m: [0, m, 0, 0], list(range(2, 33, 2)))
    short = replay_history(first32[:3], 6)
    require(all(row['decision'] is None for row in short), 'Six shots should not cross')
    require([row['shots_lower'] for row in short] == [8, 32, 8, 32], 'Censoring must use next eligible checkpoint')
    require(all(row['shots_upper'] == CAP for row in short), 'Censor upper bound changed')
    require(short[1]['history'] == [] and short[1]['unconsumed_partial_schedule_tail']['shots'] == 6,
            'Partial coarse block was consumed')
    fine_blocks, _ = schedule_blocks(first32, grid['fine'])
    coarse_blocks, _ = schedule_blocks(first32, grid['coarse'])
    require(len(fine_blocks) == 16 and len(coarse_blocks) == 1 and coarse_blocks[0]['m'] == 16,
            'Initial32 aggregation failed')
    a = replay_method(fine_blocks[:3], 'full_safe_fine', 6, grid['fine'], safe_config, rb_config)
    altered = deepcopy(fine_blocks[:3])
    altered[2].update(pair_counts=[0, 0, 1, 0], minus_successes=1, plus_successes=0)
    b = replay_method(altered, 'full_safe_fine', 6, grid['fine'], safe_config, rb_config)
    require([x['predictable'] for x in a['history']] == [x['predictable'] for x in b['history']],
            'Current outcome leaked into its prediction')
    require([x['predictable']['previous_paid_shots'] for x in a['history']] == [0, 2, 4],
            'Safe predictor did not use exactly previous paid rows')
    coarse = replay_method(coarse_blocks, 'full_safe_coarse', 32, grid['coarse'], safe_config, rb_config)
    require(coarse['history'][0]['predictable']['previous_paid_shots'] == 0 and
            coarse['history'][0]['predictable']['prediction']['center_requested'] == .5,
            'Coarse initial predictor used paid outcomes from within its block')
    full = toy_history(lambda j, m: [m, 0, 0, 0], grid['fine'])
    terminal = replay_history(full, CAP)
    require(all(row['status'] == 'observed_cap_abstention' and row['decision'] == 'abstain'
                and row['shots_lower'] == row['shots_upper'] == CAP for row in terminal),
            'Observed cap abstention must remain distinct from right censoring')
    strong = toy_history(lambda j, m: [0, m, 0, 0], grid['fine'][:20])
    positive = replay_history(strong, strong[-1]['budget'])
    mirrored = deepcopy(strong)
    for block in mirrored:
        block['pair_counts'][1], block['pair_counts'][2] = block['pair_counts'][2], block['pair_counts'][1]
        block['minus_successes'], block['plus_successes'] = block['plus_successes'], block['minus_successes']
    negative = replay_history(mirrored, mirrored[-1]['budget'])
    for pos, neg in zip(positive, negative):
        require(pos['status'] == neg['status'] == 'observed_stop' and
                pos['shots_lower'] == neg['shots_lower'] and pos['decision'] == 'reject' and
                neg['decision'] == 'accept', 'Mirrored sign/first stop check failed')
        require(all(x['certificate'] is None or x['certificate']['decision'] == 'abstain'
                    for x in pos['checkpoint_history'][:-1]), 'First certified stop was skipped')
    return {'status': 'passed', 'sources': source_hashes(), 'new_simulation_observations': 0,
            'checks': ['fixed71 exact rational weights and endpoint precision',
                       'fine versus merged initial32 physical blocks', 'predictable paid-row prefix accounting',
                       'current outcome mutation leaves its prediction unchanged',
                       'initial coarse prediction uses zero paid rows',
                       'right censoring versus observed cap abstention', 'mirror sign and first certified crossing'],
            'strong_fixed_history_stops': {r['method']: r['shots_lower'] for r in positive}}


def summarize(records, cases, references):
    output = []
    for method in METHODS:
        for kind in ('main', 'null'):
            selected = [r for r in records if r['method'] == method and r['kind'] == kind]
            groups, counts = defaultdict(list), Counter()
            for row in selected:
                groups[row['case_id']].append(row)
                truth = 'reject' if cases[row['case_id']]['C'] > 0 else 'accept'
                category = ('right_censored' if row['decision'] is None else
                            'observed_cap_abstentions' if row['decision'] == 'abstain' else
                            'observed_correct_stops' if kind == 'main' and row['decision'] == truth else
                            'observed_wrong_stops')
                counts[category] += 1
            item = {'method': method, 'kind': kind, 'trajectories': len(selected),
                    'case_count': len(groups), **{key: counts[key] for key in
                    ('right_censored', 'observed_cap_abstentions', 'observed_correct_stops', 'observed_wrong_stops')},
                    'mean_shots_lower': sum(r['shots_lower'] for r in selected)/len(selected),
                    'mean_shots_upper': sum(r['shots_upper'] for r in selected)/len(selected),
                    'per_case': [], 'comparisons': []}
            for case_id, group in sorted(groups.items()):
                item['per_case'].append({'case_id': case_id, 'trajectories': len(group),
                                         'statuses': dict(Counter(r['status'] for r in group)),
                                         'mean_shots_lower': sum(r['shots_lower'] for r in group)/len(group),
                                         'mean_shots_upper': sum(r['shots_upper'] for r in group)/len(group)})
            for ref in references:
                comparison = {'reference_method': ref}
                for endpoint in ('lower', 'upper'):
                    ratio = math.exp(sum(math.log(sum(r['shots_'+endpoint] for r in group)/
                                             sum(r['references'][ref]['shots'] for r in group))
                                         for group in groups.values())/len(groups))
                    comparison['geometric_cost_ratio_'+endpoint] = ratio
                item['comparisons'].append(comparison)
            output.append(item)
    return output


def run():
    started = time.perf_counter()
    frozen = source_hashes()
    checks = read(OUT/'interface_checks.json')
    require(checks['status'] == 'passed' and checks['sources'] == frozen, 'Run checks on this exact source first')
    audit, plan = read(AUDIT), read(BASE/'preliminary_plan.json')
    require(audit['status'] == 'passed', 'Pilot audit has not passed')
    inputs = {str((BASE/name).relative_to(ROOT).as_posix()): digest(BASE/name) for name in audit['inputs']}
    for name, expected in audit['inputs'].items():
        require(digest(BASE/name) == expected, 'Audited pilot input changed: '+name)
    for name, expected in audit['sources'].items():
        require(digest(ROOT/name) == expected, 'Audited source changed: '+name)
    inputs[AUDIT.relative_to(ROOT).as_posix()] = digest(AUDIT)
    require(plan['repetitions'] == 32 and len(plan['cases']) == 18 and plan['cap'] == CAP and plan['delta'] == DELTA,
            'Expected fixed 576-path pilot')
    grids = schedules()
    require(plan['fine_checkpoints'] == grids['fine'] and plan['coarse_checkpoints'] == grids['coarse'],
            'Schedules differ from audited pilot')
    spec = {'version': 'full_count_controls_readonly_replay_v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'source_sha256': frozen, 'input_sha256': inputs, 'methods': list(METHODS),
            'expected_paths': 576, 'expected_method_records': 2304, 'cap': CAP, 'delta': DELTA,
            'fine_checkpoints': grids['fine'], 'coarse_checkpoints': grids['coarse'],
            'safe_config': configs()[0], 'rb_config': configs()[1],
            'prediction_rule': 'old.fit_rows(previous_paid_rows)[theta][0] then safe.predict_endpoints; before current block',
            'method_stop_rule': 'first screen >= log(1/delta)-1e-6 with outward certified direction, else observed cap',
            'censoring': 'no missing continuation: next method checkpoint through4096; decision null',
            'new_simulation_observations': 0, 'scope': 'all576 fixed development paths; no favorable subset selection'}
    write_new(OUT/'spec.json', spec)
    rows, keys, physical_shots = [], set(), 0
    cases = {case['case_id']: case for case in plan['cases']}
    reference_methods = list(plan['methods'])
    with (OUT/'records.jsonl').open('x', encoding='utf-8') as record_stream:
        for i, line in enumerate((BASE/'paths.jsonl').open(encoding='utf-8'), 1):
            path = json.loads(line)
            key = (path['case_id'], path['rep'])
            require(key not in keys and path['status'] == 'complete' and path['error'] is None, 'Bad/duplicate input path')
            keys.add(key)
            require(path['cap'] == CAP and path['delta'] == DELTA and path['truth_access'] == 'simulator_only', 'Path protocol differs')
            # Strip case metadata before it can enter replay/prediction methods.
            raw = [{name: block[name] for name in ('budget', 'arm', 'm', 'pair_counts', 'plus_successes', 'minus_successes')}
                   for block in path['history']]
            results = replay_history(raw, path['shots_observed'])
            for result in results:
                result.update({'case_id': path['case_id'], 'kind': path['kind'], 'rep': path['rep'],
                               'seed': path['seed'], 'cap': CAP, 'delta': DELTA,
                               'source_path_line': i, 'source_path_sha256': object_digest(path),
                               'source_observed_history_sha256': object_digest(raw),
                               'references': {name: {'shots': path['outcomes'][name]['shots'],
                                                     'decision': path['outcomes'][name]['decision']}
                                              for name in reference_methods}})
                record_stream.write(canonical(result)+'\n')
                # Keep compact terminal rows in memory; complete traces remain on disk.
                rows.append({name: value for name, value in result.items() if name not in ('history', 'checkpoint_history')})
            physical_shots += path['shots_observed']
            if i % 16 == 0:
                record_stream.flush()
                elapsed = time.perf_counter()-started
                print(canonical({'paths_completed': i, 'method_records': len(rows), 'elapsed_seconds': elapsed,
                                 'projected_total_seconds': elapsed*576/i}), flush=True)
    require(keys == {(case_id, rep) for case_id in cases for rep in range(32)} and len(rows) == 2304,
            'Complete fixed matrix not replayed')
    require(source_hashes() == frozen, 'Replay sources changed during execution')
    for name, expected in inputs.items():
        require(digest(ROOT/name) == expected, 'Input changed during execution: '+name)
    summary = {'status': 'complete', 'version': spec['version'], 'source_sha256': frozen, 'input_sha256': inputs,
               'spec_sha256': digest(OUT/'spec.json'), 'records_sha256': digest(OUT/'records.jsonl'),
               'path_records': len(keys), 'method_records': len(rows), 'new_simulation_observations': 0,
               'previously_generated_physical_shots': physical_shots, 'elapsed_seconds': time.perf_counter()-started,
               'outcomes': summarize(rows, cases, reference_methods),
               'limits': ['Development data reused for four fixed control descriptions; this is not a new holdout.',
                          'Right-censored costs are logical intervals, not cap abstentions or confidence intervals.',
                          'Comparisons are equal-case geometric ratios of within-case mean physical-shot costs.',
                          'Stopping uses frozen run_refined.py float screens followed by outward Decimal certificates.',
                          'A performance result here does not itself establish SOTA superiority or empirical1% risk.']}
    write_new(OUT/'summary.json', summary)
    print(canonical({'status': 'complete', 'path_records': len(keys), 'method_records': len(rows),
                     'elapsed_seconds': summary['elapsed_seconds'], 'summary_sha256': digest(OUT/'summary.json')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('checks', 'run'))
    args = parser.parse_args()
    if args.command == 'checks':
        result = interface_checks()
        write_new(OUT/'interface_checks.json', result)
        print(canonical(result), flush=True)
    else:
        run()
