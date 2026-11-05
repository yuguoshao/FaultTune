"""Independent adapter audit of frozen safe/RB full-count control replays.

Rebuilds every prefix from the audited raw data, rather than invoking replay
functions. Frozen numerical kernels are shared and their hashes are checked.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
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
DIR = ROOT/'results/sota_round3/full_count_replay'
OUT = ROOT/'results/sota_round3/full_count_replay_audit/audit.json'
METHODS = ('full_safe_fine', 'full_safe_coarse', 'rb_fine', 'rb_coarse')
ARM = {'cos_numerator': 0, 'sin_numerator': 2, 'denominator': 2}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def objsha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def build_blocks(raw, fine):
    selected = []
    total, start = [0]*4, 0
    for j, block in enumerate(raw):
        total = [a+b for a, b in zip(total, block['pair_counts'])]
        if fine or block['budget'] >= 32:
            selected.append({'arm': dict(ARM), 'budget': block['budget'], 'm': sum(total),
                             'pair_counts': total, 'plus_successes': total[1]+total[3],
                             'minus_successes': total[2]+total[3],
                             'source_raw_start_index': start, 'source_raw_end_index_exclusive': j+1})
            total, start = [0]*4, j+1
    tail = {'pair_counts': total, 'shots': 2*sum(total), 'source_raw_start_index': start,
            'source_raw_end_index_exclusive': len(raw)}
    return selected, tail


def verify_record(path, record, line_number, counters):
    assert record['method'] in METHODS
    assert all(record[key] == path[key] for key in ('case_id', 'rep', 'seed', 'kind', 'cap', 'delta'))
    assert record['source_path_line'] == line_number and record['source_path_sha256'] == objsha(path)
    raw = [{key: block[key] for key in ('budget', 'arm', 'm', 'pair_counts', 'plus_successes', 'minus_successes')}
           for block in path['history']]
    assert record['source_observed_history_sha256'] == objsha(raw)
    fine = record['method'].endswith('_fine')
    source_blocks, tail = build_blocks(raw, fine)
    assert record['unconsumed_partial_schedule_tail'] == tail
    engine = safe if record['method'].startswith('full_safe') else rb
    config = engine.protocol_config(fisher.protocol_config())
    state, totals, consumed, terminal = engine.empty_state(config), [0]*4, [], None
    assert len(record['history']) == len(record['checkpoint_history']) == record['history_prefix_length']
    assert 0 <= len(record['history']) <= len(source_blocks)
    for index, history in enumerate(record['history']):
        expected = source_blocks[index]
        assert {key: value for key, value in history.items() if key != 'predictable'} == expected
        assert terminal is None
        if engine is safe:
            # Independently reconstruct the two accumulated arm rows.
            n = sum(totals)
            rows = [] if n == 0 else [dict(ARM, trials=n, successes=totals[1]+totals[3]),
                                      dict(ARM, sin_numerator=-2, trials=n, successes=totals[2]+totals[3])]
            fit = old.fit_rows(rows)
            predictable = history['predictable']
            assert predictable['paid_rows_before'] == rows and predictable['paid_rows_before_sha256'] == objsha(rows)
            assert predictable['previous_paid_shots'] == int(fit['shots']) == 2*n
            assert predictable['theta_before'] == [float(x) for x in fit['theta']]
            assert abs(float(fit['theta'][0])-min(.9999, max(.0001, (.5+totals[1]+totals[2]+2*totals[3])/(1+2*n)))) < 1e-15
            prediction = safe.predict_endpoints(float(fit['theta'][0]), config)
            assert predictable['prediction'] == prediction
            block = dict(m=expected['m'], minus_successes=expected['minus_successes'],
                         plus_successes=expected['plus_successes'], prediction=prediction)
            safe.update_block(state, block['m'], block['minus_successes'], block['plus_successes'], prediction)
            counters['past_only_predictions_reconstructed'] += 1
        else:
            block = {key: expected[key] for key in ('m', 'minus_successes', 'plus_successes')}
            rb.update_block(state, block['m'], block['minus_successes'], block['plus_successes'])
        consumed.append(block)
        totals = [a+b for a, b in zip(totals, expected['pair_counts'])]
        checkpoint = record['checkpoint_history'][index]
        assert checkpoint['counts'] == totals and checkpoint['shots'] == expected['budget'] == 2*sum(totals)
        assert checkpoint['history_prefix_length'] == index+1
        screen = engine.screen(state, .01)
        assert checkpoint['screen'] == screen
        gate = screen['log_evidence'] >= math.log(100)-1e-6 or expected['budget'] == 4096
        assert (checkpoint['certificate'] is not None) == gate
        if gate:
            cert = engine.certificate_from_blocks(consumed, delta=.01, config=config)
            assert checkpoint['certificate'] == cert
            counters['outward_certificates_recomputed'] += 1
            if cert['decision'] != 'abstain' or expected['budget'] == 4096:
                terminal = cert
        counters['prefix_screens_recomputed'] += 1
        counters['paid_method_blocks_checked'] += 1
    assert record['counts_at_last_eligible_prefix'] == totals
    assert record['last_eligible_shots'] == 2*sum(totals)
    assert record['source_raw_history_prefix_length'] == (source_blocks[len(consumed)-1]['source_raw_end_index_exclusive'] if consumed else 0)
    assert record['last_observed_shots'] == path['shots_observed']
    if terminal is not None:
        assert record['certificate'] == terminal and record['decision'] == terminal['decision']
        assert record['shots_lower'] == record['shots_upper'] == terminal['shots']
        assert record['status'] == ('observed_cap_abstention' if terminal['decision'] == 'abstain' else 'observed_stop')
    else:
        assert len(consumed) == len(source_blocks) and path['shots_observed'] < 4096
        assert record['decision'] is None and record['certificate'] is None
        assert record['status'] == 'right_censored_at_observed_path_end'
        schedule = sorted(set(range(2, 33, 2)) | {n for n in old.schedule(4096) if n > 32}) if fine else old.schedule(4096)
        assert record['shots_lower'] == next(n for n in schedule if n > path['shots_observed'])
        assert record['shots_upper'] == 4096
    assert record['references'] == {name: {'shots': outcome['shots'], 'decision': outcome['decision']}
                                      for name, outcome in path['outcomes'].items()}
    counters[record['status']] += 1
    counters['method_records_checked'] += 1


def check_summary(summary, records, cases):
    assert summary['status'] == 'complete' and summary['path_records'] == 576 and summary['method_records'] == 2304
    assert summary['new_simulation_observations'] == 0
    assert {(x['method'], x['kind']) for x in summary['outcomes']} == {(method, kind) for method in METHODS for kind in ('main', 'null')}
    for item in summary['outcomes']:
        local = [r for r in records if (r['method'], r['kind']) == (item['method'], item['kind'])]
        groups, counts = defaultdict(list), Counter()
        for row in local:
            groups[row['case_id']].append(row)
            truth = 'reject' if cases[row['case_id']]['C'] > 0 else 'accept'
            counts['right_censored' if row['decision'] is None else 'observed_cap_abstentions' if row['decision'] == 'abstain' else
                   'observed_correct_stops' if row['kind'] == 'main' and row['decision'] == truth else 'observed_wrong_stops'] += 1
        assert item['trajectories'] == len(local) and item['case_count'] == len(groups)
        for category in ('right_censored', 'observed_cap_abstentions', 'observed_correct_stops', 'observed_wrong_stops'):
            assert item[category] == counts[category]
        for endpoint in ('lower', 'upper'):
            assert item['mean_shots_'+endpoint] == sum(r['shots_'+endpoint] for r in local)/len(local)
        assert len(item['per_case']) == len(groups)
        for entry in item['per_case']:
            group = groups[entry['case_id']]
            assert entry['trajectories'] == len(group) and entry['statuses'] == dict(Counter(r['status'] for r in group))
            for endpoint in ('lower', 'upper'):
                assert entry['mean_shots_'+endpoint] == sum(r['shots_'+endpoint] for r in group)/len(group)
        assert {x['reference_method'] for x in item['comparisons']} == set(local[0]['references'])
        for comparison in item['comparisons']:
            reference = comparison['reference_method']
            for endpoint in ('lower', 'upper'):
                ratio = math.exp(sum(math.log(sum(r['shots_'+endpoint] for r in group)/
                                         sum(r['references'][reference]['shots'] for r in group))
                                     for group in groups.values())/len(groups))
                assert abs(comparison['geometric_cost_ratio_'+endpoint]-ratio) < 1e-14


def main():
    begin = time.perf_counter()
    spec, plan = read(DIR/'spec.json'), read(BASE/'preliminary_plan.json')
    # Allow the independent audit to follow the append-only replay concurrently.
    for name, expected in spec['source_sha256'].items():
        assert sha(ROOT/name) == expected
    for name, expected in spec['input_sha256'].items():
        assert sha(ROOT/name) == expected
    source_audit = read(ROOT/'results/sota_round3/paired_benchmark_audit/audit.json')
    assert source_audit['status'] == 'passed' and source_audit['inputs']['paths.jsonl'] == sha(BASE/'paths.jsonl')
    audit_sources = old.source_hashes()
    audit_sources[Path(__file__).resolve().relative_to(ROOT).as_posix()] = sha(__file__)
    counters, rows, keys = Counter(), [], set()
    with (DIR/'records.jsonl').open(encoding='utf-8') as records:
        for i, line in enumerate((BASE/'paths.jsonl').open(encoding='utf-8'), 1):
            path = json.loads(line)
            for method in METHODS:
                position = records.tell()
                while True:
                    record_line = records.readline()
                    if record_line.endswith('\n'):
                        break
                    records.seek(position)
                    if (DIR/'summary.json').exists():
                        raise AssertionError('Replay ended before completing the matrix')
                    if time.perf_counter()-begin > 3600:
                        raise TimeoutError('Replay output not complete within one-hour audit guard')
                    time.sleep(.2)
                record = json.loads(record_line)
                assert record['method'] == method
                key = (record['case_id'], record['rep'], method)
                assert key not in keys
                keys.add(key)
                verify_record(path, record, i, counters)
                rows.append({key: value for key, value in record.items() if key not in ('history', 'checkpoint_history')})
            counters['paths_checked'] += 1
            counters['previously_generated_physical_shots'] += path['shots_observed']
            if i % 32 == 0:
                print(json.dumps({'audit_paths_completed': i, 'elapsed_seconds': time.perf_counter()-begin}), flush=True)
        assert records.readline() == ''
    while not (DIR/'summary.json').exists():
        if time.perf_counter()-begin > 3600:
            raise TimeoutError('Missing final summary')
        time.sleep(.2)
    summary = read(DIR/'summary.json')
    assert keys == {(case['case_id'], rep, method) for case in plan['cases'] for rep in range(32) for method in METHODS}
    assert summary['source_sha256'] == spec['source_sha256'] and summary['input_sha256'] == spec['input_sha256']
    assert summary['spec_sha256'] == sha(DIR/'spec.json') and summary['records_sha256'] == sha(DIR/'records.jsonl')
    assert summary['previously_generated_physical_shots'] == counters['previously_generated_physical_shots']
    check_summary(summary, rows, {case['case_id']: case for case in plan['cases']})
    for mapping in (spec['source_sha256'], spec['input_sha256'], audit_sources):
        for name, expected in mapping.items():
            assert sha(ROOT/name) == expected
    result = {'status': 'passed', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'audit_sources': audit_sources, 'replay_sources': spec['source_sha256'],
              'inputs': {str((DIR/name).relative_to(ROOT).as_posix()): sha(DIR/name)
                         for name in ('spec.json', 'interface_checks.json', 'records.jsonl', 'summary.json')},
              'counts': dict(counters), 'new_simulation_observations': 0,
              'elapsed_seconds': time.perf_counter()-begin,
              'scope': 'All576 audited raw paths, all2304 controls, every past-only prediction, screen, outward certificate, first stopping prefix and censor cost interval; all summary arithmetic.',
              'independence': 'Independent replay adapter and aggregation; frozen safe/RB numerical kernels and predictor are shared and hash-checked. Underlying raw RNG audit is trusted through its passed bound audit.',
              'limits': ['Development replay only; no new holdout and no SOTA superiority conclusion.',
                         'Logical censoring cost intervals are not confidence intervals; unknown decisions remain unknown.']}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'status': 'passed', 'audit_sha256': sha(OUT), 'counts': dict(counters),
                      'elapsed_seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
