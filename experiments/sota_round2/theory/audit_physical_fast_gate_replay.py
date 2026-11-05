"""Reconstruct all gate replay prefixes and summaries without producer imports."""
from collections import defaultdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'results/sota_round3/paired_benchmark'
REPLAY = ROOT/'results/sota_round3/fast_gate_replay_v1'


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def certificate(counts, prefix, artifact):
    pairs, b, c = sum(counts), counts[1], counts[2]
    d, m = b+c, artifact['prefix_pairs']
    require(pairs <= artifact['max_physical_pairs'] and d <= artifact['max_discordances'], 'Gate horizon exceeded')
    cutoff = artifact['beta_boundaries'][d]
    ordinary = 'reject' if b >= cutoff else 'accept' if b <= d-cutoff else 'abstain'
    gate = 'reject' if prefix and prefix[1] == m else 'accept' if prefix and prefix[2] == m else 'abstain'
    decision = gate if gate != 'abstain' else ordinary
    return {'method': 'fixed_physical_prefix_plus_beta', 'decision': decision, 'shots': 2*pairs,
            'pairs': pairs, 'discordances': d, 'score': b-c, 'counts': counts.copy(),
            'prefix_pairs': m, 'first_m_counts': list(prefix) if prefix is not None else None,
            'fast_gate_decision': gate, 'beta_decision': ordinary, 'beta_integer_boundary': cutoff,
            'decision_source': 'first_m_physical_prefix' if gate != 'abstain' else 'beta_boundary' if ordinary != 'abstain' else 'none',
            'delta_exact': artifact['delta_exact'], 'boundary_sha256': artifact['boundary_sha256'],
            'boundary_artifact_sha256': artifact['artifact_sha256'], 'max_discordances': artifact['max_discordances'],
            'max_physical_pairs': artifact['max_physical_pairs'], 'finite_horizon_only': True,
            'scope': 'One fixed physical prefix and one jointly calibrated beta table; prefix provenance is the caller responsibility'}


def reconstruct(path, artifacts, plan):
    counts, prefixes, stopped, rows = [0]*4, {}, set(), []
    previous = 0
    for index, block in enumerate(path['history']):
        fresh = block['pair_counts']
        require(len(fresh) == 4 and all(type(v) is int and v >= 0 for v in fresh), 'Invalid paired categories')
        require(block['arm'] == plan['fixed_positive_arm'] and block['budget'] == plan['fine_checkpoints'][index],
                'Replay path action or monitoring differs from audited protocol')
        require(sum(fresh) == block['m'] == (block['budget']-previous)//2, 'Paid block count differs')
        counts = [a+b for a, b in zip(counts, fresh)]
        pairs = sum(counts)
        require(2*pairs == block['budget'] and block['budget'] > previous, 'Physical-prefix chronology failed')
        previous = block['budget']
        for name, artifact in artifacts.items():
            m = artifact['prefix_pairs']
            if pairs == m:
                prefixes[name] = tuple(counts)
            elif pairs > m:
                require(name in prefixes, 'Unobserved ordering required to split gate prefix')
            if name in stopped:
                continue
            cert = certificate(counts, prefixes.get(name), artifact)
            if cert['decision'] != 'abstain' or previous == path['cap']:
                rows.append({'method': name, 'status': 'observed_stop' if cert['decision'] != 'abstain' else 'observed_cap_abstention',
                             'decision': cert['decision'], 'shots_lower': cert['shots'], 'shots_upper': cert['shots'],
                             'certificate': cert, 'history_prefix_length': index+1, 'counts': counts.copy()})
                stopped.add(name)
    require(counts == path['counts'] and 2*sum(counts) == path['shots_observed'], 'Final audited counts differ')
    for name in artifacts:
        if name not in stopped:
            rows.append({'method': name, 'status': 'right_censored', 'decision': None,
                         'shots_lower': next(n for n in plan['fine_checkpoints'] if n > previous),
                         'shots_upper': path['cap'], 'last_observed_shots': previous})
    reference = path['outcomes']['beta_exact']
    for row in rows:
        row.update(case_id=path['case_id'], kind=path['kind'], rep=path['rep'], seed=path['seed'], path='shared_psr',
                   reference_shots=reference['shots'], reference_decision=reference['decision'])
    return rows


def relative_summary(rows, cases):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['path'], row['method'], row['kind']].append(row)
    summaries = []
    for (path, method, kind), local in sorted(grouped.items()):
        cell_groups = defaultdict(list)
        for row in local:
            cell_groups[row['case_id']].append(row)
        cells = []
        for case_id, group in sorted(cell_groups.items()):
            reference = sum(r['reference_shots'] for r in group)/len(group)
            cells.append({'case_id': case_id, 'reference_mean': reference,
                          'ratio_lower': sum(r['shots_lower'] for r in group)/len(group)/reference,
                          'ratio_upper': sum(r['shots_upper'] for r in group)/len(group)/reference})
        wrong = correct = censored = abstained = 0
        for row in local:
            if row['decision'] is None:
                censored += 1
            elif row['decision'] == 'abstain':
                abstained += 1
            elif kind == 'null':
                wrong += 1
            elif row['decision'] == ('reject' if cases[row['case_id']]['C'] > 0 else 'accept'):
                correct += 1
            else:
                wrong += 1
        summaries.append({'path': path, 'method': method, 'kind': kind, 'trajectories': len(local),
                          'observed_correct_stops': correct, 'observed_wrong_stops': wrong,
                          'observed_cap_abstentions': abstained, 'right_censored': censored,
                          'geometric_cost_ratio_lower': math.exp(sum(math.log(c['ratio_lower']) for c in cells)/len(cells)),
                          'geometric_cost_ratio_upper': math.exp(sum(math.log(c['ratio_upper']) for c in cells)/len(cells)),
                          'cells': cells})
    return summaries


def combined_summary(baseline, rows, plan, methods):
    joined = baseline+[dict(case_id=r['case_id'], rep=r['rep'], method=r['method'],
                            shots=r['shots_lower'], decision=r['decision']) for r in rows]
    keyed = {(r['case_id'], r['rep'], r['method']): r for r in joined}
    require(len(keyed) == len(joined) == 576*len(methods), 'Duplicate or missing combined outcome')
    cells, means = [], {}
    for case in plan['cases']:
        answer = 'reject' if case['C'] > 0 else 'accept' if case['C'] < 0 else 'abstain'
        for method in methods:
            local = [keyed[case['case_id'], rep, method] for rep in range(32)]
            mean = sum(r['shots'] for r in local)/32
            means[case['case_id'], method] = mean
            cells.append({'case_id': case['case_id'], 'kind': case['kind'], 'sign': (case['C'] > 0)-(case['C'] < 0),
                          'method': method, 'trajectories': 32, 'mean_paid_shots': mean,
                          'wrong': sum(r['decision'] not in ('abstain', answer) for r in local),
                          'correct_stops': sum(r['decision'] == answer and answer != 'abstain' for r in local),
                          'abstained': sum(r['decision'] == 'abstain' for r in local)})
    outcomes = []
    for group in ('main', 'positive', 'negative', 'null'):
        for method in methods:
            selected = [c for c in cells if c['method'] == method and
                        (c['kind'] == group or group == 'positive' and c['sign'] == 1 or group == 'negative' and c['sign'] == -1)]
            outcomes.append({'group': group, 'method': method, 'trajectories': sum(c['trajectories'] for c in selected),
                             'mean_paid_shots': sum(c['mean_paid_shots'] for c in selected)/len(selected),
                             **{field: sum(c[field] for c in selected) for field in ('wrong', 'correct_stops', 'abstained')}})
    main = [c['case_id'] for c in plan['cases'] if c['kind'] == 'main']
    ratios = {method: {reference: math.exp(sum(math.log(means[i, method]/means[i, reference]) for i in main)/len(main))
                       for reference in methods} for method in methods}
    return {'cells': cells, 'outcomes': outcomes, 'equal_case_geometric_cost_ratios': ratios,
            'method_records': len(joined)}


def main(output):
    summary = read(REPLAY/'summary.json')
    require(summary['status'] == 'complete_development_replay', 'Replay incomplete')
    inputs = dict(summary['input_sha256'])
    inputs.update({str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in
                   (REPLAY/'records.jsonl', REPLAY/'summary.json')})
    for name, expected in inputs.items():
        require(sha(ROOT/name) == expected, 'Replay input hash differs: '+name)
    for name, expected in summary['source_hashes'].items():
        require(sha(ROOT/name) == expected, 'Replay source changed: '+name)
    audited = read(ROOT/'results/sota_round3/paired_benchmark_audit/audit.json')
    require(audited['status'] == 'passed', 'Original data audit incomplete')
    for name in ('preliminary_plan.json', 'paths.jsonl', 'method_results.jsonl'):
        require(sha(BASE/name) == audited['inputs'][name], 'Input differs from audited data')
    plan = read(BASE/'preliminary_plan.json')
    artifacts = {'physical_gate_m'+str(m): read(ROOT/('results/sota_round3/fast_gate_boundaries/m'+str(m)+'.json')) for m in (5, 6, 7)}
    rows, paths = [], []
    for line in (BASE/'paths.jsonl').open(encoding='utf-8'):
        path = json.loads(line)
        require(path['status'] == 'complete' and path['cap'] == 4096, 'Invalid source path')
        paths.append((path['case_id'], path['rep']))
        rows.extend(reconstruct(path, artifacts, plan))
    require(paths == [(c['case_id'], rep) for c in plan['cases'] for rep in range(32)], 'Source path matrix differs')
    with (REPLAY/'records.jsonl').open(encoding='utf-8') as stream:
        for expected in rows:
            require(json.loads(next(stream)) == expected, 'Gate replay first-stop certificate or prefix differs')
        require(next(stream, None) is None, 'Extra replay record')
    censored = sum(r['decision'] is None for r in rows)
    require(len(paths) == summary['path_count'] == 576 and len(rows) == summary['method_records'] == 1728 and
            censored == summary['right_censored'], 'Replay totals differ')
    require(relative_summary(rows, {c['case_id']: c for c in plan['cases']}) == summary['outcomes_relative_to_beta_exact'],
            'Relative cost/error/censoring summary differs')
    combined = None
    if not censored:
        baseline = [json.loads(line) for line in (BASE/'method_results.jsonl').open(encoding='utf-8')]
        combined = combined_summary(baseline, rows, plan, plan['methods']+list(artifacts))
        for key, value in combined.items():
            require(summary['complete_combined_comparison'][key] == value, 'Combined summary differs: '+key)
    for name, expected in inputs.items():
        require(sha(ROOT/name) == expected, 'Input changed during independent replay audit')
    result = {'status': 'passed', 'created_utc': datetime.now(timezone.utc).isoformat(), 'inputs': inputs,
              'verifier_source_sha256': sha(__file__), 'paths': len(paths), 'method_records': len(rows),
              'right_censored': censored, 'all_prefixes_and_certificates_reconstructed_without_producer_import': True,
              'full_cost_error_abstention_summaries_recomputed': True, 'new_simulation_observations': 0,
              'main_comparisons': {method: {reference: combined['equal_case_geometric_cost_ratios'][method][reference]
                                           for reference in ('beta_exact', 'maxsprt', 'pair_grid', 'closed_hedge_fine_initial')}
                                   for method in artifacts} if combined else None,
              'limitations': ['All cases are development cases and the gate was explored after seeing the original pilot.',
                              'The three m values remain separate methods; no combined or hindsight-selected test is audited.',
                              'No held-out confidence interval, hardware conclusion or SOTA claim is established.']}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': 'passed', 'path': str(output), 'sha256': sha(output),
                      'right_censored': censored, 'comparisons': result['main_comparisons']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
