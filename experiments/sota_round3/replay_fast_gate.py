"""Fixed m=5/6/7 gate screening on the complete development pilot histories.

Each m is a separate method, never a union or a hindsight per-trial selector.
All first-m prefixes are extracted from actual single-pair initial blocks.
No unobserved continuation is filled in, and no new samples are generated.
"""
from pathlib import Path
import json

import fast_gate_boundary as gate
import paired_benchmark as pilot
import replay_boundaries as replay

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'results/sota_round3/paired_benchmark'
OUT = ROOT/'results/sota_round3/fast_gate_replay_v1'


def inspect_path(path, methods, checkpoints):
    counts, prefixes, stopped, rows = [0]*4, {}, set(), []
    previous = 0
    for index, block in enumerate(path['history']):
        counts = [a+b for a,b in zip(counts, block['pair_counts'])]
        pairs = sum(counts)
        if pairs*2 != block['budget'] or block['budget'] <= previous:
            raise ValueError('Paid chronological block mismatch')
        previous = block['budget']
        for name, boundary in methods.items():
            m = boundary.prefix_pairs
            if pairs == m:
                prefixes[name] = tuple(counts)
            elif pairs > m and name not in prefixes:
                raise ValueError('Prefix ends inside an aggregate block; no observed order available')
            if name in stopped:
                continue
            cert = gate.certificate_from_pairs(counts, boundary, prefixes.get(name))
            if cert['decision'] != 'abstain' or block['budget'] == path['cap']:
                rows.append({'method': name, 'status': 'observed_stop' if cert['decision'] != 'abstain' else 'observed_cap_abstention',
                             'decision': cert['decision'], 'shots_lower': cert['shots'], 'shots_upper': cert['shots'],
                             'certificate': cert, 'history_prefix_length': index+1, 'counts': counts.copy()})
                stopped.add(name)
    if counts != path['counts'] or 2*sum(counts) != path['shots_observed']:
        raise ValueError('Final cumulative counts differ')
    for name in methods:
        if name not in stopped:
            rows.append({'method': name, 'status': 'right_censored', 'decision': None,
                         'shots_lower': next(n for n in checkpoints if n > previous),
                         'shots_upper': path['cap'], 'last_observed_shots': previous})
    reference = path['outcomes']['beta_exact']
    for row in rows:
        row.update(case_id=path['case_id'], kind=path['kind'], rep=path['rep'], seed=path['seed'],
                   path='shared_psr', reference_shots=reference['shots'], reference_decision=reference['decision'])
    return rows


def main():
    paths_file, plan_file = BASE/'paths.jsonl', BASE/'preliminary_plan.json'
    audit_path = ROOT/'results/sota_round3/paired_benchmark_audit/audit.json'
    audit = pilot.read_json(audit_path)
    if audit['status'] != 'passed':
        raise ValueError('Original data audit must pass before gate screening')
    for name in ('preliminary_plan.json', 'paths.jsonl', 'method_results.jsonl'):
        if pilot.digest(BASE/name) != audit['inputs'][name]:
            raise ValueError('Input differs from independently audited data: '+name)
    for name, sha in audit['sources'].items():
        if pilot.digest(ROOT/name) != sha:
            raise ValueError('Audited source changed: '+name)
    plan = pilot.read_json(plan_file)
    methods, inputs = {}, {}
    for m in (5, 6, 7):
        path = ROOT/f'results/sota_round3/fast_gate_boundaries/m{m}.json'
        artifact = pilot.read_json(path)
        methods[f'physical_gate_m{m}'] = gate.verify_artifact(artifact)
        inputs[pilot.relative(path)] = pilot.digest(path)
    for path in (paths_file, plan_file, BASE/'method_results.jsonl', Path(__file__).resolve(), audit_path,
                 Path(pilot.__file__).resolve(), Path(replay.__file__).resolve(),
                 ROOT/'results/sota_round2/theory/physical_fast_gate_independent_audit.json'):
        inputs[pilot.relative(path)] = pilot.digest(path)
    rows, observed_keys = [], set()
    for line in paths_file.open(encoding='utf-8'):
        path = json.loads(line)
        if path['status'] != 'complete' or path['cap'] != 4096:
            raise ValueError('Replay requires a complete original path')
        key = (path['case_id'], path['rep'])
        if key in observed_keys:
            raise ValueError('Duplicate input path')
        observed_keys.add(key)
        rows.extend(inspect_path(path, methods, plan['fine_checkpoints']))
    expected = {(c['case_id'], r) for c in plan['cases'] for r in range(plan['repetitions'])}
    if observed_keys != expected or len(rows) != 1728:
        raise ValueError('Incomplete registered development matrix')
    censored = sum(row['decision'] is None for row in rows)
    result = {'status': 'complete_development_replay', 'new_simulation_observations': 0,
              'input_sha256': inputs, 'source_hashes': gate.source_hashes(), 'path_count': len(observed_keys),
              'method_records': len(rows), 'right_censored': censored,
              'outcomes_relative_to_beta_exact': replay.summarize(rows, {c['case_id']: c for c in plan['cases']}),
              'claim_boundary': 'Development screening of three fixed separately valid methods. Every original case and seed included. No per-trial best-method selector, held-out inference, or SOTA claim.'}
    if not censored:
        baseline = [json.loads(line) for line in (BASE/'method_results.jsonl').open(encoding='utf-8')]
        added = [dict(case_id=r['case_id'], kind=r['kind'], rep=r['rep'], seed=r['seed'], method=r['method'],
                      shots=r['shots_lower'], decision=r['decision'], counts=r['counts'],
                      certificate=r['certificate'], history_prefix_length=r['history_prefix_length']) for r in rows]
        combined_plan = dict(plan, methods=plan['methods']+list(methods))
        result['complete_combined_comparison'] = pilot.summarize(baseline+added, combined_plan)
    for name, expected_sha in inputs.items():
        if pilot.digest(ROOT/name) != expected_sha:
            raise ValueError('Input changed during replay')
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'records.jsonl').open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
    pilot.write_new(OUT/'summary.json', result)
    print(json.dumps({'paths': len(observed_keys), 'rows': len(rows), 'censored': censored}))
    for row in result['outcomes_relative_to_beta_exact']:
        if row['kind'] == 'main':
            print(json.dumps({k:v for k,v in row.items() if k != 'cells'}))


if __name__ == '__main__':
    main()
