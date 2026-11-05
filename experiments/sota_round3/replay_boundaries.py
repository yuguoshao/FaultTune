"""Censoring-aware deterministic replay of the 80 audited paired paths.

No continuation is simulated. An unobserved future stop is a cost interval,
never an observed cap abstention. Every available original checkpoint is used.
"""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import math

import exact_sign_boundary as beta
import generic_sign_boundary as generic

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'results/sota_round2/benchmark'
OUT = ROOT/'results/sota_round3/replay_v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_methods():
    finite_path = ROOT/'results/sota_round3/beta_boundary_N2048.json'
    finite = json.loads(finite_path.read_text())
    standard = beta.certify_threshold(2048, threshold='100')
    methods = {'beta_exact': (beta.certificate_from_pairs, beta.verify_artifact(finite)),
               'beta_ville': (beta.certificate_from_pairs, beta.verify_artifact(standard))}
    inputs = {finite_path.relative_to(ROOT).as_posix(): digest(finite_path)}
    for path in sorted((ROOT/'results/sota_round3/boundaries').glob('*.json')):
        artifact = json.loads(path.read_text())
        methods[path.stem] = (generic.certificate_from_pairs, generic.verify_artifact(artifact))
        inputs[path.relative_to(ROOT).as_posix()] = digest(path)
    return methods, inputs


def replay_path(path, methods, schedule):
    if path['status'] != 'complete' or path['cap'] != 4096:
        raise ValueError('Unexpected input path')
    counts, rows, stopped = [0]*4, [], set()
    history = path['history']
    previous_budget = 0
    for index, block in enumerate(history):
        counts = [a+b for a,b in zip(counts, block['pair_counts'])]
        budget = block['budget']
        if index+1 < len(history) and history[index+1]['budget'] == budget:
            continue
        if 2*sum(counts) != budget or budget <= previous_budget:
            raise ValueError('Checkpoint accounting mismatch')
        previous_budget = budget
        for name, (certify, boundary) in methods.items():
            if name in stopped:
                continue
            cert = certify(counts, boundary)
            if cert['decision'] != 'abstain' or budget == path['cap']:
                rows.append({'method': name, 'status': 'observed_stop' if cert['decision'] != 'abstain' else 'observed_cap_abstention',
                             'shots_lower': budget, 'shots_upper': budget,
                             'decision': cert['decision'], 'certificate': cert,
                             'history_prefix_length': index+1})
                stopped.add(name)
    if counts != path['counts'] or previous_budget != path['shots_observed']:
        raise ValueError('Terminal ledger differs')
    reference = path['outcomes']['hedge']
    for name in methods:
        if name not in stopped:
            next_look = next(s for s in schedule if s > previous_budget)
            rows.append({'method': name, 'status': 'right_censored_at_reference_stop',
                         'shots_lower': next_look, 'shots_upper': path['cap'],
                         'decision': None, 'last_observed_shots': previous_budget})
    for row in rows:
        row.update(case_id=path['case_id'], kind=path['kind'], rep=path['rep'],
                   path=path['path'], reference_shots=reference['shots'],
                   reference_decision=reference['decision'])
    return rows


def summarize(rows, cases):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['path'], row['method'], row['kind']].append(row)
    result = []
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
        wrong, correct, censored, abstained = 0,0,0,0
        for row in local:
            decision = row['decision']
            if decision is None:
                censored += 1
            elif decision == 'abstain':
                abstained += 1
            elif kind == 'null':
                wrong += 1
            elif decision == ('reject' if cases[row['case_id']]['C'] > 0 else 'accept'):
                correct += 1
            else:
                wrong += 1
        result.append({'path': path, 'method': method, 'kind': kind, 'trajectories': len(local),
                       'observed_correct_stops': correct, 'observed_wrong_stops': wrong,
                       'observed_cap_abstentions': abstained, 'right_censored': censored,
                       'geometric_cost_ratio_lower': math.exp(sum(math.log(c['ratio_lower']) for c in cells)/len(cells)),
                       'geometric_cost_ratio_upper': math.exp(sum(math.log(c['ratio_upper']) for c in cells)/len(cells)),
                       'cells': cells})
    return result


def main():
    methods, inputs = load_methods()
    plan = json.loads((BASE/'preliminary_plan.json').read_text())
    paths = [json.loads(line) for line in (BASE/'paths.jsonl').open()]
    paired = [p for p in paths if p['path'].startswith('closed_hedge_')]
    if len(paired) != 80:
        raise ValueError('Expected all 80 paired development paths')
    rows = []
    for path in paired:
        key = 'hedge_original_checkpoints' if path['path'] == 'closed_hedge_original' else 'hedge_fine_checkpoints'
        rows.extend(replay_path(path, methods, plan[key]))
    inputs.update({p.relative_to(ROOT).as_posix(): digest(p) for p in
                   (BASE/'paths.jsonl', BASE/'preliminary_plan.json', Path(__file__).resolve())})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'records.jsonl').open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
    summary = {'status': 'complete_development_replay_with_explicit_censoring',
               'input_sha256': inputs, 'path_count': len(paired), 'record_count': len(rows),
               'new_simulation_observations': 0, 'inferential_confidence_intervals': False,
               'cost_intervals': 'Logical bounds for unobserved continuation, not statistical confidence intervals. Methods inspect the same physical checkpoints as each reference path.',
               'outcomes': summarize(rows, {c['case_id']:c for c in plan['cases']}),
               'claim_boundary': 'Development screening only. Censoring is retained; no held-out or SOTA superiority claim.'}
    with (OUT/'summary.json').open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write('\n')
    for row in summary['outcomes']:
        if row['kind'] == 'main':
            print(json.dumps({k:v for k,v in row.items() if k != 'cells'}))


if __name__ == '__main__':
    main()
