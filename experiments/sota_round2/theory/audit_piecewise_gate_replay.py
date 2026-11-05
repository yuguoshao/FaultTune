"""Independent all-prefix audit of the fixed D512 piecewise gate replay."""
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path

from audit_physical_fast_gate_replay import relative_summary, combined_summary

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'results/sota_round3/paired_benchmark'
REPLAY = ROOT/'results/sota_round3/piecewise_gate_replay'
NAME = 'physical_gate_m6_piecewise_D512'


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def make_certificate(counts, first, artifact):
    b, c, pairs = counts[1], counts[2], sum(counts)
    d, cutoff = b+c, artifact['boundaries'][b+c]
    gate = 'reject' if first and first[1] == 6 else 'accept' if first and first[2] == 6 else 'abstain'
    beta = 'reject' if b >= cutoff else 'accept' if b <= d-cutoff else 'abstain'
    return {'method': 'piecewise_beta_with_fixed_first6_gate', 'decision': gate if gate != 'abstain' else beta,
            'shots': 2*pairs, 'pairs': pairs, 'discordances': d, 'counts': counts.copy(), 'score': b-c,
            'first_m_counts': first.copy() if first is not None else None, 'prefix_pairs': 6,
            'fast_gate_decision': gate, 'beta_decision': beta, 'beta_integer_boundary': cutoff,
            'continuation_stage': 'early' if d <= 512 else 'late', 'switch_discordances': 512,
            'boundary_sha256': artifact['boundary_sha256'], 'boundary_artifact_sha256': artifact['artifact_sha256'],
            'delta_exact': {'numerator': '1', 'denominator': '100'}, 'finite_horizon_only': True,
            'max_physical_pairs': 2048, 'max_discordances': 2048}


def reconstruct(source, plan, artifact):
    counts, first, history, stopped = [0]*4, None, [], False
    previous, final_index = 0, 0
    for index, block in enumerate(source['history']):
        fresh = block['pair_counts']
        require(len(fresh) == 4 and all(type(v) is int and v >= 0 for v in fresh), 'Noninteger physical outcome')
        require(block['arm'] == plan['fixed_positive_arm'] and block['budget'] == plan['fine_checkpoints'][index] and
                sum(fresh) == block['m'] == (block['budget']-previous)//2, 'Unexpected paired protocol block')
        old_pairs = sum(counts)
        counts = [a+b for a, b in zip(counts, fresh)]
        pairs = sum(counts)
        require(not old_pairs < 6 < pairs, 'Observed block straddles required first-six prefix')
        if pairs == 6:
            first = counts.copy()
        require(pairs <= 2048 and 2*pairs == block['budget'], 'Physical budget differs')
        previous, final_index = block['budget'], index+1
        cert = make_certificate(counts, first, artifact)
        history.append({'history_prefix_length': index+1, 'certificate': cert})
        if cert['decision'] != 'abstain':
            stopped = True
            break
    if stopped:
        status, decision, lower, upper = 'stopped_on_observed_checkpoint', cert['decision'], previous, previous
    elif previous == 4096:
        status, decision, lower, upper = 'cap_abstention', 'abstain', 4096, 4096
    else:
        status, decision = 'censored_existing_prefix', None
        lower, upper = next(n for n in plan['fine_checkpoints'] if n > source['shots_observed']), 4096
    reference = source['outcomes']['beta_exact']
    return dict(case_id=source['case_id'], rep=source['rep'], seed=source['seed'], kind=source['kind'], method=NAME,
                path='shared_psr', reference_shots=reference['shots'], reference_decision=reference['decision'],
                status=status, decision=decision, shots_lower=lower, shots_upper=upper,
                certificate=cert, history_prefix_length=final_index, checkpoint_history=history)


def main(output):
    result = read(REPLAY/'summary.json')
    require(result['status'] == 'complete_development_replay', 'Piecewise replay incomplete')
    inputs = dict(result['input_sha256'])
    for path in (REPLAY/'records.jsonl', REPLAY/'summary.json'):
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    for name, expected in {**inputs, **result['source_hashes']}.items():
        require(sha(ROOT/name) == expected, 'Piecewise replay binding changed: '+name)
    artifact = read(ROOT/'results/sota_round3/piecewise_gate_boundary.json')
    safety = read(ROOT/'results/sota_round2/theory/piecewise_physical_gate_independent_audit.json')
    require(safety['status'] == 'passed' and safety['artifact_file_sha256'] ==
            sha(ROOT/'results/sota_round3/piecewise_gate_boundary.json'), 'Piecewise safety audit does not match')
    plan = read(BASE/'preliminary_plan.json')
    cases = {c['case_id']: c for c in plan['cases']}
    rows, keys, regressions, decision_losses = [], [], [], []
    for line in (BASE/'paths.jsonl').open(encoding='utf-8'):
        source = json.loads(line)
        require(source['status'] == 'complete' and source['cap'] == 4096, 'Bad source path')
        keys.append((source['case_id'], source['rep']))
        row = reconstruct(source, plan, artifact)
        rows.append(row)
        if row['shots_lower'] > row['reference_shots']:
            reference = source['outcomes']['beta_exact']
            regressions.append({'case_id': row['case_id'], 'rep': row['rep'],
                                'reference_shots': row['reference_shots'], 'candidate_lower': row['shots_lower'],
                                'reference_discordances': reference['counts'][1]+reference['counts'][2],
                                'candidate_status': row['status']})
        truth = 'reject' if cases[row['case_id']]['C'] > 0 else 'accept' if cases[row['case_id']]['C'] < 0 else 'abstain'
        if truth != 'abstain' and row['reference_decision'] == truth and row['decision'] != truth:
            decision_losses.append({'case_id': row['case_id'], 'rep': row['rep'],
                                    'reference': truth, 'candidate': row['decision']})
    require(keys == [(c['case_id'], rep) for c in plan['cases'] for rep in range(32)], 'Piecewise source matrix differs')
    with (REPLAY/'records.jsonl').open(encoding='utf-8') as stream:
        for expected in rows:
            require(json.loads(next(stream)) == expected, 'Piecewise checkpoint, first stop or full certificate differs')
        require(next(stream, None) is None, 'Extra piecewise replay record')
    censored = sum(r['decision'] is None for r in rows)
    require(len(rows) == result['path_records'] == 576 and censored == result['right_censored'], 'Piecewise totals differ')
    require(relative_summary(rows, cases) == result['outcomes_relative_to_beta_exact'], 'Piecewise relative summary differs')
    combined, comparisons = None, None
    if not censored:
        baseline = [json.loads(line) for line in (BASE/'method_results.jsonl').open(encoding='utf-8')]
        combined = combined_summary(baseline, rows, plan, plan['methods']+[NAME])
        for name, value in combined.items():
            require(result['complete_combined_comparison'][name] == value, 'Combined piecewise summary differs: '+name)
        comparisons = {'fewer_shots': sum(r['shots_lower'] < r['reference_shots'] for r in rows),
                       'same_shots': sum(r['shots_lower'] == r['reference_shots'] for r in rows),
                       'more_shots': len(regressions),
                       'different_decision': sum(r['decision'] != r['reference_decision'] for r in rows)}
        require(comparisons == result['pathwise_comparison'], 'Piecewise pathwise comparison differs')
    for name, expected in inputs.items():
        require(sha(ROOT/name) == expected, 'Input changed during piecewise replay audit')
    report = {'status': 'passed', 'created_utc': datetime.now(timezone.utc).isoformat(), 'inputs': inputs,
              'verifier_source_sha256': sha(__file__), 'independent_summary_helper_sha256': sha(ROOT/'experiments/sota_round2/theory/audit_physical_fast_gate_replay.py'),
              'path_records': len(rows), 'right_censored': censored,
              'all_checkpoint_certificates_and_first_stops_reconstructed_without_producer_import': True,
              'relative_and_combined_summaries_recomputed': True,
              'pathwise_comparison': comparisons, 'cost_regression_paths': regressions,
              'correct_reference_decision_losses': decision_losses,
              'candidate_main_cost_ratios': combined['equal_case_geometric_cost_ratios'][NAME] if combined else None,
              'new_simulation_observations': 0,
              'limits': ['This verifies development replay, not independent validation.',
                         'No observed pathwise regression is not a universal dominance theorem. Beta crossings after D=512 can be delayed on other paths.',
                         'The gate is one fixed separately certified procedure, not a union of candidates.']}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': 'passed', 'path': str(output), 'sha256': sha(output),
                      'pathwise': comparisons, 'correct_reference_decision_losses': len(decision_losses),
                      'main_ratios': report['candidate_main_cost_ratios']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
