"""Append-only piecewise candidate diagnostic against a frozen15-control envelope."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'results/sota_round3'
ENVELOPE = BASE/'comparator_envelope_diagnostic.json'
GATE = BASE/'piecewise_gate_replay'
AUDIT = ROOT/'results/sota_round2/theory/piecewise_gate_replay_independent_audit.json'
OUT = BASE/'piecewise_comparator_envelope_diagnostic.json'
FROZEN_ENVELOPE = '68b0f7b18813383445226713fba67ceb1b9ba507147ae8c6ae53c3b65c6bbc13'
CANDIDATE = 'physical_gate_m6_piecewise_D512'
FULL = ('full_safe_fine', 'full_safe_coarse', 'rb_fine', 'rb_coarse')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def payload_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def main():
    paths = [ENVELOPE, GATE/'records.jsonl', GATE/'summary.json', AUDIT]
    inputs = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    prior, summary, audit = read(ENVELOPE), read(GATE/'summary.json'), read(AUDIT)
    envelope = prior['frozen_envelope']
    assert payload_sha(envelope) == prior['frozen_envelope_sha256'] == FROZEN_ENVELOPE
    assert envelope['power_qualified'] is False and envelope['executable_method'] is False
    assert len(envelope['control_pool']) == 15 and not any('gate' in name for name in envelope['control_pool'])
    assert audit['status'] == 'passed' and summary['status'] == 'complete_development_replay'
    for p in (GATE/'records.jsonl', GATE/'summary.json'):
        assert sha(p) == audit['inputs'][p.relative_to(ROOT).as_posix()]
    for name, expected in prior['input_sha256'].items():
        assert sha(ROOT/name) == expected
    groups, keys, paired, nulls = defaultdict(list), set(), Counter(), []
    for line in (GATE/'records.jsonl').open(encoding='utf-8'):
        row = json.loads(line)
        assert row['method'] == CANDIDATE and row['decision'] in ('reject', 'accept', 'abstain')
        assert row['shots_lower'] == row['shots_upper'] and 0 < row['shots_lower'] <= 4096
        assert row['decision'] != 'abstain' or row['shots_lower'] == 4096
        key = row['case_id'], row['rep']
        assert key not in keys
        keys.add(key)
        paired['fewer_shots' if row['shots_lower'] < row['reference_shots'] else
               'more_shots' if row['shots_lower'] > row['reference_shots'] else 'same_shots'] += 1
        paired['different_decision'] += row['decision'] != row['reference_decision']
        compact = {name: row[name] for name in ('case_id', 'rep', 'decision', 'shots_lower', 'reference_shots')}
        if row['kind'] == 'main':
            groups[row['case_id']].append(compact)
        else:
            assert row['kind'] == 'null'
            nulls.append(compact)
    assert len(keys) == 576 and set(groups) == set(envelope['case_order']) and len(nulls) == 64
    pairwise = {key: paired[key] for key in ('fewer_shots', 'same_shots', 'more_shots', 'different_decision')}
    assert pairwise == summary['pathwise_comparison'] == audit['pathwise_comparison']
    per_case = []
    for cell in envelope['cases']:
        rows = groups[cell['case_id']]
        assert len(rows) == 32 and {r['rep'] for r in rows} == set(range(32))
        truth = 'reject' if cell['C'] > 0 else 'accept'
        counts = Counter(r['decision'] for r in rows)
        total = sum(r['shots_lower'] for r in rows)
        controls = {r['method']: r for r in cell['all_control_statistics']}
        assert sum(r['reference_shots'] for r in rows) == controls['beta_exact']['total_cost_shots']
        ratio = Fraction(total, cell['envelope_total_cost_shots'])
        per_case.append({'case_id': cell['case_id'], 'A': cell['A'], 'B': cell['B'], 'C': cell['C'],
                         'replicates': 32, 'candidate_total_cost_shots': total, 'candidate_mean_cost_shots': total/32,
                         'candidate_correct_stops': counts[truth], 'candidate_correct_stop_rate': counts[truth]/32,
                         'candidate_wrong_stops': 32-counts[truth]-counts['abstain'], 'candidate_cap_abstentions': counts['abstain'],
                         'envelope_mean_cost_shots': cell['envelope_mean_cost_shots'],
                         'envelope_winning_controls': cell['winning_controls'],
                         'mean_cost_gap_to_envelope_shots': (total-cell['envelope_total_cost_shots'])/32,
                         'ratio_to_envelope': float(ratio), 'ratio_to_envelope_exact': [ratio.numerator, ratio.denominator],
                         'relative_cost_gap_to_envelope': float(ratio-1),
                         'ratios_to_fixed_global_controls': {name: total/controls[name]['total_cost_shots'] for name in envelope['control_pool']}})
    gm_envelope = math.exp(sum(math.log(row['ratio_to_envelope']) for row in per_case)/16)
    global_comparisons = {name: math.exp(sum(math.log(row['ratios_to_fixed_global_controls'][name]) for row in per_case)/16)
                          for name in envelope['control_pool']}
    assert abs(gm_envelope-prior['single_global_beta_exact_to_envelope_factor']*global_comparisons['beta_exact']) < 1e-14
    for name, expected in audit['candidate_main_cost_ratios'].items():
        if name != CANDIDATE:
            assert abs(global_comparisons[name]-expected) < 1e-14
    correct = sum(row['candidate_correct_stops'] for row in per_case)
    artifact = {'version': 'piecewise_candidate_vs_frozen15_cost_envelope_v1', 'status': 'complete_development_cost_diagnostic',
                'created_utc': datetime.now(timezone.utc).isoformat(), 'input_sha256': inputs,
                'producer_source': Path(__file__).resolve().relative_to(ROOT).as_posix(),
                'producer_sha256': sha(Path(__file__).resolve()), 'new_simulation_observations': 0,
                'candidate': CANDIDATE,
                'frozen_envelope_binding': {'path': ENVELOPE.relative_to(ROOT).as_posix(), 'file_sha256': inputs[ENVELOPE.relative_to(ROOT).as_posix()],
                                            'envelope_payload_sha256': FROZEN_ENVELOPE, 'control_pool_unchanged': True,
                                            'candidate_added_to_control_pool': False},
                'main': {'cases': 16, 'trajectories': 512, 'geometric_cost_ratio_to_casewise_control_envelope': gm_envelope,
                         'geometric_cost_ratio_to_single_global_beta_exact': global_comparisons['beta_exact'],
                         'global_full_count_comparisons': {name: global_comparisons[name] for name in FULL},
                         'all15_global_fixed_control_comparisons': global_comparisons,
                         'mean_cost_shots': sum(row['candidate_total_cost_shots'] for row in per_case)/512,
                         'observed_correct_stops': correct, 'observed_correct_stop_rate': correct/512,
                         'observed_wrong_stops': sum(row['candidate_wrong_stops'] for row in per_case),
                         'observed_cap_abstentions': sum(row['candidate_cap_abstentions'] for row in per_case),
                         'minimum_case_correct_stop_rate': min(row['candidate_correct_stop_rate'] for row in per_case),
                         'cases_below_envelope': sum(row['ratio_to_envelope'] < 1 for row in per_case),
                         'cases_tying_envelope': sum(row['ratio_to_envelope'] == 1 for row in per_case),
                         'cases_above_envelope': sum(row['ratio_to_envelope'] > 1 for row in per_case)},
                'null': {'trajectories': 64, 'observed_wrong_stops': sum(r['decision'] != 'abstain' for r in nulls),
                         'observed_cap_abstentions': sum(r['decision'] == 'abstain' for r in nulls)},
                'all576_pathwise_comparison_to_beta_exact': pairwise, 'per_case': per_case,
                'formal_acceptance_status': {'sota_acceptance_established': False, 'power_qualified': False,
                                             'correct_stop_target': .95, 'diagnostic_only': True},
                'limits': ['The fixed15 envelope is a development hindsight diagnostic, not an executable strategy.',
                           'The candidate and all gates remain outside the frozen control pool.',
                           'Global full-count comparisons use the same one named control on every case; envelope winners may differ by case.',
                           'Costs include all observed wrong stops and cap abstentions; no censoring or new continuation was introduced.',
                           'Observed correct-stop rate remains below95%; no power-qualified SOTA acceptance conclusion follows.',
                           'No observed regression against beta_exact in576 paths is not a universal dominance theorem or a holdout result.']}
    for name, expected in inputs.items():
        assert sha(ROOT/name) == expected
    with OUT.open('x', encoding='utf-8') as stream:
        json.dump(artifact, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': artifact['status'], 'sha256': sha(OUT), 'main': artifact['main'],
                      'per_case_gaps': [{'case_id': row['case_id'], 'gap_shots': row['mean_cost_gap_to_envelope_shots'],
                                         'ratio': row['ratio_to_envelope']} for row in per_case]}))


if __name__ == '__main__':
    main()
