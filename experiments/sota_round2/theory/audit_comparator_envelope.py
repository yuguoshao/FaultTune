"""Independent static control-envelope audit from already audited summaries."""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))


def sha(name):
    return hashlib.sha256((ROOT/name).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def case_statistics(method, mean, correct, wrong, abstained):
    total = mean*32
    require(total.is_integer() and correct+wrong+abstained == 32, 'Cell is not a complete 32-replicate statistic')
    return {'method': method, 'replicates': 32, 'total_cost_shots': int(total), 'mean_cost_shots': mean,
            'correct_stops': correct, 'wrong_stops': wrong, 'cap_abstentions': abstained,
            'correct_stop_rate': correct/32, 'right_censored': 0}


def main():
    names = ['results/sota_round3/comparator_envelope_diagnostic.json',
             'results/sota_round3/paired_benchmark/preliminary_plan.json',
             'results/sota_round3/paired_benchmark/summary.json',
             'results/sota_round3/full_count_replay/summary.json',
             'results/sota_round3/fast_gate_replay_v1/summary.json',
             'results/sota_round3/paired_benchmark_audit/audit.json',
             'results/sota_round3/full_count_replay_audit/audit.json',
             'results/sota_round2/theory/physical_fast_gate_replay_independent_audit.json']
    inputs = {name: sha(name) for name in names}
    artifact, plan, pilot, full, gate, pilot_audit, full_audit, gate_audit = [read(name) for name in names]
    require(all(a['status'] == 'passed' for a in (pilot_audit, full_audit, gate_audit)), 'A source audit is incomplete')
    require(inputs[names[2]] == pilot_audit['inputs']['summary.json'], 'Pilot summary differs from audited input')
    require(inputs[names[3]] == full_audit['inputs'][names[3]], 'Full-count summary differs from audited input')
    require(inputs[names[4]] == gate_audit['inputs'][names[4]], 'Gate summary differs from audited input')
    require(sha(artifact['producer_source']) == artifact['producer_sha256'], 'Envelope producer source changed')
    for name, expected in artifact['input_sha256'].items():
        require(sha(name) == expected, 'Envelope input binding changed: '+name)
    cases = {c['case_id']: c for c in plan['cases'] if c['kind'] == 'main'}
    controls = plan['methods']+['full_safe_fine', 'full_safe_coarse', 'rb_fine', 'rb_coarse']
    require(len(cases) == 16 and len(set(controls)) == len(controls) == 15, 'Unexpected envelope dimensions')
    statistics = {}
    for cell in pilot['cells']:
        if cell['kind'] == 'main':
            statistics[cell['method'], cell['case_id']] = case_statistics(cell['method'], cell['mean_paid_shots'],
                cell['correct_stops'], cell['wrong'], cell['abstained'])
    for method in full['outcomes']:
        if method['kind'] != 'main':
            continue
        # The audited aggregate has no wrong stops, hence each observed-stop
        # status in its per-case cells is a correct stop. Reject rather than
        # silently apply that shortcut to any future data with nonzero errors.
        require(method['right_censored'] == method['observed_wrong_stops'] == 0,
                'Per-case direction counts require raw ledger for this dataset')
        correct_total = 0
        for cell in method['per_case']:
            require(cell['trajectories'] == 32 and cell['mean_shots_lower'] == cell['mean_shots_upper'], 'Censored full-count cell')
            correct = cell['statuses'].get('observed_stop', 0)
            abstained = cell['statuses'].get('observed_cap_abstention', 0)
            correct_total += correct
            statistics[method['method'], cell['case_id']] = case_statistics(method['method'], cell['mean_shots_lower'], correct, 0, abstained)
        require(correct_total == method['observed_correct_stops'], 'Full-count aggregate and per-case statuses differ')
    for cell in gate['complete_combined_comparison']['cells']:
        if cell['kind'] == 'main' and cell['method'] in ('physical_gate_m6', 'physical_gate_m7'):
            statistics[cell['method'], cell['case_id']] = case_statistics(cell['method'], cell['mean_paid_shots'],
                cell['correct_stops'], cell['wrong'], cell['abstained'])
    envelope = artifact['frozen_envelope']
    require(envelope['control_pool'] == controls and envelope['case_order'] == list(cases) and
            len(envelope['cases']) == 16 and
            envelope['excludes_all_gate_candidates'] is True and envelope['executable_method'] is False and
            envelope['power_qualified'] is False and envelope['diagnostic_only'] is True, 'Envelope scope or pool differs')
    require(hashlib.sha256(json.dumps(envelope, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            == artifact['frozen_envelope_sha256'], 'Frozen envelope payload differs')
    totals = {}
    for saved, (case_id, case) in zip(envelope['cases'], cases.items()):
        ordered = [statistics[method, case_id] for method in controls]
        best = min(c['total_cost_shots'] for c in ordered)
        winners = [c for c in ordered if c['total_cost_shots'] == best]
        require(saved['case_id'] == case_id and saved['all_control_statistics'] == ordered and
                saved['winning_controls'] == winners and saved['envelope_total_cost_shots'] == best and
                saved['envelope_mean_cost_shots'] == best/32, 'Control envelope or exact ties differ')
        totals[case_id] = best
    compared = []
    require([c['method'] for c in artifact['comparisons']] == ['physical_gate_m6', 'physical_gate_m7', 'beta_exact'],
            'Candidate comparison registry differs')
    for comparison in artifact['comparisons']:
        method = comparison['method']
        require([c['case_id'] for c in comparison['per_case']] == list(cases), 'Candidate case matrix differs')
        ratios, beta_ratios = [], []
        for saved in comparison['per_case']:
            case_id = saved['case_id']
            stats = statistics[method, case_id]
            require(all(saved[key] == value for key, value in stats.items()), 'Candidate cell statistics differ')
            ratio = Fraction(stats['total_cost_shots'], totals[case_id])
            beta_ratio = Fraction(stats['total_cost_shots'], statistics['beta_exact', case_id]['total_cost_shots'])
            require(saved['ratio_to_envelope_exact'] == [ratio.numerator, ratio.denominator] and
                    saved['ratio_to_envelope'] == float(ratio) and saved['ratio_to_global_beta_exact'] == float(beta_ratio),
                    'Candidate per-case cost ratio differs')
            ratios.append(float(ratio)); beta_ratios.append(float(beta_ratio))
        geometric = math.exp(sum(map(math.log, ratios))/16)
        beta_geometric = math.exp(sum(map(math.log, beta_ratios))/16)
        require(comparison['geometric_cost_ratio_to_casewise_control_envelope'] == geometric and
                comparison['geometric_cost_ratio_to_single_global_beta_exact'] == beta_geometric,
                'Candidate geometric envelope ratio differs')
        for key, field in [('observed_correct_stops', 'correct_stops'), ('observed_wrong_stops', 'wrong_stops'),
                           ('observed_cap_abstentions', 'cap_abstentions')]:
            require(comparison[key] == sum(statistics[method, c][field] for c in cases), 'Candidate total differs')
        require(comparison['observed_correct_stop_rate'] == comparison['observed_correct_stops']/512 and
                comparison['minimum_case_correct_stop_rate'] == min(statistics[method, c]['correct_stop_rate'] for c in cases),
                'Candidate power descriptors differ')
        require(comparison['cases_strictly_below_envelope_cost'] == sum(r < 1 for r in ratios) and
                comparison['cases_tying_envelope_cost'] == sum(r == 1 for r in ratios) and
                comparison['cases_strictly_above_envelope_cost'] == sum(r > 1 for r in ratios),
                'Candidate case win/tie/loss counts differ')
        compared.append({'method': method, 'ratio_to_envelope': geometric, 'ratio_to_global_beta': beta_geometric})
    require(inputs == {name: sha(name) for name in names}, 'Input changed during envelope verification')
    report = {'status': 'passed', 'inputs': inputs, 'verifier_source_sha256': sha('experiments/sota_round2/theory/audit_comparator_envelope.py'),
              'cases': 16, 'control_pool_size': 15, 'all_240_control_cells_and_exact_ties_checked': True,
              'candidate_comparisons': compared, 'new_simulation_observations': 0,
              'limits': ['Independent arithmetic uses already independently audited summaries, not a new raw-ledger audit.',
                         'This is a hindsight per-case cost envelope, not an executable method or a power-qualified benchmark.',
                         'It does not relax the 95% correct-stop target or establish SOTA performance.']}
    output = ROOT/'results/sota_round2/theory/comparator_envelope_independent_audit.json'
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': 'passed', 'path': output.as_posix(), 'sha256': sha(output.relative_to(ROOT).as_posix()),
                      'comparisons': compared}))


if __name__ == '__main__':
    main()
