"""Read-only diagnostic against the per-case fastest of15 frozen controls.

This is a hindsight cost envelope on development data, not an executable
method, not power-qualified, and not a substitute for the acceptance target.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/sota_round3/comparator_envelope_diagnostic.json'
PILOT = ROOT/'results/sota_round3/paired_benchmark'
FULL = ROOT/'results/sota_round3/full_count_replay'
GATE = ROOT/'results/sota_round3/fast_gate_replay_v1'
FULL_METHODS = ('full_safe_fine', 'full_safe_coarse', 'rb_fine', 'rb_coarse')
CANDIDATES = ('physical_gate_m6', 'physical_gate_m7', 'beta_exact')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def method_case_stats(rows, case):
    assert len(rows) == 32 and {r['rep'] for r in rows} == set(range(32))
    truth = 'reject' if case['C'] > 0 else 'accept'
    assert all(r['decision'] in ('reject', 'accept', 'abstain') for r in rows)
    decisions = Counter(r['decision'] for r in rows)
    cost = sum(r['shots'] for r in rows)
    return {'method': rows[0]['method'], 'replicates': 32, 'total_cost_shots': cost,
            'mean_cost_shots': cost/32, 'correct_stops': decisions[truth],
            'wrong_stops': sum(r['decision'] not in (truth, 'abstain') for r in rows),
            'cap_abstentions': decisions['abstain'], 'correct_stop_rate': decisions[truth]/32,
            'right_censored': 0}


def main():
    paths = [PILOT/'preliminary_plan.json', PILOT/'method_results.jsonl', FULL/'records.jsonl',
             FULL/'summary.json', GATE/'records.jsonl', GATE/'summary.json',
             ROOT/'results/sota_round3/paired_benchmark_audit/audit.json',
             ROOT/'results/sota_round3/full_count_replay_audit/audit.json',
             ROOT/'results/sota_round2/theory/physical_fast_gate_replay_independent_audit.json']
    inputs = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    original_audit, full_audit, gate_audit = map(read, paths[-3:])
    assert all(a['status'] == 'passed' for a in (original_audit, full_audit, gate_audit))
    for name in ('preliminary_plan.json', 'method_results.jsonl'):
        assert sha(PILOT/name) == original_audit['inputs'][name]
    for path in (FULL/'records.jsonl', FULL/'summary.json'):
        assert sha(path) == full_audit['inputs'][path.relative_to(ROOT).as_posix()]
    for path in (GATE/'records.jsonl', GATE/'summary.json'):
        assert sha(path) == gate_audit['inputs'][path.relative_to(ROOT).as_posix()]
    plan = read(PILOT/'preliminary_plan.json')
    controls = tuple(plan['methods'])+FULL_METHODS
    assert len(controls) == len(set(controls)) == 15 and not any('physical_gate' in name for name in controls)
    cases = {c['case_id']: c for c in plan['cases'] if c['kind'] == 'main'}
    assert len(cases) == 16
    by_method_case, keys = defaultdict(list), set()
    for path, source_methods in ((PILOT/'method_results.jsonl', tuple(plan['methods'])),
                                 (FULL/'records.jsonl', FULL_METHODS),
                                 (GATE/'records.jsonl', ('physical_gate_m5', 'physical_gate_m6', 'physical_gate_m7'))):
        count = 0
        for line in path.open(encoding='utf-8'):
            source = json.loads(line)
            assert source['method'] in source_methods
            count += 1
            if source['kind'] != 'main' or source['method'] == 'physical_gate_m5':
                continue
            key = (source['method'], source['case_id'], source['rep'])
            assert key not in keys
            keys.add(key)
            if 'shots' in source:
                shots = source['shots']
            else:
                assert source['shots_lower'] == source['shots_upper'] and source['decision'] is not None
                assert source['status'] in ('observed_stop', 'observed_cap_abstention')
                shots = source['shots_lower']
            assert 0 < shots <= 4096 and (source['decision'] != 'abstain' or shots == 4096)
            by_method_case[source['method'], source['case_id']].append(
                dict({name: source[name] for name in ('method', 'case_id', 'rep', 'decision', 'seed')}, shots=shots))
        assert count == len(source_methods)*576
    all_methods = controls+('physical_gate_m6', 'physical_gate_m7')
    assert keys == {(method, case, rep) for method in all_methods for case in cases for rep in range(32)}
    stats = {(method, case): method_case_stats(by_method_case[method, case], cases[case])
             for method in all_methods for case in cases}
    for case in cases:
        for rep in range(32):
            seeds = {next(r['seed'] for r in by_method_case[method, case] if r['rep'] == rep) for method in all_methods}
            assert len(seeds) == 1
    envelope_cases = []
    for case_id, case in cases.items():
        competitors = [stats[method, case_id] for method in controls]
        best = min(row['total_cost_shots'] for row in competitors)
        winners = [row for row in competitors if row['total_cost_shots'] == best]
        envelope_cases.append({'case_id': case_id, 'A': case['A'], 'B': case['B'], 'C': case['C'],
                               'replicates': 32, 'envelope_total_cost_shots': best,
                               'envelope_mean_cost_shots': best/32,
                               'winning_controls': winners,
                               'all_control_statistics': competitors})
    envelope = {'control_pool': list(controls), 'case_count': len(cases), 'replicates_per_case': 32,
                'case_order': list(cases), 'cases': envelope_cases,
                'selection_rule': 'minimum within-case mean observed capped physical-shot cost across the fixed15 controls; keep all exact ties',
                'excludes_all_gate_candidates': True,
                'executable_method': False, 'power_qualified': False,
                'diagnostic_only': True}
    comparisons = []
    for method in CANDIDATES:
        per_case = []
        for cell in envelope_cases:
            case_id = cell['case_id']
            candidate, beta = stats[method, case_id], stats['beta_exact', case_id]
            ratio_envelope = Fraction(candidate['total_cost_shots'], cell['envelope_total_cost_shots'])
            ratio_beta = Fraction(candidate['total_cost_shots'], beta['total_cost_shots'])
            per_case.append({'case_id': case_id, **candidate,
                             'ratio_to_envelope': float(ratio_envelope),
                             'ratio_to_envelope_exact': [ratio_envelope.numerator, ratio_envelope.denominator],
                             'ratio_to_global_beta_exact': float(ratio_beta),
                             'beta_exact_mean_cost_shots': beta['mean_cost_shots'],
                             'envelope_mean_cost_shots': cell['envelope_mean_cost_shots'],
                             'envelope_winning_methods': [w['method'] for w in cell['winning_controls']]})
        ratio_e = math.exp(sum(math.log(row['ratio_to_envelope']) for row in per_case)/16)
        ratio_b = math.exp(sum(math.log(row['ratio_to_global_beta_exact']) for row in per_case)/16)
        correct = sum(row['correct_stops'] for row in per_case)
        comparisons.append({'method': method, 'case_count': 16, 'trajectories': 512,
                            'geometric_cost_ratio_to_casewise_control_envelope': ratio_e,
                            'geometric_cost_ratio_to_single_global_beta_exact': ratio_b,
                            'observed_correct_stops': correct, 'observed_correct_stop_rate': correct/512,
                            'observed_wrong_stops': sum(row['wrong_stops'] for row in per_case),
                            'observed_cap_abstentions': sum(row['cap_abstentions'] for row in per_case),
                            'minimum_case_correct_stop_rate': min(row['correct_stop_rate'] for row in per_case),
                            'cases_strictly_below_envelope_cost': sum(row['ratio_to_envelope'] < 1 for row in per_case),
                            'cases_tying_envelope_cost': sum(row['ratio_to_envelope'] == 1 for row in per_case),
                            'cases_strictly_above_envelope_cost': sum(row['ratio_to_envelope'] > 1 for row in per_case),
                            'per_case': per_case})
    factor = comparisons[-1]['geometric_cost_ratio_to_casewise_control_envelope']
    for item in comparisons:
        assert abs(item['geometric_cost_ratio_to_casewise_control_envelope']-
                   factor*item['geometric_cost_ratio_to_single_global_beta_exact']) < 1e-14
    for method in ('physical_gate_m6', 'physical_gate_m7'):
        item = next(x for x in comparisons if x['method'] == method)
        assert abs(item['geometric_cost_ratio_to_single_global_beta_exact']-
                   gate_audit['main_comparisons'][method]['beta_exact']) < 1e-14
    artifact = {'version': 'fixed15_casewise_cost_envelope_development_diagnostic_v1',
                'status': 'complete_readonly_diagnostic', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'producer_source': Path(__file__).resolve().relative_to(ROOT).as_posix(),
                'producer_sha256': sha(Path(__file__).resolve()), 'input_sha256': inputs,
                'new_simulation_observations': 0, 'new_method_selection_for_execution': False,
                'metric': 'exp(mean_over16cases(log(candidate_case_mean_cost/reference_case_mean_cost))); lower is better',
                'frozen_envelope': envelope, 'frozen_envelope_sha256': object_sha(envelope),
                'comparisons': comparisons,
                'single_global_beta_exact_to_envelope_factor': factor,
                'acceptance_target_status': {'power_target': .95, 'power_qualified': False,
                                              'description': 'The diagnostic does not relax the95% correct-stop requirement; current aggregate rates are about90% and weaker individual cases are explicitly retained.'},
                'interpretation': ['The same beta_exact is used on every case in the global-single-comparator calculation.',
                                   'The envelope instead uses the lowest mean cost of15 controls separately within each case, including all ties.',
                                   'This hindsight envelope is not an executable algorithm and must not be presented as an actual deployed comparator.',
                                   'All data are reused development observations; there is no holdout superiority conclusion or confidence interval here.',
                                   'Controls are not filtered by observed power and this is not a power-qualified SOTA acceptance comparison.',
                                   'm5/m6/m7 and future candidate gates are excluded from the frozen control pool.',
                                   'Future candidates may be compared to this immutable envelope by its payload hash in separate new artifacts.']}
    for name, expected in inputs.items():
        assert sha(ROOT/name) == expected
    with OUT.open('x', encoding='utf-8') as stream:
        json.dump(artifact, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': artifact['status'], 'sha256': sha(OUT),
                      'frozen_envelope_sha256': artifact['frozen_envelope_sha256'],
                      'comparisons': [{k: v for k, v in row.items() if k != 'per_case'} for row in comparisons],
                      'winners': [{'case_id': row['case_id'], 'mean_cost': row['envelope_mean_cost_shots'],
                                   'winners': [{k: w[k] for k in ('method', 'correct_stops', 'correct_stop_rate')}
                                               for w in row['winning_controls']]}
                                  for row in envelope_cases]}))


if __name__ == '__main__':
    main()
