"""Development-only cost and information diagnostics; no new trajectories.

All 24 old development cases and all 16 replicates supply structural costs.
Detailed deterministic evidence replay uses rep=0 in every case, selected by
label before reading outcomes. Prospective data are never accessed.
"""
from collections import defaultdict
from itertools import groupby
from pathlib import Path
import hashlib
import json
import math
import sys
import time

import numpy as np
from scipy.signal import fftconvolve
from scipy.special import gammaln, logsumexp, xlogy
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'experiments/structured_validation'))
import run_refined as runner
import weighted_fisher as fisher
from conditional_fisher import centered_cgf_float

BASE = ROOT/'results/structured_validation/refined_development_v1'
OUTPUT = ROOT/'results/sota_round2/diagnostics/development_cost_diagnosis.json'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def stats(values):
    values = np.asarray(values, dtype=float)
    return dict(zip(('min', 'p10', 'median', 'p90', 'max'),
                    map(float, np.percentile(values, [0, 10, 50, 90, 100]))), mean=float(values.mean()))


def cost_comparison(rows, cost_key, reference_key='shots'):
    by_case = defaultdict(list)
    for row in rows:
        by_case[row['case_id']].append(row)
    cases = []
    for case_id, local in sorted(by_case.items()):
        proposed = float(np.mean([r[cost_key] for r in local]))
        reference = float(np.mean([r[reference_key] for r in local]))
        cases.append({'case_id': case_id, 'ratio_of_means': proposed/reference,
                      'proposed_mean': proposed, 'reference_mean': reference})
    ratio = math.exp(sum(math.log(r['ratio_of_means']) for r in cases)/len(cases))
    return {'equal_case_geometric_ratio': ratio, 'descriptive_savings_fraction': 1-ratio,
            'cases': cases}


def kl(p, q):
    return float(xlogy(p, p/q)+xlogy(1-p, (1-p)/(1-q)))


def block_information(case, block):
    arm = block['arm']
    co = arm['cos_numerator']/arm['denominator']
    si = arm['sin_numerator']/arm['denominator']
    q = case['A']+case['B']*co
    pm, pp = q-case['C']*si, q+case['C']*si
    assert 0 < pm < 1 and 0 < pp < 1
    positive, negative = pp*(1-pm), pm*(1-pp)
    discordance = positive+negative
    paired = float(xlogy(positive, 2*positive/discordance)+xlogy(negative, 2*negative/discordance))
    full = kl(pm, q)+kl(pp, q)
    return pm, pp, full, paired


def total_count_kl(m, pm, pp):
    """Float convolution diagnostic, with explicitly measured omitted mass.

The full-data KL minus this KL is expected conditional-Fisher information.
This is numerical diagnostic integration, not an outward certificate.
"""
    arrays, offsets, retained = [], [], []
    for p in (pm, pp):
        lo, hi = int(binom.ppf(1e-15, m, p)), int(binom.ppf(1-1e-15, m, p))
        support = np.arange(lo, hi+1)
        probabilities = np.exp(binom.logpmf(support, m, p))
        arrays.append(probabilities)
        offsets.append(lo)
        retained.append(float(probabilities.sum()))
    total = (np.convolve(*arrays) if max(map(len, arrays)) < 400 else fftconvolve(*arrays))
    negative_roundoff = float(-total[total < 0].sum())
    total = np.maximum(total, 0)
    support = np.arange(len(total))+sum(offsets)
    log_null = binom.logpmf(support, 2*m, (pm+pp)/2)
    positive = total > 0
    value = float(np.sum(total[positive]*(np.log(total[positive])-log_null[positive])))
    assert math.isfinite(value) and value >= -1e-8
    return max(0., value), {'absolute_retained_mass_error': abs(1-float(total.sum())),
                           'negative_fft_roundoff_removed': negative_roundoff,
                           'support_terms': len(total)}


def main():
    started = time.perf_counter()
    assert not OUTPUT.exists(), 'Preserve existing diagnostic result'
    spec = json.loads((BASE/'spec.json').read_text())
    audit = json.loads((BASE/'audit.json').read_text())
    assert audit['status'] == 'passed'
    inputs = {name: digest(BASE/name) for name in ('spec.json', 'summary.json', 'trials.jsonl', 'audit.json')}
    assert {name: inputs[name] for name in audit['run_input_sha256']} == audit['run_input_sha256']
    assert len(spec['cases']) == 24 and spec['repetitions'] == 16 and spec['engines'] == list(runner.ENGINES)
    for name, expected in spec['sources'].items():
        assert digest(ROOT/name) == expected, name
    sources = dict(spec['sources'])
    sources[Path(__file__).resolve().relative_to(ROOT).as_posix()] = digest(Path(__file__).resolve())
    cases = {c['case_id']: c for c in spec['cases']}
    costs, hedge = {}, {}
    line_count = 0
    with (BASE/'trials.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            line_count += 1
            key = row['case_id'], row['rep'], row['method']
            assert key not in costs
            costs[key] = row['shots']
            if row['method'] == 'closed_hedge':
                hedge[(row['case_id'], row['rep'])] = row
    assert line_count == 6912 and len(hedge) == 384
    assert set(hedge) == {(case_id, rep) for case_id in cases for rep in range(16)}
    structural = []
    for key, row in sorted(hedge.items()):
        case_id, rep = key
        history, shots = row['history'], row['shots']
        last_look = history[-1]['look']
        terminal = [b for b in history if b['look'] == last_look]
        previous = shots-sum(2*b['m'] for b in terminal)
        assert shots == sum(2*b['m'] for b in history)
        full_info = paired_info = 0.
        phase_information, phase_shots = defaultdict(float), defaultdict(int)
        for block in history:
            pm, pp, full, paired = block_information(cases[case_id], block)
            full_info += block['m']*full
            paired_info += block['m']*paired
            phase_information[block['phase']] += block['m']*full
            phase_shots[block['phase']] += 2*block['m']
        item = {'case_id': case_id, 'rep': rep, 'shots': shots,
                'previous_checkpoint': previous,
                'terminal_interval_shots': shots-previous,
                'terminal_interval_fraction': (shots-previous)/shots,
                'terminal_block_shots': 2*history[-1]['m'],
                'terminal_block_fraction': 2*history[-1]['m']/shots,
                'terminal_phase': history[-1]['phase'],
                'optimistic_interval_floor': previous+2,
                'posthoc_best_beta_fisher': min(costs[(case_id, rep, 'closed_'+name)] for name in ('beta', 'fisher')),
                'posthoc_best_six_engines': min(costs[(case_id, rep, 'closed_'+name)] for name in runner.ENGINES),
                'terminal_log_hedge': float(row['certificate']['evidence_bounds'][0]),
                'exploration_shot_fraction': 2*row['exploration_pairs']/shots,
                'full_oracle_information_nats': full_info,
                'paired_oracle_information_nats': paired_info,
                'paired_information_retention': paired_info/full_info,
                'phase_shots': dict(phase_shots), 'phase_full_information_nats': dict(phase_information)}
        for engine in runner.ENGINES:
            item['closed_'+engine] = costs[(case_id, rep, 'closed_'+engine)]
        structural.append(item)

    replay = []
    numerical_errors = []
    threshold = math.log(100)
    for case_id in sorted(cases):
        row = hedge[(case_id, 0)]
        state, counts = fisher.empty_state(), np.zeros(4, dtype=np.int64)
        phase_stop = None
        all_looks = []
        full_info = total_kl = paired_info = changing_true_odds_log = 0.
        for look, grouped in groupby(row['history'], key=lambda block: block['look']):
            blocks = list(grouped)
            for block in blocks:
                counts += block['pair_counts']
                fisher.update_block(state, block['m'], block['minus_successes'], block['plus_successes'])
                b, c = counts[1:3]
                beta = float(gammaln(b+.5)+gammaln(c+.5)-gammaln(b+c+1)-2*gammaln(.5)+(b+c)*math.log(2))
                fl = fisher.screen(state)['log_evidence']
                hl = float(np.logaddexp(beta, fl)-math.log(2))
                if phase_stop is None and hl >= threshold-1e-6:
                    certificate = runner.evidence_certificate('hedge', counts, state)
                    if certificate['decision'] != 'abstain':
                        phase_stop = {'shots': state['shots'], 'phase': block['phase'],
                                      'look': look, 'certificate': certificate}
                pm, pp, full, paired = block_information(cases[case_id], block)
                loss, detail = total_count_kl(block['m'], pm, pp)
                numerical_errors.append(detail)
                assert loss <= block['m']*full+1e-7
                if block['m'] == 1:
                    assert abs(full-loss-paired) < 1e-9
                full_info += block['m']*full
                paired_info += block['m']*paired
                total_kl += loss
                odds = pp*(1-pm)/(pm*(1-pp))
                changing_true_odds_log += ((block['plus_successes']-block['minus_successes'])/2*math.log(odds)
                    -centered_cgf_float(block['m'], block['plus_successes']+block['minus_successes'], odds, 1.))
            all_looks.append({'shots': state['shots'], 'beta_log': beta, 'fisher_log': fl, 'hedge_log': hl})
        assert phase_stop is not None and state['shots'] == row['shots']
        assert abs(hl-float(row['certificate']['evidence_bounds'][0])) < 1e-5
        previous = all_looks[-2] if len(all_looks) > 1 else {'shots': 0, 'hedge_log': 0.}
        increment = hl-previous['hedge_log']
        assert increment > 0 and previous['hedge_log'] < threshold+1e-5
        fraction = min(1., max(0., (threshold-previous['hedge_log'])/increment))
        interpolated = previous['shots']+fraction*(row['shots']-previous['shots'])
        expert_logs = float(state['score'])*state['log_odds']-state['cgf']
        best = int(np.argmax(expert_logs))
        replay.append({'case_id': case_id, 'rep': 0, 'shots': row['shots'],
                       'phase_check_shots': phase_stop['shots'], 'phase_check_certificate': phase_stop,
                       'linear_log_interpolation_shots_heuristic': interpolated,
                       'previous_checkpoint': previous, 'terminal_checkpoint': all_looks[-1],
                       'hedge_log2_branch_penalty_nats': max(beta, fl)-hl,
                       'max_grid_expert_minus_fisher_mixture_nats': float(expert_logs[best])-fl,
                       'best_grid_expert_log_odds': float(state['log_odds'][best]),
                       'best_grid_expert_log_capital': float(expert_logs[best]),
                       'changing_true_odds_oracle_log_capital': changing_true_odds_log,
                       'oracle_full_information_nats': full_info,
                       'oracle_paired_information_nats': paired_info,
                       'discarded_total_count_kl_nats': total_kl,
                       'conditional_block_information_retention': 1-total_kl/full_info,
                       'paired_information_retention': paired_info/full_info})
        print(json.dumps({'replayed_case': case_id, 'completed_cases': len(replay)}), flush=True)

    report = {
        'status': 'complete_development_only_diagnostic',
        'selection': {'cases': 24, 'structural_repetitions': 16, 'detailed_replay_rep': 0,
                      'selection_rule': 'All old development cases; mechanically fixed rep 0 for detailed replay. No prospective data read.'},
        'sources': sources,
        'input_sha256': {(BASE/name).relative_to(ROOT).as_posix(): value for name, value in inputs.items()},
        'structural_summary': {
            'records': len(structural), 'first_32_shot_stops': sum(r['previous_checkpoint'] == 0 for r in structural),
            'terminal_interval_fraction': stats([r['terminal_interval_fraction'] for r in structural]),
            'terminal_block_fraction': stats([r['terminal_block_fraction'] for r in structural]),
            'exploration_shot_fraction': stats([r['exploration_shot_fraction'] for r in structural]),
            'terminal_log_hedge_minus_log100': stats([r['terminal_log_hedge']-threshold for r in structural]),
            'paired_information_retention': stats([r['paired_information_retention'] for r in structural]),
            'optimistic_terminal_interval_floor': cost_comparison(structural, 'optimistic_interval_floor'),
            'posthoc_best_beta_fisher': cost_comparison(structural, 'posthoc_best_beta_fisher'),
            'posthoc_best_six_engines': cost_comparison(structural, 'posthoc_best_six_engines'),
            'named_engine_costs': {name: cost_comparison(structural, 'closed_'+name) for name in runner.ENGINES}},
        'detailed_replay_summary': {
            'records': len(replay),
            'valid_extra_phase_checks': cost_comparison(replay, 'phase_check_shots'),
            'linear_log_interpolation_heuristic': cost_comparison(replay, 'linear_log_interpolation_shots_heuristic'),
            'branch_penalty_nats': stats([r['hedge_log2_branch_penalty_nats'] for r in replay]),
            'grid_prior_penalty_nats': stats([r['max_grid_expert_minus_fisher_mixture_nats'] for r in replay]),
            'conditional_block_information_retention': stats([r['conditional_block_information_retention'] for r in replay]),
            'paired_information_retention': stats([r['paired_information_retention'] for r in replay]),
            'max_pmf_mass_error': max(r['absolute_retained_mass_error'] for r in numerical_errors),
            'max_negative_fft_mass_removed': max(r['negative_fft_roundoff_removed'] for r in numerical_errors),
            'diagnostic_pmf_evaluations': len(numerical_errors)},
        'structural_records': structural, 'detailed_replay_records': replay,
        'interpretation_limits': [
            'All performance diagnostics are development-only, unadjusted, and not prospective superiority claims.',
            'The previous-checkpoint-plus-one-pair floor is an optimistic reclaimable-interval diagnostic, not an attainable policy or a bound on every redesigned algorithm.',
            'Linear interpolation of log evidence is a heuristic. Splitting a Fisher block changes its conditioning and cannot be assumed to preserve evidence or validity under within-block optional stopping.',
            'Extra checks after existing complete blocks are valid on the same unchanged prefixes and are independently certified here; only 24 mechanically selected histories were replayed.',
            'Posthoc best branch/engine costs use future outcomes and are unattainable selectors; they diagnose selection/mixture headroom only.',
            'Known-parameter information uses the true development simulator parameters only diagnostically. No deployed policy is given these parameters.',
            'Full Bernoulli information minus KL of the total count is expected conditional-block information. Floating truncated convolution is a numerical diagnostic, not a rigorous outward enclosure.',
            'Information-rate ratios do not directly imply equal finite-sample stopping-cost ratios. The hedge already contains block-conditional evidence, so paired-to-full information gains are not all available versus hedge.',
            'Removing mixture/prior penalties without paying for adaptive parameter learning is not a valid evidence procedure.',
            'Schedule, monitoring frequency, initialization and measurement access must be upgraded equally for candidate and all applicable strong controls in any new fair comparison.'],
        'elapsed_seconds': time.perf_counter()-started}
    for name, expected in sources.items():
        assert digest(ROOT/name) == expected
    for name, expected in inputs.items():
        assert digest(BASE/name) == expected
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(OUTPUT), 'structural': report['structural_summary'],
                      'replay': report['detailed_replay_summary'], 'elapsed_seconds': report['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
