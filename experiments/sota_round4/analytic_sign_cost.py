"""Analytic operating characteristics for EVERY-PAIR sign-boundary monitoring.

Fixed PSR produces iid independent Bernoulli pairs with p_minus=A-C and
p_plus=A+C. Discordance indicators are iid Bernoulli(r); conditional signs
are iid Bernoulli(q) and independent of the geometric waiting times. A
sign-only first-crossing time can therefore be mixed with negative-binomial
waiting times. This does NOT analyze sparse checkpoints or physical-prefix
gates. Float outputs are numerical approximations, not interval certificates.
"""
from fractions import Fraction
from itertools import product
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import time

import numpy as np
import scipy
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/sota_round3'))
import exact_sign_boundary as exact

CASE_PATH = ROOT/'results/sota_round3/paired_benchmark/preliminary_plan.json'
BETA_PATHS = (ROOT/'results/sota_round3/beta_boundary_N2048.json',
              ROOT/'results/sota_round3/cap16384_preparation/beta_exact.json')
AUDIT_PATHS = (ROOT/'results/sota_round2/theory/beta_boundary_N2048_independent_audit.json',
               ROOT/'results/sota_round2/theory/cap16384_preparation_independent_audit.json')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def psr_parameters(case):
    a, c = float(case['A']), float(case['C'])
    require(math.isfinite(a) and math.isfinite(c), 'Finite case parameters required')
    minus, plus = a-c, a+c
    require(0 <= minus <= 1 and 0 <= plus <= 1, 'Invalid fixed-PSR Bernoulli probabilities')
    positive = (1-minus)*plus
    negative = minus*(1-plus)
    rate = positive+negative
    q = positive/rate if rate else None
    return {'p_minus': minus, 'p_plus': plus, 'discordance_rate': rate,
            'conditional_plus_probability': q}


def first_crossing_distribution(boundaries, q, max_steps=None):
    """Biased iid-sign absorbing DP, with separate first positive/negative mass."""
    exact.validate_table(boundaries)
    horizon = len(boundaries)-1
    steps = horizon if max_steps is None else exact.integer(max_steps, 'max_steps', 1)
    require(steps <= horizon, 'Requested sign horizon exceeds frozen boundary')
    q = float(q)
    require(math.isfinite(q) and 0 <= q <= 1, 'Sign probability must lie in [0,1]')
    positive, negative = np.zeros(steps+1), np.zeros(steps+1)
    survival = np.zeros(steps+1)
    survival[0] = 1.
    row = np.ones(1)
    absorbed, maximum_error, zero_survival_at = 0., 0., None
    for d in range(1, steps+1):
        fresh = np.zeros(d+1)
        fresh[:-1] += (1-q)*row
        fresh[1:] += q*row
        cutoff = boundaries[d]
        negative[d] = float(fresh[:max(0, d-cutoff+1)].sum())
        positive[d] = float(fresh[min(cutoff, d+1):].sum())
        fresh[:max(0, d-cutoff+1)] = 0.
        fresh[min(cutoff, d+1):] = 0.
        survival[d] = float(fresh.sum())
        absorbed += negative[d]+positive[d]
        maximum_error = max(maximum_error, abs(absorbed+survival[d]-1))
        row = fresh
        if survival[d] == 0:
            zero_survival_at = d
            break
    return {'positive': positive, 'negative': negative, 'survival': survival,
            'computed_sign_horizon': steps, 'boundary_horizon': horizon, 'q': q,
            'boundary_table': tuple(boundaries),
            'diagnostics': {'maximum_mass_balance_error_float': maximum_error,
                            'zero_float_survival_first_step': zero_survival_at,
                            'probabilities_renormalized': False}}


def waiting_kernel(pair_cap, discordance_rate):
    """P(NB(d,r)<=T) and E[min(NB(d,r),T)] for d=1,...,T.

    NB counts physical trials through the d-th discordance, including its
    successful trial. The closed form avoids a quadratic physical-time sum.
    """
    t = exact.integer(pair_cap, 'pair_cap')
    r = float(discordance_rate)
    require(math.isfinite(r) and 0 < r <= 1, 'Waiting kernel requires 0<r<=1; handle r=0 separately')
    d = np.arange(1, t+1, dtype=float)
    if r == 1:
        return {'stop_by_cap': np.ones(t), 'wait_beyond_cap': np.zeros(t), 'truncated_pairs': d,
                'maximum_binomial_complement_error': 0.}
    by_cap = binom.sf(d-1, t, r)
    beyond = binom.cdf(d-1, t, r)
    future_tail = binom.sf(d, t+1, r)
    # Reorder as tail/r to avoid forming d/r at extremely small r.
    truncated = d*(future_tail/r)+t*beyond
    require(np.all(np.isfinite(truncated)) and np.all(truncated >= 0) and
            np.all(truncated <= t*(1+1e-10)+1e-12), 'Floating truncated waiting time is invalid')
    complement_error = float(np.max(np.abs(by_cap+beyond-1))) if t else 0.
    return {'stop_by_cap': by_cap, 'wait_beyond_cap': beyond, 'truncated_pairs': truncated,
            'maximum_binomial_complement_error': complement_error}


def oc_from_distribution(distribution, discordance_rate, pair_cap):
    """Mix independent sign first-crossing and physical waiting distributions."""
    t = exact.integer(pair_cap, 'pair_cap')
    require(t <= distribution['computed_sign_horizon'] <= distribution['boundary_horizon'],
            'Physical pair cap exceeds available sign distribution or frozen boundary')
    r = float(discordance_rate)
    require(math.isfinite(r) and 0 <= r <= 1, 'Invalid discordance rate')
    if r == 0 or t == 0:
        return {'positive_probability': 0., 'negative_probability': 0., 'abstain_probability': 1.,
                'expected_pairs': float(t), 'expected_shots': float(2*t),
                'probability_total': 1., 'maximum_binomial_complement_error': 0.}
    kernel = waiting_kernel(t, r)
    positive = distribution['positive'][1:t+1]
    negative = distribution['negative'][1:t+1]
    both = positive+negative
    p_positive = math.fsum((positive*kernel['stop_by_cap']).tolist())
    p_negative = math.fsum((negative*kernel['stop_by_cap']).tolist())
    # Explicitly retain both kinds of abstention: no sign crossing by d=T,
    # or a latent crossing that needs more than T physical pairs to arrive.
    abstain = distribution['survival'][t]+math.fsum((both*kernel['wait_beyond_cap']).tolist())
    expected = t*distribution['survival'][t]+math.fsum((both*kernel['truncated_pairs']).tolist())
    return {'positive_probability': p_positive, 'negative_probability': p_negative,
            'abstain_probability': float(abstain), 'expected_pairs': float(expected), 'expected_shots': float(2*expected),
            'probability_total': float(p_positive+p_negative+abstain),
            'maximum_binomial_complement_error': kernel['maximum_binomial_complement_error']}


def analyze_case(boundaries, case, pair_cap, distribution=None):
    """Public API for any previously frozen symmetric sign-only integer table."""
    exact.validate_table(boundaries)
    t = exact.integer(pair_cap, 'pair_cap')
    require(t <= len(boundaries)-1, 'Cannot reuse a smaller-horizon boundary at a larger physical cap')
    params = psr_parameters(case)
    q, r = params['conditional_plus_probability'], params['discordance_rate']
    if distribution is None:
        distribution = first_crossing_distribution(boundaries, .5 if q is None else q, max_steps=max(1, t))
    else:
        require(distribution['boundary_table'] == tuple(boundaries) and
                (q is None or distribution['q'] == q), 'A sign distribution for another table or q was supplied')
    result = oc_from_distribution(distribution, r, t)
    if case['C'] > 0:
        correct, wrong = result['positive_probability'], result['negative_probability']
    elif case['C'] < 0:
        correct, wrong = result['negative_probability'], result['positive_probability']
    else:
        correct, wrong = 0., result['positive_probability']+result['negative_probability']
    result.update(params, case_id=case.get('case_id'), pair_cap=t, physical_shot_cap=2*t,
                  correct_probability=correct, error_probability=wrong,
                  numerical_diagnostics=distribution['diagnostics'],
                  monitoring='Every completed physical pair; sign-only boundary; no physical-prefix gate')
    return result


def stopping_probability(distribution, rate, pair_cap, direction):
    t = exact.integer(pair_cap, 'pair_cap')
    require(t <= distribution['computed_sign_horizon'], 'Power cap exceeds sign distribution')
    if t == 0 or rate == 0:
        return 0.
    weights = distribution['positive' if direction > 0 else 'negative'][1:t+1]
    tails = binom.sf(np.arange(t), t, rate)
    return math.fsum((weights*tails).tolist())


def sufficient_cap(distribution, rate, direction, target, maximum_pair_cap):
    """Smallest integer T meeting a FLOAT power target within the valid horizon."""
    maximum = exact.integer(maximum_pair_cap, 'maximum_pair_cap', 1)
    require(maximum <= distribution['computed_sign_horizon'] and 0 < target < 1 and direction in (-1, 1),
            'Invalid power-cap search')
    endpoint = stopping_probability(distribution, rate, maximum, direction)
    if endpoint < target:
        return {'target_correct_probability': target, 'smallest_pair_cap_float': None,
                'smallest_physical_shot_cap_float': None, 'correct_probability_at_maximum': endpoint,
                'maximum_valid_pair_cap': maximum, 'status': 'target_not_reached_with_this_frozen_boundary_horizon'}
    low, high = 0, maximum
    while high-low > 1:
        middle = (low+high)//2
        if stopping_probability(distribution, rate, middle, direction) >= target:
            high = middle
        else:
            low = middle
    return {'target_correct_probability': target, 'smallest_pair_cap_float': high,
            'smallest_physical_shot_cap_float': 2*high,
            'correct_probability_at_selected_cap': stopping_probability(distribution, rate, high, direction),
            'correct_probability_at_previous_cap': stopping_probability(distribution, rate, high-1, direction),
            'maximum_valid_pair_cap': maximum,
            'status': 'floating_power_diagnostic_requires_interval_check_for_a_rigorous_threshold_claim'}


def exact_sign_distribution(table, q):
    positive, negative, survival = [Fraction(0)], [Fraction(0)], [Fraction(1)]
    row = [Fraction(1)]
    for d in range(1, len(table)):
        fresh = [Fraction(0)]*(d+1)
        for b, mass in enumerate(row):
            fresh[b] += (1-q)*mass
            fresh[b+1] += q*mass
        cutoff = table[d]
        negative.append(sum(fresh[:max(0, d-cutoff+1)], Fraction(0)))
        positive.append(sum(fresh[min(cutoff, d+1):], Fraction(0)))
        row = [mass if d-cutoff < b < cutoff else Fraction(0) for b, mass in enumerate(fresh)]
        survival.append(sum(row, Fraction(0)))
    return positive, negative, survival


def exact_binomial_tail(t, r, d):
    return sum((math.comb(t, k)*r**k*(1-r)**(t-k) for k in range(max(0, d), t+1)), Fraction(0))


def exact_mixture_oc(table, p_minus, p_plus, t):
    rate = p_minus*(1-p_plus)+(1-p_minus)*p_plus
    if rate == 0:
        return Fraction(0), Fraction(0), Fraction(1), Fraction(t)
    q = (1-p_minus)*p_plus/rate
    positive, negative, survival = exact_sign_distribution(table, q)
    p_positive, p_negative, abstain, expected = Fraction(0), Fraction(0), survival[t], t*survival[t]
    for d in range(1, t+1):
        tail = exact_binomial_tail(t, rate, d)
        truncated = Fraction(d, 1)/rate*exact_binomial_tail(t+1, rate, d+1)+t*(1-tail)
        direct_wait = sum((1-exact_binomial_tail(n, rate, d) for n in range(t)), Fraction(0))
        require(truncated == direct_wait, 'Closed truncated-NB identity differs from exact survival sum')
        p_positive += positive[d]*tail
        p_negative += negative[d]*tail
        abstain += (positive[d]+negative[d])*(1-tail)
        expected += (positive[d]+negative[d])*truncated
    return p_positive, p_negative, abstain, expected


def exact_physical_enumeration(table, p_minus, p_plus, t):
    probabilities = ((1-p_minus)*(1-p_plus), (1-p_minus)*p_plus, p_minus*(1-p_plus), p_minus*p_plus)
    totals = [Fraction(0), Fraction(0), Fraction(0), Fraction(0)]
    for word in product(range(4), repeat=t):
        probability = Fraction(1)
        plus, minus, decision, paid = 0, 0, 0, t
        for pair, category in enumerate(word, 1):
            probability *= probabilities[category]
            if not decision:
                plus += category == 1
                minus += category == 2
                d = plus+minus
                cutoff = table[d]
                if plus >= cutoff:
                    decision, paid = 1, pair
                elif plus <= d-cutoff:
                    decision, paid = -1, pair
        totals[0 if decision > 0 else 1 if decision < 0 else 2] += probability
        totals[3] += paid*probability
    return tuple(totals)


def self_checks():
    parameters = ((Fraction(1, 4), Fraction(3, 4)), (Fraction(3, 4), Fraction(1, 4)),
                  (Fraction(1, 2), Fraction(1, 2)), (Fraction(0), Fraction(3, 4)),
                  (Fraction(0), Fraction(1)), (Fraction(1), Fraction(0)),
                  (Fraction(0), Fraction(0)), (Fraction(1), Fraction(1)))
    cases, maximum_difference = 0, 0.
    table = exact.constant_boundary(6, Fraction(3))
    for t in (3, 4, 6):
        for p_minus, p_plus in parameters:
            formula = exact_mixture_oc(table, p_minus, p_plus, t)
            enumerated = exact_physical_enumeration(table, p_minus, p_plus, t)
            require(formula == enumerated and sum(formula[:3]) == 1, 'Exact sign/wait mixture differs from full physical enumeration')
            case = {'A': float((p_minus+p_plus)/2), 'C': float((p_plus-p_minus)/2)}
            floating = analyze_case(table, case, t)
            values = tuple(floating[key] for key in ('positive_probability', 'negative_probability', 'abstain_probability', 'expected_pairs'))
            maximum_difference = max(maximum_difference, *(abs(value-float(reference)) for value, reference in zip(values, formula)))
            require(all(abs(value-float(reference)) < 2e-13 for value, reference in zip(values, formula)), 'Float operating characteristics fail exact small-case check')
            cases += 1
    # Nonconstant symmetric table exercises the generic future SOTA interface.
    arbitrary = (1, 2, 2, 4, 4, 6, 6)
    require(exact_mixture_oc(arbitrary, Fraction(1, 4), Fraction(3, 4), 6) ==
            exact_physical_enumeration(arbitrary, Fraction(1, 4), Fraction(3, 4), 6), 'Nonconstant boundary identity failed')
    rejected = False
    try:
        analyze_case(table, {'A': .5, 'C': .1}, 7)
    except ValueError:
        rejected = True
    require(rejected, 'A boundary was reused beyond its registered horizon')
    return {'status': 'passed', 'exact_physical_enumeration_cases': cases+1,
            'maximum_float_difference_against_small_exact_answers': maximum_difference,
            'r_zero_r_one_q_zero_q_one_and_d_equal_T_checked': True,
            'wrong_cap_boundary_reuse_rejected': True, 'new_observations': 0,
            'scope': 'Finite exact checks validate implementation identities; float production numbers still require independent error enclosures.'}


def analyze_existing_beta_boundaries():
    plan = json.loads(CASE_PATH.read_text(encoding='utf-8'))
    cases = plan['cases']
    outputs = []
    for path in BETA_PATHS:
        start = time.perf_counter()
        artifact = json.loads(path.read_text(encoding='utf-8'))
        exact.verify_artifact(artifact)
        table, h = artifact['boundaries'], artifact['max_discordances']
        cache, rows, cap_diagnostics = {}, [], []
        for case in cases:
            params = psr_parameters(case)
            q, r = params['conditional_plus_probability'], params['discordance_rate']
            key = .5 if q is None else q
            if key not in cache:
                cache[key] = first_crossing_distribution(table, key)
            distribution = cache[key]
            rows.append(analyze_case(table, case, h, distribution))
            if case['C']:
                cap_diagnostics.append({'case_id': case['case_id'], 'targets':
                    [sufficient_cap(distribution, r, 1 if case['C'] > 0 else -1, target, h) for target in (.95, .99)]})
        main = [row for row, case in zip(rows, cases) if case['kind'] == 'main']
        null = [row for row, case in zip(rows, cases) if case['kind'] == 'null']
        outputs.append({'boundary_path': path.relative_to(ROOT).as_posix(), 'file_sha256': digest(path),
                        'artifact_sha256': artifact['artifact_sha256'], 'boundary_horizon': h, 'physical_shot_cap': 2*h,
                        'beta_threshold_exact': artifact['threshold_exact'], 'case_results': rows,
                        'equal_case_main_expected_shots': math.fsum(row['expected_shots'] for row in main)/len(main),
                        'equal_case_main_correct_probability': math.fsum(row['correct_probability'] for row in main)/len(main),
                        'worst_main_correct_probability': min(row['correct_probability'] for row in main),
                        'largest_null_false_stop_probability': max(row['error_probability'] for row in null),
                        'minimum_caps_for_target_power_float': cap_diagnostics,
                        'elapsed_seconds': time.perf_counter()-start})
    inputs = [CASE_PATH, *BETA_PATHS, *AUDIT_PATHS]
    return {'status': 'deterministic_float_operating_characteristics_not_interval_certified',
            'monitoring_protocol': 'Inspect the frozen sign-only boundary after EVERY physical pair; no sparse checkpoints or prefix gates.',
            'new_observations': 0, 'results': outputs,
            'formulas': {'stop_by_T': 'sum_d first_cross_direction[d] * P(Bin(T,r)>=d)',
                         'truncated_NB': '(d/r)*P(Bin(T+1,r)>=d+1)+T*P(Bin(T,r)<d)',
                         'expected_pairs': 'sum_{d=1}^T first_cross_total[d]*truncated_NB(d,r,T)+survival_sign[T]*T'},
            'boundary_scope': 'Each cap equals its separately certified boundary horizon in pairs. Power-cap search only shortens that same horizon; larger caps are rejected.',
            'precision_scope': 'Biased-sign DP and SciPy binomial tails use float arithmetic. No Monte Carlo confidence intervals or rigorous numerical error claims. Independent outward-error module is separate.',
            'runtime': {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__},
            'input_sha256': {path.relative_to(ROOT).as_posix(): digest(path) for path in inputs},
            'source_hashes': {**exact.source_hashes(), Path(__file__).resolve().relative_to(ROOT).as_posix(): digest(__file__)}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/sota_round4/analytic_beta_costs.json')
    parser.add_argument('--checks-output', type=Path, default=ROOT/'results/sota_round4/analytic_sign_cost_checks.json')
    args = parser.parse_args()
    checks = self_checks()
    checks['source_sha256'] = digest(__file__)
    write_new(args.checks_output, checks)
    print(json.dumps({'checks': checks['status'], 'small_exact_cases': checks['exact_physical_enumeration_cases']}), flush=True)
    result = analyze_existing_beta_boundaries()
    write_new(args.output, result)
    print(json.dumps({'status': result['status'], 'output': str(args.output), 'summaries':
        [{key: item[key] for key in ('boundary_horizon', 'equal_case_main_expected_shots',
                                    'equal_case_main_correct_probability', 'worst_main_correct_probability',
                                    'largest_null_false_stop_probability', 'elapsed_seconds')} for item in result['results']]}))
