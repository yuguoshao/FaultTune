"""Audit the released small development benchmark without new observations.

The ledger, RNG calls, DKM learner/optimism/tracking, prior-table arithmetic,
stopping prefixes and summaries are reconstructed here. Numerical projection
optimality is not independently solved: logged best responses are checked by
physical feasibility and a global convex gap. Original-runner seed replays of
every case's repetitions 0 and 3 are separately counted, not called independent
controller recomputation. Frozen interval/certificate primitives are shared.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import gammaln, logsumexp, rel_entr, xlogy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/sota_round2/benchmark'))
sys.path.insert(0, str(ROOT/'experiments/structured_validation'))
import benchmark_runner as runner
import audit_refined as old_audit
import certified_likelihood as likelihood
import weighted_fisher as fisher
from direct_tests import _Bounds, _log_marginal

DIRECTORY = ROOT/'results/sota_round2/benchmark'
OUTPUT = ROOT/'results/sota_round2/benchmark_audit'
METHOD_PATHS = ['dkm_one_shot', 'psr_six_arm_init', 'closed_hedge_original', 'closed_hedge_fine_initial']
CALIBRATIONS = ['kk21', 'kt_safe', 'structured_grid_safe']
COUNTERS = Counter()
MAXIMUM = {}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def close(left, right, name, atol=2e-9, rtol=2e-11):
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    require(a.shape == b.shape and np.all(np.isfinite(a)) and np.all(np.isfinite(b)), name+' shape/nonfinite')
    error = float(np.max(np.abs(a-b))) if a.size else 0.
    MAXIMUM[name] = max(MAXIMUM.get(name, 0.), error)
    require(np.all(np.abs(a-b) <= atol+rtol*np.abs(b)), name+' differs: '+str(error))


def original_schedule(cap):
    result, n = [], 32
    while n < cap:
        result.append(n)
        n = 2*int(math.ceil(n*1.125/2))
    return sorted(set(result+[cap]))


def inspect_release(full):
    spec = read(DIRECTORY/'spec.json')
    plan, release = spec['plan'], read(DIRECTORY/'release.json')
    require(release['status'] == 'released_by_root' and release['explicit_root_release'] is True,
            'Explicit root release absent')
    require(release['reference'] and release['sampling_cutoff_unix'] <= release['hard_deadline_unix'], 'Invalid release')
    require(spec['plan_reference'] == {'path': 'results/sota_round2/benchmark/preliminary_plan.json',
            'sha256': digest(DIRECTORY/'preliminary_plan.json')}, 'Plan binding differs')
    require(read(DIRECTORY/'preliminary_plan.json') == plan and
            release['plan_sha256'] == spec['plan_reference']['sha256'], 'Released plan differs')
    require(spec['release_reference'] == {'path': 'results/sota_round2/benchmark/release.json',
            'sha256': digest(DIRECTORY/'release.json')}, 'Release binding differs')
    require(spec['sources'] == release['sources'], 'Release/spec source map differs')
    for name, expected in release['sources'].items():
        require(digest(ROOT/name) == expected, 'Frozen source changed: '+name)
    require(release['sources']['experiments/sota_round2/literature/dkm_baseline.py'] ==
            'e8799facf1494dcf6355f2f2e6069be350f35f788a70381201f55c53e5479c9d', 'Unexpected DKM source')
    require(release['preflight'] == plan['preflight'], 'Preflight differs')
    for binding in list(plan['preflight'].values())+[release['review']]:
        require(digest(ROOT/binding['path']) == binding['sha256'], 'Review artifact changed')
        detail = read(ROOT/binding['path'])
        require(detail['status'] == 'passed', 'Review did not pass')
        for name, expected in detail.get('sources', {}).items():
            require(digest(ROOT/name) == expected, 'Review-bound source changed: '+name)
    review = read(ROOT/release['review']['path'])
    require(all(review['sources'].get(name) == expected for name, expected in release['sources'].items()),
            'Runner review does not cover release')
    require((plan['cap'], plan['delta'], plan['repetitions'], plan['seed_base']) == (4096, .01, 4, 220411000),
            'Fixed experiment budgets differ')
    require([c['case_id'] for c in plan['cases']] == [7, 9, 19, 21, 31, 33, 43, 45, 100, 101], 'Case matrix differs')
    old = plan['old_development_spec']
    require(digest(ROOT/old['path']) == old['sha256'], 'Original development specification changed')
    lookup = {c['case_id']: c for c in read(ROOT/old['path'])['cases']}
    require(plan['cases'][:8] == [lookup[i] for i in (7, 9, 19, 21, 31, 33, 43, 45)], 'Main cases changed')
    require(plan['cases'][8:] == [{'case_id': 100, 'kind': 'null', 'A': .01, 'B': 0., 'C': 0.},
                                {'case_id': 101, 'kind': 'null', 'A': .5, 'B': 0., 'C': 0.}], 'Null cases changed')
    require(plan['paths'] == METHOD_PATHS and [c['name'] for c in plan['calibrations']] == CALIBRATIONS,
            'Method registry differs')
    require((plan['expected_joint_labels'], plan['expected_path_records'], plan['expected_method_records']) ==
            (40, 160, 320), 'Expected record count differs')
    coarse = original_schedule(4096)
    fine = sorted(set(range(2, 33, 2)) | {n for n in coarse if n > 32} | {4096})
    require(plan['hedge_original_checkpoints'] == coarse and plan['hedge_fine_checkpoints'] == fine and
            plan['single_arm_checkpoints'] == fine, 'Checkpoint schedule differs')
    normalized = []
    for arm in plan['arms']:
        c, s, d = [arm[key] for key in ('cos_numerator', 'sin_numerator', 'denominator')]
        require(all(type(v) is int for v in (c, s, d)) and d > 0 and c*c+s*s == d*d, 'Non-rational arm')
        normalized.append((Fraction(c, d), Fraction(s, d)))
    require(normalized == [(Fraction(3, 5), Fraction(-4, 5)), (Fraction(3, 5), Fraction(4, 5)),
            (Fraction(0), Fraction(-1)), (Fraction(0), Fraction(1)),
            (Fraction(-3, 5), Fraction(-4, 5)), (Fraction(-3, 5), Fraction(4, 5))], 'Fixed arm set/order differs')
    if full:
        summary = read(DIRECTORY/'summary.json')
        require(summary['status'] == 'complete_development_diagnostic_no_inferential_ci', 'Producer incomplete')
        require(summary['sources'] == release['sources'], 'Summary source binding differs')
        require(summary['input_sha256'] == {name: digest(DIRECTORY/name) for name in
                ('spec.json', 'paths.jsonl', 'method_results.jsonl')}, 'Summary input hashes differ')
    return spec, plan, release


class PriorAudit:
    def __init__(self, plan):
        item = plan['calibrations'][2]
        binding = item['prior']
        artifact = read(ROOT/binding['path'])
        require(digest(ROOT/binding['path']) == binding['sha256'], 'Actual prior artifact changed')
        table = artifact['models']
        table_hash = hashlib.sha256(json.dumps(table, separators=(',', ':')).encode()).hexdigest()
        require(table_hash == binding['grid_sha256'] == artifact['grid_sha256'], 'Actual model table hash differs')
        require(artifact['parameter_order'] == binding['parameter_order'] == ['A', 'B', 'C', 'weight'], 'Prior columns differ')
        require(len(table) == binding['model_count'] == 7141, 'Prior size differs')
        self.models = []
        for row in table:
            require(len(row) == 4 and all(len(pair) == 2 and all(type(i) is int for i in pair) and pair[1] > 0
                                        for pair in row), 'Invalid rational prior entry')
            a, b, c, w = [Fraction(*pair) for pair in row]
            require(0 < a < 1 and w > 0 and b*b+c*c < min(a, 1-a)**2, 'Nonphysical/nonpositive prior')
            self.models.append((a, b, c, w))
        require(len(set(row[:3] for row in self.models)) == 7141, 'Duplicate prior model')
        require(sum((row[3] for row in self.models), Fraction(0)) == 1, 'Prior weights do not sum exactly to one')
        require(artifact['source_sha256'] == digest(ROOT/item['module_path']), 'Prior/provider binding differs')
        self.protocol, self.grid_hash = artifact['protocol'], table_hash
        self.arms = [(Fraction(a['cos_numerator'], a['denominator']), Fraction(a['sin_numerator'], a['denominator']))
                     for a in plan['arms']]
        probabilities = []
        for co, si in self.arms:
            row = [a+b*co+c*si for a, b, c, w in self.models]
            require(all(0 < p < 1 for p in row), 'Prior probability at endpoint')
            probabilities.append([float(p) for p in row])
        p = np.array(probabilities)
        self.lp, self.lf = np.log(p), np.log1p(-p)
        self.lw = np.array([math.log(float(row[3])) for row in self.models])
        self.cache = {}
        self.report = {'model_count': 7141, 'table_sha256': table_hash, 'artifact_sha256': binding['sha256'],
                       'weights_exactly_sum_to_one': True, 'all_models_strictly_physical': True,
                       'actual_table_used_for_all_numerators': True, 'table_independent_of_libm_regeneration': True}

    def scores(self, n, k):
        return np.dot(k, self.lp)+np.dot(n-k, self.lf)+self.lw

    def screen(self, n, k):
        return float(np.logaddexp(float(logsumexp(self.scores(n, k))), kt_screen(n, k))-math.log(2))

    def partial(self, n, k):
        key = (tuple(n), tuple(k))
        if key in self.cache:
            return self.cache[key]
        scores = self.scores(n, k)
        selected = sorted(range(len(self.models)), key=lambda j: (-scores[j], j))[:64]
        logs, mass = [], Fraction(0)
        for index in selected:
            a, b, c, weight = self.models[index]
            value = rational_bound(weight).log()
            for (co, si), count, successes in zip(self.arms, n, k):
                p = a+b*co+c*si
                if successes:
                    value += int(successes)*rational_bound(p).log()
                if count-successes:
                    value += int(count-successes)*rational_bound(1-p).log()
            logs.append(value)
            mass += weight
        offset = max(value.hi for value in logs)
        grid = _Bounds(offset)+sum(((value-_Bounds(offset)).exp() for value in logs), _Bounds(0)).log()
        kt = kt_bounds(n, k)
        offset = max(grid.hi, kt.hi)
        mixed = _Bounds(offset)+(((grid-_Bounds(offset)).exp()+(kt-_Bounds(offset)).exp())/2).log()
        result = {'log_lower_decimal': str(mixed.lo), 'partial_numerator_log_bounds': [str(mixed.lo), str(mixed.hi)],
                  'retained_model_indices': selected, 'retained_prior_mass': [mass.numerator, mass.denominator],
                  'grid_partial_log_bounds': [str(grid.lo), str(grid.hi)], 'kt_log_bounds': [str(kt.lo), str(kt.hi)]}
        self.cache[key] = result
        return result


def rational_bound(value):
    return _Bounds(value.numerator)/value.denominator


def kt_screen(n, k):
    return float((gammaln(k+.5)+gammaln(n-k+.5)-gammaln(n+1)-2*gammaln(.5)).sum())


def kt_bounds(n, k):
    return sum((_log_marginal(int(success), int(count), .5) for count, success in zip(n, k)), _Bounds(0))


def saturated_bounds(n, k):
    result = _Bounds(0)
    for count, successes in zip(n, k):
        count, successes = int(count), int(successes)
        if successes:
            result += successes*(_Bounds(successes)/count).log()
        if count-successes:
            result += (count-successes)*(_Bounds(count-successes)/count).log()
    return result


def inspect_certificate(cert, calibration, arms, n, k, prior, delta):
    sign = cert['sign']
    require(type(sign) is int and sign in (-1, 1) and cert['shots'] == int(n.sum()) and
            cert['delta'] == delta and cert['calibration'] == calibration, 'Certificate identity/count differs')
    rows = [dict(arm, trials=int(count), successes=int(success)) for arm, count, success in zip(arms, n, k)]
    # The optimizer is only a proposal; the shared reviewed primitive proves a
    # global upper from a concave tangent and the entire physical support.
    null = likelihood.null_loglik_upper(rows, sign, candidate=cert['null_certificate']['candidate'])
    require(null == cert['null_certificate'], 'Recomputed global null certificate differs')
    if calibration == 'kk21':
        numerator = saturated_bounds(n, k)
        threshold = runner.dkm.kk_threshold_upper(n, delta)
    elif calibration == 'kt_safe':
        numerator = kt_bounds(n, k)
        threshold = (-_Bounds(Decimal(str(delta))).log()).hi
    else:
        expected = prior.partial(n, k)
        detail = cert['numerator_certificate']
        require(all(detail.get(key) == value for key, value in expected.items()), 'Actual-prior numerator bounds differ')
        require(cert['grid_sha256'] == prior.grid_hash and cert['protocol'] == prior.protocol, 'Grid certificate prior differs')
        require(detail['bounds_scope'] == 'Bounds enclose the retained-grid-plus-KT partial numerator; only its lower endpoint bounds the FULL numerator.',
                'Partial numerator scope differs')
        numerator = _Bounds(Decimal(expected['log_lower_decimal']))
        threshold = (-_Bounds(Decimal(str(delta))).log()).hi
        COUNTERS['grid_partial_numerators_checked'] += 1
    if calibration != 'structured_grid_safe':
        require(cert['numerator_bounds'] == [str(numerator.lo), str(numerator.hi)], 'Built-in numerator bounds differ')
    evidence = (numerator-_Bounds(Decimal(null['loglik_upper_decimal']))).lo
    require(cert['evidence_lower_decimal'] == str(evidence) and cert['threshold_upper_decimal'] == str(threshold),
            'Outward evidence/threshold differs')
    decision = ('reject' if sign > 0 else 'accept') if evidence > threshold else 'abstain'
    require(cert['decision'] == decision and cert['anytime_valid_under_stated_model'] is True, 'Certified decision differs')
    COUNTERS['single_arm_certificates_recomputed'] += 1


def kl(p, q):
    # Independently assembled Bernoulli divergence, with stable endpoint terms.
    return np.maximum(0., rel_entr(p, q)+rel_entr(1-p, 1-q))


def physical(point, tolerance=1e-8):
    a, b, c = map(float, point)
    return all(math.isfinite(x) for x in (a, b, c)) and math.hypot(b, c) <= min(a, 1-a)+tolerance


class DKMReconstruction:
    def __init__(self, arms):
        self.x = np.array([[1., a['cos_numerator']/a['denominator'], a['sin_numerator']/a['denominator']] for a in arms])
        self.loss = {s: np.zeros(6) for s in (-1, 1)}
        self.gap = {-1: 0., 1: 0.}
        self.target = np.ones(6)
        self.label = 1

    def check(self, proposal, n, k):
        total = int(n.sum())
        require(proposal['batch_size'] == 1 and proposal['observed_shots'] == total, 'DKM paid prefix differs')
        if total < 6:
            require(proposal == {'arm': total, 'batch_size': 1, 'stage': 'paid_one_pull_per_arm_initialization',
                                 'observed_shots': total}, 'DKM initialization differs')
            COUNTERS['dkm_initialization_actions_checked'] += 1
            return
        require(proposal['stage'] == 'dkm_game' and proposal['design_margin'] == 1e-6 and
                proposal['schedule'] == 'per_observation_DKM_schedule', 'DKM schedule/margin differs')
        margin, factor = 1e-6, 1-2e-6
        means = np.clip(k/n, margin, 1-margin)
        argument = 1.1*1.1*math.log(total)
        bonus = argument+math.log(argument)+.5
        close(proposal['exploration_bonus'], bonus, 'dkm_exploration_bonus')
        interval = np.asarray(proposal['confidence_intervals'], dtype=float)
        require(interval.shape == (6, 2), 'DKM interval shape')
        lower, upper = interval.T
        require(np.all(lower >= margin) and np.all(upper <= 1-margin) and
                np.all(lower <= means) and np.all(means <= upper), 'DKM intervals do not contain empirical means')
        for endpoint, edge in ((lower, margin), (upper, 1-margin)):
            radius = bonus/n
            difference = kl(means, endpoint)-radius
            derivative = np.abs((endpoint-means)/(endpoint*(1-endpoint)))
            tolerance = 2e-9+2e-13*derivative
            interior = endpoint != edge
            require(np.all(np.abs(difference[interior]) <= tolerance[interior]) and
                    np.all(difference[~interior] <= tolerance[~interior]), 'KL confidence root/edge equation differs')
        fitted = np.asarray(proposal['fitted_theta'], dtype=float)
        raw_fit = (fitted-np.array([margin, 0., 0.]))/factor
        require(physical(raw_fit), 'Logged model projection is not physical')
        if proposal['model_projection_success']:
            fitted_probabilities = self.x.dot(fitted)
            require(np.all(fitted_probabilities >= lower-1.1e-8) and np.all(fitted_probabilities <= upper+1.1e-8),
                    'Successful projection lies outside its confidence box')
            self.label = 1 if fitted[2] >= 0 else -1
            COUNTERS['dkm_successful_projection_feasibility_checks'] += 1
        else:
            COUNTERS['dkm_projection_failure_previous_label_checks'] += 1
        require(proposal['selected_sign'] == self.label, 'DKM label update/stale label differs')
        loss, gap = self.loss[self.label], self.gap[self.label]
        if gap == 0:
            minima = loss == loss.min()
            weights, rate = minima.astype(float)/minima.sum(), math.inf
        else:
            rate = math.log(6)/gap
            logits = -rate*(loss-loss.min())
            weights = np.exp(logits-logsumexp(logits))
        close(proposal['allocation'], weights, 'dkm_adah_edge_allocation', atol=2e-8)
        response = np.asarray(proposal['best_response_theta'], dtype=float)
        point = (response-np.array([margin, 0., 0.]))/factor
        require(physical(point, 2e-10) and self.label*point[2] <= 2e-10 and
                proposal['best_response_success'] is True, 'Infeasible/failed opposite-model response')
        probabilities = margin+factor*self.x.dot(point)
        require(np.all(probabilities >= margin-1e-12) and np.all(probabilities <= 1-margin+1e-12),
                'Best-response model probability outside design domain')
        gradient = factor*self.x.T.dot(weights*(probabilities-means)/(probabilities*(1-probabilities)))
        direction = -gradient
        support = max(0., float(direction[0]),
                      .5*(float(direction[0])+math.hypot(float(direction[1]), max(0., -self.label*float(direction[2])))))
        oracle_gap = max(0., float(gradient.dot(point))+support)
        MAXIMUM['dkm_independent_convex_gap'] = max(MAXIMUM.get('dkm_independent_convex_gap', 0.), oracle_gap)
        require(oracle_gap <= 1.001e-5, 'Independent global best-response convex gap exceeds 1e-5 tolerance')
        close(proposal['best_response_gap_float_diagnostic'], oracle_gap, 'dkm_logged_convex_gap', atol=5e-9)
        optimistic = np.maximum(bonus/n, np.maximum(kl(lower, probabilities), kl(upper, probabilities)))
        close(proposal['optimistic_payoffs'], optimistic, 'dkm_optimistic_payoffs', atol=2e-8)
        losses = -optimistic
        average = float(weights.dot(losses))
        if math.isinf(rate):
            mixed = float(losses.min())
        else:
            centered, previous = losses-losses.min(), loss-loss.min()
            mixed = float(losses.min()-(logsumexp(-rate*(previous+centered))-logsumexp(-rate*previous))/rate)
        self.gap[self.label] += max(0., average-mixed)
        self.loss[self.label] += losses
        self.loss[self.label] -= self.loss[self.label].min()
        self.target += weights
        logged_target = np.asarray(proposal['tracking_target'], dtype=float)
        close(logged_target, self.target, 'dkm_ratio_tracking_target', atol=2e-6)
        selected = proposal['arm']
        require(selected == int(np.argmin(n/logged_target)), 'Logged arm is not ratio-tracking argmin')
        ratio = n/self.target
        require(ratio[selected]-float(ratio.min()) <= 2e-8, 'Independent cumulative-ratio arm differs')
        if selected != int(np.argmin(ratio)):
            COUNTERS['dkm_tracking_roundoff_ties'] += 1
        COUNTERS['dkm_game_actions_checked'] += 1


def margin(cert):
    return Decimal(cert['evidence_lower_decimal'])-Decimal(cert['threshold_upper_decimal'])


def inspect_checkpoint(check, active, n, k, arms, prior, delta, cap):
    total = int(n.sum())
    require(type(check['shots']) is int and all(type(value) is int for value in check['trials']+check['successes']),
            'Checkpoint contains noninteger counts')
    require(check['shots'] == total and check['trials'] == n.tolist() and check['successes'] == k.tolist(),
            'Checkpoint counts differ')
    require(set(check['calibrations']) == set(active), 'Missing/extra active calibration')
    rows = [dict(arm, trials=int(count), successes=int(success)) for arm, count, success in zip(arms, n, k)]
    fits = {sign: likelihood.null_fit(rows, sign) for sign in (-1, 1)}
    completed = {}
    for calibration in active:
        detail = check['calibrations'][calibration]
        if calibration == 'kk21':
            mean = k/n
            numerator = float((xlogy(k, mean)+xlogy(n-k, 1-mean)).sum())
            threshold = float(runner.dkm.kk_calibration_upper(str(delta), len(n)))+float((3*np.log1p(np.log(n))).sum())
        else:
            numerator = kt_screen(n, k) if calibration == 'kt_safe' else prior.screen(n, k)
            threshold = math.log(1/delta)
        close(detail['numerator_float'], numerator, 'checkpoint_'+calibration+'_numerator')
        close(detail['threshold_float'], threshold, 'checkpoint_'+calibration+'_threshold')
        for sign in (-1, 1):
            close(detail['null_loglik_proposals'][str(sign)], fits[sign]['loglik'], 'checkpoint_null_loglik')
        eligible = [s for s in (-1, 1) if total == cap or numerator-fits[s]['loglik'] >= threshold-1e-6]
        require(detail['eligible_signs'] == eligible and [c['sign'] for c in detail['certificates']] == eligible,
                'Eligible sign screening differs')
        for cert in detail['certificates']:
            require(cert['null_certificate']['candidate'] == fits[cert['sign']]['candidate'],
                    'Certified null proposal differs from checkpoint proposal')
            inspect_certificate(cert, calibration, arms, n, k, prior, delta)
        crossings = [c for c in detail['certificates'] if c['decision'] != 'abstain']
        if crossings:
            completed[calibration] = max(crossings, key=lambda c: (margin(c), c['sign']))
        elif total == cap:
            require(len(detail['certificates']) == 2, 'Cap lacks both sign certificates')
            completed[calibration] = max(detail['certificates'], key=lambda c: (margin(c), c['sign']))
        COUNTERS['single_arm_calibration_screens_checked'] += 1
    return completed


def inspect_single(record, case, plan, prior):
    arms = plan['arms']
    require(record['arms'] == arms, 'Record arm registry differs')
    require(all(type(value) is int for value in record['trials']+record['successes']), 'Noninteger terminal arm counts')
    n, k = np.zeros(6, dtype=np.int64), np.zeros(6, dtype=np.int64)
    x = np.array([[1., a['cos_numerator']/a['denominator'], a['sin_numerator']/a['denominator']] for a in arms])
    means = x.dot(np.array([case['A'], case['B'], case['C']]))
    rng = np.random.RandomState(record['seed'])
    controller = DKMReconstruction(arms) if record['path'] == 'dkm_one_shot' else None
    checks = iter(record['checkpoint_history'])
    stopped = {}
    for index, observation in enumerate(record['history']):
        total, proposal = int(n.sum()), observation['proposal']
        require(all(type(proposal[name]) is int for name in ('arm', 'batch_size', 'observed_shots')) and
                proposal['batch_size'] == 1 and proposal['observed_shots'] == total, 'Invalid observed action prefix')
        if controller is not None:
            controller.check(proposal, n, k)
        else:
            expected_arm = total if total < 6 else (2 if (total-6) % 2 == 0 else 3)
            expected = {'arm': expected_arm, 'batch_size': 1, 'observed_shots': total,
                        'stage': 'paid_one_pull_per_arm_initialization' if total < 6 else 'fixed_psr_alternation'}
            require(proposal == expected, 'PSR controller differs')
            COUNTERS['psr_actions_checked'] += 1
        arm = proposal['arm']
        require(all(type(observation[name]) is int for name in ('arm', 'trials', 'successes', 'total_shots')) and
                observation['successes'] in (0, 1) and
                observation['arm'] == arm and observation['trials'] == 1 and observation['total_shots'] == total+1,
                'Invalid single-observation ledger row')
        fresh = int(rng.binomial(1, means[arm]))
        require(fresh == observation['successes'], 'Same-seed single observation differs')
        n[arm] += 1
        k[arm] += fresh
        COUNTERS['rng_binomial_calls_checked'] += 1
        total += 1
        if total in plan['single_arm_checkpoints'] and np.all(n > 0):
            check = next(checks, None)
            require(check is not None, 'Missing eligible checkpoint')
            active = [name for name in CALIBRATIONS if name not in stopped]
            completed = inspect_checkpoint(check, active, n, k, arms, prior, plan['delta'], plan['cap'])
            for name, cert in completed.items():
                stopped[name] = {'shots': total, 'decision': cert['decision'], 'certificate': cert,
                                 'history_prefix_length': index+1, 'trials': n.tolist(), 'successes': k.tolist()}
        require(len(stopped) < 3 or index == len(record['history'])-1, 'Path continued after every method stopped')
    require(next(checks, None) is None, 'Extra checkpoint')
    require(set(stopped) == set(CALIBRATIONS) and record['outcomes'] == stopped, 'First stop/cap prefixes differ')
    require(record['trials'] == n.tolist() and record['successes'] == k.tolist() and
            record['shots_observed'] == int(n.sum()) == len(record['history']), 'Final single-arm counts differ')
    COUNTERS['terminal_certificates_checked'] += len(stopped)


def inspect_hedge(record, case, plan):
    schedule = plan['hedge_original_checkpoints' if record['path'] == 'closed_hedge_original' else 'hedge_fine_checkpoints']
    rng = np.random.RandomState(record['seed'])
    counts = np.zeros(4, dtype=np.int64)
    experts = fisher.protocol_config()['experts']
    cgf = np.zeros(len(experts))
    log_odds = np.array([math.log(e['numerator']/e['denominator']) for e in experts])
    log_weights = np.array([math.log(float(Fraction(*e['weight']))) for e in experts])
    score, last, blocks = Fraction(0), 0, []
    require(len(record['history']) == len(record['checkpoint_history']), 'Hedge block/checkpoint count differs')
    stopped = {}
    for index, (block, check) in enumerate(zip(record['history'], record['checkpoint_history'])):
        budget, m = schedule[index], (schedule[index]-last)//2
        require(all(type(block[name]) is int for name in ('look', 'budget', 'm', 'minus_successes', 'plus_successes')) and
                len(block['pair_counts']) == 4 and all(type(value) is int and value >= 0 for value in block['pair_counts']) and
                type(check['shots']) is int, 'Noninteger/negative paired ledger counts')
        require(block['look'] == index+1 and block['budget'] == budget and block['phase'] == 'psr' and
                block['m'] == m and block['arm'] == {'cos_numerator': 0, 'sin_numerator': 2, 'denominator': 2},
                'Hedge paid PSR action/checkpoint differs')
        pm, pp = case['A']-case['C'], case['A']+case['C']
        old = int(rng.binomial(m, pm))
        both = int(rng.binomial(old, pp))
        worse = int(rng.binomial(m-old, pp))
        fresh = [m-old-worse, worse, old-both, both]
        require(block['pair_counts'] == fresh and block['minus_successes'] == old and
                block['plus_successes'] == worse+both, 'Same-seed paired binomial ledger differs')
        COUNTERS['rng_binomial_calls_checked'] += 3
        counts += fresh
        plus, minus = worse+both, old
        score += Fraction(plus-minus, 2)
        canonical = {'nminus': m, 'nplus': m, 'minus_successes': minus, 'plus_successes': plus}
        blocks.append(canonical)
        for j, e in enumerate(experts):
            # Shared conditional-CGF primitive, independently accumulated state.
            cgf[j] += fisher._cgf(canonical, e['numerator'], e['denominator'])
        b, c = counts[1:3]
        beta = float(gammaln(b+.5)+gammaln(c+.5)-gammaln(b+c+1)-2*gammaln(.5)+(b+c)*math.log(2))
        fl = float(logsumexp(float(score)*log_odds-cgf+log_weights))
        screen = float(np.logaddexp(beta, fl)-math.log(2))
        require(check['shots'] == budget == 2*int(counts.sum()), 'Hedge checkpoint paid total differs')
        close(check['beta_log_float'], beta, 'hedge_beta_screen')
        close(check['fisher_log_float'], fl, 'hedge_fisher_screen')
        close(check['hedge_log_float'], screen, 'hedge_mixture_screen')
        eligible = screen >= math.log(1/plan['delta'])-1e-6 or budget == plan['cap']
        require(('certificate' in check) == eligible, 'Hedge eligibility differs')
        if eligible:
            cert = old_audit.audit_certificate('hedge', counts, {'blocks': blocks}, None, None,
                {'delta': plan['delta'], 'conditional_protocol': fisher.protocol_config()})
            require(cert == check['certificate'], 'Independent hedge certificate wrapper differs')
            COUNTERS['hedge_certificates_recomputed'] += 1
            if cert['decision'] != 'abstain' or budget == plan['cap']:
                stopped['hedge'] = {'shots': budget, 'decision': cert['decision'], 'certificate': cert,
                    'history_prefix_length': index+1, 'counts': counts.tolist(), 'exploration_pairs': 0}
        require(not stopped or index == len(record['history'])-1, 'Hedge continued after first certified stop')
        last = budget
        COUNTERS['hedge_blocks_and_screens_checked'] += 1
    require(bool(stopped) and record['outcomes'] == stopped, 'Hedge first-stop outcome differs')
    require(all(type(value) is int for value in record['counts']) and record['counts'] == counts.tolist() and
            record['shots_observed'] == 2*int(counts.sum()), 'Final hedge counts differ')
    COUNTERS['terminal_certificates_checked'] += 1


def method_rows(record):
    rows = []
    for calibration, outcome in record['outcomes'].items():
        method = record['path'] if calibration == 'hedge' else record['path']+'__'+calibration
        rows.append({'case_id': record['case_id'], 'kind': record['kind'], 'rep': record['rep'], 'seed': record['seed'],
            'path': record['path'], 'calibration': calibration, 'method': method, 'shots': outcome['shots'],
            'decision': outcome['decision'], 'history_prefix_length': outcome['history_prefix_length'],
            'certificate': outcome['certificate']})
    return rows


def inspect_record(record, key, case, plan, prior):
    case_id, rep, path = key
    require((record['case_id'], record['rep'], record['path']) == key and record['kind'] == case['kind'], 'Path key/order differs')
    require(record['status'] == 'complete' and record['error'] is None and record['truth_access'] == 'simulator_only',
            'Incomplete/errored path retained')
    require(record['seed'] == 220411000+1009*case_id+1000003*rep and record['cap'] == 4096 and record['delta'] == .01,
            'Path seed/budget differs')
    require(type(record['shots_observed']) is int and 0 < record['shots_observed'] <= 4096, 'Invalid physically paid shots')
    require(all(math.isfinite(value) and value >= 0 for value in record['timings'].values()), 'Invalid diagnostic timing')
    if path in METHOD_PATHS[:2]:
        inspect_single(record, case, plan, prior)
    else:
        inspect_hedge(record, case, plan)
    rows = method_rows(record)
    for row in rows:
        require(type(row['shots']) is int and type(row['history_prefix_length']) is int and
                0 < row['history_prefix_length'] <= len(record['history']) and
                0 < row['shots'] <= 4096 and row['decision'] in ('reject', 'accept', 'abstain') and
                (row['decision'] != 'abstain' or row['shots'] == 4096), 'Invalid unconditional method cost/outcome')
    COUNTERS['paths_checked'] += 1
    COUNTERS['method_records_checked'] += len(rows)
    COUNTERS['physically_generated_shots_accounted'] += record['shots_observed']
    return rows


def inspect_summary(rows, plan):
    summary = read(DIRECTORY/'summary.json')
    methods = [path+'__'+cal for path in METHOD_PATHS[:2] for cal in CALIBRATIONS]+METHOD_PATHS[2:]
    keyed = {(row['case_id'], row['rep'], row['method']): row for row in rows}
    expected = {(case['case_id'], rep, method) for case in plan['cases'] for rep in range(4) for method in methods}
    require(len(rows) == len(keyed) == 320 and set(keyed) == expected, 'Full method matrix differs')
    cells = []
    for case in plan['cases']:
        truth = 'reject' if case['C'] > 0 else 'accept' if case['C'] < 0 else 'abstain'
        for method in methods:
            local = [keyed[(case['case_id'], rep, method)] for rep in range(4)]
            cells.append({'case_id': case['case_id'], 'kind': case['kind'], 'method': method, 'repetitions': 4,
                'mean_shots': sum(r['shots'] for r in local)/4,
                'wrong': sum(r['decision'] not in ('abstain', truth) for r in local),
                'correct_stops': sum(r['decision'] == truth and truth != 'abstain' for r in local),
                'abstained': sum(r['decision'] == 'abstain' for r in local)})
    require(cells == summary['cells'], 'Unconditional per-case costs/errors/abstentions differ')
    outcomes = []
    for kind in ('main', 'null'):
        for method in methods:
            local = [cell for cell in cells if cell['kind'] == kind and cell['method'] == method]
            outcomes.append({'kind': kind, 'method': method, 'trajectories': sum(c['repetitions'] for c in local),
                'wrong': sum(c['wrong'] for c in local), 'correct_stops': sum(c['correct_stops'] for c in local),
                'abstained': sum(c['abstained'] for c in local),
                'mean_paid_shots': sum(c['mean_shots'] for c in local)/len(local)})
    require(outcomes == summary['outcomes'], 'Method outcome totals differ')
    cell_map = {(c['case_id'], c['method']): c for c in cells}
    main_ids = [c['case_id'] for c in plan['cases'] if c['kind'] == 'main']
    ratios = {method: {reference: math.exp(sum(math.log(cell_map[(i, method)]['mean_shots']/
        cell_map[(i, reference)]['mean_shots']) for i in main_ids)/len(main_ids)) for reference in methods} for method in methods}
    require(summary['equal_case_geometric_cost_ratios'] == ratios, 'Equal-case geometric cost ratios differ')
    require(summary['path_records'] == COUNTERS['paths_checked'] == 160 and
            summary['method_records'] == COUNTERS['method_records_checked'] == 320 and
            summary['physically_generated_shots'] == COUNTERS['physically_generated_shots_accounted'], 'Summary ledger totals differ')
    require(summary['claim_boundary'] == plan['claim_boundary'], 'Development claim boundary changed')
    return {'outcomes': outcomes, 'equal_case_geometric_cost_ratios': ratios,
            'all_costs_include_wrong_and_cap_abstentions': True, 'inferential_ci_computed': False}


def audit(smoke=False):
    started = time.perf_counter()
    spec, plan, release = inspect_release(full=not smoke)
    prior = PriorAudit(plan)
    cases = {c['case_id']: c for c in plan['cases']}
    keys = [(c['case_id'], rep, path) for c in plan['cases'] for rep in range(4) for path in METHOD_PATHS]
    if smoke:
        with (DIRECTORY/'paths.jsonl').open(encoding='utf-8') as stream:
            # Existing first complete path only: this generates no observation.
            record = json.loads(next(stream))
        inspect_record(record, keys[0], cases[keys[0][0]], plan, prior)
        print(json.dumps({'status': 'existing_first_path_smoke_passed', 'counts': dict(COUNTERS),
                          'maximum_errors': MAXIMUM, 'elapsed_seconds': time.perf_counter()-started}), flush=True)
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name in ('path_audit.jsonl', 'replays.jsonl', 'audit.json', 'failed.json'):
        require(not (OUTPUT/name).exists(), 'Preserve prior audit output: '+name)
    inputs = {name: digest(DIRECTORY/name) for name in ('spec.json', 'release.json', 'preliminary_plan.json',
                                                     'paths.jsonl', 'method_results.jsonl', 'summary.json')}
    audit_sources = {path: digest(ROOT/path) for path in ('experiments/sota_round2/audit_benchmark.py',
        'experiments/structured_validation/audit_refined.py', 'experiments/structured_validation/audit_conditional.py')}
    all_rows, replay_count, current_key = [], 0, None
    def hard_deadline():
        require(time.time() < release['hard_deadline_unix'], 'Released replay hard deadline reached')
    try:
        with (DIRECTORY/'paths.jsonl').open(encoding='utf-8') as stream, \
             (DIRECTORY/'method_results.jsonl').open(encoding='utf-8') as methods, \
             (OUTPUT/'path_audit.jsonl').open('x', encoding='utf-8') as checks, \
             (OUTPUT/'replays.jsonl').open('x', encoding='utf-8') as replay_log:
            for key in keys:
                current_key = key
                hard_deadline()
                record = json.loads(next(stream))
                before = Counter(COUNTERS)
                rows = inspect_record(record, key, cases[key[0]], plan, prior)
                for row in rows:
                    require(json.loads(next(methods)) == row, 'Flushed method ledger differs from audited stopped prefix')
                all_rows.extend(rows)
                check = {'status': 'passed', 'case_id': key[0], 'rep': key[1], 'path': key[2],
                         'checks': dict(COUNTERS-before)}
                checks.write(json.dumps(check, separators=(',', ':'))+'\n')
                checks.flush()
                if key[1] in (0, 3):
                    # Mechanically fixed before outcomes: both endpoint reps of
                    # every case/path. This is ORIGINAL-RUNNER seed replay.
                    replay_started = time.perf_counter()
                    replayed = runner.run_path(cases[key[0]], key[1], key[2], plan, hard_deadline)
                    stored_no_timing = {name: value for name, value in record.items() if name != 'timings'}
                    replayed_no_timing = {name: value for name, value in replayed.items() if name != 'timings'}
                    require(stored_no_timing == replayed_no_timing, 'Original-runner frozen-seed full replay differs: '+str(key))
                    replay_count += 1
                    replay_log.write(json.dumps({'status': 'exact_original_runner_seed_replay_passed',
                        'case_id': key[0], 'rep': key[1], 'path': key[2], 'new_simulation_observations': 0,
                        'stored_protocol_sha256': hashlib.sha256(json.dumps(stored_no_timing, sort_keys=True,
                            separators=(',', ':')).encode()).hexdigest(),
                        'elapsed_seconds': time.perf_counter()-replay_started}, separators=(',', ':'))+'\n')
                    replay_log.flush()
                if COUNTERS['paths_checked'] % 4 == 0:
                    print(json.dumps({'status': 'auditing', 'paths_checked': COUNTERS['paths_checked'],
                        'method_records_checked': COUNTERS['method_records_checked'], 'original_runner_seed_replays': replay_count,
                        'last_key': key, 'elapsed_seconds': time.perf_counter()-started}), flush=True)
            require(next(stream, None) is None and next(methods, None) is None, 'Extra path/method records')
        result_summary = inspect_summary(all_rows, plan)
        require(replay_count == 80 and COUNTERS['terminal_certificates_checked'] == 320, 'Required full replay/terminal coverage differs')
        require(inputs == {name: digest(DIRECTORY/name) for name in inputs}, 'Producer artifacts changed during audit')
        require(audit_sources == {name: digest(ROOT/name) for name in audit_sources}, 'Audit source changed while running')
        inspect_release(full=True)
        result = {'status': 'passed', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'inputs': inputs, 'sources': spec['sources'], 'audit_sources': audit_sources,
            'counts': dict(COUNTERS), 'maximum_float_differences_or_gaps': MAXIMUM, 'actual_prior': prior.report,
            'original_runner_seed_replay': {'selection': 'Every case/path, rep 0 and 3, chosen mechanically before audit',
                'paths': replay_count, 'all_nontiming_protocol_fields_exact': True,
                'fully_independent_controller_recomputation': False, 'new_simulation_observations': 0},
            'summary_reconstruction': result_summary,
            'limitations': [
                'Development comparison only; no held-out superiority claim or empirical 1% risk certification.',
                'DKM learner, optimism and ratio tracking reconstructed independently; numerical projection optimality was not independently re-solved.',
                'Logged successful projections checked for physical/confidence-box feasibility and selected-label consistency; failures retain the preceding label.',
                'All logged opposite-model best responses checked by feasibility and a floating global convex gap, not a rigorous exact-oracle sample-complexity proof.',
                'The absolute 1e-5 design-oracle tolerance can matter at weak signals; risk certificates use the full physical cone instead of this design margin.',
                'Outward Decimal/null-support, KK21 threshold and conditional-CGF primitives are reused from hash-bound reviewed sources.',
                'Original-runner repetitions 0/3 are reproducibility checks, not independent controller implementations.',
                'Actual rational prior table drives the audit; its data-selected partial sum retains original weights and only lower-bounds full Q.',
                'Official DKM Julia artifacts were inspected separately but not executed; this is the disclosed formal-schedule Bernoulli-sign adaptation.'
            ], 'elapsed_seconds': time.perf_counter()-started}
        write_new(OUTPUT/'audit.json', result)
        print(json.dumps({'status': 'passed', 'audit': str(OUTPUT/'audit.json'), 'sha256': digest(OUTPUT/'audit.json'),
                          'elapsed_seconds': result['elapsed_seconds']}), flush=True)
    except BaseException as caught:
        write_new(OUTPUT/'failed.json', {'status': 'failed', 'exception': type(caught).__name__+': '+str(caught),
            'current_key': current_key,
            'counts': dict(COUNTERS), 'original_runner_seed_replays': replay_count, 'inputs': inputs,
            'audit_sources': audit_sources, 'elapsed_seconds': time.perf_counter()-started})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true', help='Check only the first already observed path; no new simulation')
    args = parser.parse_args()
    audit(smoke=args.smoke)
