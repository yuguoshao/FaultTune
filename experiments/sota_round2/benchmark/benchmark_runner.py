"""Small development benchmark with an explicit root-controlled release gate.

The runner never mutates frozen modules. DKM and six-arm-initialized PSR each
share one observed path across their registered stopping certificates. The
two hedge paths receive explicit checkpoint lists as function arguments.
"""
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import gammaln, xlogy

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'experiments/structured_validation'))
sys.path.insert(0, str(ROOT/'experiments/sota_round2/literature'))
import dkm_baseline as dkm
import run_refined as original
import weighted_fisher as fisher

DIRECTORY = ROOT/'results/sota_round2/benchmark'
DEFAULT_PLAN = DIRECTORY/'preliminary_plan.json'
DEFAULT_RELEASE = DIRECTORY/'release.json'
MAIN_IDS = (7, 9, 19, 21, 31, 33, 43, 45)
PATH_NAMES = ('dkm_one_shot', 'psr_six_arm_init', 'closed_hedge_original', 'closed_hedge_fine_initial')
PREFLIGHT_PATHS = {
    'dkm_checks': 'results/sota_round2/literature/dkm_checks.json',
    'dkm_review': 'results/sota_round2/theory/dkm_review.json',
    'structured_grid_checks': 'results/sota_round2/structured_grid_checks.json',
    'structured_grid_review': 'results/sota_round2/theory/structured_grid_review.json'}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def fine_schedule(cap):
    return sorted(set(range(2, min(32, cap)+1, 2)) |
                  {n for n in original.schedule(cap) if n > 32} | {cap})


def check_plan(plan):
    require(plan['status'] == 'preliminary_development_design_requires_explicit_release', 'Wrong plan status')
    require(plan['cap'] == 4096 and plan['delta'] == .01 and plan['repetitions'] == 4 and
            plan['seed_base'] == 220411000, 'Fixed small-benchmark budget changed')
    require([c['case_id'] for c in plan['cases']] == list(MAIN_IDS)+[100, 101], 'Case matrix changed')
    old_spec = read_json(ROOT/'results/structured_validation/refined_development_v1/spec.json')
    require(plan['old_development_spec']['path'] == 'results/structured_validation/refined_development_v1/spec.json' and
            digest(ROOT/plan['old_development_spec']['path']) == plan['old_development_spec']['sha256'],
            'Old development specification binding changed')
    lookup = {row['case_id']: row for row in old_spec['cases']}
    require(plan['cases'][:8] == [lookup[i] for i in MAIN_IDS], 'Main cases differ from old development')
    require(plan['cases'][8:] == [
        {'case_id': 100, 'kind': 'null', 'A': .01, 'B': 0., 'C': 0.},
        {'case_id': 101, 'kind': 'null', 'A': .5, 'B': 0., 'C': 0.}], 'Fixed null cases changed')
    require(plan['arms'] == dkm.default_arms(), 'Fixed six-arm query set changed')
    require(plan['single_arm_checkpoints'] == fine_schedule(4096) and
            plan['hedge_original_checkpoints'] == original.schedule(4096) and
            plan['hedge_fine_checkpoints'] == fine_schedule(4096), 'Checkpoint schedule changed')
    require(plan['paths'] == list(PATH_NAMES), 'Path set changed')
    names = [item['name'] for item in plan['calibrations']]
    require(len(names) == len(set(names)) and names[:2] == ['kk21', 'kt_safe'], 'Invalid calibration registry')
    require(plan['expected_joint_labels'] == 40 and plan['expected_path_records'] == 160 and
            plan['expected_method_records'] == 40*(2*len(names)+2), 'Expected matrix size changed')
    for item in plan['calibrations']:
        require(item['kind'] in ('dkm_builtin', 'external_q'), 'Unknown calibration provider kind')
        if item['kind'] == 'dkm_builtin':
            require(item['name'] in ('kk21', 'kt_safe'), 'Unknown DKM built-in calibration')
        else:
            provider = (ROOT/item['module_path']).resolve()
            provider.relative_to(ROOT)
            require(provider.suffix == '.py', 'External provider must be a local Python module')
            require(item['name'] == 'structured_grid_safe' and
                    item['module_path'] == 'experiments/sota_round2/structured_grid_predictive.py',
                    'Only the reviewed fixed structured provider is registered')
            validate_prior(item)
    validate_preflight(plan)


def prepare(path=DEFAULT_PLAN, external=None, preflight=None):
    old_path = ROOT/'results/structured_validation/refined_development_v1/spec.json'
    old_spec = read_json(old_path)
    cases = {row['case_id']: row for row in old_spec['cases']}
    calibrations = [{'name': name, 'kind': 'dkm_builtin'} for name in ('kk21', 'kt_safe')]
    if external:
        name, module_path = external.split('=', 1)
        prior_path = ROOT/'results/sota_round2/structured_grid_prior.json'
        prior = read_json(prior_path)
        calibrations.append({'name': name, 'kind': 'external_q', 'module_path': module_path,
            'prior': {'path': relative(prior_path), 'sha256': digest(prior_path),
                      'grid_sha256': prior['grid_sha256'], 'model_count': len(prior['models']),
                      'parameter_order': ['A', 'B', 'C', 'weight'],
                      'actual_model_table': 'Exact rational numerator/denominator pairs in the hashed prior artifact models field'},
            'scope': 'Fixed engineering prior on rational physical A/B/C models mixed equally with KT; identical provider for DKM and PSR paths, not an oracle truth prior.'})
    roles = ('dkm_checks', 'dkm_review') + (('structured_grid_checks', 'structured_grid_review') if external else ())
    evidence_paths = dict(PREFLIGHT_PATHS)
    for item in preflight or ():
        role, target = item.split('=', 1)
        require(role in roles, 'Unknown preflight evidence role')
        evidence_paths[role] = target
    evidence = {role: {'path': relative(ROOT/evidence_paths[role]),
                       'sha256': digest(ROOT/evidence_paths[role])} for role in roles}
    plan = {
        'version': 'dkm_small_development_benchmark_v1',
        'status': 'preliminary_development_design_requires_explicit_release', 'created_utc': utc_now(),
        'cases': [cases[i] for i in MAIN_IDS]+[
            {'case_id': 100, 'kind': 'null', 'A': .01, 'B': 0., 'C': 0.},
            {'case_id': 101, 'kind': 'null', 'A': .5, 'B': 0., 'C': 0.}],
        'repetitions': 4, 'cap': 4096, 'delta': .01, 'seed_base': 220411000,
        'seed_rule': 'seed_base + 1009*case_id + 1000003*rep',
        'arms': dkm.default_arms(), 'paths': list(PATH_NAMES), 'calibrations': calibrations,
        'preflight': evidence,
        'single_arm_checkpoints': fine_schedule(4096),
        'hedge_original_checkpoints': original.schedule(4096),
        'hedge_fine_checkpoints': fine_schedule(4096),
        'expected_joint_labels': 40, 'expected_path_records': 160,
        'expected_method_records': 40*(2*len(calibrations)+2),
        'old_development_spec': {'path': relative(old_path), 'sha256': digest(old_path)},
        'decision_rule': 'At a common eligible checkpoint test both signs for each calibration. Choose the certified sign with larger evidence-minus-threshold margin; deterministic positive-sign tie break.',
        'sampler_rule': 'DKM native one-shot allocation; its label controls sampling only. PSR pays one observation of each identical fixed arm before alternating the two exact PSR arms.',
        'shared_path_rule': 'Registered calibrations share all observations and the sampler continues until every calibration stops or the cap is reached. Each method references only its stopped prefix.',
        'truth_isolation': 'Only the simulator consumes A/B/C. Samplers receive observed per-arm counts; stopping certificates receive arms, counts, sign and delta. Hedge policy receives paid-history fit rows.',
        'coupling': 'Independent joint replicate seeds. Paths start with the same seed but adaptive arms and binomial batching differ; no claim of identical per-arm prefixes across paths.',
        'timing_scope': 'Actual shared-driver CPU/wall costs, including all registered certificate work. Per-method shared-path timings are not standalone CPU benchmarks.',
        'closed_policy_scope': 'At cap 4096 the original EXPLORE_AFTER=4096 policy remains in its PSR warm-up. The fine schedule changes block partitions and therefore conditional Fisher factors; it is explicitly a separate protocol.',
        'analysis': 'Unconditional per-case mean paid shots and their equal-case geometric ratios. Include errors and cap abstentions. Report main correct-stop fraction and null false stops separately. No new inferential CI or empirical risk certification.',
        'release_rule': 'No sampling until root writes a release binding this exact plan, current provider/runner sources, a passed independent runner review and an explicit sampling cutoff.',
        'claim_boundary': 'Small development/implementation comparison on named methods and a fixed six-arm grid, not a SOTA or 1% empirical error certification.'}
    check_plan(plan)
    write_new(path, plan)
    return plan


def load_external(item):
    path = (ROOT/item['module_path']).resolve()
    spec = importlib.util.spec_from_file_location('benchmark_external_'+item['name'], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(callable(getattr(module, 'screen_log_numerator', None)) and
            callable(getattr(module, 'certificate', None)), 'External Q provider lacks the registered API')
    return module


def validate_prior(item):
    binding = item['prior']
    path = ROOT/binding['path']
    require(relative(path) == binding['path'] and digest(path) == binding['sha256'], 'Fixed prior artifact changed')
    artifact = read_json(path)
    table = artifact['models']
    table_hash = hashlib.sha256(json.dumps(table, separators=(',', ':')).encode()).hexdigest()
    require(binding['parameter_order'] == artifact['parameter_order'] == ['A', 'B', 'C', 'weight'],
            'Prior parameter order changed')
    require(binding['grid_sha256'] == artifact['grid_sha256'] == table_hash and
            binding['model_count'] == len(table) == 7141, 'Prior model table binding changed')
    module = load_external(item)
    generated = [[[value.numerator, value.denominator] for value in model] for model in module.models()]
    require(module.grid_sha256() == table_hash and generated == table,
            'Generated grid differs from exact released prior; retain artifact for cross-libm inspection')
    require(artifact['protocol'] == module.protocol_config() and
            artifact['source_sha256'] == digest(ROOT/item['module_path']), 'Prior provider source/config changed')


def validate_preflight(plan):
    roles = {'dkm_checks': 'experiments/sota_round2/literature/dkm_baseline.py',
             'dkm_review': 'experiments/sota_round2/literature/dkm_baseline.py'}
    if any(item['kind'] == 'external_q' for item in plan['calibrations']):
        roles.update({name: 'experiments/sota_round2/structured_grid_predictive.py'
                      for name in ('structured_grid_checks', 'structured_grid_review')})
    require(set(plan['preflight']) == set(roles), 'Missing or extra preflight evidence role')
    for role, source in roles.items():
        binding = plan['preflight'][role]
        path = ROOT/binding['path']
        require(relative(path) == binding['path'] and digest(path) == binding['sha256'], 'Preflight evidence changed: '+role)
        evidence = read_json(path)
        require(evidence['status'] == 'passed', 'Preflight evidence must pass: '+role)
        bound = evidence.get('sources', {}).get(source, evidence.get('source_sha256'))
        require(bound == digest(ROOT/source), 'Preflight source binding changed: '+role)
        for name, expected in evidence.get('sources', {}).items():
            require(digest(ROOT/name) == expected, 'Preflight dependent source changed: '+name)


def source_hashes(plan):
    sources = original.source_hashes()
    for path in [Path(__file__).resolve(), Path(__file__).with_name('benchmark_checks.py').resolve(),
                 ROOT/'experiments/sota_round2/literature/dkm_baseline.py']:
        sources[relative(path)] = digest(path)
    for item in plan['calibrations']:
        if item['kind'] == 'external_q':
            validate_prior(item)
            module = load_external(item)
            path = ROOT/item['module_path']
            sources[relative(path)] = digest(path)
            sources[item['prior']['path']] = digest(ROOT/item['prior']['path'])
            if callable(getattr(module, 'source_hashes', None)):
                for name, expected in module.source_hashes().items():
                    require(digest(ROOT/name) == expected, 'External provider source mismatch: '+name)
                    require(name not in sources or sources[name] == expected, 'Conflicting source binding')
                    sources[name] = expected
    return sources


class Calibration:
    def __init__(self, item):
        self.name = item['name']
        self.module = load_external(item) if item['kind'] == 'external_q' else None

    def numerator(self, arms, n, k):
        if self.module is not None:
            return float(self.module.screen_log_numerator(arms, n, k))
        if self.name == 'kk21':
            mean = k/n
            return float((xlogy(k, mean)+xlogy(n-k, 1-mean)).sum())
        return float((gammaln(k+.5)+gammaln(n-k+.5)-gammaln(n+1)-2*gammaln(.5)).sum())

    def threshold(self, n, delta):
        if self.name == 'kk21' and self.module is None:
            return float(dkm.kk_calibration_upper(str(delta), len(n)))+float((3*np.log1p(np.log(n))).sum())
        return math.log(1/delta)

    def certify(self, arms, n, k, sign, delta, proposal):
        if self.module is not None:
            return self.module.certificate(arms, n, k, sign, delta=delta, proposal=proposal)
        return dkm.certificate(arms, n, k, sign, delta=delta, calibration=self.name, proposal=proposal)


def certificate_margin(cert):
    return Decimal(cert['evidence_lower_decimal'])-Decimal(cert['threshold_upper_decimal'])


def evaluate_calibrations(arms, n, k, active, delta, force, timings):
    started = time.perf_counter()
    rows = dkm.observations(arms, n, k)
    fits = {sign: dkm.likelihood.null_fit(rows, sign) for sign in (-1, 1)}
    timings['screen_and_null_fit_seconds'] += time.perf_counter()-started
    results, checks = {}, {}
    for calibration in active:
        began = time.perf_counter()
        numerator = calibration.numerator(arms, n, k)
        threshold = calibration.threshold(n, delta)
        eligible = [sign for sign in (-1, 1) if force or numerator-fits[sign]['loglik'] >= threshold-1e-6]
        timings['screen_and_null_fit_seconds'] += time.perf_counter()-began
        certificates = []
        for sign in eligible:
            began = time.perf_counter()
            cert = calibration.certify(arms, n, k, sign, delta, fits[sign]['candidate'])
            timings['certificate_seconds'] += time.perf_counter()-began
            require(cert['shots'] == int(n.sum()), 'Certificate cost mismatch')
            require(cert['decision'] in ('accept', 'reject', 'abstain'), 'Invalid certificate decision')
            require(cert['sign'] == sign and cert['decision'] in
                    ('abstain', 'reject' if sign > 0 else 'accept'), 'Certificate direction mismatch')
            certificates.append(cert)
        checks[calibration.name] = {'numerator_float': numerator, 'threshold_float': threshold,
            'null_loglik_proposals': {str(sign): fits[sign]['loglik'] for sign in fits},
            'eligible_signs': eligible, 'certificates': certificates}
        crossing = [cert for cert in certificates if cert['decision'] != 'abstain']
        if crossing:
            results[calibration.name] = max(crossing, key=lambda cert: (certificate_margin(cert), cert['sign']))
        elif force:
            require(len(certificates) == 2, 'Cap requires both direction certificates')
            results[calibration.name] = max(certificates, key=lambda cert: (certificate_margin(cert), cert['sign']))
    return results, checks


def counts_path(arms, chooser, observe, calibrations, checkpoints, cap, delta, deadline,
                evaluator=evaluate_calibrations):
    """Pure controller/observer separation, also used by synthetic bookkeeping checks."""
    n, k = np.zeros(len(arms), dtype=np.int64), np.zeros(len(arms), dtype=np.int64)
    history, checks, stopped = [], [], {}
    timings = defaultdict(float)
    wall, cpu = time.perf_counter(), time.process_time()
    error = None
    try:
        while int(n.sum()) < cap and len(stopped) < len(calibrations):
            deadline()
            began = time.perf_counter()
            proposal = chooser(n.copy(), k.copy())
            timings['planning_seconds'] += time.perf_counter()-began
            require(all(isinstance(proposal[key], (int, np.integer)) and
                        not isinstance(proposal[key], (bool, np.bool_))
                        for key in ('arm', 'batch_size', 'observed_shots')), 'Integer controller indices/counts required')
            arm, length = int(proposal['arm']), int(proposal['batch_size'])
            require(0 <= arm < len(arms) and length == 1, 'Single-arm paths must take one paid observation')
            require(proposal['observed_shots'] == int(n.sum()), 'Controller used a different observed prefix')
            began = time.perf_counter()
            sample = observe(arm, length)
            timings['simulator_seconds'] += time.perf_counter()-began
            require(isinstance(sample, (int, np.integer)) and not isinstance(sample, (bool, np.bool_)),
                    'Integer simulator counts required')
            fresh = int(sample)
            require(0 <= fresh <= length, 'Invalid simulator count')
            n[arm] += length
            k[arm] += fresh
            history.append({'proposal': proposal, 'arm': arm, 'trials': length,
                            'successes': fresh, 'total_shots': int(n.sum())})
            total = int(n.sum())
            if (n > 0).all() and total in checkpoints:
                active = [c for c in calibrations if c.name not in stopped]
                completed, detail = evaluator(arms, n.copy(), k.copy(), active, delta, total == cap, timings)
                checks.append({'shots': total, 'trials': n.tolist(), 'successes': k.tolist(), 'calibrations': detail})
                require(set(completed).issubset({c.name for c in active}), 'Evaluator returned an inactive calibration')
                for name, certificate in completed.items():
                    require(certificate['shots'] == total and certificate['decision'] in ('accept', 'reject', 'abstain') and
                            (certificate['decision'] != 'abstain' or total == cap), 'Invalid stop/cap outcome')
                    stopped[name] = {'shots': total, 'decision': certificate['decision'],
                        'certificate': certificate, 'history_prefix_length': len(history),
                        'trials': n.tolist(), 'successes': k.tolist()}
        require(len(stopped) == len(calibrations), 'Every registered calibration needs a stop or cap abstention')
    except BaseException as caught:
        error = type(caught).__name__+': '+str(caught)
    timings.update(wall_seconds=time.perf_counter()-wall, cpu_seconds=time.process_time()-cpu)
    return {'status': 'complete' if error is None else 'incomplete', 'error': error,
            'shots_observed': int(n.sum()), 'trials': n.tolist(), 'successes': k.tolist(),
            'arms': arms, 'history': history, 'checkpoint_history': checks,
            'outcomes': stopped, 'timings': dict(timings)}


def single_arm_path(case, path_name, seed, plan, deadline):
    arms = plan['arms']
    if path_name == 'dkm_one_shot':
        sampler = dkm.DKMSampler(arms)
        chooser = lambda n, k: sampler.choose(n, k, batch_size=1)
    else:
        negative = next(i for i, a in enumerate(arms) if a['cos_numerator'] == 0 and a['sin_numerator'] == -a['denominator'])
        positive = next(i for i, a in enumerate(arms) if a['cos_numerator'] == 0 and a['sin_numerator'] == a['denominator'])
        def chooser(n, k):
            total = int(n.sum())
            empty = np.flatnonzero(n == 0)
            arm = int(empty[0]) if len(empty) else (negative if (total-len(arms)) % 2 == 0 else positive)
            return {'arm': arm, 'batch_size': 1, 'observed_shots': total,
                    'stage': 'paid_one_pull_per_arm_initialization' if len(empty) else 'fixed_psr_alternation'}
    # The simulator-only closure holds truth. It is never passed to the chooser or certificate.
    means = dkm.features(arms).dot(np.array([case['A'], case['B'], case['C']]))
    require(np.all((means >= 0) & (means <= 1)), 'Invalid simulator probabilities')
    rng = np.random.RandomState(seed)
    observe = lambda arm, length: int(rng.binomial(length, means[arm]))
    return counts_path(arms, chooser, observe, [Calibration(item) for item in plan['calibrations']],
                       set(plan['single_arm_checkpoints']), plan['cap'], plan['delta'], deadline)


def hedge_path(case, seed, cap, delta, checkpoints, deadline, binomial=None):
    """Standalone original hedge equations with an explicit local schedule.

    It omits unused safe/RB computations but preserves the paid closed-policy
    phases, queries, binomial sampling, Beta/Fisher capital and certificate.
    No mutation or monkeypatch of original.schedule is performed.
    """
    require(delta == original.DELTA, 'Frozen hedge certificate risk differs')
    if binomial is None:
        binomial = np.random.RandomState(seed).binomial
    rows, history, check_history, stopped = [], [], [], {}
    state, counts = fisher.empty_state(), np.zeros(4, dtype=np.int64)
    last, explored = 0, 0
    timings = defaultdict(float)
    wall, cpu = time.perf_counter(), time.process_time()
    error = None
    try:
        for look, budget in enumerate(checkpoints, 1):
            available = (budget-last)//2
            require(budget > last and (budget-last) % 2 == 0, 'Invalid paired checkpoint')
            if last < original.EXPLORE_AFTER:
                phases = [('psr', available)]
            else:
                target = int(math.ceil(original.SQRT_MULTIPLIER*math.sqrt(budget/2)))
                deficit = min(available, max(0, target-explored))
                phases = [('explore_a', (deficit+1)//2), ('explore_b', deficit//2), ('exploit', available-deficit)]
            for phase, number in phases:
                if not number:
                    continue
                deadline()
                began = time.perf_counter()
                if phase == 'psr':
                    arm = original.positive_arm('psr', rows)
                elif phase.startswith('explore'):
                    arm = original.rational_circle(4, 7) if phase == 'explore_a' else original.rational_circle(7, 4)
                else:
                    arm = original.positive_arm('closed', rows)
                timings['planning_seconds'] += time.perf_counter()-began
                # Truth is used only in the following simulator block.
                co, si = arm['cos_numerator']/arm['denominator'], arm['sin_numerator']/arm['denominator']
                pm, pp = case['A']+case['B']*co-case['C']*si, case['A']+case['B']*co+case['C']*si
                began = time.perf_counter()
                old = int(binomial(number, pm))
                both = int(binomial(old, pp))
                worse = int(binomial(number-old, pp))
                fresh = [number-old-worse, worse, old-both, both]
                timings['simulator_seconds'] += time.perf_counter()-began
                minus, plus = old, worse+both
                history.append({'look': look, 'budget': budget, 'phase': phase, 'arm': arm, 'm': number,
                                'pair_counts': fresh, 'minus_successes': minus, 'plus_successes': plus})
                counts += fresh
                began = time.perf_counter()
                fisher.update_block(state, number, minus, plus)
                timings['screen_seconds'] += time.perf_counter()-began
                original.merge_row(rows, arm, number, plus)
                negative = dict(arm)
                negative['sin_numerator'] = -negative['sin_numerator']
                original.merge_row(rows, negative, number, minus)
                if phase.startswith('explore'):
                    explored += number
            require(int(2*counts.sum()) == state['shots'] == budget, 'Paid paired counts do not match checkpoint')
            began = time.perf_counter()
            b, c = counts[1:3]
            beta = float(gammaln(b+.5)+gammaln(c+.5)-gammaln(b+c+1)-2*gammaln(.5)+(b+c)*math.log(2))
            fl = fisher.screen(state, delta)['log_evidence']
            screen = float(np.logaddexp(beta, fl)-math.log(2))
            timings['screen_seconds'] += time.perf_counter()-began
            check = {'shots': budget, 'beta_log_float': beta, 'fisher_log_float': fl, 'hedge_log_float': screen}
            if screen >= math.log(1/delta)-1e-6 or budget == cap:
                began = time.perf_counter()
                certificate = original.evidence_certificate('hedge', counts, state)
                timings['certificate_seconds'] += time.perf_counter()-began
                check['certificate'] = certificate
                if certificate['decision'] != 'abstain' or budget == cap:
                    stopped['hedge'] = {'shots': budget, 'decision': certificate['decision'],
                        'certificate': certificate, 'history_prefix_length': len(history),
                        'counts': list(map(int, counts)), 'exploration_pairs': explored}
            check_history.append(check)
            last = budget
            if stopped:
                break
        require(bool(stopped), 'Hedge path lacks stop or cap abstention')
    except BaseException as caught:
        error = type(caught).__name__+': '+str(caught)
    timings.update(wall_seconds=time.perf_counter()-wall, cpu_seconds=time.process_time()-cpu)
    return {'status': 'complete' if error is None else 'incomplete', 'error': error,
            'shots_observed': int(2*counts.sum()), 'counts': list(map(int, counts)),
            'history': history, 'checkpoint_history': check_history,
            'outcomes': stopped, 'timings': dict(timings),
            'schedule_adaptation': 'Explicit function argument; original module globals unchanged',
            'omitted_unused_engines': ['safe', 'rb', 'pair_grid']}


def run_path(case, rep, path_name, plan, deadline):
    seed = plan['seed_base']+1009*case['case_id']+1000003*rep
    if path_name in ('dkm_one_shot', 'psr_six_arm_init'):
        record = single_arm_path(case, path_name, seed, plan, deadline)
    else:
        key = 'hedge_original_checkpoints' if path_name == 'closed_hedge_original' else 'hedge_fine_checkpoints'
        record = hedge_path(case, seed, plan['cap'], plan['delta'], plan[key], deadline)
    record.update(case_id=case['case_id'], kind=case['kind'], rep=rep, seed=seed, path=path_name,
                  cap=plan['cap'], delta=plan['delta'], truth_access='simulator_only')
    return record


def method_rows(path_record):
    output = []
    for calibration, result in path_record['outcomes'].items():
        method = path_record['path'] if calibration == 'hedge' else path_record['path']+'__'+calibration
        output.append({'case_id': path_record['case_id'], 'kind': path_record['kind'], 'rep': path_record['rep'],
            'seed': path_record['seed'], 'path': path_record['path'], 'calibration': calibration, 'method': method,
            'shots': result['shots'], 'decision': result['decision'],
            'history_prefix_length': result['history_prefix_length'], 'certificate': result['certificate']})
    return output


def summarize(rows, plan):
    expected_methods = [path+'__'+item['name'] for path in PATH_NAMES[:2] for item in plan['calibrations']]+list(PATH_NAMES[2:])
    keyed = {(row['case_id'], row['rep'], row['method']): row for row in rows}
    expected = {(case['case_id'], rep, method) for case in plan['cases'] for rep in range(4) for method in expected_methods}
    require(len(keyed) == len(rows) and set(keyed) == expected, 'Incomplete or duplicate method matrix')
    cells, outcomes = [], []
    for case in plan['cases']:
        answer = 'reject' if case['C'] > 0 else 'accept' if case['C'] < 0 else 'abstain'
        for method in expected_methods:
            local = [keyed[(case['case_id'], rep, method)] for rep in range(4)]
            require(all(0 < r['shots'] <= plan['cap'] and
                        r['decision'] in ('accept', 'reject', 'abstain') and
                        (r['decision'] != 'abstain' or r['shots'] == plan['cap']) for r in local), 'Invalid counted cost')
            cells.append({'case_id': case['case_id'], 'kind': case['kind'], 'method': method,
                'repetitions': 4, 'mean_shots': sum(r['shots'] for r in local)/4,
                'wrong': sum(r['decision'] not in ('abstain', answer) for r in local),
                'correct_stops': sum(r['decision'] == answer and answer != 'abstain' for r in local),
                'abstained': sum(r['decision'] == 'abstain' for r in local)})
    for kind in ('main', 'null'):
        for method in expected_methods:
            local = [c for c in cells if c['kind'] == kind and c['method'] == method]
            outcomes.append({'kind': kind, 'method': method, 'trajectories': sum(c['repetitions'] for c in local),
                'wrong': sum(c['wrong'] for c in local), 'correct_stops': sum(c['correct_stops'] for c in local),
                'abstained': sum(c['abstained'] for c in local),
                'mean_paid_shots': sum(c['mean_shots'] for c in local)/len(local)})
    aggregate = {}
    main_ids = [c['case_id'] for c in plan['cases'] if c['kind'] == 'main']
    cell_map = {(c['case_id'], c['method']): c for c in cells}
    for method in expected_methods:
        aggregate[method] = {reference: math.exp(sum(math.log(cell_map[(i, method)]['mean_shots']/
            cell_map[(i, reference)]['mean_shots']) for i in main_ids)/len(main_ids))
            for reference in expected_methods}
    return {'status': 'complete_development_diagnostic_no_inferential_ci', 'cells': cells, 'outcomes': outcomes,
            'equal_case_geometric_cost_ratios': aggregate, 'method_records': len(rows),
            'claim_boundary': plan['claim_boundary']}


def validate_release(plan_path, release_path):
    plan = read_json(plan_path)
    check_plan(plan)
    release = read_json(release_path)
    require(release['status'] == 'released_by_root' and release['explicit_root_release'] is True and
            bool(release['reference'].strip()), 'Explicit root release is required')
    require(release['plan_sha256'] == digest(plan_path), 'Released plan changed')
    current = source_hashes(plan)
    require(release['sources'] == current, 'Source changed after root release')
    require(release['preflight'] == plan['preflight'], 'Released preflight evidence differs from plan')
    review_path = ROOT/release['review']['path']
    require(digest(review_path) == release['review']['sha256'], 'Runner review changed')
    review = read_json(review_path)
    require(review['status'] == 'passed' and all(review['sources'].get(name) == value for name, value in current.items()),
            'Passed runner review must cover all current implementation sources')
    require(math.isfinite(release['sampling_cutoff_unix']) and math.isfinite(release['hard_deadline_unix']) and
            release['sampling_cutoff_unix'] <= release['hard_deadline_unix'], 'Invalid release deadline')
    return plan, release


def run(plan_path=DEFAULT_PLAN, release_path=DEFAULT_RELEASE):
    plan, release = validate_release(plan_path, release_path)
    sources = source_hashes(plan)
    for name in ('spec.json', 'paths.jsonl', 'method_results.jsonl', 'progress.json', 'summary.json'):
        require(not (DIRECTORY/name).exists(), 'Preserve existing benchmark output: '+name)
    def deadline():
        if time.time() >= release['sampling_cutoff_unix']:
            raise RuntimeError('Released sampling cutoff reached; observed path retained')
    deadline()
    write_new(DIRECTORY/'spec.json', {'plan': plan, 'plan_reference': {'path': relative(plan_path), 'sha256': digest(plan_path)},
        'release_reference': {'path': relative(release_path), 'sha256': digest(release_path)}, 'sources': sources,
        'runtime': {'python': sys.version, 'numpy': np.__version__}, 'created_utc': utc_now()})
    all_rows, path_count, observed = [], 0, 0
    started = time.perf_counter()
    with (DIRECTORY/'paths.jsonl').open('x', encoding='utf-8') as raw, (DIRECTORY/'method_results.jsonl').open('x', encoding='utf-8') as methods:
        for case in plan['cases']:
            for rep in range(4):
                for path_name in plan['paths']:
                    record = run_path(case, rep, path_name, plan, deadline)
                    raw.write(json.dumps(record, separators=(',', ':'), allow_nan=False)+'\n')
                    raw.flush()
                    rows = method_rows(record)
                    for row in rows:
                        methods.write(json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n')
                    methods.flush()
                    all_rows.extend(rows)
                    path_count += 1
                    observed += record['shots_observed']
                    progress = {'status': 'running' if record['status'] == 'complete' else 'incomplete',
                        'written_path_records': path_count,
                        'completed_path_records': path_count if record['status'] == 'complete' else path_count-1,
                        'completed_method_records': len(all_rows), 'physically_generated_shots': observed,
                        'case_id': case['case_id'], 'rep': rep, 'path': path_name,
                        'elapsed_seconds': time.perf_counter()-started, 'error': record['error']}
                    pending = DIRECTORY/'progress.pending.json'
                    pending.write_text(json.dumps(progress, indent=2)+'\n', encoding='utf-8')
                    os.replace(str(pending), str(DIRECTORY/'progress.json'))
                    print(json.dumps(progress), flush=True)
                    require(record['status'] == 'complete', 'Partial observed path retained; stop this run')
    require(path_count == plan['expected_path_records'] and len(all_rows) == plan['expected_method_records'], 'Matrix incomplete')
    validate_release(plan_path, release_path)
    result = summarize(all_rows, plan)
    result.update(path_records=path_count, physically_generated_shots=observed,
        input_sha256={name: digest(DIRECTORY/name) for name in ('spec.json', 'paths.jsonl', 'method_results.jsonl')},
        sources=sources, elapsed_seconds=time.perf_counter()-started)
    write_new(DIRECTORY/'summary.json', result)
    progress['status'] = 'complete'
    pending = DIRECTORY/'progress.pending.json'
    pending.write_text(json.dumps(progress, indent=2)+'\n', encoding='utf-8')
    os.replace(str(pending), str(DIRECTORY/'progress.json'))


def replay(case_id, rep, path_name, plan_path=DEFAULT_PLAN, release_path=DEFAULT_RELEASE):
    plan, release = validate_release(plan_path, release_path)
    stored = None
    with (DIRECTORY/'paths.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if (row['case_id'], row['rep'], row['path']) == (case_id, rep, path_name):
                require(stored is None, 'Duplicate replay key')
                stored = row
    require(stored is not None and stored['status'] == 'complete', 'Replay requires one complete stored path')
    case = next(c for c in plan['cases'] if c['case_id'] == case_id)
    def replay_deadline():
        if time.time() >= release['hard_deadline_unix']:
            raise RuntimeError('Released hard deadline reached during seed replay')
    replayed = run_path(case, rep, path_name, plan, replay_deadline)
    for record in (stored, replayed):
        record.pop('timings', None)
    require(stored == replayed, 'Seed replay differs from stored protocol data')
    print(json.dumps({'status': 'exact_seed_replay_passed', 'case_id': case_id, 'rep': rep, 'path': path_name}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'replay'))
    parser.add_argument('--plan', type=Path, default=DEFAULT_PLAN)
    parser.add_argument('--release', type=Path, default=DEFAULT_RELEASE)
    parser.add_argument('--external', help='Optional registered Q calibration: name=root/relative/module.py')
    parser.add_argument('--preflight', action='append', help='Override preflight evidence path: role=root/relative/file.json')
    parser.add_argument('--case-id', type=int)
    parser.add_argument('--rep', type=int, default=0)
    parser.add_argument('--path', choices=PATH_NAMES)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.plan, args.external, args.preflight)
        print(json.dumps({'status': 'preliminary_plan_only_no_release', 'plan': str(args.plan), 'sha256': digest(args.plan)}))
    elif args.command == 'run':
        run(args.plan, args.release)
    else:
        replay(args.case_id, args.rep, args.path, args.plan, args.release)
