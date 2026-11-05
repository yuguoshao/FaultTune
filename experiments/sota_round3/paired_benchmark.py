"""Released-only shared-PSR development comparison of finite sign boundaries.

Preparing a plan or running deterministic checks never samples observations.
All methods in a joint replicate use one fixed paired PSR history. Each keeps
its own stopping prefix. Coarse hedge aggregates the first 32 physical shots;
fine hedge uses every fine block. This couples their actual observations but
does not reproduce the old runner's different binomial random-number calls.
"""
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
from numbers import Integral
from pathlib import Path
from unittest.mock import patch
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import scipy
from scipy.special import gammaln

import exact_sign_boundary as exact
import generic_sign_boundary as generic

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/structured_validation'))
import run_refined as old
import weighted_fisher as fisher

DIRECTORY = ROOT/'results/sota_round3/paired_benchmark'
DEFAULT_PLAN = DIRECTORY/'preliminary_plan.json'
DEFAULT_RELEASE = DIRECTORY/'release.json'
PREVIOUS_PLAN = ROOT/'results/sota_round2/benchmark/preliminary_plan.json'
BOUNDARY_REVIEW = ROOT/'results/sota_round2/theory/finite_boundary_source_review.json'
MAIN_IDS = (7, 9, 19, 21, 31, 33, 43, 45)
GENERIC_NAMES = ('maxsprt', 'pair_grid', 'pairwise_hedge', 'sprt_0.6', 'sprt_0.75', 'sprt_0.9', 'sprt_0.99')
METHODS = ('beta_ville100', 'beta_exact')+GENERIC_NAMES+('closed_hedge_original', 'closed_hedge_fine_initial')
PSR_ARM = {'cos_numerator': 0, 'sin_numerator': 2, 'denominator': 2}


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


def binding(path):
    return {'path': relative(path), 'sha256': digest(path)}


def check_binding(item):
    path = ROOT/item['path']
    require(relative(path) == item['path'] and digest(path) == item['sha256'], 'File binding changed: '+item['path'])
    return path


def fine_schedule(cap=4096):
    return sorted(set(range(2, min(32, cap)+1, 2)) | {n for n in old.schedule(cap) if n > 32} | {cap})


def cases():
    prior = read_json(PREVIOUS_PLAN)
    lookup = {case['case_id']: case for case in prior['cases']}
    result = []
    for mirror in (False, True):
        for identifier in MAIN_IDS:
            source = lookup[identifier]
            result.append({'case_id': identifier+(1000 if mirror else 0), 'kind': 'main',
                           'A': source['A'], 'B': source['B'], 'C': -source['C'] if mirror else source['C'],
                           'source_case_id': identifier, 'mirror': mirror,
                           'rho': source['rho'], 'B_over_R': source['B_over_R']})
    for identifier in (100, 101):
        result.append({name: lookup[identifier][name] for name in ('case_id', 'kind', 'A', 'B', 'C')})
    return result


def boundary_bindings():
    result = {'beta_exact': binding(ROOT/'results/sota_round3/beta_boundary_N2048.json')}
    for name in GENERIC_NAMES:
        result[name] = binding(ROOT/'results/sota_round3/boundaries'/f'{name}.json')
    for name, item in result.items():
        artifact = read_json(ROOT/item['path'])
        item['artifact_sha256'] = artifact['artifact_sha256']
        if name == 'beta_exact':
            item['boundary_sha256'] = artifact['boundary_sha256']
    return result


def load_boundaries(plan):
    require(set(plan['boundaries']) == {'beta_exact', *GENERIC_NAMES}, 'Boundary method registry changed')
    loaded = {}
    for name, item in plan['boundaries'].items():
        artifact = read_json(check_binding(item))
        require(artifact['max_discordances'] == plan['cap']//2 and
                Fraction(int(artifact['delta_exact']['numerator']), int(artifact['delta_exact']['denominator'])) == Fraction(1, 100),
                'Boundary horizon or exact delta differs from released physical budget')
        require(artifact['artifact_sha256'] == item['artifact_sha256'], 'Boundary payload binding changed')
        if name == 'beta_exact':
            require(artifact['boundary_sha256'] == item['boundary_sha256'], 'Exact beta table binding changed')
            loaded[name] = exact.verify_artifact(artifact)
        else:
            require(artifact['shape'] == name, 'Generic score shape differs from method name')
            loaded[name] = generic.verify_artifact(artifact)
    return loaded


def check_plan(plan, verify_boundaries=True):
    require(plan['status'] == 'preliminary_requires_explicit_root_release' and
            plan['version'] == 'paired_shared_development_v1', 'Wrong plan version/status')
    require(plan['cap'] == 4096 and plan['delta'] == .01 and plan['repetitions'] == 32 and
            plan['seed_base'] == 330411000, 'Fixed benchmark budget changed')
    require(check_binding(plan['previous_plan']) == PREVIOUS_PLAN and plan['cases'] == cases(), 'Case matrix changed')
    require(plan['fine_checkpoints'] == fine_schedule() and plan['coarse_checkpoints'] == old.schedule(4096), 'Checkpoint set changed')
    require(plan['methods'] == list(METHODS) and plan['fixed_positive_arm'] == PSR_ARM, 'Methods or fixed paired action changed')
    require(plan['expected_path_records'] == 576 and plan['expected_method_records'] == 6336 and
            plan['max_physically_generated_shots'] == 2359296, 'Expected matrix size changed')
    review = read_json(check_binding(plan['boundary_review']))
    require(review['status'] == 'passed', 'Boundary source review must pass')
    for name, expected in review['sources'].items():
        require(digest(ROOT/name) == expected, 'Boundary preflight source changed: '+name)
    if verify_boundaries:
        return load_boundaries(plan)


def prepare(path=DEFAULT_PLAN):
    plan = {'version': 'paired_shared_development_v1', 'status': 'preliminary_requires_explicit_root_release',
            'created_utc': datetime.now(timezone.utc).isoformat(), 'previous_plan': binding(PREVIOUS_PLAN),
            'boundary_review': binding(BOUNDARY_REVIEW), 'cases': cases(), 'repetitions': 32,
            'cap': 4096, 'delta': .01, 'seed_base': 330411000,
            'seed_rule': 'seed_base + 1009*case_id + 1000003*rep; mirror case_id = source_case_id + 1000',
            'methods': list(METHODS), 'boundaries': boundary_bindings(), 'fixed_positive_arm': PSR_ARM,
            'fine_checkpoints': fine_schedule(), 'coarse_checkpoints': old.schedule(4096),
            'expected_path_records': 576, 'expected_method_records': 6336, 'max_physically_generated_shots': 2359296,
            'shared_observations': 'One fixed PSR paired path per case/rep. All 11 methods see identical cumulative counts at their own checkpoints. Continue until every method has stopped or reached cap; retain each method own stopping prefix.',
            'hedge_aggregation': 'Fine hedge consumes each fine block. Original hedge aggregates the first 32 physical shots into one block, then consumes the same later blocks. This changes Fisher conditioning relative to fine but shares the underlying four-category counts.',
            'coupling_boundary': 'The coarse aggregate has the old sampling distribution, but its RNG call history differs from the old runner. No exact old-seed trajectory match is asserted.',
            'guarantee_boundary': 'Exact and generic tables use the known finite cap. Ville beta and the two original block hedges retain their original anytime rules. All methods receive the same 4096-shot evaluation cap.',
            'truth_isolation': 'A/B/C are accessed only by simulator closure and final scoring. Certificates consume paid paired counts and prespecified boundary/state. No truth-guided allocation or method selection.',
            'analysis': 'Complete-case-matrix descriptive development comparison. Equal-case geometric ratios of per-case mean paid shots across all 16 main cases; errors and cap abstentions included. Report main positive/negative correct stops and null false stops. No inferred risk certification, significance claim or per-case hindsight method selector.',
            'timing_scope': 'Shared-driver CPU/wall time includes all methods and is not standalone method runtime.',
            'release_rule': 'No sampling until root release binds this plan, sources, boundary files, passed independent runner review, sampling cutoff and hard deadline.'}
    check_plan(plan)
    write_new(path, plan)
    return plan


def source_hashes(plan):
    sources = old.source_hashes()
    for name, value in {**exact.source_hashes(), **generic.source_hashes()}.items():
        require(name not in sources or sources[name] == value, 'Conflicting source binding')
        sources[name] = value
    sources[relative(Path(__file__).resolve())] = digest(__file__)
    sources[plan['boundary_review']['path']] = plan['boundary_review']['sha256']
    return sources


def new_hedge_states():
    return {'fine': fisher.empty_state(), 'original': fisher.empty_state(), 'initial_counts': [0, 0, 0, 0]}


def update_hedges(states, fresh, budget):
    number, minus, plus = sum(fresh), fresh[2]+fresh[3], fresh[1]+fresh[3]
    fisher.update_block(states['fine'], number, minus, plus)
    if budget <= 32:
        states['initial_counts'] = [a+b for a, b in zip(states['initial_counts'], fresh)]
        if budget == 32:
            combined = states['initial_counts']
            fisher.update_block(states['original'], sum(combined), combined[2]+combined[3], combined[1]+combined[3])
    else:
        require(states['original']['shots'] >= 32, 'Coarse initial block was not completed')
        fisher.update_block(states['original'], number, minus, plus)
    require(states['fine']['shots'] == budget and
            states['original']['shots'] == (budget if budget >= 32 else 0), 'Hedge block aggregation cost mismatch')


def evaluate(counts, states, active, budget, cap, delta, boundaries, timings):
    results, details = {}, {}
    b, c = counts[1:3]
    beta = float(gammaln(b+.5)+gammaln(c+.5)-gammaln(b+c+1)-2*gammaln(.5)+(b+c)*math.log(2))
    for method in active:
        started = time.perf_counter()
        cert = None
        if method == 'beta_exact':
            cert = exact.certificate_from_pairs(counts, boundaries[method])
            details[method] = {'certificate': cert}
        elif method in GENERIC_NAMES:
            cert = generic.certificate_from_pairs(counts, boundaries[method])
            details[method] = {'certificate': cert}
        else:
            if method == 'beta_ville100':
                screen = beta
                state = states['fine']
                engine = 'beta'
            else:
                original = method == 'closed_hedge_original'
                if original and budget < 32:
                    details[method] = {'eligible': False, 'reason': 'First original physical checkpoint is 32'}
                    continue
                state = states['original' if original else 'fine']
                fisher_log = fisher.screen(state, delta)['log_evidence']
                screen = float(np.logaddexp(beta, fisher_log)-math.log(2))
                engine = 'hedge'
            details[method] = {'eligible': True, 'log_screen_float': screen}
            if screen >= math.log(1/delta)-1e-6 or budget == cap:
                cert = old.evidence_certificate(engine, counts, state)
                details[method]['certificate'] = cert
        timings[method+'_evaluation_seconds'] += time.perf_counter()-started
        if cert is not None:
            require(cert['shots'] == budget and cert['decision'] in ('accept', 'reject', 'abstain'), 'Invalid method certificate')
            if cert['decision'] != 'abstain' or budget == cap:
                results[method] = cert
    return results, details


def paired_path(observe, plan, boundaries, deadline, evaluator=evaluate):
    """Observed-data controller; no true parameter or RNG is passed to tests."""
    counts, history, checks, outcomes = [0, 0, 0, 0], [], [], {}
    states, timings = new_hedge_states(), defaultdict(float)
    started, cpu_started, last = time.perf_counter(), time.process_time(), 0
    error = None
    try:
        for budget in plan['fine_checkpoints']:
            deadline()
            require(budget > last and (budget-last) % 2 == 0 and budget <= plan['cap'], 'Invalid paired physical block')
            number = (budget-last)//2
            begin = time.perf_counter()
            fresh = observe(number)
            timings['simulator_seconds'] += time.perf_counter()-begin
            require(isinstance(fresh, (tuple, list)) and len(fresh) == 4 and
                    all(isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)) and value >= 0 for value in fresh),
                    'Four nonnegative integer observed category counts required')
            fresh = list(map(int, fresh))
            require(sum(fresh) == number, 'Observed counts differ from paid block pairs')
            counts = [a+b for a, b in zip(counts, fresh)]
            history.append({'look': len(history)+1, 'budget': budget, 'arm': PSR_ARM,
                            'm': number, 'pair_counts': fresh, 'minus_successes': fresh[2]+fresh[3],
                            'plus_successes': fresh[1]+fresh[3]})
            begin = time.perf_counter()
            update_hedges(states, fresh, budget)
            timings['hedge_update_seconds'] += time.perf_counter()-begin
            active = [method for method in plan['methods'] if method not in outcomes]
            completed, detail = evaluator(counts.copy(), states, active, budget, plan['cap'], plan['delta'], boundaries, timings)
            checks.append({'shots': budget, 'counts': counts.copy(), 'methods': detail,
                           'fine_blocks': len(states['fine']['blocks']), 'original_blocks': len(states['original']['blocks'])})
            require(set(completed).issubset(active), 'Evaluator returned inactive method')
            for method, cert in completed.items():
                require(cert['shots'] == budget and cert['decision'] in ('accept', 'reject', 'abstain') and
                        (cert['decision'] != 'abstain' or budget == plan['cap']), 'Invalid stop/cap certificate')
                outcomes[method] = {'decision': cert['decision'], 'shots': budget, 'counts': counts.copy(),
                                    'history_prefix_length': len(history), 'certificate': cert}
            last = budget
            if len(outcomes) == len(plan['methods']):
                break
        require(len(outcomes) == len(plan['methods']), 'Incomplete method outcomes')
    except BaseException as caught:
        error = type(caught).__name__+': '+str(caught)
    timings.update(wall_seconds=time.perf_counter()-started, cpu_seconds=time.process_time()-cpu_started)
    return {'status': 'complete' if error is None else 'incomplete', 'error': error,
            'shots_observed': 2*sum(counts), 'counts': counts, 'history': history,
            'checkpoint_history': checks, 'outcomes': outcomes, 'timings': dict(timings)}


def run_case(case, rep, plan, boundaries, deadline):
    seed = plan['seed_base']+1009*case['case_id']+1000003*rep
    rng = np.random.RandomState(seed)
    # Simulator-only closure. Fixed PSR has cos=0, sin=1, so B is unused.
    pm, pp = case['A']-case['C'], case['A']+case['C']
    require(0 <= pm <= 1 and 0 <= pp <= 1, 'Invalid simulator probabilities')
    def observe(number):
        minus = int(rng.binomial(number, pm))
        both = int(rng.binomial(minus, pp))
        plus_only = int(rng.binomial(number-minus, pp))
        return [number-minus-plus_only, plus_only, minus-both, both]
    result = paired_path(observe, plan, boundaries, deadline)
    result.update(case_id=case['case_id'], kind=case['kind'], rep=rep, seed=seed,
                  cap=plan['cap'], delta=plan['delta'], truth_access='simulator_only')
    return result


def method_rows(path):
    return [dict(case_id=path['case_id'], kind=path['kind'], rep=path['rep'], seed=path['seed'],
                 method=method, **outcome) for method, outcome in path['outcomes'].items()]


def summarize(rows, plan):
    keyed = {(row['case_id'], row['rep'], row['method']): row for row in rows}
    expected = {(case['case_id'], rep, method) for case in plan['cases']
                for rep in range(plan['repetitions']) for method in plan['methods']}
    require(len(rows) == len(keyed) and set(keyed) == expected, 'Incomplete or duplicate method matrix')
    cells, outcomes, means = [], [], {}
    for case in plan['cases']:
        answer = 'reject' if case['C'] > 0 else 'accept' if case['C'] < 0 else 'abstain'
        for method in plan['methods']:
            local = [keyed[(case['case_id'], rep, method)] for rep in range(plan['repetitions'])]
            require(all(type(row['shots']) is int and 0 < row['shots'] <= plan['cap'] and
                        row['decision'] in ('accept', 'reject', 'abstain') and
                        (row['decision'] != 'abstain' or row['shots'] == plan['cap']) for row in local), 'Invalid paid outcome')
            mean = sum(row['shots'] for row in local)/len(local)
            means[case['case_id'], method] = mean
            cells.append({'case_id': case['case_id'], 'kind': case['kind'], 'sign': 1 if case['C'] > 0 else -1 if case['C'] < 0 else 0,
                          'method': method, 'trajectories': len(local), 'mean_paid_shots': mean,
                          'wrong': sum(row['decision'] not in ('abstain', answer) for row in local),
                          'correct_stops': sum(row['decision'] == answer and answer != 'abstain' for row in local),
                          'abstained': sum(row['decision'] == 'abstain' for row in local)})
    for group in ('main', 'positive', 'negative', 'null'):
        for method in plan['methods']:
            local = [cell for cell in cells if cell['method'] == method and
                     (cell['kind'] == group or group == 'positive' and cell['sign'] == 1 or group == 'negative' and cell['sign'] == -1)]
            require(local, 'Empty reporting group')
            outcomes.append({'group': group, 'method': method, 'trajectories': sum(cell['trajectories'] for cell in local),
                             'mean_paid_shots': sum(cell['mean_paid_shots'] for cell in local)/len(local),
                             **{name: sum(cell[name] for cell in local) for name in ('wrong', 'correct_stops', 'abstained')}})
    main_ids = [case['case_id'] for case in plan['cases'] if case['kind'] == 'main']
    ratios = {method: {reference: math.exp(sum(math.log(means[i, method]/means[i, reference]) for i in main_ids)/len(main_ids))
                       for reference in plan['methods']} for method in plan['methods']}
    return {'status': 'complete_descriptive_development_comparison', 'method_records': len(rows), 'cells': cells,
            'outcomes': outcomes, 'equal_case_geometric_cost_ratios': ratios,
            'scope': 'All prespecified cases and outcomes included. No inferential confidence interval or empirical delta certification.'}


def validate_release(plan_path, release_path):
    plan, release = read_json(plan_path), read_json(release_path)
    boundaries = check_plan(plan)
    require(release['status'] == 'released_by_root' and release['explicit_root_release'] is True and release['reference'].strip(),
            'Explicit root release required')
    require(release['plan_sha256'] == digest(plan_path) and release['sources'] == source_hashes(plan) and
            release['boundaries'] == plan['boundaries'], 'Released plan/source/boundary binding changed')
    review = read_json(check_binding(release['review']))
    require(review['status'] == 'passed' and review['plan_sha256'] == digest(plan_path) and
            review['sources'] == source_hashes(plan) and review['boundaries'] == plan['boundaries'],
            'Passed independent runner review must bind the full released protocol')
    require(all(isinstance(release[name], (int, float)) and not isinstance(release[name], bool) and math.isfinite(release[name])
                for name in ('sampling_cutoff_unix', 'hard_deadline_unix')) and
            release['sampling_cutoff_unix'] <= release['hard_deadline_unix'], 'Invalid finite deadline')
    return plan, release, boundaries


def run(plan_path=DEFAULT_PLAN, release_path=DEFAULT_RELEASE):
    plan, release, boundaries = validate_release(plan_path, release_path)
    for name in ('spec.json', 'paths.jsonl', 'method_results.jsonl', 'progress.json', 'summary.json'):
        require(not (DIRECTORY/name).exists(), 'Preserve existing output: '+name)
    def deadline():
        if time.time() >= release['sampling_cutoff_unix']:
            raise RuntimeError('Released sampling cutoff reached; observed prefix retained')
    deadline()
    write_new(DIRECTORY/'spec.json', {'plan': plan, 'plan_reference': binding(plan_path), 'release_reference': binding(release_path),
                                    'sources': source_hashes(plan), 'runtime': {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__}})
    rows, path_count, observed = [], 0, 0
    start = time.perf_counter()
    with (DIRECTORY/'paths.jsonl').open('x', encoding='utf-8') as raw, (DIRECTORY/'method_results.jsonl').open('x', encoding='utf-8') as out:
        for case in plan['cases']:
            for rep in range(plan['repetitions']):
                path = run_case(case, rep, plan, boundaries, deadline)
                raw.write(json.dumps(path, separators=(',', ':'), allow_nan=False)+'\n')
                raw.flush()
                added = method_rows(path)
                for row in added:
                    out.write(json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n')
                out.flush()
                rows.extend(added)
                path_count += 1
                observed += path['shots_observed']
                progress = {'status': 'running' if path['status'] == 'complete' else 'incomplete',
                            'written_path_records': path_count, 'completed_method_records': len(rows),
                            'physically_generated_shots': observed, 'case_id': case['case_id'], 'rep': rep,
                            'elapsed_seconds': time.perf_counter()-start, 'error': path['error']}
                pending = DIRECTORY/'progress.pending.json'
                pending.write_text(json.dumps(progress, indent=2)+'\n', encoding='utf-8')
                os.replace(str(pending), str(DIRECTORY/'progress.json'))
                print(json.dumps(progress), flush=True)
                require(path['status'] == 'complete', 'Incomplete observed prefix retained; run stopped')
    require(path_count == plan['expected_path_records'] and len(rows) == plan['expected_method_records'], 'Run matrix incomplete')
    require(time.time() < release['hard_deadline_unix'], 'Hard deadline reached before final summary')
    validate_release(plan_path, release_path)
    result = summarize(rows, plan)
    result.update(path_records=path_count, physically_generated_shots=observed, elapsed_seconds=time.perf_counter()-start,
                  input_sha256={name: digest(DIRECTORY/name) for name in ('spec.json', 'paths.jsonl', 'method_results.jsonl')})
    write_new(DIRECTORY/'summary.json', result)
    progress['status'] = 'complete'
    pending.write_text(json.dumps(progress, indent=2)+'\n', encoding='utf-8')
    os.replace(str(pending), str(DIRECTORY/'progress.json'))


def replay(case_id, rep, plan_path=DEFAULT_PLAN, release_path=DEFAULT_RELEASE):
    plan, release, boundaries = validate_release(plan_path, release_path)
    saved = []
    for line in (DIRECTORY/'paths.jsonl').open(encoding='utf-8'):
        row = json.loads(line)
        if (row['case_id'], row['rep']) == (case_id, rep):
            saved.append(row)
    require(len(saved) == 1 and saved[0]['status'] == 'complete', 'Replay key requires one complete path')
    def deadline():
        if time.time() >= release['hard_deadline_unix']:
            raise RuntimeError('Released hard deadline reached in seed replay')
    case = next(case for case in plan['cases'] if case['case_id'] == case_id)
    repeated = run_case(case, rep, plan, boundaries, deadline)
    for record in (saved[0], repeated):
        record.pop('timings', None)
    require(saved[0] == repeated, 'Seed replay differs from stored shared protocol')
    return {'status': 'exact_seed_replay_passed', 'case_id': case_id, 'rep': rep}


def synthetic_checks():
    """Deterministic aggregate/prefix tests; RandomState is forbidden."""
    states, combined = new_hedge_states(), [0, 0, 0, 0]
    for index, budget in enumerate(range(2, 33, 2)):
        fresh = [int(j == index % 4) for j in range(4)]
        combined = [a+b for a, b in zip(combined, fresh)]
        update_hedges(states, fresh, budget)
    require(combined == [4, 4, 4, 4] and len(states['fine']['blocks']) == 16 and len(states['original']['blocks']) == 1,
            'Fine-to-coarse initial aggregate mismatch')
    coarse = states['original']['blocks'][0]
    require(coarse['nminus'] == coarse['nplus'] == 16 and coarse['minus_successes'] == coarse['plus_successes'] == 8,
            'Coarse 32-shot sufficient statistics mismatch')
    update_hedges(states, [0, 1, 1, 0], 36)
    require(states['fine']['blocks'][-1] == states['original']['blocks'][-1] and
            states['fine']['shots'] == states['original']['shots'] == 36, 'Post-initial blocks differ')
    active_history = []
    test_plan = {'fine_checkpoints': fine_schedule(32), 'cap': 32, 'delta': .01, 'methods': ['early8', 'early10', 'cap']}
    def evaluator(counts, states, active, budget, cap, delta, boundaries, timings):
        active_history.append([budget, list(active)])
        result = {}
        for name in active:
            if budget == {'early8': 8, 'early10': 10, 'cap': 32}[name]:
                result[name] = {'shots': budget, 'decision': 'abstain' if name == 'cap' else 'reject'}
        return result, {}
    def observe(number):
        return [number, 0, 0, 0]
    with patch.object(np.random, 'RandomState', side_effect=AssertionError('No RNG permitted in synthetic checks')):
        shared = paired_path(observe, test_plan, {}, lambda: None, evaluator)
        require(shared['status'] == 'complete' and shared['shots_observed'] == 32 and
                [shared['outcomes'][name]['shots'] for name in test_plan['methods']] == [8, 10, 32], 'Shared stopping accounting differs')
        require([shared['outcomes'][name]['history_prefix_length'] for name in test_plan['methods']] == [4, 5, 16], 'Early method received a later prefix')
        require(next(names for shots, names in active_history if shots == 12) == ['cap'], 'Stopped methods kept receiving checks')
        calls = 0
        def stop_after_ten():
            nonlocal calls
            calls += 1
            if calls == 6:
                raise RuntimeError('synthetic cutoff')
        partial = paired_path(observe, test_plan, {}, stop_after_ten, evaluator)
        require(partial['status'] == 'incomplete' and partial['shots_observed'] == 10 and len(partial['outcomes']) == 2,
                'Deadline discarded observations or stopped outcomes')
        for bad in ([.5, .5, 0, 0], [True, 0, 0, 0], [2, 0, 0, 0]):
            failed = paired_path(lambda number, bad=bad: bad, test_plan, {}, lambda: None, evaluator)
            require(failed['status'] == 'incomplete' and failed['shots_observed'] == 0, 'Invalid observer counts accepted')
    # One positive, one negative, one null case with error and cap costs.
    matrix_plan = {'cases': [{'case_id': 1, 'kind': 'main', 'C': 1}, {'case_id': 2, 'kind': 'main', 'C': -1},
                             {'case_id': 3, 'kind': 'null', 'C': 0}], 'methods': ['one', 'two'], 'repetitions': 4, 'cap': 32}
    rows = []
    for case in matrix_plan['cases']:
        correct = 'reject' if case['C'] > 0 else 'accept'
        decisions = [correct, 'accept' if correct == 'reject' else 'reject', 'abstain', correct] if case['kind'] == 'main' else ['abstain', 'reject', 'accept', 'abstain']
        costs = [8, 10, 32, 14] if case['kind'] == 'main' else [32, 10, 12, 32]
        for method in matrix_plan['methods']:
            rows.extend({'case_id': case['case_id'], 'rep': rep, 'method': method, 'decision': decision, 'shots': cost}
                        for rep, (decision, cost) in enumerate(zip(decisions, costs)))
    summary = summarize(rows, matrix_plan)
    require(all(cell['mean_paid_shots'] == (16 if cell['kind'] == 'main' else 21.5) for cell in summary['cells']),
            'Error or abstention costs excluded from summary')
    require(all(value == 1 for ratios in summary['equal_case_geometric_cost_ratios'].values() for value in ratios.values()),
            'Identical methods do not have unit cost ratio')
    rejected = 0
    for malformed in (rows[:-1], rows+[rows[0]]):
        try:
            summarize(malformed, matrix_plan)
        except AssertionError:
            rejected += 1
    require(rejected == 2, 'Incomplete/duplicate matrices accepted')
    return {'status': 'passed', 'scope': 'Deterministic bookkeeping only; zero simulator observations',
            'coarse_initial_fine_blocks': 16, 'coarse_initial_aggregated_blocks': 1,
            'shared_stop_shots': [8, 10, 32], 'deadline_retained_shots': 10,
            'error_and_cap_costs_included': True, 'source_sha256': digest(__file__)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'check', 'run', 'replay'))
    parser.add_argument('--plan', type=Path, default=DEFAULT_PLAN)
    parser.add_argument('--release', type=Path, default=DEFAULT_RELEASE)
    parser.add_argument('--case-id', type=int)
    parser.add_argument('--rep', type=int, default=0)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.plan)
        print(json.dumps({'status': 'preliminary_plan_only_no_sampling', 'plan': str(args.plan), 'sha256': digest(args.plan)}))
    elif args.command == 'check':
        print(json.dumps(synthetic_checks()))
    elif args.command == 'run':
        run(args.plan, args.release)
    else:
        print(json.dumps(replay(args.case_id, args.rep, args.plan, args.release)))
