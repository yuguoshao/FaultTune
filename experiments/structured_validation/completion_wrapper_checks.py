"""Independent non-sampling boundary checks of the administrative wrapper."""
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
import hashlib
import json

import completion_common as target


def run_checks():
    outcomes = []

    def rejects(name, callback):
        try:
            callback()
        except (AssertionError, ValueError, KeyError, TypeError) as error:
            outcomes.append({'case': name, 'status': 'rejected', 'message': str(error)})
        else:
            raise AssertionError('Unsafe input accepted: '+name)

    plan = {'cases': [{'case_id': 0, 'kind': 'main'}, {'case_id': 1, 'kind': 'null'}],
            'repetitions_by_kind': {'main': 2, 'null': 1},
            'cap_by_kind': {'main': 100, 'null': 20}, 'seed_base': 1000000}
    selected = {'methods': ['closed_hedge', 'closed_rb', 'psr_beta']}
    groups = list(target.layout(plan, selected))
    rows = []
    for group in groups:
        for method in reversed(group['methods']):
            rows.append({'case_id': group['case']['case_id'], 'rep': group['rep'],
                         'seed': group['seed'], 'method': method, 'policy': group['policy'],
                         'shots': 10, 'decision': 'reject'})

    def encoded(items):
        return b''.join((json.dumps(row, separators=(',', ':'))+'\n').encode() for row in items)

    def parse(items):
        return target.read_prefix(BytesIO(encoded(items)), plan, selected)

    complete = parse(rows)
    assert complete['records'] == 9 and complete['groups_consumed'] == 6
    assert complete['next_group'] is None and complete['completed_case_ids'] == [0, 1]
    assert complete['sha256'] == hashlib.sha256(encoded(rows)).hexdigest()
    assert complete['boundaries']['0'] == len(encoded(rows[:6]))
    assert complete['boundaries']['1'] == len(encoded(rows))
    assert parse([])['next_group'] == groups[0]
    prefix = parse(rows[:2])
    assert prefix['records'] == 2 and prefix['groups_consumed'] == 1
    assert prefix['next_group'] == groups[1]
    assert parse(list(reversed(rows[:2]))+rows[2:])['records'] == 9
    outcomes.extend({'case': name, 'status': 'accepted_as_required'} for name in
                    ['complete_matrix', 'empty_prefix', 'whole_group_prefix', 'either_within_group_order'])
    rejects('partial_execution_group', lambda: parse(rows[:1]))
    rejects('missing_middle_record', lambda: parse(rows[:2]+rows[3:]))
    rejects('missing_middle_whole_group', lambda: parse(rows[:3]+rows[5:]))
    rejects('duplicate_method_in_group', lambda: parse([rows[0], rows[0]]+rows[2:]))
    rejects('duplicate_complete_group', lambda: parse(rows[:2]+rows))
    rejects('unexpected_extra_record', lambda: parse(rows+[rows[-1]]))
    rejects('noncanonical_group_order', lambda: parse(rows[2:3]+rows[:2]+rows[3:]))
    rejects('unterminated_final_line', lambda: target.read_prefix(BytesIO(encoded(rows)[:-1]), plan, selected))
    rejects('invalid_json', lambda: target.read_prefix(BytesIO(b'{bad}\n'), plan, selected))
    for field, value in [('case_id', 99), ('rep', 9), ('seed', 0), ('policy', 'grid'),
                         ('method', 'closed_safe'), ('shots', 0), ('shots', 101),
                         ('shots', True), ('shots', 10.0), ('decision', 'unknown'),
                         ('decision', 'abstain')]:
        changed = deepcopy(rows)
        changed[0][field] = value
        rejects('changed_'+field+'_'+str(value), lambda changed=changed: parse(changed))
    cap_rows = deepcopy(rows)
    cap_rows[0].update(shots=100, decision='abstain')
    assert parse(cap_rows)['records'] == 9
    outcomes.append({'case': 'cap_abstention', 'status': 'accepted_as_required'})

    # No real file is modified: the original frozen loader itself rejects the
    # plan and algorithm mutations through an in-memory digest facade.
    frozen = target.frozen_common
    real_plan = frozen.load_plan()
    plan_hash = frozen.PLAN_SHA256
    control = frozen.ROOT/'results/structured_validation/run_control.json'
    expected_hashes = {frozen.PLAN: plan_hash, control: real_plan['run_control_sha256']}
    expected_hashes.update({frozen.ROOT/name: value
                           for name, value in real_plan['algorithm_sources_sha256'].items()})
    for name, changed_path in [('changed_frozen_plan_hash', frozen.PLAN),
                               ('changed_frozen_algorithm_hash', next(iter(expected_hashes.keys()-{frozen.PLAN, control}))),
                               ('changed_original_run_control_hash', control)]:
        def facade(path, changed_path=changed_path):
            return '0'*64 if Path(path) == changed_path else expected_hashes[Path(path)]
        with patch.object(frozen, 'digest', facade), patch.object(frozen, 'read_json', return_value=real_plan):
            rejects(name, frozen.load_plan)

    # Administrative fields are checked independently of any sampled outcomes.
    release = {'wrapper_sources_sha256': {'old.py': 'a'*64}}
    methods = {'methods': ['closed_hedge', 'closed_rb', 'psr_beta']}
    real_plan['expected_joint_replicate_labels'] = 3200
    sources = target.source_hashes()
    old_progress = target.read_json(target.BASE/'progress.json')
    amendment = {
        'original_plan_sha256': plan_hash,
        'algorithm_sources_sha256': real_plan['algorithm_sources_sha256'],
        'original_wrapper_sources_sha256': release['wrapper_sources_sha256'],
        'completion_sources_sha256': sources,
        'execution': {'started_utc': target.START_UTC, 'cutoff_utc': target.CUTOFF_UTC,
                      'hard_deadline_utc': target.DEADLINE_UTC, 'max_minutes': 90,
                      'cutoff_unix': target.timestamp(target.CUTOFF_UTC),
                      'hard_deadline_unix': target.timestamp(target.DEADLINE_UTC)},
        'original_records': 7937, 'expected_records': 9600, 'remaining_records': 1663,
        'original_directory': target.relative(target.BASE),
        'original_ledger_bytes': old_progress['flushed_ledger_bytes'],
        'original_complete_case_ids': old_progress['completed_case_ids'],
        'original_files_sha256': {target.relative(p): '0'*64
                                 for p in target.BASE.rglob('*') if p.is_file()}}
    variants = [('amended_plan', ('original_plan_sha256',), '0'*64),
                ('amended_algorithm_set', ('algorithm_sources_sha256',), {}),
                ('amended_old_wrapper', ('original_wrapper_sources_sha256',), {}),
                ('changed_completion_source', ('completion_sources_sha256',), {}),
                ('changed_original_inventory', ('original_files_sha256',), {}),
                ('changed_original_directory', ('original_directory',), 'elsewhere'),
                ('changed_original_byte_metadata', ('original_ledger_bytes',), 1),
                ('changed_original_case_metadata', ('original_complete_case_ids',), []),
                ('changed_start', ('execution', 'started_utc'), '2026-11-04T08:22:03Z'),
                ('changed_sampling_cutoff', ('execution', 'cutoff_unix'), 0),
                ('changed_hard_deadline', ('execution', 'hard_deadline_unix'), 0),
                ('changed_maximum_minutes', ('execution', 'max_minutes'), 91),
                ('changed_original_count', ('original_records',), 7936),
                ('changed_expected_count', ('expected_records',), 9601),
                ('changed_remaining_count', ('remaining_records',), 1664)]
    with patch.object(target, 'original_release', return_value=(real_plan, methods, release)):
        with patch.object(target, 'read_json', side_effect=lambda path: amendment if path == target.AMENDMENT else old_progress):
            target.validate_amendment(recheck_original=False)
        for name, keys, value in variants:
            changed = deepcopy(amendment)
            node = changed
            for key in keys[:-1]:
                node = node[key]
            node[keys[-1]] = value
            with patch.object(target, 'read_json', side_effect=lambda path: changed if path == target.AMENDMENT else old_progress):
                rejects(name, lambda: target.validate_amendment(recheck_original=False))
    return {'status': 'passed', 'scope': 'Synthetic BytesIO records and in-memory mutation facades only; no simulator or run_trial calls, no real file mutations.',
            'cases': outcomes, 'rejected_cases': sum(r['status']=='rejected' for r in outcomes),
            'accepted_boundary_cases': sum(r['status']=='accepted_as_required' for r in outcomes)}


if __name__ == '__main__':
    print(json.dumps(run_checks(), indent=2))
