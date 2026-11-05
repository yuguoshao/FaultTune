"""Independently verify adaptive v2, with an exclusive audit source filename."""

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from adaptive_protocol import (EARLY_RISK, HOLDOUT_RISK, POLICIES, early_schedule,
                               early_route, complete_pilot, holdout_design)
from audit_validation import (ROOT, sha256, compare_json, close,
                              independently_summarize, frozen_files)
from certificates import fisher_certificate
import run_adaptive_validation as runner


def main(directory, replay_ids):
    started = time.perf_counter()
    this_source = Path(__file__).resolve()
    source_hash_at_start = sha256(this_source)
    directory = Path(directory).resolve()
    spec = json.loads((directory/'spec.json').read_text(encoding='utf-8'))
    summary = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    assert spec == summary['spec']
    assert spec['early_risk'] == EARLY_RISK == .005
    assert spec['holdout_risk'] == HOLDOUT_RISK == .005
    assert spec['early_risk']+spec['holdout_risk'] == spec['delta'] == .01
    assert spec['cases'] == runner.case_matrix()
    assert spec['methods'] == list(runner.METHODS)
    assert spec['early_checkpoints'] == early_schedule()
    assert spec['total_shot_checkpoints'] == runner.budget_schedule()
    assert spec['design_config'] == runner.config_dict()
    for name,digest in spec['source_sha256'].items():
        assert sha256(ROOT/name) == digest, name
    source = Path(spec['baseline_reuse']['path'])
    for name,digest in spec['baseline_reuse']['sha256'].items():
        assert sha256(source/name) == digest, name
    prior_audit = json.loads((source/'audit.json').read_text(encoding='utf-8'))
    assert prior_audit['status'] == 'passed'
    assert prior_audit['artifact_sha256'] == spec['baseline_reuse']['sha256']
    originals = {}
    with (source/'trials.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['method'] in runner.METHODS[:2] and row['rep'] < spec['repetitions']:
                originals[row['case_id'],row['method'],row['rep']] = row
    cases,methods = spec['cases'],spec['methods']
    groups = {(case['case_id'],method):[] for case in cases for method in methods}
    seen = set()
    reused = early_stops = holdout_stops = holdout_abstentions = 0
    early_cache = {}
    schedule = spec['total_shot_checkpoints']
    with (directory/'trials.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            key = row['case_id'],row['method'],row['rep']
            assert key not in seen, key
            seen.add(key)
            assert key[:2] in groups
            groups[key[:2]].append(row)
            if row['method'] in runner.METHODS[:2]:
                assert row['phase'] == 'baseline_reuse'
                assert {k:v for k,v in row.items() if k != 'phase'} == originals[key], key
                reused += 1
                continue
            assert row['method'] in POLICIES
            assert row['early_risk'] == .005 and row['holdout_risk'] == .005
            assert row['total_risk_bound'] == .01
            pc = row['early_pair_counts']
            assert len(pc) == 4 and all(type(n) is int and n >= 0 for n in pc)
            transcript = tuple(pc)
            if transcript not in early_cache:
                early_cache[transcript] = early_route(pc)
            route = early_cache[transcript]
            assert row['early_certificate'] == route['certificate'], key
            early_shots = 2*sum(pc)
            assert 1 <= row['early_look_index'] <= len(spec['early_checkpoints'])
            assert early_shots == spec['early_checkpoints'][row['early_look_index']-1]
            assert len(row['pilot_counts']) == len(row['pilot_totals']) == 4
            assert row['pilot_shots'] == sum(row['pilot_totals'])
            assert all(type(s) is int and 0 <= s <= n
                       for s,n in zip(row['pilot_counts'],row['pilot_totals']))
            assert len(row['axis_successes']) == 2
            assert all(type(s) is int and 0 <= s <= 256 for s in row['axis_successes'])
            counts = row['counts']
            assert len(counts) == 4 and all(type(n) is int and n >= 0 for n in counts)
            sm,nm,sp,np_ = counts
            assert sm <= nm and sp <= np_
            assert row['shots'] == row['pilot_shots']+nm+np_
            assert row['pair_counts'] is None
            assert row['decision'] in ('accept','reject','abstain')
            if row['phase'] == 'early':
                assert route['action'] == 'stop'
                assert row['decision'] == route['decision'] == row['certificate']['decision']
                assert row['certificate'] == route['certificate']
                assert row['counts'] == [0]*4
                assert row['axis_shots'] == 0 and row['axis_successes'] == [0,0]
                assert row['shots'] == row['pilot_shots'] == early_shots
                assert row['pilot_totals'] == [0,sum(pc),0,sum(pc)]
                assert row['pilot_counts'] == [0,pc[1]+pc[3],0,pc[2]+pc[3]]
                assert row['look_index'] == row['early_look_index']
                close(row['t'],math.pi/2)
                assert row['allocation_plus'] == .5 and row['estimated_variance'] is None
                early_stops += 1
            else:
                assert row['phase'] == 'holdout'
                assert route['action'] == 'complete_pilot' and route['decision'] == 'abstain'
                assert early_shots == row['axis_shots'] == 512
                assert row['pilot_shots'] == 1024
                fitted_counts,fitted_totals = complete_pilot(pc,row['axis_successes'])
                assert row['pilot_counts'] == fitted_counts.tolist()
                assert row['pilot_totals'] == fitted_totals.tolist() == [256]*4
                assert 1 <= row['look_index'] <= len(schedule)
                assert row['shots'] == schedule[row['look_index']-1]
                assert 0 < row['t'] < math.pi and 0 < row['allocation_plus'] < 1
                assert min(nm,np_) > 0
                stage2 = row['shots']-1024
                plus_target = max(1,min(stage2-1,math.floor(row['allocation_plus']*stage2)))
                assert np_ == plus_target and nm == stage2-plus_target
                recert = fisher_certificate(*counts,look_index=row['look_index'],
                                            look_count=len(schedule),delta=.005)
                if row['decision'] == 'abstain':
                    assert row['certificate'] is None
                    assert row['shots'] == spec['total_shot_cap']
                    assert recert['decision'] == 'abstain'
                    holdout_abstentions += 1
                else:
                    assert row['certificate'] == recert and row['decision'] == recert['decision']
                    holdout_stops += 1
            assert row['shots'] <= spec['total_shot_cap']
    assert len(seen) == summary['trial_count'] == len(cases)*len(methods)*spec['repetitions']
    assert reused == summary['reused_baseline_trials'] == len(originals)
    assert early_stops+holdout_stops+holdout_abstentions == summary['new_wrapped_trials']
    cells = {(cell['case_id'],cell['method']):cell for cell in summary['cells']}
    assert len(cells) == len(groups)
    designs = shared_routing_rows = 0
    for case in cases:
        routing = runner.routing_for_case(case,spec['repetitions'])
        for method in methods:
            key = case['case_id'],method
            rows = sorted(groups[key],key=lambda row:row['rep'])
            groups[key] = rows
            assert [row['rep'] for row in rows] == list(range(spec['repetitions']))
            independent = independently_summarize(rows,case)
            for field,value in independent.items():
                compare_json(cells[key][field],value,str(key)+'/'+field)
            if method in runner.METHODS[:2]:
                assert cells[key]['baseline_reused'] is True
                continue
            seed = 91000000+case['case_id']*1009+methods.index(method)*71
            for i,row in enumerate(rows):
                assert row['validation_seed'] == seed
                assert row['early_seed'] == routing['early_seed']
                assert row['axis_seed'] == routing['axis_seed']
                for field in ('pilot_counts','pilot_totals','axis_successes'):
                    assert row[field] == routing[field][i].tolist(), (key,i,field)
                assert row['early_pair_counts'] == routing['pair_counts'][i].tolist()
                assert row['early_certificate'] == routing['certificates'][i]
                assert row['early_look_index'] == int(routing['early_looks'][i])
                assert (row['phase'] == 'holdout') == bool(routing['active'][i])
                shared_routing_rows += 1
            hard = np.flatnonzero(routing['active'])
            shifts = allocations = 0
            if len(hard):
                design = holdout_design(method,routing['pilot_counts'][hard],routing['pilot_totals'][hard])
                for j,i in enumerate(hard):
                    for field in ('t','allocation_plus','estimated_variance'):
                        close(rows[i][field],float(design[field][j]))
                    designs += 1
                shifts = int(np.sum(design['shift_was_clipped']))
                allocations = int(np.sum(design['allocation_was_clipped']))
            assert cells[key]['early_stops'] == len(rows)-len(hard)
            assert cells[key]['entered_holdout'] == len(hard)
            assert cells[key]['shift_clip_count'] == shifts
            assert cells[key]['allocation_clip_count'] == allocations
            close(cells[key]['mean_t'],math.fsum(row['t'] for row in rows)/len(rows))
            close(cells[key]['mean_allocation_plus'],math.fsum(row['allocation_plus'] for row in rows)/len(rows))
    compare_json(summary['aggregate'],runner.aggregate(summary['cells'],cases),'aggregate')
    main_ids = [case['case_id'] for case in cases if case['kind'] == 'main']
    for method in methods:
        ratio = math.exp(math.fsum(math.log(cells[cid,method]['mean_shots']/cells[cid,'learned_closed_form']['mean_shots'])
                                   for cid in main_ids)/len(main_ids))
        close(ratio,summary['aggregate']['geomean_cost_ratios_baseline_over_closed'][method])
    print(json.dumps({'ledger_passed':len(seen),'new_wrapped':summary['new_wrapped_trials'],
                      'reused_controls':reused,'replaying_cases':replay_ids}),flush=True)
    replayed = 0
    case_by_id = {case['case_id']:case for case in cases}
    for cid in replay_ids:
        for method in POLICIES:
            cell,rows = runner.run_cell(case_by_id[cid],method,spec['repetitions'])
            assert rows == groups[cid,method], (cid,method,'replay')
            for field,value in cell.items():
                if field != 'planning_seconds_batch':
                    compare_json(cells[cid,method][field],value,'replay/'+field)
            replayed += len(rows)
        print(json.dumps({'replayed_case':cid,'new_records_so_far':replayed}),flush=True)
    assert sha256(this_source) == source_hash_at_start, 'Audit source changed during execution'
    audit = {'status':'passed','ledger_records':len(seen),'new_wrapped_trajectories':summary['new_wrapped_trials'],
             'reused_control_records_exactly_matched':reused,
             'early_stop_certificates_recomputed':early_stops,
             'holdout_decision_certificates_recomputed':holdout_stops,
             'full_cap_holdout_abstention_certificates_recomputed':holdout_abstentions,
             'unique_early_transcript_certificates_recomputed':len(early_cache),
             'common_routing_records_reproduced':shared_routing_rows,
             'count_only_holdout_designs_recomputed':designs,
             'cells_and_cost_summaries_recomputed':len(cells),
             'vector_seed_replay_new_records':replayed,'replayed_case_ids':replay_ids,
             'run_source_hashes_verified':len(spec['source_sha256']),
             'elapsed_seconds':time.perf_counter()-started,
             'risk_review':'Early anytime PSR .005 plus fresh conditional holdout .005 gives total <=.01, including equality. Survival-dependent fitting affects power, not conditional holdout safety.',
             'cost_review':'Early stops pay actual early shots. Survivors pay512early+512axes+holdout. Early counts reused only for design, with no double-counting or holdout overlap.',
             'fairness_review':'Four wrapped policies share identical routing/pilot realizations. Original controls are reused and counted separately from new trajectories.',
             'scope':'Development implementation audit; no simultaneous empirical performance qualification or SOTA claim.',
             'artifact_sha256':{name:sha256(directory/name) for name in ('spec.json','summary.json','trials.jsonl')},
             'audit_source_path':str(this_source.relative_to(ROOT)).replace('\\','/'),
             'audit_source_sha256':source_hash_at_start,
             'audit_helper_source_sha256':sha256(this_source.with_name('audit_validation.py')),
             'audit_source_unchanged_during_execution':True}
    audit.update(frozen_files())
    (directory/'audit.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(audit,indent=2,sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-directory',default=str(ROOT/'results/learned_validation/development_v2'))
    parser.add_argument('--replay-cases',default='0,42,49')
    args = parser.parse_args()
    main(args.run_directory,[int(x) for x in args.replay_cases.split(',') if x])
