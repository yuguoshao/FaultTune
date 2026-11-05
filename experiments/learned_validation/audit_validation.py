"""Audit a completed paid-learning development benchmark without changing it."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/circuit_validation'))
from direct_tests import certificate_from_pairs
from certificates import fisher_certificate
from designs import choose_design, config_dict
from kl_designs import choose_kl_design
import run_validation as runner


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=3e-14, abs_tol=1e-12), (actual, expected)


def cp(successes, trials):
    return [0. if successes == 0 else float(beta.ppf(.025, successes, trials-successes+1)),
            1. if successes == trials else float(beta.ppf(.975, successes+1, trials-successes))]


def compare_json(actual, expected, path=''):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            compare_json(actual[key], expected[key], path+'/'+key)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for index, (a,b) in enumerate(zip(actual,expected)):
            compare_json(a,b,path+'/'+str(index))
    elif isinstance(expected, float):
        close(actual,expected)
    else:
        assert actual == expected, (path,actual,expected)


def independently_summarize(rows, case):
    truth = 'accept' if case['C'] < 0 else 'reject' if case['C'] > 0 else 'abstain'
    n = len(rows)
    wrong = sum(row['decision'] != 'abstain' and row['decision'] != truth for row in rows)
    correct = sum(row['decision'] == truth for row in rows) if truth != 'abstain' else 0
    abstained = sum(row['decision'] == 'abstain' for row in rows)
    costs = sorted(row['shots'] for row in rows)
    correct_costs = sorted(row['shots'] for row in rows if row['decision'] == truth and truth != 'abstain')
    rank = math.ceil(.95*n)
    n95 = correct_costs[rank-1] if len(correct_costs) >= rank else None
    return {'repetitions':n,'wrong':wrong,'correct':correct,'abstained':abstained,
            'mean_shots':math.fsum(costs)/n,
            'median_shots':float(costs[n//2] if n%2 else (costs[n//2-1]+costs[n//2])/2),
            'min_shots':costs[0],'max_shots':costs[-1],
            'error_ci95_pointwise':cp(wrong,n),
            'power_ci95_pointwise':cp(correct,n) if truth != 'abstain' else None,
            'empirical_N95_correct':n95,'N95_censored':n95 is None}


def frozen_files():
    old = []
    archive_path = ROOT/'experiments/circuit_validation/confirmation_spec_v1_sources.zip'
    with zipfile.ZipFile(str(archive_path)) as archive:
        for name in archive.namelist():
            if not name.endswith('.py'):
                continue
            target = ROOT/'experiments/circuit_validation'/name
            if not target.is_file():
                target = ROOT/name
            assert target.is_file(), name
            assert hashlib.sha256(archive.read(name)).hexdigest() == sha256(target), name
            old.append(name)
    phase_path = ROOT/'results/phase_validation/exploration_manifest.json'
    phase = json.loads(phase_path.read_text(encoding='utf-8'))
    for entry in phase['files']:
        target = ROOT/entry['path']
        assert target.stat().st_size == entry['bytes'], entry['path']
        assert sha256(target) == entry['sha256'], entry['path']
    return {'old_frozen_python_sources_unchanged':len(old),
            'previous_phase_manifest_files_unchanged':len(phase['files'])}


def main(run_directory, replay_ids):
    started = time.perf_counter()
    directory = Path(run_directory).resolve()
    spec = json.loads((directory/'spec.json').read_text(encoding='utf-8'))
    summary = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    assert summary['spec'] == spec
    for name, digest in spec['source_sha256'].items():
        assert sha256(ROOT/name) == digest, name
    assert spec['source_sha256'] == runner.source_hashes()
    assert spec['cases'] == runner.case_matrix()
    assert spec['methods'] == list(runner.METHODS)
    assert spec['design_config'] == config_dict()
    assert spec['total_shot_checkpoints'] == runner.budget_schedule()
    assert spec['pilot_per_angle'] == runner.PILOT_PER_ANGLE
    assert spec['total_shot_cap'] == runner.CAP
    assert spec['delta'] == runner.DELTA
    cases, methods = spec['cases'], spec['methods']
    case_by_id = {case['case_id']:case for case in cases}
    groups = {(case['case_id'],method):[] for case in cases for method in methods}
    seen = set()
    schedule = spec['total_shot_checkpoints']
    certified = abstain_rechecked = pair_tables = 0
    for line in (directory/'trials.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        key = row['case_id'],row['method'],row['rep']
        assert key not in seen, key
        seen.add(key)
        assert key[:2] in groups
        groups[key[:2]].append(row)
        counts = row['counts']
        assert len(counts) == 4 and all(type(n) is int and n >= 0 for n in counts)
        sm,nm,sp,np_ = counts
        assert sm <= nm and sp <= np_ and min(nm,np_) > 0
        learned = row['method'].startswith('learned_')
        expected_pilot = 4*spec['pilot_per_angle'] if learned else 0
        assert row['pilot_shots'] == expected_pilot
        assert row['pilot_totals'] == [spec['pilot_per_angle'] if learned else 0]*4
        assert len(row['pilot_counts']) == 4
        assert all(type(s) is int and 0 <= s <= n for s,n in zip(row['pilot_counts'],row['pilot_totals']))
        assert sum(row['pilot_totals']) == row['pilot_shots']
        assert row['shots'] == expected_pilot+nm+np_
        assert 1 <= row['look_index'] <= len(schedule)
        assert row['shots'] == schedule[row['look_index']-1]
        assert row['shots'] <= spec['total_shot_cap']
        stage2 = row['shots']-row['pilot_shots']
        target_plus = min(stage2-1,max(1,math.floor(row['allocation_plus']*stage2)))
        assert np_ == target_plus and nm == stage2-target_plus
        assert 0 < row['t'] < math.pi and 0 < row['allocation_plus'] < 1
        assert row['decision'] in ('accept','reject','abstain')
        if row['method'] == 'psr_mixture':
            pc = row['pair_counts']
            assert len(pc) == 4 and all(type(n) is int and n >= 0 for n in pc)
            assert nm == np_ == sum(pc)
            assert sm == pc[2]+pc[3] and sp == pc[1]+pc[3]
            recert = certificate_from_pairs(pc,delta=spec['delta'])
            pair_tables += 1
        else:
            assert row['pair_counts'] is None
            recert = fisher_certificate(*counts,look_index=row['look_index'],
                                        look_count=len(schedule),delta=spec['delta'])
        if row['decision'] != 'abstain':
            assert row['certificate'] == recert, key
            assert row['decision'] == recert['decision']
            certified += 1
        else:
            assert row['certificate'] is None
            assert row['shots'] == spec['total_shot_cap']
            assert recert['decision'] == 'abstain', key
            abstain_rechecked += 1
    expected_count = len(cases)*len(methods)*spec['repetitions']
    assert len(seen) == expected_count == summary['trial_count']
    cells = {(cell['case_id'],cell['method']):cell for cell in summary['cells']}
    assert len(cells) == len(groups)
    designs_checked = pilot_draws_checked = 0
    for case in cases:
        pilot,totals,pilot_seed = runner.pilot_for_case(case,spec['repetitions'])
        for method in methods:
            key = case['case_id'],method
            rows = sorted(groups[key],key=lambda row:row['rep'])
            groups[key] = rows
            assert [row['rep'] for row in rows] == list(range(spec['repetitions']))
            learned = method.startswith('learned_')
            count_array = np.array([row['pilot_counts'] for row in rows],dtype=np.int64)
            total_array = np.array([row['pilot_totals'] for row in rows],dtype=np.int64)
            if learned:
                assert np.array_equal(count_array,pilot)
                assert np.array_equal(total_array,totals)
                pilot_draws_checked += len(rows)
            else:
                assert not np.any(count_array) and not np.any(total_array)
            design = choose_kl_design(count_array,total_array) if method == 'learned_kl_grid' else choose_design(
                method if learned else 'psr_uniform',count_array,total_array)
            expected_seed = 81000000+case['case_id']*1009+methods.index(method)*71
            for i,row in enumerate(rows):
                assert row['validation_seed'] == expected_seed
                for field in ('t','allocation_plus','estimated_variance'):
                    close(row[field],float(design[field][i]))
                designs_checked += 1
            independent = independently_summarize(rows,case)
            for field,value in independent.items():
                compare_json(cells[key][field],value,str(key)+'/'+field)
            assert cells[key]['pilot_seed'] == pilot_seed
            assert cells[key]['validation_seed'] == expected_seed
            assert cells[key]['planning_seconds_batch'] >= 0
            assert cells[key]['shift_clip_count'] == int(np.sum(design['shift_was_clipped']))
            assert cells[key]['allocation_clip_count'] == int(np.sum(design['allocation_was_clipped']))
            close(cells[key]['mean_t'],float(np.mean(design['t'])))
            close(cells[key]['mean_allocation_plus'],float(np.mean(design['allocation_plus'])))
    # Cross-check the published aggregate from independently checked cells.
    recomputed = runner.aggregate(summary['cells'],cases)
    compare_json(summary['aggregate'],recomputed,'aggregate')
    main_ids = [case['case_id'] for case in cases if case['kind'] == 'main']
    for method in methods:
        ratio = math.exp(math.fsum(math.log(cells[cid,method]['mean_shots']/
                                           cells[cid,'learned_closed_form']['mean_shots'])
                                  for cid in main_ids)/len(main_ids))
        close(summary['aggregate']['geomean_cost_ratios_baseline_over_closed'][method],ratio)
        for field, cell_field in (('main_wrong','wrong'),('main_correct','correct'),('main_abstained','abstained')):
            assert summary['aggregate']['methods'][method][field] == sum(cells[cid,method][cell_field] for cid in main_ids)
    print(json.dumps({'ledger_checks_passed':len(seen),'replaying_case_ids':replay_ids}),flush=True)
    replayed = 0
    for cid in replay_ids:
        assert cid in case_by_id
        for method in methods:
            new_cell,new_rows = runner.run_cell(case_by_id[cid],method,spec['repetitions'])
            assert new_rows == groups[cid,method], (cid,method,'seed replay mismatch')
            for field,value in new_cell.items():
                if field != 'planning_seconds_batch':
                    compare_json(cells[cid,method][field],value,'replay/'+field)
            replayed += len(new_rows)
        print(json.dumps({'replayed_case':cid,'records_so_far':replayed}),flush=True)
    audit = {'status':'passed','ledger_records':len(seen),
             'decision_certificates_recomputed':certified,
             'full_cap_abstention_certificates_recomputed':abstain_rechecked,
             'paired_tables_checked':pair_tables,'count_only_designs_recomputed':designs_checked,
             'learned_pilot_draws_reproduced':pilot_draws_checked,
             'cells_and_cost_summaries_recomputed':len(cells),
             'vector_seed_replay_records':replayed,'replayed_case_ids':replay_ids,
             'run_source_hashes_verified':len(spec['source_sha256']),
             'elapsed_seconds':time.perf_counter()-started,
             'review':{'paid_learning':'All pilot shots charged separately to each learned method; holdout counts exclude pilot.',
                       'cost':'All methods stop on identical total-cost checkpoints; abstention charged full cap.',
                       'orientation':'All design shifts lie in (0,pi), so validation sign equals target sign.',
                       'risk':'Fisher schedule and allocation fixed conditional on pilot; delta/(2L) per direction and registered look.',
                       'screen':'Floating screen may delay an exact-tail crossing; it never authorizes a decision.',
                       'replay':'Full vector cells replayed because active repetitions share each cell PRNG stream.',
                       'limits':'Development evidence only; no simultaneous empirical performance qualification or SOTA claim.'},
             'artifact_sha256':{name:sha256(directory/name) for name in ('spec.json','summary.json','trials.jsonl')},
             'audit_source_sha256':sha256(Path(__file__).resolve())}
    audit.update(frozen_files())
    (directory/'audit.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(audit,indent=2,sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-directory',default=str(ROOT/'results/learned_validation/development_v1'))
    parser.add_argument('--replay-cases',default='0,5,70')
    args = parser.parse_args()
    main(args.run_directory,[int(x) for x in args.replay_cases.split(',') if x])
