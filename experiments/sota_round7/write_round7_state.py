"""Bind the completed mechanism screens and executable comparison controls."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = ROOT/'results/sota_round7/research_state.json'
    if output.exists():
        raise FileExistsError('Preserve completed state')
    paths = [
        'results/sota_round7/mechanisms/bitstring_nuisance_diagnostic.json',
        'results/sota_round7/mechanisms/bitstring_nuisance_peer_review.json',
        'results/sota_round7/mechanisms/bitstring_nuisance_applicability_addendum.json',
        'results/sota_round7/mechanisms/bitstring_invariance_cost.json',
        'results/sota_round7/mechanisms/bitstring_invariance_cost_independent_review.json',
        'results/sota_round7/mechanisms/noisy_multidepth_phase_obstruction.json',
        'results/sota_round7/mechanisms/noisy_multidepth_phase_budget_closure.json',
        'results/sota_round7/mechanisms/noisy_multidepth_root_review.json',
        'results/sota_round7/literature/direct_priority_and_resource_comparators.json',
        'results/sota_round7/mechanisms/resource_control_checks_v3.json',
        'results/sota_round7/mechanisms/resource_controls_peer_review.json'
    ]
    sources, checks = {}, []
    for rel in paths:
        path = ROOT/rel
        payload = json.loads(path.read_text(encoding='utf-8'))
        sources[rel] = digest(path)
        for field in ('source_sha256', 'input_sha256', 'source_files', 'frozen_sources_sha256'):
            for source, expected in payload.get(field, {}).items():
                if isinstance(expected, str) and len(expected) == 64:
                    assert digest(ROOT/source) == expected, (rel, source)
                    checks.append([rel, source])
    latest_checks = json.loads((ROOT/paths[-2]).read_text(encoding='utf-8'))
    peer = json.loads((ROOT/paths[-1]).read_text(encoding='utf-8'))
    assert latest_checks['status'] == 'passed' and latest_checks['tests_run'] == 7
    assert not peer.get('blocking_findings', [])
    assert not peer.get('blocking_issues', [])
    invariant = json.loads((ROOT/paths[3]).read_text(encoding='utf-8'))
    depth = json.loads((ROOT/paths[5]).read_text(encoding='utf-8'))
    freeze_paths = ['experiments/sota_round7/resource_controls.py',
                    'experiments/sota_round7/test_resource_controls.py',
                    'experiments/circuit_validation/direct_tests.py',
                    paths[-2], paths[-1]]
    archive = ROOT/'results/sota_round7/mechanisms/resource_controls_v3_sources.zip'
    with zipfile.ZipFile(str(archive), 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
        for rel in freeze_paths:
            bundle.write(str(ROOT/rel), rel)
    # The superseded checks retain their exact original source snapshots.
    old_versions = []
    for version, check_name in ((1, 'resource_control_checks.json'), (2, 'resource_control_checks_v2.json')):
        old_zip = ROOT/('results/sota_round7/mechanisms/resource_controls_v%s_sources.zip' % version)
        with zipfile.ZipFile(str(old_zip)) as bundle:
            old_check = json.loads(bundle.read('results/sota_round7/mechanisms/'+check_name).decode('utf-8'))
            for rel, expected in old_check['source_sha256'].items():
                raw = bundle.read(rel) if rel in bundle.namelist() else (ROOT/rel).read_bytes()
                assert hashlib.sha256(raw).hexdigest() == expected
        old_versions.append({'version': version, 'check_status': old_check['status'],
                             'archive': old_zip.relative_to(ROOT).as_posix(), 'sha256': digest(old_zip)})
    result = {
        'status': 'round7_mechanism_screens_reviews_and_controls_complete_goal_active',
        'updated_utc': datetime.now(timezone.utc).isoformat(),
        'objective': 'Continue trying to exceed applicable SOTA, retaining the original1percent error,95percent per-case correct-stop,threefold geometric cost,full-cost/abstention accounting and independent physical validation requirements.',
        'sota_goal_complete': False,
        'previous_goal_turn_classification': 'progress',
        'current_goal_turn_classification': 'progress',
        'new_simulation_observations': 0,
        'latest_operational_performance_changed': False,
        'latest_operational_result_source': 'results/sota_round6/distance_to_sota.json',
        'mechanism_results': {
            'full_bitstrings': {
                'positive_result': 'The same-duration disjoint three-qubit circuit has P_j(X,S)=Bern(x_j)*Q(S); with common s=Q(00)>0, Delta=s(x_new-x_old), so full readout permits nuisance cancellation. The enlarged common-Q information-ratio example is53.3230116211; a separately constructed physical sign-reflection pair has KL ratio55.9268095211.',
                'scope': 'These ratios compare information experiments/models and are not finite stopping-cost speedups or same-information SOTA wins. Exact-angle identities and floating gate checks remain distinguished.',
                'expanded_crosstalk_shot_lower': invariant['analytic_lower_bound']['necessary_expected_total_shots_lower'],
                'common_Q_control_shot_upper': invariant['common_Q_control']['physical_mean_upper'],
                'cross_model_invariance_cost_ratio_lower': invariant['invariance_tax']['expanded_class_necessary_cost_over_common_Q_feasible_cost_lower'],
                'mechanism_limit': 'An explicit stationary H(u) with amplitude-dependent spectator coupling makes the target null while preserving all X-only observations. Thus the cheap invariant-model control cannot be claimed uniformly safe under unknown crosstalk.'
            },
            'noisy_multidepth_phase': {
                'positive_result': 'On the declared score pair, a feasible multi-depth-resource control beats every depth1-only method by>15x in circuit counts and>3x in modeled device time.',
                'same_resource_speedup_limits': depth['maximum_possible_new_method_speedup_vs_strongest_qualified_comparator']['rounded_strict_upper_bounds'],
                'scope': 'The same multi-depth-resource comparison is bounded below3x by a common-null finite-error information proof and qualified fixed-depth controls. This applies to the declared noisy Ramsey family/score pair, not all phase-estimation problems.',
                'independent_review': 'Root verified the analytic chain, budget closure,130digit constants,4 matrix-relative-entropy calculations and3 symbolic identities.'
            }
        },
        'executable_controls': {
            'module': 'experiments/sota_round7/resource_controls.py',
            'controls': ['RamseyFixedDepthControl depth8', 'RamseyFixedDepthControl depth16', 'InvariantBitstringControl'],
            'contract': 'Paid-observation-only state machines; exact rational crossing fallback, first report absorbs, phase randomization, full readout counts, attempts charged on issue, unique pending outcome, failure/interruption handling, full-cap penalty for every undecided/uncompleted record, and exact resource ledgers.',
            'randomness_scope': 'Default Ramsey coins use system randomness under the explicit independent-fair-coin assumption. Fixed seeded replay is labelled deterministic and does not inherit a probability guarantee.',
            'validation': 'Seven deterministic test groups and independent source review passed. Tests are not Monte Carlo power/cost evidence.',
            'archive': archive.relative_to(ROOT).as_posix(), 'archive_sha256': digest(archive),
            'retained_development_versions': old_versions,
            'development_corrections': 'v1 missed a float-depth input boundary and incomplete-record score penalty. v2 fixed behavior but a test-edit placement error was retained as failed. v3 corrects the test placement and includes all reviewed accounting/seed-scope changes. Both earlier source/check snapshots are preserved.'
        },
        'literature': 'Tendick/Mitra originals did not supply the inspected fixed-basis KL reflected-pair result, but Baek2020 already defined the measurement resource quantity; priority remains unverified. Full-output and multi-depth comparators must receive equal information/actions. RPEv3 contains an erratum; its earlier constants are not used.',
        'claim_limits': [
            'No new candidate has passed the original independent performance acceptance. The previous geometric cost disadvantage remains.',
            'The two resource improvements are real within their contracts, but their advantages are also accessible to strong classical controls. They do not establish international-leading or SOTA performance.',
            'New explicit model families are feasibility probes; they do not replace or erase frozen old instances, failed preregistration checks or the original3x requirement.',
            'The exact common-Q control has demonstrated finite-cost/power bounds at its stated reference instance, not uniform95percent power over every common-Q distribution.',
            'Analytic model guarantees assume the stated physical observation law; actual execution failures remain failed trials with full-cap penalties.'
        ],
        'next_required': [
            'Develop a candidate mechanism whose gain survives the newly executable same-resource controls. Do not launch large performance experiments merely to rediscover the certified noisy-phase ceiling.',
            'For the promising full-bitstring direction, quantify which additional physically executable diagnostic action can identify amplitude-dependent spectator coupling at low paid cost, while giving it to model-aware likelihood/active-learning comparators too.',
            'Finish faithful full applicable literature controls before labelling any cost reduction SOTA. Preserve separate statuses for oracle information, theorem novelty and operational performance.',
            'If a candidate shows sufficient potential, freeze its implementation and independent physical benchmark with the original safety,power,cost and coverage criteria before full validation.'
        ],
        'source_binding_audit': {'checked_links': len(checks), 'failures': [], 'links': checks},
        'artifact_sha256': sources,
        'state_writer_sha256': digest(Path(__file__).resolve())
    }
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({'status': result['status'], 'links_checked': len(checks),
                      'tests': latest_checks['tests_run'], 'new_simulation_observations': 0}))


if __name__ == '__main__':
    main()
