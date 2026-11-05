"""Development replay of one frozen piecewise boundary, with real prefixes."""
from pathlib import Path
import json

import paired_benchmark as pilot
import piecewise_gate_boundary as piecewise
import replay_boundaries as bounded_summary

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'results/sota_round3/paired_benchmark'
OUT = ROOT/'results/sota_round3/piecewise_gate_replay'
NAME = 'physical_gate_m6_piecewise_D512'


def main():
    audit_path = ROOT/'results/sota_round3/paired_benchmark_audit/audit.json'
    audit = pilot.read_json(audit_path)
    if audit['status'] != 'passed':
        raise ValueError('Paired data audit has not passed')
    inputs = {pilot.relative(BASE/name): sha for name,sha in audit['inputs'].items()}
    for name,sha in {**inputs, **audit['sources']}.items():
        if pilot.digest(ROOT/name) != sha:
            raise ValueError('Audited input or source changed: '+name)
    boundary_path = ROOT/'results/sota_round3/piecewise_gate_boundary.json'
    boundary_audit_path = ROOT/'results/sota_round2/theory/piecewise_physical_gate_independent_audit.json'
    boundary_audit = pilot.read_json(boundary_audit_path)
    if (boundary_audit['status'] != 'passed'
            or boundary_audit['artifact_file_sha256'] != pilot.digest(boundary_path)):
        raise ValueError('Piecewise boundary differs from its passed independent audit')
    frozen_sources = piecewise.source_hashes()
    inputs.update(frozen_sources)
    for name,sha in boundary_audit['sources'].items():
        if pilot.digest(ROOT/name) != sha:
            raise ValueError('Boundary verifier source changed')
        inputs[name] = sha
    boundary = piecewise.verify_artifact(pilot.read_json(boundary_path))
    for path in (boundary_path, audit_path, boundary_audit_path, Path(__file__).resolve(), Path(pilot.__file__).resolve(),
                 Path(bounded_summary.__file__).resolve()):
        inputs[pilot.relative(path)] = pilot.digest(path)
    plan = pilot.read_json(BASE/'preliminary_plan.json')
    rows, seen = [], set()
    for line in (BASE/'paths.jsonl').open(encoding='utf-8'):
        source = json.loads(line)
        key = (source['case_id'], source['rep'])
        if key in seen or source['status'] != 'complete' or source['cap'] != 4096:
            raise ValueError('Invalid or duplicate original path')
        seen.add(key)
        observed = piecewise.replay_history(source['history'], boundary)
        censored = observed['decision'] is None
        lower = (next(n for n in plan['fine_checkpoints'] if n > source['shots_observed'])
                 if censored else observed['shots'])
        upper = source['cap'] if censored else observed['shots']
        reference = source['outcomes']['beta_exact']
        rows.append(dict(case_id=source['case_id'], rep=source['rep'], seed=source['seed'], kind=source['kind'],
                         method=NAME, path='shared_psr', reference_shots=reference['shots'],
                         reference_decision=reference['decision'], status=observed['status'],
                         decision=observed['decision'], shots_lower=lower, shots_upper=upper,
                         certificate=observed['certificate'], history_prefix_length=observed['history_prefix_length'],
                         checkpoint_history=observed['checkpoint_history']))
    expected = {(case['case_id'],rep) for case in plan['cases'] for rep in range(plan['repetitions'])}
    if seen != expected or len(rows) != 576:
        raise ValueError('Incomplete development matrix')
    result = {'status': 'complete_development_replay', 'input_sha256': inputs,
              'source_hashes': frozen_sources, 'new_simulation_observations': 0,
              'path_records': len(rows), 'right_censored': sum(r['decision'] is None for r in rows),
              'outcomes_relative_to_beta_exact': bounded_summary.summarize(rows, {c['case_id']:c for c in plan['cases']}),
              'claim_boundary': 'One development-selected fixed method on all 576 existing paths. No new seeds, held-out superiority claim, or SOTA conclusion.'}
    if result['right_censored'] == 0:
        baseline = [json.loads(line) for line in (BASE/'method_results.jsonl').open(encoding='utf-8')]
        added = [dict(case_id=r['case_id'],rep=r['rep'],seed=r['seed'],kind=r['kind'],method=NAME,
                      shots=r['shots_lower'],decision=r['decision'],certificate=r['certificate'],
                      counts=r['certificate']['counts'],history_prefix_length=r['history_prefix_length']) for r in rows]
        result['complete_combined_comparison'] = pilot.summarize(baseline+added,dict(plan,methods=plan['methods']+[NAME]))
        result['pathwise_comparison'] = {
            'fewer_shots':sum(r['shots_lower']<r['reference_shots'] for r in rows),
            'same_shots':sum(r['shots_lower']==r['reference_shots'] for r in rows),
            'more_shots':sum(r['shots_lower']>r['reference_shots'] for r in rows),
            'different_decision':sum(r['decision']!=r['reference_decision'] for r in rows)}
    for name,sha in inputs.items():
        if pilot.digest(ROOT/name) != sha:
            raise ValueError('Input changed during replay: '+name)
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'records.jsonl').open('x',encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row,allow_nan=False)+'\n')
    pilot.write_new(OUT/'summary.json',result)
    print(json.dumps({k:result[k] for k in ('path_records','right_censored','pathwise_comparison') if k in result}))
    if 'complete_combined_comparison' in result:
        comp=result['complete_combined_comparison']
        print(json.dumps(comp['equal_case_geometric_cost_ratios'][NAME]))
        print(json.dumps([r for r in comp['outcomes'] if r['method']==NAME and r['group'] in ('main','null')]))


if __name__ == '__main__':
    main()
