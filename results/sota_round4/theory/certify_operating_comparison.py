"""Certified all-pair operating characteristics of four frozen methods.

Runs only deterministic probability recursions. Case9 is emitted first.
"""
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'experiments/sota_round4'))
import interval_sign_cost as target
sys.path.insert(0, str(ROOT/'experiments/circuit_validation'))
from direct_tests import _Bounds

EXPECTED_TARGET = '14a01510987617c05e37d991c995698e99d8f6cc28d5e7e7c5afe51512b48c40'
INPUT = ROOT/'results/sota_round4/analytic_prior_comparison.json'
PLAN = ROOT/'results/sota_round3/paired_benchmark/preliminary_plan.json'
OUTPUT = ROOT/'results/sota_round4/certified_operating_comparison.json'
CASE9_OUTPUT = ROOT/'results/sota_round4/certified_operating_case9.json'
METHOD_PATHS = {
    'beta_exact': 'results/sota_round3/cap16384_preparation/beta_exact.json',
    'sprt_0.7': 'results/sota_round4/lr_mixture_boundaries/sprt_0.7.json',
    'mix_0.75_0.99_w0.5': 'results/sota_round4/lr_mixture_boundaries/mix_0.75_0.99_w0.5.json',
    'mix_0.7_0.9_0.99_equal': 'results/sota_round4/lr_mixture_boundaries/mix_0.7_0.9_0.99_equal.json',
}
CASE_ORDER = (9, 7, 19, 21, 31, 33, 43, 45)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def enclosed(pair):
    return _Bounds(Decimal.from_float(pair[0]), Decimal.from_float(pair[1]))


def describe(value):
    return {'lower': str(value.lo), 'upper': str(value.hi)}


def aggregate(rows):
    bykey = {(r['method'],r['case_id']):r for r in rows}
    case_ids = sorted({r['case_id'] for r in rows})
    output = []
    for method in METHOD_PATHS:
        values = [bykey[method,c] for c in case_ids]
        costs = [enclosed(r['operating_intervals']['expected_shots']) for r in values]
        powers = [enclosed(r['operating_intervals']['plus']) for r in values]
        total_cost, total_power = _Bounds(0), _Bounds(0)
        for value in costs:
            total_cost += value
        for value in powers:
            total_power += value
        ratios = {}
        for reference in ('beta_exact','sprt_0.7'):
            logarithm = _Bounds(0)
            per_case = []
            for case_id, cost in zip(case_ids,costs):
                denominator = enclosed(bykey[reference,case_id]['operating_intervals']['expected_shots'])
                ratio = cost/denominator
                logarithm += ratio.log()
                per_case.append({'case_id':case_id, 'ratio':describe(ratio)})
            ratios[reference] = {'geometric_mean_cost_ratio':describe((logarithm/len(case_ids)).exp()),
                                 'case_ratios':per_case}
        output.append({'method':method, 'case_count':len(case_ids),
                       'equal_case_mean_expected_shots':describe(total_cost/len(case_ids)),
                       'equal_case_mean_correct_probability':describe(total_power/len(case_ids)),
                       'minimum_case_correct_probability':{
                           'lower':str(min(p.lo for p in powers)), 'upper':str(min(p.hi for p in powers))},
                       'each_case_correct_probability_certified_at_least_95_percent':all(p.lo>=Decimal('.95') for p in powers),
                       'cost_ratios':ratios})
    return output


def main():
    started = time.perf_counter()
    assert sha(target.__file__) == EXPECTED_TARGET
    comparison = json.loads(INPUT.read_text(encoding='utf-8'))
    assert comparison['status'] == 'complete_fixed_prior_float_analytic_comparison'
    assert comparison['pair_cap'] == 8192 and comparison['physical_shot_cap'] == 16384
    for path, expected in comparison['input_sha256'].items():
        assert sha(ROOT/path) == expected
    for path, expected in comparison['source_hashes'].items():
        assert sha(ROOT/path) == expected
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    cases = {c['case_id']:c for c in plan['cases']}
    floats = {(r['method'],r['case_id']):r for r in comparison['rows']}
    audits = {
        'interval':ROOT/'results/sota_round4/theory/audit_interval_sign_cost.json',
        'mixture':ROOT/'results/sota_round4/theory/audit_lr_mixture_tables.json',
        'beta':ROOT/'results/sota_round2/theory/cap16384_preparation_independent_audit.json'}
    for path in audits.values():
        assert json.loads(path.read_text(encoding='utf-8'))['status'] == 'passed'
    assert json.loads(audits['interval'].read_text(encoding='utf-8'))['target_source_sha256'] == EXPECTED_TARGET
    artifacts = {name:json.loads((ROOT/path).read_text(encoding='utf-8')) for name,path in METHOD_PATHS.items()}
    source_paths = [Path(__file__).resolve(), Path(target.__file__), INPUT, PLAN,
                    ROOT/'experiments/circuit_validation/direct_tests.py', *audits.values(),
                    *[ROOT/path for path in METHOD_PATHS.values()]]
    input_hashes = {p.relative_to(ROOT).as_posix():sha(p) for p in source_paths}
    result_rows = []
    for case_id in CASE_ORDER:
        case = cases[case_id]
        assert case['kind'] == 'main' and case['C'] > 0
        pm, pp = float(case['A']-case['C']), float(case['A']+case['C'])
        mirror = cases[case_id+1000]
        assert mirror['C'] == -case['C'] and mirror['A'] == case['A']
        assert float(mirror['A']-mirror['C']) == pp and float(mirror['A']+mirror['C']) == pm
        for method, artifact in artifacts.items():
            cell_start = time.perf_counter()
            assert artifact['max_discordances'] == 8192
            table = artifact['boundaries']
            target.validate_table(table,8192)
            got = target.operating_intervals(table,Fraction.from_float(pm),Fraction.from_float(pp),8192)
            reference = floats[method,case_id]
            assert reference['p_minus'] == pm and reference['p_plus'] == pp
            checks = {}
            for key, float_key in (('plus','positive_probability'),('minus','negative_probability'),
                                   ('abstain','abstain_probability'),('expected_shots','expected_shots')):
                value = reference[float_key]
                inside = got[key][0] <= value <= got[key][1]
                checks[key] = {'reference_float':value,'inside_outward_interval':inside}
                assert inside, (method,case_id,key,value,got[key])
            # Verify the input float mirror uses exactly swapped binary64 arms.
            mirrored_float = floats[method,case_id+1000]
            assert mirrored_float['p_minus'] == pp and mirrored_float['p_plus'] == pm
            row = {'method':method,'case_id':case_id,'mirrored_case_id':case_id+1000,
                   'pair_cap':8192,'physical_shot_cap':16384,
                   'source_binary64_probabilities':{
                       'pminus_hex':pm.hex(),'pplus_hex':pp.hex(),
                       'pminus_rational':str(Fraction.from_float(pm)),
                       'pplus_rational':str(Fraction.from_float(pp))},
                   'boundary_path':METHOD_PATHS[method],'boundary_sha256':sha(ROOT/METHOD_PATHS[method]),
                   'operating_intervals':got,'floating_crosscheck':checks,
                   'correct_probability_at_least_95_percent':Fraction(got['plus'][0])>=Fraction(19,20),
                   'elapsed_seconds':time.perf_counter()-cell_start}
            result_rows.append(row)
            print(json.dumps({'method':method,'case_id':case_id,'correct_probability':got['plus'],
                              'expected_shots':got['expected_shots'],'seconds':row['elapsed_seconds']}),flush=True)
        if case_id == 9:
            case9 = {'status':'certified_four_method_case9_operating_intervals',
                     'rows':list(result_rows),'aggregates':aggregate(result_rows),
                     'source_sha256':input_hashes,'new_simulation_observations':0}
            write_new(CASE9_OUTPUT,case9)
            print(json.dumps({'case9_artifact':str(CASE9_OUTPUT),'sha256':sha(CASE9_OUTPUT),
                              'elapsed_seconds':time.perf_counter()-started}),flush=True)
    assert all(sha(ROOT/path)==digest for path,digest in input_hashes.items())
    result = {'status':'certified_deterministic_operating_comparison',
              'pair_cap':8192,'physical_shot_cap':16384,'methods':list(METHOD_PATHS),
              'positive_case_ids':sorted(CASE_ORDER),'calculated_cells':len(result_rows),
              'mirror_equivalent_cells':len(result_rows),
              'new_simulation_observations':0,'rows':result_rows,'aggregates':aggregate(result_rows),
              'source_sha256':input_hashes,'elapsed_seconds':time.perf_counter()-started,
              'mirror_proof':{
                  'input_check':'For every registered positive case and its mirror, the actual binary64 pminus and pplus passed to the simulator are exactly swapped.',
                  'argument':'Under arm swap the discordance probability r is unchanged and conditional sign probability q becomes1-q exactly. Map each latent sign word to its complement. The symmetric integer boundary maps plus crossing to minus crossing at the same discordance index. Physical waiting times and cap are unchanged. Therefore correct-report probability, wrong-report probability, abstention and expected paid shots are exactly equal. The numerical intervals computed for each positive case enclose the same mirror quantities.',
                  'not_recomputed':'Mirror numerical recursions are not run; equivalence follows from exact input arm-swap and the symmetric table identity.'},
              'interpretation':{
                  'protocol':'Same stationary independent Bernoulli arms, inspection after EVERY completed physical pair, one fixed table per method, same cap16384. These are not the old sparse-checkpoint or physical-prefix-gate protocols.',
                  'cost':'Unconditional expected physical shots, including wrong reports and cap abstentions. No successful-trajectory conditioning.',
                  'risk':'The fixed tables already have exact fair-null crossing risk<=.01 and the reviewed symmetric-tail guarantee. This script computes alternative operating characteristics; it does not calibrate or select a new table.',
                  'qualification':'Power>=.95 claims use the outward LOWER probability endpoint independently in every positive case. Other evaluated requirements are visible per row; equal-case means do not hide a weaker case.',
                  'arithmetic':'Certified interval_sign_cost source was independently reviewed and exact-checked. Every one of the128 supplied floating main statistics lies in its new enclosure. Aggregate arithmetic and log/exp geometric ratios use64-digit outward Decimal intervals.',
                  'claim_boundary':'Four specified classical/fixed-prior controls on eight registered model points and exact sign mirrors. Deterministic simulated-model probabilities are not hardware evidence, held-out selection validation, or field-wide SOTA superiority.'}}
    write_new(OUTPUT,result)
    print(json.dumps({'artifact':str(OUTPUT),'sha256':sha(OUTPUT),
                      'elapsed_seconds':result['elapsed_seconds'],'aggregates':result['aggregates']}),flush=True)


if __name__ == '__main__':
    main()
