"""Complete the certified comparison, preserving case9 and raw float differences.

The first runner emitted its immutable case9 artifact, then stopped because a
tiny floating wrong-sign probability fell outside the new exact-model interval.
This runner records every such difference without widening certified intervals.
"""
from fractions import Fraction
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
import certify_operating_comparison as base


def main():
    started = time.perf_counter()
    assert base.sha(base.target.__file__) == base.EXPECTED_TARGET
    case9 = json.loads(base.CASE9_OUTPUT.read_text(encoding='utf-8'))
    assert case9['status'] == 'certified_four_method_case9_operating_intervals'
    for path, expected in case9['source_sha256'].items():
        assert base.sha(base.ROOT/path) == expected
    comparison = json.loads(base.INPUT.read_text(encoding='utf-8'))
    plan = json.loads(base.PLAN.read_text(encoding='utf-8'))
    cases = {c['case_id']:c for c in plan['cases']}
    floats = {(r['method'],r['case_id']):r for r in comparison['rows']}
    artifacts = {name:json.loads((base.ROOT/path).read_text(encoding='utf-8')) for name,path in base.METHOD_PATHS.items()}
    input_hashes = dict(case9['source_sha256'])
    for path in (Path(__file__).resolve(),base.CASE9_OUTPUT):
        input_hashes[path.relative_to(base.ROOT).as_posix()] = base.sha(path)
    result_rows = list(case9['rows'])
    outside = []
    for case_id in base.CASE_ORDER:
        if case_id == 9:
            continue
        case,mirror = cases[case_id],cases[case_id+1000]
        pm,pp = float(case['A']-case['C']),float(case['A']+case['C'])
        assert case['kind']=='main' and case['C']>0
        assert mirror['C']==-case['C'] and mirror['A']==case['A']
        assert float(mirror['A']-mirror['C'])==pp and float(mirror['A']+mirror['C'])==pm
        for method,artifact in artifacts.items():
            cell_start = time.perf_counter()
            assert artifact['max_discordances']==8192
            got = base.target.operating_intervals(artifact['boundaries'],Fraction.from_float(pm),Fraction.from_float(pp),8192)
            reference = floats[method,case_id]
            assert reference['p_minus']==pm and reference['p_plus']==pp
            checks = {}
            for key,float_key in (('plus','positive_probability'),('minus','negative_probability'),
                                  ('abstain','abstain_probability'),('expected_shots','expected_shots')):
                value = reference[float_key]
                inside = got[key][0]<=value<=got[key][1]
                check = {'reference_float':value,'inside_outward_interval':inside,
                         'distance_outside_interval_float_diagnostic':max(got[key][0]-value,value-got[key][1],0.)}
                checks[key] = check
                if not inside:
                    outside.append({'method':method,'case_id':case_id,'quantity':key,
                                    'strict_interval':got[key],**check})
                # A raw floating approximation is not itself a rigorous
                # enclosure of the exact model. Retain differences visibly.
            mirrored_float = floats[method,case_id+1000]
            assert mirrored_float['p_minus']==pp and mirrored_float['p_plus']==pm
            row = {'method':method,'case_id':case_id,'mirrored_case_id':case_id+1000,
                   'pair_cap':8192,'physical_shot_cap':16384,
                   'source_binary64_probabilities':{
                       'pminus_hex':pm.hex(),'pplus_hex':pp.hex(),
                       'pminus_rational':str(Fraction.from_float(pm)),
                       'pplus_rational':str(Fraction.from_float(pp))},
                   'boundary_path':base.METHOD_PATHS[method],
                   'boundary_sha256':base.sha(base.ROOT/base.METHOD_PATHS[method]),
                   'operating_intervals':got,'floating_crosscheck':checks,
                   'correct_probability_at_least_95_percent':Fraction(got['plus'][0])>=Fraction(19,20),
                   'elapsed_seconds':time.perf_counter()-cell_start}
            result_rows.append(row)
            print(json.dumps({'method':method,'case_id':case_id,'correct_probability':got['plus'],
                              'expected_shots':got['expected_shots'],'seconds':row['elapsed_seconds']}),flush=True)
    assert len(result_rows)==32
    assert all(base.sha(base.ROOT/path)==digest for path,digest in input_hashes.items())
    result = {'status':'certified_deterministic_operating_comparison',
              'pair_cap':8192,'physical_shot_cap':16384,'methods':list(base.METHOD_PATHS),
              'positive_case_ids':sorted(base.CASE_ORDER),'calculated_cells':len(result_rows),
              'mirror_equivalent_cells':len(result_rows),'new_simulation_observations':0,
              'rows':result_rows,'aggregates':base.aggregate(result_rows),
              'source_sha256':input_hashes,'elapsed_seconds':time.perf_counter()-started,
              'floating_crosscheck_summary':{
                  'compared_quantities':128,'inside_count':128-len(outside),'outside_count':len(outside),
                  'outside_details':outside,
                  'main_cost_and_correct_probability_all_inside':not any(r['quantity'] in ('plus','expected_shots') for r in outside),
                  'intervals_widened_to_match_floats':False,
                  'interpretation':'The old calculation rounds r and q in binary64 and uses approximate SciPy binomial tails and floating sign DP. These numbers need not lie inside a much narrower certified interval for the exact binary64-arm model. Differences are recorded without substituting or padding the rigorous intervals.'},
              'mirror_proof':{
                  'input_check':'Each registered mirror swaps the actual binary64 pminus and pplus exactly.',
                  'argument':'Arm swap leaves discordance r unchanged and sends conditional q to1-q. Complementing every sign word swaps plus/minus crossings at the same discordance index because the integer table is symmetric. Physical waiting and cap are unchanged. Thus correct probability, wrong probability, abstention and total expected cost are exactly identical for mirrored cases.',
                  'not_recomputed':'Mirror recursions were not run; equality follows from exact arm-swap input checks and the symmetric boundary identity.'},
              'interpretation':{
                  'protocol':'All four methods use stationary independent Bernoulli arms, EVERY physical-pair inspection, one frozen table each, and common cap16384. This is not the old sparse-checkpoint or physical-prefix-gate protocol.',
                  'cost':'Unconditional expected physical shots including wrong reports and abstentions.',
                  'risk':'All four saved boundaries have previously independently checked exact null crossing probability<=.01 and the reviewed symmetric-tail guarantee. This artifact does not recalibrate or select a new table.',
                  'qualification':'Each-case power>=.95 uses its rigorous lower endpoint. Aggregate means do not replace this check.',
                  'arithmetic':'Operating values use the independently audited interval_sign_cost source. Geometric means, ratios and aggregate means use64-digit outward Decimal arithmetic.',
                  'claim_boundary':'Four named fixed-prior classical controls, eight registered positive model points and exact sign mirrors. Deterministic model analysis is not hardware evidence, held-out selection validation, or a field-wide SOTA comparison.'}}
    base.write_new(base.OUTPUT,result)
    print(json.dumps({'artifact':str(base.OUTPUT),'sha256':base.sha(base.OUTPUT),
                      'elapsed_seconds':result['elapsed_seconds'],'floating_crosscheck':result['floating_crosscheck_summary'],
                      'aggregates':result['aggregates']}),flush=True)


if __name__=='__main__':
    main()
