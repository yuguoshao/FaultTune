"""Independent fixed-grid evaluation of six controls and exact hybrid risk.

No sampling, producer solver import, candidate replacement, or approximate arm
reconstruction. All operating calls use exact Fraction r,q directly.
"""
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.sota_round5 import interval_hybrid_stopping as operating

DIR = ROOT/'results/sota_round5/hybrid_minimax'
PLAN = DIR/'independent_grid_plan.json'
CANDIDATE = DIR/'candidate_policy.json'
JOINT = DIR/'joint_null_exact.json'
JOINT_OUT = DIR/'independent_joint_null_review.json'
OUT = DIR/'independent_controls_a.json'
BETA = ROOT/'results/sota_round3/cap16384_preparation/beta_exact.json'
MANIFEST = ROOT/'results/sota_round4/lr_mixture_boundaries/calibration_manifest.json'
MIXTURE_AUDIT = ROOT/'results/sota_round4/theory/audit_lr_mixture_tables.json'
BETA_AUDIT = ROOT/'results/sota_round2/theory/cap16384_preparation_independent_audit.json'
HYBRID_AUDIT = ROOT/'results/sota_round5/theory/hybrid_stopping_interval_review.json'
METHODS = ('beta_exact', 'sprt_0.6', 'sprt_0.7', 'sprt_0.75', 'sprt_0.85', 'sprt_0.9')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def exact_record(value):
    return {'numerator': str(value.numerator), 'denominator': str(value.denominator)}


def directed_display(value, scale=10**15):
    low = value.numerator*scale//value.denominator
    high = -((-value.numerator*scale)//value.denominator)
    def show(n):
        integer, frac = divmod(n, scale)
        return str(integer)+'.'+str(frac).rjust(len(str(scale))-1, '0')
    return {'lower': show(low), 'upper': show(high), 'float_diagnostic_only': float(value)}


def write_new(path, obj):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write('\n')


def verify_hashes(mapping):
    for path, expected in mapping.items():
        assert sha(ROOT/path) == expected, path


def independent_full_integer_joint(prefix, denominator, boundary):
    """Full count-vector recurrence, no Fraction-prefix or suffix producer.

Every prefix step has denominator multiplier2D; each surviving incoming
integer count is multiplied byD-a. Each suffix step has multiplier2 and
removes its symmetric tails. Absorbed plus/minus integers are carried on the
same changing denominator, providing an exact conservation check each step.
"""
    h, full = len(prefix)-1, len(boundary)-1
    row, common, plus, minus = [1], 1, 0, 0
    prefix_snapshot = None
    for d in range(1, full+1):
        factor = 2*denominator if d <= h else 2
        common *= factor
        plus *= factor
        minus *= factor
        fresh = [0]*(d+1)
        for k in range(d+1):
            incoming = (row[k] if k < d else 0)+(row[k-1] if k > 0 else 0)
            if not incoming:
                continue
            if d <= h:
                numerator = prefix[d][min(k, d-k)]
                stopped = incoming*numerator
                fresh[k] = incoming*(denominator-numerator)
            elif k >= boundary[d] or k <= d-boundary[d]:
                stopped = incoming
            else:
                stopped = 0
                fresh[k] = incoming
            if 2*k > d:
                plus += stopped
            elif 2*k < d:
                minus += stopped
            else:
                assert stopped == 0
        row = fresh
        assert sum(row)+plus+minus == common
        assert plus == minus and row == row[::-1]
        if d == h:
            prefix_snapshot = {'survival_probability': exact_record(F(sum(row), common)),
                               'any_report_probability': exact_record(F(plus+minus, common))}
    risk = F(plus+minus, common)
    return risk, {'full_horizon': full, 'prefix_horizon': h,
                  'final_nonzero_surviving_states': sum(bool(x) for x in row),
                  'denominator_formula_verified': common == (2*denominator)**h*2**(full-h),
                  'integer_denominator_bit_length': common.bit_length(),
                  'exact_mass_conservation_checks': full,
                  'exact_symmetry_checks': full,
                  'prefix_snapshot': prefix_snapshot,
                  'exact_fair_one_direction_risk': exact_record(F(plus, common))}


def main():
    started = time.perf_counter()
    plan, candidate, joint = read(PLAN), read(CANDIDATE), read(JOINT)
    verify_hashes(plan['source_sha256'])
    verify_hashes(candidate['input_sha256'])
    assert tuple(plan['comparison_methods'][:6]) == METHODS
    assert plan['physical_pair_cap'] == 8192 and plan['candidate_prefix_horizon'] == 128
    assert joint['candidate_sha256'] == sha(CANDIDATE)
    p = candidate['protocol']
    policy = operating.canonical_policy(p['probability_denominator'], p['policy_numerators'])
    assert p['prefix_sign_horizon'] == len(policy[1])-1 == 128
    assert p['full_sign_horizon'] == p['physical_pair_cap'] == 8192
    assert p['continuation_start_discordance'] == 129
    assert p['continuation_boundary_sha256'] == sha(BETA)
    assert p['continuation_boundaries'] == read(BETA)['boundaries']
    assert p['truth_inputs'] == []
    operating.validate_table(p['continuation_boundaries'], 8192)
    audit = read(HYBRID_AUDIT)
    assert audit['status'] == 'passed'
    verify_hashes(audit['input_sha256'])
    paths = [Path(__file__).resolve(), PLAN, CANDIDATE, JOINT, BETA, MANIFEST,
             MIXTURE_AUDIT, BETA_AUDIT, HYBRID_AUDIT,
             Path(operating.__file__), ROOT/'experiments/sota_round5/interval_randomized_stopping.py',
             ROOT/'experiments/sota_round4/interval_sign_cost.py']
    hashes = {path.relative_to(ROOT).as_posix(): sha(path) for path in paths}
    hashes.update(plan['source_sha256'])
    print(json.dumps({'stage': 'independent_full_integer_joint_null_started'}), flush=True)
    risk, details = independent_full_integer_joint(policy[1], policy[0], p['continuation_boundaries'])
    recorded_risk = F(int(joint['risk_exact_numerator']), int(joint['risk_exact_denominator']))
    assert risk == recorded_risk
    assert F(joint['risk']['lower']) <= risk <= F(joint['risk']['upper'])
    assert joint['null_at_most_00999_exact'] == (risk <= F(999, 100000))
    risk_pass = risk <= F(1, 100)
    internal_pass = risk <= F(999, 100000)
    assert risk_pass and not internal_pass
    joint_review = {
        'status': 'passed_independent_full_integer_recomputation',
        'candidate_sha256': sha(CANDIDATE), 'producer_joint_null_sha256': sha(JOINT),
        'reviewer_source': Path(__file__).resolve().relative_to(ROOT).as_posix(),
        'reviewer_source_sha256': sha(Path(__file__).resolve()),
        'candidate_unchanged': True, 'risk_exact': exact_record(risk),
        'risk_outward': directed_display(risk),
        'independently_equal_to_producer_exact_risk': True,
        'joint_null_at_most_1_percent': risk_pass,
        'internal_00999_passed': internal_pass,
        'internal_00999_failure_retained': not internal_pass,
        'internal_failure_excess_exact': exact_record(risk-F(999, 100000)),
        'one_percent_exact_slack': exact_record(F(1, 100)-risk),
        'integer_recurrence': details,
        'proof': [
            'Under q=.5, each sign branch has weight1/2; a prefix report coin has exact probabilitya/D. The independent whole-vector integer recurrence scales the common denominator by2D at each prefix step and by2 in each suffix step, absorbing only first reports. It applies beta only fromd=129.',
            'Symmetry is checked exactly at every one of8192 steps. The folded prefix table and symmetric fallback imply equal positive and negative fair first-report probabilities, each half the reported joint risk.',
            'For constant iid q>.5, a wrong negative first-report path stops at total counts k<d/2. Its sign likelihood ratio relative to fair is(2q)^k[2(1-q)]^(d-k)<=1: its log derivative inq is(k-dq)/(q(1-q))<0 forq>=.5 and it starts at1. The same internal coin factors occur under both laws. Summing over disjoint first-report paths bounds wrong-negative risk by half the fair joint risk. Reflecting signs proves the opposite direction.',
            'An iid arrival clock with any stationaryr, capped at8192 physical pairs, only removes latent reports. Therefore actual null-report risk is at most the latent joint risk and strict wrong risk at most half that risk, uniformly over this fixed iid marked-sign resource model. The proof does not cover arbitrary adaptive variation inq.'
        ],
        'solver_claims': 'The actual frozen policy passes the1% scientific risk requirement and fails the tighter .00999 internal design guard. Neither failure record nor candidate was changed. This is not an exact LP optimality claim.',
        'new_simulation_observations': 0}
    write_new(JOINT_OUT, joint_review)
    hashes[JOINT_OUT.relative_to(ROOT).as_posix()] = sha(JOINT_OUT)
    print(json.dumps({'stage': 'joint_risk_verified', 'risk': directed_display(risk),
                      'passes_one_percent': risk_pass, 'passes_internal_00999': internal_pass,
                      'artifact': str(JOINT_OUT), 'sha256': sha(JOINT_OUT)}), flush=True)

    # Main and mirror rows are fixed in advance. Exact sign complement gives
    # mirrored probability intervals without a second numerical evaluation.
    all_points = {row['case_id']: row for row in plan['rows']}
    positives = [row for row in plan['rows'] if row['kind'] == 'main' and row['correct_sign'] == 1]
    nulls = [row for row in plan['rows'] if row['kind'] == 'null']
    assert len(positives) == 24 and len(nulls) == 4 and len(all_points) == 52
    for row in positives:
        mirror = all_points[row['case_id']+1000]
        r, q = F(row['r_exact']), F(row['q_exact'])
        assert mirror['kind'] == 'main' and mirror['correct_sign'] == -1
        assert F(mirror['r_exact']) == r and F(mirror['q_exact']) == 1-q
        # Exact feasibility of the plan's irrational arm construction:
        # s=1-sqrt(1-2r+d^2), d=r(2q-1). radicand in[0,1],
        # sqrt(radicand)<=1-d follows from r>=d; hence s>=d.
        d = r*(2*q-1)
        radical = 1-2*r+d*d
        assert 0 <= d <= r <= F(9, 20) and 0 <= radical <= 1
        assert radical <= (1-d)**2
    assert all(F(row['q_exact']) == F(1, 2) for row in nulls)
    manifest, mix_audit, beta_audit = read(MANIFEST), read(MIXTURE_AUDIT), read(BETA_AUDIT)
    assert mix_audit['status'] == beta_audit['status'] == 'passed'
    assert mix_audit['manifest_sha256'] == sha(MANIFEST)
    zero_prefix = operating.canonical_policy(1, [[0], [0]])
    rows, summaries = [], []
    for name in METHODS:
        method_started = time.perf_counter()
        if name == 'beta_exact':
            path = BETA
            prior = next(r for r in beta_audit['artifacts'] if r['path'] == path.relative_to(ROOT).as_posix())
            assert prior['status'] == 'passed' and prior['file_sha256'] == sha(path)
            bound_audit_path = BETA_AUDIT
        else:
            record = next(r for r in manifest['records'] if r['shape'] == name)
            path = ROOT/record['path']
            assert sha(path) == record['sha256']
            prior = next(r for r in mix_audit['results'] if r['shape'] == name)
            assert prior['status'] == 'passed' and prior['input_sha256'] == sha(path)
            assert prior['integer_dp_recomputed_without_producer_import'] and prior['exact_risk_within_delta']
            bound_audit_path = MIXTURE_AUDIT
        hashes[path.relative_to(ROOT).as_posix()] = sha(path)
        table = read(path)
        boundary = table['boundaries']
        assert len(boundary) == 8193 and boundary[1] == 2
        control_risk = F(int(table['fair_null_risk']['crossing_numerator']),
                         int(table['fair_null_risk']['fair_denominator']))
        assert control_risk <= F(1, 100)
        method_rows = []
        for index, point in enumerate(positives+nulls):
            r, q = F(point['r_exact']), F(point['q_exact'])
            oc = operating.operating_rq(zero_prefix, boundary, r, q, 8192)
            assert oc['exact_r'] == str(r) and oc['exact_q'] == str(q)
            row = {'control': name, 'case_id': point['case_id'], 'kind': point['kind'],
                   'correct_sign': point['correct_sign'], 'r_exact': str(r), 'q_exact': str(q),
                   'plus': oc['plus'], 'minus': oc['minus'], 'abstain': oc['abstain'],
                   'expected_shots': oc['expected_shots'],
                   'evaluation': 'direct_audited_operating_rq_with_exact_Fractions',
                   'method_sha256': sha(path)}
            if point['kind'] == 'main':
                row['correct_probability'], row['wrong_probability'] = row['plus'], row['minus']
                row['power_qualified_at_95_percent'] = F(row['correct_probability'][0]) >= F(19, 20)
                mirror_point = all_points[point['case_id']+1000]
                mirror = dict(row)
                mirror.update(case_id=mirror_point['case_id'], correct_sign=-1, q_exact=str(1-q),
                              plus=row['minus'], minus=row['plus'],
                              evaluation='derived_by_exact_sign_complement', mirrored_from_case_id=point['case_id'])
                # correct/wrong probabilities and cost are unchanged by this
                # exact coupling; plus/minus report labels alone exchange.
                method_rows.extend((row, mirror))
            else:
                row['null_report_probability'] = [float(operating.down(oc['plus'][0]+oc['minus'][0])),
                                                  min(1., float(operating.up(oc['plus'][1]+oc['minus'][1])))]
                method_rows.append(row)
            if (index+1) % 7 == 0:
                print(json.dumps({'control': name, 'direct_rows_done': index+1, 'seconds': time.perf_counter()-method_started}), flush=True)
        main = [row for row in method_rows if row['kind'] == 'main']
        assert len(main) == 48 and len(method_rows) == 52
        min_power = min(row['correct_probability'][0] for row in main)
        globally_qualified = all(row['power_qualified_at_95_percent'] for row in main)
        power_failed = [row['case_id'] for row in main if not row['power_qualified_at_95_percent']]
        sums = (sum(F(row['expected_shots'][0]) for row in main)/48,
                sum(F(row['expected_shots'][1]) for row in main)/48)
        summary = {'control': name, 'method_path': path.relative_to(ROOT).as_posix(),
                   'method_sha256': sha(path), 'risk_audit_path': bound_audit_path.relative_to(ROOT).as_posix(),
                   'risk_audit_sha256': sha(bound_audit_path),
                   'risk_exact': exact_record(control_risk),
                   'uniform_null_and_strict_wrong_at_most_one_percent': True,
                   'baseline_boundary_at_d1_is2': True,
                   'zero_prefix_equals_pure_baseline': True,
                   'minimum_correct_probability_lower': min_power,
                   'every_48_main_case_power_at_least_95_percent': globally_qualified,
                   'globally_qualified': globally_qualified, 'power_unqualified_case_ids': power_failed,
                   'mean_expected_shots_exact_endpoint_interval': {'lower': exact_record(sums[0]), 'upper': exact_record(sums[1])},
                   'direct_operating_calls': 28, 'exact_mirror_derivations': 24,
                   'elapsed_seconds': time.perf_counter()-method_started}
        rows.extend(method_rows)
        summaries.append(summary)
        print(json.dumps({'control_completed': name, 'globally_qualified': globally_qualified,
                          'minimum_power_lower': min_power, 'seconds': summary['elapsed_seconds']}), flush=True)
    verify_hashes(hashes)
    result = {'status': 'completed_independent_fixed_grid_controls_a',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'reviewer': 'stream_sota independent of candidate production',
              'source_sha256': hashes,
              'independent_grid_plan_sha256': sha(PLAN), 'candidate_policy_sha256': sha(CANDIDATE),
              'new_simulation_observations': 0, 'candidate_modified': False,
              'candidate_joint_null': joint_review,
              'method_summaries': summaries, 'rows': rows,
              'direct_operating_calls': 168, 'exact_mirror_rows': 144,
              'qualified_controls': [r['control'] for r in summaries if r['globally_qualified']],
              'proof_and_protocol': [
                  'Each control has boundary[1]=2, which cannot stop at d1. Therefore passing canonical_policy(1,[[0],[0]]) to the audited hybrid primitive exactly reproduces the pure fixed boundary rule, with fallback beginningd2 and no missed decision.',
                  'All operating inputs r,q are exact Fractions read from the independent plan. No pminus/pplus approximation or binary64 re-rounding is used. Prior independent integer DP audits bind the exact control risk certificates.',
                  'For a symmetric fixed boundary, complement every latent sign and keep the same arrival times. The stopping time and total paid cost agree pathwise, plus/minus report labels swap, and(q,r) maps exactly to(1-q,r). Thus copied mirror cost/correct-power enclosures are exact consequences, not approximate extrapolation.',
                  'The plan r,q grid has an exact physical realization with possibly irrational arms. Positivity of the radicand and radicand<=(1-d)^2 are checked with Fractions, giving0<=pminus<=pplus<=1 for the declared construction; those irrational arms are never approximated for evaluation.',
                  'A method qualifies only when its rule risk certificate passes and its outward correct-power LOWER endpoint reaches .95 at every one of48 main cases. Null points are additionally reported; arithmetic costs include errors and full-cap abstentions.'
              ],
              'limits': ['This evaluates6 named classical controls on the frozen independent parameter grid, not the entire literature SOTA.',
                         'Candidate risk passing1% does not imply candidate power/cost qualification, which is independently evaluated by another workstream.',
                         'The tighter internal .00999 guard failure remains explicit and does not lead to candidate replacement or an optimality claim.',
                         'There are no new Monte Carlo, circuit, or hardware observations. Arithmetic interval widths are numerical enclosures, not sampling confidence intervals.'],
              'elapsed_seconds': time.perf_counter()-started}
    write_new(OUT, result)
    print(json.dumps({'artifact': str(OUT), 'sha256': sha(OUT),
                      'qualified_controls': result['qualified_controls'],
                      'elapsed_seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
