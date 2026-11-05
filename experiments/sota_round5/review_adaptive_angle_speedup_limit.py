"""Independent audit of the matched-model adaptive-angle speedup ceiling.

The subject producer and its Decimal interval implementation are not imported.
Logarithms and information bounds are independently enclosed with exact
Fraction log series. The already independently audited round4 operating helper
is rerun on every positive case at the exact rational PSR probabilities.
"""
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/sota_round4'))
import interval_sign_cost as operating
import review_information_bounds as rational_review

DIR = ROOT/'results/sota_round5/theory'
SOURCE = DIR/'certify_adaptive_angle_speedup_limit.py'
ARTIFACT = DIR/'certify_adaptive_angle_speedup_limit.json'
OUT = DIR/'adaptive_angle_speedup_limit_independent_review.json'
PLAN = ROOT/'results/sota_round3/paired_benchmark/preliminary_plan.json'
BETA = ROOT/'results/sota_round3/cap16384_preparation/beta_exact.json'
SYMMETRIC_REVIEW = ROOT/'results/structured_validation/continuous_symmetric_design_review.json'
POINTWISE_REVIEW = ROOT/'results/structured_validation/continuous_psr_bound_sota_review.json'
POINTWISE_ROOT_REVIEW = ROOT/'results/structured_validation/continuous_psr_bound_root_review.json'
INTERVAL_REVIEW = ROOT/'results/sota_round4/theory/audit_interval_sign_cost.json'
INFORMATION_REVIEW = ROOT/'results/sota_round4/theory/finite_budget_information_bounds_independent_review.json'
BETA_REVIEW = ROOT/'results/sota_round2/theory/cap16384_preparation_independent_audit.json'
RISK_LEMMA = ROOT/'results/sota_round2/theory/generic_symmetric_tail_dp_lemma.json'
CASE_IDS = (9, 7, 19, 21, 31, 33, 43, 45)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def exact(record):
    return Fraction(int(record['numerator']), int(record['denominator']))


def record(value):
    return {'numerator': str(value.numerator), 'denominator': str(value.denominator)}


def verify_hashes(mapping):
    for path, expected in mapping.items():
        assert sha(ROOT/path) == expected, path


def enclosed(value, saved):
    assert Fraction(saved['lower']) <= value[0] <= value[1] <= Fraction(saved['upper'])


def main():
    started = time.perf_counter()
    subject = read(ARTIFACT)
    assert subject['status'] == 'certified_exact_parameter_adaptive_angle_cost_obstruction'
    assert subject['goal_achieved'] is False and subject['new_simulation_observations'] == 0
    verify_hashes(subject['source_sha256'])
    assert subject['source_sha256'][SOURCE.relative_to(ROOT).as_posix()] == sha(SOURCE)

    symmetric = read(SYMMETRIC_REVIEW)
    assert symmetric['status'] == 'passed'
    assert sha(ROOT/symmetric['document']) == symmetric['document_sha256']
    pointwise = read(POINTWISE_REVIEW)
    assert pointwise['status'] == 'passed' and not pointwise['blocking_findings']
    verify_hashes(pointwise['sources'])
    pointwise_root = read(POINTWISE_ROOT_REVIEW)
    assert pointwise_root['status'] == 'passed' and not pointwise_root['blocking_findings']
    assert sha(ROOT/pointwise_root['source']['path']) == pointwise_root['source']['sha256']
    interval_review = read(INTERVAL_REVIEW)
    assert interval_review['status'] == 'passed'
    verify_hashes(interval_review['source_sha256'])
    assert interval_review['target_source_sha256'] == sha(operating.__file__)
    information_review = read(INFORMATION_REVIEW)
    assert information_review['status'] == 'passed_with_explicit_scope'
    verify_hashes(information_review['input_sha256'])
    assert information_review['reviewer_sha256'] == sha(rational_review.__file__)
    beta_review = read(BETA_REVIEW)
    assert beta_review['status'] == 'passed'
    beta_audit = next(r for r in beta_review['artifacts'] if r['path'] == BETA.relative_to(ROOT).as_posix())
    assert beta_audit['status'] == 'passed' and beta_audit['file_sha256'] == sha(BETA)
    assert beta_audit['full_integer_risk_recomputed_without_producer_import']
    beta = read(BETA)
    assert beta['max_discordances'] == 8192 and len(beta['boundaries']) == 8193
    operating.validate_table(beta['boundaries'], 8192)
    fair_risk = Fraction(int(beta['fair_null_risk']['crossing_numerator']),
                         int(beta['fair_null_risk']['fair_denominator']))
    assert fair_risk <= Fraction(1, 100)
    assert read(RISK_LEMMA)['status'] == 'proved_lemma_no_shape_optimality_or_performance_claim'
    assert subject['baseline']['pair_cap'] == 8192
    assert subject['baseline']['physical_shot_cap'] == 16384
    assert subject['baseline']['every_positive_case_power_certified_at_least_95_percent']
    assert subject['baseline']['exact_null_crossing_probability_at_most_one_percent']

    dependencies = [Path(__file__).resolve(), SOURCE, ARTIFACT, PLAN, BETA,
                    Path(operating.__file__), Path(rational_review.__file__),
                    SYMMETRIC_REVIEW, POINTWISE_REVIEW, POINTWISE_ROOT_REVIEW,
                    INTERVAL_REVIEW, INFORMATION_REVIEW, BETA_REVIEW, RISK_LEMMA]
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in dependencies}
    hashes.update(subject['source_sha256'])
    kappa = rational_review.bernoulli_kl(Fraction(19, 20), Fraction(1, 100))
    enclosed(kappa, subject['kappa'])
    cases = {c['case_id']: c for c in read(PLAN)['cases']}
    assert [r['case_id'] for r in subject['rows']] == list(CASE_IDS)
    expected_main = set(CASE_IDS) | {c+1000 for c in CASE_IDS}
    assert {i for i, c in cases.items() if c['kind'] == 'main'} == expected_main
    product = Fraction(1)
    checks = []
    rounded_again_differences = []
    for row in subject['rows']:
        t0 = time.perf_counter()
        case_id = row['case_id']
        case = cases[case_id]
        A, B, C = [Fraction.from_float(case[k]) for k in ('A', 'B', 'C')]
        m = min(A, 1-A)
        assert case['kind'] == 'main' and 0 < C < m and B >= 0
        assert B*B+C*C <= m*m
        assert all(exact(row['exact_parameter_model'][k]) == Fraction.from_float(case[k])
                   for k in ('A', 'B', 'C'))
        assert all(row['parameter_binary64_hex'][k] == case[k].hex() for k in ('A', 'B', 'C'))
        pm, pp = A-C, A+C
        assert 0 < pm < pp < 1
        assert exact(row['exact_psr_probabilities']['minus']) == pm
        assert exact(row['exact_psr_probabilities']['plus']) == pp
        assert row['probabilities_were_not_rounded_again']
        rounded_pm, rounded_pp = Fraction.from_float(case['A']-case['C']), Fraction.from_float(case['A']+case['C'])
        if (rounded_pm, rounded_pp) != (pm, pp):
            rounded_again_differences.append(case_id)
        mirror = cases[row['mirror_case_id']]
        assert row['mirror_case_id'] == case_id+1000 and mirror['kind'] == 'main'
        Am, Bm, Cm = [Fraction.from_float(mirror[k]) for k in ('A', 'B', 'C')]
        assert (Am, Bm, Cm) == (A, B, -C)
        assert (Am-Cm, Am+Cm) == (pp, pm)

        # Only the already independently audited operating primitive is reused.
        # Its Fraction API preserves exact A +/- C through r and q evaluation.
        operating_fresh = operating.operating_intervals(beta['boundaries'], pm, pp, 8192)
        assert operating_fresh == row['beta_operating_intervals_exact_parameter_model']
        assert operating_fresh['exact_binary_model_probabilities'] == [str(pm), str(pp)]
        assert Fraction(operating_fresh['plus'][0]) >= Fraction(19, 20)
        assert row['beta_power_lower_endpoint_at_least_95_percent']

        # Different arithmetic route from the producer: exact rational log
        # series and residuals, with no _Bounds or mpmath import in the reviewer.
        v = C/m
        log_ratio = rational_review.add(rational_review.rational_log(1+v),
                                       rational_review.scale(rational_review.rational_log(1-v), -1))
        lam = rational_review.scale(log_ratio, 1/(2*v))
        bstar = rational_review.scale(lam, B)
        margin = m-bstar[1], m-bstar[0]
        assert margin[0] > 0 and 0 <= bstar[0] <= bstar[1] < m
        U = rational_review.scale(rational_review.bernoulli_kl((1+v)/2, Fraction(1, 2)), 2*m)
        joint = ((1-pm)*(1-pp), (1-pm)*pp, pm*(1-pp), pm*pp)
        null = ((1-A)**2, A*(1-A), A*(1-A), A*A)
        E = rational_review.scale(rational_review.discrete_kl(joint, null), Fraction(1, 2))
        candidate_lower = rational_review.divide_positive(kappa, U)
        info = row['information_bounds_exact_parameter_model']
        for value, name in ((lam, 'q_star_lambda'), (bstar, 'q_star_cosine_coefficient'),
                            (margin, 'q_star_physical_margin_m_abs_bstar'),
                            (U, 'U_pointwise_any_angle_per_shot'), (E, 'E_full_psr_per_shot'),
                            (candidate_lower, 'expected_physical_shots_lower_bound_arbitrary_adaptive_angles')):
            enclosed(value, info[name])
        assert U[1] < 2*E[0]
        if A == Fraction(1, 2):
            assert C/m == 2*C
            # Formula identity U=E is analytic; independent numerical intervals
            # necessarily overlap and are not used as proof of exact equality.
            assert max(U[0], E[0]) <= min(U[1], E[1])
        lower = exact(row['candidate95percent_expected_shots_lower_endpoint_exact'])
        high = exact(row['beta_expected_shots_upper_endpoint_exact'])
        assert lower == Fraction(info['expected_physical_shots_lower_bound_arbitrary_adaptive_angles']['lower'])
        assert high == Fraction(operating_fresh['expected_shots'][1])
        assert 0 < lower <= candidate_lower[0]
        ratio = high/lower
        assert ratio == exact(row['candidate_speedup_upper_exact'])
        assert ratio < 3 and row['individual_threefold_cost_gain_excluded']
        assert str(3*ratio.denominator-ratio.numerator) == row['exact_threefold_cross_multiplication_slack']
        product *= ratio
        checks.append({'case_id': case_id, 'mirror_case_id': row['mirror_case_id'],
                       'matched_exact_parameter_and_psr_probabilities': True,
                       'all_operating_interval_fields_recomputed_and_identical': True,
                       'certified_beta_correct_probability_lower': operating_fresh['plus'][0],
                       'certified_beta_cost_upper': operating_fresh['expected_shots'][1],
                       'independent_qstar_physical_margin': rational_review.display(margin),
                       'independent_any_angle_expected_cost_lower_bound': rational_review.display(candidate_lower),
                       'exact_ratio': record(ratio), 'ratio_float_diagnostic_only': float(ratio),
                       'individual_threefold_excluded': True,
                       'elapsed_seconds': time.perf_counter()-t0})
        print(json.dumps({'case_id': case_id, 'status': 'passed', 'seconds': checks[-1]['elapsed_seconds']}), flush=True)

    assert product == exact(subject['exact_product_of_eight_upper_speedup_ratios'])
    bracket = subject['geometric_upper_root_bracket_by_integer_bisection']
    low, high = exact(bracket['lower_exact']), exact(bracket['strict_upper_exact'])
    assert low == Fraction(bracket['lower_integer_tick'], bracket['scale'])
    assert high == Fraction(bracket['upper_integer_tick'], bracket['scale'])
    assert bracket['upper_integer_tick'] == bracket['lower_integer_tick']+1
    assert low**8 <= product < high**8
    rounded = exact(subject['upward_thousandth_geometric_speedup_bound'])
    assert rounded == Fraction(2593, 1000)
    assert product < rounded**8 < Fraction(3)**8
    assert product**2 < rounded**16 < Fraction(3)**16
    assert str(3**8*product.denominator-product.numerator) == subject['exact_geometric_threefold_integer_slack']
    assert str(rounded.numerator**8*product.denominator-product.numerator*rounded.denominator**8) == subject['exact_rounded_bound_integer_slack']
    assert subject['exact_geometric_threefold_excluded'] and subject['every_individual_case_threefold_excluded']
    verify_hashes(hashes)
    result = {
        'status': 'passed_with_explicit_scope',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'reviewer': 'stream_sota independent of producer',
        'reviewer_source': Path(__file__).resolve().relative_to(ROOT).as_posix(),
        'reviewed_artifact': ARTIFACT.relative_to(ROOT).as_posix(),
        'input_sha256': hashes,
        'new_simulation_observations': 0,
        'goal_achieved': False,
        'blocking_findings': [],
        'checks': {'positive_cases_with_full_baseline_interval_rerun': len(checks),
                   'main_cases_including_exact_mirrors': 16,
                   'independent_rational_series_information_cases': len(checks),
                   'all_target_source_bindings_verified': True,
                   'prior_symmetric_and_pointwise_theorem_review_bindings_verified': True,
                   'baseline_exact_risk_certificate_bound_to_independent_audit': True,
                   'second_rounding_would_change_psr_rates_in_cases': rounded_again_differences},
        'independent_numeric_method': {
            'logarithms': 'Exact Fraction range reduction and 80-term atanh log series with an exact positive residual bound, reusing the prior independent reviewer, not the information producer.',
            'baseline': 'All 8 outward operating interval records recomputed at exact Fraction A-C,A+C with the hash-bound independently audited round4 primitive; no producer main rerun.',
            'information': 'Independent rational-series kappa, lambda, bstar, physical margin, all-outcome pair KL E, U via Bernoulli KL, and kappa/U; each tighter enclosure lies inside the recorded interval.',
            'products': 'Exact Fraction ratios, product, integer root-bracket comparisons, and positive cross-multiplication slacks; floating ratios are diagnostic only.',
            'producer_information_code_imported': False,
            'producer_Decimal_helper_imported': False,
            'mpmath_used': False},
        'mathematical_findings': [
            {'topic': 'matched parameter semantics', 'status': 'passed', 'reason':
             'A,B,C are the exact rationals represented by parsed binary64 tokens. The model uses exact real trigonometric coordinates. At PSR coordinates cos=0,sin=+/-1 the rates are exact A-C,A+C, passed as Fractions without a second binary64 rounding. Candidate U and baseline costs therefore refer to the same exact continuous-angle law. This is not a statement about arbitrary finite-precision simulator trigonometric evaluation.'},
            {'topic': 'one globally physical null', 'status': 'passed', 'reason':
             'For every recorded case, exact physicality and 0<C<m hold. Independent rational intervals certify m-|B|*atanh(C/m)/(C/m)>0. All current B are nonnegative, so the producer margin m-B*lambda is correctly labeled here. The cited reviewed theorem supplies one qstar(x)=A+B*lambda*cos(x) with C=0 and pointwise kl(p(x),qstar(x))<=U for every real angle.'},
            {'topic': 'adaptive transcript and stopped cost', 'status': 'passed', 'reason':
             'Pad to the finite cap with deterministic stop symbols and include all paid outcomes and independent internal randomness. Identical policy, angle, stop, and output kernels under theta and qstar add zero conditional KL. Each paid observation has conditional KL at most U, so chain rule gives KL<=U E_theta[N], including unbalanced allocation, mid-pair stopping, errors and abstentions. No optional-stopping limit, grid replacement, or minimax interchange is required.'},
            {'topic': 'binary terminal event lower bound', 'status': 'passed', 'reason':
             'Use the event of a correct terminal sign report for the selected alternative. Its true probability is at least .95; under the single physical C=0 null it is at most the allowed .01 probability of any sign report. Data processing and monotonicity of Bernoulli KL for p>q give KL>=kl(.95,.01). Hence every admissible candidate has unconditional expected paid-shot cost at least kappa/U.'},
            {'topic': 'power-qualified baseline', 'status': 'passed', 'reason':
             'The frozen symmetric beta table has an independently recomputed exact fair-sign crossing risk <=.01. Every completed fixed independent PSR pair is inspected up to 8192 pairs. Null reports embed into the latent fair-sign crossing event. Rerun operating intervals certify correct power >=.95 separately on every positive main case. Wrong and capped-abstaining trajectories are charged in baseline E[N].'},
            {'topic': 'mirrors and geometric aggregation', 'status': 'passed', 'reason':
             'The plan has precisely these 8 positives and their 8 exact C-negated mirrors. Swapping plus and minus outcomes is a probability-preserving path bijection for the symmetric beta boundary, hence mirrored expected cost and correct power agree exactly. U and lambda are even in C. Each mirror candidate separately obeys the same lower bound, even for a nonsymmetric candidate. Thus the 16-case product is bounded by the square of the 8-case bound, giving the identical eighth-root geometric ceiling.'},
            {'topic': 'exact speedup comparisons', 'status': 'passed', 'reason':
             'The baseline upper endpoint divided by the candidate lower endpoint is an upper bound on each actual expected-cost speedup. All 8 exact Fraction ratios are <3. Their exact product lies below (2593/1000)^8<3^8, and its square lies below (2593/1000)^16<3^16. The reported integer bisection root bracket and both integer slacks are independently reproduced.'}],
        'kappa_independent': rational_review.display(kappa),
        'beta_exact_fair_risk': record(fair_risk),
        'rows': checks,
        'exact_product': record(product),
        'strict_geometric_upper': record(rounded),
        'root_bracket_verified': bracket,
        'eight_and_sixteen_case_geometric_threefold_excluded': True,
        'limits': [
            'The ceiling is a necessary bound, not an achievable speedup, a full SOTA comparison, or completion of the research goal.',
            'It applies to the same stationary independent Bernoulli single-harmonic query model, equal one-shot cost, truth-independent candidate kernels, finite cap, <=1% sign-report probability at every physical zero-sign null, and >=95% correct power on each compared case.',
            'It compares expected unconditional cost to this specified power-qualified beta baseline. It neither claims a globally best baseline nor bounds tasks with genuinely different resources, certified prior restrictions, or evaluation endpoints.',
            'The continuous pointwise theorem and operating primitive were already independently reviewed and hash-bound here. They were not re-proved or formally verified by a proof assistant in this audit.',
            'The production information routine must use abs(B) in its physical margin if later extended to negative B; all present positives and their C mirrors have B>=0 and pass the exact check.'
        ],
        'elapsed_seconds': time.perf_counter()-started}
    with OUT.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'sha256': sha(OUT),
                      'strict_geometric_upper': str(rounded),
                      'cases': len(checks), 'rounded_again_differences': rounded_again_differences,
                      'elapsed_seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
