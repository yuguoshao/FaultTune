"""Deterministic finite-budget lower bounds, with outward numerical enclosures.

No outcomes, seeds, replay results, or controller performance are consumed.
The source is deliberately saved beside its new round4 theorem artifact.
"""
from decimal import Decimal, ROUND_CEILING
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import mpmath as mp

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'experiments/circuit_validation'))
from direct_tests import _Bounds

OUT = Path(__file__).with_suffix('.json')
PLAN = ROOT / 'results/sota_round2/benchmark/preliminary_plan.json'
mp.mp.dps = 160


def iv(x):
    x = Fraction(x)
    return _Bounds(x.numerator) / x.denominator


def mm(x):
    x = Fraction(x)
    return mp.mpf(x.numerator) / x.denominator


def phi(x):
    x = abs(Fraction(x))
    assert 0 <= x <= 1
    if x == 0:
        return _Bounds(0)
    if x == 1:
        return _Bounds(2).log()
    y = iv(x)
    return ((1+y)*(1+y).log() + (1-y)*(1-y).log()) / 2


def kl(p, q):
    p, q = Fraction(p), Fraction(q)
    assert 0 <= p <= 1 and 0 < q < 1
    first = _Bounds(0) if p == 0 else iv(p)*iv(p/q).log()
    second = _Bounds(0) if p == 1 else iv(1-p)*iv((1-p)/(1-q)).log()
    return first + second


def mkl(p, q):
    p, q = mm(p), mm(q)
    return (p*mp.log(p/q) if p else 0) + ((1-p)*mp.log((1-p)/(1-q)) if p != 1 else 0)


def record(value, reference=None):
    result = {'lower': str(value.lo), 'upper': str(value.hi)}
    if reference is not None:
        assert mp.mpf(str(value.lo)) <= reference <= mp.mpf(str(value.hi))
        result['independent_mpmath160_value'] = mp.nstr(reference, 130)
        result['independent_mpmath160_inside_interval'] = True
    return result


def exact(x):
    x = Fraction(x)
    return {'numerator': str(x.numerator), 'denominator': str(x.denominator)}


def minimum_cap(bound, step):
    # Only the HARD CAP is integer/even valued. Expected cost is never rounded up.
    def ceil_fraction(x):
        x = Fraction(x)/step
        return -((-x.numerator)//x.denominator)*step
    left = ceil_fraction(bound.lo)
    right = ceil_fraction(bound.hi)
    assert left == right
    return left


def power_ceiling(info, cap):
    budget = info*cap
    if budget.lo >= kl(1, Fraction(1, 100)).hi:
        return {'upper': '1', 'reason': 'KL budget at this cap does not give a nontrivial power ceiling'}
    low, high = Fraction(1, 100), Fraction(1)
    for _ in range(160):
        mid = (low+high)/2
        f = kl(mid, Fraction(1, 100))
        if f.hi < budget.lo:
            low = mid
        elif f.lo > budget.hi:
            high = mid
        else:
            break
    assert high-low < Fraction(1, 10**45)
    return {'root_lower': str(iv(low).lo), 'upper': str(iv(high).hi),
            'method': 'Outward rational bisection of kl(power,.01)=cap*information on [.01,1]'}


def calculate(A, B, C, kappa, km):
    m = min(A, 1-A)
    assert 0 < C < m and B*B+C*C <= m*m
    E = iv(A)*phi(C/A) + iv(1-A)*phi(C/(1-A))
    U = 2*iv(m)*phi(C/m)
    Em = (mkl(A+C, A)+mkl(A-C, A))/2
    # Independent high-precision route uses two Bernoulli KLs.
    Um = 2*mm(m)*mkl((1+C/m)/2, Fraction(1, 2))
    v = C/m
    lam = iv((1+v)/(1-v)).log() / (2*iv(v))
    lamm = mp.atanh(mm(v))/mm(v)
    bs = iv(B)*lam
    assert (iv(m)-bs).lo > 0
    r = 2*A*(1-A)+2*C*C
    q = Fraction(1, 2)+C/r
    sign_info = iv(r)*phi(2*C/r)/2
    sign_m = mm(r)*mkl(q, Fraction(1, 2))/2
    lower_psr, lower_any = kappa/E, kappa/U
    values = {
        'E_full_psr_per_shot': record(E, Em),
        'U_pointwise_any_angle_per_shot': record(U, Um),
        'oracle_information_ratio_upper_U_over_E': record(U/E, Um/Em),
        'q_star_cosine_coefficient': record(bs, mm(B)*lamm),
        'q_star_lambda': record(lam, lamm),
        'q_star_physical_margin_m_abs_bstar': record(iv(m)-bs, mm(m)-mm(B)*lamm),
        'expected_physical_shots_lower_bound_balanced_psr': record(lower_psr, km/Em),
        'expected_physical_shots_lower_bound_arbitrary_adaptive_angles': record(lower_any, km/Um),
        'baseline_mean_necessary_for_threefold_balanced_psr_gain': record(3*lower_psr, 3*km/Em),
        'baseline_mean_necessary_for_threefold_arbitrary_angle_gain': record(3*lower_any, 3*km/Um),
        'necessary_minimum_even_cap_balanced_psr': minimum_cap(lower_psr, 2),
        'necessary_minimum_integer_cap_arbitrary_angles': minimum_cap(lower_any, 1),
        'cap4096_power_ceiling_balanced_psr': power_ceiling(E, 4096),
        'cap4096_power_ceiling_arbitrary_angles': power_ceiling(U, 4096),
        'sign_only_true_q_lr_growth_per_shot': record(sign_info, sign_m),
        'full_information_over_sign_lr_growth': record(E/sign_info, Em/sign_m),
        'discordance_probability_exact': exact(r),
        'conditional_positive_sign_probability_exact': exact(q),
    }
    assert (U-E).hi >= 0 and (2*E-U).lo > 0
    return values


def main():
    source_text = PLAN.read_text(encoding='utf-8')
    floats = json.loads(source_text)
    decimals = json.loads(source_text, parse_float=Decimal)
    decimal_cases = {c['case_id']: c for c in decimals['cases']}
    kappa = kl(Fraction(19, 20), Fraction(1, 100))
    km = mkl(Fraction(19, 20), Fraction(1, 100))
    rows = []
    for case in floats['cases']:
        if case['C'] <= 0:
            continue
        A, B, C = [Fraction.from_float(case[k]) for k in ('A', 'B', 'C')]
        decimal_case = decimal_cases[case['case_id']]
        row = {'case_id': case['case_id'],
               'exact_parameter_model': {k: {'json_decimal_token': str(decimal_case[k]),
                                            'python_binary64_hex': case[k].hex(),
                                            'python_binary64_rational': exact(Fraction.from_float(case[k]))}
                                         for k in ('A', 'B', 'C')},
               'bounds_for_exact_binary64_parameter_model': calculate(A, B, C, kappa, km)}
        # The round3 sampler rounds A +/- C once more before its binomial calls.
        pm = Fraction.from_float(case['A']-case['C'])
        pp = Fraction.from_float(case['A']+case['C'])
        effective_A, effective_C = (pm+pp)/2, (pp-pm)/2
        sampler_E = (kl(pm, effective_A)+kl(pp, effective_A))/2
        sampler_Em = (mkl(pm, effective_A)+mkl(pp, effective_A))/2
        row['actual_round3_binary64_psr_probabilities'] = {
            'minus': exact(pm), 'plus': exact(pp),
            'effective_A': exact(effective_A), 'effective_C': exact(effective_C),
            'E_per_shot': record(sampler_E, sampler_Em),
            'expected_shots_lower_bound': record(kappa/sampler_E, km/sampler_Em),
            'cap4096_power_ceiling': power_ceiling(sampler_E, 4096),
            'scope': 'Exact model at the rounded PSR probabilities passed to rng.binomial; does not model PRNG discretization.'}
        Ad, Bd, Cd = [Fraction(decimal_case[k]) for k in ('A', 'B', 'C')]
        row['literal_json_decimal_parameter_model'] = calculate(Ad, Bd, Cd, kappa, km)
        rows.append(row)
    assert [row['case_id'] for row in rows] == [7, 9, 19, 21, 31, 33, 43, 45]
    sources = [Path(__file__).resolve(), PLAN,
               ROOT/'experiments/circuit_validation/direct_tests.py',
               ROOT/'experiments/sota_round3/paired_benchmark.py',
               ROOT/'docs/CONTINUOUS_PSR_INFORMATION_BOUND.md',
               ROOT/'docs/CONTINUOUS_SYMMETRIC_DESIGN.md']
    result = {
        'status': 'proved_finite_cap_theorem_with_outward_formula_evaluation',
        'new_simulation_observations': 0,
        'empirical_performance_inputs_read': False,
        'target_power_exact': '19/20', 'null_report_budget_exact': '1/100',
        'kappa_formula': '(19/20)*log(95)+(1/20)*log(5/99)',
        'kappa': record(kappa, km),
        'theorem': {
            'setting': 'For a fixed physical theta=(A,B,C), C!=0, the same randomized observation-adaptive algorithm is run under theta and every null C=0. It stops by finite physical-shot cap M and returns +, -, or abstain. All observations, including those on wrong or abstaining paths, are paid. Angle, randomization, stopping and output kernels do not have truth access.',
            'requirements': 'P_theta(correct sign)>=.95 and P_eta(any sign report)<=.01 for every physical null eta with C=0. No power is presumed for any implemented algorithm.',
            'event': 'R is the event that the terminal output is sign(C). Its complement includes abstention and wrong reports. If p=P_theta(R), q=P_eta(R), then p>=.95 and q<=.01 imply kl(p,q)>=kl(.95,.01)=kappa.',
            'finite_transcript_chain_rule': 'Pad the transcript to cap M with a deterministic stopped symbol. Policy/randomization kernels are identical under the two models and contribute zero KL. The chain rule gives KL(P_theta transcript || P_eta transcript)=E_theta sum_{t<=N} kl(p_theta(x_t)||p_eta(x_t)). Terminal output is a common kernel. Binary data processing for R yields kappa <= this transcript KL.',
            'balanced_psr_scope': 'Every paid block is one independent Bernoulli at each of the fixed angles +/-pi/2. Blocks begin predictably and stopping/output happen only at completed block boundaries; larger predictably selected batches of complete pairs are also allowed. No additional uncharged data are allowed. Complete-pair balance is essential.',
            'balanced_psr_proof': 'Use the physical constant null eta(x)=A. A complete pair contributes kl(A-C||A)+kl(A+C||A)=2E(A,C). With N=2K physical shots, summing over started complete pairs gives transcript KL=2E*E_theta K=E*E_theta N. Hence E_theta N>=kappa/E. For coarsened recorded transcripts their KL can only be smaller; use the underlying paid-observation transcript.',
            'arbitrary_angles_scope': 'Each query angle may be chosen using all earlier paid outcomes and independent algorithm randomness, with arbitrary predictable allocation, stopping, and abstention, by a finite cap.',
            'arbitrary_angles_proof': 'Use the single physical null q*(x)=A+B*[atanh(C/m)/(C/m)]cos(x), m=min(A,1-A), when 0<|C|<m. The cited analytic theorem gives kl(p_theta(x)||q*(x))<=U=2m Phi(C/m) for EVERY x. The same null is used for the entire adaptive experiment. Chain rule gives transcript KL<=U*E_theta N, hence E_theta N>=kappa/U. No interchange of minimax or expectation is used.',
            'boundaries': 'At |C|=m physicality forces B=0; q*=A and the same bound follows by the boundary case of the information theorem. At C=0 the .95 power and .01 null-report requirements are incompatible. A=0 or1 forces C=0. Paired PSR allowing mid-pair stopping or unbalanced counts is not covered by the exact E*N identity; the U bound remains available.',
            'finite_cap_consequence': 'Since E_theta N<=M, M>=kappa/E for paired PSR and M>=kappa/U for arbitrary angles are necessary. Expected cost is real and is NOT rounded to an integer or even integer. Only a hard integer/even cap may be rounded upward as a necessary cap condition.',
            'power_ceiling': 'For any actual correct-report probability p, KL data processing implies kl(p,.01)<=M*I whenever p>=.01 and I is E or U in the corresponding scope. If M*I<log100, inversion on [.01,1] yields a strict upper bound p<1. This is a necessary information bound, not an attainable power curve.',
            'no_conditioned_cost': 'The result concerns E_theta N over ALL trajectories. E_theta[N | correct report] and cost with abstentions removed are not substituted into the bound.',
            'proof_status': 'Mathematical proof using finite chain rule, binary data processing, and the previously independently reviewed analytic pointwise theorem; not proof-assistant verified.'},
        'threefold_target': {
            'necessary_benchmark': 'If a comparator really has 95% power and expected all-trajectory cost M_base, then a 95%-power threefold gain requires M_base>=3*kappa/E for balanced PSR, or M_base>=3*kappa/U for arbitrary adaptive angles. These inequalities are necessary only.',
            'unqualified_empirical_means': 'No observed 4096-cap mean, including those at about90% aggregate power, is treated as the expected cost of a95%-power comparator. This artifact intentionally reads no empirical costs.',
            'angle_only_limit': 'The proved oracle information ratio V/E<=U/E<2 rules out a universal threefold INFORMATION-RATE gain from angle selection alone. It does not rule out a threefold finite-cost improvement over an inefficient implemented comparator.',
            'symmetric_cases': 'At A=.5, E=U=V exactly. Additional angles cannot increase full Bernoulli oracle information; evidence construction and finite-threshold overhead may still improve.'},
        'weak_signal_analysis': {
            'limit': 'For fixed0<A<1 and C->0, E=C^2/[2A(1-A)]+O_A(C^4), U=C^2/m+O_A(C^4); thus paired lower bound~2*kappa*A(1-A)/C^2 and arbitrary-angle lower bound~kappa*m/C^2.',
            'sign_growth': 'For iid PSR pairs, r=2A(1-A)+2C^2, q=1/2+C/r. The log likelihood ratio of the true-q conditional discordant-sign law against fair signs has expected growth per physical shot I_sign=(r/2)*kl(q||.5)=(r/2)*Phi(2C/r). This is a growth calculation, not a new composite-null transcript lower bound.',
            'first_order': 'As C->0 with A fixed, I_sign=C^2/[2A(1-A)]+O_A(C^4), matching E at first order. Recovering concordant/full-count information alone has no first-order multiplicative advantage in this limit. Prior mismatch, mixture regret, finite threshold/overshoot, checkpoints, and sufficient cap remain concrete opportunities.',
            'weakest_case': 'Case9 has a balanced-PSR95%-power expected-shot lower bound above4096; hence the4096 paired cap cannot meet the95% target even with an otherwise ideal stopping rule. Its larger cap must be evaluated with unconditional cost, not only successful paths.',
            'not_an_algorithm_claim': 'Information lower bounds do not prove that any candidate achieves95% power, matches the bound, or exceeds a strong literature comparator.'},
        'numerical_method': {
            'outward': '64-significant-digit directed Decimal arithmetic from frozen direct_tests._Bounds; correctly rounded Decimal.ln is enlarged one representable endpoint each way.',
            'independent_check': 'mpmath160-digit evaluation uses direct Bernoulli KL for E and transformed Bernoulli KL for U; every reported high-precision reference is checked inside its outward interval. This is an independent numerical crosscheck, not a second rigorous interval implementation.',
            'parameter_semantics': 'Main table treats each parsed Python binary64 A,B,C as an exact rational. JSON decimal tokens and exact-decimal alternative calculations are also retained. Because the sampler rounds A+/-C, exact rational rounded sampler probabilities receive a separate PSR calculation. Arbitrary-angle theorem refers to the exact parametric model, not all finite-precision angle evaluation code.',
            'safe_display': 'For a lower bound report the LOWER endpoint rounded downward; for an upper bound report the UPPER endpoint rounded upward. High-precision reference digits are diagnostic.'},
        'rows': rows,
        'source_sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    with OUT.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'artifact': str(OUT), 'sha256': hashlib.sha256(OUT.read_bytes()).hexdigest(),
                      'kappa': str(kappa.lo),
                      'rows': [{'case_id': r['case_id'],
                                'psr_lb': r['bounds_for_exact_binary64_parameter_model']['expected_physical_shots_lower_bound_balanced_psr']['lower'],
                                'any_angle_lb': r['bounds_for_exact_binary64_parameter_model']['expected_physical_shots_lower_bound_arbitrary_adaptive_angles']['lower'],
                                'psr_cap4096_power_ceiling': r['bounds_for_exact_binary64_parameter_model']['cap4096_power_ceiling_balanced_psr']['upper']}
                               for r in rows]}, indent=2))


if __name__ == '__main__':
    main()
