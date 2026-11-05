"""Independent case9 information-bound check using rational log series.

Does not import the bound producer, its interval helper, or mpmath. Positive
rational logs use range reduction and the atanh series with an exact tail
bound; all arithmetic before display is Fraction arithmetic.
"""
from datetime import datetime, timezone
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT/'results/sota_round4/theory'
SOURCE = DIR/'finite_budget_information_bounds.py'
ARTIFACT = DIR/'finite_budget_information_bounds.json'
OUT = DIR/'finite_budget_information_bounds_independent_review.json'
PLAN = ROOT/'results/sota_round2/benchmark/preliminary_plan.json'
TERMS = 80


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add(a, b):
    return a[0]+b[0], a[1]+b[1]


def scale(a, w):
    return (w*a[0], w*a[1]) if w >= 0 else (w*a[1], w*a[0])


def divide_positive(a, b):
    assert 0 < a[0] <= a[1] and 0 < b[0] <= b[1]
    return a[0]/b[1], a[1]/b[0]


def unit_log(x):
    assert 1 <= x <= 2
    z = (x-1)/(x+1)
    power, z2, total = z, z*z, Fraction(0)
    for j in range(TERMS):
        total += 2*power/(2*j+1)
        power *= z2
    remainder = 2*power/((2*TERMS+1)*(1-z2))
    return total, total+remainder


@lru_cache(maxsize=128)
def rational_log(x):
    x = Fraction(x)
    assert x > 0
    exponent = 0
    while x < 1:
        x *= 2
        exponent -= 1
    while x >= 2:
        x /= 2
        exponent += 1
    return add(unit_log(x), scale(unit_log(Fraction(2)), exponent))


def discrete_kl(probabilities, null_probabilities):
    assert sum(probabilities) == sum(null_probabilities) == 1
    answer = Fraction(0), Fraction(0)
    for p, q in zip(probabilities, null_probabilities):
        assert p >= 0 and q > 0
        if p:
            answer = add(answer, scale(rational_log(p/q), p))
    return answer


def bernoulli_kl(p, q):
    return discrete_kl((p, 1-p), (q, 1-q))


def display(value):
    with localcontext() as context:
        context.prec, context.rounding = 90, ROUND_FLOOR
        low = Decimal(value[0].numerator)/Decimal(value[0].denominator)
        context.rounding = ROUND_CEILING
        high = Decimal(value[1].numerator)/Decimal(value[1].denominator)
    return {'lower': str(low), 'upper': str(high)}


def enclosed(value, saved):
    assert Fraction(saved['lower']) <= value[0] <= value[1] <= Fraction(saved['upper'])


def main():
    artifact = json.loads(ARTIFACT.read_text(encoding='utf-8'))
    row = next(r for r in artifact['rows'] if r['case_id'] == 9)
    case = next(c for c in json.loads(PLAN.read_text())['cases'] if c['case_id'] == 9)
    A, B, C = (Fraction.from_float(case[k]) for k in ('A', 'B', 'C'))
    assert 0 < C < A < Fraction(1, 2) and B*B+C*C <= A*A
    assert all(Fraction.from_float(case[k]) == Fraction(int(row['exact_parameter_model'][k]['python_binary64_rational']['numerator']),
                                                        int(row['exact_parameter_model'][k]['python_binary64_rational']['denominator']))
               for k in ('A', 'B', 'C'))
    for name, expected in artifact['source_sha256'].items():
        assert sha(ROOT/name) == expected
    kappa = bernoulli_kl(Fraction(19, 20), Fraction(1, 100))
    enclosed(kappa, artifact['kappa'])
    pm, pp = A-C, A+C
    # A genuinely independent four-outcome pair calculation includes both
    # concordances. The producer's E formula uses Phi, not this expression.
    joint = ((1-pm)*(1-pp), (1-pm)*pp, pm*(1-pp), pm*pp)
    null = ((1-A)**2, A*(1-A), A*(1-A), A*A)
    E = scale(discrete_kl(joint, null), Fraction(1, 2))
    saved = row['bounds_for_exact_binary64_parameter_model']
    enclosed(E, saved['E_full_psr_per_shot'])
    lower = divide_positive(kappa, E)
    enclosed(lower, saved['expected_physical_shots_lower_bound_balanced_psr'])
    assert lower[0] > Fraction('4389.3758') > 4096
    ceiling = saved['cap4096_power_ceiling_balanced_psr']
    root_low, root_high = Fraction(ceiling['root_lower']), Fraction(ceiling['upper'])
    assert bernoulli_kl(root_low, Fraction(1, 100))[1] < 4096*E[0]
    assert bernoulli_kl(root_high, Fraction(1, 100))[0] > 4096*E[1]
    assert root_high < Fraction('0.911252') < Fraction(19, 20)
    required_even_cap = 2*((lower[0].numerator+2*lower[0].denominator-1)//(2*lower[0].denominator))
    assert required_even_cap == saved['necessary_minimum_even_cap_balanced_psr'] == 4390
    actual = row['actual_round3_binary64_psr_probabilities']
    assert Fraction.from_float(case['A']-case['C']) == pm and Fraction.from_float(case['A']+case['C']) == pp
    enclosed(E, actual['E_per_shot'])
    enclosed(lower, actual['expected_shots_lower_bound'])
    # The arbitrary-angle witness is physically feasible for case9. The
    # uniform pointwise KL theorem itself remains the cited analytic input.
    v = C/A
    lam = scale(add(rational_log(1+v), scale(rational_log(1-v), -1)), 1/(2*v))
    enclosed(lam, saved['q_star_lambda'])
    bstar = scale(lam, B)
    assert 0 < bstar[0] <= bstar[1] < A
    margin = A-bstar[1], A-bstar[0]
    enclosed(margin, saved['q_star_physical_margin_m_abs_bstar'])
    phi = scale(add(scale(rational_log(1+v), 1+v), scale(rational_log(1-v), 1-v)), Fraction(1, 2))
    U = scale(phi, 2*A)
    enclosed(U, saved['U_pointwise_any_angle_per_shot'])
    arbitrary_lower = divide_positive(kappa, U)
    enclosed(arbitrary_lower, saved['expected_physical_shots_lower_bound_arbitrary_adaptive_angles'])
    assert 4096*U[0] > rational_log(Fraction(100))[1]
    inputs = {p.relative_to(ROOT).as_posix(): sha(p) for p in (SOURCE, ARTIFACT, PLAN,
              ROOT/'docs/CONTINUOUS_PSR_INFORMATION_BOUND.md', ROOT/'docs/CONTINUOUS_SYMMETRIC_DESIGN.md')}
    result = {'status': 'passed_with_explicit_scope', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'reviewer_source': Path(__file__).resolve().relative_to(ROOT).as_posix(), 'reviewer_sha256': sha(Path(__file__).resolve()),
              'input_sha256': inputs, 'new_simulation_observations': 0, 'empirical_costs_or_outcomes_read': False,
              'scope': 'Mathematical source review plus independent rigorous rational-series evaluation of kappa and case9; not a rerun of all numerical rows.',
              'mathematical_findings': [
                  {'topic': 'binary data processing', 'status': 'passed',
                   'reason': 'For the same truth-independent randomized procedure under theta and a physical null eta, terminal correct-sign event R has p>=.95 and q<=.01. Since p>q, kl(p,q)>=kl(.95,.01). The null constraint is on any sign report; assigning its entire.01 budget to R gives a valid possibly loose bound.'},
                  {'topic': 'finite adaptive stopping', 'status': 'passed',
                   'reason': 'Pad the full transcript to finite cap with a deterministic stop symbol. Identical policy/randomization/decision kernels contribute zero KL. Chain rule sums observation KL over paid queries; no unbounded optional-stopping theorem is required.'},
                  {'topic': 'complete balanced independent pairs', 'status': 'essential_and_correctly_stated',
                   'reason': 'At fixed cos=0,sin=+/-1, one independent observation per arm gives joint KL=kl(A-C,A)+kl(A+C,A)=2E. Starting each complete pair is predictable, so expected transcript KL=E*expected physical shots. Batches or coarsened records do not increase information. Mid-pair stopping, unequal allocation, correlated arms, or uncharged information are outside this exact identity.'},
                  {'topic': 'B dependence', 'status': 'passed',
                   'reason': 'B vanishes exactly for fixed PSR coordinates cos=0; therefore the paired E bound depends on A,C only. A constant null eta=(A,0,0) is physical. B generally affects other angles; the arbitrary-angle witness q*=A+B*lambda*cos(x) is distinct and has been checked physical for case9.'},
                  {'topic': 'arbitrary-angle extension', 'status': 'consistent_with_cited_theorem',
                   'reason': 'The same single physical null witness must bound every queried angle, which permits the adaptive chain-rule upper bound U*E[N]. This review checks its case9 numerical witness but does not independently reprove the complete prior pointwise theorem. For case9 the U information budget at4096 yields no nontrivial power ceiling.'},
                  {'topic': 'scope of expected-cost lower bound', 'status': 'passed',
                   'reason': 'E[N]>=kappa/E applies only when the algorithm actually achieves95% correct-stop probability on this case. N includes errors and cap abstentions. Expected cost need not be integer/even; only a complete-pair hard cap can be rounded to the necessary even minimum4390.'}],
              'independent_numeric_method': {'logarithms': 'Exact Fraction range reduction x=2^k*y with1<=y<2; log(y)=2*sum z^(2j+1)/(2j+1), z=(y-1)/(y+1), with exact positive residual <=2*z^(2L+1)/((2L+1)*(1-z^2)).',
                                              'terms': TERMS, 'arithmetic': 'Exact rational endpoints; directed Decimal90 used only for displaying them.',
                                              'producer_or_interval_helper_imported': False,
                                              'mpmath_used': False, 'stronger_checks': 'Independent narrower kappa/E/lambda/U/cost intervals lie inside the saved outward intervals; saved power-root endpoints bracket the root by exact rational interval inequalities.'},
              'case9_verified': {'kappa': display(kappa), 'complete_pair_information_per_physical_shot': display(E),
                                 'necessary_expected_cost_for95pct_correct': display(lower),
                                 'cap4096_correct_probability_bracket': {'lower': str(Decimal(ceiling['root_lower'])), 'upper': str(Decimal(ceiling['upper']))},
                                 'safe_public_power_upper_percent': '91.1252', 'safe_public_expected_cost_lower': '4389.3758',
                                 'necessary_even_hard_cap': 4390, 'rounded_sampler_probabilities_match_exact_binary64_case9_model': True,
                                 'physical_null_witness_margin': display(margin), 'arbitrary_angle_necessary_expected_cost': display(arbitrary_lower)},
              'limits_for_sota_comparison': [
                  'The91.1252% ceiling is specific to fixed balanced complete independent PSR pairs at4096 shots; do not extend it to arbitrary measurement designs.',
                  'The lower bound is necessary and need not be attainable. It neither proves16384 is necessary nor establishes a95%-power algorithm at4390.',
                  'An implemented procedure with about90% aggregate power is not a95%-power cost comparator. Casewise power, not aggregate power alone, matters.',
                  'Current low observed case9 power is not shown close to the91.1252% ceiling; insufficient cap does not explain all remaining algorithmic inefficiency.',
                  'Neither this information bound nor comparison with it establishes SOTA superiority.'],
              'nonblocking_implementation_scope_note': 'The numerical calculate routine forms m-B*lambda for its margin label. All current input rows haveB>=0, so it is correct here; a future negative-B extension should explicitly use m-abs(B)*lambda or normalize the sign. This does not affect case9, the paired E formula, or the sign-invariant mathematical theorem.'}
    for name, expected in inputs.items():
        assert sha(ROOT/name) == expected
    with OUT.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'sha256': sha(OUT), 'case9_verified': result['case9_verified']}), flush=True)


if __name__ == '__main__':
    main()
