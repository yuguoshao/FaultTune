"""Independent rational and SciPy checks of the learned-design certificate.

These finite checks supplement the conditional-test argument; they do not
replace its independent-pilot, stationary-Bernoulli and fixed-look assumptions.
No private simulator probability is passed to the certificate.
"""

import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
import random
import time

from certificates import (
    _hypergeom_left_enclosure, _selected_tail, fisher_certificate,
    fisher_log_tail_screen, sufficient_final_gap,
)


ROOT = Path(__file__).resolve().parents[2]


def rational_tail(k, population, successes, draws):
    lower = max(0, draws - (population - successes))
    upper = min(k, draws, successes)
    return Fraction(sum(math.comb(successes, j)
                        * math.comb(population - successes, draws - j)
                        for j in range(lower, upper + 1)),
                    math.comb(population, draws))


def exact_small_tails():
    checked = truncated = 0
    for population in range(2, 17):
        for draws in range(1, population):
            for successes in range(1, population):
                lower = max(0, draws - (population - successes))
                upper = min(draws, successes)
                for k in range(lower, upper + 1):
                    if k * population >= draws * successes:
                        continue
                    exact = rational_tail(k, population, successes, draws)
                    for terms in (1, 3, 32):
                        bounds, details = _hypergeom_left_enclosure(
                            k, population, successes, draws, terms)
                        assert Fraction(bounds.lo) <= exact <= Fraction(bounds.hi)
                        checked += 1
                        truncated += not details['complete_tail']
    return {'rational_tail_enclosures': checked,
            'including_truncated_geometric_remainders': truncated,
            'exhaustive_population_range': [2, 16],
            'term_caps': [1, 3, 32]}


def exact_rejection_probabilities():
    """Exhaust all count tables and conditional null rejection probabilities."""
    checked = conditional_checked = 0
    max_direction = Fraction(0)
    max_total = Fraction(0)
    # This relatively large risk makes small-n tests nonvacuous. The same
    # algebra also applies to the production risk .01, checked separately.
    delta, look_count = Fraction(1, 4), 1
    alpha = delta / (2 * look_count)
    for n0 in range(1, 9):
        for n1 in range(1, 9):
            mass = {}
            for s0 in range(n0 + 1):
                for s1 in range(n1 + 1):
                    cert = fisher_certificate(s0, n0, s1, n1, 1, 1, delta=delta)
                    selected = _selected_tail(s0, n0, s1, n1)
                    if selected:
                        _, k, population, successes, draws, _ = selected
                        exact = rational_tail(k, population, successes, draws)
                        lo, hi = map(Fraction, cert['p_value_decimal_bounds'])
                        assert lo <= exact <= hi
                        if cert['decision'] != 'abstain':
                            assert exact <= alpha
                    total = s0 + s1
                    if total not in mass:
                        mass[total] = {'accept': Fraction(0), 'reject': Fraction(0)}
                    if cert['decision'] != 'abstain':
                        prob = Fraction(math.comb(n0, s0) * math.comb(n1, s1),
                                        math.comb(n0+n1, total))
                        mass[total][cert['decision']] += prob
                    checked += 1
            for event in mass.values():
                assert event['accept'] <= alpha
                assert event['reject'] <= alpha
                assert sum(event.values()) <= delta
                max_direction = max(max_direction, *event.values())
                max_total = max(max_total, sum(event.values()))
                conditional_checked += 1
    return {'exact_count_tables': checked,
            'conditional_null_events': conditional_checked,
            'arm_size_range': [1, 8], 'delta_exact': str(delta),
            'largest_direction_error_exact': str(max_direction),
            'largest_either_direction_error_exact': str(max_total)}


def large_and_boundary_checks():
    """SciPy is independent floating evidence, never decision authority."""
    from scipy.stats import hypergeom
    rng = random.Random(20261103)
    cases = [(13, 100, 3, 70), (550, 1000, 440, 900),
             (501000, 1000000, 1194000, 2400000),
             (5, 10000000, 0, 1000000),
             (0, 1000000000, 2, 100000000),
             (1000000000, 1000000000, 99999998, 100000000)]
    # Condition on total successes, place the count in tails near and away
    # from common alpha values, and then recover the two-arm table.
    for _ in range(96):
        population = rng.choice((100, 1000, 10000, 100000, 1000000))
        draws = max(1, int(population * rng.choice((.1, .25, .5, .9))))
        successes = max(1, int(population * rng.choice((.001, .01, .1, .5, .9, .999))))
        mu = draws * successes / population
        variance = draws * (successes / population) * (1 - successes / population)
        variance *= (population - draws) / (population - 1)
        lower = max(0, draws - (population - successes))
        k = max(lower, int(math.floor(mu-rng.choice((1., 2., 3.4, 3.9, 4.5, 6.))*math.sqrt(variance))))
        if k * population >= draws * successes:
            continue
        cases.append((successes-k, population-draws, k, draws))
    checked = decisions = symmetry = screens = 0
    max_relative_upper_slack = 0.
    for counts in cases:
        cert = fisher_certificate(*counts, look_index=3, look_count=80)
        selected = _selected_tail(*counts)
        direction, k, population, successes, draws, _ = selected
        reference = float(hypergeom.cdf(k, population, successes, draws))
        lo, hi = cert['p_value_bounds']
        # scipy.special's binary64 combinatorics can lose relative precision
        # for giant populations. Decimal enclosure remains the authority.
        tol = 3e-6 if population >= 100000000 else 2e-7
        assert lo <= reference * (1+tol) + 1e-300, (counts, lo, reference, hi)
        assert hi >= reference * (1-tol) - 1e-300, (counts, lo, reference, hi)
        if reference > 1e-250:
            max_relative_upper_slack = max(max_relative_upper_slack, hi/reference-1)
        if cert['decision'] != 'abstain':
            assert reference <= float(Fraction(cert['tail_alpha_exact'])) * (1+tol)
            decisions += 1
        s0,n0,s1,n1 = counts
        complemented = fisher_certificate(n0-s0,n0,n1-s1,n1,3,80)
        opposite = {'accept':'reject','reject':'accept','abstain':'abstain'}
        assert complemented['decision'] == opposite[cert['decision']]
        assert complemented['p_value_decimal_bounds'] == cert['p_value_decimal_bounds']
        symmetry += 1
        screen, log_alpha = fisher_log_tail_screen(*counts, look_count=80)
        assert math.isfinite(screen) and math.isfinite(log_alpha)
        screens += 1
        checked += 1
    return {'scipy_tail_checks': checked, 'certified_crossings_checked': decisions,
            'exact_complement_symmetry_checks': symmetry,
            'screen_finiteness_checks': screens,
            'largest_relative_upper_slack': max_relative_upper_slack,
            'note': 'Truncated tail enclosures may be conservative; screens cannot authorize decisions.'}


def boundary_and_api_checks():
    checked = 0
    # Empty arms and zero or all successes cannot establish a difference.
    for counts in ((0,0,0,0),(0,0,1,1),(0,1,0,0),(0,10,0,20),(10,10,20,20),(2,4,3,6)):
        assert fisher_certificate(*counts,1,10)['decision'] == 'abstain'
        checked += 1
    invalid = [(-1,10,0,10,1,1),(11,10,0,10,1,1),(True,10,0,10,1,1),
               (0,10.,0,10,1,1),(0,10,0,10,0,1),(0,10,0,10,2,1),
               (0,10,0,10,1,0)]
    invalid_checked = 0
    for args in invalid:
        try:
            fisher_certificate(*args)
        except ValueError:
            invalid_checked += 1
        else:
            raise AssertionError(('invalid input accepted', args))
    for delta in (0,1,-.1,float('nan'),float('inf'),True):
        try:
            fisher_certificate(0,10,1,10,1,1,delta=delta)
        except (ValueError, OverflowError):
            invalid_checked += 1
        else:
            raise AssertionError(('invalid risk accepted', delta))
    # Displayed power bound is symmetric and decreases when both arm sizes
    # are multiplied, at fixed look count and risk; this is not a power test.
    a = sufficient_final_gap(1000,3000,80)
    b = sufficient_final_gap(3000,1000,80)
    c = sufficient_final_gap(2000,6000,80)
    assert a == b
    assert c['sufficient_true_gap'] < a['sufficient_true_gap']
    assert abs(a['conditional_power_lower']-.95) < 1e-15
    return {'nondirectional_boundary_checks': checked,
            'invalid_inputs_rejected': invalid_checked,
            'power_design_function_checks': 3}


def main():
    started = time.perf_counter()
    result = {'passed': True,
              'scope': 'finite numerical cross-checks plus independent review of the statistical argument',
              'statistical_review': {
                  'null_test': 'Conditional on total successes, equality gives a central hypergeometric distribution.',
                  'composite_null': 'The conditional noncentral hypergeometric family is monotone in its odds ratio.',
                  'multiple_looks': 'Each of two directions at every predeclared look receives delta/(2L); union bound controls equality error by delta and each wrong direction by delta/2.',
                  'pilot': 'Condition on an independent pilot freezing the angle, allocation and entire finite look schedule, then integrate over that pilot.',
                  'tail_enclosure': 'Lower partial sum and decreasing-ratio geometric remainder enclose the conditional tail with outward arithmetic.',
                  'hoeffding': 'Both the selected and complementary samples have deviation cross/(n0+n1); the smaller sample gives a valid Hoeffding upper bound.',
                  'power': 'True gap strictly exceeding safety threshold plus concentration margin implies potential final-look power >=1-beta. Earlier wrong stopping costs <=delta/2; a good-pilot event costs its failure probability.',
                  'required_assumptions': 'Independent stationary Bernoulli validation observations; validation counts exclude pilot; all look-specific sample sizes fixed conditional on pilot; known orientation of the target contrast.'}}
    result['small_tails'] = exact_small_tails()
    result['conditional_tests'] = exact_rejection_probabilities()
    result['large_and_boundary'] = large_and_boundary_checks()
    result['api'] = boundary_and_api_checks()
    result['elapsed_seconds'] = time.perf_counter()-started
    result['source_sha256'] = {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (Path(__file__).resolve(),
                                            Path(__file__).resolve().with_name('certificates.py'),
                                            ROOT/'experiments/circuit_validation/direct_tests.py')}
    destination = ROOT/'results/learned_validation/certificate_checks.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__ == '__main__':
    main()
