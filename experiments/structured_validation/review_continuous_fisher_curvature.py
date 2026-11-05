"""Independent deterministic checks of the continuous Fisher curvature bound.

No simulator, ledger or prospective artifact is used. Exact-rational checks
corroborate the algebraic proof; mpmath comparisons are not outward certificates.
The optional JSON output is created exclusively and is never overwritten.
"""
from fractions import Fraction
from itertools import combinations
from math import comb, factorial, prod
from pathlib import Path
import argparse
import hashlib
import json

import mpmath as mp


ROOT = Path(__file__).resolve().parents[2]


def check():
    odds_grid = [Fraction(1, 1000), Fraction(1, 10), Fraction(1, 3),
                 Fraction(1, 2), Fraction(1), Fraction(2), Fraction(3),
                 Fraction(10), Fraction(1000)]
    variance_cases = 0
    maximum_ratio = Fraction(0)
    for m in range(19):
        for total in range(2 * m + 1):
            support = list(range(max(0, total - m), min(m, total) + 1))
            margin = min(total, 2 * m - total)
            for odds in odds_grid:
                first = [comb(m, k) * comb(m, total-k) * odds**k
                         for k in support]
                second = [comb(total, k) * comb(2*m-total, m-k) * odds**k
                          for k in support]
                ratio = Fraction(factorial(total) * factorial(2*m-total),
                                 factorial(m)**2)
                assert all(y == ratio*x for x, y in zip(first, second))
                z = sum(first)
                mean = sum(k*w for k, w in zip(support, first)) / z
                variance = sum(k*k*w for k, w in zip(support, first)) / z - mean**2
                assert 0 <= variance <= Fraction(margin, 4) <= Fraction(m, 4)
                if margin:
                    maximum_ratio = max(maximum_ratio, 4*variance/margin)
                variance_cases += 1

    covariance_checks = 0
    for size in range(2, 9):
        weights = [Fraction(2*i+1, i+1) for i in range(size)]
        for total in range(size+1):
            subsets = [(set(subset), prod(weights[i] for i in subset))
                       for subset in combinations(range(size), total)]
            z = sum(weight for subset, weight in subsets)
            for i, j in combinations(range(size), 2):
                pi = sum(w for subset, w in subsets if i in subset) / z
                pj = sum(w for subset, w in subsets if j in subset) / z
                pij = sum(w for subset, w in subsets if i in subset and j in subset) / z
                remaining = [weights[k] for k in range(size) if k not in (i, j)]

                def elementary(degree):
                    if degree < 0 or degree > len(remaining):
                        return Fraction(0)
                    return sum((prod(remaining[k] for k in subset)
                                for subset in combinations(range(len(remaining)), degree)),
                               Fraction(0))

                a, b, c = [elementary(total-shift) for shift in (0, 1, 2)]
                assert b*b >= a*c
                cov = pij-pi*pj
                assert cov == weights[i]*weights[j]*(a*c-b*b)/z**2 <= 0
                covariance_checks += 1

    mp.mp.dps = 90
    analytic_checks = 0
    largest_bound_fraction = mp.mpf(0)
    for b in range(9):
        for c in range(9):
            s = b+c
            complete = mp.mpf(factorial(2*b)*factorial(2*c)) / (
                2**s*factorial(b)*factorial(c)*factorial(s))
            for odds in odds_grid:
                w = mp.mpf(odds.numerator)/odds.denominator
                lam = mp.log(w)
                likelihood = (2*w/(1+w))**b*(2/(1+w))**c
                gprime = mp.mpf(b) - s*w/(1+w) - mp.tanh(lam/2)/2
                lower = mp.sqrt(2/(mp.pi*(s+1)))*likelihood/mp.cosh(lam/2)
                strengthened = lower*mp.exp(2*gprime*gprime/(s+1))
                assert 0 < lower <= strengthened <= complete
                largest_bound_fraction = max(largest_bound_fraction, strengthened/complete)
                analytic_checks += 1

    paths = [
        'docs/CONTINUOUS_FISHER_CURVATURE_BOUND.md',
        'docs/CONTINUOUS_FISHER_EXPLORATION.md',
        'experiments/structured_validation/review_continuous_fisher_curvature.py',
        'results/structured_validation/continuous_fisher_literature_review.json',
    ]
    return {
        'status': 'passed',
        'reviewer': 'stream_stats independent review of proposed curvature theorem',
        'sources': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        'scope': 'New mathematical bound only; no Monte Carlo, simulator, prospective files or trial ledgers.',
        'mathematical_review': [
            'Fixed-size positive-weight subset covariance is wi*wj*(e_T*e_(T-2)-e_(T-1)^2)/Z^2; Newton gives nonpositivity, including zero-extended boundary degrees.',
            'The original fixed-T representation has m plus indicators, giving Var(K)<=m/4.',
            'The alternative fixed-m representation has T odds-weighted indicators and 2m-T unit-weight indicators. Its coefficient ratio to the original law is T!*(2m-T)!/(m!)^2, independent of K.',
            'Applying negative covariance to K and m-K gives Var(K)<=min(T,2m-T)/4. No Jacobi identity is required.',
            'For S=sum(min(T,2m-T)), the finite-support CGF second derivative is the tilted variance, and the log prior contributes at most 1/4 in absolute second derivative. Thus g_second>=-(S+1)/4.',
            'The global quadratic lower bound is g(x+h)>=g(x)+g_prime(x)*h-(S+1)*h^2/8.',
            'Full Gaussian integration yields M>=exp(g(x))*sqrt(8*pi/(S+1))*exp(2*g_prime(x)^2/(S+1)). Dropping the last factor gives sqrt(2/(pi*(S+1)))*L(x)/cosh(x/2).',
            'S<=n=N/2 gives the weaker physical-shot form 2*L(x)/(sqrt(pi*(N+2))*cosh(x/2)); the inequality, not equality, connects the S and N forms.',
            'For fixed finite x, log expert loss is at most 0.5*log(S+1)+0.5*log(pi/2)+log(cosh(x/2)).',
            'The bound is simultaneous and pathwise in finite x, so post-data choices are allowed while retaining the actual location penalty. Infinite-odds endpoints are not directly covered.',
            'S=0 and n=0 are valid because S+1>0; zero or all-success margins contribute zero conditional variance.',
            'A numerically certified evaluation can serve as a lower bound for the existing signed/common-order risk argument, but current finite chord values do not automatically inherit this bound.',
        ],
        'deterministic_checks': {
            'distribution_equivalence_and_variance_cases': variance_cases,
            'm_range_inclusive': [0, 18],
            'all_valid_total_margins_checked': True,
            'rational_odds': [str(w) for w in odds_grid],
            'largest_4_variance_over_minority_margin': str(maximum_ratio),
            'exact_heterogeneous_weight_pair_covariance_checks': covariance_checks,
            'heterogeneous_subset_population_sizes': [2, 8],
            'complete_paired_beta_analytic_comparisons': analytic_checks,
            'mpmath_digits': mp.mp.dps,
            'largest_derivative_strengthened_bound_over_exact_beta': mp.nstr(largest_bound_fraction, 30),
            'mpmath_checks_are_outward_certificates': False,
        },
        'limits': [
            'Finite exact examples corroborate but do not replace the general Newton and Gaussian proof.',
            'No first-crossing cost, power within a finite cap, or SOTA advantage is established.',
            'Half-logarithmic mixture regret has classical precedent; priority of this explicit model-specific derivation is unresolved.',
            'A separately implemented floating-point or interval certificate requires its own code review and numerical checks.',
        ],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = check()
    if args.output:
        with args.output.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(report, stream, indent=2)
            stream.write('\n')
    print(json.dumps(report, indent=2))
