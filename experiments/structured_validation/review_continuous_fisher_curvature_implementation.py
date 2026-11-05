"""Read-only curvature implementation review with exact likelihood references."""
from decimal import Decimal
from fractions import Fraction
from math import comb
from pathlib import Path
import hashlib
import json

import mpmath as mp
import continuous_fisher_curvature as target

ROOT = Path(__file__).resolve().parents[2]


def run():
    mp.mp.dps = 110
    examples = [[], [(0, 0, 0)], [(1, 0, 0), (1, 1, 1)],
                [(2, 0, 2)], [(2, 1, 1)], [(3, 0, 1)],
                [(3, 3, 2)], [(8, 3, 6), (2, 0, 1)],
                [(1, 0, 1)]*20, [(1, 1, 0)]*20]
    checks = []
    for triples in examples:
        blocks = [dict(zip(('m', 'minus_successes', 'plus_successes'), row))
                  for row in triples]
        for odds in (Fraction(1, 17), Fraction(1), Fraction(17)):
            likelihood = Fraction(1)
            n = s = score = 0
            for m, minus, plus in triples:
                total = minus+plus
                denominator = sum(Fraction(comb(m, k)*comb(m, total-k))*odds**k
                                  for k in range(max(0, total-m), min(m, total)+1))
                likelihood *= odds**plus*comb(2*m, total)/denominator
                n += m
                s += min(total, 2*m-total)
                score += plus-minus
            cert = target.certificate_from_blocks(blocks, odds)
            eta = mp.log(mp.mpf(odds.numerator)/odds.denominator)
            log_l = mp.log(mp.mpf(likelihood.numerator)/likelihood.denominator)
            for margin, key in [(s, 'single_node_S_log_bound_enclosure'),
                                (n, 'single_node_n_log_bound_enclosure')]:
                reference = log_l + mp.log(2/(mp.pi*(margin+1)))/2-mp.log(mp.cosh(eta/2))
                lo, hi = [mp.mpf(value) for value in cert[key]]
                assert lo <= reference <= hi
            assert cert['shots'] == 2*n and cert['S'] == s and cert['score'] == score
            crossing = score != 0 and Decimal(cert['log_evidence_lower_bound']) >= Decimal(cert['threshold_log_bounds'][1])
            expected = ('reject' if score > 0 else 'accept') if crossing else 'abstain'
            assert cert['decision'] == expected
            checks.append({'blocks': triples, 'odds': str(odds), 'decision': expected,
                           'n': n, 'S': s, 'exact_likelihood': str(likelihood)})
    invalid = 0
    for node in [(0, 1), (-1, 2), (1, 0), (True, 1), (1.5, 2), Fraction(-1)]:
        try:
            target.certificate_from_blocks([], node)
        except (ValueError, TypeError):
            invalid += 1
        else:
            raise AssertionError('Invalid node accepted')
    for risk in [True, 0, 1, -.1, 'NaN', 'Infinity']:
        try:
            target.certificate_from_blocks([], delta=risk)
        except (ValueError, TypeError):
            invalid += 1
        else:
            raise AssertionError('Invalid risk accepted')
    author = json.loads((ROOT/'results/structured_validation/continuous_fisher_curvature_checks.json').read_text())
    paths = list(target.source_hashes()) + [
        'docs/CONTINUOUS_FISHER_CURVATURE_BOUND.md',
        'experiments/structured_validation/continuous_fisher_curvature_checks.py',
        'experiments/structured_validation/review_continuous_fisher_curvature_implementation.py',
        'results/structured_validation/continuous_fisher_curvature_checks.json',
        'results/structured_validation/continuous_fisher_curvature_review.json']
    return {'status': 'passed', 'reviewer': 'stream_stats independent implementation review',
            'sources': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
            'scope': 'Read-only source review and 30 new deterministic small inputs; no simulator or trial ledger reads and no Monte Carlo.',
            'implementation_findings': [
                'Reuses reviewed outward endpoint and Machin-pi routines. g+0.5*log(8*pi/(S+1)) is the correct whole-line lower formula.',
                'Both S and n are exact integers from validated balanced blocks; shots=2*n and score=2*U.',
                'Positive exact rational odds may be selected after data without changing the prior.',
                'Only the S lower endpoint is compared to the upper endpoint of -log(delta); zero score always abstains.',
                'Returned interval upper endpoints enclose the analytic lower bound, not the complete mixture.',
                'The derivative-square factor is conservatively omitted. No automatic finite-node power or sample-cost advantage is claimed.'],
            'independent_checks': {'exact_rational_likelihood_formula': 'Product of odds^K * choose(2m,T) / sum_j choose(m,j) choose(m,T-j) odds^j',
                                   'reference_precision_digits': mp.mp.dps,
                                   'bound_enclosure_cases': len(checks), 'invalid_inputs_rejected': invalid,
                                   'cases': checks,
                                   'high_precision_references_are_outward_certificates': False},
            'author_checks_inspected_not_rerun': {'exact_variance_cases': author['exact_variance_cases'],
                                                 'analytic_cases': len(author['analytic_cases']),
                                                 'stored_development_endpoints': len(author['records']),
                                                 'summary': author['summary']},
            'limits': ['No prospective data or full development ledger was reread.',
                       'Neither theory nor this implementation review demonstrates earlier first crossing or SOTA superiority.']}


if __name__ == '__main__':
    report = run()
    output = ROOT/'results/structured_validation/continuous_fisher_curvature_implementation_review.json'
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'status': report['status'], 'cases': len(report['independent_checks']['cases']),
                      'invalid_inputs': report['independent_checks']['invalid_inputs_rejected'],
                      'output': str(output)}))
