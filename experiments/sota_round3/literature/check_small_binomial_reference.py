"""Deterministic exact-probability checks of documented MaxSPRT conventions.

This is a new implementation of the mathematical definition. Official R code
is inspected but is not executed; these are not claimed as R runtime outputs.
No random number generator or simulation observations are used.
"""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'results/sota_round3/literature'


def glr(n, k, p=Fraction(1, 2), two=False):
    if two:
        assert p == Fraction(1, 2)
        k = max(k, n-k)
    if Fraction(k, n) <= p:
        return 0.
    result = k*math.log(float(Fraction(k, n)/p)) if k else 0.
    if k != n:
        result += (n-k)*math.log(float(Fraction(n-k, n)/(1-p)))
    return result


def boundaries(nmax, threshold, p=Fraction(1, 2), two=False, looks=None):
    looks = set(range(1, nmax+1)) if looks is None else set(looks)
    result = []
    for n in range(1, nmax+1):
        crossed = [k for k in range(n+1) if n in looks and glr(n, k, p, two) >= threshold]
        result.append(crossed)
    return result


def exact_probability(rows, p=Fraction(1, 2)):
    a, b = p.numerator, p.denominator
    survivor, absorbed = [1], 0
    for n, crossing in enumerate(rows, 1):
        fresh = [0]*(n+1)
        for k, count in enumerate(survivor):
            fresh[k] += (b-a)*count
            fresh[k+1] += a*count
        absorbed *= b
        for k in crossing:
            absorbed += fresh[k]
            fresh[k] = 0
        survivor = fresh
        assert absorbed+sum(survivor) == b**n
    return Fraction(absorbed, b**len(rows))


def calibrate(nmax, alpha, two):
    scores = sorted(set(glr(n, k, two=two) for n in range(1, nmax+1) for k in range(n+1)))
    low, high = 0, len(scores)-1
    while low < high:
        mid = (low+high)//2
        risk = exact_probability(boundaries(nmax, scores[mid], two=two))
        if risk <= alpha:
            high = mid
        else:
            low = mid+1
    threshold = scores[low]
    rows = boundaries(nmax, threshold, two=two)
    risk = exact_probability(rows)
    preceding = exact_probability(boundaries(nmax, scores[low-1], two=two)) if low else None
    assert risk <= alpha and (preceding is None or preceding > alpha)
    return {'N': nmax, 'alpha': [alpha.numerator, alpha.denominator], 'two_sided': two,
            'shape_threshold_float': threshold, 'crossing_convention': 'statistic >= threshold',
            'exact_crossing_probability': [risk.numerator, risk.denominator], 'crossing_probability_float': float(risk),
            'previous_distinct_shape_risk_float': float(preceding) if preceding is not None else None,
            'upper_crossing_integer': [min([k for k in row if 2*k > n], default=n+1) for n, row in enumerate(rows, 1)],
            'all_crossing_states': rows}


def main():
    p = Fraction(2, 9)
    rows = boundaries(25, 1.9852, p=p, looks=range(5, 26, 5))
    risk = exact_probability(rows, p)
    assert abs(float(risk)-.04775995) < 5e-9
    result = {
        'status': 'passed',
        'scope': 'Deterministic independent integer path-count calculation, no external R execution or simulation.',
        'official_manual_example': {'location': 'Sequential 4.6.3 reference manual p82, Example 2',
            'N': 25, 'alpha': .05, 'M': 1, 'z': [7, 2], 'p': [2, 9], 'GroupSizes': 5, 'Tailed': 'upper',
            'documented_cv': 1.9852, 'documented_Type_I_Error': .04775995,
            'recomputed_exact_probability': [risk.numerator, risk.denominator],
            'recomputed_probability_float': float(risk), 'all_crossing_states': rows},
        'small_fair_reference_tables': [calibrate(n, Fraction(1, 100), two) for n in (12, 20, 32, 50) for two in (False, True)],
        'sources': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
            'experiments/sota_round3/literature/check_small_binomial_reference.py',
            'results/sota_round3/literature/sequential_4.6.3.tar.gz',
            'results/sota_round3/literature/sequential_source_CV.Binomial.R',
            'results/sota_round3/literature/sequential_manual.pdf')},
        'limitations': ['The symmetric two-sided tables extend the documented upper-side statistic; current CV.Binomial source rejects Tailed=two.',
                       'Threshold floating values identify boundary shapes; integer absorbing-state tables and exact probabilities are authoritative.',
                       'These tables do not establish global optimality among boundary shapes or alternatives.']}
    target = OUT/'small_binomial_reference.json'
    assert not target.exists()
    target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'passed','path':str(target),'official_example_risk':float(risk),
                      'small_tables':[{k:r[k] for k in ('N','two_sided','shape_threshold_float','crossing_probability_float')}
                                      for r in result['small_fair_reference_tables']]}))


if __name__ == '__main__':
    main()
