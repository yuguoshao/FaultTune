"""Independent audit of the continuum certificate source and retained leaves.

Replays every retained upper enclosure, checks exact coverage and parameter
semantics, and independently checks KL calculus and capacities. No random data.
"""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time

import mpmath as mp
import sympy as sp

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.sota_round6 import continuum_null_certificate as producer

mp.mp.dps = 150


def mm(value):
    value = Fraction(value)
    return mp.mpf(value.numerator)/value.denominator


def contained(interval, value):
    return mm(interval.lo) <= value <= mm(interval.hi)


def f(p):
    return p*mp.log(p)+(1-p)*mp.log(1-p)


def independent_kl(p, q):
    return f(p)-p*mp.log(q)-(1-p)*mp.log(1-q)


def symbolic_checks():
    t = sp.symbols('t', real=True)
    p, q = sp.symbols('p q', positive=True)
    residuals = {}
    for sign in (-1, 1):
        c, s = sign*(1-t*t)/(1+t*t), 2*t/(1+t*t)
        formulas = {
            'unit_circle': c*c+s*s-1,
            'cos_first': sp.diff(c, t)+4*sign*t/(1+t*t)**2,
            'sin_first': sp.diff(s, t)-2*(1-t*t)/(1+t*t)**2,
            'cos_second': sp.diff(c, t, 2)+4*sign*(1-3*t*t)/(1+t*t)**3,
            'sin_second': sp.diff(s, t, 2)+4*t*(3-t*t)/(1+t*t)**3
        }
        for name, expression in formulas.items():
            assert sp.simplify(expression) == 0
            residuals['chart_%s_%s' % (sign, name)] = 'exact symbolic residual0'
    # On the reviewed domain 0<p,q<1 these log arguments are positive.
    # The expanded real expression avoids asking SymPy to infer p,q<1.
    divergence = p*sp.log(p)+(1-p)*sp.log(1-p)-p*sp.log(q)-(1-p)*sp.log(1-q)
    expressions = {
        'KL_p_first': sp.diff(divergence, p)-(sp.log(p)-sp.log(1-p)-sp.log(q)+sp.log(1-q)),
        'KL_q_first': sp.diff(divergence, q)-(q-p)/(q*(1-q)),
        'KL_pp_second': sp.diff(divergence, p, 2)-1/(p*(1-p)),
        'KL_pq_second': sp.diff(divergence, p, q)+1/(q*(1-q)),
        'KL_qq_second': sp.diff(divergence, q, 2)-p/q**2-(1-p)/(1-q)**2
    }
    for name, expression in expressions.items():
        assert sp.simplify(expression) == 0
        residuals[name] = 'exact symbolic residual0'
    return residuals


def main():
    started = time.monotonic()
    source = ROOT/'results/sota_round6/theory/continuum_null_certificate.json'
    registry_path = ROOT/'results/sota_round5/continuous_chernoff_refinement.json'
    certificate = json.loads(source.read_text(encoding='utf-8'))
    registry = json.loads(registry_path.read_text(encoding='utf-8'))
    for name, digest in certificate['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    symbols = symbolic_checks()
    rows = []
    leaf_count = 0
    derivative_points = 0
    assert len(certificate['rows']) == 48
    assert sorted(row['registry_index'] for row in certificate['rows']) == list(range(48))
    for row in certificate['rows']:
        index = row['registry_index']
        original = registry['rows'][index]
        theta = tuple(Fraction(value) for value in row['theta_exact'])
        null = tuple(Fraction(value) for value in row['null_exact'])
        z = Fraction(row['cosine_exact'])
        assert theta == tuple(Fraction(original['model'][key]) for key in ('A', 'B', 'C'))
        assert null == tuple(Fraction(value) for value in original['reflected_pair']['null_candidate'][:2])
        assert z == Fraction(original['reflected_pair']['cosine'])
        assert row['status'] == 'certified' and not row['unresolved_leaves']
        a, b, c = theta
        an, bn = null
        assert 0 < a < 1 and b*b+c*c < min(a, 1-a)**2
        assert 0 < an-abs(bn) <= an+abs(bn) < 1
        target = Fraction(row['acceptance_target_downward'])
        lower = Fraction(row['pair_information_interval'][0])
        upper_pair = Fraction(row['pair_information_interval'][1])
        tolerance = Fraction(row['relative_tolerance_exact'])
        assert 0 < lower <= upper_pair
        assert target <= lower*(1+tolerance)
        model = producer.Model(theta, null)
        max_upper = Fraction(0)
        for sign in (-1, 1):
            leaves = sorted((leaf for leaf in row['accepted_leaves'] if leaf['chart'] == sign),
                            key=lambda leaf: Fraction(leaf['left']))
            assert leaves and Fraction(leaves[0]['left']) == -1
            endpoint = Fraction(-1)
            for leaf in leaves:
                left, right = Fraction(leaf['left']), Fraction(leaf['right'])
                depth = leaf['depth']
                assert isinstance(depth, int) and depth >= 0
                assert left == endpoint and -1 <= left < right <= 1
                width = Fraction(2, 2**depth)
                assert right-left == width
                assert ((left+1)/width).denominator == 1
                bound = Fraction(leaf['upper'])
                assert bound <= target
                replayed = model.interval_upper(left, right, sign)
                assert Fraction(replayed) == bound
                max_upper = max(max_upper, bound)
                endpoint = right
                leaf_count += 1
            assert endpoint == 1
        assert {leaf['chart'] for leaf in row['accepted_leaves']} == {-1, 1}
        assert Fraction(row['all_angle_information_upper']) == max_upper
        reported_ratio = Fraction(row['upper_over_pair_lower'])
        assert max_upper/lower <= reported_ratio <= 1+tolerance
        # Independently recompute the exact-model capacity at 150 digits.
        am, bm, cm, zm = map(mm, (a, b, abs(c), z))
        sm = mp.sqrt(1-zm*zm)
        pm, pp = am+bm*zm-cm*sm, am+bm*zm+cm*sm
        secant = (f(pp)-f(pm))/(pp-pm)
        qc = 1/(1+mp.exp(-secant))
        weight = (qc-pm)/(pp-pm)
        dl, dr = independent_kl(pm, qc), independent_kl(pp, qc)
        assert pm < qc < pp and 0 < weight < 1
        assert abs(dl-dr) < mp.mpf('1e-130')
        assert mm(lower) <= dl <= mm(upper_pair)
        # Independent differentiation of the scalar rational chart function.
        for sign in (-1, 1):
            def direct(t):
                cosine = sign*(1-t*t)/(1+t*t)
                sine = 2*t/(1+t*t)
                p = mm(a)+mm(b)*cosine+mm(c)*sine
                q = mm(an)+mm(bn)*cosine
                return independent_kl(p, q)
            for point in (Fraction(-2, 3), Fraction(0), Fraction(3, 4)):
                enclosure = producer.enclosure(point)
                values = model.derivatives(enclosure, sign)
                for order, interval in enumerate(values):
                    reference = mp.diff(direct, mm(point), order)
                    assert contained(interval, reference), (index, sign, str(point), order)
                derivative_points += 1
        rows.append({'registry_index': index,
                     'leaf_count': len(row['accepted_leaves']),
                     'exact_cover_and_full_replay': 'passed',
                     'independent_capacity_contained_150_digits': True,
                     'independent_KL_and_two_derivative_points': 6,
                     'upper_over_pair_lower_exact': str(max_upper/lower)})
    result = {
        'status': 'passed_independent_source_and_all_48_model_leaf_audit',
        'new_simulation_observations': 0,
        'blocking_findings': [],
        'input_sha256': {
            source.relative_to(ROOT).as_posix(): hashlib.sha256(source.read_bytes()).hexdigest(),
            **certificate['source_sha256'],
            Path(__file__).resolve().relative_to(ROOT).as_posix(): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        },
        'source_review': [
            {'item': 'Directed arithmetic', 'verdict': 'passed',
             'reason': 'The imported 64-digit _Bounds uses outward ROUND_FLOOR/CEILING arithmetic. Decimal ln/exp/sqrt are correctly rounded and expanded by next_minus/next_plus. Exact Decimal endpoints are not rounded by construction, sign changes use copy_negate, and abs_upper uses copy_abs. Every rational is enclosed by interval division of exact integer numerator and denominator. No ambient Decimal arithmetic affects a returned bound.'},
            {'item': 'Rational chart coverage and calculus', 'verdict': 'passed',
             'reason': 'For each sign the chart lies on the unit circle. Its sine runs continuously and monotonically from-1 to1 while cosine has the prescribed sign, so the two closed chart domains cover the full circle. Both first and second derivatives match exact symbolic differentiation.'},
            {'item': 'Physical range intersections', 'verdict': 'passed',
             'reason': 'The exact rational physical checks exclude boundary rates. The directed A+/-sqrt(B^2+C^2) enclosure contains every true probability, and a+/-|b| contains every null probability. Intersecting these global ranges with local chart enclosures preserves containment. If interval precision cannot resolve strict physicality, construction raises rather than certifies.'},
            {'item': 'KL derivatives and Taylor bound', 'verdict': 'passed',
             'reason': 'Both KL partial derivatives and the full Hessian match independent symbolic differentiation. The chain-rule second derivative includes both quadratic derivative terms and the p/q second-derivative terms. The midpoint and halfwidth are exact rationals; directed midpoint enclosures contain the exact derivatives. F(mid)+|Fprime(mid)|h+sup|Fsecond|h^2/2 bounds every chart point, and taking its minimum with a separate direct interval upper remains valid.'},
            {'item': 'Reflected-pair lower bound', 'verdict': 'passed',
             'reason': 'The entropy secant center lies strictly between the ordered pair rates and equalizes their KL values. The capacity weight pools at that center. Every affine sine-zero null has a common rate on the pair, and a constant physical null attains that pooled value, so the exact pair capacity is a valid oracle lower bound. Intersecting the two outward KL intervals is justified by their exact equality. Negative C is handled by swapping labels through abs(C).'},
            {'item': 'Global upper/lower sandwich', 'verdict': 'passed',
             'reason': 'Each model uses one fixed physical affine null on all leaves. Exact coverage therefore gives a pointwise all-angle upper bound. For every design, its infimum over nulls is at most its divergence to this fixed null, hence at most the pointwise bound. Together with the reflected lower bound this allows arbitrary support counts and physical boundary nulls without a minimax exchange. It does not prove exact reflected-pair optimality; the certified gap is at most1e-6 relative.'},
            {'item': 'Threshold rounding and semantics', 'verdict': 'passed',
             'reason': 'The acceptance target is rounded downward from pair_lower*(1+tolerance). Exact Fraction comparisons verify every stored leaf bound is <=target and every final exact ratio is <=1+tolerance. All parameters, proposed null coefficients and selected cosine exactly equal Fraction(the recorded binary64 value); no subsequent Bernoulli probability is rounded back to binary64.'},
            {'item': 'Leaf coverage and source binding', 'verdict': 'passed',
             'reason': 'All48 original registry indices occur exactly once. For both charts in every model, sorted exact dyadic leaves meet without gaps or positive-width overlaps and cover[-1,1]. Every leaf width and lattice position match its claimed depth. Every stored leaf upper was replayed exactly and all input source hashes matched.'}
        ],
        'nonblocking_observations': [
            'max_nodes is a soft visit budget: it is checked only when a leaf cannot be accepted, so accepted leaves can carry the visit count beyond the requested number. This does not weaken a certified bound or permit unresolved coverage to be labeled certified.',
            'The arithmetic proof relies on documented Decimal correctly-rounded elementary functions and ordinary Python execution; it is not a formally verified arithmetic kernel.'
        ],
        'checks': {'models': len(rows), 'all_accepted_leaves_replayed': leaf_count,
                   'independent_capacity_precision_digits': mp.mp.dps,
                   'independent_scalar_chart_points_with_two_derivatives': derivative_points,
                   'exact_symbolic_identities': symbols},
        'models': rows,
        'elapsed_seconds': time.monotonic()-started,
        'claim_boundary': 'Passed fixed-model numerical continuum certificates and source review. These results are not a general all-parameter theorem, finite stopping-cost result, numerical epsilon for a perturbation theorem, or a SOTA claim.'
    }
    out = Path(__file__).with_suffix('.json')
    with out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'path': str(out), 'sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
                      'status': result['status'], 'checks': result['checks'],
                      'elapsed_seconds': result['elapsed_seconds']}, indent=2))


if __name__ == '__main__':
    main()
