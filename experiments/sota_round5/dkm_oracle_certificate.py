"""Paid-count-only certificates for DKM's opposite-answer best response.

This adds a rigorous convexity-gap enclosure to the old numerical oracle and
an exact two-PSR-arm best response. It is a task adapter, not a full DKM paper
or official-code reproduction. No observations are sampled in this module.
"""
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import math
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/sota_round2/literature'))
import dkm_baseline as old
from direct_tests import _Bounds

PSR_ARMS = ({'cos_numerator': 0, 'sin_numerator': -1, 'denominator': 1},
            {'cos_numerator': 0, 'sin_numerator': 1, 'denominator': 1})
OUT = ROOT/'results/sota_round5/literature/dkm_oracle_component_checks.json'


def rational(value):
    if isinstance(value, bool):
        raise ValueError('Booleans are not numerical inputs')
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise ValueError('Finite values required')
        return Fraction.from_float(float(value))
    return Fraction(value)


def iv(value):
    value = rational(value)
    return _Bounds(value.numerator)/value.denominator


def encoded(value):
    value = rational(value)
    return [str(value.numerator), str(value.denominator)]


def bounds(value):
    return [str(value.lo), str(value.hi)]


def inputs(arms, trials, successes, weights, margin):
    rows = old.observations(arms, trials, successes)
    if any(row['trials'] < 1 for row in rows):
        raise ValueError('Each registered arm requires paid initialization')
    if len(weights) != len(rows):
        raise ValueError('One weight per arm required')
    eta = rational(margin)
    if not 0 < eta < Fraction(1, 2):
        raise ValueError('Interior design margin required')
    weight = [rational(x) for x in weights]
    if any(x < 0 for x in weight) or sum(weight) <= 0:
        raise ValueError('Nonnegative weights with positive sum required')
    total = sum(weight)
    weight = [w/total for w in weight]
    means = [max(eta, min(1-eta, Fraction(row['successes'], row['trials']))) for row in rows]
    features = []
    for arm in arms:
        co, si, den = old.likelihood._coordinates(arm)
        features.append((Fraction(1), Fraction(co, den), Fraction(si, den)))
    return eta, means, weight, features


def certificate_from_paid_counts(arms, trials, successes, weights, sign, point,
                                 design_margin='0.000001', absolute_tolerance='1e-12', relative_tolerance='0.001'):
    """Enclose optimality on contracted opposite cone from exact paid counts.

``point`` is an optimizer proposal in the unit physical cone, not true A/B/C.
It must be independently feasible; an invalid proposal is rejected outright.
F is weighted KL of the clipped empirical means to eta+(1-2eta)*X*point.
"""
    if sign not in (-1, 1) or isinstance(sign, bool):
        raise ValueError('Selected answer sign must be+1 or-1')
    eta, means, weight, x = inputs(arms, trials, successes, weights, design_margin)
    if len(point) != 3:
        raise ValueError('Three optimizer coordinates required')
    a, b, c = [rational(value) for value in point]
    if not 0 <= a <= 1 or b*b+c*c > min(a, 1-a)**2 or sign*c > 0:
        raise ValueError('Proposal is outside the exact closed opposite physical cone')
    unit = (a, b, c)
    factor = 1-2*eta
    predictions = [eta+factor*sum(v*u for v, u in zip(feature, unit)) for feature in x]
    value, gradient = _Bounds(0), [Fraction(0), Fraction(0), Fraction(0)]
    for mu, q, w, feature in zip(means, predictions, weight, x):
        if not eta <= q <= 1-eta:
            raise AssertionError('Physical cone yielded an out-of-domain prediction')
        divergence = iv(mu)*iv(mu/q).log()+iv(1-mu)*iv((1-mu)/(1-q)).log()
        value += iv(w)*divergence
        scalar = factor*w*(q-mu)/(q*(1-q))
        gradient = [g+scalar*f for g, f in zip(gradient, feature)]
    tangent_support = old.likelihood.support_bounds([-iv(g) for g in gradient], sign)
    gap = iv(sum(g*u for g, u in zip(gradient, unit)))+tangent_support
    if gap.hi < 0:
        raise ArithmeticError('Certified feasible convexity gap became negative')
    gap = _Bounds(max(Decimal(0), gap.lo), max(Decimal(0), gap.hi))
    optimum = _Bounds(max(Decimal(0), (value-gap).lo), max(Decimal(0), value.hi))
    atol, rtol = rational(absolute_tolerance), rational(relative_tolerance)
    if atol < 0 or rtol < 0:
        raise ValueError('Nonnegative tolerances required')
    conservative_tolerance = iv(atol)+iv(rtol)*_Bounds(optimum.lo)
    accepted = gap.hi <= conservative_tolerance.lo
    return {'status': 'certified_approximate_best_response' if accepted else 'unresolved_best_response_accuracy',
            'selected_sign': int(sign), 'paid_shots': sum(trials), 'point_unit_exact': [encoded(v) for v in unit],
            'theta_design_exact': [encoded(eta+factor*a), encoded(factor*b), encoded(factor*c)],
            'clipped_empirical_means_exact': [encoded(mu) for mu in means],
            'normalized_weights_exact': [encoded(w) for w in weight],
            'predictions_exact': [encoded(q) for q in predictions], 'design_margin_exact': encoded(eta),
            'objective_bounds': bounds(value), 'optimum_bounds': bounds(optimum), 'convexity_gap_bounds': bounds(gap),
            'relative_gap_upper': str((gap/_Bounds(optimum.lo)).hi) if optimum.lo > 0 else None,
            'absolute_tolerance_exact': encoded(atol), 'relative_tolerance_exact': encoded(rtol),
            'conservative_acceptance_threshold_lower': str(conservative_tolerance.lo),
            'rational_gradient_exact': [encoded(g) for g in gradient],
            'proof': 'Convex differentiableF: F(u)-inf_vF(v)<=g(u).u+support_opposite(-g). The support formula is outward enclosed over the entire unit physical cone. Accept only when this upper gap<=atol+rtol*certified lower bound on infF.',
            'scope': 'Optimization certificate on the declared interior-contracted design model. Not a stopping certificate and not an exact-oracle sample-complexity theorem.',
            'truth_parameters_accepted': False}


def exact_psr_best_response_from_paid_counts(trials, successes, weights, sign, design_margin='0.000001'):
    """Exact isotonic Bernoulli KL best response for the two fixed PSR arms."""
    eta, mu, weight, unused = inputs(PSR_ARMS, trials, successes, weights, design_margin)
    if sign not in (-1, 1) or isinstance(sign, bool):
        raise ValueError('Invalid sign')
    # Means map bijectively to the two-arm feasible square using B=0.
    # Pool iff the empirical pair violates the requested opposite ordering.
    if sign*(mu[1]-mu[0]) > 0:
        pooled = sum(w*p for w, p in zip(weight, mu))
        predictions = [pooled, pooled]
    else:
        predictions = mu
    A, C = sum(predictions)/2, (predictions[1]-predictions[0])/2
    factor = 1-2*eta
    point = [(A-eta)/factor, Fraction(0), C/factor]
    cert = certificate_from_paid_counts(PSR_ARMS, trials, successes, weights, sign, point,
                                       design_margin=design_margin, absolute_tolerance='1e-40', relative_tolerance='0')
    if cert['status'] != 'certified_approximate_best_response':
        raise ArithmeticError('Exact pooled PSR solution failed its zero-gap check')
    cert['construction'] = 'Exact weighted isotonic pooling in Bernoulli mean coordinates; actual optimum, up to numerical interval enclosure of the KL value.'
    cert['optimizer_calls'] = 0
    return cert


def rational_feasible_repair(point, sign):
    """Convert a numerical optimization proposal to an exactly feasible point."""
    a, b, c = [rational(v) for v in point]
    a = min(Fraction(1), max(Fraction(0), a))
    c = min(Fraction(0), c) if sign > 0 else max(Fraction(0), c)
    m = min(a, 1-a)
    if m == 0:
        return [a, Fraction(0), Fraction(0)]
    if b*b+c*c > m*m:
        radius = math.hypot(float(b), float(c))
        contraction = rational(float(m)/radius)*Fraction(999999999999, 1000000000000)
        b, c = b*contraction, c*contraction
        while b*b+c*c > m*m:
            b, c = b*Fraction(999, 1000), c*Fraction(999, 1000)
    return [a, b, c]


def numerical_proposal_with_certificate(arms, trials, successes, weights, sign, design_margin='0.000001',
                                        absolute_tolerance='1e-12', relative_tolerance='0.001'):
    """Wrap the existing six/grid-arm proposal; never treat float success as proof."""
    eta, means, unused, unused_x = inputs(arms, trials, successes, weights, design_margin)
    sampler = old.DKMSampler(arms=arms, design_margin=float(eta))
    proposal = sampler._fit(np.array([float(x) for x in means]), np.array([float(rational(x)) for x in weights]), sign=sign)
    repaired = rational_feasible_repair(proposal['point'], sign)
    cert = certificate_from_paid_counts(arms, trials, successes, weights, sign, repaired, design_margin,
                                       absolute_tolerance, relative_tolerance)
    return {'old_float_success': proposal['success'], 'old_float_gap': proposal['oracle_gap_float_diagnostic'],
            'certificate': cert, 'proposal_only': True,
            'required_caller_action': 'Use as an accuracy-qualified best response only if status is certified_approximate_best_response; otherwise refine or report numerical unresolved status.'}


def checks():
    out = []
    for counts, successes, weights, sign in [([100, 100], [10, 20], [1, 1], 1),
                                            ([100, 100], [20, 10], [1, 1], -1),
                                            ([19, 31], [0, 31], [1, 3], 1),
                                            ([19, 31], [19, 0], [3, 1], -1),
                                            ([100000, 100000], [10000, 10001], [1, 1], 1),
                                            ([11, 17], [2, 3], [0, 1], 1)]:
        cert = exact_psr_best_response_from_paid_counts(counts, successes, weights, sign)
        assert Decimal(cert['convexity_gap_bounds'][1]) < Decimal('1e-40')
        out.append({'trials': counts, 'successes': successes, 'weights': weights, 'sign': sign,
                    'objective_bounds': cert['objective_bounds'], 'gap_upper': cert['convexity_gap_bounds'][1]})
    # A feasible but nonoptimal weak-signal proposal passes a1e-5 absolute
    # diagnostic while failing our declared1e-12+0.1% relative criterion.
    weak = certificate_from_paid_counts(PSR_ARMS, [100000, 100000], [10000, 10001], [1, 1], 1,
                                       [(Fraction('0.1000051')-Fraction('0.000001'))/Fraction('0.999998'), 0, 0])
    assert Decimal(weak['convexity_gap_bounds'][1]) < Decimal('1e-5')
    assert weak['status'] == 'unresolved_best_response_accuracy'
    try:
        certificate_from_paid_counts(PSR_ARMS, [10, 10], [1, 2], [1, 1], 1, [.1, .2, 0])
    except ValueError:
        pass
    else:
        raise AssertionError('Infeasible optimizer proposal accepted')
    six = numerical_proposal_with_certificate(old.default_arms(), [100]*6, [4, 7, 6, 10, 8, 13], [1]*6, 1)
    assert six['certificate']['truth_parameters_accepted'] is False
    return {'status': 'passed', 'new_simulation_observations': 0, 'exact_psr_cases': out,
            'weak_signal_old_absolute_gate_counterexample': weak, 'six_arm_observed_count_proposal': six,
            'claims': ['Exact two-PSR-arm best response uses only paid counts and declared allocation weights.',
                       'General-cone approximate best-response accuracy is rigorously enclosed; unresolved accuracy is explicit.',
                       'No full DKM reproduction, runtime speedup, sample-complexity theorem or SOTA advantage is claimed.']}


if __name__ == '__main__':
    result = checks()
    source_paths = [Path(__file__).resolve(), Path(old.__file__).resolve(),
                    ROOT/'experiments/structured_validation/certified_likelihood.py', ROOT/'experiments/circuit_validation/direct_tests.py']
    result['source_sha256'] = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'status': 'passed', 'result_sha256': hashlib.sha256(OUT.read_bytes()).hexdigest(),
                      'exact_psr_cases': len(result['exact_psr_cases']),
                      'weak_absolute_gap': result['weak_signal_old_absolute_gate_counterexample']['convexity_gap_bounds'],
                      'six_arm_status': result['six_arm_observed_count_proposal']['certificate']['status']}))
