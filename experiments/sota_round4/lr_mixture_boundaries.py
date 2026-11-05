"""Classical fixed LR-mixture shapes with exact finite-horizon risk calibration.

The fixed registry is independent of cases and observations. Floating scores
design symmetric integer tails; the frozen round3 integer DP certifies risk.
This is prior tuning within classical sequential testing, not a new framework.
All procedures share the horizon, delta, and all-discordance calibration rule.
"""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import time

import numpy as np
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/sota_round3'))
import generic_sign_boundary as generic

OUT = ROOT/'results/sota_round4/lr_mixture_boundaries'
REGISTRY_VERSION = 'fixed_classical_lr_mixture_priors_v1'
_VERIFIED = {}


def two_point(q, weight):
    q, weight = Fraction(str(q)), Fraction(weight)
    return {'kind': 'symmetric_two_point', 'q': [q.numerator, q.denominator],
            'weight': [weight.numerator, weight.denominator]}


def fixed_specs():
    result = {f'sprt_{q}': {'components': [two_point(q, 1)]}
              for q in ('0.6', '0.7', '0.75', '0.85', '0.9', '0.99')}
    for weight in ('0.25', '0.5', '0.75'):
        result[f'mix_0.75_0.99_w{weight}'] = {
            'components': [two_point('0.75', Fraction(weight)), two_point('0.99', 1-Fraction(weight))]}
    result['mix_0.7_0.9_0.99_equal'] = {'components': [two_point(q, Fraction(1, 3)) for q in ('0.7', '0.9', '0.99')]}
    result['mix_beta_0.75_half'] = {'components': [
        {'kind': 'jeffreys_beta', 'alpha': [1, 2], 'weight': [1, 2]}, two_point('0.75', Fraction(1, 2))]}
    for name, spec in result.items():
        spec.update({'registry_version': REGISTRY_VERSION, 'name': name,
                     'orientation_rule': 'Each two-point component puts half its weight at q and half at1-q; beta is symmetric.'})
        validate_spec(spec)
    return result


def validate_spec(spec):
    weights = []
    for component in spec['components']:
        weight = Fraction(*component['weight'])
        if weight <= 0:
            raise ValueError('Positive rational component weights required')
        weights.append(weight)
        if component['kind'] == 'symmetric_two_point':
            q = Fraction(*component['q'])
            if not Fraction(1, 2) < q < 1:
                raise ValueError('q must be strictly between half and one')
        elif component['kind'] == 'jeffreys_beta':
            if component['alpha'] != [1, 2]:
                raise ValueError('Only the fixed Jeffreys beta component is allowed')
        else:
            raise ValueError('Unknown component type')
    if not weights or sum(weights, Fraction(0)) != 1:
        raise ValueError('Component weights must sum exactly to one')


def source_hashes():
    paths = [Path(__file__).resolve(), ROOT/'experiments/sota_round3/generic_sign_boundary.py',
             ROOT/'experiments/structured_validation/expert_config.py',
             ROOT/'results/sota_round2/theory/generic_symmetric_tail_dp_lemma.json']
    return {path.relative_to(ROOT).as_posix(): generic.digest(path) for path in paths}


def score_rows(maximum, shape):
    if type(maximum) is not int or maximum < 1:
        raise ValueError('Positive integer horizon required')
    specs = fixed_specs()
    if shape not in specs:
        raise ValueError('Shape is outside the fixed registry')
    components = specs[shape]['components']
    rows = []
    for n in range(1, maximum+1):
        k = np.arange(n//2+1, n+1, dtype=float)
        c = n-k
        combined = None
        for component in components:
            weight = float(Fraction(*component['weight']))
            if component['kind'] == 'symmetric_two_point':
                q = float(Fraction(*component['q']))
                lq, lc = math.log(2*q), math.log(2*(1-q))
                score = np.logaddexp(k*lq+c*lc, c*lq+k*lc)-math.log(2)
            else:
                score = gammaln(k+.5)+gammaln(c+.5)-gammaln(n+1)-2*gammaln(.5)+n*math.log(2)
            term = score+math.log(weight)
            combined = term if combined is None else np.logaddexp(combined, term)
        if np.any(np.diff(combined) < -1e-10) or not np.all(np.isfinite(combined)):
            raise ArithmeticError('Upper-tail mixture score is not finite and increasing')
        rows.append(combined)
    return rows


def calibrate(maximum, shape, delta='0.01', log_threshold_scale=1000000):
    """Return a generic-compatible integer artifact using no outcomes/cases."""
    if type(maximum) is not int or maximum < 1 or type(log_threshold_scale) is not int or log_threshold_scale < 1:
        raise ValueError('Positive integer horizon and threshold scale required')
    risk = Fraction(str(delta))
    if not 0 < risk < 1:
        raise ValueError('Risk must lie strictly between zero and one')
    spec = fixed_specs()[shape]
    rows = score_rows(maximum, shape)
    scale, low = log_threshold_scale, 0
    high = int(math.ceil(math.log(1/float(risk))*scale))
    while generic.float_risk(generic.level_table(rows, high/scale)) > float(risk):
        high *= 2
    float_proposals = 1
    while high-low > 1:
        mid = (low+high)//2
        float_proposals += 1
        if generic.float_risk(generic.level_table(rows, mid/scale)) <= float(risk):
            high = mid
        else:
            low = mid
    table = generic.level_table(rows, high/scale)
    numerator, denominator = generic.exact_risk(table)
    adjustments = 0
    while numerator*risk.denominator > denominator*risk.numerator:
        high += 1
        adjustments += 1
        table = generic.level_table(rows, high/scale)
        numerator, denominator = generic.exact_risk(table)
    previous = generic.level_table(rows, (high-1)/scale)
    pn, pd = generic.exact_risk(previous)
    artifact = {'version': generic.VERSION, 'shape': shape,
                'status': 'integer_boundary_exact_risk_certified',
                'max_discordances': maximum, 'max_physical_pairs': maximum, 'physical_shot_cap': 2*maximum,
                'boundaries': table,
                'delta_exact': {'numerator': str(risk.numerator), 'denominator': str(risk.denominator)},
                'design_log_threshold': high/scale,
                'search': {'log_threshold_scale': scale, 'selected_tick': high,
                           'integer_acceptance_adjustments': adjustments, 'float_search_proposals': float_proposals,
                           'previous_tick_risk_exceeds_delta': pn*risk.denominator > pd*risk.numerator,
                           'previous_tick_exact_crossing_numerator': str(pn)},
                'fair_null_risk': {'crossing_numerator': str(numerator), 'fair_denominator': str(denominator),
                                   'probability_float_diagnostic': numerator/denominator},
                'boundary_encoding': 'B[n]; reject at plus>=B[n], accept at plus<=n-B[n]; zero score never stops',
                'source_hashes': generic.source_hashes(), 'mixture_source_hashes': source_hashes(),
                'fixed_prior_spec': deepcopy(spec), 'registry_sha256': generic.payload_hash(fixed_specs()),
                'monitoring_calibration': 'All discordance times throughH; fair comparison may inspect every paid physical pair through cap2H.',
                'new_simulation_observations': 0, 'case_truth_access': False,
                'claim_boundary': 'Classical fixed LR-mixture prior tuning; no novel statistical framework, no data-dependent prior, and no performance claim. Floating shape design is not exact real contour certification; frozen integer boundary and its null crossing count are exact.'}
    artifact['artifact_sha256'] = generic.payload_hash(artifact)
    return artifact


def verify_artifact(artifact):
    if artifact['mixture_source_hashes'] != source_hashes():
        raise ValueError('Mixture source binding changed')
    if artifact['shape'] not in fixed_specs() or artifact['fixed_prior_spec'] != fixed_specs()[artifact['shape']]:
        raise ValueError('Fixed prior specification changed')
    if artifact['registry_sha256'] != generic.payload_hash(fixed_specs()):
        raise ValueError('Fixed registry changed')
    if artifact['max_physical_pairs'] != artifact['max_discordances'] or artifact['physical_shot_cap'] != 2*artifact['max_discordances']:
        raise ValueError('Physical horizon mismatch')
    boundary = generic.verify_artifact(artifact)
    _VERIFIED[boundary.artifact_sha256] = boundary
    return boundary


def certificate_from_pairs(counts, boundary):
    if _VERIFIED.get(boundary.artifact_sha256) != boundary:
        raise ValueError('Call this module verify_artifact first')
    if 2*sum(counts) > 2*(len(boundary.table)-1):
        raise ValueError('Certified physical budget exceeded')
    result = generic.certificate_from_pairs(counts, boundary)
    result['method_family'] = REGISTRY_VERSION
    result['max_physical_pairs'] = len(boundary.table)-1
    result['physical_shot_cap'] = 2*(len(boundary.table)-1)
    return result


def exact_small_capital(n, k, spec):
    total = Fraction(0)
    for component in spec['components']:
        weight = Fraction(*component['weight'])
        if component['kind'] == 'symmetric_two_point':
            q = Fraction(*component['q'])
            value = ((2*q)**k*(2*(1-q))**(n-k)+(2*q)**(n-k)*(2*(1-q))**k)/2
        else:
            value = Fraction(math.comb(2*k, k)*math.comb(2*(n-k), n-k), (1 << n)*math.comb(n, k))
        total += weight*value
    return total


def checks():
    results = []
    for shape, spec in fixed_specs().items():
        rows = score_rows(12, shape)
        for n in range(0, 9):
            exact = [exact_small_capital(n, k, spec) for k in range(n+1)]
            if exact != list(reversed(exact)) or sum((math.comb(n, k)*x for k, x in enumerate(exact)), Fraction(0)) != (1 << n):
                raise AssertionError('Symmetry or mean-one exact mixture identity failed')
            for k in range(n//2+1, n+1):
                if abs(rows[n-1][k-n//2-1]-math.log(float(exact[k]))) > 2e-13:
                    raise AssertionError('Float mixture score differs from exact small count formula')
        if shape.startswith('sprt_'):
            legacy = generic.score_rows(12, shape)
            if not all(np.array_equal(a, b) for a, b in zip(rows, legacy)):
                raise AssertionError('Single-component shape differs from frozen generic implementation')
        artifact = calibrate(12, shape)
        boundary = verify_artifact(artifact)
        hits = 0
        for signs in range(1 << 12):
            k = 0
            for n in range(1, 13):
                k += (signs >> (n-1)) & 1
                b = boundary.table[n]
                if k >= b or k <= n-b:
                    hits += 1
                    break
        if hits != int(artifact['fair_null_risk']['crossing_numerator']):
            raise AssertionError('Exhaustive fair paths differ from integer DP')
        positive = certificate_from_pairs([0, 12, 0, 0], boundary)
        negative = certificate_from_pairs([0, 0, 12, 0], boundary)
        if (positive['decision'], negative['decision']) != ('reject', 'accept'):
            raise AssertionError('Mirrored runtime directions failed')
        try:
            certificate_from_pairs([13, 0, 0, 0], boundary)
        except ValueError:
            pass
        else:
            raise AssertionError('Physical horizon overrun accepted')
        results.append({'shape': shape, 'horizon': 12, 'exhaustive_paths': 4096,
                        'exact_crossings': hits, 'risk': artifact['fair_null_risk']['probability_float_diagnostic']})
    return {'status': 'passed', 'source_hashes': source_hashes(), 'fixed_registry': fixed_specs(),
            'registry_sha256': generic.payload_hash(fixed_specs()), 'cases_and_observations_read': False,
            'checks': ['exact rational prior weights and mirrored orientations', 'exact small-count fair mean-one and symmetry',
                       'float scores versus exact rational LR mixtures', 'single-q equality to frozen round3 shape',
                       'every one of4096 fair sign sequences atH12 for each shape', 'runtime mirror and physical horizon guard'],
            'results': results, 'new_simulation_observations': 0}


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('checks', 'calibrate'))
    parser.add_argument('--maximum', type=int, default=8192)
    parser.add_argument('--shape', choices=tuple(fixed_specs()))
    args = parser.parse_args()
    if args.command == 'checks':
        result = checks()
        write_new(OUT/'interface_checks.json', result)
        print(json.dumps({'status': result['status'], 'shapes': len(result['results']),
                          'source_sha256': generic.digest(__file__)}), flush=True)
        return
    check = json.loads((OUT/'interface_checks.json').read_text())
    if check['status'] != 'passed' or check['source_hashes'] != source_hashes():
        raise ValueError('Deterministic checks must pass on frozen sources before calibration')
    shapes = [args.shape] if args.shape else list(fixed_specs())
    frozen, records = source_hashes(), []
    for shape in shapes:
        start = time.perf_counter()
        result = calibrate(args.maximum, shape)
        verify_artifact(result)
        if source_hashes() != frozen:
            raise ValueError('Source changed during calibration')
        output = OUT/(shape+'.json')
        write_new(output, result)
        item = {'shape': shape, 'path': output.relative_to(ROOT).as_posix(), 'sha256': generic.digest(output),
                'artifact_sha256': result['artifact_sha256'], 'risk': result['fair_null_risk']['probability_float_diagnostic'],
                'log_threshold': result['design_log_threshold'], 'elapsed_seconds': time.perf_counter()-start}
        records.append(item)
        print(json.dumps(item), flush=True)
    if args.shape is None:
        write_new(OUT/'calibration_manifest.json', {'status': 'complete', 'source_hashes': frozen,
                  'max_discordances': args.maximum, 'physical_shot_cap': 2*args.maximum, 'delta': .01,
                  'fixed_registry_sha256': generic.payload_hash(fixed_specs()), 'records': records,
                  'new_simulation_observations': 0, 'claim_boundary': 'Fixed classical candidate and control priors; no performance inference.'})


if __name__ == '__main__':
    main()
