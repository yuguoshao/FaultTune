"""Independent full-vector verifier for a frozen integer sign-tail artifact.

This module imports no boundary producer. It recomputes integer crossing
counts and, when requested, verifies a Jeffreys contour by independent
binomial-coefficient inequalities. No simulation is performed.
"""
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def natural(value, name):
    require(type(value) is int and value >= 0, name+' must be a nonnegative integer')
    return value


def exact_text(value):
    require(type(value) is str and value and value.isascii() and value.isdigit(),
            'Exact count must be a decimal digit string')
    answer = 0
    for offset in range(0, len(value), 1000):
        block = value[offset:offset+1000]
        answer = answer*10**len(block)+int(block)
    return answer


def fraction(record):
    numerator, denominator = exact_text(record['numerator']), exact_text(record['denominator'])
    require(denominator > 0, 'Nonpositive exact denominator')
    return Fraction(numerator, denominator)


def validate_table(boundaries):
    require(type(boundaries) is list and len(boundaries) >= 2 and
            type(boundaries[0]) is int and boundaries[0] == 1, 'Invalid zero-index sentinel')
    for d, boundary in enumerate(boundaries[1:], 1):
        require(type(boundary) is int and d//2+1 <= boundary <= d+1,
                'Noninteger, asymmetric, zero-score or out-of-range tail cutoff at '+str(d))


def full_vector_dp(boundaries):
    validate_table(boundaries)
    vector, first_counts = [1], []
    previous_total = 1
    for d, upper in enumerate(boundaries[1:], 1):
        updated = [0]*(d+1)
        for b, count in enumerate(vector):
            updated[b] += count
            updated[b+1] += count
        for b in range(d+1):
            if b <= d-upper or b >= upper:
                updated[b] = 0
        total = sum(updated)
        first_counts.append(2*previous_total-total)
        require(first_counts[-1] >= 0, 'Negative crossing count')
        require(updated == list(reversed(updated)), 'Symmetric table produced asymmetric survivors')
        vector, previous_total = updated, total
    horizon = len(boundaries)-1
    denominator = 1 << horizon
    crossed = denominator-sum(vector)
    require(crossed == sum(count << (horizon-d) for d, count in enumerate(first_counts, 1)),
            'Independent first-crossing decomposition failed')
    return vector, first_counts, crossed, denominator


def jeffreys_coefficients(horizon):
    values = [1]
    for k in range(1, horizon+1):
        product = values[-1]*(4*k-2)
        require(product % k == 0, 'Central-binomial recurrence not integral')
        values.append(product//k)
    return values


def jeffreys_ge(d, b, threshold, coefficients):
    return (threshold.denominator*coefficients[b]*coefficients[d-b] >=
            threshold.numerator*(1 << d)*math.comb(d, b))


def inspect_jeffreys_table(boundaries, threshold, coefficients):
    require(threshold > 0, 'Jeffreys threshold must be positive')
    for d, boundary in enumerate(boundaries[1:], 1):
        if boundary == d+1:
            require(not jeffreys_ge(d, d, threshold, coefficients),
                    'Missing Jeffreys extreme crossing at '+str(d))
        else:
            require(jeffreys_ge(d, boundary, threshold, coefficients),
                    'Boundary is below Jeffreys threshold at '+str(d))
            if boundary > d//2+1:
                require(not jeffreys_ge(d, boundary-1, threshold, coefficients),
                        'Earlier Jeffreys tail crossing was omitted at '+str(d))


def independent_jeffreys_table(horizon, threshold, coefficients):
    result = [1]
    for d in range(1, horizon+1):
        low, high = d//2+1, d+1
        while low < high:
            middle = (low+high)//2
            if jeffreys_ge(d, middle, threshold, coefficients):
                high = middle
            else:
                low = middle+1
        result.append(low)
    return result


def inspect_risk_record(record, vector, first_counts, crossed, denominator):
    require(exact_text(record['crossing_numerator']) == crossed, 'Crossing numerator differs')
    require(exact_text(record['fair_denominator']) == denominator, 'Fair denominator differs')
    if 'surviving_total' in record:
        require(exact_text(record['surviving_total']) == sum(vector), 'Survivor total differs')
    if 'probability_fraction' in record:
        require(fraction(record['probability_fraction']) == Fraction(crossed, denominator),
                'Reported exact crossing probability differs')
    if 'first_crossing_counts' in record:
        require([exact_text(value) for value in record['first_crossing_counts']] == first_counts,
                'Per-index first-crossing counts differ')
    if 'survivor_counts' in record:
        lower = natural(record['survivor_lower_plus'], 'survivor_lower_plus')
        packed = [exact_text(value) for value in record['survivor_counts']]
        require(lower+len(packed) <= len(vector), 'Packed survivor range is invalid')
        require(not any(vector[:lower]) and not any(vector[lower+len(packed):]) and
                vector[lower:lower+len(packed)] == packed, 'Packed survivor state differs')


def audit_artifact(artifact, require_jeffreys=False):
    horizon = natural(artifact['max_discordances'], 'max_discordances')
    require(horizon >= 1, 'Positive horizon required')
    boundaries = artifact['boundaries']
    validate_table(boundaries)
    require(len(boundaries) == horizon+1, 'Horizon/table length mismatch')
    delta = fraction(artifact['delta_exact'])
    require(0 < delta < 1, 'Invalid risk budget')
    for name, expected in artifact.get('source_hashes', {}).items():
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts, 'Source must be repository relative')
        require(sha(ROOT/path) == expected, 'Bound source changed: '+name)
    if 'artifact_sha256' in artifact:
        payload = {name: value for name, value in artifact.items() if name != 'artifact_sha256'}
        actual = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                           allow_nan=False).encode()).hexdigest()
        require(actual == artifact['artifact_sha256'], 'Artifact content hash differs')
    vector, first_counts, crossed, denominator = full_vector_dp(boundaries)
    inspect_risk_record(artifact['fair_null_risk'], vector, first_counts, crossed, denominator)
    require(delta.denominator*crossed <= delta.numerator*denominator, 'Exact risk budget exceeded')
    result = {'status': 'passed', 'horizon': horizon, 'integer_dp_recomputed_without_producer_import': True,
              'all_survivor_states_and_first_crossing_counts_checked': True,
              'exact_risk_within_delta': True, 'float_probability_diagnostic': crossed/denominator,
              'jeffreys_contour_verified_independently': False,
              'lattice_lower_neighbor_verified_independently': False,
              'new_simulation_observations': 0}
    if require_jeffreys:
        threshold = fraction(artifact['threshold_exact'])
        coefficients = jeffreys_coefficients(horizon)
        inspect_jeffreys_table(boundaries, threshold, coefficients)
        result['jeffreys_contour_verified_independently'] = True
        selection = artifact['selection']
        if selection['kind'] == 'least_safe_threshold_on_fixed_positive_rational_lattice':
            scale = natural(selection['threshold_scale'], 'threshold_scale')
            index = natural(selection['chosen_lattice_index'], 'chosen_lattice_index')
            require(scale > 0 and index > 0 and threshold == Fraction(index, scale),
                    'Threshold/search lattice mismatch')
            neighbor = selection['lower_neighbor']
            require(fraction(neighbor['threshold_exact']) == Fraction(index-1, scale),
                    'Predecessor is not the next lower lattice point')
            if index > 1:
                lower_table = independent_jeffreys_table(horizon, Fraction(index-1, scale), coefficients)
                lower_vector, lower_first, lower_crossed, lower_denominator = full_vector_dp(lower_table)
                inspect_risk_record(neighbor['fair_null_risk'], lower_vector, lower_first,
                                    lower_crossed, lower_denominator)
                require(delta.denominator*lower_crossed > delta.numerator*lower_denominator,
                        'Lower lattice neighbor does not exceed risk')
            else:
                require(neighbor['excluded_nonpositive'] is True and
                        fraction(neighbor['crossing_probability_if_allowed']) == 1,
                        'Zero predecessor witness differs')
            result['lattice_lower_neighbor_verified_independently'] = True
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact', type=Path)
    parser.add_argument('--jeffreys', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    artifact = json.loads(arguments.artifact.read_text(encoding='utf-8'))
    report = audit_artifact(artifact, require_jeffreys=arguments.jeffreys)
    report['artifact_path'] = arguments.artifact.resolve().relative_to(ROOT).as_posix()
    report['artifact_file_sha256'] = sha(arguments.artifact)
    report['verifier_source_sha256'] = sha(__file__)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(report))
