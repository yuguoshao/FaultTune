"""Independent deterministic checks; no simulator or Monte Carlo calls."""
from copy import deepcopy
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import product
from pathlib import Path
import argparse
import json
import math
import sys
import time

import exact_sign_boundary as exact

sys.path.insert(0, str(exact.ROOT/'experiments/circuit_validation'))
import direct_tests as old


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def factorial_capital(s, n):
    return Fraction(math.factorial(2*s)*math.factorial(2*(n-s)),
                    2**n*math.factorial(s)*math.factorial(n-s)*math.factorial(n))


def full_vector_survivors(boundary):
    values = [1]
    for n, cutoff in enumerate(boundary[1:], 1):
        fresh = [0]*(n+1)
        for s, count in enumerate(values):
            fresh[s] += count
            fresh[s+1] += count
        values = [count if n-cutoff < s < cutoff else 0 for s, count in enumerate(fresh)]
    return values


def enumerated_survivors(boundary):
    maximum = len(boundary)-1
    values = [0]*(maximum+1)
    for word in product((0, 1), repeat=maximum):
        total, crossed = 0, False
        for n, bit in enumerate(word, 1):
            total += bit
            cutoff = boundary[n]
            crossed |= total >= cutoff or total <= n-cutoff
        if not crossed:
            values[total] += 1
    return values


def boundary_math_checks():
    comparisons = 0
    for maximum in (1, 2, 7, 16, 40):
        for threshold in (Fraction(1, 3), Fraction(1), Fraction(3, 2), Fraction(3), Fraction(10), Fraction(100)):
            table = exact.constant_boundary(maximum, threshold)
            for n in range(maximum+1):
                for s in range(n+1):
                    reference = factorial_capital(s, n)
                    require(reference == exact.jeffreys_capital(s, n), 'Capital identities differ')
                    crossing = 2*s != n and reference >= threshold
                    actual = s >= table[n] or s <= n-table[n]
                    require(crossing == actual, 'Recursive boundary differs from exact factorial comparison')
                    comparisons += 1
    # Deliberately cross at exact equality: n=2, s=2 has M=3/2.
    require(exact.constant_boundary(2, Fraction(3, 2))[2] == 2 and
            exact.constant_boundary(2, Fraction(1500001, 1000000))[2] == 3,
            'Boundary equality convention changed')
    return {'status': 'passed', 'exact_state_comparisons': comparisons,
            'threshold_equality_and_zero_score_checked': True}


def integer_dp_checks():
    comparisons = []
    for maximum in (1, 2, 6, 10, 14):
        for threshold in (Fraction(1), Fraction(3, 2), Fraction(3), Fraction(10), Fraction(100)):
            table = exact.constant_boundary(maximum, threshold)
            sparse = exact.count_surviving_paths(table)
            dense, brute = full_vector_survivors(table), enumerated_survivors(table)
            recovered = [0]*(maximum+1)
            for offset, value in enumerate(sparse['survivor_counts']):
                index = sparse['survivor_lower_plus']+offset
                if 0 <= index <= maximum:
                    recovered[index] = value
                else:
                    require(value == 0, 'Nonzero survivor outside count domain')
            require(recovered == dense == brute, 'Independent DP/enumeration survivor vectors differ')
            require(sparse['crossing_numerator'] == 2**maximum-sum(brute), 'Crossing count differs')
            comparisons.append({'maximum': maximum, 'threshold': str(threshold),
                                'crossing_numerator': sparse['crossing_numerator'], 'denominator': 2**maximum})
    # General nonconstant tables exercise the generic integer-DP interface.
    table = [1]+[n+1 if n % 3 else n for n in range(1, 13)]
    sparse = exact.count_surviving_paths(table)
    require(sparse['crossing_numerator'] == 2**12-sum(enumerated_survivors(table)),
            'Nonconstant boundary DP differs from enumeration')
    return {'status': 'passed', 'three_way_comparisons': comparisons, 'nonconstant_table_checked': True}


def ordering_checks():
    table = exact.constant_boundary(10, 3)
    wrong_positive, fair_negative, fair_either = Fraction(0), Fraction(0), Fraction(0)
    for word in product((0, 1), repeat=10):
        plus, negative_cross, either_cross = 0, False, False
        probability = Fraction(1)
        for n, bit in enumerate(word, 1):
            # Deliberately past-dependent ordered alternatives, no sampling.
            p = Fraction(3, 4) if plus % 2 else Fraction(1, 2)
            probability *= p if bit else 1-p
            plus += bit
            negative_cross |= plus <= n-table[n]
            either_cross |= negative_cross or plus >= table[n]
        if negative_cross:
            wrong_positive += probability
            fair_negative += Fraction(1, 2**10)
        if either_cross:
            fair_either += Fraction(1, 2**10)
    require(wrong_positive <= fair_negative <= fair_either,
            'Predictably ordered exact path tree violated wrong-direction domination')
    return {'status': 'passed', 'ordered_wrong_union': str(wrong_positive),
            'fair_negative_union': str(fair_negative), 'fair_symmetric_union': str(fair_either),
            'scope': 'A finite deterministic example supplements the general coupling proof'}


def old_beta_checks():
    enclosed = 0
    with localcontext() as context:
        context.prec = 100
        for n in (0, 1, 2, 9, 10, 16, 32, 100, 2048):
            for s in sorted({0, n//4, n//2, 3*n//4, n}):
                value = exact.jeffreys_capital(s, n)
                log_exact = (Decimal(value.numerator)/Decimal(value.denominator)).ln()
                bounds = old._log_marginal(s, n, .5)+n*old._Bounds(2).log()
                require(bounds.lo <= log_exact <= bounds.hi, 'Old outward log bounds miss exact capital')
                enclosed += 1
    fixed = exact.certify_threshold(128, '.01', '100')
    runtime = exact.verify_artifact(fixed)
    decisions = 0
    for n in range(129):
        for s in sorted({0, n//4, n//2, 3*n//4, n}):
            counts = [3, s, n-s, 2]
            certificate = exact.certificate_from_pairs(counts, runtime)
            previous = old.certificate_from_pairs(tuple(counts), delta=.01)
            require(certificate['decision'] == previous['decision'] and certificate['shots'] == previous['shots'],
                    'Threshold-100 integer and old beta decisions differ')
            decisions += 1
    return {'status': 'passed', 'high_precision_log_enclosures': enclosed,
            'old_threshold_100_paid_pair_decisions': decisions}


def runtime_checks():
    artifact = exact.calibrate(32, '.01', 10000)
    runtime = exact.verify_artifact(artifact)
    history = [{'m': 1, 'pair_counts': [0, 1, 0, 0], 'budget': 2*n} for n in range(1, 17)]
    replay = exact.replay_pair_history(history, artifact)
    h = exact.record_fraction(artifact['threshold_exact'])
    expected = next(n for n in range(1, 17) if factorial_capital(n, n) >= h)
    require(replay['status'] == 'stopped_on_observed_checkpoint' and replay['shots'] == 2*expected,
            'Same-checkpoint replay differs from factorial reference')
    # Multiple phases at a single physical endpoint must not add extra looks.
    grouped = [{'m': 1, 'pair_counts': [0, 1, 0, 0], 'budget': 32} for unused in range(16)]
    late = exact.replay_pair_history(grouped, artifact)
    require(late['shots'] == 32 and len(late['checkpoint_history']) == 1, 'Replay invented intermediate checkpoints')
    censored = exact.replay_pair_history(history[:1], artifact)
    require(censored['status'] == 'observed_prefix_exhausted_without_crossing' and censored['shots'] == 2,
            'Exhausted old history was treated as a completed stopping cost')
    errors = []
    def must_fail(name, function):
        try:
            function()
        except (ValueError, KeyError, TypeError):
            errors.append(name)
        else:
            raise AssertionError('Invalid input accepted: '+name)
    for name, counts in [('negative', [0, -1, 0, 0]), ('boolean', [False, 0, 0, 0]),
                         ('fractional', [0, .5, 0, 0]), ('wrong_length', [0, 0, 0]),
                         ('beyond_discordance_cap', [0, 33, 0, 0])]:
        must_fail(name, lambda counts=counts: exact.certificate_from_pairs(counts, artifact))
    require(exact.certificate_from_pairs([100, 1, 1, 200], artifact)['decision'] == 'abstain',
            'Zero discordant score must abstain despite larger concordant count')
    altered = deepcopy(artifact)
    altered['boundaries'][9] += 1
    must_fail('changed_runtime_table', lambda: exact.certificate_from_pairs([0, 9, 0, 0], altered))
    altered['boundary_sha256'] = exact.boundary_sha256(altered)
    altered['artifact_sha256'] = exact.payload_sha256(altered)
    must_fail('hashed_but_false_table', lambda: exact.verify_artifact(altered))
    counterfeit = exact.RuntimeBoundary(runtime.max_discordances, runtime.delta, runtime.threshold,
                                        runtime.boundaries, runtime.artifact_sha256,
                                        runtime.boundary_sha256, runtime.crossing_probability)
    must_fail('unverified_runtime_constructor', lambda: exact.certificate_from_pairs([0, 9, 0, 0], counterfeit))
    must_fail('zero_cap', lambda: exact.calibrate(0))
    must_fail('invalid_delta', lambda: exact.calibrate(10, '1'))
    must_fail('boolean_scale', lambda: exact.calibrate(10, '.01', True))
    must_fail('incomplete_physical_checkpoint', lambda: exact.replay_pair_history(grouped[:-1], artifact))
    return {'status': 'passed', 'fixture_calibrated_threshold': str(h), 'deterministic_fixture_stop_shots': 2*expected,
            'sparse_and_multiphase_monitoring': 'checked', 'old_prefix_censoring': 'preserved',
            'rejected_inputs': errors}


def production_boundary_checks(artifact):
    maximum = artifact['max_discordances']
    h = exact.record_fraction(artifact['threshold_exact'])
    table = artifact['boundaries']
    # Independent comb-product comparisons avoid the producer's Fraction walk.
    checked = 0
    def crosses(s, n):
        return h.denominator*math.comb(2*s, s)*math.comb(2*(n-s), n-s) >= h.numerator*2**n*math.comb(n, s)
    for n in range(1, maximum+1):
        cutoff = table[n]
        if cutoff == n+1:
            require(not crosses(n, n), 'No-crossing sentinel misses the extreme state')
        else:
            require(cutoff > n//2 and crosses(cutoff, n), 'Boundary point does not cross exactly')
            if cutoff > n//2+1:
                require(not crosses(cutoff-1, n), 'Adjacent positive state crosses below claimed boundary')
        checked += 1
    dense = full_vector_survivors(table)
    numerator = 2**maximum-sum(dense)
    require(numerator == exact.text_int(artifact['fair_null_risk']['crossing_numerator']), 'Production dense DP differs')
    delta = exact.record_fraction(artifact['delta_exact'])
    require(delta.denominator*numerator <= delta.numerator*2**maximum, 'Production risk exceeds delta')
    previous = artifact['selection']['lower_neighbor']
    lower_h = exact.record_fraction(previous['threshold_exact'])
    lower_table = exact.constant_boundary(maximum, lower_h)
    lower_risk_count = 2**maximum-sum(full_vector_survivors(lower_table))
    require(delta.denominator*lower_risk_count > delta.numerator*2**maximum, 'Production lower-neighbor witness is safe')
    return {'status': 'passed', 'independent_integer_boundary_rows': checked,
            'full_vector_integer_dp_reproduced': True, 'lower_lattice_neighbor_exceeds_delta': True,
            'threshold_exact': artifact['threshold_exact'], 'fair_null_risk': artifact['fair_null_risk']['probability_fraction']}


def main(output, boundary_path):
    started = time.perf_counter()
    result = {'status': 'running', 'scope': 'Exact deterministic arithmetic, full small-tree enumeration and fixed-count replay; zero simulator observations', 'checks': {}}
    try:
        for name, check in [('boundary_math', boundary_math_checks), ('integer_dp', integer_dp_checks),
                            ('adaptive_ordering_example', ordering_checks), ('old_beta_compatibility', old_beta_checks),
                            ('runtime_and_replay', runtime_checks)]:
            result['checks'][name] = check()
        artifact = exact.calibrate(2048, '.01', 1000000)
        result['checks']['production_N2048'] = production_boundary_checks(artifact)
        artifact_hash = exact.write_artifact(boundary_path, artifact)
        result['boundary_artifact'] = {'path': boundary_path.relative_to(exact.ROOT).as_posix(),
                                       'file_sha256': artifact_hash, 'artifact_sha256': artifact['artifact_sha256'],
                                       'boundary_sha256': artifact['boundary_sha256']}
        result['status'] = 'passed'
    except BaseException as caught:
        result.update(status='failed', error=type(caught).__name__+': '+str(caught))
        raise
    finally:
        result['sources'] = exact.source_hashes()
        for path in (Path(__file__).resolve(), Path(old.__file__).resolve()):
            result['sources'][path.relative_to(exact.ROOT).as_posix()] = exact.file_sha256(path)
        result['elapsed_seconds'] = time.perf_counter()-started
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
        print(json.dumps({'status': result['status'], 'output': str(output), 'elapsed_seconds': result['elapsed_seconds']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=exact.ROOT/'results/sota_round3/exact_sign_boundary_checks.json')
    parser.add_argument('--boundary-output', type=Path, default=exact.ROOT/'results/sota_round3/beta_boundary_N2048.json')
    args = parser.parse_args()
    main(args.output, args.boundary_output)
