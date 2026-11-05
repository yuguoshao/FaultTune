"""Deterministic independent DP/contour/prefix checks; no simulation."""
from copy import deepcopy
from fractions import Fraction
from itertools import product
from pathlib import Path
import argparse
import json
import math
import time

import exact_sign_boundary as beta
import fast_gate_boundary as fast
import piecewise_gate_boundary as piece


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def independent_counts(table, m):
    all_counts, same_prefix = [1], [1]
    for d in range(1, len(table)):
        full, restricted = [0]*(d+1), [0]*(d+1)
        for b in range(d):
            full[b] += all_counts[b]
            full[b+1] += all_counts[b]
            restricted[b] += same_prefix[b]
            restricted[b+1] += same_prefix[b]
        cutoff = table[d]
        all_counts = [count if d-cutoff < b < cutoff else 0 for b, count in enumerate(full)]
        same_prefix = [count if d-cutoff < b < cutoff and (d > m or b in (0, d)) else 0
                       for b, count in enumerate(restricted)]
    return 2**(len(table)-1)-sum(all_counts), sum(same_prefix)


def independent_bounds(table, m):
    h = len(table)-1
    crossing, restricted = independent_counts(table, m)
    return Fraction(crossing*2**m+restricted, 2**(h+m)), Fraction(crossing, 2**h)+Fraction(1, 4**m)


def small_checks():
    checked = 0
    for h in (4, 7, 10):
        for switch in (1, h//2, h-1):
            early = beta.constant_boundary(h, Fraction(3))
            table = piece.concatenate_table(early, switch, Fraction(10))
            require(table[:switch+1] == early[:switch+1], 'Early rows were changed')
            crossed, restricted = 0, 0
            m = 2
            for signs in product((0, 1), repeat=h):
                b, crossing = 0, False
                for d, bit in enumerate(signs, 1):
                    b += bit
                    crossing |= b >= table[d] or b <= d-table[d]
                crossed += crossing
                restricted += not crossing and signs[0] == signs[1]
            p, t = independent_counts(table, m)
            saved = fast.exact_risk_bounds(table, m)
            require(crossed == p == saved['beta_crossing_numerator'] and
                    restricted == t == saved['restricted_survivor_numerator'], 'Small piecewise three-way count check failed')
            checked += 1
    return {'status': 'passed', 'enumeration_dense_compact_cases': checked}


def production_checks(artifact):
    early = piece.early_artifact()['boundaries']
    table = artifact['boundaries']
    require(table[:piece.D0+1] == early[:piece.D0+1], 'Frozen first-512 table differs')
    late = beta.record_fraction(artifact['late_threshold_exact'])
    def cross(b, d):
        return late.denominator*math.comb(2*b, b)*math.comb(2*(d-b), d-b) >= late.numerator*2**d*math.comb(d, b)
    for d in range(piece.D0+1, piece.H+1):
        cutoff = table[d]
        if cutoff == d+1:
            require(not cross(d, d), 'Incorrect absent late boundary')
        else:
            require(cutoff > d//2 and cross(cutoff, d), 'Late boundary does not cross its exact threshold')
            if cutoff > d//2+1:
                require(not cross(cutoff-1, d), 'Late boundary is not the first crossing state')
    null, strict = independent_bounds(table, piece.M)
    require(null <= Fraction(1, 100) and strict <= Fraction(1, 100), 'Independent production risk exceeds delta')
    saved = artifact['risk_bounds_exact']
    require(null == Fraction(beta.text_int(saved['null_union_numerator']), beta.text_int(saved['null_union_denominator'])) and
            strict == Fraction(beta.text_int(saved['strict_wrong_numerator']), beta.text_int(saved['strict_wrong_denominator'])),
            'Saved piecewise risk differs from independent full DP')
    neighbor = artifact['selection']['lower_neighbor']
    lower = beta.record_fraction(neighbor['threshold_exact'])
    lower_table = list(early[:piece.D0+1])+list(beta.constant_boundary(piece.H, lower)[piece.D0+1:])
    lower_null, lower_strict = independent_bounds(lower_table, piece.M)
    require(lower_null > Fraction(1, 100) or lower_strict > Fraction(1, 100), 'Lower late grid point is certified')
    limit_null, limit_strict = independent_bounds(piece.never_late_table(early, piece.D0), piece.M)
    return {'status': 'passed', 'unchanged_early_rows': piece.D0+1, 'independent_late_contour_rows': piece.H-piece.D0,
            'late_threshold_exact': artifact['late_threshold_exact'], 'null_upper': float(null), 'strict_upper': float(strict),
            'lower_neighbor_null': float(lower_null), 'lower_neighbor_strict': float(lower_strict),
            'never_late_null': float(limit_null), 'never_late_strict': float(limit_strict)}


def runtime_checks(artifact):
    runtime = piece.verify_artifact(artifact)
    first = piece.certificate_from_pairs([0, 6, 0, 0], runtime, [0, 6, 0, 0])
    reversed_later = piece.certificate_from_pairs([0, 6, 8, 0], artifact, [0, 6, 0, 0])
    require(first['decision'] == reversed_later['decision'] == 'reject' and first['shots'] == 12,
            'Fixed first-six prefix gate changed')
    for d, stage in ((piece.D0, 'early'), (piece.D0+1, 'late')):
        cutoff = artifact['boundaries'][d]
        cert = piece.certificate_from_pairs([0, cutoff, d-cutoff, 0], artifact, [0, 3, 3, 0])
        require(cert['decision'] == 'reject' and cert['continuation_stage'] == stage, 'Piecewise transition differs')
    history = [{'m': 1, 'pair_counts': [0, 1, 0, 0], 'budget': 2*n} for n in range(1, 17)]
    replay = piece.replay_history(history, artifact)
    require(replay['shots'] == 12 and replay['history_prefix_length'] == 6 and replay['certificate']['first_m_counts'] == [0, 6, 0, 0],
            'Replay did not freeze the actual first-six physical prefix')
    censored = piece.replay_history([{'m': 1, 'pair_counts': [1, 0, 0, 0], 'budget': 2}], artifact)
    require(censored['status'] == 'censored_existing_prefix' and censored['decision'] is None, 'Unobserved stopping time imputed')
    invalid = []
    def fails(name, function):
        try:
            function()
        except (ValueError, TypeError, KeyError):
            invalid.append(name)
        else:
            raise AssertionError('Invalid input accepted: '+name)
    fails('aggregate_straddles_prefix', lambda: piece.replay_history([{'m': 16, 'pair_counts': [0, 16, 0, 0], 'budget': 32}], artifact))
    fails('missing_first6', lambda: piece.certificate_from_pairs([0, 6, 0, 0], artifact))
    fails('concordant_physical_overrun', lambda: piece.certificate_from_pairs([piece.H+1, 0, 0, 0], artifact, [6, 0, 0, 0]))
    modified = deepcopy(artifact)
    modified['boundaries'][piece.D0] += 1
    fails('modified_early_runtime', lambda: piece.certificate_from_pairs([0, 300, 212, 0], modified, [0, 3, 3, 0]))
    return {'status': 'passed', 'prefix_stop_physical_shots': 12, 'transition_D512_D513_checked': True,
            'replay_censoring_preserved': True, 'invalid_inputs_rejected': invalid}


def main(output, artifact_path):
    began = time.perf_counter()
    result = {'status': 'running', 'scope': 'Exact deterministic checks only; no simulator or development-history replay', 'checks': {}}
    try:
        result['checks']['small_integer_dp'] = small_checks()
        artifact = piece.calibrate()
        result['checks']['production_piecewise'] = production_checks(artifact)
        result['checks']['runtime_and_fixed_fixture'] = runtime_checks(artifact)
        file_hash = piece.write_artifact(artifact_path, artifact)
        result['artifact'] = {'path': artifact_path.relative_to(piece.ROOT).as_posix(), 'file_sha256': file_hash,
                              'artifact_sha256': artifact['artifact_sha256'], 'boundary_sha256': artifact['boundary_sha256']}
        result['status'] = 'passed'
    except BaseException as caught:
        result.update(status='failed', error=type(caught).__name__+': '+str(caught))
        raise
    finally:
        result['sources'] = piece.source_hashes()
        result['sources'][Path(__file__).resolve().relative_to(piece.ROOT).as_posix()] = beta.file_sha256(__file__)
        result['elapsed_seconds'] = time.perf_counter()-began
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
        print(json.dumps({'status': result['status'], 'elapsed_seconds': result['elapsed_seconds'], 'output': str(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=piece.ROOT/'results/sota_round3/piecewise_gate_boundary_checks.json')
    parser.add_argument('--artifact', type=Path, default=piece.ROOT/'results/sota_round3/piecewise_gate_boundary.json')
    args = parser.parse_args()
    main(args.output, args.artifact)
