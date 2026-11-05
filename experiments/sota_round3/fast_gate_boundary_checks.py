"""Exact enumeration/DP/interface checks for the single physical-prefix gate.

No randomness or simulator observations are used. Enumerated finite sample
spaces are mathematical reference experiments, not empirical calibration.
"""
from copy import deepcopy
from fractions import Fraction
from itertools import product
from pathlib import Path
import argparse
import json
import time

import exact_sign_boundary as beta
import fast_gate_boundary as fast


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sign_event(word, table):
    plus, crossed = 0, False
    for n, bit in enumerate(word, 1):
        plus += bit
        crossed |= plus >= table[n] or plus <= n-table[n]
    return crossed


def independent_full_rows(table, m):
    """Full-row beta and restricted DP, independent of compact production DP."""
    full, restricted = [1], [1]
    for n in range(1, len(table)):
        next_full, next_restricted = [0]*(n+1), [0]*(n+1)
        for s in range(n):
            next_full[s] += full[s]
            next_full[s+1] += full[s]
            next_restricted[s] += restricted[s]
            next_restricted[s+1] += restricted[s]
        cutoff = table[n]
        full = [count if n-cutoff < s < cutoff else 0 for s, count in enumerate(next_full)]
        restricted = [count if n-cutoff < s < cutoff and (n > m or s in (0, n)) else 0
                      for s, count in enumerate(next_restricted)]
    return 2**(len(table)-1)-sum(full), sum(restricted)


def enumeration_checks():
    records = []
    for h, m in ((3, 1), (4, 2), (6, 2), (7, 3), (8, 4)):
        for threshold in (Fraction(1), Fraction(3, 2), Fraction(3), Fraction(10), Fraction(100)):
            table = beta.constant_boundary(h, threshold)
            numerator, restricted, union = 0, 0, 0
            for signs in product((0, 1), repeat=h):
                crossed = sign_event(signs, table)
                same = len(set(signs[:m])) == 1
                numerator += int(crossed)
                restricted += int(same and not crossed)
                for gates in product((0, 1), repeat=m):
                    union += int(crossed or same and all(gates))
            actual = fast.exact_risk_bounds(table, m)
            full_p, full_s = independent_full_rows(table, m)
            require(numerator == full_p == actual['beta_crossing_numerator'] and
                    restricted == full_s == actual['restricted_survivor_numerator'], 'Restricted survival DP differs from enumeration')
            require(union == actual['null_union_numerator'] and actual['null_union_denominator'] == 2**(h+m),
                    'Null beta/physical-gate union formula differs from full U/V enumeration')
            # Actual H physical fair independent Bernoulli pairs (four equal categories).
            physical_crossings = 0
            for outcomes in product(range(4), repeat=h):
                b, c, crossing = 0, 0, False
                for t, value in enumerate(outcomes, 1):
                    b += int(value == 1)
                    c += int(value == 2)
                    n = b+c
                    crossing |= b >= table[n] or b <= n-table[n]
                    if t == m:
                        crossing |= all(v == 1 for v in outcomes[:m]) or all(v == 2 for v in outcomes[:m])
                physical_crossings += int(crossing)
            require(physical_crossings*actual['null_union_denominator'] <= actual['null_union_numerator']*4**h,
                    'Actual finite physical experiment exceeds the latent null envelope')
            records.append({'H': h, 'm': m, 'threshold': str(threshold), 'beta_crossings': numerator,
                            'restricted_survivors': restricted, 'latent_union_numerator': union,
                            'latent_union_denominator': 2**(h+m), 'physical_crossings': physical_crossings,
                            'physical_denominator': 4**h})
    return {'status': 'passed', 'cases': records}


def adaptive_tree_checks():
    h, m, table = 6, 2, beta.constant_boundary(6, 3)
    bounds = fast.exact_risk_bounds(table, m)
    probabilities = {}
    for model in ('adaptive_null', 'adaptive_positive'):
        bad_probability = Fraction(0)
        total_probability = Fraction(0)
        for outcomes in product(range(4), repeat=h):
            probability, b, c, bad = Fraction(1), 0, 0, False
            for t, value in enumerate(outcomes, 1):
                if model == 'adaptive_null':
                    pm = pp = Fraction(1, 2) if (b+c) % 2 else Fraction(1, 4)
                else:
                    pm, pp = (Fraction(1, 4), Fraction(3, 4)) if b % 2 else (Fraction(1, 2), Fraction(3, 4))
                q = [(1-pm)*(1-pp), (1-pm)*pp, pm*(1-pp), pm*pp]
                probability *= q[value]
                b += int(value == 1)
                c += int(value == 2)
                n = b+c
                bad |= b <= n-table[n]
                if model == 'adaptive_null':
                    bad |= b >= table[n]
                if t == m:
                    bad |= all(v == 2 for v in outcomes[:m])
                    if model == 'adaptive_null':
                        bad |= all(v == 1 for v in outcomes[:m])
            total_probability += probability
            bad_probability += probability if bad else 0
        require(total_probability == 1, 'Enumerated adaptive physical tree is not normalized')
        key = 'null_union' if model == 'adaptive_null' else 'strict_wrong'
        upper = Fraction(bounds[key+'_numerator'], bounds[key+'_denominator'])
        require(bad_probability <= upper, 'Adaptive physical tree exceeds its certified union envelope')
        probabilities[model] = {'bad_union_probability': str(bad_probability), 'upper_bound': str(upper)}
    return {'status': 'passed', 'exact_finite_tree_examples': probabilities,
            'scope': 'Examples supplement the separately reviewed general proof; no Monte Carlo'}


def runtime_checks():
    artifact = fast.calibrate(32, 5, '.01', 10000)
    runtime = fast.verify_artifact(artifact)
    before = fast.certificate_from_pairs([0, 4, 0, 0], runtime)
    first = fast.certificate_from_pairs([0, 5, 0, 0], artifact, [0, 5, 0, 0])
    reversed_later = fast.certificate_from_pairs([0, 5, 10, 0], artifact, [0, 5, 0, 0])
    negative = fast.certificate_from_pairs([0, 0, 5, 0], artifact, [0, 0, 5, 0])
    failed_gate = fast.certificate_from_pairs([5, 0, 0, 0], artifact, [5, 0, 0, 0])
    require(before['decision'] == 'abstain' and before['shots'] == 8, 'Premature fast gate')
    require(first['decision'] == reversed_later['decision'] == 'reject' and negative['decision'] == 'accept',
            'Fixed prefix direction or persistence differs')
    require(first['shots'] == 10 and reversed_later['shots'] == 30 and failed_gate['decision'] == 'abstain',
            'Gate physical paid-cost accounting differs')
    require(first['decision_source'] == 'first_m_physical_prefix' and first['beta_decision'] == 'abstain',
            'Fast-only decision not identified')
    failures = []
    def invalid(name, call):
        try:
            call()
        except (ValueError, TypeError, KeyError):
            failures.append(name)
        else:
            raise AssertionError('Invalid input accepted: '+name)
    invalid('missing_prefix_after_m', lambda: fast.certificate_from_pairs([0, 5, 0, 0], artifact))
    invalid('premature_prefix', lambda: fast.certificate_from_pairs([0, 4, 0, 0], artifact, [0, 4, 0, 0]))
    invalid('wrong_prefix_length', lambda: fast.certificate_from_pairs([0, 5, 0, 0], artifact, [0, 4, 0, 0]))
    invalid('prefix_not_contained', lambda: fast.certificate_from_pairs([0, 5, 0, 0], artifact, [0, 0, 5, 0]))
    invalid('fractional_prefix', lambda: fast.certificate_from_pairs([0, 5, 0, 0], artifact, [0, 4.5, .5, 0]))
    invalid('boolean_counts', lambda: fast.certificate_from_pairs([False, 0, 0, 0], artifact))
    invalid('discordance_overrun', lambda: fast.certificate_from_pairs([0, 33, 0, 0], artifact, [0, 5, 0, 0]))
    changed = deepcopy(artifact)
    changed['prefix_pairs'] = 6
    invalid('changed_window_cached_runtime', lambda: fast.certificate_from_pairs([0, 6, 0, 0], changed, [0, 6, 0, 0]))
    changed['boundary_sha256'] = fast.boundary_hash(changed)
    changed['artifact_sha256'] = beta.payload_sha256(changed)
    invalid('changed_window_rehashed_artifact', lambda: fast.verify_artifact(changed))
    return {'status': 'passed', 'prefix_physical_shots': 10, 'persistent_prefix_direction_checked': True,
            'invalid_inputs_rejected': failures,
            'chronology_limit': 'Componentwise count checks cannot prove prefix provenance; driver must bind recorded first m pairs'}


def independent_production_checks(artifact):
    h, m, table = artifact['max_discordances'], artifact['prefix_pairs'], artifact['beta_boundaries']
    crossing, restricted = independent_full_rows(table, m)
    null = Fraction(crossing*2**m+restricted, 2**(h+m))
    strict = Fraction(crossing, 2**h)+Fraction(1, 4**m)
    delta = beta.record_fraction(artifact['delta_exact'])
    require(null <= delta and strict <= delta, 'Production full-row risk checks exceed delta')
    saved = artifact['risk_bounds_exact']
    require(crossing == beta.text_int(saved['beta_crossing_numerator']) and
            restricted == beta.text_int(saved['restricted_survivor_numerator']), 'Production DP numerators differ')
    previous_h = beta.record_fraction(artifact['selection']['lower_neighbor']['threshold_exact'])
    previous_table = beta.constant_boundary(h, previous_h)
    previous_p, previous_s = independent_full_rows(previous_table, m)
    previous_null = Fraction(previous_p*2**m+previous_s, 2**(h+m))
    previous_strict = Fraction(previous_p, 2**h)+Fraction(1, 4**m)
    require(previous_null > delta or previous_strict > delta, 'Lower-neighbor noncertification witness differs')
    return {'status': 'passed', 'm': m, 'threshold_exact': artifact['beta_threshold_exact'],
            'null_union_upper': float(null), 'strict_wrong_upper': float(strict),
            'previous_null_upper': float(previous_null), 'previous_strict_upper': float(previous_strict),
            'beta_first_all_same_crossing': next((n for n in range(1, h+1) if table[n] <= n), None)}


def main(output, directory):
    start = time.perf_counter()
    result = {'status': 'running', 'scope': 'Independent exact finite-space enumeration and integer DP; zero simulator observations',
              'checks': {}, 'artifacts': []}
    try:
        for name, check in [('null_overlap_enumeration', enumeration_checks),
                            ('adaptive_physical_trees', adaptive_tree_checks), ('runtime_interface', runtime_checks)]:
            result['checks'][name] = check()
        for m in (5, 6, 7):
            artifact = fast.calibrate(2048, m)
            checked = independent_production_checks(artifact)
            path = directory/f'm{m}.json'
            file_hash = fast.write_artifact(path, artifact)
            result['artifacts'].append({'path': path.relative_to(fast.ROOT).as_posix(), 'file_sha256': file_hash,
                                        'artifact_sha256': artifact['artifact_sha256'], 'boundary_sha256': artifact['boundary_sha256'],
                                        'checks': checked})
        result['status'] = 'passed'
    except BaseException as caught:
        result.update(status='failed', error=type(caught).__name__+': '+str(caught))
        raise
    finally:
        result['sources'] = fast.source_hashes()
        result['sources'][Path(__file__).resolve().relative_to(fast.ROOT).as_posix()] = beta.file_sha256(__file__)
        result['elapsed_seconds'] = time.perf_counter()-start
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
        print(json.dumps({'status': result['status'], 'output': str(output), 'elapsed_seconds': result['elapsed_seconds']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=fast.ROOT/'results/sota_round3/fast_gate_boundary_checks.json')
    parser.add_argument('--directory', type=Path, default=fast.ROOT/'results/sota_round3/fast_gate_boundaries')
    args = parser.parse_args()
    main(args.output, args.directory)
