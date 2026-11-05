"""Fixed m=6 physical-prefix gate with an unchanged early beta boundary.

The released design constants are H=2048, D0=512, and the saved beta_exact
early threshold 81.174302. Only the rational late threshold is calibrated.
The concatenated symmetric integer table and the single fixed prefix gate
are certified using the already reviewed two exact risk-bound formulas.
This module neither edits prior artifacts nor generates observations.
"""
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import argparse
import hashlib
import json

import exact_sign_boundary as beta
import fast_gate_boundary as fast

ROOT = Path(__file__).resolve().parents[2]
EARLY_PATH = ROOT/'results/sota_round3/beta_boundary_N2048.json'
GENERIC_LEMMA = ROOT/'results/sota_round2/theory/generic_symmetric_tail_dp_lemma.json'
H, M, D0 = 2048, 6, 512
EARLY_THRESHOLD = Fraction(40587151, 500000)
VERSION = 'fixed_early_beta_piecewise_physical_gate_v1'
_VERIFIED = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_hashes():
    result = fast.source_hashes()
    for path in (Path(__file__).resolve(), EARLY_PATH, GENERIC_LEMMA):
        result[path.relative_to(ROOT).as_posix()] = beta.file_sha256(path)
    return result


def early_artifact():
    artifact = beta.load_boundary(EARLY_PATH)
    require(artifact['max_discordances'] == H and beta.record_fraction(artifact['threshold_exact']) == EARLY_THRESHOLD and
            beta.record_fraction(artifact['delta_exact']) == Fraction(1, 100), 'Frozen early beta reference differs')
    return artifact


def concatenate_table(early_table, switch, late_threshold):
    beta.validate_table(early_table)
    switch = beta.integer(switch, 'switch_discordances', 1)
    maximum = len(early_table)-1
    require(switch < maximum, 'Switch must precede the finite horizon')
    late = beta.constant_boundary(maximum, late_threshold)
    return tuple(early_table[:switch+1])+late[switch+1:]


def never_late_table(early_table, switch):
    return tuple(early_table[:switch+1])+tuple(d+1 for d in range(switch+1, len(early_table)))


def table_hash(artifact):
    core = {key: artifact[key] for key in ('version', 'max_discordances', 'prefix_pairs', 'switch_discordances',
                                          'delta_exact', 'early_threshold_exact', 'late_threshold_exact', 'boundaries')}
    return hashlib.sha256(json.dumps(core, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def calibrate(threshold_scale=1000000):
    scale = beta.integer(threshold_scale, 'threshold_scale', 1)
    reference, delta = early_artifact(), Fraction(1, 100)
    early = tuple(reference['boundaries'])
    limit = fast.exact_risk_bounds(never_late_table(early, D0), M)
    require(fast.certified(limit, delta), 'Keeping the early table leaves insufficient risk even with no late stops')
    minimum = (EARLY_THRESHOLD.numerator*scale+EARLY_THRESHOLD.denominator-1)//EARLY_THRESHOLD.denominator
    initial_bounds = fast.exact_risk_bounds(early, M)
    calls, cache = 0, {early: initial_bounds}
    def evaluate(index):
        nonlocal calls
        table = concatenate_table(early, D0, Fraction(index, scale))
        calls += 1
        if table not in cache:
            cache[table] = fast.exact_risk_bounds(table, M)
        return table, cache[table]
    low = high = minimum
    table, bounds = evaluate(high)
    if not fast.certified(bounds, delta):
        while not fast.certified(bounds, delta):
            low, high = high, 2*high
            table, bounds = evaluate(high)
        while high-low > 1:
            middle = (low+high)//2
            trial, trial_bounds = evaluate(middle)
            if fast.certified(trial_bounds, delta):
                high, table, bounds = middle, trial, trial_bounds
            else:
                low = middle
    previous = {'threshold_exact': beta.fraction_record(Fraction(high-1, scale))}
    if high == minimum:
        previous['excluded_below_early_threshold'] = True
    else:
        unused, previous_bounds = evaluate(high-1)
        require(not fast.certified(previous_bounds, delta), 'The lower late-threshold grid neighbor also passes')
        previous['risk_bounds_exact'] = fast.bounds_record(previous_bounds)
    require(tuple(table[:D0+1]) == early[:D0+1] and all(new >= old for new, old in zip(table, early)),
            'Piecewise table changed early decisions or relaxed beta continuation')
    result = {'version': VERSION, 'status': 'piecewise_table_and_single_gate_risk_bounds_certified',
              'max_discordances': H, 'max_physical_pairs': H, 'prefix_pairs': M, 'switch_discordances': D0,
              'delta_exact': beta.fraction_record(delta), 'early_threshold_exact': beta.fraction_record(EARLY_THRESHOLD),
              'late_threshold_exact': beta.fraction_record(Fraction(high, scale)), 'boundaries': list(table),
              'early_boundary_reference': {'path': EARLY_PATH.relative_to(ROOT).as_posix(),
                                           'file_sha256': beta.file_sha256(EARLY_PATH),
                                           'artifact_sha256': reference['artifact_sha256']},
              'risk_bounds_exact': fast.bounds_record(bounds),
              'same_threshold_plus_gate_risk_bounds': fast.bounds_record(initial_bounds),
              'never_late_stop_feasibility_bounds': fast.bounds_record(limit),
              'selection': {'kind': 'least_jointly_certified_late_threshold_on_fixed_rational_lattice',
                            'threshold_scale': scale, 'chosen_lattice_index': high, 'minimum_lattice_index': minimum,
                            'lower_neighbor': previous, 'boundary_evaluations': calls,
                            'distinct_integer_dp_evaluations': len(cache)},
              'crossing_convention': 'Persistent actual first-6 physical-prefix gate takes priority; otherwise use B[D]. D<=512 uses frozen early beta; D>512 uses the calibrated late contour.',
              'development_selection_scope': 'H, m, D0 and early reference were fixed using deterministic null feasibility and disclosed development diagnostics. This is not a held-out protocol selection or an empirical superiority claim.',
              'limits': 'One fixed prefix only; conditional independent paired Bernoulli measurements and common sign ordering required. Physical pairs<=2048. Prefix chronology requires the recorded history. No union of candidate rules is certified.',
              'source_hashes': source_hashes()}
    result['boundary_sha256'] = table_hash(result)
    result['artifact_sha256'] = beta.payload_sha256(result)
    return result


@dataclass(frozen=True)
class RuntimeBoundary:
    table: tuple
    boundary_sha256: str
    artifact_sha256: str


def verify_artifact(artifact):
    require(artifact['version'] == VERSION and artifact['status'] == 'piecewise_table_and_single_gate_risk_bounds_certified',
            'Unknown piecewise boundary artifact')
    require(artifact['source_hashes'] == source_hashes() and beta.payload_sha256(artifact) == artifact['artifact_sha256'] and
            table_hash(artifact) == artifact['boundary_sha256'], 'Piecewise artifact/source binding differs')
    require(artifact['max_discordances'] == artifact['max_physical_pairs'] == H and artifact['prefix_pairs'] == M and
            artifact['switch_discordances'] == D0 and beta.record_fraction(artifact['delta_exact']) == Fraction(1, 100) and
            beta.record_fraction(artifact['early_threshold_exact']) == EARLY_THRESHOLD, 'Fixed piecewise design changed')
    reference = early_artifact()
    binding = artifact['early_boundary_reference']
    require(binding['path'] == EARLY_PATH.relative_to(ROOT).as_posix() and binding['file_sha256'] == beta.file_sha256(EARLY_PATH) and
            binding['artifact_sha256'] == reference['artifact_sha256'], 'Early table reference differs')
    late = beta.record_fraction(artifact['late_threshold_exact'])
    require(late >= EARLY_THRESHOLD, 'Late contour cannot be below the frozen early threshold')
    table = concatenate_table(reference['boundaries'], D0, late)
    require(list(table) == artifact['boundaries'], 'Piecewise integer table differs from its two rational contours')
    bounds = fast.exact_risk_bounds(table, M)
    require(fast.bounds_record(bounds) == artifact['risk_bounds_exact'] and fast.certified(bounds, Fraction(1, 100)),
            'Piecewise exact joint risk check failed')
    no_late = fast.exact_risk_bounds(never_late_table(reference['boundaries'], D0), M)
    initial = fast.exact_risk_bounds(reference['boundaries'], M)
    require(fast.bounds_record(no_late) == artifact['never_late_stop_feasibility_bounds'] and
            fast.bounds_record(initial) == artifact['same_threshold_plus_gate_risk_bounds'], 'Feasibility bound metadata differs')
    selection = artifact['selection']
    scale = beta.integer(selection['threshold_scale'], 'threshold_scale', 1)
    index = beta.integer(selection['chosen_lattice_index'], 'chosen_lattice_index', 1)
    minimum = (EARLY_THRESHOLD.numerator*scale+EARLY_THRESHOLD.denominator-1)//EARLY_THRESHOLD.denominator
    require(selection['kind'] == 'least_jointly_certified_late_threshold_on_fixed_rational_lattice' and
            selection['minimum_lattice_index'] == minimum and index >= minimum and late == Fraction(index, scale),
            'Piecewise threshold search binding differs')
    previous = selection['lower_neighbor']
    require(beta.record_fraction(previous['threshold_exact']) == Fraction(index-1, scale), 'Incorrect lower lattice neighbor')
    if index == minimum:
        require(previous['excluded_below_early_threshold'] is True and Fraction(index-1, scale) < EARLY_THRESHOLD,
                'Invalid lower-neighbor exclusion')
    else:
        lower_table = concatenate_table(reference['boundaries'], D0, Fraction(index-1, scale))
        lower_bounds = fast.exact_risk_bounds(lower_table, M)
        require(fast.bounds_record(lower_bounds) == previous['risk_bounds_exact'] and
                not fast.certified(lower_bounds, Fraction(1, 100)), 'Lower late-threshold noncertification witness failed')
    runtime = RuntimeBoundary(table, artifact['boundary_sha256'], artifact['artifact_sha256'])
    stored = _VERIFIED.setdefault(runtime.artifact_sha256, runtime)
    require(stored == runtime, 'Cached piecewise boundary differs')
    return stored


def runtime_boundary(boundary):
    if isinstance(boundary, RuntimeBoundary):
        require(_VERIFIED.get(boundary.artifact_sha256) is boundary, 'Use a verified piecewise runtime object')
        return boundary
    require(isinstance(boundary, dict), 'Require an artifact dictionary or verified runtime object')
    runtime = _VERIFIED.get(boundary.get('artifact_sha256'))
    if runtime is None:
        runtime = verify_artifact(boundary)
    require(tuple(boundary['boundaries']) == runtime.table and boundary['boundary_sha256'] == runtime.boundary_sha256 and
            boundary['max_physical_pairs'] == H and boundary['prefix_pairs'] == M and boundary['switch_discordances'] == D0,
            'Verified piecewise decision inputs changed')
    return runtime


def certificate_from_pairs(counts, boundary, first_m_counts=None):
    boundary = runtime_boundary(boundary)
    values = fast.pair_counts(counts)
    pairs, plus, minus = sum(values), values[1], values[2]
    d = plus+minus
    require(pairs <= H and d <= H, 'Piecewise physical-pair or discordance horizon exceeded')
    prefix, gate = None, 'abstain'
    if pairs < M:
        require(first_m_counts is None, 'First-six prefix is unavailable before six physical pairs')
    else:
        prefix = fast.pair_counts(first_m_counts)
        require(sum(prefix) == M and all(a <= b for a, b in zip(prefix, values)), 'Invalid frozen first-six prefix counts')
        gate = 'reject' if prefix[1] == M else 'accept' if prefix[2] == M else 'abstain'
    cutoff = boundary.table[d]
    continuation = 'reject' if plus >= cutoff else 'accept' if plus <= d-cutoff else 'abstain'
    return {'method': 'piecewise_beta_with_fixed_first6_gate', 'decision': gate if gate != 'abstain' else continuation,
            'shots': 2*pairs, 'pairs': pairs, 'discordances': d, 'counts': list(values), 'score': plus-minus,
            'first_m_counts': list(prefix) if prefix is not None else None, 'prefix_pairs': M,
            'fast_gate_decision': gate, 'beta_decision': continuation, 'beta_integer_boundary': cutoff,
            'continuation_stage': 'early' if d <= D0 else 'late', 'switch_discordances': D0,
            'boundary_sha256': boundary.boundary_sha256, 'boundary_artifact_sha256': boundary.artifact_sha256,
            'delta_exact': {'numerator': '1', 'denominator': '100'}, 'finite_horizon_only': True,
            'max_physical_pairs': H, 'max_discordances': H}


def replay_history(history, boundary):
    """Only already recorded endpoints, with exact first-six prefix extraction."""
    counts, first, checks = [0, 0, 0, 0], None, []
    before_pairs, last_budget = 0, 0
    for index, block in enumerate(history):
        fresh = fast.pair_counts(block['pair_counts'])
        number = beta.integer(block['m'], 'block pairs', 1)
        require(sum(fresh) == number, 'Block pair counts differ from its paid cost')
        require(not before_pairs < M < before_pairs+number, 'Recorded aggregate straddles first-six boundary; cannot invent ordering')
        counts = [a+b for a, b in zip(counts, fresh)]
        before_pairs += number
        if before_pairs == M:
            first = counts.copy()
        budget = beta.integer(block['budget'], 'physical budget', 1)
        require(budget >= last_budget and 2*before_pairs <= budget, 'Invalid recorded checkpoint order')
        last_budget = budget
        if index+1 < len(history) and history[index+1]['budget'] == budget:
            continue
        require(2*before_pairs == budget, 'Incomplete recorded physical endpoint')
        cert = certificate_from_pairs(counts, boundary, first)
        checks.append({'history_prefix_length': index+1, 'certificate': cert})
        if cert['decision'] != 'abstain':
            return {'status': 'stopped_on_observed_checkpoint', 'shots': budget, 'decision': cert['decision'],
                    'history_prefix_length': index+1, 'certificate': cert, 'checkpoint_history': checks}
    cert = certificate_from_pairs(counts, boundary, first)
    complete_cap = 2*before_pairs == 2*H
    return {'status': 'cap_abstention' if complete_cap else 'censored_existing_prefix',
            'shots': 2*before_pairs, 'decision': 'abstain' if complete_cap else None,
            'history_prefix_length': len(history), 'certificate': cert, 'checkpoint_history': checks}


def write_artifact(path, artifact):
    verify_artifact(artifact)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(artifact, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return beta.file_sha256(path)


def load_boundary(path):
    artifact = json.loads(Path(path).read_text(encoding='utf-8'))
    verify_artifact(artifact)
    return artifact


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = calibrate()
    file_hash = write_artifact(args.output, result)
    print(json.dumps({'status': result['status'], 'late_threshold_exact': result['late_threshold_exact'],
                      'null_upper': result['risk_bounds_exact']['null_union_probability_float_diagnostic'],
                      'strict_upper': result['risk_bounds_exact']['strict_wrong_probability_float_diagnostic'],
                      'file_sha256': file_hash}))
