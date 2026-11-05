"""Exact finite-discordance Jeffreys sign boundaries, without simulation.

The threshold is chosen using only a declared maximum discordance count N,
delta and a fixed rational search lattice. Integer dynamic programming counts
ALL fair-sign paths that cross either tail by N. Runtime decisions use only
integer counts and the frozen boundary table; no floating evidence is used.

Validity under predictable, commonly ordered measurement pairs follows by
coupling each actual discordant sign to a latent fair-sign sequence. Do not
condition on the observed terminal discordance count: that conditioning need
not preserve fairness. Sparse physical monitoring is a subset of the latent
all-discordance crossing event. The guarantee ends at discordance count N.

This is a deterministic boundary-calibration implementation, not a claim of
novelty, simulator performance, or an infinite-horizon guarantee.
"""
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral
from pathlib import Path
import argparse
import hashlib
import json
import math


ROOT = Path(__file__).resolve().parents[2]
LEMMA_PATH = ROOT/'results/sota_round2/theory/finite_cap_jeffreys_dp_lemma.json'
VERSION = 'exact_finite_discordance_jeffreys_v1'
_VERIFIED_BOUNDARIES = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, name, minimum=0):
    require(isinstance(value, Integral) and not isinstance(value, bool), name+' must be an integer')
    value = int(value)
    require(value >= minimum, name+' is below its allowed minimum')
    return value


def rational(value, name):
    require(not isinstance(value, bool), name+' must be rational')
    try:
        result = value if isinstance(value, Fraction) else Fraction(str(value))
    except (ValueError, TypeError, ZeroDivisionError, OverflowError) as caught:
        raise ValueError(name+' must be a finite rational value') from caught
    return result


def int_text(value):
    """Decimal serialization without Python's configurable huge-int limit."""
    value = integer(value, 'serialized integer')
    parts, base = [], 10**1000
    while value >= base:
        value, remainder = divmod(value, base)
        parts.append(str(remainder).zfill(1000))
    return str(value)+''.join(reversed(parts))


def text_int(value):
    require(isinstance(value, str) and value and value.isascii() and value.isdigit(),
            'Exact integer fields require nonnegative decimal strings')
    result = 0
    for start in range(0, len(value), 1000):
        chunk = value[start:start+1000]
        result = result*10**len(chunk)+int(chunk)
    return result


def fraction_record(value):
    return {'numerator': int_text(value.numerator), 'denominator': int_text(value.denominator)}


def record_fraction(value):
    return Fraction(text_int(value['numerator']), text_int(value['denominator']))


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def payload_sha256(value):
    payload = {name: item for name, item in value.items() if name != 'artifact_sha256'}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def boundary_sha256(value):
    core = {name: value[name] for name in
            ('version', 'max_discordances', 'delta_exact', 'threshold_exact', 'boundaries')}
    return hashlib.sha256(json.dumps(core, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source_hashes():
    return {path.relative_to(ROOT).as_posix(): file_sha256(path)
            for path in (Path(__file__).resolve(), LEMMA_PATH)}


def jeffreys_capital(plus, discordances):
    """Exact M_n(s), principally for independent checking and diagnostics."""
    n, s = integer(discordances, 'discordances'), integer(plus, 'plus')
    require(s <= n, 'plus exceeds discordances')
    c = n-s
    return Fraction(math.comb(2*s, s)*math.comb(2*c, c), (1 << n)*math.comb(n, s))


def constant_boundary(max_discordances, threshold):
    """Return B[0:N+1]; crossing is s>=B[n] or s<=n-B[n].

    B[n]=n+1 means no crossing. Every B[n] is strictly above n/2,
    excluding the zero score. Exact Fraction recurrences find the first
    crossing point of each positive tail without factorial search or logs.
    """
    maximum = integer(max_discordances, 'max_discordances', 1)
    h = rational(threshold, 'threshold')
    require(h > 0, 'threshold must be positive')
    boundaries, plus, capital = [1], 0, Fraction(1)
    for n in range(1, maximum+1):
        # Append a minus symbol at the current plus count.
        capital *= Fraction(2*(n-1-plus)+1, n)
        # Move toward the positive extreme until the exact tail crosses.
        while plus < n and (2*plus <= n or capital < h):
            capital *= Fraction(2*plus+1, 2*(n-plus)-1)
            plus += 1
        boundaries.append(plus if 2*plus > n and capital >= h else n+1)
    return tuple(boundaries)


def validate_table(boundaries):
    require(isinstance(boundaries, (tuple, list)) and len(boundaries) >= 2 and boundaries[0] == 1,
            'Boundary needs the zero index followed by at least one discordance')
    for n, value in enumerate(boundaries[1:], 1):
        boundary = integer(value, 'tail boundary')
        require(n//2+1 <= boundary <= n+1, 'Tail boundary includes zero score or is out of range')


def count_surviving_paths(boundaries):
    """Exact integer DP, keeping only the contiguous surviving count range."""
    validate_table(boundaries)
    lower, counts = 0, [1]
    first_crossing_counts = []
    previous_total = 1
    for n, boundary in enumerate(boundaries[1:], 1):
        first = max(lower, n-boundary+1, 0)
        last = min(lower+len(counts), boundary-1, n)
        fresh = []
        for plus in range(first, last+1):
            old = plus-lower
            minus_parent = counts[old] if 0 <= old < len(counts) else 0
            plus_parent = counts[old-1] if 0 <= old-1 < len(counts) else 0
            fresh.append(minus_parent+plus_parent)
        total = sum(fresh)
        first_crossing_counts.append(2*previous_total-total)
        require(0 <= total <= 2*previous_total, 'Survival count recurrence is inconsistent')
        lower, counts, previous_total = first, fresh, total
    maximum = len(boundaries)-1
    denominator = 1 << maximum
    crossed = denominator-previous_total
    require(crossed == sum(value << (maximum-n) for n, value in enumerate(first_crossing_counts, 1)),
            'First-crossing decomposition differs from terminal survival count')
    return {'maximum': maximum, 'survivor_lower_plus': lower, 'survivor_counts': tuple(counts),
            'surviving_total': previous_total, 'crossing_numerator': crossed, 'fair_denominator': denominator,
            'first_crossing_counts': tuple(first_crossing_counts)}


def risk_record(counts, retain_states=True):
    result = {'crossing_numerator': int_text(counts['crossing_numerator']),
              'fair_denominator': int_text(counts['fair_denominator']),
              'surviving_total': int_text(counts['surviving_total']),
              'probability_fraction': fraction_record(Fraction(counts['crossing_numerator'], counts['fair_denominator'])),
              'probability_float_diagnostic': counts['crossing_numerator']/counts['fair_denominator'],
              'arithmetic': 'Exact integer path counts; float probability is display only'}
    if retain_states:
        result.update(survivor_lower_plus=counts['survivor_lower_plus'],
                      survivor_counts=[int_text(value) for value in counts['survivor_counts']],
                      first_crossing_counts=[int_text(value) for value in counts['first_crossing_counts']])
    return result


def within_risk(counts, delta):
    return counts['crossing_numerator']*delta.denominator <= counts['fair_denominator']*delta.numerator


def _artifact(maximum, delta, threshold, boundaries, counts, selection):
    require(within_risk(counts, delta), 'Boundary exceeds its declared exact risk budget')
    result = {'version': VERSION, 'status': 'exact_finite_horizon_boundary_certified',
              'max_discordances': maximum, 'delta_exact': fraction_record(delta),
              'threshold_exact': fraction_record(threshold), 'crossing_comparison': 'M_n(s) >= threshold; score != 0',
              'prior_exact': [1, 2], 'orientation': '[00, plus-only, minus-only, 11]; plus score reports reject',
              'physical_budget_scope': 'A 2*N physical-shot cap guarantees D<=N; the caller enforces its physical cap. This module directly enforces only the discordance cap.',
              'boundary_encoding': 'B[n]; stop reject at s>=B[n], accept at s<=n-B[n]; B[n]=n+1 means no crossing',
              'boundaries': list(boundaries), 'fair_null_risk': risk_record(counts), 'selection': selection,
              'source_hashes': source_hashes(),
              'validity': 'Uniform wrong-sign control through discordance N under predictable commonly ordered pairs. Fair latent all-index crossing controls sparse or adaptive physical monitoring; do not condition on terminal D.',
              'scope': 'Threshold selection uses only N, delta, and the declared deterministic lattice. No simulator data. Finite discordance horizon; no continuation or performance claim.'}
    result['boundary_sha256'] = boundary_sha256(result)
    result['artifact_sha256'] = payload_sha256(result)
    return result


def certify_threshold(max_discordances, delta='.01', threshold='100'):
    maximum = integer(max_discordances, 'max_discordances', 1)
    risk = rational(delta, 'delta')
    require(0 < risk < 1, 'delta must lie strictly between zero and one')
    h = rational(threshold, 'threshold')
    boundaries = constant_boundary(maximum, h)
    counts = count_surviving_paths(boundaries)
    return _artifact(maximum, risk, h, boundaries, counts, {'kind': 'fixed_declared_rational_threshold'})


def calibrate(max_discordances, delta='.01', threshold_scale=1000000):
    """Smallest safe positive threshold on the fixed 1/scale lattice.

    Both the chosen threshold and the immediately lower lattice point receive
    exact integer-DP checks. This proves lattice minimality, not unrestricted
    optimality among every possible time-dependent or real-valued boundary.
    """
    maximum = integer(max_discordances, 'max_discordances', 1)
    scale = integer(threshold_scale, 'threshold_scale', 1)
    risk = rational(delta, 'delta')
    require(0 < risk < 1, 'delta must lie strictly between zero and one')
    low, high = 0, (scale*risk.denominator+risk.numerator-1)//risk.numerator
    evaluations, cache = 0, {}
    def evaluate(index):
        nonlocal evaluations
        table = constant_boundary(maximum, Fraction(index, scale))
        evaluations += 1
        if table not in cache:
            cache[table] = count_surviving_paths(table)
        return table, cache[table]
    table, counts = evaluate(high)
    require(within_risk(counts, risk), 'The Ville upper bracket failed exact DP')
    while high-low > 1:
        middle = (low+high)//2
        candidate_table, candidate_counts = evaluate(middle)
        if within_risk(candidate_counts, risk):
            high, table, counts = middle, candidate_table, candidate_counts
        else:
            low = middle
    predecessor = None
    if low > 0:
        previous_table, previous_counts = evaluate(low)
        require(not within_risk(previous_counts, risk), 'Lower grid neighbor should exceed delta')
        predecessor = {'threshold_exact': fraction_record(Fraction(low, scale)),
                       'fair_null_risk': risk_record(previous_counts, retain_states=False)}
    else:
        # Zero is excluded; at h=0 every first discordance would cross.
        predecessor = {'threshold_exact': fraction_record(Fraction(0)), 'excluded_nonpositive': True,
                       'crossing_probability_if_allowed': fraction_record(Fraction(1))}
    selection = {'kind': 'least_safe_threshold_on_fixed_positive_rational_lattice',
                 'threshold_scale': scale, 'chosen_lattice_index': high,
                 'lower_neighbor': predecessor, 'boundary_evaluations': evaluations,
                 'distinct_integer_dp_evaluations': len(cache),
                 'performance_data_used': False, 'strict_vs_nonstrict': 'Crossing includes exact equality'}
    return _artifact(maximum, risk, Fraction(high, scale), table, counts, selection)


@dataclass(frozen=True)
class RuntimeBoundary:
    max_discordances: int
    delta: Fraction
    threshold: Fraction
    boundaries: tuple
    artifact_sha256: str
    boundary_sha256: str
    crossing_probability: Fraction

    def decision(self, plus, discordances):
        require(_VERIFIED_BOUNDARIES.get(self.artifact_sha256) is self,
                'RuntimeBoundary must be produced by verify_artifact')
        n, s = integer(discordances, 'discordances'), integer(plus, 'plus')
        require(s <= n <= self.max_discordances, 'Counts outside the certified finite discordance horizon')
        if 2*s == n:
            return 'abstain'
        boundary = self.boundaries[n]
        return 'reject' if s >= boundary else 'accept' if s <= n-boundary else 'abstain'


def verify_artifact(artifact, check_sources=True):
    require(artifact['version'] == VERSION and artifact['status'] == 'exact_finite_horizon_boundary_certified',
            'Unknown boundary artifact')
    require(payload_sha256(artifact) == artifact['artifact_sha256'], 'Boundary payload hash mismatch')
    require(boundary_sha256(artifact) == artifact['boundary_sha256'], 'Stopping boundary hash mismatch')
    if check_sources:
        require(artifact['source_hashes'] == source_hashes(), 'Boundary sources changed; review and re-certify')
    maximum = integer(artifact['max_discordances'], 'max_discordances', 1)
    require(artifact['prior_exact'] == [1, 2] and
            artifact['crossing_comparison'] == 'M_n(s) >= threshold; score != 0' and
            artifact['orientation'] == '[00, plus-only, minus-only, 11]; plus score reports reject',
            'Prior, crossing convention, or paired orientation changed')
    delta, threshold = record_fraction(artifact['delta_exact']), record_fraction(artifact['threshold_exact'])
    require(0 < delta < 1 and threshold > 0, 'Invalid risk or threshold')
    boundaries = constant_boundary(maximum, threshold)
    require(list(boundaries) == artifact['boundaries'], 'Integer boundaries differ from exact Jeffreys threshold')
    counts = count_surviving_paths(boundaries)
    require(risk_record(counts) == artifact['fair_null_risk'] and within_risk(counts, delta),
            'Exact fair-risk certificate differs or exceeds delta')
    selection = artifact['selection']
    if selection['kind'] == 'least_safe_threshold_on_fixed_positive_rational_lattice':
        scale = integer(selection['threshold_scale'], 'threshold_scale', 1)
        index = integer(selection['chosen_lattice_index'], 'chosen_lattice_index', 1)
        require(threshold == Fraction(index, scale), 'Threshold lattice binding differs')
        previous = selection['lower_neighbor']
        require(record_fraction(previous['threshold_exact']) == Fraction(index-1, scale), 'Wrong lower lattice neighbor')
        if index > 1:
            previous_counts = count_surviving_paths(constant_boundary(maximum, Fraction(index-1, scale)))
            require(previous['fair_null_risk'] == risk_record(previous_counts, retain_states=False) and
                    not within_risk(previous_counts, delta), 'Lattice minimality witness failed')
        else:
            require(previous['excluded_nonpositive'] is True and
                    record_fraction(previous['crossing_probability_if_allowed']) == 1,
                    'Excluded zero-neighbor witness differs')
    else:
        require(selection['kind'] == 'fixed_declared_rational_threshold', 'Unknown threshold selection rule')
    runtime = RuntimeBoundary(maximum, delta, threshold, boundaries, artifact['artifact_sha256'],
                              artifact['boundary_sha256'], Fraction(counts['crossing_numerator'], counts['fair_denominator']))
    existing = _VERIFIED_BOUNDARIES.setdefault(artifact['artifact_sha256'], runtime)
    require(existing == runtime, 'Cached boundary differs from newly verified artifact')
    return existing


def write_artifact(path, artifact):
    verify_artifact(artifact)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(artifact, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return file_sha256(path)


def load_boundary(path):
    artifact = json.loads(Path(path).read_text(encoding='utf-8'))
    verify_artifact(artifact)
    return artifact


def runtime_boundary(boundary):
    if isinstance(boundary, RuntimeBoundary):
        require(_VERIFIED_BOUNDARIES.get(boundary.artifact_sha256) is boundary,
                'RuntimeBoundary must be produced by verify_artifact')
        return boundary
    require(isinstance(boundary, dict), 'Require a boundary artifact dictionary')
    runtime = _VERIFIED_BOUNDARIES.get(boundary.get('artifact_sha256'))
    if runtime is None:
        runtime = verify_artifact(boundary)
    # Cache only the already certified immutable object. Recheck all decision
    # inputs even on cache hits, so a caller cannot alter the runtime table.
    require(boundary['max_discordances'] == runtime.max_discordances and
            tuple(boundary['boundaries']) == runtime.boundaries and
            record_fraction(boundary['delta_exact']) == runtime.delta and
            record_fraction(boundary['threshold_exact']) == runtime.threshold and
            boundary['boundary_sha256'] == runtime.boundary_sha256, 'Verified boundary decision inputs changed')
    return runtime


def certificate_from_pairs(counts, boundary):
    """The old beta function's paid-pair input, with integer stopping output."""
    boundary = runtime_boundary(boundary)
    require(isinstance(counts, (tuple, list)) and len(counts) == 4, 'Require four paired outcome counts')
    values = [integer(value, 'pair count') for value in counts]
    pairs, plus, minus = sum(values), values[1], values[2]
    discordances = plus+minus
    decision = boundary.decision(plus, discordances)
    cutoff = boundary.boundaries[discordances]
    return {'method': 'exact_finite_discordance_jeffreys', 'decision': decision, 'shots': 2*pairs,
            'pairs': pairs, 'discordant_pairs': discordances, 'discordances': discordances, 'score': plus-minus, 'counts': values,
            'delta_exact': fraction_record(boundary.delta), 'threshold_exact': fraction_record(boundary.threshold),
            'integer_boundary': {'plus_count': plus, 'upper': cutoff, 'lower': discordances-cutoff},
            'boundary_artifact_sha256': boundary.artifact_sha256,
            'boundary_sha256': boundary.boundary_sha256,
            'max_discordances': boundary.max_discordances, 'valid_up_to_registered_discordance_horizon': True,
            'numeric_status': 'Decision uses integer counts and the exact certified boundary table',
            'scope': 'Predictable paired queries with common sign ordering; finite discordance horizon only'}


def replay_pair_history(history, boundary):
    """Check only old physical block endpoints; never invent within-block order.

    Input uses run_refined history fields m/pair_counts/budget. Consecutive
    phases with the same budget are combined before testing. Exhausting an
    old stopped prefix is explicitly distinguished from reaching a new cap.
    """
    require(isinstance(history, (tuple, list)), 'Replay requires an ordered block list')
    counts, checks = [0, 0, 0, 0], []
    final = certificate_from_pairs(counts, boundary)
    previous_budget = 0
    for index, block in enumerate(history):
        number = integer(block['m'], 'block pairs', 1)
        require(len(block['pair_counts']) == 4, 'Four paired block counts required')
        fresh = [integer(value, 'block outcome count') for value in block['pair_counts']]
        require(sum(fresh) == number, 'Block outcome counts do not sum to paid pairs')
        counts = [old+new for old, new in zip(counts, fresh)]
        budget = integer(block['budget'], 'physical checkpoint', 1)
        require(budget >= previous_budget and 2*sum(counts) <= budget, 'Invalid block checkpoint ordering')
        previous_budget = budget
        if index+1 < len(history) and history[index+1]['budget'] == budget:
            continue
        require(2*sum(counts) == budget, 'Replay history ends in an incomplete physical checkpoint')
        final = certificate_from_pairs(counts, boundary)
        checks.append({'history_prefix_length': index+1, 'certificate': final})
        if final['decision'] != 'abstain':
            return {'status': 'stopped_on_observed_checkpoint', 'decision': final['decision'], 'shots': final['shots'],
                    'history_prefix_length': index+1, 'certificate': final, 'checkpoint_history': checks,
                    'new_observations': 0}
    return {'status': 'observed_prefix_exhausted_without_crossing', 'decision': 'abstain', 'shots': final['shots'],
            'history_prefix_length': len(history), 'certificate': final, 'checkpoint_history': checks,
            'new_observations': 0, 'scope': 'No stopping cost beyond this observed prefix is imputed'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('calibrate', 'verify'))
    parser.add_argument('--max-discordances', type=int, default=2048)
    parser.add_argument('--delta', default='.01')
    parser.add_argument('--threshold-scale', type=int, default=1000000)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    require(args.output is not None, 'An explicit boundary artifact path is required')
    if args.command == 'calibrate':
        result = calibrate(args.max_discordances, args.delta, args.threshold_scale)
        digest = write_artifact(args.output, result)
        print(json.dumps({'status': result['status'], 'path': str(args.output), 'file_sha256': digest,
                          'artifact_sha256': result['artifact_sha256'], 'threshold_exact': result['threshold_exact'],
                          'fair_crossing_probability': result['fair_null_risk']['probability_float_diagnostic']}))
    else:
        boundary = runtime_boundary(load_boundary(args.output))
        print(json.dumps({'status': 'exact_boundary_verified', 'artifact_sha256': boundary.artifact_sha256,
                          'max_discordances': boundary.max_discordances,
                          'fair_crossing_probability': float(boundary.crossing_probability)}))
