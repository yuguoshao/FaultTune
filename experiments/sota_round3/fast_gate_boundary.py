"""One fixed physical-prefix gate plus an exactly calibrated beta boundary.

At the first m physical pairs, an all-plus-only or all-minus-only prefix may
stop in that direction. Otherwise the fixed Jeffreys integer boundary is
used through discordance horizon H. Each artifact certifies ONE fixed m.

For independent paired Bernoulli measurements with common sign ordering,
the null-union upper bound is p_beta + 2**(-m) * P_fair(first m latent signs
all equal and beta never crosses through H). Under a strict alternative the
wrong-sign union is bounded by p_beta + 4**(-m). Both are checked exactly.
The null expression is an exact probability for a dominating latent event,
not an assertion of the actual finite-physical-budget false-stop probability.

No simulator or evaluation observations are read during calibration. No
window restarts, hindsight prefix selection, or union of several m values
is covered. The caller must bind first_m_counts to the actual first m pairs.
"""
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import argparse
import hashlib
import json

import exact_sign_boundary as beta

ROOT = Path(__file__).resolve().parents[2]
LEMMA_PATH = ROOT/'results/sota_round2/theory/physical_fast_gate_lemma.json'
VERSION = 'fixed_physical_prefix_plus_beta_v1'
_VERIFIED = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_hashes():
    result = beta.source_hashes()
    for path in (Path(__file__).resolve(), LEMMA_PATH):
        result[path.relative_to(ROOT).as_posix()] = beta.file_sha256(path)
    return result


def boundary_hash(artifact):
    core = {name: artifact[name] for name in ('version', 'max_discordances', 'prefix_pairs',
                                             'delta_exact', 'beta_threshold_exact', 'beta_boundaries')}
    return hashlib.sha256(json.dumps(core, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def restricted_surviving_paths(boundaries, prefix_pairs):
    """Count fair length-H strings whose first m signs agree and avoid beta."""
    beta.validate_table(boundaries)
    maximum = len(boundaries)-1
    m = beta.integer(prefix_pairs, 'prefix_pairs', 1)
    require(m <= maximum, 'Physical prefix exceeds registered discordance horizon')
    # Either constant-sign prefix crosses whenever the positive extreme does.
    if any(boundaries[n] <= n for n in range(1, m+1)):
        return 0
    lower, counts = 0, [1]+[0]*(m-1)+[1]
    for n in range(m+1, maximum+1):
        cutoff = boundaries[n]
        first = max(lower, n-cutoff+1, 0)
        last = min(lower+len(counts), cutoff-1, n)
        fresh = []
        for plus in range(first, last+1):
            index = plus-lower
            fresh.append((counts[index] if 0 <= index < len(counts) else 0)
                         +(counts[index-1] if 0 <= index-1 < len(counts) else 0))
        lower, counts = first, fresh
    return sum(counts)


def exact_risk_bounds(boundaries, prefix_pairs):
    m = beta.integer(prefix_pairs, 'prefix_pairs', 1)
    counts = beta.count_surviving_paths(boundaries)
    h = counts['maximum']
    restricted = restricted_surviving_paths(boundaries, m)
    require(0 <= restricted <= counts['surviving_total'] and restricted <= 1 << (h-m+1),
            'Restricted survivor count exceeds its containing sample space')
    p_numerator, p_denominator = counts['crossing_numerator'], counts['fair_denominator']
    null_numerator, null_denominator = (p_numerator << m)+restricted, 1 << (h+m)
    strict_numerator, strict_denominator = (p_numerator << (2*m))+(1 << h), 1 << (h+2*m)
    return {'beta_crossing_numerator': p_numerator, 'beta_denominator': p_denominator,
            'restricted_survivor_numerator': restricted, 'restricted_denominator': p_denominator,
            'null_union_numerator': null_numerator, 'null_union_denominator': null_denominator,
            'strict_wrong_numerator': strict_numerator, 'strict_wrong_denominator': strict_denominator}


def certified(bounds, delta):
    return all(bounds[name+'_numerator']*delta.denominator <= bounds[name+'_denominator']*delta.numerator
               for name in ('null_union', 'strict_wrong'))


def bounds_record(bounds):
    result = {name: beta.int_text(value) for name, value in bounds.items()}
    result['null_union_probability_float_diagnostic'] = bounds['null_union_numerator']/bounds['null_union_denominator']
    result['strict_wrong_probability_float_diagnostic'] = bounds['strict_wrong_numerator']/bounds['strict_wrong_denominator']
    result['beta_crossing_probability_float_diagnostic'] = bounds['beta_crossing_numerator']/bounds['beta_denominator']
    return result


def calibrate(max_discordances, prefix_pairs, delta='.01', threshold_scale=1000000):
    maximum = beta.integer(max_discordances, 'max_discordances', 1)
    m = beta.integer(prefix_pairs, 'prefix_pairs', 1)
    scale = beta.integer(threshold_scale, 'threshold_scale', 1)
    risk = beta.rational(delta, 'delta')
    require(m <= maximum and 0 < risk < 1, 'Invalid horizon, physical prefix or delta')
    residual = risk-Fraction(2, 1 << (2*m))
    require(residual > 0, 'Risk must exceed the two-sided physical-prefix envelope 2*4**(-m)')
    low, high = 0, (scale*residual.denominator+residual.numerator-1)//residual.numerator
    evaluations, cache = 0, {}
    def evaluate(index):
        nonlocal evaluations
        table = beta.constant_boundary(maximum, Fraction(index, scale))
        evaluations += 1
        if table not in cache:
            cache[table] = exact_risk_bounds(table, m)
        return table, cache[table]
    table, bounds = evaluate(high)
    require(certified(bounds, risk), 'The conservative Ville-plus-gate bracket failed exact certification')
    while high-low > 1:
        middle = (high+low)//2
        proposed, risks = evaluate(middle)
        if certified(risks, risk):
            high, table, bounds = middle, proposed, risks
        else:
            low = middle
    require(low > 0, 'Positive lower-neighbor witness unexpectedly absent')
    unused, lower_bounds = evaluate(low)
    require(not certified(lower_bounds, risk), 'Lower threshold lattice neighbor is also certified')
    result = {'version': VERSION, 'status': 'fixed_prefix_joint_risk_bounds_certified',
              'max_discordances': maximum, 'max_physical_pairs': maximum,
              'prefix_pairs': m, 'delta_exact': beta.fraction_record(risk),
              'beta_threshold_exact': beta.fraction_record(Fraction(high, scale)), 'beta_boundaries': list(table),
              'crossing_convention': 'Physical prefix gate first; otherwise beta M_D>=threshold with nonzero beta score',
              'risk_bounds_exact': bounds_record(bounds),
              'selection': {'kind': 'least_jointly_certified_positive_threshold_on_fixed_rational_lattice',
                            'threshold_scale': scale, 'chosen_lattice_index': high,
                            'lower_neighbor': {'threshold_exact': beta.fraction_record(Fraction(low, scale)),
                                               'risk_bounds_exact': bounds_record(lower_bounds)},
                            'boundary_evaluations': evaluations, 'distinct_integer_dp_evaluations': len(cache)},
              'prefix_contract': 'Exactly the actual first m physical paired outcomes; sum m, frozen once reached. Never reconstruct a favorable prefix from an aggregate block that crosses m.',
              'model_assumptions': 'Independent Bernoulli draws within each predictably chosen pair, with common sign ordering. Null discordance probability <=1/2 and each strict wrong-only category probability <=1/4.',
              'guarantee_scope': 'Each m is certified separately through D<=H. Null and strict-sign risk bounds both <=delta. No union across m, restart, arbitrary dependent pair model or infinite continuation is certified.',
              'risk_interpretation': 'Exact integer arithmetic certifies a dominating latent null event and a strict-sign union bound. These are upper bounds on actual physical stopping risk.',
              'source_hashes': source_hashes()}
    result['boundary_sha256'] = boundary_hash(result)
    result['artifact_sha256'] = beta.payload_sha256(result)
    return result


@dataclass(frozen=True)
class RuntimeBoundary:
    max_discordances: int
    prefix_pairs: int
    delta: Fraction
    beta_threshold: Fraction
    table: tuple
    boundary_sha256: str
    artifact_sha256: str


def verify_artifact(artifact):
    require(artifact['version'] == VERSION and artifact['status'] == 'fixed_prefix_joint_risk_bounds_certified',
            'Wrong physical-prefix boundary version/status')
    require(artifact['source_hashes'] == source_hashes(), 'Physical-prefix source binding changed')
    require(beta.payload_sha256(artifact) == artifact['artifact_sha256'] and
            boundary_hash(artifact) == artifact['boundary_sha256'], 'Physical-prefix artifact hash mismatch')
    maximum = beta.integer(artifact['max_discordances'], 'max_discordances', 1)
    require(artifact['max_physical_pairs'] == maximum and
            artifact['crossing_convention'] == 'Physical prefix gate first; otherwise beta M_D>=threshold with nonzero beta score',
            'Physical cap or crossing convention changed')
    m = beta.integer(artifact['prefix_pairs'], 'prefix_pairs', 1)
    require(m <= maximum, 'Prefix exceeds discordance horizon')
    delta = beta.record_fraction(artifact['delta_exact'])
    threshold = beta.record_fraction(artifact['beta_threshold_exact'])
    require(0 < delta < 1 and threshold > 0, 'Invalid exact risk or beta threshold')
    table = beta.constant_boundary(maximum, threshold)
    require(list(table) == artifact['beta_boundaries'], 'Beta table differs from exact rational threshold')
    bounds = exact_risk_bounds(table, m)
    require(bounds_record(bounds) == artifact['risk_bounds_exact'] and certified(bounds, delta), 'Joint risk certificate failed')
    selection = artifact['selection']
    require(selection['kind'] == 'least_jointly_certified_positive_threshold_on_fixed_rational_lattice', 'Unknown threshold selection')
    scale = beta.integer(selection['threshold_scale'], 'threshold_scale', 1)
    index = beta.integer(selection['chosen_lattice_index'], 'chosen_lattice_index', 2)
    require(threshold == Fraction(index, scale), 'Threshold differs from selected lattice index')
    previous = selection['lower_neighbor']
    lower = beta.record_fraction(previous['threshold_exact'])
    require(lower == Fraction(index-1, scale), 'Lower-neighbor lattice index differs')
    lower_bounds = exact_risk_bounds(beta.constant_boundary(maximum, lower), m)
    require(bounds_record(lower_bounds) == previous['risk_bounds_exact'] and not certified(lower_bounds, delta),
            'Lower-neighbor noncertification witness failed')
    runtime = RuntimeBoundary(maximum, m, delta, threshold, table, artifact['boundary_sha256'], artifact['artifact_sha256'])
    existing = _VERIFIED.setdefault(runtime.artifact_sha256, runtime)
    require(existing == runtime, 'Existing verified boundary differs')
    return existing


def runtime_boundary(boundary):
    if isinstance(boundary, RuntimeBoundary):
        require(_VERIFIED.get(boundary.artifact_sha256) is boundary, 'Runtime object must come from verification')
        return boundary
    require(isinstance(boundary, dict), 'Require an artifact dictionary or verified runtime boundary')
    runtime = _VERIFIED.get(boundary.get('artifact_sha256'))
    if runtime is None:
        runtime = verify_artifact(boundary)
    require(boundary['max_discordances'] == runtime.max_discordances and boundary['prefix_pairs'] == runtime.prefix_pairs and
            tuple(boundary['beta_boundaries']) == runtime.table and beta.record_fraction(boundary['delta_exact']) == runtime.delta and
            beta.record_fraction(boundary['beta_threshold_exact']) == runtime.beta_threshold and
            boundary['boundary_sha256'] == runtime.boundary_sha256, 'Verified decision inputs changed')
    return runtime


def pair_counts(counts):
    require(isinstance(counts, (list, tuple)) and len(counts) == 4, 'Require four paired category counts')
    return tuple(beta.integer(value, 'paired category count') for value in counts)


def certificate_from_pairs(counts, boundary, first_m_counts=None):
    """first_m_counts must be the immutable ACTUAL first-m physical prefix.

    Componentwise containment is checked but cannot establish chronology;
    the calling driver/audit must bind the prefix to its recorded history.
    Before m physical pairs use None. At and after m an exact prefix is required.
    """
    boundary = runtime_boundary(boundary)
    values = pair_counts(counts)
    pairs, plus, minus = sum(values), values[1], values[2]
    discordances, m = plus+minus, boundary.prefix_pairs
    require(pairs <= boundary.max_discordances and discordances <= boundary.max_discordances,
            'Certified physical-pair or discordance horizon exceeded')
    prefix, gate = None, 'abstain'
    if pairs < m:
        require(first_m_counts is None, 'First-m prefix cannot exist before m physical pairs')
    else:
        prefix = pair_counts(first_m_counts)
        require(sum(prefix) == m and all(before <= current for before, current in zip(prefix, values)),
                'First-m prefix does not match required length or cumulative containment')
        gate = 'reject' if prefix[1] == m else 'accept' if prefix[2] == m else 'abstain'
    cutoff = boundary.table[discordances]
    beta_decision = 'reject' if plus >= cutoff else 'accept' if plus <= discordances-cutoff else 'abstain'
    decision = gate if gate != 'abstain' else beta_decision
    return {'method': 'fixed_physical_prefix_plus_beta', 'decision': decision, 'shots': 2*pairs,
            'pairs': pairs, 'discordances': discordances, 'score': plus-minus, 'counts': list(values),
            'prefix_pairs': m, 'first_m_counts': list(prefix) if prefix is not None else None,
            'fast_gate_decision': gate, 'beta_decision': beta_decision, 'beta_integer_boundary': cutoff,
            'decision_source': 'first_m_physical_prefix' if gate != 'abstain' else 'beta_boundary' if beta_decision != 'abstain' else 'none',
            'delta_exact': beta.fraction_record(boundary.delta), 'boundary_sha256': boundary.boundary_sha256,
            'boundary_artifact_sha256': boundary.artifact_sha256, 'max_discordances': boundary.max_discordances,
            'max_physical_pairs': boundary.max_discordances,
            'finite_horizon_only': True,
            'scope': 'One fixed physical prefix and one jointly calibrated beta table; prefix provenance is the caller responsibility'}


def load_boundary(path):
    artifact = json.loads(Path(path).read_text(encoding='utf-8'))
    verify_artifact(artifact)
    return artifact


def write_artifact(path, artifact):
    verify_artifact(artifact)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(artifact, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return beta.file_sha256(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maximum', type=int, default=2048)
    parser.add_argument('--prefix-pairs', type=int, choices=(5, 6, 7), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    artifact = calibrate(args.maximum, args.prefix_pairs)
    file_hash = write_artifact(args.output, artifact)
    print(json.dumps({'status': artifact['status'], 'file_sha256': file_hash,
                      'prefix_pairs': args.prefix_pairs, 'beta_threshold_exact': artifact['beta_threshold_exact'],
                      'null_union_upper': artifact['risk_bounds_exact']['null_union_probability_float_diagnostic'],
                      'strict_wrong_upper': artifact['risk_bounds_exact']['strict_wrong_probability_float_diagnostic']}))
