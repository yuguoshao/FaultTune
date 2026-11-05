"""Independent physical-time exact checks of the randomized-prefix hybrid.

No producer passage, binomial, waiting-time or interval helper is used by the
reference recursions. Every paid physical pair is evolved with Fraction mass.
This checks operating characteristics, not the frozen policy's joint risk.
"""
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json
import math
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.sota_round5 import interval_hybrid_stopping as subject

OUT = ROOT/'results/sota_round5/theory/hybrid_stopping_interval_review.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_policy(horizon, style):
    denominator = 7
    rows = []
    for d in range(horizon+1):
        row = []
        for folded in range(d//2+1):
            if 2*folded == d or style == 'never':
                numerator = 0
            elif style == 'immediate':
                numerator = denominator
            elif style == 'only_last':
                numerator = 3 if d == horizon else 0
            else:
                numerator = (5*d+3*folded+2) % 8
            row.append(numerator)
        rows.append(tuple(row))
    return denominator, tuple(rows)


def make_boundary(cap, style):
    result = [1]
    for d in range(1, cap+1):
        if style == 'never':
            cutoff = d+1
        elif style == 'majority':
            cutoff = d//2+1
        elif style == 'unanimous':
            cutoff = d
        else:
            cutoff = d+1 if d % 3 == 1 else d//2+1
        assert d//2 < cutoff <= d+1
        result.append(cutoff)
    return result


def stop_probability(policy, boundary, d, k):
    denominator, rows = policy
    if d < len(rows):
        return F(rows[d][min(k, d-k)], denominator)
    return F(int(k >= boundary[d] or k <= d-boundary[d]))


def exact_physical_outcomes(policy, boundary, probabilities, cap):
    """Full physical-time DP, including concordances which never re-flip coins."""
    assert sum(probabilities) == 1 and all(p >= 0 for p in probabilities)
    outcomes = ((0, 0, probabilities[0]), (1, 1, probabilities[1]), (1, 0, probabilities[2]))
    states = {(0, 0): F(1)}
    plus = minus = direct_cost = survival_cost = F(0)
    for physical in range(1, cap+1):
        survival_cost += 2*sum(states.values())
        fresh = defaultdict(F)
        for (d, k), mass in states.items():
            for increment_d, increment_k, probability in outcomes:
                nd, nk = d+increment_d, k+increment_k
                stop = stop_probability(policy, boundary, nd, nk) if increment_d else F(0)
                stopped = mass*probability*stop
                if 2*nk > nd:
                    plus += stopped
                elif 2*nk < nd:
                    minus += stopped
                else:
                    assert stop == 0
                direct_cost += 2*physical*stopped
                fresh[nd, nk] += mass*probability*(1-stop)
        states = dict(fresh)
    abstain = sum(states.values())
    direct_cost += 2*cap*abstain
    assert plus+minus+abstain == 1 and direct_cost == survival_cost
    return {'plus': plus, 'minus': minus, 'abstain': abstain, 'expected_shots': direct_cost}


def exact_physical_arms(policy, boundary, pm, pp, cap):
    return exact_physical_outcomes(policy, boundary,
                                  ((1-pm)*(1-pp)+pm*pp, (1-pm)*pp, pm*(1-pp)), cap)


def exact_physical_rq(policy, boundary, r, q, cap):
    return exact_physical_outcomes(policy, boundary, (1-r, r*q, r*(1-q)), cap)


def exact_latent(policy, boundary, q, cap):
    states = {0: F(1)}
    plus, minus = [F(0)]*(cap+1), [F(0)]*(cap+1)
    for d in range(1, cap+1):
        fresh = defaultdict(F)
        for k, mass in states.items():
            for inc, probability in ((0, 1-q), (1, q)):
                nk = k+inc
                stop = stop_probability(policy, boundary, d, nk)
                if 2*nk > d:
                    plus[d] += mass*probability*stop
                elif 2*nk < d:
                    minus[d] += mass*probability*stop
                else:
                    assert stop == 0
                fresh[nk] += mass*probability*(1-stop)
        states = dict(fresh)
    survival = sum(states.values())
    assert sum(plus)+sum(minus)+survival == 1
    return plus, minus, survival


def enclosed(value, interval):
    low, high = map(float, interval)
    assert math.isfinite(low) and math.isfinite(high) and low <= high
    assert F(low) <= value <= F(high), (str(value), low, high)
    return high-low


def main():
    started = time.perf_counter()
    source = Path(subject.__file__).resolve()
    randomized = ROOT/'experiments/sota_round5/interval_randomized_stopping.py'
    basis = ROOT/'experiments/sota_round4/interval_sign_cost.py'
    randomized_review = ROOT/'results/sota_round5/theory/randomized_stopping_interval_review.json'
    basis_review = ROOT/'results/sota_round4/theory/audit_interval_sign_cost.json'
    dependencies = (source, randomized, basis, randomized_review, basis_review)
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in dependencies}
    old_random = json.loads(randomized_review.read_text(encoding='utf-8'))
    old_basis = json.loads(basis_review.read_text(encoding='utf-8'))
    assert old_random['status'] == old_basis['status'] == 'passed'
    for path, expected in old_random['inputs'].items():
        assert sha(ROOT/path) == expected
    assert old_basis['target_source_sha256'] == sha(basis)
    tiny, subnormal = F(1, 10**400), F(1, 1 << 1074)
    arms = ((F(0), F(0)), (F(1), F(1)), (F(0), F(1)), (F(1), F(0)),
            (F(1, 3), F(2, 3)), (F(1, 4), F(1, 4)), (F(0), tiny),
            (tiny, F(0)), (F(1, 2), tiny), (1-tiny, F(1)))
    rqs = ((F(0), F(2, 7)), (F(1), F(0)), (F(1), F(1)), (F(1), F(1, 2)),
           (tiny, tiny), (tiny, 1-tiny), (subnormal, subnormal),
           (F(1, 3), tiny), (F(1, 3), 1-tiny), (1-tiny, F(1, 3)))
    counts, widths = defaultdict(int), defaultdict(float)
    for horizon in (1, 2, 4):
        for cap in (1, 3, 5):
            relation = 'H_lt_T' if horizon < cap else 'H_eq_T' if horizon == cap else 'H_gt_T'
            for style in ('never', 'immediate', 'only_last', 'mixed'):
                policy = make_policy(horizon, style)
                assert subject.canonical_policy(*policy) == policy
                for boundary_style in ('never', 'majority', 'unanimous', 'zigzag'):
                    boundary = make_boundary(cap, boundary_style)
                    for pm, pp in arms:
                        reference = exact_physical_arms(policy, boundary, pm, pp, cap)
                        observed = subject.operating_intervals(policy, boundary, pm, pp, cap)
                        assert observed['exact_arm_probabilities'] == [str(pm), str(pp)]
                        for name, value in reference.items():
                            widths[name] = max(widths[name], enclosed(value, observed[name]))
                        counts['physical_time_fraction_arm_cases'] += 1
                        counts[relation] += 1
                    for q in (F(0), F(1), F(1, 2), F(2, 7), tiny, 1-tiny):
                        p, m, survivor = exact_latent(policy, boundary, q, cap)
                        observed = subject.first_passage_intervals(policy, boundary, q, cap)
                        for name, probabilities in (('plus', p), ('minus', m)):
                            for d, value in enumerate(probabilities):
                                enclosed(value, (observed[name][0][d], observed[name][1][d]))
                        enclosed(survivor, observed['survival'])
                        counts['latent_fraction_first_passage_cases'] += 1
            # Independent marked-arrival cases stress both tiny r and tiny q,
            # without assuming every r,q pair has independent-arm realization.
            policy, boundary = make_policy(horizon, 'mixed'), make_boundary(cap, 'zigzag')
            for r, q in rqs:
                reference = exact_physical_rq(policy, boundary, r, q, cap)
                observed = subject.operating_rq(policy, boundary, r, q, cap)
                for name, value in reference.items():
                    widths[name] = max(widths[name], enclosed(value, observed[name]))
                counts['physical_time_fraction_rq_cases'] += 1
    # Explicit witnesses pinpoint the H / H+1 transition and retained mass.
    witnesses = []
    policy = make_policy(2, 'only_last')
    boundary = make_boundary(5, 'majority')
    reference = exact_physical_rq(policy, boundary, F(1), F(1), 5)
    assert reference == {'plus': F(1), 'minus': F(0), 'abstain': F(0), 'expected_shots': F(36, 7)}
    passage = exact_latent(policy, boundary, F(1), 5)
    assert passage[0][2] == F(3, 7) and passage[0][3] == F(4, 7)
    observed = subject.operating_rq(policy, boundary, F(1), F(1), 5)
    for name, value in reference.items():
        enclosed(value, observed[name])
    witnesses.append({'name': 'H2_partial_stop_then_H3_fallback', 'exact': {k: str(v) for k, v in reference.items()},
                      'at_H_stop_mass': '3/7', 'at_Hplus1_stop_mass': '4/7',
                      'point': 'Fallback at H would incorrectly stop all paths at cost4; correct cost is36/7.'})
    policy = make_policy(1, 'never')
    reference = exact_physical_rq(policy, boundary, F(1), F(2, 3), 5)
    assert reference['expected_shots'] == F(44, 9) and reference['abstain'] == 0
    observed = subject.operating_rq(policy, boundary, F(1), F(2, 3), 5)
    for name, value in reference.items():
        enclosed(value, observed[name])
    witnesses.append({'name': 'fallback_origin_shift', 'exact': {k: str(v) for k, v in reference.items()},
                      'point': 'After d2 only absolute count k1 survives with mass4/9; d3 must use counts1,2, not counts0,1.'})
    policy = make_policy(1, 'immediate')
    reference = exact_physical_rq(policy, boundary, F(1), F(2, 3), 5)
    assert reference == {'plus': F(2, 3), 'minus': F(1, 3), 'abstain': F(0), 'expected_shots': F(2)}
    observed = subject.operating_rq(policy, boundary, F(1), F(2, 3), 5)
    for name, value in reference.items():
        enclosed(value, observed[name])
    witnesses.append({'name': 'prefix_absorption_is_final', 'exact': {k: str(v) for k, v in reference.items()},
                      'point': 'Every path stops at d1; fallback may not recycle its mass or charge later pairs.'})
    for path, expected in hashes.items():
        assert sha(ROOT/path) == expected
    result = {
        'status': 'passed', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'reviewer_source': Path(__file__).resolve().relative_to(ROOT).as_posix(),
        'reviewer_sha256': sha(Path(__file__).resolve()),
        'input_sha256': hashes, 'new_simulation_observations': 0,
        'counts': dict(counts), 'maximum_small_case_interval_widths': dict(widths),
        'explicit_phase_transition_witnesses': witnesses,
        'mathematical_review': [
            'At d<=H, the strict-majority randomized row is used once, and surviving mass is multiplied by1-stop probability. No fallback is called atH. At d=H+1 and later, the frozen boundary is applied to the total discordance count and total plus count, including the whole prefix.',
            'The randomized phase retains the full count vector with origin0, so its stop-probability arrays have matching lengths. After the phase change, tail absorption leaves a contiguous interval. Updating origin to the first surviving absolute count correctly indexes subsequent transitions; the method never returns to the randomized phase.',
            'Previously absorbed randomized mass cannot enter fallback. Rounding may leave tiny upper survival bounds after exact mass is zero, which remains a conservative enclosure and does not resurrect actual paths.',
            'WhenH>=T, the physical cap prevents reaching fallback; the randomized rule is truncated atT, with all unreported mass paying2T. WhenH<T, every latent path survivingH continues into the boundary phase. Boundary validation covers every discordance throughT, which suffices because at mostT discordances occur inT pairs.',
            'For iid stationary independent Bernoulli arms, marked discordance arrivals have rate r and independent iid signs q. The combined rule depends only on signs and fresh independent prefix stop coins, so its first passage is independent of waiting times. The already audited binomial arrival tails and truncated-negative-binomial identity apply to the entire hybrid first-passage law.',
            'Physical cost includes all paid pairs before either report and the full cap for every abstaining path. Both the direct stopped-cost calculation and the sum of survival probabilities before each paid pair agree exactly in the independent Fraction DP.',
            'Degenerate r=0 gives no reports and full-cap cost. r=1, q=0/1, below-subnormal positive r/q, least subnormal, and near-one probabilities pass exact checks. Positive r whose lower endpoint is0 safely usesT as the wait upper bound.'
        ],
        'independence': {
            'physical_dp': 'Reference state is physical time plus(discordances,plus_count); transitions enumerate concordance/plus/minus probabilities with exact Fraction arithmetic. It imports no producer waiting, binomial, passage, or interval arithmetic helper.',
            'latent_dp': 'Separate full-dictionary count recursion checks every plus/minus first-passage component and final survivor mass, without origin compression.',
            'arm_models': '10 stationary independent-arm models per configuration, including interior, fair, degenerate and below-subnormal probabilities.',
            'rq_models': '10 additional iid marked-arrival laws per horizon/cap test tiny r and q together; these check operating_rq, without claiming every r,q law is realizable by independent Bernoulli arms.',
            'configurations': 'H=1,2,4; T=1,3,5; 4 randomized prefixes and 4 deterministic symmetric fallback shapes.'
        },
        'interface_scope': 'Pass canonical immutable policy, a valid symmetric integer boundary covering the positive physical cap, and independent fresh prefix coins once per discordance. Continuous scanning occurs only at new discordances.',
        'blocking_findings': [],
        'limits': [
            'This audit verifies hybrid operating characteristics and the phase transition. It does not certify any selected production table joint error probability.',
            'Individual prefix and fallback risk guarantees do not automatically combine into one1% guarantee. The entire combined first-report rule requires a separate joint risk certificate.',
            'No current production policy performance, >=95% power, SOTA superiority, or research-goal completion is claimed.',
            'Finite exact cases and source review are not formal verification of every possible input, IEEE-754, Python, or numpy implementation.'
        ],
        'elapsed_seconds': time.perf_counter()-started}
    with OUT.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': 'passed', 'sha256': sha(OUT), 'counts': dict(counts),
                      'widths': dict(widths), 'elapsed_seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
