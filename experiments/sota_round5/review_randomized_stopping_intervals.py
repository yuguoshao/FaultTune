"""Independent exact physical-time DP checks for randomized stopping intervals."""
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.sota_round5 import interval_randomized_stopping as subject

OUT = ROOT/'results/sota_round5/theory/randomized_stopping_interval_review.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_physical(policy, pm, pp, cap):
    """Independent physical-time DP; all surviving paths pay through cap.

Each discordance gets exactly one independent stop coin. Concordant outcomes
never re-flip the coin, including after the last available policy row.
"""
    denominator, rows = policy
    horizon = len(rows)-1
    probabilities = (((0, 0), (1-pm)*(1-pp)+pm*pp),
                     ((1, 1), (1-pm)*pp), ((1, 0), pm*(1-pp)))
    states = {(0, 0): F(1)}
    plus = minus = direct_cost = survival_cost = F(0)
    for physical in range(1, cap+1):
        survival_cost += 2*sum(states.values())
        fresh = defaultdict(F)
        for (d, k), mass in states.items():
            for (inc_d, inc_k), probability in probabilities:
                nd, nk = d+inc_d, k+inc_k
                stopping = F(rows[nd][min(nk, nd-nk)], denominator) if inc_d and nd <= horizon else F(0)
                stopped = mass*probability*stopping
                if 2*nk > nd:
                    plus += stopped
                elif 2*nk < nd:
                    minus += stopped
                else:
                    assert stopping == 0
                direct_cost += 2*physical*stopped
                fresh[nd, nk] += mass*probability*(1-stopping)
        states = dict(fresh)
    abstain = sum(states.values())
    direct_cost += 2*cap*abstain
    assert plus+minus+abstain == 1 and direct_cost == survival_cost
    return {'plus': plus, 'minus': minus, 'abstain': abstain, 'expected_shots': direct_cost}


def exact_sign_passage(policy, q, steps):
    denominator, rows = policy
    states = {0: F(1)}
    plus, minus = [F(0)]*(steps+1), [F(0)]*(steps+1)
    for d in range(1, steps+1):
        fresh = defaultdict(F)
        for k, mass in states.items():
            for inc, probability in ((0, 1-q), (1, q)):
                nk = k+inc
                stop = F(rows[d][min(nk, d-nk)], denominator)
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


def make_policy(horizon, style):
    denominator = 7
    rows = []
    for d in range(horizon+1):
        row = []
        for folded in range(d//2+1):
            if d == 2*folded:
                value = 0
            elif style == 'never':
                value = 0
            elif style == 'immediate':
                value = denominator
            elif style == 'only_last':
                value = 3 if d == horizon else 0
            else:
                value = (5*d+3*folded+2) % (denominator+1)
            row.append(value)
        rows.append(row)
    return subject.canonical_policy(denominator, rows)


def enclosed(exact, lower, upper):
    lower, upper = float(lower), float(upper)
    assert math.isfinite(lower) and math.isfinite(upper) and lower <= upper
    assert F(lower) <= exact <= F(upper), (str(exact), lower, upper)
    return upper-lower


def main():
    source = Path(subject.__file__).resolve()
    basis = ROOT/'experiments/sota_round4/interval_sign_cost.py'
    basis_audit = ROOT/'results/sota_round4/theory/audit_interval_sign_cost.json'
    before = {p.relative_to(ROOT).as_posix(): sha(p) for p in (source, basis, basis_audit)}
    assert json.loads(basis_audit.read_text())['status'] == 'passed'
    tiny, subnormal = F(1, 10**400), F(1, 1 << 1074)
    models = [(F(0), F(0)), (F(1), F(1)), (F(0), F(1)), (F(1), F(0)),
              (F(1, 3), F(2, 3)), (F(1, 4), F(1, 4)), (F(1, 10), F(1, 5)),
              (F(0), tiny), (tiny, F(0)), (F(0), subnormal),
              (F(1, 2), tiny), (tiny, F(1, 2)), (1-tiny, F(1)), (1-tiny, 1-tiny)]
    policies = [make_policy(h, style) for h in (1, 2, 3, 5, 8)
                for style in ('never', 'immediate', 'only_last', 'mixed')]
    counts = defaultdict(int)
    widths = defaultdict(float)
    witnesses = []
    for policy in policies:
        h = len(policy[1])-1
        for cap in (1, 2, 3, 5, 8):
            for pm, pp in models:
                exact = exact_physical(policy, pm, pp, cap)
                observed = subject.operating_intervals(policy, pm, pp, cap)
                for name in exact:
                    widths[name] = max(widths[name], enclosed(exact[name], *observed[name]))
                counts['physical_time_fraction_dp_cases'] += 1
                counts['H_lt_cap' if h < cap else 'H_eq_cap' if h == cap else 'H_gt_cap'] += 1
                if len(witnesses) < 8 and pm == F(1, 3) and cap in (1, 5):
                    witnesses.append({'policy_horizon': h, 'cap': cap,
                                      'policy': [policy[0], [list(row) for row in policy[1]]],
                                      'pm': str(pm), 'pp': str(pp),
                                      'exact': {k: str(v) for k, v in exact.items()},
                                      'enclosures': {k: observed[k] for k in exact}})
        for q in (F(0), F(1), F(1, 2), F(2, 7), tiny, 1-tiny):
            for steps in sorted({0, 1, min(3, h), h}):
                exact_plus, exact_minus, exact_survival = exact_sign_passage(policy, q, steps)
                got = subject.policy_passage_intervals(policy, q, steps)
                for name, reference in (('plus', exact_plus), ('minus', exact_minus)):
                    for d, probability in enumerate(reference):
                        enclosed(probability, got[name][0][d], got[name][1][d])
                enclosed(exact_survival, *got['survival'])
                counts['latent_sign_fraction_dp_cases'] += 1
    # Independently verify the NB truncated-wait identity at small caps,
    # including d=T and d=1, rather than sharing any binomial helper.
    for cap in (1, 2, 3, 5, 8):
        for r in (F(1), F(1, 2), F(1, 7), tiny):
            for d in range(1, cap+1):
                arrival = sum(F(t)*math.comb(t-1, d-1)*r**d*(1-r)**(t-d) for t in range(d, cap+1))
                missed = sum(math.comb(cap, k)*r**k*(1-r)**(cap-k) for k in range(d))
                direct = arrival+cap*missed
                next_tail = sum(math.comb(cap+1, k)*r**k*(1-r)**(cap+1-k) for k in range(d+1, cap+2))
                assert direct == F(d, 1)/r*next_tail+cap*missed
                counts['negative_binomial_index_identities'] += 1
    # Common production cap with below-subnormal positive probabilities.
    # Union bound: chance of any discordance <=T*r; thus a no-report path
    # alone guarantees cost >=2T*(1-T*r), while cap gives cost <=2T.
    large_cap = 8192
    policy = make_policy(128, 'mixed')
    for pm, pp in ((F(0), tiny), (tiny, F(0)), (1-tiny, F(1))):
        got = subject.operating_intervals(policy, pm, pp, large_cap)
        r = pm*(1-pp)+pp*(1-pm)
        assert F(got['expected_shots'][0]) <= 2*large_cap*(1-large_cap*r)
        assert F(got['expected_shots'][1]) >= 2*large_cap
        assert all(math.isfinite(x) for interval in (got['plus'], got['minus'], got['abstain'], got['expected_shots']) for x in interval)
        counts['cap8192_tiny_probability_bound_checks'] += 1
    for name, value in before.items():
        assert sha(ROOT/name) == value, 'Reviewed source changed during checks'
    result = {'status': 'passed', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'reviewer_source': Path(__file__).resolve().relative_to(ROOT).as_posix(),
              'reviewer_sha256': sha(Path(__file__).resolve()), 'inputs': before,
              'counts': dict(counts), 'maximum_observed_interval_widths_small_cases': dict(widths),
              'examples': witnesses, 'new_simulation_observations': 0,
              'mathematical_review': [
                  'The latent sign DP first transitions withq/1-q, reports with the frozen rational stop probability, then keeps the complement. The sign is the strict majority; ties have zero report probability. Independent fresh decision coins are used once per discordance.',
                  'For stationary independent Bernoulli arms, discordance arrivals have rate r and conditional signs are iidBernoulli(q), independent of arrival waiting times and fresh policy randomization.',
                  'LetJ be the latent first-report discordance or infinity. P(report+ before capT)=sum_d P(J=d,+)*P(Bin(T,r)>=d), and similarly forminus.',
                  'E[min(NB(d,r),T)]=(d/r)*P(Bin(T+1,r)>=d+1)+T*P(Bin(T,r)<=d-1). Therefore the code tail slice starting2 and head slice starting0 have the correct d+1 and d-1 indices.',
                  'The common latent horizon is min(H,T). IfH<T, surviving pastH means never reporting later and every such path pays2T. IfH>T, no physical path can reach more thanT discordances, so truncating the latent policy atT and charging surviving mass2T is exact.',
                  'Expected cost includes all report directions and all cap-abstention paths. No cheap abstention atH is credited.',
                  'The r=0 branch returns full cap and no reports. q=0/1 andr=1 are correctly covered. For positive r rounding below the least subnormal, the full cap is a valid waiting-time upper bound; dividing tail by r before multiplyingd avoids the d/r overflow pitfall.'],
              'test_design': {'physical_dp': 'Exact Fraction state mass indexed by physical time and(discordances,plus signs). Every concordance leaves the policy coin untouched; afterH, all surviving mass keeps paying until cap.',
                              'probabilities': 'r=0, r=1, q=0, q=1, symmetric/interior arms, least subnormal, below-subnormal positive probabilities, and near-one rational probabilities.',
                              'policy_types': 'Never-report, immediate-report, randomized last-row-only, and nonmonotone randomized folded majority rules, with horizons1/2/3/5/8.',
                              'independence': 'Exact physical-time and latent-sign Fraction DPs use neither binomial_intervals nor any passage/output helper from the producer. Subject helpers are called only to obtain compared intervals.'},
              'interface_scope': 'Pass the immutable result of canonical_policy as the policy key, with independent fresh stop coins once per discordance. The helper does not prove that a supplied policy has1% risk.',
              'limitations': ['Validity of operating characteristics assumes stationary independent Bernoulli pairs; adaptive angle/action changes are outside this probability factorization.',
                              'Finite exact examples audit implementation and indexing; they are not a formal proof of IEEE-754 or exhaustive coverage of arbitrary horizons.',
                              'This is a probability/cost enclosure review, not a policy-risk proof, simulation result,95%-power claim, or SOTA acceptance.']}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'status': 'passed', 'sha256': sha(OUT), 'counts': dict(counts), 'widths': dict(widths)}), flush=True)


if __name__ == '__main__':
    main()
