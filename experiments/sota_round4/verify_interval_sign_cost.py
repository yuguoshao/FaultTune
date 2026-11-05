"""Independent exact finite-word checks of interval operating characteristics."""
from fractions import Fraction as F
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.sota_round4.interval_sign_cost import (
    binomial_intervals, operating_intervals, sign_first_passage_intervals)


def contains(interval, exact):
    lo, hi = map(float, interval)
    assert math.isfinite(lo) and math.isfinite(hi), (interval, str(exact))
    assert F(lo) <= exact <= F(hi), (interval, str(exact))


def physical_words(table, pm, pp):
    """Enumerate full physical outcome words, without NB/sign factorization."""
    cap = len(table)-1
    rates = ((1-pm)*(1-pp), (1-pm)*pp, pm*(1-pp), pm*pp)
    result = dict(plus=F(0), minus=F(0), abstain=F(0), expected_shots=F(0))
    for word in itertools.product(range(4), repeat=cap):
        probability = F(1)
        for outcome in word:
            probability *= rates[outcome]
        if probability == 0:
            continue
        discordances, positives, paid = 0, 0, 2*cap
        decision = 'abstain'
        for t, outcome in enumerate(word, 1):
            if outcome not in (1, 2):
                continue
            discordances += 1
            positives += int(outcome == 1)
            cutoff = table[discordances]
            if positives >= cutoff:
                decision, paid = 'plus', 2*t
                break
            if positives <= discordances-cutoff:
                decision, paid = 'minus', 2*t
                break
        result[decision] += probability
        result['expected_shots'] += probability*paid
    return result


def sign_words(table, q):
    cap = len(table)-1
    plus, minus, survival = [F(0)]*(cap+1), [F(0)]*(cap+1), F(0)
    for word in itertools.product((0, 1), repeat=cap):
        probability = q**sum(word)*(1-q)**(cap-sum(word))
        positives = 0
        for d, sign in enumerate(word, 1):
            positives += sign
            if positives >= table[d]:
                plus[d] += probability
                break
            if positives <= d-table[d]:
                minus[d] += probability
                break
        else:
            survival += probability
    return plus, minus, survival


def main():
    checks = dict(binomial_entries=0, sign_boundaries=0, physical_boundaries=0,
                  degenerate_and_extreme_models=0)
    probabilities = [F(0), F(1), F(1, 2), F(1, 3), F(7, 11), F(.1),
                     F(float(np.nextafter(0., 1.))), 1-F(1, 2**1074)]
    for n in range(17):
        for p in probabilities:
            intervals = binomial_intervals(n, p)
            masses = [F(math.comb(n, k))*p**k*(1-p)**(n-k) for k in range(n+1)]
            head = F(0)
            for k, mass in enumerate(masses):
                head += mass
                for offset, exact in ((0, mass), (2, head), (4, sum(masses[k:], F(0)))):
                    contains((intervals[offset][k], intervals[offset+1][k]), exact)
                    checks['binomial_entries'] += 1
    models = [(F(0), F(0)), (F(1), F(1)), (F(0), F(1)), (F(1), F(0)),
              (F(1, 3), F(2, 3)), (F(1, 5), F(3, 7)), (F(3, 5), F(1, 4))]
    for cap in range(1, 5):
        for cutoffs in itertools.product(*(range(d//2+1, d+2) for d in range(1, cap+1))):
            table = [1]+list(cutoffs)
            for q in (F(0), F(1), F(1, 3), F(1, 2), F(4, 5)):
                enclosed = sign_first_passage_intervals(table, q, cap)
                plus, minus, survival = sign_words(table, q)
                for name, masses in (('plus', plus), ('minus', minus)):
                    for d, mass in enumerate(masses):
                        contains((enclosed[name][0][d], enclosed[name][1][d]), mass)
                contains(enclosed['survival'], survival)
                checks['sign_boundaries'] += 1
            for pm, pp in models:
                exact = physical_words(table, pm, pp)
                enclosed = operating_intervals(table, pm, pp)
                for name, value in exact.items():
                    contains(enclosed[name], value)
                checks['physical_boundaries'] += 1
    for pm, pp in ((F(0), F(1, 2**1074)), (F(0), F(1, 2**1075)),
                   (F(1)-F(1, 2**1074), F(1)), (F(.1), F(.3))):
        for table in ([1, 1, 2, 2], [1, 2, 3, 3], [1, 2, 2, 4]):
            exact = physical_words(table, pm, pp)
            enclosed = operating_intervals(table, pm, pp)
            for name, value in exact.items():
                contains(enclosed[name], value)
            checks['degenerate_and_extreme_models'] += 1
    path = ROOT/'results/sota_round4/interval_sign_cost_exact_checks.json'
    sources = [Path(__file__).resolve(), ROOT/'experiments/sota_round4/interval_sign_cost.py']
    result = {'status': 'passed', 'new_simulation_observations': 0, 'checks': checks,
              'scope': 'Exact finite rational physical/sign words and binomial pmf/head/tail; scientific numerical correctness checks, not empirical validation.',
              'source_sha256': {str(s.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(s.read_bytes()).hexdigest() for s in sources}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
