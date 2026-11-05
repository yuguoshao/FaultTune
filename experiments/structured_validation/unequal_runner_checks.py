"""Deterministic capacity/policy and weighted-evidence checks for the runner."""
import hashlib
import json
import math
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import numpy as np

import unequal_runner as runner
import weighted_fisher as fisher
from expert_config import expert_config


def partition(nminus, nplus, total, odds):
    return sum((Fraction(math.comb(nplus, k)*math.comb(nminus, total-k))*odds**k
                for k in range(max(0, total-nminus), min(nplus, total)+1)), Fraction(0))


def run():
    pairs = [(a, b) for a in (.001, .01, .1, .3, .5, .9, .999)
             for b in (.001, .01, .1, .3, .5, .9, .999)]
    pm, pp = np.array(pairs).T
    weight, rate = runner.two_arm_capacity(pm, pp)
    reverse_weight, reverse_rate = runner.two_arm_capacity(pp, pm)
    assert np.max(np.abs(weight+reverse_weight-1)) < 1e-10
    assert np.max(np.abs(rate-reverse_rate)) < 1e-12
    dense = np.linspace(0, 1, 2001)
    for j, (minus, plus) in enumerate(pairs):
        center = (1-dense)*minus+dense*plus
        grid_rate = (1-dense)*runner.bernoulli_kl(minus, center)+dense*runner.bernoulli_kl(plus, center)
        assert rate[j] >= grid_rate.max()-1e-12
    fixtures = []
    for theta in ((.1, .085, .02), (.3, -.18, .12), (.5, .1, -.2), (.9, .08, .01)):
        rows = []
        for arm in (runner.rational_circle(1, 1), runner.rational_circle(-1, 1),
                    runner.rational_circle(4, 7), runner.rational_circle(7, 4)):
            c, s = arm["cos_numerator"]/arm["denominator"], arm["sin_numerator"]/arm["denominator"]
            trials = 100000
            row = dict(arm, trials=trials, successes=int(round(trials*(theta[0]+theta[1]*c+theta[2]*s))))
            rows.append(row)
        for policy in runner.POLICIES:
            plan = runner.plan_from_rows(policy, rows)
            arm = plan["positive_arm"]
            assert arm["sin_numerator"] > 0
            assert arm["cos_numerator"]**2+arm["sin_numerator"]**2 == arm["denominator"]**2
            assert plan["fit_shots"] == 400000 and 0 < plan["weight_plus"] < 1
            assert plan["pair_capacity_rate"] >= plan["fit_information_rate"]-1e-10
            if policy == "grid" and not plan["pair_fallback"]:
                assert len(plan["full_grid_design"]["support"]) == 2
                assert runner._exact_pair_from_support(plan["full_grid_design"]) is not None
            fixtures.append({"policy": policy, "fit_theta": plan["fit_theta"],
                             "weight_plus": plan["weight_plus"], "pair_fallback": plan["pair_fallback"]})
    assert runner._exact_pair_from_support({"support": [{"arm": 0, "weight": .5}, {"arm": 16, "weight": .5}]}) is None
    assert runner._exact_pair_from_support({"support": [{"arm": 16, "weight": .49}, {"arm": 48, "weight": .49}, {"arm": 0, "weight": .02}]}) is None
    allocation_checks = 0
    for total in range(0, 65):
        for weight in (0., .001, .2, .4999, .5, .9, 1.):
            nm, np_ = runner.allocation(total, weight)
            assert nm+np_ == total and min(nm, np_) >= 0
            if total >= 2:
                assert min(nm, np_) >= 1
            allocation_checks += 1
    # The common generic 71-expert weighted implementation must enclose
    # its exact rational mixture, including unequal and degenerate blocks.
    blocks = [dict(nminus=3, nplus=5, minus_successes=1, plus_successes=4),
              dict(nminus=7, nplus=2, minus_successes=3, plus_successes=1),
              dict(nminus=0, nplus=3, minus_successes=0, plus_successes=2)]
    exact, state = Fraction(0), fisher.empty_state()
    for b in blocks:
        fisher.update_block(state, b["nminus"], b["nplus"], b["minus_successes"], b["plus_successes"])
    for expert in expert_config():
        odds, likelihood = Fraction(expert["numerator"], expert["denominator"]), Fraction(1)
        for b in blocks:
            total, k = b["minus_successes"]+b["plus_successes"], b["plus_successes"]
            likelihood *= odds**k*math.comb(b["nminus"]+b["nplus"], total)/partition(b["nminus"], b["nplus"], total, odds)
        exact += Fraction(*expert["weight"])*likelihood
    cert = fisher.certificate_from_blocks(blocks)
    with localcontext() as context:
        context.prec = 110
        log_exact = Decimal(exact.numerator).ln()-Decimal(exact.denominator).ln()
        assert Decimal(cert["evidence_bounds"][0]) <= log_exact <= Decimal(cert["evidence_bounds"][1])
    assert Fraction(cert["score_numerator"], cert["score_denominator"]) == state["score"]
    assert abs(fisher.screen(state)["log_evidence"]-float(log_exact)) < 1e-11
    sources = runner.source_hashes()
    sources[Path(__file__).resolve().relative_to(runner.ROOT).as_posix()] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result = {"status": "passed", "capacity_pairs_against_dense_grid": len(pairs),
              "capacity_reflection_symmetry": True, "policy_fixtures": fixtures,
              "allocation_count_checks": allocation_checks, "nonpair_support_not_silently_executed": True,
              "weighted71_exact_mixture_enclosure": True, "weighted71_exact_score": True,
              "sources": sources, "monte_carlo": False}
    path = runner.ROOT/"results/structured_validation/unequal_runner_checks.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("policy_fixtures", "sources")}, indent=2))


if __name__ == "__main__":
    run()
