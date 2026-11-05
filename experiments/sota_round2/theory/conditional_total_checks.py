"""Exact tests of current-total selection, adaptive designs and the new API."""
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import json
from math import comb
from pathlib import Path

import conditional_total as ct


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/sota_round2/theory/conditional_total_checks.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def odds_from_total(m, total, past_gap=F(1, 10)):
    q = F(2*total+1, 4*m+2)
    d = min(past_gap, q/2, (1-q)/2)
    u, v = q-d, q+d
    return v*(1-u)/(u*(1-v))


def bprob(n, k, p):
    return comb(n, k)*p**k*(1-p)**(n-k)


def expectation_checks():
    checks = 0
    conditional_normalizations = 0
    probability_grid = [F(k, 4) for k in range(5)]
    for m in range(1, 5):
        factors = {}
        for km in range(m+1):
            for kp in range(m+1):
                total = km+kp
                factors[km, kp] = ct.exact_block_factors(dict(
                    m=m, total_successes=total, plus_successes=kp,
                    odds=odds_from_total(m, total)))
        for total in range(2*m+1):
            ep = en = F(0)
            for kp in range(max(0, total-m), min(m, total)+1):
                mass = F(comb(m, kp)*comb(m, total-kp), comb(2*m, total))
                p, n = factors[total-kp, kp]
                ep, en = ep+mass*p, en+mass*n
            assert ep == en == 1
            conditional_normalizations += 1
        for pm in probability_grid:
            for pp in probability_grid:
                ep = en = F(0)
                for (km, kp), (positive, negative) in factors.items():
                    mass = bprob(m, km, pm)*bprob(m, kp, pp)
                    ep, en = ep+mass*positive, en+mass*negative
                if pm >= pp:
                    assert ep <= 1
                    checks += 1
                if pm <= pp:
                    assert en <= 1
                    checks += 1
    return {"one_sided_unconditional_expectation_checks": checks,
            "exact_current_total_conditional_normalizations": conditional_normalizations}


def allocation_leak_counterexample():
    low = ct.exact_block_factors(dict(m=1, total_successes=1, plus_successes=0, odds=F(1)))[0]
    high = ct.exact_block_factors(dict(m=1, total_successes=1, plus_successes=1, odds=F(9)))[0]
    expectation = (low+high)/2
    assert expectation == F(7, 5)
    return {"m": 1, "T": 1, "null_conditional_allocation": "S=0 or 1 with equal probability",
            "invalid_rule": "Omega=1 if S=0, Omega=9 if S=1",
            "conditional_mean": str(expectation),
            "conclusion": "Current total is legal; allocation-dependent same-block odds are not"}


def certified_arithmetic_checks():
    checks = 0
    for m in range(1, 5):
        for total in range(2*m+1):
            odds = odds_from_total(m, total)
            for plus in range(max(0, total-m), min(m, total)+1):
                block = dict(m=m, total_successes=total, plus_successes=plus, odds=odds)
                exact = ct.exact_block_factors(block)
                positive, negative, diagnostics = ct.log_capital_bounds([block])
                for bound, rational in zip((positive, negative), exact):
                    reference = (ct.Bounds(rational.numerator)/rational.denominator).log()
                    assert max(bound.lo, reference.lo) <= min(bound.hi, reference.hi)
                    checks += 1
    histories = [[dict(m=8, total_successes=8, plus_successes=8, odds=F(o))] for o in (9, 81)]
    cert = ct.certificate_from_histories(histories)
    assert cert["decision"] == "reject"
    reversed_histories = [[dict(m=8, total_successes=8, plus_successes=0, odds=F(o))] for o in (9, 81)]
    reverse = ct.certificate_from_histories(reversed_histories)
    assert reverse["decision"] == "accept"
    assert cert["evidence_bounds"] == reverse["evidence_bounds"]
    mismatch_rejected = False
    try:
        ct.mixture_log_capital_bounds(histories[:1]+reversed_histories[:1])
    except ValueError:
        mismatch_rejected = True
    assert mismatch_rejected
    return {"direct_rational_log_interval_checks": checks,
            "positive_mixture_certificate": cert,
            "mirror_certificate_matches": True,
            "mismatched_expert_observations_rejected": True}


def adaptive_design_enumeration():
    """Exact four-block physical-model tree with past-adaptive angle AND m."""
    results = []
    horizon, delta = 4, F(1, 4)
    for true_c in (F(-1, 5), F(0), F(1, 5)):
        true_a, true_b = F(1, 2), F(1, 10)
        nodes, risk, mass = 0, F(0), F(0)
        # n denotes past pairs, not block count; two fixed shrinkage experts.
        stack = [(0, 0, 0, 0, (F(1), F(1)), (F(1), F(1)), F(1))]
        while stack:
            depth, n, km, kp, ep, en, probability = stack.pop()
            nodes += 1
            positive, negative = sum(ep)/2, sum(en)/2
            decision = 0
            if (positive+negative)/2 >= 1/delta:
                decision = 1 if positive > negative else -1 if negative > positive else 0
            if decision or depth == horizon:
                mass += probability
                if decision and (true_c == 0 or decision*true_c < 0):
                    risk += probability
                continue
            co, si = (F(0), F(1)) if kp >= km else (F(4, 5), F(3, 5))
            m = 1+(km+kp) % 2
            pm, pp = true_a+true_b*co-true_c*si, true_a+true_b*co+true_c*si
            # Magnitude is past-measurable; a fixed cold-start pseudogap is legal.
            gap = F(abs(kp-km)+1, 4*n+10)
            for minus in range(m+1):
                for plus in range(m+1):
                    total = minus+plus
                    factors = [ct.exact_block_factors(dict(m=m, total_successes=total,
                               plus_successes=plus, odds=odds_from_total(m, total, gap*k)))
                               for k in (F(1, 2), F(1))]
                    p = tuple(a*f[0] for a, f in zip(ep, factors))
                    q = tuple(a*f[1] for a, f in zip(en, factors))
                    prob = probability*bprob(m, minus, pm)*bprob(m, plus, pp)
                    stack.append((depth+1, n+m, km+minus, kp+plus, p, q, prob))
        assert mass == 1 and risk <= delta
        results.append({"A": str(true_a), "B": str(true_b), "C": str(true_c),
                        "tree_nodes": nodes, "wrong_risk": str(risk), "wrong_risk_float": float(risk)})
    return {"horizon_blocks": horizon, "delta": str(delta), "models": results,
            "design": "past chooses reflected rational angle and m in {1,2}; current T chooses odds; fixed mixture of two past-gap shrinkages"}


def main():
    if OUT.exists():
        raise RuntimeError("Preserve existing result")
    checks = expectation_checks()
    checks["allocation_leak_counterexample"] = allocation_leak_counterexample()
    checks["certified_arithmetic"] = certified_arithmetic_checks()
    checks["adaptive_design"] = adaptive_design_enumeration()
    paths = [Path(__file__).resolve(), Path(ct.__file__).resolve(),
             Path(ct.frozen.__file__).resolve(), ROOT/"experiments/circuit_validation/direct_tests.py",
             ROOT/"experiments/sota_round2/theory/directional_safe.py"]
    record = {"status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(),
              "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in paths},
              "checks": checks,
              "scope": "Exact finite normalization, MLR examples, adaptive physical-model trees, numerical certification and an invalid allocation-leak counterexample; no Monte Carlo or superiority claim"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("x", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    print(json.dumps({"status": "passed", "output": str(OUT), "sha256": digest(OUT),
                      "expectation_checks": checks["one_sided_unconditional_expectation_checks"],
                      "conditional_normalizations": checks["exact_current_total_conditional_normalizations"],
                      "interval_checks": checks["certified_arithmetic"]["direct_rational_log_interval_checks"],
                      "adaptive_models": checks["adaptive_design"]["models"]}, indent=2))


if __name__ == "__main__":
    main()
