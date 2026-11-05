"""Exact finite checks and a deterministic obstruction to pooled Fisher.

No simulator draws are generated. Finite paths are exhaustively enumerated,
and all risks, normalization checks and counterexamples use Fraction.
"""
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import json
from math import comb, log
from pathlib import Path

import directional_safe as safe


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/sota_round2/theory/directional_safe_checks.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encoded(value):
    return {"fraction": [value.numerator, value.denominator], "float": float(value)}


def block_check():
    probabilities = [F(k, 4) for k in range(5)]
    endpoints = [F(1, 8), F(1, 4), F(1, 2), F(3, 4), F(7, 8)]
    expectation_checks = geometric_checks = 0
    for i, u in enumerate(endpoints):
        for v in endpoints[i:]:
            factors = safe.rational_factors(u, v)
            opposite = [factors[k] for k in (0, 2, 1, 3)]
            for positive, negative in zip(factors, opposite):
                assert positive * negative <= 1
                geometric_checks += 1
            for pm in probabilities:
                for pp in probabilities:
                    mass = [(1-pm)*(1-pp), (1-pm)*pp, pm*(1-pp), pm*pp]
                    ep = sum(p*x for p, x in zip(mass, factors))
                    en = sum(p*x for p, x in zip(mass, opposite))
                    if pm >= pp:
                        assert ep <= 1
                        expectation_checks += 1
                    if pm <= pp:
                        assert en <= 1
                        expectation_checks += 1
    return {"one_sided_expectation_checks": expectation_checks,
            "pointwise_geometric_mean_checks": geometric_checks}


def predictions(n, minus, plus):
    """Four fixed rules, each chosen only from previous counts."""
    inactive = (F(1, 2), F(1, 2))
    fixed = (F(1, 4), F(3, 4))
    q = F(minus+plus+1, 2*n+2)
    d = q*(1-q)/2
    return [fixed, safe.smoothed_endpoints(minus, plus, n),
            fixed if n >= 2 else inactive,
            (q-d, q+d) if min(minus+plus, 2*n-minus-plus) >= 2 else inactive]


def exhaustive_tree(horizon=6, delta=F(1, 4)):
    """Aggregate terminal path multiplicities; probabilities are evaluated later."""
    stack = [(0, 0, 0, (F(1),)*4, (F(1),)*4)]
    terminal = Counter()
    nodes = 0
    while stack:
        n, minus, plus, positive, negative = stack.pop()
        ep, en = sum(positive)/4, sum(negative)/4
        nodes += 1
        if (ep+en)/2 >= 1/delta:
            decision = 1 if ep > en else -1 if en > ep else 0
            if decision:
                terminal[n, minus, plus, decision] += 1
                continue
        if n == horizon:
            terminal[n, minus, plus, 0] += 1
            continue
        factors = [safe.rational_factors(u, v) for u, v in predictions(n, minus, plus)]
        for xm, xp in ((0, 0), (0, 1), (1, 0), (1, 1)):
            index, opposite = 2*xm+xp, 2*xp+xm
            p = tuple(value*f[index] for value, f in zip(positive, factors))
            q = tuple(value*f[opposite] for value, f in zip(negative, factors))
            stack.append((n+1, minus+xm, plus+xp, p, q))
    tests = []
    for pm in [F(k, 4) for k in range(5)]:
        for pp in [F(k, 4) for k in range(5)]:
            total = wrong = F(0)
            for (n, minus, plus, decision), multiplicity in terminal.items():
                probability = multiplicity * pm**minus * (1-pm)**(n-minus)
                probability *= pp**plus * (1-pp)**(n-plus)
                total += probability
                if decision and (pm == pp or decision*(pp-pm) < 0):
                    wrong += probability
            assert total == 1
            assert wrong <= delta
            tests.append({"minus_probability": str(pm), "plus_probability": str(pp),
                          "wrong_direction_risk": encoded(wrong)})
    return {"horizon_pairs": horizon, "delta": str(delta), "tree_nodes": nodes,
            "null_and_alternative_models": len(tests), "tests": tests,
            "maximum_grid_wrong_risk": max(x["wrong_direction_risk"]["float"] for x in tests),
            "scope": "Exhaustive finite examples, not a replacement for the general proof"}


def pooled_fisher(m, minus, plus, odds=F(4)):
    total = minus+plus
    z = sum(F(comb(m, k)*comb(m, total-k))*odds**k
            for k in range(max(0, total-m), min(m, total)+1)) / comb(2*m, total)
    return odds**plus/z


def pooled_obstruction():
    """Even a fixed-angle iid null does not allow arbitrary pooled-time reuse."""
    conditional = []
    stopped_expectation = F(0)
    for xm, xp in ((0, 0), (0, 1), (1, 0), (1, 1)):
        first = (pooled_fisher(1, xm, xp)+pooled_fisher(1, xp, xm))/2
        second = sum((pooled_fisher(2, xm+ym, xp+yp)+
                      pooled_fisher(2, xp+yp, xm+ym))/2
                     for ym, yp in ((0, 0), (0, 1), (1, 0), (1, 1)))/4
        assert first == 1
        stopped_expectation += (first if xm == xp else second)/4
        conditional.append({"first_pair": [xm, xp], "current": str(first),
                            "conditional_next_mean": str(second)})
    assert stopped_expectation == F(91, 88) > 1
    return {"odds": 4, "null": "iid minus=plus=1/2, fixed PSR angle",
            "rule": "Stop at one pair if concordant, otherwise at two pairs",
            "conditional_values": conditional,
            "stopped_symmetric_pooled_evalue_mean": encoded(stopped_expectation),
            "conclusion": "The pooled fixed-horizon Fisher statistic is not an anytime e-process. Multiplying new independently conditioned block factors remains legal; uncorrected pooled recomputation does not."}


def direction_counterexample_and_certificate():
    wrong = F(sum(comb(128, k) for k in range(56)), 2**128)
    expected = (F(9, 4)**8+F(1, 4)**8)/2
    assert wrong > F(1, 100) and expected > 100
    history = [dict(m=64, minus_successes=40, plus_successes=24,
                    lower=[1, 2], upper=[1, 2]),
               dict(m=8, minus_successes=0, plus_successes=8,
                    lower=[1, 4], upper=[3, 4])]
    cert = safe.certificate_from_histories([history])
    assert cert["decision"] == "reject" and cert["shots"] == 144
    assert float(cert["evidence_bounds"][0])-1e-12 <= log(float(expected)) <= float(cert["evidence_bounds"][1])+1e-12
    return {"model": {"A": "1/2", "B": "0", "C": "1/2"},
            "first_block": "64 pairs at sin=0, inactive capital",
            "second_block": "8 pairs at PSR, lower=1/4 upper=3/4",
            "two_sided_capital": encoded(expected),
            "wrong_probability_if_full_raw_score_used": encoded(wrong),
            "wrong_probability_with_directional_capital_rule": 0,
            "one_explicit_wrong_raw_score_path_certificate": cert}


def main():
    if OUT.exists():
        raise RuntimeError("Preserve existing checks; use a new version")
    checks = block_check()
    checks["adaptive_fixed_mixture_enumeration"] = exhaustive_tree()
    checks["raw_direction_counterexample"] = direction_counterexample_and_certificate()
    checks["pooled_fisher_anytime_obstruction"] = pooled_obstruction()
    paths = [Path(__file__).resolve(), Path(safe.__file__).resolve(),
             ROOT/"experiments/circuit_validation/direct_tests.py",
             ROOT/"experiments/sota_round2/adaptive_safe_replay.py"]
    record = {"status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(),
              "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in paths},
              "checks": checks,
              "root_replay_source_review": {
                  "status": "passed_read_only_mathematical_interface_review",
                  "predictability": "fit_rows and start_ready run before rows/shots/successes incorporate the current block; observed current query is allowed",
                  "endpoints": "clipping and rational rounding preserve 0<u<=v<1; shrinkage changes efficiency only",
                  "mixtures": "fixed positive/negative priors normalized separately, including neutral mass; direction uses normalized directional capital",
                  "limits": "Replay remains floating screening on reused development paths; requires final outward certification and fresh validation"},
              "scope": "Deterministic exact finite checks, theorem/interface review and explicit obstructions; no new Monte Carlo or superiority claim"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("x", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    print(json.dumps({"status": "passed", "output": str(OUT), "source_sha256": record["sources"],
                      "one_sided_expectation_checks": checks["one_sided_expectation_checks"],
                      "finite_models": checks["adaptive_fixed_mixture_enumeration"]["null_and_alternative_models"],
                      "raw_direction_wrong_risk": float(checks["raw_direction_counterexample"]["wrong_probability_if_full_raw_score_used"]["float"]),
                      "pooled_optional_mean": "91/88"}, indent=2))


if __name__ == "__main__":
    main()
