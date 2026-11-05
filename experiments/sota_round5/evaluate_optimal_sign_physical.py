"""Float first-passage/negative-binomial evaluation of the frozen sign LP.

Once the H128 sign rule would abstain, this physical protocol forbids further
reports and charges every remaining pair through the physical cap. No early
abstention saving is counted. Exact small physical-state checks are independent
of the first-passage waiting decomposition. No Monte Carlo observations.
"""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "results/sota_round5/optimal_sign_stopping/prototype.json"
CASES = ROOT / "results/sota_round3/paired_benchmark/preliminary_plan.json"
CONTROLS = ROOT / "results/sota_round4/analytic_prior_comparison.json"
OUT = ARTIFACT.with_name("physical_cap16384_float.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rational(record):
    return Fraction(int(record["numerator"]), int(record["denominator"]))


def first_passage(numerators, denominator, q, exact=False):
    cast = Fraction if exact else float
    q = cast(q)
    row = [cast(1)]
    plus, minus = [cast(0)], [cast(0)]
    expected = cast(0)
    for d in range(1, len(numerators)):
        expected += sum(row)
        fresh = [cast(0)]*(d+1)
        for k, mass in enumerate(row):
            fresh[k] += (1-q)*mass
            fresh[k+1] += q*mass
        p_plus, p_minus = cast(0), cast(0)
        for k, mass in enumerate(fresh):
            numerator = numerators[d][min(k, d-k)]
            probability = Fraction(numerator, denominator) if exact else numerator/denominator
            stopped = mass*probability
            if 2*k > d:
                p_plus += stopped
            elif 2*k < d:
                p_minus += stopped
            else:
                assert stopped == 0
            fresh[k] -= stopped
        plus.append(p_plus); minus.append(p_minus)
        row = fresh
    survival = sum(row)
    if exact:
        assert sum(plus)+sum(minus)+survival == 1
    else:
        assert abs(math.fsum(plus)+math.fsum(minus)+survival-1) < 1e-12
    return plus, minus, survival, expected


def waiting_mixture(distribution, rate, pair_cap):
    plus, minus, survival, _ = distribution
    if rate == 0:
        return {"positive_probability": 0., "negative_probability": 0.,
                "abstain_probability": 1., "expected_pairs": float(pair_cap)}
    d = np.arange(1, len(plus), dtype=float)
    stop_by_cap = binom.sf(d-1, pair_cap, rate)
    beyond_cap = binom.cdf(d-1, pair_cap, rate)
    truncated_wait = d*(binom.sf(d, pair_cap+1, rate)/rate)+pair_cap*beyond_cap
    plus = np.asarray(plus[1:]); minus = np.asarray(minus[1:])
    p_plus = math.fsum((plus*stop_by_cap).tolist())
    p_minus = math.fsum((minus*stop_by_cap).tolist())
    abstain = survival+math.fsum(((plus+minus)*beyond_cap).tolist())
    # Non-reporting H128 paths pay all the way through pair_cap.
    expected = pair_cap*survival+math.fsum(((plus+minus)*truncated_wait).tolist())
    assert abs(p_plus+p_minus+abstain-1) < 1e-12
    assert 0 <= expected <= pair_cap*(1+1e-12)
    return {"positive_probability": p_plus, "negative_probability": p_minus,
            "abstain_probability": abstain, "expected_pairs": expected}


def exact_binomial_tail(n, r, k):
    return sum((math.comb(n, j)*r**j*(1-r)**(n-j) for j in range(k, n+1)), Fraction(0))


def exact_waiting_mixture(distribution, rate, pair_cap):
    plus, minus, survival, _ = distribution
    p_plus = p_minus = Fraction(0)
    abstain, expected = survival, pair_cap*survival
    for d in range(1, len(plus)):
        before = exact_binomial_tail(pair_cap, rate, d)
        truncated = sum((1-exact_binomial_tail(n, rate, d) for n in range(pair_cap)), Fraction(0))
        p_plus += plus[d]*before; p_minus += minus[d]*before
        abstain += (plus[d]+minus[d])*(1-before)
        expected += (plus[d]+minus[d])*truncated
    return p_plus, p_minus, abstain, expected


def exact_physical_dp(numerators, denominator, rate, q, pair_cap):
    """Direct three-outcome per-pair DP, without waiting-time identities."""
    sign_cap = len(numerators)-1
    live = {(0, 0): Fraction(1)}
    plus = minus = expected = Fraction(0)
    for _ in range(pair_cap):
        expected += sum(live.values(), Fraction(0))
        fresh = {}
        for (d, k), mass in live.items():
            if d == sign_cap:
                fresh[d, k] = fresh.get((d, k), Fraction(0))+mass
                continue
            for dd, kk, chance in ((d, k, 1-rate), (d+1, k+1, rate*q), (d+1, k, rate*(1-q))):
                branch = mass*chance
                stop = Fraction(0) if dd == d else branch*Fraction(numerators[dd][min(kk, dd-kk)], denominator)
                if stop:
                    if 2*kk > dd:
                        plus += stop
                    else:
                        assert 2*kk < dd
                        minus += stop
                fresh[dd, kk] = fresh.get((dd, kk), Fraction(0))+branch-stop
        live = fresh
    abstain = sum(live.values(), Fraction(0))
    assert plus+minus+abstain == 1
    return plus, minus, abstain, expected


def main():
    paths = (ARTIFACT, CASES, CONTROLS, Path(__file__).resolve())
    hashes = {path.relative_to(ROOT).as_posix(): sha(path) for path in paths}
    plan = {
        "status": "fixed_physical_protocol_before_evaluation",
        "pair_cap": 8192, "physical_shot_cap": 16384,
        "candidate_source": ARTIFACT.relative_to(ROOT).as_posix(),
        "candidate_sha256": sha(ARTIFACT),
        "monitoring": "Inspect after each physical pair, with a fresh policy coin only when that pair is discordant. A concordant pair neither changes the sign state nor redraws the policy coin.",
        "sign_horizon_abstention": "At discordance H128, all remaining non-reporting paths are forbidden to report later and are charged through the full physical cap. No early-abstention cost saving.",
        "comparison": "All original 16 main cases and 2 null cases. Original H8192 controls are compared at the same physical cap and monitoring, with per-case >=95% power qualification. Original controls may use more than128 discordances, a permitted advantage to them.",
        "new_observations": 0, "status_of_results": "Floating estimates pending independent interval verification.",
    }
    with OUT.with_name("physical_evaluation_plan.json").open("x", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2); handle.write("\n")
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    numerators = artifact["protocol"]["policy_numerators"]
    denominator = artifact["protocol"]["probability_denominator"]
    grid_checks = []
    for row in artifact["results"][0]["rows"]:
        q = Fraction(*row["q_exact"])
        dist = first_passage(numerators, denominator, float(q))
        computed = (math.fsum(dist[0]), math.fsum(dist[1]), dist[2], dist[3])
        expected = [float(rational(row[key])) for key in ("positive", "negative", "abstain", "expected_discordances")]
        error = max(abs(a-b) for a, b in zip(computed, expected))
        assert error < 1e-10
        grid_checks.append({"q_exact": row["q_exact"], "maximum_absolute_float_error_to_exact_sign_dp": error})
    small = numerators[:9]
    exact_checks = []
    for rate, q in ((Fraction(1, 5), Fraction(7, 10)), (Fraction(1, 2), Fraction(1, 2)), (Fraction(1), Fraction(1, 10))):
        a = exact_waiting_mixture(first_passage(small, denominator, q, exact=True), rate, 12)
        b = exact_physical_dp(small, denominator, rate, q, 12)
        assert a == b
        exact_checks.append({"rate": str(rate), "q": str(q), "pair_cap": 12, "sign_cap": 8, "exactly_equal": True})
    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    controls = json.loads(CONTROLS.read_text(encoding="utf-8"))
    assert controls["pair_cap"] == plan["pair_cap"]
    rows = []
    for case in cases:
        pm, pp = case["A"]-case["C"], case["A"]+case["C"]
        positive_pair = (1-pm)*pp
        negative_pair = pm*(1-pp)
        r = positive_pair+negative_pair
        q = positive_pair/r if r else .5
        dist = first_passage(numerators, denominator, q)
        row = waiting_mixture(dist, r, plan["pair_cap"])
        correct = row["positive_probability"] if case["C"] > 0 else row["negative_probability"] if case["C"] < 0 else 0.
        wrong = row["negative_probability"] if case["C"] > 0 else row["positive_probability"] if case["C"] < 0 else row["positive_probability"]+row["negative_probability"]
        row.update(case_id=case["case_id"], kind=case["kind"], expected_shots=2*row["expected_pairs"],
                   correct_probability=correct, error_probability=wrong,
                   p_minus=pm, p_plus=pp, discordance_rate=r, conditional_plus_probability=q,
                   sign_horizon_survival=dist[2], no_report_tail_paid_shots=2*plan["pair_cap"]*dist[2])
        rows.append(row)
    main_rows = [r for r in rows if r["kind"] == "main"]
    ids = {row["case_id"] for row in main_rows}
    by_id = {row["case_id"]: row for row in main_rows}
    assert len(main_rows) == 16
    summaries = []
    qualified_methods = []
    for method in (item["method"] for item in controls["aggregates"]):
        baseline = [r for r in controls["rows"] if r["kind"] == "main" and r["method"] == method]
        assert {r["case_id"] for r in baseline} == ids
        qualified = min(r["correct_probability"] for r in baseline) >= .95
        if qualified:
            qualified_methods.append(method)
        summaries.append({"control": method, "control_qualified_every_case_float": qualified,
                          "candidate_over_control_geometric_cost_float": math.exp(math.fsum(
                              math.log(by_id[r["case_id"]]["expected_shots"]/r["expected_shots"]) for r in baseline)/len(baseline)),
                          "candidate_over_control_arithmetic_cost_float": math.fsum(r["expected_shots"] for r in main_rows)/math.fsum(r["expected_shots"] for r in baseline)})
    envelope = {i: min((r for r in controls["rows"] if r["kind"] == "main" and r["case_id"] == i and r["method"] in qualified_methods), key=lambda r: r["expected_shots"]) for i in ids}
    result = {
        "status": "complete_frozen_lp_physical_float_development_evaluation",
        "protocol": plan, "rows": rows,
        "aggregate": {"minimum_case_correct_probability": min(r["correct_probability"] for r in main_rows),
                      "every_main_case_correct_at_least95_float": all(r["correct_probability"] >= .95 for r in main_rows),
                      "equal_case_mean_expected_shots": math.fsum(r["expected_shots"] for r in main_rows)/len(main_rows),
                      "maximum_main_error_probability": max(r["error_probability"] for r in main_rows),
                      "maximum_null_false_report_probability": max(r["error_probability"] for r in rows if r["kind"] == "null")},
        "control_comparisons": summaries,
        "qualified_control_hindsight_envelope": {"methods": qualified_methods,
            "candidate_over_envelope_geometric_cost_float": math.exp(math.fsum(math.log(by_id[i]["expected_shots"]/envelope[i]["expected_shots"]) for i in ids)/len(ids)),
            "scope": "Non-executable hindsight diagnostic; not a deployable baseline.",
            "selected_control_by_case": {str(i): envelope[i]["method"] for i in sorted(ids)}},
        "independent_checks": {"exact_grid_crosschecks": grid_checks, "exact_physical_dp_crosschecks": exact_checks},
        "input_sha256": hashes, "new_observations": 0,
        "claim_boundary": "This is retrospective development evaluation of an existing classical LP rule with every non-reporting path charged to the full cap. All float costs and probabilities need interval confirmation. No new algorithmic novelty, SOTA, broad uniform superiority or threefold-improvement claim.",
    }
    assert all(sha(ROOT/path) == digest for path, digest in hashes.items())
    with OUT.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False); handle.write("\n")
    print(json.dumps({"output": str(OUT), "aggregate": result["aggregate"],
                      "comparisons": summaries, "envelope": result["qualified_control_hindsight_envelope"]}, indent=2))


if __name__ == "__main__":
    main()
