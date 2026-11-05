"""Hindsight comparator and information-limit diagnostics from frozen outputs.

This does not select an executable algorithm, tune boundaries, or generate data.
Operating costs/powers come from the float analytic table and are not certified
intervals. The information lower bounds have their own outward certificates.
"""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "results/sota_round4/analytic_prior_comparison.json"
THEORY = ROOT / "results/sota_round4/theory/finite_budget_information_bounds.json"
OUT = ROOT / "results/sota_round4/qualified_comparator_diagnostic.json"
CANDIDATES = ("mix_0.75_0.99_w0.5", "mix_0.7_0.9_0.99_equal")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometric(values):
    values = list(values)
    return math.exp(math.fsum(math.log(v) for v in values) / len(values))


def main():
    hashes = {p.relative_to(ROOT).as_posix(): sha(p)
              for p in (TABLE, THEORY, Path(__file__).resolve())}
    table = json.loads(TABLE.read_text(encoding="utf-8"))
    theory = json.loads(THEORY.read_text(encoding="utf-8"))
    assert table["pair_cap"] == 8192 and table["physical_shot_cap"] == 16384
    assert theory["target_power_exact"] == "19/20"
    assert theory["null_report_budget_exact"] == "1/100"
    main_rows = [r for r in table["rows"] if r["kind"] == "main"]
    ids = sorted({r["case_id"] for r in main_rows})
    methods = sorted({r["method"] for r in main_rows})
    lookup = {(r["method"], r["case_id"]): r for r in main_rows}
    assert len(ids) == 16 and len(methods) == 12
    assert len(lookup) == len(ids) * len(methods)
    qualified = [name for name in methods
                 if all(lookup[name, i]["correct_probability"] >= .95 for i in ids)]
    controls = [name for name in qualified
                if name == "beta_exact" or name.startswith("sprt_")]
    assert controls == ["beta_exact", "sprt_0.6", "sprt_0.7", "sprt_0.75"]
    assert all(name in qualified for name in CANDIDATES)

    def envelope(names):
        selected = {i: min((lookup[name, i] for name in names),
                           key=lambda r: r["expected_shots"]) for i in ids}
        total = math.fsum(r["expected_shots"] for r in selected.values())
        return {
            "method_set": names,
            "status": "hindsight_nonexecutable_diagnostic",
            "equal_case_mean_expected_shots": total / len(ids),
            "rows": [{"case_id": i, "method": row["method"],
                      "expected_shots": row["expected_shots"],
                      "correct_probability": row["correct_probability"]}
                     for i, row in selected.items()],
            "candidate_comparisons": [{
                "candidate": name,
                "equal_case_geometric_candidate_over_envelope": geometric(
                    lookup[name, i]["expected_shots"] / selected[i]["expected_shots"] for i in ids),
                "equal_case_arithmetic_candidate_over_envelope": math.fsum(
                    lookup[name, i]["expected_shots"] for i in ids) / total,
            } for name in CANDIDATES],
            "interpretation": "An oracle chooses a different already-qualified method per case after seeing expected costs. This is a diagnostic envelope, not an implementable baseline or an achieved saving. Float near-ties do not establish strict winners.",
        }

    bounds = []
    for theory_row in theory["rows"]:
        i = theory_row["case_id"]
        beta = lookup["beta_exact", i]
        sampler = theory_row["actual_round3_binary64_psr_probabilities"]
        for name, probability_key in (("minus", "p_minus"), ("plus", "p_plus")):
            assert Fraction.from_float(beta[probability_key]) == Fraction(
                int(sampler[name]["numerator"]), int(sampler[name]["denominator"]))
        interval = sampler["expected_shots_lower_bound"]
        # Report existing rigorous endpoints verbatim, then label every derived
        # ratio as a float diagnostic (the cost has no interval certificate here).
        lower = float(interval["lower"])
        cost = beta["expected_shots"]
        bounds.append({
            "case_id": i,
            "matched_sampler_psr95_expected_shots_lower_bound": {
                "lower": interval["lower"], "upper": interval["upper"]},
            "beta_expected_shots_float": cost,
            "beta_correct_probability_float": beta["correct_probability"],
            "diagnostic_lower_bound_over_beta_cost": lower / cost,
            "diagnostic_beta_cost_over_lower_bound": cost / lower,
            "threefold_beta_cost_target_float": cost / 3,
            "diagnostic_threefold_target_below_lower_bound": cost / 3 < lower,
        })
    assert [r["case_id"] for r in bounds] == [7, 9, 19, 21, 31, 33, 43, 45]
    lb_mean = math.fsum(float(r["matched_sampler_psr95_expected_shots_lower_bound"]["lower"])
                        for r in bounds) / len(bounds)
    lb_geometric_ratio = geometric(r["diagnostic_lower_bound_over_beta_cost"] for r in bounds)
    weak_best = min((lookup[name, 9] for name in controls), key=lambda r: r["expected_shots"])
    result = {
        "status": "complete_hindsight_and_information_necessary_diagnostics",
        "pair_cap": 8192, "physical_shot_cap": 16384,
        "new_observations": 0, "tuned_parameters": 0,
        "numerical_status": "Costs and powers are existing float analytic estimates, not independently certified intervals. Information lower-bound endpoints are copied from the outward theorem artifact. Derived ratios are float diagnostics, not outward enclosures.",
        "qualification_rule": "At least .95 correct-report probability on every one of the 16 main cases, using the existing float analytic table; mean power is insufficient. The .01 uniform null-report guarantee is supplied by each frozen boundary certificate, not inferred from these cases.",
        "qualified_methods": qualified,
        "excluded_methods": [name for name in methods if name not in qualified],
        "beta_and_single_sprt_control_envelope": envelope(controls),
        "all_qualified_frozen_methods_envelope": envelope(qualified),
        "case9_comparison": [{k: row[k] for k in (
            "method", "expected_shots", "correct_probability", "error_probability", "abstain_probability")}
            for row in (lookup["beta_exact", 9], weak_best,
                        *(lookup[name, 9] for name in CANDIDATES))],
        "information_necessary_diagnostic": {
            "scope": "Complete balanced PSR pairs with rounded sampler probabilities matched exactly as rational inputs; finite cap, all-trajectory paid-shot cost, per-case correct report >=.95, uniform null any-report <=.01. General-angle U bounds are deliberately not combined with rounded-sampler costs.",
            "rows": bounds,
            "equal_positive_case_arithmetic_lower_bound_mean_float": lb_mean,
            "equal_positive_case_geometric_lower_bound_over_beta_float": lb_geometric_ratio,
            "diagnostic_maximum_geometric_beta_over_candidate_gain": 1 / lb_geometric_ratio,
            "equal_positive_case_arithmetic_beta_mean_float": math.fsum(
                r["beta_expected_shots_float"] for r in bounds) / len(bounds),
            "interpretation": "The lower bound is necessary for any qualifying balanced PSR procedure and does not give an algorithm achieving it. Relative to the displayed beta cost reference, a 3x casewise or equal-case geometric improvement is incompatible with these information inequalities. Strict comparison to the true beta expected costs still requires their independent interval certification. Eight positive cases are the theory table; the 16-case table also contains sign mirrors.",
        },
        "input_sha256": hashes,
    }
    assert all(sha(ROOT / path) == expected for path, expected in hashes.items())
    # Create a separate result without rewriting any frozen boundary or table.
    with OUT.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(OUT), "control_envelope": result[
        "beta_and_single_sprt_control_envelope"]["candidate_comparisons"],
        "information_lower_bound_geometric_ratio": lb_geometric_ratio,
        "information_lower_bound_arithmetic_mean": lb_mean}, indent=2))


if __name__ == "__main__":
    main()
