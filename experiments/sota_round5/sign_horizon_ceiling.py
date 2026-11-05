"""Exact classical Neyman-Pearson ceiling for a symmetric H-sign rule.

This reads the design only, not candidate performance, to diagnose mandatory
full-cap charging after a short sign horizon. It is not a new algorithm.
"""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "results/sota_round5/physical_minimax/development_plan.json"
OUT = PLAN.with_name("sign_horizon_ceiling.json")


def record(x):
    return {"numerator": str(x.numerator), "denominator": str(x.denominator), "float_diagnostic": float(x)}


def main():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    horizon = plan["sign_horizon"]
    alpha = Fraction(*plan["scientific_null_constraint_exact"])
    weights, budget = {}, alpha
    for j in range(horizon//2+1):
        multiplicity = 1 if j == horizon-j else 2
        null_mass = Fraction(multiplicity*math.comb(horizon, j), 2**horizon)
        acceptance = min(Fraction(1), budget/null_mass)
        weights[j] = acceptance
        budget -= acceptance*null_mass
    assert budget == 0
    boundary = max(j for j, probability in weights.items() if probability > 0)
    rows = []
    for q_record in plan["positive_q_exact"]:
        q = Fraction(*q_record)
        maximum_report = Fraction(0)
        for j, probability in weights.items():
            mass = math.comb(horizon, j)*q**j*(1-q)**(horizon-j)
            if j != horizon-j:
                mass += math.comb(horizon, j)*q**(horizon-j)*(1-q)**j
            maximum_report += probability*mass
        missing = 1-maximum_report
        rows.append({"q_exact": q_record, "any_report_probability_ceiling_exact": record(maximum_report),
                     "no_report_probability_floor_exact": record(missing),
                     "mandatory_full_cap_shots_from_no_report_floor_exact": record(plan["physical_shot_cap"]*missing)})
    result = {
        "status": "exact_symmetric_sign_horizon_neyman_pearson_ceiling",
        "horizon": horizon, "fair_null_any_report_budget": record(alpha),
        "randomized_final_count_upper_threshold": horizon-boundary,
        "randomized_final_count_boundary_probability": record(weights[boundary]),
        "rows": rows, "new_observations": 0, "candidate_performance_read": False,
        "proof": [
            "Extend every sequentially stopped sign history by fresh fictitious iid signs toH. The any-report decision is a randomized function of theseH signs; this extension does not increase the actually observed information.",
            "A sign-reflection-symmetric rule has identical any-report probability underq and1-q. Its power equals that under the symmetric alternative mixture(q,1-q).",
            "Relative to iid fair signs, the symmetric alternative likelihood depends only on|2k-H| and strictly increases with it forq>.5. The Neyman-Pearson fractional knapsack therefore accepts the most extreme final-count classes first, with randomization at at most one class.",
            "The exact rational class masses above spend the entire fair-null budget, so they maximize any-report probability over all randomized symmetricH-sign procedures, including sequential ones.",
            "A physical cap can only prevent latent reports that would require later discordances. Thus actual physical non-report probability is at least this floor. If all non-reporting paths pay the full physical cap, the displayed cap-times-floor contribution is unavoidable, before adding the cost of reporting paths.",
        ],
        "scope": "A ceiling for fixed iidq and reflection-symmetric policies using at mostH signs; not a bound for asymmetric policies, longer sign horizons, or extra informative observations. The displayed cost contribution is a lower bound, not the full optimal expected cost.",
        "source_sha256": {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in (PLAN, Path(__file__).resolve())},
    }
    with OUT.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2); handle.write("\n")
    print(json.dumps({"output": str(OUT), "threshold": horizon-boundary,
                      "boundary_probability": float(weights[boundary]),
                      "rows": [{"q": row["q_exact"], "report_ceiling": row["any_report_probability_ceiling_exact"]["float_diagnostic"],
                                "mandatory_cost_floor": row["mandatory_full_cap_shots_from_no_report_floor_exact"]["float_diagnostic"]} for row in rows]}))


if __name__ == "__main__":
    main()
