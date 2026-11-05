"""Post-freeze mean-estimator variance explanation; never read trial results.

This non-frozen diagnostic reads only the physical case probabilities already
stored in a frozen spec. It does not change an experiment, propose a stopping
rule, or estimate finite-budget validation complexity. No frozen module is
imported. All numerical outputs are explanatory calculations, not certificates.
"""

import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def formula_checks():
    """Exact rational identities, including context dependence and endpoints."""
    examples = (
        ("independent_fair_branches", [Fraction(1, 2)] * 4),
        ("opposite_deterministic_contexts", [Fraction(0), Fraction(1), Fraction(1), Fraction(0)]),
        ("matching_deterministic_contexts", [Fraction(0), Fraction(1), Fraction(0), Fraction(1)]),
        ("shared_public_context_only", [Fraction(1, 10), Fraction(9, 10), Fraction(1, 10), Fraction(9, 10)]),
    )
    weights = [Fraction(1, 2), Fraction(1, 2)]
    rows = []
    for name, probabilities in examples:
        old, new = probabilities[:2], probabilities[2:]
        conditional_delta = [q - p for p, q in zip(old, new)]
        delta = sum(w * d for w, d in zip(weights, conditional_delta))
        noise = sum(w * (p * (1-p) + q * (1-q)) for w, p, q in zip(weights, old, new))
        heterogeneous = sum(w * (d-delta)**2 for w, d in zip(weights, conditional_delta))
        second = sum(w * (p + q - 2*p*q) for w, p, q in zip(weights, old, new))
        old_mean = sum(w*p for w,p in zip(weights,old))
        new_mean = sum(w*q for w,q in zip(weights,new))
        covariance = sum(w*p*q for w,p,q in zip(weights,old,new)) - old_mean*new_mean
        independent = old_mean*(1-old_mean) + new_mean*(1-new_mean)
        assert noise + heterogeneous == second - delta**2
        assert independent - (noise + heterogeneous) == 2 * covariance
        rows.append({"example": name, "variance_identity_exact": True,
                     "public_context_covariance_identity_exact": True,
                     "conditional_noise_variance": str(noise),
                     "context_delta_variance": str(heterogeneous),
                     "paired_difference_variance": str(noise + heterogeneous),
                     "covariance_of_context_means": str(covariance)})
    return rows


def analyze_case(case):
    weights_original = list(case["weights"])
    mass = math.fsum(weights_original)
    if not math.isfinite(mass) or abs(mass - 1) > 1e-12:
        raise ValueError("frozen target weights do not form a probability distribution")
    # Removing final binary64 summation roundoff is only an arithmetic detail;
    # it does not alter the frozen spec or the implemented sampling law.
    weights = [w / mass for w in weights_original]
    probabilities = case["probabilities"]
    number = len(weights)
    if len(probabilities) != 2*number or any(not 0 <= p <= 1 for p in probabilities):
        raise ValueError("invalid frozen independent-arm probabilities")
    old, new = probabilities[:number], probabilities[number:]
    arm_variances = [p * (1-p) for p in probabilities]
    conditional_delta = [q-p for p,q in zip(old,new)]
    delta = math.fsum(w*d for w,d in zip(weights,conditional_delta))
    noise = math.fsum(w*(p*(1-p)+q*(1-q)) for w,p,q in zip(weights,old,new))
    heterogeneity = math.fsum(w*(d-delta)**2 for w,d in zip(weights,conditional_delta))
    difference_variance = noise + heterogeneity
    discordance = math.fsum(w*(p+q-2*p*q) for w,p,q in zip(weights,old,new))
    paired_coefficient = 2*difference_variance
    proportional_coefficient = 2*noise
    neyman_terms = [w*math.sqrt(v) for w,v in zip(weights+weights,arm_variances)]
    neyman_sum = math.fsum(neyman_terms)
    optimum = neyman_sum**2
    allocation = [term/neyman_sum for term in neyman_terms] if neyman_sum else None
    old_mean = math.fsum(w*p for w,p in zip(weights,old))
    new_mean = math.fsum(w*q for w,q in zip(weights,new))
    covariance = math.fsum(w*(p-old_mean)*(q-new_mean) for w,p,q in zip(weights,old,new))
    independent_target_coefficient = 2*(old_mean*(1-old_mean)+new_mean*(1-new_mean))
    checks = {
        "variance_decomposition_residual": difference_variance - (discordance-delta**2),
        "context_pairing_covariance_residual": independent_target_coefficient-paired_coefficient-4*covariance,
        "optimal_allocation_variance_residual": None if allocation is None else
            math.fsum(w*w*v/x for w,v,x in zip(weights+weights,arm_variances,allocation) if x)-optimum,
    }
    if any(abs(value) > 2e-13 for value in checks.values() if value is not None):
        raise ArithmeticError("variance identity failed numerical crosscheck")
    if optimum > proportional_coefficient + 2e-13 or proportional_coefficient > paired_coefficient + 2e-13:
        raise ArithmeticError("Cauchy/context-variance ordering failed numerical crosscheck")
    return {
        "case_id": case["case_id"], "phase": case["phase"], "epsilon": case["epsilon"],
        "strata": case["strata"], "weights": weights_original,
        "probabilities_old_contexts_then_new": probabilities,
        "conditional_delta": conditional_delta, "mixed_delta": delta,
        "conditional_noise_variance_mean": noise, "context_delta_variance": heterogeneity,
        "discordance_probability": discordance, "paired_difference_variance": difference_variance,
        "paired_per_total_shot_variance_coefficient": paired_coefficient,
        "fixed_proportional_stratum_variance_coefficient": proportional_coefficient,
        "oracle_optimal_independent_arm_variance_coefficient": optimum,
        "oracle_fixed_allocation_old_contexts_then_new": allocation,
        "independent_target_context_variance_coefficient": independent_target_coefficient,
        "covariance_of_public_context_failure_means": covariance,
        "optimal_over_paired": optimum/paired_coefficient if paired_coefficient else None,
        "paired_over_optimal": paired_coefficient/optimum if optimum else None,
        "context_heterogeneity_fraction_of_paired_variance": heterogeneity/difference_variance if difference_variance else None,
        "variance_savings_from_fixed_context_counts": paired_coefficient-proportional_coefficient,
        "variance_savings_from_oracle_allocation": proportional_coefficient-optimum,
        "ordering_holds_numerically": optimum <= proportional_coefficient+2e-13 <= paired_coefficient+4e-13,
        "numerical_identity_checks": checks,
    }


def _aggregate(rows):
    ratios = [row["paired_over_optimal"] for row in rows]
    if any(ratio is None or ratio <= 0 for ratio in ratios):
        raise ValueError("positive finite variance ratios required for this main-grid summary")
    return {"cases": len(rows),
            "paired_over_optimal_geometric_mean": math.exp(math.fsum(math.log(x) for x in ratios)/len(ratios)),
            "paired_over_optimal_min": min(ratios), "paired_over_optimal_max": max(ratios),
            "max_heterogeneity_fraction": max(row["context_heterogeneity_fraction_of_paired_variance"] for row in rows)}


def diagnostic(spec_path):
    spec_path = Path(spec_path)
    spec_bytes = spec_path.read_bytes()
    digest = hashlib.sha256(spec_bytes).hexdigest()
    declared = spec_path.with_suffix(spec_path.suffix+".sha256").read_text(encoding="ascii").split()[0]
    if digest != declared:
        raise ValueError("frozen spec sidecar hash mismatch")
    spec = json.loads(spec_bytes.decode("utf8"))
    cases = [case for case in spec["cases"] if case["benchmark_role"] == "main"]
    if len(cases) != 36:
        raise ValueError("this explanatory diagnostic expects the complete frozen 36-main-case grid")
    rows = [analyze_case(case) for case in cases]
    strata = {}
    for axis in ("epsilon", "noise", "weight_plus", "intended_sign"):
        groups = defaultdict(list)
        for row in rows:
            groups[str(row["strata"][axis])].append(row)
        strata[axis] = {name:_aggregate(group) for name,group in sorted(groups.items())}
    return {
        "schema": "faulttune.postfreeze_variance_diagnostic.v1",
        "purpose": "exploratory mechanism explanation; not a frozen acceptance endpoint",
        "input_spec": str(spec_path.resolve()), "input_spec_sha256": digest,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "raw_trial_results_read": False, "changes_frozen_experiment": False,
        "probability_source": "physical statevector/Kraus case probabilities frozen before confirmation",
        "sampling_law": "fresh iid public context per pair; old/new noise independent conditional on context",
        "coefficient_convention": "Var(mean estimator)=coefficient/T for total paid shots T, at fixed positive allocation proportions",
        "important_limit": "oracle fixed-allocation variance comparison, not a lower bound for abstaining finite-budget validation and not an automatic finite-sample speedup",
        "oracle_allocation_is_not_an_executed_policy": True,
        "numeric_status": "binary64 physical probabilities and square roots; explanatory calculations, not certified bounds",
        "exact_rational_formula_checks": formula_checks(),
        "aggregate": _aggregate(rows), "strata": strata, "per_main_case": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=REPO/"experiments/circuit_validation/confirmation_spec_v1.json")
    parser.add_argument("--output", type=Path, default=REPO/"results/variance_diagnostic.json")
    args = parser.parse_args()
    result = diagnostic(args.spec)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf8")
    print(json.dumps({"output":str(args.output.resolve()),"aggregate":result["aggregate"]},sort_keys=True))


if __name__ == "__main__":
    main()
