"""Freeze comparable validation budgets using DEVELOPMENT cases only.

For each (epsilon, plus-context weight, noise) main stratum, compute the iid
paired discordance probability for both development signs. The registered
design point is min(1, 1.10*max(discordance)), with effect magnitude epsilon.
The initial pair budget is
    ceil(1.10*d_design*(z_(1-delta/2)+z_.98)**2 / epsilon**2).
Numerically evaluate the exact conditional McNemar rejection region at that
design point, increasing the pair count by 10% if its power is below .98.

Every algorithm receives the same case-level maximum of four fixed-test shot
counts: shot_budget=4*(2*fixed_pairs). The fixed comparator takes its single
registered look at fixed_pairs, not at the larger sequential limit. All costs
still count actual executions; budget exhaustion penalties belong to scoring.

This is numerical power planning at a development-derived design distribution,
not a uniform guarantee over all unknown discordance values. In particular,
the 10% discordance inflation is a registered design margin, not a proved
worst-case theorem. Fixed McNemar's error guarantee does not rely on this power
assumption. Development pilots may assess the frozen proposal before formal
validation, but confirmation truth or outcomes must not be used to tune it.

Gray and blind controls use the maximum main-stratum budget for their epsilon;
their own failure probabilities never enter planning. They remain diagnostics
outside efficiency scoring. Confirmation keys are mechanically renamed from
development keys; this module never builds or reads confirmation cases.
"""

import math
from collections import defaultdict
from statistics import NormalDist

from case_factory import EPSILONS, NOISE_SETTINGS, PLUS_WEIGHTS, build_cases
from direct_tests import fixed_paired_power


DESIGN_VERSION = "faulttune.development_only_budget_design.v1"
DELTA = 0.01
TARGET_POWER = 0.98
DISCORDANCE_MARGIN = 1.10
NORMAL_SAMPLE_MARGIN = 1.10
RETRY_MULTIPLIER = 1.10
MAX_PLANNING_ATTEMPTS = 32
SEQUENTIAL_BUDGET_MULTIPLE = 4


def _tag(value):
    return format(value, ".6g").replace(".", "p").replace("-", "m")


def _group_key(epsilon, weight_plus, noise):
    return "e{}_w{}_{}".format(_tag(epsilon), _tag(weight_plus), noise)


def _paired_discordance(probabilities, weights):
    if len(probabilities) != 4 or len(weights) != 2:
        raise ValueError("expected old0,oldplus,new0,newplus probabilities and two weights")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("development probabilities must lie in [0,1]")
    if any(not math.isfinite(w) or w <= 0 for w in weights) or abs(math.fsum(weights) - 1) > 1e-12:
        raise ValueError("development weights must be a positive probability distribution")
    return math.fsum(w * (old * (1 - new) + (1 - old) * new)
                     for w, old, new in zip(weights, probabilities[:2], probabilities[2:]))


def _plan_stratum(rows):
    if len(rows) != 2 or {row["category"] for row in rows} != {"beneficial", "harmful"}:
        raise ValueError("each main stratum must retain both prescribed development signs")
    first = rows[0]
    epsilon, weights, noise = first["epsilon"], first["weights"], first["stratum"]
    for row in rows:
        if row["phase"] != "development":
            raise ValueError("budget planning is restricted to development cases")
        if not row.get("feasible", False):
            raise ValueError("infeasible development cell must be reported and resolved: " + row["case_id"])
        if (row["epsilon"], row["weights"], row["stratum"]) != (epsilon, weights, noise):
            raise ValueError("mixed main strata cannot share a design calculation")
    discordances = {row["case_id"]: _paired_discordance(row["probabilities"], row["weights"])
                    for row in rows}
    maximum = max(discordances.values())
    design_discordance = min(1.0, DISCORDANCE_MARGIN * maximum)
    if design_discordance < epsilon:
        raise ValueError("development-derived discordance cannot support the registered epsilon design")
    quantile_sum = NormalDist().inv_cdf(1 - DELTA / 2) + NormalDist().inv_cdf(TARGET_POWER)
    normal_pairs = NORMAL_SAMPLE_MARGIN * design_discordance * quantile_sum ** 2 / epsilon ** 2
    initial_pairs = max(1, math.ceil(normal_pairs))
    pairs, attempts, power = initial_pairs, [], None
    for unused in range(MAX_PLANNING_ATTEMPTS):
        power = fixed_paired_power(pairs, design_discordance, epsilon, DELTA)
        estimate = power["power_estimate"]
        if not math.isfinite(estimate) or not 0 <= estimate <= 1:
            raise ArithmeticError("numerical power calculation returned an invalid probability")
        attempts.append({"pairs": pairs, "power_estimate": estimate,
                         "omitted_k_probability": power["omitted_k_probability"]})
        if estimate >= TARGET_POWER:
            break
        pairs = max(pairs + 1, math.ceil(RETRY_MULTIPLIER * pairs))
    else:
        raise RuntimeError("could not obtain the requested numerical development-design power")
    fixed_shots = 2 * pairs
    return {
        "group_key": _group_key(epsilon, weights[1], noise),
        "epsilon": epsilon, "weights": list(weights), "noise_stratum": noise,
        "source_phase": "development", "source_case_ids": sorted(discordances),
        "development_discordances": discordances,
        "maximum_development_discordance": maximum,
        "design_discordance": design_discordance, "design_gap": epsilon,
        "discordance_margin": DISCORDANCE_MARGIN,
        "normal_sample_margin": NORMAL_SAMPLE_MARGIN,
        "normal_initial_pairs": initial_pairs, "planning_attempts": attempts,
        "delta": DELTA, "target_power": TARGET_POWER,
        "fixed_pairs": pairs, "fixed_shots": fixed_shots,
        "shot_budget": SEQUENTIAL_BUDGET_MULTIPLE * fixed_shots,
        "power_plan": power,
        "power_scope": "registered development-derived design point; not uniform over unknown discordance",
        "numerical_status": "SciPy floating-point power calculation; not machine-certified",
    }


def design_budgets():
    """Return {by_case, strata, diagnostic_budgets, metadata}; run no trials.

    by_case contains all development IDs and mechanically corresponding
    confirmation IDs. Both signs in a main stratum receive identical budgets.
    Requires NumPy/SciPy only through the numerical fixed-test power planner.
    """
    cases = build_cases("development")
    grouped = defaultdict(list)
    for row in cases:
        if row["phase"] != "development":
            raise ValueError("non-development case reached the budget design input")
        if row["benchmark_role"] == "main":
            grouped[(row["epsilon"], row["weights"][1], row["stratum"])].append(row)
    expected = {(epsilon, weight, noise[0]) for epsilon in EPSILONS
                for weight in PLUS_WEIGHTS for noise in NOISE_SETTINGS}
    if set(grouped) != expected:
        raise ValueError("the complete registered 18-stratum development matrix is required")
    if len(cases) != 44 or len({row["case_id"] for row in cases}) != 44:
        raise ValueError("all 44 distinct registered development cases must be retained")
    strata = {}
    for key in sorted(grouped):
        plan = _plan_stratum(grouped[key])
        strata[plan["group_key"]] = plan
    diagnostic_budgets = {}
    for epsilon in EPSILONS:
        same_gap = [plan for plan in strata.values() if plan["epsilon"] == epsilon]
        worst = max(same_gap, key=lambda plan: (plan["fixed_pairs"], plan["group_key"]))
        diagnostic_budgets[_tag(epsilon)] = {
            "epsilon": epsilon, "fixed_pairs": worst["fixed_pairs"],
            "fixed_shots": worst["fixed_shots"], "shot_budget": worst["shot_budget"],
            "source_group": worst["group_key"],
            "mapping_scheme": "maximum main-stratum pair budget at the same epsilon; diagnostic truth unused",
            "diagnostic_power_claim": None,
        }
    by_case = {}
    for row in cases:
        if not row["case_id"].startswith("dev_"):
            raise ValueError("development IDs must use the frozen dev_ naming scheme")
        if row["benchmark_role"] == "main":
            key = _group_key(row["epsilon"], row["weights"][1], row["stratum"])
            source = strata[key]
            mapping = "same epsilon/weight/noise development stratum, shared by both signs"
        else:
            source = diagnostic_budgets[_tag(row["epsilon"])]
            key = source["source_group"]
            mapping = source["mapping_scheme"]
        common = {
            "fixed_pairs": source["fixed_pairs"], "fixed_shots": source["fixed_shots"],
            "shot_budget": source["shot_budget"], "epsilon": row["epsilon"],
            "benchmark_role": row["benchmark_role"], "design_group_key": key,
            "source_phase": "development", "mapping_scheme": mapping,
            "fixed_test_use": "one look after exactly fixed_pairs; no extension based on validation data",
        }
        development_id = row["case_id"]
        confirmation_id = "confirm_" + development_id[len("dev_"):]
        by_case[development_id] = dict(common, case_id=development_id, evaluation_phase="development")
        by_case[confirmation_id] = dict(common, case_id=confirmation_id, evaluation_phase="confirmation")
    return {
        "by_case": by_case, "strata": strata, "diagnostic_budgets": diagnostic_budgets,
        "metadata": {
            "design_version": DESIGN_VERSION, "source_phase": "development",
            "uses_confirmation_truth": False, "uses_algorithm_observations": False,
            "development_case_count": len(cases), "main_stratum_count": len(strata),
            "mapped_case_id_count": len(by_case), "delta": DELTA, "target_power": TARGET_POWER,
            "discordance_margin": DISCORDANCE_MARGIN, "normal_sample_margin": NORMAL_SAMPLE_MARGIN,
            "sequential_budget_multiple_of_fixed_shots": SEQUENTIAL_BUDGET_MULTIPLE,
            "confirmation_mapping": "replace only the dev_ ID prefix with confirm_; never load confirmation cases",
            "power_design": "gap=epsilon at each inflated development discordance; numerical conditional McNemar power",
            "limits": "power is distribution-specific numerical planning, not a uniform guarantee or certified floating-point bound",
            "freeze_requirement": "freeze this proposal after development pilots and before confirmation validation",
        },
    }
