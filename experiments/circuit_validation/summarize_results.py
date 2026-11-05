"""Stream frozen circuit trial logs into complete-grid statistical summaries.

This module reads a spec and raw observations; it never selects cases, changes
budgets, tunes policies, or calls the case factory. Every intended main case
remains in the geometric averages even when a candidate or baseline fails.
Development summaries are exploratory and can never pass confirmation gates.
Cost ratios against all registered baselines remain visible when qualification
is incomplete, but their diagnostic display is not a success claim.

Only compact paid-cost/outcome tuples are retained in memory. Full event and
certificate replay is a separate engineering audit; this reader checks their
structural consistency and provenance, not the numerical proof of each stop.
"""

import argparse
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
import glob
import gzip
import hashlib
import json
import math
from pathlib import Path
import csv
from fractions import Fraction
import struct
import sys

from assessment import summarize_cell, compare_costs


SCHEMA = "faulttune.circuit_summary.v1"
SECONDARY_FIELDS = ("readout_objects", "switches", "execution_duration",
                    "modeled_device_duration", "elapsed_seconds")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _next_float(value, upward):
    if value == 0:
        return float.fromhex("0x0.0000000000001p-1022") * (1 if upward else -1)
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    bits += 1 if ((value > 0) == upward) else -1
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


def geometric_mean(values, direction="point"):
    """Keep missing/zero/infinite evidence; never drop a difficult cell."""
    values = list(values)
    if not values or any(value is None for value in values):
        return None if direction == "point" else 0.0 if direction == "lower" else math.inf
    if any(value < 0 or math.isnan(value) for value in values):
        raise ValueError("nonnegative geometric-mean inputs required")
    if any(value == 0 for value in values):
        return None if direction == "point" and any(math.isinf(value) for value in values) else 0.0
    if any(math.isinf(value) for value in values):
        return math.inf
    if direction == "point":
        return math.exp(math.fsum(math.log(value) for value in values) / len(values))
    with localcontext() as ctx:
        ctx.prec = 64
        ctx.rounding = ROUND_FLOOR if direction == "lower" else ROUND_CEILING
        logs = []
        for value in values:
            center = Decimal.from_float(float(value)).ln()
            logs.append(center.next_minus() if direction == "lower" else center.next_plus())
        mean_log = sum(logs, Decimal(0)) / len(values)
        center = mean_log.exp()
        bound = center.next_minus() if direction == "lower" else center.next_plus()
    result = float(bound)
    if direction == "lower" and Decimal.from_float(result) > bound:
        result = _next_float(result, False)
    elif direction == "upper" and Decimal.from_float(result) < bound:
        result = _next_float(result, True)
    return result


def _describe(values):
    values = list(values)
    if not values:
        return dict(n=0, mean=None, median=None, p95=None, maximum=None, minimum=None)
    ordered, n = sorted(values), len(values)
    return dict(n=n, mean=math.fsum(values) / n,
                median=ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2,
                p95=ordered[math.ceil(.95 * n) - 1], maximum=ordered[-1], minimum=ordered[0])


def _load_spec(path):
    path = Path(path)
    digest = _sha256(path)
    expected = path.with_suffix(path.suffix + ".sha256").read_text(encoding="ascii").split()[0]
    if digest != expected:
        raise ValueError("frozen spec SHA256 mismatch")
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("schema") != "faulttune.circuit_spec.v1":
        raise ValueError("unsupported experiment spec schema")
    policies, cases = spec["policies"], spec["cases"]
    if not policies or len(set(policies)) != len(policies):
        raise ValueError("frozen policy list must be nonempty and unique")
    if len({row["case_id"] for row in cases}) != len(cases):
        raise ValueError("frozen case IDs must be unique")
    if type(spec["repetitions"]) is not int or spec["repetitions"] < 1:
        raise ValueError("positive fixed repetitions required")
    for role in ("baselines", "candidates", "diagnostic_policies"):
        if len(set(spec[role])) != len(spec[role]) or not set(spec[role]).issubset(policies):
            raise ValueError("invalid frozen policy role: " + role)
    if not spec["baselines"] or not spec["candidates"]:
        raise ValueError("at least one frozen baseline and candidate required")
    if (set(spec["diagnostic_policies"]) & (set(spec["baselines"]) | set(spec["candidates"]))):
        raise ValueError("diagnostic policies cannot certify a baseline or candidate")
    return spec, digest


def _seed(spec, case_id, policy, index):
    parts = (spec["master_seed"], spec["phase"], case_id, policy, index)
    value = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return str(int.from_bytes(hashlib.blake2b(value.encode("ascii"), digest_size=16,
                                              person=b"ft-circuit-v1").digest(), "big"))


def _valid_number(value, integral=False):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value >= 0 and (not integral or int(value) == value))


def _alpha_ledger(spec):
    a = spec["analysis"]
    cells = len(spec["cases"]) * len(spec["policies"])
    spends, exact_spends, excesses = {}, {}, {}
    valid = a.get("unused_alpha_recycled") is False
    for family, count, key in (("safety", 3, "alpha_safety"), ("power", 2, "alpha_power"), ("cost", 2, "alpha_cost")):
        value = a[key]
        valid = valid and _valid_number(value) and 0 < value < 1
        exact_spends[family] = count * cells * Fraction(str(value))
        spends[family] = float(exact_spends[family])
        limit = Fraction(str(a[family + "_family_alpha"]))
        excesses[family] = max(Fraction(0), exact_spends[family] - limit)
        valid = valid and exact_spends[family] <= limit
    exact_total = sum(exact_spends.values(), Fraction(0))
    total = float(exact_total)
    excesses["total"] = max(Fraction(0), exact_total - Fraction(str(a["family_alpha"])))
    valid = valid and excesses["total"] == 0
    return {"valid": bool(valid), "cell_count": cells, "family_spends": spends,
            "total_allocated_alpha": total, "registered_analysis": a,
            "family_spends_exact": {key: str(value) for key, value in exact_spends.items()},
            "total_allocated_alpha_exact": str(exact_total),
            "budget_excesses_exact": {key: str(value) for key, value in excesses.items()},
            "budget_check": "exact Fraction(str(registered alpha)); no overspend tolerance",
            "legacy_development_rounding_only": (spec["phase"] == "development" and not valid
                                                  and all(value <= Fraction(1, 10**14) for value in excesses.values())),
            "formal_simultaneous_confidence_claim_allowed": bool(valid and spec["phase"] == "confirmation"),
            "intervals_use_exact_registered_per_endpoint_alpha": True}


def summarize_results(spec_path, trial_paths):
    """Read all supplied shards and return a fail-closed complete-grid summary."""
    spec, spec_hash = _load_spec(spec_path)
    cases = {case["case_id"]: case for case in spec["cases"]}
    policies = spec["policies"]
    cells = {(case_id, policy): [] for case_id in cases for policy in policies}
    seen = {key: bytearray(spec["repetitions"]) for key in cells}
    cell_issues, issue_counts, issue_examples = defaultdict(list), Counter(), []
    invalid_actual_cost = set()
    raw_provenance, raw_records = [], 0

    def issue(kind, location, detail=None, cell=None):
        issue_counts[kind] += 1
        record = {"issue": kind, "location": location}
        if detail is not None:
            record["detail"] = str(detail)
        if len(issue_examples) < 100:
            issue_examples.append(record)
        if cell is not None and len(cell_issues[cell]) < 12:
            cell_issues[cell].append(record)

    for raw_path in trial_paths:
        path = Path(raw_path)
        if not path.is_file():
            issue("missing_raw_file", str(path))
            raw_provenance.append({"path": str(path.resolve()), "sha256": None})
            continue
        raw_provenance.append({"path": str(path.resolve()), "sha256": _sha256(path),
                               "compressed_bytes": path.stat().st_size})
        try:
            with gzip.open(str(path), "rt", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    raw_records += 1
                    location = "{}:{}".format(path.name, line_number)
                    try:
                        record = json.loads(line)
                    except (ValueError, TypeError) as exc:
                        issue("invalid_json", location, exc)
                        continue
                    if not isinstance(record, dict):
                        issue("invalid_record_type", location)
                        continue
                    key = (record.get("case_id"), record.get("policy"))
                    index = record.get("trial_index")
                    if (not all(isinstance(part, str) for part in key) or key not in cells
                            or type(index) is not int or not 0 <= index < spec["repetitions"]):
                        issue("outside_planned_grid", location, (key, index))
                        continue
                    if seen[key][index]:
                        issue("duplicate_trial_key", location, (key, index), key)
                        continue
                    seen[key][index] = 1
                    case, budget = cases[key[0]], cases[key[0]]["shot_budget"]
                    if record.get("spec_sha256") != spec_hash:
                        issue("record_spec_sha256_mismatch", location, cell=key)
                    if record.get("phase") != spec["phase"] or record.get("schema") != "faulttune.circuit_trial.v1":
                        issue("record_phase_or_schema_mismatch", location, cell=key)
                    if str(record.get("seed")) != _seed(spec, key[0], key[1], index):
                        issue("record_seed_mismatch", location, cell=key)
                    shots = record.get("shots")
                    if not _valid_number(shots, integral=True):
                        issue("missing_or_invalid_actual_shots", location, cell=key)
                        invalid_actual_cost.add(key)
                        shots = budget
                    if shots > budget or record.get("shot_budget") != budget:
                        issue("record_shot_budget_violation", location, cell=key)
                    decision, status = record.get("decision"), record.get("status")
                    if decision not in ("accept", "reject", "abstain"):
                        issue("invalid_decision", location, decision, key)
                        decision = "abstain"
                    if status not in ("ok", "timeout", "error"):
                        issue("invalid_status", location, status, key)
                        status = "error"
                    if status != "ok":
                        issue("non_ok_trial_status", location, status, key)
                    result = record.get("result")
                    if status == "ok":
                        if not isinstance(result, dict) or result.get("decision") != decision:
                            issue("result_decision_mismatch", location, cell=key)
                        else:
                            cert = result.get("certificate")
                            if not isinstance(cert, dict) or cert.get("decision") != decision:
                                issue("certificate_decision_mismatch", location, cell=key)
                            if key[1] not in spec["diagnostic_policies"] and result.get("guarantee_in_scope") is not True:
                                issue("out_of_scope_certificate", location, cell=key)
                    cost = record.get("cost") or {}
                    if not isinstance(cost, dict):
                        cost = {}
                    if cost.get("shots") != shots:
                        issue("cost_shot_mismatch", location, cell=key)
                    arm_counts, arm_failures = cost.get("counts"), cost.get("failures")
                    if (not isinstance(arm_counts, list) or not isinstance(arm_failures, list)
                            or len(arm_counts) != 2 * len(case["contexts"]) or len(arm_failures) != len(arm_counts)
                            or not all(_valid_number(x, True) for x in arm_counts + arm_failures)
                            or sum(arm_counts) != shots or any(s > n for s, n in zip(arm_failures, arm_counts))):
                        issue("arm_cost_ledger_mismatch", location, cell=key)
                    public = record.get("public_config") or {}
                    for field, expected in (("shot_budget", budget), ("epsilon", case["epsilon"]),
                                             ("weights", case["weights"]), ("contexts", case["contexts"]),
                                             ("delta", spec["delta"]), ("phase", spec["phase"])):
                        if not isinstance(public, dict) or public.get(field) != expected:
                            issue("public_config_mismatch", location, field, key)
                    secondary = []
                    for field in SECONDARY_FIELDS:
                        value = record.get(field) if field == "elapsed_seconds" else cost.get(field)
                        if not _valid_number(value, field in ("readout_objects", "switches")):
                            issue("secondary_cost_missing_or_invalid", location, field, key)
                            value = None
                        secondary.append(value)
                    measured = spec["observation_contract"]["cost"]["measured_qubits"]
                    if secondary[0] != shots * measured:
                        issue("readout_object_cost_mismatch", location, cell=key)
                    cells[key].append((int(shots), sys.intern(decision), sys.intern(status),
                                       record.get("shot_budget", budget), *secondary))
        except (OSError, EOFError, UnicodeError) as exc:
            issue("raw_stream_read_failure", str(path), exc)

    main_cases = [case for case in spec["cases"] if case.get("intended_efficiency_eligible", case["benchmark_role"] == "main")]
    power_cases = [case for case in spec["cases"] if case["benchmark_role"] in ("main", "blind_diagnostic")]
    expected_roles = Counter(case["benchmark_role"] for case in spec["cases"])
    if expected_roles != {"main": 36, "gray_diagnostic": 6, "blind_diagnostic": 2} or len(main_cases) != 36:
        issue("registered_case_matrix_mismatch", "spec", dict(expected_roles))
    ledger = _alpha_ledger(spec)
    if not ledger["valid"]:
        issue("alpha_ledger_invalid", "spec")
    per_cell, lookup, missing_cells = [], {}, []
    for (case_id, policy), compact in cells.items():
        case = cases[case_id]
        missing = spec["repetitions"] - sum(seen[(case_id, policy)])
        if missing:
            missing_cells.append({"case_id": case_id, "policy": policy, "missing": missing,
                                  "first_missing_indices": [i for i, value in enumerate(seen[(case_id, policy)]) if not value][:12]})
        records = [dict(shots=r[0], decision=r[1], status=r[2], shot_budget=r[3]) for r in compact]
        a = spec["analysis"]
        summary = summarize_cell(records, case["true_delta"], case["epsilon"],
                                 a["alpha_safety"], a["alpha_power"], a["alpha_cost"], case["shot_budget"])
        valid = not missing and not cell_issues[(case_id, policy)] and case["feasible"]
        if (case_id, policy) in invalid_actual_cost:
            summary["actual_cost"].update(mean=None, valid=False, lower=0.0, upper=math.inf)
            for field in ("median", "p95", "minimum", "maximum"):
                summary["actual_cost"][field] = None
        if not valid:
            summary["qualified"] = False
            summary["record_integrity_pass"] = False
            summary["actual_cost"].update(valid=False, lower=0.0, upper=math.inf)
            summary["penalized_cost"].update(valid=False, lower=0.0, upper=math.inf)
        secondary = {field: _describe(r[4 + i] for r in compact if r[4 + i] is not None)
                     for i, field in enumerate(SECONDARY_FIELDS)}
        summary.update(case_id=case_id, policy=policy, phase=spec["phase"],
                       benchmark_role=case["benchmark_role"], category=case["category"],
                       stratum=case["stratum"], strata=case["strata"], weights=case["weights"],
                       intended_efficiency_eligible=case in main_cases,
                       planned_repetitions=spec["repetitions"], missing_repetitions=missing,
                       grid_complete=missing == 0, structural_integrity_pass=valid,
                       input_issue_examples=cell_issues[(case_id, policy)], secondary_costs=secondary,
                       actual_cost_unknown=(case_id, policy) in invalid_actual_cost,
                       confirmation_interval_scope=spec["phase"] == "confirmation" and valid,
                       diagnostic_policy=policy in spec["diagnostic_policies"])
        if summary["diagnostic_policy"]:
            summary["qualified"] = False
        per_cell.append(summary)
        lookup[(case_id, policy)] = summary

    complete = not missing_cells and raw_records == len(cells) * spec["repetitions"]
    structural_pass = complete and not issue_counts and all(case["feasible"] for case in cases.values()) and ledger["valid"]
    per_policy = {}
    for policy in policies:
        selected = [lookup[(case_id, policy)] for case_id in cases]
        power_selected = [lookup[(case["case_id"], policy)] for case in power_cases]
        main_selected = [lookup[(case["case_id"], policy)] for case in main_cases]
        flat = [row for (case_id, p), rows in cells.items() if p == policy for row in rows]
        counts = Counter()
        for row in selected:
            counts.update(row["decisions"])
        unsafe = policy in spec["diagnostic_policies"]
        per_policy[policy] = {
            "diagnostic_only": unsafe, "case_count": len(selected),
            "records": sum(row["n"] for row in selected), "decisions": dict(counts),
            "wrong_decisions": sum(row["safety"]["count"] for row in selected),
            "main_wrong_decisions": sum(row["safety"]["count"] for row in main_selected),
            "main_records": sum(row["n"] for row in main_selected),
            "main_correct_decisions": sum(row["power"]["count"] or 0 for row in main_selected),
            "noncompletion_count": sum(row["noncompletion_count"] for row in selected),
            "safety_cases_passed": sum(row["safety"]["passes"] and row["structural_integrity_pass"] for row in selected),
            "power_cases_passed": sum(row["power"]["passes"] is True and row["structural_integrity_pass"] for row in power_selected),
            "power_case_count": len(power_selected),
            "safety_all_cases_pass": not unsafe and all(row["safety"]["passes"] and row["structural_integrity_pass"] for row in selected),
            "power_all_required_cases_pass": not unsafe and all(row["power"]["passes"] is True and row["structural_integrity_pass"] for row in power_selected),
            "actual_shots_descriptive": _describe(r[0] for r in flat),
            "secondary_costs": {field: _describe(r[4 + i] for r in flat if r[4 + i] is not None)
                                for i, field in enumerate(SECONDARY_FIELDS)},
            "descriptive_scope": "pooled all frozen cases; heterogeneous workloads; no hardware confidence claim",
        }
        if any(p == policy for unused_case_id, p in invalid_actual_cost):
            per_policy[policy]["actual_shots_descriptive"] = dict(
                n=len(flat), mean=None, median=None, p95=None, maximum=None, minimum=None,
                valid=False, reason="at least one raw actual cost is unavailable")

    per_candidate = {}
    for candidate in spec["candidates"]:
        comparisons = {}
        for case in main_cases:
            comparisons[case["case_id"]] = compare_costs(lookup[(case["case_id"], candidate)],
                                                        {b: lookup[(case["case_id"], b)] for b in spec["baselines"]})
        def aggregate(subset):
            selected = [comparisons[case["case_id"]] for case in subset]
            result = {"case_count": len(subset), "case_ids": [case["case_id"] for case in subset],
                      "qualification_pass": all(row["qualification_pass"] for row in selected),
                      "qualification_pass_count": sum(row["qualification_pass"] for row in selected)}
            for metric in ("actual_cost", "penalized_cost"):
                ratios = [row["descriptive_all_baselines"][metric] for row in selected]
                result[metric] = {field: geometric_mean((row[field] for row in ratios), field)
                                  for field in ("point", "lower", "upper")}
                result[metric]["scope"] = "all preregistered baselines; qualification separately required"
            result["point_at_least_one"] = all(result[metric]["point"] is not None and result[metric]["point"] >= 1
                                                for metric in ("actual_cost", "penalized_cost"))
            return result
        overall = aggregate(main_cases)
        strata = {}
        for axis, function in (("epsilon", lambda c: c["epsilon"]), ("noise", lambda c: c["stratum"]),
                               ("weight", lambda c: c["weights"][1]), ("sign", lambda c: c["strata"]["intended_sign"])):
            groups = defaultdict(list)
            for case in main_cases:
                groups[str(function(case))].append(case)
            strata[axis] = {name: aggregate(rows) for name, rows in sorted(groups.items())}
        required = [candidate] + list(spec["baselines"])
        candidate_all_cases = (per_policy[candidate]["safety_all_cases_pass"]
                               and per_policy[candidate]["power_all_required_cases_pass"])
        baseline_main = all(lookup[(case["case_id"], baseline)]["qualified"]
                            for case in main_cases for baseline in spec["baselines"])
        qualifications = all(per_policy[p]["safety_all_cases_pass"] and per_policy[p]["power_all_required_cases_pass"] for p in required)
        point_target, lower_target = spec["analysis"]["speedup_point_target"], spec["analysis"]["speedup_lower_target"]
        cost_target = all(overall[metric]["point"] is not None and overall[metric]["point"] >= point_target
                          and overall[metric]["lower"] >= lower_target for metric in ("actual_cost", "penalized_cost"))
        strata_target = all(row["point_at_least_one"] for axis in strata.values() for row in axis.values())
        numerical_pass = structural_pass and qualifications and overall["qualification_pass"] and cost_target and strata_target
        per_candidate[candidate] = {
            "main": overall, "strata": strata, "per_main_case_comparison": comparisons,
            "baseline_main_qualification_pass": bool(baseline_main),
            "candidate_all_cases_qualification_pass": bool(candidate_all_cases),
            "qualification_all_44_safety_and_38_power": qualifications,
            "qualification_scope_note": "main-case comparability uses all baselines on 36 main cases; candidate scope uses 44 safety/38 power cases; overall criteria retain the stricter requirement that every baseline also pass 44 safety/38 power cases",
            "all_preregistered_baselines": list(spec["baselines"]),
            "main_cost_targets_pass": cost_target, "all_major_stratum_points_at_least_one": strata_target,
            "numerical_registered_criteria_pass": bool(numerical_pass),
            "criteria_pass": bool(numerical_pass and spec["phase"] == "confirmation" and spec.get("confirmation_freeze")),
            "scope": "development diagnostic only" if spec["phase"] != "confirmation" else "frozen confirmation statistical criteria; independent event/certificate replay separate",
        }
    return {
        "schema": SCHEMA, "phase": spec["phase"], "repetitions": spec["repetitions"],
        "policies": policies, "baselines": spec["baselines"], "candidates": spec["candidates"],
        "interval_claim": "exploratory development summary; no formal confirmation claim" if spec["phase"] != "confirmation" else
                          "fixed frozen repetitions; simultaneous endpoint coverage only when complete and structurally valid",
        "provenance": {"spec_path": str(Path(spec_path).resolve()), "spec_sha256": spec_hash,
                       "raw_files": raw_provenance, "frozen_source_hashes": spec.get("source_hashes", {}),
                       "analysis_sources": {Path(__file__).name: _sha256(__file__),
                                            "assessment.py": _sha256(Path(__file__).with_name("assessment.py"))}},
        "integrity": {"pass": structural_pass, "complete_grid_pass": complete,
                      "raw_records": raw_records, "unique_planned_records": sum(sum(v) for v in seen.values()),
                      "planned_records": len(cells) * spec["repetitions"], "missing_cells": missing_cells,
                      "issue_counts": dict(issue_counts), "issue_examples": issue_examples,
                      "all_physical_cases_feasible": all(case["feasible"] for case in cases.values()),
                      "scope": "structural provenance and cost checks; full certificate/event replay is separate"},
        "alpha_ledger": ledger, "per_cell": per_cell, "per_policy": per_policy,
        "per_candidate": per_candidate,
        "overall": {"criteria_pass": any(row["criteria_pass"] for row in per_candidate.values()),
                    "confirmation_scope": spec["phase"] == "confirmation" and bool(spec.get("confirmation_freeze")),
                    "intended_main_cases": len(main_cases), "safety_cases": len(cases), "power_cases": len(power_cases),
                    "diagnostic_policies_excluded_from_certification": spec["diagnostic_policies"]},
    }


def write_outputs(summary, prefix):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {name: prefix.with_name(prefix.name + suffix) for name, suffix in
             (("json", "_summary.json"), ("cells", "_cells.csv"), ("candidates", "_candidates.csv"), ("policies", "_policies.csv"))}
    paths["json"].write_text(json.dumps(_json_safe(summary), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    def write_csv(path, rows):
        rows = list(rows)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
            writer.writeheader()
            writer.writerows(_json_safe(row) for row in rows)
    cell_rows = []
    for row in summary["per_cell"]:
        flat = {key: row[key] for key in ("case_id", "policy", "benchmark_role", "category", "epsilon", "stratum", "n", "grid_complete", "qualified")}
        flat.update(true_delta=row["true_delta"], safety_errors=row["safety"]["count"], safety_upper=row["safety"]["upper"],
                    power_rate=row["power"]["observed_rate"], power_lower=row["power"]["lower"],
                    noncompletion_rate=row["abstention_or_failure_rate"])
        for metric in ("actual_cost", "penalized_cost"):
            for field in ("mean", "lower", "upper", "median", "p95", "maximum"):
                flat[metric + "_" + field] = row[metric][field]
        for metric in SECONDARY_FIELDS:
            flat[metric + "_mean"] = row["secondary_costs"][metric]["mean"]
        cell_rows.append(flat)
    write_csv(paths["cells"], cell_rows)
    candidate_rows = []
    for name, result in summary["per_candidate"].items():
        groups = [("main", "all", result["main"])] + [(axis, value, row) for axis, entries in result["strata"].items() for value, row in entries.items()]
        for axis, value, row in groups:
            flat = dict(candidate=name, axis=axis, stratum=value, case_count=row["case_count"], qualification_pass=row["qualification_pass"], criteria_pass=result["criteria_pass"])
            for metric in ("actual_cost", "penalized_cost"):
                for field in ("point", "lower", "upper"):
                    flat[metric + "_" + field] = row[metric][field]
            candidate_rows.append(flat)
    write_csv(paths["candidates"], candidate_rows)
    policy_rows = []
    for name, row in summary["per_policy"].items():
        flat = {key: row[key] for key in ("diagnostic_only", "records", "main_records", "wrong_decisions", "main_wrong_decisions", "main_correct_decisions", "noncompletion_count", "safety_cases_passed", "power_cases_passed")}
        flat["policy"] = name
        for metric in SECONDARY_FIELDS:
            flat[metric + "_mean"] = row["secondary_costs"][metric]["mean"]
        policy_rows.append(flat)
    write_csv(paths["policies"], policy_rows)
    return {key: str(path.resolve()) for key, path in paths.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--trials", nargs="+", required=True)
    parser.add_argument("--output-prefix", "--prefix", dest="prefix", required=True)
    args = parser.parse_args()
    paths = []
    for pattern in args.trials:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches if matches else [pattern])
    summary = summarize_results(args.spec, paths)
    outputs = write_outputs(summary, args.prefix)
    print(json.dumps(_json_safe({"outputs": outputs, "integrity": summary["integrity"]["pass"],
                                "criteria_pass": summary["overall"]["criteria_pass"],
                                "candidate_main": {name: {key: row["main"][key] for key in ("case_count", "actual_cost", "penalized_cost", "qualification_pass")}
                                                   for name, row in summary["per_candidate"].items()}}), sort_keys=True))


if __name__ == "__main__":
    main()
