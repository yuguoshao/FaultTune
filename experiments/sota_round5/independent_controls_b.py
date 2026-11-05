"""Frozen independent-grid controls, part B, and first-candidate float check.

No candidate construction occurs here. Exact rational q,r are passed directly
to the audited interval evaluator. Sign reflection supplies mirror intervals.
"""
import argparse
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import interval_hybrid_stopping as interval
import solve_hybrid_minimax as hybrid
import evaluate_optimal_sign_physical as physical

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "results/sota_round5/hybrid_minimax"
PLAN = DIRECTORY / "independent_grid_plan.json"
CANDIDATE = DIRECTORY / "candidate_policy.json"
CANDIDATE_SHA = "a2690100803c140b166068f5203b441eb24e01298485721cb11ffbb68b3b7e72"
MANIFEST = ROOT / "results/sota_round4/lr_mixture_boundaries/calibration_manifest.json"
AUDIT = ROOT / "results/sota_round5/theory/hybrid_stopping_interval_review.json"
METHODS = ("sprt_0.99", "mix_0.75_0.99_w0.25", "mix_0.75_0.99_w0.5", "mix_0.75_0.99_w0.75",
           "mix_0.7_0.9_0.99_equal", "mix_beta_0.75_half")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False); handle.write("\n")


def input_context():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert plan["status"] == "fixed_before_hybrid_candidate_and_evaluation"
    assert len(plan["rows"]) == 52 and plan["main_count"] == 48 and plan["null_count"] == 4
    assert sha(CANDIDATE) == CANDIDATE_SHA
    source_paths = (PLAN, CANDIDATE, MANIFEST, AUDIT, Path(__file__).resolve(), Path(interval.__file__),
                    ROOT/"experiments/sota_round5/interval_randomized_stopping.py",
                    ROOT/"experiments/sota_round4/interval_sign_cost.py",
                    ROOT/"experiments/sota_round5/review_hybrid_stopping_intervals.py")
    hashes = {path.relative_to(ROOT).as_posix(): sha(path) for path in source_paths}
    review = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert review["status"] == "passed"
    for path, expected in review["input_sha256"].items():
        assert sha(ROOT/path) == expected
    for path, expected in plan["source_sha256"].items():
        assert sha(ROOT/path) == expected
    return plan, hashes


def cache_interval_primitives():
    raw_passage = interval.first_passage_intervals
    raw_binomial = interval.binomial_intervals

    @lru_cache(maxsize=8)
    def passage(policy, boundary, q, steps):
        return raw_passage(policy, boundary, q, steps)

    @lru_cache(maxsize=8)
    def binomial(n, r):
        return raw_binomial(n, r)

    # operating_rq only reads these returned arrays; no numerical operations
    # are altered and no cache state is shared outside this evaluation process.
    interval.first_passage_intervals = passage
    interval.binomial_intervals = binomial
    return passage, binomial


def augment(case, method, values, derivation, reflected_from=None):
    row = {**case, "method": method, **values, "certification": derivation}
    if reflected_from is not None:
        row["reflected_from_case_id"] = reflected_from
    if case["correct_sign"] > 0:
        row["correct_probability"], row["error_probability"] = values["plus"], values["minus"]
    elif case["correct_sign"] < 0:
        row["correct_probability"], row["error_probability"] = values["minus"], values["plus"]
    else:
        row["correct_probability"] = [0., 0.]
        row["error_probability"] = [float(interval.down(values["plus"][0]+values["minus"][0])),
                                    min(1., float(interval.up(values["plus"][1]+values["minus"][1])))]
    return row


def controls_b():
    started = time.perf_counter()
    plan, hashes = input_context()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    specs = {row["shape"]: row for row in manifest["records"]}
    for method in METHODS:
        spec = specs[method]
        assert sha(ROOT/spec["path"]) == spec["sha256"]
        hashes[spec["path"]] = spec["sha256"]
    passage_cache, binomial_cache = cache_interval_primitives()
    policy = interval.canonical_policy(1, [[0], [0]])
    rows, summaries = [], []
    for method in METHODS:
        part_path = DIRECTORY/"independent_controls_b_parts"/(method+".json")
        if part_path.exists():
            part = json.loads(part_path.read_text(encoding="utf-8"))
            assert part["input_sha256"] == hashes
            rows.extend(part["rows"]); summaries.append(part["summary"])
            print(json.dumps({"resumed_method": method}), flush=True)
            continue
        artifact = json.loads((ROOT/specs[method]["path"]).read_text(encoding="utf-8"))
        boundary = tuple(artifact["boundaries"])
        assert boundary[1] == 2 and len(boundary) == plan["physical_pair_cap"]+1
        risk = artifact["fair_null_risk"]
        uniform_risk = Fraction(int(risk["crossing_numerator"]), int(risk["fair_denominator"]))
        assert uniform_risk <= Fraction(1, 100)
        direct = {}
        for case in plan["rows"]:
            if case["correct_sign"] < 0:
                continue
            values = interval.operating_rq(policy, boundary, Fraction(case["r_exact"]), Fraction(case["q_exact"]), plan["physical_pair_cap"])
            direct[case["case_id"]] = augment(case, method, values, "direct_outward_interval")
            if case["case_id"] % 4 == 3 or case["kind"] == "null":
                print(json.dumps({"method": method, "direct_case_complete": case["case_id"]}), flush=True)
        method_rows = []
        for case in plan["rows"]:
            if case["correct_sign"] >= 0:
                method_rows.append(direct[case["case_id"]]); continue
            source = direct[case["case_id"]-1000]
            assert Fraction(case["q_exact"]) == 1-Fraction(source["q_exact"])
            assert Fraction(case["r_exact"]) == Fraction(source["r_exact"])
            values = {key: list(source[key]) for key in ("plus", "minus", "abstain", "expected_shots")}
            values["plus"], values["minus"] = values["minus"], values["plus"]
            method_rows.append(augment(case, method, values, "exact_sign_reflection_of_direct_interval", source["case_id"]))
        main = [row for row in method_rows if row["kind"] == "main"]
        null = [row for row in method_rows if row["kind"] == "null"]
        qualified = all(row["correct_probability"][0] >= .95 for row in main)
        failed = any(row["correct_probability"][1] < .95 for row in main)
        summary = {"method": method, "all48_main_power_lower_at_least95": qualified,
                   "definitely_fails95_on_at_least_one_main": failed,
                   "qualification_status": "qualified" if qualified else "fails" if failed else "unresolved_interval",
                   "minimum_main_correct_probability_interval": [min(row["correct_probability"][0] for row in main), min(row["correct_probability"][1] for row in main)],
                   "maximum_null_report_probability_interval": [max(row["error_probability"][0] for row in null), max(row["error_probability"][1] for row in null)],
                   "uniform_joint_null_at_most1_percent_exact": True,
                   "uniform_joint_null_fraction": {"numerator": str(uniform_risk.numerator), "denominator": str(uniform_risk.denominator)},
                   "direct_interval_rows": 28, "exact_reflection_rows": 24}
        part = {"method": method, "rows": method_rows, "summary": summary, "input_sha256": hashes}
        write_new(part_path, part)
        rows.extend(method_rows); summaries.append(summary)
        print(json.dumps({"method_complete": method, "qualification": summary["qualification_status"],
                          "minimum_power": summary["minimum_main_correct_probability_interval"]}), flush=True)
    assert all(sha(ROOT/path) == expected for path, expected in hashes.items())
    result = {"status": "complete_independent_controls_b_outward_intervals", "rows": rows,
              "method_summaries": summaries, "methods": list(METHODS), "input_sha256": hashes,
              "independent_grid_sha256": sha(PLAN), "first_candidate_sha256": CANDIDATE_SHA,
              "baseline_adapter": "canonical_policy(1,[[0],[0]]) with every boundary[1]==2; skipping a forbidden d1 report is exactly equivalent to the pure frozen baseline.",
              "reflection_proof": "For a symmetric boundary and symmetric no-report prefix, replacing each sign by its complement exchanges plus/minus and q with1-q. Arrival rate, stopping-time law, abstention and paid cost are unchanged. This is exact and supplies all mirrored rows.",
              "cache_scope": {"first_passage": str(passage_cache.cache_info()), "binomial": str(binomial_cache.cache_info()),
                              "numerical_operations_changed": False},
              "source_of_null_guarantee": "Frozen exact full8192-sign boundary certificate; the4 physical null cases are checks, not a substitute for uniform error control.",
              "candidate_preservation": "The first candidate and its strict internal .00999 margin failure remain unchanged.",
              "elapsed_seconds": time.perf_counter()-started, "new_observations": 0}
    write_new(DIRECTORY/"independent_controls_b.json", result)
    print(json.dumps({"output": str(DIRECTORY/"independent_controls_b.json"), "elapsed_seconds": result["elapsed_seconds"],
                      "qualification": [{"method": row["method"], "status": row["qualification_status"]} for row in summaries]}), flush=True)


def candidate_float():
    started = time.perf_counter()
    plan, hashes = input_context()
    for path in (Path(hybrid.__file__), Path(physical.__file__)):
        hashes[path.relative_to(ROOT).as_posix()] = sha(path)
    artifact = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    protocol = artifact["protocol"]
    prefix, denominator, boundary = protocol["policy_numerators"], protocol["probability_denominator"], protocol["continuation_boundaries"]
    cache, rows = {}, []
    for case in plan["rows"]:
        q, r = Fraction(case["q_exact"]), Fraction(case["r_exact"])
        if q not in cache:
            cache[q] = hybrid.float_full_first_passage(prefix, denominator, boundary, float(q))
        oc = physical.waiting_mixture(cache[q], float(r), plan["physical_pair_cap"])
        positive, negative = oc["positive_probability"], oc["negative_probability"]
        correct = positive if case["correct_sign"] > 0 else negative if case["correct_sign"] < 0 else 0.
        wrong = negative if case["correct_sign"] > 0 else positive if case["correct_sign"] < 0 else positive+negative
        rows.append({**case, **oc, "expected_shots": 2*oc["expected_pairs"],
                     "correct_probability": correct, "error_probability": wrong})
    main = [row for row in rows if row["kind"] == "main"]
    assert all(sha(ROOT/path) == expected for path, expected in hashes.items())
    result = {"status": "first_frozen_candidate_independent_grid_float_crosscheck",
              "first_candidate_sha256": CANDIDATE_SHA, "independent_grid_sha256": sha(PLAN),
              "rows": rows, "minimum_main_correct_probability_float": min(row["correct_probability"] for row in main),
              "all48_main_correct_at_least95_float": all(row["correct_probability"] >= .95 for row in main),
              "equal_case_mean_expected_shots_float": sum(row["expected_shots"] for row in main)/len(main),
              "input_sha256": hashes, "new_observations": 0, "elapsed_seconds": time.perf_counter()-started,
              "limits": "Independent forward float crosscheck only; root supplies outward intervals. No change to first frozen candidate or failed internal .00999 margin."}
    write_new(DIRECTORY/"independent_candidate_float.json", result)
    print(json.dumps({key: result[key] for key in ("status", "minimum_main_correct_probability_float", "all48_main_correct_at_least95_float", "equal_case_mean_expected_shots_float", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("controls", "candidate-float"))
    if parser.parse_args().mode == "controls":
        controls_b()
    else:
        candidate_float()
