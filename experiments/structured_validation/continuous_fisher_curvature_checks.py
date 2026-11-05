"""Exact finite checks and deterministic reuse of all 48 stored endpoints."""
from collections import Counter
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time

import continuous_fisher_curvature as curvature


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/structured_validation"
OUTPUT = BASE / "continuous_fisher_curvature_checks.json"
Bounds = curvature.Bounds


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def difference(a, b):
    with localcontext() as ctx:
        ctx.prec = 100
        return str(Decimal(a) - Decimal(b))


def run_checks():
    started = time.perf_counter()
    sources = curvature.source_hashes()
    own = Path(__file__).resolve()
    sources[own.relative_to(ROOT).as_posix()] = digest(own)
    variance_checks, maximum_ratio = 0, Fraction(0)
    for m in range(1, 9):
        for total in range(2 * m + 1):
            support = range(max(0, total - m), min(m, total) + 1)
            for odds in (Fraction(1, 64), Fraction(1, 4), Fraction(1, 2),
                         Fraction(1), Fraction(2), Fraction(4), Fraction(64)):
                weights = [(k, math.comb(m, k) * math.comb(m, total - k) * odds ** k)
                           for k in support]
                z = sum(w for k, w in weights)
                mean = sum(k * w for k, w in weights) / z
                second = sum(k * k * w for k, w in weights) / z
                variance = second - mean * mean
                s = min(total, 2 * m - total)
                assert 0 <= variance <= Fraction(s, 4) <= Fraction(m, 4)
                if s:
                    maximum_ratio = max(maximum_ratio, 4 * variance / s)
                for k in support:
                    ratio = Fraction(math.comb(m, k) * math.comb(m, total - k),
                                     math.comb(total, k) * math.comb(2 * m - total, m - k))
                    assert ratio == Fraction(math.factorial(m) ** 2,
                                             math.factorial(total) * math.factorial(2 * m - total))
                if total >= 2:
                    covariance = (second - mean) / (total * (total - 1)) - (mean / total) ** 2
                    assert covariance <= 0
                other = 2 * m - total
                if other >= 2:
                    other_mean = m - mean
                    other_second = m * m - 2 * m * mean + second
                    covariance = (other_second - other_mean) / (other * (other - 1)) - (other_mean / other) ** 2
                    assert covariance <= 0
                variance_checks += 1

    analytic = []
    for positive, negative in ((0, 0), (1, 0), (3, 2), (8, 0), (20, 20), (20, 0), (0, 20)):
        blocks = ([{"m": 1, "minus_successes": 0, "plus_successes": 1}] * positive
                  + [{"m": 1, "minus_successes": 1, "plus_successes": 0}] * negative)
        d = positive + negative
        exact = Fraction(math.factorial(2 * positive) * math.factorial(2 * negative),
                         2 ** d * math.factorial(positive) * math.factorial(negative) * math.factorial(d))
        cert = curvature.certificate_from_blocks(blocks, (2 * positive + 1, 2 * negative + 1))
        expected = Bounds(exact.numerator) / exact.denominator
        lower = Bounds(Decimal(cert["log_evidence_lower_bound"])).exp()
        assert lower.hi < expected.lo
        assert cert["n"] == cert["S"] == d
        if positive == 20 and negative == 0:
            assert cert["decision"] == "reject"
        if positive == 0 and negative == 20:
            assert cert["decision"] == "accept"
        analytic.append({"case": "m1", "positive": positive, "negative": negative,
                         "complete_mixture_exact": str(exact), "certificate": cert})
    root6 = Bounds(6).sqrt()
    for minus, plus, node, expected in ((0, 2, (4, 1), 2 * root6 - 3),
                                       (2, 0, (1, 4), 2 * root6 - 3),
                                       (1, 1, (1, 1), 3 - root6)):
        cert = curvature.certificate_from_blocks([{"m": 2, "minus_successes": minus,
                                                   "plus_successes": plus}], node)
        lower = Bounds(Decimal(cert["log_evidence_lower_bound"])).exp()
        assert lower.hi < expected.lo
        analytic.append({"case": "m2 T2", "minus": minus, "plus": plus,
                         "complete_mixture_analytic_bounds": [str(expected.lo), str(expected.hi)],
                         "certificate": cert})

    replay_path = BASE / "continuous_fisher_replay_refined.json"
    replay = json.loads(replay_path.read_text())
    sources[replay_path.relative_to(ROOT).as_posix()] = digest(replay_path)
    development = BASE / "refined_development_v1"
    inputs = {}
    for name, expected in replay["development_input_sha256"].items():
        path = development / name
        assert digest(path) == expected
        inputs[path.relative_to(ROOT).as_posix()] = expected
    spec = json.loads((development / "spec.json").read_text())
    audit = json.loads((development / "audit.json").read_text())
    assert audit["status"] == "passed"
    for name, expected in audit["run_input_sha256"].items():
        path = development / name
        assert digest(path) == expected
        inputs[path.relative_to(ROOT).as_posix()] = expected
    assert replay["selection"] == {"policies": ["psr", "closed"], "rep": 0, "case_ids": None,
                                     "source_engine": "fisher", "points": 193,
                                     "penultimate": False, "records": 48}
    expected_keys = {(c["case_id"], policy + "_fisher", 0)
                     for c in spec["cases"] for policy in ("psr", "closed")}
    found = {}
    with (development / "trials.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            key = record["case_id"], record["method"], record["rep"]
            if key in expected_keys:
                assert key not in found
                found[key] = record
    assert set(found) == expected_keys and len(found) == 48
    records, seen = [], set()
    for row in replay["records"]:
        key = row["case_id"], row["method"], row["rep"]
        assert key not in seen and row["terminal_checkpoint"]
        seen.add(key)
        source = found[key]
        blocks = [{k: b[k] for k in ("m", "minus_successes", "plus_successes")}
                  for b in source["history"]]
        assert sum(2 * b["m"] for b in blocks) == row["checkpoint_shots"] == source["shots"]
        mode = row["proposal"]["mode_proposal"]
        node_index = min(range(len(row["certificate"]["nodes"])),
                         key=lambda i: (abs(math.log(row["certificate"]["nodes"][i][0]
                                                       / row["certificate"]["nodes"][i][1]) - mode), i))
        node = row["certificate"]["nodes"][node_index]
        cpu, wall = time.process_time(), time.perf_counter()
        cert = curvature.certificate_from_blocks(blocks, node, spec["delta"])
        cpu_seconds, wall_seconds = time.process_time() - cpu, time.perf_counter() - wall
        assert cert["shots"] == source["shots"] and cert["score"] == source["certificate"]["score"]
        assert Decimal(cert["single_node_S_log_bound_enclosure"][0]) >= Decimal(cert["single_node_n_log_bound_enclosure"][0])
        old_upper = row["original_terminal_evidence_log_bounds"][1]
        refined_lower = row["certificate"]["log_evidence_lower_bound"]
        records.append({"case_id": key[0], "method": key[1], "rep": key[2],
                        "checkpoint_shots": source["shots"], "node_index_in_193_grid": node_index,
                        "certificate": cert, "cpu_seconds": cpu_seconds, "wall_seconds": wall_seconds,
                        "S_lower_minus_193_lower": difference(cert["log_evidence_lower_bound"], refined_lower),
                        "S_lower_minus_71_upper": difference(cert["log_evidence_lower_bound"], old_upper),
                        "n_lower_minus_193_lower": difference(cert["single_node_n_log_bound_enclosure"][0], refined_lower)})
    assert seen == expected_keys
    for name, expected in dict(sources, **inputs).items():
        assert digest(ROOT / name) == expected, name
    return {"status": "passed", "sources": sources, "input_sha256": inputs,
            "scope": "Deterministic variance/analytic checks and all 48 existing development terminal histories; no Monte Carlo, first-crossing analysis, or prospective data.",
            "exact_variance_cases": variance_checks, "maximum_exact_4_variance_over_S": str(maximum_ratio),
            "exact_weighted_subset_identity_and_covariances_checked": True,
            "analytic_cases": analytic,
            "selection": {"cases": 24, "policies": ["psr", "closed"], "rep": 0, "checkpoints": 48,
                          "node_rule": "Stored exact rational node closest in absolute log odds to the stored floating mode; index breaks ties"},
            "records": records,
            "summary": {"S_lower_exceeds_193_lower": sum(Decimal(r["S_lower_minus_193_lower"]) > 0 for r in records),
                        "S_lower_exceeds_71_upper": sum(Decimal(r["S_lower_minus_71_upper"]) > 0 for r in records),
                        "certified_crossings": sum(r["certificate"]["decision"] != "abstain" for r in records),
                        "cpu_seconds_total": sum(r["cpu_seconds"] for r in records),
                        "wall_seconds_total": sum(r["wall_seconds"] for r in records)},
            "sources_and_inputs_unchanged": True,
            "claim_boundary": "A stronger computed lower bound is not a stopping-cost or SOTA win. Bounds below another lower bound do not order the true complete mixture.",
            "elapsed_seconds": time.perf_counter() - started}


if __name__ == "__main__":
    assert not OUTPUT.exists(), "Preserve the existing checks artifact"
    report = run_checks()
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"status": report["status"], "exact_variance_cases": report["exact_variance_cases"],
                      "summary": report["summary"], "elapsed_seconds": report["elapsed_seconds"],
                      "output": str(OUTPUT)}, indent=2))
