"""Fixed development-path screening of total-conditioned adaptive odds.

The current total T may inform a conditional alternative, but the current
plus allocation must not. No new samples or final performance claim here.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import gammaln, logsumexp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/structured_validation"))
import run_refined as old
import weighted_fisher as fisher
from conditional_fisher import centered_cgf_float
from predictive import fit_rows
from adaptive_safe_replay import digest, write_new, DENOMINATOR, SHRINKAGES


def choose_odds(past_theta, arm, m, total, mode, shrink):
    """The interface intentionally does not accept current arm allocations."""
    co, si = arm["cos_numerator"] / arm["denominator"], arm["sin_numerator"] / arm["denominator"]
    if mode == "past":
        center = float(past_theta[0] + past_theta[1] * co)
    elif mode == "total":
        center = (total + .5) / (2 * m + 1)
    else:
        raise ValueError(mode)
    center = min(1 - 2 / DENOMINATOR, max(2 / DENOMINATOR, center))
    gap = min(float(shrink) * abs(float(past_theta[2] * si)), .999 * min(center, 1 - center))
    low = min(DENOMINATOR - 1, max(1, int(round((center - gap) * DENOMINATOR))))
    high = min(DENOMINATOR - 1, max(low, int(round((center + gap) * DENOMINATOR))))
    return Fraction(high * (DENOMINATOR - low), low * (DENOMINATOR - high))


def replay(history):
    modes = ("past", "total")
    capitals = {mode: np.zeros((len(SHRINKAGES), 2)) for mode in modes}
    stopped, rows, logged = {}, [], {mode: [[] for k in SHRINKAGES] for mode in modes}
    old_state = fisher.empty_state()
    counts = np.zeros(4, dtype=np.int64)
    experts = old_state["experts"]
    odds = [Fraction(e["numerator"], e["denominator"]) for e in experts]
    positive = [j for j, o in enumerate(odds) if o > 1]
    zero = [j for j, o in enumerate(odds) if o == 1]
    negative = [odds.index(1 / odds[j]) for j in positive]
    weights = np.array([2 * float(Fraction(*experts[j]["weight"])) for j in positive]
                       + [float(Fraction(*experts[j]["weight"])) for j in zero])
    logw = np.log(weights)
    shots = 0
    for index, block in enumerate(history):
        m, km, kp = block["m"], block["minus_successes"], block["plus_successes"]
        total = km + kp
        theta = fit_rows(rows)["theta"]
        for mode in modes:
            for j, shrink in enumerate(SHRINKAGES):
                omega = choose_odds(theta, block["arm"], m, total, mode, shrink)
                eta = math.log(omega.numerator) - math.log(omega.denominator)
                cgf = centered_cgf_float(m, total, omega.numerator, omega.denominator) if omega != 1 else 0.
                linear = eta * (kp - km) / 2
                capitals[mode][j] += (linear - cgf, -linear - cgf)
                logged[mode][j].append({"m": m, "minus_successes": km, "plus_successes": kp,
                                        "odds": [omega.numerator, omega.denominator]})
        fisher.update_block(old_state, m, km, kp)
        counts += block["pair_counts"]
        shots += 2 * m
        old.merge_row(rows, block["arm"], m, kp)
        mirror = dict(block["arm"])
        mirror["sin_numerator"] *= -1
        old.merge_row(rows, mirror, m, km)
        if index + 1 < len(history) and history[index + 1]["look"] == block["look"]:
            continue
        assert shots == block["budget"]
        b, c = map(int, counts[1:3])
        beta_log = float(gammaln(b + .5) + gammaln(c + .5) - gammaln(b + c + 1)
                         - 2 * gammaln(.5) + (b + c) * math.log(2))
        flogs = float(old_state["score"]) * old_state["log_odds"] - old_state["cgf"]
        fdirectional = np.array([logsumexp(np.r_[flogs[p], flogs[zero]] + logw)
                                 for p in (positive, negative)])
        fisher_log = float(logsumexp(fdirectional) - math.log(2))
        hedge_log = float(np.logaddexp(beta_log, fisher_log) - math.log(2))
        variants = {"fisher71_control": (fisher_log, int(np.sign(b - c)), None),
                    "hedge_control": (hedge_log, int(np.sign(b - c)), None)}
        for mode in modes:
            direction_logs = logsumexp(capitals[mode], axis=0) - math.log(len(SHRINKAGES))
            mix_log = float(logsumexp(direction_logs) - math.log(2))
            mix_direction = 1 if direction_logs[0] > direction_logs[1] else -1
            variants[mode + "_mix"] = (mix_log, mix_direction, dict(mode=mode, shrink="mixture"))
            for j, shrink in enumerate(SHRINKAGES):
                value = float(logsumexp(capitals[mode][j]) - math.log(2))
                direction = 1 if capitals[mode][j, 0] > capitals[mode][j, 1] else -1
                variants[mode + "_k" + str(shrink).replace("/", "_")] = (value, direction, dict(mode=mode, shrink=str(shrink)))
            for fraction in (.25, .5):
                combined = float(np.logaddexp(math.log(fraction) + mix_log, math.log(1 - fraction) + hedge_log))
                # All branches agreeing proves that this is also the direction
                # of the larger normalized directional mixture capital. This
                # avoids numerically evaluating one-sided incomplete Beta.
                agreed = mix_direction if mix_direction == int(np.sign(b - c)) else 0
                variants[mode + "_hedge_w" + str(fraction)] = (combined, agreed,
                    dict(mode=mode, shrink="mixture", adaptive_weight=fraction))
        for name, (value, direction, config) in variants.items():
            if name not in stopped and value >= math.log(100) and direction:
                stopped[name] = {"shots": shots, "stop_block": index, "direction": direction,
                                  "log_evidence": value, "config": config,
                                  "certificate_status": "floating_screen_only"}
    for name in variants:
        stopped.setdefault(name, {"shots": None, "available_history_shots": shots,
                                  "right_censored": True, "certificate_status": "not_crossed"})
    return stopped


def main(args):
    started = time.perf_counter()
    directory = ROOT / "results/sota_round2" / args.output
    directory.mkdir(parents=True, exist_ok=True)
    source = ROOT / "results/structured_validation/refined_development_v1"
    sources = old.source_hashes()
    for path in (Path(__file__).resolve(), Path(__file__).resolve().with_name("adaptive_safe_replay.py")):
        sources[path.relative_to(ROOT).as_posix()] = digest(path)
    inputs = {name: digest(source / name) for name in ("spec.json", "summary.json", "trials.jsonl", "audit.json")}
    spec = json.loads((source / "spec.json").read_text())
    case_ids = {case["case_id"] for case in spec["cases"]}
    assert len(case_ids) == 24 and all(case["kind"] == "main" and case["C"] > 0 for case in spec["cases"])
    write_new(directory / "spec.json", {"status": "development deterministic replay only",
        "created_utc": datetime.now(timezone.utc).isoformat(), "sources": sources,
        "input_directory": source.relative_to(ROOT).as_posix(), "input_sha256": inputs,
        "repetitions": args.repetitions, "case_ids": sorted(case_ids),
        "modes": ["past", "total"], "shrinkages": [str(k) for k in SHRINKAGES],
        "hybrid_adaptive_weights": [.25, .5], "delta": .01,
        "decision": "M=(Eplus+Eminus)/2 >=100; argmax directional capital; hybrids require component direction agreement",
        "total_conditioning": "arm and block size fixed before observations; odds depend only on past rows, arm, m and current total T",
        "limits": "Floating screening only; all old 24 cases and deterministic rep prefix; no new validation observations."})
    groups = defaultdict(lambda: {"history": [], "old_costs": {}})
    with (source / "trials.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            r = json.loads(line)
            if r["policy"] != "closed" or r["rep"] >= args.repetitions:
                continue
            group = groups[r["case_id"], r["rep"]]
            group["old_costs"][r["engine"]] = r["shots"]
            group["seed"] = r["seed"]
            if len(r["history"]) > len(group["history"]):
                group["history"] = r["history"]
    assert set(groups) == {(case, rep) for case in case_ids for rep in range(args.repetitions)}
    records = []
    for (case, rep), group in sorted(groups.items()):
        output = replay(group["history"])
        matches = {engine: output[engine + "_control"]["shots"] == group["old_costs"]["fisher" if engine == "fisher71" else engine]
                   for engine in ("fisher71", "hedge")}
        records.append({"case_id": case, "rep": rep, "seed": group["seed"], "variants": output,
                        "old_costs": group["old_costs"], "control_matches": matches})
        print(json.dumps({"case_id": case, "rep": rep, "groups": len(records), "controls": matches}), flush=True)
    summaries = []
    for name in sorted(records[0]["variants"]):
        ratios, censored, wrong = [], 0, 0
        for case in sorted(case_ids):
            local = [r for r in records if r["case_id"] == case]
            costs = []
            for r in local:
                item = r["variants"][name]
                if item["shots"] is None:
                    censored += 1
                    costs.append(item["available_history_shots"])
                else:
                    costs.append(item["shots"])
                    wrong += item["direction"] != 1
            ratios.append(sum(costs) / sum(r["old_costs"]["hedge"] for r in local))
        summaries.append({"variant": name, "geometric_mean_cost_ratio_or_lower_bound": math.exp(sum(map(math.log, ratios)) / len(ratios)),
                          "right_censored": censored, "floating_wrong_stops": wrong,
                          "metric_status": "lower_bound_due_to_censoring" if censored else "floating_screen_only"})
    assert inputs == {name: digest(source / name) for name in inputs}
    assert sources == {name: digest(ROOT / name) for name in sources}
    result = {"status": "complete exploratory screening", "groups": len(records),
              "sources": sources, "input_sha256": inputs, "records": records,
              "all_controls_match": all(all(r["control_matches"].values()) for r in records),
              "summaries": sorted(summaries, key=lambda x: x["geometric_mean_cost_ratio_or_lower_bound"]),
              "elapsed_seconds": time.perf_counter() - started,
              "limits": "Selected old development paths, no new samples. Float crossings require outward certification and independent validation; censored entries are lower bounds only."}
    write_new(directory / "analysis.json", result)
    print(json.dumps({k: result[k] for k in ("status", "groups", "all_controls_match", "summaries", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="conditional_predictive_replay_v1")
    parser.add_argument("--repetitions", type=int, default=1, choices=range(1, 17))
    main(parser.parse_args())
