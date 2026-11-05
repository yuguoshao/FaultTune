"""Development-only replay of predictable full-count safe factors.

Only previously observed development histories are consumed. This is a
screening experiment, not an independent validation or an anytime-valid
floating-point implementation. Candidate stops require a later outward
certificate before any scientific performance claim.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "experiments/structured_validation"
sys.path.insert(0, str(OLD))
import run_refined as old
import safe_grid as safe
import weighted_fisher as fisher
from predictive import fit_rows

DENOMINATOR = 2 ** 40
DELTA = .01
SHRINKAGES = (Fraction(1, 4), Fraction(1, 2), Fraction(1), Fraction(2), Fraction(4))
STARTS = ("immediate", "shots32", "shots128", "minority4", "minority16")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(data)
    return value.hexdigest()


def write_new(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def start_ready(name, shots, successes):
    if name == "immediate":
        return True
    if name.startswith("shots"):
        return shots >= int(name[5:])
    if name.startswith("minority"):
        return min(successes, shots - successes) >= int(name[8:])
    raise ValueError(name)


def rational_prediction(rows, arm, shrinkage):
    """All fitted quantities use past paid rows; no case parameter is accepted."""
    theta = fit_rows(rows)["theta"]
    co = arm["cos_numerator"] / arm["denominator"]
    si = arm["sin_numerator"] / arm["denominator"]
    center = float(theta[0] + theta[1] * co)
    center = min(1 - 2 / DENOMINATOR, max(2 / DENOMINATOR, center))
    gap = min(float(shrinkage) * abs(float(theta[2] * si)),
              .999 * min(center, 1 - center))
    low = min(DENOMINATOR - 1, max(1, int(round((center - gap) * DENOMINATOR))))
    high = min(DENOMINATOR - 1, max(low, int(round((center + gap) * DENOMINATOR))))
    return low, high, DENOMINATOR


def log_factor_increment(m, minus, plus, prediction):
    """Log of likelihood(u,v) / likelihood((u+v)/2,(u+v)/2)."""
    low, high, denominator = prediction
    u, v = low / denominator, high / denominator
    center = (low + high) / (2 * denominator)
    lu, lv = math.log(u / center), math.log(v / center)
    lfu, lfv = math.log((1 - u) / (1 - center)), math.log((1 - v) / (1 - center))
    positive = minus * lu + (m - minus) * lfu + plus * lv + (m - plus) * lfv
    negative = minus * lv + (m - minus) * lfv + plus * lu + (m - plus) * lfu
    return positive, negative


def fixed_increments(block, poles):
    prediction = block["safe_prediction"]
    assert prediction["odds_poles"] == [[o.numerator, o.denominator] for o in poles]
    counts = np.array(safe._pair_counts(block["m"], block["minus_successes"],
                                      block["plus_successes"]), dtype=float)
    increments = []
    for numerator, odds in zip(prediction["lower_numerators"], poles):
        logs = safe._float_log_factors(numerator, prediction["lower_denominator"],
                                       odds.numerator, odds.denominator)
        increments.append((float(np.dot(counts, logs)), float(np.dot(counts, logs[[0, 2, 1, 3]]))))
    return np.asarray(increments)


def replay(history):
    config = safe.protocol_config(fisher.protocol_config())
    experts, poles, unused = safe._layout(config)
    weights = dict(experts)
    assert all(weights[o] == weights[1 / o] for o in poles)
    neutral = float(weights.get(Fraction(1), Fraction(0)))
    directional_weights = np.array([2 * float(weights[o]) for o in poles] + [neutral])
    assert abs(directional_weights.sum() - 1) < 1e-12
    log_fixed_weights = np.log(directional_weights, where=directional_weights > 0,
                               out=np.full_like(directional_weights, -np.inf))
    variants = {}
    for start in STARTS:
        variants["fixed_" + start] = dict(family="fixed", start=start,
                                           capital=np.zeros((len(poles) + 1, 2)))
        for shrink in SHRINKAGES:
            name = "plugin_" + start + "_k" + str(shrink).replace("/", "_")
            variants[name] = dict(family="plugin", start=start, shrink=shrink,
                                  capital=np.zeros((1, 2)))
        variants["pluginmix_" + start] = dict(family="pluginmix", start=start,
                                               capital=np.zeros((len(SHRINKAGES), 2)))
    stopped, rows = {}, []
    shots = successes = 0
    for index, block in enumerate(history):
        m, minus, plus = block["m"], block["minus_successes"], block["plus_successes"]
        fixed = fixed_increments(block, poles)
        predictions = {shrink: rational_prediction(rows, block["arm"], shrink) for shrink in SHRINKAGES}
        plugin = {shrink: log_factor_increment(m, minus, plus, prediction)
                  for shrink, prediction in predictions.items()}
        for name, state in variants.items():
            if not start_ready(state["start"], shots, successes):
                continue
            state.setdefault("first_bet_block", index)
            if state["family"] == "fixed":
                state["capital"][:-1] += fixed
            elif state["family"] == "plugin":
                state["capital"][0] += plugin[state["shrink"]]
            else:
                state["capital"] += [plugin[shrink] for shrink in SHRINKAGES]
        shots += 2 * m
        successes += minus + plus
        old.merge_row(rows, block["arm"], m, plus)
        negative = dict(block["arm"])
        negative["sin_numerator"] *= -1
        old.merge_row(rows, negative, m, minus)
        checkpoint = index + 1 == len(history) or history[index + 1]["look"] != block["look"]
        if not checkpoint:
            continue
        assert shots == block["budget"]
        for name, state in variants.items():
            if name in stopped:
                continue
            lw = log_fixed_weights if state["family"] == "fixed" else np.full(len(state["capital"]), -math.log(len(state["capital"])))
            directional = logsumexp(state["capital"] + lw[:, None], axis=0)
            log_evidence = float(logsumexp(directional) - math.log(2))
            if log_evidence >= math.log(1 / DELTA):
                stopped[name] = {"shots": shots, "first_bet_block": state.get("first_bet_block"),
                                 "stop_block": index, "log_evidence": log_evidence,
                                 "direction": 1 if directional[0] > directional[1] else -1,
                                 "directional_log_capital": list(map(float, directional)),
                                 "certificate_status": "floating_screen_only"}
    return {name: stopped.get(name, {"shots": None, "available_history_shots": shots,
                                     "right_censored": True, "certificate_status": "not_crossed"})
            for name in variants}


def main(args):
    started = time.perf_counter()
    directory = ROOT / "results/sota_round2" / args.output
    directory.mkdir(parents=True, exist_ok=True)
    source = ROOT / "results/structured_validation/refined_development_v1"
    sources = old.source_hashes()
    sources[Path(__file__).resolve().relative_to(ROOT).as_posix()] = digest(__file__)
    inputs = {name: digest(source / name) for name in ("spec.json", "summary.json", "trials.jsonl", "audit.json")}
    spec = json.loads((source / "spec.json").read_text())
    case_ids = {case["case_id"] for case in spec["cases"]}
    assert len(case_ids) == 24
    assert all(case["kind"] == "main" and case["C"] > 0 for case in spec["cases"])
    write_new(directory / "spec.json", {"status": "development deterministic replay only",
        "created_utc": datetime.now(timezone.utc).isoformat(), "sources": sources,
        "input_directory": source.relative_to(ROOT).as_posix(), "input_sha256": inputs,
        "repetitions": args.repetitions, "case_ids": sorted(case_ids), "starts": list(STARTS),
        "shrinkages": [str(value) for value in SHRINKAGES], "delta": DELTA,
        "decision": "M=(Eplus+Eminus)/2 crosses 100; direction is argmax normalized directional capital",
        "selection": "all 24 old development cases, rep range(0,repetitions); no new validation data",
        "limits": "Floating screens only. Right-censored beyond longest stored same-policy history. No SOTA or performance claim."})
    groups = defaultdict(lambda: {"history": [], "old_costs": {}})
    with (source / "trials.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record["policy"] != "closed" or record["rep"] >= args.repetitions:
                continue
            group = groups[record["case_id"], record["rep"]]
            group["old_costs"][record["engine"]] = record["shots"]
            group["seed"] = record["seed"]
            if len(record["history"]) > len(group["history"]):
                group["history"] = record["history"]
    assert set(groups) == {(case, rep) for case in case_ids for rep in range(args.repetitions)}
    records = []
    for (case_id, rep), group in sorted(groups.items()):
        assert len(group["old_costs"]) == 6
        output = replay(group["history"])
        # The fixed immediate variant must agree with the old certified safe
        # stop on the identical history, modulo a reported numerical discrepancy.
        baseline_match = output["fixed_immediate"]["shots"] == group["old_costs"]["safe"]
        records.append({"case_id": case_id, "rep": rep, "seed": group["seed"],
                        "old_costs": group["old_costs"], "variants": output,
                        "fixed_control_stop_match": baseline_match})
        print(json.dumps({"case_id": case_id, "rep": rep, "groups_completed": len(records),
                          "control_match": baseline_match}), flush=True)
    summaries = []
    for name in sorted(records[0]["variants"]):
        ratios, censored, wrong = [], 0, 0
        for case_id in sorted(case_ids):
            local = [r for r in records if r["case_id"] == case_id]
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
        ratio = math.exp(sum(map(math.log, ratios)) / len(ratios))
        summaries.append({"variant": name, "geometric_mean_cost_ratio_or_lower_bound": ratio,
                          "right_censored": censored, "floating_wrong_stops": wrong,
                          "metric_status": "lower_bound_due_to_censoring" if censored else "floating_screen_only"})
    assert inputs == {name: digest(source / name) for name in inputs}
    assert sources == {name: digest(ROOT / name) for name in sources}
    result = {"status": "complete exploratory screening", "groups": len(records),
              "sources": sources, "input_sha256": inputs,
              "all_fixed_control_stops_match": all(r["fixed_control_stop_match"] for r in records),
              "summaries": sorted(summaries, key=lambda x: x["geometric_mean_cost_ratio_or_lower_bound"]),
              "records": records, "elapsed_seconds": time.perf_counter() - started,
              "limits": "No new sampling; floating evidence and deterministic selected development paths only. Require certification and new independent validation."}
    write_new(directory / "analysis.json", result)
    print(json.dumps({k: v for k, v in result.items() if k in ("status", "groups", "all_fixed_control_stops_match", "summaries", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="adaptive_safe_replay_v1")
    parser.add_argument("--repetitions", type=int, default=1, choices=range(1, 17))
    main(parser.parse_args())
