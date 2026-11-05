"""Read-only replay of one stored refined/unequal case-policy-replicate group.

This utility generates no new seeds, changes no results and performs no new
method selection. It also works after the original experiment deadline.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import run_refined as refined
import unequal_runner as unequal

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def without_timing(value):
    """Planner wall-clock diagnostics are not protocol inputs or costs."""
    if isinstance(value, dict):
        return {key: without_timing(item) for key, item in value.items() if key != "elapsed_seconds"}
    if isinstance(value, list):
        return [without_timing(item) for item in value]
    return value


def replay(directory, case_id, policy, rep):
    started = time.perf_counter()
    directory = Path(directory).resolve()
    inputs = {name: digest(directory/name) for name in ("spec.json", "summary.json", "trials.jsonl")}
    spec = json.loads((directory/"spec.json").read_text(encoding="utf-8"))
    for name, expected in spec["sources"].items():
        assert digest(ROOT/name) == expected, "Frozen source differs: "+name
    records = []
    with (directory/"trials.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if (row["case_id"], row["policy"], row["rep"]) == (case_id, policy, rep):
                records.append(row)
    assert records, "No stored records with the requested identity."
    cases = [c for c in spec["cases"] if c["case_id"] == case_id]
    assert len(cases) == 1
    case = cases[0]
    seeds = {row["seed"] for row in records}
    assert len(seeds) == 1
    seed = seeds.pop()
    assert seed == spec["seed_base"]+1009*case_id+1000003*rep
    cap = spec["cap_by_kind"][case["kind"]] if "cap_by_kind" in spec else spec["cap"]
    engines = [row["engine"] for row in records]
    assert len(engines) == len(set(engines)), "Duplicate stored engine."
    refined_engines = [engine for engine in engines if engine != "weighted_conditional_unequal"]
    if "engines" in spec:
        refined_engines = [engine for engine in spec["engines"] if engine in refined_engines]
    else:
        refined_engines.sort()
    rebuilt = []
    if refined_engines:
        assert "experiments/structured_validation/run_refined.py" in spec["sources"], "Dataset is not a refined protocol."
        assert set(refined_engines) <= set(refined.ENGINES), "Unsupported evidence family."
        local, unused = refined.run_trial(case, policy, rep, seed, cap, tuple(refined_engines))
        rebuilt.extend(local)
    if "weighted_conditional_unequal" in engines:
        assert "experiments/structured_validation/unequal_runner.py" in spec["sources"], "Dataset is not an unequal protocol."
        local, unused = unequal.run_trial(case, policy, rep, seed, cap)
        rebuilt.append(local)
    ordered = lambda values: sorted(values, key=lambda row: row["method"])
    assert without_timing(ordered(records)) == without_timing(ordered(rebuilt)), "Stored trajectory differs from frozen seeded replay."
    assert inputs == {name: digest(directory/name) for name in inputs}, "Stored inputs changed during replay."
    for name, expected in spec["sources"].items():
        assert digest(ROOT/name) == expected, "Frozen source changed during replay: "+name
    record_hash = hashlib.sha256(json.dumps(ordered(records), sort_keys=True,
                                           separators=(",", ":")).encode("utf-8")).hexdigest()
    result = {"status": "passed", "case_id": case_id, "policy": policy, "rep": rep,
              "seed": seed, "method_trajectories_replayed": len(records),
              "records_sha256": record_hash, "input_sha256": inputs,
              "comparison_exclusions": ["elapsed_seconds: nondeterministic planner wall-clock diagnostics only"],
              "elapsed_seconds": time.perf_counter()-started,
              "scope": "One existing stored group; not a replacement for the full dataset audit."}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--case-id", type=int, required=True)
    parser.add_argument("--policy", choices=refined.POLICIES, required=True)
    parser.add_argument("--rep", type=int, required=True)
    args = parser.parse_args()
    replay(args.directory, args.case_id, args.policy, args.rep)
