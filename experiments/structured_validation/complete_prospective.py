"""Complete only missing canonical groups, or run the unchanged full-data analysis."""
import argparse
from collections import defaultdict
import json
import os
import shutil
import time

import completion_common as completion


def run():
    amendment, plan, selected, unused_release = completion.validate_amendment()
    directory = completion.DIRECTORY
    review = completion.read_json(completion.REVIEW)
    completion.require(review["status"] == "passed", "Completion wrapper review must pass")
    for name, expected in review["sources"].items():
        completion.require(completion.digest(completion.ROOT / name) == expected, "Reviewed source changed")
    completion.require(time.time() < amendment["execution"]["cutoff_unix"], "Completion sampling cutoff reached")
    for name in ("spec.json", "selection.json", "release.json", "trials.jsonl", "progress.json",
                 "summary.json", "completion_run.json"):
        completion.require(not (directory / name).exists(), "Preserve completion artifact: " + name)
    prefix = completion.original_prefix(plan, selected)
    completion.require(prefix["sha256"] == amendment["original_files_sha256"][
        completion.relative(completion.BASE / "trials.jsonl")], "Unbound original prefix")
    completion.write_new(directory / "completion_run.json", {
        "status": "released to complete only the unchanged fixed sample matrix",
        "created_utc": completion.frozen_common.utc_now(),
        "amendment_sha256": completion.digest(completion.AMENDMENT),
        "wrapper_review": {"path": completion.relative(completion.REVIEW), "sha256": completion.digest(completion.REVIEW)},
        "original_records": 7937, "remaining_records": 1663})
    for name in ("spec.json", "selection.json", "release.json"):
        with (directory / name).open("xb") as fresh, (completion.BASE / name).open("rb") as original:
            shutil.copyfileobj(original, fresh)
    spec = completion.read_json(directory / "spec.json")
    count, summaries = prefix["records"], prefix["summaries"]
    completed, boundaries = list(prefix["completed_case_ids"]), dict(prefix["boundaries"])
    current_case, status, error = prefix["next_group"]["case"]["case_id"], "running", None
    start, timing = time.perf_counter(), defaultdict(float)

    def progress(stream):
        stream.flush()
        value = {"status": status, "updated_utc": completion.frozen_common.utc_now(),
                 "completed_case_ids": completed, "completed_case_ledger_bytes": boundaries,
                 "flushed_ledger_bytes": stream.tell(), "completed_method_trajectories": count,
                 "current_case_id": current_case, "expected_method_trajectories": 9600,
                 "elapsed_seconds": time.perf_counter() - start,
                 "elapsed_seconds_scope": "completion phase only", "error": error}
        pending = directory / "progress.pending.json"
        pending.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        os.replace(str(pending), str(directory / "progress.json"))

    with (directory / "trials.jsonl").open("xb") as stream:
        with (completion.BASE / "trials.jsonl").open("rb") as original:
            shutil.copyfileobj(original, stream)
        completion.require(stream.tell() == prefix["bytes"], "Incomplete old-prefix copy")
        progress(stream)
        try:
            for index, group in enumerate(completion.layout(plan, selected)):
                if index < prefix["groups_consumed"]:
                    continue
                current_case = group["case"]["case_id"]
                completion.require(time.time() < amendment["execution"]["cutoff_unix"], "Completion sampling cutoff reached")
                cap = plan["cap_by_kind"][group["case"]["kind"]]
                if group["family"] == "refined":
                    records, spent = completion.frozen_run.refined.run_trial(
                        group["case"], group["policy"], group["rep"], group["seed"], cap, group["engines"])
                else:
                    row, spent = completion.frozen_run.unequal.run_trial(
                        group["case"], group["policy"], group["rep"], group["seed"], cap,
                        amendment["execution"]["cutoff_unix"])
                    records = [row]
                completion.validate_group(records, group, plan)
                for key, value in spent.items():
                    timing[key] += value
                for row in records:
                    stream.write((json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8"))
                    summaries.append({k: row[k] for k in ("case_id", "method", "rep", "seed", "shots", "decision")})
                    count += 1
                stream.flush()
                if group["case_end"]:
                    completed.append(current_case)
                    boundaries[str(current_case)] = stream.tell()
                    progress(stream)
                    print(json.dumps({"completed_case": current_case, "completed_records": count,
                                      "new_records": count - 7937, "elapsed_seconds": time.perf_counter() - start}), flush=True)
            completion.require(count == 9600, "Completion matrix is incomplete")
            completion.validate_amendment()
            status, current_case = "complete", None
        except Exception as caught:
            status = "incomplete_cutoff" if time.time() >= amendment["execution"]["cutoff_unix"] else "incomplete_error"
            error = type(caught).__name__ + ": " + str(caught)
            progress(stream)
            raise
        progress(stream)
    result = {"status": "complete", "spec": spec,
              "cells": completion.frozen_common.build_cells(summaries, plan["cases"], selected["methods"]),
              "new_trajectories": count, "joint_replicate_labels": plan["expected_joint_replicate_labels"],
              "additional_trajectories": count - 7937, "original_trajectories": 7937,
              "timings": dict(timing), "timings_scope": "completion phase only",
              "elapsed_seconds": time.perf_counter() - start,
              "trials_sha256": completion.digest(directory / "trials.jsonl")}
    completion.write_new(directory / "summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("spec", "cells")}, indent=2), flush=True)


def analyze():
    amendment, unused_plan, unused_selected, unused_release = completion.validate_amendment()
    completion.require(time.time() < amendment["execution"]["hard_deadline_unix"], "Completion hard deadline reached")
    import analyze as frozen_analyze
    completion.frozen_common.DIRECTORY = completion.DIRECTORY
    result = frozen_analyze.analyze()
    completion.write_new(completion.DIRECTORY / "analysis_provenance.json", {
        "status": "unchanged frozen complete-data analysis run in explicit time-extension directory",
        "sources": {completion.relative(completion.frozen_common.HERE / "analyze.py"):
                    completion.digest(completion.frozen_common.HERE / "analyze.py")},
        "input_sha256": {name: completion.digest(completion.DIRECTORY / name) for name in
                         ("amendment.json", "completion_run.json", "analysis.json")},
        "claim_boundary": result["claim_boundary"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "analyze"))
    args = parser.parse_args()
    run() if args.command == "run" else analyze()
