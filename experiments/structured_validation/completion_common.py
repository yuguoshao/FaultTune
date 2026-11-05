"""Explicit administrative extension of the unchanged prospective sample matrix.

Old files and algorithm sources are immutable. New data live in a new directory
whose ledger begins with the byte-identical old ledger. This module does not
silently alter the old run control or pretend the old executor made new data.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/structured_validation/prospective"))
import common as frozen_common
import run as frozen_run

BASE = ROOT / "results/structured_validation/prospective_v1"
DIRECTORY = ROOT / "results/structured_validation/prospective_completion_v1"
AMENDMENT = DIRECTORY / "amendment.json"
REVIEW = ROOT / "results/structured_validation/completion_wrapper_review.json"
SOURCE_NAMES = ("completion_common.py", "complete_prospective.py", "audit_prospective_completion.py")
START_UTC = "2026-11-04T08:22:02Z"
CUTOFF_UTC = "2026-11-04T09:32:02Z"
DEADLINE_UTC = "2026-11-04T09:52:02Z"
digest, read_json, write_new, require = (frozen_common.digest, frozen_common.read_json,
                                      frozen_common.write_new, frozen_common.require)


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def source_hashes():
    return {relative(Path(__file__).with_name(name)): digest(Path(__file__).with_name(name))
            for name in SOURCE_NAMES}


def original_release(recheck_development=True):
    previous = frozen_common.DIRECTORY
    try:
        frozen_common.DIRECTORY = BASE
        return frozen_common.validate_release(recheck_development=recheck_development)
    finally:
        frozen_common.DIRECTORY = previous


def layout(plan, selected):
    groups = frozen_run.execution_groups(selected["methods"])
    for case in plan["cases"]:
        reps = plan["repetitions_by_kind"][case["kind"]]
        for rep in range(reps):
            for index, (policy, family, engines) in enumerate(groups):
                methods = tuple(engines) if family == "unequal" else tuple(policy + "_" + e for e in engines)
                yield {"case": case, "rep": rep, "policy": policy, "family": family,
                       "engines": engines, "methods": methods,
                       "seed": plan["seed_base"] + 1009 * case["case_id"] + 1000003 * rep,
                       "case_end": rep == reps - 1 and index == len(groups) - 1}


def validate_group(records, group, plan):
    require(len(records) == len(group["methods"]), "Incomplete execution group")
    require({r["method"] for r in records} == set(group["methods"]), "Duplicate or unexpected method")
    for row in records:
        require(row["case_id"] == group["case"]["case_id"] and row["rep"] == group["rep"]
                and row["seed"] == group["seed"] and row["policy"] == group["policy"],
                "Record is outside its frozen case/replicate/seed/policy group")
        cap = plan["cap_by_kind"][group["case"]["kind"]]
        require(type(row["shots"]) is int and 0 < row["shots"] <= cap, "Invalid recorded cost")
        require(row["decision"] in ("accept", "reject", "abstain"), "Invalid decision")
        require(row["decision"] != "abstain" or row["shots"] == cap, "Abstention before cap")


def read_prefix(stream, plan, selected):
    """Read only whole canonical execution groups; reject even a one-row gap.

Within a group the original runner orders methods by their stopping event,
so membership, rather than lexicographic line order, is the correct invariant.
"""
    summaries, complete, boundaries = [], [], {}
    count, offset, consumed = 0, 0, 0
    sha = hashlib.sha256()
    for group in layout(plan, selected):
        rows = []
        for unused in group["methods"]:
            line = stream.readline()
            if not line:
                require(not rows, "Prefix ends inside an execution group")
                return {"summaries": summaries, "records": count, "bytes": offset,
                        "sha256": sha.hexdigest(), "groups_consumed": consumed,
                        "completed_case_ids": complete, "boundaries": boundaries,
                        "next_group": group}
            require(line.endswith(b"\n"), "Unterminated original ledger record")
            sha.update(line)
            offset += len(line)
            rows.append(json.loads(line.decode("utf-8")))
        validate_group(rows, group, plan)
        summaries.extend({k: row[k] for k in ("case_id", "method", "rep", "seed", "shots", "decision")}
                         for row in rows)
        count += len(rows)
        consumed += 1
        if group["case_end"]:
            complete.append(group["case"]["case_id"])
            boundaries[str(group["case"]["case_id"])] = offset
    require(not stream.read(1), "Unexpected records beyond the frozen sample matrix")
    return {"summaries": summaries, "records": count, "bytes": offset, "sha256": sha.hexdigest(),
            "groups_consumed": consumed, "completed_case_ids": complete, "boundaries": boundaries,
            "next_group": None}


def original_prefix(plan, selected):
    with (BASE / "trials.jsonl").open("rb") as stream:
        prefix = read_prefix(stream, plan, selected)
    progress = read_json(BASE / "progress.json")
    require(progress["status"] == "incomplete_cutoff", "Expected stopped original run")
    require(prefix["records"] == progress["completed_method_trajectories"] == 7937,
            "Original record count changed")
    require(prefix["bytes"] == progress["flushed_ledger_bytes"], "Original byte count differs")
    require(prefix["completed_case_ids"] == progress["completed_case_ids"] and
            prefix["boundaries"] == progress["completed_case_ledger_bytes"], "Original case boundaries differ")
    require(prefix["next_group"] is not None and prefix["next_group"]["case"]["case_id"] ==
            progress["current_case_id"], "Original continuation position differs")
    return prefix


def validate_amendment(recheck_original=True):
    amendment = read_json(AMENDMENT)
    plan, selected, release = original_release(recheck_development=recheck_original)
    require(amendment["original_plan_sha256"] == frozen_common.PLAN_SHA256, "Amended scientific plan")
    require(amendment["algorithm_sources_sha256"] == plan["algorithm_sources_sha256"], "Changed algorithm set")
    require(amendment["original_wrapper_sources_sha256"] == release["wrapper_sources_sha256"], "Changed old wrappers")
    require(amendment["completion_sources_sha256"] == source_hashes(), "Completion source changed after release")
    require(amendment["original_directory"] == relative(BASE), "Changed original directory")
    require(set(amendment["original_files_sha256"]) ==
            {relative(path) for path in BASE.rglob("*") if path.is_file()}, "Changed original artifact inventory")
    original_progress = read_json(BASE / "progress.json")
    require(amendment["original_ledger_bytes"] == original_progress["flushed_ledger_bytes"] and
            amendment["original_complete_case_ids"] == original_progress["completed_case_ids"],
            "Original amendment metadata differs from stopped progress")
    execution = amendment["execution"]
    require(execution["started_utc"] == START_UTC and execution["cutoff_utc"] == CUTOFF_UTC and
            execution["hard_deadline_utc"] == DEADLINE_UTC and execution["max_minutes"] == 90,
            "Administrative window changed")
    require(execution["cutoff_unix"] == timestamp(CUTOFF_UTC) and
            execution["hard_deadline_unix"] == timestamp(DEADLINE_UTC) and
            timestamp(DEADLINE_UTC) - timestamp(START_UTC) == 90 * 60, "Invalid administrative clock")
    expected = plan["expected_joint_replicate_labels"] * len(selected["methods"])
    require(amendment["original_records"] == 7937 and amendment["expected_records"] == expected == 9600 and
            amendment["remaining_records"] == expected - amendment["original_records"] == 1663,
            "Amended sample count")
    if recheck_original:
        for name, expected_hash in amendment["original_files_sha256"].items():
            require(digest(ROOT / name) == expected_hash, "Original artifact changed: " + name)
    return amendment, plan, selected, release


def prepare():
    require(not AMENDMENT.exists(), "Preserve the existing time amendment")
    plan, selected, release = original_release()
    prefix = original_prefix(plan, selected)
    originals = [path for path in BASE.rglob("*") if path.is_file()]
    result = {
        "version": "structured_prospective_completion_v1",
        "status": "user-authorized administrative time extension; scientific sample matrix unchanged",
        "authorization": "User answered: 追加最多90分钟; only complete the remaining fixed validation, audit and report.",
        "created_utc": frozen_common.utc_now(),
        "execution": {"started_utc": START_UTC, "cutoff_utc": CUTOFF_UTC,
                      "cutoff_unix": timestamp(CUTOFF_UTC), "hard_deadline_utc": DEADLINE_UTC,
                      "hard_deadline_unix": timestamp(DEADLINE_UTC), "max_minutes": 90},
        "original_plan_sha256": frozen_common.PLAN_SHA256,
        "original_directory": relative(BASE),
        "original_files_sha256": {relative(path): digest(path) for path in sorted(originals)},
        "algorithm_sources_sha256": plan["algorithm_sources_sha256"],
        "original_wrapper_sources_sha256": release["wrapper_sources_sha256"],
        "completion_sources_sha256": source_hashes(),
        "original_records": prefix["records"], "expected_records": 9600, "remaining_records": 1663,
        "original_ledger_bytes": prefix["bytes"], "original_complete_case_ids": prefix["completed_case_ids"],
        "original_execution_groups": prefix["groups_consumed"],
        "scope": "No changed candidate, comparator, observations per case, seeds, cap, risk, stopping algorithm or primary analysis. No partial primary comparison was computed. The old deadline and partial artifacts remain intact; this separate phase is disclosed in the final report."}
    write_new(AMENDMENT, result)
    validate_amendment()
    print(json.dumps({"amendment": str(AMENDMENT), "sha256": digest(AMENDMENT),
                      "original_records": prefix["records"], "remaining_records": 1663}, indent=2), flush=True)


if __name__ == "__main__":
    prepare()
