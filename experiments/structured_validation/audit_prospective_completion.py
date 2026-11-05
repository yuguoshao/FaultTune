"""Audit an explicitly authorized administrative continuation of the fixed study.

The old raw prefix and case audits remain unchanged. Old complete case audits
are reused only after their source bindings and exact raw-byte segments are
verified. Case 41 onward is audited with the original protocol auditor. The
only in-process adaptations are its administrative deadline hook and the
destination used by frozen common helpers; both are reported explicitly.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import completion_common as gate

sys.path.insert(0, str(Path(__file__).resolve().parent/"prospective"))
import audit as original_audit

ROOT = gate.ROOT
ORIGINAL = ROOT/"results/structured_validation/prospective_v1"
DIRECTORY = gate.DIRECTORY
require = gate.require


def unchanged_copies():
    for name in ("spec.json", "selection.json", "release.json"):
        require((DIRECTORY/name).read_bytes() == (ORIGINAL/name).read_bytes(),
                "Administrative continuation changed the original "+name)


def prefix_hash(path, size):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        remaining = size
        while remaining:
            block = stream.read(min(1024*1024, remaining))
            require(bool(block), "Raw ledger is shorter than its required old prefix")
            value.update(block)
            remaining -= len(block)
    return value.hexdigest()


def verify_raw_prefix(amendment):
    old_path, new_path = ORIGINAL/"trials.jsonl", DIRECTORY/"trials.jsonl"
    size = amendment["original_ledger_bytes"]
    old_name = old_path.relative_to(ROOT).as_posix()
    expected = amendment["original_files_sha256"][old_name]
    require(old_path.stat().st_size == size, "Original raw byte length changed")
    require(gate.digest(old_path) == expected, "Original raw prefix hash changed")
    require(prefix_hash(new_path, size) == expected, "Continuation raw prefix is not byte-identical")
    # Also compare actual bytes; digest equality is not used as a substitute here.
    with old_path.open("rb") as old_stream, new_path.open("rb") as new_stream:
        for block in iter(lambda: old_stream.read(1024*1024), b""):
            require(new_stream.read(len(block)) == block, "Continuation changed original raw bytes")
    return {"original_ledger_bytes": size, "input_sha256": {old_name: expected},
            "direct_byte_comparison_passed": True}


def current_sources(amendment):
    return {"original_audit_sources_sha256": original_audit.audit_hashes(),
            "completion_sources_sha256": amendment["completion_sources_sha256"]}


def check_sources(amendment, sources):
    current, plan, selected, release = gate.validate_amendment(recheck_original=False)
    require(current == amendment, "Administrative amendment changed during auditing")
    require(current_sources(current) == sources, "Audit implementation changed during auditing")
    unchanged_copies()
    return plan, selected, release


def bind_completion_run(amendment):
    path = DIRECTORY/"completion_run.json"
    record = gate.read_json(path)
    require(record["amendment_sha256"] == gate.digest(gate.AMENDMENT),
            "Completion run is not bound to this authorized amendment")
    require(record["status"] == "released to complete only the unchanged fixed sample matrix" and
            record["original_records"] == amendment["original_records"] and
            record["remaining_records"] == amendment["remaining_records"],
            "Completion producer record has a different release scope")
    review_path = ROOT/record["wrapper_review"]["path"]
    require(gate.digest(review_path) == record["wrapper_review"]["sha256"],
            "Completion wrapper review reference changed")
    review = gate.read_json(review_path)
    require(review["status"] == "passed" and all(review["sources"].get(name) == value
            for name, value in amendment["completion_sources_sha256"].items()),
            "Passed independent review does not cover all continuation sources")
    if "completion_sources_sha256" in record:
        require(record["completion_sources_sha256"] == amendment["completion_sources_sha256"],
                "Completion run producer hashes differ")
    return record


def reuse_complete_case_audits(amendment, plan, selected, sources):
    retained, offset = [], 0
    old_progress = gate.read_json(ORIGINAL/"progress.json")
    expected_ids = amendment["original_complete_case_ids"]
    require(expected_ids == old_progress["completed_case_ids"] and
            expected_ids == [c["case_id"] for c in plan["cases"][:len(expected_ids)]],
            "Original complete cases are not the frozen initial case sequence")
    with (DIRECTORY/"trials.jsonl").open("rb") as stream:
        for case_id in expected_ids:
            path = ORIGINAL/"audit_cases"/("case_%03d.json"%case_id)
            relative = path.relative_to(ROOT).as_posix()
            require(gate.digest(path) == amendment["original_files_sha256"][relative],
                    "Original case audit changed")
            item = gate.read_json(path)
            require(item["status"] == "passed_case" and item["case_id"] == case_id,
                    "Original case audit did not pass")
            require(item["audit_sources_sha256"] == sources["original_audit_sources_sha256"],
                    "Original case audit used different frozen audit sources")
            boundary = old_progress["completed_case_ledger_bytes"][str(case_id)]
            require(item["start_byte"] == offset and item["end_byte"] == boundary,
                    "Original case audit boundary differs")
            raw = stream.read(boundary-offset)
            require(len(raw) == boundary-offset and
                    hashlib.sha256(raw).hexdigest() == item["input_prefix_part_sha256"],
                    "Original case audit does not cover this exact prefix segment")
            case = plan["cases"][len(retained)]
            expected = plan["repetitions_by_kind"][case["kind"]]*len(selected["methods"])
            require(item["checks"]["terminal_records_checked"] == expected and
                    len(item["cost_rows"]) == expected, "Incomplete original case audit")
            copied = dict(item)
            copied["completion_audit_provenance"] = {
                "type": "reused_original_complete_case_audit",
                "original_case_audit": relative,
                "original_case_audit_sha256": amendment["original_files_sha256"][relative],
                "raw_segment_reverified": True,
                "protocol_audit_reexecuted_in_completion": False}
            gate.write_new(DIRECTORY/"audit_cases"/path.name, copied)
            retained.append(copied)
            offset = boundary
    return retained, offset


def finalize(amendment, plan, selected, sources, audited, offset, prefix, started):
    inputs = {name: gate.digest(DIRECTORY/name) for name in
              ("spec.json", "summary.json", "trials.jsonl", "analysis.json",
               "selection.json", "release.json", "amendment.json", "completion_run.json",
               "analysis_provenance.json")}
    spec, summary, analysis = (gate.read_json(DIRECTORY/name) for name in
                              ("spec.json", "summary.json", "analysis.json"))
    require(summary["status"] == "complete" and summary["spec"] == spec and
            summary["trials_sha256"] == inputs["trials.jsonl"], "Completion summary is not final or bound")
    require(spec["cases"] == plan["cases"] and spec["selection"] == selected and
            spec["plan_sha256"] == amendment["original_plan_sha256"] and
            spec["release_sha256"] == inputs["release.json"], "Original statistical design changed")
    require((DIRECTORY/"trials.jsonl").stat().st_size == offset,
            "Completion raw ledger contains unaudited trailing data")
    require(verify_raw_prefix(amendment) == prefix, "Old raw prefix changed after case audits")
    with (DIRECTORY/"trials.jsonl").open("rb") as stream:
        for item in audited:
            raw = stream.read(item["end_byte"]-item["start_byte"])
            require(hashlib.sha256(raw).hexdigest() == item["input_prefix_part_sha256"],
                    "Previously audited continuation case changed")
    cells = [cell for item in audited for cell in item["cells"]]
    require(cells == summary["cells"] == analysis["cells"],
            "Completed summary/analysis differs from unconditional audited cells")
    require(analysis["input_sha256"] == {name: inputs[name] for name in analysis["input_sha256"]},
            "Analysis is bound to different data")
    provenance = gate.read_json(DIRECTORY/"analysis_provenance.json")
    require(provenance["input_sha256"] == {name: inputs[name] for name in
            ("amendment.json", "completion_run.json", "analysis.json")},
            "Analysis provenance is not bound to the authorized continuation")
    analysis_source = gate.frozen_common.HERE/"analyze.py"
    require(provenance["sources"] == {analysis_source.relative_to(ROOT).as_posix(): gate.digest(analysis_source)},
            "Completion analysis did not record the unchanged frozen analysis source")
    original_audit.independently_check_analysis(audited, analysis, plan, selected)
    totals, reused, fresh = Counter(), Counter(), Counter()
    for item in audited:
        totals.update(item["checks"])
        target = reused if item["completion_audit_provenance"]["type"].startswith("reused_") else fresh
        target.update(item["checks"])
    require(totals["terminal_records_checked"] == summary["new_trajectories"] ==
            amendment["expected_records"] == plan["expected_joint_replicate_labels"]*len(selected["methods"]),
            "Final continuation trajectory count differs from the frozen sample size")
    require(len(audited) == len(plan["cases"]), "Not every frozen case is audited")
    require(inputs == {name: gate.digest(DIRECTORY/name) for name in inputs}, "Completion inputs changed")
    gate.validate_amendment(recheck_original=True)
    check_sources(amendment, sources)
    bind_completion_run(amendment)
    result = {
        "status": "passed",
        "scope": "Complete frozen statistical study under an explicitly authorized administrative continuation",
        "plan_sha256": amendment["original_plan_sha256"],
        "amendment": {"path": gate.AMENDMENT.relative_to(ROOT).as_posix(), "sha256": inputs["amendment.json"]},
        "input_sha256": inputs,
        "audit_sources_sha256": dict(sources["original_audit_sources_sha256"],
                                     **sources["completion_sources_sha256"]),
        "original_prefix": prefix,
        "audited_cases": len(audited), "checks": dict(totals),
        "reused_original_case_count": len(amendment["original_complete_case_ids"]),
        "reused_original_checks": dict(reused),
        "freshly_audited_case_count": len(audited)-len(amendment["original_complete_case_ids"]),
        "freshly_executed_checks": dict(fresh),
        "original_records": amendment["original_records"],
        "newly_generated_method_trajectories": amendment["remaining_records"],
        "old_unaudited_tail_records_now_audited": amendment["original_records"]-reused["terminal_records_checked"],
        "administrative_adaptations": [
            "Frozen common.DIRECTORY points at the separately named continuation directory in this process.",
            "Frozen audit.check_deadline is replaced in this process by the amendment hard deadline; no protocol computation is replaced.",
            "The new continuation wrapper, not the old run wrapper, produced the appended records."],
        "case_audits": [{"path": path.relative_to(ROOT).as_posix(), "sha256": gate.digest(path)}
                         for path in sorted((DIRECTORY/"audit_cases").glob("case_*.json"))],
        "all_logged_rng_blocks_replayed_or_bound_to_original_passed_audits": True,
        "all_paid_actions_and_scores_rebuilt_or_bound_to_original_passed_audits": True,
        "all_final_decisions_strictly_certified": True,
        "independent_bootstrap_rebuilt": True,
        "primary_pass": analysis["primary_numerical_threshold_met"],
        "limits": "Reused case audits are separately counted. Lazy decision proofs are not full tight-interval recomputations. Frozen normalizer primitives are reused. Fixed simulator matrix only; no full-field SOTA or novelty claim.",
        "elapsed_seconds": time.perf_counter()-started}
    gate.write_new(DIRECTORY/"audit.json", result)
    return result


def main(watch):
    amendment, plan, selected, release = gate.validate_amendment(recheck_original=True)
    require(original_audit.common is gate.frozen_common, "Frozen common module identity differs")
    gate.frozen_common.DIRECTORY = DIRECTORY
    deadline = amendment["execution"]["hard_deadline_unix"]

    def amended_deadline():
        if time.time() >= deadline:
            raise RuntimeError("Authorized continuation hard deadline reached; partial audit retained")

    original_audit.check_deadline = amended_deadline
    sources = current_sources(amendment)
    require(not (DIRECTORY/"audit.json").exists() and not (DIRECTORY/"audit_progress.json").exists(),
            "Preserve existing completion audit outputs")
    started = time.perf_counter()
    # A prepared clone may exist before sampling is released; wait for the actual producer record.
    while not (DIRECTORY/"completion_run.json").exists() or not (DIRECTORY/"progress.json").exists():
        amended_deadline()
        if not watch:
            return
        time.sleep(5)
    bind_completion_run(amendment)
    unchanged_copies()
    prefix = verify_raw_prefix(amendment)
    (DIRECTORY/"audit_cases").mkdir(exist_ok=True)
    require(not list((DIRECTORY/"audit_cases").glob("case_*.json")),
            "Use one completion auditor; existing case audits found")
    audited, offset = reuse_complete_case_audits(amendment, plan, selected, sources)

    def progress(status, error=None):
        value = {"status": status, "audited_case_ids": [r["case_id"] for r in audited],
                 "audited_ledger_bytes": offset, "amendment_sha256": gate.digest(gate.AMENDMENT),
                 "reused_original_case_count": len(amendment["original_complete_case_ids"]),
                 "freshly_audited_case_count": len(audited)-len(amendment["original_complete_case_ids"]),
                 "audit_sources": sources, "elapsed_seconds": time.perf_counter()-started, "error": error}
        pending = DIRECTORY/"audit_progress.pending.json"
        pending.write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")
        os.replace(str(pending), str(DIRECTORY/"audit_progress.json"))

    progress("running")
    print(json.dumps({"reused_original_cases":len(audited),"audited_bytes":offset}),flush=True)
    try:
        while len(audited) < len(plan["cases"]):
            amended_deadline()
            run_progress = gate.read_json(DIRECTORY/"progress.json")
            case = plan["cases"][len(audited)]
            if case["case_id"] not in run_progress["completed_case_ids"]:
                if run_progress["status"].startswith("incomplete") or not watch:
                    progress("incomplete_runner")
                    print(json.dumps({"status":"incomplete_audit","audited_cases":len(audited),
                                      "runner_status":run_progress["status"]}),flush=True)
                    return
                time.sleep(5)
                continue
            boundary = run_progress["completed_case_ledger_bytes"][str(case["case_id"])]
            require(type(boundary) is int and offset < boundary <= run_progress["flushed_ledger_bytes"],
                    "Invalid completion case byte boundary")
            with (DIRECTORY/"trials.jsonl").open("rb") as stream:
                stream.seek(offset)
                raw = stream.read(boundary-offset)
            require(len(raw) == boundary-offset and raw.endswith(b"\n"), "Incomplete flushed case bytes")
            began = time.perf_counter()
            checked = original_audit.audit_case(raw, case, plan, selected)
            checked.update(start_byte=offset, end_byte=boundary,
                           input_prefix_part_sha256=hashlib.sha256(raw).hexdigest(),
                           audit_sources_sha256=sources["original_audit_sources_sha256"],
                           completion_audit_sources_sha256=sources["completion_sources_sha256"],
                           elapsed_seconds=time.perf_counter()-began,
                           completion_audit_provenance={"type":"fresh_protocol_audit_under_administrative_extension",
                                                        "amendment_sha256":gate.digest(gate.AMENDMENT)})
            check_sources(amendment, sources)
            gate.write_new(DIRECTORY/"audit_cases"/("case_%03d.json"%case["case_id"]), checked)
            audited.append(checked)
            offset = boundary
            progress("running")
            print(json.dumps({"audited_case":case["case_id"],"audited_cases":len(audited),
                              "case_audit_seconds":checked["elapsed_seconds"],
                              "elapsed_seconds":time.perf_counter()-started}),flush=True)
        while not all((DIRECTORY/name).exists() for name in
                      ("summary.json", "analysis.json", "analysis_provenance.json")):
            amended_deadline()
            if not watch:
                progress("waiting_for_complete_analysis")
                return
            time.sleep(5)
        amended_deadline()
        result = finalize(amendment, plan, selected, sources, audited, offset, prefix, started)
        progress("complete")
        print(json.dumps(result, indent=2), flush=True)
    except BaseException as error:
        progress("incomplete_error", type(error).__name__+": "+str(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    main(parser.parse_args().watch)
