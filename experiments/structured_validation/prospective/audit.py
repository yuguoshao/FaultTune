"""Incremental audit of flushed complete cases; never generates new observations.

Every logged block RNG, paid action, score and eligible screen is rebuilt.
Every final decision is certified with outward arithmetic. Mechanical first/
last replicates, all errors and cap abstentions get the original full interval;
other decisions may use an independent nonnegative partial-sum lower bound.
Full original runner replays cover first/last replicates and every error.
"""
import argparse
from collections import Counter, defaultdict
from decimal import Decimal
from fractions import Fraction
import hashlib
from itertools import groupby
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from scipy.special import gammaln
import audit_refined as balanced
import audit_conditional as bookkeeping
import audit_unequal as unequal_audit
import fast_rb_certificate as fast_rb
import fast_weighted_fisher as fast_fisher
import run_refined as refined
import unequal_runner as unequal
import weighted_fisher as fisher
import safe_grid as safe
import rb_grid as rb
from direct_tests import _Bounds
import common
import run as prospective_run

require = common.require
AUDIT_FILES = ("audit_refined.py", "audit_conditional.py", "audit_unequal.py",
               "run_conditional.py", "fast_rb_certificate.py", "fast_weighted_fisher.py")


def check_deadline():
    if time.time() >= 1793732834:
        raise RuntimeError("Hard deadline reached; incomplete audit artifacts retained")


def audit_hashes():
    paths = [Path(__file__)]+[common.HERE.parent/name for name in AUDIT_FILES]
    return {path.relative_to(common.ROOT).as_posix(): common.digest(path) for path in paths}


def expected_decision(case):
    return "accept" if case["C"] < 0 else "reject" if case["C"] > 0 else "abstain"


def wrong(record, case):
    return record["decision"] not in ("abstain", expected_decision(case))


def original_interval_required(record, case, repetitions):
    return record["rep"] in (0, repetitions-1) or wrong(record, case) or record["decision"] == "abstain"


def adapted_spec(plan, case, engines=()):
    cap = plan["cap_by_kind"][case["kind"]]
    return {"seed_base": plan["seed_base"], "delta": plan["delta"], "cap": cap,
            "checkpoints": bookkeeping.expected_checkpoints(cap), "engines": list(engines),
            "conditional_protocol": fisher.protocol_config(), "evidence_protocol": fisher.protocol_config(),
            "explore_after_previous_shots": refined.EXPLORE_AFTER,
            "sqrt_multiplier": refined.SQRT_MULTIPLIER,
            "exploration_sqrt_multiplier": unequal.EXPLORE_MULTIPLIER,
            "exploration_coordinates": [refined.rational_circle(4,7),refined.rational_circle(7,4)],
            "grid": refined.GRID}


def validate_stored_certificate(record, proof, full, totals):
    stored = record["certificate"]
    require(stored["decision"] == record["decision"] == proof["decision"] and
            stored["shots"] == record["shots"] == proof["shots"], "Certificate decision/cost differs")
    require(stored["threshold_bounds"] == proof["threshold_bounds"], "Certificate risk threshold changed")
    for field in ("score", "score_numerator", "score_denominator"):
        if field in proof:
            require(stored[field] == proof[field], "Certificate signed score differs")
    if full:
        require(stored == proof, "Original complete Decimal certificate differs")
        totals["full_interval_final_certificates"] += 1
    else:
        totals["independent_decision_final_certificates"] += 1
        require(record["decision"] != "abstain", "Partial certificate cannot certify an abstention")
        require(Decimal(proof["evidence_bounds"][0]) >= Decimal(proof["threshold_bounds"][1]),
                "Independent lower certificate does not cross the exact threshold")
        # An independent enclosure is not a recomputation of stored tight bounds.
        require(Decimal(proof["evidence_bounds"][0]) <= Decimal(stored["evidence_bounds"][1]) and
                Decimal(stored["evidence_bounds"][0]) <= Decimal(proof["evidence_bounds"][1]),
                "Independent and stored full-mixture intervals do not overlap")
    if record["decision"] != "abstain":
        require(Decimal(stored["evidence_bounds"][0]) >= Decimal(stored["threshold_bounds"][1]),
                "Stored decided record has no certified crossing")
        totals["all_decisions_strictly_certified"] += 1
    totals["terminal_records_checked"] += 1
    return {"case_id":record["case_id"],"method":record["method"],"rep":record["rep"],
            "shots":record["shots"],"decision":record["decision"],
            "check_kind":"full_interval_recomputation" if full else "independent_decision_lower_bound",
            "evidence_bounds":proof["evidence_bounds"],"threshold_bounds":proof["threshold_bounds"],
            "certified_experts":proof.get("certified_experts"),
            "certificate_sha256":bookkeeping.stable_json_hash(proof)}


def balanced_proof(engine, counts, fstate, sstate, rstate, spec, full):
    if full or engine in ("beta", "pair_grid", "safe"):
        return balanced.audit_certificate(engine, counts, fstate, sstate, rstate, spec), True
    if engine == "rb":
        cert = fast_rb.certificate_from_blocks(rstate["blocks"], spec["delta"],
                                               spec["conditional_protocol"], state=rstate)
    else:
        cert = fast_fisher.certificate_from_blocks(fstate["blocks"], spec["delta"],
                                                   spec["conditional_protocol"]["experts"], state=fstate)
        if engine == "hedge":
            lower = _Bounds(Decimal(cert["evidence_bounds"][0]), Decimal(cert["evidence_bounds"][1]))
            mixed = bookkeeping.mixture_bounds([refined.beta_bounds(counts), lower])
            threshold = -_Bounds(Decimal(str(spec["delta"]))).log()
            score = int(counts[1]-counts[2])
            cert = {"decision":("reject" if score > 0 else "accept") if
                    score and mixed.lo >= threshold.hi else "abstain",
                    "shots":int(2*sum(counts)),"score":score,
                    "evidence_bounds":[str(mixed.lo),str(mixed.hi)],
                    "threshold_bounds":[str(threshold.lo),str(threshold.hi)]}
    if cert["decision"] == "abstain" or "certified_experts" not in cert and engine != "hedge":
        return balanced.audit_certificate(engine, counts, fstate, sstate, rstate, spec), True
    # Balanced Fisher original record uses score=2*conditional score.
    if engine == "fisher" and "score_numerator" in cert:
        cert = dict(cert)
        del cert["score_numerator"], cert["score_denominator"]
        cert["score"] = int(counts[1]-counts[2])
    return cert, False


def audit_balanced(records, case, plan, totals):
    engines = sorted(r["engine"] for r in records)
    spec = adapted_spec(plan, case, engines)
    policy, rep = records[0]["policy"], records[0]["rep"]
    seed = plan["seed_base"]+1009*case["case_id"]+1000003*rep
    rng = np.random.RandomState(seed)
    by_engine = {r["engine"]:r for r in records}
    require(len(by_engine) == len(records), "Duplicate balanced engine")
    history = max(records,key=lambda r:r["shots"])["history"]
    for record in records:
        require(record["method"] == policy+"_"+record["engine"] and record["seed"] == seed and
                record["case_id"] == case["case_id"] and record["rep"] == rep, "Balanced identity differs")
        require(record["history"] == [b for b in history if b["budget"] <= record["shots"]],
                "Same-policy engine histories do not share identical prefixes")
    fstate, sstate, rstate = fisher.empty_state(), safe.empty_state(safe.protocol_config(fisher.protocol_config())), rb.empty_state(fisher.protocol_config())
    rows, counts, proofs, stopped = [], np.zeros(4,dtype=np.int64), [], set()
    previous = explored = look_count = 0
    for look, grouped in groupby(history,key=lambda b:b["look"]):
        look_count += 1
        require(look == look_count and look <= len(spec["checkpoints"]), "Nonconsecutive balanced look")
        budget = spec["checkpoints"][look-1]
        blocks = list(grouped)
        phases = bookkeeping.expected_phases(policy,previous,budget,explored,spec)
        require([(b["phase"],b["m"]) for b in blocks] == phases, "Balanced phase budget differs")
        for block in blocks:
            require(block["budget"] == budget, "Balanced block checkpoint differs")
            arm = bookkeeping.predictable_arm(block["phase"],policy,rows,spec)
            require(arm == block["arm"], "Action was not computed from previous paid counts")
            co,si,denom = bookkeeping.arm_key(arm)
            require(all(type(v) is int for v in (co,si,denom)) and denom > 0 and si > 0 and
                    co*co+si*si == denom*denom, "Invalid exact reflected pair")
            prediction = balanced.safe_prediction_from_past(rows,arm,sstate["config"])
            require(prediction == block["safe_prediction"], "Nuisance proposal used future counts")
            m = block["m"]
            require(type(m) is int and m > 0, "Invalid block length")
            pm,pp = case["A"]+case["B"]*(co/denom)-case["C"]*(si/denom), case["A"]+case["B"]*(co/denom)+case["C"]*(si/denom)
            minus = int(rng.binomial(m,pm))
            both = int(rng.binomial(minus,pp))
            plus_only = int(rng.binomial(m-minus,pp))
            fresh = [m-minus-plus_only,plus_only,minus-both,both]
            plus = plus_only+both
            require(fresh == block["pair_counts"] and minus == block["minus_successes"] and
                    plus == block["plus_successes"], "Balanced RNG block replay differs")
            counts += fresh
            fisher.update_block(fstate,m,minus,plus)
            safe.update_block(sstate,m,minus,plus,prediction)
            rb.update_block(rstate,m,minus,plus)
            bookkeeping.add_paid_row(rows,arm,m,plus)
            bookkeeping.add_paid_row(rows,dict(arm,sin_numerator=-si),m,minus)
            if block["phase"].startswith("explore"):
                explored += m
            totals["replayed_rng_blocks"] += 1
            totals["causal_angle_and_prediction_blocks"] += 1
        require(int(counts.sum())*2 == budget == fstate["shots"] == sstate["shots"] == rstate["shots"] ==
                sum(row["trials"] for row in rows), "Balanced paid shot accounting differs")
        require(2*fstate["score"] == sstate["score"] == rstate["score"] == int(counts[1]-counts[2]),
                "Balanced score bookkeeping differs")
        b,c = counts[1:3]
        beta = float(gammaln(b+.5)+gammaln(c+.5)-gammaln(b+c+1)-2*gammaln(.5)+(b+c)*math.log(2))
        fl = fisher.screen(fstate,plan["delta"])["log_evidence"]
        screens = {"beta":beta,"pair_grid":balanced.weighted_pair_log(counts,spec["conditional_protocol"]["experts"]),
                   "fisher":fl,"hedge":float(np.logaddexp(beta,fl)-math.log(2)),
                   "safe":safe.screen(sstate,plan["delta"])["log_evidence"],"rb":rb.screen(rstate,plan["delta"])["log_evidence"]}
        for engine in engines:
            if engine in stopped:
                continue
            record = by_engine[engine]
            totals["checkpoint_screens_recomputed"] += 1
            require(budget <= record["shots"], "Recorded stop has no eligible screen")
            eligible = screens[engine] >= math.log(1/plan["delta"])-1e-6 or budget == spec["cap"]
            if not eligible:
                require(budget != record["shots"], "Stopped without eligible screen")
                continue
            if budget < record["shots"]:
                proof = balanced.audit_certificate(engine,counts,fstate,sstate,rstate,spec)
                totals["full_interval_intermediate_certificates"] += 1
                require(proof["decision"] == "abstain", "Missed an earlier exact crossing")
                continue
            require(record["counts"] == counts.tolist() and record["exploration_pairs"] == explored,
                    "Final balanced counts/exploration cost differ")
            full = original_interval_required(record,case,plan["repetitions_by_kind"][case["kind"]])
            proof, full = balanced_proof(engine,counts,fstate,sstate,rstate,spec,full)
            proofs.append(validate_stored_certificate(record,proof,full,totals))
            stopped.add(engine)
        previous = budget
    require(stopped == set(engines), "Missing final balanced decision")
    totals["seed_replayed_method_trajectories"] += len(records)
    return proofs


def audit_unequal(record, case, plan, totals):
    spec = adapted_spec(plan,case)
    policy,rep = record["policy"],record["rep"]
    seed = plan["seed_base"]+1009*case["case_id"]+1000003*rep
    require(record["seed"] == seed and record["method"] == policy+"_weighted_conditional_unequal",
            "Unequal identity/seed differs")
    rng = np.random.RandomState(seed)
    rows, state = [], fisher.empty_state()
    current_plan, regime, next_replan, planned = None,None,0,0
    previous = explored = look_count = 0
    exact_score, proof = Fraction(0), None
    for look,grouped in groupby(record["history"],key=lambda b:b["look"]):
        look_count += 1
        require(look == look_count and look <= len(spec["checkpoints"]), "Unequal look differs")
        budget = spec["checkpoints"][look-1]
        blocks = list(grouped)
        phases = unequal_audit.phase_schedule(policy,previous,budget,explored,spec)
        require([(b["phase"],b["nminus"]+b["nplus"]) for b in blocks] == phases, "Unequal phase cost differs")
        for block in blocks:
            require(block["budget"] == budget, "Unequal block budget differs")
            phase,number = block["phase"],block["nminus"]+block["nplus"]
            if phase.startswith("explore"):
                arm = spec["exploration_coordinates"][0 if phase == "explore_a" else 1]
                weight,plan_id = .5,None
            else:
                effective = "psr" if phase == "psr" else policy
                paid = sum(row["trials"] for row in rows)
                if current_plan is None or regime != effective or paid >= next_replan:
                    rebuilt = unequal.plan_from_rows(effective,rows)
                    rebuilt["plan_id"],rebuilt["look"] = planned,look
                    require(planned < len(record["plans"]), "Missing causal plan")
                    unequal_audit.equal_values(record["plans"][planned],rebuilt,"plan")
                    current_plan = record["plans"][planned]
                    planned += 1
                    regime,next_replan = effective,max(32,2*paid)
                    totals["causal_unequal_plans_rebuilt"] += 1
                arm,weight,plan_id = current_plan["positive_arm"],current_plan["weight_plus"],current_plan["plan_id"]
            require(block["positive_arm"] == arm and block["planned_weight_plus"] == weight and block["plan_id"] == plan_id,
                    "Unequal action did not use its recorded paid-data plan")
            np_ = int(math.floor(number*weight+.5))
            if number >= 2:
                np_ = min(number-1,max(1,np_))
            nm = number-np_
            require((block["nminus"],block["nplus"]) == (nm,np_), "Unequal rounded allocation differs")
            co,si,denom = (arm[k] for k in unequal.FEATURE_KEYS)
            require(all(type(v) is int for v in (co,si,denom)) and denom > 0 and si > 0 and
                    co*co+si*si == denom*denom, "Invalid exact unequal query")
            pm,pp = case["A"]+case["B"]*(co/denom)-case["C"]*(si/denom),case["A"]+case["B"]*(co/denom)+case["C"]*(si/denom)
            km,kp = int(rng.binomial(nm,pm)),int(rng.binomial(np_,pp))
            require((km,kp) == (block["minus_successes"],block["plus_successes"]), "Unequal RNG replay differs")
            fisher.update_block(state,nm,np_,km,kp)
            exact_score += Fraction(kp)-Fraction((km+kp)*np_,number)
            unequal_audit.add_row(rows,arm,np_,kp)
            unequal_audit.add_row(rows,dict(arm,sin_numerator=-si),nm,km)
            if phase.startswith("explore"):
                explored += number//2
            totals["replayed_rng_blocks"] += 1
        require(state["shots"] == budget == sum(r["trials"] for r in rows) and state["score"] == exact_score,
                "Unequal score/paid cost differs")
        screen = fisher.screen(state,plan["delta"])
        totals["checkpoint_screens_recomputed"] += 1
        require(budget <= record["shots"], "Unequal extra data after stop")
        if screen["crossing_candidate"] or budget == spec["cap"]:
            full = budget < record["shots"] or original_interval_required(record,case,plan["repetitions_by_kind"][case["kind"]])
            cert = fisher.certificate_from_blocks(state["blocks"],plan["delta"],state["experts"]) if full else \
                fast_fisher.certificate_from_blocks(state["blocks"],plan["delta"],state["experts"],state=state)
            if not full and "certified_experts" not in cert:
                full = True
            if budget < record["shots"]:
                totals["full_interval_intermediate_certificates"] += 1
                require(cert["decision"] == "abstain", "Unequal missed an earlier certified stop")
            else:
                proof = validate_stored_certificate(record,cert,full,totals)
        else:
            require(budget != record["shots"], "Unequal stop without eligible screen")
        previous = budget
    require(proof is not None and previous == record["shots"] and planned == len(record["plans"]), "Incomplete unequal history")
    require(rows == record["final_rows"] and explored == record["exploration_pairs"], "Unequal final paid state differs")
    require(record["full_grid_replans"] == sum(p["policy"] == "grid" for p in record["plans"]) and
            record["pair_fallback_replans"] == sum(p["pair_fallback"] for p in record["plans"]), "Unequal plan counts differ")
    totals["seed_replayed_method_trajectories"] += 1
    return [proof]


def audit_case(raw, case, plan, selected):
    records = [json.loads(line) for line in raw.splitlines()]
    repetitions = plan["repetitions_by_kind"][case["kind"]]
    expected = {(method,rep) for method in selected["methods"] for rep in range(repetitions)}
    seen, grouped, totals = set(),defaultdict(list),Counter()
    for record in records:
        key = record["method"],record["rep"]
        require(record["case_id"] == case["case_id"] and key in expected and key not in seen,
                "Incomplete, unexpected or duplicate case record")
        seen.add(key)
        require(record["decision"] in ("accept","reject","abstain") and type(record["shots"]) is int and
                0 < record["shots"] <= plan["cap_by_kind"][case["kind"]], "Invalid result/cost")
        require(record["decision"] != "abstain" or record["shots"] == plan["cap_by_kind"][case["kind"]],
                "Abstention before its frozen cap")
        family = "unequal" if record["method"].endswith("_weighted_conditional_unequal") else "refined"
        grouped[(record["rep"],record["policy"],family)].append(record)
    require(seen == expected, "Case is not complete")
    proofs,replays = [],[]
    for (rep,policy,family),local in sorted(grouped.items()):
        check_deadline()
        if family == "refined":
            proofs.extend(audit_balanced(local,case,plan,totals))
        else:
            require(len(local) == 1, "Duplicate unequal protocol")
            proofs.extend(audit_unequal(local[0],case,plan,totals))
        if rep in (0,repetitions-1) or any(wrong(r,case) for r in local):
            check_deadline()
            seed = plan["seed_base"]+1009*case["case_id"]+1000003*rep
            if family == "refined":
                replay,unused = refined.run_trial(case,policy,rep,seed,plan["cap_by_kind"][case["kind"]],
                                                   sorted(r["engine"] for r in local))
            else:
                replayed,unused = unequal.run_trial(case,policy,rep,seed,plan["cap_by_kind"][case["kind"]])
                replay = [replayed]
            require(sorted(replay,key=lambda r:r["method"]) == sorted(local,key=lambda r:r["method"]),
                    "Full original seeded runner replay differs")
            totals["full_runner_replay_groups"] += 1
            totals["full_runner_replay_method_records"] += len(local)
            replays.append({"rep":rep,"policy":policy,"family":family,
                            "reason":"mechanical endpoint" if rep in (0,repetitions-1) else "error",
                            "records_sha256":bookkeeping.stable_json_hash(sorted(local,key=lambda r:r["method"]))})
    return {"status":"passed_case","case_id":case["case_id"],"kind":case["kind"],"checks":dict(totals),
            "cells":common.build_cells(records,[case],selected["methods"]),
            "decision_proofs":proofs,"full_runner_replays":replays,
            "cost_rows":[{k:r[k] for k in ("case_id","method","rep","shots","decision")} for r in records]}


def independently_check_analysis(case_audits, analysis, plan, selected):
    methods = selected["methods"]
    candidate = methods.index(selected["candidate"])
    comparators = [methods.index(selected[k]) for k in ("primary_comparator","secondary_comparator")]
    rng = np.random.RandomState(plan["primary_analysis"]["bootstrap_seed"])
    count = plan["primary_analysis"]["bootstrap_resamples"]
    logs,points = np.zeros((count,2)),np.zeros(2)
    case_ratios = [[],[]]
    mains = [item for item in case_audits if item["kind"] == "main"]
    for item in mains:
        records = {(r["method"],r["rep"]):r["shots"] for r in item["cost_rows"]}
        repetitions = plan["repetitions_by_kind"]["main"]
        # Different summation path: resampling multiplicities times the cost matrix.
        matrix = np.array([[records[(method,rep)] for method in methods] for rep in range(repetitions)],dtype=float)
        indices = rng.randint(0,repetitions,size=(count,repetitions))
        weights = np.zeros((count,repetitions),dtype=np.int64)
        for replicate in range(count):
            weights[replicate] = np.bincount(indices[replicate],minlength=repetitions)
        sums = weights.dot(matrix)
        means = matrix.sum(axis=0)/repetitions
        for j,comparator in enumerate(comparators):
            ratio = means[candidate]/means[comparator]
            points[j] += math.log(ratio)
            logs[:,j] += np.log(sums[:,candidate]/sums[:,comparator])
            case_ratios[j].append({"case_id":item["case_id"],"ratio":float(ratio)})
    for j,name in enumerate(("primary","secondary")):
        actual = analysis[name]
        point = math.exp(points[j]/len(mains))
        ordered = sorted(np.exp(logs[:,j]/len(mains)).tolist())
        def quantile(q):
            index = q*(len(ordered)-1)
            lower = int(math.floor(index))
            return ordered[lower]+(index-lower)*(ordered[min(lower+1,len(ordered)-1)]-ordered[lower])
        interval = [quantile(.025),quantile(.975)]
        require(math.isclose(point,actual["geometric_mean_ratio"],rel_tol=1e-12,abs_tol=1e-12), "Bootstrap point differs")
        require(all(math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12) for a,b in zip(interval,actual["percentile_95_ci"])),
                "Independent paired-bootstrap CI differs")
        require(case_ratios[j] == actual["per_case_ratios"], "Analysis per-case ratios differ")
    require(analysis["primary_numerical_threshold_met"] == (analysis["primary"]["percentile_95_ci"][1] < 1),
            "Primary pass threshold differs")


def main(watch):
    plan,selected,release = common.validate_release()
    directory = common.DIRECTORY
    require(not (directory/"audit.json").exists(), "Preserve completed audit")
    sources = audit_hashes()
    frozen = bookkeeping.frozen_evidence_checks()
    started = time.perf_counter()
    case_directory = directory/"audit_cases"
    case_directory.mkdir(exist_ok=True)
    audited,offset = [],0
    for case in plan["cases"]:
        output = case_directory/("case_%03d.json"%case["case_id"])
        require(not output.exists(), "Use one incremental auditor; existing case audit found")
    while len(audited) < len(plan["cases"]):
        if time.time() >= 1793732834:  # frozen hard deadline, 19:07:14 UTC
            raise RuntimeError("Hard deadline reached; incomplete case audits retained")
        progress_path = directory/"progress.json"
        if not progress_path.exists():
            if not watch:
                return
            time.sleep(5)
            continue
        progress = common.read_json(progress_path)
        case = plan["cases"][len(audited)]
        if case["case_id"] not in progress["completed_case_ids"]:
            if progress["status"].startswith("incomplete") or not watch:
                print(json.dumps({"status":"incomplete_audit","audited_cases":len(audited),"runner_status":progress["status"]}),flush=True)
                return
            time.sleep(5)
            continue
        boundary = progress["completed_case_ledger_bytes"][str(case["case_id"])]
        require(type(boundary) is int and offset < boundary <= progress["flushed_ledger_bytes"], "Invalid flushed case boundary")
        with (directory/"trials.jsonl").open("rb") as stream:
            stream.seek(offset)
            raw = stream.read(boundary-offset)
        require(len(raw) == boundary-offset and raw.endswith(b"\n"), "Ledger prefix is not fully flushed")
        case_started = time.perf_counter()
        checked = audit_case(raw,case,plan,selected)
        checked.update(start_byte=offset,end_byte=boundary,input_prefix_part_sha256=hashlib.sha256(raw).hexdigest(),
                       audit_sources_sha256=sources,elapsed_seconds=time.perf_counter()-case_started)
        require(audit_hashes() == sources, "Audit source changed")
        common.validate_release(recheck_development=False)
        common.write_new(case_directory/("case_%03d.json"%case["case_id"]),checked)
        audited.append(checked)
        offset = boundary
        status = {"status":"running","audited_case_ids":[r["case_id"] for r in audited],
                  "audited_ledger_bytes":offset,"elapsed_seconds":time.perf_counter()-started,
                  "audit_sources_sha256":sources}
        temporary = directory/"audit_progress.pending.json"
        temporary.write_text(json.dumps(status,indent=2)+"\n",encoding="utf-8")
        os.replace(str(temporary),str(directory/"audit_progress.json"))
        print(json.dumps({"audited_case":case["case_id"],"audited_cases":len(audited),
                          "case_audit_seconds":checked["elapsed_seconds"],"elapsed_seconds":time.perf_counter()-started}),flush=True)
    while not all((directory/name).exists() for name in ("summary.json","analysis.json")):
        if not watch:
            return
        if time.time() >= 1793732834:
            raise RuntimeError("Hard deadline before final analysis; case audits retained")
        time.sleep(5)
    inputs = {name:common.digest(directory/name) for name in
              ("spec.json","summary.json","trials.jsonl","analysis.json","selection.json","release.json")}
    spec,summary,analysis = (common.read_json(directory/name) for name in ("spec.json","summary.json","analysis.json"))
    require(summary["status"] == "complete" and summary["spec"] == spec and summary["trials_sha256"] == inputs["trials.jsonl"],
            "Final complete summary does not match audited inputs")
    require(spec["cases"] == plan["cases"] and spec["selection"] == selected and spec["plan_sha256"] == common.PLAN_SHA256 and
            spec["release_sha256"] == inputs["release.json"], "Final spec/release differs")
    require((directory/"trials.jsonl").stat().st_size == offset, "Final ledger has unexpected trailing records")
    with (directory/"trials.jsonl").open("rb") as stream:
        for item in audited:
            raw = stream.read(item["end_byte"]-item["start_byte"])
            require(hashlib.sha256(raw).hexdigest() == item["input_prefix_part_sha256"], "Previously audited prefix changed")
    cells = [cell for item in audited for cell in item["cells"]]
    require(summary["cells"] == analysis["cells"] == cells, "Final summary/analysis differs from audited unconditional records")
    require(analysis["input_sha256"] == {name:inputs[name] for name in analysis["input_sha256"]}, "Analysis binds different inputs")
    independently_check_analysis(audited,analysis,plan,selected)
    totals = Counter()
    for item in audited:
        totals.update(item["checks"])
    require(totals["terminal_records_checked"] == summary["new_trajectories"] ==
            plan["expected_joint_replicate_labels"]*len(selected["methods"]), "Final audit trajectory count differs")
    require(inputs == {name:common.digest(directory/name) for name in inputs} and sources == audit_hashes(), "Final audit inputs changed")
    common.validate_release()
    require(bookkeeping.frozen_evidence_checks() == frozen, "Previous frozen results changed")
    result = {"status":"passed","plan_sha256":common.PLAN_SHA256,"input_sha256":inputs,
              "audit_sources_sha256":sources,"frozen_previous_evidence":frozen,
              "audited_cases":len(audited),"checks":dict(totals),
              "case_audit_sha256":{path.name:common.digest(path) for path in sorted(case_directory.glob("case_*.json"))},
              "all_logged_rng_blocks_replayed":True,"all_paid_actions_and_scores_rebuilt":True,
              "all_checkpoint_screen_paths_checked":True,"all_final_decisions_strictly_certified":True,
              "original_full_interval_rule":"Every rep 0 and last, every error, every cap abstention; also all beta/pair_grid/safe terminals and any lazy fallback.",
              "original_full_runner_replay_rule":"Every policy/family at rep 0 and last, plus every group containing an error.",
              "independent_bootstrap_rebuilt":True,
              "primary_pass":analysis["primary_numerical_threshold_met"],
              "limits":"Independent decision lower bounds prove the unchanged threshold; their overlap with stored intervals is not a full tight-interval recomputation. Frozen normalizer primitives are reused. Fixed simulator matrix only.",
              "elapsed_seconds":time.perf_counter()-started}
    common.write_new(directory/"audit.json",result)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch",action="store_true")
    main(parser.parse_args().watch)
