"""Independent, non-frozen verification of all raw confirmation observations.

No experimental module, assessment routine, or primary summarizer is imported.
The aggregate command reads every gzip JSONL record and uses integer count,
sum, and square-sum accumulators. The compare command checks the independently
derived cell statistics and all frozen cost-ratio groups against a summary.
Neither command runs experiments or changes frozen code or specifications.
"""

import argparse
from collections import Counter
from decimal import Decimal, localcontext
from fractions import Fraction
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time


class _HashingReader:
    def __init__(self, raw):
        self.raw, self.digest, self.bytes = raw, hashlib.sha256(), 0
        self.name = raw.name

    def read(self, size=-1):
        data = self.raw.read(size)
        self.digest.update(data)
        self.bytes += len(data)
        return data

    def seekable(self):
        return False


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_spec(path):
    path = Path(path)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    expected = path.with_suffix(path.suffix + ".sha256").read_text(encoding="ascii").split()[0]
    if digest != expected:
        raise ValueError("frozen spec hash mismatch")
    spec = json.loads(data.decode("utf8"))
    if spec["schema"] != "faulttune.circuit_spec.v1":
        raise ValueError("unsupported spec")
    return spec, digest


def _positive_integer(value, zero=True):
    return type(value) is int and value >= (0 if zero else 1)


def _histogram_rank(histogram, rank):
    accumulated = 0
    for value, count in sorted(histogram.items()):
        accumulated += count
        if accumulated > rank:
            return value
    raise ArithmeticError("rank outside histogram")


def _moments(n, total, squares, histogram):
    if not n:
        return {"n": 0, "sum": total, "sum_of_squares": squares, "mean": None,
                "sample_variance": None, "median": None, "p95": None,
                "minimum": None, "maximum": None}
    variance = Fraction(n*squares-total*total, n*(n-1)) if n > 1 else None
    return {"n": n, "sum": total, "sum_of_squares": squares,
            "mean": float(Fraction(total, n)),
            "sample_variance": float(variance) if variance is not None else None,
            "sample_variance_exact": str(variance) if variance is not None else None,
            "median": (_histogram_rank(histogram, (n-1)//2) + _histogram_rank(histogram, n//2))/2,
            "p95": _histogram_rank(histogram, (19*n+19)//20-1),
            "minimum": min(histogram), "maximum": max(histogram),
            "distinct_observed_costs": len(histogram)}


def aggregate_raw(spec_path, trials_path, progress_every=50000):
    spec, spec_hash = _read_spec(spec_path)
    cases = {case["case_id"]: case for case in spec["cases"]}
    cells = {}
    for case_id in cases:
        for policy in spec["policies"]:
            cells[(case_id, policy)] = {"n": 0, "seen": bytearray(spec["repetitions"]),
                                       "decisions": Counter(), "statuses": Counter(), "status_decisions": Counter(),
                                       "sum": 0, "square_sum": 0, "penalty_sum": 0, "penalty_square_sum": 0,
                                       "hist": Counter(), "penalty_hist": Counter()}
    expected = len(cells)*spec["repetitions"]
    path = Path(trials_path)
    before = path.stat()
    count, started = 0, time.time()
    with path.open("rb") as raw:
        hashing = _HashingReader(raw)
        with gzip.GzipFile(fileobj=hashing, mode="rb") as zipped:
            with io.TextIOWrapper(zipped, encoding="utf8") as text:
                for line_number, line in enumerate(text, 1):
                    record = json.loads(line)
                    count += 1
                    if record.get("schema") != "faulttune.circuit_trial.v1" or record.get("phase") != spec["phase"]:
                        raise ValueError("raw schema/phase mismatch on line {}".format(line_number))
                    if record.get("spec_sha256") != spec_hash:
                        raise ValueError("raw spec hash mismatch on line {}".format(line_number))
                    key = record.get("case_id"), record.get("policy")
                    index = record.get("trial_index")
                    if key not in cells or not _positive_integer(index) or index >= spec["repetitions"]:
                        raise ValueError("outside frozen trial grid on line {}".format(line_number))
                    cell, case = cells[key], cases[key[0]]
                    if cell["seen"][index]:
                        raise ValueError("duplicate raw key on line {}".format(line_number))
                    cell["seen"][index] = 1
                    decision, status = record.get("decision"), record.get("status")
                    if decision not in ("accept", "reject", "abstain") or status not in ("ok", "error", "timeout"):
                        raise ValueError("invalid outcome/status on line {}".format(line_number))
                    shots, budget = record.get("shots"), record.get("shot_budget")
                    if not _positive_integer(shots) or budget != case["shot_budget"] or shots > budget:
                        raise ValueError("invalid paid cost or registered budget on line {}".format(line_number))
                    if record.get("cost", {}).get("shots") != shots:
                        raise ValueError("raw header/ledger shot mismatch on line {}".format(line_number))
                    completed = status == "ok" and decision in ("accept", "reject")
                    penalty = shots if completed else max(shots, budget)
                    cell["n"] += 1
                    cell["decisions"][decision] += 1
                    cell["statuses"][status] += 1
                    cell["status_decisions"][status + ":" + decision] += 1
                    cell["sum"] += shots
                    cell["square_sum"] += shots*shots
                    cell["penalty_sum"] += penalty
                    cell["penalty_square_sum"] += penalty*penalty
                    cell["hist"][shots] += 1
                    cell["penalty_hist"][penalty] += 1
                    if progress_every and count % progress_every == 0:
                        print("Independent raw accumulation: {}/{} records, {:.1f}s".format(count, expected, time.time()-started),
                              file=sys.stderr, flush=True)
        raw_hash, raw_bytes = hashing.digest.hexdigest(), hashing.bytes
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or raw_bytes != after.st_size:
        raise ValueError("raw file changed during independent accumulation or was not completely consumed")
    rows = []
    for (case_id, policy), cell in cells.items():
        n, case = cell["n"], cases[case_id]
        decisions, counts = cell["decisions"], cell["status_decisions"]
        valid_accepts, valid_rejects = counts["ok:accept"], counts["ok:reject"]
        delta, epsilon = case["true_delta"], case["epsilon"]
        fa = decisions["accept"] if delta >= 0 else 0
        fr = decisions["reject"] if delta <= 0 else 0
        power = valid_accepts if delta <= -epsilon else valid_rejects if delta >= epsilon else None
        rows.append({"case_id":case_id, "policy":policy, "n":n, "planned_n":spec["repetitions"],
                     "missing":spec["repetitions"]-sum(cell["seen"]),
                     "decisions":dict(decisions), "statuses":dict(cell["statuses"]),
                     "status_decisions":dict(counts), "false_accept_count":fa, "false_reject_count":fr,
                     "safety_count":fa+fr, "power_count":power,
                     "completed_count":valid_accepts+valid_rejects,
                     "noncompletion_count":n-valid_accepts-valid_rejects,
                     "actual_cost":_moments(n,cell["sum"],cell["square_sum"],cell["hist"]),
                     "penalized_cost":_moments(n,cell["penalty_sum"],cell["penalty_square_sum"],cell["penalty_hist"])})
    complete = count == expected and all(row["missing"] == 0 for row in rows)
    return {"schema":"faulttune.independent_raw_moments.v1", "spec_sha256":spec_hash,
            "spec_path":str(Path(spec_path).resolve()), "trials_path":str(path.resolve()),
            "raw_sha256":raw_hash, "compressed_bytes_consumed":raw_bytes,
            "script_sha256":_file_hash(__file__), "raw_records_consumed":count,
            "expected_records":expected, "cells":len(rows), "complete_grid":complete,
            "elapsed_seconds":time.time()-started, "per_cell":rows,
            "scope":"every raw gzip JSONL record; integer sums and square sums; no primary summary or assessment code imported"}


def _decimal_geomean(ratios):
    if not ratios:
        return None
    if any(ratio <= 0 for ratio in ratios):
        return 0.0 if any(ratio == 0 for ratio in ratios) else None
    with localcontext() as ctx:
        ctx.prec = 60
        logarithms = [Decimal(r.numerator).ln()-Decimal(r.denominator).ln() for r in ratios]
        return float((sum(logarithms, Decimal(0))/len(ratios)).exp())


def compare_summary(aggregate_path, summary_path):
    aggregate = json.loads(Path(aggregate_path).read_text(encoding="utf8"))
    summary = json.loads(Path(summary_path).read_text(encoding="utf8"))
    spec, spec_hash = _read_spec(aggregate["spec_path"])
    issues, comparisons = [], 0

    def check(condition, field, detail):
        nonlocal comparisons
        comparisons += 1
        if not condition:
            issues.append({"field":field, "detail":str(detail)})

    def same_number(expected, observed, field, detail):
        if expected is None:
            check(observed is None, field, detail)
        else:
            check(isinstance(observed,(int,float)) and math.isfinite(observed)
                  and math.isclose(expected,observed,rel_tol=2e-12,abs_tol=1e-12),field,detail)

    check(aggregate["complete_grid"],"independent_complete_grid",aggregate["raw_records_consumed"])
    check(summary["provenance"]["spec_sha256"] == spec_hash == aggregate["spec_sha256"],"spec_sha256",spec_hash)
    raw_matches = [row for row in summary["provenance"]["raw_files"] if row["sha256"] == aggregate["raw_sha256"]]
    check(len(raw_matches)==1 and len(summary["provenance"]["raw_files"])==1,"raw_file_sha256",aggregate["raw_sha256"])
    check(summary["integrity"]["raw_records"]==aggregate["raw_records_consumed"],"raw_record_count",aggregate["raw_records_consumed"])
    check(summary["integrity"]["unique_planned_records"]==aggregate["expected_records"],"unique_planned_records",aggregate["expected_records"])
    cells = {(row["case_id"],row["policy"]):row for row in aggregate["per_cell"]}
    summary_cells = {(row["case_id"],row["policy"]):row for row in summary["per_cell"]}
    check(len(summary_cells)==len(summary["per_cell"]) and set(cells)==set(summary_cells),"cell_key_set",len(cells))
    for key,row in cells.items():
        other = summary_cells.get(key)
        if other is None:
            continue
        for field in ("n","decisions","statuses","false_accept_count","false_reject_count","completed_count","noncompletion_count"):
            check(row[field]==other[field],field,key)
        check(row["safety_count"]==other["safety"]["count"],"safety_count",key)
        check(row["power_count"]==other["power"]["count"],"power_count",key)
        same_number(row["noncompletion_count"]/row["n"] if row["n"] else None,
                    other["abstention_or_failure_rate"],"noncompletion_rate",key)
        for metric in ("actual_cost","penalized_cost"):
            for field in ("mean","sample_variance","median","p95","minimum","maximum"):
                same_number(row[metric][field],other[metric][field],metric+"_"+field,key)
            # Sum and square sum are independently derived from raw integer
            # costs. The primary summary exposes their exact implied moments.
    main = [case for case in spec["cases"] if case.get("intended_efficiency_eligible",case["benchmark_role"]=="main")]
    check(len(main)==36,"all_intended_main_cases",len(main))
    candidate_results = {}
    for candidate in spec["candidates"]:
        ratios = {}
        for case in main:
            case_id = case["case_id"]
            ratios[case_id] = {}
            for metric in ("actual_cost","penalized_cost"):
                means = [Fraction(cells[(case_id,b)][metric]["sum"],cells[(case_id,b)]["n"]) for b in spec["baselines"]]
                candidate_mean = Fraction(cells[(case_id,candidate)][metric]["sum"],cells[(case_id,candidate)]["n"])
                ratios[case_id][metric] = min(means)/candidate_mean
        def group(rows):
            return {metric:_decimal_geomean([ratios[case["case_id"]][metric] for case in rows])
                    for metric in ("actual_cost","penalized_cost")}
        expected_main = group(main)
        observed_candidate = summary["per_candidate"][candidate]
        check(set(observed_candidate["main"]["case_ids"])=={case["case_id"] for case in main}
              and observed_candidate["main"]["case_count"]==36,"main_case_coverage",candidate)
        check(set(observed_candidate["per_main_case_comparison"])==set(ratios),"per_case_comparison_coverage",candidate)
        for case_id, metrics in ratios.items():
            for metric,ratio in metrics.items():
                same_number(float(ratio),observed_candidate["per_main_case_comparison"][case_id]["descriptive_all_baselines"][metric]["point"],
                            "individual_ratio",(candidate,case_id,metric))
        for metric,value in expected_main.items():
            same_number(value,observed_candidate["main"][metric]["point"],"main_geometric_ratio",(candidate,metric))
        groups_out = {}
        for axis,field in (("epsilon","epsilon"),("noise","noise"),("weight","weight_plus"),("sign","intended_sign")):
            values = sorted({str(case["strata"][field]) for case in main})
            check(set(values)==set(observed_candidate["strata"][axis]),"stratum_value_set",(candidate,axis))
            groups_out[axis] = {}
            for value in values:
                subset = [case for case in main if str(case["strata"][field])==value]
                expected_group = group(subset)
                observed_group = observed_candidate["strata"][axis][value]
                check(observed_group["case_count"]==len(subset) and set(observed_group["case_ids"])=={case["case_id"] for case in subset},
                      "stratum_case_coverage",(candidate,axis,value))
                for metric,ratio in expected_group.items():
                    same_number(ratio,observed_group[metric]["point"],"stratum_geometric_ratio",(candidate,axis,value,metric))
                groups_out[axis][value] = expected_group
        candidate_results[candidate] = {"main":expected_main,"strata":groups_out}
    policy_results = {}
    case_roles = {case["case_id"]:case["benchmark_role"] for case in spec["cases"]}
    for policy in spec["policies"]:
        selected = [row for row in aggregate["per_cell"] if row["policy"]==policy]
        main_selected = [row for row in selected if case_roles[row["case_id"]]=="main"]
        expected = {"records":sum(row["n"] for row in selected),
                    "wrong_decisions":sum(row["safety_count"] for row in selected),
                    "main_wrong_decisions":sum(row["safety_count"] for row in main_selected),
                    "main_records":sum(row["n"] for row in main_selected),
                    "main_correct_decisions":sum(row["power_count"] or 0 for row in main_selected),
                    "noncompletion_count":sum(row["noncompletion_count"] for row in selected)}
        for field,value in expected.items():
            check(value==summary["per_policy"][policy][field],"policy_"+field,policy)
        policy_results[policy] = expected
    return {"schema":"faulttune.independent_raw_summary_comparison.v1", "passed":not issues,
            "spec_sha256":spec_hash, "raw_sha256":aggregate["raw_sha256"],
            "aggregate_sha256":_file_hash(aggregate_path),"summary_sha256":_file_hash(summary_path),
            "script_sha256":_file_hash(__file__), "raw_records_independently_consumed":aggregate["raw_records_consumed"],
            "cells_checked":len(cells),"comparisons":comparisons,"issues":issues,
            "independent_per_candidate":candidate_results,"independent_per_policy":policy_results,
            "scope":"full raw-record reconstruction of counts, all actual/penalized cost moments, 36-main and every major-stratum point ratio; confidence limits and certificate proofs are audited separately"}


def _write(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    accumulate=commands.add_parser("aggregate")
    accumulate.add_argument("--spec",required=True)
    accumulate.add_argument("--trials",required=True)
    accumulate.add_argument("--output",required=True)
    accumulate.add_argument("--progress-every",type=int,default=50000)
    compare=commands.add_parser("compare")
    compare.add_argument("--aggregate",required=True)
    compare.add_argument("--summary",required=True)
    compare.add_argument("--output",required=True)
    args=parser.parse_args()
    if args.command=="aggregate":
        result=aggregate_raw(args.spec,args.trials,args.progress_every)
        printed={key:result[key] for key in ("raw_records_consumed","cells","complete_grid","elapsed_seconds","raw_sha256")}
    else:
        result=compare_summary(args.aggregate,args.summary)
        printed={key:result[key] for key in ("passed","raw_records_independently_consumed","cells_checked","comparisons","issues")}
    _write(args.output,result)
    print(json.dumps(printed,sort_keys=True),flush=True)
    if ("passed" in result and not result["passed"]) or ("complete_grid" in result and not result["complete_grid"]):
        raise SystemExit(1)


if __name__=="__main__":
    main()
