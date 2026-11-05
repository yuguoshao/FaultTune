"""Descriptive development cost comparisons on the complete fixed case suite."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(unequal_directory, refined_directory, output):
    unequal_directory, refined_directory, output = map(Path, (unequal_directory, refined_directory, output))
    if output.exists():
        raise RuntimeError("preserve completed comparisons; use a new output")
    us, rs = [json.loads((directory/"summary.json").read_text(encoding="utf-8"))
              for directory in (unequal_directory, refined_directory)]
    uc, rc = [{c["case_id"]: c for c in summary["spec"]["cases"]} for summary in (us, rs)]
    if uc != rc or us["spec"]["cap"] != rs["spec"]["cap"] or us["spec"]["delta"] != rs["spec"]["delta"]:
        raise ValueError("comparisons require identical cases, cap and risk")
    if any(c["kind"] != "main" for c in uc.values()):
        raise ValueError("this development comparison is defined on the registered main suite")
    indices = sorted(uc)
    ul, rl = [{(c["case_id"], c["method"]): c for c in summary["cells"]} for summary in (us, rs)]
    comparators = {"refined_psr_beta": (rl, "psr_beta"),
                   "refined_grid_fisher": (rl, "grid_fisher"),
                   "refined_psr_fisher": (rl, "psr_fisher"),
                   "unequal_psr_fisher": (ul, "psr_weighted_conditional_unequal")}
    comparisons = {}
    for policy in us["spec"]["policies"]:
        method = policy+"_weighted_conditional_unequal"
        selected = [ul[cid, method] for cid in indices]
        rows = {}
        for label, (lookup, baseline) in comparators.items():
            ratios = [ul[cid, method]["mean_shots"]/lookup[cid, baseline]["mean_shots"] for cid in indices]
            geo = math.exp(math.fsum(math.log(value) for value in ratios)/len(ratios))
            rows[label] = {"geomean_cost_ratio": geo, "cost_reduction_percent": 100*(1-geo),
                           "faster_cases": sum(x < 1 for x in ratios), "tied_cases": sum(x == 1 for x in ratios),
                           "slower_cases": sum(x > 1 for x in ratios),
                           "minimum_case_ratio": min(ratios), "maximum_case_ratio": max(ratios),
                           "case_ratios": [{"case_id": cid, "ratio": ratio} for cid, ratio in zip(indices, ratios)]}
        comparisons[method] = {"geomean_shots": math.exp(math.fsum(math.log(c["mean_shots"]) for c in selected)/len(selected)),
                               "wrong": sum(c["wrong"] for c in selected),
                               "abstained": sum(c["abstained"] for c in selected),
                               "trials": sum(c["repetitions"] for c in selected),
                               "pair_fallback_replans": sum(c["pair_fallback_replans"] for c in selected),
                               "full_grid_replans": sum(c["full_grid_replans"] for c in selected),
                               "comparators": rows}
    audit_path = unequal_directory/"audit.json"
    audited = audit_path.exists() and json.loads(audit_path.read_text())["status"] == "passed"
    result = {"status": "complete descriptive development comparison", "case_count": len(indices),
              "unequal_repetitions": us["spec"]["repetitions"], "refined_repetitions": rs["spec"]["repetitions"],
              "unequal_implementation_audit_passed_at_comparison_time": audited,
              "metric": "Equal-case geometric mean of unconditional mean physical-shot ratios; errors and cap abstentions retained.",
              "comparisons": comparisons,
              "limits": ["Development data; no confidence interval, held-out qualification or SOTA declaration.",
                         "Cross-run comparisons include allocation, planning cadence, and sampling-stream differences; they do not isolate allocation causally.",
                         "Only numerical-grid designs with a single exact reflected support pair are executed; other supports use the recorded pair fallback."],
              "sources": {"unequal_summary": digest(unequal_directory/"summary.json"),
                          "refined_summary": digest(refined_directory/"summary.json"),
                          "comparison_source": digest(__file__)}}
    if audited:
        result["sources"]["unequal_audit"] = digest(audit_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    printed = {method: {"wrong": value["wrong"], "abstained": value["abstained"],
                       "geomean_cost_ratios": {name: details["geomean_cost_ratio"] for name, details in value["comparators"].items()}}
               for method, value in comparisons.items()}
    print(json.dumps(printed, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("unequal_directory", type=Path)
    parser.add_argument("refined_directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    compare(args.unequal_directory, args.refined_directory, args.output)
