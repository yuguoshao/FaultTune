"""Independent post hoc check of the three frozen zero-effect McNemar cells.

Only existing records are read.  No experiment, policy, certificate, assessment,
or primary summarizer module is imported.  SciPy conditional binomial CDFs
independently reconstruct every fixed-test decision.  The lower endpoints and
null tail probability are descriptive follow-up diagnostics, not additional
endpoints in the frozen simultaneous 95% family.
"""

import argparse
from collections import Counter
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy
from scipy.stats import beta, binom


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


def exact_binomial_survival(n, k, p):
    """Return Pr[Bin(n,p)>=k] by exact integer arithmetic for rational p."""
    if k <= 0:
        return Fraction(1)
    if k > n:
        return Fraction(0)
    r, d = p.numerator, p.denominator
    a = d-r
    if r == 0:
        return Fraction(0)
    if a == 0:
        return Fraction(1)
    term = a**n
    cumulative = term
    for j in range(k-1):
        numerator = term*(n-j)*r
        divisor = (j+1)*a
        term, remainder = divmod(numerator, divisor)
        if remainder:
            raise ArithmeticError("binomial integer recurrence lost exactness")
        cumulative += term
    denominator = d**n
    return Fraction(denominator-cumulative, denominator)


def conservative_cp_lower(k, n, alpha):
    """Check the rounding direction of a SciPy CP lower endpoint exactly."""
    if k == 0:
        return {"lower": 0.0, "scipy_proposal": 0.0,
                "downward_float_steps": 0, "exact_tail_le_alpha": True}
    proposal = float(beta.ppf(float(alpha), k, n-k+1))
    lower, steps = proposal, 0
    while exact_binomial_survival(n, k, Fraction.from_float(lower)) > alpha:
        lower = float(np.nextafter(lower, 0.0))
        steps += 1
        if steps > 1000:
            raise ArithmeticError("CP lower proposal requires excessive correction")
    return {"lower": lower, "scipy_proposal": proposal,
            "downward_float_steps": steps, "exact_tail_le_alpha": True,
            "scope": "one-sided lower endpoint at the stated per-endpoint alpha"}


def verify(spec_path, trials_path):
    started = time.time()
    spec_path, trials_path = Path(spec_path), Path(trials_path)
    spec_bytes = spec_path.read_bytes()
    spec_hash = hashlib.sha256(spec_bytes).hexdigest()
    expected_hash = spec_path.with_suffix(spec_path.suffix+".sha256").read_text(encoding="ascii").split()[0]
    if spec_hash != expected_hash:
        raise ValueError("frozen spec hash mismatch")
    spec = json.loads(spec_bytes.decode("utf8"))
    cases = {c["case_id"]: c for c in spec["cases"] if c["category"] == "zero"}
    if len(cases) != 3 or any(c["true_delta"] != 0 for c in cases.values()):
        raise ValueError("expected three exactly zero-effect cases")
    prefixes = tuple(('{"case_id":"'+key+'",').encode("ascii") for key in cases)
    policy_token = b'"policy":"direct_fixed"'
    records, seen = {key: [] for key in cases}, {key: set() for key in cases}
    matches, mismatches = 0, []
    nearest_threshold = float("inf")
    threshold = spec["delta"]/2
    total_records = 0
    before = trials_path.stat()
    with trials_path.open("rb") as raw:
        hashing = _HashingReader(raw)
        with gzip.GzipFile(fileobj=hashing, mode="rb") as zipped:
            for line_number, line in enumerate(zipped, 1):
                total_records += 1
                if total_records % 100000 == 0:
                    print("Independent zero check: {} raw lines, {} selected, {:.1f}s".format(
                        total_records, matches, time.time()-started), file=sys.stderr, flush=True)
                if not line.startswith(prefixes) or policy_token not in line:
                    continue
                record = json.loads(line)
                case_id, index = record["case_id"], record["trial_index"]
                case = cases[case_id]
                if record["policy"] != "direct_fixed" or record["spec_sha256"] != spec_hash:
                    raise ValueError("policy/spec mismatch on line {}".format(line_number))
                if (record["schema"] != "faulttune.circuit_trial.v1" or
                        record["phase"] != spec["phase"] or record["status"] != "ok"):
                    raise ValueError("unexpected schema/phase/status")
                if type(index) is not int or not 0 <= index < spec["repetitions"] or index in seen[case_id]:
                    raise ValueError("out-of-grid or duplicate trial index")
                seen[case_id].add(index)
                pair_counts = record["result"]["pair_counts"]
                if len(pair_counts) != 4 or any(type(v) is not int or v < 0 for v in pair_counts):
                    raise ValueError("invalid pair counts")
                if sum(pair_counts) != case["fixed_pairs"] or record["shots"] != 2*case["fixed_pairs"]:
                    raise ValueError("fixed pair/shot count mismatch")
                if record["shot_budget"] != case["shot_budget"] or record["cost"]["shots"] != record["shots"]:
                    raise ValueError("budget/cost mismatch")
                reconstructed = [0, 0, 0, 0]
                for event in record["events"]:
                    if event["kind"] != "pairs" or sum(event["joint"]) != event["pairs"]:
                        raise ValueError("invalid pair event")
                    reconstructed = [a+b for a,b in zip(reconstructed, event["joint"])]
                if reconstructed != pair_counts:
                    raise ValueError("event totals differ from final pair counts")
                b, c = pair_counts[1:3]
                discordant = b+c
                p_accept = float(binom.cdf(b, discordant, .5))
                p_reject = float(binom.cdf(c, discordant, .5))
                nearest_threshold = min(nearest_threshold, abs(p_accept-threshold), abs(p_reject-threshold))
                decision = "accept" if discordant and p_accept <= threshold else (
                    "reject" if discordant and p_reject <= threshold else "abstain")
                certificate = record["result"]["certificate"]
                if certificate["counts"] != pair_counts or certificate["method"] != "fixed_paired_mcnemar":
                    raise ValueError("unexpected certificate count/method")
                if decision != record["decision"] or decision != record["result"]["decision"] or decision != certificate["decision"]:
                    mismatches.append({"case_id":case_id, "trial_index":index,
                                       "recorded_decision":record["decision"], "independent_decision":decision})
                records[case_id].append({"trial_index":index, "pair_counts":pair_counts,
                                        "p_accept":p_accept, "p_reject":p_reject,
                                        "decision":decision, "recorded_decision":record["decision"]})
                matches += 1
        raw_hash, raw_bytes = hashing.digest.hexdigest(), hashing.bytes
    after = trials_path.stat()
    if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns) or raw_bytes != after.st_size:
        raise ValueError("raw source changed during read or was not fully consumed")
    expected = len(spec["cases"])*len(spec["policies"])*spec["repetitions"]
    if total_records != expected or any(len(indices) != spec["repetitions"] for indices in seen.values()):
        raise ValueError("raw grid or selected zero cells incomplete")
    alpha = Fraction(str(spec["analysis"]["alpha_safety"]))
    cells = []
    for case_id, case in cases.items():
        counts = Counter(record["decision"] for record in records[case_id])
        n = len(records[case_id])
        errors = counts["accept"]+counts["reject"]
        cells.append({"case_id":case_id, "epsilon":case["epsilon"], "n":n,
                      "fixed_pairs":case["fixed_pairs"], "decisions":dict(counts),
                      "errors":errors, "observed_error_rate":errors/n,
                      "cp_lower":conservative_cp_lower(errors,n,alpha),
                      "records":sorted(records[case_id],key=lambda r:r["trial_index"])})
    null_tail = float(binom.sf(35,2000,.01))
    exact_tail = float(exact_binomial_survival(2000,36,Fraction(1,100)))
    return {"schema":"faulttune.independent_zero_mcnemar.v1", "spec_sha256":spec_hash,
            "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "raw_sha256":raw_hash, "compressed_bytes_consumed":raw_bytes,
            "raw_records_consumed":total_records, "expected_raw_records":expected,
            "selected_records":matches, "expected_selected_records":3*spec["repetitions"],
            "decision_mismatches":mismatches, "all_decisions_match":not mismatches,
            "conditional_test":"B=n01,C=n10,K=B+C; accept if BinCDF(B,K,.5)<=delta/2; reject if BinCDF(C,K,.5)<=delta/2; else abstain",
            "delta":spec["delta"], "tail_threshold":threshold,
            "minimum_absolute_pvalue_distance_to_threshold":nearest_threshold,
            "post_hoc_lower_endpoint_alpha":str(alpha), "alpha_decimal":str(spec["analysis"]["alpha_safety"]),
            "null_tail_P_Bin2000_0p01_ge_36":{"scipy":null_tail,"exact_integer_then_float":exact_tail,
                                             "absolute_difference":abs(null_tail-exact_tail)},
            "per_cell":cells, "elapsed_seconds":time.time()-started,
            "versions":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__},
            "scope":"Recomputation of existing records only. No frozen source imports or new trials. Post hoc CP lower endpoints and null tail are descriptive diagnostics, not added to the frozen simultaneous confidence family."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec",required=True)
    parser.add_argument("--trials",required=True)
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    result = verify(args.spec,args.trials)
    output = Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf8")
    brief = {key:value for key,value in result.items() if key != "per_cell"}
    brief["per_cell"] = [{key:value for key,value in row.items() if key != "records"} for row in result["per_cell"]]
    print(json.dumps(brief,indent=2,sort_keys=True))
    return 0 if result["all_decisions_match"] else 1


if __name__ == "__main__":
    sys.exit(main())
