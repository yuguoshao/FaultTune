"""Deterministic continuous-mixture diagnostics on completed development data.

This does not sample or reconstruct unobserved outcomes. It evaluates the old
Fisher terminal checkpoint and (optionally) its preceding checkpoint, so the
output is not a stopping-cost comparison or an independent validation study.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize_scalar

import continuous_fisher as continuous
import conditional_fisher as fisher


ROOT = Path(__file__).resolve().parents[2]
DEVELOPMENT = ROOT / "results/structured_validation/refined_development_v1"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def focused_nodes(blocks, points=49):
    """Use floating optimization only to choose exact rational quadrature nodes.

The optimizer's accuracy does not enter the validity of the final lower bound.
Its search interval and posterior scale estimate may affect numerical power.
"""
    counts = Counter((b["m"], b["minus_successes"] + b["plus_successes"])
                     for b in blocks)
    score = sum(b["plus_successes"] - b["minus_successes"] for b in blocks) / 2

    def log_integrand(eta):
        odds = Fraction(str(math.exp(float(eta))))
        p, q = max(odds.numerator, odds.denominator), min(odds.numerator, odds.denominator)
        cgf = sum(n * fisher.centered_cgf_float(m, total, p, q)
                  for (m, total), n in counts.items())
        return score * eta - cgf - math.log(math.pi) - float(np.logaddexp(eta / 2, -eta / 2))

    fit = minimize_scalar(lambda eta: -log_integrand(eta), method="bounded",
                          bounds=(-24.0, 24.0), options={"xatol": 1e-7})
    mode = float(fit.x)
    step = 1e-3
    curvature = -(log_integrand(mode + step) - 2 * log_integrand(mode)
                  + log_integrand(mode - step)) / step ** 2
    scale = min(4.0, 1 / math.sqrt(max(curvature, 1e-12)))
    left, right = max(-30.0, mode - 8 * scale), min(30.0, mode + 8 * scale)
    nodes = [Fraction(str(math.exp(float(eta)))) for eta in np.linspace(left, right, points)]
    if len(set(nodes)) != points:
        raise ArithmeticError("Numerical node proposal produced duplicate exact odds")
    return nodes, {"mode_proposal": mode, "curvature_proposal": curvature,
                   "scale_proposal": scale, "log_odds_proposal_range": [left, right],
                   "optimizer_success": bool(fit.success), "proposal_is_a_certificate": False}


def load_records(policies, rep, case_ids=None):
    spec = json.loads((DEVELOPMENT / "spec.json").read_text(encoding="utf-8"))
    cases = [c["case_id"] for c in spec["cases"]]
    if case_ids is not None:
        if not set(case_ids) <= set(cases):
            raise ValueError("Requested case is outside the completed development study")
        cases = case_ids
    expected = {(case_id, policy + "_fisher", rep) for case_id in cases for policy in policies}
    found = {}
    with (DEVELOPMENT / "trials.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            key = record["case_id"], record["method"], record["rep"]
            if key in expected:
                if key in found:
                    raise ValueError("Duplicate source trajectory")
                found[key] = record
    if set(found) != expected:
        raise ValueError("Incomplete requested source trajectory matrix")
    return spec, [found[key] for key in sorted(expected)]


def main(args):
    if args.output.exists():
        raise FileExistsError("Preserve existing exploration output")
    if args.points < 3 or args.rep < 0:
        raise ValueError("Invalid quadrature or repetition request")
    started = time.perf_counter()
    sources = continuous.source_hashes()
    sources[Path(__file__).resolve().relative_to(ROOT).as_posix()] = digest(__file__)
    inputs = {name: digest(DEVELOPMENT / name) for name in ("spec.json", "trials.jsonl", "audit.json")}
    audit = json.loads((DEVELOPMENT / "audit.json").read_text(encoding="utf-8"))
    if audit["status"] != "passed":
        raise ValueError("Source development audit did not pass")
    for name, expected in audit["run_input_sha256"].items():
        if digest(DEVELOPMENT / name) != expected:
            raise ValueError("Completed development input binding changed: " + name)
    spec, records = load_records(args.policies, args.rep, args.case_ids)
    selection = {"policies": args.policies, "rep": args.rep, "case_ids": args.case_ids,
                 "source_engine": "fisher", "points": args.points,
                 "penultimate": args.penultimate, "records": len(records)}
    output = {"status": "running deterministic development replay", "selection": selection,
              "created_utc": datetime.now(timezone.utc).isoformat(), "sources": sources,
              "development_input_sha256": inputs, "new_simulator_observations": 0, "records": [],
              "claim_boundary": "Retrospective endpoint evidence only; no first-crossing, measurement-cost, independent-validation, or SOTA claim."}
    for record in records:
        budgets = sorted(set(b["budget"] for b in record["history"]))
        checkpoints = ([budgets[-2]] if args.penultimate and len(budgets) > 1 else []) + [budgets[-1]]
        for budget in checkpoints:
            blocks = [{k: b[k] for k in ("m", "minus_successes", "plus_successes")}
                      for b in record["history"] if b["budget"] <= budget]
            nodes, proposal = focused_nodes(blocks, args.points)
            certificate = continuous.certificate_from_blocks(blocks, nodes, delta=spec["delta"])
            terminal = budget == budgets[-1]
            original = record["certificate"]["evidence_bounds"] if terminal else None
            entry = {"case_id": record["case_id"], "rep": record["rep"], "method": record["method"],
                     "checkpoint_shots": budget, "source_terminal_shots": record["shots"],
                     "terminal_checkpoint": terminal, "original_terminal_evidence_log_bounds": original,
                     "lower_bound_exceeds_original_upper":
                         continuous.Decimal(certificate["log_evidence_lower_bound"]) > continuous.Decimal(original[1])
                         if terminal else None,
                     "proposal": proposal, "certificate": certificate}
            if certificate["shots"] != budget:
                raise ValueError("Source budget differs from actual accumulated block cost")
            output["records"].append(entry)
            print(json.dumps({"case_id": record["case_id"], "method": record["method"],
                              "checkpoint": budget, "decision": certificate["decision"],
                              "log_lower_bound": certificate["log_evidence_lower_bound"],
                              "exceeds_original_upper": entry["lower_bound_exceeds_original_upper"]}), flush=True)
    for name, expected in sources.items():
        if digest(ROOT / name) != expected:
            raise ValueError("Exploration source changed during replay: " + name)
    for name, expected in inputs.items():
        if digest(DEVELOPMENT / name) != expected:
            raise ValueError("Development input changed during replay: " + name)
    terminal = [row for row in output["records"] if row["terminal_checkpoint"]]
    output.update(status="complete deterministic development replay",
                  elapsed_seconds=time.perf_counter() - started,
                  terminal_checkpoints=len(terminal),
                  terminal_lower_bound_exceeds_old_upper=sum(row["lower_bound_exceeds_original_upper"] for row in terminal),
                  terminal_certified_crossings=sum(row["certificate"]["decision"] != "abstain" for row in terminal))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in output.items() if k not in ("records", "sources", "development_input_sha256")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", nargs="+", choices=("psr", "closed"), default=["psr", "closed"])
    parser.add_argument("--rep", type=int, default=0)
    parser.add_argument("--case-ids", nargs="+", type=int)
    parser.add_argument("--points", type=int, default=49)
    parser.add_argument("--penultimate", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
