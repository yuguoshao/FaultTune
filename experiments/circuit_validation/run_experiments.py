"""Frozen, deadline-bounded multiprocessing circuit-validation experiments.

Only development runs may guide design. A confirmation spec is written with
source hashes and a separate SHA256 before any confirmation observations.
The count backend preserves the IID observation law, not hardware wall time.
"""

import argparse
from collections import Counter
from dataclasses import asdict
from fractions import Fraction
import hashlib
import gzip
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import queue
import sys
import struct
import time
import zipfile

from case_factory import build_cases
from count_oracle import CountObservationOracle
from observation import PublicExperiment, DeadlineExceeded, stable_seed
from policies import POLICIES, BASELINES, CANDIDATES, run_policy


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
SCHEMA = "faulttune.circuit_spec.v1"
_WORKER_SPEC = None
_WORKER_HASH = None
_WORKER_DEADLINE = None


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def encoded(value, pretty=False):
    return json.dumps(json_safe(value), sort_keys=True, ensure_ascii=False,
                      allow_nan=False, indent=2 if pretty else None,
                      separators=None if pretty else (",", ":")) + "\n"


def save_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded(value, True))
    else:
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(encoded(value, True), encoding="utf-8")
        os.replace(str(temp), str(path))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes():
    return {path.relative_to(REPO).as_posix(): sha256_file(path)
            for path in sorted(ROOT.glob("*.py"))}


def endpoint_alpha(numerator, denominator):
    """JSON's decimal risk must not exceed the exact family allocation."""
    target = Fraction(numerator, denominator)
    value = float(target)
    while Fraction(str(value)) > target:
        bits = struct.unpack(">Q", struct.pack(">d", value))[0]
        value = struct.unpack(">d", struct.pack(">Q", bits - 1))[0]
    return value


def archive_sources(path, hashes):
    """Retain the exact preregistered programs even after later development."""
    with zipfile.ZipFile(str(path), "x", zipfile.ZIP_DEFLATED) as archive:
        for name, expected in hashes.items():
            data = (REPO / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError("source changed while archiving: " + name)
            archive.writestr(name, data)
    return sha256_file(path)


def verify_spec(path, check_sources=True):
    path = Path(path)
    spec = json.loads(path.read_text(encoding="utf-8"))
    digest = sha256_file(path)
    expected = path.with_suffix(path.suffix + ".sha256").read_text(encoding="ascii").split()[0]
    if digest != expected or spec["schema"] != SCHEMA:
        raise ValueError("spec hash/schema mismatch")
    if check_sources:
        mismatch = [name for name, expected_hash in spec["source_hashes"].items()
                    if not (REPO / name).is_file() or sha256_file(REPO / name) != expected_hash]
        if mismatch:
            raise ValueError("frozen sources changed: " + ", ".join(mismatch))
    return spec, digest


def write_spec(path, phase, repetitions, policies, master_seed, max_extra=262144):
    from design_budget import design_budgets
    if repetitions < 1 or not policies or len(set(policies)) != len(policies):
        raise ValueError("positive repetitions and unique policy names required")
    if any(name not in POLICIES for name in policies):
        raise ValueError("unknown policy")
    state = json.loads((ROOT / "run_state.json").read_text(encoding="utf-8"))
    if time.time() >= state["research_cutoff_epoch_seconds"]:
        raise DeadlineExceeded("research cutoff reached; no new experiment specification")
    design = design_budgets()
    cases = build_cases(phase)
    for case in cases:
        row = design["by_case"][case["case_id"]]
        case["fixed_pairs"] = row["fixed_pairs"]
        case["shot_budget"] = row["shot_budget"]
        case["budget_design"] = row
    ncell = len(cases) * len(policies)
    spec = {
        "schema": SCHEMA, "phase": phase,
        "created_epoch": time.time(), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "goal_start_epoch": state["start_epoch_seconds"],
        "deadline_epoch": state["research_cutoff_epoch_seconds"],
        "goal_hard_deadline_epoch": state["deadline_epoch_seconds"],
        "master_seed": master_seed, "repetitions": repetitions,
        "policies": policies, "baselines": [p for p in BASELINES if p in policies],
        "candidates": [p for p in CANDIDATES if p in policies],
        "diagnostic_policies": [p for p in policies if p == "calibration_only"],
        "policy_parameters": {"growth": 1.3, "initial": 32, "max_extra": max_extra,
                              "combgame_macro_budget": 512},
        "delta": .01, "cases": cases, "budget_design": design,
        "analysis": {
            "family_alpha": .05, "safety_family_alpha": .02,
            "power_family_alpha": .02, "cost_family_alpha": .01,
            "alpha_safety": endpoint_alpha(1, 50 * 3 * ncell),
            "alpha_power": endpoint_alpha(1, 50 * 2 * ncell),
            "alpha_cost": endpoint_alpha(1, 100 * 2 * ncell),
            "alpha_rounding": "decimal JSON values rounded down from exact family fractions",
            "safety_endpoints_per_cell": 3, "power_endpoints_per_cell": 2,
            "cost_intervals_per_cell": 2,
            "unused_alpha_recycled": False,
            "safety_target": .01, "power_target": .95,
            "speedup_point_target": 3, "speedup_lower_target": 2,
            "efficiency_case_rule": "all intended main cases, no dropping failed qualification or infeasible cells",
            "cost_rule": "all shots; also full shot budget for abstention/timeout/error; all baseline qualification must be confirmed",
            "aggregate_rule": "unweighted geometric mean over all 36 main cases; same simultaneous cell interval propagation; also each epsilon/noise/weight/sign stratum",
            "secondary_costs": "readout_objects, switches, normalized modeled_device_duration, classical simulation wall time; descriptive only",
        },
        "observation_contract": {
            "backend": "CountObservationOracle",
            "law": "iid public contexts; independent old/new noise conditional on context",
            "seed_rule": "stable_seed(master_seed,phase,case_id,policy,trial_index)",
            "independent_seed_per_policy": True,
            "policy_receives": "PublicExperiment and sample_arm/sample_pairs only; fixed n from development",
            "cost": {"measured_qubits": 3, "old_duration": 3.2,
                     "new_duration": 4.2, "switch_duration": .1},
        },
        "engineering_audit": {
            "full_records": "replay every batch ledger and recompute every final accept/reject certificate",
            "deterministic_replay": "fixed per-cell indices sorted({0,repetitions-1}); regenerate from seed, ignoring wall-clock fields only",
            "replay_selection_uses_outcomes": False,
            "old_manifests_required": ["experiments/local_validation/results/manifest.json",
                                       "experiments/local_validation/results/aggregate_manifest.json"],
        },
        "source_hashes": source_hashes(),
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "numpy": __import__("numpy").__version__,
                        "scipy": __import__("scipy").__version__},
        "scope": "stationary three-qubit simulator family, 36 main + 8 diagnostic cases; not hardware or universal quantum workloads",
        "confirmation_freeze": phase == "confirmation",
        "no_confirmation_feedback_tuning": True,
        "timeout_rule": "retain every interrupted/error trial, resume only absent keys; incomplete planned grid cannot pass",
        "raw_record_format": "gzip JSONL, full per-batch oracle events and independently reproducible final certificates",
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    archive_path = path.with_name(path.stem + "_sources.zip")
    spec["source_archive"] = archive_path.name
    spec["source_archive_sha256"] = archive_sources(archive_path, spec["source_hashes"])
    save_json(path, spec, exclusive=True)
    digest = sha256_file(path)
    with path.with_suffix(path.suffix + ".sha256").open("x", encoding="ascii") as handle:
        handle.write(digest + "  " + path.name + "\n")
    print(encoded({"spec": str(path), "sha256": digest, "cases": len(cases),
                   "policies": len(policies), "planned_trials": ncell * repetitions}).strip(), flush=True)
    return spec


def public_config(spec, case):
    opaque = hashlib.sha256((case["case_id"] + "::public").encode("utf-8")).hexdigest()
    return PublicExperiment(contexts=tuple(case["contexts"]), weights=tuple(case["weights"]),
                            delta=spec["delta"], epsilon=case["epsilon"],
                            shot_budget=case["shot_budget"], candidate_id=opaque[:32],
                            workload_id=opaque[32:], phase=spec["phase"],
                            **spec["observation_contract"]["cost"])


def run_trial(spec, spec_hash, case, policy, index, deadline=None, keep_events=True):
    config = public_config(spec, case)
    seed = stable_seed(spec["master_seed"], spec["phase"], case["case_id"], policy, index)
    oracle = CountObservationOracle(config, case["probabilities"], seed,
                                    deadline_epoch=deadline, keep_events=keep_events)
    started = time.time()
    result, error, status = None, None, "ok"
    try:
        if not case["feasible"]:
            raise ValueError("registered physical case is infeasible: " + str(case["infeasible_reason"]))
        result = run_policy(policy, config, oracle.sample_arm, oracle.sample_pairs,
                            fixed_pairs=case["fixed_pairs"], **spec["policy_parameters"])
    except DeadlineExceeded as exc:
        status, error = "timeout", str(exc)
    except Exception as exc:
        status, error = "error", type(exc).__name__ + ": " + str(exc)
    cost = oracle.cost_snapshot()
    record = {
        "schema": "faulttune.circuit_trial.v1", "phase": spec["phase"],
        "spec_sha256": spec_hash, "case_id": case["case_id"], "policy": policy,
        "trial_index": index, "seed": str(seed), "status": status,
        "decision": result["decision"] if result else "abstain",
        "shots": cost["shots"], "shot_budget": config.shot_budget,
        "cost": cost, "result": result, "error": error,
        "started_epoch": started, "elapsed_seconds": time.time() - started,
        "public_config": asdict(config),
    }
    if keep_events:
        record["events"] = oracle.events
    return json_safe(record)


def _initialize_worker(spec, spec_hash, deadline):
    global _WORKER_SPEC, _WORKER_HASH, _WORKER_DEADLINE
    _WORKER_SPEC, _WORKER_HASH, _WORKER_DEADLINE = spec, spec_hash, deadline


def _run_chunk(task):
    case_index, policy, indices = task
    case = _WORKER_SPEC["cases"][case_index]
    return [run_trial(_WORKER_SPEC, _WORKER_HASH, case, policy, index, _WORKER_DEADLINE)
            for index in indices]


def record_key(record):
    return record["case_id"], record["policy"], record["trial_index"]


def run_grid(spec_path, prefix, workers, chunk_size, wall_seconds=None, resume=False,
             reproduce=False):
    spec, digest = verify_spec(spec_path)
    started = time.time()
    if reproduce and (wall_seconds is None or wall_seconds <= 0):
        raise ValueError("later reproduction requires an explicit positive --wall-seconds")
    # An expired operational deadline is not part of the scientific design.
    # Explicit later reproduction keeps every case, seed, algorithm and budget.
    # Active-goal runs must never use this flag to extend their agreed deadline.
    deadline = started + wall_seconds if reproduce else spec["deadline_epoch"]
    if wall_seconds is not None:
        deadline = min(deadline, started + wall_seconds)
    if started >= deadline:
        raise DeadlineExceeded("experiment deadline already elapsed")
    if workers < 1 or chunk_size < 1:
        raise ValueError("positive workers/chunk size required")
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    trials_path = prefix.with_name(prefix.name + "_trials.jsonl.gz")
    status_path = prefix.with_name(prefix.name + "_status.json")
    seen, counts = set(), Counter()
    if trials_path.exists():
        if not resume:
            raise FileExistsError("raw results exist; choose a new prefix or explicitly resume missing keys")
        with gzip.open(str(trials_path), "rt", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record["spec_sha256"] != digest or record_key(record) in seen:
                    raise ValueError("existing results have mixed spec or duplicate trial keys")
                seen.add(record_key(record))
                counts[(record["policy"], record["status"], record["decision"])] += 1
    total = len(spec["cases"]) * len(spec["policies"]) * spec["repetitions"]
    tasks = []
    # Interleave cells at every chunk index: partial results must not be a
    # convenient subset of cases, and a live progress count covers the grid.
    for first in range(0, spec["repetitions"], chunk_size):
        for ci, case in enumerate(spec["cases"]):
            for policy in spec["policies"]:
                ids = [i for i in range(first, min(first + chunk_size, spec["repetitions"]))
                       if (case["case_id"], policy, i) not in seen]
                if ids:
                    tasks.append((ci, policy, ids))
    initial_seen = len(seen)
    def report(state):
        snapshot = {
            "schema": "faulttune.circuit_run_status.v1", "spec_sha256": digest,
            "spec": str(Path(spec_path).resolve()), "trials": str(trials_path.resolve()),
            "state": state, "completed_records": len(seen), "planned_records": total,
            "new_records": len(seen) - initial_seen, "elapsed_seconds": time.time() - started,
            "started_epoch": started, "updated_epoch": time.time(), "deadline_epoch": deadline,
            "workers": workers, "chunk_size": chunk_size,
            "purpose": "later_reproduction" if reproduce else "original_goal_experiment",
            "outcomes": [{"policy": p, "status": s, "decision": d, "count": n}
                         for (p, s, d), n in sorted(counts.items())],
        }
        save_json(status_path, snapshot)
        print(encoded({key: snapshot[key] for key in ("state", "completed_records", "planned_records",
                                                       "elapsed_seconds")}).strip(), flush=True)
        return snapshot
    report("running")
    pool = mp.Pool(workers, initializer=_initialize_worker, initargs=(spec, digest, deadline))
    last_report = time.time()
    state = "complete"
    try:
        # Bound returned full traces in RAM. Pool.imap_unordered can buffer an
        # entire fast simulation while the parent is still compressing logs.
        ready = queue.Queue()
        task_iterator = iter(tasks)
        in_flight = 0
        def submit_one():
            nonlocal in_flight
            try:
                task = next(task_iterator)
            except StopIteration:
                return False
            pool.apply_async(_run_chunk, (task,),
                             callback=lambda batch: ready.put((True, batch)),
                             error_callback=lambda error: ready.put((False, error)))
            in_flight += 1
            return True
        for unused in range(2 * workers):
            if not submit_one():
                break
        with gzip.open(str(trials_path), "at", encoding="utf-8", newline="\n", compresslevel=1) as handle:
            while in_flight:
                try:
                    success, batch = ready.get(timeout=1)
                except queue.Empty:
                    if time.time() > deadline + 15:
                        state = "deadline_incomplete"
                        break
                    if time.time() - last_report >= 15:
                        handle.flush()
                        report("running")
                        last_report = time.time()
                    continue
                in_flight -= 1
                if not success:
                    raise batch
                for record in batch:
                    key = record_key(record)
                    if key in seen:
                        raise ValueError("duplicate returned trial")
                    seen.add(key)
                    handle.write(encoded(record))
                    counts[(record["policy"], record["status"], record["decision"])] += 1
                submit_one()
                if time.time() - last_report >= 15:
                    handle.flush()
                    report("running")
                    last_report = time.time()
            handle.flush()
        if state == "complete":
            pool.close()
        else:
            pool.terminate()
        pool.join()
    except BaseException:
        pool.terminate()
        pool.join()
        report("interrupted_incomplete")
        raise
    result = report(state)
    result["raw_sha256"] = sha256_file(trials_path)
    save_json(status_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    design = sub.add_parser("spec")
    design.add_argument("--phase", choices=("development", "confirmation"), required=True)
    design.add_argument("--output", required=True)
    design.add_argument("--repetitions", type=int, required=True)
    design.add_argument("--policies", nargs="+", default=list(POLICIES))
    design.add_argument("--master-seed", type=int, default=202611031)
    design.add_argument("--max-extra", type=int, default=262144)
    run = sub.add_parser("run")
    run.add_argument("--spec", required=True)
    run.add_argument("--prefix", required=True)
    run.add_argument("--workers", type=int, default=12)
    run.add_argument("--chunk-size", type=int, default=4)
    run.add_argument("--wall-seconds", type=float)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--reproduce", action="store_true",
                     help="later reproduction after the original deadline; requires --wall-seconds")
    args = parser.parse_args()
    if args.command == "spec":
        write_spec(args.output, args.phase, args.repetitions, args.policies,
                   args.master_seed, args.max_extra)
    else:
        run_grid(args.spec, args.prefix, args.workers, args.chunk_size,
                 args.wall_seconds, args.resume, args.reproduce)


if __name__ == "__main__":
    main()
