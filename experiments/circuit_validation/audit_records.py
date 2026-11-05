"""Independent ledger and certificate audit for frozen circuit experiments.

Usage: python -B audit_records.py --spec SPEC.json --trials TRIALS.jsonl.gz
Plain JSONL is also accepted. Every retained record is checked, including
timeouts/errors. Final declared decisions are recomputed from paid counts.
Optional intermediate checks diagnose earlier exact crossings; numerical
screens are permitted to delay stopping, so that diagnostic is not a failure.
"""

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import sys
import zipfile

from direct_tests import certificate_from_pairs, direct_kl_certificate, fixed_paired_certificate
from kl_trackstop import joint_glr_certificate


AUDIT_SCHEMA = "faulttune.circuit_record_audit.v1"
TRIAL_SCHEMA = "faulttune.circuit_trial.v1"
SPEC_SCHEMA = "faulttune.circuit_spec.v1"
PAIR_POLICIES = {"direct_fixed", "direct_mixture", "direct_intervals", "calibration_only"}
HERE = Path(__file__).resolve().parent
DEFAULT_REPO = HERE.parent.parent
_WORKER_SPEC = None
_WORKER_SPEC_HASH = None


def _arm_policy(policy):
    # The frozen runner contract gives every other registered arm policy
    # arm_successes/arm_trials and the same joint GLR certificate. This also
    # covers new allocations such as combgame_glr without hardcoded names.
    return policy not in PAIR_POLICIES


def _integer(value, minimum=0):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _vector(value, length):
    return isinstance(value, (list, tuple)) and len(value) == length and all(_integer(x) for x in value)


def _close(left, right):
    return _number(left) and _number(right) and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-9)


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def derive_seed(master_seed, phase, case_id, policy, trial_index):
    """Independent implementation of the registered seed serialization."""
    encoded = json.dumps((master_seed, phase, case_id, policy, trial_index),
                         sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return int.from_bytes(hashlib.blake2b(encoded, digest_size=16, person=b"ft-circuit-v1").digest(), "big")


def expected_public_config(spec, case):
    opaque = hashlib.sha256((case["case_id"] + "::public").encode("utf-8")).hexdigest()
    result = {"contexts": list(case["contexts"]), "weights": list(case["weights"]),
              "delta": spec["delta"], "epsilon": case["epsilon"],
              "shot_budget": case["shot_budget"], "candidate_id": opaque[:32],
              "workload_id": opaque[32:], "phase": spec["phase"]}
    result.update(spec["observation_contract"]["cost"])
    return result


def _equivalent(left, right):
    """Match independently recomputed certificates, allowing numeric roundoff."""
    if isinstance(right, dict):
        return isinstance(left, dict) and all(key in left and _equivalent(left[key], value)
                                              for key, value in right.items())
    if isinstance(right, (list, tuple)):
        return isinstance(left, (list, tuple)) and len(left) == len(right) and all(
            _equivalent(a, b) for a, b in zip(left, right))
    if isinstance(right, bool) or right is None:
        return left is right
    if isinstance(right, int):
        return _integer(left) and left == right if right >= 0 else type(left) is int and left == right
    if isinstance(right, float):
        if not math.isfinite(right):
            expected = "Infinity" if right > 0 else "-Infinity" if right < 0 else "NaN"
            return left == expected or type(left) is float and left == right
        return _number(left) and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-300)
    if isinstance(right, str) and isinstance(left, str) and left != right:
        # Decimal audit strings may differ in last digits across math-library
        # builds. The independently recomputed decision remains authoritative.
        try:
            from decimal import Decimal
            a, b = Decimal(left), Decimal(right)
            return a.is_finite() and b.is_finite() and abs(a - b) <= max(abs(a), abs(b)) * Decimal("1e-12")
        except Exception:
            return False
    return left == right


def _next_total(current, limit, initial, growth, max_extra):
    target = initial if current == 0 else max(current + initial, int(math.ceil(current * growth)))
    return min(limit, current + max_extra, target)


def _certificate(policy, counts, failures, pair_counts, config):
    if policy == "direct_fixed":
        return fixed_paired_certificate(pair_counts, config["delta"])
    if policy in ("direct_mixture", "calibration_only"):
        return certificate_from_pairs(pair_counts, config["delta"])
    if policy == "direct_intervals":
        pairs = sum(pair_counts)
        return direct_kl_certificate(pair_counts[2] + pair_counts[3], pairs,
                                     pair_counts[1] + pair_counts[3], pairs, config["delta"])
    coefficients = [-weight for weight in config["weights"]] + list(config["weights"])
    return joint_glr_certificate(failures, counts, coefficients, config["delta"])


def replay_ledger(events, config, policy, policy_parameters, fixed_pairs=None,
                  allow_partial_checkpoint=False):
    """Reconstruct all paid costs and sufficient statistics from event logs."""
    errors = []

    def check(condition, code, detail):
        if not condition:
            errors.append({"code": code, "detail": detail})
        return condition

    number = len(config["contexts"])
    arms = 2 * number
    counts, failures, pair_counts = [0] * arms, [0] * arms, [0] * 4
    shots = switches = 0
    duration = 0.0
    last_arm = None
    digest = hashlib.sha256()
    checkpoints = []
    checkpoint_arm = -1
    parameters = policy_parameters
    growth, initial, maximum = parameters["growth"], parameters["initial"], parameters["max_extra"]
    macro_budget = parameters.get("combgame_macro_budget") if policy == "combgame_glr" else None
    block_size = ((config["shot_budget"] + macro_budget - 1) // macro_budget
                  if macro_budget is not None else None)
    target = (min(config["shot_budget"], block_size) if block_size is not None
              else _next_total(0, config["shot_budget"], initial * arms, growth, maximum)
              if _arm_policy(policy) else None)

    def charge(arm, amount, failed):
        nonlocal shots, switches, duration, last_arm
        if last_arm is not None and arm != last_arm:
            switches += 1
        last_arm = arm
        counts[arm] += amount
        failures[arm] += failed
        shots += amount
        duration += amount * config["old_duration" if arm < number else "new_duration"]

    for event_index, event in enumerate(events):
        if not check(isinstance(event, dict), "event_type", "event {} must be an object".format(event_index)):
            continue
        try:
            payload = json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False)
            digest.update(payload.encode("ascii") + b"\n")
        except (ValueError, TypeError, UnicodeEncodeError):
            check(False, "event_encoding", "event {} does not have the registered ASCII JSON encoding".format(event_index))
        kind = event.get("kind")
        if kind == "arm":
            check(_arm_policy(policy), "event_policy_scope", "paired policy issued a singleton-arm event")
            arm, amount, failed = event.get("arm"), event.get("shots"), event.get("failures")
            if not check(_integer(arm) and arm < arms and _integer(amount, 1)
                         and _integer(failed) and failed <= amount,
                         "arm_event_counts", "invalid arm event at {}".format(event_index)):
                continue
            check(_integer(event.get("start")) and event["start"] == counts[arm],
                  "event_offset", "arm start offset does not match paid history")
            check(arm > checkpoint_arm, "checkpoint_arm_order", "arm calls must be increasing within a frozen policy checkpoint")
            checkpoint_arm = arm
            if block_size is not None:
                check(amount == min(block_size, config["shot_budget"] - shots), "macro_block_size",
                      "CombGame must sample one whole frozen block per macro round, except the final budget remainder")
                if len(checkpoints) < arms:
                    check(arm == len(checkpoints), "macro_initialization_order",
                          "CombGame initialization must sample each arm once in index order")
            charge(arm, amount, failed)
            if target is not None:
                check(shots <= target, "checkpoint_size", "arm event crossed the registered checkpoint size")
                if shots == target:
                    checkpoints.append({"event_index": event_index, "shots": shots,
                                        "counts": counts[:], "failures": failures[:], "pair_counts": None})
                    checkpoint_arm = -1
                    if shots < config["shot_budget"]:
                        target = (min(config["shot_budget"], shots + block_size) if block_size is not None
                                  else _next_total(shots, config["shot_budget"], initial * arms, growth, maximum))
        elif kind == "pairs":
            check(policy in PAIR_POLICIES, "event_policy_scope", "arm policy issued a paired event")
            mode = event.get("mode")
            expected_mode = "fixed" if policy == "calibration_only" else "iid"
            check(mode == expected_mode, "context_scope", "paired context mode disagrees with the policy's estimand")
            amount, pieces, joint = event.get("pairs"), event.get("pieces"), event.get("joint")
            if not check(_integer(amount, 1) and isinstance(pieces, list) and _vector(joint, 4),
                         "pair_event_counts", "invalid paired event at {}".format(event_index)):
                continue
            previous_pairs = sum(pair_counts)
            if policy == "direct_fixed":
                check(event_index == 0 and amount == fixed_pairs, "fixed_sample_size",
                      "fixed test must execute exactly one preregistered pair count")
            else:
                expected_total = _next_total(previous_pairs, config["shot_budget"] // 2,
                                            initial, growth, maximum // 2)
                check(amount == expected_total - previous_pairs, "checkpoint_size",
                      "paired batch differs from the registered checkpoint schedule")
            summed_joint, total_pairs, previous_context = [0] * 4, 0, -1
            for piece in pieces:
                if not check(isinstance(piece, dict), "piece_type", "paired piece must be an object"):
                    continue
                context, size, local = piece.get("context"), piece.get("pairs"), piece.get("joint")
                if not check(_integer(context) and context < number and _integer(size, 1)
                             and _vector(local, 4) and sum(local) == size,
                             "piece_counts", "invalid context-piece counts"):
                    continue
                check(context > previous_context, "piece_context_order", "context pieces must be distinct and ascending")
                previous_context = context
                if policy == "calibration_only":
                    check(context == 0, "context_scope", "calibration-only policy sampled a different fixed context")
                check(_integer(piece.get("old_start")) and _integer(piece.get("new_start"))
                      and piece["old_start"] == counts[context]
                      and piece["new_start"] == counts[number + context],
                      "event_offset", "paired start offsets do not match paid history")
                charge(context, size, local[2] + local[3])
                charge(number + context, size, local[1] + local[3])
                summed_joint = [a + b for a, b in zip(summed_joint, local)]
                total_pairs += size
            check(total_pairs == amount and summed_joint == joint and sum(joint) == amount,
                  "pair_margins", "piece margins do not sum to the paired event")
            pair_counts = [a + b for a, b in zip(pair_counts, joint)]
            checkpoints.append({"event_index": event_index, "shots": shots,
                                "counts": counts[:], "failures": failures[:], "pair_counts": pair_counts[:]})
        else:
            check(False, "event_kind", "unknown event kind at {}".format(event_index))
        check(shots <= config["shot_budget"], "event_budget", "paid event history exceeds budget")
    if _arm_policy(policy) and shots and not allow_partial_checkpoint:
        check(bool(checkpoints) and checkpoints[-1]["shots"] == shots,
              "incomplete_checkpoint", "successful arm policy stopped in an incomplete batch")
    return {"errors": errors, "counts": counts, "failures": failures,
            "pair_counts": pair_counts, "shots": shots, "switches": switches,
            "batches": len(events), "execution_duration": duration,
            "modeled_device_duration": duration + switches * config["switch_duration"],
            "readout_objects": shots * config["measured_qubits"],
            "event_sha256": digest.hexdigest(), "checkpoints": checkpoints}


def audit_record(record, spec, spec_sha256, intermediate=False):
    """Return issues and diagnostics for one record; does not mutate inputs."""
    issues = []

    def check(condition, code, detail):
        if not condition:
            issues.append({"code": code, "detail": detail})
        return condition

    diagnostics = {"final_certificate_recomputed": False, "intermediate_checkpoints_recomputed": 0,
                   "earlier_exact_crossings": []}
    if not check(isinstance(record, dict), "record_type", "record must be an object"):
        return {"issues": issues, "diagnostics": diagnostics}
    check(record.get("schema") == TRIAL_SCHEMA, "trial_schema", "unexpected trial schema")
    check(record.get("spec_sha256") == spec_sha256, "spec_hash", "record does not identify the supplied frozen spec")
    check(record.get("phase") == spec["phase"], "phase", "record phase differs from the frozen spec")
    case = next((case for case in spec["cases"] if case["case_id"] == record.get("case_id")), None)
    policy, index = record.get("policy"), record.get("trial_index")
    valid_key = check(case is not None, "case_id", "case is not in the frozen spec")
    valid_key = check(isinstance(policy, str) and policy in spec["policies"],
                      "policy", "policy is not registered") and valid_key
    valid_key = check(_integer(index) and index < spec["repetitions"],
                      "trial_index", "trial index lies outside the frozen repetitions") and valid_key
    if not valid_key:
        return {"issues": issues, "diagnostics": diagnostics}
    expected_seed = str(derive_seed(spec["master_seed"], spec["phase"], case["case_id"], policy, index))
    check(record.get("seed") == expected_seed, "seed", "seed does not match the independent registered derivation")
    config = expected_public_config(spec, case)
    check(record.get("public_config") == config, "public_config", "public config differs from the frozen case/cost contract")
    check(config["measured_qubits"] == 3, "measurement_scope", "this circuit family requires all three readouts")
    shots, budget = record.get("shots"), record.get("shot_budget")
    check(_integer(shots) and _integer(budget, 1) and shots <= budget,
          "shot_budget", "invalid or exceeded record shot budget")
    check(budget == config["shot_budget"], "registered_budget", "record budget differs from frozen case budget")
    status, decision = record.get("status"), record.get("decision")
    check(status in ("ok", "error", "timeout"), "status", "unknown trial status")
    check(decision in ("accept", "reject", "abstain"), "decision", "unknown trial decision")
    result = record.get("result")
    if status in ("error", "timeout"):
        check(decision == "abstain" and result is None, "failed_trial_result",
              "errors/timeouts must retain abstention and no completed result")
    else:
        check(isinstance(result, dict), "result_missing", "successful trial lacks its complete policy result")
    cost = record.get("cost")
    events = record.get("events")
    if not check(isinstance(cost, dict) and isinstance(events, list), "ledger_missing",
                 "every trial, including failures, must retain cost and events"):
        return {"issues": issues, "diagnostics": diagnostics}
    check(_vector(cost.get("counts"), 2 * len(config["contexts"]))
          and _vector(cost.get("failures"), 2 * len(config["contexts"])),
          "cost_count_types", "cost counts/failures must be nonnegative integer vectors")
    check(all(_integer(cost.get(field)) for field in ("readout_objects", "switches", "batches")),
          "cost_integer_types", "readout objects, switches and batches must be nonnegative integers")
    ledger = replay_ledger(events, config, policy, spec["policy_parameters"], case.get("fixed_pairs"),
                           allow_partial_checkpoint=status != "ok")
    issues.extend(ledger["errors"])
    check(shots == ledger["shots"], "ledger_shots", "record shots differ from independently replayed events")
    for field in ("counts", "failures", "readout_objects", "switches", "batches", "event_sha256"):
        check(cost.get(field) == ledger[field], "ledger_" + field,
              "cost {} differs from replayed ledger".format(field))
    if "shots" in cost:
        check(cost["shots"] == ledger["shots"], "cost_shots", "cost shots disagree with event history")
    for field in ("execution_duration", "modeled_device_duration"):
        check(_close(cost.get(field), ledger[field]), "ledger_" + field,
              "cost {} differs from independent duration accounting".format(field))
    check(_digest(cost.get("event_sha256")), "event_digest_format", "event digest is not lowercase SHA256")
    if status != "ok" or not isinstance(result, dict):
        return {"issues": issues, "diagnostics": diagnostics}
    check(result.get("decision") == decision, "result_decision", "record and policy-result decisions disagree")
    check(result.get("guarantee_in_scope") is (policy != "calibration_only"), "guarantee_scope",
          "calibration-only is a local diagnostic; all registered target policies are in scope")
    check(_integer(result.get("checks")) and result["checks"] == len(ledger["checkpoints"]), "checkpoint_count",
          "reported checkpoint count differs from the event history")
    if policy == "combgame_glr":
        macro_budget = spec["policy_parameters"]["combgame_macro_budget"]
        block_size = (config["shot_budget"] + macro_budget - 1) // macro_budget
        metadata = result.get("sampling_metadata")
        if check(isinstance(metadata, dict), "macro_metadata_missing", "CombGame must report its public block contract"):
            expected_macro = {"macro_budget": macro_budget, "block_size": block_size,
                              "complete_blocks": ledger["shots"] // block_size,
                              "final_partial_shots": ledger["shots"] % block_size}
            for name, expected in expected_macro.items():
                check(_integer(metadata.get(name)) and metadata[name] == expected, "macro_metadata",
                      "CombGame {} does not match its paid block history".format(name))
    if policy in PAIR_POLICIES:
        check(_vector(result.get("pair_counts"), 4) and result["pair_counts"] == ledger["pair_counts"], "result_pair_counts",
              "policy paired sufficient statistics disagree with the paid ledger")
    else:
        check(_vector(result.get("arm_trials"), 2 * len(config["contexts"]))
              and _vector(result.get("arm_successes"), 2 * len(config["contexts"]))
              and result["arm_trials"] == ledger["counts"]
              and result["arm_successes"] == ledger["failures"],
              "result_arm_counts", "policy arm sufficient statistics disagree with paid observations")
    stored = result.get("certificate")
    if not check(isinstance(stored, dict), "certificate_missing", "successful result lacks a certificate"):
        return {"issues": issues, "diagnostics": diagnostics}
    check(stored.get("decision") == decision, "certificate_decision", "stored certificate decision differs from record")
    check(stored.get("shots") == ledger["shots"], "certificate_shots", "certificate omits or adds paid shots")
    if decision in ("accept", "reject"):
        try:
            recomputed = _certificate(policy, ledger["counts"], ledger["failures"], ledger["pair_counts"], config)
            diagnostics["final_certificate_recomputed"] = True
            check(recomputed["decision"] == decision, "recomputed_decision",
                  "paid observations do not reproduce the declared decision")
            check(_equivalent(stored, recomputed), "certificate_reproduction",
                  "stored certificate differs from the corresponding independently recomputed certificate")
        except Exception as error:
            check(False, "certificate_recompute_error", type(error).__name__ + ": " + str(error))
    if intermediate and policy != "direct_fixed":
        for checkpoint in ledger["checkpoints"][:-1]:
            try:
                row = _certificate(policy, checkpoint["counts"], checkpoint["failures"],
                                   checkpoint["pair_counts"], config)
                diagnostics["intermediate_checkpoints_recomputed"] += 1
                if row["decision"] != "abstain":
                    diagnostics["earlier_exact_crossings"].append(
                        {"shots": checkpoint["shots"], "event_index": checkpoint["event_index"],
                         "decision": row["decision"]})
            except Exception as error:
                check(False, "intermediate_recompute_error", type(error).__name__ + ": " + str(error))
    return {"issues": issues, "diagnostics": diagnostics}


def validate_spec(spec):
    if not isinstance(spec, dict) or spec.get("schema") != SPEC_SCHEMA:
        raise ValueError("unsupported specification schema")
    if type(spec.get("master_seed")) is not int or not _integer(spec.get("repetitions"), 1):
        raise ValueError("master seed must be an integer and repetitions positive")
    policies = spec.get("policies")
    if not isinstance(policies, list) or not policies or any(not isinstance(p, str) or not p for p in policies) or len(set(policies)) != len(policies):
        raise ValueError("registered policies must be nonempty strings and unique")
    cases = spec.get("cases")
    if not isinstance(cases, list) or not cases or len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("registered cases must be nonempty and unique")
    parameters = spec["policy_parameters"]
    if not _number(parameters["growth"]) or not 1 < parameters["growth"] <= 2:
        raise ValueError("invalid checkpoint growth")
    if not _integer(parameters["initial"], 1) or not _integer(parameters["max_extra"], 2):
        raise ValueError("initial/max_extra must permit a positive paired batch")
    if "combgame_glr" in policies and not _integer(parameters.get("combgame_macro_budget"), 1):
        raise ValueError("CombGame macro budget must be explicitly frozen as a positive integer")
    for case in cases:
        config = expected_public_config(spec, case)
        if not _integer(config["shot_budget"], 1):
            raise ValueError("case shot budget must be positive")
        if len(config["contexts"]) != len(config["weights"]) or not config["contexts"]:
            raise ValueError("contexts must match target weights")
        if any(not _number(w) or w <= 0 for w in config["weights"]) or abs(math.fsum(config["weights"]) - 1) > 1e-12:
            raise ValueError("invalid target probabilities")
        if not 0 < config["delta"] < 1 or not 0 < config["epsilon"] <= 1:
            raise ValueError("invalid registered risk or practical gap")
        if any(not _number(config[name]) or config[name] < 0
               for name in ("old_duration", "new_duration", "switch_duration")):
            raise ValueError("invalid cost duration")
        if "direct_fixed" in policies and (not _integer(case.get("fixed_pairs"), 1)
                                           or 2 * case["fixed_pairs"] > case["shot_budget"]):
            raise ValueError("fixed pair design must be a registered positive integer within budget")


class RecordAuditor:
    def __init__(self, spec, spec_sha256, intermediate_records=0, max_issue_details=1000):
        validate_spec(spec)
        self.spec, self.spec_sha256 = spec, spec_sha256
        self.intermediate_records = intermediate_records
        self.max_issue_details = max_issue_details
        self.seen_keys, self.seen_seeds = set(), set()
        self.invalid_record_numbers = set()
        self.records = self.final_certificates = 0
        self.intermediate_checked = self.intermediate_checkpoints = 0
        self.intermediate_scheduled = 0
        self.status_counts, self.issue_counts = Counter(), Counter()
        self.issues, self.earlier_crossings = [], []
        self.case_ids = {case["case_id"] for case in spec["cases"]}

    def add_issue(self, code, detail, record_number=None):
        self.issue_counts[code] += 1
        if record_number is not None:
            self.invalid_record_numbers.add(record_number)
        if len(self.issues) < self.max_issue_details:
            self.issues.append({"record_number": record_number, "code": code, "detail": detail})

    def register(self, record, record_number=None):
        """Parent-only identity/hash/seed checks before dispatching record work."""
        self.records += 1
        number = self.records if record_number is None else record_number
        if isinstance(record, dict):
            key = (record.get("phase"), record.get("case_id"), record.get("policy"), record.get("trial_index"))
            try:
                if key in self.seen_keys:
                    self.add_issue("duplicate_key", str(key), number)
                self.seen_keys.add(key)
            except TypeError:
                self.add_issue("unhashable_key", "trial identity is malformed", number)
            seed = record.get("seed")
            if isinstance(seed, str):
                if seed in self.seen_seeds:
                    self.add_issue("duplicate_seed", seed, number)
                self.seen_seeds.add(seed)
            self.status_counts[str(record.get("status"))] += 1
            if record.get("spec_sha256") != self.spec_sha256:
                self.add_issue("parent_spec_hash", "record identifies a different frozen spec", number)
            if record.get("phase") != self.spec["phase"]:
                self.add_issue("parent_phase", "record phase differs from the frozen spec", number)
            case_id, policy, index = record.get("case_id"), record.get("policy"), record.get("trial_index")
            if isinstance(case_id, str) and case_id in self.case_ids and isinstance(policy, str) and policy in self.spec["policies"] and _integer(index):
                expected = str(derive_seed(self.spec["master_seed"], self.spec["phase"], case_id, policy, index))
                if seed != expected:
                    self.add_issue("parent_seed", "seed differs from the registered independent derivation", number)
        intermediate = (self.intermediate_scheduled < self.intermediate_records
                        and isinstance(record, dict) and record.get("status") == "ok")
        if intermediate:
            self.intermediate_scheduled += 1
        return number, intermediate

    def accept_result(self, result, number, intermediate=False):
        for issue in result["issues"]:
            self.add_issue(issue["code"], issue["detail"], number)
        diagnostic = result["diagnostics"]
        self.final_certificates += int(diagnostic.get("final_certificate_recomputed", False))
        checked = diagnostic.get("intermediate_checkpoints_recomputed", 0)
        if intermediate:
            self.intermediate_checked += 1
            self.intermediate_checkpoints += checked
        if diagnostic.get("earlier_exact_crossings"):
            self.earlier_crossings.append({"record_number": number,
                                           "crossings": diagnostic["earlier_exact_crossings"]})

    def consume(self, record, record_number=None):
        number, intermediate = self.register(record, record_number)
        self.accept_result(_safe_record_check(record, self.spec, self.spec_sha256, intermediate),
                           number, intermediate)

    def finish(self, require_complete=True):
        missing = 0
        examples = []
        for case in self.spec["cases"]:
            for policy in self.spec["policies"]:
                for index in range(self.spec["repetitions"]):
                    key = (self.spec["phase"], case["case_id"], policy, index)
                    if key not in self.seen_keys:
                        missing += 1
                        if len(examples) < 20:
                            examples.append(list(key))
        if missing and require_complete:
            self.add_issue("missing_records", "{} registered trial keys are absent".format(missing))
        planned = len(self.spec["cases"]) * len(self.spec["policies"]) * self.spec["repetitions"]
        total_issues = sum(self.issue_counts.values())
        return {"schema": AUDIT_SCHEMA, "passed": total_issues == 0,
                "complete": missing == 0, "spec_sha256": self.spec_sha256,
                "phase": self.spec["phase"], "planned_records": planned,
                "records_read": self.records, "invalid_records": len(self.invalid_record_numbers),
                "unique_trial_keys": len(self.seen_keys), "unique_seeds": len(self.seen_seeds),
                "missing_records": missing, "missing_key_examples": examples,
                "status_counts": dict(self.status_counts), "final_decision_certificates_recomputed": self.final_certificates,
                "issue_counts": dict(self.issue_counts),
                "issues": sorted(self.issues, key=lambda row: (row["record_number"] is None,
                                                                row["record_number"] or 0, row["code"])),
                "omitted_issue_details": max(0, total_issues - len(self.issues)),
                "intermediate_audit": {"records_requested": self.intermediate_records,
                                       "successful_records_checked": self.intermediate_checked,
                                       "checkpoints_recomputed": self.intermediate_checkpoints,
                                       "earlier_exact_crossings": sorted(self.earlier_crossings, key=lambda row: row["record_number"]),
                                       "scope": "diagnostic only: fast numerical screens may delay exact crossings; screens and adaptive allocation choices are not replayed"},
                "scope": "all logged batches and final declared certificates, with independent ledger/seed/spec checks; no resampling of RNG streams or validation of the physical model",
                "failure_records": "timeouts/errors remain in status counts and must retain their paid event history"}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key: " + key)
        result[key] = value
    return result


def strict_json_loads(text):
    def invalid_constant(value):
        raise ValueError("nonstandard JSON numeric constant: " + value)
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=invalid_constant)


def _safe_record_check(record, spec, spec_hash, intermediate):
    try:
        return audit_record(record, spec, spec_hash, intermediate=intermediate)
    except Exception as error:
        return {"issues": [{"code": "record_audit_exception", "detail": type(error).__name__ + ": " + str(error)}],
                "diagnostics": {}}


def _initialize_worker(spec, spec_hash):
    global _WORKER_SPEC, _WORKER_SPEC_HASH
    _WORKER_SPEC, _WORKER_SPEC_HASH = spec, spec_hash


def _audit_chunk(tasks):
    return [(number, _safe_record_check(record, _WORKER_SPEC, _WORKER_SPEC_HASH, intermediate), intermediate)
            for number, record, intermediate in tasks]


def _record_stream(trials_path, auditor):
    opener = gzip.open if trials_path.suffix.lower() == ".gz" else open
    try:
        with opener(trials_path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = strict_json_loads(line)
                except (ValueError, TypeError) as error:
                    auditor.records += 1
                    auditor.add_issue("record_json", str(error), line_number)
                    continue
                yield line_number, row
    except (OSError, EOFError, UnicodeDecodeError) as error:
        auditor.add_issue("trials_read_error", type(error).__name__ + ": " + str(error))


def _registered_chunks(stream, auditor, chunk_records):
    chunk = []
    for number, record in stream:
        number, intermediate = auditor.register(record, number)
        chunk.append((number, record, intermediate))
        if len(chunk) == chunk_records:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _source_evidence(spec, repo_root, source_archive=None):
    """Authenticate selected source provenance without executing/extracting ZIP contents."""
    rows, issues = {}, []
    expected_hashes = spec.get("source_hashes", {})
    for name, expected in expected_hashes.items():
        path = (repo_root / name).resolve()
        try:
            path.relative_to(repo_root)
            current = hashlib.sha256(path.read_bytes()).hexdigest()
        except (OSError, ValueError):
            current = None
        rows[name] = {"expected": expected, "current_sha256": current,
                      "current_matches": current == expected,
                      "archived_sha256": None, "archived_matches": None}
    current_matches = all(row["current_matches"] for row in rows.values())
    archive = {"selected": source_archive is not None, "path": None, "sha256": None,
               "expected_sha256": None, "container_hash_matches": None}
    archived_matches = None
    if source_archive is not None:
        path = Path(source_archive).resolve()
        archive["path"] = str(path)
        declared_archive = spec.get("source_archive", {})
        if isinstance(declared_archive, dict):
            archive["expected_sha256"] = declared_archive.get("sha256")
        archived_hashes = {}
        try:
            archive["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            if archive["expected_sha256"] is not None:
                archive["container_hash_matches"] = archive["sha256"] == archive["expected_sha256"]
                if not archive["container_hash_matches"]:
                    issues.append({"code": "archive_container_hash", "detail": "source archive differs from its frozen container SHA256"})
            with zipfile.ZipFile(path) as handle:
                seen = set()
                for member in handle.infolist():
                    if member.is_dir():
                        continue
                    name = member.filename
                    if name in seen:
                        issues.append({"code": "archive_duplicate_entry", "detail": "duplicate source archive entry: " + name})
                    seen.add(name)
                    if name not in expected_hashes:
                        issues.append({"code": "archive_extra_entry", "detail": "archive entry is absent from frozen source hashes: " + name})
                    archived_hashes[name] = hashlib.sha256(handle.read(member)).hexdigest()
        except (OSError, zipfile.BadZipFile, RuntimeError) as error:
            issues.append({"code": "source_archive_read", "detail": type(error).__name__ + ": " + str(error)})
        for name, row in rows.items():
            row["archived_sha256"] = archived_hashes.get(name)
            row["archived_matches"] = row["archived_sha256"] == row["expected"]
            if not row["archived_matches"]:
                issues.append({"code": "archived_source_hash", "detail": "archive source missing or changed: " + name})
        archived_matches = all(row["archived_matches"] for row in rows.values()) and not issues
    else:
        for name, row in rows.items():
            if not row["current_matches"]:
                issues.append({"code": "source_hash", "detail": "frozen current source changed or is unavailable: " + name})
    return {"rows": rows, "issues": issues, "current_source_matches": current_matches,
            "archived_source_matches": archived_matches, "archive": archive,
            "selected_mode": "archive_explicit" if source_archive is not None else "current"}


def audit_files(spec_path, trials_path, output_path=None, repo_root=DEFAULT_REPO,
                intermediate_records=0, require_complete=True, check_sources=True,
                workers=1, chunk_records=16, progress_every=0, source_archive=None):
    if not _integer(workers, 1) or not _integer(chunk_records, 1) or not _integer(intermediate_records):
        raise ValueError("positive workers/chunk_records and nonnegative intermediate_records required")
    if source_archive is not None and not check_sources:
        raise ValueError("an explicitly selected archive cannot be combined with skipped source verification")
    spec_path, trials_path, repo_root = Path(spec_path), Path(trials_path), Path(repo_root).resolve()
    spec_bytes = spec_path.read_bytes()
    spec = strict_json_loads(spec_bytes.decode("utf-8"))
    spec_hash = hashlib.sha256(spec_bytes).hexdigest()
    auditor = RecordAuditor(spec, spec_hash, intermediate_records)
    sidecar = spec_path.with_suffix(spec_path.suffix + ".sha256")
    sidecar_parts = sidecar.read_text(encoding="ascii").split() if sidecar.is_file() else []
    if not sidecar_parts or sidecar_parts[0] != spec_hash:
        auditor.add_issue("spec_sidecar_hash", "frozen specification sidecar hash is absent or mismatched")
    sources = {"rows": {}, "current_source_matches": None, "archived_source_matches": None,
               "archive": None, "selected_mode": "skipped"}
    if check_sources:
        sources = _source_evidence(spec, repo_root, source_archive)
        for issue in sources["issues"]:
            auditor.add_issue(issue["code"], issue["detail"])
    stream = _record_stream(trials_path, auditor)
    completed, next_progress = 0, progress_every
    if workers == 1:
        for number, record in stream:
            auditor.consume(record, number)
            completed += 1
            if progress_every and completed >= next_progress:
                print("Audited {} records".format(completed), file=sys.stderr, flush=True)
                next_progress += progress_every
    else:
        context = multiprocessing.get_context("spawn")
        with context.Pool(workers, initializer=_initialize_worker, initargs=(spec, spec_hash)) as pool:
            chunks = _registered_chunks(stream, auditor, chunk_records)
            for results in pool.imap_unordered(_audit_chunk, chunks, chunksize=1):
                for number, result, intermediate in results:
                    auditor.accept_result(result, number, intermediate)
                completed += len(results)
                if progress_every and completed >= next_progress:
                    print("Audited {} records with {} workers".format(completed, workers), file=sys.stderr, flush=True)
                    next_progress += progress_every
    result = auditor.finish(require_complete)
    own_path = Path(__file__).resolve()
    own_relative = own_path.relative_to(DEFAULT_REPO).as_posix()
    own_hash = hashlib.sha256(own_path.read_bytes()).hexdigest()
    audit_frozen = spec.get("source_hashes", {}).get(own_relative) == own_hash
    certificate_modules = {}
    for name in ("direct_tests.py", "kl_trackstop.py"):
        path = HERE / name
        relative = path.relative_to(DEFAULT_REPO).as_posix()
        current_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        frozen_hash = spec.get("source_hashes", {}).get(relative)
        certificate_modules[relative] = {"current_sha256": current_hash, "frozen_sha256": frozen_hash,
                                          "matches_frozen": current_hash == frozen_hash}
    result.update({"spec_path": str(spec_path.resolve()), "trials_path": str(trials_path.resolve()),
                   "source_hash_verification_enabled": check_sources,
                   "workers": workers, "chunk_records": chunk_records,
                   "source_hash_checks": sources["rows"],
                   "source_verification_mode": sources["selected_mode"],
                   "current_source_matches": sources["current_source_matches"],
                   "archived_source_matches": sources["archived_source_matches"],
                   "source_archive": sources["archive"],
                   "audit_tool_sha256": own_hash, "audit_tool_frozen_in_spec": audit_frozen,
                   "current_certificate_module_provenance": certificate_modules,
                   "audit_version_scope": "the independent checker and its numerical certificate modules may be newer than the frozen experimental source; their current hashes are reported separately",
                   "confirmation_engineering_eligible": result["passed"] and result["complete"]
                       and check_sources and spec["phase"] == "confirmation"})
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--intermediate-records", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-records", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--skip-source-check", action="store_true")
    parser.add_argument("--source-archive", type=Path,
                        help="explicitly authenticate frozen ZIP entries instead of requiring current sources to match")
    args = parser.parse_args()
    if args.intermediate_records < 0 or args.workers < 1 or args.chunk_records < 1 or args.progress_every < 0:
        parser.error("workers/chunk-records must be positive; intermediate-records/progress-every nonnegative")
    result = audit_files(args.spec, args.trials, args.output, args.repo_root,
                         args.intermediate_records, not args.allow_partial, not args.skip_source_check,
                         args.workers, args.chunk_records, args.progress_every, args.source_archive)
    print(json.dumps({key: result[key] for key in ("passed", "complete", "records_read", "planned_records",
                                                  "invalid_records", "missing_records", "issue_counts",
                                                  "final_decision_certificates_recomputed")}, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
