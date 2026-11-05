"""Paid, independently sampled circuit observations for fixed-candidate validation.

The circuit simulator supplies private probabilities to the runner. Policies
receive only PublicExperiment and the two sampling methods. This is a research
API contract, not a sandbox against hostile Python introspection.

NumPy is used to draw the exact IID sampling law in batches. Simulating the
unitary once and sampling its outcome law does not create free device shots:
every delivered old/new outcome is charged as a separate circuit execution.
"""

import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass

import numpy as np


class BudgetExhausted(RuntimeError):
    pass


class DeadlineExceeded(RuntimeError):
    pass


class TranscriptExhausted(RuntimeError):
    pass


def stable_seed(*parts):
    value = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return int.from_bytes(hashlib.blake2b(value.encode("ascii"), digest_size=16,
                                         person=b"ft-circuit-v1").digest(), "big")


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(name + " must be a positive integer")


@dataclass(frozen=True)
class PublicExperiment:
    contexts: tuple = ("spectator_0", "spectator_plus")
    weights: tuple = (0.5, 0.5)
    delta: float = 0.01
    epsilon: float = 0.001
    shot_budget: int = 1048576
    measured_qubits: int = 3
    # Reset/readout plus two H layers cost 2.2 normalized time units. The
    # physical control window adds tau_old=1 or tau_new=2.
    old_duration: float = 3.2
    new_duration: float = 4.2
    switch_duration: float = 0.1
    candidate_id: str = "fixed-candidate"
    workload_id: str = "fixed-workload"
    phase: str = "development"

    def __post_init__(self):
        contexts, weights = tuple(self.contexts), tuple(self.weights)
        if not contexts or len(set(contexts)) != len(contexts) or len(contexts) != len(weights):
            raise ValueError("contexts must be distinct and match weights")
        if any(not isinstance(c, str) or not c for c in contexts):
            raise ValueError("contexts must be nonempty strings")
        if any(not math.isfinite(w) or w <= 0 for w in weights):
            raise ValueError("context weights must be finite and positive")
        if abs(math.fsum(weights) - 1.0) > 1e-12:
            raise ValueError("context weights must sum to one")
        if not 0 < self.delta < 0.5 or not 0 < self.epsilon <= 1:
            raise ValueError("invalid error budget or practical gap")
        positive_integer(self.shot_budget, "shot_budget")
        positive_integer(self.measured_qubits, "measured_qubits")
        if any(not math.isfinite(v) or v < 0 for v in
               (self.old_duration, self.new_duration, self.switch_duration)):
            raise ValueError("durations must be finite and nonnegative")
        object.__setattr__(self, "contexts", contexts)
        object.__setattr__(self, "weights", weights)

    @property
    def arms(self):
        return 2 * len(self.contexts)

    @property
    def coefficients(self):
        return tuple(-w for w in self.weights) + self.weights


class CircuitObservationOracle:
    """One arm is one (old/new, context) circuit; all are independently paid.

    Arm order is old contexts, followed by new contexts. sample_pairs chooses
    a shared PUBLIC context, then draws INDEPENDENT old and new measurement
    noise. Its joint-count order is 00,01,10,11 (old bit first). Reordering a
    batch by context and branch reduces switches without changing its IID law.
    """

    def __init__(self, config, probabilities, seed, deadline_epoch=None,
                 transcripts=None, keep_events=False):
        if not isinstance(config, PublicExperiment):
            raise ValueError("expected PublicExperiment")
        probabilities = tuple(probabilities)
        if len(probabilities) != config.arms or any(not 0 <= p <= 1 for p in probabilities):
            raise ValueError("one valid failure probability is required per arm")
        self._config = config
        self._probabilities = probabilities
        self._seed = seed
        self._deadline = deadline_epoch
        self._streams = [np.random.Generator(np.random.PCG64(stable_seed(seed, "arm", arm)))
                         for arm in range(config.arms)]
        self._context_stream = np.random.Generator(np.random.PCG64(stable_seed(seed, "context")))
        self._transcripts = None
        if transcripts is not None:
            self._transcripts = {}
            for arm, values in transcripts.items():
                self._check_arm(arm)
                values = tuple(values)
                if any(v not in (0, 1) for v in values):
                    raise ValueError("transcripts must contain bits")
                self._transcripts[arm] = np.asarray(values, dtype=np.uint8)
        self.counts = [0] * config.arms
        self.failures = [0] * config.arms
        self.shots = self.switches = self.batches = 0
        self.execution_duration = 0.0
        self._last_arm = None
        self._digest = hashlib.sha256()
        self.events = [] if keep_events else None

    def _check_arm(self, arm):
        if isinstance(arm, bool) or not isinstance(arm, int) or not 0 <= arm < self._config.arms:
            raise ValueError("invalid circuit arm")

    def _preflight(self, shots):
        positive_integer(shots, "shots")
        if self._deadline is not None and time.time() >= self._deadline:
            raise DeadlineExceeded("goal/experiment wall-clock deadline reached")
        if self.shots + shots > self._config.shot_budget:
            raise BudgetExhausted("requested executions exceed the remaining shot budget")

    def _check_transcript(self, arm, shots):
        if self._transcripts is not None and self.counts[arm] + shots > len(self._transcripts.get(arm, ())):
            raise TranscriptExhausted("circuit arm has insufficient fixed observations")

    def _draw(self, arm, shots):
        if self._transcripts is not None:
            start = self.counts[arm]
            return self._transcripts[arm][start:start + shots]
        # Each uniform consumes one PCG64 output, so the kth outcome on an
        # arm is invariant to how callers split its requests into batches.
        return (self._streams[arm].random(shots) < self._probabilities[arm]).astype(np.uint8)

    def _charge(self, arm, shots, failures):
        if shots == 0:
            return
        if self._last_arm is not None and arm != self._last_arm:
            self.switches += 1
        self._last_arm = arm
        self.counts[arm] += shots
        self.failures[arm] += failures
        self.shots += shots
        duration = self._config.old_duration if arm < len(self._config.contexts) else self._config.new_duration
        self.execution_duration += shots * duration

    def _record(self, event):
        self.batches += 1
        encoded = json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self._digest.update(encoded.encode("ascii") + b"\n")
        if self.events is not None:
            self.events.append(event)

    def sample_arm(self, arm, shots):
        self._check_arm(arm)
        self._preflight(shots)
        self._check_transcript(arm, shots)
        start = self.counts[arm]
        failures = int(self._draw(arm, shots).sum())
        self._charge(arm, shots, failures)
        self._record({"kind": "arm", "arm": arm, "start": start,
                      "shots": shots, "failures": failures})
        return failures

    def sample_pairs(self, pairs, context=None):
        positive_integer(pairs, "pairs")
        self._preflight(2 * pairs)
        number = len(self._config.contexts)
        if context is not None and (isinstance(context, bool) or not isinstance(context, int)
                                    or not 0 <= context < number):
            raise ValueError("invalid public context")
        context_state = copy.deepcopy(self._context_stream.bit_generator.state)
        if context is None:
            draws = self._context_stream.random(pairs)
            cumulative = np.cumsum(self._config.weights)
            # Accepted normalized weights can accumulate just below one.
            # Every draw lies in [0, 1), so close the last interval exactly.
            cumulative[-1] = 1.0
            context_counts = np.bincount(np.searchsorted(cumulative, draws,
                                                        side="right"), minlength=number)
        else:
            context_counts = np.zeros(number, dtype=np.int64)
            context_counts[context] = pairs
        try:
            for c, count in enumerate(context_counts):
                if count:
                    self._check_transcript(c, int(count))
                    self._check_transcript(number + c, int(count))
        except TranscriptExhausted:
            self._context_stream.bit_generator.state = context_state
            raise
        joint = np.zeros(4, dtype=np.int64)
        pieces = []
        for c, count in enumerate(context_counts):
            count = int(count)
            if count == 0:
                continue
            old_start, new_start = self.counts[c], self.counts[number + c]
            old = self._draw(c, count)
            new = self._draw(number + c, count)
            local_joint = np.bincount(2 * old + new, minlength=4)
            joint += local_joint
            old_failures, new_failures = int(old.sum()), int(new.sum())
            self._charge(c, count, old_failures)
            self._charge(number + c, count, new_failures)
            pieces.append({"context": c, "pairs": count, "old_start": old_start,
                           "new_start": new_start, "joint": local_joint.tolist()})
        event = {"kind": "pairs", "mode": "iid" if context is None else "fixed",
                 "pairs": pairs, "pieces": pieces, "joint": joint.tolist()}
        self._record(event)
        return {"joint": joint.tolist(), "context_counts": context_counts.tolist(),
                "pieces": pieces}

    @property
    def remaining(self):
        return self._config.shot_budget - self.shots

    def cost_snapshot(self):
        return {"shots": self.shots, "counts": self.counts[:], "failures": self.failures[:],
                "readout_objects": self.shots * self._config.measured_qubits,
                "switches": self.switches, "batches": self.batches,
                "execution_duration": self.execution_duration,
                "modeled_device_duration": self.execution_duration + self.switches * self._config.switch_duration,
                "event_sha256": self._digest.hexdigest()}

    def audit_truth(self):
        number = len(self._config.contexts)
        old = math.fsum(w * self._probabilities[c] for c, w in enumerate(self._config.weights))
        new = math.fsum(w * self._probabilities[number + c] for c, w in enumerate(self._config.weights))
        return {"probabilities": self._probabilities, "old_failure": old, "new_failure": new,
                "true_delta": new - old, "seed": str(self._seed),
                "phase": self._config.phase,
                "noise_pairing": "independent old/new noise; common public iid context only"}
