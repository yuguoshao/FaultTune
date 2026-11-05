"""Exact count-law simulation of paid circuit observations, without shot arrays.

This backend has the SAME distribution of returned counts as independent
old/new circuit executions. It replaces explicit Bernoulli arrays with
binomial margins and an independent hypergeometric overlap draw. It does not
reuse a hidden uniform to correlate the two branches or reduce their noise.

The same seed and the same action trace are reproducible. Changing batch
boundaries generally changes the realized counts: NumPy's binomial and
hypergeometric draws consume a variable number of random words. Therefore
this backend does NOT preserve a kth-shot common-random-number path. Use
independent seeds per method in benchmark comparisons. The original
CircuitObservationOracle remains available for explicit small-case traces
and for experiments requiring batch-invariant per-arm uniform streams.

The simulator can aggregate millions of executions quickly; the inherited
ledger still charges every old/new execution and every configuration switch.
"""

import numpy as np

from observation import CircuitObservationOracle, positive_integer, stable_seed


class CountObservationOracle(CircuitObservationOracle):
    """Distribution-exact aggregate subclass with the parent's public API.

    Conditional on n pairs in a context, draw independently
    A~Bin(n,p_old), B~Bin(n,p_new). Given these margins, independent uniform
    failure subsets have overlap H~Hypergeom(A,n-A,B). Thus the four cells
    [n-A-B+H, B-H, A-H, H] have exactly the multinomial law of n independent
    old/new Bernoulli pairs. A multinomial public-context count draw completes
    the iid target-context law. NumPy PRNG sampling is the numerical primitive;
    "exact" refers to this probability law, not exact arithmetic or hardware.

    Fixed transcripts are deliberately rejected: they specify shot identities
    and must be processed by the original uniform/transcript oracle.
    """

    def __init__(self, config, probabilities, seed, deadline_epoch=None,
                 transcripts=None, keep_events=False):
        if transcripts is not None:
            raise ValueError("CountObservationOracle does not accept transcripts; use CircuitObservationOracle")
        super().__init__(config, probabilities, seed, deadline_epoch=deadline_epoch,
                         transcripts=None, keep_events=keep_events)
        self._pairing_streams = [
            np.random.Generator(np.random.PCG64(stable_seed(seed, "aggregate-pairing", context)))
            for context in range(len(config.contexts))
        ]

    def sample_arm(self, arm, shots):
        self._check_arm(arm)
        self._preflight(shots)
        start = self.counts[arm]
        failures = int(self._streams[arm].binomial(shots, self._probabilities[arm]))
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
        if context is None:
            # NumPy treats the final category as the remaining probability,
            # as does the original oracle's last cumulative interval.
            context_counts = self._context_stream.multinomial(pairs, self._config.weights)
        else:
            context_counts = np.zeros(number, dtype=np.int64)
            context_counts[context] = pairs
        joint = [0, 0, 0, 0]
        pieces = []
        pending_charges = []
        for c, count in enumerate(context_counts):
            count = int(count)
            if count == 0:
                continue
            old_arm, new_arm = c, number + c
            old_start, new_start = self.counts[old_arm], self.counts[new_arm]
            old_failures = int(self._streams[old_arm].binomial(count, self._probabilities[old_arm]))
            new_failures = int(self._streams[new_arm].binomial(count, self._probabilities[new_arm]))
            both = int(self._pairing_streams[c].hypergeometric(old_failures, count - old_failures, new_failures))
            local_joint = [count - old_failures - new_failures + both,
                           new_failures - both, old_failures - both, both]
            joint = [left + right for left, right in zip(joint, local_joint)]
            pending_charges.extend(((old_arm, count, old_failures), (new_arm, count, new_failures)))
            pieces.append({"context": c, "pairs": count, "old_start": old_start,
                           "new_start": new_start, "joint": local_joint})
        # Charge only after the entire count draw is ready. Budget/deadline and
        # action validation happen before any random generator is consumed.
        for arm, count, failures in pending_charges:
            self._charge(arm, count, failures)
        event = {"kind": "pairs", "mode": "iid" if context is None else "fixed",
                 "pairs": pairs, "pieces": pieces, "joint": joint}
        self._record(event)
        return {"joint": joint, "context_counts": context_counts.tolist(), "pieces": pieces}

    def audit_truth(self):
        result = super().audit_truth()
        result.update({
            "sampling_backend": "binomial margins plus independent hypergeometric pairing",
            "batch_partition_invariant": False,
            "comparison_seed_policy": "independent per-method seeds; no shared kth-shot CRN claim",
            "aggregate_law": "same count law as iid public contexts and conditionally independent old/new executions",
        })
        return result
