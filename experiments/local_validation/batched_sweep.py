"""Reduce action switching without changing the pooled geometric-checkpoint test.

Python 3.8+, standard library. Each checkpoint takes equally many samples from
every arm. Within the next doubling block, finish one partition group before
moving to the next. Under stationary, action-independent per-arm streams, this
only changes the schedule: complete-checkpoint data, certificates, total object
observations and rounds match aggregate_probe.sweep_kl exactly.

For G > 1 groups and final per-arm depth m=2**k, the switch count is
G*(k+1)-1. For G=1 it is zero. These are action switches, not measured device
latencies. Incomplete blocks remain charged but do not update the certificate.
"""

from aggregate_probe import checkpoint_alpha, interval_decision, kl_interval
from probe import BudgetExhausted


def sweep_kl_batched(config, measure):
    """Apply the original pooled test after group-batched doubling blocks."""
    complete = successes = checkpoint = 0
    target = 1
    group_count = len(config.partition)
    latest = [-config.n * config.p0 - config.local_gain,
              config.n * (1.0 - config.p0) - config.local_gain]
    try:
        while True:
            extra_per_arm = target - complete
            for action in config.partition:
                for _ in range(extra_per_arm):
                    successes += sum(measure(action))
            # Only a complete, balanced checkpoint can be scored. If a
            # measurement above exhausts a budget, complete/latest stay old.
            complete = target
            lower, upper = kl_interval(successes, config.n * complete,
                                       checkpoint_alpha(config.delta, checkpoint))
            offset = config.n * config.p0 + config.local_gain
            latest = [config.n * lower - offset, config.n * upper - offset]
            decision = interval_decision(latest)
            if decision:
                return {"decision": decision, "interval": latest,
                        "checkpoint_counts": [complete] * group_count}
            checkpoint += 1
            target *= 2
    except BudgetExhausted:
        return {"decision": "abstain", "interval": latest,
                "checkpoint_counts": [complete] * group_count}
