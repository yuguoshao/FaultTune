# Aggregate validation with graph-constrained measurements

This second experiment validates the **sum** of multiple remote effects from a
fixed candidate update. It extends the [first-round experiment](README.md) with
legal simultaneous measurements. See the [research note](../../docs/AGGREGATE_VALIDATION_RESEARCH.md)
for proofs, the negative result for the current adaptive policy, prior work,
and the next falsifiable research question.

Python 3.8+ and the standard library suffice. This is an independent Bernoulli
observation model, not a quantum circuit simulation or a FaultTune engine test.

## Model and files

The known baseline is p0 and the known local improvement is a. Unknown monitor
probabilities p_j define `Delta = sum(p_j - p0) - a`. A legal action is a nonempty
independent set in a public conflict graph. It returns independent Bernoulli
bits, costs one round, and costs one object observation per returned bit.
Each monitor's probability is stationary and independent of the selected action.
Switches exclude the first action and have no assumed conversion to device time.

| File | Purpose |
| --- | --- |
| [group_oracle.py](group_oracle.py) | Public graph/partition API, hidden streams, atomic costs and budgets |
| [aggregate_probe.py](aggregate_probe.py) | Pooled sweep, separate group certificates, adaptive heuristic, invalid legacy control |
| [batched_sweep.py](batched_sweep.py) | Same pooled checkpoints with fewer group switches |
| [aggregate_math.py](aggregate_math.py) | Exact rational finite-event counterexamples and numerical null information rates |
| [test_aggregate.py](test_aggregate.py) | Oracle contracts, independent heterogeneous-tail checks, policy invariants |
| [test_batched_sweep.py](test_batched_sweep.py) | Prefix equivalence, switch accounting and budget interruption |
| [test_aggregate_math.py](test_aggregate_math.py) | Independent probability and optimization checks |
| [verify_aggregate.py](verify_aggregate.py) | Recompute saved summaries, check pairing, hashes, links and optionally tests |
| [aggregate_manifest.json](results/aggregate_manifest.json) | Commands, revisions, hashes and verification results |

The original `probe.py`, `stress_probe.py` and `audit_math.py` supply reused
utilities and the first-round comparator. They remain unchanged.

## Recorded experiment

All runs use N=16, p0=0.1, a=0.3, delta=0.05 and seed 20261104. Both budgets
are 262144. The kth random uniform for each arm is shared across policies,
graphs and cases. It does not depend on scheduling order.

- `null`: all p_j=0.1, Delta=-0.3.
- `four_cluster`: arms 0,2,4,6 have p_j=0.2, Delta=+0.1.
- `four_spread`: arms 0,1,14,15 have p_j=0.2, Delta=+0.1.
- `many_weak`: all p_j=0.1375, Delta=+0.3.

Cluster/spread describe path color classes; the same positions are retained
on the star graph. The CLI also offers a beneficial-weak case, which is **not**
part of the recorded main experiments. Pilot files are not final evidence.

Run from the repository root:

```powershell
python -B experiments/local_validation/aggregate_probe.py --trials 100 --output experiments/local_validation/results/aggregate_path.json --raw-output experiments/local_validation/results/aggregate_path_trials.jsonl

python -B experiments/local_validation/aggregate_probe.py --graphs star --policies sweep_kl group_kl_balanced group_kl_adaptive --trials 100 --output experiments/local_validation/results/aggregate_star.json --raw-output experiments/local_validation/results/aggregate_star_trials.jsonl

python -B experiments/local_validation/aggregate_probe.py --graphs empty path clique --policies sweep_kl --cases null four_cluster --trials 30 --output experiments/local_validation/results/aggregate_topology.json --raw-output experiments/local_validation/results/aggregate_topology_trials.jsonl

python -B experiments/local_validation/aggregate_math.py --output experiments/local_validation/results/aggregate_analytic.json
```

These are 1600 + 1200 + 180 = **2980** policy trials. Batched scheduling is
verified through pathwise reasoning and tests; it adds no Monte Carlo trials.
`elapsed_seconds` measures Python runtime only. Seeds reproduce decisions and
sample costs with this implementation; timestamps and measured runtimes vary.
Raw JSONL is retained locally but excluded by `.gitignore`.

To check the existing snapshot and run all behavioral/math tests:

```powershell
python -B experiments/local_validation/verify_aggregate.py --tests
```

To run only tests, independently of the saved result files:

```powershell
python -B -m unittest discover -s experiments/local_validation -p 'test_*.py' -v
```

The audit checks every saved summary against its raw records, cost accounting,
the experiment source hash, 60 cross-topology triplets, 400 path/star sweep pairs,
local document links, and all hashes from the unchanged first-round manifest.
After **intentionally replacing this snapshot** with the exact commands above,
refresh its new manifest using:

```powershell
python -B experiments/local_validation/verify_aggregate.py --write-manifest --tests
```

That command runs the tests and records their actual count. It does not modify
the first-round manifest. Fresh runs have different timestamp/runtime hashes;
hash equality certifies a saved snapshot, not cross-machine timing reproduction.
Use different output paths to keep both old and new runs.

## Guarantee and interpretation

The three valid policies inspect only geometric checkpoints and allocate
summable error budgets. Accept requires a strictly negative total upper bound;
reject requires a strictly positive total lower bound. Budget exhaustion yields
`abstain`, with partial observations charged but only a completed-checkpoint
certificate used. No finite stopping promise is made at Delta=0.

For heterogeneous independent bits, a Chernoff bound on their average remains
valid. Pooling all arms is valid when their prefix lengths agree. Naively
pooling asynchronous groups changes the estimand and is not valid here.
The adaptive rule chooses a group by interval width divided by the next block's
round count. It is a heuristic, not an optimal experimental design.

`legacy_guard` is intentionally outside its single-effect guarantee when
multiple effects accumulate. Its observed 37%--81% false-acceptance rates are
counterexamples to extending that guarantee, not comparisons between equally
correct procedures. The new policies recorded zero errors in these instances,
but three runs abstained; finite Monte Carlo observations do not prove an error
rate or a simultaneous confidence statement across cells.

The separate-group adaptive method loses to pooled sweep in both rounds and
objects on every recorded path/star case. On the star it saves rounds relative
to separate-group balanced sampling while increasing object cost. Fewer switches
can also be obtained by batching pooled sweep, without changing its complete
checkpoint data under this model.

The analytic artifact distinguishes exact integer/Fraction counterexamples from
double-precision information-rate calculations. Null lower bounds require
terminating uniformly correct procedures over all nonzero-sign instances;
do not apply them directly to non-null cases or budgeted high-abstention runs.
No claim of algorithmic novelty, gate-fidelity improvement or hardware speedup
is made from this scalar simulation.
