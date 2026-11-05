# Local validation under hidden crosstalk

This standalone experiment validates a **fixed** proposed calibration update.
It separates the observation API from the hidden scoring model and asks how many
observations are needed to accept or reject that update. It runs on Python 3.8+
with the standard library. Neither the FaultTune extension nor NumPy is required.

The [research note](../../docs/LOCAL_VALIDATION_RESEARCH.md) contains the assumptions,
proofs, measured results, prior work, and the next research question. These are
specializations of classical sequential tests, not a claimed novel optimizer.

## Model

There are N potential remote monitors. The old baseline probability p0 and local
gain a are known. In the nominal model, the proposed setting leaves every monitor
at p0, or changes exactly one to p1. Its additive surrogate change is
`-a + (p1-p0) * has_hidden_effect`. One query returns one independent Bernoulli
observation from a selected monitor and costs one **object sample**.

This is a scalar observation model, not a quantum circuit or hardware simulation.
One object sample is not automatically one hardware circuit execution. The
candidate and device do not drift during this initial experiment.

## Files

| File | Purpose |
| --- | --- |
| `probe.py` | Environment, seven nominal policies, budget accounting, raw and aggregate results |
| `audit_math.py` | Independent likelihood-walk probability calculation and theoretical bounds |
| `stress_probe.py` | Off-model effect size and a classical least-favorable-boundary correction |
| `test_probe.py` | Behavioral tests and independent mathematical checks |
| `results/` | Saved summaries, source manifest, and optional ignored per-trial logs |

## Run

Run these commands from the repository root. `-B` avoids bytecode files.

```powershell
python -B -m unittest discover -s experiments/local_validation -p test_probe.py -v

python -B experiments/local_validation/audit_math.py --output experiments/local_validation/results/analytic.json

python -B experiments/local_validation/probe.py --sizes 8 32 128 --trials 200 --source-sha 4f4c319+uncommitted-local-validation --output experiments/local_validation/results/nominal.json --raw-output experiments/local_validation/results/nominal_trials.jsonl

python -B experiments/local_validation/probe.py --sizes 32 --trials 500 --budget 300 --policies fixed_exact_scan sequential_scan sequential_round_robin local_only --source-sha 4f4c319+uncommitted-local-validation --output experiments/local_validation/results/budget_300.json --raw-output experiments/local_validation/results/budget_300_trials.jsonl

python -B experiments/local_validation/stress_probe.py --trials 1000 --output experiments/local_validation/results/misspecification.json
```

These reproduce the recorded run, overwriting the named outputs. For another
checkout, set `--source-sha` to its base revision and retain the script hash.
Random seeds and the aggregate statistics reproduce; generation timestamps may
differ. The manifest describes the saved snapshot and is not automatically updated
by these commands. Change `--output` paths to retain multiple runs.

## Strategies and guarantees

- `local_only` is an intentionally invalid zero-measurement control.
- `full_fixed` / `fixed_scan` use a Hoeffding midpoint threshold. The scan can stop
  at the first detected anomaly; the full version always purchases the full batch.
- `fixed_exact` / `fixed_exact_scan` calibrate a common integer threshold using
  the binomial distribution. Numerical design uses doubles and a conservative
  probability margin; "exact" refers to the binomial model, not interval arithmetic.
- `sequential_scan` / `sequential_round_robin` use likelihood thresholds
  `N/delta` and `delta`, freeze cleared monitors, and stop at the first alert.
- `harm_boundary_scan`, in the separate stress script, uses `p0 + local_gain`
  as the least harmful alternative. Its false-acceptance guarantee covers a
  single unknown effect that offsets the local gain. Its beneficial-power
  guarantee covers only the all-p0 null. Multiple weak effects are outside it.

Policies receive public configuration and `measure(index)`, not hidden state or
the final score. This is an interface contract, not a sandbox for untrusted code.
Each monitor has an independently seeded random stream; its kth uniform is shared
across policies, regardless of measurement order.

`accept`, `reject`, and `abstain` are reported separately. Abstention is never
counted as success, and conditional error rates among only decided runs are not
the guaranteed quantities. The complete-test KL lower bound requires high null
acceptance power; it cannot be applied unchanged to high-abstention runs.

## Interpretation

The numerical DP provides an independent check of decision probabilities. Its
unresolved mass bounds probability truncation, up to floating-point error; it is
not an error bound for the predicted mean sample count. Monte Carlo intervals
are marginal Wilson intervals and do not constitute simultaneous certification.

The main result of the stress experiment is the cost of an explicit modeling
assumption. A test designed for p1=0.4 wrongly accepts roughly 23% of harmful
updates when the true hidden probability is 0.3. Testing against p0+a=0.25
controls that error under the stated single-effect family, while increasing
the cost of confirming safe updates. These comparisons are not evidence of
hardware improvement or international algorithmic leadership.
