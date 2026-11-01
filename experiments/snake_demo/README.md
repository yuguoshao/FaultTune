# Snake exploration scripts

These scripts reuse the toy `GateFrequencyModel` from
[`examples/gate_frequency.py`](../../examples/gate_frequency.py). They provide a small
entry point for inspecting Snake's decisions and comparing parameter scopes. No hardware
is accessed, and their scores are synthetic objectives, not measured gate or logical error rates.

## Contents

| Path | Purpose |
| --- | --- |
| `explore_snake.py` | Print per-step parameters and fixed constraints; compare `d_P` settings. |
| `comparison_checks/local_global_counterexample.py` | Isolate a local/global objective mismatch in the current demo model. |
| `local_results/` | Optional, ignored directory for generated outputs. Created when saving results. |

## Run

Build the Python package using the repository's [installation instructions](../../README.md#installation).
The commands below assume an activated environment with `faulttune` and NumPy installed,
and are run from the repository root. On the existing local checkout, `.venv/bin/python`
can be used instead of `python`.

```bash
# Inspect all four steps on a 2x2 lattice.
python experiments/snake_demo/explore_snake.py --size 2 --scopes 1 --trace

# Compare scopes on the same 4x4 synthetic device, saving fresh output.
python experiments/snake_demo/explore_snake.py --size 4 --scopes 0 1 2 \
  --output experiments/snake_demo/local_results/scope_4x4.json

# Change only the optimizer's random seed, keeping the device fixed.
python experiments/snake_demo/explore_snake.py --seed 53 --optimizer-seed 1

# Reproduce the isolated objective-consistency counterexample.
python experiments/snake_demo/comparison_checks/local_global_counterexample.py \
  --output experiments/snake_demo/local_results/local_global_counterexample.json

# Run the original examples and their existing tests.
python examples/quickstart.py
python examples/gate_frequency.py
python -m pytest python/tests/test_examples.py -q
```

Both scripts resolve repository imports relative to their own files, so they can also be
invoked by absolute path from another working directory. An explicit `--output` is interpreted
relative to the current working directory; missing parent directories are created. Without
`--output`, results are only printed. Generated logs and JSON results are not versioned.

## Interpreting the results

`--seed` generates the synthetic device; `--optimizer-seed` controls the optimizer's randomness.
Keep the device seed fixed when comparing strategies. `full_model_mean_score` re-evaluates the
complete final assignment; local step scores should not be summed as a substitute. Evaluation
counts and elapsed time are reported separately. These scope comparisons do not enforce equal
total evaluation budgets, and one device instance does not establish a general performance claim.

The counterexample holds all other frequencies fixed and changes one node. The current demo
model's local plan omits detuning terms belonging to fixed edges that still depend on that node.
Its local score decreases while the complete assignment score increases. The script isolates
this effect by changing model arrays in memory; it does not modify the source model or establish
a defect in the general Snake algorithm. It describes the current implementation and will need
review if that model is corrected.

