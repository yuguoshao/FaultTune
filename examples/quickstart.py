"""Quickstart: the Python walkthrough of the README as a runnable script.

It calibrates the gate frequencies of a 4x4 lattice of qubits with a small NumPy error model and
shows the things most users do:

1. describe the processor (a graph) and the calibration goal,
2. write an error model (here a toy one; yours holds your device data),
3. configure and run the Snake,
4. read the learned parameters, then save them to JSON and load them back,
5. re-calibrate one drifted qubit, which touches only its neighbourhood.

Run it with `python examples/quickstart.py` (after `maturin develop --release`, or
`pip install faulttune`). For a larger, fuller tour see `gate_frequency.py` and the notebook
`gate_frequency.ipynb`. The output is the same on every run.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import faulttune as ft

# 1. Describe the processor: 16 qubits on a 4x4 grid. Nearest-neighbour couplers are engineered
#    edges; diagonals are parasitic edges that only carry crosstalk constraints.
graph = ft.ProcessorGraph.square_lattice(4, 4)
print(
    f"Processor: {len(graph.nodes())} nodes, {len(graph.edges())} edges "
    f"({len(graph.all_elements())} elements in total)"
)

# Calibrate every node and every engineered edge. Parasitic edges have no parameters.
goal = graph.goal(exclude=[ft.ElementKind.PARASITIC_EDGE])
print(f"Calibration goal: {len(goal)} parameters")


# 2. The error model. A real model would wrap measured data (relaxation spectra, crosstalk
#    strengths); this toy one draws random defects for every element from a seed.
class GateFrequencyModel(ft.ErrorModel):
    """Frequencies in GHz. Each element has three relaxation defects to stay away from, and
    gates that run at similar frequencies close to each other (the step's constraints and its
    other parameters) disturb one another through crosstalk."""

    CROSSTALK_WIDTH = 0.04  # GHz

    def __init__(self, n_elements: int, seed: int = 2024) -> None:
        rng = np.random.default_rng(seed)
        self.centre = rng.uniform(5.0, 7.0, (n_elements, 3))  # where the defects sit (GHz)
        self.width = rng.uniform(0.005, 0.020, (n_elements, 3))  # half widths (GHz)
        self.strength = rng.uniform(0.5, 2.0, (n_elements, 3))

    def options(self, element: ft.Element) -> NDArray[np.float64]:
        """The k candidate values (frequencies in GHz) for one element."""
        return np.linspace(5.0, 7.0, 100)

    def error(self, step: ft.Step, candidates: NDArray[np.float64]) -> NDArray[np.float64]:
        """Estimated error of many candidate assignments at once.

        candidates: array of shape (n_candidates, len(step.parameters)).
        step.constraints maps each constraining element to its fixed value.
        Returns an array of shape (n_candidates,).
        """
        rows = [p.index for p in step.parameters]
        # Relaxation: a Lorentzian bump around every defect of every parameter.
        d = candidates[:, :, None] - self.centre[rows]  # (n_candidates, n_parameters, 3)
        w2 = self.width[rows] ** 2
        total = (self.strength[rows] * w2 / (d * d + w2)).sum(axis=(1, 2))
        # Crosstalk with the fixed neighbours ...
        x2 = self.CROSSTALK_WIDTH**2
        for value in step.constraints.values():
            total += (x2 / ((candidates - value) ** 2 + x2)).sum(axis=1)
        # ... and among the parameters of the step (each pair once).
        n = candidates.shape[1]
        diff = candidates[:, :, None] - candidates[:, None, :]
        pairs = np.triu(np.ones((n, n)), k=1)
        total += (pairs * x2 / (diff * diff + x2)).sum(axis=(1, 2))
        return total

    def drift(self, element: ft.Element, frequency: float) -> None:
        """A new defect appears at `frequency` on `element`: the device has changed."""
        self.centre[element.index, 0] = frequency
        self.width[element.index, 0] = 0.015
        self.strength[element.index, 0] = 3.0


model = GateFrequencyModel(len(graph.all_elements()))

# 3. Run the Snake with the distances of the paper's XEB example.
snake = ft.Snake(
    graph,
    goal,
    parameter_distance=1,  # d_P: optimize a qubit together with its couplers
    constraint_distance=2,  # d_R: see the calibrated gates up to two hops away
    traversal_distance=2,  # d_T: the Snake may jump two hops per step
    heuristic="breadth_first",
    optimizer="coordinate_descent",
)
params = snake.calibrate(model)  # a ParameterDatabase
report = snake.last_report
assert report is not None
print(f"\nCalibrated {len(params)} parameters:")
print(f"  passes            {report.passes}")
print(f"  steps             {report.steps}")
print(f"  evaluations       {report.evaluations}")
print("  worst steps (largest error):")
for step in report.worst_steps[:3]:
    print(
        f"    centre {step.centre.label:<10} {len(step.parameters)} parameters, "
        f"{len(step.constraints)} constraints, error {step.error:.4f}"
    )

# 4. Read the parameters. `params` maps elements to values and keeps the ordered history of the
#    steps that set them.
print("\nQubit frequencies (GHz), laid out like the lattice:")
for row in range(4):
    print("  " + "  ".join(f"{params[graph.node(row, col)]:6.3f}" for col in range(4)))
print("Coupler frequencies (GHz), first five:")
couplers = [e for e in params.to_dict() if e.is_edge]
for coupler in couplers[:5]:
    print(f"  {coupler.label:<10} {params[coupler]:6.3f}")

# Save to JSON and load it back; the file records which graph it belongs to.
with tempfile.TemporaryDirectory(prefix="faulttune-quickstart-") as directory:
    path = Path(directory) / "calibration.json"
    params.save(path)
    loaded = ft.ParameterDatabase.load(path, graph)
identical = loaded.same_outcome(params)
print(f"\nSaved and loaded {len(loaded)} parameters; identical to the original: {identical}")
assert identical

# 5. Qubit q1_3 has drifted: a new defect sits right at its frequency. Forget its neighbourhood
#    and re-learn it. Everything outside the neighbourhood keeps its value and acts as a
#    constraint.
qubit = graph.node(1, 3)
model.drift(qubit, params[qubit])
before = params.to_dict()
report = snake.recalibrate(params, qubit, model)
after = params.to_dict()
changed = [e for e in graph.all_elements() if before.get(e) != after.get(e)]
print(
    f"\nRe-calibrated {qubit.label}: {report.steps} steps, {report.evaluations} evaluations, "
    f"{len(changed)} parameters changed:"
)
for element in changed:
    print(f"  {element.label:<10} {before[element]:6.3f} -> {after[element]:6.3f}")

# Self-checks: the whole goal is calibrated, and only the neighbourhood of the qubit changed.
neighbourhood = {qubit} | {e for e in goal if e.is_edge and qubit in (e.endpoints or ())}
assert goal <= params.calibrated
assert qubit in changed and set(changed) <= neighbourhood
