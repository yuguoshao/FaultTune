"""The README example, with the annotations a strictly typed project would add."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import faulttune as ft

# 1. Describe the processor: 16 qubits on a 4x4 grid.
graph = ft.ProcessorGraph.square_lattice(4, 4)
goal = graph.goal(exclude=[ft.ElementKind.PARASITIC_EDGE])


# 2. Write the error model in Python.
class GateFrequencyModel(ft.ErrorModel):
    def __init__(self, device_data: dict[str, float]) -> None:
        self.data = device_data  # relaxation spectra, crosstalk strengths, ...

    def options(self, element: ft.Element) -> NDArray[np.float64]:
        """The k candidate values (frequencies in GHz) for one element."""
        return np.linspace(5.0, 7.0, 100)

    def prepare(self, step: ft.Step) -> None:
        """Optional: collect calibration data for this step."""

    def error(self, step: ft.Step, candidates: NDArray[np.float64]) -> NDArray[np.float64]:
        """Estimated error of many candidate assignments at once."""
        return np.zeros(candidates.shape[0])


# 3. Run the Snake.
snake = ft.Snake(
    graph,
    goal,
    parameter_distance=1,  # d_P
    constraint_distance=2,  # d_R
    traversal_distance=2,  # d_T
    heuristic="breadth_first",
    optimizer="coordinate_descent",
)
device_data = {"t1": 20.0}
model = GateFrequencyModel(device_data)
params = snake.calibrate(model)

# 4. Use, save and re-calibrate the results.
print(params.to_dict())  # {element: value}
params.save("calibration.json")

qubit = graph.node(1, 3)  # this qubit has drifted
snake.recalibrate(params, qubit, model)
