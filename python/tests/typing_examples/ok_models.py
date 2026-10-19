"""Error models: ErrorModel subclasses and duck-typed objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, assert_type

import numpy as np
from numpy.typing import ArrayLike, NDArray

import faulttune as ft


@dataclass
class Spectrum:
    """Calibration data collected by ``prepare``."""

    t1: float
    shift: NDArray[np.float64]


class SpectrumModel(ft.ErrorModel):
    """A subclass with per-step data and a constraint-aware error."""

    def __init__(self, ghz: float = 6.0) -> None:
        self.ghz = ghz

    def options(self, element: ft.Element) -> ArrayLike:
        # Structured options: the indices 0..k, interpreted by the model itself.
        if element.is_node:
            return range(10)
        return [4.5, 5.5, 6.5]

    def prepare(self, step: ft.Step) -> Spectrum:
        assert_type(step.parameters, list[ft.Element])
        assert_type(step.centre, ft.Element)
        return Spectrum(t1=20.0, shift=np.zeros(len(step.parameters)))

    def error(self, step: ft.Step, candidates: NDArray[np.float64]) -> NDArray[np.float64]:
        data: Any = step.data  # whatever ``prepare`` returned
        penalty = 0.0
        for other, value in step.constraints.items():
            assert_type(other, ft.Element)
            assert_type(value, float)
            penalty += float(np.abs(candidates - value).min())
        collision = np.abs(candidates[:, 0] - self.ghz) < 0.01
        errors: NDArray[np.float64] = np.where(collision, np.inf, penalty + data.t1)
        return errors


class DuckModel:
    """Not an ErrorModel: ``options`` and ``error`` are all that is needed."""

    def options(self, element: ft.Element) -> NDArray[np.float64]:
        return np.arange(5, dtype=np.float64)

    def error(self, step: ft.Step, candidates: NDArray[np.float64]) -> list[float]:
        return [0.0] * len(candidates)

    def prepare(self, step: ft.Step) -> int:
        return step.index


def calibrate_both() -> ft.ParameterDatabase:
    graph = ft.ProcessorGraph.square_lattice(3, 3)
    snake = ft.Snake(graph, graph.goal(exclude=ft.ElementKind.PARASITIC_EDGE), threads=1)
    first = snake.calibrate(SpectrumModel())
    assert_type(first, ft.ParameterDatabase)
    return snake.calibrate(DuckModel(), database=first)


assert calibrate_both().step_count > 0
