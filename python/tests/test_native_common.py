"""Helpers shared by the ``test_native_*`` modules (this module has no tests of its own)."""

from __future__ import annotations

import os
import signal
import threading
import time
from typing import Any

import numpy as np

import faulttune as ft


class TargetModel(ft.ErrorModel):
    """A cheap model with a known optimum and a little coupling between neighbours.

    Every element wants the value ``target(element)``. With ``repel > 0`` a parameter is also
    pushed away from the values of the step's constraints, which couples the steps. It records
    how it was called so that tests can check the batch contract.
    """

    def __init__(self, k: int = 12, repel: float = 0.0, delay: float = 0.0) -> None:
        self.k = k
        self.repel = repel
        self.delay = delay
        self.error_calls = 0
        self.prepare_calls = 0
        self.option_calls: list[ft.Element] = []
        self.shapes: list[tuple[int, ...]] = []
        self.dtypes: set[str] = set()
        self.threads: set[int] = set()
        self.lock = threading.Lock()

    @staticmethod
    def target(element: ft.Element) -> float:
        return ((element.index * 37) % 101) / 100.0

    def options(self, element: ft.Element) -> Any:
        self.option_calls.append(element)
        return np.linspace(0.0, 1.0, self.k)

    def error(self, step: ft.Step, candidates: Any) -> Any:
        with self.lock:
            self.error_calls += 1
            self.shapes.append(candidates.shape)
            self.dtypes.add(str(candidates.dtype))
            self.threads.add(threading.get_ident())
        if self.delay:
            time.sleep(self.delay)
        targets = np.array([self.target(p) for p in step.parameters])
        errors = ((candidates - targets) ** 2).sum(axis=1)
        if self.repel:
            for value in step.constraints.values():
                errors = errors + self.repel * np.exp(-((candidates - value) ** 2) / 0.02).sum(
                    axis=1
                )
        return errors


class GatedModel(TargetModel):
    """A model whose ``error`` waits at the steps from ``from_index`` on until ``gate`` is set.

    Tests that stop a run from ``on_step`` (which is delivered asynchronously) open the gate
    from the callback: the engine cannot get past the gated step before the callback has
    run, whatever the load of the machine, so the run is certain to still be going.
    """

    def __init__(self, gate: threading.Event, from_index: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.gate = gate
        self.from_index = from_index

    def error(self, step: ft.Step, candidates: Any) -> Any:
        if step.index >= self.from_index and not self.gate.wait(60):
            raise AssertionError("the gate was never opened")
        return super().error(step, candidates)


class InterruptingModel(TargetModel):
    """A model that, at one step, raises SIGINT in this process and then waits a little.

    The wait gives the event pump (which checks for signals every ~100 ms) time to see the
    interrupt and stop the run, so the run cannot finish before the interrupt is handled.
    """

    def __init__(self, at_step: int, action: Any = None, wait: float = 1.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.at_step = at_step
        self.action = action
        self.wait = wait
        self.fired = False

    def error(self, step: ft.Step, candidates: Any) -> Any:
        if step.index == self.at_step and not self.fired:
            self.fired = True
            if self.action is not None:
                self.action()
            else:
                os.kill(os.getpid(), signal.SIGINT)
            time.sleep(self.wait)
        return super().error(step, candidates)


class PreparingModel(TargetModel):
    """A model whose ``prepare`` hands data to ``error`` through ``step.data``."""

    def prepare(self, step: ft.Step) -> Any:
        self.prepare_calls += 1
        assert step.data is None, "step.data must be None while prepare runs"
        return {"offset": 0.25, "centre": step.centre.label}

    def error(self, step: ft.Step, candidates: Any) -> Any:
        assert step.data is not None and step.data["centre"] == step.centre.label
        return super().error(step, candidates) + step.data["offset"]


def lattice(rows: int = 4, cols: int = 4) -> tuple[ft.ProcessorGraph, ft.ElementSet]:
    """A square lattice and the usual goal (everything except the parasitic edges)."""
    graph = ft.ProcessorGraph.square_lattice(rows, cols)
    return graph, graph.goal(exclude=[ft.ElementKind.PARASITIC_EDGE])


def assert_same_database(a: ft.ParameterDatabase, b: ft.ParameterDatabase) -> None:
    """Same values and same history, ignoring timings."""
    assert a.same_outcome(b)
    assert a.to_dict() == b.to_dict()
