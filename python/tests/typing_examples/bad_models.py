"""Error models, heuristics and optimizers that do not fit. See bad_arguments.py."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

import faulttune as ft

graph = ft.ProcessorGraph.square_lattice(3, 3)
goal = graph.goal()
snake = ft.Snake(graph, goal)


class OnlyOptions:
    def options(self, element: ft.Element) -> NDArray[np.float64]:
        return np.arange(3.0)


class OnlyError:
    def error(self, step: ft.Step, candidates: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.zeros(len(candidates))


class WrongStep:
    def options(self, element: ft.Element) -> NDArray[np.float64]:
        return np.arange(3.0)

    def error(self, step: str, candidates: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.zeros(len(candidates))


class WrongOptions:
    def options(self, element: str) -> NDArray[np.float64]:
        return np.arange(3.0)

    def error(self, step: ft.Step, candidates: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.zeros(len(candidates))


class TooFewArguments:
    def options(self) -> NDArray[np.float64]:
        return np.arange(3.0)

    def error(self, step: ft.Step, candidates: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.zeros(len(candidates))


class NoError(ft.ErrorModel):
    def options(self, element: ft.Element) -> NDArray[np.float64]:
        return np.arange(3.0)


class BadOverride(ft.ErrorModel):
    def options(self, element: ft.Element) -> NDArray[np.float64]:
        return np.arange(3.0)

    def error(self, step: str, candidates: NDArray[np.float64]) -> NDArray[np.float64]:  # E: override
        return np.zeros(len(candidates))


snake.calibrate(OnlyOptions())  # E: arg-type: "OnlyOptions"
snake.calibrate(OnlyError())  # E: arg-type: "OnlyError"
snake.calibrate(WrongStep())  # E: arg-type: "WrongStep"
snake.calibrate(WrongOptions())  # E: arg-type: "WrongOptions"
snake.calibrate(TooFewArguments())  # E: arg-type: "TooFewArguments"
snake.calibrate(NoError())  # E: abstract: Cannot instantiate abstract class "NoError" with abstract attribute "error"
snake.calibrate(object())  # E: arg-type: "object"
snake.calibrate(lambda step: 0)  # E: arg-type
ft.ParameterDatabase(graph).replay_journal("j", OnlyOptions())  # E: arg-type: "OnlyOptions"


# Heuristics.
def heuristic_ok(ctx: ft.TraversalContext, options: list[ft.Element]) -> list[ft.Element]:
    return options


def heuristic_one_arg(options: list[ft.Element]) -> list[ft.Element]:
    return options


def heuristic_wrong_context(ctx: ft.Snake, options: list[ft.Element]) -> list[ft.Element]:
    return options


def heuristic_returns_ints(ctx: ft.TraversalContext, options: list[ft.Element]) -> list[int]:
    return [e.index for e in options]


def heuristic_returns_none(ctx: ft.TraversalContext, options: list[ft.Element]) -> None:
    return None


ft.Snake(graph, goal, heuristic=heuristic_ok)
ft.Snake(graph, goal, heuristic=heuristic_one_arg)  # E: arg-type
ft.Snake(graph, goal, heuristic=heuristic_wrong_context)  # E: arg-type
ft.Snake(graph, goal, heuristic=heuristic_returns_ints)  # E: arg-type
ft.Snake(graph, goal, heuristic=heuristic_returns_none)  # E: arg-type
ft.Snake(graph, goal, heuristic=object())  # E: arg-type: "object"


# Optimizers.
def optimizer_ok(
    space: ft.SearchSpace, objective: ft.Objective, rng: np.random.Generator
) -> tuple[list[int], float]:
    return [0] * space.width, 0.0


def optimizer_no_rng(space: ft.SearchSpace, objective: ft.Objective) -> tuple[list[int], float]:
    return [0] * space.width, 0.0


def optimizer_returns_indices_only(
    space: ft.SearchSpace, objective: ft.Objective, rng: np.random.Generator
) -> list[int]:
    return [0] * space.width


def optimizer_float_indices(
    space: ft.SearchSpace, objective: ft.Objective, rng: np.random.Generator
) -> tuple[list[float], float]:
    return [0.5] * space.width, 0.0


def optimizer_swapped(
    objective: ft.Objective, space: ft.SearchSpace, rng: np.random.Generator
) -> tuple[list[int], float]:
    return [], 0.0


ft.Snake(graph, goal, optimizer=optimizer_ok)
ft.Snake(graph, goal, optimizer=optimizer_no_rng)  # E: arg-type
ft.Snake(graph, goal, optimizer=optimizer_returns_indices_only)  # E: arg-type
ft.Snake(graph, goal, optimizer=optimizer_float_indices)  # E: arg-type
ft.Snake(graph, goal, optimizer=optimizer_swapped)  # E: arg-type
ft.Snake(graph, goal, optimizer=ft.Retry(1))  # E: arg-type: "Retry"


# Seed policies.
def seed_ok(thread: list[ft.Element]) -> ft.Element:
    return thread[0]


def seed_returns_int(thread: list[ft.Element]) -> int:
    return 0


def seed_wrong_arg(thread: ft.ElementSet) -> ft.Element:
    return next(iter(thread))


ft.Snake(graph, goal, seed_policy=seed_ok)
ft.Snake(graph, goal, seed_policy=seed_returns_int)  # E: arg-type
ft.Snake(graph, goal, seed_policy=seed_wrong_arg)  # E: arg-type
ft.Snake(graph, goal, seed_policy=[1])  # E: list-item
ft.Snake(graph, goal, seed_policy=graph.node(0, 0))  # E: arg-type: "Element"


def objective_misuse(objective: ft.Objective, rng: np.random.Generator) -> None:
    objective(np.zeros((2, 2)))  # E: arg-type
    objective(["a"])  # E: arg-type
    objective()  # E: call-arg
    objective(indices=[0])  # E: call-arg
    rng.random()


def iter_misuse(options: Iterable[ft.Element]) -> None:
    for o in options:
        o.label = "x"  # E: misc: read-only
