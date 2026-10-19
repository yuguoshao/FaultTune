"""Custom heuristics, optimizers and seed policies, and the built-in ones."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

import faulttune as ft

graph = ft.ProcessorGraph.square_lattice(4, 4)
goal = graph.goal(exclude=ft.ElementKind.PARASITIC_EDGE)


# A heuristic is a callable (context, options) -> the options in visiting order.
def nearest_first(ctx: ft.TraversalContext, options: list[ft.Element]) -> list[ft.Element]:
    assert ctx.graph is graph or True
    done = ctx.calibrated_count
    return sorted(options, key=lambda e: (e.index + done) % 7)


class Weighted:
    """A heuristic with a priority frontier: a callable plus ``frontier`` and ``priority``."""

    frontier: Literal["priority"] = "priority"

    def __call__(self, ctx: ft.TraversalContext, options: list[ft.Element]) -> Iterable[ft.Element]:
        return options

    def priority(self, ctx: ft.TraversalContext, element: ft.Element) -> float:
        return -float(element.index) if ctx.is_calibrated(ctx.current) else 0.0


class Stack:
    """A depth-first style heuristic: a plain callable with a ``frontier`` attribute."""

    frontier = "lifo"

    def __call__(self, ctx: ft.TraversalContext, options: Sequence[ft.Element]) -> list[ft.Element]:
        history: list[ft.StepRecord] = ctx.history()
        return list(reversed(options)) if history else list(options)


# An optimizer is a callable (space, objective, rng) -> (indices, error).
def random_optimizer(
    space: ft.SearchSpace, objective: ft.Objective, rng: np.random.Generator
) -> tuple[NDArray[np.int64], float]:
    n = min(space.budget, 32)
    batch = np.stack([rng.integers(0, k, size=n) for k in space.sizes], axis=1)
    errors = objective(batch)
    best = int(np.argmin(errors))
    return batch[best], float(errors[best])


def first_optimizer(
    space: ft.SearchSpace, objective: ft.Objective, rng: np.random.Generator
) -> tuple[list[int], np.float64]:
    zeros = [0] * space.width
    return zeros, objective(zeros)[0]


def pick_seed(thread_elements: list[ft.Element]) -> ft.Element:
    return max(thread_elements)


snakes = [
    ft.Snake(graph, goal, heuristic="depth_first", optimizer="exhaustive"),
    ft.Snake(graph, goal, heuristic="random", optimizer="random_search", seed_policy="random"),
    ft.Snake(graph, goal, heuristic=nearest_first, optimizer=random_optimizer),
    ft.Snake(graph, goal, heuristic=Weighted(), optimizer=first_optimizer),
    ft.Snake(graph, goal, heuristic=Stack(), seed_policy=pick_seed),
    ft.Snake(graph, goal, heuristic=lambda ctx, options: options[::-1]),
    ft.Snake(graph, goal, optimizer=ft.CoordinateDescent(random_starts=8, max_sweeps=5)),
    ft.Snake(graph, goal, optimizer=ft.Exhaustive(batch_size=100, max_size=10_000)),
    ft.Snake(graph, goal, optimizer=ft.RandomSearch(samples=200)),
    ft.Snake(graph, goal, seed_policy=[graph.node(0, 0), graph.node(3, 3)]),
    ft.Snake(graph, goal, seed_policy=graph.element_set([graph.node(0, 0)])),
    ft.Snake(graph, goal, seed_policy=lambda thread: thread[0]),
    ft.Snake(graph, goal, algorithm=ft.AlgorithmSubgraphs.all_active(graph)),
    ft.Snake(graph, goal, algorithm=ft.AlgorithmSubgraphs.xeb_square_lattice(graph, 4, 4)),
    ft.Snake(graph, goal, algorithm=ft.AlgorithmSubgraphs(graph, [graph.nodes(), graph.edges()])),
    ft.Snake(graph, goal, algorithm=None, heuristic=None, optimizer=None, seed_policy=None),
]

# The error policies and the cache modes.
policies = [
    ft.Snake(graph, goal, on_model_error="abort"),
    ft.Snake(graph, goal, on_model_error="skip"),
    ft.Snake(graph, goal, on_model_error=("retry", 3)),
    ft.Snake(graph, goal, on_model_error=ft.Retry(2), cache="across_steps"),
    ft.Snake(graph, goal, cache="step", threads=4, seed=7, evaluation_budget=10_000),
    ft.Snake(graph, goal, cache=50_000, expiry_radius=2),
]

# The objective takes a batch of indices or a single one.
def probe(space: ft.SearchSpace, objective: ft.Objective, rng: np.random.Generator) -> tuple[list[int], float]:
    one = objective([0] * space.width)
    many = objective([[0] * space.width, [1] * space.width])
    typed = objective(np.zeros((2, space.width), dtype=np.int32))
    return [0] * space.width, float(one[0] + many[0] + typed[0])


class Trivial:
    """A duck-typed model for the run at the end of this example."""

    def options(self, element: ft.Element) -> NDArray[np.float64]:
        return np.arange(4.0)

    def error(self, step: ft.Step, candidates: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(candidates.sum(axis=1))


for custom in [*snakes, *policies, ft.Snake(graph, goal, optimizer=probe)]:
    assert custom.calibrate(Trivial()).calibrated == goal
