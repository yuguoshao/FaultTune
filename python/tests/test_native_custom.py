"""Custom heuristics, optimizers, seed policies and algorithm subgraphs written in Python."""

from __future__ import annotations

import gc
import itertools
import traceback
import weakref
from typing import Any

import numpy as np
import pytest
from test_native_common import TargetModel, assert_same_database, lattice

import faulttune as ft


def centres(db: ft.ParameterDatabase) -> list[str]:
    return [s.centre.label for s in db.steps()]


# ---------------------------------------------------------------------------- heuristics


def test_builtin_heuristics_visit_in_different_orders() -> None:
    graph, goal = lattice(4, 4)
    orders = {}
    for name in ("breadth_first", "depth_first", "random"):
        db = ft.Snake(graph, goal, heuristic=name, seed=2).calibrate(TargetModel())
        assert db.calibrated == goal
        orders[name] = centres(db)
    assert orders["breadth_first"][0] == orders["depth_first"][0] == "q0_0"
    assert len({tuple(o) for o in orders.values()}) >= 2


class Recorder:
    """A heuristic that wraps another one and records what it was asked and what it said."""

    def __init__(self, order: Any, frontier: str = "fifo", priority: Any = None) -> None:
        self.order = order
        self.frontier = frontier
        self.events: list[tuple[str, list[str], list[str]]] = []
        if priority is not None:
            self.priority = priority  # type: ignore[method-assign]

    def __call__(self, ctx: ft.TraversalContext, options: list[ft.Element]) -> list[ft.Element]:
        ordered = self.order(options)
        self.events.append(
            (ctx.current.label, [o.label for o in options], [o.label for o in ordered])
        )
        return ordered


def replay_traversal(recorder: Recorder, kind: str, priority: Any = None) -> list[str]:
    """The centres a frontier of ``kind`` must visit, given what the heuristic returned.

    This is an independent model of the engine's traversal: one thread of nodes, so the
    frontier starts with the seed (the first centre).
    """
    first = recorder.events[0][0]
    answers = {current: ordered for current, _, ordered in recorder.events}
    visited: list[str] = []
    frontier: list[tuple[float, int, str]] = [(0.0, 0, first)]
    counter = 1
    while frontier:
        if kind == "fifo":
            _, _, g = frontier.pop(0)
        elif kind == "lifo":
            _, _, g = frontier.pop()
        else:
            best = min(range(len(frontier)), key=lambda i: frontier[i][:2])
            _, _, g = frontier.pop(best)
        if g in visited:
            continue
        visited.append(g)
        ordered = answers.get(g, [])
        pushed = list(reversed(ordered)) if kind == "lifo" else ordered
        for o in pushed:
            score = priority(o) if kind == "priority" else 0.0
            frontier.append((score, counter, o))
            counter += 1
    return visited


def test_callable_heuristic_decides_the_order() -> None:
    graph, goal = lattice(4, 4)
    heuristic = Recorder(lambda options: sorted(options, key=lambda e: -e.index))
    snake = ft.Snake(graph, goal, heuristic=heuristic)
    db = snake.calibrate(TargetModel())
    assert db.calibrated == goal and heuristic.events
    assert snake.config["heuristic"] is heuristic
    # Called once per step, with the step's centre as the current element.
    assert [c for c, _, _ in heuristic.events] == centres(db)
    # The first step's options (the three nodes around the corner) are visited highest id first.
    assert heuristic.events[0] == ("q0_0", ["q0_1", "q1_0", "q1_1"], ["q1_1", "q1_0", "q0_1"])
    assert centres(db)[:3] == ["q0_0", "q1_1", "q1_0"]
    assert replay_traversal(heuristic, "fifo") == centres(db)


@pytest.mark.parametrize("kind", ["fifo", "lifo", "priority"])
def test_frontier_kinds_follow_the_heuristic(kind: str) -> None:
    graph, goal = lattice(5, 5)

    def priority(label: str) -> float:
        return -float(graph[label].index)

    recorder = Recorder(
        lambda options: sorted(options, key=lambda e: e.index % 7),
        frontier=kind,
        priority=(lambda ctx, element: priority(element.label)) if kind == "priority" else None,
    )
    db = ft.Snake(graph, goal, heuristic=recorder).calibrate(TargetModel())
    assert db.calibrated == goal
    expected = replay_traversal(recorder, kind, priority)
    assert centres(db) == expected
    assert len(expected) == 25


def test_frontier_kinds_differ() -> None:
    graph, goal = lattice(5, 5)
    runs = {}
    for kind in ("fifo", "lifo", "priority"):
        recorder = Recorder(
            lambda options: sorted(options, key=lambda e: -e.index),
            frontier=kind,
            priority=(lambda ctx, element: -float(element.index)) if kind == "priority" else None,
        )
        runs[kind] = centres(ft.Snake(graph, goal, heuristic=recorder).calibrate(TargetModel()))
    assert len({tuple(run) for run in runs.values()}) == 3
    assert runs["priority"][1] == "q1_1"


def test_heuristic_context() -> None:
    graph, goal = lattice(4, 4)
    stash: list[ft.TraversalContext] = []
    counts: list[int] = []
    history_lengths: list[int] = []
    seen_centres: list[str] = []

    def probe(ctx: ft.TraversalContext, options: list[ft.Element]) -> list[ft.Element]:
        stash.append(ctx)
        counts.append(ctx.calibrated_count)
        records = ctx.history()
        history_lengths.append(len(records))
        assert all(isinstance(r, ft.StepRecord) for r in records)
        assert records[-1].centre == ctx.current
        assert ctx.is_calibrated(ctx.current)
        assert all(not ctx.is_calibrated(o) for o in options)
        assert len(ctx.calibrated) == ctx.calibrated_count
        assert ctx.current in ctx.calibrated
        seen_centres.append(ctx.current.label)
        with pytest.raises(ValueError, match="different graph"):
            ctx.is_calibrated(ft.ProcessorGraph.square_lattice(4, 4).node(0, 0))
        with pytest.raises(TypeError):
            ctx.is_calibrated("q0_0")  # type: ignore[arg-type]
        return list(options)

    db = ft.Snake(graph, goal, heuristic=probe).calibrate(TargetModel())
    assert history_lengths == list(range(1, len(history_lengths) + 1))
    assert counts == sorted(counts) and counts[0] >= 1
    assert seen_centres == centres(db)[: len(seen_centres)]
    # The context is only valid during its call.
    with pytest.raises(RuntimeError, match="only be used during"):
        stash[0].current
    with pytest.raises(RuntimeError, match="only be used during"):
        stash[0].history()
    with pytest.raises(RuntimeError, match="only be used during"):
        stash[0].calibrated_count


def test_heuristic_may_drop_elements() -> None:
    graph, goal = lattice(4, 4)
    db = ft.Snake(graph, goal, heuristic=lambda ctx, options: options[:1]).calibrate(
        TargetModel()
    )
    assert db.calibrated == goal, "the completion loop picks up what the traversal skipped"
    assert ft.Snake(graph, goal, heuristic=lambda ctx, options: []).calibrate(TargetModel())


@pytest.mark.parametrize(
    ("heuristic", "error", "cause"),
    [
        (lambda ctx, options: 1 / 0, ft.ModelError, ZeroDivisionError),
        (lambda ctx, options: 5, ft.ModelError, TypeError),
        (lambda ctx, options: [5], ft.ModelError, TypeError),
        (lambda ctx, options: ["q0_0"], ft.ModelError, TypeError),
        (
            lambda ctx, options: [ft.ProcessorGraph.square_lattice(4, 4).node(0, 0)],
            ft.ModelError,
            ValueError,
        ),
        (lambda ctx, options: [ctx.current], ft.ModelError, ValueError),
    ],
)
def test_failing_heuristics_raise_model_error(heuristic: Any, error: type, cause: type) -> None:
    graph, goal = lattice(4, 4)
    db = ft.ParameterDatabase(graph)
    with pytest.raises(error) as info:
        ft.Snake(graph, goal, heuristic=heuristic).calibrate(TargetModel(), database=db)
    assert isinstance(info.value.__cause__, cause)
    assert "heuristic" in str(info.value)
    assert isinstance(info.value.element, ft.Element)
    assert 1 <= db.step_count < 16, "the run was stopped, committed steps are kept"


def test_heuristic_protocol_is_checked() -> None:
    graph, goal = lattice(3, 3)

    class BadFrontier:
        frontier = "sideways"

        def __call__(self, ctx: Any, options: Any) -> Any:
            return options

    class NoPriority(BadFrontier):
        frontier = "priority"

    with pytest.raises(ValueError, match="frontier"):
        ft.Snake(graph, goal, heuristic=BadFrontier())
    with pytest.raises(ValueError, match="priority"):
        ft.Snake(graph, goal, heuristic=NoPriority())
    with pytest.raises(TypeError, match="callable"):
        ft.Snake(graph, goal, heuristic=object())


def test_failing_priority() -> None:
    class Bad:
        frontier = "priority"

        def __call__(self, ctx: ft.TraversalContext, options: list[ft.Element]) -> Any:
            return options

        def priority(self, ctx: ft.TraversalContext, element: ft.Element) -> float:
            return "high"  # type: ignore[return-value]

    graph, goal = lattice(3, 3)
    with pytest.raises(ft.ModelError, match="priority") as info:
        ft.Snake(graph, goal, heuristic=Bad()).calibrate(TargetModel())
    assert isinstance(info.value.__cause__, TypeError)


# ---------------------------------------------------------------------------- optimizers


def brute_force(space: ft.SearchSpace, objective: ft.Objective, rng: Any) -> tuple[Any, float]:
    assert isinstance(space, ft.SearchSpace) and isinstance(objective, ft.Objective)
    assert isinstance(rng, np.random.Generator)
    grid = np.array(list(itertools.product(*[range(k) for k in space.sizes])), dtype=np.int64)
    errors = objective(grid)
    assert errors.dtype == np.float64 and errors.shape == (len(grid),)
    best = int(np.argmin(errors))
    return grid[best], float(errors[best])


def test_callable_optimizer_matches_exhaustive_search() -> None:
    graph, goal = lattice(3, 3)
    model = dict(k=8)
    reference = ft.Snake(graph, goal, optimizer="exhaustive").calibrate(TargetModel(**model))
    custom = ft.Snake(graph, goal, optimizer=brute_force).calibrate(TargetModel(**model))
    assert custom.calibrated == goal
    assert custom.to_dict() == reference.to_dict()
    for ours, theirs in zip(custom.steps(), reference.steps(), strict=True):
        assert ours.error == pytest.approx(theirs.error, abs=1e-12)
        assert ours.evaluations == theirs.evaluations > 0


def test_optimizer_receives_space_objective_and_rng() -> None:
    graph, goal = lattice(3, 3)
    seen: list[Any] = []

    def spy(space: ft.SearchSpace, objective: ft.Objective, rng: Any) -> tuple[Any, float]:
        seen.append((space.sizes, space.budget, space.width, space.size, rng.integers(0, 10**9)))
        assert objective.sizes == space.sizes
        # A single row (1-D) works, and the answer is a float64 array of shape (1,).
        single = objective(np.zeros(space.width, dtype=np.int64))
        assert single.shape == (1,)
        # Duplicates inside a batch and across calls go through the engine's de-duplication.
        batch = np.array([[0] * space.width] * 3, dtype=np.int64)
        again = objective(batch)
        assert again.shape == (3,) and (again == single[0]).all()
        return np.zeros(space.width, dtype=np.int64), float(single[0])

    ft.Snake(graph, goal, optimizer=spy, evaluation_budget=777).calibrate(TargetModel(k=5))
    assert seen
    for sizes, budget, width, size, _ in seen:
        assert sizes == [5] * width and budget == 777 and size == 5**width
    draws = [d for *_, d in seen]
    assert len(set(draws)) > 1, "the generators differ from step to step"

    # The same engine seed gives the same generators; another seed gives other ones.
    def draws_for(seed: int) -> list[int]:
        out: list[int] = []

        def opt(space: ft.SearchSpace, objective: ft.Objective, rng: Any) -> tuple[Any, float]:
            out.append(int(rng.integers(0, 10**9)))
            return np.zeros(space.width, dtype=np.int64), 0.0

        ft.Snake(graph, goal, optimizer=opt, seed=seed).calibrate(TargetModel(k=5))
        return out

    assert draws_for(1) == draws_for(1) != draws_for(2)


def test_optimizer_evaluations_are_counted() -> None:
    graph, goal = lattice(3, 3)

    def two_rows(space: ft.SearchSpace, objective: ft.Objective, rng: Any) -> tuple[Any, float]:
        rows = np.array([[0] * space.width, [1] * space.width], dtype=np.int64)
        errors = objective(rows)
        best = int(np.argmin(errors))
        return rows[best], float(errors[best])

    db = ft.Snake(graph, goal, optimizer=two_rows).calibrate(TargetModel(k=5))
    assert all(s.evaluations == 2 for s in db.steps())


@pytest.mark.parametrize(
    ("optimizer", "error", "text"),
    [
        (lambda s, o, r: 5, ft.InvalidConfigError, "pair"),
        (lambda s, o, r: ([0], 0.0, 1), ft.InvalidConfigError, "pair"),
        (lambda s, o, r: ([0] * 9, 0.0), ft.InvalidConfigError, "shape"),
        (lambda s, o, r: ([-1] * s.width, 0.0), ft.InvalidConfigError, "index -1"),
        (lambda s, o, r: ([99] * s.width, 0.0), ft.InvalidConfigError, "index 99"),
        (lambda s, o, r: ([0] * s.width, "low"), ft.InvalidConfigError, "numeric"),
        (lambda s, o, r: (["a"] * s.width, 0.0), ft.InvalidConfigError, "invalid result"),
        (lambda s, o, r: ([[0] * s.width], 0.0), ft.InvalidConfigError, "shape"),
        (lambda s, o, r: ([0.7] * s.width, 0.0), ft.InvalidConfigError, "integers"),
    ],
)
def test_invalid_optimizer_results(optimizer: Any, error: type, text: str) -> None:
    graph, goal = lattice(3, 3)
    with pytest.raises(error, match=text):
        ft.Snake(graph, goal, optimizer=optimizer).calibrate(TargetModel(k=5))


def test_optimizer_exceptions() -> None:
    graph, goal = lattice(3, 3)

    def failing(space: ft.SearchSpace, objective: ft.Objective, rng: Any) -> Any:
        raise NotImplementedError("not yet")

    with pytest.raises(ft.ModelError, match="optimizer") as info:
        ft.Snake(graph, goal, optimizer=failing).calibrate(TargetModel())
    assert isinstance(info.value.__cause__, NotImplementedError)
    assert traceback.extract_tb(info.value.__cause__.__traceback__)[-1].name == "failing"

    def bad_index(space: ft.SearchSpace, objective: ft.Objective, rng: Any) -> Any:
        objective([[space.sizes[0]] + [0] * (space.width - 1)])

    with pytest.raises(ft.ModelError) as info:
        ft.Snake(graph, goal, optimizer=bad_index).calibrate(TargetModel(k=5))
    assert isinstance(info.value.__cause__, ValueError) and "out of range" in str(info.value)

    def bad_shape(space: ft.SearchSpace, objective: ft.Objective, rng: Any) -> Any:
        objective(np.zeros((2, space.width + 1), dtype=np.int64))

    with pytest.raises(ft.ModelError) as info:
        ft.Snake(graph, goal, optimizer=bad_shape).calibrate(TargetModel(k=5))
    assert "shape (n, " in str(info.value.__cause__)


def test_objective_failures_keep_the_engines_own_error() -> None:
    """A model error seen through the objective is the engine's error, not the optimizer's."""
    graph, goal = lattice(3, 3)

    class Failing(TargetModel):
        def error(self, step: ft.Step, candidates: Any) -> Any:
            raise ValueError("model failed")

    def swallowing(space: ft.SearchSpace, objective: ft.Objective, rng: Any) -> Any:
        try:
            objective(np.zeros((1, space.width), dtype=np.int64))
        except Exception:  # the optimizer hides the exception ...
            return np.zeros(space.width, dtype=np.int64), 0.0
        raise AssertionError("unreachable")

    # ... but the failure is still the run's error.
    with pytest.raises(ft.ModelError) as info:
        ft.Snake(graph, goal, optimizer=swallowing).calibrate(Failing())
    assert isinstance(info.value.__cause__, ValueError) and "model failed" in str(info.value.__cause__)

    # The skip policy sees the model error too.
    snake = ft.Snake(graph, goal, optimizer=swallowing, on_model_error="skip")
    db = snake.calibrate(Failing())
    assert len(db) == 0 and snake.last_report is not None
    assert len(snake.last_report.skipped) == len(goal)

    class Nan(TargetModel):
        def error(self, step: ft.Step, candidates: Any) -> Any:
            return np.full(len(candidates), np.nan)

    with pytest.raises(ft.NonFiniteError):
        ft.Snake(graph, goal, optimizer=swallowing).calibrate(Nan())


def test_objective_cannot_outlive_the_call() -> None:
    graph, goal = lattice(3, 3)
    stash: list[Any] = []

    def keep(space: ft.SearchSpace, objective: ft.Objective, rng: Any) -> Any:
        stash.append(objective)
        return np.zeros(space.width, dtype=np.int64), 0.0

    ft.Snake(graph, goal, optimizer=keep).calibrate(TargetModel(k=3))
    with pytest.raises(RuntimeError, match="only be used during"):
        stash[0](np.zeros((1, len(stash[0].sizes)), dtype=np.int64))


def test_objective_cannot_be_reentered_from_the_model() -> None:
    graph, goal = lattice(3, 3)
    holder: list[Any] = []

    class Reentrant(TargetModel):
        def error(self, step: ft.Step, candidates: Any) -> Any:
            objective = holder[0]
            objective(np.zeros((1, len(objective.sizes)), dtype=np.int64))
            return super().error(step, candidates)

    def opt(space: ft.SearchSpace, objective: ft.Objective, rng: Any) -> Any:
        holder[:] = [objective]
        errors = objective(np.zeros((1, space.width), dtype=np.int64))
        return np.zeros(space.width, dtype=np.int64), float(errors[0])

    with pytest.raises(ft.ModelError) as info:
        ft.Snake(graph, goal, optimizer=opt).calibrate(Reentrant(k=3))
    assert isinstance(info.value.__cause__, RuntimeError)


def test_optimizer_objects() -> None:
    graph, goal = lattice(3, 3)
    reference = ft.Snake(graph, goal, optimizer="coordinate_descent").calibrate(TargetModel(k=9))
    explicit = ft.Snake(graph, goal, optimizer=ft.CoordinateDescent()).calibrate(TargetModel(k=9))
    assert_same_database(reference, explicit)
    d = ft.CoordinateDescent(random_starts=2, max_sweeps=1, max_start_attempts=5)
    assert (d.random_starts, d.max_sweeps, d.max_start_attempts) == (2, 1, 5)
    assert repr(d) == "CoordinateDescent(random_starts=2, max_sweeps=1, max_start_attempts=5)"
    assert ft.Snake(graph, goal, optimizer=d).calibrate(TargetModel(k=9)).calibrated == goal
    r = ft.RandomSearch(samples=50, batch_size=20)
    assert (r.samples, r.batch_size) == (50, 20) and "RandomSearch(samples=50" in repr(r)
    db = ft.Snake(graph, goal, optimizer=r).calibrate(TargetModel(k=9))
    assert all(s.evaluations <= 50 for s in db.steps())
    assert ft.Snake(graph, goal, optimizer="random_search").calibrate(TargetModel(k=9))
    e = ft.Exhaustive(batch_size=10, max_size=10_000)
    assert (e.batch_size, e.max_size) == (10, 10_000) and "Exhaustive(batch_size=10" in repr(e)
    assert ft.Snake(graph, goal, optimizer=e).calibrate(TargetModel(k=9)).calibrated == goal
    assert ft.Exhaustive().max_size == 1_000_000 and ft.RandomSearch().samples == 1000
    for bad in (
        lambda: ft.CoordinateDescent(random_starts=0),
        lambda: ft.CoordinateDescent(max_sweeps=0),
        lambda: ft.CoordinateDescent(max_start_attempts=-1),
        lambda: ft.RandomSearch(samples=0),
        lambda: ft.RandomSearch(batch_size=-3),
        lambda: ft.Exhaustive(batch_size=0),
        lambda: ft.Exhaustive(max_size=0),
    ):
        with pytest.raises(ValueError):
            bad()
    with pytest.raises(TypeError):
        ft.CoordinateDescent(random_starts="two")  # type: ignore[arg-type]


def test_exhaustive_search_finds_the_optimum_that_coordinate_descent_misses() -> None:
    """A needle in a haystack: only the combination (2, 2) is good."""
    graph = ft.ProcessorGraph.square_lattice(1, 2)
    goal = graph.element_set(graph.nodes())

    class Needle(ft.ErrorModel):
        def options(self, element: ft.Element) -> Any:
            return [0.0, 1.0, 2.0]

        def error(self, step: ft.Step, candidates: Any) -> Any:
            assert candidates.shape[1] == 2
            both = (candidates[:, 0] == 2) & (candidates[:, 1] == 2)
            return np.where(both, 0.0, 1.0 + candidates.sum(axis=1))

    exhaustive = ft.Snake(graph, goal, parameter_distance=2, optimizer="exhaustive")
    db = exhaustive.calibrate(Needle())
    assert db.step_count == 1 and db.steps()[0].error == 0.0
    assert set(db.to_dict().values()) == {2.0}
    assert db.steps()[0].evaluations == 9


# ---------------------------------------------------------------------------- seed policies


def test_fixed_seed_policy() -> None:
    graph, goal = lattice(4, 4)
    seed = graph.node(2, 2)
    db = ft.Snake(graph, goal, seed_policy=[seed]).calibrate(TargetModel())
    assert centres(db)[0] == "q2_2"
    lowest = ft.Snake(graph, goal).calibrate(TargetModel())
    assert centres(lowest)[0] == "q0_0"
    # The first listed element that is in a thread wins; unlisted threads fall back.
    db = ft.Snake(graph, goal, seed_policy=[graph.edges()[0], seed]).calibrate(TargetModel())
    assert centres(db)[0] == "q2_2"
    assert ft.Snake(graph, goal, seed_policy=graph.element_set([seed])).config["seed_policy"]


def test_random_seed_policy() -> None:
    graph, goal = lattice(4, 4)
    firsts = set()
    for seed in range(8):
        a = ft.Snake(graph, goal, seed_policy="random", seed=seed).calibrate(TargetModel())
        b = ft.Snake(graph, goal, seed_policy="random", seed=seed).calibrate(TargetModel())
        assert_same_database(a, b)
        firsts.add(centres(a)[0])
    assert len(firsts) > 1


def test_callable_seed_policy() -> None:
    graph, goal = lattice(4, 4)
    asked: list[list[ft.Element]] = []

    def last_of_thread(thread: list[ft.Element]) -> ft.Element:
        asked.append(thread)
        assert isinstance(thread, list) and all(isinstance(e, ft.Element) for e in thread)
        assert thread == sorted(thread)
        return thread[-1]

    db = ft.Snake(graph, goal, seed_policy=last_of_thread).calibrate(TargetModel())
    assert asked and asked[0] == graph.nodes()
    assert centres(db)[0] == "q3_3"
    assert db.calibrated == goal


@pytest.mark.parametrize(
    ("policy", "cause"),
    [
        (lambda thread: 1 / 0, ZeroDivisionError),
        (lambda thread: "q0_0", TypeError),
        (lambda thread: None, TypeError),
        (lambda thread: ft.ProcessorGraph.square_lattice(4, 4).node(0, 0), ValueError),
        (lambda thread: [e for e in thread][0].endpoints, TypeError),
    ],
)
def test_failing_seed_policies(policy: Any, cause: type) -> None:
    graph, goal = lattice(4, 4)
    with pytest.raises(ft.ModelError, match="seed policy") as info:
        ft.Snake(graph, goal, seed_policy=policy).calibrate(TargetModel())
    assert isinstance(info.value.__cause__, cause)


def test_seed_policy_outside_the_thread_is_refused() -> None:
    graph, goal = lattice(4, 4)
    edge = graph.edges()[0]
    with pytest.raises(ft.ModelError, match="not in the thread") as info:
        ft.Snake(graph, goal, seed_policy=lambda thread: edge).calibrate(TargetModel())
    assert isinstance(info.value.__cause__, ValueError)


# ---------------------------------------------------------------------------- algorithms


def test_algorithm_subgraphs() -> None:
    graph = ft.ProcessorGraph.square_lattice(4, 4)
    everything = ft.AlgorithmSubgraphs.all_active(graph)
    assert len(everything) == 1 and everything.subgraphs == [graph.all_elements()]
    assert everything.graph is graph and "AlgorithmSubgraphs(1" in repr(everything)
    xeb = ft.AlgorithmSubgraphs.xeb_square_lattice(graph, 4, 4)
    assert len(xeb) == 5 and all(isinstance(s, ft.ElementSet) for s in xeb.subgraphs)
    assert xeb.subgraphs[0] == graph.element_set(graph.nodes())
    custom = ft.AlgorithmSubgraphs(graph, [graph.nodes(), graph.element_set(graph.edges())])
    assert len(custom) == 2
    with pytest.raises(ft.InvalidConfigError, match="not a 3x3 square lattice"):
        ft.AlgorithmSubgraphs.xeb_square_lattice(graph, 3, 3)
    with pytest.raises(ft.InvalidConfigError, match="at least one"):
        ft.AlgorithmSubgraphs(graph, [])
    with pytest.raises(ft.InvalidConfigError, match="at most 64"):
        ft.AlgorithmSubgraphs(graph, [graph.nodes()] * 65)
    with pytest.raises(TypeError, match="subgraphs"):
        ft.AlgorithmSubgraphs(graph, graph.all_elements())
    with pytest.raises(TypeError):
        ft.AlgorithmSubgraphs(graph, 5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="graph"):
        ft.AlgorithmSubgraphs("graph", [])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="different graph"):
        ft.AlgorithmSubgraphs(graph, [ft.ProcessorGraph.square_lattice(4, 4).nodes()])


def test_algorithm_subgraphs_change_the_constraints() -> None:
    graph, goal = lattice(4, 4)
    totals = {}
    for name, algorithm in [
        ("all", None),
        ("explicit_all", ft.AlgorithmSubgraphs.all_active(graph)),
        ("xeb", ft.AlgorithmSubgraphs.xeb_square_lattice(graph, 4, 4)),
    ]:
        snake = ft.Snake(graph, goal, algorithm=algorithm)
        db = snake.calibrate(TargetModel(repel=0.1))
        assert db.calibrated == goal
        totals[name] = sum(len(s.constraints) for s in db.steps())
    assert totals["all"] == totals["explicit_all"]
    assert totals["xeb"] < totals["all"], "fewer elements are active together under XEB"


def test_goal_element_that_is_never_active_is_refused() -> None:
    graph, goal = lattice(3, 3)
    nodes_only = ft.AlgorithmSubgraphs(graph, [graph.nodes()])
    snake = ft.Snake(graph, goal, algorithm=nodes_only)
    with pytest.raises(ft.InvalidConfigError, match="not active in any algorithm subgraph"):
        snake.calibrate(TargetModel())
    assert snake.last_report is None
    ok = ft.Snake(graph, graph.element_set(graph.nodes()), algorithm=nodes_only)
    assert ok.calibrate(TargetModel()).step_count > 0


def test_objective_refuses_non_integer_indices() -> None:
    graph, goal = lattice(3, 3)
    seen: list[type] = []

    def opt(space: ft.SearchSpace, objective: ft.Objective, rng: Any) -> tuple[Any, float]:
        for bad in (np.zeros((3, space.width)) + 0.7, [["0"] * space.width]):
            try:
                objective(bad)
            except TypeError as err:
                assert "integers" in str(err)
                seen.append(TypeError)
        return np.zeros(space.width, dtype=np.int64), 0.0

    ft.Snake(graph, goal, optimizer=opt).calibrate(TargetModel(k=5))
    assert seen and set(seen) == {TypeError}
    assert len(seen) % 2 == 0


class Keeper:
    """A callback that keeps the Snake it belongs to (a common pattern for stateful ones)."""

    snake: Any = None

    def heuristic(self, ctx: ft.TraversalContext, options: list[ft.Element]) -> list[ft.Element]:
        return list(options)

    def optimizer(self, space: ft.SearchSpace, objective: ft.Objective, rng: Any) -> tuple[Any, float]:
        zeros = np.zeros(space.width, dtype=np.int64)
        return zeros, float(objective(zeros)[0])

    def seed_policy(self, thread_elements: list[ft.Element]) -> ft.Element:
        return thread_elements[0]


@pytest.mark.parametrize("kind", ["heuristic", "optimizer", "seed_policy"])
def test_a_callback_that_refers_to_its_snake_does_not_leak(kind: str) -> None:
    graph, goal = lattice(3, 3)
    keeper = Keeper()
    reference = weakref.ref(keeper)
    snake = ft.Snake(graph, goal, **{kind: getattr(keeper, kind)})
    keeper.snake = snake
    gc.collect()
    # Alive and reachable from the outside: the collector must leave it alone.
    assert snake.calibrate(TargetModel(k=4)).calibrated == goal
    del keeper, snake
    gc.collect()
    assert reference() is None, "the Snake <-> callback cycle was not collected"
