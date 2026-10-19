"""End-to-end calibration through the native extension."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import pytest
from test_native_common import (
    GatedModel,
    PreparingModel,
    TargetModel,
    assert_same_database,
    lattice,
)

import faulttune as ft


def test_readme_example() -> None:
    """The example of the README, with a model that has a known optimum."""
    graph = ft.ProcessorGraph.square_lattice(4, 4)
    goal = graph.goal(exclude=[ft.ElementKind.PARASITIC_EDGE])

    class GateFrequencyModel(ft.ErrorModel):
        def __init__(self, device_data: Any) -> None:
            self.data = device_data

        def options(self, element: ft.Element) -> Any:
            return np.linspace(5.0, 7.0, 100)

        def prepare(self, step: ft.Step) -> Any:
            return None

        def error(self, step: ft.Step, candidates: Any) -> Any:
            wanted = 5.0 + 2.0 * self.data[[p.index for p in step.parameters]]
            return ((candidates - wanted) ** 2).sum(axis=1)

    device_data = np.random.default_rng(1).random(len(graph))
    snake = ft.Snake(
        graph,
        goal,
        parameter_distance=1,
        constraint_distance=2,
        traversal_distance=2,
        heuristic="breadth_first",
        optimizer="coordinate_descent",
    )
    params = snake.calibrate(GateFrequencyModel(device_data))

    assert params.calibrated == goal and len(params) == 40
    values = params.to_dict()
    assert set(values) == set(goal)
    options = np.linspace(5.0, 7.0, 100)
    for element, value in values.items():
        assert value in options
        # A separable model: every element lands on its own optimum (within one grid step).
        assert abs(value - (5.0 + 2.0 * device_data[element.index])) <= 1.0 / 99 + 1e-12
    assert params.to_dict(labels=True)["q0_0"] == values[graph.node(0, 0)]
    assert params[graph.node(0, 0)] == params["q0_0"] == values[graph.node(0, 0)]

    qubit = graph.node(1, 3)
    report = snake.recalibrate(params, qubit, GateFrequencyModel(device_data))
    assert isinstance(report, ft.Report) and report.steps >= 1
    assert params.calibrated == goal
    assert any(isinstance(h, ft.Invalidation) for h in params.history)


def test_batch_contract() -> None:
    graph, goal = lattice()
    model = TargetModel(k=10)
    snake = ft.Snake(graph, goal)
    snake.calibrate(model)

    assert model.error_calls > 0
    assert model.dtypes == {"float64"}
    assert all(len(shape) == 2 and shape[0] >= 1 and 1 <= shape[1] <= 5 for shape in model.shapes)
    assert max(shape[0] for shape in model.shapes) <= 10, "one sweep over k options at most"
    # One options() call per element that was a parameter of some step, none twice.
    assert len(model.option_calls) == len(set(model.option_calls)) <= len(goal)
    # Batching: a few calls per step, not one per candidate.
    report = snake.last_report
    assert report is not None
    assert model.error_calls < report.evaluations


def test_candidate_columns_follow_step_parameters() -> None:
    graph, goal = lattice(3, 3)
    seen: list[tuple[ft.Step, np.ndarray]] = []

    class Spy(TargetModel):
        def error(self, step: ft.Step, candidates: Any) -> Any:
            seen.append((step, candidates.copy()))
            return super().error(step, candidates)

    ft.Snake(graph, goal).calibrate(Spy(k=7))
    options = np.linspace(0.0, 1.0, 7)
    for step, candidates in seen:
        assert candidates.shape[1] == len(step.parameters)
        assert np.isin(candidates, options).all()
        assert step.centre in step.parameters or step.centre not in goal
        assert all(p in goal for p in step.parameters)
        values = step.constraints
        assert list(values) == sorted(values), "constraints ascend by element id"
        assert all(isinstance(v, float) for v in values.values())


def test_step_objects() -> None:
    graph, goal = lattice(3, 3)
    steps: dict[int, list[ft.Step]] = {}

    class Spy(PreparingModel):
        def prepare(self, step: ft.Step) -> Any:
            steps.setdefault(id(step), []).append(step)
            return super().prepare(step)

        def error(self, step: ft.Step, candidates: Any) -> Any:
            steps[id(step)].append(step)
            assert step.graph == graph
            assert step.index >= 0
            with pytest.raises(AttributeError):
                step.data = 1  # type: ignore[misc]
            return super().error(step, candidates)

    model = Spy(k=6)
    ft.Snake(graph, goal).calibrate(model)
    assert model.prepare_calls == sum(1 for _ in steps)
    for same in steps.values():
        assert len(same) >= 2 and all(s is same[0] for s in same), "one Step object per step"


def test_prepare_data_reaches_error() -> None:
    graph, goal = lattice(3, 3)
    model = PreparingModel(k=6)
    db = ft.Snake(graph, goal).calibrate(model)
    assert model.prepare_calls == db.step_count > 0


def test_default_prepare_is_not_called() -> None:
    """A model that does not override ``prepare`` costs no Python call per step."""
    graph, goal = lattice(3, 3)
    called: list[ft.Step] = []
    original = ft.ErrorModel.prepare

    def counting(self: ft.ErrorModel, step: ft.Step) -> Any:
        called.append(step)
        return original(self, step)

    ft.ErrorModel.prepare = counting  # type: ignore[method-assign]
    try:
        # The base-class method is what a non-overriding model inherits: not called.
        ft.Snake(graph, goal).calibrate(TargetModel(k=4))
        assert not called

        class Overriding(TargetModel):
            def prepare(self, step: ft.Step) -> Any:
                called.append(step)
                return None

        db = ft.Snake(graph, goal).calibrate(Overriding(k=4))
        assert len(called) == db.step_count
    finally:
        ft.ErrorModel.prepare = original  # type: ignore[method-assign]


def test_duck_typed_model() -> None:
    graph, goal = lattice(3, 3)

    class Duck:
        def options(self, element: ft.Element) -> list[float]:
            return [0.0, 0.5, 1.0]

        def error(self, step: ft.Step, candidates: Any) -> list[float]:
            return [float(row.sum()) for row in candidates]

    db = ft.Snake(graph, goal).calibrate(Duck())
    assert set(db.to_dict().values()) == {0.0}, "the minimum of sum() is all zeros"

    class NoPrepareWithList(Duck):
        def prepare(self, step: ft.Step) -> str:
            return "x"

    ft.Snake(graph, goal).calibrate(NoPrepareWithList())


def test_models_are_checked() -> None:
    graph, goal = lattice(3, 3)
    snake = ft.Snake(graph, goal)

    class NoError:
        def options(self, element: ft.Element) -> list[float]:
            return [0.0]

    with pytest.raises(TypeError, match=r"error\(\)"):
        snake.calibrate(NoError())
    with pytest.raises(TypeError, match=r"options\(\)"):
        snake.calibrate(object())
    with pytest.raises(TypeError, match="callable"):
        snake.calibrate(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ft.ErrorModel()  # type: ignore[abstract]
    with pytest.raises(TypeError, match="on_step"):
        snake.calibrate(TargetModel(), on_step=5)  # type: ignore[arg-type]
    assert snake.last_report is None, "nothing ran"


def test_on_step_runs_on_the_main_thread() -> None:
    graph, goal = lattice()
    main = threading.get_ident()
    where: list[int] = []
    records: list[ft.StepRecord] = []

    def on_step(record: ft.StepRecord) -> None:
        where.append(threading.get_ident())
        records.append(record)

    model = TargetModel(k=8)
    db = ft.Snake(graph, goal).calibrate(model, on_step=on_step)
    assert where and set(where) == {main}
    assert main not in model.threads, "the model runs on the engine's threads"
    assert len(records) == db.step_count

    options = np.linspace(0.0, 1.0, 8)
    for record in records:
        assert record.values is not None
        assert list(record.values) == record.parameters
        for parameter, index in zip(record.parameters, record.option_indices, strict=True):
            assert record.values[parameter] == options[index]
        assert record.pass_ == record.pass_index == 0
        assert record.duration >= record.prepare_time + record.error_time - 1e-9
        assert record.evaluations >= 1 and record.error >= 0
    # Committed values of the last writer of each element are the database's values.
    final: dict[ft.Element, float] = {}
    for record in records:
        assert record.values is not None
        final.update(record.values)
    assert final == db.to_dict()


def test_step_records_in_the_database_have_no_values() -> None:
    graph, goal = lattice(3, 3)
    db = ft.Snake(graph, goal).calibrate(TargetModel(k=5))
    steps = db.steps()
    assert [s.index for s in steps] == list(range(db.step_count))
    assert all(s.values is None for s in steps)
    assert all(isinstance(h, ft.StepRecord) for h in db.history)
    for element in db:
        assert db.step_of(element) < db.step_count
        assert 0 <= db.option_index(element) < 5
        assert db[element] == np.linspace(0.0, 1.0, 5)[db.option_index(element)]
        step = steps[db.step_of(element)]
        assert element in step.parameters


def test_on_step_error_cancels_and_is_reraised() -> None:
    graph, goal = lattice()
    calls = []

    def on_step(record: ft.StepRecord) -> None:
        calls.append(record)
        raise KeyError("from the callback")

    snake = ft.Snake(graph, goal)
    db = ft.ParameterDatabase(graph)
    with pytest.raises(KeyError, match="from the callback"):
        snake.calibrate(TargetModel(), database=db, on_step=on_step)
    assert len(calls) == 1, "no further callbacks after the first failure"
    assert db.step_count >= 1, "committed steps are kept"
    # The database is usable again, and the run can be resumed.
    snake.calibrate(TargetModel(), database=db)
    assert db.calibrated == goal


def test_database_argument_resumes_and_returns_the_same_object() -> None:
    graph, goal = lattice()
    snake = ft.Snake(graph, goal)
    db = ft.ParameterDatabase(graph)
    assert len(db) == 0 and db.step_count == 0 and not list(db)

    gate = threading.Event()

    def stop_early(record: ft.StepRecord) -> None:
        if record.index >= 2:
            snake.cancel()
            gate.set()

    with pytest.raises(ft.CancelledError) as info:
        snake.calibrate(GatedModel(gate, 4), database=db, on_step=stop_early)
    done = db.step_count
    assert 3 <= done < 16
    assert info.value.partial is db
    assert isinstance(info.value.report, ft.Report)
    assert snake.last_report is info.value.report

    resumed = snake.calibrate(TargetModel(), database=db)
    assert resumed is db
    assert db.calibrated == goal and db.step_count > done

    reference = ft.Snake(graph, goal).calibrate(TargetModel())
    assert reference.calibrated == db.calibrated
    assert reference.to_dict() == db.to_dict()


def test_calibrating_a_calibrated_database_does_nothing() -> None:
    graph, goal = lattice(3, 3)
    snake = ft.Snake(graph, goal)
    db = snake.calibrate(TargetModel())
    before = db.copy()
    again = snake.calibrate(TargetModel(), database=db)
    assert again is db and snake.last_report is not None and snake.last_report.steps == 0
    assert_same_database(db, before)


def test_report() -> None:
    graph, goal = lattice()
    snake = ft.Snake(graph, goal)
    assert snake.last_report is None
    db = snake.calibrate(TargetModel(k=9))
    report = snake.last_report
    assert report is not None
    assert report.passes >= 1 and report.steps == db.step_count
    assert report.evaluations == sum(s.evaluations for s in db.steps())
    assert report.wall_time > 0 and report.prepare_time >= 0 and report.error_time > 0
    assert report.cache_hits == report.cache_misses == 0 and report.cache_hit_rate is None
    assert report.skipped == []
    assert 1 <= len(report.worst_steps) <= 10
    errors = [s.error for s in report.worst_steps]
    assert errors == sorted(errors, reverse=True)
    assert all(s.values is not None for s in report.worst_steps)
    assert "Report" in repr(report)


def test_cache_across_steps_reports_hits() -> None:
    graph, goal = lattice()
    snake = ft.Snake(graph, goal, cache="across_steps")
    snake.calibrate(TargetModel(k=6, repel=0.0))
    report = snake.last_report
    assert report is not None and report.cache_hits + report.cache_misses > 0
    assert report.cache_hit_rate is not None
    capped = ft.Snake(graph, goal, cache=500)
    capped.calibrate(TargetModel(k=6))
    assert capped.last_report is not None and capped.last_report.cache_misses > 0
    assert snake.config["cache"] == 1 << 20 and capped.config["cache"] == 500


def test_snake_properties_and_config() -> None:
    graph, goal = lattice()
    snake = ft.Snake(
        graph,
        goal,
        parameter_distance=0,
        constraint_distance=1,
        traversal_distance=3,
        expiry_radius=2,
        seed=7,
        threads=2,
        on_model_error=("retry", 3),
        evaluation_budget=999,
    )
    assert snake.graph is graph and snake.goal == goal
    config = snake.config
    assert config["parameter_distance"] == 0 and config["constraint_distance"] == 1
    assert config["traversal_distance"] == 3 and config["expiry_radius"] == 2
    assert config["seed"] == 7 and config["threads"] == 2
    assert config["on_model_error"] == ("retry", 3) and config["evaluation_budget"] == 999
    assert config["heuristic"] == "breadth_first" and config["optimizer"] == "coordinate_descent"
    assert config["seed_policy"] == "lowest_id" and config["algorithm"] is None
    config["seed"] = 99
    assert snake.config["seed"] == 7, "config is a copy"
    assert "Snake" in repr(snake)
    default = ft.Snake(graph, goal).config
    assert (default["parameter_distance"], default["constraint_distance"]) == (1, 2)
    assert default["traversal_distance"] == 2 and default["threads"] is None


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"parameter_distance": -1}, ValueError, "parameter_distance"),
        ({"constraint_distance": 1.5}, TypeError, "constraint_distance"),
        ({"traversal_distance": True}, TypeError, "traversal_distance"),
        ({"traversal_distance": 1000}, ft.InvalidConfigError, "traversal_distance"),
        ({"expiry_radius": "1"}, TypeError, "expiry_radius"),
        ({"seed": -1}, ValueError, "seed"),
        ({"seed": 1.0}, TypeError, "seed"),
        ({"seed": 2**64}, ValueError, "seed"),
        ({"evaluation_budget": 2**63}, ValueError, "evaluation_budget"),
        ({"cache": 2**63}, ValueError, "cache"),
        ({"threads": 2**70}, ValueError, "threads"),
        ({"threads": 0}, ValueError, "threads"),
        ({"threads": "2"}, TypeError, "threads"),
        ({"evaluation_budget": 0}, ValueError, "evaluation_budget"),
        ({"on_model_error": "explode"}, ValueError, "on_model_error"),
        ({"on_model_error": "retry"}, ValueError, "retry"),
        ({"on_model_error": ("retry", 0)}, ValueError, "retries"),
        ({"on_model_error": 3}, ValueError, "on_model_error"),
        ({"cache": "forever"}, ValueError, "cache"),
        ({"cache": 0}, ValueError, "cache"),
        ({"cache": 2.5}, TypeError, "cache"),
        ({"heuristic": "sideways"}, ValueError, "heuristic"),
        ({"heuristic": 5}, TypeError, "heuristic"),
        ({"optimizer": "annealing"}, ValueError, "optimizer"),
        ({"optimizer": 5}, TypeError, "optimizer"),
        ({"seed_policy": "best"}, ValueError, "seed_policy"),
        ({"seed_policy": 5}, TypeError, "seed_policy"),
        ({"algorithm": "xeb"}, TypeError, "algorithm"),
        ({"no_such_option": 1}, TypeError, "no_such_option"),
    ],
)
def test_invalid_arguments_name_the_argument(kwargs: dict[str, Any], error: type, match: str) -> None:
    graph, goal = lattice(3, 3)
    with pytest.raises(error, match=match):
        ft.Snake(graph, goal, **kwargs)


def test_invalid_goal_and_graph() -> None:
    graph, goal = lattice(3, 3)
    other = ft.ProcessorGraph.square_lattice(3, 3)
    with pytest.raises(TypeError, match="graph"):
        ft.Snake("graph", goal)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="different graph"):
        ft.Snake(graph, other.goal())
    with pytest.raises(TypeError, match="goal"):
        ft.Snake(graph, 5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="goal"):
        ft.Snake(graph, graph.node(0, 0))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="different graph"):
        ft.Snake(graph, goal, algorithm=ft.AlgorithmSubgraphs.all_active(other))
    with pytest.raises(ValueError, match="different graph"):
        ft.Snake(graph, goal, seed_policy=[other.node(0, 0)])
    # An iterable of elements is a goal as well.
    snake = ft.Snake(graph, graph.nodes(), parameter_distance=0)
    assert snake.goal == graph.element_set(graph.nodes())


def test_empty_goal_calibrates_nothing() -> None:
    graph, _ = lattice(3, 3)
    snake = ft.Snake(graph, graph.element_set([]))
    db = snake.calibrate(TargetModel())
    assert len(db) == 0 and db.step_count == 0
    assert snake.last_report is not None and snake.last_report.steps == 0


def test_goal_of_only_nodes_with_zero_parameter_distance() -> None:
    graph = ft.ProcessorGraph.square_lattice(3, 3)
    goal = graph.element_set(graph.nodes())
    model = TargetModel(k=30)
    db = ft.Snake(graph, goal, parameter_distance=0, constraint_distance=1).calibrate(model)
    assert db.step_count == 9 and len(db) == 9
    for element, value in db.to_dict().items():
        assert abs(value - model.target(element)) <= 0.5 / 29 + 1e-12
    assert all(shape[1] == 1 for shape in model.shapes)


def test_same_inputs_same_outputs() -> None:
    graph, goal = lattice(5, 5)
    a = ft.Snake(graph, goal, seed=3, heuristic="random").calibrate(TargetModel(repel=0.3))
    b = ft.Snake(graph, goal, seed=3, heuristic="random").calibrate(TargetModel(repel=0.3))
    assert_same_database(a, b)


def test_concurrent_use_of_a_snake_is_refused() -> None:
    graph, goal = lattice()
    snake = ft.Snake(graph, goal)
    started = threading.Event()
    release = threading.Event()

    class Blocking(TargetModel):
        def error(self, step: ft.Step, candidates: Any) -> Any:
            started.set()
            release.wait(10)
            return super().error(step, candidates)

    result: list[object] = []
    worker = threading.Thread(target=lambda: result.append(snake.calibrate(Blocking())))
    worker.start()
    try:
        assert started.wait(10)
        with pytest.raises(RuntimeError, match="already running"):
            snake.calibrate(TargetModel())
        other = ft.Snake(graph, goal)  # another Snake is independent
        other.calibrate(TargetModel(k=3))
    finally:
        release.set()
        worker.join(30)
    assert len(result) == 1


def test_database_is_busy_while_it_is_calibrated() -> None:
    graph, goal = lattice()
    snake = ft.Snake(graph, goal)
    db = ft.ParameterDatabase(graph)
    seen: list[str] = []

    def on_step(record: ft.StepRecord) -> None:
        for use in (lambda: len(db), lambda: db.to_dict(), lambda: db.history, lambda: db.copy()):
            with pytest.raises(RuntimeError, match="in use"):
                use()
        with pytest.raises(RuntimeError, match="in use"):
            ft.Snake(graph, goal).calibrate(TargetModel(), database=db)
        with pytest.raises(RuntimeError, match="already running"):
            snake.calibrate(TargetModel())
        seen.append(repr(db))

    snake.calibrate(TargetModel(), database=db, on_step=on_step)
    assert seen and all("in use" in r for r in seen)
    assert len(db) == len(goal), "usable again afterwards"
