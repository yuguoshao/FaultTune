"""The error-model contract: return shapes, failures, exception chaining and policies."""

from __future__ import annotations

import math
import traceback
from typing import Any

import numpy as np
import pytest
from test_native_common import TargetModel, lattice

import faulttune as ft


def calibrate(model: Any, **kwargs: Any) -> ft.ParameterDatabase:
    graph, goal = lattice(3, 3)
    snake_args = {k: kwargs.pop(k) for k in list(kwargs) if k not in ("database", "on_step")}
    return ft.Snake(graph, goal, **snake_args).calibrate(model, **kwargs)


class Returns(TargetModel):
    """A model whose ``error`` returns whatever ``make`` builds from the candidates."""

    def __init__(self, make: Any, k: int = 6) -> None:
        super().__init__(k=k)
        self.make = make

    def error(self, step: ft.Step, candidates: Any) -> Any:
        return self.make(candidates)


@pytest.mark.parametrize(
    ("make", "shape_text"),
    [
        (lambda c: np.zeros(len(c) + 1), r"\(\d+,\)"),
        (lambda c: np.zeros(len(c) - 1), r"\(\d+,\)"),
        (lambda c: np.zeros((len(c), 1)), r"\(\d+, 1\)"),
        (lambda c: np.zeros((1, len(c))), r"\(1, \d+\)"),
        (lambda c: 0.5, r"\(\)"),
        (lambda c: c, r"\(\d+, \d+\)"),
        (lambda c: np.zeros(0), r"\(0,\)"),
        (lambda c: [], r"\(0,\)"),
    ],
)
def test_wrong_return_shapes_raise_model_error(make: Any, shape_text: str) -> None:
    with pytest.raises(ft.ModelError, match=rf"shape \(\d+,\).*returned shape {shape_text}") as info:
        calibrate(Returns(make))
    assert isinstance(info.value.element, ft.Element)
    assert info.value.__cause__ is None, "no Python exception was involved"
    assert isinstance(info.value, ft.FaultTuneError)
    assert "Returns.error()" in str(info.value)


def test_unconvertible_returns_raise_model_error_with_cause() -> None:
    with pytest.raises(ft.ModelError, match="did not return an array of floats") as info:
        calibrate(Returns(lambda c: ["a"] * len(c)))
    assert isinstance(info.value.__cause__, (TypeError, ValueError))
    # Numeric strings are not silently converted either.
    with pytest.raises(ft.ModelError, match="did not return an array of floats"):
        calibrate(Returns(lambda c: ["0.5"] * len(c)))
    with pytest.raises(ft.ModelError, match=r"returned None.*forget to return"):
        calibrate(Returns(lambda c: None))


def test_flexible_return_types_are_accepted() -> None:
    def squares(c: Any) -> Any:
        return np.asarray(c).sum(axis=1)

    reference = calibrate(Returns(squares)).to_dict(labels=True)
    assert calibrate(Returns(lambda c: list(squares(c)))).to_dict(labels=True) == reference
    assert calibrate(Returns(lambda c: tuple(squares(c)))).to_dict(labels=True) == reference
    assert calibrate(Returns(lambda c: squares(c).astype(np.float32))).step_count > 0
    assert calibrate(Returns(lambda c: (squares(c) * 100).astype(np.int64))).step_count > 0
    assert calibrate(Returns(lambda c: np.asfortranarray(squares(c)))).step_count > 0


def test_error_exception_is_chained() -> None:
    class Boom(TargetModel):
        def error(self, step: ft.Step, candidates: Any) -> Any:
            if step.centre.label == "q1_1":
                raise ValueError("boom in the model")
            return super().error(step, candidates)

    graph, goal = lattice(3, 3)
    db = ft.ParameterDatabase(graph)
    with pytest.raises(ft.ModelError) as info:
        ft.Snake(graph, goal).calibrate(Boom(), database=db)
    err = info.value
    assert err.element == graph.node(1, 1)
    assert "q1_1" in str(err) and "Boom.error()" in str(err) and "boom in the model" in str(err)
    assert isinstance(err.__cause__, ValueError) and str(err.__cause__) == "boom in the model"
    # The traceback of the cause points into the user's model.
    frames = traceback.extract_tb(err.__cause__.__traceback__)
    assert frames[-1].name == "error" and frames[-1].filename == __file__
    text = "".join(traceback.format_exception(err))
    assert "direct cause of the following exception" in text and "boom in the model" in text
    assert len(db) > 0, "the steps committed before the failure are kept"
    assert err.partial is db

    # Without database= the partial database rides on the exception.
    with pytest.raises(ft.ModelError) as info:
        ft.Snake(graph, goal).calibrate(Boom())
    partial = info.value.partial
    assert isinstance(partial, ft.ParameterDatabase) and len(partial) > 0
    # ... and resuming it with a healthy model finishes the job.
    ft.Snake(graph, goal).calibrate(TargetModel(), database=partial)
    assert partial.calibrated == goal


def test_prepare_exception_is_chained() -> None:
    class BadPrepare(TargetModel):
        def prepare(self, step: ft.Step) -> Any:
            raise OSError("instrument offline")

    with pytest.raises(ft.ModelError, match=r"BadPrepare.prepare\(\) raised OSError") as info:
        calibrate(BadPrepare())
    assert isinstance(info.value.__cause__, OSError)
    assert isinstance(info.value.element, ft.Element)


@pytest.mark.parametrize(
    ("make", "text"),
    [
        (lambda e: [], "returned no options"),
        (lambda e: np.zeros((2, 2)), "must return a 1-D array"),
        (lambda e: 3.0, "must return a 1-D array"),
        (lambda e: [0.0, float("nan")], "NaN or infinite"),
        (lambda e: [0.0, math.inf], "NaN or infinite"),
        (lambda e: "ab", "did not return an array of floats"),
        (lambda e: ["0.5", "0.7"], "did not return an array of floats"),
        (lambda e: None, "returned None"),
    ],
)
def test_bad_options_raise_model_error(make: Any, text: str) -> None:
    class BadOptions(TargetModel):
        def options(self, element: ft.Element) -> Any:
            return make(element)

    with pytest.raises(ft.ModelError, match=text) as info:
        calibrate(BadOptions())
    assert "BadOptions.options()" in str(info.value)
    assert isinstance(info.value.element, ft.Element)


def test_options_exception_is_chained() -> None:
    class Failing(TargetModel):
        def options(self, element: ft.Element) -> Any:
            raise RuntimeError("no such element in the device data")

    graph, goal = lattice(3, 3)
    with pytest.raises(ft.ModelError, match="Failing.options") as info:
        ft.Snake(graph, goal).calibrate(Failing())
    assert isinstance(info.value.__cause__, RuntimeError)
    assert frames_end(info.value.__cause__) == "options"
    assert isinstance(info.value.partial, ft.ParameterDatabase)


def frames_end(exc: BaseException) -> str:
    return traceback.extract_tb(exc.__traceback__)[-1].name


def test_options_with_other_array_likes() -> None:
    class Listy(TargetModel):
        def options(self, element: ft.Element) -> Any:
            return [0.0, 0.25, 0.5, 1]

    db = calibrate(Listy())
    assert set(db.to_dict().values()) <= {0.0, 0.25, 0.5, 1.0}

    class Ranged(TargetModel):
        def options(self, element: ft.Element) -> Any:
            return range(5)

    assert set(calibrate(Ranged()).to_dict().values()) <= {0.0, 1.0, 2.0, 3.0, 4.0}


def test_nan_errors_are_refused_and_infinity_is_allowed() -> None:
    with pytest.raises(ft.NonFiniteError) as info:
        calibrate(Returns(lambda c: np.full(len(c), np.nan)))
    assert isinstance(info.value.element, ft.Element) and "NaN" in str(info.value)

    class Forbidding(TargetModel):
        def error(self, step: ft.Step, candidates: Any) -> Any:
            errors = super().error(step, candidates)
            return np.where(candidates[:, 0] < 0.5, np.inf, errors)

    graph, goal = lattice(3, 3)
    db = ft.Snake(graph, goal, parameter_distance=0).calibrate(Forbidding(k=10))
    assert all(value >= 0.5 for value in db.to_dict().values())
    assert all(np.isfinite(s.error) for s in db.steps())


def test_everything_forbidden_gives_an_infinite_step_error() -> None:
    graph, goal = lattice(3, 3)
    db = ft.Snake(graph, goal, optimizer=ft.RandomSearch(samples=20)).calibrate(
        Returns(lambda c: np.full(len(c), np.inf))
    )
    assert len(db) == len(goal)
    assert all(s.error == math.inf for s in db.steps())


def test_search_space_too_large() -> None:
    graph, goal = lattice(3, 3)
    snake = ft.Snake(graph, goal, optimizer="exhaustive")
    with pytest.raises(ft.SearchSpaceTooLargeError) as info:
        snake.calibrate(TargetModel(k=200))
    assert info.value.budget == 1_000_000 and info.value.size >= 200**3
    assert isinstance(info.value, ft.FaultTuneError)
    # A small space is fine, and an explicit limit applies.
    ft.Snake(graph, goal, optimizer="exhaustive").calibrate(TargetModel(k=5))
    with pytest.raises(ft.SearchSpaceTooLargeError) as info:
        ft.Snake(graph, goal, optimizer=ft.Exhaustive(max_size=50)).calibrate(TargetModel(k=5))
    assert info.value.budget == 50
    with pytest.raises(ft.SearchSpaceTooLargeError):
        ft.Snake(graph, goal, optimizer="exhaustive", evaluation_budget=10).calibrate(
            TargetModel(k=5)
        )


def test_skip_policy_gives_up_on_failing_centres() -> None:
    class Picky(TargetModel):
        def error(self, step: ft.Step, candidates: Any) -> Any:
            if step.centre.label in ("q1_1", "q0_0"):
                raise ValueError("cannot measure")
            return super().error(step, candidates)

    graph, goal = lattice(3, 3)
    snake = ft.Snake(graph, goal, on_model_error="skip")
    db = snake.calibrate(Picky())
    report = snake.last_report
    assert report is not None
    assert {e.label for e in report.skipped} == {"q1_1", "q0_0"}
    assert db.calibrated == goal - graph.element_set(report.skipped) or len(db) < len(goal)
    assert all(e not in db for e in report.skipped)
    assert snake.config["on_model_error"] == "skip"


def test_retry_policy() -> None:
    class Flaky(TargetModel):
        def __init__(self, failures: int) -> None:
            super().__init__(k=6)
            self.failures = failures
            self.attempts = 0

        def prepare(self, step: ft.Step) -> Any:
            self.attempts += 1
            if self.attempts <= self.failures:
                raise OSError("flaky instrument")
            return None

    graph, goal = lattice(3, 3)
    flaky = Flaky(2)
    db = ft.Snake(graph, goal, on_model_error=("retry", 2)).calibrate(flaky)
    assert len(db) == len(goal) and flaky.attempts == db.step_count + 2

    flaky = Flaky(2)
    db = ft.Snake(graph, goal, on_model_error=ft.Retry(2)).calibrate(flaky)
    assert len(db) == len(goal)

    with pytest.raises(ft.ModelError):
        ft.Snake(graph, goal, on_model_error=("retry", 1)).calibrate(Flaky(2))
    with pytest.raises(ft.ModelError):
        ft.Snake(graph, goal).calibrate(Flaky(1))
    assert ft.Retry(3).times == 3 and ft.Retry().times == 1 and repr(ft.Retry(3)) == "Retry(3)"
    assert ft.Retry(3) == ft.Retry(3) and ft.Retry(3) != ft.Retry(4)
    with pytest.raises(ValueError, match="retries"):
        ft.Retry(0)


def test_model_may_keep_state_between_calls() -> None:
    class Counting(TargetModel):
        def __init__(self) -> None:
            super().__init__(k=5)
            self.total = 0

        def error(self, step: ft.Step, candidates: Any) -> Any:
            self.total += len(candidates)
            return super().error(step, candidates)

    model = Counting()
    graph, goal = lattice(3, 3)
    snake = ft.Snake(graph, goal)
    snake.calibrate(model)
    assert snake.last_report is not None and model.total >= snake.last_report.evaluations


def test_constraints_reach_the_model() -> None:
    graph, goal = lattice(3, 3)
    seen: list[dict[ft.Element, float]] = []

    class Watching(TargetModel):
        def error(self, step: ft.Step, candidates: Any) -> Any:
            seen.append(dict(step.constraints))
            return super().error(step, candidates)

    db = ft.Snake(graph, goal).calibrate(Watching(k=6))
    assert any(constraints for constraints in seen), "later steps are constrained"
    for constraints in seen:
        for element, value in constraints.items():
            assert element.kind != ft.ElementKind.PARASITIC_EDGE or element in goal
            assert value in np.linspace(0, 1, 6)
            assert db.step_of(element) >= 0


class FlakyOptions(TargetModel):
    """``options`` fails for one element the first ``failures`` times it is asked."""

    def __init__(self, failures: int) -> None:
        super().__init__(k=6)
        self.failures = failures
        self.attempts = 0

    def options(self, element: ft.Element) -> Any:
        if element.label == "q1_1":
            self.attempts += 1
            if self.attempts <= self.failures:
                raise OSError("device busy")
        return super().options(element)


def test_retry_policy_covers_options() -> None:
    model = FlakyOptions(failures=2)
    db = calibrate(model, on_model_error=("retry", 2))
    assert model.attempts == 3 and len(db) > 0
    again = FlakyOptions(failures=2)
    with pytest.raises(ft.ModelError, match="options") as info:
        calibrate(again, on_model_error=ft.Retry(1))
    assert isinstance(info.value.__cause__, OSError)
    assert again.attempts == 2, "one try and one retry"
    # The default (abort) tries once, and so does "skip", which cannot skip options().
    for policy in ("abort", "skip"):
        once = FlakyOptions(failures=1)
        with pytest.raises(ft.ModelError, match="options"):
            calibrate(once, on_model_error=policy)
        assert once.attempts == 1
