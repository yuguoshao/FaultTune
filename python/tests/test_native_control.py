"""Threads, cancellation and Ctrl-C."""

from __future__ import annotations

import os
import signal
import threading
import time
from typing import Any

import pytest
from test_native_common import (
    GatedModel,
    InterruptingModel,
    TargetModel,
    assert_same_database,
    lattice,
)

import faulttune as ft


def islands(rows: int = 4, count: int = 4) -> tuple[ft.ProcessorGraph, ft.ElementSet]:
    """A wide lattice and a goal of ``count`` islands that are too far apart to interact.

    Every island is a subgoal of its own, so that the engine can run them in parallel.
    """
    graph = ft.ProcessorGraph.square_lattice(rows, 12 * count)
    assert graph.coords is not None
    coords = graph.coords
    wanted = [
        e
        for e in graph
        if e.kind != ft.ElementKind.PARASITIC_EDGE
        and any(12 * i - 0.1 <= coords[e.index][0] <= 12 * i + 3.1 for i in range(count))
    ]
    return graph, graph.element_set(wanted)


def test_thread_count_does_not_change_the_result() -> None:
    graph, goal = islands(4, 6)
    results = []
    for threads in (1, 2, 4, 8, None):
        snake = ft.Snake(graph, goal, threads=threads, seed=3, heuristic="random")
        results.append(snake.calibrate(TargetModel(k=8, repel=0.3)))
    subgoals = {s.subgoal for s in results[0].steps()}
    assert len(subgoals) == 6, "one subgoal per island, or the test proves nothing"
    for other in results[1:]:
        assert_same_database(results[0], other)


def test_threads_run_subgoals_concurrently() -> None:
    graph, goal = islands(4, 4)

    class Rendezvous(TargetModel):
        """The first steps of two subgoals can only finish if they run at the same time."""

        def __init__(self, timeout: float = 30) -> None:
            super().__init__(k=5)
            self.barrier = threading.Barrier(2, timeout=timeout)
            self.met: set[ft.Element] = set()

        def error(self, step: ft.Step, candidates: Any) -> Any:
            with self.lock:  # only the first call of each subgoal's first step waits
                first = step.index == 0 and step.centre not in self.met
                self.met.add(step.centre)
            if first:
                self.barrier.wait()
            return super().error(step, candidates)

    model = Rendezvous()
    db = ft.Snake(graph, goal, threads=4).calibrate(model)
    assert db.calibrated == goal
    assert len(model.threads) > 1, "the model was called from several threads"
    # With one thread there is nobody to meet: the model's barrier breaks.
    with pytest.raises(ft.ModelError) as info:
        ft.Snake(graph, goal, threads=1).calibrate(Rendezvous(timeout=0.5))
    assert isinstance(info.value.__cause__, threading.BrokenBarrierError)
    one = TargetModel(k=5)
    ft.Snake(graph, goal, threads=1).calibrate(one)
    assert len(one.threads) == 1


def test_recalibration_is_the_same_at_any_thread_count() -> None:
    graph, goal = lattice(10, 10)
    corners = [graph.node(1, 1), graph.node(1, 8), graph.node(8, 1), graph.node(8, 8)]
    results = []
    for threads in (1, 4):
        snake = ft.Snake(graph, goal, threads=threads, seed=9)
        db = snake.calibrate(TargetModel(k=8, repel=0.2))
        snake.recalibrate(db, corners, TargetModel(k=9, repel=0.2))
        results.append(db)
    assert_same_database(*results)
    assert max(s.subgoal for s in results[0].steps()) >= 3


def test_other_python_threads_keep_running_during_a_calibration() -> None:
    graph, goal = lattice(5, 5)
    ticks = 0
    stop = threading.Event()

    def spin() -> None:
        nonlocal ticks
        while not stop.is_set():
            ticks += 1
            time.sleep(0.001)

    worker = threading.Thread(target=spin)
    worker.start()
    try:
        ft.Snake(graph, goal).calibrate(TargetModel(k=8, delay=0.01))
    finally:
        stop.set()
        worker.join()
    assert ticks > 10, "the GIL is released while the engine waits for the model"


def test_calibrate_from_a_non_main_thread() -> None:
    graph, goal = lattice(4, 4)
    out: list[Any] = []
    caught: list[BaseException] = []

    def work() -> None:
        try:
            main = threading.get_ident()
            seen: list[int] = []
            db = ft.Snake(graph, goal).calibrate(
                TargetModel(), on_step=lambda r: seen.append(threading.get_ident())
            )
            out.append((main, set(seen), db))
        except BaseException as err:  # noqa: BLE001
            caught.append(err)

    thread = threading.Thread(target=work)
    thread.start()
    thread.join(60)
    assert not caught, caught
    main, seen, db = out[0]
    assert seen == {main}, "on_step runs on the thread that called calibrate"
    assert db.calibrated == goal


def test_no_threads_are_leaked() -> None:
    graph, goal = lattice(4, 4)
    before = threading.active_count()
    for _ in range(5):
        ft.Snake(graph, goal).calibrate(TargetModel(k=5))
    assert threading.active_count() == before


def test_cancel_from_another_thread() -> None:
    graph, goal = lattice(5, 5)
    snake = ft.Snake(graph, goal)
    db = ft.ParameterDatabase(graph)

    def cancel_from_another_thread() -> None:
        worker = threading.Thread(target=snake.cancel)
        worker.start()
        worker.join()

    # The model calls cancel() from a thread of its own while the fourth step runs.
    model = InterruptingModel(3, action=cancel_from_another_thread, wait=0.0, k=6)
    with pytest.raises(ft.CancelledError) as info:
        snake.calibrate(model, database=db)
    err = info.value
    assert err.partial is db and isinstance(err.report, ft.Report)
    assert 3 <= db.step_count <= 5, "stopped part-way"
    assert err.report.steps == db.step_count
    assert snake.last_report is err.report
    calls_when_stopped = model.error_calls
    time.sleep(0.1)
    assert model.error_calls == calls_when_stopped, "the engine has stopped for good"
    # Every committed step is there; the run can be resumed.
    ft.Snake(graph, goal).calibrate(TargetModel(k=6), database=db)
    assert db.calibrated == goal
    assert "cancelled" in str(err)


def test_cancel_without_a_running_call_is_a_no_op() -> None:
    graph, goal = lattice(3, 3)
    snake = ft.Snake(graph, goal)
    snake.cancel()
    db = snake.calibrate(TargetModel())
    assert db.calibrated == goal
    snake.cancel()
    assert snake.calibrate(TargetModel(), database=db) is db


def test_a_snake_can_be_cancelled_and_reused() -> None:
    graph, goal = lattice(4, 4)
    snake = ft.Snake(graph, goal)

    for _ in range(2):
        gate = threading.Event()

        def once(record: ft.StepRecord) -> None:
            snake.cancel()
            gate.set()

        with pytest.raises(ft.CancelledError):
            snake.calibrate(GatedModel(gate, 1), on_step=once)
    assert snake.calibrate(TargetModel()).calibrated == goal


def test_cancel_without_partial_database_returns_it_on_the_exception() -> None:
    graph, goal = lattice(4, 4)
    snake = ft.Snake(graph, goal)

    gate = threading.Event()

    def once(record: ft.StepRecord) -> None:
        snake.cancel()
        gate.set()

    with pytest.raises(ft.CancelledError) as info:
        snake.calibrate(GatedModel(gate, 1), on_step=once)
    partial = info.value.partial
    assert isinstance(partial, ft.ParameterDatabase) and 1 <= partial.step_count < 16
    snake.calibrate(TargetModel(), database=partial)
    assert partial.calibrated == goal


class SigintTimer:
    """Sends SIGINT to this process from a timer thread, as Ctrl-C would."""

    def __init__(self, delay: float) -> None:
        self.timer = threading.Timer(delay, lambda: os.kill(os.getpid(), signal.SIGINT))
        self.sent_at = 0.0

    def __enter__(self) -> SigintTimer:
        self.timer.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.timer.cancel()
        self.timer.join()


@pytest.mark.skipif(os.name == "nt", reason="SIGINT delivery is POSIX-specific here")
def test_ctrl_c_raises_keyboard_interrupt_with_a_usable_partial_database() -> None:
    graph, goal = lattice(5, 5)
    snake = ft.Snake(graph, goal)
    db = ft.ParameterDatabase(graph)
    model = InterruptingModel(3, k=6)
    with pytest.raises(KeyboardInterrupt):
        snake.calibrate(model, database=db)
    assert 3 <= db.step_count <= 5, "stopped part-way, after the step that was running"
    calls_when_stopped = model.error_calls
    time.sleep(0.1)
    assert model.error_calls == calls_when_stopped, "the engine was stopped before the raise"
    # The partial database is usable: resume it.
    ft.Snake(graph, goal).calibrate(TargetModel(k=6), database=db)
    assert db.calibrated == goal


@pytest.mark.skipif(os.name == "nt", reason="SIGINT delivery is POSIX-specific here")
def test_ctrl_c_attaches_the_partial_database_when_none_was_given() -> None:
    graph, goal = lattice(5, 5)
    snake = ft.Snake(graph, goal)
    with pytest.raises(KeyboardInterrupt) as info:
        snake.calibrate(InterruptingModel(3, k=6))
    partial = info.value.partial  # type: ignore[attr-defined]
    assert isinstance(partial, ft.ParameterDatabase) and 3 <= partial.step_count <= 5
    assert snake.calibrate(TargetModel(k=6), database=partial).calibrated == goal
    # The Snake is reusable after an interrupt.
    assert snake.calibrate(TargetModel(k=6)).calibrated == goal


@pytest.mark.skipif(os.name == "nt", reason="SIGINT delivery is POSIX-specific here")
def test_ctrl_c_during_a_custom_python_callback() -> None:
    graph, goal = lattice(5, 5)

    def slow_heuristic(ctx: ft.TraversalContext, options: list[ft.Element]) -> list[ft.Element]:
        time.sleep(0.01)
        return list(options)

    snake = ft.Snake(graph, goal, heuristic=slow_heuristic)
    with SigintTimer(0.1):
        with pytest.raises(KeyboardInterrupt):
            snake.calibrate(TargetModel(k=4))


@pytest.mark.skipif(os.name == "nt", reason="SIGINT delivery is POSIX-specific here")
def test_checkpoint_survives_ctrl_c(tmp_path: Any) -> None:
    graph, goal = lattice(5, 5)
    journal = tmp_path / "j"
    snake = ft.Snake(graph, goal)
    db = ft.ParameterDatabase(graph)
    with pytest.raises(KeyboardInterrupt):
        snake.calibrate(InterruptingModel(3, k=6), database=db, checkpoint=journal)
    restored = ft.ParameterDatabase(graph)
    restored.replay_journal(journal, TargetModel(k=6))
    assert_same_database(restored, db)


def test_on_step_callback_that_raises_keyboard_interrupt() -> None:
    graph, goal = lattice(4, 4)

    def interrupt(record: ft.StepRecord) -> None:
        raise KeyboardInterrupt

    db = ft.ParameterDatabase(graph)
    with pytest.raises(KeyboardInterrupt):
        ft.Snake(graph, goal).calibrate(TargetModel(), database=db, on_step=interrupt)
    assert db.step_count >= 1
    assert ft.Snake(graph, goal).calibrate(TargetModel(), database=db) is db


def test_last_report_belongs_to_the_last_run() -> None:
    graph, goal = lattice(3, 3)
    snake = ft.Snake(graph, goal)
    assert snake.last_report is None
    snake.calibrate(TargetModel(k=6))
    assert snake.last_report is not None

    class Bad(TargetModel):
        def error(self, step: ft.Step, candidates: Any) -> Any:
            raise RuntimeError("boom")

    with pytest.raises(ft.ModelError):
        snake.calibrate(Bad(k=6))
    assert snake.last_report is None, "the report of the previous run must not be shown"
    snake.calibrate(TargetModel(k=6))
    assert snake.last_report is not None


def test_delivered_records_match_the_database_by_pass_subgoal_and_centre() -> None:
    graph, goal = islands(3, 3)
    seen: list[ft.StepRecord] = []
    snake = ft.Snake(graph, goal, threads=4)
    db = snake.calibrate(TargetModel(k=5, repel=0.2), on_step=seen.append)
    key = lambda r: (r.pass_, r.subgoal, r.centre)  # noqa: E731
    assert sorted(map(key, seen)) == sorted(map(key, db.steps()))
    by_key = {key(r): r for r in db.steps()}
    for record in seen:
        stored = by_key[key(record)]
        assert record.option_indices == stored.option_indices
        assert record.error == stored.error
        assert record.values is not None and stored.values is None
    # Within one subgoal the records arrive in the order of the history.
    for subgoal in {r.subgoal for r in seen}:
        arrived = [r.centre for r in seen if r.subgoal == subgoal]
        assert arrived == [r.centre for r in db.steps() if r.subgoal == subgoal]
