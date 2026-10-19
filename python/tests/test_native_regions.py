"""Local re-calibration and stitching."""

from __future__ import annotations

import threading

import pytest
from test_native_common import GatedModel, TargetModel, assert_same_database, lattice

import faulttune as ft


class Drifted(TargetModel):
    """The device after a drift: element targets shift by a fixed amount."""

    @staticmethod
    def target(element: ft.Element) -> float:
        return (TargetModel.target(element) + 0.3) % 1.0


def test_recalibrate_relearns_only_the_neighbourhood() -> None:
    graph, goal = lattice(5, 5)
    snake = ft.Snake(graph, goal)
    db = snake.calibrate(TargetModel(k=21))
    before = db.to_dict()
    steps_before = db.step_count

    qubit = graph.node(2, 2)
    seen: list[ft.StepRecord] = []
    report = snake.recalibrate(db, qubit, Drifted(k=21), on_step=seen.append)

    assert isinstance(report, ft.Report) and report.steps == len(seen) >= 1
    assert db.step_count == steps_before + report.steps
    assert db.calibrated == goal, "everything is calibrated again"
    invalidation = [h for h in db.history if isinstance(h, ft.Invalidation)][-1]
    region = graph.element_set(invalidation.elements)
    assert qubit in region and len(region) > 1
    assert region <= goal
    # Everything outside the forgotten region keeps its value; the region has new ones.
    after = db.to_dict()
    for element in goal - region:
        assert after[element] == before[element]
    assert any(after[e] != before[e] for e in region)
    # The re-learned steps only touch the region.
    for record in seen:
        assert set(record.parameters) <= set(region)
    # Everything in the region constrains against the untouched surroundings.
    assert any(record.constraints for record in seen)


def test_recalibrate_accepts_several_forms() -> None:
    graph, goal = lattice(5, 5)
    snake = ft.Snake(graph, goal)
    db = snake.calibrate(TargetModel())
    q1, q2 = graph.node(0, 0), graph.node(4, 4)
    for elements in (q1, [q1, q2], (e for e in [q1, q2]), graph.element_set([q1, q2]), []):
        snake.recalibrate(db, elements, TargetModel())
        assert db.calibrated == goal
    assert snake.last_report is not None and snake.last_report.steps == 0, "the empty list"
    two = ft.Snake(graph, goal)
    two.recalibrate(db, [q1, q2], TargetModel())
    assert two.last_report is not None and two.last_report.steps >= 2


def test_recalibrate_checks_its_arguments() -> None:
    graph, goal = lattice(4, 4)
    snake = ft.Snake(graph, goal)
    db = snake.calibrate(TargetModel())
    other = ft.ProcessorGraph.square_lattice(4, 4)
    parasitic = [e for e in graph.edges() if e.kind == ft.ElementKind.PARASITIC_EDGE][0]
    with pytest.raises(ft.InvalidConfigError, match="not in the calibration goal"):
        snake.recalibrate(db, parasitic, TargetModel())
    with pytest.raises(ValueError, match="different graph"):
        snake.recalibrate(db, other.node(0, 0), TargetModel())
    with pytest.raises(TypeError, match="elements"):
        snake.recalibrate(db, "q0_0", TargetModel())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="elements"):
        snake.recalibrate(db, 7, TargetModel())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        snake.recalibrate("db", graph.node(0, 0), TargetModel())  # type: ignore[arg-type]
    with pytest.raises(ft.GraphMismatchError):
        snake.recalibrate(ft.ParameterDatabase(ft.ProcessorGraph.square_lattice(3, 3)),
                          graph.node(0, 0), TargetModel())
    assert db.calibrated == goal, "nothing was forgotten by the failed calls"
    assert not any(isinstance(h, ft.Invalidation) for h in db.history)


def test_expiry_radius_widens_the_region() -> None:
    graph, goal = lattice(6, 6)
    sizes = []
    for radius in (0, 1, 2):
        snake = ft.Snake(graph, goal, expiry_radius=radius)
        db = snake.calibrate(TargetModel())
        snake.recalibrate(db, graph.node(3, 3), TargetModel())
        invalidation = [h for h in db.history if isinstance(h, ft.Invalidation)][-1]
        sizes.append(len(invalidation.elements))
    assert sizes[0] == 1 < sizes[1] < sizes[2]


def quadrants(
    graph: ft.ProcessorGraph, goal: ft.ElementSet, rows: int, cols: int
) -> list[ft.ElementSet]:
    """The four quadrants of the goal, by the layout coordinates of the elements."""
    assert graph.coords is not None
    coords = graph.coords
    top = rows / 2 - 0.5
    left = cols / 2 - 0.5

    def quadrant(upper: bool, west: bool) -> ft.ElementSet:
        return goal & graph.element_set(
            e
            for e in graph
            if (coords[e.index][1] <= top) == upper and (coords[e.index][0] <= left) == west
        )

    return [quadrant(True, True), quadrant(True, False), quadrant(False, True), quadrant(False, False)]


def test_stitch_plan_and_stitch() -> None:
    graph, goal = lattice(8, 8)
    regions = quadrants(graph, goal, 8, 8)
    assert all(regions) and sum(len(r) for r in regions) <= len(goal)
    snake = ft.Snake(graph, goal, parameter_distance=1, constraint_distance=2)
    plan = snake.stitch_plan(regions)
    assert isinstance(plan, ft.StitchPlan) and "StitchPlan" in repr(plan)
    assert plan.regions == regions
    assert plan.stage_one | plan.gap == goal
    assert plan.stage_one.isdisjoint(plan.gap) and plan.gap and plan.stage_one

    db = ft.ParameterDatabase(graph)
    steps = []
    report = snake.stitch(db, plan, TargetModel(k=8, repel=0.2), on_step=steps.append)
    assert db.calibrated == goal and report.steps == db.step_count == len(steps)
    assert report.passes >= 2, "two stages"
    stage_one_steps = [s for s in db.steps() if s.pass_ == 0]
    assert max(s.subgoal for s in stage_one_steps) >= 3, "the quadrants are separate subgoals"
    # The gap is calibrated after the regions, under constraints from both sides.
    last_stage_one = max(db.step_of(e) for e in plan.stage_one)
    assert all(db.step_of(e) > 0 for e in plan.gap)
    assert any(s.index > last_stage_one for s in db.steps())

    # Passing the regions directly gives the same result as passing the plan.
    again = ft.ParameterDatabase(graph)
    snake.stitch(again, regions, TargetModel(k=8, repel=0.2))
    assert_same_database(db, again)


def test_stitch_gives_the_same_result_for_any_thread_count() -> None:
    graph, goal = lattice(8, 8)
    regions = quadrants(graph, goal, 8, 8)
    results = []
    for threads in (1, 2, 4, None):
        snake = ft.Snake(graph, goal, threads=threads, seed=11, heuristic="random")
        db = ft.ParameterDatabase(graph)
        snake.stitch(db, regions, TargetModel(k=8, repel=0.3))
        results.append(db)
    for other in results[1:]:
        assert_same_database(results[0], other)


def test_stitch_checks_its_arguments() -> None:
    graph, goal = lattice(6, 6)
    snake = ft.Snake(graph, goal)
    nodes = graph.element_set(graph.nodes())
    with pytest.raises(ft.InvalidConfigError, match="overlap"):
        snake.stitch_plan([nodes, nodes])
    parasitic = graph.element_set(e for e in graph if e not in goal)
    with pytest.raises(ft.InvalidConfigError, match="outside the goal"):
        snake.stitch_plan([parasitic])
    with pytest.raises(TypeError, match="regions"):
        snake.stitch_plan(nodes)
    with pytest.raises(TypeError, match="regions"):
        snake.stitch_plan(5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="different graph"):
        snake.stitch_plan([ft.ProcessorGraph.square_lattice(6, 6).all_elements()])
    plan = snake.stitch_plan([nodes])
    twin = ft.ProcessorGraph.square_lattice(6, 6)
    other = ft.Snake(twin, twin.all_elements())
    with pytest.raises(ValueError, match="different graph"):
        other.stitch(ft.ParameterDatabase(twin), plan, TargetModel())
    db = ft.ParameterDatabase(graph)
    with pytest.raises(TypeError, match="regions"):
        snake.stitch(db, 5, TargetModel())  # type: ignore[arg-type]
    assert len(db) == 0
    snake.stitch(db, plan, TargetModel())
    assert db.calibrated == goal


def test_recalibrate_scattered_elements_runs_subgoals_in_parallel() -> None:
    graph, goal = lattice(10, 10)
    corners = [graph.node(1, 1), graph.node(1, 8), graph.node(8, 1), graph.node(8, 8)]
    outcomes = []
    for threads in (1, 4):
        snake = ft.Snake(graph, goal, threads=threads, seed=5)
        db = snake.calibrate(TargetModel(k=8, repel=0.2))
        snake.recalibrate(db, corners, Drifted(k=8, repel=0.2))
        outcomes.append((db, max(s.subgoal for s in db.steps()[-12:])))
    assert outcomes[0][1] >= 3, "four distant regions are four subgoals"
    assert_same_database(outcomes[0][0], outcomes[1][0])


def test_on_step_in_recalibrate_can_cancel() -> None:
    graph, goal = lattice(6, 6)
    snake = ft.Snake(graph, goal, expiry_radius=3)
    db = snake.calibrate(TargetModel())

    gate = threading.Event()

    class GatedDrift(GatedModel, Drifted):
        pass

    def stop(record: ft.StepRecord) -> None:
        snake.cancel()
        gate.set()

    with pytest.raises(ft.CancelledError) as info:
        snake.recalibrate(db, [graph.node(2, 2)], GatedDrift(gate, 1), on_step=stop)
    assert info.value.partial is db and isinstance(info.value.report, ft.Report)
    assert len(db) < len(goal), "the forgotten elements are not all back"
    # A later call finishes the job.
    snake.calibrate(TargetModel(), database=db)
    assert db.calibrated == goal
