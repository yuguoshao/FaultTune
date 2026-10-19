"""on_step, checkpoint, recalibrate, stitch, the database and the reports."""

from __future__ import annotations

import pathlib
from typing import assert_type

import numpy as np
from numpy.typing import NDArray

import faulttune as ft


class Model(ft.ErrorModel):
    def options(self, element: ft.Element) -> NDArray[np.float64]:
        return np.linspace(0.0, 1.0, 8)

    def error(self, step: ft.Step, candidates: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray((candidates**2).sum(axis=1))


graph = ft.ProcessorGraph.square_lattice(4, 6)
goal = graph.goal(exclude=ft.ElementKind.PARASITIC_EDGE)
snake = ft.Snake(graph, goal, seed=3)
model = Model()
seen: list[ft.StepRecord] = []


def on_step(record: ft.StepRecord) -> None:
    assert_type(record.centre, ft.Element)
    assert_type(record.values, dict[ft.Element, float] | None)
    assert_type(record.option_indices, list[int])
    print(record.pass_, record.subgoal, record.index, record.error, record.duration)


# calibrate: on_step (function, bound method, lambda), checkpoint (str and Path).
db = snake.calibrate(model, on_step=on_step, checkpoint="run.journal")
assert_type(db, ft.ParameterDatabase)
db2 = snake.calibrate(model, database=ft.ParameterDatabase(graph), on_step=seen.append)
db3 = snake.calibrate(model, on_step=lambda record: print(record), checkpoint=pathlib.Path("j"))

# Resume from a journal.
replayed: int = ft.ParameterDatabase(graph).replay_journal("run.journal", model)
records: list[ft.StepRecord] = ft.read_journal(pathlib.Path("run.journal"), graph)

# recalibrate: one element, a list, a set.
report = snake.recalibrate(db, graph.node(1, 3), model)
assert_type(report, ft.Report)
snake.recalibrate(db, [graph.node(1, 3), graph.node(2, 2)], model, on_step=on_step)
snake.recalibrate(db, graph.element_set([graph.node(0, 0)]), model, checkpoint="again.journal")
snake.recalibrate(db, goal & {graph.node(0, 0)}, model)
snake.recalibrate(db, (e for e in goal if e.is_node), model)

# stitch: a plan, or the regions themselves.
left = graph.element_set(e for e in goal if e.is_node and e.label.endswith(("_0", "_1")))
right = graph.element_set(e for e in goal if e.is_node and e.label.endswith(("_4", "_5")))
plan = snake.stitch_plan([left, right])
assert_type(plan, ft.StitchPlan)
assert_type(plan.gap, ft.ElementSet)
assert_type(plan.regions, list[ft.ElementSet])
stitched = ft.ParameterDatabase(graph)
snake.stitch(stitched, plan, model)
snake.stitch(stitched, [left, right], model, on_step=on_step)
snake.stitch(stitched, ([e for e in left], [e for e in right]), model, checkpoint="stitch.journal")

# The database.
element = graph.node(0, 0)
value: float = db[element]
by_label: float = db["q0_0"]
maybe = db.get(element)
assert_type(maybe, float | None)
parasitic = graph.goal(include=ft.ElementKind.PARASITIC_EDGE).to_list()[0]
fallback = db.get(parasitic, "absent")
assert_type(fallback, float | str)
assert_type(db.get(element, 0.0), float)
assert_type(db.to_dict(), dict[ft.Element, float])
assert_type(db.to_dict(labels=True), dict[str, float])
assert_type(db.to_dict(labels=False), dict[ft.Element, float])
flag = bool(len(db))
assert_type(db.to_dict(flag), dict[ft.Element, float] | dict[str, float])
assert_type(db.calibrated, ft.ElementSet)
assert_type(db.history, list[ft.StepRecord | ft.Invalidation])
assert_type(db.steps(), list[ft.StepRecord])
for item in db.history:
    if isinstance(item, ft.Invalidation):
        assert_type(item.elements, list[ft.Element])
    else:
        assert_type(item, ft.StepRecord)
for calibrated in db:
    assert_type(calibrated, ft.Element)
if element in db and "q0_0" in db:
    index: int = db.option_index(element) + db.step_of(element)
db.save(pathlib.Path("db.json"))
loaded = ft.ParameterDatabase.load("db.json", graph)
assert_type(loaded, ft.ParameterDatabase)
text: str = loaded.to_json()
again = ft.ParameterDatabase.from_json(graph, text)
same: bool = again.same_outcome(loaded.copy())

# Reports and records.
assert_type(report.worst_steps, list[ft.StepRecord])
assert_type(report.cache_hit_rate, float | None)
assert_type(report.skipped, list[ft.Element])
total: float = report.wall_time + report.prepare_time + report.error_time
steps: int = report.steps + report.passes + report.evaluations
last = snake.last_report
if last is not None:
    assert_type(last.steps, int)
config = snake.config
assert_type(config["parameter_distance"], int)
assert_type(config["threads"], int | None)
snake.cancel()
