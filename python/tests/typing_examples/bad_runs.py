"""Misuse of calibrate, recalibrate, stitch and of the results. See bad_arguments.py."""

from __future__ import annotations

from typing import assert_type

import numpy as np
from numpy.typing import NDArray

import faulttune as ft


class Model(ft.ErrorModel):
    def options(self, element: ft.Element) -> NDArray[np.float64]:
        return np.arange(3.0)

    def error(self, step: ft.Step, candidates: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.zeros(len(candidates))


graph = ft.ProcessorGraph.square_lattice(3, 3)
goal = graph.goal()
snake = ft.Snake(graph, goal)
model = Model()
db = ft.ParameterDatabase(graph)
node = graph.node(0, 0)


def on_step_ok(record: ft.StepRecord) -> None: ...
def on_step_str(record: str) -> None: ...
def on_step_two(record: ft.StepRecord, extra: int) -> None: ...


# calibrate
snake.calibrate(model, db)  # E: call-arg: Too many positional arguments
snake.calibrate(model, on_step=on_step_str)  # E: arg-type
snake.calibrate(model, on_step=on_step_two)  # E: arg-type
snake.calibrate(model, on_step="print")  # E: arg-type: "str"
snake.calibrate(model, checkpoint=3)  # E: arg-type: "int"
snake.calibrate(model, database=graph)  # E: arg-type: "ProcessorGraph"
snake.calibrate(model, on_step=on_step_ok, callback=print)  # E: call-arg: Unexpected keyword argument "callback"
snake.calibrate()  # E: call-arg: Missing positional argument "model"
snake.calibrate("model")  # E: arg-type: "str"
snake.calibrate(42)  # E: arg-type: "int"

# recalibrate and stitch
snake.recalibrate(db, "q0_0", model)  # E: arg-type: "str"
snake.recalibrate(db, node)  # E: call-arg: Missing positional argument "model"
snake.recalibrate(node, db, model)  # E: arg-type: "Element"
snake.recalibrate(db, [1, 2], model)  # E: list-item ;; E: list-item
snake.recalibrate(db, node, model, True)  # E: call-arg: Too many ;; E: arg-type: "bool"
snake.stitch(db, goal, model)  # E: arg-type: "ElementSet"
snake.stitch(db, [goal], model, on_step=on_step_str)  # E: arg-type
snake.stitch(db, 3, model)  # E: arg-type: "int"
snake.stitch(db, [goal], model, checkpoint=[1])  # E: arg-type

# Results are read-only, and typed.
report = snake.recalibrate(db, node, model)
report.steps = 3  # E: misc: Property "steps" defined in "Report" is read-only
node.label = "other"  # E: misc: Property "label" defined in "Element" is read-only
db.step_count = 0  # E: misc: Property "step_count" defined in "ParameterDatabase" is read-only
report.no_such_field  # E: attr-defined: "Report" has no attribute "no_such_field"
db.get()  # E: call-overload
db.to_dict(labels="yes")  # E: call-overload
ft.ElementKind.OTHER  # E: attr-defined: has no attribute "OTHER"
count: str = report.steps  # E: assignment: "int"
rate: float = report.cache_hit_rate  # E: assignment: "float | None"
node.endpoints[0]  # E: index: is not indexable
last: ft.Report = snake.last_report  # E: assignment: "Report | None"
assert_type(db.to_dict(labels=True), dict[ft.Element, float])  # E: assert-type: dict[str, float]
value: str = db["q0_0"]  # E: assignment: "float"
record: ft.StepRecord = db.history[0]  # E: assignment: "StepRecord | Invalidation"
snake.config["threads"] = "many"  # E: typeddict-item
snake.config["nope"]  # E: typeddict-item


# The classes of the native module cannot be subclassed.
class MySnake(ft.Snake):  # E: misc: Cannot inherit from final class "Snake"
    pass


class MyRetry(ft.Retry):  # E: misc: Cannot inherit from final class "Retry"
    pass


# Exceptions.
try:
    snake.calibrate(model)
except ft.CancelledError as err:
    err.report.steps  # E: union-attr: Item "None" of "Report | None" has no attribute "steps"
    err.nonexistent  # E: attr-defined
except ft.ModelError as err:
    err.element.label  # E: union-attr: Item "None" of "Element | None" has no attribute "label"
except ft.SearchSpaceTooLargeError as err:
    err.size.label  # E: attr-defined: "int" has no attribute "label"
