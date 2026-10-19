"""Building graphs, sets, and handling the exceptions."""

from __future__ import annotations

from typing import assert_type

import numpy as np
from numpy.typing import NDArray

import faulttune as ft

# A graph from the builder.
builder = ft.GraphBuilder()
builder.add_node("a", (0.0, 0.0))
builder.add_node("b", pos=(1.0, 0.0))
builder.add_node("c")
builder.add_edge("a", "b")
builder.add_edge("b", "c", ft.ElementKind.PARASITIC_EDGE, label="bc")
builder.add_edge("a", "c", ft.ElementKind.CUSTOM_EDGE, custom_type=7)
built = builder.build()
assert_type(built, ft.ProcessorGraph)
restored = ft.ProcessorGraph.from_json(built.to_json())
assert_type(restored.coords, NDArray[np.float32] | None)

# Lookups and iteration.
lattice = ft.ProcessorGraph.square_lattice(3, 3)
by_index: ft.Element = lattice[3]
by_label: ft.Element = lattice["q0_1"]
edge: ft.Element = lattice.edge("q0_0", lattice.node(0, 1), ft.ElementKind.ENGINEERED_EDGE)
for element in lattice:
    assert_type(element, ft.Element)
    assert_type(element.kind, ft.ElementKind)
    assert_type(element.endpoints, tuple[ft.Element, ft.Element] | None)
    if element.kind == ft.ElementKind.NODE and element < by_index:
        print(int(element.kind), element.index, str(element), element.custom_type)

# Sets: operators, comparisons and the goal selectors.
nodes = lattice.goal(include=ft.ElementKind.NODE)
edges = lattice.goal(include=[ft.ElementKind.ENGINEERED_EDGE, ft.ElementKind.PARASITIC_EDGE])
engineered = lattice.goal(exclude=(ft.ElementKind.PARASITIC_EDGE,), predicate=lambda e: e.index > 0)
union = nodes | edges
assert_type(union, ft.ElementSet)
assert_type(nodes & [by_index], ft.ElementSet)
assert_type([by_index] | nodes, ft.ElementSet)
assert_type(nodes - edges, ft.ElementSet)
assert_type(nodes ^ edges, ft.ElementSet)
if nodes <= union and not nodes.isdisjoint(edges) and by_index in nodes and union:
    print(len(nodes), nodes.to_list()[0], hash(nodes), nodes.graph == lattice)

# Algorithm subgraphs.
subgraphs = ft.AlgorithmSubgraphs(lattice, [nodes, edges])
assert_type(subgraphs.subgraphs, list[ft.ElementSet])
assert len(subgraphs) == 2


def handle(snake: ft.Snake, model: ft.ErrorModel) -> ft.ParameterDatabase | None:
    try:
        return snake.calibrate(model)
    except ft.CancelledError as err:
        assert_type(err.report, ft.Report | None)
        return err.partial
    except ft.ModelError as err:
        assert_type(err.element, ft.Element | None)
        assert isinstance(err.__cause__, ValueError)
        return err.partial
    except ft.NonFiniteError as err:
        print(err.element)
    except ft.SearchSpaceTooLargeError as err:
        print(err.size, err.budget)
    except (ft.InvalidConfigError, ft.GraphMismatchError) as err:
        assert isinstance(err, ValueError)
    except ft.FaultTuneError as err:
        print(err.partial)
    return None


try:
    ft.ProcessorGraph.from_json("{")
except ft.FormatError as err:
    assert isinstance(err, ft.FaultTuneError)
except ft.InvalidGraphError:
    pass

version: str = ft.__version__


class Failing(ft.ErrorModel):
    def options(self, element: ft.Element) -> NDArray[np.float64]:
        return np.arange(3.0)

    def error(self, step: ft.Step, candidates: NDArray[np.float64]) -> NDArray[np.float64]:
        raise ValueError("boom")


partial = handle(ft.Snake(lattice, nodes), Failing())
assert partial is not None and len(partial) == 0
