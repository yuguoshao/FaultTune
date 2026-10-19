"""Graphs, elements and element sets of the native extension."""

from __future__ import annotations

import numpy as np
import pytest

import faulttune as ft


@pytest.fixture
def graph() -> ft.ProcessorGraph:
    return ft.ProcessorGraph.square_lattice(3, 4)


def test_square_lattice_counts(graph: ft.ProcessorGraph) -> None:
    assert len(graph) == 12 + 17 + 12
    assert len(graph.nodes()) == 12
    assert len(graph.edges()) == 29
    assert len(graph.elements()) == len(graph) == len(list(graph))
    assert len(graph.fingerprint) == 16
    int(graph.fingerprint, 16)
    assert graph == graph and graph != ft.ProcessorGraph.square_lattice(3, 4)
    assert graph.fingerprint == ft.ProcessorGraph.square_lattice(3, 4).fingerprint
    assert "ProcessorGraph" in repr(graph)


def test_element_basics(graph: ft.ProcessorGraph) -> None:
    q = graph.node(1, 2)
    assert q.label == str(q) == "q1_2"
    assert q.kind is not None and q.kind == ft.ElementKind.NODE
    assert q.endpoints is None and q.is_node and not q.is_edge
    assert graph[q.label] == q == graph.element("q1_2") == graph[q.index]
    assert "q1_2" in repr(q)

    right = graph.node(1, 3)
    edge = graph.edge(q, right)
    assert edge.kind == ft.ElementKind.ENGINEERED_EDGE
    assert edge.endpoints == (q, right)
    assert edge.label == "q1_2-q1_3"
    assert graph.edge(right, q) == edge, "edges are undirected"
    assert graph.edge("q1_2", "q1_3") == edge, "labels work as well"
    diagonal = graph.edge(q, graph.node(2, 3))
    assert diagonal.kind == ft.ElementKind.PARASITIC_EDGE
    assert graph.edge(q, graph.node(2, 3), ft.ElementKind.PARASITIC_EDGE) == diagonal
    with pytest.raises(KeyError):
        graph.edge(q, graph.node(2, 3), ft.ElementKind.ENGINEERED_EDGE)
    with pytest.raises(KeyError):
        graph.edge(graph.node(0, 0), graph.node(2, 3))
    with pytest.raises(ValueError, match="must be a node"):
        graph.edge(edge, q)
    with pytest.raises(ValueError, match="edge kind"):
        graph.edge(q, right, ft.ElementKind.NODE)


def test_element_identity_hash_and_order(graph: ft.ProcessorGraph) -> None:
    a, b = graph.node(0, 0), graph.node(0, 1)
    assert a == graph.node(0, 0) and hash(a) == hash(graph.node(0, 0))
    assert a != b and a < b <= b and b > a >= a
    assert len({a, b, graph.node(0, 0)}) == 2
    assert {a: 1}[graph.node(0, 0)] == 1
    assert sorted([b, a]) == [a, b]

    other = ft.ProcessorGraph.square_lattice(3, 4)  # same shape, different graph
    assert a != other.node(0, 0)
    assert a != "q0_0" and not (a == 0)
    assert len({a, other.node(0, 0)}) == 2
    with pytest.raises(TypeError):
        a < other.node(0, 1)  # noqa: B015


def test_lookups_report_what_is_missing(graph: ft.ProcessorGraph) -> None:
    with pytest.raises(KeyError):
        graph.node(7, 7)
    with pytest.raises(KeyError):
        graph.node(-1, 0)
    with pytest.raises(KeyError):
        graph.element("nope")
    with pytest.raises(KeyError):
        graph["nope"]
    with pytest.raises(IndexError):
        graph[len(graph)]
    assert graph[-1] == graph.elements()[-1]
    with pytest.raises(TypeError):
        graph[1.5]  # type: ignore[index]
    assert graph.node(0, 0) in graph and "q0_0" in graph and "zz" not in graph
    assert ft.ProcessorGraph.square_lattice(3, 4).node(0, 0) not in graph
    assert 5 not in graph


def test_lattice_arguments_are_checked() -> None:
    with pytest.raises(ValueError, match="rows"):
        ft.ProcessorGraph.square_lattice(-1, 3)
    with pytest.raises(ValueError, match="cols"):
        ft.ProcessorGraph.square_lattice(3, -3)
    with pytest.raises(ValueError, match="too large"):
        ft.ProcessorGraph.square_lattice(10**6, 10**6)
    assert len(ft.ProcessorGraph.square_lattice(0, 0)) == 0


def test_json_round_trip(graph: ft.ProcessorGraph) -> None:
    clone = ft.ProcessorGraph.from_json(graph.to_json())
    assert clone.fingerprint == graph.fingerprint
    assert [e.label for e in clone] == [e.label for e in graph]
    assert np.array_equal(clone.coords, graph.coords)
    with pytest.raises(ft.FormatError):
        ft.ProcessorGraph.from_json("{not json")
    with pytest.raises(ft.FaultTuneError):
        ft.ProcessorGraph.from_json('{"format": "other/9", "elements": []}')
    with pytest.raises(ft.InvalidGraphError):
        ft.ProcessorGraph.from_json(
            '{"format": "faulttune.graph/1", "elements": ['
            '{"label": "a", "kind": "engineered", "endpoints": ["x", "y"]}]}'
        )


def test_coords(graph: ft.ProcessorGraph) -> None:
    coords = graph.coords
    assert coords is not None and coords.dtype == np.float32
    assert coords.shape == (len(graph), 2)
    assert tuple(coords[graph.node(2, 3).index]) == (3.0, 2.0)
    assert coords is not graph.coords, "a fresh array each time"
    builder = ft.GraphBuilder()
    builder.add_node("a")
    assert builder.build().coords is None


def test_graph_builder() -> None:
    b = ft.GraphBuilder()
    assert b.add_node("a", pos=(0, 0)) == "a"
    b.add_node("b", pos=(1, 0))
    b.add_node("c", pos=(2, 0))
    assert b.add_edge("a", "b") == "a-b"
    b.add_edge("b", "c", ft.ElementKind.PARASITIC_EDGE, label="bc")
    b.add_edge("a", "c", ft.ElementKind.CUSTOM_EDGE, custom_type=7)
    with pytest.raises(ft.InvalidGraphError):
        b.add_node("a")
    with pytest.raises(KeyError):
        b.add_edge("a", "zzz")
    with pytest.raises(ValueError, match="custom_type"):
        b.add_edge("a", "b", ft.ElementKind.CUSTOM_EDGE)
    with pytest.raises(ValueError, match="CUSTOM_EDGE"):
        b.add_edge("a", "b", custom_type=3)
    g = b.build()
    assert [e.label for e in g] == ["a", "b", "c", "a-b", "bc", "a-c"]
    assert g["bc"].kind == ft.ElementKind.PARASITIC_EDGE
    custom = g["a-c"]
    assert custom.kind == ft.ElementKind.CUSTOM_EDGE and custom.custom_type == 7
    assert g["a-b"].custom_type is None
    assert g.coords is not None and tuple(g.coords[custom.index]) == (1.0, 0.0)
    with pytest.raises(ValueError, match="built already"):
        b.build()
    with pytest.raises(ValueError, match="built already"):
        b.add_node("z")
    assert ft.ProcessorGraph.from_json(g.to_json()).fingerprint == g.fingerprint


def test_goal_filters(graph: ft.ProcessorGraph) -> None:
    kinds = ft.ElementKind
    no_parasitic = graph.goal(exclude=[kinds.PARASITIC_EDGE])
    assert len(no_parasitic) == 12 + 17
    assert graph.goal(exclude=kinds.PARASITIC_EDGE) == no_parasitic, "a single kind works"
    assert len(graph.goal()) == len(graph)
    assert len(graph.goal(include=[kinds.NODE])) == 12
    assert graph.goal(include=[kinds.NODE, kinds.ENGINEERED_EDGE]) == no_parasitic
    assert graph.goal(include=[kinds.NODE, kinds.ENGINEERED_EDGE], exclude=[kinds.NODE]) == (
        graph.goal(include=[kinds.ENGINEERED_EDGE])
    )
    row0 = graph.goal(include=[kinds.NODE], predicate=lambda e: e.label.startswith("q0_"))
    assert row0.to_list() == [graph.node(0, c) for c in range(4)]
    assert graph.goal(exclude=None, include=None, predicate=None) == graph.all_elements()
    assert len(graph.all_elements()) == len(graph)
    with pytest.raises(TypeError, match="exclude"):
        graph.goal(exclude=["node"])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="predicate"):
        graph.goal(predicate=5)  # type: ignore[arg-type]
    with pytest.raises(ZeroDivisionError):
        graph.goal(predicate=lambda e: 1 / 0)


def test_element_set_basics(graph: ft.ProcessorGraph) -> None:
    nodes = graph.element_set(graph.nodes())
    q00, q01 = graph.node(0, 0), graph.node(0, 1)
    assert len(nodes) == 12 and nodes
    assert q00 in nodes and graph.edges()[0] not in nodes
    assert "q0_0" not in nodes and 3 not in nodes
    assert list(nodes) == nodes.to_list() == graph.nodes(), "ascending id order"
    assert nodes.graph == graph
    assert not graph.element_set([])
    assert "ElementSet" in repr(nodes)

    a = graph.element_set([q00, q01])
    b = graph.element_set([q01, graph.node(0, 2)])
    assert (a | b).to_list() == [q00, q01, graph.node(0, 2)]
    assert (a & b).to_list() == [q01]
    assert (a - b).to_list() == [q00]
    assert (a ^ b).to_list() == [q00, graph.node(0, 2)]
    assert (a | [graph.node(1, 1)]).to_list() == [q00, q01, graph.node(1, 1)]
    assert [q01] | a == a, "reflected operators accept iterables"
    assert (a & {q00}) == graph.element_set([q00])
    assert [q00, q01, graph.node(0, 2)] - a == graph.element_set([graph.node(0, 2)])
    assert graph.element_set([q00]) <= a and graph.element_set([q00]) < a
    assert a <= a and not a < a and a >= a and nodes > a and not a >= nodes
    assert a == graph.element_set([q01, q00]) and a != b
    assert hash(a) == hash(graph.element_set([q01, q00]))
    assert {a, graph.element_set([q00, q01])} == {a}
    assert a.isdisjoint(graph.element_set([graph.node(2, 3)])) and not a.isdisjoint(b)
    assert a.isdisjoint([graph.node(2, 3)])


def test_element_sets_check_their_inputs(graph: ft.ProcessorGraph) -> None:
    other = ft.ProcessorGraph.square_lattice(3, 4)
    with pytest.raises(ValueError, match="different graph"):
        graph.element_set([other.node(0, 0)])
    with pytest.raises(ValueError, match="different graph"):
        graph.element_set([graph.node(0, 0)]) | other.element_set([other.node(0, 0)])
    with pytest.raises(TypeError, match="Element"):
        graph.element_set(["q0_0"])
    with pytest.raises(TypeError, match="iterable"):
        graph.element_set(graph.node(0, 0))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="iterable"):
        graph.element_set("q0_0")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        graph.element_set(5)  # type: ignore[arg-type]
    a = graph.element_set([graph.node(0, 0)])
    assert (a == other.element_set([other.node(0, 0)])) is False
    assert (a != other.element_set([other.node(0, 0)])) is True
    assert a.__eq__(5) is NotImplemented
    assert graph.element_set(e for e in graph.nodes()) == graph.element_set(graph.nodes())


def test_building_a_big_lattice_lets_other_threads_run() -> None:
    import threading
    import time

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
        started = time.monotonic()
        graph = ft.ProcessorGraph.square_lattice(300, 300)
        text = graph.to_json()
        ft.ProcessorGraph.from_json(text)
        elapsed = time.monotonic() - started
    finally:
        stop.set()
        worker.join()
    # The other thread would tick about once per millisecond if the GIL were free.
    assert ticks > elapsed * 100, (ticks, elapsed)
