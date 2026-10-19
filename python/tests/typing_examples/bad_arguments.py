"""Wrong argument types, unknown arguments and bad literal options.

Every line that must produce an error carries an ``# E: <code>[: <text>]`` comment: the
mypy error code, and optionally a text that the message must contain. Several expected
errors on one line are separated by ``;;``. A line without such a comment must have no error.
"""

from __future__ import annotations

import faulttune as ft

graph = ft.ProcessorGraph.square_lattice(3, 3)
goal = graph.goal()
node = graph.node(0, 0)

# Wrong argument types.
ft.Snake(graph, goal, parameter_distance="1")  # E: arg-type: "str"
ft.Snake(graph, goal, threads=1.5)  # E: arg-type: "float"
ft.Snake(graph, goal, seed="abc")  # E: arg-type: "str"
ft.Snake("graph", goal)  # E: arg-type: "str"
ft.Snake(graph, node)  # E: arg-type: "Element"
ft.Snake(graph, goal, algorithm=graph)  # E: arg-type: "ProcessorGraph"
ft.Snake(graph, goal, evaluation_budget=None, expiry_radius=[1])  # E: arg-type: "list[int]"
ft.Retry("3")  # E: arg-type: "str"
ft.CoordinateDescent(random_starts=2.5)  # E: arg-type: "float"
ft.Exhaustive(max_size="big")  # E: arg-type: "str"
ft.RandomSearch(samples=None)  # E: arg-type: "None"
ft.ParameterDatabase(goal)  # E: arg-type: "ElementSet"
ft.ProcessorGraph.square_lattice(3.0, 3)  # E: arg-type: "float"
ft.ProcessorGraph.square_lattice("3", "3")  # E: arg-type ;; E: arg-type
graph.node("a", "b")  # E: arg-type ;; E: arg-type
graph.goal(exclude="node")  # E: arg-type: "str"
graph.goal(predicate=3)  # E: arg-type: "int"
graph.edge(1, 2)  # E: arg-type ;; E: arg-type
graph.element_set(node)  # E: arg-type: "Element"
goal | 5  # E: operator
goal.isdisjoint(7)  # E: arg-type: "int"
ft.ParameterDatabase(graph).save(42)  # E: arg-type: "int"
ft.read_journal(graph, "journal")  # E: arg-type ;; E: arg-type
ft.AlgorithmSubgraphs(graph, goal)  # E: arg-type: "ElementSet"
ft.AlgorithmSubgraphs(graph, 5)  # E: arg-type: "int"
ft.Snake(graph, goal).stitch_plan(goal)  # E: arg-type: "ElementSet"
ft.GraphBuilder().add_node(1)  # E: arg-type: "int"
ft.GraphBuilder().add_edge("a", "b", "engineered")  # E: arg-type: "str"

# Bad literal options.
ft.Snake(graph, goal, heuristic="bfs")  # E: arg-type: "Literal['bfs']"
ft.Snake(graph, goal, heuristic="Breadth_First")  # E: arg-type
ft.Snake(graph, goal, optimizer="nelder_mead")  # E: arg-type: "Literal['nelder_mead']"
ft.Snake(graph, goal, optimizer="coordinate descent")  # E: arg-type
ft.Snake(graph, goal, seed_policy="highest_id")  # E: arg-type: "Literal['highest_id']"
ft.Snake(graph, goal, on_model_error="ignore")  # E: arg-type: "Literal['ignore']"
ft.Snake(graph, goal, on_model_error="retry")  # E: arg-type: "Literal['retry']"
ft.Snake(graph, goal, on_model_error=("retry", "3"))  # E: arg-type
ft.Snake(graph, goal, on_model_error=("skip", 3))  # E: arg-type
ft.Snake(graph, goal, cache="always")  # E: arg-type: "Literal['always']"
ft.Snake(graph, goal, cache=1.5)  # E: arg-type: "float"

# Unknown keywords and wrong call shapes.
ft.Snake(graph, goal, parameter_dist=2)  # E: call-arg: Unexpected keyword argument "parameter_dist"
ft.Snake(graph, goal, 1, 2, 2)  # E: call-arg: Too many positional arguments
ft.Snake(graph)  # E: call-arg: Missing positional argument "goal"
ft.Snake(goal=goal)  # E: call-arg: Missing positional argument "graph"
ft.CoordinateDescent(max_sweep=3)  # E: call-arg: Unexpected keyword argument "max_sweep"
graph.goal(exclude=[ft.ElementKind.NODE], kind=ft.ElementKind.NODE)  # E: call-arg
graph.edge("a")  # E: call-arg: Missing positional argument "b"
ft.GraphBuilder().add_edge("a", "b", ft.ElementKind.CUSTOM_EDGE, 7)  # E: call-arg: Too many ;; E: arg-type: "int"
ft.Retry(times=1, extra=2)  # E: call-arg
ft.read_journal("path")  # E: call-arg: Missing positional argument "graph"
