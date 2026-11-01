"""Minimal counterexample for the demo model's local/global objective consistency.

Only runtime model arrays are changed; the FaultTune source is untouched.
All values are valid 5--7 GHz options. This is not a hardware experiment.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "examples"))
import numpy as np
import faulttune as ft
from gate_frequency import GateFrequencyModel

def run() -> dict:
    graph = ft.ProcessorGraph.square_lattice(2, 2)
    goal = graph.goal(exclude=[ft.ElementKind.PARASITIC_EDGE])
    model = GateFrequencyModel(graph, seed=53)

    # Isolate the target dependence: detuning of fixed edges on a variable endpoint.
    model.coupling[:] = 0.0
    for centre, width_sq, strength in model.defects:
        strength[:] = 0.0
    node = graph.node(0, 0)
    centre, width_sq, strength = model.defects[node.index]
    centre[0], width_sq[0], strength[0] = 5.0, 0.015**2, 0.5

    before = {element: 5.0 for element in goal}
    after = dict(before)
    after[node] = 7.0
    constraints = {element: value for element, value in before.items() if element != node}
    plan = model.plan([node], constraints)
    local = model.evaluate(plan, np.array([[5.0], [7.0]]))
    global_scores = [model.assignment_error(before), model.assignment_error(after)]

    result = {
        "scope": "demo GateFrequencyModel only; isolated synthetic 2x2 case",
        "parameter": node.label,
        "fixed_others_GHz": 5.0,
        "before_GHz": 5.0,
        "after_GHz": 7.0,
        "local_score_before": float(local[0]),
        "local_score_after": float(local[1]),
        "global_score_before": global_scores[0],
        "global_score_after": global_scores[1],
        "local_detuning_terms": len(plan.detuning),
        "affected_fixed_edges": [
            element.label for element in goal
            if element.is_edge and node in (element.endpoints or ())
        ],
    }
    assert local[1] < local[0], result
    assert global_scores[1] > global_scores[0], result
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optionally save JSON; otherwise only print it")
    args = parser.parse_args()
    result = run()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
