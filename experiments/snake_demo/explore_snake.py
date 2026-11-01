"""Small, editable experiment using FaultTune's existing toy physics model.

From the repository root, run:
    python experiments/snake_demo/explore_snake.py
No hardware is accessed. Scores are toy objectives, not physical error rates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "examples"))

import faulttune as ft
from gate_frequency import GateFrequencyModel


def run(size: int, scopes: list[int], seed: int, optimizer_seed: int, trace: bool) -> list[dict]:
    graph = ft.ProcessorGraph.square_lattice(size, size)
    goal = graph.goal(exclude=[ft.ElementKind.PARASITIC_EDGE])
    activity = ft.AlgorithmSubgraphs.xeb_square_lattice(graph, size, size)
    # Keep this device fixed when comparing different optimization scopes.
    model = GateFrequencyModel(graph, seed=seed)
    results = []
    for scope in scopes:
        snake = ft.Snake(
            graph, goal, algorithm=activity,
            parameter_distance=scope, constraint_distance=2, traversal_distance=2,
            heuristic="breadth_first", optimizer="coordinate_descent", threads=1,
            seed=optimizer_seed,
        )

        def show_step(record: ft.StepRecord) -> None:
            if not trace:
                return
            print(f"  step {record.index}: centre={record.centre.label}")
            print("    optimize:", [e.label for e in record.parameters])
            print("    fixed constraints:", [e.label for e in record.constraints])
            print("    chosen values (GHz):", {
                e.label: round(v, 6) for e, v in (record.values or {}).items()
            })
            print(f"    local score={record.error:.6f}; evaluations={record.evaluations}")

        start = perf_counter()
        db = snake.calibrate(model, on_step=show_step)
        elapsed = perf_counter() - start
        report = snake.last_report
        assert report is not None and goal <= db.calibrated
        # Re-score the COMPLETE final assignment. Local step errors need not add up
        # to a consistent global score and should not be used for this comparison.
        result = {
            "size": size, "device_seed": seed, "optimizer_seed": optimizer_seed,
            "parameter_distance": scope,
            "parameters": len(db), "steps": report.steps,
            "evaluations": report.evaluations, "wall_seconds": elapsed,
            "full_model_mean_score": model.mean_error(db),
            "all_6ghz_baseline_mean_score": model.baseline_mean_error(db),
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--scopes", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--seed", type=int, default=53, help="Seed for the toy device")
    parser.add_argument("--optimizer-seed", type=int, default=0)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.size < 2:
        parser.error("--size must be at least 2")
    records = run(args.size, args.scopes, args.seed, args.optimizer_seed, args.trace)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2) + "\n")
