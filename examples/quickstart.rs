//! Quickstart: the Rust walkthrough of the README as a runnable program.
//!
//! It calibrates the gate frequencies of a 4x4 lattice of qubits with the toy
//! `GateFrequencyModel` and shows the five things most users do:
//!
//! 1. describe the processor (a graph) and the calibration goal,
//! 2. pick an error model (here the toy model; yours holds your device data),
//! 3. configure and run the Snake,
//! 4. read the learned parameters by label,
//! 5. re-calibrate one drifted qubit, which touches only its neighbourhood.
//!
//! Run it with `cargo run --release -p faulttune --example quickstart`.
//! For a larger, fuller tour see the `gate_frequency` example.

use std::error::Error;
use std::time::Duration;

use faulttune::heuristics::BreadthFirst;
use faulttune::optimizers::CoordinateDescent;
use faulttune::toy::GateFrequencyModel;
use faulttune::{
    AlgorithmSubgraphs, EdgeType, ElementKind, ProcessorGraph, Report, Snake, SnakeConfig,
};

fn main() -> Result<(), Box<dyn Error>> {
    // 1. Describe the processor: 16 qubits on a 4x4 grid. Nearest-neighbour couplers are
    //    engineered edges; diagonals are parasitic edges that only carry crosstalk constraints.
    let graph = ProcessorGraph::square_lattice(4, 4);
    println!(
        "Processor: {} nodes, {} edges ({} elements in total)",
        graph.nodes().count(),
        graph.edges().count(),
        graph.len()
    );

    // Calibrate every node and every engineered edge. Parasitic edges have no parameters.
    let goal = graph.goal(|element| element.kind() != ElementKind::Edge(EdgeType::Parasitic));
    println!("Calibration goal: {} parameters", goal.len());

    // 2. The error model. This toy model knows relaxation defects and crosstalk of a random
    //    device (the seed picks the device). A real model would wrap measured data.
    let model = GateFrequencyModel::new(&graph, 2024);

    // 3. Run the Snake with the distances of the paper's XEB example.
    let config = SnakeConfig {
        parameter_distance: 1,  // d_P: optimize a qubit together with its couplers
        constraint_distance: 2, // d_R: see the calibrated gates up to two hops away
        traversal_distance: 2,  // d_T: the Snake may jump two hops per step
        ..SnakeConfig::default()
    };
    let snake = Snake::new(&graph, goal, config)
        .algorithm(AlgorithmSubgraphs::all_active(&graph)) // everything runs at once
        .heuristic(BreadthFirst)
        .optimizer(CoordinateDescent::default());

    let (mut params, report) = snake.calibrate_with_report(&model)?;
    println!("\nCalibrated {} parameters:", params.len());
    print_report(&report, &graph);

    // 4. Read the parameters by label. `params` is a ParameterDatabase: values by element,
    //    plus the ordered history of the steps that set them.
    println!("\nQubit frequencies (GHz), laid out like the lattice:");
    for row in 0..4 {
        let line: Vec<String> = (0..4)
            .filter_map(|col| graph.lattice_node(row, col))
            .filter_map(|q| params.value(q))
            .map(|f| format!("{f:6.3}"))
            .collect();
        println!("  {}", line.join("  "));
    }
    println!("Coupler frequencies (GHz), first five:");
    for (id, entry) in params
        .iter()
        .filter(|(id, _)| graph.kind(*id).is_edge())
        .take(5)
    {
        println!("  {:<10} {:6.3}", graph.label(id), entry.value);
    }

    println!(
        "\nMean error per gate: {:.4} (a naive 6 GHz everywhere would give {:.4})",
        model.mean_error(&graph, &params),
        model.baseline_mean_error(&graph, &params),
    );

    // 5. Qubit q1_3 has drifted past its threshold: forget its neighbourhood and re-learn it.
    //    Everything outside the neighbourhood keeps its value and acts as a constraint.
    //    (The toy model is unchanged here; on a real device the measured data would have
    //    changed, and the new calibration with it. The gate_frequency example simulates that.)
    let qubit = graph
        .lattice_node(1, 3)
        .ok_or("the lattice has no qubit q1_3")?;
    let before = params.clone();
    let report = snake.recalibrate(&mut params, &[qubit], &model)?;
    let changed: Vec<&str> = graph
        .ids()
        .filter(|&id| before.value(id) != params.value(id))
        .map(|id| graph.label(id))
        .collect();
    println!(
        "\nRe-calibrated {}: {} steps, {} evaluations, {} parameters changed: {}",
        graph.label(qubit),
        report.steps,
        report.evaluations,
        changed.len(),
        changed.join(", ")
    );
    Ok(())
}

/// Prints the interesting numbers of a run report.
fn print_report(report: &Report, graph: &ProcessorGraph) {
    println!("  passes            {}", report.passes);
    println!("  steps             {}", report.steps);
    println!("  evaluations       {}", report.evaluations);
    println!("  wall time         {}", millis(report.wall_time));
    println!(
        "  time in the model {} (error) + {} (prepare)",
        millis(report.error_time),
        millis(report.prepare_time)
    );
    if let Some(rate) = report.cache_hit_rate() {
        println!("  cache hit rate    {:.1}%", 100.0 * rate);
    }
    println!("  worst steps (largest error):");
    for step in report.worst_steps.iter().take(3) {
        println!(
            "    centre {:<10} {} parameters, {} constraints, error {:.4}",
            graph.label(step.centre),
            step.parameters.len(),
            step.constraints.len(),
            step.error
        );
    }
}

/// A duration in milliseconds with two decimals.
fn millis(d: Duration) -> String {
    format!("{:.2} ms", d.as_secs_f64() * 1e3)
}
