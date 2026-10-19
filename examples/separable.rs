//! A known optimum: comparing the step optimizers on a separable model.
//!
//! In a `SeparableModel` every element has a private target and the error of an assignment is
//! the sum of squared distances to the targets. The best value of each element is therefore known
//! exactly (the option nearest to its target), and so is the global optimum of the whole
//! calibration. That makes it a test bench for the three step optimizers:
//!
//! * `CoordinateDescent` (the default) must find the global optimum for every `d_P`, because a
//!   sweep over one parameter of a separable function already finds its best option.
//! * `Exhaustive` is exact by construction, but refuses a step whose search space exceeds its
//!   budget (`k^|P|` grows far too fast with `d_P`).
//! * `RandomSearch` with a fixed number of samples is fine for one parameter and deteriorates as
//!   the number of parameters per step grows.
//!
//! The tables show, for `k = 4` and `k = 30` options per element, how many elements ended at
//! their best value and how far the total error is above the global optimum. The program checks
//! its own claims and fails if `CoordinateDescent` misses the optimum.
//!
//! Run it with `cargo run --release -p faulttune --example separable`.

use std::error::Error;
use std::sync::Arc;
use std::time::{Duration, Instant};

use faulttune::heuristics::BreadthFirst;
use faulttune::optimizers::{CoordinateDescent, Exhaustive, RandomSearch};
use faulttune::toy::SeparableModel;
use faulttune::{
    EdgeType, ElementKind, FaultTuneError, Optimizer, ParameterDatabase, ProcessorGraph, Snake,
    SnakeConfig,
};

/// The lattice is 6x6: 36 qubits and 60 couplers to calibrate.
const SIZE: usize = 6;

/// What one calibration produced.
struct Outcome {
    steps: u64,
    evaluations: u64,
    wall: Duration,
    /// Elements that ended at their best value.
    at_optimum: usize,
    calibrated: usize,
    /// Total error minus the global optimum of the calibrated elements.
    excess: f64,
    /// The reason the optimizer gave up, if it did.
    refusal: Option<FaultTuneError>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let graph = ProcessorGraph::square_lattice(SIZE, SIZE);
    let goal = graph.goal(|element| element.kind() != ElementKind::Edge(EdgeType::Parasitic));
    println!(
        "{SIZE}x{SIZE} lattice, {} elements to calibrate, d_R = 2, d_T = 2, breadth-first.",
        goal.len()
    );

    let mut coordinate_descent_is_exact = true;
    for k in [4, 30] {
        println!("\n{k} options per element");
        println!(
            "  {:>3}  {:<18} {:>6} {:>12} {:>10} {:>11} {:>14}",
            "d_P", "optimizer", "steps", "evaluations", "wall time", "at optimum", "excess error"
        );
        let model = SeparableModel::new(&graph, k, 5.0, 7.0, 11);
        for parameter_distance in [0, 1, 2] {
            let optimizers: [(&str, Arc<dyn Optimizer>); 3] = [
                ("CoordinateDescent", Arc::new(CoordinateDescent::default())),
                ("Exhaustive", Arc::new(Exhaustive::default())),
                (
                    "RandomSearch(2000)",
                    Arc::new(RandomSearch {
                        samples: 2000,
                        batch_size: 1000,
                    }),
                ),
            ];
            for (name, optimizer) in optimizers {
                let config = SnakeConfig {
                    parameter_distance,
                    ..SnakeConfig::default()
                };
                let snake = Snake::new(&graph, goal.clone(), config)
                    .heuristic(BreadthFirst)
                    .optimizer_arc(optimizer);
                let outcome = run(&snake, &model);
                print_row(parameter_distance, name, &outcome);
                if name == "CoordinateDescent" {
                    coordinate_descent_is_exact &=
                        outcome.refusal.is_none() && outcome.at_optimum == outcome.calibrated;
                }
            }
        }
    }

    println!(
        "\nCoordinateDescent found the global optimum in every case: {coordinate_descent_is_exact}"
    );
    if !coordinate_descent_is_exact {
        return Err("CoordinateDescent missed the optimum of a separable model".into());
    }
    Ok(())
}

/// Calibrates and compares the result with the known optimum.
fn run(snake: &Snake<'_>, model: &SeparableModel) -> Outcome {
    let mut db = ParameterDatabase::new(snake.graph());
    let started = Instant::now();
    let result = snake.calibrate_into(&mut db, model);
    let wall = started.elapsed();

    // The steps committed before a refusal are still in `db`.
    let (mut at_optimum, mut optimum_error) = (0, 0.0);
    for (id, entry) in db.iter() {
        let best = model.optimum(id);
        at_optimum += usize::from(entry.value == best);
        let d = best - model.target(id);
        optimum_error += d * d;
    }
    Outcome {
        steps: db.step_count(),
        evaluations: db.steps().map(|s| s.evaluations).sum(),
        wall,
        at_optimum,
        calibrated: db.len(),
        excess: model.total_error(&db) - optimum_error,
        refusal: result.err(),
    }
}

fn print_row(parameter_distance: u32, name: &str, outcome: &Outcome) {
    if outcome.steps == 0 {
        // Refused at the very first step: there is nothing to compare.
        println!(
            "  {:>3}  {:<18} {:>6} {:>12} {:>10} {:>11} {:>14}",
            parameter_distance, name, "-", "-", "-", "-", "-"
        );
    } else {
        print_numbers(parameter_distance, name, outcome);
    }
    if let Some(err) = &outcome.refusal {
        println!("       stopped early: {err}");
    }
}

fn print_numbers(parameter_distance: u32, name: &str, outcome: &Outcome) {
    println!(
        "  {:>3}  {:<18} {:>6} {:>12} {:>7.1} ms {:>5}/{:<5} {:>14.3e}",
        parameter_distance,
        name,
        outcome.steps,
        outcome.evaluations,
        outcome.wall.as_secs_f64() * 1e3,
        outcome.at_optimum,
        outcome.calibrated,
        outcome.excess
    );
}
