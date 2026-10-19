//! The Rust walkthrough of the README, end to end, written in the shape of its snippets and
//! using only the public API of the crate.
//!
//! The README leaves the bodies of the user's `GateFrequencyModel` open (`/* ... */`); here the
//! struct wraps the toy model of `faulttune::toy`, whose physics it delegates to. Everything
//! else (the `use` lines, the method chain, the `?` operators) follows the README. If a snippet
//! stops compiling, the README or the public API is out of date.
//!
//! The file is self-contained.

use std::error::Error;
use std::sync::{Arc, Mutex};

// README, "1. Describe the processor".
use faulttune::{EdgeType, ElementKind, ProcessorGraph};

// README, "2. Supply an error model".
use faulttune::{ElementId, ErrorModel, ModelError, Step};

// README, "3. Run the Snake".
use faulttune::{
    heuristics::BreadthFirst, optimizers::CoordinateDescent, AlgorithmSubgraphs, Snake, SnakeConfig,
};

// Beyond the README snippets: what a user touches next.
use faulttune::heuristics::{DepthFirst, Random};
use faulttune::optimizers::{Exhaustive, RandomSearch};
use faulttune::{
    Event, FaultTuneError, HistoryRecord, Neighbourhoods, Observer, ParameterDatabase,
    RecordingObserver,
};

// ---------------------------------------------------------------------------------------------
// README, "2. Supply an error model"
// ---------------------------------------------------------------------------------------------

struct GateFrequencyModel {
    // Measured device data: relaxation spectra, crosstalk strengths, ...
    device: faulttune::toy::GateFrequencyModel,
}

impl ErrorModel for GateFrequencyModel {
    type Value = f64; // frequency in GHz
    type Data = (); // per-step calibration data; none in this example

    /// The `k` candidate values for one element.
    fn options(&self, element: ElementId) -> Vec<f64> {
        self.device.options(element)
    }

    /// Collect any calibration data this step needs.
    fn prepare(&self, step: &Step<f64>) -> Result<(), ModelError> {
        let _ = step;
        Ok(())
    }

    /// Estimated error of one candidate assignment of the step's parameters,
    /// given the values already fixed for the constraining neighbours
    /// (`step.constraints`). Return `f64::INFINITY` for forbidden candidates.
    fn error(&self, step: &Step<f64>, data: &(), candidate: &[f64]) -> f64 {
        self.device.error(step, data, candidate)
    }
}

fn gate_frequency_model(graph: &ProcessorGraph, seed: u64) -> GateFrequencyModel {
    GateFrequencyModel {
        device: faulttune::toy::GateFrequencyModel::new(graph, seed),
    }
}

// ---------------------------------------------------------------------------------------------
// The walkthrough
// ---------------------------------------------------------------------------------------------

/// Steps 1 to 4 of the README, one after the other.
#[test]
fn readme_walkthrough() -> Result<(), Box<dyn Error>> {
    // 1. Describe the processor.
    // 16 qubits on a 4x4 grid: nearest-neighbour couplers are engineered edges,
    // diagonals are parasitic edges that only carry crosstalk constraints.
    let graph = ProcessorGraph::square_lattice(4, 4);

    // Calibrate every node and every engineered edge.
    let goal = graph.goal(|element| element.kind() != ElementKind::Edge(EdgeType::Parasitic));

    assert_eq!(graph.nodes().count(), 16);
    assert_eq!(
        graph
            .edges()
            .filter(|&e| graph.kind(e) == ElementKind::Edge(EdgeType::Engineered))
            .count(),
        24
    );
    assert_eq!(
        graph
            .edges()
            .filter(|&e| graph.kind(e) == ElementKind::Edge(EdgeType::Parasitic))
            .count(),
        18
    );
    assert_eq!(goal.len(), 16 + 24);

    // 3. Run the Snake.
    let config = SnakeConfig {
        parameter_distance: 1,  // d_P
        constraint_distance: 2, // d_R
        traversal_distance: 2,  // d_T
        ..SnakeConfig::default()
    };

    let snake = Snake::new(&graph, goal, config)
        .algorithm(AlgorithmSubgraphs::all_active(&graph)) // or the subgraphs of your algorithm
        .heuristic(BreadthFirst)
        .optimizer(CoordinateDescent::default());

    let model = gate_frequency_model(&graph, 2024);
    let mut params = snake.calibrate(&model)?; // a ParameterDatabase

    // Calibration is finished when P* = G*.
    assert_eq!(params.calibrated(), snake.goal());
    assert_eq!(params.len(), 40);
    for id in snake.goal().iter() {
        let value = *params.value(id).expect("a calibrated element has a value");
        assert!(
            (5.0..=7.0).contains(&value),
            "{} = {value} GHz is outside the band",
            graph.label(id)
        );
        // The value is one of the element's options.
        let entry = params.get(id).unwrap();
        assert_eq!(model.options(id)[entry.option_index as usize], value);
    }
    // Parasitic edges have no parameters of their own.
    for id in graph.edges() {
        if graph.kind(id) == ElementKind::Edge(EdgeType::Parasitic) {
            assert!(!params.is_calibrated(id));
        }
    }
    // The Snake beats the naive "everything at 6 GHz" configuration by a wide margin.
    let tuned = model.device.mean_error(&graph, &params);
    let naive = model.device.baseline_mean_error(&graph, &params);
    assert!(
        tuned < naive,
        "the calibration ({tuned}) must beat the naive baseline ({naive})"
    );

    // 4. Re-calibrate after drift.
    // Qubit 7 has drifted past its threshold: forget its neighbourhood and re-learn it.
    let qubit_7 = ElementId::new(7);
    assert_eq!(graph.label(qubit_7), "q1_3", "the README's qubit 7 is q1_3");
    assert_eq!(graph.lattice_node(1, 3), Some(qubit_7));
    let before = params.clone();
    snake.recalibrate(&mut params, &[qubit_7], &model)?;

    // The goal is calibrated again, and only the neighbourhood of qubit 7 was touched.
    assert_eq!(params.calibrated(), snake.goal());
    let hoods = Neighbourhoods::build(&graph, snake.config().expiry_radius);
    let region: Vec<ElementId> = hoods
        .within(qubit_7, snake.config().expiry_radius)
        .iter()
        .copied()
        .filter(|&e| snake.goal().contains(e))
        .collect();
    assert!(region.contains(&qubit_7));
    // The paper's radius is 1: the qubit and its engineered edges (a border qubit has 3 of them).
    assert_eq!(region.len(), 1 + 3);
    for id in snake.goal().iter() {
        let (old, new) = (before.get(id).unwrap(), params.get(id).unwrap());
        if !region.contains(&id) {
            assert_eq!(old.value.to_bits(), new.value.to_bits());
            assert_eq!(
                old,
                new,
                "{} lies outside the removed region",
                graph.label(id)
            );
        } else {
            assert!(
                new.step >= before.step_count(),
                "{} was re-learned",
                graph.label(id)
            );
        }
    }
    // The history is an audit trail: the old steps are untouched, an Invalidated record
    // follows, then the new steps.
    let history = params.history();
    assert_eq!(&history[..before.history().len()], before.history());
    match &history[before.history().len()] {
        HistoryRecord::Invalidated(removed) => {
            let mut expected = region.clone();
            expected.sort();
            assert_eq!(removed, &expected);
        }
        other => panic!("expected an Invalidated record, found {other:?}"),
    }
    assert!(history.len() > before.history().len() + 1);
    Ok(())
}

/// The snippet of the crate documentation (`goal_excluding`) does the same as the README goal.
#[test]
fn lib_docs_goal_shape() {
    let graph = ProcessorGraph::square_lattice(4, 4);
    let goal = graph.goal_excluding(&[ElementKind::Edge(EdgeType::Parasitic)]);
    let readme_goal =
        graph.goal(|element| element.kind() != ElementKind::Edge(EdgeType::Parasitic));
    assert_eq!(goal, readme_goal);
}

/// The Python example's `graph.node(1, 3)` is `lattice_node(1, 3)` here, and `params.to_dict()`
/// corresponds to iterating the database.
#[test]
fn python_example_equivalents() -> Result<(), FaultTuneError> {
    let graph = ProcessorGraph::square_lattice(4, 4);
    let goal = graph.goal(|e| e.kind() != ElementKind::Edge(EdgeType::Parasitic));
    let snake = Snake::new(&graph, goal, SnakeConfig::default());
    let model = gate_frequency_model(&graph, 1);
    let params = snake.calibrate(&model)?;
    let as_dict: std::collections::BTreeMap<String, f64> = params
        .iter()
        .map(|(id, entry)| (graph.label(id).to_owned(), entry.value))
        .collect();
    assert_eq!(as_dict.len(), 40);
    assert!(as_dict.contains_key("q1_3"));
    assert!(as_dict.contains_key("q1_2-q1_3"));
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// The XEB setting and the figures quoted by the README
// ---------------------------------------------------------------------------------------------

/// "In the paper's gate-frequency example for XEB (`d_P = 1`, `d_R = 2`, `d_T = 2`), each step
/// optimized 1 to 5 parameters": the default configuration is that setting.
#[test]
fn xeb_setting_steps_have_one_to_five_parameters() -> Result<(), Box<dyn Error>> {
    let config = SnakeConfig::default();
    assert_eq!(
        (
            config.parameter_distance,
            config.constraint_distance,
            config.traversal_distance
        ),
        (1, 2, 2)
    );
    for (rows, cols) in [(4, 4), (3, 5), (6, 6)] {
        let graph = ProcessorGraph::square_lattice(rows, cols);
        let goal = graph.goal(|e| e.kind() != ElementKind::Edge(EdgeType::Parasitic));
        let xeb = AlgorithmSubgraphs::xeb_square_lattice(&graph, rows, cols)?;
        assert_eq!(xeb.len(), 5, "the XEB algorithm has five subgraphs");
        let recorder = RecordingObserver::new();
        let snake = Snake::new(&graph, goal, config.clone())
            .algorithm(xeb)
            .heuristic(BreadthFirst)
            .optimizer(CoordinateDescent::default())
            .observer(recorder.clone());
        let model = gate_frequency_model(&graph, 5);
        let params = snake.calibrate(&model)?;
        assert_eq!(params.calibrated(), snake.goal());
        let committed: Vec<_> = recorder
            .events()
            .into_iter()
            .filter_map(|e| match e {
                Event::StepCommitted(record) => Some(record),
                _ => None,
            })
            .collect();
        assert!(!committed.is_empty());
        for record in &committed {
            assert!(
                (1..=5).contains(&record.parameters.len()),
                "{rows}x{cols}: step at {} has {} parameters",
                graph.label(record.centre),
                record.parameters.len()
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// README, "Pluggable parts"
// ---------------------------------------------------------------------------------------------

/// Every built-in optimizer and heuristic of the table can be plugged in with the same chain.
#[test]
fn pluggable_builtin_parts() -> Result<(), FaultTuneError> {
    let graph = ProcessorGraph::square_lattice(3, 3);
    let goal = graph.goal(|e| e.kind() != ElementKind::Edge(EdgeType::Parasitic));
    let model = gate_frequency_model(&graph, 3);
    let config = SnakeConfig {
        parameter_distance: 0,
        ..SnakeConfig::default()
    };
    let make = || Snake::new(&graph, goal.clone(), config.clone());

    let dbs = [
        make()
            .heuristic(BreadthFirst)
            .optimizer(CoordinateDescent::default())
            .calibrate(&model)?,
        make()
            .heuristic(DepthFirst)
            .optimizer(Exhaustive::default())
            .calibrate(&model)?,
        make()
            .heuristic(Random)
            .optimizer(RandomSearch::default())
            .calibrate(&model)?,
    ];
    for db in &dbs {
        assert_eq!(db.calibrated(), &goal);
    }
    // With d_P = 0 each step has one parameter with 100 options, so exhaustive search is
    // optimal for every step, and the coordinate descent finds at least a good value.
    let exhaustive_error = model.device.mean_error(&graph, &dbs[1]);
    let naive = model.device.baseline_mean_error(&graph, &dbs[1]);
    assert!(exhaustive_error < naive);
    Ok(())
}

/// A user-written optimizer and traversal heuristic plug in through the public traits. The
/// trait signatures name `rand::rngs::StdRng`, so users need the same `rand` version.
#[test]
fn pluggable_custom_parts() -> Result<(), FaultTuneError> {
    use faulttune::{
        FrontierKind, IndexBatch, Objective, Optimizer, SearchSpace, Solution, TraversalContext,
        TraversalHeuristic,
    };
    use rand::rngs::StdRng;

    /// Evaluates the all-zero candidate and then each single-coordinate change once.
    struct OneSweep;
    impl Optimizer for OneSweep {
        fn optimize(
            &self,
            space: &SearchSpace,
            objective: &mut dyn Objective,
            _rng: &mut StdRng,
        ) -> Result<Solution, FaultTuneError> {
            let width = space.width();
            let mut best = vec![0u32; width];
            let mut out = [0.0];
            objective.evaluate(IndexBatch::new(&best, width), &mut out)?;
            let mut best_error = out[0];
            let mut evaluations = 1;
            for j in 0..width {
                let k = space.sizes[j];
                let mut rows = Vec::new();
                for option in 0..k {
                    let mut row = best.clone();
                    row[j] = option;
                    rows.extend(row);
                }
                let mut errors = vec![0.0; k as usize];
                objective.evaluate(IndexBatch::new(&rows, width), &mut errors)?;
                evaluations += u64::from(k);
                for (option, &error) in errors.iter().enumerate() {
                    if error < best_error {
                        best_error = error;
                        best[j] = option as u32;
                    }
                }
            }
            Ok(Solution {
                indices: best,
                error: best_error,
                evaluations,
            })
        }
    }

    /// Visits the options in descending ID order, last in first out.
    struct HighestFirst;
    impl TraversalHeuristic for HighestFirst {
        fn frontier(&self) -> FrontierKind {
            FrontierKind::Lifo
        }
        fn order(
            &self,
            _ctx: &TraversalContext<'_>,
            options: &mut Vec<ElementId>,
            _rng: &mut StdRng,
        ) {
            options.sort();
        }
    }

    let graph = ProcessorGraph::square_lattice(4, 4);
    let goal = graph.goal(|e| e.kind() != ElementKind::Edge(EdgeType::Parasitic));
    let model = gate_frequency_model(&graph, 8);
    let snake = Snake::new(&graph, goal, SnakeConfig::default())
        .heuristic(HighestFirst)
        .optimizer(OneSweep);
    let db = snake.calibrate(&model)?;
    assert_eq!(db.calibrated(), snake.goal());
    let naive = model.device.baseline_mean_error(&graph, &db);
    assert!(model.device.mean_error(&graph, &db) < naive);
    Ok(())
}

/// A user observer (a closure) sees the run, and `Arc`s of observers can be shared.
#[test]
fn observers_see_the_run() -> Result<(), FaultTuneError> {
    let graph = ProcessorGraph::square_lattice(4, 4);
    let goal = graph.goal(|e| e.kind() != ElementKind::Edge(EdgeType::Parasitic));
    let model = gate_frequency_model(&graph, 4);

    let committed = Arc::new(Mutex::new(0usize));
    let counter = Arc::clone(&committed);
    let recorder = Arc::new(RecordingObserver::new());
    let snake = Snake::new(&graph, goal, SnakeConfig::default())
        .observer(move |event: &Event| {
            if matches!(event, Event::StepCommitted(_)) {
                *counter.lock().unwrap() += 1;
            }
        })
        .observer_arc(recorder.clone() as Arc<dyn Observer>);
    let (db, report) = snake.calibrate_with_report(&model)?;

    assert_eq!(*committed.lock().unwrap() as u64, report.steps);
    assert_eq!(report.steps, db.step_count());
    assert!(report.evaluations > 0);
    assert!(report.skipped.is_empty());
    let events = recorder.events();
    assert!(matches!(
        events.first(),
        Some(Event::PassStarted { pass: 0, .. })
    ));
    match events.last() {
        Some(Event::Finished(finished)) => assert_eq!(finished.steps, report.steps),
        other => panic!("the last event must be Finished, found {other:?}"),
    }
    Ok(())
}

/// The error type is usable with `?` in a function returning `Box<dyn Error>`, and the
/// documented misuse is reported as an error, not a panic.
#[test]
fn errors_are_ordinary_errors() {
    let graph = ProcessorGraph::square_lattice(2, 2);
    let goal = graph.all_elements();
    let model = gate_frequency_model(&graph, 1);
    // The configuration is validated before anything runs.
    let bad = SnakeConfig {
        threads: Some(0),
        ..SnakeConfig::default()
    };
    let result = Snake::new(&graph, goal.clone(), bad).calibrate(&model);
    let err = result.expect_err("zero threads is invalid");
    assert!(matches!(err, FaultTuneError::InvalidConfig(_)), "{err:?}");
    let boxed: Box<dyn Error> = Box::new(err);
    assert!(boxed.to_string().contains("threads"));

    // A database of another graph is rejected.
    let other = ProcessorGraph::square_lattice(3, 3);
    let mut foreign: ParameterDatabase<f64> = ParameterDatabase::new(&other);
    let snake = Snake::new(
        &graph,
        graph.goal_excluding(&[ElementKind::Edge(EdgeType::Parasitic)]),
        SnakeConfig::default(),
    );
    let err = snake
        .calibrate_into(&mut foreign, &model)
        .expect_err("a foreign database must be refused");
    assert!(
        matches!(err, FaultTuneError::GraphMismatch { .. }),
        "{err:?}"
    );
}
