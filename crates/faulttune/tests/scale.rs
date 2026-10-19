//! Scale tests (ARCHITECTURE sections 1.2 and 12): the engine's own work per step must not grow
//! with the size of the processor, so the total engine time is linear in the number of elements.
//!
//! * The ordinary tests use a 30x30 lattice (900 qubits, 2640 goal elements) with the toy
//!   `GateFrequencyModel`. They finish in a few seconds in a debug build.
//! * The `#[ignore]`d tests use a 100x100 lattice (10,000 qubits, the design target) and print
//!   timings. Run them with
//!
//!   ```text
//!   cargo test -p faulttune --release --test scale -- --ignored --nocapture --test-threads=1
//!   ```
//!
//! Timing assertions are deliberately loose (they catch a hang or an accidental quadratic
//! algorithm, not a 20% regression); the printed numbers are the measurement.
//!
//! The file is self-contained.

use std::time::{Duration, Instant};

use faulttune::toy::{GateFrequencyModel, SeparableModel};
use faulttune::{
    AlgorithmSubgraphs, EdgeType, ElementId, ElementKind, ElementSet, Event, Neighbourhoods,
    ParameterDatabase, ProcessorGraph, RecordingObserver, Report, Snake, SnakeConfig,
};

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

fn lattice(n: usize) -> (ProcessorGraph, ElementSet) {
    let graph = ProcessorGraph::square_lattice(n, n);
    let goal = graph.goal_excluding(&[ElementKind::Edge(EdgeType::Parasitic)]);
    (graph, goal)
}

fn config(threads: Option<usize>) -> SnakeConfig {
    SnakeConfig {
        parameter_distance: 1,
        constraint_distance: 2,
        traversal_distance: 2,
        seed: 2024,
        threads,
        ..SnakeConfig::default()
    }
}

fn nodes(graph: &ProcessorGraph) -> usize {
    graph.nodes().count()
}

/// The goal cut into `side x side` blocks of a `n x n` lattice (edges belong to the block of
/// their midpoint).
fn blocks(graph: &ProcessorGraph, goal: &ElementSet, n: usize, side: usize) -> Vec<ElementSet> {
    let position = |id: ElementId| -> (usize, usize) {
        let (r, c) = graph.label(id)[1..].split_once('_').unwrap();
        (r.parse().unwrap(), c.parse().unwrap())
    };
    let mut parts = vec![ElementSet::new(graph.len()); side * side];
    for id in goal.iter() {
        // Twice the row and column of the midpoint.
        let (r2, c2) = match graph.endpoints(id) {
            None => {
                let (r, c) = position(id);
                (2 * r, 2 * c)
            }
            Some([a, b]) => {
                let ((ra, ca), (rb, cb)) = (position(a), position(b));
                (ra + rb, ca + cb)
            }
        };
        let (br, bc) = (r2 * side / (2 * n), c2 * side / (2 * n));
        parts[br * side + bc].insert(id);
    }
    parts
}

/// The statistics of a run that the printed lines and the assertions use.
struct Timing {
    wall: Duration,
    report: Report,
}

impl Timing {
    /// Wall time not spent in the model (CPU-like sums of `prepare` and the error evaluations
    /// are subtracted; meaningful for runs on one thread).
    fn engine_overhead(&self) -> Duration {
        self.wall
            .saturating_sub(self.report.error_time + self.report.prepare_time)
    }

    fn overhead_per_step(&self) -> Duration {
        self.engine_overhead() / self.report.steps.max(1) as u32
    }
}

fn calibrate_timed<M: faulttune::ErrorModel>(
    snake: &Snake<'_>,
    model: &M,
) -> (ParameterDatabase<M::Value>, Timing) {
    let started = Instant::now();
    let (db, report) = snake.calibrate_with_report(model).unwrap();
    (
        db,
        Timing {
            wall: started.elapsed(),
            report,
        },
    )
}

/// Checks a finished gate-frequency calibration: complete, one step per qubit (the node steps
/// absorb their engineered edges), the evaluation bound of ARCHITECTURE 12.2, and an error far
/// below the naive "everything at 6 GHz" baseline.
fn check_gate_frequency_result(
    graph: &ProcessorGraph,
    goal: &ElementSet,
    model: &GateFrequencyModel,
    db: &ParameterDatabase<f64>,
    report: &Report,
) {
    assert_eq!(db.calibrated(), goal);
    assert_eq!(report.steps, nodes(graph) as u64, "one step per qubit");
    // At most 4 random starts and 3 sweeps over 5 parameters of 100 options per step.
    let bound = report.steps * (4 + 3 * 5 * GateFrequencyModel::NUM_OPTIONS as u64);
    assert!(
        report.evaluations <= bound,
        "{} evaluations exceed {bound}",
        report.evaluations
    );
    let tuned = model.mean_error(graph, db);
    let naive = model.baseline_mean_error(graph, db);
    assert!(
        tuned * 5.0 < naive,
        "tuned {tuned} should be far below the baseline {naive}"
    );
}

// ---------------------------------------------------------------------------------------------
// 30 x 30 (debug builds)
// ---------------------------------------------------------------------------------------------

#[test]
fn thirty_by_thirty_gate_frequencies_in_a_few_seconds() {
    let (graph, goal) = lattice(30);
    assert_eq!(graph.nodes().count(), 900);
    assert_eq!(goal.len(), 900 + 2 * 30 * 29);
    let model = GateFrequencyModel::new(&graph, 5);
    let snake = Snake::new(&graph, goal.clone(), config(None));
    let (db, timing) = calibrate_timed(&snake, &model);
    println!(
        "30x30: {:?} wall, {} steps, {} evaluations, {} pass(es), prepare {:?}, error {:?}",
        timing.wall,
        timing.report.steps,
        timing.report.evaluations,
        timing.report.passes,
        timing.report.prepare_time,
        timing.report.error_time
    );
    check_gate_frequency_result(&graph, &goal, &model, &db, &timing.report);
    assert_eq!(
        timing.report.passes, 1,
        "a lattice is one connected subgoal and one pass"
    );
    assert!(
        timing.wall < Duration::from_secs(60),
        "30x30 took {:?}; it takes a few seconds in a debug build",
        timing.wall
    );
}

#[test]
fn thirty_by_thirty_with_the_xeb_subgraphs() {
    let (graph, goal) = lattice(30);
    let xeb = AlgorithmSubgraphs::xeb_square_lattice(&graph, 30, 30).unwrap();
    let model = GateFrequencyModel::new(&graph, 6);
    let recorder = RecordingObserver::new();
    let snake = Snake::new(&graph, goal.clone(), config(None))
        .algorithm(xeb)
        .observer(recorder.clone());
    let (db, timing) = calibrate_timed(&snake, &model);
    println!(
        "30x30 XEB: {:?} wall, {} steps",
        timing.wall, timing.report.steps
    );
    check_gate_frequency_result(&graph, &goal, &model, &db, &timing.report);
    // Steps have 1 to 5 parameters whatever the size.
    let max = recorder
        .events()
        .iter()
        .filter_map(|e| match e {
            Event::StepCommitted(s) => Some(s.parameters.len()),
            _ => None,
        })
        .max();
    assert_eq!(max, Some(3), "a breadth-first walk from the corner seed");
    assert!(timing.wall < Duration::from_secs(60), "{:?}", timing.wall);
}

#[test]
fn engine_work_per_step_does_not_grow_with_the_processor() {
    // A model that costs next to nothing, on one thread, so that what is left of the wall
    // time is the engine (traversal, scoping, bookkeeping, the optimizer's own work).
    let per_step = |n: usize| -> (Duration, usize) {
        let (graph, goal) = lattice(n);
        let model = SeparableModel::new(&graph, 20, 5.0, 7.0, 1);
        let snake = Snake::new(&graph, goal, config(Some(1)));
        // The best of two runs filters out scheduling noise.
        let best = (0..2)
            .map(|_| calibrate_timed(&snake, &model).1.overhead_per_step())
            .min()
            .unwrap();
        (best, graph.len())
    };
    let (small, small_len) = per_step(10);
    let (large, large_len) = per_step(60);
    println!(
        "engine overhead per step: {small:?} at {small_len} elements, {large:?} at {large_len} elements"
    );
    // 40 times more elements; the cost per step may not follow (a linear growth would give a
    // ratio near 40).
    assert!(
        large < small * 2 + Duration::from_micros(50),
        "the work per step grew from {small:?} to {large:?}"
    );
}

#[test]
fn the_neighbourhood_index_grows_linearly_with_the_processor() {
    let bytes_per_element = |n: usize| {
        let graph = ProcessorGraph::square_lattice(n, n);
        let hoods = Neighbourhoods::build(&graph, 2);
        hoods.memory_bytes() as f64 / graph.len() as f64
    };
    let small = bytes_per_element(10);
    let large = bytes_per_element(60);
    println!(
        "neighbourhood index at depth 2: {small:.0} bytes per element (10x10), {large:.0} (60x60)"
    );
    // ARCHITECTURE 12.1: about 17 MB for 50,000 elements, that is about 340 bytes each.
    assert!(large < 500.0, "{large} bytes per element");
    assert!(
        large < 1.4 * small,
        "per-element memory grew from {small} to {large}"
    );
}

#[test]
fn a_stitched_thirty_by_thirty_is_the_same_at_every_thread_count() {
    let (graph, goal) = lattice(30);
    let model = GateFrequencyModel::new(&graph, 7);
    let regions = blocks(&graph, &goal, 30, 2);
    let mut runs = Vec::new();
    for threads in [1usize, 3, 8] {
        let recorder = RecordingObserver::new();
        let snake =
            Snake::new(&graph, goal.clone(), config(Some(threads))).observer(recorder.clone());
        let plan = snake.stitch_plan(regions.clone()).unwrap();
        assert!(!plan.gap.is_empty() && !plan.stage_one.is_empty());
        let mut db = ParameterDatabase::new(&graph);
        let started = Instant::now();
        let report = snake.stitch(&mut db, &plan, &model).unwrap();
        println!(
            "30x30 stitched, {threads} thread(s): {:?}, {} steps",
            started.elapsed(),
            report.steps
        );
        assert_eq!(db.calibrated(), &goal);
        // Stage one has four independent subgoals.
        let first_pass_subgoals = recorder.events().iter().find_map(|e| match e {
            Event::PassStarted {
                pass: 0, subgoals, ..
            } => Some(*subgoals),
            _ => None,
        });
        assert_eq!(first_pass_subgoals, Some(4));
        runs.push(db);
    }
    assert!(runs[0].same_outcome(&runs[1]), "1 and 3 threads");
    assert!(runs[0].same_outcome(&runs[2]), "1 and 8 threads");
    let tuned = model.mean_error(&graph, &runs[0]);
    let naive = model.baseline_mean_error(&graph, &runs[0]);
    assert!(tuned * 5.0 < naive, "tuned {tuned}, baseline {naive}");
}

#[test]
fn scattered_drift_on_thirty_by_thirty_recalibrates_many_small_subgoals() {
    let (graph, goal) = lattice(30);
    let model = GateFrequencyModel::new(&graph, 8);
    let recorder = RecordingObserver::new();
    let snake = Snake::new(&graph, goal.clone(), config(None)).observer(recorder.clone());
    let mut db = snake.calibrate(&model).unwrap();
    let before = db.clone();

    // 25 qubits on a grid with spacing 6: their removal regions (the qubit and its edges) are
    // more than max(d_P, d_R) = 2 apart, so every one is a subgoal of its own.
    let drifted: Vec<ElementId> = (0..5)
        .flat_map(|i| (0..5).map(move |j| (3 + 6 * i, 3 + 6 * j)))
        .map(|(r, c)| graph.lattice_node(r, c).unwrap())
        .collect();
    let recorder_len = recorder.events().len();
    let started = Instant::now();
    let report = snake.recalibrate(&mut db, &drifted, &model).unwrap();
    println!(
        "30x30 recalibration of {} scattered qubits: {:?}, {} steps",
        drifted.len(),
        started.elapsed(),
        report.steps
    );
    assert_eq!(db.calibrated(), &goal);
    assert_eq!(
        report.steps,
        drifted.len() as u64,
        "one step per drifted qubit"
    );
    let events = recorder.events();
    let subgoals = events[recorder_len..].iter().find_map(|e| match e {
        Event::PassStarted { subgoals, .. } => Some(*subgoals),
        _ => None,
    });
    assert_eq!(subgoals, Some(drifted.len()));

    // Only the removal regions changed.
    let hoods = Neighbourhoods::build(&graph, 1);
    let mut region = ElementSet::new(graph.len());
    for &d in &drifted {
        for &e in hoods.within(d, 1) {
            if goal.contains(e) {
                region.insert(e);
            }
        }
    }
    for id in goal.iter() {
        if region.contains(id) {
            assert!(db.get(id).unwrap().step >= before.step_count());
        } else {
            assert_eq!(
                db.get(id),
                before.get(id),
                "{} lies outside the removal",
                graph.label(id)
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// 100 x 100 (heavy: run with --ignored --release)
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "heavy: 10,000 qubits; run with `cargo test --release --test scale -- --ignored --nocapture`"]
fn heavy_hundred_by_hundred_gate_frequencies() {
    let started = Instant::now();
    let (graph, goal) = lattice(100);
    println!(
        "100x100: graph with {} elements built in {:?}",
        graph.len(),
        started.elapsed()
    );
    let started = Instant::now();
    let model = GateFrequencyModel::new(&graph, 5);
    println!("100x100: model built in {:?}", started.elapsed());
    let started = Instant::now();
    let hoods = Neighbourhoods::build(&graph, 2);
    println!(
        "100x100: neighbourhood index of depth 2 built in {:?}, {:.1} MB",
        started.elapsed(),
        hoods.memory_bytes() as f64 / 1e6
    );
    drop(hoods);

    let snake = Snake::new(&graph, goal.clone(), config(None));
    let (db, timing) = calibrate_timed(&snake, &model);
    println!(
        "100x100 sequential: {:?} wall, {} steps, {} evaluations, prepare {:?}, error {:?}, \
         engine overhead {:?} ({:?} per step)",
        timing.wall,
        timing.report.steps,
        timing.report.evaluations,
        timing.report.prepare_time,
        timing.report.error_time,
        timing.engine_overhead(),
        timing.overhead_per_step()
    );
    check_gate_frequency_result(&graph, &goal, &model, &db, &timing.report);
    println!(
        "100x100: mean error {:.4} (naive baseline {:.4})",
        model.mean_error(&graph, &db),
        model.baseline_mean_error(&graph, &db)
    );
    assert!(timing.wall < Duration::from_secs(1800), "{:?}", timing.wall);
}

#[test]
#[ignore = "heavy: 10,000 qubits; run with `cargo test --release --test scale -- --ignored --nocapture`"]
fn heavy_hundred_by_hundred_with_the_xeb_subgraphs() {
    let (graph, goal) = lattice(100);
    let xeb = AlgorithmSubgraphs::xeb_square_lattice(&graph, 100, 100).unwrap();
    let model = GateFrequencyModel::new(&graph, 6);
    let snake = Snake::new(&graph, goal.clone(), config(None)).algorithm(xeb);
    let (db, timing) = calibrate_timed(&snake, &model);
    println!(
        "100x100 XEB: {:?} wall, {} steps, {} evaluations",
        timing.wall, timing.report.steps, timing.report.evaluations
    );
    check_gate_frequency_result(&graph, &goal, &model, &db, &timing.report);
}

#[test]
#[ignore = "heavy: 10,000 qubits; run with `cargo test --release --test scale -- --ignored --nocapture`"]
fn heavy_hundred_by_hundred_stitched_speed_up() {
    let (graph, goal) = lattice(100);
    let model = GateFrequencyModel::new(&graph, 5);
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    println!("100x100 stitched: {cores} cores available");
    let mut reference: Option<ParameterDatabase<f64>> = None;
    for (side, threads) in [
        (2usize, 1usize),
        (2, 4),
        (4, 1),
        (4, 4),
        (4, cores.max(2)),
        (8, cores.max(2)),
    ] {
        let regions = blocks(&graph, &goal, 100, side);
        let snake = Snake::new(&graph, goal.clone(), config(Some(threads)));
        let plan = snake.stitch_plan(regions).unwrap();
        let mut db = ParameterDatabase::new(&graph);
        let started = Instant::now();
        let report = snake.stitch(&mut db, &plan, &model).unwrap();
        let wall = started.elapsed();
        println!(
            "100x100 stitched in {side}x{side} regions, {threads} thread(s): {wall:?} wall, \
             stage one {} elements, gap {}, {} steps",
            plan.stage_one.len(),
            plan.gap.len(),
            report.steps
        );
        assert_eq!(db.calibrated(), &goal);
        let tuned = model.mean_error(&graph, &db);
        let naive = model.baseline_mean_error(&graph, &db);
        assert!(tuned * 5.0 < naive);
        // The same plan gives the same result at every thread count.
        if side == 4 && threads == 1 {
            reference = Some(db);
        } else if side == 4 {
            assert!(
                reference.as_ref().unwrap().same_outcome(&db),
                "{threads} threads differ"
            );
        }
    }
}

#[test]
#[ignore = "heavy: 10,000 qubits; run with `cargo test --release --test scale -- --ignored --nocapture`"]
fn heavy_hundred_by_hundred_engine_overhead_with_a_free_model() {
    let (graph, goal) = lattice(100);
    let model = SeparableModel::new(&graph, 20, 5.0, 7.0, 1);
    let snake = Snake::new(&graph, goal.clone(), config(Some(1)));
    let (db, timing) = calibrate_timed(&snake, &model);
    println!(
        "100x100 with a free model: {:?} wall, {} steps, engine overhead {:?} ({:?} per step)",
        timing.wall,
        timing.report.steps,
        timing.engine_overhead(),
        timing.overhead_per_step()
    );
    assert_eq!(db.calibrated(), &goal);
    // The overhead per step is that of a small lattice, within a loose factor.
    let (small_graph, small_goal) = lattice(10);
    let small_model = SeparableModel::new(&small_graph, 20, 5.0, 7.0, 1);
    let small = Snake::new(&small_graph, small_goal, config(Some(1)));
    let small_timing = calibrate_timed(&small, &small_model).1;
    println!(
        "10x10 with a free model: {:?} per step",
        small_timing.overhead_per_step()
    );
    assert!(timing.overhead_per_step() < small_timing.overhead_per_step() * 5);
}

#[test]
#[ignore = "heavy: 10,000 qubits; run with `cargo test --release --test scale -- --ignored --nocapture`"]
fn heavy_hundred_by_hundred_scattered_recalibration() {
    let (graph, goal) = lattice(100);
    let model = GateFrequencyModel::new(&graph, 5);
    let snake = Snake::new(&graph, goal.clone(), config(None));
    let mut db = snake.calibrate(&model).unwrap();
    let before = db.clone();
    // 400 drifted qubits on a grid with spacing 5.
    let drifted: Vec<ElementId> = (0..20)
        .flat_map(|i| (0..20).map(move |j| (2 + 5 * i, 2 + 5 * j)))
        .map(|(r, c)| graph.lattice_node(r, c).unwrap())
        .collect();
    let started = Instant::now();
    let report = snake.recalibrate(&mut db, &drifted, &model).unwrap();
    println!(
        "100x100 recalibration of {} scattered qubits: {:?}, {} steps, {} pass(es)",
        drifted.len(),
        started.elapsed(),
        report.steps,
        report.passes
    );
    assert_eq!(db.calibrated(), &goal);
    assert_eq!(report.steps, drifted.len() as u64);
    // A full calibration is 10,000 steps; this touched 400 of them.
    assert!(db.step_count() - before.step_count() == drifted.len() as u64);
}
