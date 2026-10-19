//! Benchmarks of the engine on square lattices from 10x10 to 100x100 qubits (ARCHITECTURE
//! sections 12 and 13), with the cheap `SeparableModel` so that the engine itself is measured
//! and not a physics model.
//!
//! * `calibrate`: a whole calibration from scratch (nodes and engineered edges, the paper's XEB
//!   distances `d_P = 1`, `d_R = 2`, `d_T = 2`). Criterion reports the time per run and, because
//!   the throughput is set to the number of steps, the time per step as well. The neighbourhood
//!   index is built before the measurement (see `neighbourhoods`).
//! * `engine_overhead`: the same runs, but the reported time is the wall time minus the time
//!   spent inside the error model (`Report::error_time` + `Report::prepare_time`), so the time
//!   per step is the engine's own work: scopes, traversal, de-duplication and the database.
//!   The architecture promises that this does not grow with the lattice.
//! * `neighbourhoods`: `Neighbourhoods::build` at depth 2, per element.
//! * `segment`: `faulttune::segment::segment` of a fully uncalibrated 100x100 goal, with
//!   everything active at once and with the XEB subgraphs.
//! * `stitched`: a stitched 100x100 calibration (the lattice cut into 2x2 and 4x4 regions) with
//!   1, 2, 4 and 8 worker threads. The result is checked to be the same for every thread count.
//!
//! Run everything with `cargo bench -p faulttune --bench engine`, or a quick pass with
//! `cargo bench -p faulttune --bench engine -- --warm-up-time 1 --measurement-time 2 --sample-size 10`.
//! Filter with a group name, for example `... --bench engine -- engine_overhead`.

use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use faulttune::segment::segment;
use faulttune::toy::SeparableModel;
use faulttune::{
    ActivityMasks, AlgorithmSubgraphs, Distances, EdgeType, ElementKind, ElementSet,
    Neighbourhoods, ParameterDatabase, ProcessorGraph, Scope, Snake, SnakeConfig,
};

/// Lattice sizes of the benchmarks (rows = columns).
const SIZES: [usize; 4] = [10, 20, 50, 100];

/// Options per element in the `calibrate` and `engine_overhead` groups: few, so that the model
/// is cheap and the engine's own work stays visible.
const CHEAP_OPTIONS: usize = 16;

/// Options per element in the `stitched` group: the paper's `k` of about 100, so that a step
/// has real work for the worker threads.
const PAPER_OPTIONS: usize = 100;

/// The distances of the paper's XEB example, which are also the default of `SnakeConfig`.
const DISTANCES: Distances = Distances {
    parameter: 1,
    constraint: 2,
    traversal: 2,
};

/// A lattice, its goal (nodes and engineered edges) and a model for it.
struct Fixture {
    graph: ProcessorGraph,
    goal: ElementSet,
    model: SeparableModel,
}

impl Fixture {
    fn new(size: usize, options: usize) -> Self {
        let graph = ProcessorGraph::square_lattice(size, size);
        let goal = graph.goal(|element| element.kind() != ElementKind::Edge(EdgeType::Parasitic));
        let model = SeparableModel::new(&graph, options, 5.0, 7.0, 1);
        Self { graph, goal, model }
    }

    /// A Snake with the default (XEB) distances, one worker thread unless told otherwise.
    fn snake(&self, threads: usize) -> Snake<'_> {
        let config = SnakeConfig {
            threads: Some(threads),
            ..SnakeConfig::default()
        };
        Snake::new(&self.graph, self.goal.clone(), config)
    }
}

/// Calibrates once to build the neighbourhood index inside `snake` (it is kept for later
/// calls) and to learn the number of steps of a run.
fn warm_up(snake: &Snake<'_>, model: &SeparableModel) -> u64 {
    let (db, report) = snake
        .calibrate_with_report(model)
        .expect("calibrating a lattice with the separable model cannot fail");
    assert_eq!(db.len(), snake.goal().len(), "the goal must be calibrated");
    report.steps
}

/// Whole calibrations from scratch.
fn calibrate(c: &mut Criterion) {
    let mut group = c.benchmark_group("calibrate");
    for size in SIZES {
        let fixture = Fixture::new(size, CHEAP_OPTIONS);
        let snake = fixture.snake(1);
        let steps = warm_up(&snake, &fixture.model);
        group.throughput(Throughput::Elements(steps));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{size}x{size}")),
            &size,
            |b, _| {
                b.iter_with_large_drop(|| {
                    snake
                        .calibrate(black_box(&fixture.model))
                        .expect("calibration cannot fail")
                })
            },
        );
    }
    group.finish();
}

/// The same calibrations, timed without the time inside the error model.
fn engine_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("engine_overhead");
    for size in SIZES {
        let fixture = Fixture::new(size, CHEAP_OPTIONS);
        let snake = fixture.snake(1);
        let steps = warm_up(&snake, &fixture.model);
        group.throughput(Throughput::Elements(steps));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{size}x{size}")),
            &size,
            |b, _| {
                b.iter_custom(|iterations| {
                    let mut engine_time = Duration::ZERO;
                    for _ in 0..iterations {
                        let (db, report) = snake
                            .calibrate_with_report(black_box(&fixture.model))
                            .expect("calibration cannot fail");
                        engine_time += report
                            .wall_time
                            .saturating_sub(report.error_time + report.prepare_time);
                        drop(black_box(db));
                    }
                    engine_time
                });
            },
        );
    }
    group.finish();
}

/// Building the neighbourhood index at depth 2 (the largest of the default distances).
fn neighbourhoods(c: &mut Criterion) {
    let mut group = c.benchmark_group("neighbourhoods");
    for size in SIZES {
        let graph = ProcessorGraph::square_lattice(size, size);
        group.throughput(Throughput::Elements(graph.len() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{size}x{size}")),
            &size,
            |b, _| b.iter_with_large_drop(|| Neighbourhoods::build(black_box(&graph), 2)),
        );
    }
    group.finish();
}

/// Segmenting a fully uncalibrated 100x100 goal into subgoals and threads.
fn segmentation(c: &mut Criterion) {
    let size = 100;
    let graph = ProcessorGraph::square_lattice(size, size);
    let goal = graph.goal(|element| element.kind() != ElementKind::Edge(EdgeType::Parasitic));
    let index = Neighbourhoods::build(&graph, 2);
    let all_active = AlgorithmSubgraphs::all_active(&graph).masks();
    let xeb = AlgorithmSubgraphs::xeb_square_lattice(&graph, size, size)
        .expect("a square lattice has XEB subgraphs")
        .masks();

    let mut group = c.benchmark_group("segment");
    group.throughput(Throughput::Elements(goal.len() as u64));
    for (name, activity) in [("all_active", &all_active), ("xeb", &xeb)] {
        let scope = scope_of(&graph, &index, activity);
        group.bench_function(
            BenchmarkId::from_parameter(format!("100x100/{name}")),
            |b| b.iter_with_large_drop(|| segment(black_box(&scope), black_box(&goal))),
        );
    }
    group.finish();
}

fn scope_of<'a>(
    graph: &'a ProcessorGraph,
    index: &'a Neighbourhoods,
    activity: &'a ActivityMasks,
) -> Scope<'a> {
    Scope::new(graph, index, activity, DISTANCES)
}

/// The goal elements of a `grid` x `grid` arrangement of regions, by layout coordinates (edges
/// sit at the midpoint of their endpoints).
fn regions(fixture: &Fixture, size: usize, grid: usize) -> Vec<ElementSet> {
    let coords = fixture.graph.coords().expect("lattices have a layout");
    let cell = size as f32 / grid as f32;
    let mut regions = vec![ElementSet::new(fixture.graph.len()); grid * grid];
    for e in &fixture.goal {
        let [x, y] = coords[e.index()];
        let column = ((x / cell) as usize).min(grid - 1);
        let row = ((y / cell) as usize).min(grid - 1);
        regions[row * grid + column].insert(e);
    }
    regions
}

/// A stitched 100x100 calibration with several worker threads.
fn stitched(c: &mut Criterion) {
    let size = 100;
    let fixture = Fixture::new(size, PAPER_OPTIONS);
    let mut group = c.benchmark_group("stitched");
    for grid in [2, 4] {
        let plan = fixture
            .snake(1)
            .stitch_plan(regions(&fixture, size, grid))
            .expect("quadrants of the goal make a valid plan");
        // The reference result of one thread, which every other thread count must reproduce.
        let mut reference = ParameterDatabase::new(&fixture.graph);
        let report = fixture
            .snake(1)
            .stitch(&mut reference, &plan, &fixture.model)
            .expect("stitching cannot fail");
        group.throughput(Throughput::Elements(report.steps));

        for threads in [1, 2, 4, 8] {
            let snake = fixture.snake(threads);
            let mut db = ParameterDatabase::new(&fixture.graph);
            snake
                .stitch(&mut db, &plan, &fixture.model)
                .expect("stitching cannot fail");
            assert!(
                db.same_outcome(&reference),
                "{threads} threads gave a different calibration than 1 thread"
            );
            group.bench_with_input(
                BenchmarkId::new(
                    format!("{grid}x{grid}_regions"),
                    format!("{threads}_threads"),
                ),
                &threads,
                |b, _| {
                    b.iter_with_large_drop(|| {
                        let mut db = ParameterDatabase::new(&fixture.graph);
                        snake
                            .stitch(&mut db, black_box(&plan), black_box(&fixture.model))
                            .expect("stitching cannot fail");
                        db
                    })
                },
            );
        }
    }
    group.finish();
}

criterion_group! {
    name = benches;
    // Modest sample sizes: a full run takes a few minutes. The command line overrides these.
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    targets = calibrate, engine_overhead, neighbourhoods, segmentation, stitched
}
criterion_main!(benches);
