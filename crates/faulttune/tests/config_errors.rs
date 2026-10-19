//! Configuration and input errors (ARCHITECTURE section 8): every way `SnakeConfig::validate`
//! and the `Snake::calibrate*`, `recalibrate`, `stitch_plan` and `stitch` methods refuse their
//! input, checked through the public API.
//!
//! The contract these tests pin down:
//!
//! * Bad input is an error value, never a panic, and it is found *before any step runs*: the
//!   model is not called, no event is emitted, and the database is left exactly as it was.
//! * `Snake::new` and the builder methods never fail; the problems surface when a `calibrate_*`
//!   method is called, as `FaultTuneError::InvalidConfig` with a message that names the problem.
//! * A database of another graph is a `GraphMismatch` (with both fingerprints), not an
//!   `InvalidConfig`.
//!
//! The file is self-contained.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use faulttune::engine::{MAX_DISTANCE, MAX_THREADS};
use faulttune::optimizers::{CoordinateDescent, Exhaustive, RandomSearch};
use faulttune::rand::rngs::StdRng;
use faulttune::toy::SeparableModel;
use faulttune::{
    AlgorithmSubgraphs, CachePolicy, CandidateBatch, EdgeType, ElementId, ElementKind, ElementSet,
    ErrorModel, Event, FaultTuneError, IndexBatch, ModelError, Objective, Optimizer,
    ParameterDatabase, ProcessorGraph, RecordingObserver, SearchSpace, SeedPolicy, Snake,
    SnakeConfig, Solution, Step, StitchPlan,
};

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

fn lattice(rows: usize, cols: usize) -> (ProcessorGraph, ElementSet) {
    let graph = ProcessorGraph::square_lattice(rows, cols);
    let goal = graph.goal_excluding(&[ElementKind::Edge(EdgeType::Parasitic)]);
    (graph, goal)
}

fn node(graph: &ProcessorGraph, r: usize, c: usize) -> ElementId {
    graph.lattice_node(r, c).expect("a node of the lattice")
}

/// A model that counts every call, to prove that the model was not asked for anything.
struct Counting {
    inner: SeparableModel,
    calls: AtomicUsize,
}

impl Counting {
    fn new(graph: &ProcessorGraph) -> Self {
        Self {
            inner: SeparableModel::new(graph, 10, 5.0, 7.0, 1),
            calls: AtomicUsize::new(0),
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl ErrorModel for Counting {
    type Value = f64;
    type Data = ();

    fn options(&self, element: ElementId) -> Vec<f64> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.options(element)
    }

    fn prepare(&self, step: &Step<'_, f64>) -> Result<(), ModelError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.prepare(step)
    }

    fn error(&self, step: &Step<'_, f64>, data: &(), candidate: &[f64]) -> f64 {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.error(step, data, candidate)
    }

    fn error_batch(
        &self,
        step: &Step<'_, f64>,
        data: &(),
        candidates: &CandidateBatch<'_, f64>,
        out: &mut [f64],
    ) -> Result<(), ModelError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.error_batch(step, data, candidates, out)
    }
}

/// The message of an `InvalidConfig`, or a test failure.
fn invalid_message<T: std::fmt::Debug>(result: Result<T, FaultTuneError>) -> String {
    match result {
        Err(FaultTuneError::InvalidConfig(message)) => message,
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}

/// Asserts an `InvalidConfig` whose message contains every needle.
#[track_caller]
fn assert_invalid<T: std::fmt::Debug>(result: Result<T, FaultTuneError>, needles: &[&str]) {
    let message = invalid_message(result);
    for needle in needles {
        assert!(
            message.contains(needle),
            "the message {message:?} should mention {needle:?}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// SnakeConfig::validate
// ---------------------------------------------------------------------------------------------

#[test]
fn validate_accepts_the_defaults_and_the_documented_extremes() {
    SnakeConfig::default().validate().unwrap();
    let ok = |cfg: SnakeConfig| cfg.validate().unwrap();
    ok(SnakeConfig {
        threads: Some(1),
        ..SnakeConfig::default()
    });
    ok(SnakeConfig {
        threads: Some(MAX_THREADS),
        ..SnakeConfig::default()
    });
    ok(SnakeConfig {
        evaluation_budget: 1,
        ..SnakeConfig::default()
    });
    ok(SnakeConfig {
        cache: CachePolicy::AcrossSteps { capacity: 1 },
        ..SnakeConfig::default()
    });
    ok(SnakeConfig {
        parameter_distance: 0,
        constraint_distance: 0,
        traversal_distance: 0,
        expiry_radius: 0,
        ..SnakeConfig::default()
    });
    ok(SnakeConfig {
        parameter_distance: MAX_DISTANCE,
        constraint_distance: MAX_DISTANCE,
        traversal_distance: MAX_DISTANCE,
        expiry_radius: MAX_DISTANCE,
        ..SnakeConfig::default()
    });
    // The documented XEB setting is the default.
    let xeb = SnakeConfig::default();
    assert_eq!(
        (
            xeb.parameter_distance,
            xeb.constraint_distance,
            xeb.traversal_distance,
            xeb.expiry_radius
        ),
        (1, 2, 2, 1)
    );
    assert_eq!(xeb.threads, None);
    assert_eq!(xeb.on_model_error, faulttune::ModelErrorPolicy::Abort);
}

#[test]
fn validate_rejects_each_bad_field_and_names_it() {
    let default = SnakeConfig::default;
    let cases: Vec<(&str, SnakeConfig)> = vec![
        (
            "threads",
            SnakeConfig {
                threads: Some(0),
                ..default()
            },
        ),
        (
            "threads",
            SnakeConfig {
                threads: Some(MAX_THREADS + 1),
                ..default()
            },
        ),
        (
            "threads",
            SnakeConfig {
                threads: Some(usize::MAX),
                ..default()
            },
        ),
        (
            "evaluation_budget",
            SnakeConfig {
                evaluation_budget: 0,
                ..default()
            },
        ),
        (
            "capacity",
            SnakeConfig {
                cache: CachePolicy::AcrossSteps { capacity: 0 },
                ..default()
            },
        ),
        (
            "parameter_distance",
            SnakeConfig {
                parameter_distance: MAX_DISTANCE + 1,
                ..default()
            },
        ),
        (
            "constraint_distance",
            SnakeConfig {
                constraint_distance: MAX_DISTANCE + 1,
                ..default()
            },
        ),
        (
            "traversal_distance",
            SnakeConfig {
                traversal_distance: MAX_DISTANCE + 1,
                ..default()
            },
        ),
        (
            "expiry_radius",
            SnakeConfig {
                expiry_radius: MAX_DISTANCE + 1,
                ..default()
            },
        ),
        (
            "parameter_distance",
            SnakeConfig {
                parameter_distance: u32::MAX,
                ..default()
            },
        ),
        (
            "traversal_distance",
            SnakeConfig {
                traversal_distance: 1000,
                ..default()
            },
        ),
    ];
    for (needle, config) in cases {
        let err = config.validate().unwrap_err();
        let FaultTuneError::InvalidConfig(message) = &err else {
            panic!("{config:?}: expected InvalidConfig, got {err:?}");
        };
        assert!(
            message.contains(needle),
            "{config:?}: {message:?} should mention {needle:?}"
        );
        // The Display form says what kind of error it is.
        assert!(
            err.to_string().starts_with("invalid configuration"),
            "{err}"
        );
    }
}

#[test]
fn distances_report_the_offending_value_and_the_limit() {
    let message = invalid_message(
        SnakeConfig {
            constraint_distance: 99,
            ..SnakeConfig::default()
        }
        .validate(),
    );
    assert!(
        message.contains("99") && message.contains(&MAX_DISTANCE.to_string()),
        "{message}"
    );
}

// ---------------------------------------------------------------------------------------------
// Errors are found before anything runs
// ---------------------------------------------------------------------------------------------

#[test]
fn construction_never_fails_and_every_entry_point_reports_a_bad_configuration() {
    let (graph, goal) = lattice(4, 4);
    let model = Counting::new(&graph);
    let events = RecordingObserver::new();

    // A valid Snake first, to get a plan and a database with content.
    let good = Snake::new(&graph, goal.clone(), SnakeConfig::default());
    let plan = good.stitch_plan(vec![goal.clone()]).unwrap();
    let mut db = good.calibrate(&model).unwrap();
    let before = db.clone();
    let calls_before = model.calls();

    for config in [
        SnakeConfig {
            threads: Some(0),
            ..SnakeConfig::default()
        },
        SnakeConfig {
            evaluation_budget: 0,
            ..SnakeConfig::default()
        },
        SnakeConfig {
            traversal_distance: MAX_DISTANCE + 1,
            ..SnakeConfig::default()
        },
    ] {
        // Building never fails ...
        let snake = Snake::new(&graph, goal.clone(), config).observer(events.clone());
        // ... and every method that would run refuses with an InvalidConfig.
        assert_invalid(snake.calibrate(&model), &[]);
        assert_invalid(snake.calibrate_with_report(&model), &[]);
        assert_invalid(snake.calibrate_into(&mut db, &model), &[]);
        assert_invalid(snake.calibrate_region(&mut db, &goal, &model), &[]);
        assert_invalid(
            snake.recalibrate(&mut db, &[node(&graph, 1, 1)], &model),
            &[],
        );
        assert_invalid(snake.stitch_plan(vec![goal.clone()]), &[]);
        assert_invalid(snake.stitch(&mut db, &plan, &model), &[]);
    }
    // Nothing happened: the database is as it was (no invalidation either), the model was not
    // called, no event was emitted.
    assert!(db.same_outcome(&before));
    assert_eq!(db.history(), before.history());
    assert_eq!(model.calls(), calls_before);
    assert!(events.events().is_empty(), "{:?}", events.events());
}

// ---------------------------------------------------------------------------------------------
// Goal, algorithm subgraphs, region
// ---------------------------------------------------------------------------------------------

#[test]
fn a_goal_over_another_number_of_elements_is_refused() {
    let (graph, goal) = lattice(4, 4);
    let model = Counting::new(&graph);
    let bigger = ProcessorGraph::square_lattice(5, 5);
    let smaller = ProcessorGraph::square_lattice(2, 2);
    for wrong in [
        bigger.all_elements(),
        smaller.all_elements(),
        ElementSet::new(0),
    ] {
        let capacity = wrong.capacity();
        let events = RecordingObserver::new();
        let snake = Snake::new(&graph, wrong, SnakeConfig::default()).observer(events.clone());
        let message = invalid_message(snake.calibrate(&model));
        assert!(message.contains("goal"), "{message}");
        assert!(
            message.contains(&capacity.to_string()) && message.contains(&graph.len().to_string()),
            "the message should give both sizes: {message}"
        );
        let mut db = ParameterDatabase::new(&graph);
        assert_invalid(snake.calibrate_into(&mut db, &model), &["goal"]);
        assert_invalid(snake.calibrate_region(&mut db, &goal, &model), &["goal"]);
        assert_invalid(snake.stitch_plan(vec![]), &["goal"]);
        assert!(db.is_empty() && db.history().is_empty());
        assert!(events.events().is_empty());
    }
    assert_eq!(model.calls(), 0);
}

#[test]
fn an_empty_goal_or_an_empty_graph_is_a_valid_no_op() {
    let (graph, _) = lattice(3, 3);
    let model = Counting::new(&graph);
    let events = RecordingObserver::new();
    let snake = Snake::new(&graph, ElementSet::new(graph.len()), SnakeConfig::default())
        .observer(events.clone());
    let (db, report) = snake.calibrate_with_report(&model).unwrap();
    assert!(db.is_empty());
    assert_eq!((report.steps, report.passes, report.evaluations), (0, 0, 0));
    assert!(matches!(events.events().as_slice(), [Event::Finished(r)] if r.steps == 0));
    assert_eq!(model.calls(), 0);

    let empty = ProcessorGraph::square_lattice(0, 0);
    assert!(empty.is_empty());
    let snake = Snake::new(&empty, empty.all_elements(), SnakeConfig::default());
    let model = SeparableModel::new(&empty, 3, 0.0, 1.0, 0);
    let db = snake.calibrate(&model).unwrap();
    assert!(db.is_empty());
}

#[test]
fn algorithm_subgraphs_over_another_graph_are_refused() {
    let (graph, goal) = lattice(4, 4);
    let model = Counting::new(&graph);
    let other = ProcessorGraph::square_lattice(3, 3);
    // The sets have the capacity of the other graph (and are the right kind of thing otherwise).
    let wrong = AlgorithmSubgraphs::all_active(&other);
    let snake = Snake::new(&graph, goal.clone(), SnakeConfig::default()).algorithm(wrong);
    assert_invalid(snake.calibrate(&model), &["algorithm subgraphs"]);
    let mut db = ParameterDatabase::new(&graph);
    assert_invalid(
        snake.calibrate_into(&mut db, &model),
        &["algorithm subgraphs"],
    );
    assert_invalid(
        snake.recalibrate(&mut db, &[node(&graph, 0, 0)], &model),
        &["algorithm"],
    );
    assert_invalid(snake.stitch_plan(vec![goal.clone()]), &["algorithm"]);
    assert_eq!(model.calls(), 0);

    // Replacing the subgraphs by the right ones repairs the same configuration.
    let snake = Snake::new(&graph, goal, SnakeConfig::default())
        .algorithm(AlgorithmSubgraphs::all_active(&other))
        .algorithm(AlgorithmSubgraphs::all_active(&graph));
    snake.calibrate(&model).unwrap();
}

#[test]
fn building_algorithm_subgraphs_checks_its_arguments() {
    let (graph, _) = lattice(3, 3);
    let n = graph.len();
    assert_invalid(AlgorithmSubgraphs::new(vec![]), &["at least one"]);
    assert_invalid(
        AlgorithmSubgraphs::new(vec![ElementSet::new(n); 65]),
        &["65", "64"],
    );
    assert_invalid(
        AlgorithmSubgraphs::new(vec![ElementSet::new(n), ElementSet::new(n + 1)]),
        &["capacity"],
    );
    // 64 subgraphs are allowed.
    AlgorithmSubgraphs::new(vec![graph.all_elements(); 64]).unwrap();
}

#[test]
fn a_goal_element_that_is_never_active_is_refused_by_name() {
    let (graph, goal) = lattice(4, 4);
    let model = Counting::new(&graph);

    // The subgraph covers the goal except for one engineered edge.
    let missing = graph.edges().find(|&e| goal.contains(e)).unwrap();
    let mut almost = goal.clone();
    almost.remove(missing);
    let snake = Snake::new(&graph, goal.clone(), SnakeConfig::default())
        .algorithm(AlgorithmSubgraphs::new(vec![almost]).unwrap());
    let label = graph.label(missing).to_owned();
    assert_invalid(snake.calibrate(&model), &[&label, "not active"]);

    // Nothing is active at all.
    let nobody = AlgorithmSubgraphs::new(vec![ElementSet::new(graph.len())]).unwrap();
    let snake = Snake::new(&graph, goal.clone(), SnakeConfig::default()).algorithm(nobody);
    assert_invalid(snake.calibrate(&model), &["not active"]);
    assert_eq!(model.calls(), 0);

    // The XEB subgraphs leave the parasitic edges inactive: they cannot be in the goal.
    let xeb = AlgorithmSubgraphs::xeb_square_lattice(&graph, 4, 4).unwrap();
    let with_parasitic = graph.all_elements();
    let snake = Snake::new(&graph, with_parasitic, SnakeConfig::default()).algorithm(xeb.clone());
    assert_invalid(snake.calibrate(&model), &["not active"]);
    // Without them it is fine.
    Snake::new(&graph, goal, SnakeConfig::default())
        .algorithm(xeb)
        .calibrate(&model)
        .unwrap();
}

#[test]
fn a_region_of_another_size_is_refused_and_a_partial_region_is_honoured() {
    let (graph, goal) = lattice(5, 5);
    let model = Counting::new(&graph);
    let snake = Snake::new(&graph, goal.clone(), SnakeConfig::default());
    let mut db = ParameterDatabase::new(&graph);

    let wrong = ProcessorGraph::square_lattice(2, 2).all_elements();
    assert_invalid(snake.calibrate_region(&mut db, &wrong, &model), &["region"]);
    assert_invalid(
        snake.calibrate_region(&mut db, &ElementSet::new(0), &model),
        &["region"],
    );
    assert!(db.is_empty() && model.calls() == 0);

    // An empty region is a no-op.
    let report = snake
        .calibrate_region(&mut db, &ElementSet::new(graph.len()), &model)
        .unwrap();
    assert_eq!((report.steps, report.passes), (0, 0));
    assert!(db.is_empty());

    // A region may reach outside the goal: only the goal elements in it are calibrated, and
    // nothing else is touched. Here: the upper-left 3x3 nodes with all their edges.
    let mut region = ElementSet::new(graph.len());
    for r in 0..3 {
        for c in 0..3 {
            let n = node(&graph, r, c);
            region.insert(n);
            for &e in graph.incident(n) {
                region.insert(e);
            }
        }
    }
    assert!(
        !region.is_subset(&goal),
        "the region contains parasitic edges"
    );
    snake.calibrate_region(&mut db, &region, &model).unwrap();
    assert_eq!(db.calibrated(), &region.intersection(&goal));

    // The rest is done by the next call.
    snake.calibrate_into(&mut db, &model).unwrap();
    assert_eq!(db.calibrated(), &goal);
}

// ---------------------------------------------------------------------------------------------
// The database
// ---------------------------------------------------------------------------------------------

#[test]
fn a_database_of_another_graph_is_a_graph_mismatch() {
    // Two different graphs with the same number of elements (3x4 and 4x3 lattices) and one with
    // a different number.
    let (graph, goal) = lattice(3, 4);
    let model = Counting::new(&graph);
    let swapped = ProcessorGraph::square_lattice(4, 3);
    assert_eq!(swapped.len(), graph.len());
    assert_ne!(swapped.fingerprint(), graph.fingerprint());
    let smaller = ProcessorGraph::square_lattice(2, 2);

    let snake = Snake::new(&graph, goal.clone(), SnakeConfig::default());
    let mut own = snake.calibrate(&model).unwrap();
    let own_before = own.clone();
    let calls = model.calls();

    for foreign_graph in [&swapped, &smaller] {
        let mut foreign = ParameterDatabase::<f64>::new(foreign_graph);
        let err = snake.calibrate_into(&mut foreign, &model).unwrap_err();
        match err {
            FaultTuneError::GraphMismatch { expected, found } => {
                assert_eq!(expected, graph.fingerprint());
                assert_eq!(found, foreign_graph.fingerprint());
            }
            other => panic!("expected GraphMismatch, got {other:?}"),
        }
        assert!(foreign.is_empty());
        assert!(matches!(
            snake.calibrate_region(&mut foreign, &goal, &model),
            Err(FaultTuneError::GraphMismatch { .. })
        ));
        assert!(matches!(
            snake.recalibrate(&mut foreign, &[node(&graph, 1, 1)], &model),
            Err(FaultTuneError::GraphMismatch { .. })
        ));
        assert!(
            foreign.history().is_empty(),
            "nothing was invalidated or committed"
        );
        let plan = snake.stitch_plan(vec![goal.clone()]).unwrap();
        assert!(matches!(
            snake.stitch(&mut foreign, &plan, &model),
            Err(FaultTuneError::GraphMismatch { .. })
        ));
    }
    // The model was never called, and a database of the right graph is untouched by all this.
    assert_eq!(model.calls(), calls);
    assert!(own.same_outcome(&own_before));
    snake.calibrate_into(&mut own, &model).unwrap();
    assert!(
        own.same_outcome(&own_before),
        "a complete database is a no-op"
    );
}

#[cfg(feature = "serde")]
#[test]
fn a_saved_calibration_does_not_load_onto_another_device() {
    let (graph, goal) = lattice(3, 3);
    let model = Counting::new(&graph);
    let db = Snake::new(&graph, goal, SnakeConfig::default())
        .calibrate(&model)
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("calibration.json");
    db.save(&graph, &path).unwrap();
    ParameterDatabase::<f64>::load(&graph, &path).unwrap();

    let other = ProcessorGraph::square_lattice(3, 4);
    let err = ParameterDatabase::<f64>::load(&other, &path).unwrap_err();
    assert!(
        matches!(err, FaultTuneError::GraphMismatch { .. }),
        "{err:?}"
    );
    // Saving a database with a graph it does not belong to is refused as well.
    let err = db.save(&other, dir.path().join("wrong.json")).unwrap_err();
    assert!(
        matches!(err, FaultTuneError::GraphMismatch { .. }),
        "{err:?}"
    );
    // A missing file is an I/O error, a file that is not a calibration a format error.
    let err = ParameterDatabase::<f64>::load(&graph, dir.path().join("missing.json")).unwrap_err();
    assert!(matches!(err, FaultTuneError::Io(_)), "{err:?}");
    std::fs::write(dir.path().join("junk.json"), "not json").unwrap();
    let err = ParameterDatabase::<f64>::load(&graph, dir.path().join("junk.json")).unwrap_err();
    assert!(matches!(err, FaultTuneError::Format(_)), "{err:?}");
}

// ---------------------------------------------------------------------------------------------
// recalibrate
// ---------------------------------------------------------------------------------------------

#[test]
fn recalibrate_refuses_elements_outside_the_goal_or_the_graph_and_changes_nothing() {
    let (graph, goal) = lattice(5, 5);
    let model = Counting::new(&graph);
    let events = RecordingObserver::new();
    let snake = Snake::new(&graph, goal.clone(), SnakeConfig::default());
    let mut db = snake.calibrate(&model).unwrap();
    let snake = Snake::new(&graph, goal.clone(), SnakeConfig::default()).observer(events.clone());
    let before = db.clone();
    let calls = model.calls();

    let parasitic = graph.edges().find(|&e| !goal.contains(e)).unwrap();
    let valid = node(&graph, 2, 2);
    let beyond = ElementId::new(graph.len() as u32);
    let cases: Vec<(Vec<ElementId>, &str)> = vec![
        (vec![parasitic], "goal"),
        // A valid element first does not make the call partial: nothing is forgotten.
        (vec![valid, parasitic], "goal"),
        (vec![valid, beyond], "not in the graph"),
        (vec![beyond], "not in the graph"),
        (vec![ElementId::new(u32::MAX)], "not in the graph"),
    ];
    for (elements, needle) in cases {
        let message = invalid_message(snake.recalibrate(&mut db, &elements, &model));
        assert!(message.contains(needle), "{elements:?}: {message}");
        assert!(db.same_outcome(&before), "{elements:?}");
        assert_eq!(
            db.history(),
            before.history(),
            "{elements:?}: not even an Invalidated record"
        );
    }
    assert_eq!(db.calibrated(), &goal);
    assert_eq!(model.calls(), calls);
    assert!(
        events.events().is_empty(),
        "no Invalidated or Finished event for refused input"
    );

    // The label of the offending element is in the message.
    let message = invalid_message(snake.recalibrate(&mut db, &[parasitic], &model));
    assert!(message.contains(graph.label(parasitic)), "{message}");
}

#[test]
fn recalibrate_with_no_elements_or_an_empty_database_is_well_defined() {
    let (graph, goal) = lattice(4, 4);
    let model = Counting::new(&graph);
    let snake = Snake::new(&graph, goal.clone(), SnakeConfig::default());

    // No elements: nothing is forgotten and nothing runs.
    let mut db = snake.calibrate(&model).unwrap();
    let before = db.clone();
    let report = snake.recalibrate(&mut db, &[], &model).unwrap();
    assert_eq!((report.steps, report.passes), (0, 0));
    assert!(db.same_outcome(&before));
    assert_eq!(db.history(), before.history());

    // A database with nothing calibrated: there is nothing to forget, and the region around the
    // element is calibrated (and no more).
    let mut empty = ParameterDatabase::<f64>::new(&graph);
    let centre = node(&graph, 1, 1);
    snake.recalibrate(&mut empty, &[centre], &model).unwrap();
    let mut expected = ElementSet::new(graph.len());
    expected.insert(centre);
    for &e in graph.incident(centre) {
        if goal.contains(e) {
            expected.insert(e);
        }
    }
    assert_eq!(empty.calibrated(), &expected);
    assert!(empty
        .history()
        .iter()
        .all(|h| matches!(h, faulttune::HistoryRecord::Step(_))));

    // The expiry radius widens the region: radius 0 forgets the element alone.
    let mut db = before.clone();
    let cfg = SnakeConfig {
        expiry_radius: 0,
        ..SnakeConfig::default()
    };
    let snake0 = Snake::new(&graph, goal.clone(), cfg);
    let steps_before = db.step_count();
    snake0.recalibrate(&mut db, &[centre], &model).unwrap();
    assert_eq!(db.calibrated(), &goal);
    let changed: Vec<ElementId> = goal
        .iter()
        .filter(|&e| db.get(e).unwrap().step >= steps_before)
        .collect();
    assert_eq!(changed, vec![centre]);

    // A radius above the limit is a configuration error, found before anything is forgotten.
    let cfg = SnakeConfig {
        expiry_radius: MAX_DISTANCE + 1,
        ..SnakeConfig::default()
    };
    let bad = Snake::new(&graph, goal, cfg);
    let mut db = before.clone();
    assert_invalid(
        bad.recalibrate(&mut db, &[centre], &model),
        &["expiry_radius"],
    );
    assert!(db.same_outcome(&before));
}

// ---------------------------------------------------------------------------------------------
// Stitch plans
// ---------------------------------------------------------------------------------------------

fn halves(graph: &ProcessorGraph, goal: &ElementSet) -> (ElementSet, ElementSet) {
    // Top and bottom half of a 6x6 lattice by the row of the node (or the upper end point).
    let mut top = ElementSet::new(graph.len());
    let mut bottom = ElementSet::new(graph.len());
    for id in goal.iter() {
        let n = graph.endpoints(id).map_or(id, |[a, _]| a);
        let row: usize = graph.label(n)[1..]
            .split('_')
            .next()
            .unwrap()
            .parse()
            .unwrap();
        if row < 3 {
            top.insert(id)
        } else {
            bottom.insert(id)
        };
    }
    (top, bottom)
}

#[test]
fn stitch_plans_refuse_overlapping_regions_and_elements_outside_the_goal() {
    let (graph, goal) = lattice(6, 6);
    let model = Counting::new(&graph);
    let snake = Snake::new(&graph, goal.clone(), SnakeConfig::default());
    let (top, bottom) = halves(&graph, &goal);

    // Overlapping regions.
    let mut overlapping = bottom.clone();
    overlapping.insert(top.first().unwrap());
    assert_invalid(
        snake.stitch_plan(vec![top.clone(), overlapping]),
        &["overlap"],
    );
    // Elements outside the goal (a parasitic edge).
    let parasitic = graph.edges().find(|&e| !goal.contains(e)).unwrap();
    let mut outside = bottom.clone();
    outside.insert(parasitic);
    assert_invalid(
        snake.stitch_plan(vec![top.clone(), outside]),
        &["outside the goal"],
    );
    // A region over another number of elements.
    let small = ProcessorGraph::square_lattice(2, 2).all_elements();
    assert_invalid(snake.stitch_plan(vec![top.clone(), small]), &["capacity"]);
    assert_eq!(model.calls(), 0);

    // Good plans: the same region twice is an overlap, no regions at all is one big gap, and a
    // single region covering the goal is one big stage one.
    assert_invalid(
        snake.stitch_plan(vec![top.clone(), top.clone()]),
        &["overlap"],
    );
    let plan: StitchPlan = snake.stitch_plan(vec![]).unwrap();
    assert_eq!((plan.stage_one.len(), plan.gap.len()), (0, goal.len()));
    let plan = snake.stitch_plan(vec![goal.clone()]).unwrap();
    assert_eq!((plan.stage_one.len(), plan.gap.len()), (goal.len(), 0));
    let plan = snake
        .stitch_plan(vec![top.clone(), bottom.clone()])
        .unwrap();
    assert!(plan.stage_one.is_disjoint(&plan.gap));
    assert_eq!(plan.stage_one.union(&plan.gap), goal);
    assert!(!plan.gap.is_empty());

    // All three kinds of plan calibrate the whole goal when run.
    for regions in [vec![], vec![goal.clone()], vec![top, bottom]] {
        let plan = snake.stitch_plan(regions).unwrap();
        let mut db = ParameterDatabase::new(&graph);
        snake.stitch(&mut db, &plan, &model).unwrap();
        assert_eq!(db.calibrated(), &goal);
    }
}

#[test]
fn a_plan_of_another_graph_is_refused_by_stitch() {
    let (graph, goal) = lattice(4, 4);
    let model = Counting::new(&graph);
    let snake = Snake::new(&graph, goal, SnakeConfig::default());
    let (other_graph, other_goal) = lattice(3, 3);
    let other_snake = Snake::new(&other_graph, other_goal.clone(), SnakeConfig::default());
    let plan = other_snake.stitch_plan(vec![other_goal]).unwrap();
    let mut db = ParameterDatabase::new(&graph);
    assert_invalid(snake.stitch(&mut db, &plan, &model), &["stitch plan"]);
    assert!(db.is_empty() && model.calls() == 0);
}

// ---------------------------------------------------------------------------------------------
// Errors that come from the model or the optimizer contract
// ---------------------------------------------------------------------------------------------

/// A model that violates the contract in one way.
struct NoOptions {
    empty_for: ElementId,
    inner: SeparableModel,
}

impl ErrorModel for NoOptions {
    type Value = f64;
    type Data = ();

    fn options(&self, element: ElementId) -> Vec<f64> {
        if element == self.empty_for {
            Vec::new()
        } else {
            self.inner.options(element)
        }
    }

    fn prepare(&self, _step: &Step<'_, f64>) -> Result<(), ModelError> {
        Ok(())
    }

    fn error(&self, step: &Step<'_, f64>, data: &(), candidate: &[f64]) -> f64 {
        self.inner.error(step, data, candidate)
    }
}

#[test]
fn a_model_without_options_for_an_element_is_a_configuration_error() {
    let (graph, goal) = lattice(4, 4);
    let victim = node(&graph, 2, 2);
    let model = NoOptions {
        empty_for: victim,
        inner: SeparableModel::new(&graph, 10, 5.0, 7.0, 1),
    };
    let snake = Snake::new(&graph, goal, SnakeConfig::default());
    let mut db = ParameterDatabase::new(&graph);
    let message = invalid_message(snake.calibrate_into(&mut db, &model));
    assert!(
        message.contains("options") && message.contains(graph.label(victim)),
        "{message}"
    );
    // The steps before it are kept as always.
    assert!(!db.is_empty() && !db.is_calibrated(victim));
}

/// An optimizer scripted by a closure.
type Script =
    dyn Fn(&SearchSpace, &mut dyn Objective) -> Result<Solution, FaultTuneError> + Send + Sync;

struct Scripted(Box<Script>);

impl Optimizer for Scripted {
    fn optimize(
        &self,
        space: &SearchSpace,
        objective: &mut dyn Objective,
        _rng: &mut StdRng,
    ) -> Result<Solution, FaultTuneError> {
        (self.0)(space, objective)
    }
}

fn scripted(
    f: impl Fn(&SearchSpace, &mut dyn Objective) -> Result<Solution, FaultTuneError>
        + Send
        + Sync
        + 'static,
) -> Scripted {
    Scripted(Box::new(f))
}

#[test]
fn an_optimizer_that_breaks_its_contract_is_a_configuration_error_not_a_panic() {
    let (graph, goal) = lattice(4, 4);
    let model = SeparableModel::new(&graph, 10, 5.0, 7.0, 1);
    let run = |optimizer: Scripted| {
        let snake = Snake::new(&graph, goal.clone(), SnakeConfig::default()).optimizer(optimizer);
        let mut db = ParameterDatabase::new(&graph);
        let result = snake.calibrate_into(&mut db, &model);
        assert!(db.is_empty(), "nothing was committed from a bad solution");
        result
    };

    // A solution of the wrong width.
    assert_invalid(
        run(scripted(|_, _| {
            Ok(Solution {
                indices: vec![],
                error: 0.0,
                evaluations: 0,
            })
        })),
        &["does not fit"],
    );
    assert_invalid(
        run(scripted(|space, _| {
            Ok(Solution {
                indices: vec![0; space.width() + 1],
                error: 0.0,
                evaluations: 0,
            })
        })),
        &["does not fit"],
    );
    // A solution naming an option that does not exist.
    assert_invalid(
        run(scripted(|space, _| {
            Ok(Solution {
                indices: vec![u32::MAX; space.width()],
                error: 0.0,
                evaluations: 0,
            })
        })),
        &["does not fit"],
    );
    assert_invalid(
        run(scripted(|space, _| {
            let mut indices = vec![0; space.width()];
            indices[0] = space.sizes[0];
            Ok(Solution {
                indices,
                error: 0.0,
                evaluations: 0,
            })
        })),
        &["does not fit"],
    );
    // A candidate with an option index out of range, passed to the objective.
    assert_invalid(
        run(scripted(|space, objective| {
            let row: Vec<u32> = space.sizes.clone();
            let mut out = [0.0];
            objective.evaluate(IndexBatch::new(&row, row.len()), &mut out)?;
            Ok(Solution {
                indices: vec![0; space.width()],
                error: out[0],
                evaluations: 1,
            })
        })),
        &["option"],
    );
    // A batch of the wrong width.
    assert_invalid(
        run(scripted(|space, objective| {
            let row = vec![0u32; space.width() + 1];
            let mut out = [0.0];
            objective.evaluate(IndexBatch::new(&row, row.len()), &mut out)?;
            Ok(Solution {
                indices: vec![0; space.width()],
                error: 0.0,
                evaluations: 1,
            })
        })),
        &["width"],
    );
    // An output buffer of the wrong length.
    assert_invalid(
        run(scripted(|space, objective| {
            let row = vec![0u32; space.width()];
            let mut out = [0.0, 0.0];
            objective.evaluate(IndexBatch::new(&row, row.len()), &mut out)?;
            Ok(Solution {
                indices: vec![0; space.width()],
                error: 0.0,
                evaluations: 1,
            })
        })),
        &["output"],
    );
    // An error of the optimizer itself is passed on as it is.
    let err = run(scripted(|_, _| {
        Err(FaultTuneError::Format("custom".into()))
    }))
    .unwrap_err();
    assert!(matches!(err, FaultTuneError::Format(m) if m == "custom"));
    // A NaN error in the solution is the same fatal condition as a NaN from the model.
    let err = run(scripted(|space, _| {
        Ok(Solution {
            indices: vec![0; space.width()],
            error: f64::NAN,
            evaluations: 1,
        })
    }))
    .unwrap_err();
    assert!(
        matches!(err, FaultTuneError::NonFiniteError { .. }),
        "{err:?}"
    );
}

#[test]
fn exhaustive_search_refuses_a_space_beyond_its_budget() {
    let (graph, goal) = lattice(5, 5);
    // The first step, centred on an interior node, has 5 parameters of 100 options each:
    // 10^10 combinations, far beyond the default budget of 10^6.
    let model = SeparableModel::new(&graph, 100, 5.0, 7.0, 1);
    let snake = Snake::new(&graph, goal.clone(), SnakeConfig::default())
        .seed_policy(SeedPolicy::Fixed(vec![node(&graph, 2, 2)]))
        .optimizer(Exhaustive::default());
    let mut db = ParameterDatabase::new(&graph);
    match snake.calibrate_into(&mut db, &model).unwrap_err() {
        FaultTuneError::SearchSpaceTooLarge { size, budget } => {
            assert!(size > u128::from(budget));
            assert_eq!(budget, faulttune::optimize::DEFAULT_BUDGET);
        }
        other => panic!("expected SearchSpaceTooLarge, got {other:?}"),
    }
    assert!(db.is_empty());

    // The evaluation budget of the configuration is a limit as well; d_P = 0 fits in it.
    let tight = SnakeConfig {
        parameter_distance: 0,
        evaluation_budget: 50,
        ..SnakeConfig::default()
    };
    let snake = Snake::new(&graph, goal.clone(), tight).optimizer(Exhaustive::default());
    match snake.calibrate(&model).unwrap_err() {
        FaultTuneError::SearchSpaceTooLarge { size, budget } => {
            assert_eq!((size, budget), (100, 50))
        }
        other => panic!("expected SearchSpaceTooLarge, got {other:?}"),
    }
    let roomy = SnakeConfig {
        parameter_distance: 0,
        evaluation_budget: 100,
        ..SnakeConfig::default()
    };
    let db = Snake::new(&graph, goal.clone(), roomy)
        .optimizer(Exhaustive::default())
        .calibrate(&model)
        .unwrap();
    assert_eq!(db.calibrated(), &goal);
}

#[test]
fn the_evaluation_budget_limits_the_budget_aware_optimizers() {
    let (graph, goal) = lattice(4, 4);
    let model = SeparableModel::new(&graph, 100, 5.0, 7.0, 1);
    for budget in [1u64, 7, 150] {
        let cfg = SnakeConfig {
            evaluation_budget: budget,
            ..SnakeConfig::default()
        };
        for name in ["coordinate descent", "random search"] {
            let snake = Snake::new(&graph, goal.clone(), cfg.clone());
            let snake = if name == "random search" {
                snake.optimizer(RandomSearch::default())
            } else {
                snake.optimizer(CoordinateDescent::default())
            };
            let (db, report) = snake.calibrate_with_report(&model).unwrap();
            assert_eq!(db.calibrated(), &goal, "{name}, budget {budget}");
            for step in db.steps() {
                assert!(
                    step.evaluations <= budget,
                    "{name}, budget {budget}: a step used {} evaluations",
                    step.evaluations
                );
            }
            assert!(
                report.evaluations <= budget * report.steps,
                "{name}, budget {budget}"
            );
        }
    }
}

#[test]
fn seed_policies_cannot_wedge_a_run() {
    let (graph, goal) = lattice(4, 4);
    let model = SeparableModel::new(&graph, 10, 5.0, 7.0, 1);
    let outside = ElementId::new(graph.len() as u32 + 7);

    // A custom policy that names an element outside the graph: no thread can be started, the
    // pass makes no progress and the run says so.
    let events = RecordingObserver::new();
    let snake = Snake::new(&graph, goal.clone(), SnakeConfig::default())
        .seed_policy(SeedPolicy::Custom(Arc::new(move |_, _| outside)))
        .observer(events.clone());
    let mut db = ParameterDatabase::new(&graph);
    let err = snake.calibrate_into(&mut db, &model).unwrap_err();
    assert!(matches!(err, FaultTuneError::NoProgress), "{err:?}");
    assert!(err.to_string().contains("no progress"));
    assert!(db.is_empty());
    let log = events.events();
    assert!(matches!(
        log.first(),
        Some(Event::PassStarted { pass: 0, .. })
    ));
    assert!(matches!(log.last(), Some(Event::Finished(_))));

    // A custom policy naming an element of another thread (a node for an edge thread) stalls
    // the edge threads only; with d_P = 0 the edges cannot be reached as parameters.
    let cfg = SnakeConfig {
        parameter_distance: 0,
        ..SnakeConfig::default()
    };
    let first = node(&graph, 0, 0);
    let snake = Snake::new(&graph, goal.clone(), cfg)
        .seed_policy(SeedPolicy::Custom(Arc::new(move |_, _| first)));
    let mut db = ParameterDatabase::new(&graph);
    let err = snake.calibrate_into(&mut db, &model).unwrap_err();
    assert!(matches!(err, FaultTuneError::NoProgress), "{err:?}");
    assert_eq!(
        db.len(),
        graph.nodes().count(),
        "the node thread was calibrated, the edges stalled"
    );

    // The built-in policies always make progress.
    for policy in [
        SeedPolicy::LowestId,
        SeedPolicy::Random,
        SeedPolicy::Fixed(vec![]),
        SeedPolicy::Fixed(vec![outside]),
        SeedPolicy::Fixed(vec![node(&graph, 3, 3), outside]),
    ] {
        let snake = Snake::new(&graph, goal.clone(), SnakeConfig::default()).seed_policy(policy);
        assert_eq!(snake.calibrate(&model).unwrap().calibrated(), &goal);
    }
}

#[test]
fn small_distances_are_valid_configurations_that_still_complete() {
    let (graph, goal) = lattice(4, 4);
    let model = SeparableModel::new(&graph, 10, 5.0, 7.0, 1);
    for (d_p, d_r, d_t) in [
        (0, 0, 0),
        (0, 0, 2),
        (1, 0, 2),
        (0, 2, 1),
        (1, 1, 1),
        (2, 3, 3),
    ] {
        let cfg = SnakeConfig {
            parameter_distance: d_p,
            constraint_distance: d_r,
            traversal_distance: d_t,
            ..SnakeConfig::default()
        };
        let db = Snake::new(&graph, goal.clone(), cfg)
            .calibrate(&model)
            .unwrap();
        assert_eq!(
            db.calibrated(),
            &goal,
            "d_P = {d_p}, d_R = {d_r}, d_T = {d_t}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// The error types themselves
// ---------------------------------------------------------------------------------------------

#[test]
fn errors_can_cross_threads_and_chain_to_the_models_own_error() {
    fn assert_send_sync<T: Send + Sync + 'static>() {}
    assert_send_sync::<FaultTuneError>();
    assert_send_sync::<ModelError>();

    #[derive(Debug)]
    struct Offline;
    impl std::fmt::Display for Offline {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("the instrument is offline")
        }
    }
    impl std::error::Error for Offline {}

    struct Broken(SeparableModel);
    impl ErrorModel for Broken {
        type Value = f64;
        type Data = ();
        fn options(&self, e: ElementId) -> Vec<f64> {
            self.0.options(e)
        }
        fn prepare(&self, _: &Step<'_, f64>) -> Result<(), ModelError> {
            Err(ModelError::with_source("measurement failed", Offline))
        }
        fn error(&self, step: &Step<'_, f64>, data: &(), c: &[f64]) -> f64 {
            self.0.error(step, data, c)
        }
    }

    let (graph, goal) = lattice(3, 3);
    let model = Broken(SeparableModel::new(&graph, 5, 5.0, 7.0, 1));
    let err = Snake::new(&graph, goal, SnakeConfig::default())
        .calibrate(&model)
        .unwrap_err();
    let FaultTuneError::Model { element, .. } = &err else {
        panic!("{err:?}")
    };
    // The first step is centred on the lowest node.
    assert_eq!(*element, graph.nodes().next().unwrap());
    assert!(err.to_string().contains("measurement failed"), "{err}");

    // error -> ModelError -> the original error.
    let model_error = std::error::Error::source(&err).expect("a model error has a source");
    assert_eq!(model_error.to_string(), "measurement failed");
    let original = model_error
        .source()
        .expect("the model's own error is chained");
    assert_eq!(original.to_string(), "the instrument is offline");
    assert!(original.downcast_ref::<Offline>().is_some());

    // The error crosses a thread boundary and keeps its chain.
    let handle = std::thread::spawn(move || {
        std::error::Error::source(&err)
            .and_then(|e| e.source())
            .map(|e| e.to_string())
    });
    assert_eq!(
        handle.join().unwrap().as_deref(),
        Some("the instrument is offline")
    );

    // ModelError constructors.
    assert_eq!(ModelError::from("a").message(), "a");
    assert_eq!(ModelError::from(String::from("b")).message(), "b");
    let wrapped = ModelError::from_error(Offline);
    assert_eq!(wrapped.message(), "the instrument is offline");
    assert!(wrapped.source_ref().is_some());
    assert!(ModelError::new("c").into_source().is_none());
}
