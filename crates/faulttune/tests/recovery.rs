//! Recovery tests (ARCHITECTURE section 13, level "Recovery"): inject model errors, NaN,
//! panics and cancellation into running calibrations, check that every committed step survives
//! in the database, and that resuming with `calibrate_into` completes the calibration with a
//! valid, collision-free result. Then "kill" a run, damage its checkpoint journal the way a
//! crash does, and rebuild the database from an earlier save plus the journal.
//!
//! Only the public API of the crate is used. The file is self-contained.
//!
//! The model under test is the toy `CollisionModel` (hard frequency collisions as `+inf`
//! errors) wrapped in a saboteur. A calibration is *valid* when every goal element has a value
//! that is one of its options, the history is consistent with the values, and the model reports
//! no collision.
#![cfg(feature = "serde")]

use std::fs::OpenOptions;
use std::io::Write as _;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use faulttune::toy::CollisionModel;
use faulttune::{
    read_journal, CancellationToken, CandidateBatch, CheckpointObserver, EdgeType, ElementId,
    ElementKind, ElementSet, ErrorModel, Event, FaultTuneError, HistoryRecord, ModelError,
    ModelErrorPolicy, ParameterDatabase, ProcessorGraph, RecordingObserver, Snake, SnakeConfig,
    Step, StepRecord,
};

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

/// A `rows x cols` lattice and the goal of the README: nodes and engineered edges.
fn lattice(rows: usize, cols: usize) -> (ProcessorGraph, ElementSet) {
    let graph = ProcessorGraph::square_lattice(rows, cols);
    let goal = graph.goal_excluding(&[ElementKind::Edge(EdgeType::Parasitic)]);
    (graph, goal)
}

fn config(d_p: u32, d_r: u32, d_t: u32) -> SnakeConfig {
    SnakeConfig {
        parameter_distance: d_p,
        constraint_distance: d_r,
        traversal_distance: d_t,
        seed: 7,
        ..SnakeConfig::default()
    }
}

/// A collision model for which a collision-free assignment always exists (interacting
/// elements are within `d_R = 2` of each other, so every pair is seen by some step).
fn collision_model(graph: &ProcessorGraph, seed: u64) -> CollisionModel {
    let probe = CollisionModel::new(graph, 1, 5.0, 7.0, 0.01, 1, seed);
    let k = probe.required_options().max(100);
    assert!(k < 10_000, "the bound must be finite for this setup");
    CollisionModel::new(graph, k, 5.0, 7.0, 0.01, 1, seed)
}

fn node(graph: &ProcessorGraph, r: usize, c: usize) -> ElementId {
    graph.lattice_node(r, c).expect("a node of the lattice")
}

/// The steps committed according to a recorded event stream, in event order.
fn committed(events: &[Event]) -> Vec<StepRecord> {
    events
        .iter()
        .filter_map(|e| match e {
            Event::StepCommitted(s) => Some(s.clone()),
            _ => None,
        })
        .collect()
}

/// Checks that `db` is a complete, collision-free and internally consistent calibration.
fn assert_valid(
    db: &ParameterDatabase<f64>,
    goal: &ElementSet,
    model: &CollisionModel,
    context: &str,
) {
    assert_eq!(db.calibrated(), goal, "{context}: P* must equal G*");
    assert_eq!(
        model.collisions(db),
        vec![],
        "{context}: a valid calibration has no frequency collision"
    );
    let steps: Vec<&StepRecord> = db.steps().collect();
    assert_eq!(steps.len() as u64, db.step_count(), "{context}");
    for (i, step) in steps.iter().enumerate() {
        assert_eq!(
            step.index, i as u64,
            "{context}: step indices are 0, 1, 2, ..."
        );
        assert!(
            step.error.is_finite(),
            "{context}: step {i} has no feasible value"
        );
    }
    for (id, entry) in db.iter() {
        assert!(goal.contains(id), "{context}: {id} is outside the goal");
        let options = model.options(id);
        assert_eq!(
            options[entry.option_index as usize], entry.value,
            "{context}: the value of {id} is not its option"
        );
        let step = steps[entry.step as usize];
        let j = step
            .parameters
            .iter()
            .position(|&p| p == id)
            .unwrap_or_else(|| panic!("{context}: step {} does not list {id}", entry.step));
        assert_eq!(step.option_indices[j], entry.option_index, "{context}");
    }
}

/// Everything `before` held is still held by `after`: the history is a prefix and the entries
/// that were calibrated are untouched (a step never overwrites a calibrated element).
fn assert_survives(before: &ParameterDatabase<f64>, after: &ParameterDatabase<f64>, ctx: &str) {
    let n = before.history().len();
    assert!(after.history().len() >= n, "{ctx}: history shrank");
    assert_eq!(
        &after.history()[..n],
        before.history(),
        "{ctx}: history changed"
    );
    for (id, entry) in before.iter() {
        assert_eq!(after.get(id), Some(entry), "{ctx}: {id} changed");
    }
}

// ---------------------------------------------------------------------------------------------
// The saboteur
// ---------------------------------------------------------------------------------------------

/// What goes wrong for the step centred on `Saboteur::target`.
#[derive(Copy, Clone, Debug)]
enum Fault {
    /// `prepare` fails its first `n` calls (a flaky measurement).
    PrepareFails(usize),
    /// `error_batch` fails its first `n` calls.
    BatchFails(usize),
    /// Every error is NaN.
    NanEverywhere,
    /// Errors are NaN for candidates whose first value is above 6.5 GHz only.
    NanOnSomeCandidates,
    /// Every error is `+inf`: "forbidden", which is not a failure.
    InfiniteEverywhere,
    /// `prepare` panics.
    PanicInPrepare,
    /// `error_batch` panics.
    PanicInError,
    /// `options` panics (for the target element).
    PanicInOptions,
}

struct Saboteur<'a> {
    inner: &'a CollisionModel,
    target: ElementId,
    fault: Fault,
    prepare_calls: AtomicUsize,
    batch_calls: AtomicUsize,
}

impl<'a> Saboteur<'a> {
    fn new(inner: &'a CollisionModel, target: ElementId, fault: Fault) -> Self {
        Self {
            inner,
            target,
            fault,
            prepare_calls: AtomicUsize::new(0),
            batch_calls: AtomicUsize::new(0),
        }
    }

    fn prepare_calls(&self) -> usize {
        self.prepare_calls.load(Ordering::SeqCst)
    }

    fn batch_calls(&self) -> usize {
        self.batch_calls.load(Ordering::SeqCst)
    }
}

impl ErrorModel for Saboteur<'_> {
    type Value = f64;
    type Data = ();

    fn options(&self, element: ElementId) -> Vec<f64> {
        if element == self.target && matches!(self.fault, Fault::PanicInOptions) {
            panic!("the options of {element} blew up");
        }
        self.inner.options(element)
    }

    fn prepare(&self, step: &Step<'_, f64>) -> Result<(), ModelError> {
        if step.centre == self.target {
            let call = self.prepare_calls.fetch_add(1, Ordering::SeqCst) + 1;
            match self.fault {
                Fault::PrepareFails(n) if call <= n => {
                    return Err(ModelError::new(format!("measurement {call} failed")));
                }
                Fault::PanicInPrepare => panic!("prepare of {} blew up", step.centre),
                _ => {}
            }
        }
        self.inner.prepare(step)
    }

    fn error(&self, step: &Step<'_, f64>, data: &(), candidate: &[f64]) -> f64 {
        self.inner.error(step, data, candidate)
    }

    fn error_batch(
        &self,
        step: &Step<'_, f64>,
        data: &(),
        candidates: &CandidateBatch<'_, f64>,
        out: &mut [f64],
    ) -> Result<(), ModelError> {
        if step.centre == self.target {
            let call = self.batch_calls.fetch_add(1, Ordering::SeqCst) + 1;
            match self.fault {
                Fault::BatchFails(n) if call <= n => {
                    return Err(ModelError::new(format!("evaluation {call} failed")));
                }
                Fault::PanicInError => panic!("the error of {} blew up", step.centre),
                _ => {}
            }
        }
        self.inner.error_batch(step, data, candidates, out)?;
        if step.centre == self.target {
            match self.fault {
                Fault::NanEverywhere => out.fill(f64::NAN),
                Fault::NanOnSomeCandidates => {
                    for (slot, row) in out.iter_mut().zip(candidates.rows()) {
                        if row[0] > 6.5 {
                            *slot = f64::NAN;
                        }
                    }
                }
                Fault::InfiniteEverywhere => out.fill(f64::INFINITY),
                _ => {}
            }
        }
        Ok(())
    }
}

/// A fault that never stops happening.
const ALWAYS: usize = usize::MAX;

fn snake_for<'g>(graph: &'g ProcessorGraph, goal: &ElementSet, cfg: SnakeConfig) -> Snake<'g> {
    Snake::new(graph, goal.clone(), cfg)
}

// ---------------------------------------------------------------------------------------------
// Model errors: Abort
// ---------------------------------------------------------------------------------------------

#[test]
fn abort_keeps_committed_steps_names_the_centre_and_resume_completes() {
    let (graph, goal) = lattice(6, 6);
    let healthy = collision_model(&graph, 11);
    for (name, fault) in [
        ("prepare", Fault::PrepareFails(ALWAYS)),
        ("error_batch", Fault::BatchFails(ALWAYS)),
    ] {
        for target in [node(&graph, 3, 3), node(&graph, 5, 5), node(&graph, 0, 1)] {
            let ctx = format!("{name} fails at {}", graph.label(target));
            let broken = Saboteur::new(&healthy, target, fault);
            let events = RecordingObserver::new();
            let snake = snake_for(&graph, &goal, config(1, 2, 2)).observer(events.clone());
            let mut db = ParameterDatabase::new(&graph);

            let err = snake.calibrate_into(&mut db, &broken).unwrap_err();
            match &err {
                FaultTuneError::Model { element, source } => {
                    assert_eq!(*element, target, "{ctx}");
                    assert!(source.message().contains("failed"), "{ctx}: {source}");
                }
                other => panic!("{ctx}: expected a model error, got {other:?}"),
            }
            // The message points at the model, and the model's error is the source.
            assert!(err.to_string().contains("failed"), "{ctx}: {err}");
            assert!(std::error::Error::source(&err).is_some(), "{ctx}");

            // Everything committed before the failure is in the database.
            let log = events.events();
            let steps = committed(&log);
            assert_eq!(steps.len() as u64, db.step_count(), "{ctx}");
            assert!(!db.is_empty(), "{ctx}: steps before the failure are kept");
            assert!(!db.is_calibrated(target), "{ctx}");
            assert!(db.len() < goal.len(), "{ctx}");
            assert!(
                matches!(log.last(), Some(Event::Finished(r)) if r.steps == db.step_count()),
                "{ctx}: a failed run still ends with Finished"
            );
            assert!(db.steps().all(|s| s.centre != target), "{ctx}");

            // Resume with the same (still failing) model fails the same way and loses nothing.
            let snapshot = db.clone();
            let again = snake.calibrate_into(&mut db, &broken).unwrap_err();
            assert!(matches!(again, FaultTuneError::Model { element, .. } if element == target));
            assert_survives(&snapshot, &db, &ctx);

            // Resume with the healthy model completes.
            let after_failure = db.clone();
            let report = snake.calibrate_into(&mut db, &healthy).unwrap();
            assert!(report.steps > 0, "{ctx}");
            assert_survives(&after_failure, &db, &ctx);
            assert_valid(&db, &goal, &healthy, &ctx);
        }
    }
}

#[test]
fn a_failure_in_the_seed_step_leaves_an_empty_database_that_still_resumes() {
    let (graph, goal) = lattice(5, 5);
    let healthy = collision_model(&graph, 3);
    // The default seed policy starts at the lowest ID: the first node.
    let seed = graph.nodes().next().unwrap();
    let broken = Saboteur::new(&healthy, seed, Fault::PrepareFails(ALWAYS));
    let snake = snake_for(&graph, &goal, config(1, 2, 2));
    let mut db = ParameterDatabase::new(&graph);
    let err = snake.calibrate_into(&mut db, &broken).unwrap_err();
    assert!(matches!(err, FaultTuneError::Model { element, .. } if element == seed));
    assert!(db.is_empty() && db.history().is_empty());
    snake.calibrate_into(&mut db, &healthy).unwrap();
    assert_valid(&db, &goal, &healthy, "seed failure");
}

// ---------------------------------------------------------------------------------------------
// Model errors: Retry
// ---------------------------------------------------------------------------------------------

#[test]
fn retry_rides_out_a_flaky_model_and_changes_nothing_else() {
    let (graph, goal) = lattice(6, 6);
    let healthy = collision_model(&graph, 21);
    let target = node(&graph, 2, 2);
    let reference = snake_for(&graph, &goal, config(1, 2, 2))
        .calibrate(&healthy)
        .unwrap();
    for (name, fault) in [
        ("prepare", Fault::PrepareFails(2)),
        ("error_batch", Fault::BatchFails(2)),
    ] {
        let flaky = Saboteur::new(&healthy, target, fault);
        let cfg = SnakeConfig {
            on_model_error: ModelErrorPolicy::Retry { times: 2 },
            ..config(1, 2, 2)
        };
        let events = RecordingObserver::new();
        let snake = snake_for(&graph, &goal, cfg).observer(events.clone());
        let (db, report) = snake.calibrate_with_report(&flaky).unwrap();
        assert_valid(&db, &goal, &healthy, name);
        // Retries are invisible: no failure event, the same outcome as a healthy model
        // (retries do not consume random numbers, and they are not counted as evaluations).
        assert!(
            !events
                .events()
                .iter()
                .any(|e| matches!(e, Event::StepFailed { .. })),
            "{name}"
        );
        assert!(report.skipped.is_empty(), "{name}");
        assert!(
            db.same_outcome(&reference),
            "{name}: retries must not change the result"
        );
        // The failing call was made 3 times (two failures and the success).
        match fault {
            Fault::PrepareFails(_) => assert_eq!(flaky.prepare_calls(), 3, "{name}"),
            _ => assert!(flaky.batch_calls() >= 3, "{name}"),
        }
    }
}

#[test]
fn retry_gives_up_after_the_allowed_attempts_and_the_run_can_resume() {
    let (graph, goal) = lattice(6, 6);
    let healthy = collision_model(&graph, 22);
    let target = node(&graph, 3, 1);
    for (name, fault) in [
        ("prepare", Fault::PrepareFails(3)),
        ("error_batch", Fault::BatchFails(3)),
    ] {
        let flaky = Saboteur::new(&healthy, target, fault);
        let cfg = SnakeConfig {
            on_model_error: ModelErrorPolicy::Retry { times: 2 },
            ..config(1, 2, 2)
        };
        let snake = snake_for(&graph, &goal, cfg);
        let mut db = ParameterDatabase::new(&graph);
        let err = snake.calibrate_into(&mut db, &flaky).unwrap_err();
        assert!(
            matches!(err, FaultTuneError::Model { element, .. } if element == target),
            "{name}: {err:?}"
        );
        // One attempt plus two retries were made before giving up.
        match fault {
            Fault::PrepareFails(_) => assert_eq!(flaky.prepare_calls(), 3, "{name}"),
            _ => assert_eq!(flaky.batch_calls(), 3, "{name}"),
        }
        assert!(!db.is_empty() && !db.is_calibrated(target), "{name}");
        let kept = db.clone();
        // The hardware has recovered (the fault only affects the first three calls): resuming
        // with the same model object completes.
        snake.calibrate_into(&mut db, &flaky).unwrap();
        assert_survives(&kept, &db, name);
        assert_valid(&db, &goal, &healthy, name);
    }
}

#[test]
fn retry_zero_behaves_like_abort() {
    let (graph, goal) = lattice(4, 4);
    let healthy = collision_model(&graph, 23);
    let target = node(&graph, 1, 1);
    let flaky = Saboteur::new(&healthy, target, Fault::PrepareFails(1));
    let cfg = SnakeConfig {
        on_model_error: ModelErrorPolicy::Retry { times: 0 },
        ..config(1, 2, 2)
    };
    let err = snake_for(&graph, &goal, cfg).calibrate(&flaky).unwrap_err();
    assert!(matches!(err, FaultTuneError::Model { .. }));
    assert_eq!(flaky.prepare_calls(), 1);
}

// ---------------------------------------------------------------------------------------------
// Model errors: SkipElement
// ---------------------------------------------------------------------------------------------

#[test]
fn skip_element_excludes_the_centre_reports_it_and_a_later_call_finishes_it() {
    let (graph, goal) = lattice(6, 6);
    let healthy = collision_model(&graph, 31);
    // The seed, an interior element and the last corner.
    for target in [node(&graph, 0, 0), node(&graph, 3, 2), node(&graph, 5, 5)] {
        for fault in [Fault::PrepareFails(ALWAYS), Fault::BatchFails(ALWAYS)] {
            let ctx = format!("{fault:?} at {}", graph.label(target));
            let broken = Saboteur::new(&healthy, target, fault);
            let cfg = SnakeConfig {
                on_model_error: ModelErrorPolicy::SkipElement,
                ..config(1, 2, 2)
            };
            let events = RecordingObserver::new();
            let snake = snake_for(&graph, &goal, cfg).observer(events.clone());
            let mut db = ParameterDatabase::new(&graph);
            let report = snake.calibrate_into(&mut db, &broken).unwrap();

            assert_eq!(report.skipped, vec![target], "{ctx}");
            assert!(!db.is_calibrated(target), "{ctx}");
            let mut expected = goal.clone();
            expected.remove(target);
            assert_eq!(
                db.calibrated(),
                &expected,
                "{ctx}: everything else is calibrated"
            );
            let failures: Vec<_> = events
                .events()
                .into_iter()
                .filter_map(|e| match e {
                    Event::StepFailed { centre, error } => Some((centre, error)),
                    _ => None,
                })
                .collect();
            assert_eq!(
                failures.len(),
                1,
                "{ctx}: the skipped element is tried once"
            );
            assert_eq!(failures[0].0, target, "{ctx}");
            assert!(failures[0].1.contains("failed"), "{ctx}: {}", failures[0].1);
            assert_eq!(report.steps, db.step_count(), "{ctx}");
            // The collision model is still satisfied by everything calibrated.
            assert!(healthy.collisions(&db).is_empty(), "{ctx}");

            // A second call retries the skipped element: with the failing model it is skipped
            // again (and nothing else is touched) ...
            let kept = db.clone();
            let again = snake.calibrate_into(&mut db, &broken).unwrap();
            assert_eq!(again.skipped, vec![target], "{ctx}");
            assert_eq!(again.steps, 0, "{ctx}");
            assert!(db.same_outcome(&kept), "{ctx}");

            // ... and with the healthy model it is calibrated under all the constraints.
            let report = snake.calibrate_into(&mut db, &healthy).unwrap();
            assert!(report.skipped.is_empty(), "{ctx}");
            assert_survives(&kept, &db, &ctx);
            assert_valid(&db, &goal, &healthy, &ctx);
        }
    }
}

#[test]
fn skip_element_lists_several_skipped_elements_in_ascending_order() {
    let (graph, goal) = lattice(6, 6);
    let healthy = collision_model(&graph, 32);

    // A model that fails for two different centres.
    struct TwoBroken<'a> {
        inner: &'a CollisionModel,
        broken: [ElementId; 2],
    }
    impl ErrorModel for TwoBroken<'_> {
        type Value = f64;
        type Data = ();
        fn options(&self, e: ElementId) -> Vec<f64> {
            self.inner.options(e)
        }
        fn prepare(&self, step: &Step<'_, f64>) -> Result<(), ModelError> {
            if self.broken.contains(&step.centre) {
                Err(ModelError::new("sensor offline"))
            } else {
                Ok(())
            }
        }
        fn error(&self, step: &Step<'_, f64>, data: &(), c: &[f64]) -> f64 {
            self.inner.error(step, data, c)
        }
    }

    let (a, b) = (node(&graph, 4, 4), node(&graph, 1, 2));
    let model = TwoBroken {
        inner: &healthy,
        broken: [a, b],
    };
    let cfg = SnakeConfig {
        on_model_error: ModelErrorPolicy::SkipElement,
        ..config(1, 2, 2)
    };
    let snake = snake_for(&graph, &goal, cfg);
    let (db, report) = snake.calibrate_with_report(&model).unwrap();
    let mut expected = vec![a, b];
    expected.sort();
    assert_eq!(report.skipped, expected);
    assert_eq!(db.len(), goal.len() - 2);
}

#[test]
fn skip_element_does_not_hide_nan_or_panics() {
    let (graph, goal) = lattice(5, 5);
    let healthy = collision_model(&graph, 33);
    let target = node(&graph, 2, 2);
    let cfg = SnakeConfig {
        on_model_error: ModelErrorPolicy::SkipElement,
        ..config(1, 2, 2)
    };
    let snake = snake_for(&graph, &goal, cfg);
    let nan = Saboteur::new(&healthy, target, Fault::NanEverywhere);
    let err = snake.calibrate(&nan).unwrap_err();
    assert!(matches!(err, FaultTuneError::NonFiniteError { element } if element == target));

    let panicking = Saboteur::new(&healthy, target, Fault::PanicInPrepare);
    let outcome = catch_unwind(AssertUnwindSafe(|| snake.calibrate(&panicking)));
    assert!(outcome.is_err(), "SkipElement must not swallow a panic");
}

// ---------------------------------------------------------------------------------------------
// NaN and +inf
// ---------------------------------------------------------------------------------------------

#[test]
fn nan_is_fatal_under_every_policy_and_is_never_retried() {
    let (graph, goal) = lattice(6, 6);
    let healthy = collision_model(&graph, 41);
    let target = node(&graph, 3, 3);
    for fault in [Fault::NanEverywhere, Fault::NanOnSomeCandidates] {
        for policy in [
            ModelErrorPolicy::Abort,
            ModelErrorPolicy::Retry { times: 5 },
            ModelErrorPolicy::SkipElement,
        ] {
            let ctx = format!("{fault:?} with {policy:?}");
            let nan = Saboteur::new(&healthy, target, fault);
            let cfg = SnakeConfig {
                on_model_error: policy,
                ..config(1, 2, 2)
            };
            let events = RecordingObserver::new();
            let snake = snake_for(&graph, &goal, cfg).observer(events.clone());
            let mut db = ParameterDatabase::new(&graph);

            let err = snake.calibrate_into(&mut db, &nan).unwrap_err();
            assert!(
                matches!(err, FaultTuneError::NonFiniteError { element } if element == target),
                "{ctx}: {err:?}"
            );
            assert!(err.to_string().contains("NaN"), "{ctx}: {err}");
            // A NaN answer is a bug of the model, not a transient failure: it is not retried
            // (the failing batch was asked exactly once) and not skipped.
            assert_eq!(nan.batch_calls(), 1, "{ctx}");
            assert!(
                !events
                    .events()
                    .iter()
                    .any(|e| matches!(e, Event::StepFailed { .. })),
                "{ctx}"
            );
            // The steps before it are kept, and it is not among them.
            let log = events.events();
            assert_eq!(committed(&log).len() as u64, db.step_count(), "{ctx}");
            assert!(!db.is_empty() && !db.is_calibrated(target), "{ctx}");
            assert!(matches!(log.last(), Some(Event::Finished(_))), "{ctx}");

            let kept = db.clone();
            snake.calibrate_into(&mut db, &healthy).unwrap();
            assert_survives(&kept, &db, &ctx);
            assert_valid(&db, &goal, &healthy, &ctx);
        }
    }
}

#[test]
fn infinite_errors_mean_forbidden_not_failed() {
    let (graph, goal) = lattice(5, 5);
    let healthy = collision_model(&graph, 42);
    let target = node(&graph, 2, 2);
    let model = Saboteur::new(&healthy, target, Fault::InfiniteEverywhere);
    let events = RecordingObserver::new();
    let snake = snake_for(&graph, &goal, config(1, 2, 2)).observer(events.clone());
    let (db, report) = snake.calibrate_with_report(&model).unwrap();
    assert_eq!(
        db.calibrated(),
        &goal,
        "a forbidden step still commits something"
    );
    assert!(report.skipped.is_empty());
    assert!(!events
        .events()
        .iter()
        .any(|e| matches!(e, Event::StepFailed { .. })));
    let bad = db
        .steps()
        .find(|s| s.centre == target)
        .expect("the target was a step centre");
    assert_eq!(bad.error, f64::INFINITY);
    // The worst step of the run is the one with no feasible candidate.
    assert_eq!(report.worst_steps[0].centre, target);
    assert_eq!(report.worst_steps[0].error, f64::INFINITY);
}

// ---------------------------------------------------------------------------------------------
// Panics
// ---------------------------------------------------------------------------------------------

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_owned())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_default()
}

#[test]
fn a_panicking_model_keeps_committed_steps_and_the_snake_stays_usable() {
    let (graph, goal) = lattice(6, 6);
    let healthy = collision_model(&graph, 51);
    let target = node(&graph, 3, 3);
    for fault in [
        Fault::PanicInPrepare,
        Fault::PanicInError,
        Fault::PanicInOptions,
    ] {
        let ctx = format!("{fault:?}");
        let bomb = Saboteur::new(&healthy, target, fault);
        let events = RecordingObserver::new();
        let snake = snake_for(&graph, &goal, config(1, 2, 2)).observer(events.clone());
        let mut db = ParameterDatabase::new(&graph);

        let outcome = catch_unwind(AssertUnwindSafe(|| snake.calibrate_into(&mut db, &bomb)));
        let payload = outcome.expect_err("the panic of the model must reach the caller");
        assert!(panic_message(&*payload).contains("blew up"), "{ctx}");

        // The steps committed before the panic were merged, and the observers saw exactly them.
        let log = events.events();
        assert_eq!(committed(&log).len() as u64, db.step_count(), "{ctx}");
        assert!(!db.is_empty(), "{ctx}");
        assert!(!db.is_calibrated(target), "{ctx}");
        // A run that panics does not announce its end.
        assert!(
            !log.iter().any(|e| matches!(e, Event::Finished(_))),
            "{ctx}"
        );

        // The same Snake and the same database take the resumed run.
        let kept = db.clone();
        snake.calibrate_into(&mut db, &healthy).unwrap();
        assert_survives(&kept, &db, &ctx);
        assert_valid(&db, &goal, &healthy, &ctx);
    }
}

/// The nodes and edges of the goal split into the four quadrants of a lattice, by position.
fn quadrants(
    graph: &ProcessorGraph,
    goal: &ElementSet,
    rows: usize,
    cols: usize,
) -> Vec<ElementSet> {
    let coords = |id: ElementId| -> (usize, usize) {
        // Twice the average row and column of the element's end points.
        let label_pos = |n: ElementId| {
            let label = graph.label(n);
            let (r, c) = label[1..].split_once('_').unwrap();
            (r.parse::<usize>().unwrap(), c.parse::<usize>().unwrap())
        };
        match graph.endpoints(id) {
            None => {
                let (r, c) = label_pos(id);
                (2 * r, 2 * c)
            }
            Some([a, b]) => {
                let ((ra, ca), (rb, cb)) = (label_pos(a), label_pos(b));
                (ra + rb, ca + cb)
            }
        }
    };
    let mut parts = vec![ElementSet::new(graph.len()); 4];
    for id in goal.iter() {
        let (r2, c2) = coords(id);
        let q = usize::from(r2 >= rows) * 2 + usize::from(c2 >= cols);
        parts[q].insert(id);
    }
    parts
}

#[test]
fn a_panic_in_one_parallel_subgoal_keeps_the_steps_of_the_others() {
    let (graph, goal) = lattice(10, 10);
    let healthy = collision_model(&graph, 52);
    let regions = quadrants(&graph, &goal, 10, 10);
    let cfg = SnakeConfig {
        threads: Some(4),
        ..config(1, 2, 2)
    };
    let plan = snake_for(&graph, &goal, cfg.clone())
        .stitch_plan(regions)
        .unwrap();

    // The subgoals of stage one are ordered by their smallest element; the last of them is
    // the one whose failure leaves all earlier ones running to their end.
    let stage_one_regions: Vec<ElementSet> = plan
        .regions
        .iter()
        .map(|r| r.intersection(&plan.stage_one))
        .collect();
    let last = (0..4)
        .max_by_key(|&i| stage_one_regions[i].first().unwrap())
        .unwrap();
    let target = stage_one_regions[last]
        .iter()
        .filter(|&e| graph.kind(e) == ElementKind::Node)
        .max()
        .unwrap();

    let outcomes: Vec<ParameterDatabase<f64>> = [1usize, 4]
        .into_iter()
        .map(|threads| {
            let cfg = SnakeConfig {
                threads: Some(threads),
                ..cfg.clone()
            };
            let snake = snake_for(&graph, &goal, cfg);
            let bomb = Saboteur::new(&healthy, target, Fault::PanicInError);
            let mut db = ParameterDatabase::new(&graph);
            let outcome = catch_unwind(AssertUnwindSafe(|| snake.stitch(&mut db, &plan, &bomb)));
            let payload = outcome.expect_err("the panic must propagate");
            assert!(panic_message(&*payload).contains("blew up"));
            assert!(!db.is_calibrated(target));
            // Every subgoal before the failing one ran to its end and was merged.
            for (i, region) in stage_one_regions.iter().enumerate() {
                if i != last && region.first().unwrap() < stage_one_regions[last].first().unwrap() {
                    assert!(
                        region.is_subset(db.calibrated()),
                        "threads {threads}: region {i}"
                    );
                }
            }
            // And the failing subgoal's own earlier steps are there too.
            assert!(db
                .steps()
                .any(|s| stage_one_regions[last].contains(s.centre)));

            let kept = db.clone();
            snake.calibrate_into(&mut db, &healthy).unwrap();
            assert_survives(&kept, &db, &format!("threads {threads}"));
            assert_valid(&db, &goal, &healthy, &format!("threads {threads}"));
            kept
        })
        .collect();
    assert!(
        outcomes[0].same_outcome(&outcomes[1]),
        "what survives a panic does not depend on the thread count"
    );
}

// ---------------------------------------------------------------------------------------------
// Cancellation
// ---------------------------------------------------------------------------------------------

/// An observer that cancels `token` when the `k`-th `StepCommitted` event after the last reset
/// of `seen` arrives.
fn cancel_after(
    token: &CancellationToken,
    seen: &Arc<AtomicUsize>,
    k: usize,
) -> impl Fn(&Event) + Send + Sync + 'static {
    let (token, seen) = (token.clone(), Arc::clone(seen));
    move |event: &Event| {
        if matches!(event, Event::StepCommitted(_)) && seen.fetch_add(1, Ordering::SeqCst) + 1 == k
        {
            token.cancel();
        }
    }
}

fn expect_cancelled(result: Result<faulttune::Report, FaultTuneError>) -> faulttune::Report {
    match result {
        Err(FaultTuneError::Cancelled { report }) => *report,
        other => panic!("expected Cancelled, got {other:?}"),
    }
}

#[test]
fn cancelling_after_k_steps_keeps_exactly_k_steps_and_resume_completes() {
    let (graph, goal) = lattice(7, 7);
    let healthy = collision_model(&graph, 61);
    let total = snake_for(&graph, &goal, config(1, 2, 2))
        .calibrate_with_report(&healthy)
        .unwrap()
        .1
        .steps as usize;
    assert!(total > 20);
    for k in [1, 2, 5, 17, total - 1] {
        let ctx = format!("cancel after {k} of {total} steps");
        let token = CancellationToken::new();
        let seen = Arc::new(AtomicUsize::new(0));
        let events = RecordingObserver::new();
        let snake = snake_for(&graph, &goal, config(1, 2, 2))
            .cancellation(token.clone())
            .observer(events.clone())
            .observer(cancel_after(&token, &seen, k));
        let mut db = ParameterDatabase::new(&graph);

        let report = expect_cancelled(snake.calibrate_into(&mut db, &healthy));
        // The cancellation is checked between steps: the step whose event cancelled is kept
        // and the next one never starts.
        assert_eq!(db.step_count(), k as u64, "{ctx}");
        assert_eq!(report.steps, k as u64, "{ctx}");
        assert!(token.is_cancelled(), "{ctx}");

        // Event sequence: one pass, k steps, the end of the pass, Finished.
        let log = events.events();
        assert!(
            matches!(log.first(), Some(Event::PassStarted { pass: 0, .. })),
            "{ctx}"
        );
        assert_eq!(committed(&log).len(), k, "{ctx}");
        let n = log.len();
        assert!(
            matches!(&log[n - 2], Event::PassFinished { pass: 0, calibrated } if *calibrated == db.len()),
            "{ctx}: {:?}",
            log[n - 2]
        );
        assert!(
            matches!(&log[n - 1], Event::Finished(r) if r.steps == k as u64),
            "{ctx}"
        );
        assert_eq!(n, k + 3, "{ctx}");

        // Resume: the token is reset, the pass numbering continues, and the result is valid.
        token.reset();
        let kept = db.clone();
        let resumed = RecordingObserver::new();
        let snake = snake_for(&graph, &goal, config(1, 2, 2)).observer(resumed.clone());
        let report = snake.calibrate_into(&mut db, &healthy).unwrap();
        assert!(report.steps > 0, "{ctx}");
        assert!(matches!(
            resumed.events().first(),
            Some(Event::PassStarted { pass: 1, uncalibrated, .. }) if *uncalibrated == goal.len() - kept.len()
        ));
        assert_survives(&kept, &db, &ctx);
        assert_valid(&db, &goal, &healthy, &ctx);
        assert!(db.steps().skip(k).all(|s| s.pass >= 1), "{ctx}");
    }
}

#[test]
fn cancelling_before_the_run_starts_commits_nothing() {
    let (graph, goal) = lattice(5, 5);
    let healthy = collision_model(&graph, 62);
    let token = CancellationToken::new();
    token.cancel();
    let events = RecordingObserver::new();
    let snake = snake_for(&graph, &goal, config(1, 2, 2))
        .cancellation(token.clone())
        .observer(events.clone());
    let mut db = ParameterDatabase::new(&graph);
    let report = expect_cancelled(snake.calibrate_into(&mut db, &healthy));
    assert_eq!(report.steps, 0);
    assert!(db.is_empty() && db.history().is_empty());
    assert!(matches!(events.events().last(), Some(Event::Finished(_))));
    // The token is a switch that stays on until it is reset.
    expect_cancelled(snake.calibrate_into(&mut db, &healthy));
    token.reset();
    snake.calibrate_into(&mut db, &healthy).unwrap();
    assert_valid(&db, &goal, &healthy, "cancelled before the start");
}

#[test]
fn cancelling_at_the_very_last_step_leaves_a_complete_database() {
    let (graph, goal) = lattice(5, 5);
    let healthy = collision_model(&graph, 63);
    let total = snake_for(&graph, &goal, config(1, 2, 2))
        .calibrate_with_report(&healthy)
        .unwrap()
        .1
        .steps as usize;
    let token = CancellationToken::new();
    let seen = Arc::new(AtomicUsize::new(0));
    let snake = snake_for(&graph, &goal, config(1, 2, 2))
        .cancellation(token.clone())
        .observer(cancel_after(&token, &seen, total));
    let mut db = ParameterDatabase::new(&graph);
    // Either outcome is acceptable (the work was finished, or the cancellation was seen before
    // the run noticed it was done), but the database must be complete and valid.
    match snake.calibrate_into(&mut db, &healthy) {
        Ok(report) => assert_eq!(report.steps, total as u64),
        Err(FaultTuneError::Cancelled { report }) => assert_eq!(report.steps, total as u64),
        Err(other) => panic!("unexpected {other:?}"),
    }
    assert_valid(&db, &goal, &healthy, "cancelled at the last step");
    // Resuming a complete database is a no-op.
    token.reset();
    let report = snake.calibrate_into(&mut db, &healthy).unwrap();
    assert_eq!((report.steps, report.passes, report.evaluations), (0, 0, 0));
}

#[test]
fn repeated_cancellation_always_makes_progress_and_ends_with_a_valid_result() {
    let (graph, goal) = lattice(7, 7);
    let healthy = collision_model(&graph, 64);
    let token = CancellationToken::new();
    let seen = Arc::new(AtomicUsize::new(0));
    // One thread: with several subgoals in a pass the number of steps after a cancellation
    // would depend on the scheduling.
    let cfg = SnakeConfig {
        threads: Some(1),
        ..config(1, 2, 2)
    };
    let snake = snake_for(&graph, &goal, cfg)
        .cancellation(token.clone())
        .observer(cancel_after(&token, &seen, 3));
    let mut db = ParameterDatabase::new(&graph);
    let mut attempts = 0;
    let mut previous: Option<ParameterDatabase<f64>> = None;
    loop {
        attempts += 1;
        assert!(attempts < 100, "no convergence");
        seen.store(0, Ordering::SeqCst);
        token.reset();
        let before = db.step_count();
        match snake.calibrate_into(&mut db, &healthy) {
            Ok(_) => break,
            Err(FaultTuneError::Cancelled { report }) => {
                assert_eq!(report.steps, 3);
                assert_eq!(db.step_count(), before + 3);
            }
            Err(other) => panic!("unexpected {other:?}"),
        }
        if let Some(previous) = previous.take() {
            assert_survives(&previous, &db, &format!("attempt {attempts}"));
        }
        previous = Some(db.clone());
    }
    assert!(
        attempts > 5,
        "the run was interrupted many times ({attempts})"
    );
    assert_valid(&db, &goal, &healthy, "after repeated cancellation");
    // The passes are numbered consecutively over all the attempts.
    let passes: Vec<u32> = db.steps().map(|s| s.pass).collect();
    assert!(
        passes.windows(2).all(|w| w[1] == w[0] || w[1] == w[0] + 1),
        "{passes:?}"
    );
}

#[test]
fn cancelling_parallel_subgoals_keeps_at_least_the_steps_that_triggered_it() {
    let (graph, goal) = lattice(10, 10);
    let healthy = collision_model(&graph, 65);
    let cfg = SnakeConfig {
        threads: Some(4),
        ..config(1, 2, 2)
    };
    let snake = snake_for(&graph, &goal, cfg.clone());
    let plan = snake.stitch_plan(quadrants(&graph, &goal, 10, 10)).unwrap();
    for k in [1usize, 10, 40] {
        let token = CancellationToken::new();
        let seen = Arc::new(AtomicUsize::new(0));
        let events = RecordingObserver::new();
        let snake = snake_for(&graph, &goal, cfg.clone())
            .cancellation(token.clone())
            .observer(events.clone())
            .observer(cancel_after(&token, &seen, k));
        let mut db = ParameterDatabase::new(&graph);
        let report = expect_cancelled(snake.stitch(&mut db, &plan, &healthy));
        // Subgoals that were running finish their current step, so there can be more than k.
        assert!(db.step_count() >= k as u64, "k = {k}");
        assert!(db.step_count() < 4 * 25 + 40, "k = {k}: the run stopped");
        assert_eq!(report.steps, db.step_count(), "k = {k}");
        assert_eq!(
            committed(&events.events()).len() as u64,
            db.step_count(),
            "k = {k}"
        );
        // The merged history is consistent: consecutive global indices.
        for (i, s) in db.steps().enumerate() {
            assert_eq!(s.index, i as u64);
        }
        token.reset();
        let kept = db.clone();
        snake.calibrate_into(&mut db, &healthy).unwrap();
        assert_survives(&kept, &db, &format!("k = {k}"));
        assert_valid(&db, &goal, &healthy, &format!("k = {k}"));
    }
}

#[test]
fn resuming_on_a_pass_boundary_equals_the_uninterrupted_run() {
    // With d_P = 2 one step absorbs everything the traversal could reach, so a pass makes few
    // steps and the calibration takes many passes: there are many boundaries to cut at.
    let (graph, goal) = lattice(8, 8);
    let healthy = collision_model(&graph, 66);
    let uninterrupted = snake_for(&graph, &goal, config(2, 2, 2))
        .calibrate(&healthy)
        .unwrap();
    let passes = uninterrupted.steps().map(|s| s.pass).max().unwrap() + 1;
    assert!(
        passes >= 4,
        "the setup should need several passes, found {passes}"
    );
    assert_valid(&uninterrupted, &goal, &healthy, "uninterrupted");

    for cut in 0..passes - 1 {
        let token = CancellationToken::new();
        let hook = {
            let token = token.clone();
            move |event: &Event| {
                if matches!(event, Event::PassFinished { pass, .. } if *pass == cut) {
                    token.cancel();
                }
            }
        };
        let snake = snake_for(&graph, &goal, config(2, 2, 2))
            .cancellation(token.clone())
            .observer(hook);
        let mut db = ParameterDatabase::new(&graph);
        expect_cancelled(snake.calibrate_into(&mut db, &healthy));
        assert_eq!(db.steps().map(|s| s.pass).max(), Some(cut));
        token.reset();
        snake.calibrate_into(&mut db, &healthy).unwrap();
        assert!(
            db.same_outcome(&uninterrupted),
            "cut after pass {cut}: the resumed run must equal the uninterrupted one"
        );
    }
}

#[test]
fn resuming_in_the_middle_of_a_pass_is_valid_though_not_necessarily_identical() {
    let (graph, goal) = lattice(8, 8);
    let healthy = collision_model(&graph, 67);
    let uninterrupted = snake_for(&graph, &goal, config(1, 2, 2))
        .calibrate(&healthy)
        .unwrap();
    let mut differs = false;
    for k in [4usize, 13, 30] {
        let token = CancellationToken::new();
        let seen = Arc::new(AtomicUsize::new(0));
        let snake = snake_for(&graph, &goal, config(1, 2, 2))
            .cancellation(token.clone())
            .observer(cancel_after(&token, &seen, k));
        let mut db = ParameterDatabase::new(&graph);
        expect_cancelled(snake.calibrate_into(&mut db, &healthy));
        token.reset();
        snake.calibrate_into(&mut db, &healthy).unwrap();
        assert_valid(&db, &goal, &healthy, &format!("cut after {k} steps"));
        // The first k steps are those of the uninterrupted run (same pass, same random
        // numbers); only what follows can differ.
        for (a, b) in db.steps().zip(uninterrupted.steps()).take(k) {
            assert!(a.same_outcome(b), "step {} of the prefix differs", a.index);
        }
        differs |= !db.same_outcome(&uninterrupted);
    }
    // Not asserted to differ (it need not), but a cut is allowed to.
    let _ = differs;
}

// ---------------------------------------------------------------------------------------------
// Kill and resume through the checkpoint journal
// ---------------------------------------------------------------------------------------------

/// Cuts the last line of the journal at `path` to its first `keep` bytes (the newline goes too),
/// as a crash in the middle of writing it would. Returns the length of the whole last line,
/// without its newline.
fn tear_last_line(path: &Path, keep: usize) -> usize {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(
        bytes.last(),
        Some(&b'\n'),
        "a flushed journal ends with a newline"
    );
    let body = &bytes[..bytes.len() - 1];
    let start = body.iter().rposition(|&b| b == b'\n').map_or(0, |p| p + 1);
    let line_len = body.len() - start;
    assert!(keep <= line_len);
    let file = OpenOptions::new().write(true).open(path).unwrap();
    file.set_len((start + keep) as u64).unwrap();
    line_len
}

/// Appends raw bytes to a file (a half-written line of the step the process died in).
fn append_bytes(path: &Path, bytes: &[u8]) {
    let mut file = OpenOptions::new().append(true).open(path).unwrap();
    file.write_all(bytes).unwrap();
}

/// A recovered database must equal the database the dead process had, record by record.
fn assert_same_history(a: &ParameterDatabase<f64>, b: &ParameterDatabase<f64>, ctx: &str) {
    assert!(
        a.same_outcome(b),
        "{ctx}: the databases differ ({} and {} steps)",
        a.step_count(),
        b.step_count()
    );
}

#[test]
fn journal_replay_after_a_kill_on_a_pass_boundary_resumes_to_the_uninterrupted_result() {
    let (graph, goal) = lattice(8, 8);
    let healthy = collision_model(&graph, 71);
    let uninterrupted = snake_for(&graph, &goal, config(2, 2, 2))
        .calibrate(&healthy)
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let (journal, earlier) = (
        dir.path().join("run.jsonl"),
        dir.path().join("earlier.json"),
    );

    // The last save is an empty database: the process died before it could save again.
    ParameterDatabase::<f64>::new(&graph)
        .save(&graph, &earlier)
        .unwrap();

    // The run, journaled, is killed at the end of pass 4.
    let observer = Arc::new(CheckpointObserver::create(&journal, &graph).unwrap());
    let token = CancellationToken::new();
    let hook = {
        let token = token.clone();
        move |event: &Event| {
            if matches!(event, Event::PassFinished { pass: 4, .. }) {
                token.cancel();
            }
        }
    };
    let snake = snake_for(&graph, &goal, config(2, 2, 2))
        .cancellation(token)
        .observer(Arc::clone(&observer))
        .observer(hook);
    let mut dead = ParameterDatabase::new(&graph);
    expect_cancelled(snake.calibrate_into(&mut dead, &healthy));
    assert!(observer.take_error().is_none());
    assert_eq!(dead.steps().map(|s| s.pass).max(), Some(4));
    // The process died while writing the next step: half a line is left at the end.
    append_bytes(
        &journal,
        br#"{"type":"step","pass":5,"subgoal":0,"centre":"q3_3","parame"#,
    );
    drop(snake);
    drop(observer);

    // Recovery: the last save, plus the journal.
    let steps = read_journal(&journal, &graph).unwrap();
    assert_eq!(
        steps.len() as u64,
        dead.step_count(),
        "the torn half line is ignored"
    );
    let mut recovered = ParameterDatabase::<f64>::load(&graph, &earlier).unwrap();
    let applied = recovered
        .replay_journal(&graph, &steps, |e| healthy.options(e))
        .unwrap();
    assert_eq!(applied, steps.len());
    assert_same_history(&recovered, &dead, "replay onto an empty save");
    // Replaying again applies nothing.
    let again = recovered
        .replay_journal(&graph, &steps, |e| healthy.options(e))
        .unwrap();
    assert_eq!(again, 0);
    assert_same_history(&recovered, &dead, "second replay");

    // Resume: the cut fell on a pass boundary, so the result equals the uninterrupted run.
    let resumer = snake_for(&graph, &goal, config(2, 2, 2));
    resumer.calibrate_into(&mut recovered, &healthy).unwrap();
    assert_same_history(&recovered, &uninterrupted, "resumed after the kill");
    assert_valid(&recovered, &goal, &healthy, "resumed after the kill");
}

#[test]
fn journal_recovery_from_a_later_save_and_a_torn_last_line() {
    let (graph, goal) = lattice(7, 7);
    let healthy = collision_model(&graph, 72);
    let dir = tempfile::tempdir().unwrap();
    let (journal, earlier) = (
        dir.path().join("run.jsonl"),
        dir.path().join("earlier.json"),
    );

    // Run 1: 10 steps, journaled, then the periodic save of the database.
    let token = CancellationToken::new();
    let seen = Arc::new(AtomicUsize::new(0));
    let first = {
        let observer = CheckpointObserver::create(&journal, &graph).unwrap();
        let snake = snake_for(&graph, &goal, config(1, 2, 2))
            .cancellation(token.clone())
            .observer(observer)
            .observer(cancel_after(&token, &seen, 10));
        let mut db = ParameterDatabase::new(&graph);
        expect_cancelled(snake.calibrate_into(&mut db, &healthy));
        db
    };
    assert_eq!(first.step_count(), 10);
    first.save(&graph, &earlier).unwrap();

    // Run 2 continues the same history in the same journal and is killed after 12 more steps.
    token.reset();
    seen.store(0, Ordering::SeqCst);
    let mut dead = first.clone();
    {
        let observer = CheckpointObserver::append(&journal, &graph).unwrap();
        let snake = snake_for(&graph, &goal, config(1, 2, 2))
            .cancellation(token.clone())
            .observer(observer)
            .observer(cancel_after(&token, &seen, 12));
        expect_cancelled(snake.calibrate_into(&mut dead, &healthy));
    }
    assert_eq!(dead.step_count(), 22);

    // The kill tore the last line (the 22nd step) in half.
    let line_len = tear_last_line(&journal, 20);
    assert!(line_len > 40);
    let steps = read_journal(&journal, &graph).unwrap();
    assert_eq!(steps.len(), 21, "the torn step is lost, the rest is read");

    // Recovery from the save taken after 10 steps: the first 10 journal steps are recognised
    // as already applied, the other 11 are applied.
    let mut recovered = ParameterDatabase::<f64>::load(&graph, &earlier).unwrap();
    assert_eq!(recovered.step_count(), 10);
    let applied = recovered
        .replay_journal(&graph, &steps, |e| healthy.options(e))
        .unwrap();
    assert_eq!(applied, 11);
    assert_eq!(recovered.step_count(), 21);
    for (a, b) in recovered.steps().zip(dead.steps()) {
        assert!(
            a.same_outcome(b),
            "step {} differs after the replay",
            a.index
        );
    }
    // Everything the recovered database has, the dead one had (it only lacks the last step).
    for (id, entry) in recovered.iter() {
        let lost = dead.steps().last().unwrap();
        if !lost.parameters.contains(&id) {
            assert_eq!(dead.get(id), Some(entry));
        }
    }

    // Resume with the journal appended to (the torn line is repaired first). The result is
    // valid, and the journal now describes the whole history of the recovered database.
    {
        let observer = CheckpointObserver::append(&journal, &graph).unwrap();
        let snake = snake_for(&graph, &goal, config(1, 2, 2)).observer(Arc::new(observer));
        snake.calibrate_into(&mut recovered, &healthy).unwrap();
    }
    assert_valid(&recovered, &goal, &healthy, "resumed from the journal");
    let all = read_journal(&journal, &graph).unwrap();
    assert_eq!(all.len() as u64, recovered.step_count());
    let mut rebuilt = ParameterDatabase::<f64>::load(&graph, &earlier).unwrap();
    rebuilt
        .replay_journal(&graph, &all, |e| healthy.options(e))
        .unwrap();
    assert_same_history(
        &rebuilt,
        &recovered,
        "a second crash would recover the same state",
    );
}

#[test]
fn a_torn_last_line_is_ignored_wherever_it_is_cut() {
    let (graph, goal) = lattice(5, 5);
    let healthy = collision_model(&graph, 73);
    let dir = tempfile::tempdir().unwrap();
    let journal = dir.path().join("run.jsonl");
    let token = CancellationToken::new();
    let seen = Arc::new(AtomicUsize::new(0));
    let observer = CheckpointObserver::create(&journal, &graph).unwrap();
    let snake = snake_for(&graph, &goal, config(1, 2, 2))
        .cancellation(token.clone())
        .observer(observer)
        .observer(cancel_after(&token, &seen, 6));
    let mut dead = ParameterDatabase::new(&graph);
    expect_cancelled(snake.calibrate_into(&mut dead, &healthy));
    drop(snake);
    let intact = std::fs::read(&journal).unwrap();
    assert_eq!(read_journal(&journal, &graph).unwrap().len(), 6);

    // A cut anywhere strictly inside the last line loses that one step.
    let line_len = tear_last_line(&journal, 0);
    for keep in (0..line_len).step_by(7).chain([1, line_len - 1]) {
        std::fs::write(&journal, &intact).unwrap();
        tear_last_line(&journal, keep);
        let steps = read_journal(&journal, &graph).unwrap();
        assert_eq!(steps.len(), 5, "cut after {keep} of {line_len} bytes");
    }
    // A complete last line is a step even if its newline never reached the file.
    std::fs::write(&journal, &intact).unwrap();
    tear_last_line(&journal, line_len);
    assert_eq!(read_journal(&journal, &graph).unwrap().len(), 6);
}

#[test]
fn the_journal_of_a_parallel_run_replays_to_the_same_database() {
    let (graph, goal) = lattice(10, 10);
    let healthy = collision_model(&graph, 74);
    let cfg = SnakeConfig {
        threads: Some(4),
        ..config(1, 2, 2)
    };
    let plan = snake_for(&graph, &goal, cfg.clone())
        .stitch_plan(quadrants(&graph, &goal, 10, 10))
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let journal = dir.path().join("parallel.jsonl");

    let token = CancellationToken::new();
    let seen = Arc::new(AtomicUsize::new(0));
    let observer = Arc::new(CheckpointObserver::create(&journal, &graph).unwrap());
    let snake = snake_for(&graph, &goal, cfg.clone())
        .cancellation(token.clone())
        .observer(Arc::clone(&observer))
        .observer(cancel_after(&token, &seen, 45));
    let mut dead = ParameterDatabase::new(&graph);
    expect_cancelled(snake.stitch(&mut dead, &plan, &healthy));
    assert!(observer.take_error().is_none());
    drop(snake);

    // The steps of the quadrants interleaved in the journal, but the reader puts them back in
    // the order in which the engine merged them, so the replay equals the database.
    let steps = read_journal(&journal, &graph).unwrap();
    assert_eq!(steps.len() as u64, dead.step_count());
    // The reader returns the canonical order (the order of the history), whatever the order of
    // the lines in the file was.
    let keys: Vec<(u32, u32, u64)> = steps.iter().map(|s| (s.pass, s.subgoal, s.index)).collect();
    assert!(keys.windows(2).all(|w| w[0] < w[1]), "{keys:?}");
    let mut recovered = ParameterDatabase::<f64>::new(&graph);
    recovered
        .replay_journal(&graph, &steps, |e| healthy.options(e))
        .unwrap();
    assert_same_history(&recovered, &dead, "parallel journal");

    // The replay does not depend on the order of the slice it is given either.
    let mut reversed = steps.clone();
    reversed.reverse();
    let mut from_reversed = ParameterDatabase::<f64>::new(&graph);
    from_reversed
        .replay_journal(&graph, &reversed, |e| healthy.options(e))
        .unwrap();
    assert_same_history(&from_reversed, &dead, "reversed journal");

    // And the recovered run resumes to a valid calibration.
    snake_for(&graph, &goal, cfg)
        .calibrate_into(&mut recovered, &healthy)
        .unwrap();
    assert_valid(&recovered, &goal, &healthy, "parallel journal, resumed");
}

#[test]
fn a_journal_of_another_graph_is_refused_not_applied() {
    let (graph, goal) = lattice(4, 4);
    let healthy = collision_model(&graph, 75);
    let dir = tempfile::tempdir().unwrap();
    let journal = dir.path().join("run.jsonl");
    let observer = CheckpointObserver::create(&journal, &graph).unwrap();
    snake_for(&graph, &goal, config(1, 2, 2))
        .observer(observer)
        .calibrate(&healthy)
        .unwrap();

    let other = ProcessorGraph::square_lattice(2, 2);
    let err = read_journal(&journal, &other).unwrap_err();
    assert!(
        matches!(
            err,
            FaultTuneError::Format(_) | FaultTuneError::GraphMismatch { .. }
        ),
        "{err:?}"
    );
    // Replaying steps of the right graph onto a database of the wrong one fails too.
    let steps = read_journal(&journal, &graph).unwrap();
    let mut foreign = ParameterDatabase::<f64>::new(&other);
    let err = foreign
        .replay_journal(&other, &steps, |e| healthy.options(e))
        .unwrap_err();
    assert!(matches!(err, FaultTuneError::Format(_)), "{err:?}");
}

#[test]
fn recalibration_removals_are_journaled_and_restored_by_the_replay() {
    use faulttune::persist::{read_journal_entries, JournalEntry};

    let (graph, goal) = lattice(5, 5);
    let healthy = collision_model(&graph, 76);
    let dir = tempfile::tempdir().unwrap();
    let (journal, earlier) = (
        dir.path().join("run.jsonl"),
        dir.path().join("earlier.json"),
    );
    ParameterDatabase::<f64>::new(&graph)
        .save(&graph, &earlier)
        .unwrap();

    let observer = Arc::new(CheckpointObserver::create(&journal, &graph).unwrap());
    let snake = snake_for(&graph, &goal, config(1, 2, 2)).observer(Arc::clone(&observer));
    let mut db = snake.calibrate(&healthy).unwrap();
    snake
        .recalibrate(&mut db, &[node(&graph, 2, 2)], &healthy)
        .unwrap();
    assert!(observer.take_error().is_none());
    assert!(db
        .history()
        .iter()
        .any(|h| matches!(h, HistoryRecord::Invalidated(_))));

    let entries = read_journal_entries(&journal, &graph).unwrap();
    assert!(
        entries
            .iter()
            .any(|e| matches!(e, JournalEntry::Invalidated { .. })),
        "the removal made by recalibrate is in the journal"
    );
    let mut recovered = ParameterDatabase::<f64>::load(&graph, &earlier).unwrap();
    recovered
        .replay_entries(&graph, &entries, |e| healthy.options(e))
        .unwrap();
    assert_same_history(&recovered, &db, "journal with a recalibration");
}

#[test]
fn replaying_onto_a_database_that_is_ahead_of_the_journal_applies_nothing() {
    let (graph, goal) = lattice(6, 6);
    let healthy = collision_model(&graph, 77);
    let dir = tempfile::tempdir().unwrap();
    let journal = dir.path().join("run.jsonl");
    let observer = CheckpointObserver::create(&journal, &graph).unwrap();
    let db = snake_for(&graph, &goal, config(1, 2, 2))
        .observer(observer)
        .calibrate(&healthy)
        .unwrap();

    // The journal lost its tail (a torn last line): the database is ahead of it.
    tear_last_line(&journal, 10);
    let steps = read_journal(&journal, &graph).unwrap();
    assert_eq!(steps.len() as u64 + 1, db.step_count());
    let mut ahead = db.clone();
    let applied = ahead
        .replay_journal(&graph, &steps, |e| healthy.options(e))
        .unwrap();
    assert_eq!(
        applied, 0,
        "every step of the journal is already in the database"
    );
    assert!(ahead.same_outcome(&db));
    assert_eq!(ahead.history(), db.history());
}

#[test]
fn a_resumed_run_may_keep_a_journal_of_its_own() {
    let (graph, goal) = lattice(7, 7);
    let healthy = collision_model(&graph, 78);
    let dir = tempfile::tempdir().unwrap();
    let (first, second) = (
        dir.path().join("first.jsonl"),
        dir.path().join("second.jsonl"),
    );

    // Run 1 is killed after 15 steps; run 2 resumes on the same database with a new journal.
    let token = CancellationToken::new();
    let seen = Arc::new(AtomicUsize::new(0));
    let mut live = ParameterDatabase::new(&graph);
    {
        let snake = snake_for(&graph, &goal, config(1, 2, 2))
            .cancellation(token.clone())
            .observer(CheckpointObserver::create(&first, &graph).unwrap())
            .observer(cancel_after(&token, &seen, 15));
        expect_cancelled(snake.calibrate_into(&mut live, &healthy));
    }
    assert_eq!(live.step_count(), 15);
    snake_for(&graph, &goal, config(1, 2, 2))
        .observer(CheckpointObserver::create(&second, &graph).unwrap())
        .calibrate_into(&mut live, &healthy)
        .unwrap();
    assert_valid(&live, &goal, &healthy, "two runs, two journals");

    // Recovery from nothing at all: replay the journals, oldest first.
    let mut recovered = ParameterDatabase::<f64>::new(&graph);
    for path in [&first, &second] {
        recovered
            .replay_journal(&graph, &read_journal(path, &graph).unwrap(), |e| {
                healthy.options(e)
            })
            .unwrap();
    }
    assert_same_history(&recovered, &live, "journals replayed one after the other");

    // Recovery from a save taken after run 1: only the second journal is needed, and replaying
    // the first one as well changes nothing.
    let mut after_first = ParameterDatabase::<f64>::new(&graph);
    after_first
        .replay_journal(&graph, &read_journal(&first, &graph).unwrap(), |e| {
            healthy.options(e)
        })
        .unwrap();
    assert_eq!(after_first.step_count(), 15);
    let save = dir.path().join("after_first.json");
    after_first.save(&graph, &save).unwrap();
    let mut from_save = ParameterDatabase::<f64>::load(&graph, &save).unwrap();
    for path in [&first, &second] {
        from_save
            .replay_journal(&graph, &read_journal(path, &graph).unwrap(), |e| {
                healthy.options(e)
            })
            .unwrap();
    }
    assert_same_history(
        &from_save,
        &live,
        "recovery from a save taken between the runs",
    );
}
