//! Unit tests of the engine; the broader suites live in `tests/`.

use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::Mutex;

use super::*;
use crate::error::ModelError;
use crate::events::RecordingObserver;
use crate::graph::{EdgeType, ElementKind};
use crate::model::Step;
use crate::toy::{CollisionModel, SeparableModel};

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
        ..SnakeConfig::default()
    }
}

/// A model that wraps another and misbehaves on chosen steps.
struct Saboteur<'a> {
    inner: &'a SeparableModel,
    /// Centre element whose steps misbehave.
    centre: ElementId,
    mode: Mode,
    calls: AtomicUsize,
}

#[derive(Clone, Copy)]
enum Mode {
    /// `prepare` fails the first `n` times it is called for the centre.
    FailPrepare(usize),
    /// `error_batch` always fails for the centre.
    FailError,
    /// `error` returns NaN for the centre.
    Nan,
    /// `error` panics for the centre.
    Panic,
}

impl ErrorModel for Saboteur<'_> {
    type Value = f64;
    type Data = ();

    fn options(&self, element: ElementId) -> Vec<f64> {
        self.inner.options(element)
    }

    fn prepare(&self, step: &Step<'_, f64>) -> Result<(), ModelError> {
        if step.centre == self.centre {
            if let Mode::FailPrepare(n) = self.mode {
                if self.calls.fetch_add(1, AtomicOrdering::SeqCst) < n {
                    return Err(ModelError::new("flaky measurement"));
                }
            }
        }
        Ok(())
    }

    fn error(&self, step: &Step<'_, f64>, data: &(), candidate: &[f64]) -> f64 {
        if step.centre == self.centre {
            match self.mode {
                Mode::Nan => return f64::NAN,
                Mode::Panic => panic!("the model blew up"),
                _ => {}
            }
        }
        self.inner.error(step, data, candidate)
    }

    fn error_batch(
        &self,
        step: &Step<'_, f64>,
        data: &(),
        candidates: &crate::model::CandidateBatch<'_, f64>,
        out: &mut [f64],
    ) -> Result<(), ModelError> {
        if step.centre == self.centre && matches!(self.mode, Mode::FailError) {
            return Err(ModelError::new("the instrument is offline"));
        }
        for (slot, row) in out.iter_mut().zip(candidates.rows()) {
            *slot = self.error(step, data, row);
        }
        Ok(())
    }
}

#[test]
fn separable_model_reaches_its_optimum_for_every_distance() {
    let (graph, goal) = lattice(4, 5);
    let model = SeparableModel::new(&graph, 40, 5.0, 7.0, 1);
    for d_p in 0..=2 {
        for d_t in 1..=2 {
            let snake = Snake::new(&graph, goal.clone(), config(d_p, 2, d_t));
            let (db, report) = snake.calibrate_with_report(&model).unwrap();
            assert_eq!(db.calibrated(), &goal, "d_P={d_p} d_T={d_t}");
            for element in goal.iter() {
                assert_eq!(db.value(element), Some(&model.optimum(element)));
            }
            assert!(report.skipped.is_empty());
            assert_eq!(report.steps, db.step_count());
        }
    }
}

#[test]
fn empty_and_tiny_graphs() {
    let empty = ProcessorGraph::square_lattice(0, 0);
    let snake = Snake::new(&empty, empty.all_elements(), SnakeConfig::default());
    let model = SeparableModel::new(&empty, 4, 0.0, 1.0, 0);
    let (db, report) = snake.calibrate_with_report(&model).unwrap();
    assert!(db.is_empty());
    assert_eq!((report.passes, report.steps), (0, 0));

    let (one, goal) = lattice(1, 1);
    let model = SeparableModel::new(&one, 4, 0.0, 1.0, 0);
    let db = Snake::new(&one, goal, SnakeConfig::default())
        .calibrate(&model)
        .unwrap();
    assert_eq!(db.len(), 1);
}

#[test]
fn threads_do_not_change_the_result() {
    let (graph, goal) = lattice(10, 10);
    let model = SeparableModel::new(&graph, 30, 5.0, 7.0, 3);
    // Scattered drifted elements give many small subgoals.
    let reference = {
        let snake = Snake::new(
            &graph,
            goal.clone(),
            SnakeConfig {
                threads: Some(1),
                ..config(1, 2, 2)
            },
        );
        snake.calibrate(&model).unwrap()
    };
    let drifted: Vec<ElementId> = [0, 9, 44, 55, 90, 99]
        .iter()
        .map(|&i| graph.lattice_node(i / 10, i % 10).unwrap())
        .collect();
    let mut outcomes = Vec::new();
    for threads in [1usize, 2, 4, 8] {
        let cfg = SnakeConfig {
            threads: Some(threads),
            seed: 11,
            ..config(1, 2, 2)
        };
        let snake = Snake::new(&graph, goal.clone(), cfg);
        let mut db = reference.clone();
        snake.recalibrate(&mut db, &drifted, &model).unwrap();
        // A stitched run from scratch with four quadrants.
        let plan = snake.stitch_plan(quadrants(&graph, &goal, 10)).unwrap();
        let mut stitched = ParameterDatabase::new(&graph);
        snake.stitch(&mut stitched, &plan, &model).unwrap();
        outcomes.push((db, stitched));
    }
    for (db, stitched) in &outcomes[1..] {
        assert!(
            db.same_outcome(&outcomes[0].0),
            "recalibration differs between thread counts"
        );
        assert!(
            stitched.same_outcome(&outcomes[0].1),
            "stitched run differs between thread counts"
        );
    }
}

/// Four quadrants of an `n × n` lattice, as goal subsets.
fn quadrants(graph: &ProcessorGraph, goal: &ElementSet, n: usize) -> Vec<ElementSet> {
    let mut regions = vec![ElementSet::new(graph.len()); 4];
    let position = |label: &str| -> (usize, usize) {
        // Labels are q{r}_{c} or q{r}_{c}-q{r2}_{c2}: use the first endpoint.
        let first = label.split('-').next().unwrap();
        let (r, c) = first[1..].split_once('_').unwrap();
        (r.parse().unwrap(), c.parse().unwrap())
    };
    for element in goal.iter() {
        let (r, c) = position(graph.label(element));
        regions[usize::from(r >= n / 2) * 2 + usize::from(c >= n / 2)].insert(element);
    }
    regions
}

#[test]
fn stitching_runs_regions_as_separate_subgoals() {
    let (graph, goal) = lattice(12, 12);
    let model = SeparableModel::new(&graph, 20, 5.0, 7.0, 5);
    let observer = RecordingObserver::new();
    let snake = Snake::new(&graph, goal.clone(), config(1, 2, 2)).observer(observer.clone());
    let plan = snake.stitch_plan(quadrants(&graph, &goal, 12)).unwrap();
    let mut db = ParameterDatabase::new(&graph);
    let report = snake.stitch(&mut db, &plan, &model).unwrap();
    assert_eq!(db.calibrated(), &goal);
    let first_pass_subgoals = observer
        .events()
        .iter()
        .find_map(|e| match e {
            Event::PassStarted { subgoals, .. } => Some(*subgoals),
            _ => None,
        })
        .unwrap();
    assert!(
        first_pass_subgoals >= 4,
        "stage one should split into the quadrants"
    );
    assert!(report.passes >= 2, "both stages ran");
}

#[test]
fn same_seed_same_result() {
    let (graph, goal) = lattice(6, 6);
    let model = SeparableModel::new(&graph, 25, 5.0, 7.0, 2);
    let run = |seed| {
        let cfg = SnakeConfig {
            seed,
            ..config(1, 2, 2)
        };
        Snake::new(&graph, goal.clone(), cfg)
            .heuristic(crate::heuristics::Random)
            .calibrate(&model)
            .unwrap()
    };
    assert!(run(5).same_outcome(&run(5)));
    assert!(run(5).same_outcome(&run(5)));
}

#[test]
fn cross_step_cache_does_not_change_results() {
    let (graph, goal) = lattice(6, 6);
    let model = CollisionModel::new(&graph, 60, 5.0, 7.0, 0.02, 2, 4);
    let plain = Snake::new(&graph, goal.clone(), config(1, 2, 2))
        .calibrate(&model)
        .unwrap();
    let cached_cfg = SnakeConfig {
        cache: CachePolicy::across_steps(),
        ..config(1, 2, 2)
    };
    let cached_snake = Snake::new(&graph, goal.clone(), cached_cfg);
    let (cached, report) = cached_snake.calibrate_with_report(&model).unwrap();
    assert!(plain.same_outcome(&cached));
    assert!(report.cache_misses > 0);
    // A cache shared between two runs is hit by the second one.
    let shared = Arc::new(CrossStepCache::new(1 << 16));
    let snake = Snake::new(&graph, goal.clone(), config(1, 2, 2)).shared_cache(shared.clone());
    let (_, first) = snake.calibrate_with_report(&model).unwrap();
    let (again, second) = snake.calibrate_with_report(&model).unwrap();
    assert!(again.same_outcome(&plain));
    assert!(second.cache_hits > first.cache_hits);
    assert_eq!(
        second.cache_misses, 0,
        "everything was cached by the first run"
    );
}

#[test]
fn collision_model_is_collision_free() {
    let (graph, goal) = lattice(7, 7);
    let model = CollisionModel::new(&graph, 150, 5.0, 7.0, 0.01, 1, 9);
    assert!(
        150 >= model.required_options(),
        "needs {}",
        model.required_options()
    );
    for d_p in 0..=2 {
        let db = Snake::new(&graph, goal.clone(), config(d_p, 2, 2))
            .calibrate(&model)
            .unwrap();
        assert!(model.collisions(&db).is_empty(), "d_P = {d_p}");
        assert!(db.steps().all(|s| s.error.is_finite()));
    }
}

fn saboteur(model: &SeparableModel, centre: ElementId, mode: Mode) -> Saboteur<'_> {
    Saboteur {
        inner: model,
        centre,
        mode,
        calls: AtomicUsize::new(0),
    }
}

#[test]
fn abort_keeps_committed_steps_and_names_the_centre() {
    let (graph, goal) = lattice(5, 5);
    let base = SeparableModel::new(&graph, 20, 5.0, 7.0, 1);
    let centre = graph.lattice_node(2, 2).unwrap();
    let model = saboteur(&base, centre, Mode::FailError);
    let snake = Snake::new(&graph, goal.clone(), config(1, 2, 2));
    let mut db = ParameterDatabase::new(&graph);
    let err = snake.calibrate_into(&mut db, &model).unwrap_err();
    match err {
        FaultTuneError::Model { element, source } => {
            assert_eq!(element, centre);
            assert!(source.to_string().contains("offline"));
        }
        other => panic!("unexpected error {other:?}"),
    }
    assert!(
        !db.is_empty(),
        "steps committed before the failure are kept"
    );
    assert!(!db.is_calibrated(centre));
    // Resuming with a healthy model finishes the job.
    snake.calibrate_into(&mut db, &base).unwrap();
    assert_eq!(db.calibrated(), &goal);
}

#[test]
fn retry_survives_a_flaky_prepare_and_gives_up_eventually() {
    let (graph, goal) = lattice(4, 4);
    let base = SeparableModel::new(&graph, 20, 5.0, 7.0, 1);
    let centre = graph.lattice_node(1, 1).unwrap();
    let flaky = saboteur(&base, centre, Mode::FailPrepare(2));
    let cfg = SnakeConfig {
        on_model_error: ModelErrorPolicy::Retry { times: 2 },
        ..config(1, 2, 2)
    };
    let db = Snake::new(&graph, goal.clone(), cfg)
        .calibrate(&flaky)
        .unwrap();
    assert_eq!(db.calibrated(), &goal);
    // One retry fewer than the failures is not enough.
    let flaky = saboteur(&base, centre, Mode::FailPrepare(2));
    let cfg = SnakeConfig {
        on_model_error: ModelErrorPolicy::Retry { times: 1 },
        ..config(1, 2, 2)
    };
    let err = Snake::new(&graph, goal, cfg).calibrate(&flaky).unwrap_err();
    assert!(matches!(err, FaultTuneError::Model { .. }));
}

#[test]
fn skip_excludes_the_element_and_reports_it() {
    let (graph, goal) = lattice(4, 4);
    let base = SeparableModel::new(&graph, 20, 5.0, 7.0, 1);
    let centre = graph.lattice_node(1, 2).unwrap();
    let model = saboteur(&base, centre, Mode::FailError);
    let cfg = SnakeConfig {
        on_model_error: ModelErrorPolicy::SkipElement,
        ..config(0, 2, 2)
    };
    let observer = RecordingObserver::new();
    let snake = Snake::new(&graph, goal.clone(), cfg).observer(observer.clone());
    let (db, report) = snake.calibrate_with_report(&model).unwrap();
    assert_eq!(report.skipped, vec![centre]);
    assert!(!db.is_calibrated(centre));
    assert_eq!(db.len(), goal.len() - 1);
    let failed = observer
        .events()
        .iter()
        .filter(|e| matches!(e, Event::StepFailed { centre: c, .. } if *c == centre))
        .count();
    assert_eq!(
        failed, 1,
        "the skipped element is tried once, not once per pass"
    );
}

#[test]
fn nan_is_always_fatal() {
    let (graph, goal) = lattice(4, 4);
    let base = SeparableModel::new(&graph, 20, 5.0, 7.0, 1);
    let centre = graph.lattice_node(1, 1).unwrap();
    let model = saboteur(&base, centre, Mode::Nan);
    for policy in [ModelErrorPolicy::Abort, ModelErrorPolicy::SkipElement] {
        let cfg = SnakeConfig {
            on_model_error: policy,
            ..config(1, 2, 2)
        };
        let mut db = ParameterDatabase::new(&graph);
        let err = Snake::new(&graph, goal.clone(), cfg)
            .calibrate_into(&mut db, &model)
            .unwrap_err();
        assert!(
            matches!(err, FaultTuneError::NonFiniteError { element } if element == centre),
            "{err:?}"
        );
    }
}

#[test]
fn a_panicking_model_keeps_committed_steps_and_reraises() {
    let (graph, goal) = lattice(5, 5);
    let base = SeparableModel::new(&graph, 20, 5.0, 7.0, 1);
    let centre = graph.lattice_node(3, 3).unwrap();
    let model = saboteur(&base, centre, Mode::Panic);
    let snake = Snake::new(&graph, goal.clone(), config(1, 2, 2));
    let db = Mutex::new(ParameterDatabase::new(&graph));
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut guard = db.lock().unwrap_or_else(|e| e.into_inner());
        let _ = snake.calibrate_into(&mut guard, &model);
    }));
    let payload = outcome.expect_err("the panic must propagate");
    let message = payload
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_default();
    assert!(message.contains("blew up"), "{message}");
    let db = db.into_inner().unwrap_or_else(|e| e.into_inner());
    assert!(!db.is_empty(), "steps before the panic were merged");
    assert!(!db.is_calibrated(centre));
}

#[test]
fn cancellation_returns_a_partial_report_and_resume_completes() {
    let (graph, goal) = lattice(8, 8);
    let model = SeparableModel::new(&graph, 20, 5.0, 7.0, 1);
    let token = CancellationToken::new();
    let seen = Arc::new(AtomicUsize::new(0));
    let observer = {
        let (token, seen) = (token.clone(), Arc::clone(&seen));
        move |event: &Event| {
            if matches!(event, Event::StepCommitted(_))
                && seen.fetch_add(1, AtomicOrdering::SeqCst) == 4
            {
                token.cancel();
            }
        }
    };
    let finished = RecordingObserver::new();
    let snake = Snake::new(&graph, goal.clone(), config(1, 2, 2))
        .cancellation(token.clone())
        .observer(observer)
        .observer(finished.clone());
    let mut db = ParameterDatabase::new(&graph);
    let err = snake.calibrate_into(&mut db, &model).unwrap_err();
    let FaultTuneError::Cancelled { report } = err else {
        panic!("expected Cancelled")
    };
    assert!(report.steps >= 5 && (report.steps as usize) < goal.len());
    assert_eq!(report.steps, db.step_count());
    assert!(matches!(finished.events().last(), Some(Event::Finished(_))));
    // Resume.
    token.reset();
    let report = snake.calibrate_into(&mut db, &model).unwrap();
    assert_eq!(db.calibrated(), &goal);
    assert!(report.steps > 0);
    for element in goal.iter() {
        assert_eq!(db.value(element), Some(&model.optimum(element)));
    }
}

#[test]
fn recalibrate_changes_only_its_region() {
    let (graph, goal) = lattice(7, 7);
    let model = CollisionModel::new(&graph, 150, 5.0, 7.0, 0.01, 1, 3);
    let snake = Snake::new(&graph, goal.clone(), config(1, 2, 2));
    let mut db = snake.calibrate(&model).unwrap();
    let before = db.clone();
    let drifted = graph.lattice_node(3, 3).unwrap();
    let observer = RecordingObserver::new();
    let snake = Snake::new(&graph, goal.clone(), config(1, 2, 2)).observer(observer.clone());
    let report = snake.recalibrate(&mut db, &[drifted], &model).unwrap();
    let region: ElementSet = {
        let hood = Neighbourhoods::build(&graph, 1);
        ElementSet::from_ids(
            graph.len(),
            hood.within(drifted, 1)
                .iter()
                .copied()
                .filter(|&e| goal.contains(e)),
        )
    };
    assert!(report.steps > 0);
    assert_eq!(db.calibrated(), &goal);
    for element in goal.iter() {
        if !region.contains(element) {
            assert_eq!(
                db.get(element),
                before.get(element),
                "outside the region: {element:?}"
            );
        }
    }
    assert!(model.collisions(&db).is_empty());
    let invalidated: Vec<_> = db
        .history()
        .iter()
        .filter_map(|h| match h {
            HistoryRecord::Invalidated(elements) => Some(elements.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(invalidated.len(), 1);
    assert_eq!(invalidated[0], region.to_vec());
    assert!(observer
        .events()
        .iter()
        .any(|e| matches!(e, Event::Invalidated { elements, .. } if *elements == region.to_vec())));
    // Old steps are never rewritten: the history only grew.
    assert!(db.history().starts_with(before.history()));
}

#[test]
fn invalid_inputs_are_errors_not_panics() {
    let (graph, goal) = lattice(4, 4);
    let model = SeparableModel::new(&graph, 10, 5.0, 7.0, 1);
    let bad = |cfg: SnakeConfig| {
        Snake::new(&graph, goal.clone(), cfg)
            .calibrate(&model)
            .unwrap_err()
    };
    assert!(matches!(
        bad(SnakeConfig {
            threads: Some(0),
            ..config(1, 2, 2)
        }),
        FaultTuneError::InvalidConfig(_)
    ));
    assert!(matches!(
        bad(SnakeConfig {
            evaluation_budget: 0,
            ..config(1, 2, 2)
        }),
        FaultTuneError::InvalidConfig(_)
    ));
    assert!(matches!(
        bad(config(1, 2, 1000)),
        FaultTuneError::InvalidConfig(_)
    ));
    let acr = SnakeConfig {
        cache: CachePolicy::AcrossSteps { capacity: 0 },
        ..config(1, 2, 2)
    };
    assert!(matches!(bad(acr), FaultTuneError::InvalidConfig(_)));

    // Wrong sizes.
    let other = ProcessorGraph::square_lattice(2, 2);
    let wrong_goal = Snake::new(&graph, other.all_elements(), SnakeConfig::default());
    assert!(matches!(
        wrong_goal.calibrate(&model),
        Err(FaultTuneError::InvalidConfig(_))
    ));

    // A database of another graph.
    let snake = Snake::new(&graph, goal.clone(), SnakeConfig::default());
    let mut foreign = ParameterDatabase::<f64>::new(&other);
    assert!(snake.calibrate_into(&mut foreign, &model).is_err());

    // Goal elements that are never active.
    let nobody = AlgorithmSubgraphs::new(vec![ElementSet::new(graph.len())]).unwrap();
    let inactive = Snake::new(&graph, goal.clone(), SnakeConfig::default()).algorithm(nobody);
    assert!(matches!(
        inactive.calibrate(&model),
        Err(FaultTuneError::InvalidConfig(_))
    ));

    // Recalibrating something outside the goal changes nothing.
    let mut db = snake.calibrate(&model).unwrap();
    let before = db.clone();
    let parasitic = graph.edges().find(|&e| !goal.contains(e)).unwrap();
    assert!(snake.recalibrate(&mut db, &[parasitic], &model).is_err());
    assert!(snake
        .recalibrate(&mut db, &[ElementId::new(9999)], &model)
        .is_err());
    assert!(db.same_outcome(&before));
}

#[test]
fn events_are_consistent_with_the_report() {
    let (graph, goal) = lattice(5, 5);
    let model = SeparableModel::new(&graph, 15, 5.0, 7.0, 1);
    let observer = RecordingObserver::new();
    let snake = Snake::new(&graph, goal.clone(), config(1, 2, 2)).observer(observer.clone());
    let (_, report) = snake.calibrate_with_report(&model).unwrap();
    let events = observer.events();
    let committed = events
        .iter()
        .filter(|e| matches!(e, Event::StepCommitted(_)))
        .count();
    assert_eq!(committed as u64, report.steps);
    let started = events
        .iter()
        .filter(|e| matches!(e, Event::PassStarted { .. }))
        .count();
    let finished = events
        .iter()
        .filter(|e| matches!(e, Event::PassFinished { .. }))
        .count();
    assert_eq!(
        (started, finished),
        (report.passes as usize, report.passes as usize)
    );
    assert!(matches!(
        events.first(),
        Some(Event::PassStarted { pass: 0, .. })
    ));
    let Some(Event::Finished(last)) = events.last() else {
        panic!("no Finished event")
    };
    assert_eq!(last.steps, report.steps);
    assert!(report.worst_steps.len() <= MAX_WORST_STEPS);
    assert!(report
        .worst_steps
        .windows(2)
        .all(|w| w[0].error >= w[1].error));
    let calibrated: usize = events
        .iter()
        .filter_map(|e| match e {
            Event::PassFinished { calibrated, .. } => Some(*calibrated),
            _ => None,
        })
        .sum();
    assert_eq!(calibrated, goal.len());
}

#[test]
fn the_snake_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Snake<'static>>();
    assert_send_sync::<CancellationToken>();
}

#[test]
fn a_seed_outside_its_thread_cannot_loop_forever() {
    let (graph, goal) = lattice(3, 3);
    let model = SeparableModel::new(&graph, 10, 5.0, 7.0, 1);
    // A custom policy that always names an element that is already calibrated.
    let first = graph.lattice_node(0, 0).unwrap();
    let policy = SeedPolicy::Custom(Arc::new(move |_, _| first));
    let snake = Snake::new(&graph, goal, config(0, 2, 2)).seed_policy(policy);
    let result = snake.calibrate(&model);
    // Either it completes (the seed is in the thread that contains it) or it reports no progress.
    assert!(result.is_ok() || matches!(result, Err(FaultTuneError::NoProgress)));
}

#[test]
fn resuming_on_a_pass_boundary_equals_an_uninterrupted_run() {
    // With d_P = 2 the first step of a thread absorbs everything its traversal could reach, so
    // each pass makes few steps and a 5x5 lattice needs many passes.
    let (graph, goal) = lattice(8, 8);
    let model = CollisionModel::new(&graph, 200, 5.0, 7.0, 0.01, 1, 6);
    let cfg = SnakeConfig {
        seed: 17,
        ..config(2, 2, 2)
    };
    let build = |token: Option<CancellationToken>| {
        let mut snake = Snake::new(&graph, goal.clone(), cfg.clone())
            .heuristic(crate::heuristics::Random)
            .seed_policy(SeedPolicy::Random)
            .optimizer(crate::optimizers::RandomSearch::default());
        if let Some(token) = token {
            snake = snake.cancellation(token);
        }
        snake
    };
    let uninterrupted = build(None).calibrate(&model).unwrap();
    let mut passes: Vec<u32> = uninterrupted.steps().map(|s| s.pass).collect();
    passes.dedup();
    assert!(
        passes.len() > 3,
        "the setup should need several passes: {passes:?}"
    );
    assert!(passes.windows(2).all(|w| w[1] == w[0] + 1), "{passes:?}");

    for cut in [0u32, 1, 2] {
        let token = CancellationToken::new();
        let hook = {
            let token = token.clone();
            move |event: &Event| {
                if matches!(event, Event::PassFinished { pass, .. } if *pass == cut) {
                    token.cancel();
                }
            }
        };
        let snake = build(Some(token.clone())).observer(hook);
        let mut db = ParameterDatabase::new(&graph);
        let err = snake.calibrate_into(&mut db, &model).unwrap_err();
        assert!(matches!(err, FaultTuneError::Cancelled { .. }));
        assert_eq!(db.steps().map(|s| s.pass).max(), Some(cut));
        token.reset();
        snake.calibrate_into(&mut db, &model).unwrap();
        assert!(db.same_outcome(&uninterrupted), "cut after pass {cut}");
    }
}

/// A model whose steps fail slowly for one centre and panic at once for another.
struct Duel<'a> {
    inner: &'a SeparableModel,
    fails: ElementId,
    panics: ElementId,
}

impl ErrorModel for Duel<'_> {
    type Value = f64;
    type Data = ();

    fn options(&self, element: ElementId) -> Vec<f64> {
        self.inner.options(element)
    }

    fn prepare(&self, step: &Step<'_, f64>) -> Result<(), ModelError> {
        if step.centre == self.fails {
            std::thread::sleep(std::time::Duration::from_millis(150));
            return Err(ModelError::new("the earlier subgoal fails"));
        }
        if step.centre == self.panics {
            panic!("the later subgoal panics");
        }
        Ok(())
    }

    fn error(&self, step: &Step<'_, f64>, data: &(), candidate: &[f64]) -> f64 {
        self.inner.error(step, data, candidate)
    }
}

#[test]
fn the_earliest_failing_subgoal_decides_at_every_thread_count() {
    let (graph, goal) = lattice(10, 10);
    let base = SeparableModel::new(&graph, 10, 5.0, 7.0, 1);
    let first = graph.lattice_node(0, 0).unwrap();
    let last = graph.lattice_node(9, 9).unwrap();
    let duel = Duel {
        inner: &base,
        fails: first,
        panics: last,
    };
    for threads in [1usize, 2, 4] {
        let cfg = SnakeConfig {
            threads: Some(threads),
            ..config(0, 2, 2)
        };
        let snake = Snake::new(&graph, goal.clone(), cfg);
        let mut db = snake.calibrate(&base).unwrap();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            snake.recalibrate(&mut db, &[first, last], &duel)
        }));
        match outcome {
            Ok(Err(FaultTuneError::Model { element, .. })) => assert_eq!(element, first),
            Ok(other) => panic!("threads={threads}: unexpected result {other:?}"),
            Err(_) => panic!("threads={threads}: the later panic beat the earlier error"),
        }
    }
}

#[test]
fn reconfiguring_a_used_snake_rebuilds_its_activity_masks() {
    let (graph, goal) = lattice(4, 4);
    let model = SeparableModel::new(&graph, 10, 5.0, 7.0, 1);
    let snake = Snake::new(&graph, goal.clone(), SnakeConfig::default());
    snake.calibrate(&model).unwrap();
    let nobody = AlgorithmSubgraphs::new(vec![ElementSet::new(graph.len())]).unwrap();
    let snake = snake.algorithm(nobody);
    assert!(matches!(
        snake.calibrate(&model),
        Err(FaultTuneError::InvalidConfig(_))
    ));
}

#[test]
fn evaluations_do_not_depend_on_the_cache() {
    let (graph, goal) = lattice(5, 5);
    let model = CollisionModel::new(&graph, 120, 5.0, 7.0, 0.01, 1, 2);
    let plain = Snake::new(&graph, goal.clone(), config(1, 2, 2))
        .calibrate_with_report(&model)
        .unwrap();
    let cached_cfg = SnakeConfig {
        cache: CachePolicy::across_steps(),
        ..config(1, 2, 2)
    };
    let cached = Snake::new(&graph, goal.clone(), cached_cfg)
        .calibrate_with_report(&model)
        .unwrap();
    assert!(plain.1.evaluations > 0);
    assert_eq!(plain.1.evaluations, cached.1.evaluations);
    let per_step: Vec<u64> = plain.0.steps().map(|s| s.evaluations).collect();
    assert_eq!(
        per_step,
        cached.0.steps().map(|s| s.evaluations).collect::<Vec<_>>()
    );
}

#[test]
fn stitch_emits_one_finished_event_with_the_combined_report() {
    let (graph, goal) = lattice(8, 8);
    let model = SeparableModel::new(&graph, 10, 5.0, 7.0, 1);
    let observer = RecordingObserver::new();
    let snake = Snake::new(&graph, goal.clone(), config(1, 2, 2)).observer(observer.clone());
    let plan = snake.stitch_plan(quadrants(&graph, &goal, 8)).unwrap();
    let mut db = ParameterDatabase::new(&graph);
    let report = snake.stitch(&mut db, &plan, &model).unwrap();
    let finished: Vec<_> = observer
        .events()
        .into_iter()
        .filter_map(|e| match e {
            Event::Finished(r) => Some(r),
            _ => None,
        })
        .collect();
    assert_eq!(finished.len(), 1);
    assert_eq!(finished[0].steps, report.steps);
    assert_eq!(report.steps, db.step_count());
}
