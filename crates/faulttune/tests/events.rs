//! The observer interface (ARCHITECTURE sections 5.4 and 11): which events a calibration run
//! emits, in which order and how many, what the `Report` says, and which invariants every
//! `StepRecord` satisfies. Everything is checked through the public API against the
//! definitions of the paper (parameters within `d_P` of the centre, constraints within `d_R` of
//! a parameter and co-active with it, calibrated before the step).
//!
//! The file is self-contained.

use std::collections::{BTreeMap, BTreeSet};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use faulttune::heuristics::{BreadthFirst, DepthFirst, Random};
use faulttune::toy::{CollisionModel, SeparableModel};
use faulttune::{
    ActivityMasks, AlgorithmSubgraphs, CachePolicy, CancellationToken, CrossStepCache, EdgeType,
    ElementId, ElementKind, ElementSet, ErrorModel, Event, FaultTuneError, HistoryRecord,
    ModelError, ModelErrorPolicy, Neighbourhoods, Observer, ParameterDatabase, ProcessorGraph,
    RecordingObserver, Report, Snake, SnakeConfig, Step, StepRecord, TraversalHeuristic,
};

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

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
        seed: 3,
        ..SnakeConfig::default()
    }
}

fn node(graph: &ProcessorGraph, r: usize, c: usize) -> ElementId {
    graph.lattice_node(r, c).expect("a node of the lattice")
}

fn steps_of(events: &[Event]) -> Vec<StepRecord> {
    events
        .iter()
        .filter_map(|e| match e {
            Event::StepCommitted(s) => Some(s.clone()),
            _ => None,
        })
        .collect()
}

fn count(events: &[Event], pick: impl Fn(&Event) -> bool) -> usize {
    events.iter().filter(|e| pick(e)).count()
}

/// The outcome of a step without the timings and the index (events carry the ordinal within
/// the subgoal, the database the global index).
type Outcome = (
    u32,
    u32,
    ElementId,
    Vec<ElementId>,
    Vec<u32>,
    Vec<ElementId>,
    u64,
    u64,
);

fn outcome(s: &StepRecord) -> Outcome {
    (
        s.pass,
        s.subgoal,
        s.centre,
        s.parameters.clone(),
        s.option_indices.clone(),
        s.constraints.clone(),
        s.error.to_bits(),
        s.evaluations,
    )
}

/// The recorded events of a plain run, with what is needed to check them.
struct Run {
    events: Vec<Event>,
    db: ParameterDatabase<f64>,
    report: Report,
}

// ---------------------------------------------------------------------------------------------
// The checker
// ---------------------------------------------------------------------------------------------

/// What the checker needs to know about the run it checks.
struct Expect<'a> {
    graph: &'a ProcessorGraph,
    goal: &'a ElementSet,
    config: &'a SnakeConfig,
    masks: &'a ActivityMasks,
    model: &'a SeparableModel,
}

/// Checks the event stream of one `calibrate` call from an empty database and returns the
/// steps in event order. Panics, with `ctx`, if anything is off.
///
/// The invariants checked, for every `StepCommitted(s)`:
///
/// * `s.pass` is the pass in progress; `s.index` counts the steps of its subgoal from 0;
/// * the centre is a goal element that is not calibrated yet, and the first parameter;
/// * the parameters are distinct, uncalibrated goal elements within `d_P` of the centre;
/// * the constraints are ascending, calibrated *before* the step (in an earlier pass, or
///   earlier in the same subgoal), and they are exactly the calibrated elements within `d_R` of a
///   parameter that can be active at the same time as it;
/// * one option index per parameter, within the element's options;
/// * `prepare_time + error_time <= duration`, and at least one evaluation.
fn check_stream(x: &Expect<'_>, events: &[Event], ctx: &str) -> Vec<StepRecord> {
    let d = x.config;
    let hoods = Neighbourhoods::build(
        x.graph,
        d.parameter_distance
            .max(d.constraint_distance)
            .max(d.traversal_distance)
            .max(1),
    );
    let n = x.graph.len();
    // Calibrated before the pass in progress, plus the writes of each subgoal of the pass.
    let mut base = ElementSet::new(n);
    let mut overlays: BTreeMap<u32, ElementSet> = BTreeMap::new();
    let mut ordinals: BTreeMap<u32, u64> = BTreeMap::new();
    let mut open: Option<(u32, usize)> = None; // the pass in progress and its step count
    let mut pass_params = 0usize;
    let mut next_pass = 0u32;
    let mut steps = Vec::new();
    let mut finished = 0;
    let mut last_was_finished = false;

    for (i, event) in events.iter().enumerate() {
        assert!(!last_was_finished, "{ctx}: event {i} follows Finished");
        match event {
            Event::PassStarted {
                pass,
                subgoals,
                uncalibrated,
            } => {
                assert!(
                    open.is_none(),
                    "{ctx}: pass {pass} starts inside another pass"
                );
                assert_eq!(*pass, next_pass, "{ctx}: passes are numbered 0, 1, 2, ...");
                assert!(*subgoals >= 1, "{ctx}");
                assert_eq!(
                    *uncalibrated,
                    x.goal.len() - base.len(),
                    "{ctx}: pass {pass} starts with the rest of the goal"
                );
                assert!(*subgoals <= *uncalibrated, "{ctx}");
                open = Some((*pass, 0));
                pass_params = 0;
                overlays.clear();
                ordinals.clear();
            }
            Event::StepCommitted(s) => {
                let (pass, in_pass) = open.as_mut().unwrap_or_else(|| {
                    panic!("{ctx}: a step is committed outside any pass (event {i})")
                });
                *in_pass += 1;
                assert_eq!(s.pass, *pass, "{ctx}");
                let overlay = overlays
                    .entry(s.subgoal)
                    .or_insert_with(|| ElementSet::new(n));
                let ordinal = ordinals.entry(s.subgoal).or_insert(0);
                assert_eq!(
                    s.index, *ordinal,
                    "{ctx}: the ordinal within subgoal {}",
                    s.subgoal
                );
                *ordinal += 1;
                let before = base.union(overlay);
                let at = format!(
                    "{ctx}: pass {} subgoal {} step {} at {}",
                    s.pass,
                    s.subgoal,
                    s.index,
                    x.graph.label(s.centre)
                );

                // Centre and parameters.
                assert!(
                    x.goal.contains(s.centre) && !before.contains(s.centre),
                    "{at}"
                );
                assert_eq!(
                    s.parameters.first(),
                    Some(&s.centre),
                    "{at}: the centre comes first"
                );
                assert_eq!(
                    set(&s.parameters).len(),
                    s.parameters.len(),
                    "{at}: distinct parameters"
                );
                for &p in &s.parameters {
                    assert!(
                        x.goal.contains(p),
                        "{at}: {} is not in the goal",
                        x.graph.label(p)
                    );
                    assert!(
                        !before.contains(p),
                        "{at}: {} was calibrated before",
                        x.graph.label(p)
                    );
                    assert!(
                        hoods.is_within(s.centre, p, d.parameter_distance),
                        "{at}: parameter {} is farther than d_P = {} from the centre",
                        x.graph.label(p),
                        d.parameter_distance
                    );
                }
                if d.parameter_distance == 0 {
                    assert_eq!(s.parameters, vec![s.centre], "{at}");
                }
                assert_eq!(s.option_indices.len(), s.parameters.len(), "{at}");
                for (&p, &k) in s.parameters.iter().zip(&s.option_indices) {
                    assert!((k as usize) < x.model.options(p).len(), "{at}");
                }

                // Constraints.
                assert!(
                    s.constraints.windows(2).all(|w| w[0] < w[1]),
                    "{at}: ascending"
                );
                let expected: BTreeSet<ElementId> = before
                    .iter()
                    .filter(|&c| {
                        s.parameters.iter().any(|&p| {
                            hoods.is_within(c, p, d.constraint_distance) && x.masks.co_active(c, p)
                        })
                    })
                    .collect();
                for &c in &s.constraints {
                    assert!(
                        before.contains(c),
                        "{at}: constraint {} is not calibrated yet",
                        x.graph.label(c)
                    );
                    assert!(
                        s.parameters
                            .iter()
                            .any(|&p| hoods.is_within(c, p, d.constraint_distance)),
                        "{at}: constraint {} is farther than d_R = {} from every parameter",
                        x.graph.label(c),
                        d.constraint_distance
                    );
                    assert!(!s.parameters.contains(&c), "{at}");
                }
                assert_eq!(
                    set(&s.constraints),
                    expected,
                    "{at}: the constraints are exactly the definition"
                );

                // Bookkeeping.
                assert!(s.evaluations >= 1, "{at}");
                assert!(
                    s.error.is_finite() && s.error >= 0.0,
                    "{at}: error {}",
                    s.error
                );
                assert!(s.prepare_time + s.error_time <= s.duration, "{at}: timings");
                pass_params += s.parameters.len();
                for &p in &s.parameters {
                    overlay.insert(p);
                }
                steps.push(s.clone());
            }
            Event::StepFailed { .. } => panic!("{ctx}: unexpected StepFailed"),
            Event::Invalidated { .. } => panic!("{ctx}: unexpected Invalidated"),
            Event::PassFinished { pass, calibrated } => {
                let (open_pass, in_pass) = open
                    .take()
                    .unwrap_or_else(|| panic!("{ctx}: PassFinished without a pass"));
                assert_eq!(*pass, open_pass, "{ctx}");
                assert!(in_pass >= 1, "{ctx}: every pass commits at least one step");
                assert_eq!(
                    *calibrated, pass_params,
                    "{ctx}: pass {pass} calibrated the parameters of its steps"
                );
                for overlay in overlays.values() {
                    base.union_with(overlay);
                }
                next_pass += 1;
            }
            Event::Finished(_) => {
                assert!(open.is_none(), "{ctx}: Finished inside a pass");
                finished += 1;
                last_was_finished = true;
            }
        }
    }
    assert_eq!(
        finished, 1,
        "{ctx}: exactly one Finished, and it is the last event"
    );
    assert!(last_was_finished, "{ctx}");
    assert_eq!(
        &base, x.goal,
        "{ctx}: every goal element is calibrated at the end"
    );
    // From scratch, every goal element is a parameter of exactly one step.
    let all: Vec<ElementId> = steps
        .iter()
        .flat_map(|s| s.parameters.iter().copied())
        .collect();
    assert_eq!(all.len(), x.goal.len(), "{ctx}");
    assert_eq!(set(&all).len(), all.len(), "{ctx}");
    steps
}

fn set(ids: &[ElementId]) -> BTreeSet<ElementId> {
    ids.iter().copied().collect()
}

/// The report, the events, and the database tell the same story.
fn check_report(x: &Expect<'_>, run: &Run, steps: &[StepRecord], ctx: &str) {
    let Run { events, db, report } = run;
    let started = count(events, |e| matches!(e, Event::PassStarted { .. }));
    let finished = count(events, |e| matches!(e, Event::PassFinished { .. }));
    assert_eq!(
        (started, finished),
        (report.passes as usize, report.passes as usize),
        "{ctx}"
    );
    assert_eq!(report.steps as usize, steps.len(), "{ctx}");
    assert_eq!(report.steps, db.step_count(), "{ctx}");
    assert_eq!(
        report.evaluations,
        steps.iter().map(|s| s.evaluations).sum::<u64>(),
        "{ctx}"
    );
    assert_eq!(
        report.prepare_time,
        steps.iter().map(|s| s.prepare_time).sum(),
        "{ctx}"
    );
    assert_eq!(
        report.error_time,
        steps.iter().map(|s| s.error_time).sum(),
        "{ctx}"
    );
    assert!(report.skipped.is_empty(), "{ctx}");
    assert_eq!(
        (report.cache_hits, report.cache_misses),
        (0, 0),
        "{ctx}: the cache is off"
    );
    assert_eq!(report.cache_hit_rate(), None, "{ctx}");
    assert!(report.wall_time > std::time::Duration::ZERO, "{ctx}");

    // The Finished event carries the very report that is returned.
    let Some(Event::Finished(last)) = events.last() else {
        panic!("{ctx}: no Finished")
    };
    assert_eq!(format!("{last:?}"), format!("{report:?}"), "{ctx}");

    // The worst steps: at most 10, worst first, and they are the steps with the largest error.
    assert_eq!(report.worst_steps.len(), steps.len().min(10), "{ctx}");
    assert!(
        report
            .worst_steps
            .windows(2)
            .all(|w| w[0].error >= w[1].error),
        "{ctx}: worst first"
    );
    let mut errors: Vec<f64> = steps.iter().map(|s| s.error).collect();
    errors.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let reported: Vec<f64> = report.worst_steps.iter().map(|s| s.error).collect();
    assert_eq!(reported, errors[..reported.len()], "{ctx}");
    for worst in &report.worst_steps {
        // They are the database's records (with the global index).
        let in_db = db
            .steps()
            .nth(worst.index as usize)
            .expect("a step of the database");
        assert_eq!(in_db, worst, "{ctx}");
    }

    // The database holds exactly the committed steps, in the order the subgoals were merged:
    // by pass, then subgoal, then commit order within the subgoal.
    let mut ordered = steps.to_vec();
    ordered.sort_by_key(|s| (s.pass, s.subgoal, s.index));
    let history: Vec<&StepRecord> = db.steps().collect();
    assert_eq!(history.len(), ordered.len(), "{ctx}");
    for (i, (h, e)) in history.iter().zip(&ordered).enumerate() {
        assert_eq!(h.index, i as u64, "{ctx}: global indices are 0, 1, 2, ...");
        assert_eq!(outcome(h), outcome(e), "{ctx}: history step {i}");
    }
    // And the values: option indices of the events index the model's options.
    for s in steps {
        for (&p, &k) in s.parameters.iter().zip(&s.option_indices) {
            let entry = db.get(p).expect("calibrated");
            assert_eq!(entry.option_index, k, "{ctx}");
            assert_eq!(entry.value, x.model.options(p)[k as usize], "{ctx}");
        }
    }
    assert!(
        db.history()
            .iter()
            .all(|h| matches!(h, HistoryRecord::Step(_))),
        "{ctx}"
    );
}

fn run_snake(
    graph: &ProcessorGraph,
    goal: &ElementSet,
    cfg: SnakeConfig,
    heuristic: impl TraversalHeuristic + 'static,
    algorithm: Option<AlgorithmSubgraphs>,
    model: &SeparableModel,
) -> Run {
    let events = RecordingObserver::new();
    let mut snake = Snake::new(graph, goal.clone(), cfg)
        .heuristic(heuristic)
        .observer(events.clone());
    if let Some(a) = algorithm {
        snake = snake.algorithm(a);
    }
    let (db, report) = snake.calibrate_with_report(model).unwrap();
    Run {
        events: events.events(),
        db,
        report,
    }
}

// ---------------------------------------------------------------------------------------------
// The sequence of events of a run
// ---------------------------------------------------------------------------------------------

#[test]
fn a_single_pass_run_emits_pass_started_steps_pass_finished_finished() {
    let (graph, goal) = lattice(5, 5);
    let model = SeparableModel::new(&graph, 12, 5.0, 7.0, 1);
    let run = run_snake(&graph, &goal, config(1, 2, 2), BreadthFirst, None, &model);
    let events = &run.events;
    let n = events.len();
    assert_eq!(n, run.report.steps as usize + 3);
    assert!(matches!(
        &events[0],
        Event::PassStarted { pass: 0, subgoals: 1, uncalibrated } if *uncalibrated == goal.len()
    ));
    assert!(events[1..n - 2]
        .iter()
        .all(|e| matches!(e, Event::StepCommitted(_))));
    assert!(matches!(
        &events[n - 2],
        Event::PassFinished { pass: 0, calibrated } if *calibrated == goal.len()
    ));
    assert!(matches!(&events[n - 1], Event::Finished(r) if r.steps == run.report.steps));
    // A lattice with d_P = 1 has one step per node.
    assert_eq!(run.report.steps, graph.nodes().count() as u64);
    assert_eq!(run.report.passes, 1);
}

#[test]
fn every_step_satisfies_the_definitions_for_many_settings() {
    let (graph, goal) = lattice(5, 5);
    let (graph2, goal2) = lattice(4, 6);
    let model = SeparableModel::new(&graph, 12, 5.0, 7.0, 1);
    let model2 = SeparableModel::new(&graph2, 12, 5.0, 7.0, 1);
    let distances = [
        (0, 0, 0),
        (0, 0, 2),
        (0, 2, 2),
        (1, 0, 2),
        (1, 1, 1),
        (1, 2, 2),
        (1, 3, 2),
        (2, 2, 2),
        (2, 3, 1),
        (3, 1, 2),
    ];
    let mut runs = 0;
    for (g, goal, model, rows, cols) in [
        (&graph, &goal, &model, 5, 5),
        (&graph2, &goal2, &model2, 4, 6),
    ] {
        for (d_p, d_r, d_t) in distances {
            for heuristic in ["breadth-first", "depth-first", "random"] {
                for xeb in [false, true] {
                    let algorithm = if xeb {
                        AlgorithmSubgraphs::xeb_square_lattice(g, rows, cols).unwrap()
                    } else {
                        AlgorithmSubgraphs::all_active(g)
                    };
                    let masks = algorithm.masks();
                    let cfg = config(d_p, d_r, d_t);
                    let ctx = format!(
                        "{rows}x{cols} d = ({d_p}, {d_r}, {d_t}), {heuristic}, xeb = {xeb}"
                    );
                    let run = match heuristic {
                        "breadth-first" => {
                            run_snake(g, goal, cfg.clone(), BreadthFirst, Some(algorithm), model)
                        }
                        "depth-first" => {
                            run_snake(g, goal, cfg.clone(), DepthFirst, Some(algorithm), model)
                        }
                        _ => run_snake(g, goal, cfg.clone(), Random, Some(algorithm), model),
                    };
                    let x = Expect {
                        graph: g,
                        goal,
                        config: &cfg,
                        masks: &masks,
                        model,
                    };
                    let steps = check_stream(&x, &run.events, &ctx);
                    check_report(&x, &run, &steps, &ctx);
                    runs += 1;
                }
            }
        }
    }
    assert_eq!(runs, 120);
}

#[test]
fn multi_pass_runs_number_and_count_their_passes() {
    // With d_P = 2 one step absorbs a whole neighbourhood and the walk strands elements, so
    // the completion loop needs many passes.
    let (graph, goal) = lattice(8, 8);
    let model = SeparableModel::new(&graph, 12, 5.0, 7.0, 1);
    let cfg = config(2, 2, 2);
    let masks = AlgorithmSubgraphs::all_active(&graph).masks();
    let run = run_snake(&graph, &goal, cfg.clone(), BreadthFirst, None, &model);
    assert!(run.report.passes >= 4, "{} passes", run.report.passes);
    let x = Expect {
        graph: &graph,
        goal: &goal,
        config: &cfg,
        masks: &masks,
        model: &model,
    };
    let steps = check_stream(&x, &run.events, "d_P = 2");
    check_report(&x, &run, &steps, "d_P = 2");
    // The uncalibrated counts of the passes decrease strictly and the calibrated counts add up.
    let started: Vec<usize> = run
        .events
        .iter()
        .filter_map(|e| match e {
            Event::PassStarted { uncalibrated, .. } => Some(*uncalibrated),
            _ => None,
        })
        .collect();
    assert!(started.windows(2).all(|w| w[0] > w[1]), "{started:?}");
    let calibrated: usize = run
        .events
        .iter()
        .filter_map(|e| match e {
            Event::PassFinished { calibrated, .. } => Some(*calibrated),
            _ => None,
        })
        .sum();
    assert_eq!(calibrated, goal.len());
}

#[test]
fn parallel_subgoals_emit_consistent_events_at_every_thread_count() {
    let (graph, goal) = lattice(10, 10);
    let model = SeparableModel::new(&graph, 12, 5.0, 7.0, 1);
    let masks = AlgorithmSubgraphs::all_active(&graph).masks();
    // Four quadrants, calibrated as separate subgoals and stitched afterwards.
    let quadrant = |id: ElementId| -> usize {
        let position = |n: ElementId| -> (usize, usize) {
            let (r, c) = graph.label(n)[1..].split_once('_').unwrap();
            (r.parse().unwrap(), c.parse().unwrap())
        };
        let (r2, c2) = match graph.endpoints(id) {
            None => {
                let (r, c) = position(id);
                (2 * r, 2 * c)
            }
            Some([a, b]) => (position(a).0 + position(b).0, position(a).1 + position(b).1),
        };
        usize::from(r2 >= 10) * 2 + usize::from(c2 >= 10)
    };
    let mut regions = vec![ElementSet::new(graph.len()); 4];
    for id in goal.iter() {
        regions[quadrant(id)].insert(id);
    }

    let mut reference: Option<(Vec<Outcome>, ParameterDatabase<f64>)> = None;
    for threads in [1usize, 2, 4, 8] {
        let ctx = format!("{threads} thread(s)");
        let cfg = SnakeConfig {
            threads: Some(threads),
            ..config(1, 2, 2)
        };
        let events = RecordingObserver::new();
        let snake = Snake::new(&graph, goal.clone(), cfg.clone()).observer(events.clone());
        let plan = snake.stitch_plan(regions.clone()).unwrap();
        let mut db = ParameterDatabase::new(&graph);
        let report = snake.stitch(&mut db, &plan, &model).unwrap();
        let log = events.events();

        // Stage one has four subgoals; the passes of both stages are numbered consecutively
        // and there is a single Finished event for the whole stitched run.
        let passes: Vec<(u32, usize)> = log
            .iter()
            .filter_map(|e| match e {
                Event::PassStarted { pass, subgoals, .. } => Some((*pass, *subgoals)),
                _ => None,
            })
            .collect();
        assert_eq!(passes[0], (0, 4), "{ctx}: {passes:?}");
        assert!(
            passes.windows(2).all(|w| w[1].0 == w[0].0 + 1),
            "{ctx}: {passes:?}"
        );
        assert_eq!(count(&log, |e| matches!(e, Event::Finished(_))), 1, "{ctx}");
        assert!(
            matches!(log.last(), Some(Event::Finished(r)) if r.steps == report.steps),
            "{ctx}"
        );
        assert_eq!(report.passes as usize, passes.len(), "{ctx}");

        // The stream is consistent subgoal by subgoal (the order between subgoals is not
        // defined), and the database is the same whatever the scheduling was.
        let cfg1 = config(1, 2, 2);
        let x = Expect {
            graph: &graph,
            goal: &goal,
            config: &cfg1,
            masks: &masks,
            model: &model,
        };
        let steps = stitched_stream_steps(&x, &log, &ctx);
        assert_eq!(steps.len() as u64, db.step_count(), "{ctx}");
        let mut ordered = steps.clone();
        ordered.sort_by_key(|s| (s.pass, s.subgoal, s.index));
        let history: Vec<Outcome> = db.steps().map(outcome).collect();
        let from_events: Vec<Outcome> = ordered.iter().map(outcome).collect();
        assert_eq!(
            history, from_events,
            "{ctx}: the merge order is by pass and subgoal"
        );
        match &reference {
            None => reference = Some((history, db)),
            Some((outcomes, first_db)) => {
                assert_eq!(&history, outcomes, "{ctx}");
                assert!(first_db.same_outcome(&db), "{ctx}");
            }
        }
    }
}

/// Like `check_stream`, for a run in stages: the passes of the second stage start from a
/// partly calibrated database, so only the structure and the per-step definitions are checked.
fn stitched_stream_steps(x: &Expect<'_>, events: &[Event], ctx: &str) -> Vec<StepRecord> {
    let hoods = Neighbourhoods::build(x.graph, 2);
    let n = x.graph.len();
    let mut base = ElementSet::new(n);
    let mut overlays: BTreeMap<u32, ElementSet> = BTreeMap::new();
    let mut open = None;
    let mut steps = Vec::new();
    for event in events {
        match event {
            Event::PassStarted { pass, .. } => {
                assert!(open.replace(*pass).is_none(), "{ctx}");
                overlays.clear();
            }
            Event::StepCommitted(s) => {
                assert_eq!(open, Some(s.pass), "{ctx}: a step outside its pass");
                let overlay = overlays
                    .entry(s.subgoal)
                    .or_insert_with(|| ElementSet::new(n));
                let before = base.union(overlay);
                assert!(!before.contains(s.centre), "{ctx}");
                let expected: BTreeSet<ElementId> = before
                    .iter()
                    .filter(|&c| {
                        s.parameters
                            .iter()
                            .any(|&p| hoods.is_within(c, p, x.config.constraint_distance))
                    })
                    .collect();
                assert_eq!(set(&s.constraints), expected, "{ctx}");
                for &p in &s.parameters {
                    assert!(!before.contains(p), "{ctx}");
                    assert!(
                        hoods.is_within(s.centre, p, x.config.parameter_distance),
                        "{ctx}"
                    );
                    overlay.insert(p);
                }
                steps.push(s.clone());
            }
            Event::PassFinished { pass, .. } => {
                assert_eq!(open.take(), Some(*pass), "{ctx}");
                for overlay in overlays.values() {
                    base.union_with(overlay);
                }
            }
            Event::Finished(_) => assert!(open.is_none(), "{ctx}"),
            other => panic!("{ctx}: unexpected {other:?}"),
        }
    }
    assert_eq!(&base, x.goal, "{ctx}");
    steps
}

// ---------------------------------------------------------------------------------------------
// Observers
// ---------------------------------------------------------------------------------------------

#[test]
fn every_observer_sees_every_event_in_registration_order() {
    let (graph, goal) = lattice(4, 4);
    let model = SeparableModel::new(&graph, 8, 5.0, 7.0, 1);

    struct Tagged {
        id: usize,
        log: Arc<Mutex<Vec<(usize, usize)>>>,
        seen: AtomicUsize,
    }
    impl Observer for Tagged {
        fn on_event(&self, _event: &Event) {
            let k = self.seen.fetch_add(1, Ordering::SeqCst);
            self.log.lock().unwrap().push((self.id, k));
        }
    }
    let log = Arc::new(Mutex::new(Vec::new()));
    let tagged = |id| Tagged {
        id,
        log: Arc::clone(&log),
        seen: AtomicUsize::new(0),
    };
    let recorder = RecordingObserver::new();
    let counter = Arc::new(AtomicUsize::new(0));
    let closure = {
        let counter = Arc::clone(&counter);
        move |_: &Event| {
            counter.fetch_add(1, Ordering::SeqCst);
        }
    };
    let shared: Arc<dyn Observer> = Arc::new(recorder.clone());
    let snake = Snake::new(&graph, goal, config(1, 2, 2))
        .observer(tagged(0))
        .observer(tagged(1))
        .observer(closure)
        .observer_arc(Arc::clone(&shared))
        .observer(recorder.clone());
    snake.calibrate(&model).unwrap();

    let events = recorder.events();
    // The recorder was registered twice (directly and through the Arc), so it saw each
    // event twice, in the same order.
    assert_eq!(events.len() % 2, 0);
    let n = events.len() / 2;
    assert_eq!(counter.load(Ordering::SeqCst), n);
    for i in 0..n {
        assert_eq!(
            format!("{:?}", events[2 * i]),
            format!("{:?}", events[2 * i + 1])
        );
    }
    // Observer 0 gets every event before observer 1 does.
    let log = log.lock().unwrap();
    assert_eq!(log.len(), 2 * n);
    for (i, pair) in log.chunks(2).enumerate() {
        assert_eq!(pair, [(0, i), (1, i)]);
    }
}

#[test]
fn recording_observers_share_their_buffer_and_events_can_be_cloned() {
    let (graph, goal) = lattice(3, 3);
    let model = SeparableModel::new(&graph, 8, 5.0, 7.0, 1);
    let a = RecordingObserver::new();
    let b = a.clone();
    Snake::new(&graph, goal, config(1, 2, 2))
        .observer(a.clone())
        .calibrate(&model)
        .unwrap();
    assert_eq!(a.events().len(), b.events().len());
    assert!(!a.events().is_empty());
    let copy: Vec<Event> = a.events().to_vec();
    assert_eq!(format!("{copy:?}"), format!("{:?}", b.events()));
}

#[test]
fn a_panicking_observer_does_not_lose_the_step_it_was_told_about() {
    let (graph, goal) = lattice(5, 5);
    let model = SeparableModel::new(&graph, 8, 5.0, 7.0, 1);
    let seen = Arc::new(AtomicUsize::new(0));
    let bomb = {
        let seen = Arc::clone(&seen);
        move |event: &Event| {
            if matches!(event, Event::StepCommitted(_)) && seen.fetch_add(1, Ordering::SeqCst) == 4
            {
                panic!("the progress bar broke");
            }
        }
    };
    let snake = Snake::new(&graph, goal.clone(), config(1, 2, 2)).observer(bomb);
    let mut db = ParameterDatabase::new(&graph);
    let outcome = catch_unwind(AssertUnwindSafe(|| snake.calibrate_into(&mut db, &model)));
    assert!(
        outcome.is_err(),
        "the panic of an observer reaches the caller"
    );
    // The event is emitted after the commit: the fifth step is in the database.
    assert_eq!(db.step_count(), 5);
    // The same Snake carries on (the observer panics only once).
    snake.calibrate_into(&mut db, &model).unwrap();
    assert_eq!(db.calibrated(), &goal);
}

// ---------------------------------------------------------------------------------------------
// The report
// ---------------------------------------------------------------------------------------------

#[test]
fn the_report_of_a_resumed_run_covers_only_that_call() {
    let (graph, goal) = lattice(6, 6);
    let model = SeparableModel::new(&graph, 12, 5.0, 7.0, 1);
    let token = CancellationToken::new();
    let seen = Arc::new(AtomicUsize::new(0));
    let events = RecordingObserver::new();
    let canceller = {
        let (token, seen) = (token.clone(), Arc::clone(&seen));
        move |event: &Event| {
            if matches!(event, Event::StepCommitted(_)) && seen.fetch_add(1, Ordering::SeqCst) == 9
            {
                token.cancel();
            }
        }
    };
    let snake = Snake::new(&graph, goal.clone(), config(1, 2, 2))
        .cancellation(token.clone())
        .observer(events.clone())
        .observer(canceller);
    let mut db = ParameterDatabase::new(&graph);
    let FaultTuneError::Cancelled { report: first } =
        snake.calibrate_into(&mut db, &model).unwrap_err()
    else {
        panic!("expected Cancelled")
    };
    assert_eq!(first.steps, 10);
    assert_eq!(first.passes, 1);
    token.reset();
    let log_len = events.events().len();
    let second = snake.calibrate_into(&mut db, &model).unwrap();
    let log = &events.events()[log_len..];

    assert_eq!(first.steps + second.steps, db.step_count());
    assert_eq!(second.steps as usize, steps_of(log).len());
    assert_eq!(
        second.evaluations,
        steps_of(log).iter().map(|s| s.evaluations).sum::<u64>()
    );
    // Pass numbers continue across calls on one database.
    assert!(matches!(&log[0], Event::PassStarted { pass: 1, .. }));
    assert!(steps_of(log).iter().all(|s| s.pass >= 1));
    // The worst steps of the second report come from the second call only.
    assert!(second.worst_steps.iter().all(|s| s.index >= first.steps));
    // Both calls ended with a Finished event carrying their own report.
    let all = events.events();
    let finished: Vec<&Report> = all
        .iter()
        .filter_map(|e| match e {
            Event::Finished(r) => Some(r),
            _ => None,
        })
        .collect();
    assert_eq!(finished.len(), 2);
    assert_eq!(
        (finished[0].steps, finished[1].steps),
        (first.steps, second.steps)
    );
}

#[test]
fn the_cache_statistics_of_the_report_count_the_lookups_of_that_run() {
    let (graph, goal) = lattice(5, 5);
    // A pure model: the cross-step cache is safe.
    let model = CollisionModel::new(&graph, 120, 5.0, 7.0, 0.01, 1, 4);

    // Off by default: no lookups.
    let (_, report) = Snake::new(&graph, goal.clone(), config(1, 2, 2))
        .calibrate_with_report(&model)
        .unwrap();
    assert_eq!((report.cache_hits, report.cache_misses), (0, 0));

    // On: every distinct candidate is looked up once, so the lookups are the evaluations; in
    // one fresh run every lookup of a first evaluation is a miss (nothing is shared by steps
    // that differ in their constraints), and a cache kept for a second run answers everything.
    let cfg = SnakeConfig {
        cache: CachePolicy::AcrossSteps { capacity: 1 << 20 },
        ..config(1, 2, 2)
    };
    let (db_a, first) = Snake::new(&graph, goal.clone(), cfg.clone())
        .calibrate_with_report(&model)
        .unwrap();
    assert_eq!(first.cache_hits + first.cache_misses, first.evaluations);
    let shared = Arc::new(CrossStepCache::new(1 << 20));
    let snake = Snake::new(&graph, goal.clone(), cfg).shared_cache(Arc::clone(&shared));
    let (db_b, cold) = snake.calibrate_with_report(&model).unwrap();
    let (db_c, warm) = snake.calibrate_with_report(&model).unwrap();
    assert_eq!(cold.cache_hits + cold.cache_misses, cold.evaluations);
    assert_eq!(
        warm.cache_misses, 0,
        "everything was cached by the first run"
    );
    assert_eq!(warm.cache_hits, warm.evaluations);
    assert_eq!(warm.cache_hit_rate(), Some(1.0));
    // The cache changes neither the evaluation counts nor the result.
    assert_eq!(cold.evaluations, warm.evaluations);
    assert!(db_a.same_outcome(&db_b) && db_b.same_outcome(&db_c));
    let (db_off, off) = Snake::new(&graph, goal, config(1, 2, 2))
        .calibrate_with_report(&model)
        .unwrap();
    assert_eq!(off.evaluations, first.evaluations);
    assert!(db_off.same_outcome(&db_a));
}

// ---------------------------------------------------------------------------------------------
// Failures and cancellation in the event stream
// ---------------------------------------------------------------------------------------------

/// A model that fails (or answers NaN) for the step centred on one element.
struct Failing {
    inner: SeparableModel,
    centre: ElementId,
    nan: bool,
}

impl ErrorModel for Failing {
    type Value = f64;
    type Data = ();
    fn options(&self, e: ElementId) -> Vec<f64> {
        self.inner.options(e)
    }
    fn prepare(&self, step: &Step<'_, f64>) -> Result<(), ModelError> {
        if step.centre == self.centre && !self.nan {
            Err(ModelError::new("detector offline"))
        } else {
            Ok(())
        }
    }
    fn error(&self, step: &Step<'_, f64>, data: &(), c: &[f64]) -> f64 {
        if step.centre == self.centre && self.nan {
            f64::NAN
        } else {
            self.inner.error(step, data, c)
        }
    }
}

#[test]
fn finished_is_the_last_event_of_every_run_that_returns() {
    let (graph, goal) = lattice(5, 5);
    let base = SeparableModel::new(&graph, 8, 5.0, 7.0, 1);
    let victim = node(&graph, 2, 2);

    // Abort, NaN, cancellation and success: always one Finished, always last.
    type Case = (&'static str, ModelErrorPolicy, bool, bool);
    let cases: [Case; 4] = [
        ("model error", ModelErrorPolicy::Abort, false, false),
        ("nan", ModelErrorPolicy::SkipElement, true, false),
        ("cancelled", ModelErrorPolicy::Abort, false, true),
        ("skipped", ModelErrorPolicy::SkipElement, false, false),
    ];
    for (name, policy, nan, cancel) in cases {
        let model = Failing {
            inner: base.clone(),
            centre: victim,
            nan,
        };
        let token = CancellationToken::new();
        let events = RecordingObserver::new();
        let seen = Arc::new(AtomicUsize::new(0));
        let canceller = {
            let (token, seen) = (token.clone(), Arc::clone(&seen));
            move |event: &Event| {
                if cancel
                    && matches!(event, Event::StepCommitted(_))
                    && seen.fetch_add(1, Ordering::SeqCst) == 2
                {
                    token.cancel();
                }
            }
        };
        let cfg = SnakeConfig {
            on_model_error: policy,
            ..config(1, 2, 2)
        };
        let snake = Snake::new(&graph, goal.clone(), cfg)
            .cancellation(token)
            .observer(events.clone())
            .observer(canceller);
        let mut db = ParameterDatabase::new(&graph);
        let result = snake.calibrate_into(&mut db, &model);
        let log = events.events();
        assert_eq!(
            count(&log, |e| matches!(e, Event::Finished(_))),
            1,
            "{name}"
        );
        let Some(Event::Finished(report)) = log.last() else {
            panic!("{name}: Finished is not last")
        };
        assert_eq!(report.steps, db.step_count(), "{name}");
        assert_eq!(report.steps as usize, steps_of(&log).len(), "{name}");
        // Passes started are passes finished, whatever the ending.
        assert_eq!(
            count(&log, |e| matches!(e, Event::PassStarted { .. })),
            count(&log, |e| matches!(e, Event::PassFinished { .. })),
            "{name}"
        );
        match name {
            "model error" => assert!(matches!(result, Err(FaultTuneError::Model { .. }))),
            "nan" => assert!(matches!(result, Err(FaultTuneError::NonFiniteError { .. }))),
            "cancelled" => assert!(matches!(result, Err(FaultTuneError::Cancelled { .. }))),
            _ => assert!(result.is_ok()),
        }
    }
}

#[test]
fn a_skipped_step_is_announced_once_and_commits_nothing() {
    let (graph, goal) = lattice(5, 5);
    let victim = node(&graph, 2, 2);
    let model = Failing {
        inner: SeparableModel::new(&graph, 8, 5.0, 7.0, 1),
        centre: victim,
        nan: false,
    };
    let cfg = SnakeConfig {
        on_model_error: ModelErrorPolicy::SkipElement,
        ..config(1, 2, 2)
    };
    let events = RecordingObserver::new();
    let (db, report) = Snake::new(&graph, goal.clone(), cfg)
        .observer(events.clone())
        .calibrate_with_report(&model)
        .unwrap();
    let log = events.events();
    let failures: Vec<(usize, ElementId, &str)> = log
        .iter()
        .enumerate()
        .filter_map(|(i, e)| match e {
            Event::StepFailed { centre, error } => Some((i, *centre, error.as_str())),
            _ => None,
        })
        .collect();
    assert_eq!(failures.len(), 1);
    let (position, centre, message) = failures[0];
    assert_eq!(centre, victim);
    assert!(message.contains("detector offline"), "{message}");
    assert_eq!(report.skipped, vec![victim]);
    // The failure is inside a pass, and no step has the skipped element as its centre.
    let started = log[..position]
        .iter()
        .rposition(|e| matches!(e, Event::PassStarted { .. }))
        .unwrap();
    let finished = log[position..]
        .iter()
        .position(|e| matches!(e, Event::PassFinished { .. }))
        .unwrap();
    assert!(started < position && finished > 0);
    assert!(steps_of(&log)
        .iter()
        .all(|s| s.centre != victim && !s.parameters.contains(&victim)));
    assert_eq!(db.len(), goal.len() - 1);
    // The skipped element is not counted as calibrated by its pass.
    let calibrated: usize = log
        .iter()
        .filter_map(|e| match e {
            Event::PassFinished { calibrated, .. } => Some(*calibrated),
            _ => None,
        })
        .sum();
    assert_eq!(calibrated, goal.len() - 1);
}

// ---------------------------------------------------------------------------------------------
// Re-calibration
// ---------------------------------------------------------------------------------------------

#[test]
fn recalibrate_announces_the_removal_once_before_its_first_pass() {
    let (graph, goal) = lattice(6, 6);
    let model = SeparableModel::new(&graph, 12, 5.0, 7.0, 1);
    let events = RecordingObserver::new();
    let snake = Snake::new(&graph, goal.clone(), config(1, 2, 2)).observer(events.clone());
    let mut db = snake.calibrate(&model).unwrap();
    let first_call = events.events().len();

    let (a, b) = (node(&graph, 1, 1), node(&graph, 4, 4));
    let report = snake.recalibrate(&mut db, &[a, b], &model).unwrap();
    let log = &events.events()[first_call..];

    // The removal: the elements within the expiry radius 1 of the two qubits.
    let mut expected: Vec<ElementId> = [a, b]
        .iter()
        .flat_map(|&q| std::iter::once(q).chain(graph.incident(q).iter().copied()))
        .filter(|&e| goal.contains(e))
        .collect();
    expected.sort();
    expected.dedup();
    let announced: Vec<(usize, u32, Vec<ElementId>)> = log
        .iter()
        .enumerate()
        .filter_map(|(i, e)| match e {
            Event::Invalidated { pass, elements } => Some((i, *pass, elements.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(announced.len(), 1, "one announcement per call");
    let (position, pass, elements) = &announced[0];
    assert_eq!(*position, 0, "before the first pass");
    assert_eq!(elements, &expected, "ascending, without duplicates");
    // The number of the pass that follows, and the database's own record of the removal.
    assert_eq!(*pass, 1);
    assert!(matches!(&log[1], Event::PassStarted { pass: 1, .. }));
    let removal = db
        .history()
        .iter()
        .rev()
        .find_map(|h| match h {
            HistoryRecord::Invalidated(e) => Some(e.clone()),
            _ => None,
        })
        .unwrap();
    assert_eq!(&removal, elements);
    // The two regions are far apart: two subgoals, one step each.
    assert!(
        matches!(&log[1], Event::PassStarted { subgoals: 2, uncalibrated, .. } if *uncalibrated == expected.len())
    );
    assert_eq!(report.steps, 2);
    assert!(matches!(log.last(), Some(Event::Finished(r)) if r.steps == 2));
    assert_eq!(steps_of(log).len(), 2);
}

#[test]
fn recalibrate_announces_nothing_when_nothing_was_forgotten() {
    let (graph, goal) = lattice(4, 4);
    let model = SeparableModel::new(&graph, 8, 5.0, 7.0, 1);
    let events = RecordingObserver::new();
    let snake = Snake::new(&graph, goal.clone(), config(1, 2, 2)).observer(events.clone());

    // An empty database: there is nothing to forget.
    let mut db = ParameterDatabase::new(&graph);
    snake
        .recalibrate(&mut db, &[node(&graph, 1, 1)], &model)
        .unwrap();
    assert_eq!(
        count(&events.events(), |e| matches!(e, Event::Invalidated { .. })),
        0
    );
    assert!(db
        .history()
        .iter()
        .all(|h| matches!(h, HistoryRecord::Step(_))));

    // No elements: no pass and no announcement, only the end of the (empty) run.
    let before = events.events().len();
    snake.calibrate_into(&mut db, &model).unwrap();
    let before_empty = events.events().len();
    assert!(before_empty > before);
    snake.recalibrate(&mut db, &[], &model).unwrap();
    let log = &events.events()[before_empty..];
    assert!(
        matches!(log, [Event::Finished(r)] if r.steps == 0 && r.passes == 0),
        "{log:?}"
    );
}

#[test]
fn recalibrating_twice_announces_each_removal_with_the_next_pass_number() {
    let (graph, goal) = lattice(5, 5);
    let model = SeparableModel::new(&graph, 8, 5.0, 7.0, 1);
    let events = RecordingObserver::new();
    let snake = Snake::new(&graph, goal, config(1, 2, 2)).observer(events.clone());
    let mut db = snake.calibrate(&model).unwrap();
    let q = node(&graph, 2, 2);
    snake.recalibrate(&mut db, &[q], &model).unwrap();
    snake.recalibrate(&mut db, &[q], &model).unwrap();
    let log = events.events();
    let passes: Vec<u32> = log
        .iter()
        .filter_map(|e| match e {
            Event::Invalidated { pass, .. } => Some(*pass),
            _ => None,
        })
        .collect();
    assert_eq!(passes, vec![1, 2]);
    // Every invalidation is followed by the pass it announced.
    for (i, e) in log.iter().enumerate() {
        if let Event::Invalidated { pass, .. } = e {
            assert!(matches!(&log[i + 1], Event::PassStarted { pass: p, .. } if p == pass));
        }
    }
    // The steps of the three calls have increasing pass numbers.
    let step_passes: Vec<u32> = db.steps().map(|s| s.pass).collect();
    assert!(
        step_passes.windows(2).all(|w| w[0] <= w[1]),
        "{step_passes:?}"
    );
    assert_eq!(step_passes.last(), Some(&2));
}

// ---------------------------------------------------------------------------------------------
// What the model is told
// ---------------------------------------------------------------------------------------------

/// What the model saw in `prepare` for one step.
struct Seen {
    index: u64,
    centre: ElementId,
    parameters: Vec<ElementId>,
    constraints: Vec<(ElementId, f64)>,
}

/// A model that records the steps it is given, and passes data from `prepare` to `error`.
struct Spy<'g> {
    graph: &'g ProcessorGraph,
    inner: SeparableModel,
    seen: Mutex<Vec<Seen>>,
}

impl ErrorModel for Spy<'_> {
    type Value = f64;
    /// The index and centre of the step `prepare` was called for.
    type Data = (u64, ElementId);

    fn options(&self, element: ElementId) -> Vec<f64> {
        self.inner.options(element)
    }

    fn prepare(&self, step: &Step<'_, f64>) -> Result<(u64, ElementId), ModelError> {
        assert!(
            std::ptr::eq(step.graph, self.graph),
            "the step carries the graph"
        );
        self.seen.lock().unwrap().push(Seen {
            index: step.index,
            centre: step.centre,
            parameters: step.parameters.to_vec(),
            constraints: step.constraints.to_vec(),
        });
        Ok((step.index, step.centre))
    }

    fn error(&self, step: &Step<'_, f64>, data: &(u64, ElementId), candidate: &[f64]) -> f64 {
        assert_eq!(
            *data,
            (step.index, step.centre),
            "the data of prepare reaches error"
        );
        self.inner.error(step, &(), candidate)
    }
}

#[test]
fn the_model_is_told_what_the_events_report() {
    let (graph, goal) = lattice(7, 7);
    for (d_p, d_r) in [(1, 2), (2, 2), (0, 3)] {
        let spy = Spy {
            graph: &graph,
            inner: SeparableModel::new(&graph, 12, 5.0, 7.0, 1),
            seen: Mutex::new(Vec::new()),
        };
        let events = RecordingObserver::new();
        let db = Snake::new(&graph, goal.clone(), config(d_p, d_r, 2))
            .observer(events.clone())
            .calibrate(&spy)
            .unwrap();
        let steps = steps_of(&events.events());
        let seen = spy.seen.lock().unwrap();
        assert_eq!(seen.len(), steps.len());
        for step in &steps {
            let told = seen
                .iter()
                .find(|s| s.centre == step.centre)
                .unwrap_or_else(|| panic!("no prepare for {}", graph.label(step.centre)));
            let at = format!("d_P = {d_p}: step at {}", graph.label(step.centre));
            // The ordinal of the step within its subgoal, the same in the event.
            assert_eq!(told.index, step.index, "{at}");
            assert_eq!(told.parameters, step.parameters, "{at}");
            let ids: Vec<ElementId> = told.constraints.iter().map(|&(c, _)| c).collect();
            assert_eq!(ids, step.constraints, "{at}");
            // The constraint values are the values the database holds for those elements.
            for &(c, value) in &told.constraints {
                assert_eq!(
                    db.value(c),
                    Some(&value),
                    "{at}: constraint {}",
                    graph.label(c)
                );
            }
        }
    }
}
