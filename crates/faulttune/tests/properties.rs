//! Property tests of the whole engine (ARCHITECTURE §13, "Properties"), on random small
//! lattices and random configurations, through the public API only.
//!
//! * every calibration ends with every goal element calibrated or a reported failure, with no
//!   element skipped silently, and its history, events and report agree with each other;
//! * parallel runs (2, 4 and 8 threads) give exactly the databases of a sequential run, for
//!   calibration, re-calibration and stitching;
//! * the same seed gives the same result twice, also with the random heuristic, seed policy and
//!   optimizer;
//! * the number of evaluations of every step is deterministic, and the accounting of
//!   evaluations, cache hits and model calls is exact;
//! * `CachePolicy::AcrossSteps` (and a shared cache) never changes a result;
//! * `recalibrate` changes only entries inside the removal region, keeps every other entry bit
//!   for bit and only appends to the history;
//! * `stitch` gives a complete database whose stage-one regions were separate subgoals.
//!
//! The file is self-contained.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use faulttune::heuristics::{BreadthFirst, DepthFirst, Random};
use faulttune::optimizers::{CoordinateDescent, Exhaustive, RandomSearch};
use faulttune::toy::{CollisionModel, GateFrequencyModel, SeparableModel};
use faulttune::{
    AlgorithmSubgraphs, CachePolicy, CandidateBatch, CrossStepCache, EdgeType, ElementId,
    ElementKind, ElementSet, ErrorModel, Event, FaultTuneError, HistoryRecord, ModelError,
    ModelErrorPolicy, Neighbourhoods, ParameterDatabase, ProcessorGraph, RecordingObserver, Report,
    SeedPolicy, Snake, SnakeConfig, Step, StepRecord,
};
use proptest::prelude::*;
use proptest::test_runner::TestCaseError;

// ---------------------------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Heur {
    Breadth,
    Depth,
    Random,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Opt {
    Coordinate,
    Exhaustive,
    RandomSearch,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Seeds {
    Lowest,
    Random,
    Fixed(u32),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Goal {
    NodesAndEngineered,
    Everything,
    NodesOnly,
    EngineeredOnly,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ModelKind {
    Separable,
    Collision,
    Gate,
}

/// Everything that defines one random calibration problem.
#[derive(Clone, Debug)]
struct Scenario {
    rows: usize,
    cols: usize,
    d_p: u32,
    d_r: u32,
    d_t: u32,
    expiry: u32,
    heuristic: Heur,
    optimizer: Opt,
    seeds: Seeds,
    xeb: bool,
    goal: Goal,
    model: ModelKind,
    /// Options per element of the separable model, extra options of the collision model.
    k: usize,
    /// Interaction distance of the collision model (clamped to `d_r`).
    interaction: u32,
    seed: u64,
    /// 0: step only, 1: tiny cross-step cache, 2: large cross-step cache.
    cache: u8,
}

/// One model type for all the scenarios.
enum AnyModel {
    Separable(SeparableModel),
    Collision(CollisionModel),
    Gate(GateFrequencyModel),
}

impl ErrorModel for AnyModel {
    type Value = f64;
    type Data = ();

    fn options(&self, element: ElementId) -> Vec<f64> {
        match self {
            Self::Separable(m) => m.options(element),
            Self::Collision(m) => m.options(element),
            Self::Gate(m) => m.options(element),
        }
    }

    fn prepare(&self, step: &Step<'_, f64>) -> Result<(), ModelError> {
        match self {
            Self::Separable(m) => m.prepare(step),
            Self::Collision(m) => m.prepare(step),
            Self::Gate(m) => m.prepare(step),
        }
    }

    fn error(&self, step: &Step<'_, f64>, data: &(), candidate: &[f64]) -> f64 {
        match self {
            Self::Separable(m) => m.error(step, data, candidate),
            Self::Collision(m) => m.error(step, data, candidate),
            Self::Gate(m) => m.error(step, data, candidate),
        }
    }

    fn error_batch(
        &self,
        step: &Step<'_, f64>,
        data: &(),
        candidates: &CandidateBatch<'_, f64>,
        out: &mut [f64],
    ) -> Result<(), ModelError> {
        match self {
            Self::Separable(m) => m.error_batch(step, data, candidates, out),
            Self::Collision(m) => m.error_batch(step, data, candidates, out),
            Self::Gate(m) => m.error_batch(step, data, candidates, out),
        }
    }
}

/// Counts the evaluations the model is asked for (it relies on the default `error_batch`).
struct Counting<'a> {
    inner: &'a AnyModel,
    calls: AtomicU64,
}

impl<'a> Counting<'a> {
    fn new(inner: &'a AnyModel) -> Self {
        Self {
            inner,
            calls: AtomicU64::new(0),
        }
    }

    fn calls(&self) -> u64 {
        self.calls.load(Ordering::SeqCst)
    }
}

impl ErrorModel for Counting<'_> {
    type Value = f64;
    type Data = ();

    fn options(&self, element: ElementId) -> Vec<f64> {
        self.inner.options(element)
    }

    fn prepare(&self, step: &Step<'_, f64>) -> Result<(), ModelError> {
        self.inner.prepare(step)
    }

    fn error(&self, step: &Step<'_, f64>, data: &(), candidate: &[f64]) -> f64 {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.error(step, data, candidate)
    }
}

/// A model whose `prepare` fails for the steps centred on the `bad` elements.
struct Flaky<'a> {
    inner: &'a AnyModel,
    bad: Vec<bool>,
}

impl ErrorModel for Flaky<'_> {
    type Value = f64;
    type Data = ();

    fn options(&self, element: ElementId) -> Vec<f64> {
        self.inner.options(element)
    }

    fn prepare(&self, step: &Step<'_, f64>) -> Result<(), ModelError> {
        if self.bad[step.centre.index()] {
            Err(ModelError::new(format!(
                "measurement of {} failed",
                step.graph.label(step.centre)
            )))
        } else {
            self.inner.prepare(step)
        }
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
        self.inner.error_batch(step, data, candidates, out)
    }
}

/// The largest number of parameters a step can have: `max_g |within(g, d_P) ∩ goal|`.
fn max_parameters(graph: &ProcessorGraph, goal: &ElementSet, d_p: u32) -> usize {
    let hoods = Neighbourhoods::build(graph, d_p);
    goal.iter()
        .map(|g| {
            hoods
                .within(g, d_p)
                .iter()
                .filter(|&&e| goal.contains(e))
                .count()
        })
        .max()
        .unwrap_or(0)
}

impl Scenario {
    fn graph(&self) -> ProcessorGraph {
        ProcessorGraph::square_lattice(self.rows, self.cols)
    }

    fn goal_set(&self, graph: &ProcessorGraph) -> ElementSet {
        match self.goal {
            Goal::NodesAndEngineered => {
                graph.goal(|e| e.kind() != ElementKind::Edge(EdgeType::Parasitic))
            }
            Goal::Everything => graph.all_elements(),
            Goal::NodesOnly => graph.goal(|e| e.kind() == ElementKind::Node),
            Goal::EngineeredOnly => {
                graph.goal(|e| e.kind() == ElementKind::Edge(EdgeType::Engineered))
            }
        }
    }

    fn model(&self, graph: &ProcessorGraph) -> AnyModel {
        match self.model {
            ModelKind::Separable => {
                AnyModel::Separable(SeparableModel::new(graph, self.k, 5.0, 7.0, self.seed))
            }
            ModelKind::Gate => AnyModel::Gate(GateFrequencyModel::new(graph, self.seed)),
            ModelKind::Collision => {
                let interaction = self.interaction.min(self.d_r);
                for separation in [0.04, 0.02, 0.01, 0.005, 0.0025] {
                    let probe = CollisionModel::new(graph, 1, 5.0, 7.0, separation, interaction, 0);
                    let required = probe.required_options();
                    if required <= 150 {
                        return AnyModel::Collision(CollisionModel::new(
                            graph,
                            required + self.k,
                            5.0,
                            7.0,
                            separation,
                            interaction,
                            self.seed,
                        ));
                    }
                }
                AnyModel::Separable(SeparableModel::new(
                    graph,
                    self.k.max(2),
                    5.0,
                    7.0,
                    self.seed,
                ))
            }
        }
    }

    /// Fixes the combinations that would be invalid or too expensive.
    fn normalised(mut self) -> Self {
        // Parasitic edges are in no XEB subgraph, and the gate model has no options for them.
        if (self.xeb || self.model == ModelKind::Gate) && self.goal == Goal::Everything {
            self.goal = Goal::NodesAndEngineered;
        }
        if self.model == ModelKind::Collision {
            self.interaction = self.interaction.min(self.d_r);
        }
        if self.optimizer == Opt::Exhaustive {
            let graph = self.graph();
            let goal = self.goal_set(&graph);
            let model = self.model(&graph);
            let options = model.options(ElementId::new(0)).len() as u128;
            let size = options.checked_pow(max_parameters(&graph, &goal, self.d_p) as u32);
            if size.is_none_or(|s| s > 4000) {
                self.optimizer = Opt::Coordinate;
            }
        }
        self
    }

    fn config(&self, threads: Option<usize>, cache: CachePolicy) -> SnakeConfig {
        SnakeConfig {
            parameter_distance: self.d_p,
            constraint_distance: self.d_r,
            traversal_distance: self.d_t,
            expiry_radius: self.expiry,
            seed: self.seed,
            threads,
            cache,
            ..SnakeConfig::default()
        }
    }

    fn cache_policy(&self) -> CachePolicy {
        match self.cache {
            0 => CachePolicy::StepOnly,
            1 => CachePolicy::AcrossSteps { capacity: 7 },
            _ => CachePolicy::AcrossSteps { capacity: 1 << 16 },
        }
    }

    fn algorithm(&self, graph: &ProcessorGraph) -> AlgorithmSubgraphs {
        if self.xeb {
            AlgorithmSubgraphs::xeb_square_lattice(graph, self.rows, self.cols)
                .expect("the XEB subgraphs of a square lattice")
        } else {
            AlgorithmSubgraphs::all_active(graph)
        }
    }

    /// A Snake for this scenario with the scenario's own cache policy.
    fn snake<'g>(&self, graph: &'g ProcessorGraph, threads: Option<usize>) -> Snake<'g> {
        self.snake_with(graph, self.config(threads, self.cache_policy()))
    }

    fn snake_with<'g>(&self, graph: &'g ProcessorGraph, config: SnakeConfig) -> Snake<'g> {
        let goal = self.goal_set(graph);
        let snake = Snake::new(graph, goal, config).algorithm(self.algorithm(graph));
        let snake = match self.heuristic {
            Heur::Breadth => snake.heuristic(BreadthFirst),
            Heur::Depth => snake.heuristic(DepthFirst),
            Heur::Random => snake.heuristic(Random),
        };
        let snake = match self.optimizer {
            Opt::Coordinate => snake.optimizer(CoordinateDescent::default()),
            Opt::Exhaustive => snake.optimizer(Exhaustive::default()),
            Opt::RandomSearch => snake.optimizer(RandomSearch {
                samples: 40,
                batch_size: 16,
            }),
        };
        let n = graph.len() as u32;
        match self.seeds {
            Seeds::Lowest => snake,
            Seeds::Random => snake.seed_policy(SeedPolicy::Random),
            Seeds::Fixed(x) => snake.seed_policy(SeedPolicy::Fixed(vec![
                ElementId::new(x % n),
                ElementId::new((x / 7) % n),
            ])),
        }
    }
}

fn scenario() -> impl Strategy<Value = Scenario> {
    (
        (1usize..=5, 1usize..=5),
        (0u32..=2, 0u32..=3, 1u32..=3, 0u32..=2),
        (0u8..3, 0u8..3, 0u8..3, 0u32..1000),
        (any::<bool>(), 0u8..4),
        (0u8..3, 2usize..=10, 0u32..=3),
        (any::<u64>(), 0u8..3),
    )
        .prop_map(
            |(
                (rows, cols),
                (d_p, d_r, d_t, expiry),
                (heuristic, optimizer, seeds, fixed),
                (xeb, goal),
                (model, k, interaction),
                (seed, cache),
            )| {
                Scenario {
                    rows,
                    cols,
                    d_p,
                    d_r,
                    d_t,
                    expiry,
                    heuristic: [Heur::Breadth, Heur::Depth, Heur::Random][heuristic as usize],
                    // Weight the cheap, general optimizer more than the others.
                    optimizer: [Opt::Coordinate, Opt::Coordinate, Opt::Exhaustive]
                        [optimizer as usize % 3]
                        .min_by_seed(seed),
                    seeds: [Seeds::Lowest, Seeds::Random, Seeds::Fixed(fixed)][seeds as usize],
                    xeb,
                    goal: [
                        Goal::NodesAndEngineered,
                        Goal::Everything,
                        Goal::NodesOnly,
                        Goal::EngineeredOnly,
                    ][goal as usize],
                    model: [ModelKind::Separable, ModelKind::Collision, ModelKind::Gate]
                        [model as usize],
                    k,
                    interaction,
                    seed,
                    cache,
                }
                .normalised()
            },
        )
}

/// Scenarios on wide lattices, where stitched regions have a stage-one core of their own.
fn wide_scenario() -> impl Strategy<Value = Scenario> {
    (scenario(), 1usize..=4, 8usize..=16).prop_map(|(mut sc, rows, cols)| {
        sc.rows = rows;
        sc.cols = cols;
        sc.normalised()
    })
}

/// Mostly small lattices, sometimes wide ones.
fn mixed_scenario() -> impl Strategy<Value = Scenario> {
    prop_oneof![2 => scenario(), 1 => wide_scenario()]
}

/// Cut coordinates that split a lattice into `parts` strips of about the same width.
fn cuts_for(sc: &Scenario, by_rows: bool, parts: usize, jitter: &[usize]) -> Vec<usize> {
    let extent = if by_rows { sc.rows } else { sc.cols };
    (1..parts)
        .map(|i| (i * extent / parts + jitter[i - 1]).min(extent))
        .collect()
}

impl Opt {
    /// Spreads the random search over some of the scenarios.
    fn min_by_seed(self, seed: u64) -> Self {
        if self == Opt::Coordinate && matches!(seed % 5, 0) {
            Opt::RandomSearch
        } else {
            self
        }
    }
}

/// The proptest configuration. The environment variable `FAULTTUNE_PROPTEST_CASES` overrides the
/// number of cases (for longer runs, for example overnight).
fn config_for_tests(cases: u32) -> ProptestConfig {
    let cases = std::env::var("FAULTTUNE_PROPTEST_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(cases);
    ProptestConfig {
        cases,
        // Do not write regression files next to the sources.
        failure_persistence: None,
        ..ProptestConfig::default()
    }
}

// ---------------------------------------------------------------------------------------------
// Helpers on runs
// ---------------------------------------------------------------------------------------------

/// The result of a run together with the events it emitted.
struct Run {
    db: ParameterDatabase<f64>,
    report: Report,
    events: Vec<Event>,
}

fn run_scenario(
    sc: &Scenario,
    graph: &ProcessorGraph,
    model: &impl ErrorModel<Value = f64>,
    config: SnakeConfig,
) -> Result<Run, FaultTuneError> {
    let recorder = RecordingObserver::new();
    let snake = sc.snake_with(graph, config).observer(recorder.clone());
    let (db, report) = snake.calibrate_with_report(model)?;
    Ok(Run {
        db,
        report,
        events: recorder.events(),
    })
}

fn committed(events: &[Event]) -> Vec<&StepRecord> {
    events
        .iter()
        .filter_map(|e| match e {
            Event::StepCommitted(r) => Some(r),
            _ => None,
        })
        .collect()
}

fn fail(error: FaultTuneError) -> TestCaseError {
    TestCaseError::fail(format!("unexpected error: {error}"))
}

/// The step records of a database as `(index, evaluations)`.
fn evaluations_of(db: &ParameterDatabase<f64>) -> Vec<(u64, u64)> {
    db.steps().map(|s| (s.index, s.evaluations)).collect()
}

/// Two runs agree on everything but timings and the order of events across subgoals.
fn same_run(a: &Run, b: &Run) -> Result<(), TestCaseError> {
    prop_assert!(a.db.same_outcome(&b.db), "databases differ");
    prop_assert_eq!(a.report.passes, b.report.passes);
    prop_assert_eq!(a.report.steps, b.report.steps);
    prop_assert_eq!(a.report.evaluations, b.report.evaluations);
    prop_assert_eq!(&a.report.skipped, &b.report.skipped);
    let key = |r: &&StepRecord| (r.pass, r.subgoal, r.index);
    let mut ea = committed(&a.events);
    let mut eb = committed(&b.events);
    ea.sort_by_key(key);
    eb.sort_by_key(key);
    prop_assert_eq!(ea.len(), eb.len());
    for (x, y) in ea.iter().zip(&eb) {
        prop_assert!(x.same_outcome(y), "event streams differ");
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Property: completeness and consistency
// ---------------------------------------------------------------------------------------------

/// Checks everything that must hold for a completed from-scratch run.
fn check_complete_run(
    sc: &Scenario,
    graph: &ProcessorGraph,
    model: &impl ErrorModel<Value = f64>,
    run: &Run,
) -> Result<(), TestCaseError> {
    let goal = sc.goal_set(graph);
    let Run { db, report, events } = run;

    // P* = G*, nothing skipped.
    prop_assert_eq!(db.calibrated(), &goal);
    prop_assert!(report.skipped.is_empty());
    prop_assert!(db
        .history()
        .iter()
        .all(|h| matches!(h, HistoryRecord::Step(_))));
    prop_assert_eq!(report.steps, db.step_count());
    prop_assert_eq!(report.steps as usize, db.history().len());

    // Every goal element is a parameter of exactly one step; values are options.
    let mut learned_at: HashMap<ElementId, u64> = HashMap::new();
    for step in db.steps() {
        for (&p, &option) in step.parameters.iter().zip(&step.option_indices) {
            prop_assert!(goal.contains(p), "{} is not in the goal", graph.label(p));
            prop_assert!(
                learned_at.insert(p, step.index).is_none(),
                "{} was calibrated twice",
                graph.label(p)
            );
            let entry = db.get(p).expect("calibrated");
            prop_assert_eq!(entry.step, step.index);
            prop_assert_eq!(entry.option_index, option);
            prop_assert_eq!(entry.value, model.options(p)[option as usize]);
        }
    }
    prop_assert_eq!(learned_at.len(), goal.len());

    // Step records are consistent with the definitions of the Snake.
    let hoods = Neighbourhoods::build(graph, 3);
    let masks = sc.algorithm(graph).masks();
    for (position, step) in db.steps().enumerate() {
        prop_assert_eq!(step.index, position as u64);
        prop_assert!(step.parameters.contains(&step.centre));
        let mut distinct = step.parameters.clone();
        distinct.sort();
        distinct.dedup();
        prop_assert_eq!(distinct.len(), step.parameters.len());
        for &p in &step.parameters {
            prop_assert!(hoods.is_within(step.centre, p, sc.d_p));
        }
        prop_assert!(step.constraints.windows(2).all(|w| w[0] < w[1]));
        for &c in &step.constraints {
            prop_assert!(!step.parameters.contains(&c));
            // A constraint is a calibrated element, learned by an earlier step ...
            let at = learned_at.get(&c).copied();
            prop_assert!(
                at.is_some_and(|at| at < step.index),
                "constraint {} is not older",
                c
            );
            // ... within d_R of a parameter it can be active with.
            prop_assert!(
                step.parameters
                    .iter()
                    .any(|&p| hoods.is_within(p, c, sc.d_r) && masks.co_active(p, c)),
                "{} constrains nothing",
                graph.label(c)
            );
        }
        prop_assert!(step.evaluations >= 1);
        prop_assert!(!step.error.is_nan());
    }
    // The converse: every calibrated element within reach of a step's parameters and active
    // with them, that was learned earlier, is a constraint.
    for step in db.steps() {
        for &p in &step.parameters {
            for &c in hoods.within(p, sc.d_r) {
                let older = learned_at.get(&c).is_some_and(|&at| at < step.index);
                if older && masks.co_active(p, c) && !step.parameters.contains(&c) {
                    // Elements learned by an earlier step of another subgoal of the same pass
                    // cannot be within d_R (subgoals are more than max(d_P, d_R) apart).
                    prop_assert!(
                        step.constraints.contains(&c),
                        "{} should constrain the step at {}",
                        graph.label(c),
                        graph.label(step.centre)
                    );
                }
            }
        }
    }

    // The sums of the report.
    prop_assert_eq!(
        report.evaluations,
        db.steps().map(|s| s.evaluations).sum::<u64>()
    );

    // The event stream is well formed and agrees with the database.
    prop_assert!(matches!(events.last(), Some(Event::Finished(_))));
    if let Some(Event::Finished(finished)) = events.last() {
        prop_assert_eq!(finished.steps, report.steps);
        prop_assert_eq!(finished.evaluations, report.evaluations);
        prop_assert_eq!(finished.passes, report.passes);
    }
    let mut next_pass = 0u32;
    let mut open = false;
    let mut calibrated_total = 0usize;
    for event in events {
        match event {
            Event::PassStarted {
                pass,
                subgoals,
                uncalibrated,
            } => {
                prop_assert!(!open);
                prop_assert_eq!(*pass, next_pass);
                prop_assert!(*subgoals >= 1 && *uncalibrated >= *subgoals);
                if *pass == 0 {
                    prop_assert_eq!(*uncalibrated, goal.len());
                }
                open = true;
            }
            Event::PassFinished { pass, calibrated } => {
                prop_assert!(open);
                prop_assert_eq!(*pass, next_pass);
                calibrated_total += *calibrated;
                next_pass += 1;
                open = false;
            }
            Event::StepCommitted(_) => prop_assert!(open),
            Event::StepFailed { .. } | Event::Invalidated { .. } => {
                prop_assert!(false, "unexpected event {:?}", event)
            }
            Event::Finished(_) => {}
        }
    }
    prop_assert!(!open);
    prop_assert_eq!(next_pass, report.passes);
    prop_assert_eq!(calibrated_total, goal.len());

    // Events per (pass, subgoal) are the steps of the database, in commit order, with the
    // ordinal within the subgoal as index.
    let mut by_subgoal: BTreeMap<(u32, u32), Vec<&StepRecord>> = BTreeMap::new();
    for record in committed(events) {
        by_subgoal
            .entry((record.pass, record.subgoal))
            .or_default()
            .push(record);
    }
    let mut from_db: BTreeMap<(u32, u32), Vec<&StepRecord>> = BTreeMap::new();
    for record in db.steps() {
        from_db
            .entry((record.pass, record.subgoal))
            .or_default()
            .push(record);
    }
    prop_assert_eq!(by_subgoal.len(), from_db.len());
    for (key, events) in &by_subgoal {
        let steps = &from_db[key];
        prop_assert_eq!(events.len(), steps.len());
        for (ordinal, (event, step)) in events.iter().zip(steps).enumerate() {
            prop_assert_eq!(event.index, ordinal as u64);
            prop_assert_eq!(event.centre, step.centre);
            prop_assert_eq!(&event.parameters, &step.parameters);
            prop_assert_eq!(&event.option_indices, &step.option_indices);
            prop_assert_eq!(&event.constraints, &step.constraints);
            prop_assert_eq!(event.evaluations, step.evaluations);
        }
    }
    // The global order of the database is by pass, then by subgoal.
    prop_assert!(db
        .steps()
        .zip(db.steps().skip(1))
        .all(|(a, b)| (a.pass, a.subgoal) <= (b.pass, b.subgoal)));
    Ok(())
}

/// Calibration ends with `P* = G*` and a consistent history, whatever the configuration.
fn check_calibration_completes_with_a_consistent_history(
    sc: Scenario,
) -> Result<(), TestCaseError> {
    let graph = sc.graph();
    let model = sc.model(&graph);
    let run =
        run_scenario(&sc, &graph, &model, sc.config(Some(1), sc.cache_policy())).map_err(fail)?;
    check_complete_run(&sc, &graph, &model, &run)?;
    Ok(())
}

proptest! {
    #![proptest_config(config_for_tests(60))]

    /// Calibration ends with `P* = G*` and a consistent history, whatever the configuration.
    #[test]
    fn calibration_completes_with_a_consistent_history(sc in scenario()) {
        check_calibration_completes_with_a_consistent_history(sc)?;
    }
}

/// With a failing model the run ends with every goal element calibrated or reported as
/// skipped (never lost), and aborting keeps the committed steps so that a second call
/// resumes.
fn check_failures_are_reported_and_progress_is_kept(
    sc: Scenario,
    salt: u64,
) -> Result<(), TestCaseError> {
    let graph = sc.graph();
    let goal = sc.goal_set(&graph);
    let model = sc.model(&graph);
    let bad: Vec<bool> = graph
        .ids()
        .map(|id| matches!((u64::from(id.raw()) * 2654435761 + salt) % 4, 0))
        .collect();
    let flaky = Flaky {
        inner: &model,
        bad: bad.clone(),
    };

    // SkipElement: complete or reported.
    let recorder = RecordingObserver::new();
    let config = SnakeConfig {
        on_model_error: ModelErrorPolicy::SkipElement,
        ..sc.config(Some(1), sc.cache_policy())
    };
    let snake = sc.snake_with(&graph, config).observer(recorder.clone());
    let (db, report) = snake.calibrate_with_report(&flaky).map_err(fail)?;
    let skipped: BTreeSet<ElementId> = report.skipped.iter().copied().collect();
    prop_assert_eq!(skipped.len(), report.skipped.len());
    let calibrated: BTreeSet<ElementId> = db.calibrated().iter().collect();
    prop_assert!(calibrated.is_disjoint(&skipped));
    let union: BTreeSet<ElementId> = calibrated.union(&skipped).copied().collect();
    prop_assert_eq!(union, goal.iter().collect::<BTreeSet<_>>());
    for &s in &skipped {
        prop_assert!(
            bad[s.index()],
            "{} was skipped but its model never fails",
            graph.label(s)
        );
    }
    let failed_events: BTreeSet<ElementId> = recorder
        .events()
        .iter()
        .filter_map(|e| match e {
            Event::StepFailed { centre, .. } => Some(*centre),
            _ => None,
        })
        .collect();
    prop_assert_eq!(failed_events, skipped);
    // No step is centred on a bad element.
    prop_assert!(db.steps().all(|s| !bad[s.centre.index()]));

    // Abort: either the run never needed a bad centre, or it stops with a model error
    // that names a bad centre, and the database keeps the steps committed before.
    let mut partial = ParameterDatabase::new(&graph);
    let snake = sc.snake(&graph, Some(1));
    match snake.calibrate_into(&mut partial, &flaky) {
        Ok(report) => {
            prop_assert!(report.skipped.is_empty());
            prop_assert_eq!(partial.calibrated(), &goal);
        }
        Err(FaultTuneError::Model { element, .. }) => {
            prop_assert!(bad[element.index()]);
            prop_assert!(partial.steps().all(|s| !bad[s.centre.index()]));
            prop_assert!(!goal.is_subset(partial.calibrated()));
            let kept = partial.history().to_vec();
            // Resume with a model that works.
            snake.calibrate_into(&mut partial, &model).map_err(fail)?;
            prop_assert_eq!(partial.calibrated(), &goal);
            prop_assert_eq!(&partial.history()[..kept.len()], &kept[..]);
        }
        Err(other) => prop_assert!(false, "unexpected error {other}"),
    }
    Ok(())
}

proptest! {
    #![proptest_config(config_for_tests(40))]

    /// With a failing model the run ends with every goal element calibrated or reported as
    /// skipped (never lost), and aborting keeps the committed steps so that a second call
    /// resumes.
    #[test]
    fn failures_are_reported_and_progress_is_kept(sc in scenario(), salt in 0u64..1000) {
        check_failures_are_reported_and_progress_is_kept(sc, salt)?;
    }
}

// ---------------------------------------------------------------------------------------------
// Property: parallel == sequential
// ---------------------------------------------------------------------------------------------

/// A coordinate of an element on the lattice: the row or the column of its first endpoint.
fn coordinate(graph: &ProcessorGraph, id: ElementId, by_rows: bool) -> usize {
    let node = graph.endpoints(id).map_or(id, |[a, _]| a);
    let label = graph.label(node);
    let mut parts = label[1..].split('_');
    let row: usize = parts.next().unwrap().parse().unwrap();
    let col: usize = parts.next().unwrap().parse().unwrap();
    if by_rows {
        row
    } else {
        col
    }
}

/// Regions of the goal cut at the given coordinates; goal elements whose ID is a multiple of
/// `leave_out` (if non-zero) belong to no region.
fn regions(
    graph: &ProcessorGraph,
    goal: &ElementSet,
    by_rows: bool,
    cuts: &[usize],
    leave_out: u32,
) -> Vec<ElementSet> {
    let mut cuts: Vec<usize> = cuts.to_vec();
    cuts.sort_unstable();
    cuts.dedup();
    let mut regions = vec![ElementSet::new(graph.len()); cuts.len() + 1];
    for e in goal.iter() {
        if leave_out != 0 && e.raw() % leave_out == 0 {
            continue;
        }
        let c = coordinate(graph, e, by_rows);
        regions[cuts.partition_point(|&cut| cut <= c)].insert(e);
    }
    regions
}

/// The elements drifted: a selection of the goal.
fn pick(goal: &ElementSet, picks: &[usize]) -> Vec<ElementId> {
    let all = goal.to_vec();
    if all.is_empty() {
        return Vec::new();
    }
    let mut chosen: Vec<ElementId> = picks.iter().map(|&p| all[p % all.len()]).collect();
    chosen.sort();
    chosen.dedup();
    chosen
}

/// What one thread count produced: a calibration, a re-calibration of scattered elements of the
/// sequential calibration, and a stitched calibration.
struct Outcome {
    run: Run,
    recalibrated: ParameterDatabase<f64>,
    recal_report: Report,
    stitched: ParameterDatabase<f64>,
    stitch_report: Report,
}

/// Runs the three operations of [`Outcome`] with `threads` worker threads. The re-calibration
/// starts from `base` (the sequential calibration) when given, so that only the re-calibration
/// itself is compared across thread counts.
fn outcome(
    sc: &Scenario,
    graph: &ProcessorGraph,
    model: &AnyModel,
    threads: usize,
    drifted: &[ElementId],
    parts: &[ElementSet],
    base: Option<&ParameterDatabase<f64>>,
) -> Result<Outcome, TestCaseError> {
    let config = sc.config(Some(threads), sc.cache_policy());
    let run = run_scenario(sc, graph, model, config.clone()).map_err(fail)?;
    let snake = sc.snake_with(graph, config);

    let mut recalibrated = base.unwrap_or(&run.db).clone();
    let recal_report = snake
        .recalibrate(&mut recalibrated, drifted, model)
        .map_err(fail)?;

    let plan = snake.stitch_plan(parts.to_vec()).map_err(fail)?;
    let mut stitched = ParameterDatabase::new(graph);
    let stitch_report = snake.stitch(&mut stitched, &plan, model).map_err(fail)?;
    prop_assert_eq!(stitched.calibrated(), &sc.goal_set(graph));
    Ok(Outcome {
        run,
        recalibrated,
        recal_report,
        stitched,
        stitch_report,
    })
}

/// Calibration, re-calibration of scattered elements and stitching give identical results
/// at 1, 2, 4 and 8 threads.
fn check_parallel_equals_sequential(
    sc: Scenario,
    picks: Vec<usize>,
    parts_count: usize,
    jitter: Vec<usize>,
    by_rows: bool,
    leave_out: u32,
) -> Result<(), TestCaseError> {
    let graph = sc.graph();
    let goal = sc.goal_set(&graph);
    let model = sc.model(&graph);
    let drifted = pick(&goal, &picks);
    let cuts = cuts_for(&sc, by_rows, parts_count, &jitter);
    let parts = regions(&graph, &goal, by_rows, &cuts, leave_out);

    let sequential = outcome(&sc, &graph, &model, 1, &drifted, &parts, None)?;
    for threads in [2usize, 4, 8] {
        let parallel = outcome(
            &sc,
            &graph,
            &model,
            threads,
            &drifted,
            &parts,
            Some(&sequential.run.db),
        )?;
        same_run(&sequential.run, &parallel.run)?;
        prop_assert!(
            sequential.recalibrated.same_outcome(&parallel.recalibrated),
            "recalibrate differs at {threads} threads"
        );
        prop_assert_eq!(sequential.recal_report.steps, parallel.recal_report.steps);
        prop_assert_eq!(
            sequential.recal_report.evaluations,
            parallel.recal_report.evaluations
        );
        prop_assert!(
            sequential.stitched.same_outcome(&parallel.stitched),
            "stitch differs at {threads} threads"
        );
        prop_assert_eq!(sequential.stitch_report.steps, parallel.stitch_report.steps);
        prop_assert_eq!(
            sequential.stitch_report.evaluations,
            parallel.stitch_report.evaluations
        );
        prop_assert_eq!(
            sequential.stitch_report.passes,
            parallel.stitch_report.passes
        );
    }
    Ok(())
}

proptest! {
    #![proptest_config(config_for_tests(40))]

    /// Calibration, re-calibration of scattered elements and stitching give identical results
    /// at 1, 2, 4 and 8 threads.
    #[test]
    fn parallel_equals_sequential(
        sc in mixed_scenario(),
        picks in proptest::collection::vec(0usize..10_000, 1..8),
        parts_count in 2usize..=4,
        jitter in proptest::collection::vec(0usize..3, 3),
        by_rows in any::<bool>(),
        leave_out in 0u32..6,
    ) {
        check_parallel_equals_sequential(sc, picks, parts_count, jitter, by_rows, leave_out)?;
    }
}

/// The parallel test above is only meaningful if some runs have several subgoals: this
/// scenario has hundreds, and the result is still the sequential one.
#[test]
fn many_subgoals_run_in_parallel_with_the_sequential_result() {
    let sc = Scenario {
        rows: 6,
        cols: 6,
        d_p: 0,
        d_r: 0,
        d_t: 1,
        expiry: 1,
        heuristic: Heur::Random,
        optimizer: Opt::RandomSearch,
        seeds: Seeds::Random,
        xeb: false,
        goal: Goal::NodesAndEngineered,
        model: ModelKind::Gate,
        k: 4,
        interaction: 0,
        seed: 17,
        cache: 0,
    };
    let graph = sc.graph();
    let model = sc.model(&graph);
    let reference = run_scenario(
        &sc,
        &graph,
        &model,
        sc.config(Some(1), CachePolicy::StepOnly),
    )
    .unwrap();
    let Some(Event::PassStarted { subgoals, .. }) = reference.events.first() else {
        panic!("the first event is PassStarted");
    };
    assert_eq!(
        *subgoals,
        sc.goal_set(&graph).len(),
        "with d_P = d_R = 0 every element is a subgoal"
    );
    for threads in [2, 4, 8] {
        let run = run_scenario(
            &sc,
            &graph,
            &model,
            sc.config(Some(threads), CachePolicy::StepOnly),
        )
        .unwrap();
        same_run(&reference, &run).unwrap();
    }
    let by_default =
        run_scenario(&sc, &graph, &model, sc.config(None, CachePolicy::StepOnly)).unwrap();
    same_run(&reference, &by_default).unwrap();
}

// ---------------------------------------------------------------------------------------------
// Property: the same seed gives the same result
// ---------------------------------------------------------------------------------------------

/// Two runs with the same inputs and seed agree, also across separately built Snakes and
/// models, across `calibrate` and `calibrate_into`, and when one Snake is reused.
fn check_the_same_seed_gives_the_same_result(
    sc: Scenario,
    threads: Option<usize>,
) -> Result<(), TestCaseError> {
    let graph = sc.graph();
    let config = sc.config(threads, sc.cache_policy());
    let first = run_scenario(&sc, &graph, &sc.model(&graph), config.clone()).map_err(fail)?;
    let second = run_scenario(&sc, &graph, &sc.model(&graph), config.clone()).map_err(fail)?;
    same_run(&first, &second)?;

    // The same Snake twice, and into a fresh database.
    let model = sc.model(&graph);
    let snake = sc.snake_with(&graph, config);
    let once = snake.calibrate(&model).map_err(fail)?;
    let twice = snake.calibrate(&model).map_err(fail)?;
    prop_assert!(once.same_outcome(&twice));
    prop_assert!(once.same_outcome(&first.db));
    let mut into = ParameterDatabase::new(&graph);
    snake.calibrate_into(&mut into, &model).map_err(fail)?;
    prop_assert!(into.same_outcome(&once));
    Ok(())
}

proptest! {
    #![proptest_config(config_for_tests(50))]

    /// Two runs with the same inputs and seed agree, also across separately built Snakes and
    /// models, across `calibrate` and `calibrate_into`, and when one Snake is reused.
    #[test]
    fn the_same_seed_gives_the_same_result(sc in scenario(), threads in prop_oneof![Just(None), Just(Some(1usize)), Just(Some(3usize))]) {
        check_the_same_seed_gives_the_same_result(sc, threads)?;
    }
}

/// The seed matters when something random is involved (the test would pass vacuously if every
/// seed gave the same result).
#[test]
fn different_seeds_can_differ() {
    let sc = Scenario {
        rows: 5,
        cols: 5,
        d_p: 1,
        d_r: 2,
        d_t: 2,
        expiry: 1,
        heuristic: Heur::Random,
        optimizer: Opt::RandomSearch,
        seeds: Seeds::Random,
        xeb: false,
        goal: Goal::NodesAndEngineered,
        model: ModelKind::Gate,
        k: 4,
        interaction: 0,
        seed: 1,
        cache: 0,
    };
    let graph = sc.graph();
    let model = sc.model(&graph);
    let mut config = sc.config(Some(1), CachePolicy::StepOnly);
    let a = run_scenario(&sc, &graph, &model, config.clone()).unwrap();
    config.seed = 2;
    let b = run_scenario(&sc, &graph, &model, config).unwrap();
    assert!(
        !a.db.same_outcome(&b.db),
        "two seeds gave the same random traversal"
    );
}

// ---------------------------------------------------------------------------------------------
// Property: evaluations
// ---------------------------------------------------------------------------------------------

/// The evaluations of every step are deterministic (across runs, thread counts and cache
/// policies), bounded by the search space, and the accounting of model calls, evaluations,
/// cache hits and misses is exact.
fn check_evaluations_per_step_are_deterministic_and_accounted(
    sc: Scenario,
) -> Result<(), TestCaseError> {
    let graph = sc.graph();
    let model = sc.model(&graph);

    // Step only: the model sees exactly the distinct candidates of every step.
    let counted = Counting::new(&model);
    let base = run_scenario(
        &sc,
        &graph,
        &counted,
        sc.config(Some(1), CachePolicy::StepOnly),
    )
    .map_err(fail)?;
    prop_assert_eq!(counted.calls(), base.report.evaluations);
    prop_assert_eq!(base.report.cache_hits + base.report.cache_misses, 0);
    let evaluations = evaluations_of(&base.db);

    // Bounded by the search space: at least one, at most the number of combinations.
    for step in base.db.steps() {
        let combos = step
            .parameters
            .iter()
            .map(|&p| model.options(p).len() as u128)
            .product::<u128>();
        prop_assert!(u128::from(step.evaluations) >= 1);
        prop_assert!(u128::from(step.evaluations) <= combos);
        if sc.optimizer == Opt::Exhaustive {
            prop_assert_eq!(u128::from(step.evaluations), combos);
        }
        if sc.optimizer == Opt::RandomSearch {
            prop_assert!(step.evaluations <= 40);
        }
    }

    // Again, and at other thread counts: the same evaluations.
    for threads in [1usize, 4] {
        let again = run_scenario(
            &sc,
            &graph,
            &model,
            sc.config(Some(threads), CachePolicy::StepOnly),
        )
        .map_err(fail)?;
        prop_assert_eq!(&evaluations, &evaluations_of(&again.db));
    }

    // With the cross-step cache: the same evaluations per step (they are counted before
    // the cache), hits and misses account for all of them, and the model is asked only for
    // the misses.
    for capacity in [3usize, 1 << 16] {
        for threads in [1usize, 4] {
            let counted = Counting::new(&model);
            let cached = run_scenario(
                &sc,
                &graph,
                &counted,
                sc.config(Some(threads), CachePolicy::AcrossSteps { capacity }),
            )
            .map_err(fail)?;
            prop_assert_eq!(&evaluations, &evaluations_of(&cached.db));
            prop_assert_eq!(cached.report.evaluations, base.report.evaluations);
            prop_assert_eq!(
                cached.report.cache_hits + cached.report.cache_misses,
                cached.report.evaluations
            );
            prop_assert_eq!(counted.calls(), cached.report.cache_misses);
            prop_assert!(cached.report.cache_misses <= cached.report.evaluations);
            if let Some(rate) = cached.report.cache_hit_rate() {
                prop_assert!((0.0..=1.0).contains(&rate));
            }
            prop_assert!(cached.db.same_outcome(&base.db));
        }
    }
    Ok(())
}

proptest! {
    #![proptest_config(config_for_tests(40))]

    /// The evaluations of every step are deterministic (across runs, thread counts and cache
    /// policies), bounded by the search space, and the accounting of model calls, evaluations,
    /// cache hits and misses is exact.
    #[test]
    fn evaluations_per_step_are_deterministic_and_accounted(sc in scenario()) {
        check_evaluations_per_step_are_deterministic_and_accounted(sc)?;
    }
}

// ---------------------------------------------------------------------------------------------
// Property: caches never change a result
// ---------------------------------------------------------------------------------------------

/// `AcrossSteps` (tiny and large, own or shared) equals `StepOnly`, and a warm shared cache
/// answers a repeated run without a single model call.
fn check_across_steps_equals_step_only(sc: Scenario, capacity: usize) -> Result<(), TestCaseError> {
    let graph = sc.graph();
    let model = sc.model(&graph);
    let reference = run_scenario(
        &sc,
        &graph,
        &model,
        sc.config(Some(1), CachePolicy::StepOnly),
    )
    .map_err(fail)?;

    for threads in [1usize, 4] {
        for policy in [
            CachePolicy::AcrossSteps { capacity },
            CachePolicy::across_steps(),
        ] {
            let run = run_scenario(&sc, &graph, &model, sc.config(Some(threads), policy))
                .map_err(fail)?;
            prop_assert!(
                run.db.same_outcome(&reference.db),
                "{policy:?} at {threads} threads changed the result"
            );
        }
    }

    // A shared cache, large enough to keep everything: the second run is all hits.
    let cache = Arc::new(CrossStepCache::new(1 << 18));
    let counted = Counting::new(&model);
    let config = sc.config(Some(2), CachePolicy::StepOnly);
    let first = sc
        .snake_with(&graph, config.clone())
        .shared_cache(Arc::clone(&cache))
        .calibrate(&counted)
        .map_err(fail)?;
    prop_assert!(first.same_outcome(&reference.db));
    let calls_after_first = counted.calls();
    prop_assert_eq!(calls_after_first, cache.stats().misses);
    let (second, report) = sc
        .snake_with(&graph, config.clone())
        .shared_cache(Arc::clone(&cache))
        .calibrate_with_report(&counted)
        .map_err(fail)?;
    prop_assert!(second.same_outcome(&reference.db));
    prop_assert_eq!(
        counted.calls(),
        calls_after_first,
        "the warm cache must answer everything"
    );
    prop_assert_eq!(report.cache_misses, 0);
    prop_assert_eq!(report.cache_hits, report.evaluations);

    // Another traversal on the same shared cache still equals its own uncached run.
    let other = Scenario {
        heuristic: Heur::Depth,
        ..sc.clone()
    };
    let other_reference = run_scenario(
        &other,
        &graph,
        &model,
        sc.config(Some(1), CachePolicy::StepOnly),
    )
    .map_err(fail)?;
    let shared = other
        .snake_with(&graph, config)
        .shared_cache(cache)
        .calibrate(&model)
        .map_err(fail)?;
    prop_assert!(shared.same_outcome(&other_reference.db));
    Ok(())
}

proptest! {
    #![proptest_config(config_for_tests(40))]

    /// `AcrossSteps` (tiny and large, own or shared) equals `StepOnly`, and a warm shared cache
    /// answers a repeated run without a single model call.
    #[test]
    fn across_steps_equals_step_only(sc in scenario(), capacity in 1usize..200) {
        check_across_steps_equals_step_only(sc, capacity)?;
    }
}

// ---------------------------------------------------------------------------------------------
// Property: re-calibration
// ---------------------------------------------------------------------------------------------

/// The goal elements within the expiry radius of `elements`.
fn removal_region(
    graph: &ProcessorGraph,
    goal: &ElementSet,
    elements: &[ElementId],
    radius: u32,
) -> BTreeSet<ElementId> {
    let hoods = Neighbourhoods::build(graph, radius);
    let mut region = BTreeSet::new();
    for &e in elements {
        for &x in hoods.within(e, radius) {
            if goal.contains(x) {
                region.insert(x);
            }
        }
    }
    region
}

/// Checks one re-calibration of `before` into `after`.
fn check_recalibration(
    sc: &Scenario,
    graph: &ProcessorGraph,
    drifted: &[ElementId],
    before: &ParameterDatabase<f64>,
    after: &ParameterDatabase<f64>,
    events: &[Event],
) -> Result<(), TestCaseError> {
    let goal = sc.goal_set(graph);
    let region = removal_region(graph, &goal, drifted, sc.expiry);
    let calibrated_before: BTreeSet<ElementId> = before.calibrated().iter().collect();
    let removed: Vec<ElementId> = region.intersection(&calibrated_before).copied().collect();

    // The goal is complete again.
    prop_assert_eq!(after.calibrated(), &goal);

    // Entries outside the region are bit for bit the old ones.
    for id in goal.iter() {
        let new = after.get(id).expect("calibrated");
        if region.contains(&id) {
            prop_assert!(
                new.step >= before.step_count(),
                "{} was not re-learned",
                graph.label(id)
            );
        } else {
            let old = before.get(id).expect("calibrated before");
            prop_assert_eq!(new.value.to_bits(), old.value.to_bits());
            prop_assert_eq!(new.option_index, old.option_index);
            prop_assert_eq!(new.step, old.step);
        }
    }

    // The history is append-only: the old records are a prefix, then the Invalidated record,
    // then steps that only touch the region.
    let old_len = before.history().len();
    prop_assert!(after.history().len() >= old_len);
    prop_assert_eq!(&after.history()[..old_len], before.history());
    let mut rest = after.history()[old_len..].iter();
    if removed.is_empty() {
        prop_assert!(matches!(
            rest.clone().next(),
            None | Some(HistoryRecord::Step(_))
        ));
    } else {
        match rest.next() {
            Some(HistoryRecord::Invalidated(listed)) => prop_assert_eq!(listed, &removed),
            other => prop_assert!(false, "expected an Invalidated record, found {:?}", other),
        }
    }
    let mut relearned = BTreeSet::new();
    let mut next_index = before.step_count();
    for record in rest {
        let HistoryRecord::Step(step) = record else {
            return Err(TestCaseError::fail("a second Invalidated record"));
        };
        prop_assert_eq!(step.index, next_index);
        next_index += 1;
        for &p in &step.parameters {
            prop_assert!(
                region.contains(&p),
                "{} is outside the removal region",
                graph.label(p)
            );
            prop_assert!(relearned.insert(p));
        }
        prop_assert!(region.contains(&step.centre));
    }
    prop_assert_eq!(&relearned, &region);
    prop_assert_eq!(after.step_count(), next_index);

    // The observer heard of the removal exactly once, before the first pass.
    let invalidations: Vec<&Vec<ElementId>> = events
        .iter()
        .filter_map(|e| match e {
            Event::Invalidated { elements, .. } => Some(elements),
            _ => None,
        })
        .collect();
    prop_assert_eq!(invalidations.len(), usize::from(!removed.is_empty()));
    if let Some(listed) = invalidations.first() {
        prop_assert_eq!(*listed, &removed);
        let at = events
            .iter()
            .position(|e| matches!(e, Event::Invalidated { .. }))
            .unwrap();
        let first_pass = events
            .iter()
            .position(|e| matches!(e, Event::PassStarted { .. }));
        prop_assert!(first_pass.is_none_or(|p| at < p));
    }
    // The step events are the new steps.
    let new_steps = after.steps().skip(before.step_count() as usize).count();
    prop_assert_eq!(committed(events).len(), new_steps);
    Ok(())
}

/// Re-calibration, twice in a row on different elements, changes only the removal region,
/// keeps the rest bit for bit and only appends to the history.
fn check_recalibrate_changes_only_the_removal_region(
    sc: Scenario,
    first: Vec<usize>,
    second: Vec<usize>,
) -> Result<(), TestCaseError> {
    let graph = sc.graph();
    let goal = sc.goal_set(&graph);
    if goal.is_empty() {
        return Ok(());
    }
    let model = sc.model(&graph);
    let config = sc.config(Some(2), sc.cache_policy());
    let mut db = sc
        .snake_with(&graph, config.clone())
        .calibrate(&model)
        .map_err(fail)?;
    for picks in [first, second] {
        let drifted = pick(&goal, &picks);
        let before = db.clone();
        let recorder = RecordingObserver::new();
        let snake = sc
            .snake_with(&graph, config.clone())
            .observer(recorder.clone());
        snake.recalibrate(&mut db, &drifted, &model).map_err(fail)?;
        check_recalibration(&sc, &graph, &drifted, &before, &db, &recorder.events())?;
    }
    Ok(())
}

proptest! {
    #![proptest_config(config_for_tests(60))]

    /// Re-calibration, twice in a row on different elements, changes only the removal region,
    /// keeps the rest bit for bit and only appends to the history.
    #[test]
    fn recalibrate_changes_only_the_removal_region(
        sc in scenario(),
        first in proptest::collection::vec(0usize..10_000, 1..6),
        second in proptest::collection::vec(0usize..10_000, 1..6),
    ) {
        check_recalibrate_changes_only_the_removal_region(sc, first, second)?;
    }
}

/// Re-calibrating an element outside the goal, or outside the graph, is an error that changes
/// nothing.
#[test]
fn recalibrate_refuses_bad_elements_without_changing_anything() {
    let graph = ProcessorGraph::square_lattice(3, 3);
    let goal = graph.goal(|e| e.kind() != ElementKind::Edge(EdgeType::Parasitic));
    let model = SeparableModel::new(&graph, 8, 0.0, 1.0, 1);
    let snake = Snake::new(&graph, goal.clone(), SnakeConfig::default());
    let mut db = snake.calibrate(&model).unwrap();
    let before = db.clone();
    let parasitic = graph
        .edges()
        .find(|&e| graph.kind(e) == ElementKind::Edge(EdgeType::Parasitic))
        .unwrap();
    for bad in [parasitic, ElementId::new(10_000)] {
        let err = snake
            .recalibrate(&mut db, &[graph.lattice_node(0, 0).unwrap(), bad], &model)
            .expect_err("not a goal element");
        assert!(matches!(err, FaultTuneError::InvalidConfig(_)), "{err:?}");
        assert!(
            db.same_outcome(&before),
            "a refused recalibration must not touch the database"
        );
    }
    // An empty list of elements is a no-op.
    let report = snake.recalibrate(&mut db, &[], &model).unwrap();
    assert_eq!(report.steps, 0);
    assert!(db.same_outcome(&before));
}

// ---------------------------------------------------------------------------------------------
// Property: stitching
// ---------------------------------------------------------------------------------------------

/// `stitch` gives a complete database; its stage-one regions were separate subgoals whose
/// steps stay inside their region, and stage two only touches the gap.
fn check_stitch_calibrates_regions_separately_and_then_the_gap(
    sc: Scenario,
    parts_count: usize,
    jitter: Vec<usize>,
    by_rows: bool,
    leave_out: u32,
    threads: usize,
) -> Result<(), TestCaseError> {
    let graph = sc.graph();
    let goal = sc.goal_set(&graph);
    let model = sc.model(&graph);
    let parts = regions(
        &graph,
        &goal,
        by_rows,
        &cuts_for(&sc, by_rows, parts_count, &jitter),
        leave_out,
    );
    let recorder = RecordingObserver::new();
    let snake = sc
        .snake_with(&graph, sc.config(Some(threads), sc.cache_policy()))
        .observer(recorder.clone());
    let plan = snake.stitch_plan(parts.clone()).map_err(fail)?;

    // The plan: stage one and the gap split the goal; regions are far apart.
    prop_assert!(plan.stage_one.is_disjoint(&plan.gap));
    prop_assert_eq!(plan.stage_one.union(&plan.gap), goal.clone());
    let separation = sc.d_p.max(sc.d_r);
    let hoods = Neighbourhoods::build(&graph, separation.max(1));
    let region_of = |e: ElementId| parts.iter().position(|r| r.contains(e));
    for a in plan.stage_one.iter() {
        prop_assert!(region_of(a).is_some());
        for &b in hoods.within(a, separation) {
            if plan.stage_one.contains(b) {
                prop_assert_eq!(
                    region_of(a),
                    region_of(b),
                    "stage-one regions are too close"
                );
            }
        }
    }

    let mut db = ParameterDatabase::new(&graph);
    let report = snake.stitch(&mut db, &plan, &model).map_err(fail)?;
    prop_assert_eq!(db.calibrated(), &goal);
    prop_assert!(report.skipped.is_empty());
    prop_assert_eq!(report.steps, db.step_count());
    prop_assert_eq!(
        report.evaluations,
        db.steps().map(|s| s.evaluations).sum::<u64>()
    );

    // `stitch` is one call: a single `Finished` event closes both stages. The stage of a step
    // follows from its centre: stage one calibrates only elements of `plan.stage_one`, stage two
    // only elements of the gap, and the two partition the goal.
    let events = recorder.events();
    prop_assert_eq!(
        events
            .iter()
            .filter(|e| matches!(e, Event::Finished(_)))
            .count(),
        1
    );
    prop_assert!(matches!(events.last(), Some(Event::Finished(_))));
    let stage_one_records: Vec<&StepRecord> = committed(&events)
        .into_iter()
        .filter(|r| plan.stage_one.contains(r.centre))
        .collect();
    let stage_two_records: Vec<&StepRecord> = committed(&events)
        .into_iter()
        .filter(|r| !plan.stage_one.contains(r.centre))
        .collect();

    // Stage one: every step stays inside one region, a subgoal never spans two regions,
    // and each non-empty region got a subgoal of its own.
    let mut subgoal_region: HashMap<(u32, u32), usize> = HashMap::new();
    let mut stage_one_steps = 0;
    for record in stage_one_records.iter().copied() {
        stage_one_steps += 1;
        prop_assert!(plan.stage_one.contains(record.centre));
        let region = region_of(record.centre);
        prop_assert!(region.is_some());
        for &p in &record.parameters {
            prop_assert_eq!(
                region_of(p),
                region,
                "a parameter left the region of its centre"
            );
            prop_assert!(plan.stage_one.contains(p));
        }
        for &c in &record.constraints {
            prop_assert_eq!(
                region_of(c),
                region,
                "a constraint came from another region"
            );
        }
        let entry = subgoal_region
            .entry((record.pass, record.subgoal))
            .or_insert(region.unwrap());
        prop_assert_eq!(*entry, region.unwrap(), "one subgoal served two regions");
    }
    let non_empty_regions = parts
        .iter()
        .filter(|r| r.iter().any(|e| plan.stage_one.contains(e)))
        .count();
    if non_empty_regions > 0 {
        let first_pass = events.iter().find_map(|e| match e {
            Event::PassStarted {
                pass: 0, subgoals, ..
            } => Some(*subgoals),
            _ => None,
        });
        prop_assert!(first_pass.is_some_and(|s| s >= non_empty_regions));
        let regions_seen: BTreeSet<usize> = subgoal_region.values().copied().collect();
        prop_assert_eq!(regions_seen.len(), non_empty_regions);
    } else {
        prop_assert_eq!(stage_one_steps, 0);
    }

    // Stage two only touches the gap.
    for record in stage_two_records.iter().copied() {
        prop_assert!(plan.gap.contains(record.centre));
        for &p in &record.parameters {
            prop_assert!(plan.gap.contains(p));
        }
    }
    let stage_two_steps = stage_two_records.len();
    prop_assert_eq!(stage_one_steps + stage_two_steps, db.step_count() as usize);

    // Whatever the thread count, the result is the sequential one.
    if threads != 1 {
        let sequential = sc.snake_with(&graph, sc.config(Some(1), sc.cache_policy()));
        let mut reference = ParameterDatabase::new(&graph);
        sequential
            .stitch(&mut reference, &plan, &model)
            .map_err(fail)?;
        prop_assert!(db.same_outcome(&reference));
    }
    Ok(())
}

proptest! {
    #![proptest_config(config_for_tests(60))]

    /// `stitch` gives a complete database; its stage-one regions were separate subgoals whose
    /// steps stay inside their region, and stage two only touches the gap.
    #[test]
    fn stitch_calibrates_regions_separately_and_then_the_gap(
        sc in mixed_scenario(),
        parts_count in 2usize..=4,
        jitter in proptest::collection::vec(0usize..3, 3),
        by_rows in any::<bool>(),
        leave_out in 0u32..6,
        threads in prop_oneof![Just(1usize), Just(2usize), Just(4usize)],
    ) {
        check_stitch_calibrates_regions_separately_and_then_the_gap(sc, parts_count, jitter, by_rows, leave_out, threads)?;
    }
}

/// Stage-one regions of a big lattice run in parallel: events of different regions interleave
/// in time (at least in the log, whatever the thread count) but stay inside their regions.
#[test]
fn stitched_regions_interleave_but_stay_inside() {
    let sc = Scenario {
        rows: 6,
        cols: 16,
        d_p: 1,
        d_r: 2,
        d_t: 2,
        expiry: 1,
        heuristic: Heur::Breadth,
        optimizer: Opt::Coordinate,
        seeds: Seeds::Lowest,
        xeb: false,
        goal: Goal::NodesAndEngineered,
        model: ModelKind::Gate,
        k: 4,
        interaction: 0,
        seed: 3,
        cache: 0,
    };
    let graph = sc.graph();
    let goal = sc.goal_set(&graph);
    let model = sc.model(&graph);
    let parts = regions(&graph, &goal, false, &[5, 11], 0);
    let recorder = RecordingObserver::new();
    let snake = sc
        .snake_with(&graph, sc.config(Some(4), CachePolicy::StepOnly))
        .observer(recorder.clone());
    let plan = snake.stitch_plan(parts.clone()).unwrap();
    assert!(!plan.gap.is_empty() && !plan.stage_one.is_empty());
    let mut db = ParameterDatabase::new(&graph);
    snake.stitch(&mut db, &plan, &model).unwrap();
    assert_eq!(db.calibrated(), &goal);

    let events = recorder.events();
    let region_of = |e: ElementId| parts.iter().position(|r| r.contains(e)).unwrap();
    let mut regions_in_order: Vec<usize> = Vec::new();
    for event in &events {
        if let Event::StepCommitted(record) = event {
            if !plan.stage_one.contains(record.centre) {
                continue;
            }
            let region = region_of(record.centre);
            assert!(record.parameters.iter().all(|&p| region_of(p) == region));
            regions_in_order.push(region);
        }
    }
    for (region, members) in parts.iter().enumerate() {
        assert!(
            regions_in_order.contains(&region),
            "region {region} has {} stage-one elements; steps per region: {:?}",
            members
                .iter()
                .filter(|&e| plan.stage_one.contains(e))
                .count(),
            regions_in_order
        );
    }
    // Within one region the commit order is the traversal; across regions the log is whatever
    // the scheduler made of it, so only the per-region order is compared with the sequential
    // run.
    let sequential_log = {
        let recorder = RecordingObserver::new();
        let snake = sc
            .snake_with(&graph, sc.config(Some(1), CachePolicy::StepOnly))
            .observer(recorder.clone());
        let mut db = ParameterDatabase::new(&graph);
        snake.stitch(&mut db, &plan, &model).unwrap();
        recorder.events()
    };
    let per_region = |events: &[Event]| {
        let mut map: BTreeMap<usize, Vec<ElementId>> = BTreeMap::new();
        for event in events {
            if let Event::StepCommitted(record) = event {
                if !plan.stage_one.contains(record.centre) {
                    continue;
                }
                map.entry(region_of(record.centre))
                    .or_default()
                    .push(record.centre);
            }
        }
        map
    };
    assert_eq!(per_region(&events), per_region(&sequential_log));
}
