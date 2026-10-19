//! The Snake: configuration and the calibration loop (ARCHITECTURE §4, §6).
//!
//! ```text
//! calibrate_into(db, goal, region = all):
//!     failed ← ∅
//!     loop:
//!         U ← (goal ∩ region) ∖ calibrated(db) ∖ failed
//!         if U = ∅: return report
//!         subgoals ← segment(U)
//!         snapshot ← read-only view of db
//!         results ← for each subgoal S, in parallel:
//!             overlay ← empty set of changes on top of snapshot
//!             for each thread T of S, nodes first:
//!                 seed ← seed_policy(T)
//!                 if seed is calibrated in overlay: continue
//!                 traverse(seed, S, overlay)
//!             return overlay
//!         merge results into db, in subgoal order
//!         if this pass neither calibrated nor skipped any element: fail with NoProgress
//! ```

mod objective;
mod subgoal;
#[cfg(test)]
mod tests;
mod trace;

use std::panic::resume_unwind;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use crate::activity::{ActivityMasks, AlgorithmSubgraphs};
use crate::cache::{CachePolicy, CrossStepCache};
use crate::database::{HistoryRecord, ParameterDatabase, StepRecord};
use crate::error::FaultTuneError;
use crate::events::{worst_first, Event, Observer, Report, MAX_WORST_STEPS};
use crate::graph::{ElementId, ProcessorGraph};
use crate::model::ErrorModel;
use crate::neighbourhoods::Neighbourhoods;
use crate::optimize::{CoordinateDescent, Optimizer, DEFAULT_BUDGET};
use crate::scope::{Distances, Scope};
use crate::segment::{segment, SeedPolicy, StitchPlan, Subgoal};
use crate::sets::ElementSet;
use crate::traverse::{BreadthFirst, TraversalHeuristic};

use subgoal::{run_task, Outcome, PassCtx, Shared, TaskResult};

/// The largest distance (`d_P`, `d_R`, `d_T` or the expiry radius) a configuration may use.
///
/// The neighbourhood index grows quickly with its depth; 64 is far beyond anything a
/// processor needs.
pub const MAX_DISTANCE: u32 = 64;

/// Stack size of the worker threads of a parallel run (16 MiB).
const WORKER_STACK_SIZE: usize = 16 << 20;

/// The largest number of worker threads a configuration may ask for.
pub const MAX_THREADS: usize = 4096;

/// What to do when the error model fails for a step.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum ModelErrorPolicy {
    /// Stop with `FaultTuneError::Model` (the default). Committed steps are kept.
    #[default]
    Abort,
    /// Call the failing model method again up to `times` more times, then abort. For flaky
    /// hardware measurements.
    Retry {
        /// Number of retries after the first failure.
        times: u32,
    },
    /// Give up on the step's centre: exclude it for the rest of the run and list it in
    /// `Report::skipped`.
    SkipElement,
}

/// Configuration of a Snake run.
#[derive(Clone, Debug)]
pub struct SnakeConfig {
    /// `d_P`: how many parameters are optimized together in one step.
    pub parameter_distance: u32,
    /// `d_R`: how far away calibrated elements can be and still constrain the step.
    pub constraint_distance: u32,
    /// `d_T`: how far the Snake can move from one step to the next. Two nodes are two incidence
    /// steps apart, so `d_T` must be at least 2 for the Snake to walk from node to node; with
    /// `d_T = 1` every pass makes a single step (correct, but quadratic in the processor size).
    pub traversal_distance: u32,
    /// Radius around an expired element that `recalibrate` forgets (paper: 1).
    pub expiry_radius: u32,
    /// Seed for all randomness. The same inputs and seed give the same result at any thread count.
    pub seed: u64,
    /// Worker threads for parallel subgoals; `None` uses all cores.
    pub threads: Option<usize>,
    /// What to do when the model fails.
    pub on_model_error: ModelErrorPolicy,
    /// Which caches to use.
    pub cache: CachePolicy,
    /// Evaluation budget handed to optimizers (`SearchSpace::budget`).
    pub evaluation_budget: u64,
}

impl Default for SnakeConfig {
    /// The XEB setting of the paper: `d_P = 1`, `d_R = 2`, `d_T = 2`.
    fn default() -> Self {
        Self {
            parameter_distance: 1,
            constraint_distance: 2,
            traversal_distance: 2,
            expiry_radius: 1,
            seed: 0,
            threads: None,
            on_model_error: ModelErrorPolicy::Abort,
            cache: CachePolicy::StepOnly,
            evaluation_budget: DEFAULT_BUDGET,
        }
    }
}

impl SnakeConfig {
    /// The three Snake distances.
    pub fn distances(&self) -> Distances {
        Distances {
            parameter: self.parameter_distance,
            constraint: self.constraint_distance,
            traversal: self.traversal_distance,
        }
    }

    /// Checks the configuration (for example `threads != Some(0)`, `evaluation_budget > 0`,
    /// `AcrossSteps` capacity > 0, distances small enough to index).
    ///
    /// Distances are limited to [`MAX_DISTANCE`]. Every problem is an
    /// [`FaultTuneError::InvalidConfig`].
    pub fn validate(&self) -> Result<(), FaultTuneError> {
        let invalid = |message: String| Err(FaultTuneError::InvalidConfig(message));
        if self.threads == Some(0) {
            return invalid("threads must be at least 1 (use None for all cores)".into());
        }
        if self.threads.is_some_and(|t| t > MAX_THREADS) {
            return invalid(format!("threads must be at most {MAX_THREADS}"));
        }
        if self.evaluation_budget == 0 {
            return invalid("evaluation_budget must be positive".into());
        }
        if let CachePolicy::AcrossSteps { capacity: 0 } = self.cache {
            return invalid("the cross-step cache needs a capacity above 0".into());
        }
        for (name, value) in [
            ("parameter_distance", self.parameter_distance),
            ("constraint_distance", self.constraint_distance),
            ("traversal_distance", self.traversal_distance),
            ("expiry_radius", self.expiry_radius),
        ] {
            if value > MAX_DISTANCE {
                return invalid(format!(
                    "{name} is {value}, but at most {MAX_DISTANCE} is supported"
                ));
            }
        }
        Ok(())
    }
}

/// A shared flag that cancels a run. Clones share the flag.
///
/// Checked between steps. A cancelled run finishes the steps in progress, merges what it has,
/// and returns `FaultTuneError::Cancelled` with the partial report; nothing committed is lost.
#[derive(Clone, Debug, Default)]
pub struct CancellationToken(Arc<AtomicBool>);

impl CancellationToken {
    /// A new token that is not cancelled.
    pub fn new() -> Self {
        Self::default()
    }

    /// Requests cancellation.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    /// Whether cancellation was requested.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }

    /// Clears the flag so the token can be reused.
    pub fn reset(&self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

#[derive(Clone)]
enum GraphRef<'g> {
    Borrowed(&'g ProcessorGraph),
    Shared(Arc<ProcessorGraph>),
}

impl GraphRef<'_> {
    fn get(&self) -> &ProcessorGraph {
        match self {
            Self::Borrowed(g) => g,
            Self::Shared(g) => g,
        }
    }
}

/// The Snake Optimizer.
///
/// ```ignore
/// let snake = Snake::new(&graph, goal, SnakeConfig::default())
///     .algorithm(AlgorithmSubgraphs::all_active(&graph))
///     .heuristic(BreadthFirst)
///     .optimizer(CoordinateDescent::default());
/// let mut params = snake.calibrate(&model)?;
/// snake.recalibrate(&mut params, &[qubit_7], &model)?;
/// ```
pub struct Snake<'g> {
    graph: GraphRef<'g>,
    goal: ElementSet,
    config: SnakeConfig,
    algorithm: Option<AlgorithmSubgraphs>,
    heuristic: Arc<dyn TraversalHeuristic>,
    optimizer: Arc<dyn Optimizer>,
    seed_policy: SeedPolicy,
    observers: Vec<Arc<dyn Observer>>,
    cancellation: CancellationToken,
    shared_cache: Option<Arc<CrossStepCache>>,
    /// Built on first use and kept: the neighbourhood index and the activity masks.
    setup: OnceLock<Setup>,
}

/// The data derived from the graph, the algorithm and the distances; built once per Snake.
struct Setup {
    neighbourhoods: Neighbourhoods,
    activity: ActivityMasks,
}

impl<'g> Snake<'g> {
    /// A Snake for `goal` (the elements whose parameters must be learned) on `graph`.
    /// Defaults: all elements active at once, breadth-first traversal, coordinate descent,
    /// lowest-ID seeds. Configuration errors are reported by the `calibrate_*` methods.
    pub fn new(graph: &'g ProcessorGraph, goal: ElementSet, config: SnakeConfig) -> Self {
        Self::with_graph(GraphRef::Borrowed(graph), goal, config)
    }

    fn with_graph(graph: GraphRef<'g>, goal: ElementSet, config: SnakeConfig) -> Self {
        Self {
            graph,
            goal,
            config,
            algorithm: None,
            heuristic: Arc::new(BreadthFirst),
            optimizer: Arc::new(CoordinateDescent::default()),
            seed_policy: SeedPolicy::default(),
            observers: Vec::new(),
            cancellation: CancellationToken::new(),
            shared_cache: None,
            setup: OnceLock::new(),
        }
    }

    /// Sets the algorithm subgraphs `A_i` (default: everything active at once).
    pub fn algorithm(mut self, subgraphs: AlgorithmSubgraphs) -> Self {
        self.algorithm = Some(subgraphs);
        // The activity masks were derived from the old subgraphs.
        self.setup = OnceLock::new();
        self
    }

    /// Sets the traversal heuristic.
    pub fn heuristic(mut self, heuristic: impl TraversalHeuristic + 'static) -> Self {
        self.heuristic = Arc::new(heuristic);
        self
    }

    /// Sets the traversal heuristic from a shared trait object.
    pub fn heuristic_arc(mut self, heuristic: Arc<dyn TraversalHeuristic>) -> Self {
        self.heuristic = heuristic;
        self
    }

    /// Sets the step optimizer.
    pub fn optimizer(mut self, optimizer: impl Optimizer + 'static) -> Self {
        self.optimizer = Arc::new(optimizer);
        self
    }

    /// Sets the step optimizer from a shared trait object.
    pub fn optimizer_arc(mut self, optimizer: Arc<dyn Optimizer>) -> Self {
        self.optimizer = optimizer;
        self
    }

    /// Sets how thread seeds are chosen.
    pub fn seed_policy(mut self, policy: SeedPolicy) -> Self {
        self.seed_policy = policy;
        self
    }

    /// Adds an observer (events are delivered to all observers).
    pub fn observer(mut self, observer: impl Observer + 'static) -> Self {
        self.observers.push(Arc::new(observer));
        self
    }

    /// Adds an observer from a shared trait object.
    pub fn observer_arc(mut self, observer: Arc<dyn Observer>) -> Self {
        self.observers.push(observer);
        self
    }

    /// Uses `token` to cancel runs.
    pub fn cancellation(mut self, token: CancellationToken) -> Self {
        self.cancellation = token;
        self
    }

    /// The processor graph.
    pub fn graph(&self) -> &ProcessorGraph {
        self.graph.get()
    }

    /// The calibration goal `G*`.
    pub fn goal(&self) -> &ElementSet {
        &self.goal
    }

    /// The configuration.
    pub fn config(&self) -> &SnakeConfig {
        &self.config
    }

    /// The cancellation token in use.
    pub fn cancellation_token(&self) -> &CancellationToken {
        &self.cancellation
    }

    /// Shares `cache` across `calibrate_*` calls (and across Snakes).
    ///
    /// By default [`CachePolicy::AcrossSteps`] gives every call a fresh cache, because a cache
    /// shared silently between different models would return wrong errors. Passing a cache
    /// here is the caller's assertion that every model used with it is pure and the same.
    /// The cache is used whatever the configured [`CachePolicy`] is.
    pub fn shared_cache(mut self, cache: Arc<CrossStepCache>) -> Self {
        self.shared_cache = Some(cache);
        self
    }

    /// Calibrates the whole goal from scratch.
    pub fn calibrate<M: ErrorModel>(
        &self,
        model: &M,
    ) -> Result<ParameterDatabase<M::Value>, FaultTuneError> {
        self.calibrate_with_report(model).map(|(db, _)| db)
    }

    /// Like [`calibrate`](Self::calibrate) but also returns the run report.
    pub fn calibrate_with_report<M: ErrorModel>(
        &self,
        model: &M,
    ) -> Result<(ParameterDatabase<M::Value>, Report), FaultTuneError> {
        let mut db = ParameterDatabase::new(self.graph());
        let report = self.run(&mut db, None, model)?;
        Ok((db, report))
    }

    /// Calibrates whatever part of the goal `db` has not calibrated yet. When it returns an
    /// error, every step committed before the error is still in `db`. Also the way to resume an
    /// interrupted run.
    pub fn calibrate_into<M: ErrorModel>(
        &self,
        db: &mut ParameterDatabase<M::Value>,
        model: &M,
    ) -> Result<Report, FaultTuneError> {
        self.run(db, None, model)
    }

    /// Like [`calibrate_into`](Self::calibrate_into) but only for goal elements in `region`.
    pub fn calibrate_region<M: ErrorModel>(
        &self,
        db: &mut ParameterDatabase<M::Value>,
        region: &ElementSet,
        model: &M,
    ) -> Result<Report, FaultTuneError> {
        self.run(db, Some(region), model)
    }

    /// Local re-calibration: forgets `within(g, expiry_radius)` of each expired element (a
    /// `HistoryRecord::Invalidated` is appended) and re-learns exactly that region, still subject
    /// to the constraints of everything around it.
    ///
    /// Every element must be in the goal. The forgotten elements are announced with
    /// [`Event::Invalidated`] before the first pass. If the configuration or the database is
    /// invalid nothing is forgotten.
    pub fn recalibrate<M: ErrorModel>(
        &self,
        db: &mut ParameterDatabase<M::Value>,
        elements: &[ElementId],
        model: &M,
    ) -> Result<Report, FaultTuneError> {
        self.check_inputs(db.capacity(), db.check_graph(self.graph()), None)?;
        let setup = self.setup();
        let graph = self.graph();
        let mut region = ElementSet::new(graph.len());
        for &e in elements {
            if e.index() >= graph.len() {
                return Err(FaultTuneError::InvalidConfig(format!(
                    "element {} is not in the graph",
                    e.raw()
                )));
            }
            if !self.goal.contains(e) {
                return Err(FaultTuneError::InvalidConfig(format!(
                    "element {} is not in the calibration goal and cannot be re-calibrated",
                    graph.label(e)
                )));
            }
            for &x in setup.neighbourhoods.within(e, self.config.expiry_radius) {
                if self.goal.contains(x) {
                    region.insert(x);
                }
            }
        }
        let removed: Vec<ElementId> = region.iter().filter(|&x| db.is_calibrated(x)).collect();
        if !removed.is_empty() {
            db.invalidate(&removed);
            let pass = next_pass(db);
            self.emit_with(|| Event::Invalidated {
                pass,
                elements: removed,
            });
        }
        self.run(db, Some(&region), model)
    }

    /// Computes a [`StitchPlan`] for `regions` of this Snake's goal.
    pub fn stitch_plan(&self, regions: Vec<ElementSet>) -> Result<StitchPlan, FaultTuneError> {
        self.config.validate()?;
        self.check_inputs(self.graph().len(), Ok(()), None)?;
        let setup = self.setup();
        let scope = Scope::new(
            self.graph(),
            &setup.neighbourhoods,
            &setup.activity,
            self.config.distances(),
        );
        StitchPlan::new(&scope, &self.goal, regions)
    }

    /// Runs a stitch plan: stage one calibrates the regions minus the gap (separate subgoals,
    /// in parallel); stage two calibrates the gap under constraints from both sides. The
    /// returned report covers both stages.
    pub fn stitch<M: ErrorModel>(
        &self,
        db: &mut ParameterDatabase<M::Value>,
        plan: &StitchPlan,
        model: &M,
    ) -> Result<Report, FaultTuneError> {
        let n = self.graph().len();
        if plan.stage_one.capacity() != n || plan.gap.capacity() != n {
            return Err(FaultTuneError::InvalidConfig(format!(
                "the stitch plan is for a graph of {} elements, but this one has {n}",
                plan.gap.capacity()
            )));
        }
        let mut first = self.run_stage(db, Some(&plan.stage_one), model)?;
        if first.ending.is_err() || first.cancelled {
            return self.finish(first);
        }
        let second = self.run_stage(db, Some(&plan.gap), model)?;
        first.report.absorb(&second.report);
        first.ending = second.ending;
        first.cancelled = second.cancelled;
        self.finish(first)
    }

    /// Sends `event` to every observer.
    fn emit(&self, event: &Event) {
        for observer in &self.observers {
            observer.on_event(event);
        }
    }

    /// Builds and sends an event, but only if somebody is listening.
    fn emit_with(&self, make: impl FnOnce() -> Event) {
        if !self.observers.is_empty() {
            self.emit(&make());
        }
    }

    /// The neighbourhood index and the activity masks, built on first use.
    ///
    /// Only call this after [`check_inputs`](Self::check_inputs) succeeded: it relies on the
    /// configuration and the algorithm subgraphs being consistent with the graph.
    fn setup(&self) -> &Setup {
        self.setup.get_or_init(|| {
            let depth = [
                self.config.parameter_distance,
                self.config.constraint_distance,
                self.config.traversal_distance,
                self.config.expiry_radius,
            ]
            .into_iter()
            .max()
            .unwrap_or(0);
            let activity = match &self.algorithm {
                Some(subgraphs) => subgraphs.masks(),
                None => AlgorithmSubgraphs::all_active(self.graph()).masks(),
            };
            Setup {
                neighbourhoods: Neighbourhoods::build(self.graph(), depth),
                activity,
            }
        })
    }

    /// Validates everything that does not depend on the model: the configuration, the sizes of
    /// the goal, the algorithm subgraphs and the optional region, the database's graph, and that
    /// every goal element can be active in some algorithm subgraph.
    fn check_inputs(
        &self,
        db_capacity: usize,
        db_graph: Result<(), FaultTuneError>,
        region: Option<&ElementSet>,
    ) -> Result<(), FaultTuneError> {
        self.config.validate()?;
        let graph = self.graph();
        let n = graph.len();
        let invalid = |message: String| Err(FaultTuneError::InvalidConfig(message));
        if self.goal.capacity() != n {
            return invalid(format!(
                "the goal is a set over {} elements, but the graph has {n}",
                self.goal.capacity()
            ));
        }
        if let Some(subgraphs) = &self.algorithm {
            if subgraphs.iter().any(|a| a.capacity() != n) {
                return invalid(format!(
                    "the algorithm subgraphs are sets over a graph of a different size than \
                     this one ({n} elements)"
                ));
            }
        }
        if let Some(region) = region {
            if region.capacity() != n {
                return invalid(format!(
                    "the region is a set over {} elements, but the graph has {n}",
                    region.capacity()
                ));
            }
        }
        db_graph?;
        if db_capacity != n {
            return invalid(format!(
                "the database holds {db_capacity} elements, but the graph has {n}"
            ));
        }
        let setup = self.setup();
        if let Some(inactive) = self.goal.iter().find(|&e| !setup.activity.is_active(e)) {
            return invalid(format!(
                "goal element {} is not active in any algorithm subgraph, so it could never be \
                 calibrated",
                graph.label(inactive)
            ));
        }
        Ok(())
    }

    /// The completion loop (ARCHITECTURE §4.1) for one `calibrate_*` call, ending with the
    /// `Finished` event.
    fn run<M: ErrorModel>(
        &self,
        db: &mut ParameterDatabase<M::Value>,
        region: Option<&ElementSet>,
        model: &M,
    ) -> Result<Report, FaultTuneError> {
        let stage = self.run_stage(db, region, model)?;
        self.finish(stage)
    }

    /// Emits `Finished` and turns a stage's outcome into the result of a call.
    fn finish(&self, stage: Stage) -> Result<Report, FaultTuneError> {
        let Stage {
            report,
            ending,
            cancelled,
        } = stage;
        self.emit_with(|| Event::Finished(report.clone()));
        match ending {
            Err(err) => Err(err),
            Ok(()) if cancelled => Err(FaultTuneError::Cancelled {
                report: Box::new(report),
            }),
            Ok(()) => Ok(report),
        }
    }

    /// One stage of a call: the completion loop without the closing `Finished` event. The outer
    /// `Err` is a validation failure (nothing ran); failures of the run itself are in
    /// [`Stage::ending`] together with the report of what was done.
    fn run_stage<M: ErrorModel>(
        &self,
        db: &mut ParameterDatabase<M::Value>,
        region: Option<&ElementSet>,
        model: &M,
    ) -> Result<Stage, FaultTuneError> {
        let started = Instant::now();
        self.check_inputs(db.capacity(), db.check_graph(self.graph()), region)?;
        let graph = self.graph();
        let n = graph.len();
        let setup = self.setup();
        let scope = Scope::new(
            graph,
            &setup.neighbourhoods,
            &setup.activity,
            self.config.distances(),
        );
        let cache = match (&self.shared_cache, self.config.cache) {
            (Some(shared), _) => Some(Arc::clone(shared)),
            (None, CachePolicy::AcrossSteps { capacity }) => {
                Some(Arc::new(CrossStepCache::new(capacity)))
            }
            (None, CachePolicy::StepOnly) => None,
        };
        let cache_before = cache.as_ref().map(|c| c.stats());
        let mut target = self.goal.clone();
        if let Some(region) = region {
            target.intersect_with(region);
        }
        let shared = Shared {
            snake: self,
            model,
            graph,
            scope,
            options: (0..n).map(|_| OnceLock::new()).collect(),
            cache: cache.clone(),
        };
        let span = trace::Span::calibrate(n, self.goal.len(), target.len(), self.config.seed);
        let _entered = span.enter();
        let history_start = db.history().len();
        // Pass numbers continue across the calls on one database, so that `(pass, subgoal,
        // centre)` identifies a step for the whole life of the database and a resumed run draws
        // the random numbers an uninterrupted run would have drawn.
        let pass_base = next_pass(db);
        let mut failed = ElementSet::new(n);
        let mut pool = Pool::default();
        let mut passes = 0u32;
        let mut ending: Result<(), FaultTuneError> = Ok(());
        let mut cancelled = false;

        loop {
            let mut uncalibrated = target.difference(db.calibrated());
            uncalibrated.difference_with(&failed);
            if uncalibrated.is_empty() {
                break;
            }
            let pass = pass_base + passes;
            let subgoals = segment(&scope, &uncalibrated);
            if subgoals.len() > 1 {
                if let Err(err) = pool.ensure(self.config.threads) {
                    ending = Err(err);
                    break;
                }
            }
            self.emit_with(|| Event::PassStarted {
                pass,
                subgoals: subgoals.len(),
                uncalibrated: uncalibrated.len(),
            });
            let ctx = PassCtx {
                pass,
                first_failed: AtomicUsize::new(usize::MAX),
                span: trace::Span::pass(&span, pass, subgoals.len(), uncalibrated.len()),
            };
            let calibrated_before = db.calibrated().len();
            let results = self.run_pass(&shared, &ctx, &*db, &subgoals, &mut pool);

            // Merge in subgoal order, keeping the steps of failed subgoals too. The failure that
            // is reported (or re-raised, for a panic) is the one of the lowest-numbered failing
            // subgoal, whatever the scheduling was; later failures are dropped.
            let mut skipped = 0usize;
            let mut failure = None;
            for result in results {
                db.merge(result.overlay);
                skipped += result.failed.len();
                for element in result.failed {
                    failed.insert(element);
                }
                match result.outcome {
                    Outcome::Done => {}
                    Outcome::Cancelled => cancelled = true,
                    Outcome::Failed(err) => {
                        failure.get_or_insert(Failure::Error(err));
                    }
                    Outcome::Panicked(payload) => {
                        failure.get_or_insert(Failure::Panic(payload));
                    }
                }
            }
            passes += 1;
            let calibrated = db.calibrated().len() - calibrated_before;
            self.emit_with(|| Event::PassFinished { pass, calibrated });
            match failure {
                Some(Failure::Panic(payload)) => resume_unwind(payload),
                Some(Failure::Error(err)) => {
                    ending = Err(err);
                    break;
                }
                None => {}
            }
            if cancelled {
                break;
            }
            if calibrated == 0 && skipped == 0 {
                ending = Err(FaultTuneError::NoProgress);
                break;
            }
        }

        let report = self.build_report(
            db,
            history_start,
            passes,
            &failed,
            cache.as_deref().zip(cache_before),
            started.elapsed(),
        );
        Ok(Stage {
            report,
            ending,
            cancelled,
        })
    }

    /// Runs the subgoals of one pass, in parallel when there is more than one and the
    /// `parallel` feature and the configuration allow it. The results are in subgoal order.
    fn run_pass<M: ErrorModel>(
        &self,
        shared: &Shared<'_, '_, M>,
        ctx: &PassCtx,
        snapshot: &ParameterDatabase<M::Value>,
        subgoals: &[Subgoal],
        pool: &mut Pool,
    ) -> Vec<TaskResult<M::Value>> {
        #[cfg(feature = "parallel")]
        if subgoals.len() > 1 && self.config.threads != Some(1) {
            use rayon::prelude::*;
            if let Some(workers) = pool.workers.as_ref() {
                return workers.install(|| {
                    subgoals
                        .par_iter()
                        .map(|subgoal| run_task(shared, ctx, snapshot, subgoal))
                        .collect()
                });
            }
        }
        let _ = &pool;
        subgoals
            .iter()
            .map(|subgoal| run_task(shared, ctx, snapshot, subgoal))
            .collect()
    }

    /// Summarises the steps committed since `history_start`.
    fn build_report<V>(
        &self,
        db: &ParameterDatabase<V>,
        history_start: usize,
        passes: u32,
        failed: &ElementSet,
        cache: Option<(&CrossStepCache, crate::cache::CacheStats)>,
        wall_time: Duration,
    ) -> Report {
        let mut report = Report {
            passes,
            wall_time,
            skipped: failed.to_vec(),
            ..Report::default()
        };
        let mut steps: Vec<&StepRecord> = Vec::new();
        for record in &db.history()[history_start..] {
            if let HistoryRecord::Step(step) = record {
                report.steps += 1;
                report.evaluations += step.evaluations;
                report.prepare_time += step.prepare_time;
                report.error_time += step.error_time;
                steps.push(step);
            }
        }
        steps.sort_by(|a, b| worst_first(a, b));
        report.worst_steps = steps.into_iter().take(MAX_WORST_STEPS).cloned().collect();
        if let Some((cache, before)) = cache {
            let after = cache.stats();
            report.cache_hits = after.hits.saturating_sub(before.hits);
            report.cache_misses = after.misses.saturating_sub(before.misses);
        }
        report
    }
}

/// The rayon pool of one `calibrate_*` call, started the first time a pass has several subgoals.
#[derive(Default)]
struct Pool {
    #[cfg(feature = "parallel")]
    workers: Option<rayon::ThreadPool>,
}

impl Pool {
    /// Starts the workers if they are wanted (more than one thread allowed) and not running yet.
    #[cfg(feature = "parallel")]
    fn ensure(&mut self, threads: Option<usize>) -> Result<(), FaultTuneError> {
        if threads == Some(1) || self.workers.is_some() {
            return Ok(());
        }
        // The same generous stack the Python bindings give their engine thread: user callbacks
        // (models, heuristics) run on these workers and should not behave differently from
        // `threads = 1`.
        let mut builder = rayon::ThreadPoolBuilder::new().stack_size(WORKER_STACK_SIZE);
        if let Some(threads) = threads {
            builder = builder.num_threads(threads);
        }
        let workers = builder.build().map_err(|err| {
            FaultTuneError::InvalidConfig(format!("could not start the thread pool: {err}"))
        })?;
        self.workers = Some(workers);
        Ok(())
    }

    /// Without the `parallel` feature every pass runs on the calling thread.
    #[cfg(not(feature = "parallel"))]
    fn ensure(&mut self, _threads: Option<usize>) -> Result<(), FaultTuneError> {
        Ok(())
    }
}

/// The outcome of one stage of a `calibrate_*` call.
struct Stage {
    report: Report,
    ending: Result<(), FaultTuneError>,
    cancelled: bool,
}

/// How a pass failed: an error to return, or a panic to re-raise.
enum Failure {
    Error(FaultTuneError),
    Panic(Box<dyn std::any::Any + Send>),
}

/// The number the next pass over `db` gets: one more than the highest pass in its history.
fn next_pass<V>(db: &ParameterDatabase<V>) -> u32 {
    db.steps().map(|s| s.pass).max().map_or(0, |p| p + 1)
}

impl Snake<'static> {
    /// Like [`Snake::new`] but shares ownership of the graph, so the Snake has no borrowed
    /// lifetime (used by the Python bindings).
    pub fn new_shared(graph: Arc<ProcessorGraph>, goal: ElementSet, config: SnakeConfig) -> Self {
        Self::with_graph(GraphRef::Shared(graph), goal, config)
    }
}
