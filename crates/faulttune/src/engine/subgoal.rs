//! Running one subgoal of a pass: the traversal and the calibration step
//! (ARCHITECTURE §4.3 and §4.4).

use std::any::Any;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;

use super::objective::{call_with_policy, EvalScratch, StepObjective};
use super::trace;
use super::{ModelErrorPolicy, Snake};
use crate::cache::{CrossStepCache, StepDedup, StepKey};
use crate::database::{Overlay, ParameterDatabase, StepRecord};
use crate::error::{FaultTuneError, ModelError};
use crate::events::Event;
use crate::graph::{ElementId, ProcessorGraph};
use crate::model::{ErrorModel, Step};
use crate::optimize::SearchSpace;
use crate::scope::Scope;
use crate::segment::Subgoal;
use crate::sets::ElementSet;
use crate::traverse::{Frontier, TraversalContext};

/// The finalizer of SplitMix64.
fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// The seed of the RNG of a subgoal: a fixed splitmix-style mix of the run seed, the pass and
/// the smallest element of the subgoal.
pub(super) fn subgoal_seed(seed: u64, pass: u32, first: ElementId) -> u64 {
    let h = mix64(seed);
    let h = mix64(
        h ^ u64::from(pass)
            .wrapping_add(1)
            .wrapping_mul(0xD1B5_4A32_D192_ED03),
    );
    mix64(
        h ^ u64::from(first.raw())
            .wrapping_add(1)
            .wrapping_mul(0xA24B_AED4_963E_E407),
    )
}

/// What is shared by every subgoal of one `calibrate_*` call.
pub(super) struct Shared<'a, 'g, M: ErrorModel> {
    pub(super) snake: &'a Snake<'g>,
    pub(super) model: &'a M,
    pub(super) graph: &'a ProcessorGraph,
    pub(super) scope: Scope<'a>,
    /// The option table: filled on first use, at most once per element per call.
    pub(super) options: Vec<OnceLock<Vec<M::Value>>>,
    pub(super) cache: Option<Arc<CrossStepCache>>,
}

impl<M: ErrorModel> Shared<'_, '_, M> {
    /// The options of an element, from the model on first use.
    fn options_of(&self, element: ElementId) -> Result<&[M::Value], FaultTuneError> {
        let slot = self.options.get(element.index()).ok_or_else(|| {
            FaultTuneError::InvalidConfig(format!("element {} is not in the graph", element.raw()))
        })?;
        let table = slot.get_or_init(|| self.model.options(element));
        if table.is_empty() {
            return Err(FaultTuneError::InvalidConfig(format!(
                "the model gave no options for element {}",
                self.graph.label(element)
            )));
        }
        Ok(table)
    }
}

/// What is shared by the subgoals of one pass.
pub(super) struct PassCtx {
    pub(super) pass: u32,
    /// The smallest index of a subgoal that failed so far (`usize::MAX` if none did). Later
    /// subgoals stop at their next step; earlier ones carry on, because their failure would
    /// take priority.
    pub(super) first_failed: AtomicUsize,
    pub(super) span: trace::Span,
}

/// How a subgoal ended.
pub(super) enum Outcome {
    /// It ran to its end (or stopped because an earlier subgoal failed).
    Done,
    /// The cancellation token stopped it.
    Cancelled,
    /// An error.
    Failed(FaultTuneError),
    /// A panic, to be re-raised after the pass was merged.
    Panicked(Box<dyn Any + Send>),
}

/// Everything a subgoal produced, kept even when it failed.
pub(super) struct TaskResult<V> {
    pub(super) overlay: Overlay<V>,
    pub(super) failed: Vec<ElementId>,
    pub(super) outcome: Outcome,
}

/// Runs a subgoal against `snapshot`. The overlay lives outside the unwind guard, so the steps
/// committed before an error or a panic are never lost.
pub(super) fn run_task<M: ErrorModel>(
    run: &Shared<'_, '_, M>,
    pass: &PassCtx,
    snapshot: &ParameterDatabase<M::Value>,
    subgoal: &Subgoal,
) -> TaskResult<M::Value> {
    let mut overlay = Overlay::new(snapshot);
    let mut failed = Vec::new();
    let result = catch_unwind(AssertUnwindSafe(|| {
        Worker::new(run, pass, snapshot, subgoal, &mut overlay, &mut failed).run()
    }));
    let outcome = match result {
        Ok(Ok(Ending::Done)) => Outcome::Done,
        Ok(Ok(Ending::Cancelled)) => Outcome::Cancelled,
        Ok(Err(err)) => {
            pass.first_failed.fetch_min(subgoal.index, Ordering::SeqCst);
            Outcome::Failed(err)
        }
        Err(payload) => {
            pass.first_failed.fetch_min(subgoal.index, Ordering::SeqCst);
            Outcome::Panicked(payload)
        }
    };
    TaskResult {
        overlay,
        failed,
        outcome,
    }
}

enum Ending {
    Done,
    Cancelled,
}

/// Buffers reused from step to step.
struct Buffers<V> {
    parameters: Vec<ElementId>,
    constraint_ids: Vec<ElementId>,
    constraints: Vec<(ElementId, V)>,
    constraint_options: Vec<(ElementId, u32)>,
    traversal: Vec<ElementId>,
    sizes: Vec<u32>,
    eval: EvalScratch<V>,
}

struct Worker<'r, 'a, 'g, M: ErrorModel> {
    run: &'r Shared<'a, 'g, M>,
    pass: &'r PassCtx,
    snapshot: &'r ParameterDatabase<M::Value>,
    subgoal: &'r Subgoal,
    overlay: &'r mut Overlay<M::Value>,
    failed: &'r mut Vec<ElementId>,
    /// Elements of the subgoal that can still be calibrated here.
    available: ElementSet,
    rng: StdRng,
    bufs: Buffers<M::Value>,
    frontier: Frontier,
    span: trace::Span,
}

impl<'r, 'a, 'g, M: ErrorModel> Worker<'r, 'a, 'g, M> {
    fn new(
        run: &'r Shared<'a, 'g, M>,
        pass: &'r PassCtx,
        snapshot: &'r ParameterDatabase<M::Value>,
        subgoal: &'r Subgoal,
        overlay: &'r mut Overlay<M::Value>,
        failed: &'r mut Vec<ElementId>,
    ) -> Self {
        let first = subgoal
            .elements
            .first()
            .copied()
            .unwrap_or(ElementId::new(0));
        Self {
            run,
            pass,
            snapshot,
            subgoal,
            overlay,
            failed,
            available: subgoal.set.clone(),
            rng: StdRng::seed_from_u64(subgoal_seed(run.snake.config.seed, pass.pass, first)),
            bufs: Buffers {
                parameters: Vec::new(),
                constraint_ids: Vec::new(),
                constraints: Vec::new(),
                constraint_options: Vec::new(),
                traversal: Vec::new(),
                sizes: Vec::new(),
                eval: EvalScratch::default(),
            },
            frontier: Frontier::new(run.snake.heuristic.frontier()),
            span: trace::Span::subgoal(
                &pass.span,
                subgoal.index,
                subgoal.elements.len(),
                subgoal.threads.len(),
            ),
        }
    }

    /// Whether to stop before the next seed or step.
    fn stop(&self) -> Option<Ending> {
        if self.run.snake.cancellation.is_cancelled() {
            Some(Ending::Cancelled)
        } else if self.pass.first_failed.load(Ordering::SeqCst) < self.subgoal.index {
            Some(Ending::Done)
        } else {
            None
        }
    }

    /// The traversal of every thread of the subgoal, nodes first.
    fn run(mut self) -> Result<Ending, FaultTuneError> {
        let span = self.span.clone();
        let _entered = span.enter();
        let subgoal = self.subgoal;
        let snake = self.run.snake;
        for thread in &subgoal.threads {
            if let Some(end) = self.stop() {
                return Ok(end);
            }
            let seed = snake.seed_policy.seed(thread, &mut self.rng);
            if !self.available.contains(seed) {
                continue;
            }
            self.frontier.clear();
            self.frontier.push(seed);
            while let Some(g) = self.frontier.pop() {
                if !self.available.contains(g) {
                    continue;
                }
                if let Some(end) = self.stop() {
                    return Ok(end);
                }
                self.step(g)?;
                self.run
                    .scope
                    .traversal_options(g, &self.available, &mut self.bufs.traversal);
                let ctx = TraversalContext {
                    graph: self.run.graph,
                    neighbourhoods: self.run.scope.neighbourhoods,
                    calibrated: self.overlay.calibrated(),
                    current: g,
                    history: self.overlay.steps(),
                };
                self.frontier.extend_ordered(
                    &*snake.heuristic,
                    &ctx,
                    &mut self.bufs.traversal,
                    &mut self.rng,
                );
            }
        }
        Ok(Ending::Done)
    }

    /// One step centred on `g`; a model failure follows the error policy.
    fn step(&mut self, g: ElementId) -> Result<(), FaultTuneError> {
        match self.try_step(g) {
            Ok(()) => Ok(()),
            Err(FaultTuneError::Model { source, .. })
                if self.run.snake.config.on_model_error == ModelErrorPolicy::SkipElement =>
            {
                self.skip(g, &source);
                Ok(())
            }
            Err(err) => Err(err),
        }
    }

    /// Gives up on the centre of a step whose model failed.
    fn skip(&mut self, g: ElementId, source: &ModelError) {
        self.available.remove(g);
        self.failed.push(g);
        self.run.snake.emit_with(|| Event::StepFailed {
            centre: g,
            error: source.to_string(),
        });
    }

    fn try_step(&mut self, g: ElementId) -> Result<(), FaultTuneError> {
        let started = Instant::now();
        let Worker {
            run,
            pass,
            snapshot,
            subgoal,
            overlay,
            available,
            rng,
            bufs,
            span,
            ..
        } = self;
        let run: &Shared<'_, '_, M> = run;
        let snake = run.snake;
        let policy = snake.config.on_model_error;

        // Parameters and constraints.
        run.scope.parameters(g, available, &mut bufs.parameters);
        if bufs.parameters.is_empty() {
            return Err(FaultTuneError::InvalidConfig(format!(
                "internal error: the step centred on {} has no parameters",
                run.graph.label(g)
            )));
        }
        run.scope.constraints(
            &bufs.parameters,
            overlay.calibrated(),
            &mut bufs.constraint_ids,
        );
        bufs.constraints.clear();
        bufs.constraint_options.clear();
        for &c in &bufs.constraint_ids {
            let (value, option) = overlay.value(snapshot, c).ok_or_else(|| {
                FaultTuneError::InvalidConfig(format!(
                    "internal error: constraint {} is calibrated but has no value",
                    run.graph.label(c)
                ))
            })?;
            bufs.constraints.push((c, value.clone()));
            bufs.constraint_options.push((c, option));
        }
        let width = bufs.parameters.len();
        let index = overlay.steps().len() as u64;
        let step_span = trace::Span::step(
            span,
            index,
            run.graph.label(g),
            width,
            bufs.constraints.len(),
        );
        let _entered = step_span.enter();

        // Option tables and sizes.
        let mut tables: Vec<&[M::Value]> = Vec::with_capacity(width);
        bufs.sizes.clear();
        for &p in &bufs.parameters {
            let table = run.options_of(p)?;
            let size = u32::try_from(table.len()).map_err(|_| {
                FaultTuneError::InvalidConfig(format!(
                    "element {} has more than {} options",
                    run.graph.label(p),
                    u32::MAX
                ))
            })?;
            bufs.sizes.push(size);
            tables.push(table);
        }

        let step = Step {
            index,
            centre: g,
            parameters: &bufs.parameters,
            constraints: &bufs.constraints,
            graph: run.graph,
        };

        // Prepare.
        let prepare_started = Instant::now();
        let prepared = call_with_policy(policy, || run.model.prepare(&step));
        let prepare_time = prepare_started.elapsed();
        let data = prepared.map_err(|source| FaultTuneError::Model { element: g, source })?;

        // Optimize.
        let mut space = SearchSpace {
            sizes: std::mem::take(&mut bufs.sizes),
            budget: snake.config.evaluation_budget,
        };
        let mut objective = StepObjective {
            model: run.model,
            step,
            data: &data,
            tables: &tables,
            policy,
            cache: run.cache.as_deref().map(|cache| {
                (
                    cache,
                    StepKey::new(&bufs.parameters, &bufs.constraint_options),
                )
            }),
            dedup: StepDedup::new(width),
            scratch: &mut bufs.eval,
            error_time: std::time::Duration::ZERO,
            distinct: 0,
        };
        let solved = snake.optimizer.optimize(&space, &mut objective, rng);
        let (error_time, distinct) = (objective.error_time, objective.distinct);
        drop(objective);
        let sizes = std::mem::take(&mut space.sizes);
        bufs.sizes = sizes;
        let solution = solved?;

        // Check the solution, then commit it.
        if solution.indices.len() != width
            || solution
                .indices
                .iter()
                .zip(&tables)
                .any(|(&k, table)| k as usize >= table.len())
        {
            return Err(FaultTuneError::InvalidConfig(
                "the optimizer returned a solution that does not fit the search space".into(),
            ));
        }
        if solution.error.is_nan() {
            return Err(FaultTuneError::NonFiniteError { element: g });
        }
        let values: Vec<M::Value> = solution
            .indices
            .iter()
            .zip(&tables)
            .map(|(&k, table)| table[k as usize].clone())
            .collect();
        let record = StepRecord {
            index,
            pass: pass.pass,
            subgoal: subgoal.index as u32,
            centre: g,
            parameters: bufs.parameters.clone(),
            option_indices: solution.indices,
            constraints: bufs.constraint_ids.clone(),
            error: solution.error,
            evaluations: distinct,
            prepare_time,
            error_time,
            duration: started.elapsed(),
        };
        step_span.record_outcome(distinct, solution.error);
        overlay.commit(record, values);
        for &p in &bufs.parameters {
            available.remove(p);
        }
        if let Some(committed) = overlay.steps().last() {
            snake.emit_with(|| Event::StepCommitted(committed.clone()));
        }
        Ok(())
    }
}
