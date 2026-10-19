//! The objective the engine hands to an optimizer: de-duplication, the optional cross-step
//! cache and the error model behind one `evaluate` call (ARCHITECTURE §4.4 and §7).

use std::time::{Duration, Instant};

use crate::cache::{CrossStepCache, StepDedup, StepKey};
use crate::engine::ModelErrorPolicy;
use crate::error::{FaultTuneError, ModelError};
use crate::model::{CandidateBatch, ErrorModel, Step};
use crate::optimize::{IndexBatch, Objective};

/// Runs `call`, repeating it after a failure as `policy` allows: `Retry { times }` makes up to
/// `times` further attempts, every other policy makes exactly one. Returns the last error if all
/// attempts failed.
pub(super) fn call_with_policy<T>(
    policy: ModelErrorPolicy,
    mut call: impl FnMut() -> Result<T, ModelError>,
) -> Result<T, ModelError> {
    let retries = match policy {
        ModelErrorPolicy::Retry { times } => times,
        ModelErrorPolicy::Abort | ModelErrorPolicy::SkipElement => 0,
    };
    let mut failures = 0;
    loop {
        match call() {
            Ok(value) => return Ok(value),
            Err(_) if failures < retries => failures += 1,
            Err(err) => return Err(err),
        }
    }
}

/// Buffers an objective reuses from one step (and one batch) to the next.
pub(super) struct EvalScratch<V> {
    /// Rows of the current batch that still have to be evaluated by the model.
    pending: Vec<usize>,
    /// Rows that repeat an earlier pending row of the same batch.
    repeats: Vec<usize>,
    /// Candidate values of the pending rows, row-major.
    values: Vec<V>,
    /// The model's answers for the pending rows.
    errors: Vec<f64>,
}

impl<V> Default for EvalScratch<V> {
    fn default() -> Self {
        Self {
            pending: Vec::new(),
            repeats: Vec::new(),
            values: Vec::new(),
            errors: Vec::new(),
        }
    }
}

/// Evaluates batches of option-index candidates for one step.
///
/// For each batch: rows already seen in this step are answered from the within-step
/// de-duplication map; the rest are looked up in the cross-step cache (when there is one);
/// what remains is evaluated with a single `error_batch` call. A NaN anywhere in the model's
/// answer is a [`FaultTuneError::NonFiniteError`], whatever the error policy.
pub(super) struct StepObjective<'a, M: ErrorModel> {
    pub(super) model: &'a M,
    pub(super) step: Step<'a, M::Value>,
    pub(super) data: &'a M::Data,
    /// The option table of each parameter, in `step.parameters` order.
    pub(super) tables: &'a [&'a [M::Value]],
    pub(super) policy: ModelErrorPolicy,
    pub(super) cache: Option<(&'a CrossStepCache, StepKey)>,
    pub(super) dedup: StepDedup,
    pub(super) scratch: &'a mut EvalScratch<M::Value>,
    /// Time spent in `error_batch` (including retries).
    pub(super) error_time: Duration,
    /// Number of distinct candidates asked for so far: after within-step de-duplication and
    /// before the cross-step cache.
    pub(super) distinct: u64,
}

impl<M: ErrorModel> StepObjective<'_, M> {
    fn internal(message: String) -> FaultTuneError {
        FaultTuneError::InvalidConfig(message)
    }
}

impl<M: ErrorModel> Objective for StepObjective<'_, M> {
    fn evaluate(&mut self, batch: IndexBatch<'_>, out: &mut [f64]) -> Result<(), FaultTuneError> {
        let width = self.step.parameters.len();
        let centre = self.step.centre;
        if batch.width() != width || out.len() != batch.len() {
            return Err(Self::internal(format!(
                "the optimizer passed a batch of {} candidates of width {} and {} output \
                 slots for a step with {width} parameters",
                batch.len(),
                batch.width(),
                out.len()
            )));
        }
        let scratch = &mut *self.scratch;
        scratch.pending.clear();
        scratch.repeats.clear();

        // (a) and (b): what is known already, and what is left to evaluate.
        for (i, slot) in out.iter_mut().enumerate() {
            let row = batch.row(i);
            for (j, &k) in row.iter().enumerate() {
                if k as usize >= self.tables[j].len() {
                    return Err(Self::internal(format!(
                        "the optimizer proposed option {k} for a parameter with {} options",
                        self.tables[j].len()
                    )));
                }
            }
            match self.dedup.lookup(row) {
                // A NaN is the placeholder of a row that is pending in this very batch.
                Some(known) if known.is_nan() => scratch.repeats.push(i),
                Some(known) => *slot = known,
                None => {
                    self.distinct += 1;
                    if let Some((cache, key)) = self.cache {
                        if let Some(cached) = cache.get(key, row) {
                            if cached.is_nan() {
                                return Err(FaultTuneError::NonFiniteError { element: centre });
                            }
                            self.dedup.insert(row, cached);
                            *slot = cached;
                            continue;
                        }
                    }
                    self.dedup.insert(row, f64::NAN);
                    scratch.pending.push(i);
                }
            }
        }

        // (c) and (d): one model call for the rest.
        if !scratch.pending.is_empty() {
            scratch.values.clear();
            scratch.values.reserve(scratch.pending.len() * width);
            for &i in &scratch.pending {
                for (j, &k) in batch.row(i).iter().enumerate() {
                    scratch.values.push(self.tables[j][k as usize].clone());
                }
            }
            scratch.errors.clear();
            scratch.errors.resize(scratch.pending.len(), 0.0);
            let candidates = CandidateBatch::new(&scratch.values, width);
            let started = Instant::now();
            let answered = call_with_policy(self.policy, || {
                self.model
                    .error_batch(&self.step, self.data, &candidates, &mut scratch.errors)
            });
            self.error_time += started.elapsed();
            answered.map_err(|source| FaultTuneError::Model {
                element: centre,
                source,
            })?;
            if scratch.errors.iter().any(|e| e.is_nan()) {
                return Err(FaultTuneError::NonFiniteError { element: centre });
            }
            // (e): remember the answers.
            for (&i, &error) in scratch.pending.iter().zip(&scratch.errors) {
                let row = batch.row(i);
                self.dedup.insert(row, error);
                if let Some((cache, key)) = self.cache {
                    cache.insert(key, row, error);
                }
                out[i] = error;
            }
        }
        for &i in &scratch.repeats {
            // The first occurrence was evaluated above, so the answer is there.
            out[i] = self.dedup.lookup(batch.row(i)).unwrap_or(f64::NAN);
        }
        Ok(())
    }
}
