//! The error-model extension point (ARCHITECTURE §5.1).
//!
//! FaultTune provides the Snake machinery; the user provides the physics through
//! an [`ErrorModel`]: the candidate values of each element, how calibration data is
//! collected for a step, and the estimated error of candidate assignments.

use crate::error::ModelError;
use crate::graph::{ElementId, ProcessorGraph};

/// Everything the model needs to know about one calibration step.
///
/// A step centres on one element `g`. Its *parameters* are the uncalibrated elements
/// within `d_P` of `g`; they are optimized together. Its *constraints* are the
/// already-calibrated, simultaneously-active elements within `d_R` of the parameters,
/// together with their fixed values. The model never has to look values up elsewhere.
pub struct Step<'a, V> {
    /// Ordinal of the step within its subgoal and pass, starting at 0. It is **not** the
    /// final index in the [`ParameterDatabase`](crate::ParameterDatabase) history, which is
    /// assigned when subgoals are merged (so that results do not depend on scheduling).
    pub index: u64,
    /// The element the step is centred on.
    pub centre: ElementId,
    /// The parameters being optimized: the columns of every candidate, in this order.
    pub parameters: &'a [ElementId],
    /// The active constraints: calibrated elements and their fixed values, ascending by ID.
    pub constraints: &'a [(ElementId, V)],
    /// The processor graph.
    pub graph: &'a ProcessorGraph,
}

impl<V> Clone for Step<'_, V> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<V> Copy for Step<'_, V> {}

impl<V> Step<'_, V> {
    /// The fixed value of constraint `id`, if `id` is one of this step's constraints.
    pub fn constraint(&self, id: ElementId) -> Option<&V> {
        self.constraints
            .binary_search_by_key(&id, |(e, _)| *e)
            .ok()
            .map(|i| &self.constraints[i].1)
    }
}

/// A batch of candidate assignments: `len()` rows of `width()` values each,
/// stored row-major. Column `j` belongs to `step.parameters[j]`.
pub struct CandidateBatch<'a, V> {
    values: &'a [V],
    width: usize,
}

impl<'a, V> CandidateBatch<'a, V> {
    /// Wraps a row-major slice.
    ///
    /// # Panics
    /// Panics if `width == 0` while `values` is non-empty, or `values.len()` is not a multiple of `width`.
    pub fn new(values: &'a [V], width: usize) -> Self {
        if width == 0 {
            assert!(values.is_empty(), "a zero-width batch cannot hold values");
        } else {
            assert_eq!(
                values.len() % width,
                0,
                "batch length must be a multiple of its width"
            );
        }
        Self { values, width }
    }

    /// Number of candidates (rows).
    pub fn len(&self) -> usize {
        self.values.len().checked_div(self.width).unwrap_or(0)
    }

    /// Whether the batch has no candidates.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of values per candidate (the number of parameters).
    pub fn width(&self) -> usize {
        self.width
    }

    /// Candidate `i`.
    pub fn row(&self, i: usize) -> &'a [V] {
        &self.values[i * self.width..(i + 1) * self.width]
    }

    /// Iterates over the candidates.
    pub fn rows(&self) -> impl Iterator<Item = &'a [V]> + '_ {
        (0..self.len()).map(move |i| self.row(i))
    }

    /// The underlying row-major slice (`len() * width()` values).
    pub fn as_flat(&self) -> &'a [V] {
        self.values
    }
}

impl<V> Clone for CandidateBatch<'_, V> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<V> Copy for CandidateBatch<'_, V> {}

/// The user-supplied physics: candidate values and the error of candidate assignments.
///
/// All methods take `&self`, so one model can serve parallel subgoals. Per-step state
/// travels in [`ErrorModel::Data`] instead of being stored in the model. A model that
/// drives hardware which cannot be shared can serialise access with its own lock.
pub trait ErrorModel: Sync {
    /// The type of a parameter value (for example `f64`, a frequency in GHz).
    type Value: Clone + Send + Sync;
    /// Per-step calibration data returned by [`prepare`](Self::prepare). Use `()` if there is none.
    type Data: Send;

    /// The `k` candidate values for one element. Called once per element per run.
    fn options(&self, element: ElementId) -> Vec<Self::Value>;

    /// Collects the calibration data a step needs, for example by measuring hardware.
    fn prepare(&self, step: &Step<'_, Self::Value>) -> Result<Self::Data, ModelError>;

    /// Estimated error of one candidate assignment of the step's parameters (lower is
    /// better). Return `f64::INFINITY` for forbidden candidates. Returning NaN is an error.
    fn error(
        &self,
        step: &Step<'_, Self::Value>,
        data: &Self::Data,
        candidate: &[Self::Value],
    ) -> f64;

    /// Estimated errors of many candidates at once, written to `out` (`out.len() ==
    /// candidates.len()`). Override for speed; the default calls [`error`](Self::error) per row.
    fn error_batch(
        &self,
        step: &Step<'_, Self::Value>,
        data: &Self::Data,
        candidates: &CandidateBatch<'_, Self::Value>,
        out: &mut [f64],
    ) -> Result<(), ModelError> {
        debug_assert_eq!(out.len(), candidates.len());
        for (slot, row) in out.iter_mut().zip(candidates.rows()) {
            *slot = self.error(step, data, row);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_rows() {
        let flat = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let batch = CandidateBatch::new(&flat, 2);
        assert_eq!(batch.len(), 3);
        assert_eq!(batch.row(1), &[3.0, 4.0]);
        assert_eq!(batch.rows().count(), 3);
        assert!(CandidateBatch::<f64>::new(&[], 0).is_empty());
    }

    #[test]
    #[should_panic]
    fn ragged_batch_panics() {
        CandidateBatch::new(&[1.0, 2.0, 3.0], 2);
    }

    struct Quadratic;
    impl ErrorModel for Quadratic {
        type Value = f64;
        type Data = ();
        fn options(&self, _: ElementId) -> Vec<f64> {
            vec![0.0, 1.0]
        }
        fn prepare(&self, _: &Step<'_, f64>) -> Result<(), ModelError> {
            Ok(())
        }
        fn error(&self, _: &Step<'_, f64>, _: &(), c: &[f64]) -> f64 {
            c.iter().map(|x| x * x).sum()
        }
    }

    #[test]
    fn default_error_batch_loops_over_rows() {
        let graph = ProcessorGraph::square_lattice(1, 2);
        let params = [ElementId::new(0), ElementId::new(1)];
        let constraints: [(ElementId, f64); 0] = [];
        let step = Step {
            index: 0,
            centre: params[0],
            parameters: &params,
            constraints: &constraints,
            graph: &graph,
        };
        let flat = [1.0, 2.0, 3.0, 4.0];
        let mut out = [0.0; 2];
        Quadratic
            .error_batch(&step, &(), &CandidateBatch::new(&flat, 2), &mut out)
            .unwrap();
        assert_eq!(out, [5.0, 25.0]);
    }
}
