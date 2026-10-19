//! Step optimizers (ARCHITECTURE §5.2).
//!
//! Optimizers never see values, only **option indices**: for each parameter the model's
//! `options(element)` gives `k` values and a candidate is a vector of indices, one per
//! parameter. This keeps optimizers generic over the value type and makes candidates cheap
//! to hash and compare.

mod coordinate;
mod exhaustive;
mod random;

use rand::rngs::StdRng;

pub use coordinate::CoordinateDescent;
pub use exhaustive::Exhaustive;
pub use random::RandomSearch;

use crate::error::FaultTuneError;

/// The default evaluation budget of a step (`10^6` candidates).
pub const DEFAULT_BUDGET: u64 = 1_000_000;

/// The search space of one step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SearchSpace {
    /// Number of options `k_i` of each parameter, in `Step::parameters` order. Never 0.
    pub sizes: Vec<u32>,
    /// Evaluation budget: the most candidates a budget-aware optimizer should evaluate.
    pub budget: u64,
}

impl SearchSpace {
    /// A search space with the default budget.
    pub fn new(sizes: Vec<u32>) -> Self {
        Self {
            sizes,
            budget: DEFAULT_BUDGET,
        }
    }

    /// Number of parameters.
    pub fn width(&self) -> usize {
        self.sizes.len()
    }

    /// Number of combinations `∏ k_i`, saturating at `u128::MAX`.
    pub fn size(&self) -> u128 {
        self.sizes
            .iter()
            .fold(1u128, |acc, &k| acc.saturating_mul(u128::from(k)))
    }
}

/// A batch of candidates as option indices: `len()` rows of `width()` indices, row-major.
#[derive(Copy, Clone, Debug)]
pub struct IndexBatch<'a> {
    data: &'a [u32],
    width: usize,
}

impl<'a> IndexBatch<'a> {
    /// Wraps a row-major slice. Panics if `data.len()` is not a multiple of a non-zero `width`.
    pub fn new(data: &'a [u32], width: usize) -> Self {
        if width == 0 {
            assert!(data.is_empty());
        } else {
            assert_eq!(
                data.len() % width,
                0,
                "batch length must be a multiple of its width"
            );
        }
        Self { data, width }
    }

    /// Number of candidates.
    pub fn len(&self) -> usize {
        self.data.len().checked_div(self.width).unwrap_or(0)
    }

    /// Whether the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Indices per candidate.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Candidate `i`.
    pub fn row(&self, i: usize) -> &'a [u32] {
        &self.data[i * self.width..(i + 1) * self.width]
    }

    /// The row-major slice.
    pub fn as_flat(&self) -> &'a [u32] {
        self.data
    }
}

/// Evaluates batches of candidates. The engine implements it on top of the error model,
/// with within-step de-duplication and the optional cross-step cache.
pub trait Objective {
    /// Writes the error of each candidate to `out` (`out.len() == batch.len()`). Lower is
    /// better; `+∞` marks a forbidden candidate.
    fn evaluate(&mut self, batch: IndexBatch<'_>, out: &mut [f64]) -> Result<(), FaultTuneError>;
}

/// The result of optimizing one step.
#[derive(Clone, Debug, PartialEq)]
pub struct Solution {
    /// Best option index per parameter.
    pub indices: Vec<u32>,
    /// Its error (possibly `+∞` if no feasible candidate was found).
    pub error: f64,
    /// Number of candidates the optimizer asked the objective to evaluate.
    pub evaluations: u64,
}

/// Solves one step: finds a good assignment of option indices.
pub trait Optimizer: Send + Sync {
    /// Searches `space`, evaluating candidates through `objective`.
    fn optimize(
        &self,
        space: &SearchSpace,
        objective: &mut dyn Objective,
        rng: &mut StdRng,
    ) -> Result<Solution, FaultTuneError>;
}
