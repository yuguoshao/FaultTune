//! Exhaustive search.

use rand::rngs::StdRng;

use super::{IndexBatch, Objective, Optimizer, SearchSpace, Solution};
use crate::error::FaultTuneError;

/// Tries every combination, in batches. Refuses with
/// [`FaultTuneError::SearchSpaceTooLarge`] if `∏ k_i` exceeds the budget
/// (`min(space.budget, self.max_size)`; default `10^6`).
///
/// Combinations are enumerated in lexicographic order (the last parameter varies fastest)
/// and ties are resolved in favour of the first combination in that order. The random
/// number generator is not used. A `batch_size` of 0 is treated as 1.
///
/// A space without parameters has exactly one (empty) candidate. The objective is called once
/// with a zero-width batch, which cannot return an error: the solution has an empty index
/// vector, `evaluations == 0` and a NaN error, and callers must not use that error.
///
/// `batch_size` is capped at 2^20 rows per call.
///
/// Use when `d_P = 0` or `k` is small.
#[derive(Clone, Debug)]
pub struct Exhaustive {
    /// Candidates per objective call.
    pub batch_size: usize,
    /// Upper bound on the search space size, applied together with `SearchSpace::budget`.
    pub max_size: u64,
}

impl Default for Exhaustive {
    fn default() -> Self {
        Self {
            batch_size: 4096,
            max_size: super::DEFAULT_BUDGET,
        }
    }
}

impl Optimizer for Exhaustive {
    fn optimize(
        &self,
        space: &SearchSpace,
        objective: &mut dyn Objective,
        _rng: &mut StdRng,
    ) -> Result<Solution, FaultTuneError> {
        check_space(space)?;
        let width = space.width();
        if width == 0 {
            return solve_empty(objective);
        }

        let limit = space.budget.min(self.max_size);
        let size = space.size();
        if size > u128::from(limit) {
            return Err(FaultTuneError::SearchSpaceTooLarge {
                size,
                budget: limit,
            });
        }
        // `size <= limit <= u64::MAX` here.
        let total = size as u64;
        let batch_size = clamp_batch_size(self.batch_size, total);

        let mut counter = vec![0u32; width];
        let mut buf: Vec<u32> = Vec::with_capacity(batch_size * width);
        let mut out = vec![0.0f64; batch_size];
        let mut best: Option<(Vec<u32>, f64)> = None;
        let mut remaining = total;
        while remaining > 0 {
            let n = remaining.min(batch_size as u64) as usize;
            buf.clear();
            for _ in 0..n {
                buf.extend_from_slice(&counter);
                increment(&mut counter, &space.sizes);
            }
            objective.evaluate(IndexBatch::new(&buf, width), &mut out[..n])?;
            if let Some((row, error)) = best_of(&out[..n], None) {
                let improves = match &best {
                    None => true,
                    Some((_, current)) => is_better(error, *current),
                };
                if improves {
                    let start = row * width;
                    best = Some((buf[start..start + width].to_vec(), error));
                }
            }
            remaining -= n as u64;
        }
        let (indices, error) = best.expect("the space has at least one candidate");
        Ok(Solution {
            indices,
            error,
            evaluations: total,
        })
    }
}

/// Largest number of rows in one objective call, whatever `batch_size` is configured.
///
/// It bounds the scratch buffers (about `4 * width + 8` bytes per row), so an absurd
/// `batch_size` cannot ask the allocator for gigabytes.
pub(super) const MAX_BATCH: usize = 1 << 20;

/// The effective batch size: at least 1, at most [`MAX_BATCH`] and at most `total`.
pub(super) fn clamp_batch_size(batch_size: usize, total: u64) -> usize {
    (batch_size.clamp(1, MAX_BATCH) as u64).min(total) as usize
}

/// Advances a mixed-radix counter (last digit fastest). Wraps to all zeros after the last
/// combination.
fn increment(counter: &mut [u32], sizes: &[u32]) {
    for (digit, &size) in counter.iter_mut().zip(sizes).rev() {
        *digit += 1;
        if *digit < size {
            return;
        }
        *digit = 0;
    }
}

/// Rejects a space in which some parameter has no options.
pub(super) fn check_space(space: &SearchSpace) -> Result<(), FaultTuneError> {
    match space.sizes.iter().position(|&k| k == 0) {
        Some(j) => Err(FaultTuneError::InvalidConfig(format!(
            "parameter {j} of the search space has no options"
        ))),
        None => Ok(()),
    }
}

/// Whether error `new` is strictly better than `old`. NaN is worse than every number and
/// is never better than anything.
pub(super) fn is_better(new: f64, old: f64) -> bool {
    new < old || (old.is_nan() && !new.is_nan())
}

/// The first best entry of `errors` (see [`is_better`]) as `(position, error)`, ignoring
/// position `skip`. `None` if there is nothing to look at.
pub(super) fn best_of(errors: &[f64], skip: Option<usize>) -> Option<(usize, f64)> {
    let mut best: Option<(usize, f64)> = None;
    for (i, &e) in errors.iter().enumerate() {
        if Some(i) == skip {
            continue;
        }
        match best {
            Some((_, b)) if !is_better(e, b) => {}
            _ => best = Some((i, e)),
        }
    }
    best
}

/// Solves a space without parameters: the objective is called once with an empty batch.
///
/// An [`IndexBatch`] of width 0 has no rows, so the objective has nowhere to put an error
/// and none can be read back. The solution therefore reports `evaluations == 0` (no row was
/// handed over) and an error of NaN, which stands for "unknown" rather than a made-up number.
/// The engine must not record `Solution::error` for a step without parameters.
pub(super) fn solve_empty(objective: &mut dyn Objective) -> Result<Solution, FaultTuneError> {
    objective.evaluate(IndexBatch::new(&[], 0), &mut [])?;
    Ok(Solution {
        indices: Vec::new(),
        error: f64::NAN,
        evaluations: 0,
    })
}

#[cfg(test)]
pub(super) mod test_support {
    use super::*;

    /// An objective backed by a closure that records how it was called.
    pub struct FnObjective<F> {
        sizes: Vec<u32>,
        f: F,
        pub calls: usize,
        pub rows: u64,
        pub max_batch: usize,
    }

    impl<F: FnMut(&[u32]) -> f64> FnObjective<F> {
        pub fn new(sizes: &[u32], f: F) -> Self {
            Self {
                sizes: sizes.to_vec(),
                f,
                calls: 0,
                rows: 0,
                max_batch: 0,
            }
        }
    }

    impl<F: FnMut(&[u32]) -> f64> Objective for FnObjective<F> {
        fn evaluate(
            &mut self,
            batch: IndexBatch<'_>,
            out: &mut [f64],
        ) -> Result<(), FaultTuneError> {
            assert_eq!(out.len(), batch.len(), "out must match the batch");
            assert_eq!(batch.width(), self.sizes.len());
            self.calls += 1;
            self.rows += batch.len() as u64;
            self.max_batch = self.max_batch.max(batch.len());
            for (i, o) in out.iter_mut().enumerate() {
                let row = batch.row(i);
                for (&x, &k) in row.iter().zip(&self.sizes) {
                    assert!(x < k, "index {x} out of range for {k} options");
                }
                *o = (self.f)(row);
            }
            Ok(())
        }
    }

    /// An objective that always fails.
    pub struct Failing;

    impl Objective for Failing {
        fn evaluate(&mut self, _: IndexBatch<'_>, _: &mut [f64]) -> Result<(), FaultTuneError> {
            Err(FaultTuneError::InvalidConfig("boom".into()))
        }
    }

    /// A deterministic pseudo-random table cost: unique values with overwhelming probability.
    pub fn table_cost(seed: u64, j: usize, option: u32) -> f64 {
        let mut x = seed
            ^ (j as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ u64::from(option).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^= x >> 31;
        (x >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;
    use proptest::prelude::*;
    use rand::SeedableRng;

    fn rng() -> StdRng {
        StdRng::seed_from_u64(1)
    }

    #[test]
    fn increment_is_lexicographic() {
        let sizes = [2, 3];
        let mut c = vec![0, 0];
        let mut seen = vec![c.clone()];
        for _ in 0..5 {
            increment(&mut c, &sizes);
            seen.push(c.clone());
        }
        assert_eq!(
            seen,
            vec![
                vec![0, 0],
                vec![0, 1],
                vec![0, 2],
                vec![1, 0],
                vec![1, 1],
                vec![1, 2]
            ]
        );
        increment(&mut c, &sizes);
        assert_eq!(c, vec![0, 0], "wraps around");
    }

    #[test]
    fn is_better_orders_nan_last() {
        assert!(is_better(1.0, 2.0));
        assert!(!is_better(2.0, 1.0));
        assert!(!is_better(1.0, 1.0));
        assert!(is_better(1.0, f64::INFINITY));
        assert!(is_better(1.0, f64::NAN));
        assert!(is_better(f64::INFINITY, f64::NAN));
        assert!(!is_better(f64::NAN, 1.0));
        assert!(!is_better(f64::NAN, f64::NAN));
    }

    #[test]
    fn best_of_takes_first_and_skips() {
        assert_eq!(best_of(&[], None), None);
        assert_eq!(best_of(&[3.0, 1.0, 1.0, 2.0], None), Some((1, 1.0)));
        assert_eq!(best_of(&[3.0, 1.0, 2.0], Some(1)), Some((2, 2.0)));
        assert_eq!(best_of(&[1.0], Some(0)), None);
        let (i, e) = best_of(&[f64::NAN, f64::NAN, 5.0, f64::NAN], None).unwrap();
        assert_eq!((i, e), (2, 5.0));
        let (i, e) = best_of(&[f64::NAN, f64::INFINITY], None).unwrap();
        assert_eq!((i, e), (1, f64::INFINITY));
        let (i, e) = best_of(&[f64::NAN, f64::NAN], None).unwrap();
        assert_eq!(i, 0);
        assert!(e.is_nan());
    }

    #[test]
    fn finds_global_optimum_of_coupled_objective() {
        let sizes = [4u32, 5, 3];
        let f = |x: &[u32]| {
            let (a, b, c) = (f64::from(x[0]), f64::from(x[1]), f64::from(x[2]));
            (a - b).powi(2) + (b + c - 4.0).powi(2) + 0.01 * a
        };
        let mut obj = FnObjective::new(&sizes, f);
        let sol = Exhaustive::default()
            .optimize(&SearchSpace::new(sizes.to_vec()), &mut obj, &mut rng())
            .unwrap();
        let mut best = (vec![], f64::INFINITY);
        for a in 0..4 {
            for b in 0..5 {
                for c in 0..3 {
                    let e = f(&[a, b, c]);
                    if e < best.1 {
                        best = (vec![a, b, c], e);
                    }
                }
            }
        }
        assert_eq!(sol.indices, best.0);
        assert_eq!(sol.error, best.1);
        assert_eq!(sol.evaluations, 60);
        assert_eq!(obj.rows, 60);
    }

    #[test]
    fn batches_cover_everything_once_in_order() {
        let sizes = [3u32, 4, 2];
        let mut seen: Vec<Vec<u32>> = Vec::new();
        let mut obj = FnObjective::new(&sizes, |x| {
            seen.push(x.to_vec());
            1.0
        });
        let ex = Exhaustive {
            batch_size: 7,
            ..Default::default()
        };
        let sol = ex
            .optimize(&SearchSpace::new(sizes.to_vec()), &mut obj, &mut rng())
            .unwrap();
        assert_eq!(obj.calls, 4); // 24 rows in batches of 7
        assert_eq!(obj.max_batch, 7);
        assert_eq!(sol.evaluations, 24);
        let mut expected = Vec::new();
        for a in 0..3 {
            for b in 0..4 {
                for c in 0..2 {
                    expected.push(vec![a, b, c]);
                }
            }
        }
        assert_eq!(seen, expected);
        assert_eq!(sol.indices, vec![0, 0, 0], "ties keep the first candidate");
    }

    #[test]
    fn tie_across_batches_keeps_first() {
        let sizes = [10u32];
        let mut obj = FnObjective::new(&sizes, |x| if x[0] % 2 == 1 { 0.0 } else { 1.0 });
        let ex = Exhaustive {
            batch_size: 3,
            ..Default::default()
        };
        let sol = ex
            .optimize(&SearchSpace::new(sizes.to_vec()), &mut obj, &mut rng())
            .unwrap();
        assert_eq!(sol.indices, vec![1]);
    }

    #[test]
    fn batch_size_edge_cases() {
        let sizes = [5u32];
        for batch_size in [0usize, 1, 2, 5, 6, 1000] {
            let mut obj = FnObjective::new(&sizes, |x| f64::from((x[0] as i32 - 3).unsigned_abs()));
            let ex = Exhaustive {
                batch_size,
                ..Default::default()
            };
            let sol = ex
                .optimize(&SearchSpace::new(sizes.to_vec()), &mut obj, &mut rng())
                .unwrap();
            assert_eq!(sol.indices, vec![3], "batch_size {batch_size}");
            assert_eq!(sol.error, 0.0);
            assert_eq!(obj.rows, 5);
            assert!(obj.max_batch <= batch_size.max(1));
        }
    }

    #[test]
    fn too_large_is_refused_without_evaluating() {
        let sizes = [10u32, 10, 10];
        let mut obj = FnObjective::new(&sizes, |_| 0.0);
        let mut space = SearchSpace::new(sizes.to_vec());
        space.budget = 999;
        match Exhaustive::default().optimize(&space, &mut obj, &mut rng()) {
            Err(FaultTuneError::SearchSpaceTooLarge { size, budget }) => {
                assert_eq!((size, budget), (1000, 999));
            }
            other => panic!("unexpected {other:?}"),
        }
        assert_eq!(obj.calls, 0);

        // Exactly at the limit is fine.
        space.budget = 1000;
        let sol = Exhaustive::default()
            .optimize(&space, &mut obj, &mut rng())
            .unwrap();
        assert_eq!(sol.evaluations, 1000);
    }

    #[test]
    fn max_size_applies_with_budget() {
        let sizes = [10u32, 10];
        let mut obj = FnObjective::new(&sizes, |_| 0.0);
        let space = SearchSpace::new(sizes.to_vec());
        let ex = Exhaustive {
            max_size: 99,
            ..Default::default()
        };
        match ex.optimize(&space, &mut obj, &mut rng()) {
            Err(FaultTuneError::SearchSpaceTooLarge { size, budget }) => {
                assert_eq!((size, budget), (100, 99));
            }
            other => panic!("unexpected {other:?}"),
        }
        assert_eq!(obj.calls, 0);
    }

    #[test]
    fn huge_space_saturates_without_overflow() {
        let sizes = vec![u32::MAX; 8];
        let mut obj = FnObjective::new(&sizes, |_| 0.0);
        let space = SearchSpace::new(sizes);
        match Exhaustive::default().optimize(&space, &mut obj, &mut rng()) {
            Err(FaultTuneError::SearchSpaceTooLarge { size, .. }) => {
                assert_eq!(size, u128::MAX);
            }
            other => panic!("unexpected {other:?}"),
        }
        assert_eq!(obj.calls, 0);
    }

    #[test]
    fn zero_options_is_invalid_config() {
        let sizes = [3u32, 0, 2];
        let mut obj = FnObjective::new(&sizes, |_| 0.0);
        let r =
            Exhaustive::default().optimize(&SearchSpace::new(sizes.to_vec()), &mut obj, &mut rng());
        assert!(matches!(r, Err(FaultTuneError::InvalidConfig(_))));
        assert_eq!(obj.calls, 0);
    }

    #[test]
    fn zero_width_space_is_evaluated_once() {
        let mut obj = FnObjective::new(&[], |_| 0.0);
        let sol = Exhaustive::default()
            .optimize(&SearchSpace::new(vec![]), &mut obj, &mut rng())
            .unwrap();
        assert!(sol.indices.is_empty());
        assert_eq!(sol.evaluations, 0, "no row was handed to the objective");
        assert_eq!(obj.calls, 1);
        assert_eq!(obj.rows, 0, "a zero-width batch has no rows");
    }

    #[test]
    fn zero_width_error_is_not_a_made_up_number() {
        // The objective would answer 5.0 for the empty candidate, but a zero-width batch
        // cannot carry that answer back: the error must not pose as a real value.
        let mut obj = FnObjective::new(&[], |_| 5.0);
        let sol = Exhaustive::default()
            .optimize(&SearchSpace::new(vec![]), &mut obj, &mut rng())
            .unwrap();
        assert!(sol.error.is_nan(), "got {}", sol.error);
    }

    #[test]
    fn huge_batch_size_neither_panics_nor_overallocates() {
        // ~2^64 combinations of width 2 with an unbounded batch size used to ask `Vec` for
        // more than `isize::MAX` bytes. The first batch is capped; the failing objective
        // stops the run right after it.
        let sizes = [u32::MAX, u32::MAX];
        let mut space = SearchSpace::new(sizes.to_vec());
        space.budget = u64::MAX;
        let ex = Exhaustive {
            batch_size: usize::MAX,
            max_size: u64::MAX,
        };
        let r = ex.optimize(&space, &mut Failing, &mut rng());
        assert!(matches!(r, Err(FaultTuneError::InvalidConfig(m)) if m == "boom"));
    }

    #[test]
    fn batches_are_capped_at_max_batch() {
        let sizes = [MAX_BATCH as u32 + 5];
        let mut obj = FnObjective::new(&sizes, |x| f64::from(x[0]));
        let ex = Exhaustive {
            batch_size: usize::MAX,
            max_size: u64::MAX,
        };
        let mut space = SearchSpace::new(sizes.to_vec());
        space.budget = u64::MAX;
        let sol = ex.optimize(&space, &mut obj, &mut rng()).unwrap();
        assert_eq!(sol.indices, vec![0]);
        assert_eq!(sol.evaluations, MAX_BATCH as u64 + 5);
        assert_eq!(obj.calls, 2);
        assert_eq!(obj.max_batch, MAX_BATCH);
    }

    #[test]
    fn single_option_parameters() {
        let sizes = [1u32, 1, 1];
        let mut obj = FnObjective::new(&sizes, |_| 2.5);
        let sol = Exhaustive::default()
            .optimize(&SearchSpace::new(sizes.to_vec()), &mut obj, &mut rng())
            .unwrap();
        assert_eq!(sol.indices, vec![0, 0, 0]);
        assert_eq!(sol.error, 2.5);
        assert_eq!(sol.evaluations, 1);
    }

    #[test]
    fn infinite_and_nan_errors_do_not_panic() {
        let sizes = [4u32];
        let mut obj = FnObjective::new(&sizes, |x| match x[0] {
            0 => f64::NAN,
            1 => f64::INFINITY,
            2 => 7.0,
            _ => f64::NAN,
        });
        let sol = Exhaustive::default()
            .optimize(&SearchSpace::new(sizes.to_vec()), &mut obj, &mut rng())
            .unwrap();
        assert_eq!(sol.indices, vec![2]);
        assert_eq!(sol.error, 7.0);

        let mut all_nan = FnObjective::new(&sizes, |_| f64::NAN);
        let sol = Exhaustive::default()
            .optimize(&SearchSpace::new(sizes.to_vec()), &mut all_nan, &mut rng())
            .unwrap();
        assert!(sol.error.is_nan());
        assert_eq!(sol.indices, vec![0]);

        let mut all_inf = FnObjective::new(&sizes, |_| f64::INFINITY);
        let sol = Exhaustive::default()
            .optimize(&SearchSpace::new(sizes.to_vec()), &mut all_inf, &mut rng())
            .unwrap();
        assert_eq!(sol.error, f64::INFINITY);
        assert_eq!(sol.indices, vec![0]);
    }

    #[test]
    fn objective_errors_propagate() {
        let r =
            Exhaustive::default().optimize(&SearchSpace::new(vec![2]), &mut Failing, &mut rng());
        assert!(matches!(r, Err(FaultTuneError::InvalidConfig(m)) if m == "boom"));
    }

    #[test]
    fn does_not_touch_the_rng() {
        let mut a = rng();
        let b = rng();
        let mut obj = FnObjective::new(&[3], |x| f64::from(x[0]));
        Exhaustive::default()
            .optimize(&SearchSpace::new(vec![3]), &mut obj, &mut a)
            .unwrap();
        assert_eq!(a, b);
    }

    proptest! {
        #[test]
        fn matches_a_naive_search(
            sizes in proptest::collection::vec(1u32..5, 1..5),
            seed in any::<u64>(),
            batch_size in 1usize..40,
        ) {
            let cost = |x: &[u32]| -> f64 {
                // Coupled and with plenty of ties (values on a coarse grid).
                let h = x.iter().enumerate().fold(seed, |acc, (j, &o)| {
                    acc.rotate_left(7) ^ (table_cost(acc, j, o) * 1e6) as u64
                });
                (h % 5) as f64
            };
            let mut obj = FnObjective::new(&sizes, cost);
            let ex = Exhaustive { batch_size, ..Default::default() };
            let sol = ex
                .optimize(&SearchSpace::new(sizes.clone()), &mut obj, &mut rng())
                .unwrap();

            // Naive lexicographic scan keeping the first minimum.
            let total: u64 = sizes.iter().map(|&k| u64::from(k)).product();
            let mut best: Option<(Vec<u32>, f64)> = None;
            for n in 0..total {
                let mut rest = n;
                let mut x = vec![0u32; sizes.len()];
                for (j, &k) in sizes.iter().enumerate().rev() {
                    x[j] = (rest % u64::from(k)) as u32;
                    rest /= u64::from(k);
                }
                let e = cost(&x);
                if best.as_ref().is_none_or(|(_, b)| e < *b) {
                    best = Some((x, e));
                }
            }
            let (bx, be) = best.unwrap();
            prop_assert_eq!(sol.indices, bx);
            prop_assert_eq!(sol.error, be);
            prop_assert_eq!(sol.evaluations, total);
            prop_assert_eq!(obj.rows, total);
        }
    }
}
