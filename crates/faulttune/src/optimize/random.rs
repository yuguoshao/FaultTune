//! Random search.

use rand::rngs::StdRng;
use rand::Rng;

use super::exhaustive::{best_of, check_space, clamp_batch_size, is_better, solve_empty};
use super::{IndexBatch, Objective, Optimizer, SearchSpace, Solution};
use crate::error::FaultTuneError;

/// Samples a fixed number of candidates uniformly at random (with replacement) and keeps the
/// best. Useful as a baseline and in tests.
///
/// Exactly `min(samples, space.budget)` candidates are evaluated, but at least one. A
/// `batch_size` of 0 is treated as 1 and a `batch_size` above 2^20 is capped there. Ties are
/// resolved in favour of the earliest sample.
///
/// A space without parameters has a single empty candidate; the objective is called once
/// with a zero-width batch, which cannot return an error, so the solution has
/// `evaluations == 0` and a NaN error that callers must not use.
#[derive(Clone, Debug)]
pub struct RandomSearch {
    /// Number of candidates to sample (capped by `SearchSpace::budget`).
    pub samples: usize,
    /// Candidates per objective call.
    pub batch_size: usize,
}

impl Default for RandomSearch {
    fn default() -> Self {
        Self {
            samples: 1000,
            batch_size: 1000,
        }
    }
}

impl Optimizer for RandomSearch {
    fn optimize(
        &self,
        space: &SearchSpace,
        objective: &mut dyn Objective,
        rng: &mut StdRng,
    ) -> Result<Solution, FaultTuneError> {
        check_space(space)?;
        let width = space.width();
        if width == 0 {
            return solve_empty(objective);
        }

        let total = (self.samples as u64).min(space.budget).max(1);
        let batch_size = clamp_batch_size(self.batch_size, total);

        let mut buf: Vec<u32> = Vec::with_capacity(batch_size * width);
        let mut out = vec![0.0f64; batch_size];
        let mut best: Option<(Vec<u32>, f64)> = None;
        let mut remaining = total;
        while remaining > 0 {
            let n = remaining.min(batch_size as u64) as usize;
            buf.clear();
            for _ in 0..n {
                for &k in &space.sizes {
                    buf.push(rng.random_range(0..k));
                }
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
        let (indices, error) = best.expect("at least one sample was evaluated");
        Ok(Solution {
            indices,
            error,
            evaluations: total,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::exhaustive::test_support::*;
    use super::super::exhaustive::Exhaustive;
    use super::*;
    use proptest::prelude::*;
    use rand::SeedableRng;

    fn rng(seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }

    fn coupled(x: &[u32]) -> f64 {
        x.windows(2)
            .map(|w| (f64::from(w[0]) - f64::from(w[1])).powi(2))
            .sum::<f64>()
            + x.iter().map(|&v| f64::from(v) * 0.01).sum::<f64>()
    }

    #[test]
    fn counts_samples_and_batches() {
        let sizes = [5u32, 5, 5];
        let mut obj = FnObjective::new(&sizes, coupled);
        let rs = RandomSearch {
            samples: 25,
            batch_size: 10,
        };
        let sol = rs
            .optimize(&SearchSpace::new(sizes.to_vec()), &mut obj, &mut rng(3))
            .unwrap();
        assert_eq!(sol.evaluations, 25);
        assert_eq!(obj.rows, 25);
        assert_eq!(obj.calls, 3);
        assert_eq!(obj.max_batch, 10);
    }

    #[test]
    fn budget_caps_samples() {
        let sizes = [5u32, 5];
        let mut obj = FnObjective::new(&sizes, coupled);
        let mut space = SearchSpace::new(sizes.to_vec());
        space.budget = 7;
        let sol = RandomSearch::default()
            .optimize(&space, &mut obj, &mut rng(3))
            .unwrap();
        assert_eq!(sol.evaluations, 7);
        assert_eq!(obj.rows, 7);
    }

    #[test]
    fn at_least_one_sample() {
        for (samples, budget) in [(0usize, 100u64), (10, 0), (0, 0)] {
            let sizes = [3u32];
            let mut obj = FnObjective::new(&sizes, |x| f64::from(x[0]));
            let mut space = SearchSpace::new(sizes.to_vec());
            space.budget = budget;
            let rs = RandomSearch {
                samples,
                batch_size: 4,
            };
            let sol = rs.optimize(&space, &mut obj, &mut rng(1)).unwrap();
            assert_eq!(sol.evaluations, 1);
            assert_eq!(sol.error, f64::from(sol.indices[0]));
        }
    }

    #[test]
    fn zero_batch_size_is_one() {
        let sizes = [3u32];
        let mut obj = FnObjective::new(&sizes, |x| f64::from(x[0]));
        let rs = RandomSearch {
            samples: 5,
            batch_size: 0,
        };
        rs.optimize(&SearchSpace::new(sizes.to_vec()), &mut obj, &mut rng(1))
            .unwrap();
        assert_eq!((obj.calls, obj.rows, obj.max_batch), (5, 5, 1));
    }

    #[test]
    fn finds_optimum_with_many_samples() {
        let sizes = [4u32, 4, 4];
        let mut exhaustive_obj = FnObjective::new(&sizes, coupled);
        let space = SearchSpace::new(sizes.to_vec());
        let opt = Exhaustive::default()
            .optimize(&space, &mut exhaustive_obj, &mut rng(0))
            .unwrap();
        let mut obj = FnObjective::new(&sizes, coupled);
        let rs = RandomSearch {
            samples: 2000,
            batch_size: 128,
        };
        let sol = rs.optimize(&space, &mut obj, &mut rng(5)).unwrap();
        assert_eq!(sol.indices, opt.indices);
        assert_eq!(sol.error, opt.error);
    }

    #[test]
    fn deterministic_for_a_seed_and_seeds_differ() {
        let sizes = [50u32, 50, 50];
        let space = SearchSpace::new(sizes.to_vec());
        let rs = RandomSearch {
            samples: 40,
            batch_size: 16,
        };
        let run = |seed| {
            let mut seen = Vec::new();
            let mut obj = FnObjective::new(&sizes, |x| {
                seen.push(x.to_vec());
                coupled(x)
            });
            let sol = rs.optimize(&space, &mut obj, &mut rng(seed)).unwrap();
            (sol, seen)
        };
        let (a, seen_a) = run(11);
        let (b, seen_b) = run(11);
        let (_, seen_c) = run(12);
        assert_eq!(a, b);
        assert_eq!(seen_a, seen_b);
        assert_ne!(seen_a, seen_c);
    }

    #[test]
    fn samples_cover_every_option() {
        let sizes = [3u32, 2];
        let mut seen = std::collections::HashSet::new();
        let mut obj = FnObjective::new(&sizes, |x| {
            seen.insert(x.to_vec());
            0.0
        });
        let rs = RandomSearch {
            samples: 500,
            batch_size: 100,
        };
        let sol = rs
            .optimize(&SearchSpace::new(sizes.to_vec()), &mut obj, &mut rng(2))
            .unwrap();
        assert_eq!(seen.len(), 6);
        // All errors are equal, so the very first sample is returned.
        assert_eq!(sol.error, 0.0);
    }

    #[test]
    fn invalid_and_empty_spaces() {
        let r = RandomSearch::default().optimize(
            &SearchSpace::new(vec![2, 0]),
            &mut FnObjective::new(&[2, 0], |_| 0.0),
            &mut rng(1),
        );
        assert!(matches!(r, Err(FaultTuneError::InvalidConfig(_))));

        let mut obj = FnObjective::new(&[], |_| 0.0);
        let sol = RandomSearch::default()
            .optimize(&SearchSpace::new(vec![]), &mut obj, &mut rng(1))
            .unwrap();
        assert!(sol.indices.is_empty());
        assert_eq!((sol.evaluations, obj.calls, obj.rows), (0, 1, 0));
        assert!(sol.error.is_nan(), "a zero-width error is unknown, not 0.0");

        let mut obj = FnObjective::new(&[1], |_| 4.0);
        let sol = RandomSearch::default()
            .optimize(&SearchSpace::new(vec![1]), &mut obj, &mut rng(1))
            .unwrap();
        assert_eq!((sol.indices, sol.error), (vec![0], 4.0));
    }

    #[test]
    fn huge_batch_size_is_capped() {
        // 2^64 samples of width 2 in one batch used to overflow the buffer allocation.
        let sizes = [3u32, 3];
        let mut space = SearchSpace::new(sizes.to_vec());
        space.budget = u64::MAX;
        let rs = RandomSearch {
            samples: usize::MAX,
            batch_size: usize::MAX,
        };
        let r = rs.optimize(&space, &mut Failing, &mut rng(1));
        assert!(matches!(r, Err(FaultTuneError::InvalidConfig(m)) if m == "boom"));
    }

    #[test]
    fn nan_never_wins_and_errors_propagate() {
        let sizes = [6u32];
        let mut obj = FnObjective::new(&sizes, |x| if x[0] == 4 { 1.0 } else { f64::NAN });
        let rs = RandomSearch {
            samples: 300,
            batch_size: 50,
        };
        let sol = rs
            .optimize(&SearchSpace::new(sizes.to_vec()), &mut obj, &mut rng(8))
            .unwrap();
        assert_eq!((sol.indices, sol.error), (vec![4], 1.0));

        let r =
            RandomSearch::default().optimize(&SearchSpace::new(vec![2]), &mut Failing, &mut rng(1));
        assert!(matches!(r, Err(FaultTuneError::InvalidConfig(_))));
    }

    proptest! {
        #[test]
        fn returned_error_is_the_error_of_the_returned_indices(
            sizes in proptest::collection::vec(1u32..30, 1..6),
            samples in 1usize..200,
            batch_size in 0usize..64,
            seed in any::<u64>(),
        ) {
            let mut obj = FnObjective::new(&sizes, |x| {
                x.iter().enumerate().map(|(j, &o)| table_cost(7, j, o)).sum()
            });
            let rs = RandomSearch { samples, batch_size };
            let sol = rs
                .optimize(&SearchSpace::new(sizes.clone()), &mut obj, &mut rng(seed))
                .unwrap();
            let recomputed: f64 =
                sol.indices.iter().enumerate().map(|(j, &o)| table_cost(7, j, o)).sum();
            prop_assert_eq!(sol.error, recomputed);
            prop_assert_eq!(sol.evaluations, samples as u64);
            prop_assert_eq!(obj.rows, samples as u64);
            for (&x, &k) in sol.indices.iter().zip(&sizes) {
                prop_assert!(x < k);
            }
        }
    }
}
