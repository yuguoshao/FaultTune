//! Coordinate descent: the default optimizer.

use rand::rngs::StdRng;
use rand::Rng;

use super::exhaustive::{best_of, check_space, is_better, solve_empty};
use super::{IndexBatch, Objective, Optimizer, SearchSpace, Solution};
use crate::error::FaultTuneError;

/// Starts from the best of a few random points, then repeatedly sweeps one parameter at a
/// time over all of its options (one batch per parameter), moving to the best option.
/// Stops when a full sweep makes no improvement, or after `max_sweeps` sweeps.
///
/// With `d_P = 1` and `k = 100` a step can have up to `100^5 = 10^10` combinations, far too
/// many to search exhaustively; a sweep costs only `Σ k_i` evaluations.
///
/// If every random start is forbidden (`+∞`), up to `max_start_attempts` further random
/// points are tried before sweeping from the best (infinite) start; any finite candidate
/// found by a sweep then counts as an improvement, which lets the search leave the
/// forbidden plateau. The search for a feasible start stops at the first batch that contains
/// one, so a model that rarely returns `+∞` never pays for the attempts.
///
/// **Limitation.** A model that can only say `+∞` ("forbidden") gives the search no gradient on
/// the forbidden plateau: a sweep over one parameter cannot see progress while another colliding
/// pair remains. For steps with many mutually constraining parameters (for example `d_P = 2`) a
/// random point is feasible with a small probability, and `max_start_attempts` bounds the effort
/// spent looking for one; if none is found the step is committed with error `+∞` (reported, never
/// silent). Models for such steps should return a graded penalty (a large finite number that grows
/// with the number of violations) instead of a flat `+∞`.
///
/// A sweep moves along parameter `j` only to an option that is *strictly* better than the
/// current point, so ties keep the current value (and otherwise the lowest option index).
/// The current value is part of every sweep batch (the engine de-duplicates it); parameters
/// with a single option are skipped. NaN counts as worse than every number.
///
/// The optimizer never asks the objective for more than `space.budget` candidates in total,
/// except that it always evaluates at least one point. A sweep batch that would exceed the
/// budget is not evaluated, but later (smaller) parameters that still fit are; a batch that
/// exactly fills the budget is evaluated. This differs from a literal reading of "one batch
/// for each parameter" only in that single-option parameters are skipped.
///
/// A `random_starts` of 0 is treated as 1. A space without parameters has a single empty
/// candidate; the objective is called once with a zero-width batch, which cannot return an
/// error, so the solution has `evaluations == 0` and a NaN error that callers must not use.
#[derive(Clone, Debug)]
pub struct CoordinateDescent {
    /// Random starting points evaluated first (as one batch).
    pub random_starts: usize,
    /// Maximum number of full sweeps over all parameters.
    pub max_sweeps: usize,
    /// Extra random points tried when all `random_starts` are infinite.
    pub max_start_attempts: usize,
}

impl Default for CoordinateDescent {
    fn default() -> Self {
        Self {
            random_starts: 4,
            max_sweeps: 3,
            max_start_attempts: 20_000,
        }
    }
}

/// Whether an error is a usable value: neither `+∞` (forbidden) nor NaN.
fn is_feasible(error: f64) -> bool {
    error < f64::INFINITY
}

/// Size of the batches used to look for a feasible start.
const START_RETRY_BATCH: u64 = 256;

/// Scratch space shared by the phases of one run.
struct Run<'a> {
    objective: &'a mut dyn Objective,
    width: usize,
    buf: Vec<u32>,
    out: Vec<f64>,
    /// Candidates handed to the objective so far.
    used: u64,
}

impl Run<'_> {
    /// Evaluates the rows currently in `buf`, leaving the errors in `out`.
    fn evaluate(&mut self) -> Result<(), FaultTuneError> {
        let n = self.buf.len() / self.width;
        self.out.resize(n, 0.0);
        self.objective
            .evaluate(IndexBatch::new(&self.buf, self.width), &mut self.out)?;
        self.used += n as u64;
        Ok(())
    }

    /// Evaluates `n` uniformly random points; returns the first best as `(row, error)`.
    fn random_points(
        &mut self,
        sizes: &[u32],
        n: usize,
        rng: &mut StdRng,
    ) -> Result<(usize, f64), FaultTuneError> {
        self.buf.clear();
        for _ in 0..n {
            for &k in sizes {
                self.buf.push(rng.random_range(0..k));
            }
        }
        self.evaluate()?;
        Ok(best_of(&self.out, None).expect("n >= 1"))
    }
}

impl Optimizer for CoordinateDescent {
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
        let sizes = &space.sizes;
        let budget = space.budget.max(1);
        let mut run = Run {
            objective,
            width,
            buf: Vec::new(),
            out: Vec::new(),
            used: 0,
        };

        // Phase 1: the best of a few random points, retrying while it is forbidden.
        let n = (self.random_starts.max(1) as u64).min(budget) as usize;
        let (row, mut current_error) = run.random_points(sizes, n, rng)?;
        let mut current: Vec<u32> = run.buf[row * width..(row + 1) * width].to_vec();
        let mut extra = self.max_start_attempts as u64;
        while !is_feasible(current_error) && extra > 0 && run.used < budget {
            let n = extra.min(START_RETRY_BATCH).min(budget - run.used) as usize;
            let (row, error) = run.random_points(sizes, n, rng)?;
            extra -= n as u64;
            if is_better(error, current_error) {
                current_error = error;
                current.copy_from_slice(&run.buf[row * width..(row + 1) * width]);
            }
        }

        // Phase 2: sweeps, one batch per parameter.
        for _ in 0..self.max_sweeps {
            let mut moved = false;
            for j in 0..width {
                let k = sizes[j];
                if k == 1 {
                    continue;
                }
                if run.used + u64::from(k) > budget {
                    // This batch does not fit; a later, smaller one still might.
                    continue;
                }
                run.buf.clear();
                for option in 0..k {
                    run.buf.extend_from_slice(&current);
                    let last = run.buf.len() - width + j;
                    run.buf[last] = option;
                }
                run.evaluate()?;
                let skip = current[j] as usize;
                if let Some((option, error)) = best_of(&run.out, Some(skip)) {
                    if is_better(error, current_error) {
                        current[j] = option as u32;
                        current_error = error;
                        moved = true;
                    }
                }
            }
            if !moved {
                break;
            }
        }

        Ok(Solution {
            indices: current,
            error: current_error,
            evaluations: run.used,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::exhaustive::test_support::*;
    use super::super::exhaustive::Exhaustive;
    use super::*;
    use proptest::prelude::*;
    use rand::{Rng, SeedableRng};

    fn rng(seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }

    fn separable(seed: u64) -> impl Fn(&[u32]) -> f64 {
        move |x: &[u32]| {
            x.iter()
                .enumerate()
                .map(|(j, &o)| table_cost(seed, j, o))
                .sum()
        }
    }

    fn argmin_separable(seed: u64, sizes: &[u32]) -> (Vec<u32>, f64) {
        let mut idx = Vec::new();
        let mut total = 0.0;
        for (j, &k) in sizes.iter().enumerate() {
            let (o, c) = (0..k)
                .map(|o| (o, table_cost(seed, j, o)))
                .fold((0, f64::INFINITY), |b, c| if c.1 < b.1 { c } else { b });
            idx.push(o);
            total += c;
        }
        (idx, total)
    }

    #[test]
    fn separable_objective_one_sweep_finds_global_optimum() {
        for sizes in [
            vec![7u32],
            vec![5, 9],
            vec![3, 4, 5],
            vec![10, 1, 10, 2],
            vec![8, 8, 8, 8, 8],
            vec![2, 2, 2, 2, 2, 2, 2, 2],
            vec![100, 100, 100],
        ] {
            for seed in 0..5u64 {
                let (best, _) = argmin_separable(seed, &sizes);
                let cd = CoordinateDescent {
                    random_starts: 1,
                    max_sweeps: 1,
                    max_start_attempts: 0,
                };
                let mut obj = FnObjective::new(&sizes, separable(seed));
                let sol = cd
                    .optimize(&SearchSpace::new(sizes.clone()), &mut obj, &mut rng(seed))
                    .unwrap();
                assert_eq!(sol.indices, best, "sizes {sizes:?} seed {seed}");
                let expected: f64 = best
                    .iter()
                    .enumerate()
                    .map(|(j, &o)| table_cost(seed, j, o))
                    .sum();
                assert_eq!(sol.error, expected);
                // One start plus one batch of k_j rows per parameter with k_j > 1.
                let sweep: u64 = sizes
                    .iter()
                    .filter(|&&k| k > 1)
                    .map(|&k| u64::from(k))
                    .sum();
                assert_eq!(sol.evaluations, 1 + sweep);
                assert_eq!(obj.rows, sol.evaluations);
            }
        }
    }

    #[test]
    fn default_settings_find_separable_optimum_and_stop_early() {
        let sizes = [6u32, 6, 6];
        let (best, _) = argmin_separable(9, &sizes);
        let mut obj = FnObjective::new(&sizes, separable(9));
        let sol = CoordinateDescent::default()
            .optimize(&SearchSpace::new(sizes.to_vec()), &mut obj, &mut rng(4))
            .unwrap();
        assert_eq!(sol.indices, best);
        // 4 starts + a moving sweep + a verifying sweep (18 each); a third sweep never runs.
        assert_eq!(sol.evaluations, 4 + 18 + 18);
        assert_eq!(obj.calls, 1 + 3 + 3);
    }

    #[test]
    fn sweep_batches_hold_one_coordinate_at_a_time() {
        // Record every batch to check the structure: each sweep batch differs from the
        // current point only in coordinate j and holds all k_j options in order.
        struct Recorder(Vec<(Vec<u32>, usize)>);
        impl Objective for Recorder {
            fn evaluate(
                &mut self,
                batch: IndexBatch<'_>,
                out: &mut [f64],
            ) -> Result<(), FaultTuneError> {
                assert_eq!(out.len(), batch.len());
                self.0.push((batch.as_flat().to_vec(), batch.len()));
                for (i, o) in out.iter_mut().enumerate() {
                    *o = batch.row(i).iter().map(|&v| f64::from(v)).sum();
                }
                Ok(())
            }
        }
        let sizes = vec![3u32, 4];
        let mut rec = Recorder(Vec::new());
        let cd = CoordinateDescent {
            random_starts: 2,
            max_sweeps: 1,
            max_start_attempts: 0,
        };
        let sol = cd
            .optimize(&SearchSpace::new(sizes), &mut rec, &mut rng(0))
            .unwrap();
        assert_eq!(sol.indices, vec![0, 0]);
        assert_eq!(rec.0.len(), 3);
        assert_eq!(rec.0[0].1, 2, "the starts are one batch");
        let sweep_a = &rec.0[1].0;
        let sweep_b = &rec.0[2].0;
        assert_eq!(sweep_a.len(), 3 * 2);
        assert_eq!(sweep_b.len(), 4 * 2);
        // The first sweep batch varies coordinate 0 only.
        let col1: Vec<u32> = sweep_a.chunks(2).map(|r| r[1]).collect();
        assert!(col1.windows(2).all(|w| w[0] == w[1]));
        let col0: Vec<u32> = sweep_a.chunks(2).map(|r| r[0]).collect();
        assert_eq!(col0, vec![0, 1, 2]);
        // The second varies coordinate 1 only, from the point reached after the first move.
        let col0: Vec<u32> = sweep_b.chunks(2).map(|r| r[0]).collect();
        assert!(col0.iter().all(|&v| v == 0));
        let col1: Vec<u32> = sweep_b.chunks(2).map(|r| r[1]).collect();
        assert_eq!(col1, vec![0, 1, 2, 3]);
    }

    #[test]
    fn coupled_convex_objective_reaches_the_exhaustive_optimum_through_sweeps() {
        // A strictly convex, diagonally dominant coupled chain with a known unique optimum.
        // The random start is a single point, so only the sweeps can get to the optimum.
        let sizes = [9u32, 9, 9, 9];
        let target = [1.0, 7.0, 3.0, 5.0];
        let f = move |x: &[u32]| {
            let own: f64 = x
                .iter()
                .zip(&target)
                .map(|(&v, t)| (f64::from(v) - t).powi(2))
                .sum();
            let chain: f64 = x
                .windows(2)
                .map(|w| (f64::from(w[0]) - f64::from(w[1])).powi(2))
                .sum();
            own + 0.1 * chain
        };
        let space = SearchSpace::new(sizes.to_vec());
        let opt = Exhaustive::default()
            .optimize(&space, &mut FnObjective::new(&sizes, f), &mut rng(0))
            .unwrap();
        let cd = CoordinateDescent {
            random_starts: 1,
            max_sweeps: 50,
            max_start_attempts: 0,
        };
        let one_sweep = CoordinateDescent {
            max_sweeps: 1,
            ..cd.clone()
        };
        let mut needed_more_than_one_sweep = false;
        for seed in 0..40 {
            let sol = cd
                .optimize(&space, &mut FnObjective::new(&sizes, f), &mut rng(seed))
                .unwrap();
            assert_eq!(
                (&sol.indices, sol.error),
                (&opt.indices, opt.error),
                "seed {seed}"
            );
            let single = one_sweep
                .optimize(&space, &mut FnObjective::new(&sizes, f), &mut rng(seed))
                .unwrap();
            needed_more_than_one_sweep |= single.error > opt.error;
        }
        assert!(
            needed_more_than_one_sweep,
            "the objective must be coupled enough that one sweep is not always enough"
        );
    }

    #[test]
    fn every_result_is_a_point_of_the_space_with_its_own_error() {
        // A non-convex coupled objective: sweeps can stall, but never beat the optimum, and
        // the reported error is always the error of the reported indices.
        let sizes = [6u32, 6, 5];
        let f = |x: &[u32]| {
            let (a, b, c) = (f64::from(x[0]), f64::from(x[1]), f64::from(x[2]));
            (a - b).powi(2) + (b - c).powi(2) + (a + b + c - 9.0).powi(2) + 0.001 * a
        };
        let space = SearchSpace::new(sizes.to_vec());
        let opt = Exhaustive::default()
            .optimize(&space, &mut FnObjective::new(&sizes, f), &mut rng(0))
            .unwrap();
        for seed in 0..20 {
            let sol = CoordinateDescent::default()
                .optimize(&space, &mut FnObjective::new(&sizes, f), &mut rng(seed))
                .unwrap();
            assert_eq!(sol.error, f(&sol.indices));
            assert!(sol.error >= opt.error);
        }
        // With enough random starts the best start is already the optimum.
        let cd = CoordinateDescent {
            random_starts: 2000,
            ..Default::default()
        };
        let sol = cd
            .optimize(&space, &mut FnObjective::new(&sizes, f), &mut rng(1))
            .unwrap();
        assert_eq!((sol.indices, sol.error), (opt.indices, opt.error));
    }

    #[test]
    fn coupled_objective_with_several_sweeps_descends() {
        // A coupled valley: the answer needs more than one sweep to reach.
        let sizes = [20u32, 20];
        let f = |x: &[u32]| {
            let (a, b) = (f64::from(x[0]), f64::from(x[1]));
            (a - 12.0).powi(2) + (b - a / 2.0).powi(2)
        };
        let space = SearchSpace::new(sizes.to_vec());
        let one = CoordinateDescent {
            random_starts: 1,
            max_sweeps: 1,
            max_start_attempts: 0,
        };
        let many = CoordinateDescent {
            random_starts: 1,
            max_sweeps: 10,
            max_start_attempts: 0,
        };
        let mut worse_somewhere = false;
        for seed in 0..30 {
            let a = one
                .optimize(&space, &mut FnObjective::new(&sizes, f), &mut rng(seed))
                .unwrap();
            let b = many
                .optimize(&space, &mut FnObjective::new(&sizes, f), &mut rng(seed))
                .unwrap();
            assert!(b.error <= a.error, "more sweeps never hurt");
            worse_somewhere |= b.error < a.error;
        }
        assert!(
            worse_somewhere,
            "extra sweeps must matter for a coupled valley"
        );
    }

    #[test]
    fn escapes_an_infinite_plateau() {
        // Only one point in 20^3 is feasible; random starts will not find it, but a sweep
        // finds a finite value as soon as one coordinate matches... so make feasibility
        // reachable one coordinate at a time: infinite unless x0 == 7, then cost on x1, x2.
        let sizes = [20u32, 20, 20];
        let f = |x: &[u32]| {
            if x[0] != 7 {
                f64::INFINITY
            } else {
                f64::from(x[1]) + f64::from(x[2])
            }
        };
        let cd = CoordinateDescent {
            random_starts: 1,
            max_sweeps: 3,
            max_start_attempts: 0,
        };
        let mut obj = FnObjective::new(&sizes, f);
        let sol = cd
            .optimize(&SearchSpace::new(sizes.to_vec()), &mut obj, &mut rng(5))
            .unwrap();
        assert_eq!(sol.indices, vec![7, 0, 0]);
        assert_eq!(sol.error, 0.0);
    }

    #[test]
    fn extra_start_attempts_find_a_feasible_start() {
        let sizes = [10u32, 10];
        // Feasible with probability 1/10 per point; coordinate sweeps cannot help since
        // the optimizer is given zero sweeps.
        let f = |x: &[u32]| {
            if x[0] == 3 {
                f64::from(x[1])
            } else {
                f64::INFINITY
            }
        };
        let cd = CoordinateDescent {
            random_starts: 1,
            max_sweeps: 0,
            max_start_attempts: 500,
        };
        for seed in 0..10 {
            let mut obj = FnObjective::new(&sizes, f);
            let sol = cd
                .optimize(&SearchSpace::new(sizes.to_vec()), &mut obj, &mut rng(seed))
                .unwrap();
            assert!(sol.error.is_finite(), "seed {seed}");
            assert_eq!(sol.indices[0], 3);
            assert_eq!(sol.error, f64::from(sol.indices[1]));
            assert!(sol.evaluations <= 1 + 500);
            assert_eq!(sol.evaluations, obj.rows);
            // Retries run in batches, and stop after the first batch with a feasible point.
            assert!(obj.max_batch <= START_RETRY_BATCH as usize);
        }
    }

    #[test]
    fn start_attempts_are_limited_and_counted() {
        let sizes = [4u32, 4];
        let mut obj = FnObjective::new(&sizes, |_| f64::INFINITY);
        let cd = CoordinateDescent {
            random_starts: 3,
            max_sweeps: 0,
            max_start_attempts: 100,
        };
        let sol = cd
            .optimize(&SearchSpace::new(sizes.to_vec()), &mut obj, &mut rng(1))
            .unwrap();
        assert_eq!(sol.error, f64::INFINITY);
        assert_eq!(sol.evaluations, 103);
        assert_eq!(obj.rows, 103);
        assert_eq!(obj.calls, 1 + 1); // the 100 retries fit in one batch
    }

    #[test]
    fn fully_infinite_space_returns_infinity_after_sweeps() {
        let sizes = [3u32, 3];
        let mut obj = FnObjective::new(&sizes, |_| f64::INFINITY);
        let sol = CoordinateDescent::default()
            .optimize(&SearchSpace::new(sizes.to_vec()), &mut obj, &mut rng(1))
            .unwrap();
        assert_eq!(sol.error, f64::INFINITY);
        // Starts, retries, then one sweep (no move, stops).
        let attempts = CoordinateDescent::default().max_start_attempts as u64;
        assert_eq!(sol.evaluations, 4 + attempts + 6);
    }

    #[test]
    fn single_option_parameters_are_skipped() {
        let sizes = [1u32];
        let mut obj = FnObjective::new(&sizes, |_| 3.0);
        let sol = CoordinateDescent::default()
            .optimize(&SearchSpace::new(sizes.to_vec()), &mut obj, &mut rng(1))
            .unwrap();
        assert_eq!((sol.indices, sol.error), (vec![0], 3.0));
        assert_eq!(sol.evaluations, 4);
        assert_eq!(obj.calls, 1);

        let sizes = [1u32, 5, 1];
        let mut obj = FnObjective::new(&sizes, |x| (f64::from(x[1]) - 2.0).abs());
        let sol = CoordinateDescent::default()
            .optimize(&SearchSpace::new(sizes.to_vec()), &mut obj, &mut rng(1))
            .unwrap();
        assert_eq!(sol.indices, vec![0, 2, 0]);
        assert_eq!(sol.error, 0.0);
    }

    #[test]
    fn width_one() {
        let sizes = [50u32];
        let mut obj = FnObjective::new(&sizes, |x| (f64::from(x[0]) - 37.0).abs());
        let sol = CoordinateDescent::default()
            .optimize(&SearchSpace::new(sizes.to_vec()), &mut obj, &mut rng(2))
            .unwrap();
        assert_eq!(sol.indices, vec![37]);
        assert_eq!(sol.error, 0.0);
    }

    #[test]
    fn width_zero_and_zero_options() {
        let mut obj = FnObjective::new(&[], |_| 0.0);
        let sol = CoordinateDescent::default()
            .optimize(&SearchSpace::new(vec![]), &mut obj, &mut rng(1))
            .unwrap();
        assert!(sol.indices.is_empty());
        assert_eq!((sol.evaluations, obj.calls, obj.rows), (0, 1, 0));
        assert!(sol.error.is_nan(), "a zero-width error is unknown, not 0.0");

        let mut obj = FnObjective::new(&[2, 0], |_| 0.0);
        let r = CoordinateDescent::default().optimize(
            &SearchSpace::new(vec![2, 0]),
            &mut obj,
            &mut rng(1),
        );
        assert!(matches!(r, Err(FaultTuneError::InvalidConfig(_))));
        assert_eq!(obj.calls, 0);
    }

    #[test]
    fn ties_keep_the_current_value() {
        let sizes = [5u32, 5];
        let mut obj = FnObjective::new(&sizes, |_| 1.0);
        let cd = CoordinateDescent {
            random_starts: 1,
            ..Default::default()
        };
        // The sole start is the first random point; a flat objective never moves.
        let mut probe = rng(6);
        let start: Vec<u32> = sizes.iter().map(|&k| probe.random_range(0..k)).collect();
        let sol = cd
            .optimize(&SearchSpace::new(sizes.to_vec()), &mut obj, &mut rng(6))
            .unwrap();
        assert_eq!(sol.indices, start);
        assert_eq!(sol.evaluations, 1 + 10, "one sweep, no move, stop");
    }

    #[test]
    fn re_evaluated_current_point_is_not_a_move() {
        // A noisy objective: the very first point looks worse the first time than when it is
        // evaluated again as part of a sweep batch. That is not an improvement.
        let sizes = [4u32, 4];
        let mut first: Option<Vec<u32>> = None;
        let mut obj = FnObjective::new(&sizes, |x| match &first {
            None => {
                first = Some(x.to_vec());
                5.0
            }
            Some(p) if p.as_slice() == x => 0.1,
            Some(_) => 5.0,
        });
        let cd = CoordinateDescent {
            random_starts: 1,
            ..Default::default()
        };
        let sol = cd
            .optimize(&SearchSpace::new(sizes.to_vec()), &mut obj, &mut rng(0))
            .unwrap();
        assert_eq!(sol.error, 5.0);
        assert_eq!(sol.evaluations, 1 + 4 + 4, "one sweep without a move");
    }

    #[test]
    fn best_start_wins_with_first_found_tie_break() {
        let sizes = [1000u32];
        let cd = CoordinateDescent {
            random_starts: 5,
            max_sweeps: 0,
            max_start_attempts: 0,
        };
        let mut seen = Vec::new();
        let mut obj = FnObjective::new(&sizes, |x| {
            seen.push(x[0]);
            f64::from(x[0] % 3)
        });
        let sol = cd
            .optimize(&SearchSpace::new(sizes.to_vec()), &mut obj, &mut rng(3))
            .unwrap();
        let first_best = seen
            .iter()
            .copied()
            .fold(None::<u32>, |b, v| match b {
                Some(bv) if v % 3 >= bv % 3 => Some(bv),
                _ => Some(v),
            })
            .unwrap();
        assert_eq!(sol.indices, vec![first_best]);
    }

    #[test]
    fn budget_is_respected() {
        let sizes = [10u32, 10, 10];
        for budget in [1u64, 2, 4, 5, 13, 14, 15, 24, 25, 34, 35, 44, 100] {
            let mut obj = FnObjective::new(&sizes, separable(1));
            let mut space = SearchSpace::new(sizes.to_vec());
            space.budget = budget;
            let sol = CoordinateDescent::default()
                .optimize(&space, &mut obj, &mut rng(1))
                .unwrap();
            assert!(obj.rows <= budget.max(1), "budget {budget}: {}", obj.rows);
            assert_eq!(sol.evaluations, obj.rows);
            assert_eq!(sol.error, separable(1)(&sol.indices));
        }
    }

    #[test]
    fn budget_zero_still_evaluates_one_point() {
        let sizes = [10u32, 10];
        let mut obj = FnObjective::new(&sizes, separable(1));
        let mut space = SearchSpace::new(sizes.to_vec());
        space.budget = 0;
        let sol = CoordinateDescent::default()
            .optimize(&space, &mut obj, &mut rng(1))
            .unwrap();
        assert_eq!(sol.evaluations, 1);
        assert_eq!(obj.calls, 1);
    }

    #[test]
    fn budget_stops_before_a_batch_that_would_not_fit() {
        let sizes = [10u32, 10];
        let mut obj = FnObjective::new(&sizes, separable(2));
        let mut space = SearchSpace::new(sizes.to_vec());
        // 1 start + 10 for parameter 0 fits (11); parameter 1 would make 21.
        space.budget = 20;
        let cd = CoordinateDescent {
            random_starts: 1,
            ..Default::default()
        };
        let sol = cd.optimize(&space, &mut obj, &mut rng(1)).unwrap();
        assert_eq!(sol.evaluations, 11);
    }

    #[test]
    fn a_batch_that_exactly_fits_the_budget_is_evaluated() {
        let sizes = [10u32, 10];
        let cd = CoordinateDescent {
            random_starts: 1,
            max_sweeps: 3,
            max_start_attempts: 0,
        };
        // 1 start + 10 + 10 fits exactly; one less leaves the second parameter out.
        for (budget, expected) in [(21u64, 21u64), (20, 11), (11, 11), (10, 1)] {
            let mut obj = FnObjective::new(&sizes, |_| 1.0);
            let mut space = SearchSpace::new(sizes.to_vec());
            space.budget = budget;
            let sol = cd.optimize(&space, &mut obj, &mut rng(1)).unwrap();
            assert_eq!(sol.evaluations, expected, "budget {budget}");
            assert_eq!(obj.rows, expected, "budget {budget}");
        }
    }

    #[test]
    fn a_later_parameter_that_still_fits_is_swept() {
        // Budget 12 = 1 start + 10 for parameter 0, leaving 1 row: neither two-option
        // parameter (2 rows each) fits any more, so the sweep ends after parameter 0.
        let sizes = [10u32, 2, 2];
        let mut space = SearchSpace::new(sizes.to_vec());
        space.budget = 1 + 10 + 1;
        let cd = CoordinateDescent {
            random_starts: 1,
            max_sweeps: 1,
            max_start_attempts: 0,
        };
        let mut obj = FnObjective::new(&sizes, |x| f64::from(x[0] + x[1] + x[2]));
        let sol = cd.optimize(&space, &mut obj, &mut rng(3)).unwrap();
        assert_eq!(sol.evaluations, 11, "11 + 2 > 12: nothing else fits");

        // Parameter 0 cannot be swept (1 + 10 > 6) but the two-option ones can.
        space.budget = 6;
        let mut obj = FnObjective::new(&sizes, |x| f64::from(x[0] + x[1] + x[2]));
        let sol = cd.optimize(&space, &mut obj, &mut rng(3)).unwrap();
        assert_eq!(sol.evaluations, 1 + 2 + 2);
        assert_eq!(sol.indices[1..], [0, 0]);
        assert!(obj.rows <= 6);
    }

    #[test]
    fn deterministic_for_a_seed_and_seeds_give_different_paths() {
        let sizes = [30u32, 30, 30];
        let space = SearchSpace::new(sizes.to_vec());
        let run = |seed| {
            let mut rows = Vec::new();
            let mut obj = FnObjective::new(&sizes, |x| {
                rows.push(x.to_vec());
                let (a, b, c) = (f64::from(x[0]), f64::from(x[1]), f64::from(x[2]));
                (a - b).powi(2) + (b - c).powi(2) + (c - 17.0).abs()
            });
            let sol = CoordinateDescent::default()
                .optimize(&space, &mut obj, &mut rng(seed))
                .unwrap();
            (sol, rows)
        };
        let (a, rows_a) = run(21);
        let (b, rows_b) = run(21);
        assert_eq!(a, b);
        assert_eq!(rows_a, rows_b);
        let (_, rows_c) = run(22);
        assert_ne!(rows_a, rows_c);
    }

    #[test]
    fn nan_is_worse_than_everything_and_never_panics() {
        let sizes = [5u32, 5];
        let f = |x: &[u32]| {
            if x[0] == 2 && x[1] == 3 {
                0.5
            } else if (x[0] + x[1]).is_multiple_of(2) {
                f64::NAN
            } else {
                10.0 + f64::from(x[0])
            }
        };
        for seed in 0..20 {
            let sol = CoordinateDescent::default()
                .optimize(
                    &SearchSpace::new(sizes.to_vec()),
                    &mut FnObjective::new(&sizes, f),
                    &mut rng(seed),
                )
                .unwrap();
            assert!(!sol.error.is_nan(), "a number exists in every sweep line");
        }
        // Everything NaN: no panic, result is NaN.
        let sol = CoordinateDescent::default()
            .optimize(
                &SearchSpace::new(sizes.to_vec()),
                &mut FnObjective::new(&sizes, |_| f64::NAN),
                &mut rng(1),
            )
            .unwrap();
        assert!(sol.error.is_nan());
    }

    #[test]
    fn objective_errors_propagate() {
        let r = CoordinateDescent::default().optimize(
            &SearchSpace::new(vec![3, 3]),
            &mut Failing,
            &mut rng(1),
        );
        assert!(matches!(r, Err(FaultTuneError::InvalidConfig(m)) if m == "boom"));

        // Failing only in the sweep phase.
        struct FailAfter(usize);
        impl Objective for FailAfter {
            fn evaluate(
                &mut self,
                _: IndexBatch<'_>,
                out: &mut [f64],
            ) -> Result<(), FaultTuneError> {
                if self.0 == 0 {
                    return Err(FaultTuneError::InvalidConfig("late".into()));
                }
                self.0 -= 1;
                out.fill(1.0);
                Ok(())
            }
        }
        let r = CoordinateDescent::default().optimize(
            &SearchSpace::new(vec![3, 3]),
            &mut FailAfter(2),
            &mut rng(1),
        );
        assert!(matches!(r, Err(FaultTuneError::InvalidConfig(m)) if m == "late"));
    }

    proptest! {
        #[test]
        fn separable_optimum_for_random_shapes(
            sizes in proptest::collection::vec(1u32..12, 1..7),
            seed in any::<u64>(),
            rng_seed in any::<u64>(),
            starts in 1usize..5,
        ) {
            let (best, best_cost) = argmin_separable(seed, &sizes);
            let cd = CoordinateDescent { random_starts: starts, max_sweeps: 1, max_start_attempts: 8 };
            let mut obj = FnObjective::new(&sizes, separable(seed));
            let sol = cd
                .optimize(&SearchSpace::new(sizes.clone()), &mut obj, &mut rng(rng_seed))
                .unwrap();
            prop_assert_eq!(sol.indices, best);
            prop_assert!((sol.error - best_cost).abs() < 1e-12);
            prop_assert_eq!(sol.evaluations, obj.rows);
        }

        #[test]
        fn result_is_valid_and_consistent_on_arbitrary_objectives(
            sizes in proptest::collection::vec(1u32..6, 1..5),
            seed in any::<u64>(),
            budget in 0u64..120,
            sweeps in 0usize..4,
            starts in 0usize..6,
            attempts in 0usize..80,
        ) {
            let cost = |x: &[u32]| -> f64 {
                let h = x.iter().enumerate().fold(0.0, |acc, (j, &o)| {
                    acc * 1.7 + table_cost(seed, j, o)
                });
                // Roughly a fifth of the points are forbidden.
                if ((h * 100.0) as u64).is_multiple_of(5) { f64::INFINITY } else { h }
            };
            let mut obj = FnObjective::new(&sizes, cost);
            let mut space = SearchSpace::new(sizes.clone());
            space.budget = budget;
            let cd = CoordinateDescent { random_starts: starts, max_sweeps: sweeps, max_start_attempts: attempts };
            let sol = cd.optimize(&space, &mut obj, &mut rng(seed)).unwrap();
            prop_assert_eq!(sol.evaluations, obj.rows);
            prop_assert!(obj.rows <= budget.max(1));
            prop_assert_eq!(sol.error, cost(&sol.indices));
            for (&x, &k) in sol.indices.iter().zip(&sizes) {
                prop_assert!(x < k);
            }
        }
    }
}
