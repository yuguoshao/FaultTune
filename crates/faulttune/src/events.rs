//! Observability: events, observers and the run report (ARCHITECTURE §5.4, §11).

use std::cmp::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::database::StepRecord;
use crate::graph::ElementId;

/// Something that happens during a calibration run.
///
/// Events from different subgoals can interleave (and arrive from different threads) when
/// subgoals run in parallel; the order *within* a subgoal is the commit order.
#[derive(Clone, Debug)]
pub enum Event {
    /// A pass of the completion loop begins.
    PassStarted {
        /// 0-based pass number.
        pass: u32,
        /// Number of subgoals the uncalibrated goal was split into.
        subgoals: usize,
        /// Number of goal elements still uncalibrated.
        uncalibrated: usize,
    },
    /// A step was committed. `StepRecord::index` is the ordinal within its subgoal and pass
    /// (the global index is assigned when the engine merges the subgoals after the pass), and
    /// `pass` and `subgoal` are final. The committed values are
    /// `model.options(element)[option_indices[i]]`. Because the events of parallel subgoals
    /// interleave, the order in which these events arrive is not the order of the history:
    /// that is `(pass, subgoal, index)` (see the `persist` module).
    StepCommitted(StepRecord),
    /// A step failed and was skipped (`ModelErrorPolicy::SkipElement`).
    StepFailed {
        /// Centre of the failed step.
        centre: ElementId,
        /// Description of the failure.
        error: String,
    },
    /// `recalibrate` removed these elements (ascending) from `P*` before re-learning them.
    /// Emitted once, after the database was updated and before the first pass of the
    /// re-calibration, so a journal can record the removal.
    Invalidated {
        /// The number the next pass will have. Pass numbers continue across the `calibrate_*`
        /// calls on one database, so `(pass, subgoal, centre)` identifies a step for the whole
        /// life of a database, and an invalidation sorts between the steps of earlier passes
        /// and those of pass `pass`.
        pass: u32,
        /// The elements that were calibrated and are now forgotten.
        elements: Vec<ElementId>,
    },
    /// A pass finished.
    PassFinished {
        /// 0-based pass number.
        pass: u32,
        /// Number of elements calibrated in the pass.
        calibrated: usize,
    },
    /// The run ended: successfully, after a cancellation, or after an error. It is the last
    /// event of every `calibrate_*` call that did not panic.
    Finished(Report),
}

/// Receives [`Event`]s. Must be thread-safe: events can arrive from worker threads.
pub trait Observer: Send + Sync {
    /// Called for every event.
    fn on_event(&self, event: &Event);
}

impl<F: Fn(&Event) + Send + Sync> Observer for F {
    fn on_event(&self, event: &Event) {
        self(event)
    }
}

impl<T: Observer + ?Sized> Observer for Arc<T> {
    fn on_event(&self, event: &Event) {
        (**self).on_event(event)
    }
}

/// An observer that stores every event, for tests and simple scripts.
#[derive(Clone, Debug, Default)]
pub struct RecordingObserver {
    events: Arc<Mutex<Vec<Event>>>,
}

impl RecordingObserver {
    /// A new, empty recorder. Clones share the same buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// A copy of the events recorded so far.
    pub fn events(&self) -> Vec<Event> {
        self.events
            .lock()
            .expect("recording observer mutex")
            .clone()
    }
}

impl Observer for RecordingObserver {
    fn on_event(&self, event: &Event) {
        self.events
            .lock()
            .expect("recording observer mutex")
            .push(event.clone());
    }
}

/// Summary of a calibration run.
#[derive(Clone, Debug, Default)]
pub struct Report {
    /// Number of passes of the completion loop.
    pub passes: u32,
    /// Number of steps committed.
    pub steps: u64,
    /// Total candidates evaluated (sum of `StepRecord::evaluations`).
    pub evaluations: u64,
    /// Cross-step cache hits (0 when the cache is off).
    pub cache_hits: u64,
    /// Cross-step cache misses (0 when the cache is off).
    pub cache_misses: u64,
    /// Wall-clock time of the run.
    pub wall_time: Duration,
    /// Total time spent in `ErrorModel::prepare`, summed over steps (CPU-like, not wall).
    pub prepare_time: Duration,
    /// Total time spent evaluating errors, summed over steps.
    pub error_time: Duration,
    /// Elements excluded from the run after a model error (`ModelErrorPolicy::SkipElement`).
    pub skipped: Vec<ElementId>,
    /// The steps with the largest error of this run, worst first (at most 10).
    pub worst_steps: Vec<StepRecord>,
}

impl Report {
    /// Fraction of cross-step cache lookups that hit, or `None` if there were none.
    pub fn cache_hit_rate(&self) -> Option<f64> {
        let total = self.cache_hits + self.cache_misses;
        (total > 0).then(|| self.cache_hits as f64 / total as f64)
    }

    /// Adds the report of another stage of a multi-stage run (for example the regions and the
    /// gap of a stitched calibration) to this one.
    ///
    /// Counters and times are summed (saturating), `skipped` is merged with `other`'s elements
    /// (ascending, without duplicates), and `worst_steps` keeps the [`MAX_WORST_STEPS`] steps with the largest error
    /// of both reports, worst first, ties broken by the lower step index. The number of
    /// passes adds up: the stages ran one after another.
    pub fn absorb(&mut self, other: &Report) {
        self.passes = self.passes.saturating_add(other.passes);
        self.steps = self.steps.saturating_add(other.steps);
        self.evaluations = self.evaluations.saturating_add(other.evaluations);
        self.cache_hits = self.cache_hits.saturating_add(other.cache_hits);
        self.cache_misses = self.cache_misses.saturating_add(other.cache_misses);
        self.wall_time = self.wall_time.saturating_add(other.wall_time);
        self.prepare_time = self.prepare_time.saturating_add(other.prepare_time);
        self.error_time = self.error_time.saturating_add(other.error_time);
        self.skipped.extend_from_slice(&other.skipped);
        self.skipped.sort_unstable();
        self.skipped.dedup();
        self.worst_steps.extend_from_slice(&other.worst_steps);
        self.worst_steps.sort_by(worst_first);
        self.worst_steps.truncate(MAX_WORST_STEPS);
    }
}

/// The number of steps a [`Report`] keeps in `worst_steps`.
pub const MAX_WORST_STEPS: usize = 10;

/// Orders steps by decreasing error (a NaN of either sign counts as the largest), then by
/// increasing index.
pub(crate) fn worst_first(a: &StepRecord, b: &StepRecord) -> Ordering {
    let by_error = match (a.error.is_nan(), b.error.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        // Neither is NaN, so this is never `None`; -0.0 and 0.0 tie.
        (false, false) => b.error.partial_cmp(&a.error).unwrap_or(Ordering::Equal),
    };
    by_error.then(a.index.cmp(&b.index))
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::thread;

    fn step(index: u64, error: f64) -> StepRecord {
        StepRecord {
            index,
            pass: 0,
            subgoal: 0,
            centre: ElementId::new(index as u32),
            parameters: vec![ElementId::new(index as u32)],
            option_indices: vec![0],
            constraints: Vec::new(),
            error,
            evaluations: 1,
            prepare_time: Duration::ZERO,
            error_time: Duration::ZERO,
            duration: Duration::ZERO,
        }
    }

    fn report(steps: &[(u64, f64)]) -> Report {
        Report {
            worst_steps: steps.iter().map(|&(i, e)| step(i, e)).collect(),
            ..Report::default()
        }
    }

    fn indices(r: &Report) -> Vec<u64> {
        r.worst_steps.iter().map(|s| s.index).collect()
    }

    #[test]
    fn absorb_sums_counters_and_times() {
        let mut a = Report {
            passes: 2,
            steps: 10,
            evaluations: 100,
            cache_hits: 5,
            cache_misses: 7,
            wall_time: Duration::from_millis(1500),
            prepare_time: Duration::from_millis(300),
            error_time: Duration::from_millis(700),
            skipped: vec![ElementId::new(4)],
            worst_steps: vec![],
        };
        let b = Report {
            passes: 3,
            steps: 20,
            evaluations: 250,
            cache_hits: 1,
            cache_misses: 2,
            wall_time: Duration::from_millis(500),
            prepare_time: Duration::from_millis(100),
            error_time: Duration::from_millis(50),
            skipped: vec![ElementId::new(9), ElementId::new(1)],
            worst_steps: vec![],
        };
        a.absorb(&b);
        assert_eq!(a.passes, 5);
        assert_eq!(a.steps, 30);
        assert_eq!(a.evaluations, 350);
        assert_eq!((a.cache_hits, a.cache_misses), (6, 9));
        assert_eq!(a.wall_time, Duration::from_millis(2000));
        assert_eq!(a.prepare_time, Duration::from_millis(400));
        assert_eq!(a.error_time, Duration::from_millis(750));
        // Skipped elements are merged ascending and without duplicates.
        assert_eq!(
            a.skipped,
            vec![ElementId::new(1), ElementId::new(4), ElementId::new(9)]
        );
        let mut again = a.clone();
        again.absorb(&b);
        assert_eq!(
            again.skipped, a.skipped,
            "absorbing the same skips twice adds nothing"
        );
        assert_eq!(a.cache_hit_rate(), Some(6.0 / 15.0));
        // `other` is untouched.
        assert_eq!(b.steps, 20);
        assert_eq!(b.skipped.len(), 2);
    }

    #[test]
    fn absorb_into_and_from_default() {
        let src = Report {
            passes: 1,
            steps: 3,
            evaluations: 9,
            worst_steps: vec![step(2, 0.5), step(0, 0.1)],
            ..Report::default()
        };
        let mut empty = Report::default();
        empty.absorb(&src);
        assert_eq!(empty.steps, 3);
        assert_eq!(indices(&empty), vec![2, 0]);
        let mut again = src.clone();
        again.absorb(&Report::default());
        assert_eq!(again.steps, 3);
        assert_eq!(indices(&again), vec![2, 0]);
        assert_eq!(empty.cache_hit_rate(), None);
    }

    #[test]
    fn absorb_merges_worst_steps_descending() {
        let mut a = report(&[(0, 0.9), (1, 0.5), (2, 0.1)]);
        let b = report(&[(10, 0.7), (11, 0.95), (12, 0.0)]);
        a.absorb(&b);
        assert_eq!(indices(&a), vec![11, 0, 10, 1, 2, 12]);
    }

    #[test]
    fn absorb_keeps_the_ten_largest() {
        let a_steps: Vec<(u64, f64)> = (0..10).map(|i| (i, i as f64)).collect();
        let b_steps: Vec<(u64, f64)> = (10..20).map(|i| (i, i as f64 - 5.0)).collect();
        let mut a = report(&a_steps);
        let b = report(&b_steps);
        a.absorb(&b);
        assert_eq!(a.worst_steps.len(), 10);
        let errors: Vec<f64> = a.worst_steps.iter().map(|s| s.error).collect();
        assert_eq!(
            errors,
            vec![14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 9.0, 8.0, 8.0, 7.0]
        );
        // Errors of 9 (steps 9 and 14) and 8 (steps 8 and 13): ties go to the lower index.
        assert_eq!(indices(&a), vec![19, 18, 17, 16, 15, 9, 14, 8, 13, 7]);
    }

    #[test]
    fn absorb_breaks_ties_by_lower_index() {
        let mut a = report(&[(7, 1.0), (3, 1.0)]);
        let b = report(&[(5, 1.0), (1, 1.0), (9, 2.0)]);
        a.absorb(&b);
        assert_eq!(indices(&a), vec![9, 1, 3, 5, 7]);
    }

    #[test]
    fn absorb_truncates_an_oversized_self() {
        let steps: Vec<(u64, f64)> = (0..15).map(|i| (i, i as f64)).collect();
        let mut a = report(&steps);
        a.absorb(&Report::default());
        assert_eq!(a.worst_steps.len(), MAX_WORST_STEPS);
        assert_eq!(a.worst_steps[0].index, 14);
        assert_eq!(a.worst_steps[9].index, 5);
    }

    #[test]
    fn absorb_orders_infinite_and_nan_errors_first() {
        let mut a = report(&[(0, 1.0), (1, f64::INFINITY)]);
        let b = report(&[(2, f64::NAN), (3, f64::INFINITY), (4, f64::NEG_INFINITY)]);
        a.absorb(&b);
        // NaN counts as the largest (it signals a bug and must stay visible).
        assert_eq!(indices(&a), vec![2, 1, 3, 0, 4]);
    }

    #[test]
    fn absorbing_a_clone_of_itself_doubles_counters() {
        let mut a = Report {
            passes: 1,
            steps: 4,
            evaluations: 8,
            cache_hits: 3,
            cache_misses: 1,
            wall_time: Duration::from_secs(2),
            ..report(&[(0, 3.0), (1, 2.0)])
        };
        let copy = a.clone();
        a.absorb(&copy);
        assert_eq!((a.passes, a.steps, a.evaluations), (2, 8, 16));
        assert_eq!(a.wall_time, Duration::from_secs(4));
        assert_eq!(indices(&a), vec![0, 0, 1, 1]);
    }

    #[test]
    fn absorb_saturates_instead_of_overflowing() {
        let mut a = Report {
            passes: u32::MAX,
            steps: u64::MAX,
            wall_time: Duration::MAX,
            ..Report::default()
        };
        let b = Report {
            passes: 1,
            steps: 1,
            wall_time: Duration::from_secs(1),
            ..Report::default()
        };
        a.absorb(&b);
        assert_eq!(a.passes, u32::MAX);
        assert_eq!(a.steps, u64::MAX);
        assert_eq!(a.wall_time, Duration::MAX);
    }

    proptest! {
        #[test]
        fn absorb_property(
            xs in prop::collection::vec((0u64..40, 0u32..6), 0..25),
            ys in prop::collection::vec((0u64..40, 0u32..6), 0..25),
            p in 0u32..100,
            q in 0u32..100,
        ) {
            let to_report = |v: &[(u64, u32)], passes: u32| {
                let mut r = report(&v.iter().map(|&(i, e)| (i, e as f64)).collect::<Vec<_>>());
                r.worst_steps.sort_by(worst_first);
                r.worst_steps.truncate(MAX_WORST_STEPS);
                r.passes = passes;
                r.steps = v.len() as u64;
                r
            };
            let mut a = to_report(&xs, p);
            let b = to_report(&ys, q);
            let mut all: Vec<(u64, f64)> = a
                .worst_steps
                .iter()
                .chain(&b.worst_steps)
                .map(|s| (s.index, s.error))
                .collect();
            a.absorb(&b);
            prop_assert_eq!(a.passes, p + q);
            prop_assert_eq!(a.steps, (xs.len() + ys.len()) as u64);
            prop_assert!(a.worst_steps.len() <= MAX_WORST_STEPS);
            // Reference: sort by (error desc, index asc) and keep ten.
            all.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap().then(x.0.cmp(&y.0)));
            all.truncate(MAX_WORST_STEPS);
            let got: Vec<(u64, f64)> = a.worst_steps.iter().map(|s| (s.index, s.error)).collect();
            prop_assert_eq!(got, all);
        }
    }

    #[test]
    fn absorb_keeps_nan_of_either_sign_as_the_worst() {
        // The sign bit of a NaN is arbitrary (0.0 / 0.0 is negative on x86).
        for nan in [f64::NAN, -f64::NAN, f64::from_bits(0xfff8_0000_0000_0001)] {
            let mut a = report(&[(1, 1.0), (2, f64::INFINITY)]);
            a.absorb(&report(&[(3, nan)]));
            assert_eq!(indices(&a), vec![3, 2, 1], "NaN {:#x}", nan.to_bits());

            // Truncation must not drop it either.
            let mut full = report(&(0..10).map(|i| (i, i as f64)).collect::<Vec<_>>());
            full.absorb(&report(&[(50, nan), (51, -f64::INFINITY)]));
            assert_eq!(full.worst_steps.len(), MAX_WORST_STEPS);
            assert_eq!(full.worst_steps[0].index, 50);
        }
        // -0.0 and 0.0 tie, so the lower index comes first.
        let mut z = report(&[(8, 0.0)]);
        z.absorb(&report(&[(2, -0.0)]));
        assert_eq!(indices(&z), vec![2, 8]);
        // Two NaNs tie: the lower index first, whichever report they come from.
        let mut a = report(&[(9, f64::NAN)]);
        a.absorb(&report(&[(4, -f64::NAN)]));
        assert_eq!(indices(&a), vec![4, 9]);
        let mut a = report(&[(4, -f64::NAN)]);
        a.absorb(&report(&[(9, f64::NAN)]));
        assert_eq!(indices(&a), vec![4, 9]);
        let mut a = report(&[(4, f64::NAN), (9, f64::NAN)]);
        a.absorb(&report(&[(6, -f64::NAN), (1, f64::NAN)]));
        assert_eq!(indices(&a), vec![1, 4, 6, 9]);
    }

    fn passed(pass: u32) -> Event {
        Event::PassFinished {
            pass,
            calibrated: pass as usize,
        }
    }

    #[test]
    fn closures_are_observers() {
        let count = AtomicUsize::new(0);
        let observer = |event: &Event| {
            if matches!(event, Event::PassFinished { .. }) {
                count.fetch_add(1, AtomicOrdering::SeqCst);
            }
        };
        observer.on_event(&passed(0));
        observer.on_event(&Event::StepFailed {
            centre: ElementId::new(1),
            error: "x".into(),
        });
        observer.on_event(&passed(1));
        assert_eq!(count.load(AtomicOrdering::SeqCst), 2);
    }

    #[test]
    fn observers_work_behind_dyn_and_arc() {
        let recorder = RecordingObserver::new();
        let shared: Arc<dyn Observer> = Arc::new(recorder.clone());
        shared.on_event(&passed(3));
        let boxed: Box<dyn Observer> = Box::new(shared.clone());
        boxed.on_event(&passed(4));
        let plain = Arc::new(recorder.clone());
        plain.on_event(&passed(5));
        let events = recorder.events();
        assert_eq!(events.len(), 3);
        assert!(matches!(events[2], Event::PassFinished { pass: 5, .. }));
    }

    #[test]
    fn recording_observer_clones_share_one_buffer() {
        let a = RecordingObserver::new();
        let b = a.clone();
        assert!(a.events().is_empty());
        a.on_event(&passed(0));
        b.on_event(&Event::StepCommitted(step(7, 0.5)));
        b.on_event(&Event::Finished(report(&[(1, 2.0)])));
        let events = a.events();
        assert_eq!(events.len(), 3);
        assert!(matches!(events[0], Event::PassFinished { pass: 0, .. }));
        match &events[1] {
            Event::StepCommitted(s) => assert_eq!(s.index, 7),
            e => panic!("unexpected {e:?}"),
        }
        match &events[2] {
            Event::Finished(r) => assert_eq!(r.worst_steps[0].index, 1),
            e => panic!("unexpected {e:?}"),
        }
        // `events()` returns a copy.
        let mut copy = b.events();
        copy.clear();
        assert_eq!(a.events().len(), 3);
    }

    #[test]
    fn recording_observer_from_several_threads() {
        const THREADS: usize = 8;
        const PER_THREAD: usize = 250;
        let recorder = RecordingObserver::new();
        thread::scope(|scope| {
            for t in 0..THREADS {
                let recorder = recorder.clone();
                scope.spawn(move || {
                    for k in 0..PER_THREAD {
                        recorder.on_event(&Event::StepCommitted(step(
                            (t * PER_THREAD + k) as u64,
                            0.0,
                        )));
                    }
                });
            }
        });
        let events = recorder.events();
        assert_eq!(events.len(), THREADS * PER_THREAD);
        // No event lost or duplicated, and each thread's events are in its own order.
        let mut seen: Vec<u64> = Vec::new();
        let mut last_per_thread = [None::<u64>; THREADS];
        for event in &events {
            let Event::StepCommitted(s) = event else {
                panic!("unexpected event {event:?}");
            };
            seen.push(s.index);
            let t = s.index as usize / PER_THREAD;
            if let Some(prev) = last_per_thread[t] {
                assert!(s.index > prev, "thread {t} events reordered");
            }
            last_per_thread[t] = Some(s.index);
        }
        seen.sort_unstable();
        assert_eq!(seen, (0..(THREADS * PER_THREAD) as u64).collect::<Vec<_>>());
    }

    #[test]
    fn closure_observer_from_several_threads() {
        let count = AtomicUsize::new(0);
        let observer = |_: &Event| {
            count.fetch_add(1, AtomicOrdering::SeqCst);
        };
        thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    for k in 0..100 {
                        observer.on_event(&passed(k));
                    }
                });
            }
        });
        assert_eq!(count.load(AtomicOrdering::SeqCst), 400);
    }
}
