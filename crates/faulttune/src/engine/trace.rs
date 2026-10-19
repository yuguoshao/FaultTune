//! Tracing spans for `calibrate`, `pass`, `subgoal` and `step` (ARCHITECTURE §11).
//!
//! With the `tracing` feature the functions create real [`tracing`] spans. Without it they are
//! empty, so the engine code calls them unconditionally and the whole module compiles to
//! nothing.

#[cfg(feature = "tracing")]
mod imp {
    /// A handle to a span of the calibration run.
    #[derive(Clone)]
    pub(crate) struct Span(tracing::Span);

    /// Keeps a span entered while it is alive.
    pub(crate) type Entered<'a> = tracing::span::Entered<'a>;

    impl Span {
        /// The span of a whole `calibrate_*` call.
        pub(crate) fn calibrate(elements: usize, goal: usize, region: usize, seed: u64) -> Self {
            Self(tracing::info_span!(
                "calibrate",
                elements = elements as u64,
                goal = goal as u64,
                region = region as u64,
                seed
            ))
        }

        /// The span of one pass of the completion loop.
        pub(crate) fn pass(parent: &Span, pass: u32, subgoals: usize, uncalibrated: usize) -> Self {
            Self(tracing::info_span!(
                parent: &parent.0,
                "pass",
                pass,
                subgoals = subgoals as u64,
                uncalibrated = uncalibrated as u64
            ))
        }

        /// The span of one subgoal within a pass.
        pub(crate) fn subgoal(
            parent: &Span,
            index: usize,
            elements: usize,
            threads: usize,
        ) -> Self {
            Self(tracing::info_span!(
                parent: &parent.0,
                "subgoal",
                index = index as u64,
                elements = elements as u64,
                threads = threads as u64
            ))
        }

        /// The span of one calibration step. `evaluations` and `error` are filled in by
        /// [`record_outcome`](Self::record_outcome).
        pub(crate) fn step(
            parent: &Span,
            index: u64,
            centre: &str,
            parameters: usize,
            constraints: usize,
        ) -> Self {
            Self(tracing::debug_span!(
                parent: &parent.0,
                "step",
                index,
                centre,
                parameters = parameters as u64,
                constraints = constraints as u64,
                evaluations = tracing::field::Empty,
                error = tracing::field::Empty
            ))
        }

        /// Records the result of a step on its span.
        pub(crate) fn record_outcome(&self, evaluations: u64, error: f64) {
            self.0.record("evaluations", evaluations);
            self.0.record("error", error);
        }

        /// Enters the span until the returned guard is dropped.
        pub(crate) fn enter(&self) -> Entered<'_> {
            self.0.enter()
        }
    }
}

#[cfg(not(feature = "tracing"))]
mod imp {
    /// A stand-in for a span handle; every operation does nothing.
    #[derive(Clone)]
    pub(crate) struct Span;

    /// The guard returned by [`Span::enter`].
    pub(crate) type Entered<'a> = std::marker::PhantomData<&'a ()>;

    impl Span {
        pub(crate) fn calibrate(_: usize, _: usize, _: usize, _: u64) -> Self {
            Self
        }

        pub(crate) fn pass(_: &Span, _: u32, _: usize, _: usize) -> Self {
            Self
        }

        pub(crate) fn subgoal(_: &Span, _: usize, _: usize, _: usize) -> Self {
            Self
        }

        pub(crate) fn step(_: &Span, _: u64, _: &str, _: usize, _: usize) -> Self {
            Self
        }

        pub(crate) fn record_outcome(&self, _: u64, _: f64) {}

        pub(crate) fn enter(&self) -> Entered<'_> {
            std::marker::PhantomData
        }
    }
}

pub(crate) use imp::Span;
