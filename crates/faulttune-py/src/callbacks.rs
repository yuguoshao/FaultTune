//! User code that steers the engine: traversal heuristics, step optimizers and seed policies
//! written in Python. They run on engine threads with the GIL attached for the duration of
//! each call.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use faulttune::optimizers::{CoordinateDescent, Exhaustive, RandomSearch};
use faulttune::{
    CancellationToken, ElementId, FaultTuneError, FrontierKind, IndexBatch, Objective, Optimizer,
    SearchSpace, SeedPolicy, Solution, Thread, TraversalContext, TraversalHeuristic,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::gc::PyVisit;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::PyTraverseError;
use rand::rngs::StdRng;
use rand::Rng;

use crate::errors::{describe, Failure, FailureSlot};
use crate::graph::{
    elements_list, extract_element, new_element, new_set, PyElement, PyElementSet, PyGraph,
};
use crate::npy::{default_rng, format_shape, to_f64_array, to_i64_array};
use crate::records::PyStepRecord;
use crate::scoped::{ScopedContext, ScopedObjective};

use numpy::PyArray1;

/// A Python object that a part of the engine (a heuristic, an optimizer, a seed policy) calls.
///
/// The engine parts live inside Rust types that the garbage collector cannot look into, so a
/// callback that refers back to its `Snake` (a stateful heuristic that keeps the Snake, say)
/// would never be collected. Every `Held` is therefore also registered with the `Snake`
/// object, which shows it to the collector (`__traverse__`) and lets go of it when the
/// collector breaks a cycle (`__clear__`).
#[derive(Clone)]
pub struct Held(Arc<Mutex<Option<Py<PyAny>>>>);

impl Held {
    /// Holds `object`.
    pub fn new(object: Py<PyAny>) -> Self {
        Self(Arc::new(Mutex::new(Some(object))))
    }

    /// The object (`RuntimeError` after the garbage collector cleared the Snake).
    pub fn get<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &*self.0.lock().unwrap_or_else(PoisonError::into_inner) {
            Some(object) => Ok(object.bind(py).clone()),
            None => Err(PyRuntimeError::new_err(
                "the Snake was cleared by the garbage collector",
            )),
        }
    }

    /// Shows the object to the garbage collector.
    pub fn visit(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        let guard = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        visit.call(&*guard)
    }

    /// Lets go of the object (used by the garbage collector to break a cycle).
    pub fn clear(&self) {
        let taken = self.0.lock().unwrap_or_else(PoisonError::into_inner).take();
        drop(taken);
    }
}

/// What every callback needs to report a failure and to stop the run.
#[derive(Clone)]
pub struct Reporter {
    pub graph: Arc<Py<PyGraph>>,
    pub failures: Arc<FailureSlot>,
    pub cancel: CancellationToken,
}

impl Reporter {
    /// Remembers the failure of user code and tells the engine to stop.
    fn fail(&self, py: Python<'_>, what: &str, element: Option<ElementId>, err: PyErr) {
        self.failures.record(Failure {
            element,
            message: format!("{what} raised {}", describe(py, &err)),
            cause: Some(err),
        });
        self.cancel.cancel();
    }

    fn arc(&self) -> Arc<faulttune::ProcessorGraph> {
        Arc::clone(&self.graph.get().inner)
    }
}

// ---------------------------------------------------------------------------------------
// Traversal heuristics
// ---------------------------------------------------------------------------------------

/// What a heuristic sees: the graph, the calibration status and the step history.
///
/// It is only valid during the heuristic call it is passed to. Using it later raises
/// `RuntimeError`.
#[pyclass(frozen, module = "faulttune", name = "TraversalContext")]
pub struct PyTraversalContext {
    graph: Py<PyGraph>,
    scoped: ScopedContext,
}

const EXPIRED: &str = "the TraversalContext can only be used during the heuristic call it was \
                       passed to";

impl PyTraversalContext {
    fn with<R>(&self, f: impl FnOnce(&TraversalContext<'_>) -> R) -> PyResult<R> {
        self.scoped
            .with(f)
            .ok_or_else(|| PyRuntimeError::new_err(EXPIRED))
    }

    fn arc(&self) -> Arc<faulttune::ProcessorGraph> {
        Arc::clone(&self.graph.get().inner)
    }
}

#[pymethods]
impl PyTraversalContext {
    /// The processor graph.
    #[getter]
    fn graph(&self, py: Python<'_>) -> Py<PyGraph> {
        self.graph.clone_ref(py)
    }

    /// The element whose step was just committed.
    #[getter]
    fn current<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyElement>> {
        let id = self.with(|ctx| ctx.current)?;
        new_element(py, &self.arc(), id)
    }

    /// Whether `element` is calibrated (in the database so far, or by this subgoal's steps).
    fn is_calibrated(&self, element: &Bound<'_, PyAny>) -> PyResult<bool> {
        let id = extract_element(&self.arc(), element, "element")?;
        self.with(|ctx| ctx.calibrated.contains(id))
    }

    /// Number of calibrated elements.
    #[getter]
    fn calibrated_count(&self) -> PyResult<usize> {
        self.with(|ctx| ctx.calibrated.len())
    }

    /// The calibrated elements as an `ElementSet`.
    #[getter]
    fn calibrated<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyElementSet>> {
        let set = self.with(|ctx| ctx.calibrated.clone())?;
        new_set(py, &self.arc(), set)
    }

    /// The steps committed so far by the current subgoal in this pass, in commit order.
    /// Built on demand; the cost is only paid when this method is called.
    fn history<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let records = self.with(|ctx| ctx.history.to_vec())?;
        let graph = self.arc();
        let list = PyList::empty(py);
        for record in records {
            list.append(Bound::new(py, PyStepRecord::new(&graph, record, None))?)?;
        }
        Ok(list)
    }
}

/// A traversal heuristic written in Python: `callable(context, options) -> sequence of
/// Elements`, with an optional `frontier` attribute and `priority(context, element)` method.
pub struct PyHeuristic {
    callable: Held,
    frontier: FrontierKind,
    has_priority: bool,
    reporter: Reporter,
}

impl PyHeuristic {
    /// Wraps `obj`; `ValueError`/`TypeError` if it does not follow the protocol.
    pub fn new(obj: &Bound<'_, PyAny>, reporter: Reporter) -> PyResult<Self> {
        if !obj.is_callable() {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "heuristic must be \"breadth_first\", \"depth_first\", \"random\" or a callable \
                 taking (context, options)",
            ));
        }
        let frontier = match obj.getattr("frontier") {
            Err(_) => FrontierKind::Fifo,
            Ok(f) => match f.extract::<String>().as_deref() {
                Ok("fifo") => FrontierKind::Fifo,
                Ok("lifo") => FrontierKind::Lifo,
                Ok("priority") => FrontierKind::Priority,
                _ => {
                    return Err(PyValueError::new_err(
                        "heuristic.frontier must be \"fifo\", \"lifo\" or \"priority\"",
                    ))
                }
            },
        };
        let has_priority = obj.getattr("priority").is_ok_and(|p| p.is_callable());
        if frontier == FrontierKind::Priority && !has_priority {
            return Err(PyValueError::new_err(
                "a heuristic with frontier=\"priority\" needs a priority(context, element) method",
            ));
        }
        Ok(Self {
            callable: Held::new(obj.clone().unbind()),
            frontier,
            has_priority,
            reporter,
        })
    }

    /// The held callable, to register it with the `Snake` that owns this heuristic.
    pub fn held(&self) -> Held {
        self.callable.clone()
    }

    fn make_context<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTraversalContext>> {
        Bound::new(
            py,
            PyTraversalContext {
                graph: self.reporter.graph.clone_ref(py),
                scoped: ScopedContext::new(),
            },
        )
    }

    fn reorder(
        &self,
        py: Python<'_>,
        ctx: &TraversalContext<'_>,
        options: &[ElementId],
    ) -> PyResult<Vec<ElementId>> {
        let graph = self.reporter.arc();
        let pctx = self.make_context(py)?;
        let offered = elements_list(py, &graph, options.iter().copied())?;
        let returned = pctx.get().scoped.scope(ctx, || {
            self.callable
                .get(py)
                .and_then(|callable| callable.call1((&pctx, &offered)))
        })?;
        let allowed: HashSet<ElementId> = options.iter().copied().collect();
        let mut ordered = Vec::with_capacity(options.len());
        for item in returned.try_iter()? {
            let id = extract_element(&graph, &item?, "an item returned by the heuristic")?;
            if !allowed.contains(&id) {
                return Err(PyValueError::new_err(format!(
                    "the heuristic returned element {:?}, which was not among the options it \
                     was given",
                    graph.label(id)
                )));
            }
            ordered.push(id);
        }
        Ok(ordered)
    }
}

impl TraversalHeuristic for PyHeuristic {
    fn frontier(&self) -> FrontierKind {
        self.frontier
    }

    fn order(&self, ctx: &TraversalContext<'_>, options: &mut Vec<ElementId>, _rng: &mut StdRng) {
        if self.reporter.failures.is_set() {
            return;
        }
        Python::attach(|py| match self.reorder(py, ctx, options) {
            Ok(ordered) => *options = ordered,
            Err(err) => self
                .reporter
                .fail(py, "the traversal heuristic", Some(ctx.current), err),
        })
    }

    fn priority(&self, ctx: &TraversalContext<'_>, element: ElementId) -> f64 {
        if !self.has_priority || self.reporter.failures.is_set() {
            return 0.0;
        }
        Python::attach(|py| {
            let call = || -> PyResult<f64> {
                let graph = self.reporter.arc();
                let pctx = self.make_context(py)?;
                let handle = new_element(py, &graph, element)?;
                let score = pctx.get().scoped.scope(ctx, || {
                    self.callable
                        .get(py)
                        .and_then(|callable| callable.call_method1("priority", (&pctx, handle)))
                })?;
                score.extract::<f64>()
            };
            match call() {
                Ok(score) => score,
                Err(err) => {
                    self.reporter
                        .fail(py, "the heuristic's priority()", Some(ctx.current), err);
                    0.0
                }
            }
        })
    }
}

// ---------------------------------------------------------------------------------------
// Seed policies
// ---------------------------------------------------------------------------------------

/// A seed policy from a Python callable `(thread_elements) -> Element`.
pub fn callable_seed_policy(callable: Held, reporter: Reporter) -> SeedPolicy {
    SeedPolicy::Custom(Arc::new(move |thread: &Thread, _rng: &mut StdRng| {
        let lowest = thread
            .elements
            .iter()
            .copied()
            .min()
            .unwrap_or(ElementId::new(0));
        if reporter.failures.is_set() {
            return lowest;
        }
        Python::attach(|py| {
            let graph = reporter.arc();
            let call = || -> PyResult<ElementId> {
                let members = elements_list(py, &graph, thread.elements.iter().copied())?;
                let seed = callable.get(py)?.call1((members,))?;
                let id = extract_element(&graph, &seed, "the seed returned by the seed policy")?;
                if !thread.elements.contains(&id) {
                    return Err(PyValueError::new_err(format!(
                        "the seed policy returned element {:?}, which is not in the thread it \
                         was asked about",
                        graph.label(id)
                    )));
                }
                Ok(id)
            };
            match call() {
                Ok(id) => id,
                Err(err) => {
                    reporter.fail(py, "the seed policy", Some(lowest), err);
                    lowest
                }
            }
        })
    }))
}

// ---------------------------------------------------------------------------------------
// Optimizers
// ---------------------------------------------------------------------------------------

fn positive(name: &str, value: i64) -> PyResult<usize> {
    if value < 1 {
        return Err(PyValueError::new_err(format!(
            "{name} must be at least 1, got {value}"
        )));
    }
    Ok(value as usize)
}

/// Coordinate descent (the default optimizer): sweeps one parameter at a time.
#[pyclass(frozen, module = "faulttune", name = "CoordinateDescent")]
pub struct PyCoordinateDescent {
    pub inner: CoordinateDescent,
}

#[pymethods]
impl PyCoordinateDescent {
    /// `random_starts` random points are evaluated first, then at most `max_sweeps` sweeps
    /// follow; `max_start_attempts` more random points are tried if every start is forbidden.
    #[new]
    #[pyo3(signature = (random_starts=4, max_sweeps=3, max_start_attempts=20_000))]
    fn new(random_starts: i64, max_sweeps: i64, max_start_attempts: i64) -> PyResult<Self> {
        if max_start_attempts < 0 {
            return Err(PyValueError::new_err(format!(
                "max_start_attempts must be at least 0, got {max_start_attempts}"
            )));
        }
        Ok(Self {
            inner: CoordinateDescent {
                random_starts: positive("random_starts", random_starts)?,
                max_sweeps: positive("max_sweeps", max_sweeps)?,
                max_start_attempts: max_start_attempts as usize,
            },
        })
    }

    /// Random starting points evaluated first.
    #[getter]
    fn random_starts(&self) -> usize {
        self.inner.random_starts
    }

    /// Maximum number of sweeps over all parameters.
    #[getter]
    fn max_sweeps(&self) -> usize {
        self.inner.max_sweeps
    }

    /// Extra random points tried when every start is forbidden.
    #[getter]
    fn max_start_attempts(&self) -> usize {
        self.inner.max_start_attempts
    }

    fn __repr__(&self) -> String {
        format!(
            "CoordinateDescent(random_starts={}, max_sweeps={}, max_start_attempts={})",
            self.inner.random_starts, self.inner.max_sweeps, self.inner.max_start_attempts
        )
    }
}

/// Exhaustive search: tries every combination, in batches.
#[pyclass(frozen, module = "faulttune", name = "Exhaustive")]
pub struct PyExhaustive {
    pub inner: Exhaustive,
}

#[pymethods]
impl PyExhaustive {
    /// Refuses with `SearchSpaceTooLargeError` if a step has more than `max_size`
    /// combinations (or more than the evaluation budget).
    #[new]
    #[pyo3(signature = (batch_size=4096, max_size=1_000_000))]
    fn new(batch_size: i64, max_size: i64) -> PyResult<Self> {
        Ok(Self {
            inner: Exhaustive {
                batch_size: positive("batch_size", batch_size)?,
                max_size: positive("max_size", max_size)? as u64,
            },
        })
    }

    /// Candidates per objective call.
    #[getter]
    fn batch_size(&self) -> usize {
        self.inner.batch_size
    }

    /// Largest search space that is searched.
    #[getter]
    fn max_size(&self) -> u64 {
        self.inner.max_size
    }

    fn __repr__(&self) -> String {
        format!(
            "Exhaustive(batch_size={}, max_size={})",
            self.inner.batch_size, self.inner.max_size
        )
    }
}

/// Random search: samples a fixed number of candidates.
#[pyclass(frozen, module = "faulttune", name = "RandomSearch")]
pub struct PyRandomSearch {
    pub inner: RandomSearch,
}

#[pymethods]
impl PyRandomSearch {
    /// Evaluates `samples` random candidates, `batch_size` at a time.
    #[new]
    #[pyo3(signature = (samples=1000, batch_size=1000))]
    fn new(samples: i64, batch_size: i64) -> PyResult<Self> {
        Ok(Self {
            inner: RandomSearch {
                samples: positive("samples", samples)?,
                batch_size: positive("batch_size", batch_size)?,
            },
        })
    }

    /// Number of candidates sampled.
    #[getter]
    fn samples(&self) -> usize {
        self.inner.samples
    }

    /// Candidates per objective call.
    #[getter]
    fn batch_size(&self) -> usize {
        self.inner.batch_size
    }

    fn __repr__(&self) -> String {
        format!(
            "RandomSearch(samples={}, batch_size={})",
            self.inner.samples, self.inner.batch_size
        )
    }
}

/// The search space of one step, as handed to a Python optimizer.
#[pyclass(frozen, module = "faulttune", name = "SearchSpace")]
pub struct PySearchSpace {
    sizes: Vec<u32>,
    budget: u64,
}

#[pymethods]
impl PySearchSpace {
    /// Number of options of each parameter, in candidate-column order.
    #[getter]
    fn sizes(&self) -> Vec<u32> {
        self.sizes.clone()
    }

    /// The evaluation budget: the most candidates a budget-aware optimizer should evaluate.
    #[getter]
    fn budget(&self) -> u64 {
        self.budget
    }

    /// Number of parameters.
    #[getter]
    fn width(&self) -> usize {
        self.sizes.len()
    }

    /// Number of combinations (the product of `sizes`).
    #[getter]
    fn size(&self) -> u128 {
        self.sizes
            .iter()
            .fold(1u128, |acc, &k| acc.saturating_mul(u128::from(k)))
    }

    fn __repr__(&self) -> String {
        format!(
            "SearchSpace(sizes={:?}, budget={})",
            self.sizes, self.budget
        )
    }
}

/// The objective handed to a Python optimizer: `objective(indices)` takes an integer array of
/// shape `(n, width)` of option indices and returns the `(n,)` float64 errors. It goes through
/// the engine's de-duplication and cache and counts evaluations. Only valid during the call.
#[pyclass(frozen, module = "faulttune", name = "Objective")]
pub struct PyObjective {
    scoped: ScopedObjective,
    sizes: Vec<u32>,
    requested: AtomicU64,
    failure: Mutex<Option<FaultTuneError>>,
}

#[pymethods]
impl PyObjective {
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        indices: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let width = self.sizes.len();
        let (shape, flat) = to_i64_array(py, indices)?;
        let rows = match shape.as_slice() {
            [n, w] if *w == width => *n,
            [w] if *w == width => 1,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "indices must have shape (n, {width}), but has shape {}",
                    format_shape(&shape)
                )))
            }
        };
        let mut batch = Vec::with_capacity(flat.len());
        for (i, &k) in flat.iter().enumerate() {
            let size = self.sizes[i % width.max(1)];
            if k < 0 || k >= i64::from(size) {
                return Err(PyValueError::new_err(format!(
                    "index {k} is out of range for parameter {} with {size} options",
                    i % width.max(1)
                )));
            }
            batch.push(k as u32);
        }
        let mut errors = vec![0.0f64; rows];
        let outcome = self
            .scoped
            .with(|objective| objective.evaluate(IndexBatch::new(&batch, width), &mut errors));
        match outcome {
            None => Err(PyRuntimeError::new_err(
                "the objective can only be used during the optimizer call it was passed to, \
                 and not from inside the error model",
            )),
            Some(Err(err)) => {
                let message = err.to_string();
                *self.failure.lock().unwrap_or_else(PoisonError::into_inner) = Some(err);
                Err(crate::errors::generic(py, &message))
            }
            Some(Ok(())) => {
                self.requested.fetch_add(rows as u64, Ordering::Relaxed);
                Ok(PyArray1::from_vec(py, errors))
            }
        }
    }

    /// The option counts of the step's parameters.
    #[getter]
    fn sizes(&self) -> Vec<u32> {
        self.sizes.clone()
    }
}

/// A step optimizer written in Python: `optimizer(space, objective, rng) -> (indices, error)`.
pub struct PyOptimizer {
    callable: Held,
    reporter: Reporter,
}

impl PyOptimizer {
    /// Wraps a callable.
    pub fn new(callable: Held, reporter: Reporter) -> Self {
        Self { callable, reporter }
    }

    fn parse(
        py: Python<'_>,
        result: &Bound<'_, PyAny>,
        sizes: &[u32],
    ) -> PyResult<(Vec<u32>, f64)> {
        let (indices, error) = result
            .extract::<(Bound<'_, PyAny>, Bound<'_, PyAny>)>()
            .map_err(|_| {
                PyValueError::new_err("the optimizer must return a pair (indices, error)")
            })?;
        let (shape, values) = to_i64_array(py, &indices)?;
        if shape != [sizes.len()] {
            return Err(PyValueError::new_err(format!(
                "the optimizer must return indices of shape ({},), but returned shape {}",
                sizes.len(),
                format_shape(&shape)
            )));
        }
        let mut out = Vec::with_capacity(values.len());
        for (j, &k) in values.iter().enumerate() {
            if k < 0 || k >= i64::from(sizes[j]) {
                return Err(PyValueError::new_err(format!(
                    "the optimizer returned index {k} for parameter {j}, which has {} options",
                    sizes[j]
                )));
            }
            out.push(k as u32);
        }
        let error = error
            .extract::<f64>()
            .or_else(|_| {
                to_f64_array(py, &error).and_then(|(shape, v)| {
                    if shape.is_empty() {
                        Ok(v[0])
                    } else {
                        Err(PyValueError::new_err("the error must be a number"))
                    }
                })
            })
            .map_err(|_| PyValueError::new_err("the optimizer must return a numeric error"))?;
        Ok((out, error))
    }
}

impl Optimizer for PyOptimizer {
    fn optimize(
        &self,
        space: &SearchSpace,
        objective: &mut dyn Objective,
        rng: &mut StdRng,
    ) -> Result<Solution, FaultTuneError> {
        let seed: u64 = rng.random();
        Python::attach(|py| {
            let invalid = |message: String| FaultTuneError::InvalidConfig(message);
            let prepared = (|| -> PyResult<_> {
                let rng = default_rng(py, seed)?;
                let pspace = Bound::new(
                    py,
                    PySearchSpace {
                        sizes: space.sizes.clone(),
                        budget: space.budget,
                    },
                )?;
                let pobj = Bound::new(
                    py,
                    PyObjective {
                        scoped: ScopedObjective::new(),
                        sizes: space.sizes.clone(),
                        requested: AtomicU64::new(0),
                        failure: Mutex::new(None),
                    },
                )?;
                Ok((rng, pspace, pobj))
            })();
            let (rng, pspace, pobj) = match prepared {
                Ok(parts) => parts,
                Err(err) => {
                    self.reporter.fail(py, "the optimizer setup", None, err);
                    return Err(invalid("the Python optimizer could not be started".into()));
                }
            };
            let called = pobj.get().scoped.scope(objective, || {
                self.callable
                    .get(py)
                    .and_then(|callable| callable.call1((&pspace, &pobj, rng)))
            });
            // A failure of the objective (the error model failed, NaN, ...) is the engine's
            // own error: give it back unchanged, whatever the optimizer did with the exception.
            let failure = pobj
                .get()
                .failure
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .take();
            if let Some(err) = failure {
                return Err(err);
            }
            let evaluations = pobj.get().requested.load(Ordering::Relaxed);
            let result = match called {
                Ok(result) => result,
                Err(err) => {
                    self.reporter.fail(py, "the optimizer", None, err);
                    return Err(invalid("the Python optimizer raised an exception".into()));
                }
            };
            match Self::parse(py, &result, &space.sizes) {
                Ok((indices, error)) => Ok(Solution {
                    indices,
                    error,
                    evaluations,
                }),
                Err(err) => Err(invalid(format!("invalid result of the optimizer: {err}"))),
            }
        })
    }
}
