//! The `Snake` class: configuration parsing and the calibration entry points.

use std::path::PathBuf;
use std::sync::Arc;

use faulttune::heuristics::{BreadthFirst, DepthFirst, Random};
use faulttune::{
    CachePolicy, CancellationToken, ElementSet, ModelErrorPolicy, SeedPolicy, Snake, SnakeConfig,
    TraversalHeuristic,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyInt, PyString, PyTuple};

use crate::algorithm::{extract_regions, PyAlgorithmSubgraphs, PyRetry, PyStitchPlan};
use crate::callbacks::{
    callable_seed_policy, Held, PyCoordinateDescent, PyExhaustive, PyHeuristic, PyOptimizer,
    PyRandomSearch, Reporter,
};
use crate::database::PyDatabase;
use crate::errors::{self, FailureSlot};
use crate::graph::{
    extract_element, extract_element_list, extract_set, new_set, PyElement, PyElementSet, PyGraph,
};
use crate::records::PyReport;
use crate::run::{Dispatch, Engine, Job, Request};

fn type_name(obj: &Bound<'_, PyAny>) -> String {
    obj.get_type()
        .name()
        .map(|n| n.to_string())
        .unwrap_or_else(|_| "object".into())
}

/// An integer argument (`bool` is refused), or `default` when it is `None`/missing.
fn int_arg(name: &str, value: Option<&Bound<'_, PyAny>>, default: i64) -> PyResult<i64> {
    match value {
        None => Ok(default),
        Some(v) if v.is_none() => Ok(default),
        Some(v) if v.is_instance_of::<PyBool>() => Err(PyTypeError::new_err(format!(
            "{name} must be an integer, not bool"
        ))),
        Some(v) => v.extract::<i64>().map_err(|_| {
            if v.is_instance_of::<PyInt>() {
                PyValueError::new_err(format!("{name} is out of range: {v}"))
            } else {
                PyTypeError::new_err(format!("{name} must be an integer, not {}", type_name(v)))
            }
        }),
    }
}

fn distance_arg(name: &str, value: Option<&Bound<'_, PyAny>>, default: i64) -> PyResult<u32> {
    let v = int_arg(name, value, default)?;
    if v < 0 {
        return Err(PyValueError::new_err(format!(
            "{name} must be non-negative, got {v}"
        )));
    }
    u32::try_from(v).map_err(|_| PyValueError::new_err(format!("{name} is too large: {v}")))
}

/// `on_model_error`: "abort", "skip", ("retry", n) or `Retry(n)`.
fn parse_policy(value: Option<&Bound<'_, PyAny>>) -> PyResult<ModelErrorPolicy> {
    let bad = || {
        PyValueError::new_err(
            "on_model_error must be \"abort\", \"skip\", (\"retry\", n) or Retry(n)",
        )
    };
    let Some(v) = value.filter(|v| !v.is_none()) else {
        return Ok(ModelErrorPolicy::Abort);
    };
    if let Ok(s) = v.cast::<PyString>() {
        return match s.to_str()? {
            "abort" => Ok(ModelErrorPolicy::Abort),
            "skip" => Ok(ModelErrorPolicy::SkipElement),
            "retry" => Err(PyValueError::new_err(
                "on_model_error=\"retry\" needs a count: use (\"retry\", n) or Retry(n)",
            )),
            _ => Err(bad()),
        };
    }
    if let Ok(retry) = v.extract::<PyRetry>() {
        return Ok(ModelErrorPolicy::Retry {
            times: retry.times(),
        });
    }
    if let Ok(pair) = v.cast::<PyTuple>() {
        if pair.len() == 2
            && pair
                .get_item(0)?
                .extract::<String>()
                .is_ok_and(|s| s == "retry")
        {
            let times = int_arg("on_model_error retry count", Some(&pair.get_item(1)?), 1)?;
            return Ok(ModelErrorPolicy::Retry {
                times: PyRetry::checked(times)?.times(),
            });
        }
    }
    Err(bad())
}

/// `cache`: "step", "across_steps" or a capacity.
fn parse_cache(value: Option<&Bound<'_, PyAny>>) -> PyResult<CachePolicy> {
    let Some(v) = value.filter(|v| !v.is_none()) else {
        return Ok(CachePolicy::StepOnly);
    };
    if let Ok(s) = v.cast::<PyString>() {
        return match s.to_str()? {
            "step" => Ok(CachePolicy::StepOnly),
            "across_steps" => Ok(CachePolicy::across_steps()),
            _ => Err(PyValueError::new_err(
                "cache must be \"step\", \"across_steps\" or a capacity (an integer)",
            )),
        };
    }
    let capacity = int_arg("cache", Some(v), 0).map_err(|e| {
        if e.is_instance_of::<PyTypeError>(v.py()) {
            PyTypeError::new_err(
                "cache must be \"step\", \"across_steps\" or a capacity (an integer)",
            )
        } else {
            e
        }
    })?;
    if capacity < 1 {
        return Err(PyValueError::new_err(format!(
            "a cache capacity must be at least 1, got {capacity}"
        )));
    }
    Ok(CachePolicy::AcrossSteps {
        capacity: capacity as usize,
    })
}

/// The Python-visible `Snake`.
#[pyclass(frozen, module = "faulttune", name = "Snake")]
pub struct PySnake {
    graph: Py<PyGraph>,
    /// The Python callbacks the engine calls, shown to the garbage collector.
    held: Vec<Held>,
    goal: ElementSet,
    config: SnakeConfig,
    settings: Py<PyDict>,
    engine: Engine,
}

impl PySnake {
    fn graph_arc(&self) -> Arc<faulttune::ProcessorGraph> {
        Arc::clone(&self.graph.get().inner)
    }
}

#[pymethods]
impl PySnake {
    /// The Snake Optimizer on `graph`, calibrating the elements of `goal`.
    ///
    /// `goal` is an `ElementSet` or an iterable of `Element`s. Keyword arguments:
    ///
    /// * `parameter_distance`, `constraint_distance`, `traversal_distance` (`d_P`, `d_R`,
    ///   `d_T`; defaults 1, 2, 2) and `expiry_radius` (used by `recalibrate`; default 1);
    /// * `seed` (default 0) and `threads` (default: all cores): the result does not depend
    ///   on the number of threads;
    /// * `on_model_error`: `"abort"` (default), `"skip"`, `("retry", n)` or `Retry(n)`;
    /// * `cache`: `"step"` (default), `"across_steps"` or a capacity;
    /// * `evaluation_budget`: candidates per step handed to the optimizer as its budget;
    /// * `heuristic`: `"breadth_first"` (default), `"depth_first"`, `"random"` or a callable
    ///   `(context, options) -> sequence of Elements`, with an optional `frontier` attribute
    ///   (`"fifo"`, `"lifo"`, `"priority"`) and `priority(context, element)` method;
    /// * `optimizer`: `"coordinate_descent"` (default), `"exhaustive"`, `"random_search"`, a
    ///   `CoordinateDescent`/`Exhaustive`/`RandomSearch` instance or a callable
    ///   `(space, objective, rng) -> (indices, error)`;
    /// * `seed_policy`: `"lowest_id"` (default), `"random"`, a list of Elements to start from
    ///   or a callable `(thread_elements) -> Element`;
    /// * `algorithm`: `None` (everything active at once) or an `AlgorithmSubgraphs`.
    #[new]
    #[pyo3(signature = (
        graph,
        goal,
        *,
        parameter_distance=None,
        constraint_distance=None,
        traversal_distance=None,
        expiry_radius=None,
        seed=None,
        threads=None,
        on_model_error=None,
        cache=None,
        evaluation_budget=None,
        heuristic=None,
        optimizer=None,
        seed_policy=None,
        algorithm=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        graph: &Bound<'_, PyAny>,
        goal: &Bound<'_, PyAny>,
        parameter_distance: Option<&Bound<'_, PyAny>>,
        constraint_distance: Option<&Bound<'_, PyAny>>,
        traversal_distance: Option<&Bound<'_, PyAny>>,
        expiry_radius: Option<&Bound<'_, PyAny>>,
        seed: Option<&Bound<'_, PyAny>>,
        threads: Option<&Bound<'_, PyAny>>,
        on_model_error: Option<&Bound<'_, PyAny>>,
        cache: Option<&Bound<'_, PyAny>>,
        evaluation_budget: Option<&Bound<'_, PyAny>>,
        heuristic: Option<&Bound<'_, PyAny>>,
        optimizer: Option<&Bound<'_, PyAny>>,
        seed_policy: Option<&Bound<'_, PyAny>>,
        algorithm: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let graph = graph.cast::<PyGraph>().map_err(|_| {
            PyTypeError::new_err(format!(
                "graph must be a faulttune.ProcessorGraph, not {}",
                type_name(graph)
            ))
        })?;
        let core_graph = Arc::clone(&graph.get().inner);
        let goal = extract_set(&core_graph, goal, "goal")?;

        let mut config = SnakeConfig {
            parameter_distance: distance_arg("parameter_distance", parameter_distance, 1)?,
            constraint_distance: distance_arg("constraint_distance", constraint_distance, 2)?,
            traversal_distance: distance_arg("traversal_distance", traversal_distance, 2)?,
            expiry_radius: distance_arg("expiry_radius", expiry_radius, 1)?,
            on_model_error: parse_policy(on_model_error)?,
            cache: parse_cache(cache)?,
            ..SnakeConfig::default()
        };
        config.seed = match seed.filter(|s| !s.is_none()) {
            None => 0,
            Some(s) if s.is_instance_of::<PyBool>() => {
                return Err(PyTypeError::new_err("seed must be an integer, not bool"))
            }
            Some(s) => match s.extract::<u64>() {
                Ok(v) => v,
                Err(_) => match s.extract::<i64>() {
                    Ok(v) => {
                        return Err(PyValueError::new_err(format!(
                            "seed must be non-negative, got {v}"
                        )))
                    }
                    Err(_) if s.is_instance_of::<PyInt>() => {
                        return Err(PyValueError::new_err(format!(
                            "seed must be in the range 0 to 2**64 - 1, got {s}"
                        )))
                    }
                    Err(_) => {
                        return Err(PyTypeError::new_err(format!(
                            "seed must be an integer, not {}",
                            type_name(s)
                        )))
                    }
                },
            },
        };
        config.threads = match threads.filter(|t| !t.is_none()) {
            None => None,
            Some(t) => {
                let n = int_arg("threads", Some(t), 1)?;
                if n < 1 {
                    return Err(PyValueError::new_err(format!(
                        "threads must be at least 1 (or None for all cores), got {n}"
                    )));
                }
                Some(n as usize)
            }
        };
        let budget = int_arg("evaluation_budget", evaluation_budget, 1_000_000)?;
        if budget < 1 {
            return Err(PyValueError::new_err(format!(
                "evaluation_budget must be at least 1, got {budget}"
            )));
        }
        config.evaluation_budget = budget as u64;
        config.validate().map_err(|e| errors::plain(py, e))?;

        let failures = Arc::new(FailureSlot::default());
        let token = CancellationToken::new();
        let graph_handle: Py<PyGraph> = graph.clone().unbind();
        let reporter = Reporter {
            graph: Arc::new(graph_handle.clone_ref(py)),
            failures: Arc::clone(&failures),
            cancel: token.clone(),
        };

        let mut held: Vec<Held> = Vec::new();
        let heuristic_arc: Arc<dyn TraversalHeuristic> = match heuristic.filter(|h| !h.is_none()) {
            None => Arc::new(BreadthFirst),
            Some(h) => match h.extract::<String>().ok().as_deref() {
                Some("breadth_first") => Arc::new(BreadthFirst),
                Some("depth_first") => Arc::new(DepthFirst),
                Some("random") => Arc::new(Random),
                Some(other) => {
                    return Err(PyValueError::new_err(format!(
                        "unknown heuristic {other:?}: use \"breadth_first\", \"depth_first\", \
                         \"random\" or a callable"
                    )))
                }
                None => {
                    let heuristic = PyHeuristic::new(h, reporter.clone())?;
                    held.push(heuristic.held());
                    Arc::new(heuristic)
                }
            },
        };

        let optimizer_arc: Arc<dyn faulttune::Optimizer> = match optimizer.filter(|o| !o.is_none())
        {
            None => Arc::new(faulttune::optimizers::CoordinateDescent::default()),
            Some(o) => {
                if let Ok(name) = o.extract::<String>() {
                    match name.as_str() {
                        "coordinate_descent" => {
                            Arc::new(faulttune::optimizers::CoordinateDescent::default())
                        }
                        "exhaustive" => Arc::new(faulttune::optimizers::Exhaustive::default()),
                        "random_search" => Arc::new(faulttune::optimizers::RandomSearch::default()),
                        other => {
                            return Err(PyValueError::new_err(format!(
                                "unknown optimizer {other:?}: use \"coordinate_descent\", \
                                     \"exhaustive\", \"random_search\", an optimizer object or \
                                     a callable"
                            )))
                        }
                    }
                } else if let Ok(cd) = o.cast::<PyCoordinateDescent>() {
                    Arc::new(cd.get().inner.clone())
                } else if let Ok(ex) = o.cast::<PyExhaustive>() {
                    Arc::new(ex.get().inner.clone())
                } else if let Ok(rs) = o.cast::<PyRandomSearch>() {
                    Arc::new(rs.get().inner.clone())
                } else if o.is_callable() {
                    let callable = Held::new(o.clone().unbind());
                    held.push(callable.clone());
                    Arc::new(PyOptimizer::new(callable, reporter.clone()))
                } else {
                    return Err(PyTypeError::new_err(format!(
                        "optimizer must be a name, a CoordinateDescent/Exhaustive/\
                             RandomSearch instance or a callable (space, objective, rng), not {}",
                        type_name(o)
                    )));
                }
            }
        };

        let seed_policy_value = match seed_policy.filter(|s| !s.is_none()) {
            None => SeedPolicy::LowestId,
            Some(s) => {
                if let Ok(name) = s.extract::<String>() {
                    match name.as_str() {
                        "lowest_id" => SeedPolicy::LowestId,
                        "random" => SeedPolicy::Random,
                        other => {
                            return Err(PyValueError::new_err(format!(
                                "unknown seed_policy {other:?}: use \"lowest_id\", \"random\", \
                                 a list of Elements or a callable"
                            )))
                        }
                    }
                } else if s.is_callable() {
                    let callable = Held::new(s.clone().unbind());
                    held.push(callable.clone());
                    callable_seed_policy(callable, reporter.clone())
                } else {
                    SeedPolicy::Fixed(extract_element_list(&core_graph, s, "seed_policy")?)
                }
            }
        };

        let algorithm_object = match algorithm.filter(|a| !a.is_none()) {
            None => None,
            Some(a) => {
                let a = a.cast::<PyAlgorithmSubgraphs>().map_err(|_| {
                    PyTypeError::new_err(format!(
                        "algorithm must be None or an AlgorithmSubgraphs, not {}",
                        type_name(a)
                    ))
                })?;
                if !Arc::ptr_eq(&a.get().graph.get().inner, &core_graph) {
                    return Err(PyValueError::new_err(
                        "algorithm belongs to a different graph",
                    ));
                }
                Some(a.clone())
            }
        };

        let dispatch = Arc::new(Dispatch::default());
        let mut snake = Snake::new_shared(Arc::clone(&core_graph), goal.clone(), config.clone())
            .heuristic_arc(heuristic_arc)
            .optimizer_arc(optimizer_arc)
            .seed_policy(seed_policy_value)
            .observer_arc(dispatch.clone())
            .cancellation(token.clone());
        if let Some(a) = &algorithm_object {
            snake = snake.algorithm(a.get().inner.clone());
        }

        let settings = PyDict::new(py);
        settings.set_item("parameter_distance", config.parameter_distance)?;
        settings.set_item("constraint_distance", config.constraint_distance)?;
        settings.set_item("traversal_distance", config.traversal_distance)?;
        settings.set_item("expiry_radius", config.expiry_radius)?;
        settings.set_item("seed", config.seed)?;
        settings.set_item("threads", config.threads)?;
        settings.set_item(
            "on_model_error",
            match config.on_model_error {
                ModelErrorPolicy::Abort => "abort".into_pyobject(py)?.into_any(),
                ModelErrorPolicy::SkipElement => "skip".into_pyobject(py)?.into_any(),
                ModelErrorPolicy::Retry { times } => ("retry", times).into_pyobject(py)?.into_any(),
            },
        )?;
        settings.set_item(
            "cache",
            match config.cache {
                CachePolicy::StepOnly => "step".into_pyobject(py)?.into_any(),
                CachePolicy::AcrossSteps { capacity } => capacity.into_pyobject(py)?.into_any(),
            },
        )?;
        settings.set_item("evaluation_budget", config.evaluation_budget)?;
        settings.set_item("heuristic", heuristic.filter(|h| !h.is_none()))?;
        settings.set_item("optimizer", optimizer.filter(|o| !o.is_none()))?;
        settings.set_item("seed_policy", seed_policy.filter(|s| !s.is_none()))?;
        settings.set_item("algorithm", algorithm_object)?;
        for (key, default) in [
            ("heuristic", "breadth_first"),
            ("optimizer", "coordinate_descent"),
            ("seed_policy", "lowest_id"),
        ] {
            if settings.get_item(key)?.is_none_or(|v| v.is_none()) {
                settings.set_item(key, default)?;
            }
        }

        let option_retries = match config.on_model_error {
            ModelErrorPolicy::Retry { times } => times,
            _ => 0,
        };
        Ok(Self {
            held,
            graph: graph_handle.clone_ref(py),
            goal,
            config,
            settings: settings.unbind(),
            engine: Engine::new(
                graph_handle,
                Arc::new(snake),
                dispatch,
                token,
                failures,
                option_retries,
            ),
        })
    }

    fn __traverse__(&self, visit: pyo3::gc::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        visit.call(&self.graph)?;
        visit.call(&self.settings)?;
        for held in &self.held {
            held.visit(&visit)?;
        }
        Ok(())
    }

    fn __clear__(&self, py: Python<'_>) {
        for held in &self.held {
            held.clear();
        }
        self.settings.bind(py).clear();
    }

    /// The processor graph.
    #[getter]
    fn graph(&self, py: Python<'_>) -> Py<PyGraph> {
        self.graph.clone_ref(py)
    }

    /// The calibration goal as an `ElementSet`.
    #[getter]
    fn goal<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyElementSet>> {
        new_set(py, &self.graph_arc(), self.goal.clone())
    }

    /// The configuration as a dict (a copy: changing it does not change the Snake).
    #[getter]
    fn config<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.settings.bind(py).copy()
    }

    /// The report of the last finished (or cancelled) run, or `None`.
    #[getter]
    fn last_report(&self, py: Python<'_>) -> Option<Py<PyReport>> {
        self.engine
            .last_report
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .map(|r| r.clone_ref(py))
    }

    /// Calibrates the whole goal and returns the `ParameterDatabase`.
    ///
    /// With `database=` it continues that database (the way to resume an interrupted run)
    /// and returns it. `on_step(record)` is called on the calling thread for every committed
    /// step. `checkpoint=path` appends every committed step to a journal file (see
    /// `ParameterDatabase.replay_journal`).
    ///
    /// The engine runs on its own thread; Ctrl-C and `cancel()` stop it between steps, and
    /// every committed step is kept (in `database`, or as `.partial` of the exception).
    ///
    /// Things to know:
    ///
    /// * While a call that was given `database=` runs, the engine owns that database: using it
    ///   from `on_step` (or from another thread) raises `RuntimeError`. To watch progress or
    ///   save part-way, use the `StepRecord` that `on_step` receives (its `values` are set),
    ///   or `checkpoint=` and `ParameterDatabase.replay_journal`.
    /// * Ctrl-C takes effect between steps. Repeated Ctrl-C presses are swallowed until the
    ///   running step has finished; a model whose steps take long (a hardware measurement)
    ///   should itself check for interruption. After Ctrl-C or an exception in `on_step`,
    ///   steps that `on_step` has not seen yet are not delivered; they are in the database.
    /// * `on_model_error` covers `error()` and `prepare()`. For `options()` only the retry
    ///   count applies: `"skip"` cannot skip an element whose options are unknown, so the
    ///   run stops with `ModelError` as for `"abort"`.
    #[pyo3(signature = (model, *, database=None, on_step=None, checkpoint=None))]
    fn calibrate<'py>(
        &self,
        py: Python<'py>,
        model: &Bound<'py, PyAny>,
        database: Option<&Bound<'py, PyDatabase>>,
        on_step: Option<&Bound<'py, PyAny>>,
        checkpoint: Option<PathBuf>,
    ) -> PyResult<Py<PyDatabase>> {
        let (db, _report) = self.engine.run(
            py,
            Request {
                job: Job::Calibrate,
                database,
                model,
                on_step: on_step.filter(|c| !c.is_none()),
                checkpoint,
            },
        )?;
        Ok(db)
    }

    /// Local re-calibration: forgets the neighbourhood (`expiry_radius`) of each expired
    /// element in `database` and re-learns exactly that region. `elements` is one `Element` or
    /// an iterable of them. Returns the `Report`.
    #[pyo3(signature = (database, elements, model, *, on_step=None, checkpoint=None))]
    fn recalibrate<'py>(
        &self,
        py: Python<'py>,
        database: &Bound<'py, PyDatabase>,
        elements: &Bound<'py, PyAny>,
        model: &Bound<'py, PyAny>,
        on_step: Option<&Bound<'py, PyAny>>,
        checkpoint: Option<PathBuf>,
    ) -> PyResult<Py<PyReport>> {
        let graph = self.graph_arc();
        let ids = if elements.is_instance_of::<PyElement>() {
            vec![extract_element(&graph, elements, "elements")?]
        } else {
            extract_element_list(&graph, elements, "elements")?
        };
        let (_db, report) = self.engine.run(
            py,
            Request {
                job: Job::Recalibrate(ids),
                database: Some(database),
                model,
                on_step: on_step.filter(|c| !c.is_none()),
                checkpoint,
            },
        )?;
        Ok(report)
    }

    /// Plans the stitching of constraint-disjoint `regions` (each an `ElementSet` or an
    /// iterable of `Element`s, subsets of the goal).
    fn stitch_plan<'py>(
        &self,
        py: Python<'py>,
        regions: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyStitchPlan>> {
        let graph = self.graph_arc();
        let regions = extract_regions(&graph, regions)?;
        let plan = self
            .engine
            .snake
            .stitch_plan(regions)
            .map_err(|e| errors::plain(py, e))?;
        Bound::new(py, PyStitchPlan::new(graph, plan))
    }

    /// Calibrates `regions_or_plan` into `database`: the regions minus the gap between them
    /// first (as separate subgoals, in parallel), then the gap under the constraints from
    /// both sides. `regions_or_plan` is a `StitchPlan` (from `stitch_plan`) or the regions.
    /// Returns the `Report` of both stages.
    #[pyo3(signature = (database, regions_or_plan, model, *, on_step=None, checkpoint=None))]
    fn stitch<'py>(
        &self,
        py: Python<'py>,
        database: &Bound<'py, PyDatabase>,
        regions_or_plan: &Bound<'py, PyAny>,
        model: &Bound<'py, PyAny>,
        on_step: Option<&Bound<'py, PyAny>>,
        checkpoint: Option<PathBuf>,
    ) -> PyResult<Py<PyReport>> {
        let graph = self.graph_arc();
        let plan = match regions_or_plan.cast::<PyStitchPlan>() {
            Ok(plan) => {
                let plan = plan.get();
                if !Arc::ptr_eq(plan.graph(), &graph) {
                    return Err(PyValueError::new_err(
                        "the stitch plan was made for a different graph",
                    ));
                }
                plan.plan.clone()
            }
            Err(_) => {
                let regions = extract_regions(&graph, regions_or_plan)?;
                self.engine
                    .snake
                    .stitch_plan(regions)
                    .map_err(|e| errors::plain(py, e))?
            }
        };
        let (_db, report) = self.engine.run(
            py,
            Request {
                job: Job::Stitch(plan),
                database: Some(database),
                model,
                on_step: on_step.filter(|c| !c.is_none()),
                checkpoint,
            },
        )?;
        Ok(report)
    }

    /// Asks the running `calibrate`/`recalibrate`/`stitch` call to stop. Callable from any
    /// thread; the running call then raises `CancelledError` (with the committed steps kept).
    /// Does nothing when no call is running.
    fn cancel(&self) {
        self.engine.cancel();
    }

    fn __repr__(&self) -> String {
        format!(
            "Snake(goal={} elements, parameter_distance={}, constraint_distance={}, \
             traversal_distance={})",
            self.goal.len(),
            self.config.parameter_distance,
            self.config.constraint_distance,
            self.config.traversal_distance
        )
    }
}
