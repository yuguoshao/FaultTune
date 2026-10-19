//! The error-model adapter (ARCHITECTURE §9.3): runs a Python model inside the Rust engine.

use std::sync::Arc;

use faulttune::{CandidateBatch, ElementId, ErrorModel, ModelError, Step};
use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::errors::{describe, Failure, FailureSlot};
use crate::graph::PyGraph;
use crate::npy::{format_shape, to_f64_array};
use crate::records::{OptionTables, PyStep};

/// Per-step data of the adapter: the Python `Step` object, which carries what `prepare`
/// returned and is passed to every `error` call of the step.
pub struct StepData {
    step: Py<PyStep>,
}

/// A Python object with `options()` and `error()` (and optionally `prepare()`) methods, as an
/// [`ErrorModel`] with `f64` values.
///
/// Every call takes the GIL for its duration only. The model's exceptions become
/// [`ModelError`]s that keep the original `PyErr` as their source, so that they can be
/// re-raised with `raise ... from original`.
pub struct PyErrorModel {
    model: Py<PyAny>,
    graph: Py<PyGraph>,
    has_prepare: bool,
    name: String,
    tables: Arc<OptionTables>,
    failures: Arc<FailureSlot>,
    /// How many times a failing `options()` call is repeated (`on_model_error=Retry`).
    retries: u32,
}

impl PyErrorModel {
    /// Checks that `model` can serve as an error model and wraps it.
    ///
    /// `TypeError` unless it has callable `options` and `error` attributes. `prepare` is used
    /// only if it exists and is not the do-nothing default of `faulttune.ErrorModel`. A failing
    /// `options()` call is repeated `retries` more times before it is recorded as a failure.
    pub fn new(
        model: &Bound<'_, PyAny>,
        graph: Py<PyGraph>,
        tables: Arc<OptionTables>,
        failures: Arc<FailureSlot>,
        retries: u32,
    ) -> PyResult<Self> {
        let py = model.py();
        let name = model
            .get_type()
            .name()
            .map(|n| n.to_string())
            .unwrap_or_else(|_| "model".into());
        for method in ["options", "error"] {
            match model.getattr(method) {
                Ok(attr) if attr.is_callable() => {}
                _ => {
                    return Err(PyTypeError::new_err(format!(
                        "model must have a callable {method}() method (subclass \
                         faulttune.ErrorModel); got an object of type {name}"
                    )))
                }
            }
        }
        let has_prepare = match model.getattr("prepare") {
            Ok(prepare) if prepare.is_callable() => !is_default_prepare(py, &prepare),
            _ => false,
        };
        Ok(Self {
            model: model.clone().unbind(),
            graph,
            has_prepare,
            name,
            tables,
            failures,
            retries,
        })
    }

    /// `fetch_options`, repeated up to `retries` more times while it fails.
    pub fn fetch_options_with_retries(
        &self,
        py: Python<'_>,
        element: ElementId,
    ) -> Result<Vec<f64>, Failure> {
        let mut attempt = 0;
        loop {
            match self.fetch_options(py, element) {
                Err(_) if attempt < self.retries => attempt += 1,
                other => return other,
            }
        }
    }

    /// The options of `element` as the model gives them, or the failure to describe.
    pub fn fetch_options(&self, py: Python<'_>, element: ElementId) -> Result<Vec<f64>, Failure> {
        let graph = &self.graph.get().inner;
        let label = graph.label(element).to_string();
        let fail = |message: String, cause: Option<PyErr>| Failure {
            element: Some(element),
            message,
            cause,
        };
        let handle = crate::graph::new_element(py, graph, element)
            .map_err(|e| fail("could not create the element handle".into(), Some(e)))?;
        let result = self
            .model
            .bind(py)
            .call_method1("options", (handle,))
            .map_err(|e| {
                fail(
                    format!(
                        "{}.options() raised {} for element {label}",
                        self.name,
                        describe(py, &e)
                    ),
                    Some(e),
                )
            })?;
        if result.is_none() {
            return Err(fail(
                format!(
                    "{}.options() for element {label} returned None; it must return an array \
                     of floats",
                    self.name
                ),
                None,
            ));
        }
        let (shape, values) = to_f64_array(py, &result).map_err(|e| {
            fail(
                format!(
                    "{}.options() for element {label} did not return an array of floats ({})",
                    self.name,
                    describe(py, &e)
                ),
                Some(e),
            )
        })?;
        if shape.len() != 1 {
            return Err(fail(
                format!(
                    "{}.options() for element {label} must return a 1-D array, but returned \
                     shape {}",
                    self.name,
                    format_shape(&shape)
                ),
                None,
            ));
        }
        if values.is_empty() {
            return Err(fail(
                format!(
                    "{}.options() for element {label} returned no options; it must return at \
                     least one",
                    self.name
                ),
                None,
            ));
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(fail(
                format!(
                    "{}.options() for element {label} returned NaN or infinite values",
                    self.name
                ),
                None,
            ));
        }
        Ok(values)
    }

    /// One `error(step, candidates)` call for `n` rows of `width` values (row-major).
    fn call_error(
        &self,
        py: Python<'_>,
        step: &Py<PyStep>,
        flat: &[f64],
        width: usize,
    ) -> Result<Vec<f64>, ModelError> {
        let n = flat.len().checked_div(width).unwrap_or(0);
        let candidates = PyArray1::from_slice(py, flat)
            .reshape([n, width])
            .map_err(|e| ModelError::with_source("could not build the candidates array", e))?;
        let result = self
            .model
            .bind(py)
            .call_method1("error", (step.bind(py), candidates))
            .map_err(|e| {
                ModelError::with_source(
                    format!("{}.error() raised {}", self.name, describe(py, &e)),
                    e,
                )
            })?;
        if result.is_none() {
            return Err(ModelError::new(format!(
                "{}.error() returned None; it must return an array of {n} errors (did it forget \
                 to return?)",
                self.name
            )));
        }
        let (shape, values) = to_f64_array(py, &result).map_err(|e| {
            ModelError::with_source(
                format!(
                    "{}.error() did not return an array of floats ({})",
                    self.name,
                    describe(py, &e)
                ),
                e,
            )
        })?;
        if shape != [n] {
            return Err(ModelError::new(format!(
                "{}.error() must return an array of shape ({n},), one error per candidate, but \
                 returned shape {}",
                self.name,
                format_shape(&shape)
            )));
        }
        Ok(values)
    }
}

/// Whether `prepare` is the do-nothing method inherited from `faulttune.ErrorModel`.
fn is_default_prepare(py: Python<'_>, prepare: &Bound<'_, PyAny>) -> bool {
    let Ok(function) = prepare.getattr("__func__") else {
        return false;
    };
    py.import("faulttune._model")
        .and_then(|m| m.getattr("ErrorModel"))
        .and_then(|c| c.getattr("prepare"))
        .is_ok_and(|default| default.is(&function))
}

impl ErrorModel for PyErrorModel {
    type Value = f64;
    type Data = StepData;

    fn options(&self, element: ElementId) -> Vec<f64> {
        Python::attach(|py| match self.fetch_options_with_retries(py, element) {
            Ok(values) => {
                self.tables.set(element, Arc::from(values.as_slice()));
                values
            }
            Err(failure) => {
                // The core API cannot fail here: an empty table makes the engine stop with an
                // error, and the failure is raised in its place. (`on_model_error="skip"`
                // therefore does not cover `options()`; `Retry` does.)
                self.failures.record(failure);
                Vec::new()
            }
        })
    }

    fn prepare(&self, step: &Step<'_, f64>) -> Result<StepData, ModelError> {
        Python::attach(|py| {
            let pystep = Py::new(
                py,
                PyStep::new(
                    self.graph.clone_ref(py),
                    step.index,
                    step.centre,
                    step.parameters,
                    step.constraints,
                ),
            )
            .map_err(|e| ModelError::with_source("could not create the Step object", e))?;
            if self.has_prepare {
                let data = self
                    .model
                    .bind(py)
                    .call_method1("prepare", (pystep.bind(py),))
                    .map_err(|e| {
                        ModelError::with_source(
                            format!("{}.prepare() raised {}", self.name, describe(py, &e)),
                            e,
                        )
                    })?;
                if !data.is_none() {
                    pystep.get().set_data(data.unbind());
                }
            }
            Ok(StepData { step: pystep })
        })
    }

    fn error(&self, step: &Step<'_, f64>, data: &StepData, candidate: &[f64]) -> f64 {
        let _ = step;
        // The engine always calls `error_batch`; this single-candidate path goes through it.
        Python::attach(
            |py| match self.call_error(py, &data.step, candidate, candidate.len()) {
                Ok(values) => values.first().copied().unwrap_or(f64::NAN),
                Err(err) => {
                    self.failures.record(Failure {
                        element: Some(step.centre),
                        message: err.message().to_string(),
                        cause: err
                            .source_ref()
                            .and_then(|s| s.downcast_ref::<PyErr>())
                            .map(|p| p.clone_ref(py)),
                    });
                    f64::NAN
                }
            },
        )
    }

    fn error_batch(
        &self,
        _step: &Step<'_, f64>,
        data: &StepData,
        candidates: &CandidateBatch<'_, f64>,
        out: &mut [f64],
    ) -> Result<(), ModelError> {
        Python::attach(|py| {
            let values =
                self.call_error(py, &data.step, candidates.as_flat(), candidates.width())?;
            out.copy_from_slice(&values);
            Ok(())
        })
    }
}
