//! The `faulttune` exception hierarchy and the conversion of core errors.
//!
//! The classes are created at import time with `type(name, bases, namespace)`, which allows
//! several bases (for example `InvalidConfigError` is also a `ValueError`) and gives each class
//! `__module__ == "faulttune"`.

use std::sync::{Arc, Mutex, PoisonError};

use faulttune::{ElementId, FaultTuneError, ProcessorGraph};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyTuple, PyType};

use crate::graph::new_element;

/// The exception classes of the package.
pub struct Exceptions {
    pub fault_tune: Py<PyType>,
    pub invalid_graph: Py<PyType>,
    pub invalid_config: Py<PyType>,
    pub search_space_too_large: Py<PyType>,
    pub model: Py<PyType>,
    pub non_finite: Py<PyType>,
    pub no_progress: Py<PyType>,
    pub graph_mismatch: Py<PyType>,
    pub cancelled: Py<PyType>,
    pub format: Py<PyType>,
}

static EXCEPTIONS: PyOnceLock<Exceptions> = PyOnceLock::new();

fn make_class<'py>(
    py: Python<'py>,
    name: &str,
    bases: &[&Bound<'py, PyType>],
    doc: &str,
) -> PyResult<Bound<'py, PyType>> {
    let namespace = PyDict::new(py);
    namespace.set_item("__module__", "faulttune")?;
    namespace.set_item("__doc__", doc)?;
    let bases = PyTuple::new(py, bases.iter().copied())?;
    py.get_type::<PyType>()
        .call1((name, bases, namespace))?
        .cast_into::<PyType>()
        .map_err(Into::into)
}

fn build(py: Python<'_>) -> PyResult<Exceptions> {
    let builtins = py.import("builtins")?;
    let exception = builtins.getattr("Exception")?.cast_into::<PyType>()?;
    let value_error = builtins.getattr("ValueError")?.cast_into::<PyType>()?;
    let base = make_class(
        py,
        "FaultTuneError",
        &[&exception],
        "Base class of every error raised by FaultTune itself.",
    )?;
    let sub = |name: &str, extra: &[&Bound<'_, PyType>], doc: &str| -> PyResult<Py<PyType>> {
        let mut bases = vec![&base];
        bases.extend(extra.iter().copied());
        Ok(make_class(py, name, &bases, doc)?.unbind())
    };
    Ok(Exceptions {
        invalid_graph: sub(
            "InvalidGraphError",
            &[&value_error],
            "The processor graph is malformed.",
        )?,
        invalid_config: sub(
            "InvalidConfigError",
            &[&value_error],
            "The configuration, goal or another input is invalid.",
        )?,
        search_space_too_large: sub(
            "SearchSpaceTooLargeError",
            &[],
            "The exhaustive optimizer was asked to search too many combinations.\n\n\
             Attributes: size (number of combinations) and budget.",
        )?,
        model: sub(
            "ModelError",
            &[],
            "The error model (or another piece of user code) failed.\n\n\
             The original exception is the __cause__. Attribute: element, the centre of the \
             failing step (None if it is not known).",
        )?,
        non_finite: sub(
            "NonFiniteError",
            &[],
            "The error model returned NaN.\n\nAttribute: element, the centre of the step.",
        )?,
        no_progress: sub(
            "NoProgressError",
            &[],
            "A calibration pass made no progress.",
        )?,
        graph_mismatch: sub(
            "GraphMismatchError",
            &[&value_error],
            "A saved calibration belongs to a different graph.",
        )?,
        cancelled: sub(
            "CancelledError",
            &[],
            "The calibration was cancelled. Every committed step is kept.\n\n\
             Attributes: report (what was done, or None) and partial (the database).",
        )?,
        format: sub(
            "FormatError",
            &[&value_error],
            "A file or string is not in a recognised format.",
        )?,
        fault_tune: base.unbind(),
    })
}

/// The exception classes (created on first use).
pub fn exceptions(py: Python<'_>) -> PyResult<&'static Exceptions> {
    EXCEPTIONS.get_or_try_init(py, || build(py))
}

/// Adds the exception classes to the module.
pub fn init(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let e = exceptions(py)?;
    for (name, class) in [
        ("FaultTuneError", &e.fault_tune),
        ("InvalidGraphError", &e.invalid_graph),
        ("InvalidConfigError", &e.invalid_config),
        ("SearchSpaceTooLargeError", &e.search_space_too_large),
        ("ModelError", &e.model),
        ("NonFiniteError", &e.non_finite),
        ("NoProgressError", &e.no_progress),
        ("GraphMismatchError", &e.graph_mismatch),
        ("CancelledError", &e.cancelled),
        ("FormatError", &e.format),
    ] {
        m.add(name, class.bind(py))?;
    }
    Ok(())
}

/// `TypeName: message` for an exception.
pub fn describe(py: Python<'_>, err: &PyErr) -> String {
    let ty = err
        .get_type(py)
        .qualname()
        .map(|n| n.to_string())
        .unwrap_or_else(|_| "exception".into());
    let text = err.value(py).to_string();
    if text.is_empty() {
        ty
    } else {
        format!("{ty}: {text}")
    }
}

/// A `faulttune.ModelError` with an optional element and an optional chained cause.
pub fn model_error(
    py: Python<'_>,
    graph: Option<&Arc<ProcessorGraph>>,
    element: Option<ElementId>,
    message: &str,
    cause: Option<PyErr>,
) -> PyErr {
    let build = || -> PyResult<PyErr> {
        let class = exceptions(py)?.model.bind(py);
        let instance = class.call1((message,))?;
        let element = match (graph, element) {
            (Some(graph), Some(id)) => new_element(py, graph, id)?.into_any(),
            _ => py.None().into_bound(py),
        };
        instance.setattr("element", element)?;
        Ok(PyErr::from_value(instance))
    };
    match build() {
        Ok(err) => {
            if let Some(cause) = cause {
                err.set_cause(py, Some(cause));
            }
            err
        }
        // Building the exception failed (an interpreter in distress): report that instead.
        Err(err) => err,
    }
}

/// A plain `faulttune.FaultTuneError` with a message.
pub fn generic(py: Python<'_>, message: &str) -> PyErr {
    exceptions(py)
        .and_then(|e| e.fault_tune.bind(py).call1((message,)))
        .map(PyErr::from_value)
        .unwrap_or_else(|err| err)
}

/// The Python exception for a core error, without the context only the caller has.
pub fn plain(py: Python<'_>, err: FaultTuneError) -> PyErr {
    convert(py, None, err, None)
}

/// The Python exception for a core error.
///
/// `graph` lets exceptions carry `Element`s (`ModelError.element`); `report` becomes
/// `CancelledError.report`. `CancelledError.partial` is set by the caller.
pub fn convert(
    py: Python<'_>,
    graph: Option<&Arc<ProcessorGraph>>,
    err: FaultTuneError,
    report: Option<Py<PyAny>>,
) -> PyErr {
    let result = (|| -> PyResult<PyErr> {
        let e = exceptions(py)?;
        let text = describe_core(graph, &err);
        let simple = |class: &Py<PyType>| -> PyResult<PyErr> {
            Ok(PyErr::from_value(class.bind(py).call1((text.as_str(),))?))
        };
        match err {
            FaultTuneError::InvalidGraph(_) => simple(&e.invalid_graph),
            FaultTuneError::InvalidConfig(_) => simple(&e.invalid_config),
            FaultTuneError::SearchSpaceTooLarge { size, budget } => {
                let instance = e.search_space_too_large.bind(py).call1((text.as_str(),))?;
                instance.setattr("size", size)?;
                instance.setattr("budget", budget)?;
                Ok(PyErr::from_value(instance))
            }
            FaultTuneError::Model { element, source } => {
                let cause = source
                    .source_ref()
                    .and_then(|s| s.downcast_ref::<PyErr>())
                    .map(|p| p.clone_ref(py));
                Ok(model_error(py, graph, Some(element), &text, cause))
            }
            FaultTuneError::NonFiniteError { element } => {
                let instance = e.non_finite.bind(py).call1((text.as_str(),))?;
                let element = match graph {
                    Some(graph) => new_element(py, graph, element)?.into_any(),
                    None => py.None().into_bound(py),
                };
                instance.setattr("element", element)?;
                Ok(PyErr::from_value(instance))
            }
            FaultTuneError::NoProgress => simple(&e.no_progress),
            FaultTuneError::GraphMismatch { .. } => simple(&e.graph_mismatch),
            FaultTuneError::Cancelled { .. } => {
                let instance = e.cancelled.bind(py).call1((text.as_str(),))?;
                instance.setattr("report", report.unwrap_or_else(|| py.None()))?;
                instance.setattr("partial", py.None())?;
                Ok(PyErr::from_value(instance))
            }
            FaultTuneError::Io(io) => Ok(PyErr::from(io)),
            FaultTuneError::Format(_) => simple(&e.format),
            // `FaultTuneError` is non-exhaustive: a variant added later still gets an error.
            other => Ok(PyErr::from_value(
                e.fault_tune.bind(py).call1((other.to_string(),))?,
            )),
        }
    })();
    result.unwrap_or_else(|err| err)
}

/// The message of a core error, with element labels instead of raw ids when the graph is known.
fn describe_core(graph: Option<&Arc<ProcessorGraph>>, err: &FaultTuneError) -> String {
    let label = |id: ElementId| match graph {
        Some(g) if id.index() < g.len() => g.label(id).to_string(),
        _ => format!("#{}", id.raw()),
    };
    match err {
        FaultTuneError::Model { element, source } => format!(
            "the error model failed for the step centred on {}: {source}",
            label(*element)
        ),
        FaultTuneError::NonFiniteError { element } => format!(
            "the error model returned NaN for the step centred on {}",
            label(*element)
        ),
        FaultTuneError::Io(io) => io.to_string(),
        other => other.to_string(),
    }
}

/// A failure of user code that cannot be returned to the engine (for example an exception in
/// `options()`, a heuristic or a seed policy, none of which can fail in the core API).
///
/// It is remembered, the engine is told to stop, and the failure is raised once the engine has
/// stopped.
pub struct Failure {
    /// The element it concerns, if known.
    pub element: Option<ElementId>,
    /// A description.
    pub message: String,
    /// The original exception.
    pub cause: Option<PyErr>,
}

/// Holds the first [`Failure`] of a run.
#[derive(Default)]
pub struct FailureSlot(Mutex<Option<Failure>>);

impl FailureSlot {
    /// Remembers `failure` unless an earlier one is held.
    pub fn record(&self, failure: Failure) {
        let mut slot = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        if slot.is_none() {
            *slot = Some(failure);
        }
    }

    /// Takes the held failure, if any.
    pub fn take(&self) -> Option<Failure> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner).take()
    }

    /// Whether a failure is held.
    pub fn is_set(&self) -> bool {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_some()
    }
}

impl Failure {
    /// The exception to raise for this failure.
    pub fn into_pyerr(self, py: Python<'_>, graph: &Arc<ProcessorGraph>) -> PyErr {
        model_error(py, Some(graph), self.element, &self.message, self.cause)
    }
}
