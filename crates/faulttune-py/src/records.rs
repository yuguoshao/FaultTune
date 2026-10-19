//! Read-only result types: `Step`, `StepRecord`, `Invalidation`, `Report`.

use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use faulttune::{ElementId, HistoryRecord, ProcessorGraph, Report, StepRecord};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::graph::{elements_list, new_element, PyElement, PyGraph};

/// The option values of every element whose `options()` the model has been asked for.
///
/// The model adapter fills it (once per element); it is how a committed `StepRecord`, which
/// only carries option indices, gets its values back.
pub struct OptionTables {
    tables: Vec<OnceLock<Arc<[f64]>>>,
}

impl OptionTables {
    /// Empty tables for a graph of `n` elements.
    pub fn new(n: usize) -> Self {
        Self {
            tables: (0..n).map(|_| OnceLock::new()).collect(),
        }
    }

    /// Stores the options of `id` (the first call wins).
    pub fn set(&self, id: ElementId, values: Arc<[f64]>) {
        if let Some(slot) = self.tables.get(id.index()) {
            let _ = slot.set(values);
        }
    }

    /// The options of `id`, if known.
    pub fn get(&self, id: ElementId) -> Option<&Arc<[f64]>> {
        self.tables.get(id.index()).and_then(OnceLock::get)
    }

    /// The committed values of a step (aligned with `record.parameters`), if all are known.
    pub fn values_of(&self, record: &StepRecord) -> Option<Vec<f64>> {
        record
            .parameters
            .iter()
            .zip(&record.option_indices)
            .map(|(&p, &k)| self.get(p).and_then(|t| t.get(k as usize).copied()))
            .collect()
    }
}

fn seconds(d: std::time::Duration) -> f64 {
    d.as_secs_f64()
}

/// One committed calibration step.
///
/// `index` is the position in the database history for records read from a database
/// (`db.history`, `db.steps()`); for the records delivered to `on_step` while a run is in
/// progress it is only the ordinal within the step's subgoal and pass, because the global
/// index is assigned when the engine merges the subgoals of a pass. Subgoals run in
/// parallel, so the order in which their records reach `on_step` may differ between runs
/// and from the order of the history. What identifies a step in both places is
/// `(pass_, subgoal, centre)`: use it to match a delivered record with `db.steps()`.
/// Steps that `on_step` has not seen yet when a run is stopped by Ctrl-C or by an error in
/// `on_step` are not delivered; they are in the database all the same.
#[pyclass(frozen, module = "faulttune", name = "StepRecord")]
pub struct PyStepRecord {
    pub graph: Arc<ProcessorGraph>,
    pub record: StepRecord,
    pub values: Option<Vec<f64>>,
}

impl PyStepRecord {
    /// Wraps a record.
    pub fn new(graph: &Arc<ProcessorGraph>, record: StepRecord, values: Option<Vec<f64>>) -> Self {
        Self {
            graph: Arc::clone(graph),
            record,
            values,
        }
    }
}

#[pymethods]
impl PyStepRecord {
    /// Index of the step (see the class documentation).
    #[getter]
    fn index(&self) -> u64 {
        self.record.index
    }

    /// The pass of the `calibrate` call that ran the step (0-based).
    #[getter]
    fn pass_(&self) -> u32 {
        self.record.pass
    }

    /// Alias of `pass_`.
    #[getter]
    fn pass_index(&self) -> u32 {
        self.record.pass
    }

    /// The subgoal within the pass (0-based).
    #[getter]
    fn subgoal(&self) -> u32 {
        self.record.subgoal
    }

    /// The element the step was centred on.
    #[getter]
    fn centre<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyElement>> {
        new_element(py, &self.graph, self.record.centre)
    }

    /// The parameters that were optimized, in candidate-column order.
    #[getter]
    fn parameters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        elements_list(py, &self.graph, self.record.parameters.iter().copied())
    }

    /// The committed option index of each parameter.
    #[getter]
    fn option_indices(&self) -> Vec<u32> {
        self.record.option_indices.clone()
    }

    /// The elements that constrained the step.
    #[getter]
    fn constraints<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        elements_list(py, &self.graph, self.record.constraints.iter().copied())
    }

    /// The error of the committed assignment (possibly `inf`).
    #[getter]
    fn error(&self) -> f64 {
        self.record.error
    }

    /// Number of distinct candidates evaluated.
    #[getter]
    fn evaluations(&self) -> u64 {
        self.record.evaluations
    }

    /// Total time of the step in seconds.
    #[getter]
    fn duration(&self) -> f64 {
        seconds(self.record.duration)
    }

    /// Time spent in `prepare`, in seconds.
    #[getter]
    fn prepare_time(&self) -> f64 {
        seconds(self.record.prepare_time)
    }

    /// Time spent evaluating errors, in seconds.
    #[getter]
    fn error_time(&self) -> f64 {
        seconds(self.record.error_time)
    }

    /// The committed values as `{parameter: value}`, or `None` for records read from a
    /// database history (which keeps option indices only).
    #[getter]
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(values) = &self.values else {
            return Ok(None);
        };
        let dict = PyDict::new(py);
        for (&p, &v) in self.record.parameters.iter().zip(values) {
            dict.set_item(new_element(py, &self.graph, p)?, v)?;
        }
        Ok(Some(dict))
    }

    /// Whether `other` describes the same outcome, ignoring the timings.
    fn same_outcome(&self, other: &PyStepRecord) -> bool {
        Arc::ptr_eq(&self.graph, &other.graph) && self.record.same_outcome(&other.record)
    }

    fn __repr__(&self) -> String {
        format!(
            "StepRecord(index={}, centre={:?}, parameters={}, error={})",
            self.record.index,
            self.graph.label(self.record.centre),
            self.record.parameters.len(),
            self.record.error
        )
    }
}

/// Elements removed from the calibration by `recalibrate`.
#[pyclass(frozen, module = "faulttune", name = "Invalidation")]
pub struct PyInvalidation {
    graph: Arc<ProcessorGraph>,
    elements: Vec<ElementId>,
}

#[pymethods]
impl PyInvalidation {
    /// The elements that were forgotten, ascending.
    #[getter]
    fn elements<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        elements_list(py, &self.graph, self.elements.iter().copied())
    }

    fn __repr__(&self) -> String {
        format!("Invalidation({} elements)", self.elements.len())
    }
}

/// A history entry as a Python object.
pub fn history_item<'py>(
    py: Python<'py>,
    graph: &Arc<ProcessorGraph>,
    record: &HistoryRecord,
) -> PyResult<Bound<'py, PyAny>> {
    match record {
        HistoryRecord::Step(step) => {
            Ok(Bound::new(py, PyStepRecord::new(graph, step.clone(), None))?.into_any())
        }
        HistoryRecord::Invalidated(elements) => Ok(Bound::new(
            py,
            PyInvalidation {
                graph: Arc::clone(graph),
                elements: elements.clone(),
            },
        )?
        .into_any()),
    }
}

/// Summary of a calibration run.
#[pyclass(frozen, module = "faulttune", name = "Report")]
pub struct PyReport {
    graph: Arc<ProcessorGraph>,
    report: Report,
    values: Vec<Option<Vec<f64>>>,
}

impl PyReport {
    /// Wraps a report; `tables` gives the worst steps their values when known.
    pub fn new(graph: &Arc<ProcessorGraph>, report: Report, tables: Option<&OptionTables>) -> Self {
        let values = report
            .worst_steps
            .iter()
            .map(|s| tables.and_then(|t| t.values_of(s)))
            .collect();
        Self {
            graph: Arc::clone(graph),
            report,
            values,
        }
    }
}

#[pymethods]
impl PyReport {
    /// Number of passes of the completion loop.
    #[getter]
    fn passes(&self) -> u32 {
        self.report.passes
    }

    /// Number of steps committed.
    #[getter]
    fn steps(&self) -> u64 {
        self.report.steps
    }

    /// Total candidates evaluated.
    #[getter]
    fn evaluations(&self) -> u64 {
        self.report.evaluations
    }

    /// Cross-step cache hits (0 when the cache is off).
    #[getter]
    fn cache_hits(&self) -> u64 {
        self.report.cache_hits
    }

    /// Cross-step cache misses (0 when the cache is off).
    #[getter]
    fn cache_misses(&self) -> u64 {
        self.report.cache_misses
    }

    /// Fraction of cross-step cache lookups that hit, or `None` if there were none.
    #[getter]
    fn cache_hit_rate(&self) -> Option<f64> {
        self.report.cache_hit_rate()
    }

    /// Wall-clock time of the run in seconds.
    #[getter]
    fn wall_time(&self) -> f64 {
        seconds(self.report.wall_time)
    }

    /// Total time spent in `prepare`, summed over steps, in seconds.
    #[getter]
    fn prepare_time(&self) -> f64 {
        seconds(self.report.prepare_time)
    }

    /// Total time spent evaluating errors, summed over steps, in seconds.
    #[getter]
    fn error_time(&self) -> f64 {
        seconds(self.report.error_time)
    }

    /// Elements given up on after a model error (`on_model_error="skip"`).
    #[getter]
    fn skipped<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        elements_list(py, &self.graph, self.report.skipped.iter().copied())
    }

    /// The steps with the largest error of the run, worst first (at most 10).
    #[getter]
    fn worst_steps<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for (step, values) in self.report.worst_steps.iter().zip(&self.values) {
            list.append(Bound::new(
                py,
                PyStepRecord::new(&self.graph, step.clone(), values.clone()),
            )?)?;
        }
        Ok(list)
    }

    fn __repr__(&self) -> String {
        format!(
            "Report(passes={}, steps={}, evaluations={}, wall_time={:.3}s, skipped={})",
            self.report.passes,
            self.report.steps,
            self.report.evaluations,
            seconds(self.report.wall_time),
            self.report.skipped.len()
        )
    }
}

/// One calibration step as seen by the error model.
///
/// A step centres on one element. Its `parameters` are the elements optimized together (the
/// columns of every candidate, in this order); its `constraints` map the already-calibrated
/// elements that constrain it to their fixed values. One `Step` object is created per step
/// and passed to `prepare` and to every `error` call of that step.
#[pyclass(frozen, module = "faulttune", name = "Step")]
pub struct PyStep {
    graph: Py<PyGraph>,
    index: u64,
    centre: ElementId,
    parameters: Vec<ElementId>,
    constraints: Vec<(ElementId, f64)>,
    data: Mutex<Option<Py<PyAny>>>,
    parameter_cache: OnceLock<Vec<Py<PyElement>>>,
    constraint_cache: OnceLock<Vec<(Py<PyElement>, f64)>>,
}

impl PyStep {
    /// A step; `data` is `None` until `prepare` has returned.
    pub fn new(
        graph: Py<PyGraph>,
        index: u64,
        centre: ElementId,
        parameters: &[ElementId],
        constraints: &[(ElementId, f64)],
    ) -> Self {
        Self {
            graph,
            index,
            centre,
            parameters: parameters.to_vec(),
            constraints: constraints.to_vec(),
            data: Mutex::new(None),
            parameter_cache: OnceLock::new(),
            constraint_cache: OnceLock::new(),
        }
    }

    /// Stores what `prepare` returned.
    pub fn set_data(&self, data: Py<PyAny>) {
        *self.data.lock().unwrap_or_else(PoisonError::into_inner) = Some(data);
    }

    fn graph_arc(&self) -> Arc<ProcessorGraph> {
        Arc::clone(&self.graph.get().inner)
    }
}

#[pymethods]
impl PyStep {
    /// Ordinal of the step within its subgoal and pass, starting at 0.
    #[getter]
    fn index(&self) -> u64 {
        self.index
    }

    /// The element the step is centred on.
    #[getter]
    fn centre<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyElement>> {
        new_element(py, &self.graph_arc(), self.centre)
    }

    /// The parameters being optimized: the columns of every candidate, in this order.
    #[getter]
    fn parameters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        if self.parameter_cache.get().is_none() {
            let graph = self.graph_arc();
            let built = self
                .parameters
                .iter()
                .map(|&p| new_element(py, &graph, p).map(Bound::unbind))
                .collect::<PyResult<Vec<_>>>()?;
            let _ = self.parameter_cache.set(built);
        }
        let cached = self.parameter_cache.get().map(Vec::as_slice).unwrap_or(&[]);
        PyList::new(py, cached.iter().map(|e| e.bind(py)))
    }

    /// The active constraints as `{element: fixed value}`, ascending by element id.
    #[getter]
    fn constraints<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        if self.constraint_cache.get().is_none() {
            let graph = self.graph_arc();
            let built = self
                .constraints
                .iter()
                .map(|&(c, v)| Ok((new_element(py, &graph, c)?.unbind(), v)))
                .collect::<PyResult<Vec<_>>>()?;
            let _ = self.constraint_cache.set(built);
        }
        let dict = PyDict::new(py);
        for (element, value) in self
            .constraint_cache
            .get()
            .map(Vec::as_slice)
            .unwrap_or(&[])
        {
            dict.set_item(element.bind(py), *value)?;
        }
        Ok(dict)
    }

    /// What `prepare` returned (`None` while `prepare` runs, and if it returned nothing).
    #[getter]
    fn data(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.data
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .as_ref()
            .map(|d| d.clone_ref(py))
    }

    /// The processor graph.
    #[getter]
    fn graph(&self, py: Python<'_>) -> Py<PyGraph> {
        self.graph.clone_ref(py)
    }

    fn __repr__(&self) -> String {
        let graph = self.graph_arc();
        format!(
            "Step(index={}, centre={:?}, parameters={}, constraints={})",
            self.index,
            graph.label(self.centre),
            self.parameters.len(),
            self.constraints.len()
        )
    }
}
