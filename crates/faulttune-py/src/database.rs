//! `ParameterDatabase`: the learned values and the calibration history.

use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use faulttune::persist::read_journal_entries;
use faulttune::{read_journal, ElementId, ParameterDatabase, ProcessorGraph};
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::errors::{self, Failure, FailureSlot};
use crate::graph::{
    elements_list, extract_element_or_label, new_element, new_set, PyElement, PyElementSet, PyGraph,
};
use crate::model::PyErrorModel;
use crate::records::{history_item, OptionTables, PyStepRecord};

/// Where the database is: here, or lent to a running calibration.
enum State {
    Ready(ParameterDatabase<f64>),
    Lent,
}

/// Learned values and the ordered calibration history for one graph.
///
/// `db[element]` is the value of a calibrated element (`KeyError` otherwise). While a
/// `calibrate`, `recalibrate` or `stitch` call that was given this database runs, the engine
/// owns it and any other use raises `RuntimeError`.
#[pyclass(frozen, module = "faulttune", name = "ParameterDatabase")]
pub struct PyDatabase {
    graph: Py<PyGraph>,
    state: Mutex<State>,
}

const IN_USE: &str = "this ParameterDatabase is in use by a running calibration; use it again \
                      once the call has returned";

impl PyDatabase {
    /// Wraps a core database of `graph`.
    pub fn from_core(graph: Py<PyGraph>, db: ParameterDatabase<f64>) -> Self {
        Self {
            graph,
            state: Mutex::new(State::Ready(db)),
        }
    }

    /// The core graph.
    pub fn arc(&self) -> Arc<ProcessorGraph> {
        Arc::clone(&self.graph.get().inner)
    }

    /// The Python graph.
    pub fn graph_object(&self, py: Python<'_>) -> Py<PyGraph> {
        self.graph.clone_ref(py)
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Runs `f` on the database (`RuntimeError` while it is lent).
    fn with<R>(&self, f: impl FnOnce(&ParameterDatabase<f64>) -> R) -> PyResult<R> {
        match &*self.lock() {
            State::Ready(db) => Ok(f(db)),
            State::Lent => Err(PyRuntimeError::new_err(IN_USE)),
        }
    }

    /// Takes the database out for the duration of a calibration (`RuntimeError` if it is
    /// lent already). Give it back with [`restore`](Self::restore).
    pub fn lend(&self) -> PyResult<ParameterDatabase<f64>> {
        let mut state = self.lock();
        match std::mem::replace(&mut *state, State::Lent) {
            State::Ready(db) => Ok(db),
            State::Lent => Err(PyRuntimeError::new_err(IN_USE)),
        }
    }

    /// Puts a lent database back.
    pub fn restore(&self, db: ParameterDatabase<f64>) {
        *self.lock() = State::Ready(db);
    }

    /// Number of steps committed so far (`RuntimeError` while the database is lent).
    pub fn step_count_now(&self) -> PyResult<u64> {
        self.with(|db| db.step_count())
    }

    /// Checks that this database belongs to the very graph object of a Snake. A database of
    /// another graph is left to the core to refuse (`GraphMismatchError`); one of a graph
    /// that is structurally identical but another object is refused here, because its
    /// `Element`s would not be those of the Snake.
    pub fn ensure_graph(&self, snake_graph: &Py<PyGraph>) -> PyResult<()> {
        let (mine, theirs) = (self.graph.get(), snake_graph.get());
        if Arc::ptr_eq(&mine.inner, &theirs.inner) {
            return Ok(());
        }
        let same_device = self.with(|db| db.graph_fingerprint())? == theirs.inner.fingerprint();
        if same_device {
            return Err(PyValueError::new_err(
                "database belongs to a different ProcessorGraph object than the Snake's, \
                 although the two are structurally identical: build the database with \
                 ParameterDatabase(snake.graph), or load it with the Snake's graph",
            ));
        }
        Ok(())
    }

    fn element_id(&self, element: &Bound<'_, PyAny>) -> PyResult<ElementId> {
        extract_element_or_label(&self.graph.get().inner, element, "element")
    }

    fn not_calibrated(&self, id: ElementId) -> PyErr {
        PyKeyError::new_err(format!(
            "element {:?} is not calibrated",
            self.graph.get().inner.label(id)
        ))
    }
}

#[pymethods]
impl PyDatabase {
    /// An empty database for `graph`.
    #[new]
    fn py_new(graph: &Bound<'_, PyGraph>) -> Self {
        let g = graph.get();
        Self::from_core(graph.clone().unbind(), ParameterDatabase::new(&g.inner))
    }

    /// The processor graph.
    #[getter]
    fn graph(&self, py: Python<'_>) -> Py<PyGraph> {
        self.graph.clone_ref(py)
    }

    /// Fingerprint of the graph this database belongs to (16 hex digits).
    #[getter]
    fn fingerprint(&self) -> PyResult<String> {
        self.with(|db| format!("{:016x}", db.graph_fingerprint()))
    }

    fn __getitem__(&self, element: &Bound<'_, PyAny>) -> PyResult<f64> {
        let id = self.element_id(element)?;
        self.with(|db| db.value(id).copied())?
            .ok_or_else(|| self.not_calibrated(id))
    }

    /// The value of `element`, or `default` if it is not calibrated.
    #[pyo3(signature = (element, default=None))]
    fn get(
        &self,
        py: Python<'_>,
        element: &Bound<'_, PyAny>,
        default: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let id = self.element_id(element)?;
        match self.with(|db| db.value(id).copied())? {
            Some(v) => Ok(v.into_pyobject(py)?.into_any().unbind()),
            None => Ok(default.unwrap_or_else(|| py.None())),
        }
    }

    fn __contains__(&self, element: &Bound<'_, PyAny>) -> PyResult<bool> {
        match self.element_id(element) {
            Ok(id) => self.with(|db| db.is_calibrated(id)),
            Err(e) if e.is_instance_of::<PyKeyError>(element.py()) => Ok(false),
            Err(e) if e.is_instance_of::<PyTypeError>(element.py()) => Ok(false),
            // An element of another graph is not in this database (as for ElementSet).
            Err(e)
                if e.is_instance_of::<PyValueError>(element.py())
                    && element.is_instance_of::<PyElement>() =>
            {
                Ok(false)
            }
            Err(e) => Err(e),
        }
    }

    fn __len__(&self) -> PyResult<usize> {
        self.with(|db| db.len())
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyIterator>> {
        let ids = self.with(|db| db.calibrated().to_vec())?;
        elements_list(py, &self.arc(), ids)?.into_any().try_iter()
    }

    /// The calibrated values as `{element: value}` (or `{label: value}` with
    /// `labels=True`), in ascending element order.
    #[pyo3(signature = (labels=false))]
    fn to_dict<'py>(&self, py: Python<'py>, labels: bool) -> PyResult<Bound<'py, PyDict>> {
        let graph = self.arc();
        let items: Vec<(ElementId, f64)> =
            self.with(|db| db.iter().map(|(id, e)| (id, e.value)).collect())?;
        let dict = PyDict::new(py);
        for (id, value) in items {
            if labels {
                dict.set_item(graph.label(id), value)?;
            } else {
                dict.set_item(new_element(py, &graph, id)?, value)?;
            }
        }
        Ok(dict)
    }

    /// The index of the committed value among the element's options.
    fn option_index(&self, element: &Bound<'_, PyAny>) -> PyResult<u32> {
        let id = self.element_id(element)?;
        self.with(|db| db.get(id).map(|e| e.option_index))?
            .ok_or_else(|| self.not_calibrated(id))
    }

    /// The index (in the history) of the step that set the element.
    fn step_of(&self, element: &Bound<'_, PyAny>) -> PyResult<u64> {
        let id = self.element_id(element)?;
        self.with(|db| db.get(id).map(|e| e.step))?
            .ok_or_else(|| self.not_calibrated(id))
    }

    /// The append-only history: `StepRecord`s and `Invalidation`s in order.
    #[getter]
    fn history<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let graph = self.arc();
        let records = self.with(|db| db.history().to_vec())?;
        let list = PyList::empty(py);
        for record in &records {
            list.append(history_item(py, &graph, record)?)?;
        }
        Ok(list)
    }

    /// The committed steps in order (invalidations left out).
    fn steps<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let graph = self.arc();
        let steps: Vec<_> = self.with(|db| db.steps().cloned().collect())?;
        let list = PyList::empty(py);
        for step in steps {
            list.append(Bound::new(py, PyStepRecord::new(&graph, step, None))?)?;
        }
        Ok(list)
    }

    /// Number of steps ever committed.
    #[getter]
    fn step_count(&self) -> PyResult<u64> {
        self.with(|db| db.step_count())
    }

    /// The set of calibrated elements.
    #[getter]
    fn calibrated<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyElementSet>> {
        let set = self.with(|db| db.calibrated().clone())?;
        new_set(py, &self.arc(), set)
    }

    /// Writes the database as JSON, atomically.
    fn save(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        let graph = self.arc();
        let result = py.detach(|| self.with(|db| db.save(&graph, &path)));
        result?.map_err(|e| errors::plain(py, e))
    }

    /// Reads a database written by `save`. `GraphMismatchError` if `graph` is not the
    /// graph it was saved for.
    #[classmethod]
    fn load(
        _cls: &Bound<'_, pyo3::types::PyType>,
        py: Python<'_>,
        path: PathBuf,
        graph: &Bound<'_, PyGraph>,
    ) -> PyResult<Self> {
        let g = Arc::clone(&graph.get().inner);
        let db = py
            .detach(|| ParameterDatabase::<f64>::load(&g, &path))
            .map_err(|e| errors::plain(py, e))?;
        Ok(Self::from_core(graph.clone().unbind(), db))
    }

    /// The database as a JSON string (the format of `save`).
    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        let graph = self.arc();
        self.with(|db| db.to_json_string(&graph))?
            .map_err(|e| errors::plain(py, e))
    }

    /// Parses the JSON written by `to_json` or `save`.
    #[classmethod]
    fn from_json(
        _cls: &Bound<'_, pyo3::types::PyType>,
        py: Python<'_>,
        graph: &Bound<'_, PyGraph>,
        text: &str,
    ) -> PyResult<Self> {
        let db = ParameterDatabase::<f64>::from_json_str(&graph.get().inner, text)
            .map_err(|e| errors::plain(py, e))?;
        Ok(Self::from_core(graph.clone().unbind(), db))
    }

    /// Whether `other` holds the same values and the same history, ignoring timings.
    fn same_outcome(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let other = other
            .cast::<PyDatabase>()
            .map_err(|_| PyTypeError::new_err("other must be a ParameterDatabase"))?;
        let other = other.get();
        if std::ptr::eq(self, other) {
            return self.with(|_| true);
        }
        let (a, b) = (self.lock(), other.lock());
        match (&*a, &*b) {
            (State::Ready(a), State::Ready(b)) => Ok(a.same_outcome(b)),
            _ => Err(PyRuntimeError::new_err(IN_USE)),
        }
    }

    /// A copy of the database (same graph, same values, same history).
    fn copy(&self, py: Python<'_>) -> PyResult<Self> {
        let db = self.with(|db| db.clone())?;
        Ok(Self::from_core(self.graph.clone_ref(py), db))
    }

    /// Applies the entries of a checkpoint journal that this database does not have yet (the
    /// steps, and the removals that `recalibrate` journaled) and returns how many were
    /// applied. The values come from `model.options`.
    ///
    /// Steps are recognised by their `(pass, subgoal, centre)`, so a replay onto an empty
    /// database reproduces the journaled run, a replay onto a database saved part-way through
    /// the run (before, inside or after a `recalibrate`) applies exactly what is missing, and
    /// a second replay applies nothing. The journals of the several runs of one database can
    /// be replayed one after the other.
    fn replay_journal(
        &self,
        py: Python<'_>,
        path: PathBuf,
        model: &Bound<'_, PyAny>,
    ) -> PyResult<usize> {
        let graph = self.arc();
        let failures = Arc::new(FailureSlot::default());
        let model = PyErrorModel::new(
            model,
            self.graph.clone_ref(py),
            Arc::new(OptionTables::new(graph.len())),
            Arc::clone(&failures),
            0,
        )?;
        let entries = read_journal_entries(&path, &graph).map_err(|e| errors::plain(py, e))?;
        let mut db = self.lend()?;
        let mut failure: Option<Failure> = None;
        let result = db.replay_entries(&graph, &entries, |element| {
            match model.fetch_options(py, element) {
                Ok(values) => values,
                Err(f) => {
                    failure.get_or_insert(f);
                    Vec::new()
                }
            }
        });
        self.restore(db);
        match (result, failure) {
            (_, Some(f)) => Err(f.into_pyerr(py, &graph)),
            (Ok(applied), None) => Ok(applied),
            (Err(e), None) => Err(errors::plain(py, e)),
        }
    }

    fn __repr__(&self) -> String {
        match &*self.lock() {
            State::Ready(db) => format!(
                "ParameterDatabase(calibrated={}, steps={})",
                db.len(),
                db.step_count()
            ),
            State::Lent => "ParameterDatabase(<in use by a running calibration>)".into(),
        }
    }
}

/// Reads a checkpoint journal written by `Snake.calibrate(..., checkpoint=path)`.
///
/// A torn last line is ignored. The records carry option indices, not values.
#[pyfunction]
#[pyo3(name = "read_journal")]
pub fn py_read_journal<'py>(
    py: Python<'py>,
    path: PathBuf,
    graph: &Bound<'py, PyGraph>,
) -> PyResult<Bound<'py, PyList>> {
    let g = Arc::clone(&graph.get().inner);
    let records = read_journal(&path, &g).map_err(|e| errors::plain(py, e))?;
    let list = PyList::empty(py);
    for record in records {
        list.append(Bound::new(py, PyStepRecord::new(&g, record, None))?)?;
    }
    Ok(list)
}
