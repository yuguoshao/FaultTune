//! `AlgorithmSubgraphs`, `StitchPlan` and the `Retry` policy: small configuration objects.

use std::sync::Arc;

use faulttune::{AlgorithmSubgraphs, ElementSet, ProcessorGraph, StitchPlan};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::errors;
use crate::graph::{extract_set, new_set, PyElementSet, PyGraph};

/// Retry a failing model call `times` more times before giving up (`on_model_error`).
#[pyclass(frozen, module = "faulttune", name = "Retry", from_py_object)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PyRetry {
    times: u32,
}

impl PyRetry {
    /// Number of retries.
    pub fn times(&self) -> u32 {
        self.times
    }

    /// A retry policy of `times` retries (at least 1).
    pub fn checked(times: i64) -> PyResult<Self> {
        if times < 1 {
            return Err(PyValueError::new_err(format!(
                "the number of retries must be at least 1, got {times}"
            )));
        }
        u32::try_from(times)
            .map(|times| Self { times })
            .map_err(|_| PyValueError::new_err("the number of retries is too large"))
    }
}

#[pymethods]
impl PyRetry {
    /// `Retry(times)`: call the failing model method again up to `times` more times, then
    /// abort. For flaky hardware measurements.
    #[new]
    #[pyo3(signature = (times=1))]
    fn new(times: i64) -> PyResult<Self> {
        Self::checked(times)
    }

    /// Number of retries after the first failure.
    #[getter(times)]
    fn get_times(&self) -> u32 {
        self.times
    }

    fn __repr__(&self) -> String {
        format!("Retry({})", self.times)
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other.extract::<PyRetry>().is_ok_and(|o| o == *self)
    }

    fn __hash__(&self) -> u64 {
        u64::from(self.times)
    }
}

/// The algorithm subgraphs `A_i`: the elements that are active at the same time at each
/// moment of the target algorithm. Only elements that can be active together constrain each
/// other.
#[pyclass(frozen, module = "faulttune", name = "AlgorithmSubgraphs")]
pub struct PyAlgorithmSubgraphs {
    pub graph: Py<PyGraph>,
    pub inner: AlgorithmSubgraphs,
}

impl PyAlgorithmSubgraphs {
    fn wrap(graph: &Bound<'_, PyGraph>, inner: AlgorithmSubgraphs) -> Self {
        Self {
            graph: graph.clone().unbind(),
            inner,
        }
    }
}

#[pymethods]
impl PyAlgorithmSubgraphs {
    /// Subgraphs from explicit element collections: `AlgorithmSubgraphs(graph, [A_0, A_1,
    /// ...])`, each an `ElementSet` or an iterable of `Element`s. At most 64 subgraphs.
    #[new]
    fn new(
        py: Python<'_>,
        graph: &Bound<'_, PyGraph>,
        subgraphs: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let g = Arc::clone(&graph.get().inner);
        if subgraphs.is_instance_of::<PyElementSet>() {
            return Err(PyTypeError::new_err(
                "subgraphs must be an iterable of ElementSets (or of iterables of Elements), \
                 not a single ElementSet",
            ));
        }
        let iter = subgraphs
            .try_iter()
            .map_err(|_| PyTypeError::new_err("subgraphs must be an iterable of ElementSets"))?;
        let mut sets = Vec::new();
        for item in iter {
            sets.push(extract_set(&g, &item?, "a subgraph")?);
        }
        let inner = AlgorithmSubgraphs::new(sets).map_err(|e| errors::plain(py, e))?;
        Ok(Self::wrap(graph, inner))
    }

    /// A single subgraph holding every element: everything can be active at once.
    #[staticmethod]
    fn all_active(graph: &Bound<'_, PyGraph>) -> Self {
        Self::wrap(graph, AlgorithmSubgraphs::all_active(&graph.get().inner))
    }

    /// The five XEB subgraphs of a `rows x cols` lattice built by
    /// `ProcessorGraph.square_lattice` (the paper's Fig. 3).
    #[staticmethod]
    fn xeb_square_lattice(
        py: Python<'_>,
        graph: &Bound<'_, PyGraph>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Self> {
        let inner = AlgorithmSubgraphs::xeb_square_lattice(&graph.get().inner, rows, cols)
            .map_err(|e| errors::plain(py, e))?;
        Ok(Self::wrap(graph, inner))
    }

    /// The processor graph.
    #[getter]
    fn graph(&self, py: Python<'_>) -> Py<PyGraph> {
        self.graph.clone_ref(py)
    }

    /// The subgraphs as a list of `ElementSet`s.
    #[getter]
    fn subgraphs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let graph = Arc::clone(&self.graph.get().inner);
        let list = PyList::empty(py);
        for set in self.inner.iter() {
            list.append(new_set(py, &graph, set.clone())?)?;
        }
        Ok(list)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("AlgorithmSubgraphs({} subgraphs)", self.inner.len())
    }
}

/// A plan for calibrating a large processor in constraint-disjoint regions in parallel and
/// then stitching them together: `stage_one` (the regions minus the gap) is calibrated first,
/// then the `gap` under the constraints from both sides.
#[pyclass(frozen, module = "faulttune", name = "StitchPlan")]
pub struct PyStitchPlan {
    graph: Arc<ProcessorGraph>,
    pub plan: StitchPlan,
}

impl PyStitchPlan {
    /// Wraps a plan of `graph`.
    pub fn new(graph: Arc<ProcessorGraph>, plan: StitchPlan) -> Self {
        Self { graph, plan }
    }

    /// The graph the plan was made for.
    pub fn graph(&self) -> &Arc<ProcessorGraph> {
        &self.graph
    }
}

#[pymethods]
impl PyStitchPlan {
    /// The regions the plan was made from, as `ElementSet`s.
    #[getter]
    fn regions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for region in &self.plan.regions {
            list.append(new_set(py, &self.graph, region.clone())?)?;
        }
        Ok(list)
    }

    /// Elements calibrated in stage one (the regions minus the gap).
    #[getter]
    fn stage_one<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyElementSet>> {
        new_set(py, &self.graph, self.plan.stage_one.clone())
    }

    /// Elements calibrated in stage two.
    #[getter]
    fn gap<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyElementSet>> {
        new_set(py, &self.graph, self.plan.gap.clone())
    }

    fn __repr__(&self) -> String {
        format!(
            "StitchPlan(regions={}, stage_one={}, gap={})",
            self.plan.regions.len(),
            self.plan.stage_one.len(),
            self.plan.gap.len()
        )
    }
}

/// The regions of a stitch call as element sets.
pub fn extract_regions(
    graph: &Arc<ProcessorGraph>,
    regions: &Bound<'_, PyAny>,
) -> PyResult<Vec<ElementSet>> {
    if regions.is_instance_of::<PyElementSet>() {
        return Err(PyTypeError::new_err(
            "regions must be an iterable of ElementSets (or of iterables of Elements), not a \
             single ElementSet",
        ));
    }
    let iter = regions.try_iter().map_err(|_| {
        PyTypeError::new_err(
            "regions must be a StitchPlan or an iterable of ElementSets (or of iterables of \
             Elements)",
        )
    })?;
    let mut out = Vec::new();
    for (i, region) in iter.enumerate() {
        out.push(extract_set(graph, &region?, &format!("region {i}"))?);
    }
    Ok(out)
}
