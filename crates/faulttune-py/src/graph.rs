//! Graph, element and element-set wrappers (`ProcessorGraph`, `Element`, `ElementSet`,
//! `ElementKind`, `GraphBuilder`).

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use faulttune::{EdgeType, ElementId, ElementKind, ElementSet, GraphBuilder, ProcessorGraph};
use numpy::{PyArray1, PyArrayMethods};
use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyIndexError, PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyIterator, PyList, PyString};
use pyo3::IntoPyObjectExt;

use crate::errors;

/// Largest lattice (in nodes) that `square_lattice` builds.
const MAX_LATTICE_NODES: usize = 4_000_000;

/// What an element is: a node or an edge of some type.
#[pyclass(
    eq,
    eq_int,
    frozen,
    hash,
    module = "faulttune",
    name = "ElementKind",
    from_py_object
)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum PyElementKind {
    /// A node, usually a qubit.
    #[pyo3(name = "NODE")]
    Node,
    /// A designed coupling.
    #[pyo3(name = "ENGINEERED_EDGE")]
    EngineeredEdge,
    /// Unwanted crosstalk: carries constraints but has no parameters of its own.
    #[pyo3(name = "PARASITIC_EDGE")]
    ParasiticEdge,
    /// A user-defined edge type (see `Element.custom_type`).
    #[pyo3(name = "CUSTOM_EDGE")]
    CustomEdge,
}

impl PyElementKind {
    /// The Python kind of a core kind.
    pub fn of(kind: ElementKind) -> Self {
        match kind {
            ElementKind::Node => Self::Node,
            ElementKind::Edge(EdgeType::Engineered) => Self::EngineeredEdge,
            ElementKind::Edge(EdgeType::Parasitic) => Self::ParasiticEdge,
            ElementKind::Edge(EdgeType::Custom(_)) => Self::CustomEdge,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Node => "NODE",
            Self::EngineeredEdge => "ENGINEERED_EDGE",
            Self::ParasiticEdge => "PARASITIC_EDGE",
            Self::CustomEdge => "CUSTOM_EDGE",
        }
    }
}

/// A node or an edge of a processor graph: a hashable, comparable handle.
///
/// Elements are equal when they have the same id in the same graph; elements of different
/// graphs are never equal. They order by id within one graph. `str(element)` is the label.
#[pyclass(frozen, module = "faulttune", name = "Element")]
pub struct PyElement {
    pub graph: Arc<ProcessorGraph>,
    pub id: ElementId,
}

#[pymethods]
impl PyElement {
    /// The label, for example `"q1_3"` or `"q1_3-q1_4"`.
    #[getter]
    fn label(&self) -> &str {
        self.graph.label(self.id)
    }

    /// The kind of the element.
    #[getter]
    fn kind(&self) -> PyElementKind {
        PyElementKind::of(self.graph.kind(self.id))
    }

    /// The dense integer id of the element within its graph.
    #[getter]
    fn index(&self) -> u32 {
        self.id.raw()
    }

    /// The two end nodes of an edge, or `None` for a node.
    #[getter]
    fn endpoints(&self, py: Python<'_>) -> PyResult<Option<(Py<PyElement>, Py<PyElement>)>> {
        match self.graph.endpoints(self.id) {
            Some([a, b]) => Ok(Some((
                new_element(py, &self.graph, a)?.unbind(),
                new_element(py, &self.graph, b)?.unbind(),
            ))),
            None => Ok(None),
        }
    }

    /// The number of a user-defined edge type, or `None` for other elements.
    #[getter]
    fn custom_type(&self) -> Option<u16> {
        match self.graph.kind(self.id) {
            ElementKind::Edge(EdgeType::Custom(n)) => Some(n),
            _ => None,
        }
    }

    /// Whether this element is a node.
    #[getter]
    fn is_node(&self) -> bool {
        self.graph.kind(self.id).is_node()
    }

    /// Whether this element is an edge.
    #[getter]
    fn is_edge(&self) -> bool {
        self.graph.kind(self.id).is_edge()
    }

    fn __str__(&self) -> &str {
        self.graph.label(self.id)
    }

    fn __repr__(&self) -> String {
        format!(
            "Element({:?}, kind={})",
            self.graph.label(self.id),
            PyElementKind::of(self.graph.kind(self.id)).name()
        )
    }

    fn __hash__(&self) -> isize {
        let mut hasher = DefaultHasher::new();
        (Arc::as_ptr(&self.graph) as usize).hash(&mut hasher);
        self.id.hash(&mut hasher);
        hasher.finish() as isize
    }

    fn __richcmp__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        op: CompareOp,
    ) -> PyResult<Py<PyAny>> {
        let Ok(other) = other.cast::<PyElement>() else {
            return Ok(py.NotImplemented());
        };
        let other = other.get();
        let same_graph = Arc::ptr_eq(&self.graph, &other.graph);
        match op {
            CompareOp::Eq => (same_graph && self.id == other.id).into_py_any(py),
            CompareOp::Ne => (!(same_graph && self.id == other.id)).into_py_any(py),
            _ if !same_graph => Ok(py.NotImplemented()),
            CompareOp::Lt => (self.id < other.id).into_py_any(py),
            CompareOp::Le => (self.id <= other.id).into_py_any(py),
            CompareOp::Gt => (self.id > other.id).into_py_any(py),
            CompareOp::Ge => (self.id >= other.id).into_py_any(py),
        }
    }
}

/// Creates the Python handle of element `id`.
pub fn new_element<'py>(
    py: Python<'py>,
    graph: &Arc<ProcessorGraph>,
    id: ElementId,
) -> PyResult<Bound<'py, PyElement>> {
    Bound::new(
        py,
        PyElement {
            graph: Arc::clone(graph),
            id,
        },
    )
}

/// A Python list of the elements `ids`.
pub fn elements_list<'py>(
    py: Python<'py>,
    graph: &Arc<ProcessorGraph>,
    ids: impl IntoIterator<Item = ElementId>,
) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for id in ids {
        list.append(new_element(py, graph, id)?)?;
    }
    Ok(list)
}

/// A new Python element set.
pub fn new_set<'py>(
    py: Python<'py>,
    graph: &Arc<ProcessorGraph>,
    set: ElementSet,
) -> PyResult<Bound<'py, PyElementSet>> {
    Bound::new(
        py,
        PyElementSet {
            graph: Arc::clone(graph),
            set,
        },
    )
}

fn type_name(obj: &Bound<'_, PyAny>) -> String {
    obj.get_type()
        .name()
        .map(|n| n.to_string())
        .unwrap_or_else(|_| "object".into())
}

/// The element `obj` of `graph`. `what` names the argument in error messages.
///
/// `TypeError` if `obj` is not an `Element`, `ValueError` if it belongs to another graph.
pub fn extract_element(
    graph: &Arc<ProcessorGraph>,
    obj: &Bound<'_, PyAny>,
    what: &str,
) -> PyResult<ElementId> {
    let element = obj.cast::<PyElement>().map_err(|_| {
        PyTypeError::new_err(format!(
            "{what} must be a faulttune.Element, not {}",
            type_name(obj)
        ))
    })?;
    let element = element.get();
    if !Arc::ptr_eq(graph, &element.graph) {
        return Err(PyValueError::new_err(format!(
            "{what}: element {:?} belongs to a different graph",
            element.graph.label(element.id)
        )));
    }
    Ok(element.id)
}

/// Like [`extract_element`] but also accepts an element label (a `str`).
pub fn extract_element_or_label(
    graph: &Arc<ProcessorGraph>,
    obj: &Bound<'_, PyAny>,
    what: &str,
) -> PyResult<ElementId> {
    if let Ok(label) = obj.cast::<PyString>() {
        let label = label.to_str()?;
        return graph
            .id_of(label)
            .ok_or_else(|| PyKeyError::new_err(format!("no element labelled {label:?}")));
    }
    extract_element(graph, obj, what)
}

/// A set of elements of `graph` from an `ElementSet` or any iterable of `Element`s.
pub fn extract_set(
    graph: &Arc<ProcessorGraph>,
    obj: &Bound<'_, PyAny>,
    what: &str,
) -> PyResult<ElementSet> {
    if let Ok(set) = obj.cast::<PyElementSet>() {
        let set = set.get();
        if !Arc::ptr_eq(graph, &set.graph) {
            return Err(PyValueError::new_err(format!(
                "{what} is an ElementSet of a different graph"
            )));
        }
        return Ok(set.set.clone());
    }
    if obj.is_instance_of::<PyString>() || obj.is_instance_of::<PyElement>() {
        return Err(PyTypeError::new_err(format!(
            "{what} must be an ElementSet or an iterable of Elements, not {}",
            type_name(obj)
        )));
    }
    let iter = obj.try_iter().map_err(|_| {
        PyTypeError::new_err(format!(
            "{what} must be an ElementSet or an iterable of Elements, not {}",
            type_name(obj)
        ))
    })?;
    let mut set = ElementSet::new(graph.len());
    for item in iter {
        set.insert(extract_element(
            graph,
            &item?,
            &format!("an item of {what}"),
        )?);
    }
    Ok(set)
}

/// The elements of `obj` (an `ElementSet` or an iterable of `Element`s) in the order given.
pub fn extract_element_list(
    graph: &Arc<ProcessorGraph>,
    obj: &Bound<'_, PyAny>,
    what: &str,
) -> PyResult<Vec<ElementId>> {
    if let Ok(set) = obj.cast::<PyElementSet>() {
        let set = set.get();
        if !Arc::ptr_eq(graph, &set.graph) {
            return Err(PyValueError::new_err(format!(
                "{what} is an ElementSet of a different graph"
            )));
        }
        return Ok(set.set.to_vec());
    }
    if obj.is_instance_of::<PyString>() || obj.is_instance_of::<PyElement>() {
        return Err(PyTypeError::new_err(format!(
            "{what} must be an iterable of Elements, not {}",
            type_name(obj)
        )));
    }
    let iter = obj.try_iter().map_err(|_| {
        PyTypeError::new_err(format!(
            "{what} must be an iterable of Elements, not {}",
            type_name(obj)
        ))
    })?;
    iter.map(|item| extract_element(graph, &item?, &format!("an item of {what}")))
        .collect()
}

/// An immutable set of elements of one graph.
///
/// Supports `len`, iteration (ascending id order), `in`, `|`, `&`, `-`, `^`, the subset
/// comparisons and `isdisjoint`. Operators accept another `ElementSet` or any iterable of
/// `Element`s of the same graph.
#[pyclass(frozen, module = "faulttune", name = "ElementSet")]
pub struct PyElementSet {
    pub graph: Arc<ProcessorGraph>,
    pub set: ElementSet,
}

impl PyElementSet {
    fn operand(&self, other: &Bound<'_, PyAny>) -> PyResult<ElementSet> {
        extract_set(&self.graph, other, "the other operand")
    }

    fn derived<'py>(&self, py: Python<'py>, set: ElementSet) -> PyResult<Bound<'py, PyElementSet>> {
        new_set(py, &self.graph, set)
    }
}

#[pymethods]
impl PyElementSet {
    /// The graph these elements belong to.
    #[getter]
    fn graph(&self) -> PyGraph {
        PyGraph {
            inner: Arc::clone(&self.graph),
        }
    }

    fn __len__(&self) -> usize {
        self.set.len()
    }

    fn __bool__(&self) -> bool {
        !self.set.is_empty()
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyIterator>> {
        elements_list(py, &self.graph, self.set.iter())?
            .into_any()
            .try_iter()
    }

    fn __contains__(&self, item: &Bound<'_, PyAny>) -> bool {
        match item.cast::<PyElement>() {
            Ok(e) => {
                let e = e.get();
                Arc::ptr_eq(&self.graph, &e.graph) && self.set.contains(e.id)
            }
            Err(_) => false,
        }
    }

    fn __or__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyElementSet>> {
        self.derived(py, self.set.union(&self.operand(other)?))
    }

    fn __ror__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyElementSet>> {
        self.__or__(py, other)
    }

    fn __and__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyElementSet>> {
        self.derived(py, self.set.intersection(&self.operand(other)?))
    }

    fn __rand__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyElementSet>> {
        self.__and__(py, other)
    }

    fn __sub__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyElementSet>> {
        self.derived(py, self.set.difference(&self.operand(other)?))
    }

    fn __rsub__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyElementSet>> {
        self.derived(py, self.operand(other)?.difference(&self.set))
    }

    fn __xor__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyElementSet>> {
        let other = self.operand(other)?;
        let both = self.set.intersection(&other);
        self.derived(py, self.set.union(&other).difference(&both))
    }

    fn __rxor__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyElementSet>> {
        self.__xor__(py, other)
    }

    fn __richcmp__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        op: CompareOp,
    ) -> PyResult<Py<PyAny>> {
        let Ok(other) = other.cast::<PyElementSet>() else {
            return Ok(py.NotImplemented());
        };
        let other = other.get();
        if !Arc::ptr_eq(&self.graph, &other.graph) {
            return match op {
                CompareOp::Eq => false.into_py_any(py),
                CompareOp::Ne => true.into_py_any(py),
                _ => Ok(py.NotImplemented()),
            };
        }
        let (a, b) = (&self.set, &other.set);
        match op {
            CompareOp::Eq => (a == b).into_py_any(py),
            CompareOp::Ne => (a != b).into_py_any(py),
            CompareOp::Le => a.is_subset(b).into_py_any(py),
            CompareOp::Lt => (a != b && a.is_subset(b)).into_py_any(py),
            CompareOp::Ge => b.is_subset(a).into_py_any(py),
            CompareOp::Gt => (a != b && b.is_subset(a)).into_py_any(py),
        }
    }

    fn __hash__(&self) -> isize {
        let mut hasher = DefaultHasher::new();
        (Arc::as_ptr(&self.graph) as usize).hash(&mut hasher);
        self.set.hash(&mut hasher);
        hasher.finish() as isize
    }

    /// Whether the set shares no element with `other` (an `ElementSet` or an iterable).
    fn isdisjoint(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self.set.is_disjoint(&self.operand(other)?))
    }

    /// The elements as a list, in ascending id order.
    fn to_list<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        elements_list(py, &self.graph, self.set.iter())
    }

    fn __repr__(&self) -> String {
        const SHOWN: usize = 8;
        let labels: Vec<&str> = self
            .set
            .iter()
            .take(SHOWN)
            .map(|id| self.graph.label(id))
            .collect();
        let more = self.set.len().saturating_sub(SHOWN);
        if more > 0 {
            format!(
                "ElementSet({} elements: {labels:?} and {more} more)",
                self.set.len()
            )
        } else {
            format!("ElementSet({labels:?})")
        }
    }
}

/// An immutable processor graph: nodes and typed edges, all addressed as elements.
///
/// Build one with `ProcessorGraph.square_lattice`, `ProcessorGraph.from_json` or `GraphBuilder`.
#[pyclass(frozen, module = "faulttune", name = "ProcessorGraph")]
pub struct PyGraph {
    pub inner: Arc<ProcessorGraph>,
}

impl PyGraph {
    fn handle(&self, py: Python<'_>, id: ElementId) -> PyResult<Py<PyElement>> {
        Ok(new_element(py, &self.inner, id)?.unbind())
    }

    fn kind_set(&self, kinds: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<PyElementKind>> {
        if let Ok(kind) = kinds.extract::<PyElementKind>() {
            return Ok(vec![kind]);
        }
        let iter = kinds.try_iter().map_err(|_| {
            PyTypeError::new_err(format!(
                "{what} must be an ElementKind or an iterable of ElementKinds, not {}",
                type_name(kinds)
            ))
        })?;
        iter.map(|k| {
            k?.extract::<PyElementKind>()
                .map_err(|_| PyTypeError::new_err(format!("{what} must contain only ElementKinds")))
        })
        .collect()
    }
}

#[pymethods]
impl PyGraph {
    /// A `rows x cols` square lattice.
    ///
    /// Nodes `q{row}_{col}` come first (row-major), then the nearest-neighbour engineered
    /// edges, then the diagonal parasitic edges.
    #[staticmethod]
    fn square_lattice(py: Python<'_>, rows: i64, cols: i64) -> PyResult<Self> {
        for (name, value) in [("rows", rows), ("cols", cols)] {
            if value < 0 {
                return Err(PyValueError::new_err(format!(
                    "{name} must be non-negative, got {value}"
                )));
            }
        }
        let nodes = (rows as usize).checked_mul(cols as usize);
        if nodes.is_none_or(|n| n > MAX_LATTICE_NODES) {
            return Err(PyValueError::new_err(format!(
                "a {rows} x {cols} lattice is too large (at most {MAX_LATTICE_NODES} nodes)"
            )));
        }
        Ok(Self {
            // Building a big lattice takes seconds: let other Python threads run meanwhile.
            inner: Arc::new(
                py.detach(|| ProcessorGraph::square_lattice(rows as usize, cols as usize)),
            ),
        })
    }

    /// Parses a graph from its JSON description (see `to_json`).
    #[staticmethod]
    fn from_json(py: Python<'_>, text: &str) -> PyResult<Self> {
        let graph = py
            .detach(|| ProcessorGraph::from_json(text))
            .map_err(|e| errors::plain(py, e))?;
        Ok(Self {
            inner: Arc::new(graph),
        })
    }

    /// The JSON description of the graph; `from_json` reads it back (same ids, same
    /// fingerprint).
    fn to_json(&self) -> String {
        self.inner.to_json()
    }

    /// The node `q{row}_{col}` of a lattice-labelled graph (`KeyError` if there is none).
    fn node(&self, py: Python<'_>, row: i64, col: i64) -> PyResult<Py<PyElement>> {
        let found = (row >= 0 && col >= 0)
            .then(|| self.inner.lattice_node(row as usize, col as usize))
            .flatten();
        match found {
            Some(id) => self.handle(py, id),
            None => Err(PyKeyError::new_err(format!(
                "no node labelled q{row}_{col}"
            ))),
        }
    }

    /// The edge joining nodes `a` and `b`, optionally of the given kind (`KeyError` if
    /// there is none). `a` and `b` are nodes of this graph or their labels.
    #[pyo3(signature = (a, b, kind=None))]
    fn edge(
        &self,
        py: Python<'_>,
        a: &Bound<'_, PyAny>,
        b: &Bound<'_, PyAny>,
        kind: Option<PyElementKind>,
    ) -> PyResult<Py<PyElement>> {
        let graph = &self.inner;
        let (na, nb) = (
            extract_element_or_label(graph, a, "a")?,
            extract_element_or_label(graph, b, "b")?,
        );
        for (name, n) in [("a", na), ("b", nb)] {
            if !graph.kind(n).is_node() {
                return Err(PyValueError::new_err(format!(
                    "{name} must be a node, but {:?} is an edge",
                    graph.label(n)
                )));
            }
        }
        let wanted = match kind {
            None => None,
            Some(PyElementKind::Node) => {
                return Err(PyValueError::new_err("kind must be an edge kind, not NODE"))
            }
            Some(k) => Some(k),
        };
        let found = graph.incident(na).iter().copied().find(|&e| {
            let Some([x, y]) = graph.endpoints(e) else {
                return false;
            };
            ((x == na && y == nb) || (x == nb && y == na))
                && wanted.is_none_or(|k| PyElementKind::of(graph.kind(e)) == k)
        });
        match found {
            Some(e) => self.handle(py, e),
            None => Err(PyKeyError::new_err(format!(
                "no edge between {:?} and {:?}",
                graph.label(na),
                graph.label(nb)
            ))),
        }
    }

    /// The element with this label (`KeyError` if there is none).
    fn element(&self, py: Python<'_>, label: &str) -> PyResult<Py<PyElement>> {
        match self.inner.id_of(label) {
            Some(id) => PyGraph::handle(self, py, id),
            None => Err(PyKeyError::new_err(format!(
                "no element labelled {label:?}"
            ))),
        }
    }

    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyElement>> {
        if let Ok(label) = key.cast::<PyString>() {
            return self.element(py, label.to_str()?);
        }
        if let Ok(i) = key.extract::<i64>() {
            let n = self.inner.len() as i64;
            let j = if i < 0 { i + n } else { i };
            if (0..n).contains(&j) {
                return self.handle(py, ElementId::new(j as u32));
            }
            return Err(PyIndexError::new_err(format!(
                "element index {i} out of range for a graph of {n} elements"
            )));
        }
        Err(PyTypeError::new_err(format!(
            "graph indices must be labels (str) or integers, not {}",
            type_name(key)
        )))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyIterator>> {
        self.elements(py)?.into_any().try_iter()
    }

    fn __contains__(&self, item: &Bound<'_, PyAny>) -> bool {
        if let Ok(label) = item.cast::<PyString>() {
            return label
                .to_str()
                .map(|l| self.inner.id_of(l).is_some())
                .unwrap_or(false);
        }
        match item.cast::<PyElement>() {
            Ok(e) => Arc::ptr_eq(&self.inner, &e.get().graph),
            Err(_) => false,
        }
    }

    /// All nodes, in id order.
    fn nodes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        elements_list(py, &self.inner, self.inner.nodes())
    }

    /// All edges, in id order.
    fn edges<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        elements_list(py, &self.inner, self.inner.edges())
    }

    /// All elements, in id order.
    fn elements<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        elements_list(py, &self.inner, self.inner.ids())
    }

    /// A hash of the labels, kinds and incidence (16 hex digits).
    #[getter]
    fn fingerprint(&self) -> String {
        format!("{:016x}", self.inner.fingerprint())
    }

    /// Layout coordinates as a float32 array of shape `(n_elements, 2)`, or `None`.
    #[getter]
    fn coords<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, numpy::PyArray2<f32>>>> {
        match self.inner.coords() {
            None => Ok(None),
            Some(coords) => {
                let flat: Vec<f32> = coords.iter().flat_map(|c| c.iter().copied()).collect();
                let array = PyArray1::from_vec(py, flat).reshape([coords.len(), 2])?;
                Ok(Some(array))
            }
        }
    }

    /// The set of elements to calibrate.
    ///
    /// An element is selected when its kind is in `include` (default: every kind), not in
    /// `exclude`, and `predicate(element)` is true (default: always). `include` and
    /// `exclude` take an `ElementKind` or an iterable of them.
    #[pyo3(signature = (exclude=None, include=None, predicate=None))]
    fn goal<'py>(
        &self,
        py: Python<'py>,
        exclude: Option<&Bound<'py, PyAny>>,
        include: Option<&Bound<'py, PyAny>>,
        predicate: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyElementSet>> {
        let exclude = match exclude {
            Some(k) if !k.is_none() => self.kind_set(k, "exclude")?,
            _ => Vec::new(),
        };
        let include = match include {
            Some(k) if !k.is_none() => Some(self.kind_set(k, "include")?),
            _ => None,
        };
        let predicate = predicate.filter(|p| !p.is_none());
        if let Some(p) = predicate {
            if !p.is_callable() {
                return Err(PyTypeError::new_err(format!(
                    "predicate must be callable, not {}",
                    type_name(p)
                )));
            }
        }
        let mut set = ElementSet::new(self.inner.len());
        for id in self.inner.ids() {
            let kind = PyElementKind::of(self.inner.kind(id));
            if exclude.contains(&kind) || include.as_ref().is_some_and(|i| !i.contains(&kind)) {
                continue;
            }
            if let Some(p) = predicate {
                if !p.call1((new_element(py, &self.inner, id)?,))?.is_truthy()? {
                    continue;
                }
            }
            set.insert(id);
        }
        new_set(py, &self.inner, set)
    }

    /// The set of every element.
    fn all_elements<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyElementSet>> {
        new_set(py, &self.inner, self.inner.all_elements())
    }

    /// An `ElementSet` of this graph from any iterable of its `Element`s.
    fn element_set<'py>(
        &self,
        py: Python<'py>,
        elements: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyElementSet>> {
        let set = extract_set(&self.inner, elements, "elements")?;
        new_set(py, &self.inner, set)
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .cast::<PyGraph>()
            .is_ok_and(|o| Arc::ptr_eq(&self.inner, &o.get().inner))
    }

    fn __hash__(&self) -> isize {
        (Arc::as_ptr(&self.inner) as usize >> 4) as isize
    }

    fn __repr__(&self) -> String {
        format!(
            "ProcessorGraph(elements={}, nodes={}, edges={}, fingerprint='{:016x}')",
            self.inner.len(),
            self.inner.nodes().count(),
            self.inner.edges().count(),
            self.inner.fingerprint()
        )
    }
}

/// Builds a `ProcessorGraph` of any topology. Elements get ids in insertion order.
#[pyclass(module = "faulttune", name = "GraphBuilder")]
pub struct PyGraphBuilder {
    builder: Option<GraphBuilder>,
}

impl PyGraphBuilder {
    fn builder(&mut self) -> PyResult<&mut GraphBuilder> {
        self.builder
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("this GraphBuilder has been built already"))
    }
}

#[pymethods]
impl PyGraphBuilder {
    #[new]
    fn new() -> Self {
        Self {
            builder: Some(GraphBuilder::new()),
        }
    }

    /// Adds a node and returns its label. `pos` is an optional `(x, y)` layout coordinate;
    /// the graph has coordinates only if every node has one.
    #[pyo3(signature = (label, pos=None))]
    fn add_node(
        &mut self,
        py: Python<'_>,
        label: String,
        pos: Option<(f32, f32)>,
    ) -> PyResult<String> {
        let builder = self.builder()?;
        let result = match pos {
            Some((x, y)) => builder.add_node_at(label.clone(), [x, y]),
            None => builder.add_node(label.clone()),
        };
        result.map_err(|e| errors::plain(py, e))?;
        Ok(label)
    }

    /// Adds an edge between the nodes labelled `a` and `b` and returns its label.
    ///
    /// `kind` is `ENGINEERED_EDGE` (default), `PARASITIC_EDGE` or `CUSTOM_EDGE`; a custom
    /// edge also needs `custom_type`, a number of your choosing. The label defaults to
    /// `"{a}-{b}"`.
    #[pyo3(signature = (a, b, kind=PyElementKind::EngineeredEdge, *, label=None, custom_type=None))]
    fn add_edge(
        &mut self,
        py: Python<'_>,
        a: &str,
        b: &str,
        kind: PyElementKind,
        label: Option<String>,
        custom_type: Option<u16>,
    ) -> PyResult<String> {
        let edge_type = match (kind, custom_type) {
            (PyElementKind::Node, _) => {
                return Err(PyValueError::new_err("kind must be an edge kind, not NODE"))
            }
            (PyElementKind::EngineeredEdge, None) => EdgeType::Engineered,
            (PyElementKind::ParasiticEdge, None) => EdgeType::Parasitic,
            (PyElementKind::CustomEdge, Some(n)) => EdgeType::Custom(n),
            (PyElementKind::CustomEdge, None) => {
                return Err(PyValueError::new_err(
                    "a CUSTOM_EDGE needs custom_type (an integer)",
                ))
            }
            (_, Some(_)) => {
                return Err(PyValueError::new_err(
                    "custom_type is only for kind=CUSTOM_EDGE",
                ))
            }
        };
        let builder = self.builder()?;
        let find = |builder: &GraphBuilder, name: &str, what: &str| {
            builder
                .find_label(name)
                .ok_or_else(|| PyKeyError::new_err(format!("{what}: no node labelled {name:?}")))
        };
        let (ia, ib) = (find(builder, a, "a")?, find(builder, b, "b")?);
        let label = label.unwrap_or_else(|| format!("{a}-{b}"));
        builder
            .add_edge_labelled(label.clone(), ia, ib, edge_type)
            .map_err(|e| errors::plain(py, e))?;
        Ok(label)
    }

    /// Finishes the graph. The builder cannot be used afterwards.
    fn build(&mut self, py: Python<'_>) -> PyResult<PyGraph> {
        let builder = self
            .builder
            .take()
            .ok_or_else(|| PyValueError::new_err("this GraphBuilder has been built already"))?;
        let graph = builder.build().map_err(|e| errors::plain(py, e))?;
        Ok(PyGraph {
            inner: Arc::new(graph),
        })
    }
}
