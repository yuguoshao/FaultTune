//! The processor graph: nodes and typed edges, all addressed as *elements*.
//!
//! Every node and every edge is an element with a dense integer [`ElementId`], so
//! per-element data can live in plain vectors (ARCHITECTURE §3.1). Graphs are
//! immutable once built; use [`GraphBuilder`] or a ready-made builder such as
//! [`ProcessorGraph::square_lattice`].

mod builder;
#[cfg(feature = "serde")]
mod json;
mod lattice;

use std::collections::HashMap;
use std::fmt;

pub use builder::GraphBuilder;

use crate::sets::ElementSet;

/// Dense identifier of a node or an edge within one [`ProcessorGraph`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct ElementId(pub(crate) u32);

impl ElementId {
    /// Wraps a raw index. The caller must make sure it belongs to the graph it is used with.
    #[inline]
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    /// The ID as a vector index.
    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// The raw `u32`.
    #[inline]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

impl fmt::Display for ElementId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.0)
    }
}

/// The type of an edge.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum EdgeType {
    /// A designed coupling, for example a coupler used for a two-qubit gate.
    Engineered,
    /// Unwanted crosstalk. It has no parameters of its own but carries constraints.
    Parasitic,
    /// A user-defined edge type, identified by a number of the user's choosing.
    Custom(u16),
}

/// Whether an element is a node or an edge (and if so, of which type).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ElementKind {
    /// A node, usually a qubit.
    Node,
    /// An edge of the given type.
    Edge(EdgeType),
}

impl ElementKind {
    /// `true` for [`ElementKind::Node`].
    pub fn is_node(self) -> bool {
        matches!(self, Self::Node)
    }

    /// `true` for any edge kind.
    pub fn is_edge(self) -> bool {
        matches!(self, Self::Edge(_))
    }

    /// The edge type, or `None` for a node.
    pub fn edge_type(self) -> Option<EdgeType> {
        match self {
            Self::Node => None,
            Self::Edge(t) => Some(t),
        }
    }
}

/// A borrowed view of one element of a graph.
#[derive(Copy, Clone)]
pub struct Element<'g> {
    graph: &'g ProcessorGraph,
    id: ElementId,
}

impl<'g> Element<'g> {
    /// The element's ID.
    pub fn id(&self) -> ElementId {
        self.id
    }

    /// The element's kind.
    pub fn kind(&self) -> ElementKind {
        self.graph.kind(self.id)
    }

    /// The element's label, for example `"q1_3"` or `"q1_3-q1_4"`.
    pub fn label(&self) -> &'g str {
        self.graph.label(self.id)
    }

    /// The two endpoints, if this is an edge.
    pub fn endpoints(&self) -> Option<[ElementId; 2]> {
        self.graph.endpoints(self.id)
    }

    /// `true` if this is a node.
    pub fn is_node(&self) -> bool {
        self.kind().is_node()
    }

    /// `true` if this is an edge.
    pub fn is_edge(&self) -> bool {
        self.kind().is_edge()
    }

    /// The graph this element belongs to.
    pub fn graph(&self) -> &'g ProcessorGraph {
        self.graph
    }
}

impl fmt::Debug for Element<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Element({} {:?} {:?})",
            self.id,
            self.label(),
            self.kind()
        )
    }
}

/// Compressed rows: row `i` is `data[offsets[i]..offsets[i + 1]]`.
#[derive(Clone, Debug, Default)]
struct Csr {
    offsets: Vec<u32>,
    data: Vec<ElementId>,
}

impl Csr {
    fn row(&self, i: usize) -> &[ElementId] {
        &self.data[self.offsets[i] as usize..self.offsets[i + 1] as usize]
    }
}

/// An immutable processor graph.
///
/// Nodes and edges are *elements*. The *incidence graph* links each node to its
/// incident edges; distances between elements (ARCHITECTURE §3.2) are measured on it.
#[derive(Clone, Debug)]
pub struct ProcessorGraph {
    kinds: Vec<ElementKind>,
    labels: Vec<String>,
    by_label: HashMap<String, ElementId>,
    endpoints: Vec<Option<[ElementId; 2]>>,
    /// node → incident edges (ascending ID); edge → its two endpoints `[a, b]`.
    incidence: Csr,
    coords: Option<Vec<[f32; 2]>>,
    fingerprint: u64,
}

impl ProcessorGraph {
    pub(crate) fn from_parts(
        kinds: Vec<ElementKind>,
        labels: Vec<String>,
        endpoints: Vec<Option<[ElementId; 2]>>,
        coords: Option<Vec<[f32; 2]>>,
    ) -> Self {
        let n = kinds.len();
        let mut by_label = HashMap::with_capacity(n);
        for (i, label) in labels.iter().enumerate() {
            by_label.insert(label.clone(), ElementId(i as u32));
        }
        // Incidence rows: nodes list their incident edges, edges list their endpoints.
        let mut degree = vec![0u32; n];
        for ep in endpoints.iter().flatten() {
            degree[ep[0].index()] += 1;
            degree[ep[1].index()] += 1;
        }
        let mut offsets = Vec::with_capacity(n + 1);
        offsets.push(0u32);
        let mut total = 0u32;
        for i in 0..n {
            total += if endpoints[i].is_some() { 2 } else { degree[i] };
            offsets.push(total);
        }
        let mut data = vec![ElementId(0); total as usize];
        let mut fill = offsets.clone();
        for i in 0..n {
            if let Some([a, b]) = endpoints[i] {
                let at = fill[i] as usize;
                data[at] = a;
                data[at + 1] = b;
                fill[i] += 2;
                for node in [a, b] {
                    let at = fill[node.index()] as usize;
                    data[at] = ElementId(i as u32);
                    fill[node.index()] += 1;
                }
            }
        }
        let fingerprint = fingerprint(&kinds, &labels, &endpoints);
        Self {
            kinds,
            labels,
            by_label,
            endpoints,
            incidence: Csr { offsets, data },
            coords,
            fingerprint,
        }
    }

    /// A [`GraphBuilder`] for arbitrary topologies.
    pub fn builder() -> GraphBuilder {
        GraphBuilder::new()
    }

    /// A `rows × cols` square lattice.
    ///
    /// Nodes `q{row}_{col}` get IDs `0..rows*cols` in row-major order. After them come
    /// the nearest-neighbour **engineered** edges (for each node in row-major order: the
    /// edge to its right neighbour, then the edge to its lower neighbour), then the
    /// diagonal **parasitic** edges (for each node in row-major order: the edge to its
    /// lower-right neighbour, then to its lower-left neighbour). An edge is labelled
    /// `"{a}-{b}"` after its endpoints, lower ID first, for example `"q1_3-q1_4"`.
    pub fn square_lattice(rows: usize, cols: usize) -> Self {
        lattice::square_lattice(rows, cols)
    }

    /// Number of elements (nodes plus edges).
    #[inline]
    pub fn len(&self) -> usize {
        self.kinds.len()
    }

    /// Whether the graph has no elements.
    pub fn is_empty(&self) -> bool {
        self.kinds.is_empty()
    }

    /// The kind of an element.
    #[inline]
    pub fn kind(&self, id: ElementId) -> ElementKind {
        self.kinds[id.index()]
    }

    /// The label of an element.
    #[inline]
    pub fn label(&self, id: ElementId) -> &str {
        &self.labels[id.index()]
    }

    /// Looks an element up by label.
    pub fn id_of(&self, label: &str) -> Option<ElementId> {
        self.by_label.get(label).copied()
    }

    /// The endpoints `[a, b]` of an edge, or `None` for a node.
    #[inline]
    pub fn endpoints(&self, id: ElementId) -> Option<[ElementId; 2]> {
        self.endpoints[id.index()]
    }

    /// A view of one element.
    pub fn element(&self, id: ElementId) -> Element<'_> {
        assert!(id.index() < self.len(), "element {id} is not in this graph");
        Element { graph: self, id }
    }

    /// All elements in ID order.
    pub fn elements(&self) -> impl Iterator<Item = Element<'_>> + '_ {
        (0..self.len() as u32).map(move |i| Element {
            graph: self,
            id: ElementId(i),
        })
    }

    /// All element IDs in ascending order.
    pub fn ids(&self) -> impl Iterator<Item = ElementId> + '_ {
        (0..self.len() as u32).map(ElementId)
    }

    /// IDs of all nodes in ascending order.
    pub fn nodes(&self) -> impl Iterator<Item = ElementId> + '_ {
        self.ids().filter(move |&id| self.kind(id).is_node())
    }

    /// IDs of all edges in ascending order.
    pub fn edges(&self) -> impl Iterator<Item = ElementId> + '_ {
        self.ids().filter(move |&id| self.kind(id).is_edge())
    }

    /// The neighbours of `id` in the *incidence graph*: for a node, its incident
    /// edges (ascending ID); for an edge, its two endpoints `[a, b]`.
    #[inline]
    pub fn incident(&self, id: ElementId) -> &[ElementId] {
        self.incidence.row(id.index())
    }

    /// The first edge joining nodes `a` and `b` (in either order), optionally of a given type.
    pub fn edge_between(
        &self,
        a: ElementId,
        b: ElementId,
        edge_type: Option<EdgeType>,
    ) -> Option<ElementId> {
        if !self.kind(a).is_node() {
            return None;
        }
        self.incident(a).iter().copied().find(|&e| {
            let [x, y] = self.endpoints[e.index()].expect("incident element of a node is an edge");
            ((x == a && y == b) || (x == b && y == a))
                && edge_type.is_none_or(|t| self.kind(e) == ElementKind::Edge(t))
        })
    }

    /// The node labelled `q{row}_{col}`, the naming used by [`ProcessorGraph::square_lattice`].
    pub fn lattice_node(&self, row: usize, col: usize) -> Option<ElementId> {
        self.id_of(&format!("q{row}_{col}"))
    }

    /// Optional layout coordinates `[x, y]` for every element (edges sit at the midpoint
    /// of their endpoints), for plotting. `None` if the graph has no layout.
    pub fn coords(&self) -> Option<&[[f32; 2]]> {
        self.coords.as_deref()
    }

    /// A hash of the labels, kinds and incidence. Saved calibrations record it so that
    /// loading one onto the wrong device fails loudly. Layout coordinates are not hashed.
    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    /// The set of elements for which `predicate` is true.
    ///
    /// ```
    /// use faulttune::{EdgeType, ElementKind, ProcessorGraph};
    /// let graph = ProcessorGraph::square_lattice(4, 4);
    /// let goal = graph.goal(|e| e.kind() != ElementKind::Edge(EdgeType::Parasitic));
    /// assert_eq!(goal.len(), 16 + 24);
    /// ```
    pub fn goal(&self, predicate: impl Fn(Element<'_>) -> bool) -> ElementSet {
        let mut set = ElementSet::new(self.len());
        for element in self.elements() {
            if predicate(element) {
                set.insert(element.id());
            }
        }
        set
    }

    /// The set of elements whose kind is *not* in `excluded`.
    pub fn goal_excluding(&self, excluded: &[ElementKind]) -> ElementSet {
        self.goal(|e| !excluded.contains(&e.kind()))
    }

    /// The set of every element.
    pub fn all_elements(&self) -> ElementSet {
        ElementSet::full(self.len())
    }
}

/// FNV-1a over the structural description of the graph (stable across platforms and releases).
fn fingerprint(
    kinds: &[ElementKind],
    labels: &[String],
    endpoints: &[Option<[ElementId; 2]>],
) -> u64 {
    struct Fnv(u64);
    impl Fnv {
        fn write(&mut self, bytes: &[u8]) {
            for &b in bytes {
                self.0 ^= u64::from(b);
                self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    }
    let mut h = Fnv(0xcbf2_9ce4_8422_2325);
    h.write(b"faulttune.graph/1");
    h.write(&(kinds.len() as u64).to_le_bytes());
    for i in 0..kinds.len() {
        match kinds[i] {
            ElementKind::Node => h.write(&[0]),
            ElementKind::Edge(EdgeType::Engineered) => h.write(&[1]),
            ElementKind::Edge(EdgeType::Parasitic) => h.write(&[2]),
            ElementKind::Edge(EdgeType::Custom(n)) => {
                h.write(&[3]);
                h.write(&n.to_le_bytes());
            }
        }
        h.write(labels[i].as_bytes());
        h.write(&[0xff]);
        if let Some([a, b]) = endpoints[i] {
            h.write(&a.0.to_le_bytes());
            h.write(&b.0.to_le_bytes());
        }
    }
    h.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lattice_counts_and_labels() {
        let g = ProcessorGraph::square_lattice(4, 4);
        assert_eq!(g.nodes().count(), 16);
        let engineered = g
            .edges()
            .filter(|&e| g.kind(e) == ElementKind::Edge(EdgeType::Engineered))
            .count();
        let parasitic = g
            .edges()
            .filter(|&e| g.kind(e) == ElementKind::Edge(EdgeType::Parasitic))
            .count();
        assert_eq!(engineered, 24);
        assert_eq!(parasitic, 18);
        assert_eq!(g.len(), 58);
        let q = g.lattice_node(1, 3).unwrap();
        assert_eq!(g.label(q), "q1_3");
        assert!(
            g.id_of("q1_3-q1_4").is_none(),
            "no node q1_4 on a 4-column lattice"
        );
        assert!(g.id_of("q1_2-q1_3").is_some());
        assert!(g.id_of("q1_3-q2_3").is_some());
        assert!(g.id_of("q1_3-q2_2").is_some());
    }

    #[test]
    fn incidence_is_consistent() {
        let g = ProcessorGraph::square_lattice(3, 5);
        for id in g.ids() {
            for &other in g.incident(id) {
                assert!(
                    g.incident(other).contains(&id),
                    "incidence must be symmetric"
                );
                assert_ne!(g.kind(id).is_node(), g.kind(other).is_node());
            }
        }
        let centre = g.lattice_node(1, 2).unwrap();
        assert_eq!(
            g.incident(centre).len(),
            8,
            "4 engineered + 4 parasitic edges"
        );
        let corner = g.lattice_node(0, 0).unwrap();
        assert_eq!(g.incident(corner).len(), 3);
    }

    #[test]
    fn fingerprint_distinguishes_graphs() {
        let a = ProcessorGraph::square_lattice(3, 3);
        let b = ProcessorGraph::square_lattice(3, 3);
        let c = ProcessorGraph::square_lattice(3, 4);
        assert_eq!(a.fingerprint(), b.fingerprint());
        assert_ne!(a.fingerprint(), c.fingerprint());
    }

    #[test]
    fn goal_excludes_parasitic() {
        let g = ProcessorGraph::square_lattice(4, 4);
        let goal = g.goal(|e| e.kind() != ElementKind::Edge(EdgeType::Parasitic));
        assert_eq!(goal.len(), 40);
        let same = g.goal_excluding(&[ElementKind::Edge(EdgeType::Parasitic)]);
        assert_eq!(goal, same);
    }

    #[test]
    fn edge_between_finds_edges() {
        let g = ProcessorGraph::square_lattice(2, 2);
        let a = g.lattice_node(0, 0).unwrap();
        let b = g.lattice_node(1, 1).unwrap();
        let e = g.edge_between(a, b, None).unwrap();
        assert_eq!(g.kind(e), ElementKind::Edge(EdgeType::Parasitic));
        assert!(g.edge_between(a, b, Some(EdgeType::Engineered)).is_none());
        assert_eq!(g.edge_between(b, a, None), Some(e));
    }
}
