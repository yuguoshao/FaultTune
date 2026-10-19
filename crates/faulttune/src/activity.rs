//! Algorithm subgraphs and per-element activity masks (ARCHITECTURE §3.4).
//!
//! The target quantum algorithm activates different sets of elements at different
//! moments; `A_i` is the set of elements active at the same time in moment `i`. Only
//! elements that can be active together constrain each other.

use crate::error::FaultTuneError;
use crate::graph::{EdgeType, ElementId, ElementKind, ProcessorGraph};
use crate::sets::ElementSet;

/// At most this many algorithm subgraphs are supported (one bit of a `u64` mask each).
pub const MAX_SUBGRAPHS: usize = 64;

/// The algorithm subgraphs `A_0 … A_{m-1}`.
#[derive(Clone, Debug)]
pub struct AlgorithmSubgraphs {
    subgraphs: Vec<ElementSet>,
}

impl AlgorithmSubgraphs {
    /// Wraps explicit subgraphs. Errors if there are none, more than [`MAX_SUBGRAPHS`],
    /// or the sets have different capacities.
    pub fn new(subgraphs: Vec<ElementSet>) -> Result<Self, FaultTuneError> {
        if subgraphs.is_empty() {
            return Err(FaultTuneError::InvalidConfig(
                "at least one algorithm subgraph is required".into(),
            ));
        }
        if subgraphs.len() > MAX_SUBGRAPHS {
            return Err(FaultTuneError::InvalidConfig(format!(
                "{} algorithm subgraphs given, but at most {MAX_SUBGRAPHS} are supported",
                subgraphs.len()
            )));
        }
        let capacity = subgraphs[0].capacity();
        if let Some(i) = subgraphs.iter().position(|s| s.capacity() != capacity) {
            return Err(FaultTuneError::InvalidConfig(format!(
                "algorithm subgraph {i} has capacity {}, but subgraph 0 has capacity {capacity}",
                subgraphs[i].capacity()
            )));
        }
        Ok(Self { subgraphs })
    }

    /// A single subgraph `A_0` containing every element: everything can be active at once.
    pub fn all_active(graph: &ProcessorGraph) -> Self {
        Self {
            subgraphs: vec![graph.all_elements()],
        }
    }

    /// The five XEB subgraphs of a `rows × cols` square lattice built by
    /// [`ProcessorGraph::square_lattice`] (the paper's Fig. 3): one subgraph with every
    /// qubit (single-qubit layer) and four with the engineered edges that fire together
    /// in each two-qubit layer pattern, each pattern being a matching.
    ///
    /// # Construction
    ///
    /// The four two-qubit layer patterns are a proper 4-edge-colouring of the engineered
    /// edges, each pattern a matching:
    ///
    /// * a horizontal edge `(r, c)–(r, c+1)` is in pattern 0 if `r + c` is even, else in
    ///   pattern 1;
    /// * a vertical edge `(r, c)–(r+1, c)` is in pattern 2 if `r + c` is even, else in
    ///   pattern 3.
    ///
    /// Along a row (or a column) the patterns alternate, so no two edges of a pattern share a
    /// qubit, and an interior qubit takes part in each of the four patterns exactly once.
    ///
    /// * `A_0` (the single-qubit layer) holds every node.
    /// * `A_1 … A_4` hold the edges of patterns `0 … 3` plus **every node**: qubits that do
    ///   not take part in a gate idle at their frequency, so they are active.
    /// * Parasitic edges belong to no subgraph.
    ///
    /// With these subgraphs and `d_T = 2`, an interior node has 8 traversal options and an
    /// interior horizontal engineered edge has 6 (the counts of the paper's Fig. 4).
    ///
    /// # Errors
    /// `InvalidConfig` if `graph` is not exactly the `rows × cols` lattice that
    /// [`ProcessorGraph::square_lattice`] builds: nodes labelled `q{r}_{c}`, an engineered
    /// edge between every pair of nearest neighbours, a parasitic edge between every pair of
    /// diagonal neighbours, and nothing else.
    pub fn xeb_square_lattice(
        graph: &ProcessorGraph,
        rows: usize,
        cols: usize,
    ) -> Result<Self, FaultTuneError> {
        let invalid = |msg: String| {
            FaultTuneError::InvalidConfig(format!("not a {rows}x{cols} square lattice: {msg}"))
        };
        let node_at = |r: usize, c: usize| -> Result<ElementId, FaultTuneError> {
            match graph.lattice_node(r, c) {
                Some(id) if graph.kind(id) == ElementKind::Node => Ok(id),
                _ => Err(invalid(format!("no node labelled q{r}_{c}"))),
            }
        };
        let edge_at =
            |a: ElementId, b: ElementId, ty: EdgeType| -> Result<ElementId, FaultTuneError> {
                graph.edge_between(a, b, Some(ty)).ok_or_else(|| {
                    invalid(format!(
                        "no {ty:?} edge between {} and {}",
                        graph.label(a),
                        graph.label(b)
                    ))
                })
            };

        // Expected element counts of the lattice.
        let (expected_nodes, expected_engineered, expected_parasitic) = if rows == 0 || cols == 0 {
            (0, 0, 0)
        } else {
            let count = rows
                .checked_mul(cols)
                .ok_or_else(|| invalid("dimensions overflow".into()))?;
            (
                count,
                rows * (cols - 1) + (rows - 1) * cols,
                2 * (rows - 1) * (cols - 1),
            )
        };
        let count_kind = |kind| graph.ids().filter(|&id| graph.kind(id) == kind).count();
        let found = (
            count_kind(ElementKind::Node),
            count_kind(ElementKind::Edge(EdgeType::Engineered)),
            count_kind(ElementKind::Edge(EdgeType::Parasitic)),
        );
        if found != (expected_nodes, expected_engineered, expected_parasitic)
            || graph.len() != expected_nodes + expected_engineered + expected_parasitic
        {
            return Err(invalid(format!(
                "expected {expected_nodes} nodes, {expected_engineered} engineered and \
                 {expected_parasitic} parasitic edges ({} elements), found {} nodes, {} engineered and \
                 {} parasitic edges ({} elements)",
                expected_nodes + expected_engineered + expected_parasitic,
                found.0,
                found.1,
                found.2,
                graph.len()
            )));
        }

        let capacity = graph.len();
        let mut single = ElementSet::new(capacity);
        for r in 0..rows {
            for c in 0..cols {
                single.insert(node_at(r, c)?);
            }
        }
        let mut subgraphs = vec![single.clone()];
        subgraphs.extend((0..4).map(|_| single.clone()));
        for r in 0..rows {
            for c in 0..cols {
                let here = node_at(r, c)?;
                let parity_even = (r + c) % 2 == 0;
                if c + 1 < cols {
                    let e = edge_at(here, node_at(r, c + 1)?, EdgeType::Engineered)?;
                    subgraphs[1 + usize::from(!parity_even)].insert(e);
                }
                if r + 1 < rows {
                    let e = edge_at(here, node_at(r + 1, c)?, EdgeType::Engineered)?;
                    subgraphs[3 + usize::from(!parity_even)].insert(e);
                }
                // Parasitic diagonals only have to exist; they are in no subgraph.
                if r + 1 < rows && c + 1 < cols {
                    edge_at(here, node_at(r + 1, c + 1)?, EdgeType::Parasitic)?;
                    edge_at(node_at(r, c + 1)?, node_at(r + 1, c)?, EdgeType::Parasitic)?;
                }
            }
        }
        Self::new(subgraphs)
    }

    /// Number of subgraphs.
    pub fn len(&self) -> usize {
        self.subgraphs.len()
    }

    /// Whether there are no subgraphs (never true for a constructed value).
    pub fn is_empty(&self) -> bool {
        self.subgraphs.is_empty()
    }

    /// Subgraph `A_i`.
    pub fn get(&self, i: usize) -> Option<&ElementSet> {
        self.subgraphs.get(i)
    }

    /// Iterates over the subgraphs.
    pub fn iter(&self) -> impl Iterator<Item = &ElementSet> + '_ {
        self.subgraphs.iter()
    }

    /// One 64-bit activity mask per element: bit `i` is set if the element is in `A_i`.
    pub fn masks(&self) -> ActivityMasks {
        let capacity = self.subgraphs.first().map_or(0, ElementSet::capacity);
        let mut masks = vec![0u64; capacity];
        for (i, subgraph) in self.subgraphs.iter().enumerate() {
            for id in subgraph {
                masks[id.index()] |= 1u64 << i;
            }
        }
        ActivityMasks { masks }
    }
}

/// Per-element activity masks. Two elements can be active at the same time exactly when
/// their masks share a bit.
#[derive(Clone, Debug)]
pub struct ActivityMasks {
    masks: Vec<u64>,
}

impl ActivityMasks {
    /// The mask of an element.
    #[inline]
    pub fn mask(&self, id: ElementId) -> u64 {
        self.masks[id.index()]
    }

    /// Whether `a` and `b` can be active at the same time.
    #[inline]
    pub fn co_active(&self, a: ElementId, b: ElementId) -> bool {
        self.masks[a.index()] & self.masks[b.index()] != 0
    }

    /// Whether the element belongs to at least one subgraph.
    #[inline]
    pub fn is_active(&self, id: ElementId) -> bool {
        self.masks[id.index()] != 0
    }

    /// Number of elements the masks cover.
    pub fn len(&self) -> usize {
        self.masks.len()
    }

    /// Whether the masks cover no elements.
    pub fn is_empty(&self) -> bool {
        self.masks.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;

    fn id(i: u32) -> ElementId {
        ElementId::new(i)
    }

    fn set(capacity: usize, ids: &[u32]) -> ElementSet {
        ElementSet::from_ids(capacity, ids.iter().map(|&i| id(i)))
    }

    fn is_invalid_config<T: std::fmt::Debug>(r: Result<T, FaultTuneError>) -> bool {
        matches!(r, Err(FaultTuneError::InvalidConfig(_)))
    }

    // ----- new / all_active / masks -----

    #[test]
    fn new_validates_count_and_capacity() {
        assert!(is_invalid_config(AlgorithmSubgraphs::new(vec![])));
        let too_many = vec![ElementSet::new(4); MAX_SUBGRAPHS + 1];
        assert!(is_invalid_config(AlgorithmSubgraphs::new(too_many)));
        let mixed = vec![ElementSet::new(4), ElementSet::new(5)];
        assert!(is_invalid_config(AlgorithmSubgraphs::new(mixed)));
        let one = AlgorithmSubgraphs::new(vec![ElementSet::new(4)]).unwrap();
        assert_eq!(one.len(), 1);
        assert!(!one.is_empty());
        let max = AlgorithmSubgraphs::new(vec![ElementSet::new(4); MAX_SUBGRAPHS]).unwrap();
        assert_eq!(max.len(), 64);
    }

    #[test]
    fn accessors() {
        let subs = AlgorithmSubgraphs::new(vec![set(5, &[0, 1]), set(5, &[3])]).unwrap();
        assert_eq!(subs.get(0), Some(&set(5, &[0, 1])));
        assert_eq!(subs.get(1), Some(&set(5, &[3])));
        assert_eq!(subs.get(2), None);
        assert_eq!(subs.iter().count(), 2);
    }

    #[test]
    fn masks_set_the_bit_of_every_subgraph() {
        let subs =
            AlgorithmSubgraphs::new(vec![set(5, &[0, 1]), set(5, &[1, 2]), set(5, &[1])]).unwrap();
        let masks = subs.masks();
        assert_eq!(masks.len(), 5);
        assert!(!masks.is_empty());
        assert_eq!(masks.mask(id(0)), 0b001);
        assert_eq!(masks.mask(id(1)), 0b111);
        assert_eq!(masks.mask(id(2)), 0b010);
        assert_eq!(masks.mask(id(3)), 0);
        assert!(masks.is_active(id(0)));
        assert!(!masks.is_active(id(4)));
        // Co-active exactly when the masks share a bit.
        assert!(masks.co_active(id(0), id(1)));
        assert!(masks.co_active(id(1), id(2)));
        assert!(!masks.co_active(id(0), id(2)));
        assert!(!masks.co_active(id(3), id(3)));
        assert!(!masks.co_active(id(0), id(3)));
    }

    #[test]
    fn masks_use_all_64_bits() {
        let subs = AlgorithmSubgraphs::new(
            (0..64)
                .map(|i| set(2, &[if i == 63 { 1 } else { 0 }]))
                .collect(),
        )
        .unwrap();
        let masks = subs.masks();
        assert_eq!(masks.mask(id(1)), 1 << 63);
        assert_eq!(masks.mask(id(0)), u64::MAX >> 1);
    }

    #[test]
    fn masks_of_empty_universe() {
        let subs = AlgorithmSubgraphs::new(vec![ElementSet::new(0)]).unwrap();
        assert!(subs.masks().is_empty());
    }

    #[test]
    fn all_active_covers_everything() {
        let graph = ProcessorGraph::square_lattice(3, 3);
        let subs = AlgorithmSubgraphs::all_active(&graph);
        assert_eq!(subs.len(), 1);
        assert_eq!(subs.get(0).unwrap().len(), graph.len());
        let masks = subs.masks();
        for a in graph.ids() {
            assert_eq!(masks.mask(a), 1);
            for b in graph.ids().filter(|&b| b != a) {
                assert!(masks.co_active(a, b));
            }
        }
    }

    // ----- XEB subgraphs -----

    fn engineered(
        graph: &ProcessorGraph,
        rows: usize,
        cols: usize,
    ) -> Vec<(usize, usize, bool, ElementId)> {
        // (r, c, horizontal, edge id)
        let mut out = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                let here = graph.lattice_node(r, c).unwrap();
                if c + 1 < cols {
                    let right = graph.lattice_node(r, c + 1).unwrap();
                    out.push((
                        r,
                        c,
                        true,
                        graph
                            .edge_between(here, right, Some(EdgeType::Engineered))
                            .unwrap(),
                    ));
                }
                if r + 1 < rows {
                    let down = graph.lattice_node(r + 1, c).unwrap();
                    out.push((
                        r,
                        c,
                        false,
                        graph
                            .edge_between(here, down, Some(EdgeType::Engineered))
                            .unwrap(),
                    ));
                }
            }
        }
        out
    }

    #[test]
    fn xeb_structure_on_several_lattices() {
        for (rows, cols) in [(1, 1), (1, 2), (1, 7), (6, 1), (2, 2), (3, 5), (6, 6)] {
            let graph = ProcessorGraph::square_lattice(rows, cols);
            let subs = AlgorithmSubgraphs::xeb_square_lattice(&graph, rows, cols).unwrap();
            assert_eq!(subs.len(), 5);
            for i in 0..5 {
                assert_eq!(subs.get(i).unwrap().capacity(), graph.len());
            }
            // Every subgraph contains every node.
            for i in 0..5 {
                for n in graph.nodes() {
                    assert!(subs.get(i).unwrap().contains(n), "{rows}x{cols} A_{i}");
                }
            }
            // A_0 has nothing else.
            assert_eq!(subs.get(0).unwrap().len(), rows * cols);
            // Parasitic edges are in no subgraph.
            for e in graph.edges() {
                if graph.kind(e).edge_type() == Some(EdgeType::Parasitic) {
                    assert!((0..5).all(|i| !subs.get(i).unwrap().contains(e)));
                }
            }
            // Each engineered edge is in exactly one of A_1..A_4 and none else.
            for (r, c, horizontal, e) in engineered(&graph, rows, cols) {
                let containing: Vec<usize> = (0..5)
                    .filter(|&i| subs.get(i).unwrap().contains(e))
                    .collect();
                let even = (r + c) % 2 == 0;
                let pattern = match (horizontal, even) {
                    (true, true) => 0,
                    (true, false) => 1,
                    (false, true) => 2,
                    (false, false) => 3,
                };
                assert_eq!(containing, vec![1 + pattern], "{rows}x{cols} edge {e}");
            }
            // Each pattern is a matching: no node is covered twice.
            for i in 1..5 {
                let mut covered = vec![0; graph.len()];
                for e in subs
                    .get(i)
                    .unwrap()
                    .iter()
                    .filter(|&e| graph.kind(e).is_edge())
                {
                    for n in graph.endpoints(e).unwrap() {
                        covered[n.index()] += 1;
                    }
                }
                assert!(covered.iter().all(|&k| k <= 1), "{rows}x{cols} pattern {i}");
            }
        }
    }

    #[test]
    fn xeb_interior_nodes_take_part_in_every_pattern() {
        let graph = ProcessorGraph::square_lattice(6, 6);
        let subs = AlgorithmSubgraphs::xeb_square_lattice(&graph, 6, 6).unwrap();
        for r in 1..5 {
            for c in 1..5 {
                let n = graph.lattice_node(r, c).unwrap();
                for i in 1..5 {
                    let k = graph
                        .incident(n)
                        .iter()
                        .filter(|&&e| {
                            graph.kind(e).edge_type() == Some(EdgeType::Engineered)
                                && subs.get(i).unwrap().contains(e)
                        })
                        .count();
                    assert_eq!(k, 1, "node ({r},{c}) pattern {i}");
                }
            }
        }
    }

    #[test]
    fn xeb_edge_counts_match_fig3() {
        // On a 6x6 lattice there are 2 * 6 * 5 = 60 engineered edges, split over 4 patterns.
        let graph = ProcessorGraph::square_lattice(6, 6);
        let subs = AlgorithmSubgraphs::xeb_square_lattice(&graph, 6, 6).unwrap();
        let sizes: Vec<usize> = (1..5).map(|i| subs.get(i).unwrap().len() - 36).collect();
        assert_eq!(sizes.iter().sum::<usize>(), 60);
        // Horizontal edges: 6 rows x 5 gaps = 30, split by the parity of r + c into 15 + 15.
        // Vertical edges: likewise 15 + 15. A shifted parity rule moves edges between patterns.
        assert_eq!(sizes, vec![15, 15, 15, 15]);
    }

    #[test]
    fn xeb_activity_masks() {
        let graph = ProcessorGraph::square_lattice(5, 5);
        let masks = AlgorithmSubgraphs::xeb_square_lattice(&graph, 5, 5)
            .unwrap()
            .masks();
        let n = graph.lattice_node(1, 1).unwrap();
        assert_eq!(masks.mask(n), 0b11111);
        for e in graph.edges() {
            match graph.kind(e).edge_type() {
                Some(EdgeType::Parasitic) => {
                    assert!(!masks.is_active(e));
                    assert!(!masks.co_active(e, n));
                }
                _ => {
                    let m = masks.mask(e);
                    assert_eq!(m.count_ones(), 1);
                    assert!(m & 1 == 0, "engineered edges are not in A_0");
                    assert!(masks.co_active(e, n));
                }
            }
        }
        // Horizontal pattern-0 and pattern-1 edges are never active together.
        let h = |r, c| {
            graph
                .edge_between(
                    graph.lattice_node(r, c).unwrap(),
                    graph.lattice_node(r, c + 1).unwrap(),
                    Some(EdgeType::Engineered),
                )
                .unwrap()
        };
        assert!(!masks.co_active(h(1, 1), h(1, 2)));
        assert!(masks.co_active(h(1, 1), h(1, 3)));
        assert!(masks.co_active(h(1, 1), h(2, 0)));
        assert!(!masks.co_active(h(1, 1), h(2, 1)));
    }

    #[test]
    fn xeb_empty_lattice() {
        let graph = ProcessorGraph::square_lattice(0, 0);
        let subs = AlgorithmSubgraphs::xeb_square_lattice(&graph, 0, 0).unwrap();
        assert_eq!(subs.len(), 5);
        assert!(subs.iter().all(ElementSet::is_empty));
        // 0 x n is the same empty graph.
        assert!(AlgorithmSubgraphs::xeb_square_lattice(&graph, 0, 4).is_ok());
    }

    #[test]
    fn xeb_rejects_graphs_that_are_not_the_lattice() {
        let lattice = ProcessorGraph::square_lattice(3, 3);
        // Wrong dimensions.
        assert!(is_invalid_config(AlgorithmSubgraphs::xeb_square_lattice(
            &lattice, 3, 4
        )));
        assert!(is_invalid_config(AlgorithmSubgraphs::xeb_square_lattice(
            &lattice, 2, 2
        )));
        assert!(is_invalid_config(AlgorithmSubgraphs::xeb_square_lattice(
            &lattice, 0, 0
        )));
        // Transposed dimensions are rejected for non-square lattices.
        let wide = ProcessorGraph::square_lattice(2, 5);
        assert!(is_invalid_config(AlgorithmSubgraphs::xeb_square_lattice(
            &wide, 5, 2
        )));
        // A custom graph with other labels.
        let mut b = GraphBuilder::new();
        let a = b.add_node("a").unwrap();
        let c = b.add_node("c").unwrap();
        b.add_edge(a, c, EdgeType::Engineered).unwrap();
        let custom = b.build().unwrap();
        assert!(is_invalid_config(AlgorithmSubgraphs::xeb_square_lattice(
            &custom, 1, 2
        )));
    }

    #[test]
    fn xeb_rejects_lattice_with_wrong_edges() {
        // 2x2 lattice labels, but the diagonals are missing.
        let mut b = GraphBuilder::new();
        let q: Vec<_> = [(0, 0), (0, 1), (1, 0), (1, 1)]
            .iter()
            .map(|&(r, c)| b.add_node(format!("q{r}_{c}")).unwrap())
            .collect();
        b.add_edge(q[0], q[1], EdgeType::Engineered).unwrap();
        b.add_edge(q[0], q[2], EdgeType::Engineered).unwrap();
        b.add_edge(q[1], q[3], EdgeType::Engineered).unwrap();
        b.add_edge(q[2], q[3], EdgeType::Engineered).unwrap();
        let no_diagonals = b.clone().build().unwrap();
        assert!(is_invalid_config(AlgorithmSubgraphs::xeb_square_lattice(
            &no_diagonals,
            2,
            2
        )));
        // Diagonals of the wrong type.
        let mut wrong_type = b.clone();
        wrong_type
            .add_edge(q[0], q[3], EdgeType::Custom(1))
            .unwrap();
        wrong_type
            .add_edge(q[1], q[2], EdgeType::Custom(1))
            .unwrap();
        let wrong_type = wrong_type.build().unwrap();
        assert!(is_invalid_config(AlgorithmSubgraphs::xeb_square_lattice(
            &wrong_type,
            2,
            2
        )));
        // The right diagonals make it valid; one more edge makes it invalid again.
        b.add_edge(q[0], q[3], EdgeType::Parasitic).unwrap();
        b.add_edge(q[1], q[2], EdgeType::Parasitic).unwrap();
        let mut extra = b.clone();
        assert!(AlgorithmSubgraphs::xeb_square_lattice(&b.build().unwrap(), 2, 2).is_ok());
        extra
            .add_edge_labelled("extra", q[0], q[1], EdgeType::Parasitic)
            .unwrap();
        assert!(is_invalid_config(AlgorithmSubgraphs::xeb_square_lattice(
            &extra.build().unwrap(),
            2,
            2
        )));
    }
}
