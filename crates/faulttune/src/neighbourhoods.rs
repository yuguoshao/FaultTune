//! Connectivity subgraphs `X_g^d` and the neighbourhood index (ARCHITECTURE §3.2–3.3).
//!
//! Distance is measured on the **incidence graph** (vertices: all elements; each node is
//! linked to its incident edges). `X_g^d` is the set of elements within incidence distance
//! `r(g, d)` of `g`, where
//!
//! * `r = 2d` if `g` is an edge, and
//! * `r = max(2d − 1, 0)` if `g` is a node.
//!
//! "`a` is within `d` of `b`" is symmetric for every pair of element kinds. The incidence
//! graph is bipartite (nodes on one side, edges on the other), so the incidence distance
//! between two elements of the same kind is even and between a node and an edge is odd.
//! For a node `a` and an edge `b`, `dist ≤ 2d − 1` and `dist ≤ 2d` are therefore the same
//! condition; for two nodes `dist ≤ 2d − 1` and `dist ≤ 2d − 2` agree; for two edges both
//! radii are `2d`. This is why the relation is symmetric.

use crate::graph::{ElementId, ElementKind, ProcessorGraph};

/// For every element, all elements up to the largest radius needed, sorted by incidence
/// distance (then by ID), in one compressed array. `within(g, d)` is a slice: no search at
/// run time, which is what keeps the engine's work per step independent of processor size.
///
/// Layout (per element `g`):
///
/// ```text
/// ids[offsets[g] ..]        g, then the elements at distance 1 (ascending ID), at distance 2, …
/// ring_ends[ring_start[g] + r]   end (relative to offsets[g]) of the prefix of distance <= r
/// ```
///
/// Rings are stored only up to the largest distance actually reached, so a small connected
/// component costs nothing for a large `max_distance`. No per-entry distance is stored: it is
/// implied by the ring an entry sits in.
///
/// Offsets are `u32`. Memory is dominated by the balls themselves (4 bytes per entry) and grows
/// quickly with `max_distance`: on a 100 x 100 lattice (49,402 elements) it is about 17 MB at
/// `max_distance = 2`, 37 MB at 3 and 65 MB at 4 (see [`memory_bytes`](Self::memory_bytes)).
#[derive(Clone, Debug)]
pub struct Neighbourhoods {
    max_distance: u32,
    /// `true` for edges: the radius `r(g, d)` depends on the kind of `g` only.
    is_edge: Vec<bool>,
    /// `n + 1` offsets: the ball of `g` is `ids[offsets[g]..offsets[g + 1]]`.
    offsets: Vec<u32>,
    /// All balls, concatenated.
    ids: Vec<ElementId>,
    /// `n + 1` offsets into `ring_ends`: element `g` owns `ring_start[g]..ring_start[g + 1]`.
    ring_start: Vec<u32>,
    /// For each element and each ring `r` up to its largest reached distance: the end of
    /// the prefix with distance `<= r`, relative to the start of the element's ball.
    ring_ends: Vec<u32>,
}

/// Converts a table position to the compact `u32` offset type.
///
/// # Panics
/// Panics if the index would hold more than `u32::MAX` entries (over 16 GB of IDs): lower
/// `max_distance` or split the processor.
#[inline]
fn offset_u32(position: usize) -> u32 {
    u32::try_from(position)
        .expect("neighbourhood index exceeds 2^32 entries; use a smaller max_distance")
}

/// The incidence radius for an element that is (not) an edge.
#[inline]
fn radius_of(is_edge: bool, d: u32) -> u32 {
    let twice = d.saturating_mul(2);
    if is_edge {
        twice
    } else {
        twice.saturating_sub(1)
    }
}

impl Neighbourhoods {
    /// The incidence radius `r(g, d)` for an element of the given kind.
    pub fn incidence_radius(kind: ElementKind, d: u32) -> u32 {
        radius_of(kind.is_edge(), d)
    }

    /// Builds the index for every `d <= max_distance`.
    ///
    /// Runs one breadth-first search over the incidence graph from every element, truncated
    /// at `r(g, max_distance)`. Time and memory are proportional to the total size of all
    /// balls, which is linear in the number of elements when balls stay small.
    pub fn build(graph: &ProcessorGraph, max_distance: u32) -> Self {
        let n = graph.len();
        let is_edge: Vec<bool> = graph.ids().map(|id| graph.kind(id).is_edge()).collect();

        let mut offsets: Vec<u32> = Vec::with_capacity(n + 1);
        let mut ring_start: Vec<u32> = Vec::with_capacity(n + 1);
        let mut ids: Vec<ElementId> = Vec::new();
        let mut ring_ends: Vec<u32> = Vec::new();
        offsets.push(0);
        ring_start.push(0);

        // `stamp[v] == s + 1` marks `v` as seen by the search from `s`; 0 means never seen.
        let mut stamp = vec![0u32; n];
        let mut frontier: Vec<ElementId> = Vec::new();
        let mut next: Vec<ElementId> = Vec::new();

        for s in 0..n {
            let source = ElementId::new(s as u32);
            let mark = s as u32 + 1;
            let radius = radius_of(is_edge[s], max_distance);
            let base = ids.len();

            ids.push(source);
            stamp[s] = mark;
            ring_ends.push(1);
            frontier.clear();
            frontier.push(source);

            let mut reached = 0u32;
            while reached < radius {
                next.clear();
                for &u in &frontier {
                    for &v in graph.incident(u) {
                        if stamp[v.index()] != mark {
                            stamp[v.index()] = mark;
                            next.push(v);
                        }
                    }
                }
                if next.is_empty() {
                    break;
                }
                next.sort_unstable();
                ids.extend_from_slice(&next);
                ring_ends.push((ids.len() - base) as u32);
                std::mem::swap(&mut frontier, &mut next);
                reached += 1;
            }

            offsets.push(offset_u32(ids.len()));
            ring_start.push(offset_u32(ring_ends.len()));
        }

        ids.shrink_to_fit();
        ring_ends.shrink_to_fit();
        Self {
            max_distance,
            is_edge,
            offsets,
            ids,
            ring_start,
            ring_ends,
        }
    }

    /// The largest `d` the index supports.
    pub fn max_distance(&self) -> u32 {
        self.max_distance
    }

    /// Number of elements the index covers.
    pub fn len(&self) -> usize {
        self.is_edge.len()
    }

    /// Whether the index covers no elements.
    pub fn is_empty(&self) -> bool {
        self.is_edge.is_empty()
    }

    /// Approximate heap memory used by the index, in bytes.
    pub fn memory_bytes(&self) -> usize {
        use std::mem::size_of;
        self.is_edge.capacity()
            + self.offsets.capacity() * size_of::<u32>()
            + self.ids.capacity() * size_of::<ElementId>()
            + self.ring_start.capacity() * size_of::<u32>()
            + self.ring_ends.capacity() * size_of::<u32>()
    }

    /// The largest ring index (incidence distance) stored for element `gi`.
    #[inline]
    fn last_ring(&self, gi: usize) -> usize {
        (self.ring_start[gi + 1] - self.ring_start[gi]) as usize - 1
    }

    /// The ring index that corresponds to `X_g^d`, clipped to the rings that exist.
    #[inline]
    fn ring_limit(&self, gi: usize, d: u32) -> usize {
        assert!(
            d <= self.max_distance,
            "distance {d} exceeds the neighbourhood index (max distance {})",
            self.max_distance
        );
        (radius_of(self.is_edge[gi], d) as usize).min(self.last_ring(gi))
    }

    /// The elements of `g`'s ball at incidence distance `<= r` (a prefix of the ball).
    #[inline]
    fn prefix(&self, gi: usize, r: usize) -> &[ElementId] {
        let base = self.offsets[gi] as usize;
        let end = self.ring_ends[self.ring_start[gi] as usize + r] as usize;
        &self.ids[base..base + end]
    }

    /// `X_g^d`: the elements within `d` of `g`, including `g` itself (first), sorted by
    /// incidence distance and then by ID.
    ///
    /// # Panics
    /// Panics if `d > self.max_distance()`.
    #[inline]
    pub fn within(&self, g: ElementId, d: u32) -> &[ElementId] {
        let gi = g.index();
        self.prefix(gi, self.ring_limit(gi, d))
    }

    /// Like [`within`](Self::within) but also yields each element's incidence distance from `g`.
    ///
    /// # Panics
    /// Panics if `d > self.max_distance()`.
    pub fn within_with_distance(
        &self,
        g: ElementId,
        d: u32,
    ) -> impl Iterator<Item = (ElementId, u32)> + '_ {
        let gi = g.index();
        let last = self.ring_limit(gi, d);
        (0..=last).flat_map(move |r| self.ring(g, r as u32).iter().map(move |&e| (e, r as u32)))
    }

    /// The elements at incidence distance exactly `r` from `g` (ascending by ID); empty if
    /// `r` is beyond the indexed ball of `g`.
    pub fn ring(&self, g: ElementId, r: u32) -> &[ElementId] {
        let gi = g.index();
        let r = r as usize;
        if r > self.last_ring(gi) {
            return &[];
        }
        let full = self.prefix(gi, r);
        if r == 0 {
            full
        } else {
            &full[self.prefix(gi, r - 1).len()..]
        }
    }

    /// The incidence distance from `a` to `b`, looking only at distances `<= limit`.
    fn distance_upto(&self, a: ElementId, b: ElementId, limit: usize) -> Option<u32> {
        let (ai, bi) = (a.index(), b.index());
        if ai >= self.len() || bi >= self.len() {
            return None;
        }
        let top = limit.min(self.last_ring(ai));
        // Distances between same-kind elements are even, between different kinds odd.
        let first = usize::from(self.is_edge[ai] != self.is_edge[bi]);
        (first..=top)
            .step_by(2)
            .find(|&r| self.ring(a, r as u32).binary_search(&b).is_ok())
            .map(|r| r as u32)
    }

    /// Incidence distance from `a` to `b`, if `b` is inside `a`'s indexed ball.
    ///
    /// Returns `None` if `b` is farther than the radius indexed for `a`, is unreachable
    /// from `a`, or either ID is outside the graph.
    pub fn incidence_distance(&self, a: ElementId, b: ElementId) -> Option<u32> {
        self.distance_upto(a, b, usize::MAX)
    }

    /// Whether `b ∈ X_a^d` (equivalently `a ∈ X_b^d`).
    ///
    /// # Panics
    /// Panics if `d > self.max_distance()`.
    pub fn is_within(&self, a: ElementId, b: ElementId, d: u32) -> bool {
        assert!(
            d <= self.max_distance,
            "distance {d} exceeds the neighbourhood index (max distance {})",
            self.max_distance
        );
        if a.index() >= self.len() {
            return false;
        }
        let radius = radius_of(self.is_edge[a.index()], d) as usize;
        self.distance_upto(a, b, radius).is_some()
    }
}

/// Helpers shared by the unit tests of this module and of `scope`, `segment` and `traverse`.
#[cfg(test)]
pub(crate) mod test_support {
    use std::collections::VecDeque;

    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use crate::graph::{EdgeType, ElementId, GraphBuilder, ProcessorGraph};

    /// A random multigraph: `1..=max_nodes` nodes and up to `max_edges` edges (parallel edges
    /// allowed) of mixed types, including a custom type.
    pub fn random_graph(seed: u64, max_nodes: usize, max_edges: usize) -> ProcessorGraph {
        let mut rng = StdRng::seed_from_u64(seed);
        let nodes = rng.random_range(1..=max_nodes);
        let mut b = GraphBuilder::new();
        let ids: Vec<ElementId> = (0..nodes)
            .map(|i| b.add_node(format!("n{i}")).unwrap())
            .collect();
        if nodes >= 2 {
            for k in 0..rng.random_range(0..=max_edges) {
                let a = rng.random_range(0..nodes);
                let mut c = rng.random_range(0..nodes - 1);
                if c >= a {
                    c += 1;
                }
                let ty = match rng.random_range(0..4) {
                    0 | 1 => EdgeType::Engineered,
                    2 => EdgeType::Parasitic,
                    _ => EdgeType::Custom(7),
                };
                b.add_edge_labelled(format!("e{k}"), ids[a], ids[c], ty)
                    .unwrap();
            }
        }
        b.build().unwrap()
    }

    /// Reference incidence distances from `src` by a plain whole-graph BFS.
    pub fn reference_distances(graph: &ProcessorGraph, src: ElementId) -> Vec<Option<u32>> {
        let mut dist = vec![None; graph.len()];
        dist[src.index()] = Some(0);
        let mut queue = VecDeque::from([src]);
        while let Some(u) = queue.pop_front() {
            let du = dist[u.index()].unwrap();
            for &v in graph.incident(u) {
                if dist[v.index()].is_none() {
                    dist[v.index()] = Some(du + 1);
                    queue.push_back(v);
                }
            }
        }
        dist
    }

    /// The reference radius `r(g, d)`.
    pub fn reference_radius(graph: &ProcessorGraph, g: ElementId, d: u32) -> u32 {
        if graph.kind(g).is_edge() {
            2 * d
        } else {
            (2 * d).saturating_sub(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use proptest::prelude::*;

    use super::test_support::*;
    use super::*;
    use crate::graph::{EdgeType, GraphBuilder};

    fn id(i: usize) -> ElementId {
        ElementId::new(i as u32)
    }

    fn set_of(slice: &[ElementId]) -> BTreeSet<ElementId> {
        slice.iter().copied().collect()
    }

    /// Checks the whole contract of `Neighbourhoods` against plain BFS for one graph.
    fn check_against_reference(graph: &ProcessorGraph, max_distance: u32) {
        let nb = Neighbourhoods::build(graph, max_distance);
        assert_eq!(nb.max_distance(), max_distance);
        assert_eq!(nb.len(), graph.len());
        for a in graph.ids() {
            let dist = reference_distances(graph, a);
            for d in 0..=max_distance {
                let radius = reference_radius(graph, a, d);
                let within = nb.within(a, d);
                // First element is the centre; no duplicates.
                assert_eq!(within[0], a);
                assert_eq!(set_of(within).len(), within.len());
                // Exactly the elements within the radius.
                let expected: BTreeSet<_> = graph
                    .ids()
                    .filter(|b| dist[b.index()].is_some_and(|x| x <= radius))
                    .collect();
                assert_eq!(set_of(within), expected, "X_{a}^{d}");
                // Sorted by (distance, id), with the right distances.
                let mut last = (0u32, ElementId::new(0));
                for (i, (b, x)) in nb.within_with_distance(a, d).enumerate() {
                    assert_eq!(Some(x), dist[b.index()]);
                    assert_eq!(within[i], b);
                    if i > 0 {
                        assert!((x, b) > last, "not sorted by (distance, id)");
                    }
                    last = (x, b);
                }
                assert_eq!(nb.within_with_distance(a, d).count(), within.len());
            }
            // Distances inside the indexed ball are right; outside, None.
            let top = reference_radius(graph, a, max_distance);
            for b in graph.ids() {
                let want = dist[b.index()].filter(|&x| x <= top);
                assert_eq!(nb.incidence_distance(a, b), want);
            }
        }
    }

    /// `is_within` is symmetric, agrees with `within` and with the reference, for all pairs.
    fn check_is_within(graph: &ProcessorGraph, max_distance: u32) {
        let nb = Neighbourhoods::build(graph, max_distance);
        for a in graph.ids() {
            let dist = reference_distances(graph, a);
            for b in graph.ids() {
                for d in 0..=max_distance {
                    let got = nb.is_within(a, b, d);
                    assert_eq!(got, nb.is_within(b, a, d), "asymmetric: {a} {b} d={d}");
                    assert_eq!(got, nb.within(a, d).contains(&b), "{a} {b} d={d}");
                    let want = dist[b.index()].is_some_and(|x| x <= reference_radius(graph, a, d));
                    assert_eq!(got, want, "{a} {b} d={d}");
                }
            }
        }
    }

    // ----- construction edge cases -----

    #[test]
    fn empty_graph() {
        let graph = ProcessorGraph::square_lattice(0, 0);
        let nb = Neighbourhoods::build(&graph, 3);
        assert!(nb.is_empty());
        assert_eq!(nb.len(), 0);
        assert_eq!(nb.incidence_distance(id(0), id(0)), None);
        assert!(!nb.is_within(id(0), id(0), 1));
    }

    #[test]
    fn single_node() {
        let graph = ProcessorGraph::square_lattice(1, 1);
        let nb = Neighbourhoods::build(&graph, 2);
        for d in 0..=2 {
            assert_eq!(nb.within(id(0), d), &[id(0)]);
        }
        assert!(nb.is_within(id(0), id(0), 0));
        assert_eq!(nb.incidence_distance(id(0), id(0)), Some(0));
    }

    #[test]
    fn two_nodes_one_edge() {
        let graph = ProcessorGraph::square_lattice(1, 2);
        assert_eq!(graph.len(), 3);
        let nb = Neighbourhoods::build(&graph, 2);
        let (a, b, e) = (id(0), id(1), id(2));
        assert_eq!(nb.within(a, 0), &[a]);
        assert_eq!(nb.within(a, 1), &[a, e]);
        assert_eq!(nb.within(a, 2), &[a, e, b]);
        assert_eq!(nb.within(e, 0), &[e]);
        assert_eq!(nb.within(e, 1), &[e, a, b]);
    }

    #[test]
    fn max_distance_zero() {
        let graph = ProcessorGraph::square_lattice(3, 3);
        let nb = Neighbourhoods::build(&graph, 0);
        assert_eq!(nb.max_distance(), 0);
        for g in graph.ids() {
            assert_eq!(nb.within(g, 0), &[g]);
            assert!(nb.is_within(g, g, 0));
        }
        assert!(!nb.is_within(id(0), id(1), 0));
        assert_eq!(nb.incidence_distance(id(0), id(1)), None);
        check_against_reference(&graph, 0);
    }

    #[test]
    #[should_panic(expected = "exceeds the neighbourhood index")]
    fn within_beyond_max_distance_panics() {
        let graph = ProcessorGraph::square_lattice(2, 2);
        Neighbourhoods::build(&graph, 1).within(id(0), 2);
    }

    #[test]
    #[should_panic(expected = "exceeds the neighbourhood index")]
    fn is_within_beyond_max_distance_panics() {
        let graph = ProcessorGraph::square_lattice(2, 2);
        Neighbourhoods::build(&graph, 1).is_within(id(0), id(1), 2);
    }

    #[test]
    fn incidence_radius_formula() {
        let node = ElementKind::Node;
        let edge = ElementKind::Edge(EdgeType::Parasitic);
        let nodes: Vec<u32> = (0..5)
            .map(|d| Neighbourhoods::incidence_radius(node, d))
            .collect();
        let edges: Vec<u32> = (0..5)
            .map(|d| Neighbourhoods::incidence_radius(edge, d))
            .collect();
        assert_eq!(nodes, [0, 1, 3, 5, 7]);
        assert_eq!(edges, [0, 2, 4, 6, 8]);
        assert_eq!(Neighbourhoods::incidence_radius(edge, u32::MAX), u32::MAX);
    }

    // ----- paper figures -----

    #[test]
    fn fig2_interior_node() {
        let graph = ProcessorGraph::square_lattice(5, 5);
        let nb = Neighbourhoods::build(&graph, 3);
        let n = graph.lattice_node(2, 2).unwrap();

        // X_n^0 = {n}
        assert_eq!(nb.within(n, 0), &[n]);

        // X_n^1 = n plus its 8 incident edges: 4 engineered and 4 parasitic.
        let x1 = nb.within(n, 1);
        assert_eq!(x1.len(), 9);
        assert_eq!(x1[0], n);
        assert_eq!(set_of(&x1[1..]), set_of(graph.incident(n)));
        let engineered = x1[1..]
            .iter()
            .filter(|&&e| graph.kind(e).edge_type() == Some(EdgeType::Engineered))
            .count();
        let parasitic = x1[1..]
            .iter()
            .filter(|&&e| graph.kind(e).edge_type() == Some(EdgeType::Parasitic))
            .count();
        assert_eq!((engineered, parasitic), (4, 4));

        // X_n^2 = the 3x3 block of nodes around n plus ALL their incident edges.
        let x2 = nb.within(n, 2);
        let block: BTreeSet<ElementId> = (1..=3)
            .flat_map(|r| (1..=3).map(move |c| (r, c)))
            .map(|(r, c)| graph.lattice_node(r, c).unwrap())
            .collect();
        let mut expected = block.clone();
        for e in graph.edges() {
            let [a, b] = graph.endpoints(e).unwrap();
            if block.contains(&a) || block.contains(&b) {
                expected.insert(e);
            }
        }
        assert_eq!(set_of(x2), expected);
        // 9 nodes + 24 engineered + 28 parasitic edges.
        assert_eq!(x2.len(), 9 + 24 + 28);
        let nodes_in = x2.iter().filter(|&&e| graph.kind(e).is_node()).count();
        assert_eq!(nodes_in, 9);
        // The edge counts, by type.
        let count = |t| {
            x2.iter()
                .filter(|&&e| graph.kind(e).edge_type() == Some(t))
                .count()
        };
        assert_eq!(count(EdgeType::Engineered), 24);
        assert_eq!(count(EdgeType::Parasitic), 28);
    }

    #[test]
    fn fig2_edge() {
        let graph = ProcessorGraph::square_lattice(5, 5);
        let nb = Neighbourhoods::build(&graph, 3);
        let a = graph.lattice_node(2, 2).unwrap();
        let b = graph.lattice_node(2, 3).unwrap();
        let e = graph
            .edge_between(a, b, Some(EdgeType::Engineered))
            .unwrap();

        assert_eq!(nb.within(e, 0), &[e]);

        // X_e^1 = e, its two endpoints and every edge incident to them.
        let mut expected: BTreeSet<ElementId> = [e, a, b].into_iter().collect();
        expected.extend(graph.incident(a));
        expected.extend(graph.incident(b));
        let x1 = nb.within(e, 1);
        assert_eq!(set_of(x1), expected);
        // 3 + (8 + 8 - 1 shared) edges = 3 elements besides... 2 nodes + 15 edges.
        assert_eq!(x1.len(), 2 + 15);
        // Sorted: e first, then the two endpoints, then edges.
        assert_eq!(x1[0], e);
        assert_eq!(&x1[1..3], &[a, b]);

        // X_e^2 contains the nodes at graph distance 1 of the endpoints.
        let x2 = nb.within(e, 2);
        for r in 1..=3 {
            for c in 1..=4 {
                assert!(x2.contains(&graph.lattice_node(r, c).unwrap()), "{r},{c}");
            }
        }
        assert!(!x2.contains(&graph.lattice_node(0, 0).unwrap()));
    }

    #[test]
    fn corner_and_boundary_elements() {
        let graph = ProcessorGraph::square_lattice(5, 5);
        let nb = Neighbourhoods::build(&graph, 2);
        // Corner node: 2 engineered + 1 parasitic incident edge.
        let corner = graph.lattice_node(0, 0).unwrap();
        assert_eq!(nb.within(corner, 1).len(), 1 + 3);
        // 2x2 block of nodes around the corner, plus every edge touching it.
        let x2 = nb.within(corner, 2);
        let nodes: BTreeSet<_> = x2
            .iter()
            .copied()
            .filter(|&e| graph.kind(e).is_node())
            .collect();
        let expected_nodes: BTreeSet<_> = [(0, 0), (0, 1), (1, 0), (1, 1)]
            .iter()
            .map(|&(r, c)| graph.lattice_node(r, c).unwrap())
            .collect();
        assert_eq!(nodes, expected_nodes);
        // Boundary (non-corner) node: 3 engineered + 2 parasitic incident edges.
        let side = graph.lattice_node(0, 2).unwrap();
        assert_eq!(nb.within(side, 1).len(), 1 + 5);
        let nodes_side = nb
            .within(side, 2)
            .iter()
            .filter(|&&e| graph.kind(e).is_node())
            .count();
        assert_eq!(nodes_side, 6);
        check_against_reference(&graph, 2);
    }

    #[test]
    fn distance_beyond_diameter_gives_whole_component() {
        let graph = ProcessorGraph::square_lattice(3, 4);
        let nb = Neighbourhoods::build(&graph, 12);
        for g in graph.ids() {
            assert_eq!(nb.within(g, 12).len(), graph.len());
            assert_eq!(nb.within(g, 8), nb.within(g, 12));
        }
        // Memory does not blow up with the huge radius: rings stop at the eccentricity.
        assert!(nb.memory_bytes() < 100_000);
    }

    #[test]
    fn disconnected_components_do_not_mix() {
        let mut b = GraphBuilder::new();
        let a0 = b.add_node("a0").unwrap();
        let a1 = b.add_node("a1").unwrap();
        let c0 = b.add_node("c0").unwrap();
        let lone = b.add_node("lone").unwrap();
        let ea = b.add_edge(a0, a1, EdgeType::Engineered).unwrap();
        let graph = b.build().unwrap();
        let nb = Neighbourhoods::build(&graph, 3);
        assert_eq!(set_of(nb.within(a0, 3)), [a0, a1, ea].into_iter().collect());
        assert_eq!(nb.within(c0, 3), &[c0]);
        assert_eq!(nb.within(lone, 3), &[lone]);
        assert_eq!(nb.incidence_distance(a0, c0), None);
        assert!(!nb.is_within(a0, c0, 3));
        check_against_reference(&graph, 3);
        check_is_within(&graph, 3);
    }

    #[test]
    fn parallel_edges_and_custom_types() {
        let mut b = GraphBuilder::new();
        let x = b.add_node("x").unwrap();
        let y = b.add_node("y").unwrap();
        let e1 = b
            .add_edge_labelled("e1", x, y, EdgeType::Engineered)
            .unwrap();
        let e2 = b
            .add_edge_labelled("e2", x, y, EdgeType::Custom(5))
            .unwrap();
        let graph = b.build().unwrap();
        let nb = Neighbourhoods::build(&graph, 2);
        assert_eq!(nb.within(x, 1), &[x, e1, e2]);
        assert_eq!(nb.incidence_distance(e1, e2), Some(2));
        assert!(nb.is_within(e1, e2, 1));
        assert!(nb.is_within(e2, e1, 1));
        assert!(!nb.is_within(e1, e2, 0));
    }

    // ----- exhaustive checks on lattices -----

    #[test]
    fn lattices_match_reference() {
        for (rows, cols) in [(1, 1), (1, 2), (1, 5), (5, 1), (2, 2), (3, 5), (4, 4)] {
            let graph = ProcessorGraph::square_lattice(rows, cols);
            check_against_reference(&graph, 3);
        }
    }

    #[test]
    fn is_within_symmetric_on_lattices() {
        for (rows, cols) in [(1, 6), (6, 1), (2, 2), (3, 5)] {
            let graph = ProcessorGraph::square_lattice(rows, cols);
            check_is_within(&graph, 3);
        }
    }

    #[test]
    fn nodes_only_lattice_row_matches_graph_hops() {
        // On a 1xN lattice (a path) X_n^d holds the nodes within d - 1 hops (just n for
        // d = 0 and d = 1) and the edges incident to them.
        let graph = ProcessorGraph::square_lattice(1, 9);
        let nb = Neighbourhoods::build(&graph, 3);
        let n = graph.lattice_node(0, 4).unwrap();
        for d in 0..=3usize {
            let hops = d.saturating_sub(1);
            let nodes: BTreeSet<usize> = nb
                .within(n, d as u32)
                .iter()
                .filter(|&&e| graph.kind(e).is_node())
                .map(|e| e.index())
                .collect();
            let expected: BTreeSet<usize> =
                (4usize.saturating_sub(hops)..=(4 + hops).min(8)).collect();
            assert_eq!(nodes, expected, "d={d}");
            let edges = nb
                .within(n, d as u32)
                .iter()
                .filter(|&&e| graph.kind(e).is_edge())
                .count();
            let expected_edges = if d == 0 { 0 } else { 2 * d };
            assert_eq!(edges, expected_edges, "d={d}");
        }
    }

    #[test]
    fn ring_accessor() {
        let graph = ProcessorGraph::square_lattice(3, 3);
        let nb = Neighbourhoods::build(&graph, 2);
        let n = graph.lattice_node(1, 1).unwrap();
        assert_eq!(nb.ring(n, 0), &[n]);
        assert_eq!(set_of(nb.ring(n, 1)), set_of(graph.incident(n)));
        assert!(nb.ring(n, 2).iter().all(|&e| graph.kind(e).is_node()));
        assert_eq!(nb.ring(n, 2).len(), 8);
        assert!(nb.ring(n, 40).is_empty());
        let total: usize = (0..=3).map(|r| nb.ring(n, r).len()).sum();
        assert_eq!(total, nb.within(n, 2).len());
    }

    // ----- random graphs -----

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(48))]

        #[test]
        fn random_graphs_match_reference(seed in any::<u64>(), max_distance in 0u32..=3) {
            let graph = random_graph(seed, 9, 16);
            check_against_reference(&graph, max_distance);
        }

        #[test]
        fn random_graphs_is_within_symmetric(seed in any::<u64>()) {
            let graph = random_graph(seed, 8, 14);
            check_is_within(&graph, 3);
        }

        #[test]
        fn within_is_monotone_in_d(seed in any::<u64>()) {
            let graph = random_graph(seed, 9, 16);
            let nb = Neighbourhoods::build(&graph, 3);
            for g in graph.ids() {
                for d in 0..3 {
                    let small = nb.within(g, d);
                    let large = nb.within(g, d + 1);
                    prop_assert!(large.starts_with(small));
                }
            }
        }
    }

    // ----- compact offsets -----

    #[test]
    fn offset_conversion_accepts_u32_range() {
        assert_eq!(offset_u32(0), 0);
        assert_eq!(offset_u32(u32::MAX as usize), u32::MAX);
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    #[should_panic(expected = "exceeds 2^32 entries")]
    fn offset_conversion_rejects_overflow() {
        offset_u32(u32::MAX as usize + 1);
    }

    #[test]
    fn memory_grows_with_max_distance_but_stays_bounded_at_depth_three() {
        let graph = ProcessorGraph::square_lattice(100, 100);
        let two = Neighbourhoods::build(&graph, 2).memory_bytes();
        let three = Neighbourhoods::build(&graph, 3).memory_bytes();
        assert!(three > two);
        // Documented figure: about 37 MB at max_distance 3 (4 bytes per ball entry).
        assert!((three as f64) < 45e6, "{three} bytes");
    }

    // ----- scale -----

    #[test]
    fn ten_thousand_qubit_lattice_is_small_and_fast() {
        let graph = ProcessorGraph::square_lattice(100, 100);
        assert!(graph.len() > 49_000);
        let started = std::time::Instant::now();
        let nb = Neighbourhoods::build(&graph, 2);
        let elapsed = started.elapsed();
        if !cfg!(debug_assertions) {
            assert!(
                elapsed < std::time::Duration::from_secs(1),
                "indexing took {elapsed:?}"
            );
        }
        let mb = nb.memory_bytes() as f64 / 1e6;
        // ARCHITECTURE §12.1 budgets about 25 MB for 10,000 qubits at max_distance 2.
        assert!(mb < 25.0, "index uses {mb} MB");
        // An interior node sees a 3x3 block and an interior edge a bigger region.
        let n = graph.lattice_node(50, 50).unwrap();
        assert_eq!(nb.within(n, 2).len(), 61);
    }
}
