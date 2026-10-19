//! Scoping a calibration step: parameters, constraints and traversal options
//! (ARCHITECTURE §4.4, paper §"calibration parameters / constraints / traversal options").
//!
//! Everything here is pure and synchronous: data in, data out. Every function writes into a
//! caller-supplied buffer and only visits the neighbourhood slices of the elements it is
//! given, so its work is proportional to those slices and never to the processor size.
//!
//! # Paper figures reproduced by the tests
//!
//! With the XEB algorithm subgraphs ([`xeb_square_lattice`](crate::activity::AlgorithmSubgraphs::xeb_square_lattice)) and
//! `d_T = 2`:
//!
//! * an interior node has exactly 8 traversal options (4 nearest and 4 diagonal neighbours);
//! * an interior engineered edge has exactly 6 when only engineered edges are available
//!   (for a horizontal edge: 2 above, 2 below and 2 in line with it), because only edges of
//!   the same two-qubit layer pattern can be active together;
//! * with `d_P = 1`, a node-centred step on the goal (nodes and engineered edges) has 1 to
//!   5 parameters: the node and up to 4 incident engineered edges.
//!
//! A sequential Snake over a 6x6 lattice (`d_P = 1`, `d_R = 2`, `d_T = 2`, breadth-first
//! from the lowest node ID) gives 1 to 3 parameters (1 to 5 from an interior seed) and 0 to
//! 18 constraints per step (0 to 27 depth-first from an interior seed on an 8x8 lattice).
//!
//! **Open discrepancy with the paper.** The paper reports 0 to 8 constraints per step for
//! these distances. The definition implemented here (ARCHITECTURE §4.4: every calibrated,
//! co-active element within `d_R` of *any* parameter, edges included) gives larger numbers.
//! No reading of the definition reproduces 8, except counting nodes only under a depth-first
//! walk. Counting around the centre only, or around all parameters, or breadth-first versus
//! depth-first, gives maxima between 4 and 27. Which definition the paper uses is a design
//! decision, not a code defect, so the code follows the documented definition and the tests
//! pin the numbers it produces. See `sequential_snake_on_6x6_parameter_bound_and_observed_constraints`.

use crate::activity::ActivityMasks;
use crate::graph::{ElementId, ProcessorGraph};
use crate::neighbourhoods::Neighbourhoods;
use crate::sets::ElementSet;

/// The three distances that control the Snake.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Distances {
    /// `d_P`: how many parameters are optimized together in one step.
    pub parameter: u32,
    /// `d_R`: how far away calibrated elements can be and still constrain the step.
    pub constraint: u32,
    /// `d_T`: how far the Snake can move from one step to the next.
    pub traversal: u32,
}

impl Distances {
    /// The largest of the three distances.
    pub fn max(&self) -> u32 {
        self.parameter.max(self.constraint).max(self.traversal)
    }

    /// `max(d_P, d_R)`: the separation that makes subgoals independent of each other.
    pub fn separation(&self) -> u32 {
        self.parameter.max(self.constraint)
    }
}

/// Read-only context for scoping steps.
#[derive(Copy, Clone)]
pub struct Scope<'a> {
    /// The processor graph.
    pub graph: &'a ProcessorGraph,
    /// Neighbourhood index; must support at least `distances.max()`.
    pub neighbourhoods: &'a Neighbourhoods,
    /// Activity masks of all elements.
    pub activity: &'a ActivityMasks,
    /// The three Snake distances.
    pub distances: Distances,
}

impl<'a> Scope<'a> {
    /// Bundles the pieces.
    ///
    /// # Panics
    /// Panics if the neighbourhood index is too shallow for `distances`, or if the
    /// neighbourhood index or the activity masks do not cover exactly the elements of `graph`.
    pub fn new(
        graph: &'a ProcessorGraph,
        neighbourhoods: &'a Neighbourhoods,
        activity: &'a ActivityMasks,
        distances: Distances,
    ) -> Self {
        assert_eq!(
            neighbourhoods.len(),
            graph.len(),
            "the neighbourhood index covers {} elements, but the graph has {}",
            neighbourhoods.len(),
            graph.len()
        );
        assert_eq!(
            activity.len(),
            graph.len(),
            "the activity masks cover {} elements, but the graph has {}",
            activity.len(),
            graph.len()
        );
        assert!(
            neighbourhoods.max_distance() >= distances.max(),
            "neighbourhood index (max distance {}) is too shallow for distances {:?}",
            neighbourhoods.max_distance(),
            distances
        );
        Self {
            graph,
            neighbourhoods,
            activity,
            distances,
        }
    }

    /// Calibration parameters `P = within(centre, d_P) ∩ available`, written to `out`
    /// (cleared first) in `within` order: the centre first if it is available, then by
    /// incidence distance and ID. `available` is the set of elements that may still be
    /// calibrated here: the subgoal minus everything already calibrated.
    pub fn parameters(&self, centre: ElementId, available: &ElementSet, out: &mut Vec<ElementId>) {
        out.clear();
        out.extend(
            self.neighbourhoods
                .within(centre, self.distances.parameter)
                .iter()
                .copied()
                .filter(|&e| available.contains(e)),
        );
    }

    /// Active constraints `R`: every calibrated element `c` such that, for some parameter
    /// `p`, `c ∈ within(p, d_R)` and `c` and `p` can be active at the same time. Written to
    /// `out` (cleared first), sorted ascending by ID, without duplicates.
    pub fn constraints(
        &self,
        parameters: &[ElementId],
        calibrated: &ElementSet,
        out: &mut Vec<ElementId>,
    ) {
        out.clear();
        for &p in parameters {
            out.extend(
                self.neighbourhoods
                    .within(p, self.distances.constraint)
                    .iter()
                    .copied()
                    .filter(|&c| calibrated.contains(c) && self.activity.co_active(c, p)),
            );
        }
        out.sort_unstable();
        out.dedup();
    }

    /// Traversal options of `centre`: elements `e ∈ within(centre, d_T) ∩ available` with
    /// the same kind as `centre`, `e != centre`, that can be active at the same time as
    /// `centre`. Written to `out` (cleared first) in `within` order.
    pub fn traversal_options(
        &self,
        centre: ElementId,
        available: &ElementSet,
        out: &mut Vec<ElementId>,
    ) {
        out.clear();
        let kind = self.graph.kind(centre);
        out.extend(
            self.neighbourhoods
                .within(centre, self.distances.traversal)
                .iter()
                .copied()
                .filter(|&e| {
                    e != centre
                        && available.contains(e)
                        && self.graph.kind(e) == kind
                        && self.activity.co_active(centre, e)
                }),
        );
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, VecDeque};

    use proptest::prelude::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use super::*;
    use crate::activity::AlgorithmSubgraphs;
    use crate::graph::{EdgeType, ElementKind};
    use crate::neighbourhoods::test_support::random_graph;

    fn id(i: u32) -> ElementId {
        ElementId::new(i)
    }

    /// Everything needed to scope steps on a `rows x cols` lattice with the XEB subgraphs.
    struct Fixture {
        graph: ProcessorGraph,
        neighbourhoods: Neighbourhoods,
        activity: ActivityMasks,
        goal: ElementSet,
    }

    impl Fixture {
        fn xeb(rows: usize, cols: usize, max_distance: u32) -> Self {
            let graph = ProcessorGraph::square_lattice(rows, cols);
            let activity = AlgorithmSubgraphs::xeb_square_lattice(&graph, rows, cols)
                .unwrap()
                .masks();
            let goal = graph.goal_excluding(&[ElementKind::Edge(EdgeType::Parasitic)]);
            let neighbourhoods = Neighbourhoods::build(&graph, max_distance);
            Self {
                graph,
                neighbourhoods,
                activity,
                goal,
            }
        }

        fn scope(&self, p: u32, r: u32, t: u32) -> Scope<'_> {
            Scope::new(
                &self.graph,
                &self.neighbourhoods,
                &self.activity,
                Distances {
                    parameter: p,
                    constraint: r,
                    traversal: t,
                },
            )
        }

        fn node(&self, r: usize, c: usize) -> ElementId {
            self.graph.lattice_node(r, c).unwrap()
        }

        fn engineered(&self, a: (usize, usize), b: (usize, usize)) -> ElementId {
            self.graph
                .edge_between(
                    self.node(a.0, a.1),
                    self.node(b.0, b.1),
                    Some(EdgeType::Engineered),
                )
                .unwrap()
        }

        fn horizontal(&self, r: usize, c: usize) -> ElementId {
            self.engineered((r, c), (r, c + 1))
        }

        fn vertical(&self, r: usize, c: usize) -> ElementId {
            self.engineered((r, c), (r + 1, c))
        }
    }

    fn options(scope: &Scope<'_>, centre: ElementId, available: &ElementSet) -> Vec<ElementId> {
        let mut out = vec![id(999)]; // stale content must be cleared
        scope.traversal_options(centre, available, &mut out);
        out
    }

    fn sorted(v: &[ElementId]) -> Vec<ElementId> {
        let mut v = v.to_vec();
        v.sort_unstable();
        v
    }

    // ----- Distances and Scope::new -----

    #[test]
    fn distances_helpers() {
        let d = Distances {
            parameter: 1,
            constraint: 3,
            traversal: 2,
        };
        assert_eq!(d.max(), 3);
        assert_eq!(d.separation(), 3);
        let d = Distances {
            parameter: 0,
            constraint: 0,
            traversal: 5,
        };
        assert_eq!(d.max(), 5);
        assert_eq!(d.separation(), 0);
    }

    #[test]
    #[should_panic(expected = "too shallow")]
    fn scope_rejects_shallow_index() {
        let f = Fixture::xeb(3, 3, 1);
        let _ = f.scope(1, 2, 2);
    }

    // ----- traversal options: Fig. 4 -----

    #[test]
    fn fig4_interior_node_has_eight_options() {
        let f = Fixture::xeb(7, 7, 2);
        let scope = f.scope(1, 2, 2);
        let centre = f.node(3, 3);
        let got = options(&scope, centre, &f.goal);
        let mut want = Vec::new();
        for r in 2..=4 {
            for c in 2..=4 {
                if (r, c) != (3, 3) {
                    want.push(f.node(r, c));
                }
            }
        }
        assert_eq!(got.len(), 8);
        assert_eq!(sorted(&got), sorted(&want));
        // 4 nearest and 4 diagonal neighbours; `within` order: nearest by ID.
        assert!(got.iter().all(|&e| f.graph.kind(e).is_node()));
    }

    #[test]
    fn fig4_interior_horizontal_edge_has_six_options() {
        let f = Fixture::xeb(7, 7, 2);
        let scope = f.scope(1, 2, 2);
        for (r, c) in [(3, 2), (3, 3), (2, 2), (4, 3)] {
            let centre = f.horizontal(r, c);
            let got = options(&scope, centre, &f.goal);
            let want = vec![
                // two in line
                f.horizontal(r, c - 2),
                f.horizontal(r, c + 2),
                // two above
                f.horizontal(r - 1, c - 1),
                f.horizontal(r - 1, c + 1),
                // two below
                f.horizontal(r + 1, c - 1),
                f.horizontal(r + 1, c + 1),
            ];
            assert_eq!(sorted(&got), sorted(&want), "horizontal edge at ({r},{c})");
        }
    }

    #[test]
    fn fig4_interior_vertical_edge_has_six_options() {
        let f = Fixture::xeb(7, 7, 2);
        let scope = f.scope(1, 2, 2);
        let (r, c) = (2, 3);
        let centre = f.vertical(r, c);
        let got = options(&scope, centre, &f.goal);
        let want = vec![
            f.vertical(r - 2, c),
            f.vertical(r + 2, c),
            f.vertical(r - 1, c - 1),
            f.vertical(r + 1, c - 1),
            f.vertical(r - 1, c + 1),
            f.vertical(r + 1, c + 1),
        ];
        assert_eq!(sorted(&got), sorted(&want));
    }

    #[test]
    fn traversal_options_with_parasitic_edges_available_are_unchanged() {
        // Parasitic edges differ in kind (and are in no subgraph), so they never appear.
        let f = Fixture::xeb(7, 7, 2);
        let scope = f.scope(1, 2, 2);
        let everything = f.graph.all_elements();
        let centre = f.horizontal(3, 3);
        assert_eq!(options(&scope, centre, &everything).len(), 6);
        assert_eq!(options(&scope, f.node(3, 3), &everything).len(), 8);
    }

    #[test]
    fn all_active_gives_more_edge_options_than_xeb() {
        let f = Fixture::xeb(7, 7, 2);
        let all_active = AlgorithmSubgraphs::all_active(&f.graph).masks();
        let scope = Scope::new(
            &f.graph,
            &f.neighbourhoods,
            &all_active,
            Distances {
                parameter: 1,
                constraint: 2,
                traversal: 2,
            },
        );
        let centre = f.horizontal(3, 3);
        let wide = options(&scope, centre, &f.goal);
        let xeb = options(&f.scope(1, 2, 2), centre, &f.goal);
        assert!(wide.len() > xeb.len());
        assert!(xeb.iter().all(|e| wide.contains(e)));
    }

    #[test]
    fn traversal_options_respect_available_and_exclude_centre() {
        let f = Fixture::xeb(5, 5, 2);
        let scope = f.scope(1, 2, 2);
        let centre = f.node(2, 2);
        let mut available = f.goal.clone();
        available.remove(f.node(2, 3));
        available.remove(f.node(1, 1));
        let got = options(&scope, centre, &available);
        assert_eq!(got.len(), 6);
        assert!(!got.contains(&f.node(2, 3)));
        assert!(!got.contains(&centre));
        // The centre itself being unavailable changes nothing.
        available.remove(centre);
        assert_eq!(options(&scope, centre, &available).len(), 6);
        // Nothing available: no options.
        assert!(options(&scope, centre, &ElementSet::new(f.graph.len())).is_empty());
    }

    #[test]
    fn traversal_options_boundary_and_d_t_zero() {
        let f = Fixture::xeb(5, 5, 2);
        let corner = f.node(0, 0);
        assert_eq!(options(&f.scope(1, 2, 2), corner, &f.goal).len(), 3);
        // d_T = 0: only the centre is within reach, which is excluded.
        assert!(options(&f.scope(1, 2, 0), f.node(2, 2), &f.goal).is_empty());
        // d_T = 1: a node reaches no other node (only its incident edges).
        assert!(options(&f.scope(1, 2, 1), f.node(2, 2), &f.goal).is_empty());
    }

    #[test]
    fn traversal_options_in_within_order() {
        let f = Fixture::xeb(7, 7, 2);
        let scope = f.scope(1, 2, 2);
        let centre = f.node(3, 3);
        let got = options(&scope, centre, &f.goal);
        let order: Vec<ElementId> = f
            .neighbourhoods
            .within(centre, 2)
            .iter()
            .copied()
            .filter(|e| got.contains(e))
            .collect();
        assert_eq!(got, order);
    }

    #[test]
    fn traversal_options_of_inactive_centre_are_empty() {
        // A parasitic edge is in no subgraph, so it is never co-active with anything.
        let f = Fixture::xeb(5, 5, 2);
        let scope = f.scope(1, 2, 2);
        let para = f
            .graph
            .edges()
            .find(|&e| f.graph.kind(e).edge_type() == Some(EdgeType::Parasitic))
            .unwrap();
        assert!(options(&scope, para, &f.graph.all_elements()).is_empty());
    }

    // ----- parameters: Fig. 7 -----

    fn parameters(scope: &Scope<'_>, centre: ElementId, available: &ElementSet) -> Vec<ElementId> {
        let mut out = vec![id(999)];
        scope.parameters(centre, available, &mut out);
        out
    }

    #[test]
    fn fig7_node_step_has_one_to_five_parameters() {
        let f = Fixture::xeb(5, 5, 2);
        let scope = f.scope(1, 2, 2);
        // Interior node: the node and its 4 incident engineered edges.
        let n = f.node(2, 2);
        let p = parameters(&scope, n, &f.goal);
        assert_eq!(p.len(), 5);
        assert_eq!(p[0], n, "the centre comes first");
        assert!(p[1..]
            .iter()
            .all(|&e| f.graph.kind(e).edge_type() == Some(EdgeType::Engineered)));
        assert_eq!(
            sorted(&p[1..]),
            sorted(
                &f.graph
                    .incident(n)
                    .iter()
                    .copied()
                    .filter(|&e| f.graph.kind(e).edge_type() == Some(EdgeType::Engineered))
                    .collect::<Vec<_>>()
            )
        );
        // Boundary: 4, corner: 3.
        assert_eq!(parameters(&scope, f.node(0, 2), &f.goal).len(), 4);
        assert_eq!(parameters(&scope, f.node(0, 0), &f.goal).len(), 3);
        assert_eq!(parameters(&scope, f.node(4, 4), &f.goal).len(), 3);
        // Once its edges are calibrated, only the node is left.
        let mut available = f.goal.clone();
        for &e in &p[1..] {
            available.remove(e);
        }
        assert_eq!(parameters(&scope, n, &available), vec![n]);
        // Once the node itself is calibrated, only the remaining edges are.
        let mut available = f.goal.clone();
        available.remove(n);
        let p = parameters(&scope, n, &available);
        assert_eq!(p.len(), 4);
        assert!(!p.contains(&n));
        // Nothing available: no parameters.
        assert!(parameters(&scope, n, &ElementSet::new(f.graph.len())).is_empty());
    }

    #[test]
    fn parameters_d_p_zero_is_a_single_parameter() {
        let f = Fixture::xeb(5, 5, 2);
        let scope = f.scope(0, 2, 2);
        for centre in [f.node(2, 2), f.horizontal(2, 2), f.vertical(1, 3)] {
            assert_eq!(parameters(&scope, centre, &f.goal), vec![centre]);
            let mut available = f.goal.clone();
            available.remove(centre);
            assert!(parameters(&scope, centre, &available).is_empty());
        }
    }

    #[test]
    fn parameters_of_edge_centre() {
        let f = Fixture::xeb(5, 5, 2);
        let scope = f.scope(1, 2, 2);
        let e = f.horizontal(2, 2);
        let p = parameters(&scope, e, &f.goal);
        // The edge, its 2 endpoints, and the 6 other engineered edges at those endpoints.
        assert_eq!(p.len(), 1 + 2 + 6);
        assert_eq!(p[0], e);
        assert!(p[1..3].iter().all(|&n| f.graph.kind(n).is_node()));
    }

    #[test]
    fn parameters_d_p_two_covers_the_block() {
        let f = Fixture::xeb(5, 5, 2);
        let scope = f.scope(2, 2, 2);
        let p = parameters(&scope, f.node(2, 2), &f.goal);
        // 9 nodes and the engineered edges with an endpoint in the 3x3 block (24).
        assert_eq!(p.len(), 9 + 24);
    }

    // ----- constraints -----

    fn constraints(
        scope: &Scope<'_>,
        parameters: &[ElementId],
        calibrated: &ElementSet,
    ) -> Vec<ElementId> {
        let mut out = vec![id(999)];
        scope.constraints(parameters, calibrated, &mut out);
        out
    }

    #[test]
    fn constraints_are_sorted_deduplicated_and_calibrated() {
        let f = Fixture::xeb(6, 6, 2);
        let scope = f.scope(1, 2, 2);
        let mut calibrated = ElementSet::new(f.graph.len());
        // Calibrate the left two columns' goal elements.
        for (r, c) in (0..6).flat_map(|r| (0..2).map(move |c| (r, c))) {
            calibrated.insert(f.node(r, c));
        }
        let centre = f.node(2, 3);
        let mut available = f.goal.clone();
        available.difference_with(&calibrated);
        let params = parameters(&scope, centre, &available);
        let cons = constraints(&scope, &params, &calibrated);
        assert!(!cons.is_empty());
        assert!(cons.windows(2).all(|w| w[0] < w[1]), "sorted and distinct");
        assert!(cons.iter().all(|&c| calibrated.contains(c)));
        // Check against a brute-force definition.
        let want: BTreeSet<ElementId> = f
            .graph
            .ids()
            .filter(|&c| {
                calibrated.contains(c)
                    && params
                        .iter()
                        .any(|&p| f.neighbourhoods.is_within(p, c, 2) && f.activity.co_active(c, p))
            })
            .collect();
        assert_eq!(cons.iter().copied().collect::<BTreeSet<_>>(), want);
    }

    #[test]
    fn constraints_without_calibrated_elements_or_parameters_are_empty() {
        let f = Fixture::xeb(5, 5, 2);
        let scope = f.scope(1, 2, 2);
        let params = parameters(&scope, f.node(2, 2), &f.goal);
        assert!(constraints(&scope, &params, &ElementSet::new(f.graph.len())).is_empty());
        assert!(constraints(&scope, &[], &f.goal).is_empty());
    }

    #[test]
    fn constraints_require_co_activity() {
        let f = Fixture::xeb(7, 7, 2);
        let scope = f.scope(0, 2, 2);
        // Single parameter: the horizontal edge h(3,3) (pattern 0). Calibrated: h(3,4)
        // (pattern 1, shares the node q3_4) and h(3,5)... (pattern 0).
        let p = f.horizontal(3, 3);
        let mut calibrated = ElementSet::new(f.graph.len());
        let other_pattern = f.horizontal(3, 4);
        let same_pattern = f.horizontal(3, 5);
        let vertical = f.vertical(3, 3);
        let parasitic = f
            .graph
            .edge_between(f.node(3, 3), f.node(4, 4), Some(EdgeType::Parasitic))
            .unwrap();
        for e in [other_pattern, same_pattern, vertical, parasitic] {
            calibrated.insert(e);
        }
        let cons = constraints(&scope, &[p], &calibrated);
        assert_eq!(cons, vec![same_pattern]);
        // A node parameter is active in every pattern: it is constrained by all engineered
        // edges within reach, but never by parasitic edges.
        let node_cons = constraints(&scope, &[f.node(3, 4)], &calibrated);
        assert!(node_cons.contains(&other_pattern));
        assert!(node_cons.contains(&vertical));
        assert!(!node_cons.contains(&parasitic));
    }

    #[test]
    fn constraints_use_d_r() {
        let f = Fixture::xeb(7, 7, 3);
        let mut calibrated = ElementSet::new(f.graph.len());
        calibrated.insert(f.node(3, 5)); // two hops to the right of (3,3)
        let p = [f.node(3, 3)];
        // Two hops apart is incidence distance 4, so d_R = 3 (radius 5) is the first to reach.
        for d_r in 0..=2 {
            assert!(
                constraints(&f.scope(0, d_r, 0), &p, &calibrated).is_empty(),
                "d_R={d_r}"
            );
        }
        assert_eq!(
            constraints(&f.scope(0, 3, 0), &p, &calibrated),
            vec![f.node(3, 5)]
        );
    }

    /// Constraints are symmetric: `g` constrains `g'` exactly when `g'` constrains `g`.
    fn assert_constraints_symmetric(scope: &Scope<'_>, graph: &ProcessorGraph) {
        let d_r = scope.distances.constraint;
        for g in graph.ids() {
            for h in graph.ids().filter(|&h| h != g) {
                let calibrated_h = ElementSet::from_ids(graph.len(), [h]);
                let calibrated_g = ElementSet::from_ids(graph.len(), [g]);
                let mut a = Vec::new();
                let mut b = Vec::new();
                scope.constraints(&[g], &calibrated_h, &mut a);
                scope.constraints(&[h], &calibrated_g, &mut b);
                // h constrains the step on g  <=>  g constrains the step on h.
                assert_eq!(a.is_empty(), b.is_empty(), "{g} {h}");
                let expected =
                    scope.neighbourhoods.is_within(g, h, d_r) && scope.activity.co_active(g, h);
                assert_eq!(!a.is_empty(), expected, "{g} {h}");
            }
        }
    }

    #[test]
    fn constraints_are_symmetric_on_xeb_lattices() {
        for (rows, cols) in [(1, 5), (5, 1), (3, 4), (4, 4)] {
            let f = Fixture::xeb(rows, cols, 3);
            for r in 0..=3 {
                assert_constraints_symmetric(&f.scope(1, r, 2), &f.graph);
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn constraints_are_symmetric_on_random_graphs(seed in any::<u64>(), d_r in 0u32..=3) {
            let graph = random_graph(seed, 8, 14);
            let mut rng = StdRng::seed_from_u64(seed ^ 0x5eed);
            let subgraphs: Vec<ElementSet> = (0..rng.random_range(1..=4))
                .map(|_| {
                    let mut s = ElementSet::new(graph.len());
                    for e in graph.ids() {
                        if rng.random_bool(0.6) {
                            s.insert(e);
                        }
                    }
                    s
                })
                .collect();
            let activity = AlgorithmSubgraphs::new(subgraphs).unwrap().masks();
            let nb = Neighbourhoods::build(&graph, 3);
            let scope = Scope::new(
                &graph,
                &nb,
                &activity,
                Distances { parameter: 1, constraint: d_r, traversal: 2 },
            );
            assert_constraints_symmetric(&scope, &graph);
        }

        #[test]
        fn scope_outputs_are_subsets_of_their_definitions(seed in any::<u64>()) {
            let graph = random_graph(seed, 9, 16);
            let nb = Neighbourhoods::build(&graph, 3);
            let activity = AlgorithmSubgraphs::all_active(&graph).masks();
            let scope = Scope::new(
                &graph,
                &nb,
                &activity,
                Distances { parameter: 1, constraint: 2, traversal: 3 },
            );
            let mut rng = StdRng::seed_from_u64(seed);
            let mut available = ElementSet::new(graph.len());
            let mut calibrated = ElementSet::new(graph.len());
            for e in graph.ids() {
                match rng.random_range(0..3) {
                    0 => { available.insert(e); }
                    1 => { calibrated.insert(e); }
                    _ => {}
                }
            }
            let mut params = Vec::new();
            let mut cons = Vec::new();
            let mut opts = Vec::new();
            for g in graph.ids() {
                scope.parameters(g, &available, &mut params);
                prop_assert!(params.iter().all(|&p| available.contains(p) && nb.is_within(g, p, 1)));
                let expected = nb.within(g, 1).iter().filter(|&&e| available.contains(e)).count();
                prop_assert_eq!(params.len(), expected);
                scope.constraints(&params, &calibrated, &mut cons);
                prop_assert!(cons.iter().all(|&c| calibrated.contains(c)
                    && params.iter().any(|&p| nb.is_within(p, c, 2))));
                scope.traversal_options(g, &available, &mut opts);
                prop_assert!(opts.iter().all(|&o| o != g
                    && available.contains(o)
                    && graph.kind(o) == graph.kind(g)
                    && nb.is_within(g, o, 3)));
            }
        }
    }

    // ----- the whole-sequence simulation -----

    /// Per-step statistics of a sequential Snake simulation.
    struct Simulation {
        /// Number of parameters of each step.
        parameters: Vec<usize>,
        /// Number of constraints of each step.
        constraints: Vec<usize>,
        /// Number of constraints of each step that are nodes.
        node_constraints: Vec<usize>,
        /// Calibration status at the end.
        calibrated: ElementSet,
    }

    impl Simulation {
        fn range(values: &[usize]) -> (usize, usize) {
            (*values.iter().min().unwrap(), *values.iter().max().unwrap())
        }
    }

    /// Traversal order of the simulation.
    #[derive(Copy, Clone, PartialEq)]
    enum Order {
        /// First in, first out.
        Bfs,
        /// Last in, first out, options pushed so that the first one pops first.
        Dfs,
    }

    /// A simple sequential Snake using only the scope functions: it starts at `seed`, marks
    /// each step's parameters calibrated, and follows the traversal options.
    fn simulate(
        f: &Fixture,
        distances: (u32, u32, u32),
        seed: ElementId,
        order: Order,
    ) -> Simulation {
        let scope = f.scope(distances.0, distances.1, distances.2);
        let mut available = f.goal.clone();
        let mut calibrated = ElementSet::new(f.graph.len());
        let mut queue = VecDeque::from([seed]);
        let (mut params, mut cons, mut opts) = (Vec::new(), Vec::new(), Vec::new());
        let mut sim = Simulation {
            parameters: Vec::new(),
            constraints: Vec::new(),
            node_constraints: Vec::new(),
            calibrated: ElementSet::new(0),
        };
        loop {
            let next = match order {
                Order::Bfs => queue.pop_front(),
                Order::Dfs => queue.pop_back(),
            };
            let Some(g) = next else { break };
            if calibrated.contains(g) {
                continue;
            }
            scope.parameters(g, &available, &mut params);
            scope.constraints(&params, &calibrated, &mut cons);
            sim.parameters.push(params.len());
            sim.constraints.push(cons.len());
            sim.node_constraints
                .push(cons.iter().filter(|&&c| f.graph.kind(c).is_node()).count());
            for &p in &params {
                calibrated.insert(p);
                available.remove(p);
            }
            scope.traversal_options(g, &available, &mut opts);
            match order {
                Order::Bfs => queue.extend(opts.iter().copied()),
                Order::Dfs => queue.extend(opts.iter().rev().copied()),
            }
        }
        sim.calibrated = calibrated;
        sim
    }

    /// Sequential Snake on a 6x6 lattice (XEB subgraphs, `d_P = 1`, `d_R = 2`, `d_T = 2`),
    /// breadth-first from the lowest node ID (the corner `q0_0`).
    ///
    /// Observed with this implementation of the definitions in ARCHITECTURE §4.4:
    ///
    /// * parameters per step: 1 to 3 (never 5, because a breadth-first walk from a corner
    ///   arrives at each node with two of its four edges already calibrated); the paper's
    ///   bound of 1 to 5 holds;
    /// * constraints per step: 0 to 18. This is **more** than the paper's 0 to 8 (an open
    ///   discrepancy, not a match: the test asserts what this implementation produces). The
    ///   definition (every calibrated co-active element within `d_R = 2` of *any* parameter,
    ///   where an edge parameter reaches 4 incidence steps) counts engineered edges and nodes
    ///   alike; the paper's 8 is matched only if constraints are counted among nodes under
    ///   a depth-first walk (see `sequential_snake_depth_first_and_interior_seed`). The
    ///   discrepancy is reported in the implementation notes; the numbers asserted below are
    ///   the ones this implementation produces.
    #[test]
    fn sequential_snake_on_6x6_parameter_bound_and_observed_constraints() {
        let f = Fixture::xeb(6, 6, 2);
        let sim = simulate(&f, (1, 2, 2), f.graph.nodes().next().unwrap(), Order::Bfs);
        // Every node and engineered edge got calibrated, each exactly once.
        assert_eq!(sim.calibrated, f.goal);
        assert_eq!(sim.parameters.iter().sum::<usize>(), f.goal.len());
        // One step per node: node steps absorb their edges (Fig. 7).
        assert_eq!(sim.parameters.len(), 36);

        let (pmin, pmax) = Simulation::range(&sim.parameters);
        let (cmin, cmax) = Simulation::range(&sim.constraints);
        // The paper's parameter bound.
        assert!(pmin >= 1 && pmax <= 5);
        // What this walk produces.
        assert_eq!((pmin, pmax), (1, 3));
        assert_eq!(sim.parameters[0], 3, "a corner seed has 2 edges");
        assert_eq!(
            *sim.parameters.last().unwrap(),
            1,
            "the last node has no edges left"
        );
        assert_eq!((cmin, cmax), (0, 18));
        assert_eq!(sim.constraints[0], 0, "the seed step has no constraints");
        // Among the nodes alone, at most 5 constrain a step.
        assert_eq!(Simulation::range(&sim.node_constraints), (0, 5));
    }

    /// Interior seed and depth-first order: the 5-parameter steps of Fig. 7 occur, and the
    /// paper's "up to 8" constraints is what the *node* constraints reach.
    #[test]
    fn sequential_snake_depth_first_and_interior_seed() {
        let f = Fixture::xeb(7, 7, 2);
        let seed = f.node(3, 3);
        let bfs = simulate(&f, (1, 2, 2), seed, Order::Bfs);
        assert_eq!(bfs.calibrated, f.goal);
        assert_eq!(Simulation::range(&bfs.parameters), (1, 5));
        assert_eq!(bfs.parameters[0], 5, "an interior seed absorbs 4 edges");
        assert_eq!(Simulation::range(&bfs.constraints), (0, 18));

        let dfs = simulate(&f, (1, 2, 2), seed, Order::Dfs);
        assert_eq!(dfs.calibrated, f.goal);
        assert_eq!(Simulation::range(&dfs.parameters), (1, 5));
        assert_eq!(Simulation::range(&dfs.constraints), (0, 26));
        assert_eq!(Simulation::range(&dfs.node_constraints), (0, 8));
    }

    #[test]
    fn sequential_snake_d_p_zero_takes_one_parameter_per_step() {
        let f = Fixture::xeb(4, 4, 2);
        // With d_P = 0 and d_T = 2 only nodes are visited (edges are never traversal
        // options of a node), so only the nodes get calibrated.
        let sim = simulate(&f, (0, 2, 2), f.graph.nodes().next().unwrap(), Order::Bfs);
        assert!(sim.parameters.iter().all(|&p| p == 1));
        assert_eq!(sim.parameters.len(), 16);
        assert_eq!(sim.calibrated.len(), 16);
    }

    #[test]
    fn sequential_snake_covers_other_lattices() {
        for (rows, cols) in [(1, 6), (6, 1), (2, 2), (3, 5), (5, 5)] {
            let f = Fixture::xeb(rows, cols, 2);
            let sim = simulate(&f, (1, 2, 2), f.graph.nodes().next().unwrap(), Order::Bfs);
            assert_eq!(sim.calibrated, f.goal, "{rows}x{cols}");
            assert!(sim.parameters.iter().all(|&p| (1..=5).contains(&p)));
        }
    }

    /// Smallest per-call time in nanoseconds of `step`, over several batches.
    fn best_time_per_call(mut step: impl FnMut()) -> f64 {
        const CALLS: u32 = 200;
        let mut best = f64::MAX;
        for _ in 0..7 {
            let started = std::time::Instant::now();
            for _ in 0..CALLS {
                step();
            }
            best = best.min(started.elapsed().as_nanos() as f64 / f64::from(CALLS));
        }
        best
    }

    #[test]
    fn work_is_independent_of_lattice_size() {
        // The same step in the middle of a 1x20 and of a 1x200000 lattice (10,000 times the
        // elements, but a tiny ball per element, so the step itself stays cheap). Work
        // proportional to the neighbourhood slices takes the same time on both. Per-call work
        // that visits or allocates something proportional to the processor does not: a 400 KB
        // buffer per call costs tens of microseconds against a step of about a microsecond.
        // Checked to fail by injecting `vec![1u8; self.graph.len()]` into `parameters`,
        // `constraints` and `traversal_options` one at a time (a 2D lattice of 50,000 elements
        // is too small for that: its memset is cheaper than the step).
        let small = Fixture::xeb(1, 20, 2);
        let large = Fixture::xeb(1, 200_000, 2);
        let measure = |f: &Fixture, centre: ElementId| {
            let scope = f.scope(1, 2, 2);
            let mut calibrated = ElementSet::new(f.graph.len());
            calibrated.insert(f.node(0, centre_col(f) - 3));
            let (mut params, mut cons, mut opts) = (Vec::new(), Vec::new(), Vec::new());
            best_time_per_call(|| {
                scope.parameters(centre, &f.goal, &mut params);
                scope.constraints(&params, &calibrated, &mut cons);
                scope.traversal_options(centre, &f.goal, &mut opts);
                std::hint::black_box((&params, &cons, &opts));
            })
        };
        fn centre_col(f: &Fixture) -> usize {
            f.graph.nodes().count() / 2
        }
        let (a, b) = (
            measure(&small, small.node(0, centre_col(&small))),
            measure(&large, large.node(0, centre_col(&large))),
        );
        assert!(
            b < 3.0 * a.max(1500.0),
            "small {a} ns, large {b} ns per step"
        );
    }

    #[test]
    fn scope_functions_reuse_the_output_buffer() {
        // With enough capacity, no output buffer is reallocated or grown with the lattice.
        let f = Fixture::xeb(60, 60, 2);
        let scope = f.scope(1, 2, 2);
        let centre = f.node(30, 30);
        let mut out: Vec<ElementId> = Vec::with_capacity(256);
        let ptr = out.as_ptr();
        scope.parameters(centre, &f.goal, &mut out);
        scope.traversal_options(centre, &f.goal, &mut out);
        let params = {
            let mut p = Vec::new();
            scope.parameters(centre, &f.goal, &mut p);
            p
        };
        scope.constraints(&params, &f.goal, &mut out);
        assert_eq!(out.as_ptr(), ptr);
        assert_eq!(out.capacity(), 256);
    }

    #[test]
    #[should_panic(expected = "neighbourhood index covers")]
    fn scope_rejects_a_neighbourhood_index_of_another_graph() {
        let small = ProcessorGraph::square_lattice(1, 2);
        let big = Fixture::xeb(3, 3, 2);
        let nb = Neighbourhoods::build(&small, 2);
        let _ = Scope::new(
            &big.graph,
            &nb,
            &big.activity,
            Distances {
                parameter: 1,
                constraint: 2,
                traversal: 2,
            },
        );
    }

    #[test]
    #[should_panic(expected = "activity masks cover")]
    fn scope_rejects_activity_masks_of_another_graph() {
        let small = ProcessorGraph::square_lattice(1, 2);
        let big = Fixture::xeb(3, 3, 2);
        let masks = AlgorithmSubgraphs::all_active(&small).masks();
        let _ = Scope::new(
            &big.graph,
            &big.neighbourhoods,
            &masks,
            Distances {
                parameter: 1,
                constraint: 2,
                traversal: 2,
            },
        );
    }
}
