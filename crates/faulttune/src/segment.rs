//! Segmentation of the uncalibrated goal into subgoals and traversal threads, seed
//! policies, and stitch plans (ARCHITECTURE §4.2 and §4.5).
//!
//! **Subgoals.** Two uncalibrated elements are linked if one is within
//! `max(d_P, d_R)` of the other; subgoals are the connected components, ordered by their
//! smallest element ID. Elements of different subgoals are therefore more than
//! `max(d_P, d_R)` apart, so no subgoal ever reads another's writes and subgoals can be
//! calibrated in parallel with exactly the same result as sequentially.
//!
//! **Threads.** Within a subgoal, two elements are linked if they are within `d_T`, can be
//! active at the same time, and have the same [`ElementKind`]. Threads are the connected
//! components. They are ordered: node threads first, then edge threads (by edge type
//! `Engineered < Parasitic < Custom(n)`), then by smallest element ID.
//!
//! Segmentation deliberately ignores activity masks when forming subgoals.

use std::sync::Arc;

use rand::rngs::StdRng;
use rand::Rng;

use crate::error::FaultTuneError;
use crate::graph::{ElementId, ElementKind};
use crate::scope::Scope;
use crate::sets::ElementSet;

/// A piece of a subgoal that the traversal rule cannot cross between. Each thread is
/// seeded once and walked to completion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Thread {
    /// Position within the subgoal's thread order.
    pub index: usize,
    /// The kind shared by all elements of the thread.
    pub kind: ElementKind,
    /// Members, ascending by ID.
    pub elements: Vec<ElementId>,
}

/// A region of uncalibrated elements far enough from every other region that no element
/// in one constrains an element in another.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Subgoal {
    /// Position in the pass (ordered by smallest member ID).
    pub index: usize,
    /// Members, ascending by ID.
    pub elements: Vec<ElementId>,
    /// The same members as a set (capacity = graph element count).
    pub set: ElementSet,
    /// The subgoal's traversal threads, in processing order.
    pub threads: Vec<Thread>,
}

/// Label of an element that no component has claimed yet.
const UNASSIGNED: u32 = u32::MAX;

/// Connected components of `members` (ascending IDs) under the relation "`v` is within
/// `radius` of `u`, `contains(v)` holds and `linked(u, v)` holds".
///
/// The relation must be symmetric. Components are returned in order of their smallest member,
/// each with its members ascending. `label` is scratch space of at least `members.len()`
/// entries, one per member and indexed by the member's position in `members` (found by binary
/// search), all `UNASSIGNED` on entry and again on exit. Its size therefore depends on the set
/// being segmented, never on the processor. Time is linear in the total size of the members'
/// neighbourhoods (times a logarithmic position lookup).
fn components(
    scope: &Scope<'_>,
    members: &[ElementId],
    radius: u32,
    contains: impl Fn(ElementId) -> bool,
    linked: impl Fn(ElementId, ElementId) -> bool,
    label: &mut [u32],
) -> Vec<Vec<ElementId>> {
    let mut count = 0u32;
    let mut stack = Vec::new();
    for (start_pos, &start) in members.iter().enumerate() {
        if label[start_pos] != UNASSIGNED {
            continue;
        }
        label[start_pos] = count;
        stack.push(start);
        while let Some(u) = stack.pop() {
            for &v in scope.neighbourhoods.within(u, radius) {
                if v == u || !contains(v) {
                    continue;
                }
                // `contains(v)` holds, so `v` is one of the members.
                let Ok(pos) = members.binary_search(&v) else {
                    continue;
                };
                if label[pos] == UNASSIGNED && linked(u, v) {
                    label[pos] = count;
                    stack.push(v);
                }
            }
        }
        count += 1;
    }
    // Members are ascending, so every bucket comes out ascending too.
    let mut buckets: Vec<Vec<ElementId>> = vec![Vec::new(); count as usize];
    for (pos, &m) in members.iter().enumerate() {
        buckets[label[pos] as usize].push(m);
        label[pos] = UNASSIGNED;
    }
    buckets
}

/// Wraps components as subgoals without threads.
fn subgoals_from(components: Vec<Vec<ElementId>>, capacity: usize) -> Vec<Subgoal> {
    components
        .into_iter()
        .enumerate()
        .map(|(index, elements)| Subgoal {
            index,
            set: ElementSet::from_ids(capacity, elements.iter().copied()),
            elements,
            threads: Vec::new(),
        })
        .collect()
}

/// The threads of one subgoal given as ascending `members` and the same `set`.
fn threads_of(scope: &Scope<'_>, members: &[ElementId], set: &ElementSet) -> Vec<Thread> {
    let graph = scope.graph;
    let activity = scope.activity;
    let components = components(
        scope,
        members,
        scope.distances.traversal,
        |v| set.contains(v),
        |u, v| graph.kind(u) == graph.kind(v) && activity.co_active(u, v),
        &mut vec![UNASSIGNED; members.len()],
    );
    let mut threads: Vec<Thread> = components
        .into_iter()
        .map(|elements| Thread {
            index: 0,
            kind: graph.kind(elements[0]),
            elements,
        })
        .collect();
    // Node threads first, then edge threads by edge type, then by smallest element.
    threads.sort_by_key(|t| (t.kind, t.elements[0]));
    for (index, thread) in threads.iter_mut().enumerate() {
        thread.index = index;
    }
    threads
}

/// Panics with a clear message if a set does not cover exactly the graph's elements.
fn check_capacity(scope: &Scope<'_>, set: &ElementSet) {
    assert_eq!(
        set.capacity(),
        scope.graph.len(),
        "the element set has capacity {}, but the graph has {} elements",
        set.capacity(),
        scope.graph.len()
    );
}

/// The subgoals of `uncalibrated`, without threads.
fn subgoals_of(scope: &Scope<'_>, uncalibrated: &ElementSet) -> Vec<Subgoal> {
    check_capacity(scope, uncalibrated);
    let members = uncalibrated.to_vec();
    let components = components(
        scope,
        &members,
        scope.distances.separation(),
        |v| uncalibrated.contains(v),
        |_, _| true,
        &mut vec![UNASSIGNED; members.len()],
    );
    subgoals_from(components, uncalibrated.capacity())
}

/// Splits `uncalibrated` into subgoals, each already split into threads.
///
/// Scratch memory is proportional to `uncalibrated.len()`. Each [`Subgoal`] carries its
/// members as a full-capacity [`ElementSet`] (the skeleton's field type), so `k` subgoals cost
/// `O(k * N / 64)` words in total for a processor of `N` elements.
///
/// # Panics
/// Panics if `uncalibrated.capacity()` differs from the number of elements of the graph.
pub fn segment(scope: &Scope<'_>, uncalibrated: &ElementSet) -> Vec<Subgoal> {
    let mut subgoals = subgoals_of(scope, uncalibrated);
    for subgoal in &mut subgoals {
        subgoal.threads = threads_of(scope, &subgoal.elements, &subgoal.set);
    }
    subgoals
}

/// Splits only into subgoals (no threads); `Subgoal::threads` is empty.
///
/// # Panics
/// Panics if `uncalibrated.capacity()` differs from the number of elements of the graph.
pub fn segment_subgoals(scope: &Scope<'_>, uncalibrated: &ElementSet) -> Vec<Subgoal> {
    subgoals_of(scope, uncalibrated)
}

/// Splits one subgoal's elements into threads, in processing order.
///
/// Besides the `O(capacity / 64)` cost of listing the members of `subgoal`, the work is
/// proportional to the subgoal and its neighbourhoods.
///
/// # Panics
/// Panics if `subgoal.capacity()` differs from the number of elements of the graph.
pub fn segment_threads(scope: &Scope<'_>, subgoal: &ElementSet) -> Vec<Thread> {
    check_capacity(scope, subgoal);
    threads_of(scope, &subgoal.to_vec(), subgoal)
}

/// A user function that chooses the seed of a thread: see [`SeedPolicy::Custom`].
pub type SeedFn = dyn Fn(&Thread, &mut StdRng) -> ElementId + Send + Sync;

/// Chooses the seed element of a thread.
#[derive(Clone, Default)]
pub enum SeedPolicy {
    /// The smallest element ID in the thread (deterministic; the default).
    #[default]
    LowestId,
    /// The first listed element contained in the thread; the lowest ID if none is.
    Fixed(Vec<ElementId>),
    /// A uniformly random member, drawn from the subgoal's seeded RNG.
    Random,
    /// A user function. Returning an element outside the thread can make a pass stall
    /// (`FaultTuneError::NoProgress`).
    Custom(Arc<SeedFn>),
}

impl std::fmt::Debug for SeedPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LowestId => f.write_str("LowestId"),
            Self::Fixed(ids) => f.debug_tuple("Fixed").field(ids).finish(),
            Self::Random => f.write_str("Random"),
            Self::Custom(_) => f.write_str("Custom(..)"),
        }
    }
}

impl SeedPolicy {
    /// The seed for `thread`.
    ///
    /// `Fixed` looks the preferred IDs up with a linear scan, so it also works for threads
    /// whose members are not sorted. Only `Random` draws from `rng`, and it draws exactly once.
    ///
    /// # Panics
    /// Panics if `thread` has no elements (threads built by [`segment`] never are empty) and
    /// the policy is not `Custom`.
    pub fn seed(&self, thread: &Thread, rng: &mut StdRng) -> ElementId {
        let lowest = || {
            thread
                .elements
                .iter()
                .copied()
                .min()
                .expect("a thread has at least one element")
        };
        match self {
            Self::LowestId => lowest(),
            Self::Fixed(preferred) => preferred
                .iter()
                .copied()
                .find(|id| thread.elements.contains(id))
                .unwrap_or_else(lowest),
            Self::Random => {
                assert!(
                    !thread.elements.is_empty(),
                    "a thread has at least one element"
                );
                thread.elements[rng.random_range(0..thread.elements.len())]
            }
            Self::Custom(choose) => choose(thread, rng),
        }
    }
}

/// A plan for calibrating a large processor in constraint-disjoint regions in parallel and
/// then *stitching* them together (ARCHITECTURE §4.5).
///
/// `gap` holds every goal element that lies in no region or lies within `max(d_P, d_R)` of
/// an element of a *different* region; `stage_one` is the rest of the regions. Stage-one
/// cores of different regions are more than `max(d_P, d_R)` apart, so they segment into
/// separate subgoals and run in parallel. Stage two calibrates the gap under constraints
/// from both sides.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StitchPlan {
    /// The user-chosen regions (disjoint, subsets of the goal).
    pub regions: Vec<ElementSet>,
    /// Elements calibrated in stage one: the regions minus the gap.
    pub stage_one: ElementSet,
    /// Elements calibrated in stage two.
    pub gap: ElementSet,
}

impl StitchPlan {
    /// Computes the plan for `goal` split into `regions`.
    ///
    /// Errors with `InvalidConfig` if regions overlap or contain elements outside `goal`.
    pub fn new(
        scope: &Scope<'_>,
        goal: &ElementSet,
        regions: Vec<ElementSet>,
    ) -> Result<Self, FaultTuneError> {
        let capacity = goal.capacity();
        if capacity != scope.graph.len() {
            return Err(FaultTuneError::InvalidConfig(format!(
                "the goal has capacity {capacity}, but the graph has {} elements",
                scope.graph.len()
            )));
        }
        let mut covered = ElementSet::new(capacity);
        let mut region_of = vec![UNASSIGNED; capacity];
        for (i, region) in regions.iter().enumerate() {
            if region.capacity() != capacity {
                return Err(FaultTuneError::InvalidConfig(format!(
                    "region {i} has capacity {}, but the goal has capacity {capacity}",
                    region.capacity()
                )));
            }
            if !region.is_subset(goal) {
                return Err(FaultTuneError::InvalidConfig(format!(
                    "region {i} contains elements outside the goal"
                )));
            }
            if !region.is_disjoint(&covered) {
                return Err(FaultTuneError::InvalidConfig(format!(
                    "region {i} overlaps an earlier region"
                )));
            }
            covered.union_with(region);
            for e in region {
                region_of[e.index()] = i as u32;
            }
        }

        // Goal elements outside every region, plus region elements that sit within the
        // separation of an element of a different region (on either side of a boundary).
        let mut gap = goal.difference(&covered);
        let separation = scope.distances.separation();
        for (i, region) in regions.iter().enumerate() {
            for g in region {
                let near_other = scope.neighbourhoods.within(g, separation).iter().any(|&h| {
                    let owner = region_of[h.index()];
                    owner != UNASSIGNED && owner != i as u32
                });
                if near_other {
                    gap.insert(g);
                }
            }
        }
        let stage_one = covered.difference(&gap);
        Ok(Self {
            regions,
            stage_one,
            gap,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use proptest::prelude::*;
    use rand::SeedableRng;

    use super::*;
    use crate::activity::{ActivityMasks, AlgorithmSubgraphs};
    use crate::graph::{EdgeType, GraphBuilder, ProcessorGraph};
    use crate::neighbourhoods::test_support::random_graph;
    use crate::neighbourhoods::Neighbourhoods;
    use crate::scope::Distances;

    fn id(i: u32) -> ElementId {
        ElementId::new(i)
    }

    fn ids(v: &[u32]) -> Vec<ElementId> {
        v.iter().map(|&i| id(i)).collect()
    }

    /// Graph, neighbourhood index and activity masks.
    struct World {
        graph: ProcessorGraph,
        neighbourhoods: Neighbourhoods,
        activity: ActivityMasks,
    }

    impl World {
        fn new(graph: ProcessorGraph, activity: ActivityMasks) -> Self {
            let neighbourhoods = Neighbourhoods::build(&graph, 3);
            Self {
                graph,
                neighbourhoods,
                activity,
            }
        }

        fn xeb(rows: usize, cols: usize) -> Self {
            let graph = ProcessorGraph::square_lattice(rows, cols);
            let activity = AlgorithmSubgraphs::xeb_square_lattice(&graph, rows, cols)
                .unwrap()
                .masks();
            Self::new(graph, activity)
        }

        fn all_active(graph: ProcessorGraph) -> Self {
            let activity = AlgorithmSubgraphs::all_active(&graph).masks();
            Self::new(graph, activity)
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

        fn set(&self, members: &[ElementId]) -> ElementSet {
            ElementSet::from_ids(self.graph.len(), members.iter().copied())
        }

        fn goal(&self) -> ElementSet {
            self.graph
                .goal_excluding(&[ElementKind::Edge(EdgeType::Parasitic)])
        }
    }

    fn rng(seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }

    // ----- naive references -----

    /// Connected components by repeated merging over all pairs (quadratic, obviously right).
    fn reference_components(
        members: &[ElementId],
        linked: impl Fn(ElementId, ElementId) -> bool,
    ) -> BTreeSet<Vec<ElementId>> {
        let mut parent: Vec<usize> = (0..members.len()).collect();
        fn find(parent: &mut [usize], x: usize) -> usize {
            let mut root = x;
            while parent[root] != root {
                root = parent[root];
            }
            parent[x] = root;
            root
        }
        for (i, &a) in members.iter().enumerate() {
            for (j, &b) in members.iter().enumerate().skip(i + 1) {
                if linked(a, b) {
                    let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                    parent[a] = b;
                }
            }
        }
        let mut groups: std::collections::BTreeMap<usize, Vec<ElementId>> = Default::default();
        for (i, &m) in members.iter().enumerate() {
            let root = find(&mut parent, i);
            groups.entry(root).or_default().push(m);
        }
        groups.into_values().collect()
    }

    /// Checks every promise of `segment` against references, for one scope and set.
    fn check_segmentation(scope: &Scope<'_>, uncalibrated: &ElementSet) {
        let nb = scope.neighbourhoods;
        let graph = scope.graph;
        let sep = scope.distances.separation();
        let d_t = scope.distances.traversal;
        let subgoals = segment(scope, uncalibrated);

        // Partition of the uncalibrated set; sorted members; consistent set.
        let mut union = ElementSet::new(uncalibrated.capacity());
        for (i, sg) in subgoals.iter().enumerate() {
            assert_eq!(sg.index, i);
            assert!(!sg.elements.is_empty());
            assert!(sg.elements.windows(2).all(|w| w[0] < w[1]));
            assert_eq!(sg.set.capacity(), uncalibrated.capacity());
            assert_eq!(sg.set.to_vec(), sg.elements);
            assert!(union.is_disjoint(&sg.set), "subgoals overlap");
            union.union_with(&sg.set);
        }
        assert_eq!(&union, uncalibrated);
        // Ordered by smallest element.
        assert!(subgoals
            .windows(2)
            .all(|w| w[0].elements[0] < w[1].elements[0]));
        // Different subgoals are more than `sep` apart.
        for (i, a) in subgoals.iter().enumerate() {
            for b in &subgoals[i + 1..] {
                for &x in &a.elements {
                    for &y in &b.elements {
                        assert!(!nb.is_within(x, y, sep), "{x} and {y} are within {sep}");
                    }
                }
            }
        }
        // Equal to the naive connected components.
        let reference =
            reference_components(&uncalibrated.to_vec(), |x, y| nb.is_within(x, y, sep));
        let got: BTreeSet<Vec<ElementId>> = subgoals.iter().map(|s| s.elements.clone()).collect();
        assert_eq!(got, reference);

        // Subgoals alone agree with the full segmentation, minus the threads.
        let plain = segment_subgoals(scope, uncalibrated);
        assert_eq!(plain.len(), subgoals.len());
        for (p, s) in plain.iter().zip(&subgoals) {
            assert!(p.threads.is_empty());
            assert_eq!(
                (p.index, &p.elements, &p.set),
                (s.index, &s.elements, &s.set)
            );
        }

        for sg in &subgoals {
            // Threads partition the subgoal.
            let mut covered = ElementSet::new(uncalibrated.capacity());
            let mut total = 0;
            for (i, t) in sg.threads.iter().enumerate() {
                assert_eq!(t.index, i);
                assert!(!t.elements.is_empty());
                assert!(t.elements.windows(2).all(|w| w[0] < w[1]));
                assert!(t.elements.iter().all(|&e| graph.kind(e) == t.kind));
                let set = ElementSet::from_ids(uncalibrated.capacity(), t.elements.iter().copied());
                assert!(covered.is_disjoint(&set));
                covered.union_with(&set);
                total += t.elements.len();
            }
            assert_eq!(covered, sg.set);
            assert_eq!(total, sg.elements.len());
            // Order: kind, then smallest element.
            assert!(sg
                .threads
                .windows(2)
                .all(|w| (w[0].kind, w[0].elements[0]) < (w[1].kind, w[1].elements[0])));
            // Equal to the naive components of the thread relation.
            let reference = reference_components(&sg.elements, |x, y| {
                graph.kind(x) == graph.kind(y)
                    && scope.activity.co_active(x, y)
                    && nb.is_within(x, y, d_t)
            });
            let got: BTreeSet<Vec<ElementId>> =
                sg.threads.iter().map(|t| t.elements.clone()).collect();
            assert_eq!(got, reference);
            // `segment_threads` agrees.
            assert_eq!(segment_threads(scope, &sg.set), sg.threads);
        }

        // Deterministic.
        assert_eq!(segment(scope, uncalibrated), subgoals);
    }

    // ----- segment: examples -----

    #[test]
    fn empty_uncalibrated_set_has_no_subgoals() {
        let w = World::xeb(3, 3);
        let scope = w.scope(1, 2, 2);
        let none = ElementSet::new(w.graph.len());
        assert!(segment(&scope, &none).is_empty());
        assert!(segment_subgoals(&scope, &none).is_empty());
        assert!(segment_threads(&scope, &none).is_empty());
    }

    #[test]
    fn empty_graph() {
        let graph = ProcessorGraph::square_lattice(0, 0);
        let w = World::all_active(graph);
        let scope = w.scope(1, 2, 2);
        assert!(segment(&scope, &ElementSet::new(0)).is_empty());
    }

    #[test]
    fn single_element_and_two_elements() {
        let w = World::all_active(ProcessorGraph::square_lattice(1, 2));
        let scope = w.scope(1, 1, 1);
        let one = w.set(&ids(&[2]));
        let sg = segment(&scope, &one);
        assert_eq!(sg.len(), 1);
        assert_eq!(sg[0].elements, ids(&[2]));
        assert_eq!(sg[0].threads.len(), 1);
        assert_eq!(
            sg[0].threads[0].kind,
            ElementKind::Edge(EdgeType::Engineered)
        );
        check_segmentation(&scope, &one);
        // The two nodes of the 1x2 lattice are two hops apart.
        let both = w.set(&ids(&[0, 1]));
        assert_eq!(
            segment(&scope, &both).len(),
            2,
            "d = 1 does not link two nodes"
        );
        let wide = w.scope(2, 1, 2);
        let sg = segment(&wide, &both);
        assert_eq!(sg.len(), 1);
        assert_eq!(sg[0].threads.len(), 1, "d_T = 2 links them");
        check_segmentation(&wide, &both);
    }

    #[test]
    fn zero_distances_make_every_element_alone() {
        let w = World::all_active(ProcessorGraph::square_lattice(3, 3));
        let scope = w.scope(0, 0, 0);
        let everything = w.graph.all_elements();
        let sg = segment(&scope, &everything);
        assert_eq!(sg.len(), w.graph.len());
        for (i, s) in sg.iter().enumerate() {
            assert_eq!(s.elements, vec![id(i as u32)]);
            assert_eq!(s.threads.len(), 1);
        }
        check_segmentation(&scope, &everything);
    }

    #[test]
    fn separation_is_the_larger_of_d_p_and_d_r() {
        let w = World::all_active(ProcessorGraph::square_lattice(1, 12));
        let nodes = w.graph.goal(|e| e.is_node());
        let by = |p, r| segment_subgoals(&w.scope(p, r, 1), &nodes).len();
        // Neighbouring nodes are linked by d = 2; nodes 2 hops apart by d = 3.
        assert_eq!(by(0, 0), 12);
        assert_eq!(by(1, 1), 12);
        assert_eq!(by(2, 1), 1);
        assert_eq!(by(1, 2), 1);
        // Only every other node: linked at d = 3, not at d = 2.
        let spaced = w.set(&(0..12).step_by(2).map(id).collect::<Vec<_>>());
        let count = |p, r| segment_subgoals(&w.scope(p, r, 1), &spaced).len();
        assert_eq!(count(2, 2), 6);
        assert_eq!(count(3, 0), 1);
        assert_eq!(count(0, 3), 1);
    }

    #[test]
    fn islands_in_a_path_are_separate_subgoals_ordered_by_smallest_id() {
        let w = World::all_active(ProcessorGraph::square_lattice(1, 20));
        let scope = w.scope(1, 2, 2);
        // Nodes 0..=3 and 15..=19 (calibrated gap in between).
        let nodes: Vec<u32> = (0..=3).chain(15..=19).collect();
        let uncalibrated = w.set(&ids(&nodes));
        let sg = segment(&scope, &uncalibrated);
        assert_eq!(sg.len(), 2);
        assert_eq!(sg[0].elements, ids(&[0, 1, 2, 3]));
        assert_eq!(sg[1].elements, ids(&[15, 16, 17, 18, 19]));
        assert_eq!((sg[0].index, sg[1].index), (0, 1));
        check_segmentation(&scope, &uncalibrated);
        // A subgoal that starts at a small ID but extends far still sorts by its smallest.
        let split = w.set(&ids(&[5, 6, 12, 13]));
        let sg = segment(&scope, &split);
        assert_eq!(sg.len(), 2);
        assert_eq!(sg[0].elements[0], id(5));
    }

    #[test]
    fn whole_lattice_is_one_subgoal_with_ordered_threads() {
        let w = World::xeb(6, 6);
        let scope = w.scope(1, 2, 2);
        let goal = w.goal();
        let sg = segment(&scope, &goal);
        assert_eq!(sg.len(), 1);
        assert_eq!(sg[0].set, goal);
        let kinds: Vec<ElementKind> = sg[0].threads.iter().map(|t| t.kind).collect();
        // One node thread, then the engineered edges, one thread per layer pattern.
        assert_eq!(kinds[0], ElementKind::Node);
        assert!(kinds[1..]
            .iter()
            .all(|&k| k == ElementKind::Edge(EdgeType::Engineered)));
        assert_eq!(sg[0].threads[0].elements.len(), 36);
        assert_eq!(
            sg[0].threads.len(),
            5,
            "{:?}",
            sg[0]
                .threads
                .iter()
                .map(|t| t.elements.len())
                .collect::<Vec<_>>()
        );
        check_segmentation(&scope, &goal);
    }

    #[test]
    fn threads_are_ordered_by_kind_then_smallest_id() {
        // Parasitic edges are in no XEB subgraph, so each is its own thread.
        let w = World::xeb(3, 3);
        let scope = w.scope(1, 2, 2);
        let everything = w.graph.all_elements();
        let sg = segment(&scope, &everything);
        assert_eq!(sg.len(), 1);
        let threads = &sg[0].threads;
        assert_eq!(threads[0].kind, ElementKind::Node);
        let engineered = threads
            .iter()
            .filter(|t| t.kind == ElementKind::Edge(EdgeType::Engineered))
            .count();
        let parasitic: Vec<&Thread> = threads
            .iter()
            .filter(|t| t.kind == ElementKind::Edge(EdgeType::Parasitic))
            .collect();
        assert!(engineered >= 1);
        assert_eq!(
            parasitic.len(),
            8,
            "one singleton thread per parasitic edge"
        );
        assert!(parasitic.iter().all(|t| t.elements.len() == 1));
        // Engineered threads come before parasitic ones.
        let first_parasitic = threads
            .iter()
            .position(|t| t.kind.edge_type() == Some(EdgeType::Parasitic))
            .unwrap();
        assert_eq!(first_parasitic, 1 + engineered);
        check_segmentation(&scope, &everything);
    }

    #[test]
    fn custom_edge_types_sort_after_the_built_in_ones() {
        let mut b = GraphBuilder::new();
        let n: Vec<_> = (0..4)
            .map(|i| b.add_node(format!("n{i}")).unwrap())
            .collect();
        b.add_edge(n[0], n[1], EdgeType::Custom(9)).unwrap();
        b.add_edge(n[1], n[2], EdgeType::Parasitic).unwrap();
        b.add_edge(n[2], n[3], EdgeType::Custom(2)).unwrap();
        b.add_edge(n[0], n[3], EdgeType::Engineered).unwrap();
        let w = World::all_active(b.build().unwrap());
        // d_T = 0 keeps every element in its own thread.
        let scope = w.scope(1, 1, 0);
        let all = w.graph.all_elements();
        let threads = segment_threads(&scope, &all);
        let kinds: Vec<ElementKind> = threads.iter().map(|t| t.kind).collect();
        let mut sorted = kinds.clone();
        sorted.sort();
        assert_eq!(kinds, sorted);
        assert_eq!(kinds[0], ElementKind::Node);
        assert_eq!(kinds[4], ElementKind::Edge(EdgeType::Engineered));
        assert_eq!(kinds[5], ElementKind::Edge(EdgeType::Parasitic));
        assert_eq!(kinds[6], ElementKind::Edge(EdgeType::Custom(2)));
        assert_eq!(kinds[7], ElementKind::Edge(EdgeType::Custom(9)));
        check_segmentation(&scope, &all);
    }

    #[test]
    fn calibrated_elements_split_a_lattice_into_subgoals() {
        // Calibrate a full middle column band of a lattice: the two sides separate.
        let w = World::xeb(5, 9);
        let scope = w.scope(1, 2, 2);
        let mut uncalibrated = w.goal();
        for r in 0..5 {
            for c in 3..=5 {
                let n = w.graph.lattice_node(r, c).unwrap();
                uncalibrated.remove(n);
                for &e in w.graph.incident(n) {
                    uncalibrated.remove(e);
                }
            }
        }
        let sg = segment(&scope, &uncalibrated);
        assert_eq!(sg.len(), 2);
        check_segmentation(&scope, &uncalibrated);
        // Narrow the band to one column: its two sides are two hops apart, which is still
        // too far for d_R = 2 but within reach of d_R = 3.
        let mut narrow = w.goal();
        for r in 0..5 {
            let n = w.graph.lattice_node(r, 4).unwrap();
            narrow.remove(n);
            for &e in w.graph.incident(n) {
                narrow.remove(e);
            }
        }
        assert_eq!(segment(&scope, &narrow).len(), 2);
        assert_eq!(segment(&w.scope(1, 3, 2), &narrow).len(), 1);
        check_segmentation(&w.scope(1, 3, 2), &narrow);
        // Removing only the middle node and its edges leaves the rest joined around it.
        let mut hole = w.goal();
        let n = w.graph.lattice_node(2, 4).unwrap();
        hole.remove(n);
        for &e in w.graph.incident(n) {
            hole.remove(e);
        }
        assert_eq!(segment(&scope, &hole).len(), 1);
    }

    // ----- segment: properties -----

    fn random_world(seed: u64) -> (World, ElementSet) {
        use rand::Rng;
        let graph = random_graph(seed, 9, 15);
        let mut r = rng(seed ^ 0xabcdef);
        let subgraphs: Vec<ElementSet> = (0..r.random_range(1..=3))
            .map(|_| {
                let mut s = ElementSet::new(graph.len());
                for e in graph.ids() {
                    if r.random_bool(0.7) {
                        s.insert(e);
                    }
                }
                s
            })
            .collect();
        let activity = AlgorithmSubgraphs::new(subgraphs).unwrap().masks();
        let mut uncalibrated = ElementSet::new(graph.len());
        for e in graph.ids() {
            if r.random_bool(0.6) {
                uncalibrated.insert(e);
            }
        }
        (World::new(graph, activity), uncalibrated)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn segmentation_properties_on_random_graphs(
            seed in any::<u64>(),
            d_p in 0u32..=3,
            d_r in 0u32..=3,
            d_t in 0u32..=3,
        ) {
            let (w, uncalibrated) = random_world(seed);
            check_segmentation(&w.scope(d_p, d_r, d_t), &uncalibrated);
        }

        #[test]
        fn segmentation_properties_on_lattices(
            rows in 1usize..=5,
            cols in 1usize..=5,
            mask_seed in any::<u64>(),
            d_p in 0u32..=2,
            d_r in 0u32..=3,
        ) {
            use rand::Rng;
            let w = World::xeb(rows, cols);
            let mut r = rng(mask_seed);
            let mut uncalibrated = w.goal();
            for e in w.graph.ids() {
                if r.random_bool(0.4) {
                    uncalibrated.remove(e);
                }
            }
            check_segmentation(&w.scope(d_p, d_r, 2), &uncalibrated);
        }
    }

    #[test]
    fn segmentation_scales_linearly_enough_for_ten_thousand_qubits() {
        // Not a timing assertion; checks a big instance works and obeys the invariants.
        let w = World::xeb(60, 60);
        let scope = w.scope(1, 2, 2);
        let goal = w.goal();
        let sg = segment(&scope, &goal);
        assert_eq!(sg.len(), 1);
        assert_eq!(sg[0].elements.len(), goal.len());
        assert_eq!(
            sg[0]
                .threads
                .iter()
                .map(|t| t.elements.len())
                .sum::<usize>(),
            goal.len()
        );
    }

    // ----- scratch space is proportional to the set, not the processor -----

    #[test]
    fn components_scratch_is_indexed_by_member_position_not_by_id() {
        // Regression: scratch used to be sized to the whole processor. Here the scratch has
        // exactly one slot per member while the member IDs are large; indexing by ID panics.
        let w = World::xeb(30, 30);
        let scope = w.scope(1, 2, 2);
        let members = vec![
            w.graph.lattice_node(25, 25).unwrap(),
            w.graph.lattice_node(25, 26).unwrap(),
            w.graph.lattice_node(2, 2).unwrap(),
        ];
        let mut members = members;
        members.sort_unstable();
        assert!(members.iter().all(|m| m.index() > 2));
        let set = w.set(&members);
        let mut label = vec![UNASSIGNED; members.len()];
        let comps = components(
            &scope,
            &members,
            scope.distances.separation(),
            |v| set.contains(v),
            |_, _| true,
            &mut label,
        );
        assert_eq!(comps.len(), 2);
        assert_eq!(comps.iter().map(Vec::len).sum::<usize>(), 3);
        assert!(
            label.iter().all(|&l| l == UNASSIGNED),
            "scratch is restored"
        );
    }

    #[test]
    fn single_element_segmentation_on_a_big_lattice() {
        let w = World::xeb(60, 60);
        let scope = w.scope(1, 2, 2);
        let one = w.set(&[w.graph.lattice_node(30, 30).unwrap()]);
        let sg = segment(&scope, &one);
        assert_eq!(sg.len(), 1);
        assert_eq!(sg[0].elements.len(), 1);
        assert_eq!(sg[0].threads.len(), 1);
    }

    #[test]
    #[should_panic(expected = "capacity")]
    fn segment_rejects_a_set_with_the_wrong_capacity() {
        let w = World::xeb(3, 3);
        segment(&w.scope(1, 2, 2), &ElementSet::new(3));
    }

    #[test]
    #[should_panic(expected = "capacity")]
    fn segment_threads_rejects_a_set_with_the_wrong_capacity() {
        let w = World::xeb(3, 3);
        segment_threads(&w.scope(1, 2, 2), &ElementSet::new(1000));
    }

    // ----- seed policies -----

    fn thread(kind: ElementKind, members: &[u32]) -> Thread {
        Thread {
            index: 0,
            kind,
            elements: ids(members),
        }
    }

    #[test]
    fn lowest_id_seed() {
        let t = thread(ElementKind::Node, &[4, 7, 9]);
        assert_eq!(SeedPolicy::LowestId.seed(&t, &mut rng(0)), id(4));
        assert_eq!(SeedPolicy::default().seed(&t, &mut rng(0)), id(4));
        // Does not consume randomness.
        let mut a = rng(5);
        SeedPolicy::LowestId.seed(&t, &mut a);
        assert_eq!(
            rand::Rng::random::<u64>(&mut a),
            rand::Rng::random::<u64>(&mut rng(5))
        );
    }

    #[test]
    fn fixed_seed_picks_first_listed_member() {
        let t = thread(ElementKind::Node, &[4, 7, 9]);
        let pick = |list: &[u32]| SeedPolicy::Fixed(ids(list)).seed(&t, &mut rng(0));
        assert_eq!(pick(&[9, 4]), id(9), "first *listed*, not lowest");
        assert_eq!(pick(&[1, 7, 9]), id(7), "skips non-members");
        assert_eq!(pick(&[1, 2]), id(4), "no member: lowest id");
        assert_eq!(pick(&[]), id(4));
    }

    #[test]
    fn fixed_seed_does_not_require_sorted_threads() {
        // Regression: `Fixed` used a binary search and missed members of unsorted threads.
        let t = thread(ElementKind::Node, &[9, 2, 7, 4, 8]);
        for want in [9u32, 2, 7, 4, 8] {
            let got = SeedPolicy::Fixed(ids(&[want])).seed(&t, &mut rng(0));
            assert_eq!(got, id(want));
        }
        assert_eq!(SeedPolicy::Fixed(ids(&[1])).seed(&t, &mut rng(0)), id(2));
    }

    #[test]
    fn random_seed_is_a_deterministic_uniform_member() {
        let t = thread(ElementKind::Node, &[4, 7, 9, 12]);
        let draw = |seed| SeedPolicy::Random.seed(&t, &mut rng(seed));
        for seed in 0..10 {
            assert!(t.elements.contains(&draw(seed)));
            assert_eq!(draw(seed), draw(seed));
        }
        let mut r = rng(1);
        let mut counts = std::collections::BTreeMap::new();
        for _ in 0..4000 {
            *counts
                .entry(SeedPolicy::Random.seed(&t, &mut r))
                .or_insert(0usize) += 1;
        }
        assert_eq!(counts.len(), 4);
        assert!(
            counts.values().all(|&c| (800..1200).contains(&c)),
            "{counts:?}"
        );
        // A single-member thread.
        let one = thread(ElementKind::Node, &[3]);
        assert_eq!(SeedPolicy::Random.seed(&one, &mut rng(0)), id(3));
    }

    #[test]
    fn custom_seed_calls_the_function() {
        let t = thread(ElementKind::Node, &[4, 7, 9]);
        let last = SeedPolicy::Custom(Arc::new(|t, _| *t.elements.last().unwrap()));
        assert_eq!(last.seed(&t, &mut rng(0)), id(9));
        // It may return an element outside the thread: the policy passes it through.
        let outside = SeedPolicy::Custom(Arc::new(|_, _| ElementId::new(100)));
        assert_eq!(outside.seed(&t, &mut rng(0)), id(100));
        let with_rng = SeedPolicy::Custom(Arc::new(|t, rng| {
            t.elements[rand::Rng::random_range(rng, 0..t.elements.len())]
        }));
        assert_eq!(
            with_rng.seed(&t, &mut rng(3)),
            with_rng.seed(&t, &mut rng(3))
        );
        assert_eq!(format!("{last:?}"), "Custom(..)");
        assert_eq!(
            format!("{:?}", SeedPolicy::Fixed(ids(&[1]))),
            "Fixed([ElementId(1)])"
        );
        assert_eq!(format!("{:?}", SeedPolicy::Random), "Random");
        assert_eq!(format!("{:?}", SeedPolicy::LowestId), "LowestId");
        let cloned = last.clone();
        assert_eq!(cloned.seed(&t, &mut rng(0)), id(9));
    }

    #[test]
    #[should_panic(expected = "at least one element")]
    fn seed_of_empty_thread_panics() {
        let t = thread(ElementKind::Node, &[]);
        SeedPolicy::LowestId.seed(&t, &mut rng(0));
    }

    #[test]
    #[should_panic(expected = "at least one element")]
    fn random_seed_of_empty_thread_panics() {
        let t = thread(ElementKind::Node, &[]);
        SeedPolicy::Random.seed(&t, &mut rng(0));
    }

    // ----- StitchPlan -----

    fn invalid<T: std::fmt::Debug>(r: Result<T, FaultTuneError>) -> bool {
        matches!(r, Err(FaultTuneError::InvalidConfig(_)))
    }

    /// Splits a lattice's goal into quadrants by the layout coordinates of each element.
    fn quadrants(w: &World, rows: usize, cols: usize) -> Vec<ElementSet> {
        let coords = w.graph.coords().expect("lattices have a layout");
        let mut regions = vec![ElementSet::new(w.graph.len()); 4];
        let goal = w.goal();
        for e in &goal {
            let [x, y] = coords[e.index()];
            let right = usize::from(x * 2.0 >= (cols - 1) as f32);
            let bottom = usize::from(y * 2.0 >= (rows - 1) as f32);
            regions[2 * bottom + right].insert(e);
        }
        regions
    }

    /// Checks every promise of a stitch plan.
    fn check_plan(scope: &Scope<'_>, goal: &ElementSet, plan: &StitchPlan) {
        let nb = scope.neighbourhoods;
        let sep = scope.distances.separation();
        let mut covered = ElementSet::new(goal.capacity());
        let mut region_of = vec![None; goal.capacity()];
        for (i, r) in plan.regions.iter().enumerate() {
            for e in r {
                region_of[e.index()] = Some(i);
            }
            covered.union_with(r);
        }
        // stage_one and gap partition the goal.
        assert!(plan.stage_one.is_disjoint(&plan.gap));
        assert_eq!(plan.stage_one.union(&plan.gap), *goal);
        // The stage-one cores are inside the regions.
        assert!(plan.stage_one.is_subset(&covered));
        // Goal elements outside every region are in the gap.
        assert!(goal.difference(&covered).is_subset(&plan.gap));
        // Cores of different regions are more than `sep` apart.
        for a in &plan.stage_one {
            for b in &plan.stage_one {
                if region_of[a.index()] != region_of[b.index()] {
                    assert!(!nb.is_within(a, b, sep), "{a} and {b}");
                }
            }
        }
        // The gap is minimal: a region element is in it only because it is close to another
        // region's element.
        for g in plan.gap.intersection(&covered).iter() {
            let near_other = covered
                .iter()
                .any(|h| region_of[h.index()] != region_of[g.index()] && nb.is_within(g, h, sep));
            assert!(near_other, "{g} was gapped for no reason");
        }
        for g in &plan.stage_one {
            let near_other = covered
                .iter()
                .any(|h| region_of[h.index()] != region_of[g.index()] && nb.is_within(g, h, sep));
            assert!(!near_other);
        }
        // Different regions' cores segment into separate subgoals.
        let subgoals = segment_subgoals(scope, &plan.stage_one);
        for sg in &subgoals {
            let owners: BTreeSet<_> = sg.elements.iter().map(|e| region_of[e.index()]).collect();
            assert_eq!(owners.len(), 1, "a stage-one subgoal spans several regions");
        }
    }

    #[test]
    fn quadrants_give_four_stage_one_subgoals() {
        let (rows, cols) = (12, 12);
        let w = World::xeb(rows, cols);
        let scope = w.scope(1, 2, 2);
        let goal = w.goal();
        let regions = quadrants(&w, rows, cols);
        assert!(regions.iter().all(|r| !r.is_empty()));
        let plan = StitchPlan::new(&scope, &goal, regions.clone()).unwrap();
        assert_eq!(plan.regions, regions);
        check_plan(&scope, &goal, &plan);
        let subgoals = segment_subgoals(&scope, &plan.stage_one);
        assert!(subgoals.len() >= 4, "{} subgoals", subgoals.len());
        assert_eq!(subgoals.len(), 4);
        // The gap is a cross through the middle, not the whole lattice.
        assert!(!plan.gap.is_empty());
        assert!(plan.gap.len() < goal.len() / 2);
        // Without a plan the lattice is a single subgoal.
        assert_eq!(segment_subgoals(&scope, &goal).len(), 1);
        // Stage two joins the regions: segmenting the gap alone is a single subgoal here.
        assert_eq!(segment_subgoals(&scope, &plan.gap).len(), 1);
    }

    #[test]
    fn wider_separation_widens_the_gap() {
        let (rows, cols) = (12, 12);
        let w = World::xeb(rows, cols);
        let goal = w.goal();
        let regions = quadrants(&w, rows, cols);
        let narrow = StitchPlan::new(&w.scope(1, 1, 1), &goal, regions.clone()).unwrap();
        let wide = StitchPlan::new(&w.scope(1, 3, 1), &goal, regions).unwrap();
        assert!(wide.gap.len() > narrow.gap.len());
        assert!(narrow.gap.is_subset(&wide.gap));
        check_plan(&w.scope(1, 1, 1), &goal, &narrow);
        check_plan(&w.scope(1, 3, 1), &goal, &wide);
    }

    #[test]
    fn single_region_and_no_regions() {
        let w = World::xeb(6, 6);
        let scope = w.scope(1, 2, 2);
        let goal = w.goal();
        // No regions: everything is the gap.
        let plan = StitchPlan::new(&scope, &goal, vec![]).unwrap();
        assert!(plan.stage_one.is_empty());
        assert_eq!(plan.gap, goal);
        // One region covering the goal: nothing is gapped.
        let plan = StitchPlan::new(&scope, &goal, vec![goal.clone()]).unwrap();
        assert_eq!(plan.stage_one, goal);
        assert!(plan.gap.is_empty());
        // One region covering part of it: the rest is the gap.
        let half: Vec<ElementId> = goal.iter().filter(|e| e.index() % 2 == 0).collect();
        let half = w.set(&half);
        let plan = StitchPlan::new(&scope, &goal, vec![half.clone()]).unwrap();
        assert_eq!(plan.stage_one, half);
        assert_eq!(plan.gap, goal.difference(&half));
        // Empty regions are fine.
        let plan = StitchPlan::new(&scope, &goal, vec![ElementSet::new(w.graph.len()); 2]).unwrap();
        assert_eq!(plan.gap, goal);
    }

    #[test]
    fn stitch_plan_rejects_bad_regions() {
        let w = World::xeb(6, 6);
        let scope = w.scope(1, 2, 2);
        let goal = w.goal();
        let a = w.set(&ids(&[0, 1, 2]));
        let overlapping = w.set(&ids(&[2, 3]));
        assert!(invalid(StitchPlan::new(
            &scope,
            &goal,
            vec![a.clone(), overlapping]
        )));
        // A region that is also listed twice.
        assert!(invalid(StitchPlan::new(
            &scope,
            &goal,
            vec![a.clone(), a.clone()]
        )));
        // An element outside the goal: a parasitic edge.
        let parasitic = w
            .graph
            .edges()
            .find(|&e| w.graph.kind(e).edge_type() == Some(EdgeType::Parasitic))
            .unwrap();
        let outside = w.set(&[parasitic]);
        assert!(invalid(StitchPlan::new(&scope, &goal, vec![outside])));
        // A goal that excludes part of a region.
        let small_goal = w.set(&ids(&[0, 1]));
        assert!(invalid(StitchPlan::new(
            &scope,
            &small_goal,
            vec![a.clone()]
        )));
        // Wrong capacity.
        assert!(invalid(StitchPlan::new(
            &scope,
            &goal,
            vec![ElementSet::new(3)]
        )));
        assert!(invalid(StitchPlan::new(
            &scope,
            &ElementSet::new(3),
            vec![]
        )));
        // Valid disjoint regions are accepted.
        assert!(StitchPlan::new(&scope, &goal, vec![a, w.set(&ids(&[5, 6]))]).is_ok());
    }

    #[test]
    fn stitch_plan_is_deterministic() {
        let w = World::xeb(8, 8);
        let scope = w.scope(1, 2, 2);
        let goal = w.goal();
        let regions = quadrants(&w, 8, 8);
        assert_eq!(
            StitchPlan::new(&scope, &goal, regions.clone()).unwrap(),
            StitchPlan::new(&scope, &goal, regions).unwrap()
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(48))]

        #[test]
        fn stitch_plan_properties(
            seed in any::<u64>(),
            region_count in 0usize..=4,
            d_p in 0u32..=2,
            d_r in 0u32..=3,
        ) {
            use rand::Rng;
            let (w, _) = random_world(seed);
            let mut r = rng(seed.wrapping_add(1));
            let mut goal = ElementSet::new(w.graph.len());
            for e in w.graph.ids() {
                if r.random_bool(0.8) {
                    goal.insert(e);
                }
            }
            // Random disjoint regions of the goal; some goal elements stay in none.
            let mut regions = vec![ElementSet::new(w.graph.len()); region_count];
            if region_count > 0 {
                for e in &goal {
                    let pick = r.random_range(0..=region_count);
                    if pick < region_count {
                        regions[pick].insert(e);
                    }
                }
            }
            let scope = w.scope(d_p, d_r, 2);
            let plan = StitchPlan::new(&scope, &goal, regions).unwrap();
            check_plan(&scope, &goal, &plan);
        }
    }
}
