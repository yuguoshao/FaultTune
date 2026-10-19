//! Traversal frontiers and heuristics (ARCHITECTURE §4.3, §5.3).
//!
//! The paper describes traversal as a recursive function. FaultTune uses an explicit
//! frontier (a work list) instead: a last-in-first-out frontier visits elements in the same
//! order as the paper's recursion; first-in-first-out gives true breadth-first traversal; a
//! priority frontier supports scored heuristics. Nothing here can overflow the stack.

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, VecDeque};

use rand::rngs::StdRng;
use rand::Rng;

use crate::database::StepRecord;
use crate::graph::{ElementId, ProcessorGraph};
use crate::neighbourhoods::Neighbourhoods;
use crate::sets::ElementSet;

/// How the frontier hands out elements.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum FrontierKind {
    /// Last in, first out: reproduces the paper's recursion order (depth-first).
    Lifo,
    /// First in, first out: breadth-first.
    Fifo,
    /// Lowest [`TraversalHeuristic::priority`] first; ties by insertion order.
    Priority,
}

/// Read access for heuristics: the graph, the calibration status and the step history.
pub struct TraversalContext<'a> {
    /// The processor graph.
    pub graph: &'a ProcessorGraph,
    /// Neighbourhood index (for distances).
    pub neighbourhoods: &'a Neighbourhoods,
    /// The current calibration status `P*` (database snapshot plus this subgoal's writes).
    pub calibrated: &'a ElementSet,
    /// The element whose step was just committed.
    pub current: ElementId,
    /// Steps committed so far by the current subgoal in this pass, in commit order.
    pub history: &'a [StepRecord],
}

/// Orders the traversal options after each step.
pub trait TraversalHeuristic: Send + Sync {
    /// The kind of frontier this heuristic wants. Default: [`FrontierKind::Fifo`].
    fn frontier(&self) -> FrontierKind {
        FrontierKind::Fifo
    }

    /// Orders `options` in place (it may also drop elements). `options[0]` is visited
    /// first among them, for every frontier kind except `Priority`.
    fn order(&self, ctx: &TraversalContext<'_>, options: &mut Vec<ElementId>, rng: &mut StdRng);

    /// The score used by a [`FrontierKind::Priority`] frontier (lower pops first).
    fn priority(&self, ctx: &TraversalContext<'_>, element: ElementId) -> f64 {
        let _ = (ctx, element);
        0.0
    }
}

/// Visit the nearest options first (ascending incidence distance from the current element,
/// then ID), using a FIFO frontier: true breadth-first traversal.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct BreadthFirst;

/// Follow the first option as far as possible before backing up (LIFO frontier), reproducing
/// the paper's recursion. Options are ordered by ascending incidence distance, then ID.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct DepthFirst;

/// Visit options in a random order drawn from the subgoal's seeded RNG (FIFO frontier).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Random;

impl TraversalHeuristic for BreadthFirst {
    fn frontier(&self) -> FrontierKind {
        FrontierKind::Fifo
    }
    fn order(&self, ctx: &TraversalContext<'_>, options: &mut Vec<ElementId>, _rng: &mut StdRng) {
        sort_by_proximity(ctx, options);
    }
}

impl TraversalHeuristic for DepthFirst {
    fn frontier(&self) -> FrontierKind {
        FrontierKind::Lifo
    }
    fn order(&self, ctx: &TraversalContext<'_>, options: &mut Vec<ElementId>, _rng: &mut StdRng) {
        sort_by_proximity(ctx, options);
    }
}

impl TraversalHeuristic for Random {
    fn frontier(&self) -> FrontierKind {
        FrontierKind::Fifo
    }
    fn order(&self, _ctx: &TraversalContext<'_>, options: &mut Vec<ElementId>, rng: &mut StdRng) {
        // Fisher-Yates: uniform over all permutations, one RNG draw per swap.
        for i in (1..options.len()).rev() {
            let j = rng.random_range(0..=i);
            options.swap(i, j);
        }
    }
}

/// Sorts `options` by ascending incidence distance from `ctx.current`, then by ID. An option
/// outside the indexed ball of the current element sorts last.
fn sort_by_proximity(ctx: &TraversalContext<'_>, options: &mut [ElementId]) {
    options.sort_by_cached_key(|&e| {
        (
            ctx.neighbourhoods
                .incidence_distance(ctx.current, e)
                .unwrap_or(u32::MAX),
            e,
        )
    });
}

/// A scored entry of a priority frontier: lowest score first, ties by insertion order.
#[derive(Copy, Clone, Debug)]
struct Scored {
    priority: f64,
    sequence: u64,
    element: ElementId,
}

impl PartialEq for Scored {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Scored {}

impl PartialOrd for Scored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Scored {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .total_cmp(&other.priority)
            .then(self.sequence.cmp(&other.sequence))
    }
}

/// The work list of a traversal.
///
/// `Lifo` and `Fifo` frontiers hand elements out in stack or queue order. A `Priority`
/// frontier hands out the lowest score first (compared with [`f64::total_cmp`]), and equal
/// scores in insertion order. The same element may be pushed several times; the traversal
/// skips elements that are already calibrated when they pop.
pub struct Frontier {
    kind: FrontierKind,
    /// Storage of `Lifo` (a stack: push and pop at the back) and `Fifo` (a queue).
    queue: VecDeque<ElementId>,
    /// Storage of `Priority`.
    heap: BinaryHeap<Reverse<Scored>>,
    /// Insertion counter of the priority frontier, for stable ties.
    next_sequence: u64,
}

impl Frontier {
    /// An empty frontier of the given kind.
    pub fn new(kind: FrontierKind) -> Self {
        Self {
            kind,
            queue: VecDeque::new(),
            heap: BinaryHeap::new(),
            next_sequence: 0,
        }
    }

    fn push_scored(&mut self, element: ElementId, priority: f64) {
        let sequence = self.next_sequence;
        self.next_sequence += 1;
        self.heap.push(Reverse(Scored {
            priority,
            sequence,
            element,
        }));
    }

    /// Pushes a single element (a seed), with priority 0 for a priority frontier.
    pub fn push(&mut self, element: ElementId) {
        match self.kind {
            FrontierKind::Lifo | FrontierKind::Fifo => self.queue.push_back(element),
            FrontierKind::Priority => self.push_scored(element, 0.0),
        }
    }

    /// Orders `options` with the heuristic and pushes them, so that `options[0]` is popped
    /// first among them for `Lifo` (they are pushed in reverse) and `Fifo`; a `Priority`
    /// frontier scores each with [`TraversalHeuristic::priority`].
    ///
    /// `options` is left in the heuristic's order (it is not cleared), so the caller can
    /// reuse the buffer.
    pub fn extend_ordered(
        &mut self,
        heuristic: &dyn TraversalHeuristic,
        ctx: &TraversalContext<'_>,
        options: &mut Vec<ElementId>,
        rng: &mut StdRng,
    ) {
        heuristic.order(ctx, options, rng);
        match self.kind {
            FrontierKind::Lifo => self.queue.extend(options.iter().rev().copied()),
            FrontierKind::Fifo => self.queue.extend(options.iter().copied()),
            FrontierKind::Priority => {
                for &element in options.iter() {
                    let priority = heuristic.priority(ctx, element);
                    self.push_scored(element, priority);
                }
            }
        }
    }

    /// Removes and returns the next element.
    pub fn pop(&mut self) -> Option<ElementId> {
        match self.kind {
            FrontierKind::Lifo => self.queue.pop_back(),
            FrontierKind::Fifo => self.queue.pop_front(),
            FrontierKind::Priority => self.heap.pop().map(|Reverse(s)| s.element),
        }
    }

    /// Number of elements waiting in the frontier.
    pub fn len(&self) -> usize {
        self.queue.len() + self.heap.len()
    }

    /// Whether the frontier is empty.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty() && self.heap.is_empty()
    }

    /// Removes every element.
    pub fn clear(&mut self) {
        self.queue.clear();
        self.heap.clear();
        self.next_sequence = 0;
    }

    /// The frontier kind.
    pub fn kind(&self) -> FrontierKind {
        self.kind
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use proptest::prelude::*;
    use rand::SeedableRng;

    use super::*;
    use crate::activity::{ActivityMasks, AlgorithmSubgraphs};
    use crate::graph::{EdgeType, ElementKind};
    use crate::neighbourhoods::test_support::random_graph;
    use crate::scope::{Distances, Scope};

    fn id(i: u32) -> ElementId {
        ElementId::new(i)
    }

    fn ids(v: &[u32]) -> Vec<ElementId> {
        v.iter().map(|&i| id(i)).collect()
    }

    /// Graph, index and activity for traversal tests.
    struct World {
        graph: ProcessorGraph,
        neighbourhoods: Neighbourhoods,
        activity: ActivityMasks,
    }

    impl World {
        fn lattice(rows: usize, cols: usize) -> Self {
            let graph = ProcessorGraph::square_lattice(rows, cols);
            let neighbourhoods = Neighbourhoods::build(&graph, 3);
            let activity = AlgorithmSubgraphs::all_active(&graph).masks();
            Self {
                graph,
                neighbourhoods,
                activity,
            }
        }

        fn scope(&self, traversal: u32) -> Scope<'_> {
            Scope::new(
                &self.graph,
                &self.neighbourhoods,
                &self.activity,
                Distances {
                    parameter: 0,
                    constraint: 1,
                    traversal,
                },
            )
        }

        fn ctx<'a>(
            &'a self,
            calibrated: &'a ElementSet,
            current: ElementId,
        ) -> TraversalContext<'a> {
            TraversalContext {
                graph: &self.graph,
                neighbourhoods: &self.neighbourhoods,
                calibrated,
                current,
                history: &[],
            }
        }
    }

    fn rng(seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }

    /// A heuristic whose priority is a fixed function of the element ID.
    struct ByScore(fn(ElementId) -> f64);

    impl TraversalHeuristic for ByScore {
        fn frontier(&self) -> FrontierKind {
            FrontierKind::Priority
        }
        fn order(&self, _: &TraversalContext<'_>, _: &mut Vec<ElementId>, _: &mut StdRng) {}
        fn priority(&self, _: &TraversalContext<'_>, element: ElementId) -> f64 {
            (self.0)(element)
        }
    }

    /// Orders options with the RNG but visits them depth-first: exercises RNG-order equality.
    struct ShuffledLifo;

    impl TraversalHeuristic for ShuffledLifo {
        fn frontier(&self) -> FrontierKind {
            FrontierKind::Lifo
        }
        fn order(
            &self,
            ctx: &TraversalContext<'_>,
            options: &mut Vec<ElementId>,
            rng: &mut StdRng,
        ) {
            Random.order(ctx, options, rng);
        }
    }

    /// Leaves the options as they are, with the given frontier kind.
    struct Plain(FrontierKind);

    impl TraversalHeuristic for Plain {
        fn frontier(&self) -> FrontierKind {
            self.0
        }
        fn order(&self, _: &TraversalContext<'_>, _: &mut Vec<ElementId>, _: &mut StdRng) {}
    }

    fn drain(frontier: &mut Frontier) -> Vec<ElementId> {
        std::iter::from_fn(|| frontier.pop()).collect()
    }

    // ----- Frontier -----

    #[test]
    fn push_and_pop_orders() {
        let mut lifo = Frontier::new(FrontierKind::Lifo);
        let mut fifo = Frontier::new(FrontierKind::Fifo);
        let mut prio = Frontier::new(FrontierKind::Priority);
        for f in [&mut lifo, &mut fifo, &mut prio] {
            assert!(f.is_empty());
            assert_eq!(f.len(), 0);
            assert_eq!(f.pop(), None);
            for i in [3, 1, 2] {
                f.push(id(i));
            }
            assert!(!f.is_empty());
            assert_eq!(f.len(), 3);
        }
        assert_eq!(drain(&mut lifo), ids(&[2, 1, 3]));
        assert_eq!(drain(&mut fifo), ids(&[3, 1, 2]));
        // Equal priorities: insertion order.
        assert_eq!(drain(&mut prio), ids(&[3, 1, 2]));
        assert_eq!(lifo.kind(), FrontierKind::Lifo);
        assert_eq!(fifo.kind(), FrontierKind::Fifo);
        assert_eq!(prio.kind(), FrontierKind::Priority);
    }

    #[test]
    fn extend_ordered_pops_first_option_first() {
        let w = World::lattice(3, 3);
        let calibrated = ElementSet::new(w.graph.len());
        let ctx = w.ctx(&calibrated, id(0));
        for kind in [FrontierKind::Lifo, FrontierKind::Fifo] {
            let mut f = Frontier::new(kind);
            let mut options = ids(&[5, 2, 9, 7]);
            f.extend_ordered(&Plain(kind), &ctx, &mut options, &mut rng(0));
            assert_eq!(
                options,
                ids(&[5, 2, 9, 7]),
                "options keep the heuristic's order"
            );
            assert_eq!(drain(&mut f), ids(&[5, 2, 9, 7]), "{kind:?}");
        }
    }

    #[test]
    fn lifo_stacks_batches_and_fifo_queues_them() {
        let w = World::lattice(3, 3);
        let calibrated = ElementSet::new(w.graph.len());
        let ctx = w.ctx(&calibrated, id(0));
        let mut lifo = Frontier::new(FrontierKind::Lifo);
        let mut fifo = Frontier::new(FrontierKind::Fifo);
        for f in [&mut lifo, &mut fifo] {
            let kind = f.kind();
            f.push(id(0));
            f.extend_ordered(&Plain(kind), &ctx, &mut ids(&[1, 2]), &mut rng(0));
        }
        // Lifo: the last batch pops first, and within it options[0] first.
        let mut order = Vec::new();
        order.push(lifo.pop().unwrap());
        lifo.extend_ordered(
            &Plain(FrontierKind::Lifo),
            &ctx,
            &mut ids(&[7, 8]),
            &mut rng(0),
        );
        order.extend(drain(&mut lifo));
        assert_eq!(order, ids(&[1, 7, 8, 2, 0]));
        // Fifo: strictly in arrival order.
        let mut order = Vec::new();
        order.push(fifo.pop().unwrap());
        fifo.extend_ordered(
            &Plain(FrontierKind::Fifo),
            &ctx,
            &mut ids(&[7, 8]),
            &mut rng(0),
        );
        order.extend(drain(&mut fifo));
        assert_eq!(order, ids(&[0, 1, 2, 7, 8]));
    }

    #[test]
    fn priority_pops_lowest_score_first_with_stable_ties() {
        let w = World::lattice(3, 3);
        let calibrated = ElementSet::new(w.graph.len());
        let ctx = w.ctx(&calibrated, id(0));
        // Score = id % 3: ids 3, 0, 6 tie at 0; 4, 1 at 1; 8, 2, 5 at 2.
        let h = ByScore(|e| f64::from(e.raw() % 3));
        let mut f = Frontier::new(FrontierKind::Priority);
        let mut options = ids(&[8, 4, 3, 2, 1, 0, 6, 5]);
        f.extend_ordered(&h, &ctx, &mut options, &mut rng(0));
        assert_eq!(drain(&mut f), ids(&[3, 0, 6, 4, 1, 8, 2, 5]));
        // A later batch with a better score jumps the queue; equal scores queue behind.
        f.extend_ordered(&h, &ctx, &mut ids(&[4, 5]), &mut rng(0));
        f.extend_ordered(&h, &ctx, &mut ids(&[7, 3]), &mut rng(0));
        f.push(id(1)); // priority 0.0, after the earlier 0.0 entries
        assert_eq!(drain(&mut f), ids(&[3, 1, 4, 7, 5]));
    }

    #[test]
    fn priority_uses_total_order_for_special_floats() {
        let w = World::lattice(3, 3);
        let calibrated = ElementSet::new(w.graph.len());
        let ctx = w.ctx(&calibrated, id(0));
        let h = ByScore(|e| match e.raw() {
            0 => f64::NAN,
            1 => f64::INFINITY,
            2 => f64::NEG_INFINITY,
            3 => 0.0,
            4 => -0.0,
            5 => -1.5,
            _ => 7.0,
        });
        let mut f = Frontier::new(FrontierKind::Priority);
        f.extend_ordered(&h, &ctx, &mut ids(&[0, 1, 2, 3, 4, 5, 6]), &mut rng(0));
        // -inf, -1.5, -0.0, +0.0, 7, +inf, NaN (total_cmp order).
        assert_eq!(drain(&mut f), ids(&[2, 5, 4, 3, 6, 1, 0]));
    }

    #[test]
    fn priority_can_depend_on_the_context() {
        // Prefer elements closest to the current one; the heuristic reads the context.
        struct Nearest;
        impl TraversalHeuristic for Nearest {
            fn frontier(&self) -> FrontierKind {
                FrontierKind::Priority
            }
            fn order(&self, _: &TraversalContext<'_>, _: &mut Vec<ElementId>, _: &mut StdRng) {}
            fn priority(&self, ctx: &TraversalContext<'_>, e: ElementId) -> f64 {
                f64::from(
                    ctx.neighbourhoods
                        .incidence_distance(ctx.current, e)
                        .unwrap(),
                )
            }
        }
        let w = World::lattice(5, 5);
        let calibrated = ElementSet::new(w.graph.len());
        let centre = w.graph.lattice_node(2, 2).unwrap();
        let ctx = w.ctx(&calibrated, centre);
        let far = w.graph.lattice_node(0, 0).unwrap();
        let near = w.graph.lattice_node(2, 3).unwrap();
        let mid = w.graph.lattice_node(1, 1).unwrap();
        let mut f = Frontier::new(FrontierKind::Priority);
        f.extend_ordered(&Nearest, &ctx, &mut vec![far, near, mid], &mut rng(0));
        assert_eq!(drain(&mut f), vec![near, mid, far]);
    }

    #[test]
    fn empty_options_push_nothing() {
        let w = World::lattice(2, 2);
        let calibrated = ElementSet::new(w.graph.len());
        let ctx = w.ctx(&calibrated, id(0));
        for kind in [
            FrontierKind::Lifo,
            FrontierKind::Fifo,
            FrontierKind::Priority,
        ] {
            let mut f = Frontier::new(kind);
            f.extend_ordered(&Plain(kind), &ctx, &mut Vec::new(), &mut rng(0));
            f.extend_ordered(&ByScore(|_| 0.0), &ctx, &mut Vec::new(), &mut rng(0));
            assert!(f.is_empty());
            assert_eq!(f.pop(), None);
        }
        for h in [
            &BreadthFirst as &dyn TraversalHeuristic,
            &DepthFirst,
            &Random,
        ] {
            let mut f = Frontier::new(h.frontier());
            let mut options = Vec::new();
            f.extend_ordered(h, &ctx, &mut options, &mut rng(0));
            assert!(f.is_empty());
            assert!(options.is_empty());
        }
    }

    #[test]
    fn frontier_is_reusable_after_clear() {
        let w = World::lattice(3, 3);
        let calibrated = ElementSet::new(w.graph.len());
        let ctx = w.ctx(&calibrated, id(0));
        for kind in [
            FrontierKind::Lifo,
            FrontierKind::Fifo,
            FrontierKind::Priority,
        ] {
            let mut f = Frontier::new(kind);
            let h = ByScore(|e| f64::from(e.raw() % 2));
            let run = |f: &mut Frontier| {
                f.push(id(8));
                f.extend_ordered(&Plain(kind), &ctx, &mut ids(&[4, 3, 2]), &mut rng(0));
                f.extend_ordered(&h, &ctx, &mut ids(&[7, 6]), &mut rng(0));
                drain(f)
            };
            let first = run(&mut f);
            assert_eq!(first.len(), 6);
            // Partially fill, clear, and fill again: identical behaviour to a fresh frontier.
            f.push(id(1));
            f.push(id(2));
            f.clear();
            assert!(f.is_empty());
            assert_eq!(f.pop(), None);
            assert_eq!(run(&mut f), first, "{kind:?}");
            assert_eq!(f.kind(), kind);
        }
    }

    // ----- heuristics -----

    #[test]
    fn heuristic_frontier_kinds() {
        assert_eq!(BreadthFirst.frontier(), FrontierKind::Fifo);
        assert_eq!(DepthFirst.frontier(), FrontierKind::Lifo);
        assert_eq!(Random.frontier(), FrontierKind::Fifo);
        // The trait defaults.
        assert_eq!(ByScoreDefault.frontier(), FrontierKind::Fifo);
        let w = World::lattice(2, 2);
        let calibrated = ElementSet::new(w.graph.len());
        assert_eq!(
            ByScoreDefault.priority(&w.ctx(&calibrated, id(0)), id(1)),
            0.0
        );
    }

    struct ByScoreDefault;
    impl TraversalHeuristic for ByScoreDefault {
        fn order(&self, _: &TraversalContext<'_>, _: &mut Vec<ElementId>, _: &mut StdRng) {}
    }

    #[test]
    fn breadth_and_depth_first_sort_by_distance_then_id() {
        let w = World::lattice(5, 5);
        let calibrated = ElementSet::new(w.graph.len());
        let n = w.graph.lattice_node(2, 2).unwrap();
        let ctx = w.ctx(&calibrated, n);
        // All nodes within d = 3 of n, in reverse ID order.
        let mut nodes: Vec<ElementId> = w
            .neighbourhoods
            .within(n, 3)
            .iter()
            .copied()
            .filter(|&e| w.graph.kind(e).is_node() && e != n)
            .collect();
        nodes.sort_unstable_by(|a, b| b.cmp(a));
        for h in [&BreadthFirst as &dyn TraversalHeuristic, &DepthFirst] {
            let mut options = nodes.clone();
            h.order(&ctx, &mut options, &mut rng(0));
            let keys: Vec<(u32, ElementId)> = options
                .iter()
                .map(|&e| (w.neighbourhoods.incidence_distance(n, e).unwrap(), e))
                .collect();
            assert!(keys.windows(2).all(|p| p[0] < p[1]), "{keys:?}");
            assert_eq!(
                options.iter().copied().collect::<BTreeSet<_>>(),
                nodes.iter().copied().collect::<BTreeSet<_>>()
            );
            assert_eq!(keys[0].0, 2);
            assert_eq!(keys.last().unwrap().0, 4);
        }
        // Edge-centred: edge options by distance (2 before 4), then ID.
        let e = w
            .graph
            .edge_between(
                n,
                w.graph.lattice_node(2, 3).unwrap(),
                Some(EdgeType::Engineered),
            )
            .unwrap();
        let ctx = w.ctx(&calibrated, e);
        let mut options: Vec<ElementId> = w
            .neighbourhoods
            .within(e, 2)
            .iter()
            .copied()
            .filter(|&x| w.graph.kind(x) == ElementKind::Edge(EdgeType::Engineered) && x != e)
            .collect();
        options.reverse();
        BreadthFirst.order(&ctx, &mut options, &mut rng(0));
        let dists: Vec<u32> = options
            .iter()
            .map(|&x| w.neighbourhoods.incidence_distance(e, x).unwrap())
            .collect();
        assert!(dists.windows(2).all(|p| p[0] <= p[1]));
        assert!(dists.contains(&2) && dists.contains(&4));
    }

    #[test]
    fn proximity_sort_puts_unindexed_options_last_and_is_idempotent() {
        let w = World::lattice(1, 8);
        let calibrated = ElementSet::new(w.graph.len());
        // The index only reaches d = 3 (incidence radius 5); q0_7 is farther from q0_0.
        let a = w.graph.lattice_node(0, 0).unwrap();
        let far = w.graph.lattice_node(0, 7).unwrap();
        let near = w.graph.lattice_node(0, 1).unwrap();
        let ctx = w.ctx(&calibrated, a);
        let mut options = vec![far, near];
        DepthFirst.order(&ctx, &mut options, &mut rng(0));
        assert_eq!(options, vec![near, far]);
        let again = options.clone();
        DepthFirst.order(&ctx, &mut options, &mut rng(0));
        assert_eq!(options, again);
    }

    #[test]
    fn random_is_a_deterministic_permutation() {
        let w = World::lattice(4, 4);
        let calibrated = ElementSet::new(w.graph.len());
        let ctx = w.ctx(&calibrated, id(0));
        let input: Vec<ElementId> = (0..16).map(id).collect();
        let shuffled = |seed| {
            let mut v = input.clone();
            Random.order(&ctx, &mut v, &mut rng(seed));
            v
        };
        for seed in 0..20 {
            let a = shuffled(seed);
            assert_eq!(a, shuffled(seed), "same seed, same order");
            let mut sorted = a.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, input, "a permutation");
        }
        assert!((0..20).any(|s| shuffled(s) != input), "actually shuffles");
        assert_ne!(shuffled(1), shuffled(2));
        // Edge cases.
        let mut empty = Vec::new();
        Random.order(&ctx, &mut empty, &mut rng(0));
        assert!(empty.is_empty());
        let mut one = vec![id(5)];
        let mut r = rng(0);
        Random.order(&ctx, &mut one, &mut r);
        assert_eq!(one, vec![id(5)]);
        // One element consumes no randomness.
        assert_eq!(
            rand::Rng::random::<u64>(&mut r),
            rand::Rng::random::<u64>(&mut rng(0))
        );
    }

    #[test]
    fn random_is_roughly_uniform() {
        let w = World::lattice(2, 2);
        let calibrated = ElementSet::new(w.graph.len());
        let ctx = w.ctx(&calibrated, id(0));
        let mut counts = std::collections::BTreeMap::new();
        let mut r = rng(42);
        for _ in 0..6000 {
            let mut v = ids(&[0, 1, 2]);
            Random.order(&ctx, &mut v, &mut r);
            *counts.entry(v).or_insert(0usize) += 1;
        }
        assert_eq!(counts.len(), 6, "all 3! permutations occur");
        assert!(
            counts.values().all(|&c| (800..1200).contains(&c)),
            "{counts:?}"
        );
    }

    // ----- traversal orders -----

    /// Visits the available elements from `seed` with an explicit frontier.
    fn frontier_order(
        w: &World,
        scope: &Scope<'_>,
        goal: &ElementSet,
        seed: ElementId,
        heuristic: &dyn TraversalHeuristic,
        seed_rng: u64,
    ) -> Vec<ElementId> {
        let mut available = goal.clone();
        let mut calibrated = ElementSet::new(w.graph.len());
        let mut rng = rng(seed_rng);
        let mut frontier = Frontier::new(heuristic.frontier());
        frontier.push(seed);
        let mut visited = Vec::new();
        let mut options = Vec::new();
        while let Some(g) = frontier.pop() {
            if calibrated.contains(g) {
                continue;
            }
            calibrated.insert(g);
            available.remove(g);
            visited.push(g);
            scope.traversal_options(g, &available, &mut options);
            let ctx = w.ctx(&calibrated, g);
            frontier.extend_ordered(heuristic, &ctx, &mut options, &mut rng);
        }
        visited
    }

    /// The paper's recursive traversal, written out literally.
    fn recursive_order(
        w: &World,
        scope: &Scope<'_>,
        goal: &ElementSet,
        seed: ElementId,
        heuristic: &dyn TraversalHeuristic,
        seed_rng: u64,
    ) -> Vec<ElementId> {
        struct State<'a> {
            w: &'a World,
            scope: &'a Scope<'a>,
            heuristic: &'a dyn TraversalHeuristic,
            available: ElementSet,
            calibrated: ElementSet,
            rng: StdRng,
            visited: Vec<ElementId>,
        }
        fn recurse(s: &mut State<'_>, g: ElementId) {
            if s.calibrated.contains(g) {
                return;
            }
            s.calibrated.insert(g);
            s.available.remove(g);
            s.visited.push(g);
            let mut options = Vec::new();
            s.scope.traversal_options(g, &s.available, &mut options);
            let ctx = s.w.ctx(&s.calibrated, g);
            s.heuristic.order(&ctx, &mut options, &mut s.rng);
            for o in options {
                recurse(s, o);
            }
        }
        let mut state = State {
            w,
            scope,
            heuristic,
            available: goal.clone(),
            calibrated: ElementSet::new(w.graph.len()),
            rng: rng(seed_rng),
            visited: Vec::new(),
        };
        recurse(&mut state, seed);
        state.visited
    }

    #[test]
    fn lifo_reproduces_the_recursive_pseudo_code() {
        let w = World::lattice(5, 6);
        let goal = w
            .graph
            .goal_excluding(&[ElementKind::Edge(EdgeType::Parasitic)]);
        for d_t in [1, 2, 3] {
            let scope = w.scope(d_t);
            for seed in [
                w.graph.lattice_node(0, 0).unwrap(),
                w.graph.lattice_node(2, 3).unwrap(),
                id(40),
            ] {
                for h in [&DepthFirst as &dyn TraversalHeuristic, &ShuffledLifo] {
                    let a = frontier_order(&w, &scope, &goal, seed, h, 5);
                    let b = recursive_order(&w, &scope, &goal, seed, h, 5);
                    assert_eq!(a, b, "d_T={d_t} seed={seed}");
                    assert!(!a.is_empty());
                }
            }
        }
    }

    #[test]
    fn lifo_order_on_a_path_is_the_recursion() {
        // A 1x5 lattice with d_T = 2 on nodes: from q0_2 the options are q0_1, q0_3 (nearest
        // first by ID), then q0_0 and q0_4 are reached from them.
        let w = World::lattice(1, 5);
        let goal = w.graph.goal(|e| e.is_node());
        let scope = w.scope(2);
        let order = frontier_order(&w, &scope, &goal, id(2), &DepthFirst, 0);
        assert_eq!(order, ids(&[2, 1, 0, 3, 4]));
        let order = frontier_order(&w, &scope, &goal, id(2), &BreadthFirst, 0);
        assert_eq!(order, ids(&[2, 1, 3, 0, 4]));
    }

    #[test]
    fn fifo_visits_in_breadth_first_levels() {
        // Nodes of a lattice with d_T = 2 form a "king graph": the level of a node is its
        // Chebyshev distance from the seed.
        let w = World::lattice(7, 7);
        let goal = w.graph.goal(|e| e.is_node());
        let scope = w.scope(2);
        let seed = w.graph.lattice_node(3, 3).unwrap();
        let rc = |e: ElementId| (e.index() / 7, e.index() % 7);
        for h in [&BreadthFirst as &dyn TraversalHeuristic, &Random] {
            let order = frontier_order(&w, &scope, &goal, seed, h, 3);
            assert_eq!(order.len(), 49);
            let levels: Vec<usize> = order
                .iter()
                .map(|&e| {
                    let (r, c) = rc(e);
                    r.abs_diff(3).max(c.abs_diff(3))
                })
                .collect();
            assert!(levels.windows(2).all(|p| p[0] <= p[1]), "{levels:?}");
            assert_eq!(levels.iter().filter(|&&l| l == 1).count(), 8);
            assert_eq!(levels.iter().filter(|&&l| l == 2).count(), 16);
            assert_eq!(levels.iter().filter(|&&l| l == 3).count(), 24);
        }
        // Deterministic order for BreadthFirst: ring by ring, nearest first, ascending IDs.
        let order = frontier_order(&w, &scope, &goal, seed, &BreadthFirst, 0);
        let first_ring: Vec<usize> = order[1..9].iter().map(|e| e.index()).collect();
        assert_eq!(first_ring, vec![16, 17, 18, 23, 25, 30, 31, 32]);
    }

    #[test]
    fn random_traversal_is_deterministic_and_covers_everything() {
        let w = World::lattice(5, 5);
        let goal = w.graph.goal(|e| e.is_node());
        let scope = w.scope(2);
        let seed = id(0);
        let a = frontier_order(&w, &scope, &goal, seed, &Random, 11);
        let b = frontier_order(&w, &scope, &goal, seed, &Random, 11);
        let c = frontier_order(&w, &scope, &goal, seed, &Random, 12);
        assert_eq!(a, b);
        assert_ne!(a, c);
        let mut sorted = a.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..25).map(id).collect::<Vec<_>>());
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(40))]

        #[test]
        fn lifo_equals_recursion_on_random_graphs(
            seed in any::<u64>(),
            rng_seed in any::<u64>(),
            d_t in 1u32..=3,
            shuffled in any::<bool>(),
        ) {
            let graph = random_graph(seed, 10, 18);
            let neighbourhoods = Neighbourhoods::build(&graph, 3);
            let activity = AlgorithmSubgraphs::all_active(&graph).masks();
            let w = World { graph, neighbourhoods, activity };
            let goal = w.graph.all_elements();
            let scope = w.scope(d_t);
            let h: &dyn TraversalHeuristic = if shuffled { &ShuffledLifo } else { &DepthFirst };
            for start in w.graph.ids() {
                let a = frontier_order(&w, &scope, &goal, start, h, rng_seed);
                let b = recursive_order(&w, &scope, &goal, start, h, rng_seed);
                prop_assert_eq!(a, b);
            }
        }

        #[test]
        fn every_frontier_visits_the_same_set(seed in any::<u64>(), d_t in 1u32..=3) {
            let graph = random_graph(seed, 10, 18);
            let neighbourhoods = Neighbourhoods::build(&graph, 3);
            let activity = AlgorithmSubgraphs::all_active(&graph).masks();
            let w = World { graph, neighbourhoods, activity };
            let goal = w.graph.all_elements();
            let scope = w.scope(d_t);
            let start = id(0);
            let reference: BTreeSet<ElementId> =
                frontier_order(&w, &scope, &goal, start, &DepthFirst, 0).into_iter().collect();
            for h in [&BreadthFirst as &dyn TraversalHeuristic, &Random, &ByScore(|e| f64::from(e.raw() % 4))] {
                let visited = frontier_order(&w, &scope, &goal, start, h, 9);
                prop_assert_eq!(visited.len(), visited.iter().collect::<BTreeSet<_>>().len());
                prop_assert_eq!(visited.into_iter().collect::<BTreeSet<_>>(), reference.clone());
            }
        }
    }
}
