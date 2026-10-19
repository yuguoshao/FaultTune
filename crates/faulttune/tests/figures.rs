//! The figures of the paper, reproduced through the public API only (ARCHITECTURE section 13,
//! level "Paper figures"; section 16.5 records the results).
//!
//! * Fig. 2: the connectivity subgraphs `X_g^d`.
//! * Fig. 3: the algorithm subgraphs of cross-entropy benchmarking (XEB).
//! * Fig. 4: the traversal options with `d_T = 2`: 8 for a node, 6 for an engineered edge.
//! * Fig. 7: the real Snake on 6x6 and 7x7 lattices with the paper's distances (`d_P = 1`,
//!   `d_R = 2`, `d_T = 2`): the committed steps have 1 to 5 parameters, and the number of
//!   constraints per step is recorded.
//!
//! **The constraint count of the paper is not reproduced.** The paper reports 0 to 8 active
//! constraints per step for these distances. The definition of ARCHITECTURE section 4.4 (every
//! calibrated element that is within `d_R` of *any* parameter of the step and can be active at
//! the same time as it, edges included) gives a larger maximum: 18 for a breadth-first walk and
//! up to 26 for a depth-first walk from an interior seed. Counting nodes only, a depth-first
//! walk reaches exactly 8 and a breadth-first walk 5. Which definition the paper uses is an open
//! question (ARCHITECTURE section 15), not a defect, so the definition is not weakened and the
//! tests pin the numbers it produces.
//!
//! The file is self-contained.

use std::collections::BTreeSet;

use faulttune::heuristics::{BreadthFirst, DepthFirst};
use faulttune::toy::GateFrequencyModel;
use faulttune::{
    ActivityMasks, AlgorithmSubgraphs, Distances, EdgeType, ElementId, ElementKind, ElementSet,
    Event, FaultTuneError, Neighbourhoods, ProcessorGraph, RecordingObserver, Scope, SeedPolicy,
    Snake, SnakeConfig, StepRecord, TraversalHeuristic,
};

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

const ENGINEERED: ElementKind = ElementKind::Edge(EdgeType::Engineered);
const PARASITIC: ElementKind = ElementKind::Edge(EdgeType::Parasitic);

fn node(graph: &ProcessorGraph, r: usize, c: usize) -> ElementId {
    graph.lattice_node(r, c).expect("a node of the lattice")
}

/// The engineered edge between two lattice nodes.
fn engineered(graph: &ProcessorGraph, a: (usize, usize), b: (usize, usize)) -> ElementId {
    graph
        .edge_between(
            node(graph, a.0, a.1),
            node(graph, b.0, b.1),
            Some(EdgeType::Engineered),
        )
        .expect("an engineered edge")
}

fn set(ids: &[ElementId]) -> BTreeSet<ElementId> {
    ids.iter().copied().collect()
}

/// The lattice position `(row, col)` of a node.
fn position(graph: &ProcessorGraph, id: ElementId) -> (usize, usize) {
    let (r, c) = graph.label(id)[1..]
        .split_once('_')
        .expect("a label q<r>_<c>");
    (r.parse().unwrap(), c.parse().unwrap())
}

/// The engineered edges incident to a node.
fn engineered_incident(graph: &ProcessorGraph, n: ElementId) -> Vec<ElementId> {
    graph
        .incident(n)
        .iter()
        .copied()
        .filter(|&e| graph.kind(e) == ENGINEERED)
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Fig. 2: connectivity subgraphs
// ---------------------------------------------------------------------------------------------

#[test]
fn fig2_connectivity_subgraphs() {
    let graph = ProcessorGraph::square_lattice(5, 5);
    let hoods = Neighbourhoods::build(&graph, 3);
    let centre = node(&graph, 2, 2);

    // X_n^0 is the node alone, X_e^0 the edge alone.
    assert_eq!(hoods.within(centre, 0), &[centre]);
    for e in graph.edges() {
        assert_eq!(hoods.within(e, 0), &[e], "X_e^0 is the edge alone");
    }

    // X_n^1: the node and its incident edges (4 engineered, 4 parasitic).
    let x1 = set(hoods.within(centre, 1));
    let mut expected = set(graph.incident(centre));
    expected.insert(centre);
    assert_eq!(x1, expected);
    assert_eq!(x1.len(), 9);
    let kinds = |ids: &BTreeSet<ElementId>, kind: ElementKind| {
        ids.iter().filter(|&&e| graph.kind(e) == kind).count()
    };
    assert_eq!(kinds(&x1, ElementKind::Node), 1);
    assert_eq!(kinds(&x1, ENGINEERED), 4);
    assert_eq!(kinds(&x1, PARASITIC), 4);

    // X_n^2: the 3x3 block of nodes around the node, with all their incident edges.
    let x2 = set(hoods.within(centre, 2));
    let block: Vec<ElementId> = (1..=3)
        .flat_map(|r| (1..=3).map(move |c| (r, c)))
        .map(|(r, c)| node(&graph, r, c))
        .collect();
    let mut expected = set(&block);
    for &n in &block {
        expected.extend(graph.incident(n).iter().copied());
    }
    assert_eq!(x2, expected);
    assert_eq!(x2.len(), 61, "9 nodes and the 52 edges that touch them");
    assert_eq!(kinds(&x2, ElementKind::Node), 9);

    // X_e^1 of an edge: the edge, its two end points and every edge incident to them.
    let edge = engineered(&graph, (2, 2), (2, 3));
    let [a, b] = graph.endpoints(edge).unwrap();
    let mut expected = set(&[edge, a, b]);
    expected.extend(graph.incident(a).iter().copied());
    expected.extend(graph.incident(b).iter().copied());
    assert_eq!(set(hoods.within(edge, 1)), expected);
}

#[test]
fn fig2_the_relation_is_symmetric_for_every_pair_of_elements() {
    // The paper requires "a is within d of b" to be symmetric; segmentation relies on it.
    let graph = ProcessorGraph::square_lattice(4, 5);
    let hoods = Neighbourhoods::build(&graph, 3);
    for d in 0..=3 {
        for a in graph.ids() {
            let from_a = set(hoods.within(a, d));
            for b in graph.ids() {
                assert_eq!(
                    from_a.contains(&b),
                    hoods.is_within(a, b, d),
                    "is_within disagrees with within for {a} and {b} at d = {d}"
                );
                assert_eq!(
                    hoods.is_within(a, b, d),
                    hoods.is_within(b, a, d),
                    "{a} and {b} at d = {d}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Fig. 3: the XEB algorithm subgraphs
// ---------------------------------------------------------------------------------------------

#[test]
fn fig3_xeb_algorithm_subgraphs() {
    for (rows, cols) in [(6, 6), (7, 7), (4, 7)] {
        let graph = ProcessorGraph::square_lattice(rows, cols);
        let xeb = AlgorithmSubgraphs::xeb_square_lattice(&graph, rows, cols).unwrap();
        assert_eq!(xeb.len(), 5, "XEB needs five algorithm subgraphs");
        let all_nodes: BTreeSet<ElementId> = graph.nodes().collect();
        let engineered_edges: BTreeSet<ElementId> = graph
            .edges()
            .filter(|&e| graph.kind(e) == ENGINEERED)
            .collect();

        // A_0 is the single-qubit layer: all the nodes and nothing else.
        assert_eq!(
            xeb.get(0).unwrap().iter().collect::<BTreeSet<_>>(),
            all_nodes
        );

        // A_1 .. A_4 each hold all the nodes (idle qubits are active) and a matching of
        // engineered edges; the four matchings partition the engineered edges.
        let mut union = BTreeSet::new();
        for i in 1..=4 {
            let a = xeb.get(i).unwrap();
            let edges: Vec<ElementId> = a
                .iter()
                .filter(|&e| graph.kind(e) != ElementKind::Node)
                .collect();
            assert!(
                edges.iter().all(|&e| graph.kind(e) == ENGINEERED),
                "parasitic edges belong to no subgraph"
            );
            assert!(
                all_nodes.iter().all(|&n| a.contains(n)),
                "A_{i} holds every node"
            );
            let mut touched = BTreeSet::new();
            for &e in &edges {
                for end in graph.endpoints(e).unwrap() {
                    assert!(
                        touched.insert(end),
                        "A_{i} is a matching: {} is used twice",
                        end
                    );
                }
                assert!(union.insert(e), "the matchings are disjoint");
            }
            assert!(!edges.is_empty());
        }
        assert_eq!(
            union, engineered_edges,
            "the matchings cover every engineered edge"
        );
        assert!(graph
            .edges()
            .filter(|&e| graph.kind(e) == PARASITIC)
            .all(|e| { xeb.iter().all(|a| !a.contains(e)) }));

        // Two elements can be active at the same time exactly when they share a subgraph.
        let masks = xeb.masks();
        assert_eq!(masks.len(), graph.len());
        for e in graph.ids().filter(|&e| graph.kind(e) == ENGINEERED) {
            assert!(masks.is_active(e));
            // Every engineered edge is in exactly one of A_1 .. A_4.
            assert_eq!(masks.mask(e).count_ones(), 1, "{}", graph.label(e));
            assert!(masks.co_active(e, graph.endpoints(e).unwrap()[0]));
        }
        for p in graph.ids().filter(|&e| graph.kind(e) == PARASITIC) {
            assert!(!masks.is_active(p));
        }
        // Two edges that share a qubit are never active together.
        for n in graph.nodes() {
            let incident = engineered_incident(&graph, n);
            for (i, &a) in incident.iter().enumerate() {
                for &b in &incident[i + 1..] {
                    assert!(
                        !masks.co_active(a, b),
                        "{} and {}",
                        graph.label(a),
                        graph.label(b)
                    );
                }
            }
        }
    }
}

#[test]
fn fig3_xeb_subgraphs_refuse_a_graph_that_is_not_the_lattice() {
    let graph = ProcessorGraph::square_lattice(6, 6);
    for (rows, cols) in [(6, 7), (5, 6), (7, 6), (0, 0)] {
        let err = AlgorithmSubgraphs::xeb_square_lattice(&graph, rows, cols).unwrap_err();
        assert!(
            matches!(err, FaultTuneError::InvalidConfig(_)),
            "{rows}x{cols}: {err:?}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Fig. 4: traversal options
// ---------------------------------------------------------------------------------------------

#[test]
fn fig4_traversal_options_with_d_t_2() {
    let (rows, cols) = (7, 7);
    let graph = ProcessorGraph::square_lattice(rows, cols);
    let xeb = AlgorithmSubgraphs::xeb_square_lattice(&graph, rows, cols).unwrap();
    let masks: ActivityMasks = xeb.masks();
    let hoods = Neighbourhoods::build(&graph, 2);
    let scope = Scope::new(
        &graph,
        &hoods,
        &masks,
        Distances {
            parameter: 1,
            constraint: 2,
            traversal: 2,
        },
    );
    // Everything of the goal is still available (nothing calibrated yet).
    let available = graph.goal_excluding(&[PARASITIC]);
    let mut out = Vec::new();

    // An interior node has 8 options: the 4 nearest and the 4 diagonal neighbours.
    for (r, c) in [(3, 3), (2, 4), (1, 1), (5, 5)] {
        let centre = node(&graph, r, c);
        scope.traversal_options(centre, &available, &mut out);
        let got: BTreeSet<(usize, usize)> = out.iter().map(|&n| position(&graph, n)).collect();
        let expected: BTreeSet<(usize, usize)> = [
            (r - 1, c - 1),
            (r - 1, c),
            (r - 1, c + 1),
            (r, c - 1),
            (r, c + 1),
            (r + 1, c - 1),
            (r + 1, c),
            (r + 1, c + 1),
        ]
        .into_iter()
        .collect();
        assert_eq!(got, expected, "node q{r}_{c}");
        assert_eq!(out.len(), 8);
    }
    // At the border fewer neighbours exist: 5 on a side, 3 at a corner.
    scope.traversal_options(node(&graph, 0, 3), &available, &mut out);
    assert_eq!(out.len(), 5);
    scope.traversal_options(node(&graph, 0, 0), &available, &mut out);
    assert_eq!(out.len(), 3);

    // An interior engineered edge has 6 options among the engineered edges: for a horizontal
    // edge 2 above, 2 below and 2 in line with it (and the vertical ones likewise). Only edges
    // of the same two-qubit layer pattern can be active at the same time, which is why the
    // neighbouring parallel edges (sharing a qubit with a pattern member) are not options.
    let h = |r, c| engineered(&graph, (r, c), (r, c + 1));
    let v = |r, c| engineered(&graph, (r, c), (r + 1, c));
    let horizontal = h(3, 3);
    scope.traversal_options(horizontal, &available, &mut out);
    let expected = set(&[h(3, 1), h(3, 5), h(2, 2), h(2, 4), h(4, 2), h(4, 4)]);
    assert_eq!(
        out.len(),
        6,
        "an interior horizontal edge has 6 traversal options"
    );
    assert_eq!(set(&out), expected);
    // The edge itself is never an option, and no edge sharing a qubit with it is (they are
    // never active at the same time).
    assert!(!out.contains(&horizontal));
    for end in graph.endpoints(horizontal).unwrap() {
        for e in engineered_incident(&graph, end) {
            assert!(!out.contains(&e));
        }
    }
    // The same for a vertical edge.
    let vertical = v(3, 3);
    scope.traversal_options(vertical, &available, &mut out);
    let expected = set(&[v(1, 3), v(5, 3), v(2, 2), v(4, 2), v(2, 4), v(4, 4)]);
    assert_eq!(
        out.len(),
        6,
        "an interior vertical edge has 6 traversal options"
    );
    assert_eq!(set(&out), expected);
}

// ---------------------------------------------------------------------------------------------
// Fig. 7: the Snake on 6x6 and 7x7 lattices
// ---------------------------------------------------------------------------------------------

/// A run of the real Snake with the XEB subgraphs and the paper's distances.
struct XebRun {
    graph: ProcessorGraph,
    steps: Vec<StepRecord>,
}

fn run_xeb(
    rows: usize,
    cols: usize,
    heuristic: impl TraversalHeuristic + 'static,
    seed: Option<(usize, usize)>,
) -> XebRun {
    let graph = ProcessorGraph::square_lattice(rows, cols);
    let goal = graph.goal(|e| e.kind() != PARASITIC);
    let xeb = AlgorithmSubgraphs::xeb_square_lattice(&graph, rows, cols).unwrap();
    let config = SnakeConfig {
        parameter_distance: 1,  // d_P
        constraint_distance: 2, // d_R
        traversal_distance: 2,  // d_T
        seed: 5,
        ..SnakeConfig::default()
    };
    let recorder = RecordingObserver::new();
    let mut snake = Snake::new(&graph, goal.clone(), config)
        .algorithm(xeb)
        .heuristic(heuristic)
        .observer(recorder.clone());
    if let Some((r, c)) = seed {
        snake = snake.seed_policy(SeedPolicy::Fixed(vec![node(&graph, r, c)]));
    }
    let model = GateFrequencyModel::new(&graph, 5);
    let db = snake.calibrate(&model).unwrap();
    assert_eq!(db.calibrated(), &goal);
    drop(snake);
    let steps = recorder
        .events()
        .into_iter()
        .filter_map(|e| match e {
            Event::StepCommitted(s) => Some(s),
            _ => None,
        })
        .collect();
    XebRun { graph, steps }
}

impl XebRun {
    fn parameter_range(&self) -> (usize, usize) {
        let counts = self.steps.iter().map(|s| s.parameters.len());
        (counts.clone().min().unwrap(), counts.max().unwrap())
    }

    fn constraint_range(&self) -> (usize, usize) {
        let counts = self.steps.iter().map(|s| s.constraints.len());
        (counts.clone().min().unwrap(), counts.max().unwrap())
    }

    /// The largest number of constraints that are nodes.
    fn max_node_constraints(&self) -> usize {
        self.steps
            .iter()
            .map(|s| {
                s.constraints
                    .iter()
                    .filter(|&&c| self.graph.kind(c) == ElementKind::Node)
                    .count()
            })
            .max()
            .unwrap()
    }
}

/// Replays the events in commit order (a single subgoal) and checks every step against the
/// definitions, recomputed here from the public neighbourhood index and activity masks.
fn check_steps_against_the_definitions(run: &XebRun, rows: usize, cols: usize, label: &str) {
    let graph = &run.graph;
    let xeb = AlgorithmSubgraphs::xeb_square_lattice(graph, rows, cols).unwrap();
    let masks = xeb.masks();
    let hoods = Neighbourhoods::build(graph, 2);
    let goal = graph.goal(|e| e.kind() != PARASITIC);
    let mut calibrated = ElementSet::new(graph.len());
    for (i, step) in run.steps.iter().enumerate() {
        let ctx = format!("{label}: step {i} at {}", graph.label(step.centre));

        // Parameters: the uncalibrated goal elements within d_P = 1 of the centre.
        let mut expected: BTreeSet<ElementId> = hoods
            .within(step.centre, 1)
            .iter()
            .copied()
            .filter(|&e| goal.contains(e) && !calibrated.contains(e))
            .collect();
        assert_eq!(set(&step.parameters), expected, "{ctx}: parameters");
        assert_eq!(
            step.parameters[0], step.centre,
            "{ctx}: the centre comes first"
        );
        assert_eq!(step.parameters.len(), step.option_indices.len(), "{ctx}");
        assert!(
            (1..=5).contains(&step.parameters.len()),
            "{ctx}: 1 to 5 parameters"
        );

        // Constraints: calibrated elements within d_R = 2 of some parameter and co-active
        // with it.
        expected = graph
            .ids()
            .filter(|&c| {
                calibrated.contains(c)
                    && step
                        .parameters
                        .iter()
                        .any(|&p| hoods.is_within(c, p, 2) && masks.co_active(c, p))
            })
            .collect();
        assert_eq!(set(&step.constraints), expected, "{ctx}: constraints");
        assert!(
            step.constraints.windows(2).all(|w| w[0] < w[1]),
            "{ctx}: ascending"
        );

        // A node-centred step has the node and its still uncalibrated engineered edges.
        if graph.kind(step.centre) == ElementKind::Node {
            let free_edges = engineered_incident(graph, step.centre)
                .into_iter()
                .filter(|&e| !calibrated.contains(e))
                .count();
            assert_eq!(step.parameters.len(), 1 + free_edges, "{ctx}");
            let (r, c) = position(graph, step.centre);
            let on_border = r == 0 || c == 0 || r + 1 == rows || c + 1 == cols;
            let at_corner = (r == 0 || r + 1 == rows) && (c == 0 || c + 1 == cols);
            let bound = if at_corner {
                3
            } else if on_border {
                4
            } else {
                5
            };
            assert!(
                step.parameters.len() <= bound,
                "{ctx}: at most {bound} parameters here"
            );
        }
        for &p in &step.parameters {
            calibrated.insert(p);
        }
    }
    assert_eq!(
        &calibrated, &goal,
        "{label}: every goal element was a parameter exactly once"
    );
}

#[test]
fn fig7_steps_have_one_to_five_parameters_on_6x6_and_7x7() {
    for (rows, cols) in [(6, 6), (7, 7)] {
        for seed in [None, Some((rows / 2, cols / 2))] {
            let run = run_xeb(rows, cols, BreadthFirst, seed);
            let label = format!("{rows}x{cols} breadth-first, seed {seed:?}");
            let (lo, hi) = run.parameter_range();
            assert_eq!(lo, 1, "{label}: the last steps have a single parameter");
            if seed.is_some() {
                // From an interior seed the very first step has the node and its four
                // engineered edges.
                assert_eq!(hi, 5, "{label}");
                assert_eq!(run.steps[0].parameters.len(), 5, "{label}");
            } else {
                // The lowest-ID seed is the corner q0_0: a corner has only two edges.
                assert_eq!(run.steps[0].parameters.len(), 3, "{label}");
                assert!(
                    hi <= 3,
                    "{label}: a breadth-first walk from a corner finds {hi}"
                );
            }
            check_steps_against_the_definitions(&run, rows, cols, &label);
        }
    }
}

#[test]
fn fig7_a_node_step_with_no_calibrated_edge_has_five_parameters() {
    // Interior node-centred steps have 5 parameters whenever none of the node's four edges was
    // calibrated before; steps further along the walk usually share an edge with an earlier
    // step. Count both kinds on a depth-first walk, which has plenty of each.
    let run = run_xeb(7, 7, DepthFirst, Some((3, 3)));
    let graph = &run.graph;
    let mut calibrated = ElementSet::new(graph.len());
    let mut fresh_interior = 0;
    let mut shared_interior = 0;
    for step in &run.steps {
        let (r, c) = position(graph, step.centre);
        let interior = (1..6).contains(&r) && (1..6).contains(&c);
        let free = engineered_incident(graph, step.centre)
            .into_iter()
            .filter(|&e| !calibrated.contains(e))
            .count();
        if interior {
            if free == 4 {
                assert_eq!(step.parameters.len(), 5, "q{r}_{c}");
                fresh_interior += 1;
            } else {
                assert_eq!(step.parameters.len(), 1 + free, "q{r}_{c}");
                assert!(step.parameters.len() < 5);
                shared_interior += 1;
            }
        }
        for &p in &step.parameters {
            calibrated.insert(p);
        }
    }
    assert!(
        fresh_interior >= 1,
        "the interior seed is a fresh interior step"
    );
    assert!(shared_interior >= 1);
}

#[test]
fn fig7_constraints_per_step_are_recorded_and_exceed_the_papers_eight() {
    // (rows, depth-first?, interior seed?, expected min..=max of the constraint counts,
    //  expected maximum of the node-only counts)
    type Seed = Option<(usize, usize)>;
    type Range = (usize, usize);
    let cases: [(usize, bool, Seed, Range, usize); 6] = [
        (6, false, None, (0, 18), 5),
        (7, false, None, (0, 18), 5),
        (6, false, Some((3, 3)), (0, 18), 5),
        (7, false, Some((3, 3)), (0, 18), 5),
        (6, true, Some((3, 3)), (0, 26), 8),
        (7, true, Some((3, 3)), (0, 26), 8),
    ];
    for (rows, depth_first, seed, expected, node_only) in cases {
        let label = format!("{rows}x{rows}, depth_first = {depth_first}, seed {seed:?}");
        let run = if depth_first {
            run_xeb(rows, rows, DepthFirst, seed)
        } else {
            run_xeb(rows, rows, BreadthFirst, seed)
        };
        let observed = run.constraint_range();
        // The first step of a run has nothing to be constrained by.
        assert_eq!(observed.0, 0, "{label}");
        assert_eq!(run.steps[0].constraints.len(), 0, "{label}");
        // The paper (Fig. 7) reports 0 to 8 constraints per step. The literal definition of
        // ARCHITECTURE 4.4 gives a larger maximum, and the number is pinned here as observed.
        assert!(
            observed.1 > 8,
            "{label}: the definition gives more than the paper's 8"
        );
        assert_eq!(observed, expected, "{label}: observed constraints per step");
        // Counting nodes only, a depth-first walk reaches the paper's 8 exactly.
        assert_eq!(
            run.max_node_constraints(),
            node_only,
            "{label}: node constraints only"
        );
        check_steps_against_the_definitions(&run, rows, rows, &label);
    }
}

#[test]
fn fig7_the_counts_do_not_depend_on_the_model_or_the_optimizer() {
    // The traversal, and with it the scope of every step, depends on the graph, the distances
    // and the algorithm subgraphs only, not on the values the optimizer picks.
    let reference = run_xeb(6, 6, BreadthFirst, None);
    let graph = ProcessorGraph::square_lattice(6, 6);
    let goal = graph.goal(|e| e.kind() != PARASITIC);
    let xeb = AlgorithmSubgraphs::xeb_square_lattice(&graph, 6, 6).unwrap();
    let recorder = RecordingObserver::new();
    let snake = Snake::new(
        &graph,
        goal,
        SnakeConfig {
            seed: 99,
            ..SnakeConfig::default()
        },
    )
    .algorithm(xeb)
    .optimizer(faulttune::optimizers::RandomSearch::default())
    .observer(recorder.clone());
    snake
        .calibrate(&faulttune::toy::SeparableModel::new(
            &graph, 10, 5.0, 7.0, 1,
        ))
        .unwrap();
    let steps: Vec<StepRecord> = recorder
        .events()
        .into_iter()
        .filter_map(|e| match e {
            Event::StepCommitted(s) => Some(s),
            _ => None,
        })
        .collect();
    let scope_of = |s: &StepRecord| (s.centre, s.parameters.clone(), s.constraints.clone());
    assert_eq!(
        steps.iter().map(scope_of).collect::<Vec<_>>(),
        reference.steps.iter().map(scope_of).collect::<Vec<_>>()
    );
}
