# FaultTune

**A Rust implementation of the Snake Optimizer.** FaultTune learns the control parameters of a quantum processor, or any other system whose parameters sit on a graph and interfere with their neighbours. It works through the graph one small neighbourhood at a time, choosing parameter values that keep computational errors low.

> **Status: early development.** No code has been published yet. The API below describes the planned design and may change before the first release.

---

## Background

FaultTune is based on **The Snake Optimizer for Learning Quantum Processor Control Parameters** by Paul V. Klimov, Julian Kelly, John M. Martinis and Hartmut Neven (Google AI Quantum and UC Santa Barbara, 2020, [arXiv:2006.04594](https://arxiv.org/abs/2006.04594), [Google Research page](https://research.google/pubs/the-snake-optimizer-for-learning-quantum-processor-control-parameters/)).

Google used the Snake to choose the frequencies at which quantum logic gates run on the 53-qubit Sycamore processor, as part of its quantum supremacy experiment. According to the paper, the Snake found a full-processor configuration about **10⁴ times faster than a human expert**, with about **20% lower median two-qubit gate error**.

The name **FaultTune** describes the goal: *tune* control parameters so that the processor makes as few *faults* (computational errors) as possible.

## The problem

A quantum processor must be calibrated before it can run algorithms, and calibration has to finish much faster than the hardware drifts. Calibration is split into a stack of layers. Each layer learns one kind of parameter for the relevant parts of the processor, such as qubits, couplers or readout resonators.

- **Independent layers** are easy. If the parts don't interact much, each parameter can be learned on its own and in parallel. An example is mapping control voltages to qubit frequencies.
- **Interdependent layers** are hard. If the parts interact through designed couplings or unwanted crosstalk, every parameter depends on every other one, and they must be learned together.

The main interdependent example is **gate-frequency calibration**. Every single-qubit and two-qubit gate needs an operating frequency, and the frequencies must not collide with each other. For `N` qubits on a square grid, this is a combinatorial problem with `O(N)` dimensions and a search space of size `k^O(N)`, where `k` (about 100 in practice) is the number of frequency options per gate. That search space is larger than the processor's own `2^N` state space. Exhaustive search is impossible, and general-purpose global optimizers are too slow even for processors with tens of qubits.

## How the Snake works

The Snake turns one huge optimization into many small ones:

1. **Map the processor onto a graph.** Qubits become nodes. Interactions become edges, either *engineered* (couplers used for two-qubit gates) or *parasitic* (crosstalk). Parasitic edges have no parameters of their own, but they carry constraints.
2. **Walk the graph.** Start from a seed element and move step by step to nearby uncalibrated elements. On a grid this step-by-step path looks like the arcade game *Snake*, which is where the name comes from.
3. **Optimize locally at each step.** At each step, build an error model over a small set of nearby uncalibrated parameters and optimize it. Already-calibrated neighbours act as constraints.
4. **Freeze the result.** Store the learned values and move on. Later steps must respect them.

Each step on its own is greedy. But each step is constrained by everything calibrated before it, so the process as a whole behaves like a much higher-dimensional optimization. It still returns a solution that satisfies every processor constraint.

### Key concepts

| Term | Meaning |
| --- | --- |
| **Processor graph** `G` | All relevant elements: nodes (usually qubits) and edges (interactions), with edge types such as engineered and parasitic. |
| **Calibration goal** `G*` | The subset of `G` whose parameters must be learned. For gate frequencies these are the nodes (single-qubit gate frequencies) and engineered edges (two-qubit gate frequency trajectories). |
| **Calibration status** `P*` | An ordered record of what has been calibrated so far. Calibration is finished when `P* = G*`. |
| **Connectivity subgraph** `X_g^d` | Every element within `d` edge steps of element `g`. It is used to define the scope of parameters, constraints and traversal. |
| **Algorithm subgraphs** `A_i` | The elements that are active *at the same time* at each moment of the target quantum algorithm. Only elements that can be active together constrain each other. For cross-entropy benchmarking (XEB) there are five such subgraphs. |
| **Calibration parameters** | The uncalibrated elements within `d_P` of the current element. They are optimized together in one step. |
| **Calibration constraints** | The already-calibrated, simultaneously active elements within `d_R` of those parameters. Constraints must be symmetric: if `g` constrains `g'`, then `g'` constrains `g`. |
| **Error model** | A function that maps a candidate assignment of the step's parameters to a performance metric (for example, estimated gate error). It is built from the parameters, the active constraints and calibration data. For gate frequencies it can include relaxation, dephasing, leakage and control errors. |
| **Calibration subgoals** | Regions of uncalibrated elements far enough apart that no element in one region constrains an element in another. They may share already-calibrated neighbours, and they can be calibrated in parallel. |
| **Traversal threads** | Pieces of a subgoal that the traversal rule cannot cross between. Each thread is seeded once and walked to completion. |

### The three distances

The Snake is controlled mainly by three distances:

| Distance | Controls |
| --- | --- |
| `d_P` (parameter distance) | How many parameters are optimized together in one step. This sets the size of each sub-problem. |
| `d_R` (constraint distance) | How far away calibrated elements can be and still constrain the current step. |
| `d_T` (traversal distance) | How far the Snake can move from one step to the next. |

`d_P` is the key trade-off between speed and solution quality:

- **`d_P = 0`**: each step optimizes one parameter over `k` options. Calibration is fast, but the number of reachable solutions drops sharply.
- **`d_P ≈ √N`**: one step covers the whole remaining goal. This is the full global optimization over `k^|G* \ P*|` options that the Snake exists to avoid.
- **In between**: each step optimizes between 1 and about `d_P²` parameters, and full calibration takes between about `|G* \ P*| / d_P²` and `|G* \ P*|` steps.

In the paper's gate-frequency example for XEB (`d_P = 1`, `d_R = 2`, `d_T = 2`), each step optimized 1 to 5 parameters under 0 to 8 active constraints.

### The algorithm

1. Split the uncalibrated part of the goal into **calibration subgoals** that cannot constrain each other, and process them in parallel.
2. Split each subgoal into **traversal threads**, and pick a seed element in each.
3. From the seed, **calibrate the element**: collect the parameters within `d_P` and the active constraints within `d_R`, gather calibration data, build the error model, optimize it, and add the results to `P*`.
4. Get the **traversal options**: uncalibrated elements within `d_T` that can be active together with the current element and are of the same type (node or edge). Sort them with the **traversal heuristic** (breadth-first, depth-first, random, …), then recurse into each one that is still uncalibrated.

### Local re-calibration and stitching

- **Re-calibration.** Calibrated values drift. When an element's performance falls past a threshold, its calibration is considered expired. The Snake removes that element's neighbourhood from `P*` and runs the same algorithm again, so only that local region is re-calibrated, still subject to all surrounding constraints.
- **Stitching.** Calibrated regions separated by uncalibrated gaps can be joined by running the same algorithm on the gaps. A large processor can therefore be split on purpose into constraint-disjoint regions, calibrated in parallel, and then stitched together.

## Planned API

FaultTune provides the Snake machinery: graphs, scopes, traversal, segmentation, parallelism and the parameter database. You provide the physics: the error model and how calibration data is collected.

### 1. Describe the processor

```rust
use faulttune::{ProcessorGraph, ElementKind};

// 16 qubits on a 4×4 grid: nearest-neighbour couplers are engineered edges,
// diagonals are parasitic edges that only carry crosstalk constraints.
let graph = ProcessorGraph::square_lattice(4, 4);

// Calibrate every node and every engineered edge.
let goal = graph.goal(|element| element.kind() != ElementKind::ParasiticEdge);
```

### 2. Supply an error model

```rust
use faulttune::{ElementId, ErrorModel, Step, Assignment};

struct GateFrequencyModel {
    // Measured device data: relaxation spectra, crosstalk strengths, ...
}

impl ErrorModel for GateFrequencyModel {
    type Value = f64; // frequency in GHz

    /// The `k` candidate values for one element.
    fn options(&self, element: ElementId) -> Vec<f64> {
        /* ... */
    }

    /// Collect any calibration data this step needs (optional).
    fn prepare(&mut self, step: &Step<f64>) {}

    /// Estimated error of a candidate assignment of the step's parameters,
    /// given the values already fixed for the constraining neighbours.
    fn error(&self, step: &Step<f64>, candidate: &Assignment<f64>) -> f64 {
        /* ... */
    }
}
```

### 3. Run the Snake

```rust
use faulttune::{Snake, SnakeConfig, AlgorithmSubgraphs, heuristics::BreadthFirst, optimizers::Exhaustive};

let config = SnakeConfig {
    parameter_distance: 1,  // d_P
    constraint_distance: 2, // d_R
    traversal_distance: 2,  // d_T
};

let mut snake = Snake::new(&graph, goal, config)
    .algorithm(AlgorithmSubgraphs::all_active(&graph)) // or the subgraphs of your algorithm
    .heuristic(BreadthFirst)
    .optimizer(Exhaustive);

let mut model = GateFrequencyModel { /* ... */ };
let mut params = snake.calibrate(&mut model)?; // a ParameterDatabase
```

### 4. Re-calibrate after drift

```rust
// Qubit 7 has drifted past its threshold: forget its neighbourhood and re-learn it.
snake.recalibrate(&mut params, qubit_7, &mut model)?;
```

### Pluggable parts

| Part | Provided by FaultTune | You can supply your own |
| --- | --- | --- |
| `ErrorModel` | Toy models for examples and tests | Yes, normally required |
| `Optimizer` (solves one step) | Exhaustive search, random search | Yes |
| `TraversalHeuristic` | Breadth-first, depth-first, random | Yes |
| `AlgorithmSubgraphs` | "Everything active at once" | Yes, from your algorithm's structure |
| `ProcessorGraph` | Square lattice builder | Yes, any graph with typed edges |

## Beyond quantum hardware

Nothing in the Snake is specific to qubits. The paper notes that the same ideas may apply to other optimization problems, such as playing games, financial trading or traffic routing. FaultTune is a good fit for any problem with these properties:

- The parameters sit on the nodes and edges of a graph.
- Interactions are mostly local.
- A global search over all parameters at once is too large.
- You need one good solution quickly, not a proof of the best one.

## Roadmap

- [ ] Processor graph with typed nodes and edges, and connectivity subgraphs
- [ ] Builders for calibration parameters, active constraints and traversal options
- [ ] Traversal heuristics: breadth-first, depth-first, random
- [ ] Segmentation into calibration subgoals and traversal threads
- [ ] Exhaustive and random-search step optimizers
- [ ] Parallel calibration of subgoals (via `rayon`, behind a feature flag)
- [ ] Local re-calibration and calibration stitching
- [ ] Worked example: a toy gate-frequency model on a square lattice
- [ ] Extensions suggested in the paper: caching error-model evaluations, re-optimizing the worst performers, parallel traversal threads, and reinforcement-learning-guided traversal

## References

- P. V. Klimov, J. Kelly, J. M. Martinis, H. Neven, *The Snake Optimizer for Learning Quantum Processor Control Parameters*, [arXiv:2006.04594](https://arxiv.org/abs/2006.04594) (2020).
- F. Arute et al., *Quantum supremacy using a programmable superconducting processor*, Nature 574, 505 (2019).

The paper cites a patent filing for the Snake Optimizer (P. V. Klimov, US16/434,513, filed 2019). Check its status before distributing or using FaultTune commercially.

## License

To be decided.
