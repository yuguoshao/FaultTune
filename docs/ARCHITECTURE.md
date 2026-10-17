# FaultTune architecture

This document describes the planned architecture of FaultTune: a Rust implementation of the Snake Optimizer ([Klimov et al., arXiv:2006.04594](https://arxiv.org/abs/2006.04594)) with a Python interface. It is a design for code that has not been written yet. Read the [README](../README.md) first for the algorithm and its terminology.

## Contents

1. [Requirements](#1-requirements)
2. [System overview](#2-system-overview)
3. [Data model](#3-data-model)
4. [Calibration engine](#4-calibration-engine)
5. [Extension points](#5-extension-points)
6. [Concurrency and determinism](#6-concurrency-and-determinism)
7. [Caching](#7-caching)
8. [Error handling](#8-error-handling)
9. [Python bindings](#9-python-bindings)
10. [Persistence and checkpoints](#10-persistence-and-checkpoints)
11. [Observability](#11-observability)
12. [Performance and scale](#12-performance-and-scale)
13. [Testing strategy](#13-testing-strategy)
14. [Trade-offs](#14-trade-offs)
15. [What to revisit as FaultTune grows](#15-what-to-revisit-as-faulttune-grows)

### Notation

| Paper | FaultTune |
| --- | --- |
| Processor graph `G` | `ProcessorGraph` |
| Calibration goal `G*` | `goal: ElementSet` |
| Calibration status `P*` | the calibrated entries and history of a `ParameterDatabase` |
| Connectivity subgraph `X_g^d` | `Neighbourhoods::within(g, d)` |
| Algorithm subgraphs `A_i`, active elements `A_g` | `AlgorithmSubgraphs`, per-element activity mask |
| Distances `d_P`, `d_R`, `d_T` | `SnakeConfig::{parameter_distance, constraint_distance, traversal_distance}` |
| Calibration parameters `P_g^{d_P}` | `Step::parameters` |
| Active constraints `R_P^{d_R}` | `Step::constraints` |
| Calibration subgoal `G*_i` | `Subgoal` |
| Traversal thread `T_ij` | `Thread` |
| `build_calibration_data` | `ErrorModel::prepare` |
| `build_error_model` and `optimize_error_model` | `ErrorModel::error_batch`, called by an `Optimizer` |

---

## 1. Requirements

### 1.1 Functional requirements

| ID | Requirement |
| --- | --- |
| F1 | Represent a processor as a graph of nodes and typed edges (for example engineered and parasitic), with any topology. Provide builders for common layouts, starting with the square lattice. |
| F2 | Compute connectivity subgraphs `X_g^d` for any element and distance. |
| F3 | Represent the target algorithm as algorithm subgraphs, and decide whether two elements can be active at the same time. |
| F4 | Run the full Snake: segment into subgoals and threads, seed, traverse, and at each step build the parameters and constraints, collect data, optimize, and commit. |
| F5 | Let users supply the error model, step optimizer, traversal heuristic and seed policy. Ship sensible built-ins for all but the error model. |
| F6 | Calibrate subgoals in parallel. The result must be the same as a sequential run. |
| F7 | Re-calibrate the neighbourhood of drifted elements, stitch calibrated regions, and resume an interrupted run. |
| F8 | Store learned values together with the ordered calibration history, and save and load them. |
| F9 | Expose all of the above to Python, including error models written in Python with NumPy. |
| F10 | Report progress and per-step results while a calibration runs. |

### 1.2 Non-functional requirements

| Area | Requirement |
| --- | --- |
| Speed | Calibration must finish much faster than the hardware drifts. The engine's own work per step must not grow with processor size, so total engine time is linear in the number of elements. Model evaluation and measurement should dominate the run time. |
| Scale | Support processors from tens of qubits up to about 10,000 qubits on one machine. |
| Determinism | The same inputs and seed give the same parameters and history, whatever the thread count. |
| Reliability | An error, crash or interrupt never loses the steps already committed. |
| Usability | Python users need only `pip install`, with no Rust toolchain. Errors from models point to the model, not to engine internals. |
| Portability | Linux, macOS and Windows on x86-64 and ARM64. |

### 1.3 Assumptions and constraints

- Each element's search space is a finite list of options (`k` of them, about 100 in the paper). Continuous parameters are discretized by the error model.
- The error model is supplied by the user and may be slow. It may measure real hardware inside `prepare`.
- Interactions are mostly local, so connectivity subgraphs stay small (tens of elements) as the processor grows.
- At most 64 algorithm subgraphs per calibration (XEB needs 5). This limit is revisited in [§15](#15-what-to-revisit-as-faulttune-grows).
- The team is small, so the design favours a single Rust implementation wrapped by Python over two parallel implementations.

### 1.4 Non-goals

- Hardware drivers, pulse sequencing or measurement code. FaultTune calls the user's code for these.
- Physics models beyond the toy models used for examples and tests.
- The calibration stack itself (choosing which layer to run next). The paper treats this as a separate system (Optimus).
- A general-purpose global optimizer.

---

## 2. System overview

### 2.1 Components

```text
 ┌──────────────────────────── Python user code ────────────────────────────┐
 │  graph description · ErrorModel subclass (NumPy) · scripts / notebooks   │
 └───────────────────────────────────┬──────────────────────────────────────┘
                                     │ import faulttune
 ┌───────────────────────────────────▼──────────────────────────────────────┐
 │  python/faulttune      pure-Python helpers, ErrorModel base class, .pyi │
 │  crates/faulttune-py   PyO3 bindings: type wrappers, model adapter,      │
 │                        GIL handling, event pump, exception mapping       │
 └───────────────────────────────────┬──────────────────────────────────────┘
                                     │ Rust API
 ┌───────────────────────────────────▼──────────────────────────────────────┐
 │  crates/faulttune  (core)                                                │
 │                                                                          │
 │   graph ─► neighbourhoods ─► scope ─► segment ─► traverse ─┐             │
 │     │            activity ──────┘                          ▼             │
 │     │                                                   engine ─► events │
 │     └──────────────► database ◄────────────────────────┘   │             │
 │                         │                    model · optimize · cache    │
 │                         ▼                                                │
 │                     persistence (JSON, journal)                          │
 └──────────────────────────────────────────────────────────────────────────┘
```

The core crate holds all of the algorithm. The Python layer converts types and runs callbacks; it contains no calibration logic of its own.

### 2.2 Repository layout

```text
FaultTune/
├── Cargo.toml                 # workspace
├── pyproject.toml             # maturin build for the Python package
├── crates/
│   ├── faulttune/             # core library
│   │   └── src/
│   │       ├── lib.rs         # public re-exports
│   │       ├── graph/         # ProcessorGraph, ElementId, ElementKind, builders
│   │       ├── sets.rs        # ElementSet (bitset)
│   │       ├── neighbourhoods.rs
│   │       ├── activity.rs    # AlgorithmSubgraphs, activity masks
│   │       ├── scope.rs       # parameters, constraints, traversal options
│   │       ├── segment.rs     # subgoals, threads, seeds
│   │       ├── traverse.rs    # frontiers and heuristics
│   │       ├── model.rs       # ErrorModel, Step
│   │       ├── optimize/      # Optimizer, SearchSpace, built-in optimizers
│   │       ├── cache.rs
│   │       ├── database.rs    # ParameterDatabase, history, overlays
│   │       ├── persist.rs     # JSON format, journal
│   │       ├── engine.rs      # Snake, SnakeConfig, calibration loop
│   │       ├── events.rs      # Observer, Event, Report
│   │       └── error.rs
│   └── faulttune-py/          # PyO3 bindings
├── python/faulttune/          # __init__.py, _model.py, *.pyi, py.typed
├── examples/                  # Rust and Python examples, notebook
├── benches/                   # criterion benchmarks
└── docs/
```

### 2.3 Module rules

- `graph`, `sets`, `neighbourhoods`, `activity`, `scope` and `segment` are pure and synchronous: they take data and return data. Most of the paper's definitions live here, so they can be unit-tested directly against the paper's figures.
- Only `engine` knows about threads, models, optimizers and observers.
- Optional dependencies sit behind Cargo features: `parallel` (rayon, on by default), `serde` (on by default), `tracing`.

---

## 3. Data model

### 3.1 Elements and the processor graph

Every node and edge is an **element** with a dense integer ID, so per-element data can live in plain vectors instead of hash maps.

```rust
pub struct ElementId(u32);

pub enum ElementKind {
    Node,
    Edge(EdgeType), // EdgeType::Engineered, EdgeType::Parasitic, or a user-defined type
}

pub struct ProcessorGraph {
    kinds: Vec<ElementKind>,                // indexed by ElementId
    labels: Vec<String>,                    // "q1_3", "q1_3-q1_4", ...
    by_label: HashMap<String, ElementId>,
    endpoints: Vec<Option<[ElementId; 2]>>, // for edges
    incidence: Csr<ElementId>,              // node → edges, edge → endpoints
    coords: Option<Vec<[f32; 2]>>,          // optional layout, for plots
}
```

Graphs are immutable once built. A `GraphBuilder` adds nodes and edges, and `square_lattice(rows, cols)` creates nearest-neighbour engineered edges plus diagonal parasitic edges. Graphs can also be loaded from a JSON description, which is convenient from Python.

Each graph has a **fingerprint**: a hash of its labels, kinds and incidence. Saved calibrations record it, so loading a calibration onto the wrong device fails loudly.

### 3.2 Distance and connectivity subgraphs

The paper defines `X_g^d` as the elements "within `d` edge-traversals" of `g`. FaultTune makes this precise using the **incidence graph**, whose vertices are all elements and which links each node to its incident edges:

- `X_g^d` = the elements within incidence distance `r(g, d)` of `g`, where
- `r = 2d` if `g` is an edge, and `r = max(2d − 1, 0)` if `g` is a node.

In words, `d` counts graph edges along a walk, and an edge element counts as already standing on its own edge. This definition reproduces the paper's figures:

| Paper figure | Check |
| --- | --- |
| Fig. 2 | `X_n^1` is a node and its incident edges; `X_n^2` is the 3×3 block of nodes around it with all their incident edges; `X_e^0` is the edge alone. |
| Fig. 4 (`d_T = 2`, XEB) | A node has 8 traversal options (4 nearest and 4 diagonal neighbours). A horizontal edge has 6 (2 above, 2 below and 2 in line with it). |
| Fig. 7 (`d_P = 1`) | A node-centred step optimizes 1 to 5 parameters: the node and up to 4 incident engineered edges. |

The relation "`a` is within `d` of `b`" is symmetric for every pair of element kinds. The paper requires this, and segmentation relies on it.

### 3.3 Neighbourhood index

`Neighbourhoods` is built once per graph and configuration. For every element it stores all elements up to the largest radius needed, sorted by incidence distance, in one compressed array:

```text
offsets[g] ─► [(g,0) (e1,1) (e2,1) … (n1,2) (n2,2) … (e9,3) …]
radius_end[g][r] ─► end of the prefix with distance ≤ r
```

`within(g, d)` is therefore a slice, with no search at run time. This is what keeps the engine's work per step independent of processor size.

### 3.4 Element sets and activity

- `ElementSet` is a fixed-size bitset over element IDs. It is used for `G*`, the uncalibrated set, subgoal membership and calibration status. Membership tests are O(1), and steps never iterate over a whole set; they iterate over a neighbourhood slice and test membership.
- `AlgorithmSubgraphs` stores each `A_i` as an `ElementSet`. From these the engine builds one 64-bit **activity mask** per element (bit `i` is set if the element is in `A_i`). Two elements can be active at the same time exactly when their masks share a bit. The default `all_active` is a single subgraph `A_0 = G*`, as in the paper.

### 3.5 Values and search spaces

Optimizers never see values, only **option indices**. For each element the model's `options(element)` gives a list of `k` values. The engine caches it in an `OptionTable`, and a candidate for a step is a vector of indices, one per parameter. This keeps optimizers generic over the value type and makes candidates cheap to hash and compare.

In Rust the value type is generic (`ErrorModel::Value`). In Python it is `f64` (see [§9.2](#92-type-mapping)).

### 3.6 Parameter database

```rust
pub struct ParameterDatabase<V> {
    graph_fingerprint: u64,
    entries: Vec<Option<Entry<V>>>, // current values, indexed by ElementId
    history: Vec<HistoryRecord>,     // append-only
}

pub struct Entry<V> {
    value: V,
    option_index: u32,
    step: u64, // the step that set it
}

pub enum HistoryRecord {
    Step(StepRecord),           // centre, parameters, constraints, error, evaluations, duration
    Invalidated(Vec<ElementId>), // removed by re-calibration
}
```

`P*` is the set of elements with an entry, ordered by the step that set them. History is append-only, so it is also an audit trail: re-calibration adds an `Invalidated` record instead of rewriting the past.

---

## 4. Calibration engine

### 4.1 Top-level loop

```text
calibrate_into(db, goal, region = all):
    failed ← ∅
    loop:
        U ← (goal ∩ region) ∖ calibrated(db) ∖ failed
        if U = ∅: return report
        subgoals ← segment(U)                        # §4.2
        snapshot ← read-only view of db
        results ← for each subgoal S, in parallel:   # §6
            overlay ← empty set of changes on top of snapshot
            for each thread T of S, nodes first:
                seed ← seed_policy(T)
                if seed is calibrated in overlay: continue
                traverse(seed, S, overlay)           # §4.3
            return overlay
        merge results into db, in subgoal order
        if this pass neither calibrated nor skipped any element: fail with NoProgress
```

The outer loop is a **completion loop**. A single pass of the paper's algorithm can leave elements uncalibrated. When a step calibrates an element as a *parameter* of another centre, traversal never expands from it, so elements reachable only through it can be missed. The completion loop re-segments whatever is left and runs again. Every pass calibrates at least the seed of its first thread, so the loop always ends.

### 4.2 Segmentation

**Subgoals.** Two uncalibrated elements are linked if one is within `max(d_P, d_R)` of the other. Subgoals are the connected components of this relation, ordered by their smallest element ID.

This guarantees that elements in different subgoals are more than `max(d_P, d_R)` apart. Therefore:

1. A step in one subgoal never takes a parameter from another, because parameters are within `d_P` of the step's centre. The engine also intersects parameters with the subgoal as a safety net.
2. A value committed in one subgoal never constrains a step in another, because constraints are within `d_R` of a parameter.
3. Elements calibrated before the pass are shared by all subgoals, but they are read-only during the pass.

So no subgoal ever reads another's writes, and running subgoals in parallel gives exactly the same result as running them one after another.

Segmentation deliberately ignores activity masks. Using them would allow more, smaller subgoals, but it would make the correctness argument depend on each user's algorithm subgraphs. See [§14](#14-trade-offs).

**Threads.** Within a subgoal, two elements are linked if they are within `d_T`, can be active at the same time, and have the same kind. Threads are the connected components. Node threads run before edge threads, so with `d_P ≥ 1` the edges are usually calibrated as parameters of node steps, and their own threads have nothing left to do. This matches Fig. 7, where one node seed calibrates the whole graph.

**Seeds.** The default `SeedPolicy::LowestId` is deterministic. `Fixed(labels)` lets users start from a known-good qubit, and `Random` uses the run's seeded RNG.

### 4.3 Traversal

The paper describes traversal as a recursive function. FaultTune uses an explicit **frontier** (a work list) instead:

```text
traverse(seed, S, overlay):
    frontier.push(seed)
    while g ← frontier.pop():
        if g is calibrated: continue
        step(g, S, overlay)                                   # §4.4
        options ← within(g, d_T) ∩ S, uncalibrated,
                  active at the same time as g, same kind as g
        frontier.push_all(heuristic.order(options))
```

A last-in-first-out frontier visits elements in the same order as the paper's recursion. A first-in-first-out frontier gives true breadth-first traversal, which plain recursion cannot do, and a priority frontier supports scored heuristics. The work list also avoids stack overflow on large processors, where recursion depth could reach thousands of elements.

### 4.4 One calibration step

```text
 centre g
    │
    ├─► parameters   P = within(g, d_P) ∩ S, uncalibrated              (scope.rs)
    ├─► constraints  R = calibrated elements c such that, for some p in P,
    │                    c is within(p, d_R) and c, p can be active together
    ▼
 Step { index, centre, parameters, constraints (with values), graph }
    │
    ├─► model.prepare(&step) ──────────────────────────► data
    ▼
 SearchSpace { k_1 × … × k_|P| option indices, evaluation budget }
    │
    ├─► optimizer.optimize(space, objective)
    │        objective(batch of index vectors)
    │            └─► within-step dedupe ─► cross-step cache (opt-in) ─►
    │                model.error_batch(step, data, values) ─► errors
    ▼
 Solution { best indices, error, evaluations }
    │
    ▼
 overlay.commit(P, values, StepRecord) ─► observer: StepCommitted
```

Constraints reach the model as `(element, value)` pairs. The model never has to look values up in the database. A step's work is proportional to the size of its neighbourhoods, plus the cost of the model.

### 4.5 Re-calibration, stitching and resuming

All three use the same building block: run the loop from a partly filled database.

- **Re-calibration.** `recalibrate(db, elements, model)` removes `within(g, expiry_radius)` around each expired element from `P*` (the paper uses radius 1). It records an `Invalidated` entry and then runs `calibrate_into` restricted to the removed region. Constraints from the untouched surroundings apply as usual. Deciding *when* an element has expired is left to the user's monitoring.
- **Stitching.** Calibrating from scratch gives a single subgoal, because the whole uncalibrated processor is connected. To get parallelism, a `StitchPlan` takes user-chosen regions (for example quadrants) and computes a gap between them that is wide enough to separate them by more than `max(d_P, d_R)`. Stage one calibrates the regions minus the gap; they become separate subgoals and run in parallel. Stage two calibrates the gap, joining the regions under constraints from both sides.
- **Resuming.** A run interrupted by an error, crash or Ctrl-C keeps every committed step (see [§10](#10-persistence-and-checkpoints)). Calling `calibrate_into` again continues with whatever is still uncalibrated.

---

## 5. Extension points

These traits are the public contract. Everything else in the engine is internal.

### 5.1 Error model

```rust
pub trait ErrorModel: Sync {
    type Value: Clone + Send + Sync;
    /// Per-step calibration data returned by `prepare`. Use `()` if there is none.
    type Data: Send;

    /// The candidate values for one element. Called once per element per run.
    fn options(&self, element: ElementId) -> Vec<Self::Value>;

    /// Collect calibration data for a step, for example by measuring hardware.
    fn prepare(&self, step: &Step<'_, Self::Value>) -> Result<Self::Data, ModelError>;

    /// Estimated error of one candidate. Lower is better; +∞ marks a forbidden candidate.
    fn error(&self, step: &Step<'_, Self::Value>, data: &Self::Data, candidate: &[Self::Value]) -> f64;

    /// Estimated errors of many candidates. Override for speed; the default calls `error`.
    fn error_batch(
        &self,
        step: &Step<'_, Self::Value>,
        data: &Self::Data,
        candidates: &CandidateBatch<'_, Self::Value>,
        out: &mut [f64],
    ) -> Result<(), ModelError> { /* loops over `error` */ }
}

pub struct Step<'a, V> {
    pub index: u64,
    pub centre: ElementId,
    pub parameters: &'a [ElementId],     // candidate columns, in this order
    pub constraints: &'a [(ElementId, V)],
    pub graph: &'a ProcessorGraph,
}
```

All methods take `&self`, so one model can serve parallel subgoals. Per-step state travels in `Data` instead of being stored in the model. A model that drives hardware which cannot be shared can serialize access with its own lock. Returning `+∞` is how a model expresses hard constraints such as forbidden frequency collisions.

### 5.2 Optimizer

```rust
pub trait Optimizer: Send + Sync {
    fn optimize(
        &self,
        space: &SearchSpace,           // option count per parameter, evaluation budget
        objective: &mut dyn Objective, // evaluates batches of index vectors
        rng: &mut StdRng,
    ) -> Result<Solution, FaultTuneError>;
}
```

| Built-in | Behaviour | Use when |
| --- | --- | --- |
| `Exhaustive` | Tries every combination, in batches. Refuses with `SearchSpaceTooLarge` if `∏ k_i` exceeds the budget (default 10⁶). | `d_P = 0`, or small `k` |
| `CoordinateDescent` | Starts from the best of a few random points, then repeatedly sweeps one parameter at a time over all of its options, as one batch per parameter. Stops when a sweep makes no improvement. | Default. With `d_P = 1` and `k = 100`, a step has up to 100⁵ = 10¹⁰ combinations, far too many to search exhaustively. |
| `RandomSearch` | Samples a fixed number of candidates. | Baselines and tests |

### 5.3 Traversal heuristic and seed policy

```rust
pub trait TraversalHeuristic: Send + Sync {
    fn frontier(&self) -> FrontierKind; // Lifo, Fifo or Priority
    fn order(&self, ctx: &TraversalContext, options: &mut Vec<ElementId>, rng: &mut StdRng);
}
```

`TraversalContext` gives read access to the graph, the current calibration status and the step history. That is enough for heuristics such as "most-constrained first", and later for a learned policy (see [§15](#15-what-to-revisit-as-faulttune-grows)). The built-ins are `BreadthFirst`, `DepthFirst` and `Random`.

### 5.4 Observer

```rust
pub trait Observer: Send + Sync {
    fn on_event(&self, event: &Event);
}

pub enum Event {
    PassStarted { pass: u32, subgoals: usize, uncalibrated: usize },
    StepCommitted(StepRecord), // includes the committed values
    StepFailed { centre: ElementId, error: String },
    PassFinished { pass: u32, calibrated: usize },
    Finished(Report),
}
```

### 5.5 Configuration

```rust
pub struct SnakeConfig {
    pub parameter_distance: u32,  // d_P
    pub constraint_distance: u32, // d_R
    pub traversal_distance: u32,  // d_T
    pub expiry_radius: u32,       // used by recalibrate; default 1
    pub seed: u64,                // for all randomness
    pub threads: Option<usize>,   // worker threads; default all cores
    pub on_model_error: ModelErrorPolicy,
    pub cache: CachePolicy,
}
```

---

## 6. Concurrency and determinism

- **Parallel unit.** The unit of parallel work is the subgoal, run on a rayon pool. Threads within a subgoal run one after another, because they can constrain each other.
- **No locks on the hot path.** Each pass takes a read-only snapshot of the database. Each subgoal writes only to its own overlay, and the overlays are merged after the pass. The segmentation guarantee in [§4.2](#42-segmentation) means overlays never touch the same elements and never read each other.
- **Deterministic randomness.** Every subgoal gets its own RNG, seeded from `(config.seed, pass, smallest element ID in the subgoal)`. Results therefore don't depend on thread scheduling or core count.
- **Deterministic history.** Overlays are merged in subgoal order, so the history order is stable too.
- **Cancellation.** A shared `CancellationToken` is checked between steps. A cancelled run finishes the steps in progress, merges what it has, and returns `Cancelled` with the partial report. Nothing committed is lost.

---

## 7. Caching

| Cache | Scope | Safe because | Default |
| --- | --- | --- | --- |
| Option table | One run | `options` depends only on the element | Always on |
| Within-step dedupe | One step | The same candidate within one step, with the same data, has the same error. Coordinate descent revisits its current point every sweep. | Always on |
| Cross-step cache | One run, or saved between runs | Only correct if the model's error is a pure function of the parameters, their values and the constraint values. It is wrong if `prepare` measures fresh data. | Off. Turned on with `CachePolicy::AcrossSteps`, which is the user's assertion that the model is pure. |

The cross-step cache is a bounded LRU map. Its key is the parameter IDs, the candidate's option indices, and the constraint IDs and option indices. This is the "cache error model evaluations" extension the paper suggests. It mainly helps when trying several traversal settings on the same model.

---

## 8. Error handling

One error type, `FaultTuneError`, covers the library:

| Variant | Cause |
| --- | --- |
| `InvalidGraph`, `InvalidConfig` | Bad input, detected before any step runs |
| `SearchSpaceTooLarge { size, budget }` | `Exhaustive` was asked to search too much |
| `Model { element, source }` | The error model failed; `source` holds the model's own error |
| `NonFiniteError { element }` | The model returned NaN |
| `NoProgress` | A pass made no progress (only possible with a custom seed policy that returns elements outside its thread) |
| `GraphMismatch` | Loaded a calibration made for a different graph |
| `Cancelled` | The run was cancelled; the database keeps every committed step |
| `Io`, `Format` | Saving or loading failed |

Policies:

- **Model errors** follow `ModelErrorPolicy`: `Abort` (the default), `Retry { times }` for flaky hardware measurements, or `SkipElement`. Skipped elements are excluded for the rest of the run and listed in the report.
- **NaN is an error; +∞ is allowed.** NaN almost always means a bug in the model, so it fails loudly. +∞ means "forbidden".
- **Progress is never lost.** `calibrate_into` writes into the caller's database. When it returns an error, every step committed before the error is still there.

---

## 9. Python bindings

### 9.1 Layers

```text
 Python caller (main thread)
   │  snake.calibrate(model, on_step=...)
   ▼
 faulttune-py
   ├─ starts the engine on Rust worker threads, with the GIL released
   ├─ main thread: event pump, waking every ~100 ms
   │     · checks for Ctrl-C (Python only delivers signals to the main thread)
   │     · calls on_step(...) for queued StepCommitted events
   ├─ PyErrorModel adapter: takes the GIL for each call into the Python model
   ▼
 faulttune core
```

The engine runs on worker threads with the GIL released, so other Python threads keep running. The main thread runs a small **event pump**. This solves two problems:

- Python only delivers Ctrl-C to the main thread. The pump checks for signals and turns Ctrl-C into a cancellation.
- Progress callbacks run on the main thread, so tools such as `tqdm` or notebook widgets work normally.

### 9.2 Type mapping

| Rust | Python |
| --- | --- |
| `ProcessorGraph`, `GraphBuilder` | `faulttune.ProcessorGraph` (with `square_lattice`, `from_json`, `node`, `edge`, `goal`) |
| `ElementId` | `faulttune.Element`: a hashable handle with `.label` and `.kind` |
| `ElementKind`, `EdgeType` | `faulttune.ElementKind` enum |
| `ErrorModel<Value = f64>` | subclass of `faulttune.ErrorModel` |
| `Step` | `faulttune.Step`: read-only, with `.centre`, `.parameters`, `.constraints` (a dict) and `.data` |
| `Snake`, `SnakeConfig` | `faulttune.Snake(graph, goal, **config)` |
| `ParameterDatabase<f64>` | `faulttune.ParameterDatabase` (indexing, `to_dict`, `history`, `save`, `load`) |
| `Report`, `StepRecord` | `faulttune.Report`, `faulttune.StepRecord` |
| `FaultTuneError` variants | `faulttune.FaultTuneError` and subclasses |

Values are `float` in Python. A model whose options are structured (for example a two-qubit gate trajectory described by several numbers) can return option indices `0..k` from `options` and interpret them itself.

### 9.3 Error model adapter

`PyErrorModel` implements `ErrorModel<Value = f64, Data = Option<PyObject>>`:

- `options(element)` calls the Python method once per element and converts the result (any array-like) to a list of floats.
- `prepare(step)` calls the Python method if it is overridden. Whatever it returns becomes `step.data` in later `error` calls.
- `error_batch` builds a `float64` NumPy array of shape `(n_candidates, n_parameters)` from Rust memory, calls `error(step, candidates)` once for the whole batch, and checks that the result has shape `(n_candidates,)`.

One Python call per batch, rather than per candidate, is what keeps Python models fast. Coordinate descent sends batches of `k` candidates, so a step costs about `sweeps × |P|` Python calls instead of thousands.

Calls into Python take the GIL, so Python models serialize across parallel subgoals. NumPy releases the GIL inside large array operations, so heavy NumPy models still get some overlap. Free-threaded Python is discussed in [§15](#15-what-to-revisit-as-faulttune-grows).

### 9.4 Exceptions and interrupts

- A Python exception raised by the model becomes `FaultTuneError::Model`. It is re-raised as `faulttune.ModelError`, chained to the original exception (`raise ... from`), so the traceback points into the user's model.
- Ctrl-C cancels the run as described in [§6](#6-concurrency-and-determinism), then raises `KeyboardInterrupt`. If the caller passed a `database=`, it holds every committed step. Otherwise the partial database is attached to the exception as `.partial`.
- Custom heuristics and seed policies can be Python callables. They are called once per step, so their cost is small.

### 9.5 Packaging

- Built with maturin from `crates/faulttune-py`. The pure-Python package lives in `python/faulttune`.
- Uses the stable ABI (`abi3`, Python 3.11 and newer), so one wheel per platform covers every supported Python version.
- CI builds wheels for manylinux (x86-64 and ARM64), macOS (x86-64 and ARM64) and Windows (x86-64), plus an sdist, and publishes them to PyPI on tagged releases.
- Type stubs (`.pyi`) ship with the package and are checked against the compiled module in CI with mypy's `stubtest`.

---

## 10. Persistence and checkpoints

**Saved calibrations** are JSON, keyed by element label so they are readable and diff well:

```json
{
  "format": "faulttune.parameters/1",
  "graph": { "fingerprint": "9f2c…", "elements": 214 },
  "values": {
    "q1_3":      { "value": 6.12, "option": 56, "step": 17 },
    "q1_3-q1_4": { "value": 5.87, "option": 43, "step": 17 }
  },
  "history": [
    { "type": "step", "index": 17, "centre": "q1_3",
      "parameters": ["q1_3", "q1_3-q1_4"], "constraints": ["q0_3", "q1_2"],
      "error": 0.0031, "evaluations": 1200, "duration_ms": 4 }
  ]
}
```

- The `format` field is versioned. Newer versions of FaultTune read older formats.
- Loading checks the graph fingerprint and fails with `GraphMismatch` on a different device.

**Checkpoints** are an append-only journal, written by a built-in `CheckpointObserver`. Each `StepCommitted` event becomes one JSON line. Each line is flushed as it is written, and a torn last line is ignored on replay, so a crash loses at most the step being written. To resume, load the last saved database, replay the journal, and call `calibrate_into` again. This matters most when `prepare` measures hardware and a full run takes hours.

---

## 11. Observability

- **Events.** `Observer` events (see [§5.4](#54-observer)) drive progress bars, checkpoints and custom logging.
- **Tracing.** With the `tracing` feature, the engine emits spans for `calibrate`, `pass`, `subgoal` and `step`. They carry fields such as the centre label, parameter count, evaluations and error, so a slow step can be found with any `tracing` subscriber.
- **Report.** `Report` summarizes a run: passes, steps, total evaluations, cache hit rate, wall time, time spent in `prepare` compared with `error`, skipped elements, and the worst steps by error. The worst steps are the natural input to the paper's "re-optimize the worst performers" extension, via `recalibrate`.

---

## 12. Performance and scale

### 12.1 Size estimates (square lattice with diagonal crosstalk edges)

| Qubits `N` | Elements in `G` (≈5N) | Calibrated parameters in `G*` (≈3N) | Neighbourhood index |
| --- | --- | --- | --- |
| 53 | ≈ 265 | ≈ 160 | ≈ 0.1 MB |
| 1,000 | ≈ 5,000 | ≈ 3,000 | ≈ 2 MB |
| 10,000 | ≈ 50,000 | ≈ 30,000 | ≈ 25 MB |

### 12.2 Evaluation counts

With `d_P = 1`, `k = 100` and `CoordinateDescent` doing up to 3 sweeps over at most 5 parameters, a step needs at most about 3 × 5 × 100 = **1,500 evaluations**, plus a few for the random starting points. There are at most `|G*|` steps (usually close to `N`, since node steps absorb their edges). So:

| Qubits | Evaluations (upper bound) | At 1 µs each (Rust model) | At 10 µs each |
| --- | --- | --- | --- |
| 53 | ≈ 0.24 million | ≈ 0.2 s | ≈ 2 s |
| 1,000 | ≈ 4.5 million | ≈ 5 s | ≈ 45 s |
| 10,000 | ≈ 45 million | ≈ 45 s | ≈ 8 min |

These are single-core figures before parallelism and caching. They will be replaced by benchmark results. The engine's own overhead per step is a few neighbourhood slices and bitset tests, which is microseconds and negligible next to the model.

### 12.3 Where the time really goes

- **Python models.** About 15 Python calls per step with batching, so call overhead is small. The NumPy work itself dominates.
- **Hardware in the loop.** If `prepare` measures hardware for several seconds per step, a 1,000-qubit run of about 1,000 steps takes hours on one subgoal. This is why stitching ([§4.5](#45-re-calibration-stitching-and-resuming)) is a core feature: with `m` stitched regions, stage one runs about `m` times faster. The checkpoint journal makes these long runs safe to interrupt.
- **Parallel speed-up** is limited by the number of subgoals. Calibration from scratch gives one subgoal unless a `StitchPlan` is used. Re-calibrating scattered drifted elements naturally gives many small subgoals.

---

## 13. Testing strategy

| Level | What is tested |
| --- | --- |
| Paper figures | Unit tests for `scope` and `segment` that rebuild the paper's examples: `X_g^d` shapes (Fig. 2), the XEB algorithm subgraphs (Fig. 3), 8 node and 6 edge traversal options (Fig. 4), and steps of 1–5 parameters with 0–8 constraints (Fig. 7). |
| Properties | `proptest` on random graphs and calibration states: subgoals partition the uncalibrated set; elements in different subgoals are more than `max(d_P, d_R)` apart; threads partition subgoals; every run ends with `P* = G*` or a reported failure; parallel and sequential runs give identical databases; the same seed gives the same result. |
| Known optima | Toy models where the best answer is known. With a separable model the Snake must find the global optimum for every `d_P`. A frequency-collision model must never produce a collision when a collision-free assignment exists within the options. |
| Recovery | Inject model errors, NaN and cancellation mid-run; check that committed steps survive and that resuming completes the calibration. Kill a run and resume from the journal. |
| Python | `pytest` for the bindings: NumPy batch shapes, exception chaining, Ctrl-C handling, callbacks on the main thread, save/load round trips. `stubtest` for the type stubs. |
| Benchmarks | `criterion` benchmarks on lattices from 10×10 to 100×100, tracking engine overhead per step and total time with a cheap Rust model. |

---

## 14. Trade-offs

| Decision | Alternative | Why this choice | Cost |
| --- | --- | --- | --- |
| Rust core with Python bindings | Pure Python | One implementation, fast traversal, real parallelism, and Python ease of use on top | A harder build (maturin, wheels per platform) and a higher bar for contributors |
| Distance measured on the incidence graph | Distance on nodes only, or on the line graph | Reproduces the paper's figures exactly, and treats nodes and edges uniformly | The formula is less obvious; it is documented in [§3.2](#32-distance-and-connectivity-subgraphs) and tested against the figures |
| Segmentation ignores activity masks | Use co-activity to split subgoals more finely | Correctness does not depend on user-supplied subgraphs | Fewer, larger subgoals, so less parallelism |
| Explicit frontier | Recursion, as in the paper | Real breadth-first and priority orders, and no stack overflow | Differs from the paper's pseudocode; the LIFO frontier reproduces the paper's order |
| Completion loop | A single pass | Guarantees every element is calibrated | Extra passes on some graphs, each with segmentation work |
| Search over option indices | Optimizers working on values | Optimizers are generic, and candidates are cheap to hash and cache | Continuous optimizers need the model to discretize |
| Batch error evaluation | One call per candidate | Makes Python models practical | A slightly larger model API (Rust gets a default batch method) |
| Snapshot and overlays | A shared database behind a lock | No locking, and deterministic results | Relies on the segmentation guarantee, which property tests enforce |
| Cross-step cache off by default | Always cache | A stale cache would silently give wrong calibrations when the model measures fresh data | Users must opt in to get the speed-up |
| `f64` values in Python | Arbitrary Python objects | Fast NumPy batches and simple JSON | Structured options need an index-based model |

---

## 15. What to revisit as FaultTune grows

- **Hardware-in-the-loop throughput.** Let `prepare` be asynchronous, and batch simultaneous measurements across subgoals in a pass, since real hardware can often measure distant regions at once.
- **Free-threaded Python.** Once free-threaded CPython is common, ship per-version free-threaded wheels so Python models run truly in parallel. This means giving up the single `abi3` wheel for those builds.
- **Distributed calibration.** Subgoals are natural units of work to send to other machines for very large processors or expensive models.
- **Learned traversal.** The paper suggests reinforcement learning to guide traversal. The heuristic trait already exposes the state such a policy needs, but training will likely need a richer, replayable environment interface.
- **More precise segmentation.** Use activity masks to split subgoals more finely, once property tests can cover user-supplied subgraphs.
- **More than 64 algorithm subgraphs.** Replace the `u64` activity mask with a small bitset.
- **Continuous and model-aware optimizers**, for example optimizers that use gradients when the model provides them.
- **Other topologies.** Add builders for layouts beyond the square lattice, and check that neighbourhoods stay small on them.

### Open questions

- The paper defines connectivity in words; the incidence-graph formula is inferred from its figures. It should be confirmed against the original implementation or the authors if possible.
- Should hard constraints get their own API instead of being expressed as `+∞` errors? A separate API would let optimizers skip forbidden candidates without evaluating them.
- Which default `CoordinateDescent` settings (random starts, maximum sweeps) work well on realistic models? This needs benchmarks on a physics-based example.
