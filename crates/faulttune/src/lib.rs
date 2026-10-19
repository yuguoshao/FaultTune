//! # FaultTune
//!
//! A Rust implementation of the **Snake Optimizer** (Klimov, Kelly, Martinis, Neven,
//! [arXiv:2006.04594](https://arxiv.org/abs/2006.04594)) for learning the control parameters
//! of a quantum processor, or of any system whose parameters sit on a graph and interfere
//! with their neighbours.
//!
//! FaultTune provides the Snake machinery: graphs, scopes, traversal, segmentation,
//! parallelism and the parameter database. You provide the physics: an [`ErrorModel`].
//!
//! See `README.md` for the algorithm and `docs/ARCHITECTURE.md` for the design.
//!
//! ```ignore
//! use faulttune::{heuristics::BreadthFirst, optimizers::CoordinateDescent};
//! use faulttune::{AlgorithmSubgraphs, ProcessorGraph, Snake, SnakeConfig};
//!
//! let graph = ProcessorGraph::square_lattice(4, 4);
//! let goal = graph.goal_excluding(&[faulttune::ElementKind::Edge(faulttune::EdgeType::Parasitic)]);
//! let snake = Snake::new(&graph, goal, SnakeConfig::default())
//!     .algorithm(AlgorithmSubgraphs::all_active(&graph))
//!     .heuristic(BreadthFirst)
//!     .optimizer(CoordinateDescent::default());
//! let params = snake.calibrate(&model)?;
//! ```

#![warn(missing_docs)]

pub mod activity;
pub mod cache;
pub mod database;
pub mod engine;
pub mod error;
pub mod events;
pub mod graph;
pub mod model;
pub mod neighbourhoods;
pub mod optimize;
#[cfg(feature = "serde")]
pub mod persist;
pub mod scope;
pub mod segment;
pub mod sets;
pub mod toy;
pub mod traverse;

pub use activity::{ActivityMasks, AlgorithmSubgraphs};
pub use cache::{CachePolicy, CacheStats, CrossStepCache, StepDedup, StepKey};
pub use database::{Entry, HistoryRecord, Overlay, ParameterDatabase, StepRecord};
pub use engine::{CancellationToken, ModelErrorPolicy, Snake, SnakeConfig};
pub use error::{FaultTuneError, ModelError};
pub use events::{Event, Observer, RecordingObserver, Report};
pub use graph::{EdgeType, Element, ElementId, ElementKind, GraphBuilder, ProcessorGraph};
pub use model::{CandidateBatch, ErrorModel, Step};
pub use neighbourhoods::Neighbourhoods;
pub use optimize::{IndexBatch, Objective, Optimizer, SearchSpace, Solution};
#[cfg(feature = "serde")]
pub use persist::{read_journal, read_journal_entries, CheckpointObserver, JournalEntry};
pub use scope::{Distances, Scope};
pub use segment::{SeedFn, SeedPolicy, StitchPlan, Subgoal, Thread};
pub use sets::ElementSet;
pub use traverse::{Frontier, FrontierKind, TraversalContext, TraversalHeuristic};

/// The `rand` crate this version of FaultTune is built against. Custom [`Optimizer`]s and
/// [`TraversalHeuristic`]s receive a [`rand::rngs::StdRng`]; use this re-export to name it without
/// adding a matching `rand` dependency.
pub use rand;

/// Built-in traversal heuristics.
pub mod heuristics {
    pub use crate::traverse::{BreadthFirst, DepthFirst, Random};
}

/// Built-in step optimizers.
pub mod optimizers {
    pub use crate::optimize::{CoordinateDescent, Exhaustive, RandomSearch};
}
