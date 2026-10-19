//! Square-lattice builder (see [`ProcessorGraph::square_lattice`]).

use super::{EdgeType, ElementId, ProcessorGraph};
use crate::graph::GraphBuilder;

pub(super) fn square_lattice(rows: usize, cols: usize) -> ProcessorGraph {
    let mut b = GraphBuilder::new();
    let node = |r: usize, c: usize| ElementId::new((r * cols + c) as u32);
    for r in 0..rows {
        for c in 0..cols {
            b.add_node_at(format!("q{r}_{c}"), [c as f32, r as f32])
                .expect("lattice labels are unique");
        }
    }
    // Engineered edges: right neighbour, then lower neighbour, for each node in row-major order.
    for r in 0..rows {
        for c in 0..cols {
            if c + 1 < cols {
                b.add_edge(node(r, c), node(r, c + 1), EdgeType::Engineered)
                    .expect("lattice edges are valid");
            }
            if r + 1 < rows {
                b.add_edge(node(r, c), node(r + 1, c), EdgeType::Engineered)
                    .expect("lattice edges are valid");
            }
        }
    }
    // Parasitic diagonals: lower-right neighbour, then lower-left neighbour.
    for r in 0..rows {
        for c in 0..cols {
            if r + 1 < rows && c + 1 < cols {
                b.add_edge(node(r, c), node(r + 1, c + 1), EdgeType::Parasitic)
                    .expect("lattice edges are valid");
            }
            if r + 1 < rows && c >= 1 {
                b.add_edge(node(r, c), node(r + 1, c - 1), EdgeType::Parasitic)
                    .expect("lattice edges are valid");
            }
        }
    }
    b.build().expect("a lattice is a valid graph")
}
