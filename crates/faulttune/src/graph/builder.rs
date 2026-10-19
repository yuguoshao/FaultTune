//! [`GraphBuilder`]: build a [`ProcessorGraph`] of any topology.

use std::collections::HashMap;

use super::{EdgeType, ElementId, ElementKind, ProcessorGraph};
use crate::error::FaultTuneError;

/// Builds a [`ProcessorGraph`]. Elements get IDs in insertion order.
///
/// ```
/// use faulttune::{EdgeType, GraphBuilder};
/// let mut b = GraphBuilder::new();
/// let a = b.add_node("a").unwrap();
/// let c = b.add_node("c").unwrap();
/// let e = b.add_edge(a, c, EdgeType::Engineered).unwrap();
/// let graph = b.build().unwrap();
/// assert_eq!(graph.label(e), "a-c");
/// ```
#[derive(Debug, Default, Clone)]
pub struct GraphBuilder {
    kinds: Vec<ElementKind>,
    labels: Vec<String>,
    endpoints: Vec<Option<[ElementId; 2]>>,
    coords: Vec<Option<[f32; 2]>>,
    by_label: HashMap<String, ElementId>,
}

impl GraphBuilder {
    /// An empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    fn push(
        &mut self,
        kind: ElementKind,
        label: String,
        endpoints: Option<[ElementId; 2]>,
        coord: Option<[f32; 2]>,
    ) -> Result<ElementId, FaultTuneError> {
        if label.is_empty() {
            return Err(FaultTuneError::InvalidGraph("empty element label".into()));
        }
        if self.by_label.contains_key(&label) {
            return Err(FaultTuneError::InvalidGraph(format!(
                "duplicate element label {label:?}"
            )));
        }
        let id = ElementId(
            u32::try_from(self.kinds.len())
                .map_err(|_| FaultTuneError::InvalidGraph("more than u32::MAX elements".into()))?,
        );
        self.by_label.insert(label.clone(), id);
        self.kinds.push(kind);
        self.labels.push(label);
        self.endpoints.push(endpoints);
        self.coords.push(coord);
        Ok(id)
    }

    /// The element added with this label, if any.
    pub fn find_label(&self, label: &str) -> Option<ElementId> {
        self.by_label.get(label).copied()
    }

    /// Adds a node.
    pub fn add_node(&mut self, label: impl Into<String>) -> Result<ElementId, FaultTuneError> {
        self.push(ElementKind::Node, label.into(), None, None)
    }

    /// Adds a node with a layout coordinate `[x, y]` (used only for plotting).
    pub fn add_node_at(
        &mut self,
        label: impl Into<String>,
        coord: [f32; 2],
    ) -> Result<ElementId, FaultTuneError> {
        self.push(ElementKind::Node, label.into(), None, Some(coord))
    }

    /// Adds an edge between nodes `a` and `b`, labelled `"{a}-{b}"` after the node labels.
    pub fn add_edge(
        &mut self,
        a: ElementId,
        b: ElementId,
        edge_type: EdgeType,
    ) -> Result<ElementId, FaultTuneError> {
        let (la, lb) = (self.node_label(a)?, self.node_label(b)?);
        let label = format!("{la}-{lb}");
        self.add_edge_labelled(label, a, b, edge_type)
    }

    /// Adds an edge with an explicit label.
    pub fn add_edge_labelled(
        &mut self,
        label: impl Into<String>,
        a: ElementId,
        b: ElementId,
        edge_type: EdgeType,
    ) -> Result<ElementId, FaultTuneError> {
        self.node_label(a)?;
        self.node_label(b)?;
        if a == b {
            return Err(FaultTuneError::InvalidGraph(format!(
                "edge from node {a} to itself (self-loops are not allowed)"
            )));
        }
        self.push(
            ElementKind::Edge(edge_type),
            label.into(),
            Some([a, b]),
            None,
        )
    }

    fn node_label(&self, id: ElementId) -> Result<&str, FaultTuneError> {
        match (self.kinds.get(id.index()), self.labels.get(id.index())) {
            (Some(ElementKind::Node), Some(label)) => Ok(label),
            (Some(_), _) => Err(FaultTuneError::InvalidGraph(format!(
                "element {id} is not a node"
            ))),
            _ => Err(FaultTuneError::InvalidGraph(format!("unknown node {id}"))),
        }
    }

    /// Finishes the graph. Layout coordinates are kept only if every node has one.
    pub fn build(self) -> Result<ProcessorGraph, FaultTuneError> {
        let all_nodes_placed = self
            .kinds
            .iter()
            .zip(&self.coords)
            .all(|(kind, coord)| !kind.is_node() || coord.is_some());
        let has_nodes = self.kinds.iter().any(|k| k.is_node());
        let coords = (all_nodes_placed && has_nodes).then(|| {
            (0..self.kinds.len())
                .map(|i| match self.endpoints[i] {
                    None => self.coords[i].expect("node has a coordinate"),
                    Some([a, b]) => {
                        let (pa, pb) = (
                            self.coords[a.index()].expect("node has a coordinate"),
                            self.coords[b.index()].expect("node has a coordinate"),
                        );
                        [(pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0]
                    }
                })
                .collect::<Vec<_>>()
        });
        Ok(ProcessorGraph::from_parts(
            self.kinds,
            self.labels,
            self.endpoints,
            coords,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_input() {
        let mut b = GraphBuilder::new();
        let a = b.add_node("a").unwrap();
        assert!(b.add_node("a").is_err(), "duplicate label");
        assert!(b.add_node("").is_err(), "empty label");
        assert!(b.add_edge(a, a, EdgeType::Engineered).is_err(), "self loop");
        let c = b.add_node("c").unwrap();
        let e = b.add_edge(a, c, EdgeType::Engineered).unwrap();
        assert!(
            b.add_edge(a, e, EdgeType::Engineered).is_err(),
            "endpoint must be a node"
        );
        assert!(b
            .add_edge(a, ElementId::new(99), EdgeType::Engineered)
            .is_err());
        assert!(
            b.add_edge(a, c, EdgeType::Engineered).is_err(),
            "same label twice"
        );
        assert!(b
            .add_edge_labelled("a-c/p", a, c, EdgeType::Parasitic)
            .is_ok());
    }

    #[test]
    fn coords_need_every_node_placed() {
        let mut b = GraphBuilder::new();
        let a = b.add_node_at("a", [0.0, 0.0]).unwrap();
        let c = b.add_node_at("c", [2.0, 0.0]).unwrap();
        let e = b.add_edge(a, c, EdgeType::Engineered).unwrap();
        let g = b.build().unwrap();
        assert_eq!(g.coords().unwrap()[e.index()], [1.0, 0.0]);

        let mut b = GraphBuilder::new();
        b.add_node_at("a", [0.0, 0.0]).unwrap();
        b.add_node("c").unwrap();
        assert!(b.build().unwrap().coords().is_none());
    }
}
