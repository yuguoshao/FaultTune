//! JSON description of a graph (feature `serde`).
//!
//! ```json
//! { "format": "faulttune.graph/1",
//!   "elements": [
//!     { "label": "q0", "kind": "node", "pos": [0.0, 0.0] },
//!     { "label": "q1", "kind": "node", "pos": [1.0, 0.0] },
//!     { "label": "q0-q1", "kind": "engineered", "endpoints": ["q0", "q1"] }
//!   ] }
//! ```
//!
//! `kind` is `"node"`, `"engineered"`, `"parasitic"` or `"custom:<n>"`. Elements are
//! listed in ID order, so a round trip preserves IDs and the fingerprint.

use serde::{Deserialize, Serialize};

use super::{EdgeType, ElementKind, GraphBuilder, ProcessorGraph};
use crate::error::FaultTuneError;

const FORMAT: &str = "faulttune.graph/1";

#[derive(Serialize, Deserialize)]
struct GraphDoc {
    format: String,
    elements: Vec<ElementDoc>,
}

#[derive(Serialize, Deserialize)]
struct ElementDoc {
    label: String,
    kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pos: Option<[f32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    endpoints: Option<[String; 2]>,
}

fn kind_to_str(kind: ElementKind) -> String {
    match kind {
        ElementKind::Node => "node".into(),
        ElementKind::Edge(EdgeType::Engineered) => "engineered".into(),
        ElementKind::Edge(EdgeType::Parasitic) => "parasitic".into(),
        ElementKind::Edge(EdgeType::Custom(n)) => format!("custom:{n}"),
    }
}

fn kind_from_str(s: &str) -> Result<ElementKind, FaultTuneError> {
    Ok(match s {
        "node" => ElementKind::Node,
        "engineered" => ElementKind::Edge(EdgeType::Engineered),
        "parasitic" => ElementKind::Edge(EdgeType::Parasitic),
        other => match other
            .strip_prefix("custom:")
            .and_then(|n| n.parse::<u16>().ok())
        {
            Some(n) => ElementKind::Edge(EdgeType::Custom(n)),
            None => {
                return Err(FaultTuneError::Format(format!(
                    "unknown element kind {other:?}"
                )))
            }
        },
    })
}

impl ProcessorGraph {
    /// Serialises the graph to the JSON description shown in the module docs.
    pub fn to_json(&self) -> String {
        let coords = self.coords();
        let elements = self
            .elements()
            .map(|e| ElementDoc {
                label: e.label().to_string(),
                kind: kind_to_str(e.kind()),
                pos: if e.is_node() {
                    coords.map(|c| c[e.id().index()])
                } else {
                    None
                },
                endpoints: e
                    .endpoints()
                    .map(|[a, b]| [self.label(a).to_string(), self.label(b).to_string()]),
            })
            .collect();
        serde_json::to_string(&GraphDoc {
            format: FORMAT.into(),
            elements,
        })
        .expect("graph description serialises")
    }

    /// Parses a graph from its JSON description.
    pub fn from_json(json: &str) -> Result<Self, FaultTuneError> {
        let doc: GraphDoc = serde_json::from_str(json)?;
        if doc.format != FORMAT {
            return Err(FaultTuneError::Format(format!(
                "unsupported graph format {:?} (expected {FORMAT:?})",
                doc.format
            )));
        }
        let mut builder = GraphBuilder::new();
        for el in &doc.elements {
            match kind_from_str(&el.kind)? {
                ElementKind::Node => {
                    match el.pos {
                        Some(pos) => builder.add_node_at(el.label.clone(), pos)?,
                        None => builder.add_node(el.label.clone())?,
                    };
                }
                ElementKind::Edge(ty) => {
                    let [la, lb] = el.endpoints.as_ref().ok_or_else(|| {
                        FaultTuneError::InvalidGraph(format!(
                            "edge {:?} has no endpoints",
                            el.label
                        ))
                    })?;
                    let find = |l: &String| {
                        builder_lookup(&builder, l).ok_or_else(|| {
                            FaultTuneError::InvalidGraph(format!(
                                "edge {:?} refers to unknown node {l:?} (nodes must be listed before their edges)",
                                el.label
                            ))
                        })
                    };
                    let (a, b) = (find(la)?, find(lb)?);
                    builder.add_edge_labelled(el.label.clone(), a, b, ty)?;
                }
            }
        }
        builder.build()
    }
}

fn builder_lookup(builder: &GraphBuilder, label: &str) -> Option<super::ElementId> {
    builder.find_label(label)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_preserves_ids_and_fingerprint() {
        let g = ProcessorGraph::square_lattice(3, 4);
        let back = ProcessorGraph::from_json(&g.to_json()).unwrap();
        assert_eq!(back.fingerprint(), g.fingerprint());
        assert_eq!(back.len(), g.len());
        assert_eq!(back.coords(), g.coords());
        for id in g.ids() {
            assert_eq!(back.label(id), g.label(id));
            assert_eq!(back.kind(id), g.kind(id));
            assert_eq!(back.endpoints(id), g.endpoints(id));
        }
    }

    #[test]
    fn rejects_bad_documents() {
        assert!(ProcessorGraph::from_json("not json").is_err());
        assert!(ProcessorGraph::from_json(r#"{"format":"other/9","elements":[]}"#).is_err());
        let dangling = r#"{"format":"faulttune.graph/1","elements":[
            {"label":"a","kind":"node"},
            {"label":"a-b","kind":"engineered","endpoints":["a","b"]}]}"#;
        assert!(ProcessorGraph::from_json(dangling).is_err());
        let bad_kind = r#"{"format":"faulttune.graph/1","elements":[{"label":"a","kind":"blob"}]}"#;
        assert!(ProcessorGraph::from_json(bad_kind).is_err());
    }
}
