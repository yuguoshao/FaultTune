//! Error types.
//!
//! [`FaultTuneError`] is the one error type of the library (see ARCHITECTURE §8).
//! [`ModelError`] is what a user's [`ErrorModel`](crate::ErrorModel) returns when it fails.

use std::error::Error;
use std::fmt;

use crate::events::Report;
use crate::graph::ElementId;

/// An error raised by a user-supplied [`ErrorModel`](crate::ErrorModel).
///
/// It carries a message and, optionally, the model's own error as the
/// [`source`](Error::source). The Python bindings store the original Python
/// exception here so it can be re-raised with a chained traceback.
#[derive(Debug)]
pub struct ModelError {
    message: String,
    source: Option<Box<dyn Error + Send + Sync + 'static>>,
}

impl ModelError {
    /// A model error with a message only.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            source: None,
        }
    }

    /// A model error wrapping another error.
    pub fn with_source(
        message: impl Into<String>,
        source: impl Error + Send + Sync + 'static,
    ) -> Self {
        Self {
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }

    /// A model error that adopts `source`'s message.
    pub fn from_error(source: impl Error + Send + Sync + 'static) -> Self {
        Self {
            message: source.to_string(),
            source: Some(Box::new(source)),
        }
    }

    /// The message.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// The wrapped error, if any.
    pub fn source_ref(&self) -> Option<&(dyn Error + Send + Sync + 'static)> {
        self.source.as_deref()
    }

    /// Takes the wrapped error out, if any.
    pub fn into_source(self) -> Option<Box<dyn Error + Send + Sync + 'static>> {
        self.source
    }
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for ModelError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.source {
            Some(s) => Some(&**s as &(dyn Error + 'static)),
            None => None,
        }
    }
}

impl From<String> for ModelError {
    fn from(message: String) -> Self {
        Self::new(message)
    }
}

impl From<&str> for ModelError {
    fn from(message: &str) -> Self {
        Self::new(message)
    }
}

/// The error type of the FaultTune library.
#[derive(Debug)]
#[non_exhaustive]
pub enum FaultTuneError {
    /// The processor graph is malformed (detected while building or loading it).
    InvalidGraph(String),
    /// The configuration, goal or another input is invalid (detected before any step runs).
    InvalidConfig(String),
    /// `Exhaustive` was asked to search more combinations than its budget allows.
    SearchSpaceTooLarge {
        /// Number of combinations (`∏ k_i`), saturating at `u128::MAX`.
        size: u128,
        /// The budget that was exceeded.
        budget: u64,
    },
    /// The error model failed while calibrating the step centred on `element`.
    Model {
        /// Centre of the failing step.
        element: ElementId,
        /// The model's own error.
        source: ModelError,
    },
    /// The error model returned NaN for the step centred on `element`.
    NonFiniteError {
        /// Centre of the failing step.
        element: ElementId,
    },
    /// A pass made no progress (only possible with a custom seed policy).
    NoProgress,
    /// A saved calibration belongs to a different graph.
    GraphMismatch {
        /// Fingerprint of the graph in use.
        expected: u64,
        /// Fingerprint recorded in the saved calibration.
        found: u64,
    },
    /// The run was cancelled. The database keeps every committed step.
    Cancelled {
        /// Report of the work done before the cancellation.
        report: Box<Report>,
    },
    /// Reading or writing a file failed.
    Io(std::io::Error),
    /// A file or string is not in a recognised format.
    Format(String),
}

impl fmt::Display for FaultTuneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGraph(msg) => write!(f, "invalid graph: {msg}"),
            Self::InvalidConfig(msg) => write!(f, "invalid configuration: {msg}"),
            Self::SearchSpaceTooLarge { size, budget } => write!(
                f,
                "search space of {size} combinations exceeds the budget of {budget}"
            ),
            Self::Model { element, source } => write!(
                f,
                "the error model failed for the step centred on element {}: {source}",
                element.raw()
            ),
            Self::NonFiniteError { element } => write!(
                f,
                "the error model returned NaN for the step centred on element {}",
                element.raw()
            ),
            Self::NoProgress => f.write_str("a calibration pass made no progress"),
            Self::GraphMismatch { expected, found } => write!(
                f,
                "graph mismatch: expected fingerprint {expected:016x}, found {found:016x}"
            ),
            Self::Cancelled { .. } => f.write_str("the calibration was cancelled"),
            Self::Io(err) => write!(f, "I/O error: {err}"),
            Self::Format(msg) => write!(f, "format error: {msg}"),
        }
    }
}

impl Error for FaultTuneError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Model { source, .. } => Some(source),
            Self::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for FaultTuneError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

#[cfg(feature = "serde")]
impl From<serde_json::Error> for FaultTuneError {
    fn from(err: serde_json::Error) -> Self {
        Self::Format(err.to_string())
    }
}
