//! PyO3 bindings for FaultTune: the native extension module `faulttune._native`.
//!
//! The bindings wrap the core crate and contain no calibration logic. The pure-Python package
//! `faulttune` (in `python/faulttune`) re-exports these types and adds the `ErrorModel` base
//! class. See `docs/ARCHITECTURE.md`, section 9.

mod algorithm;
mod callbacks;
mod database;
mod errors;
mod graph;
mod model;
mod npy;
mod records;
mod run;
mod scoped;
mod snake;

use pyo3::prelude::*;

/// The native extension module of the `faulttune` package.
#[pymodule]
fn _native(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    errors::init(py, m)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<graph::PyElementKind>()?;
    m.add_class::<graph::PyElement>()?;
    m.add_class::<graph::PyElementSet>()?;
    m.add_class::<graph::PyGraph>()?;
    m.add_class::<graph::PyGraphBuilder>()?;
    m.add_class::<records::PyStep>()?;
    m.add_class::<records::PyStepRecord>()?;
    m.add_class::<records::PyInvalidation>()?;
    m.add_class::<records::PyReport>()?;
    m.add_class::<callbacks::PyTraversalContext>()?;
    m.add_class::<callbacks::PyCoordinateDescent>()?;
    m.add_class::<callbacks::PyExhaustive>()?;
    m.add_class::<callbacks::PyRandomSearch>()?;
    m.add_class::<callbacks::PySearchSpace>()?;
    m.add_class::<callbacks::PyObjective>()?;
    m.add_class::<database::PyDatabase>()?;
    m.add_class::<algorithm::PyRetry>()?;
    m.add_class::<algorithm::PyAlgorithmSubgraphs>()?;
    m.add_class::<algorithm::PyStitchPlan>()?;
    m.add_class::<snake::PySnake>()?;
    m.add_function(wrap_pyfunction!(database::py_read_journal, m)?)?;
    Ok(())
}
