//! Small helpers around NumPy (conversion of array-likes, random generators).

use numpy::{PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyDict;

struct Functions {
    asarray: Py<PyAny>,
    float64: Py<PyAny>,
    int64: Py<PyAny>,
    default_rng: Py<PyAny>,
}

static FUNCTIONS: PyOnceLock<Functions> = PyOnceLock::new();

fn functions(py: Python<'_>) -> PyResult<&'static Functions> {
    FUNCTIONS.get_or_try_init(py, || {
        let numpy = py.import("numpy")?;
        Ok(Functions {
            asarray: numpy.getattr("asarray")?.unbind(),
            float64: numpy.getattr("float64")?.unbind(),
            int64: numpy.getattr("int64")?.unbind(),
            default_rng: numpy.getattr("random")?.getattr("default_rng")?.unbind(),
        })
    })
}

/// `numpy.asarray(obj, dtype=dtype)`, after checking that the dtype `kind` of
/// `numpy.asarray(obj)` is one of `allowed` (empty arrays always pass: `[]` is a float array).
///
/// Without the check NumPy would silently turn the string `"0.5"` into a float and truncate
/// `0.7` to the index `0`.
fn as_array<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    dtype: &Py<PyAny>,
    allowed: &str,
    wanted: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let asarray = functions(py)?.asarray.bind(py);
    let raw = asarray.call1((obj,))?;
    let kind: String = raw.getattr("dtype")?.getattr("kind")?.extract()?;
    let size: usize = raw.getattr("size")?.extract()?;
    if size > 0 && !allowed.contains(kind.as_str()) {
        let name = raw.getattr("dtype")?.str()?;
        return Err(PyTypeError::new_err(format!(
            "expected {wanted}, got an array of dtype {name}"
        )));
    }
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", dtype.bind(py))?;
    asarray.call((raw,), Some(&kwargs))
}

/// `shape` as Python prints it: `(3,)`, `(2, 3)`, `()`.
pub fn format_shape(shape: &[usize]) -> String {
    match shape {
        [] => "()".into(),
        [n] => format!("({n},)"),
        many => format!(
            "({})",
            many.iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// The shape and the values (C order) of `numpy.asarray(obj, dtype=float64)`.
pub fn to_f64_array(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<(Vec<usize>, Vec<f64>)> {
    let f = functions(py)?;
    let array = as_array(py, obj, &f.float64, "biufO", "numbers")?;
    let array = array.cast::<PyArrayDyn<f64>>()?;
    let readonly = array.readonly();
    Ok((
        array.shape().to_vec(),
        readonly.as_array().iter().copied().collect(),
    ))
}

/// The shape and the values (C order) of `numpy.asarray(obj, dtype=int64)`.
///
/// Anything but an integer array is refused (`TypeError`): floats would be truncated by the
/// cast and silently select another index.
pub fn to_i64_array(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<(Vec<usize>, Vec<i64>)> {
    let f = functions(py)?;
    let array = as_array(py, obj, &f.int64, "iu", "integers")?;
    let array = array.cast::<PyArrayDyn<i64>>()?;
    let readonly = array.readonly();
    Ok((
        array.shape().to_vec(),
        readonly.as_array().iter().copied().collect(),
    ))
}

/// `numpy.random.default_rng(seed)`.
pub fn default_rng<'py>(py: Python<'py>, seed: u64) -> PyResult<Bound<'py, PyAny>> {
    functions(py)?.default_rng.bind(py).call1((seed,))
}
