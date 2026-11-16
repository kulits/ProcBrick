//! ProcBrick Python bindings.

use crate::store::bad;
use crate::trees;
use crate::{appearance, coding};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes};
use std::io;
use std::path::PathBuf;

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<crate::colliders::Colliders>()?;
    module.add_class::<crate::geometry_pack::GeometryPacker>()?;
    module.add_class::<crate::sampler::Sampler>()?;
    module.add_class::<Appearance>()?;
    module.add_class::<Records>()?;
    module.add_class::<Coding>()?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

/// Invalid data (`store::bad`) as ValueError, file system errors as OSError.
pub(crate) fn io_err(error: io::Error) -> PyErr {
    match error.kind() {
        io::ErrorKind::Other => PyValueError::new_err(error.to_string()),
        _ => error.into(),
    }
}

/// Codec of trees on the atlas in `assets`, given the part id of each class.
#[pyclass(frozen)]
pub(crate) struct Coding(trees::Coding);

#[pymethods]
impl Coding {
    #[new]
    fn new(assets: PathBuf, representatives: Vec<i32>) -> PyResult<Self> {
        Ok(Self(trees::Coding::open(&assets, representatives).map_err(io_err)?))
    }

    /// Record of a tree: part ids and edge rows as the sampler gives them, and, if given, the float64 poses the record
    /// must decode to.
    fn encode<'py>(
        &self,
        py: Python<'py>,
        ids: Vec<u32>,
        edges: Vec<[i64; 7]>,
        poses: Option<&[u8]>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let block = py.detach(|| self.0.encode(&ids, &edges, poses)).map_err(io_err)?;
        Ok(PyBytes::new(py, &block))
    }

    /// Part counts, part ids, poses, and edge rows of the first `parts[n]` parts (None: all) of record `indices[n]`,
    /// as `Coding::read` gives them.
    fn read<'py>(
        &self,
        py: Python<'py>,
        records: &Records,
        indices: Vec<i64>,
        parts: Option<Vec<usize>>,
    ) -> PyResult<Read<'py>> {
        if parts.as_ref().is_some_and(|parts| parts.len() != indices.len()) {
            return Err(io_err(bad("indices and parts must have the same length")));
        }
        let indices = positions(indices, records.0.count, "tree")?;
        let read = py.detach(|| self.0.read(&records.0, &indices, parts.as_deref())).map_err(io_err)?;
        let (sizes, ids, poses, edges) = read;
        Ok((sizes, PyBytes::new(py, &ids), PyBytes::new(py, &poses), PyBytes::new(py, &edges)))
    }
}

/// Part counts, part ids, poses, and edge rows, as `Coding::read` gives them.
type Read<'py> = (Vec<usize>, Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>);

/// Memory-mapped records of trees or render appearance.
#[pyclass(frozen)]
pub(crate) struct Records(coding::Records);

#[pymethods]
impl Records {
    #[new]
    fn new(py: Python<'_>, root: PathBuf) -> PyResult<Self> {
        Ok(Self(py.detach(|| coding::Records::open(&root)).map_err(io_err)?))
    }

    #[getter]
    fn count(&self) -> usize {
        self.0.count
    }
}

/// Indices into a directory of `count` records, negative ones wrapping.
fn positions(indices: Vec<i64>, count: usize, what: &str) -> PyResult<Vec<usize>> {
    let count = count as i128;
    let position = |i: i64| {
        let t = if i < 0 { count + i as i128 } else { i as i128 };
        match (0..count).contains(&t) {
            true => Ok(t as usize),
            false => Err(PyIndexError::new_err(format!("{what} {i} is out of range ({count})"))),
        }
    };
    indices.into_iter().map(position).collect()
}

/// Codec of the appearance of renders, given the class of each part id.
#[pyclass(frozen)]
pub(crate) struct Appearance(appearance::Appearance);

#[pymethods]
impl Appearance {
    #[new]
    fn new(assets: PathBuf, class_ids: Vec<i64>) -> PyResult<Self> {
        Ok(Self(appearance::Appearance::open(&assets, &class_ids).map_err(io_err)?))
    }

    /// Record of a render's appearance: part ids with print variants and material ids, one per part.
    fn encode<'py>(&self, py: Python<'py>, variants: Vec<u32>, materials: Vec<u16>) -> PyResult<Bound<'py, PyBytes>> {
        let record = py.detach(|| self.0.encode(&variants, &materials)).map_err(io_err)?;
        Ok(PyBytes::new(py, &record))
    }
    /// Part ids with print variants (u32) and material ids (u16) of renders `indices`, as little-endian bytes, given
    /// the part ids of their parts in order (`part_ids`, render k from `starts[k]` to `starts[k + 1]`).
    fn read<'py>(
        &self,
        py: Python<'py>,
        records: &Records,
        indices: Vec<i64>,
        part_ids: Vec<u32>,
        starts: Vec<usize>,
    ) -> PyResult<(Bound<'py, PyByteArray>, Bound<'py, PyByteArray>)> {
        let indices = positions(indices, records.0.count, "render")?;
        let extents = starts.len() == indices.len() + 1
            && starts.first() == Some(&0)
            && starts.last() == Some(&part_ids.len())
            && starts.windows(2).all(|w| w[0] <= w[1]);
        if !extents {
            return Err(io_err(bad("the renders' parts are not given in order")));
        }
        let (mut variants, mut materials) = (vec![0u32; part_ids.len()], vec![0u16; part_ids.len()]);
        py.detach(|| self.0.read(&records.0, &indices, &part_ids, &starts, &mut variants, &mut materials))
            .map_err(io_err)?;
        let variants = PyByteArray::new(py, &variants.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let materials = PyByteArray::new(py, &materials.iter().flat_map(|m| m.to_le_bytes()).collect::<Vec<_>>());
        Ok((variants, materials))
    }
}
