//! Collider bounds and clearances.

use crate::geometry::Geometry;
use crate::geometry_pack::Packed;
use crate::parts::Parts;
use crate::python::io_err;
use crate::store::bad;
use crate::trees::pose;
use pyo3::prelude::*;
use std::path::Path;

/// Colliders of a geometry pack by part id.
#[pyclass]
pub(crate) struct Colliders {
    geometry: Geometry,
    meshes: Vec<Option<u32>>, // by part id
}

impl Colliders {
    fn mesh(&self, part: u32) -> PyResult<u32> {
        let mesh = self.meshes.get(part as usize).copied().flatten();
        mesh.ok_or_else(|| io_err(bad(&format!("part {part} has no collider"))))
    }
}

#[pymethods]
impl Colliders {
    /// Colliders of geometry pack `geometry` for the parts of `assets/parts.json`, `cache_bytes` of them cached.
    #[new]
    fn new(assets: &str, geometry: &str, cache_bytes: usize) -> PyResult<Self> {
        let raw = std::fs::read(Path::new(assets).join("parts.json"))?;
        let parts = Parts::read(&raw).map_err(io_err)?;
        let mut meshes = vec![None; parts.parts.iter().map(|p| p.0 as usize + 1).max().unwrap_or(0)];
        for &(part, mesh, _) in &parts.parts {
            meshes[part as usize] = Some(mesh);
        }
        let packed = Packed::open(Path::new(geometry), &raw).map_err(io_err)?;
        Ok(Self { geometry: Geometry::new(packed, cache_bytes), meshes })
    }

    /// Box of each part's collider in the part's frame: lo x, y, z, hi x, y, z (LDU; NaN without a collider).
    fn bounds(&self, part_ids: Vec<u32>) -> PyResult<Vec<[f64; 6]>> {
        let boxed = |part: u32| -> PyResult<[f64; 6]> {
            let Ok(mesh) = self.mesh(part) else {
                return Ok([f64::NAN; 6]);
            };
            let bounds = self.geometry.bounds_of(mesh)?;
            let (lo, hi) = (bounds.mins.to_array(), bounds.maxs.to_array());
            Ok([lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]])
        };
        part_ids.into_iter().map(boxed).collect()
    }

    /// Clearance of each pair (i, j) of parts `part_ids` placed at row-major 4 x 4 `poses` (LDU): the least
    /// distance between their colliders, 0 when they meet, infinity beyond `radius`.
    fn clearances(
        &mut self,
        py: Python<'_>,
        part_ids: Vec<u32>,
        poses: Vec<f64>,
        pairs: Vec<(usize, usize)>,
        radius: f64,
    ) -> PyResult<Vec<f64>> {
        if poses.len() != 16 * part_ids.len() || pairs.iter().any(|&(i, j)| i.max(j) >= part_ids.len()) {
            return Err(io_err(bad("one pose per part and pairs of its parts are required")));
        }
        let placed: Vec<_> = poses.chunks_exact(16).map(|m| pose(m.try_into().unwrap())).collect();
        let meshes = part_ids.iter().map(|&part| self.mesh(part)).collect::<PyResult<Vec<u32>>>()?;
        py.detach(|| {
            pairs
                .iter()
                .map(|&(i, j)| {
                    let (a, b) = (self.geometry.get(meshes[i])?, self.geometry.get(meshes[j])?);
                    a.clearance(&b, placed[i].inv_mul(&placed[j]), radius).map_err(io_err)
                })
                .collect()
        })
    }
}
