//! Sampler inputs: the parts' connectors and colliders.

use crate::geometry::Geometry;
use crate::geometry_pack::Packed;
use crate::parts::{self, has, Connector, Fixed, Kind};
use crate::store::bad;
use crate::store::Store;
use crate::trees::PoseDictionary;
use parry3d_f64::bounding_volume::Aabb;
use std::collections::HashMap;
use std::io;
use std::path::Path;
use std::sync::OnceLock;

/// Part of the sampling pool: its collider, its connectors, and how each pairs (`Kind`).
pub(crate) struct Part {
    pub id: u32,
    pub mesh: u32,
    pub bounds: OnceLock<Aabb>, // of its collider, read from the geometry pack when first needed
    pub connectors: Vec<Connector>,
    pub kinds: Vec<Kind>,
    pub covered: bool, // whether the atlas covers it (has a connector with a profile)
}

impl Part {
    /// Fixed connectors: index and label.
    pub fn fixed_connectors(&self) -> impl Iterator<Item = (usize, Fixed)> + '_ {
        self.kinds.iter().enumerate().filter_map(|(k, kind)| match kind {
            Kind::Fixed(fixed) => Some((k, *fixed)),
            Kind::Profiled(_) => None,
        })
    }
}

/// Sampler inputs: the pool's parts, which connector types join which, the pose dictionary, each frame's connector
/// length, and the colliders.
pub(crate) struct Inputs {
    pub parts: Vec<Part>,
    pub receivers: Vec<Vec<usize>>, // per connector type, the types that can join it
    pub poses: PoseDictionary,      // the connector frames and joint rotations, which place parts as the decoder does
    pub lengths: Vec<f64>,
    pub geometry: Geometry,
}

impl Inputs {
    /// Inputs for the parts `pool` (sorted) of `assets` (the atlas's connectors, parts.json, and the pose dictionary)
    /// with the colliders of the geometry pack, `cache_bytes` of them cached.
    pub fn new(atlas: &Store, assets: &Path, geometry: &Path, pool: &[u32], cache_bytes: usize) -> io::Result<Self> {
        let parts::Assets { poses, types, parts: listed, raw, connectors } = parts::open(assets, &atlas.chart)?;
        let packed = Packed::open(geometry, &raw)?;
        let meshes: HashMap<u32, u32> = listed.parts.iter().map(|&(id, mesh, _)| (id, mesh)).collect();
        let mut parts = Vec::with_capacity(pool.len());
        for &id in pool {
            let mesh =
                *meshes.get(&id).ok_or_else(|| bad(&format!("part {id} of the sampling pool is not in parts.json")))?;
            let own = connectors.get(id as usize).map_or(&[][..], Vec::as_slice);
            if own
                .iter()
                .any(|c| c.profile.is_some_and(|p| atlas.profile(id as usize, c.connector.id as usize) != Some(p)))
            {
                return Err(bad("the pose dictionary and the atlas have different connectors"));
            }
            packed.get(mesh).map_err(|_| bad(&format!("the geometry pack has no collider for part {id}")))?;
            parts.push(Part {
                id,
                mesh,
                bounds: OnceLock::new(),
                connectors: own.iter().map(|c| c.connector).collect(),
                kinds: own.iter().map(|c| c.kind()).collect(),
                covered: own.iter().any(|c| c.profile.is_some()),
            });
        }
        if parts.is_empty() {
            return Err(bad("the sampling pool is empty"));
        }
        let mut lengths = listed.lengths;
        lengths.resize(poses.frames.len(), 0.0);
        let receivers = types.compatible.iter().map(|bits| (0..types.count()).filter(move |&r| has(bits, r)));
        let receivers = receivers.map(Iterator::collect).collect();
        Ok(Self { parts, receivers, poses, lengths, geometry: Geometry::new(packed, cache_bytes) })
    }
}
