//! ProcBrick Rust core.

pub mod appearance;
pub mod coding;
pub mod parts;
pub mod store;
pub mod trees;

#[cfg(feature = "python")]
mod colliders;
#[cfg(feature = "python")]
mod contacts;
#[cfg(feature = "python")]
mod geometry;
#[cfg(feature = "python")]
mod geometry_pack;
#[cfg(feature = "python")]
mod inputs;
#[cfg(feature = "python")]
mod mates;
#[cfg(feature = "python")]
mod python;
#[cfg(feature = "python")]
mod sampler;

#[cfg(feature = "python")]
type FastMap<K, V> = std::collections::HashMap<K, V, rustc_hash::FxBuildHasher>;
#[cfg(feature = "python")]
type FastLru<K, V> = lru::LruCache<K, V, rustc_hash::FxBuildHasher>;
