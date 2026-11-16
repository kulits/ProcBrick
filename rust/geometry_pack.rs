//! Memory-mapped collider cache.

use crate::geometry::Collider;
use crate::parts::Parts;
use crate::python::io_err;
use crate::store::{bad, le};
use memmap2::{Mmap, MmapOptions};
use pyo3::prelude::*;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 8] = b"GEOMPACK";
const HEADER: usize = 88; // magic, the SHA-256 of parts.json, u64 count, u64 index offset, the index's SHA-256
const RECORD: usize = 20; // per collider: u32 mesh, u64 offset, u64 bytes

/// Geometry pack, memory-mapped, and the position of each collider's index record.
pub(crate) struct Packed {
    data: Mmap,
    index: HashMap<u32, usize>,
}

impl Packed {
    /// Pack at `path`, which must have been built for the parts.json whose bytes are `parts`.
    pub fn open(path: &Path, parts: &[u8]) -> io::Result<Self> {
        let mut file = File::open(path)?;
        // warm the page cache with large sequential reads, not thousands of remote page faults
        let mut buffer = vec![0u8; 1 << 20];
        while file.read(&mut buffer)? != 0 {}
        // SAFETY: the pack, built offline, is not modified while mapped
        let data = unsafe { MmapOptions::new().map(&file) }?;
        if data.len() < HEADER || &data[..8] != MAGIC {
            return Err(bad(&format!("{} is not a geometry pack", path.display())));
        }
        if Sha256::digest(parts)[..] != data[8..40] {
            return Err(bad(&format!("{} was built for other parts", path.display())));
        }
        let (count, start) = (le(&data, 40, 8) as usize, le(&data, 48, 8) as usize);
        if start < HEADER || count.checked_mul(RECORD).and_then(|n| n.checked_add(start)) != Some(data.len()) {
            return Err(bad("the geometry pack is truncated"));
        }
        if Sha256::digest(&data[start..])[..] != data[56..88] {
            return Err(bad("the geometry pack's index is corrupt"));
        }
        let mut index = HashMap::with_capacity(count);
        for pos in (start..data.len()).step_by(RECORD) {
            let (offset, bytes) = (le(&data, pos + 4, 8) as usize, le(&data, pos + 12, 8) as usize);
            if offset < HEADER
                || offset.checked_add(bytes).filter(|end| *end <= start).is_none()
                || index.insert(le(&data, pos, 4) as u32, pos).is_some()
            {
                return Err(bad("the geometry pack's index is corrupt"));
            }
        }
        Ok(Self { data, index })
    }

    /// Byte range of the packed collider of `mesh`.
    fn span(&self, mesh: u32) -> io::Result<Range<usize>> {
        let pos = *self.index.get(&mesh).ok_or_else(|| bad(&format!("the geometry pack has no collider {mesh}")))?;
        let offset = le(&self.data, pos + 4, 8) as usize;
        Ok(offset..offset + le(&self.data, pos + 12, 8) as usize)
    }

    /// Packed collider of `mesh`.
    pub fn get(&self, mesh: u32) -> io::Result<&[u8]> {
        Ok(&self.data[self.span(mesh)?])
    }

    /// Drop the pages of a decoded collider from the mapping (they stay in the page cache).
    pub fn release(&self, mesh: u32) {
        #[cfg(not(unix))] // other platforms keep the pages until the pack is closed
        let _ = mesh;
        #[cfg(unix)]
        if let Ok(span) = self.span(mesh) {
            // SAFETY: a read-only file mapping, whose pages are read again from the file when touched
            let _ =
                unsafe { self.data.unchecked_advise_range(memmap2::UncheckedAdvice::DontNeed, span.start, span.len()) };
        }
    }
}

/// Geometry pack writer: the collider of each mesh is added from its OBJ file, then `finish` checks that every
/// collider of parts.json was added, writes the index, and moves the pack into place. Until then the pack is written
/// to a temporary file beside it, removed if the packer is dropped unfinished.
#[pyclass]
pub(crate) struct GeometryPacker {
    path: PathBuf,
    temporary: PathBuf,
    writer: Option<BufWriter<File>>,
    index: Vec<u8>,
    meshes: HashSet<u32>,
    written: HashMap<[u8; 32], (u64, u64)>, // identical encodings are stored once
}

#[pymethods]
impl GeometryPacker {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let path = PathBuf::from(path);
        if path.exists() {
            let message = format!("{} exists", path.display());
            return Err(io_err(io::Error::new(io::ErrorKind::AlreadyExists, message)));
        }
        let mut name = path.file_name().ok_or_else(|| io_err(bad("the geometry pack needs a file name")))?.to_owned();
        name.push(format!(".{}.partial", std::process::id()));
        let temporary = path.with_file_name(name);
        let mut writer = BufWriter::new(File::create(&temporary).map_err(io_err)?);
        let header = writer.write_all(&[0; HEADER]);
        let packer = Self {
            path,
            temporary,
            writer: Some(writer),
            index: Vec::new(),
            meshes: HashSet::new(),
            written: HashMap::new(),
        };
        header.map_err(io_err)?; // the packer, dropped, removes the temporary file
        Ok(packer)
    }

    /// Add the collider of `mesh` from its OBJ file.
    fn add(&mut self, mesh: u32, raw: &[u8]) -> PyResult<()> {
        let bytes = Collider::from_obj(raw).map_err(io_err)?.encode();
        let writer = self.writer.as_mut().ok_or_else(|| io_err(bad("the geometry pack is finished")))?;
        if !self.meshes.insert(mesh) {
            return Err(io_err(bad(&format!("collider {mesh} was added twice"))));
        }
        let digest: [u8; 32] = Sha256::digest(&bytes).into();
        let (offset, size) = match self.written.get(&digest) {
            Some(&written) => written,
            None => {
                let written = (writer.stream_position().map_err(io_err)?, bytes.len() as u64);
                writer.write_all(&bytes).map_err(io_err)?;
                *self.written.entry(digest).or_insert(written)
            }
        };
        self.index.extend(mesh.to_le_bytes().into_iter().chain(offset.to_le_bytes()).chain(size.to_le_bytes()));
        Ok(())
    }

    /// Write the index and header for `parts`, the bytes of parts.json.
    fn finish(&mut self, parts: &[u8]) -> PyResult<()> {
        let rows = Parts::read(parts).map_err(io_err)?.geometry;
        if rows.len() != self.meshes.len() || !rows.iter().all(|(mesh, _)| self.meshes.contains(mesh)) {
            return Err(io_err(bad("the colliders added are not those of parts.json")));
        }
        let mut writer = self.writer.take().ok_or_else(|| io_err(bad("the geometry pack is finished")))?;
        let mut header = MAGIC.to_vec();
        header.extend_from_slice(&Sha256::digest(parts));
        header.extend_from_slice(&(self.meshes.len() as u64).to_le_bytes());
        header.extend_from_slice(&writer.stream_position().map_err(io_err)?.to_le_bytes());
        header.extend_from_slice(&Sha256::digest(&self.index));
        writer.write_all(&self.index).map_err(io_err)?;
        writer.seek(SeekFrom::Start(0)).map_err(io_err)?;
        writer.write_all(&header).map_err(io_err)?;
        let file = writer.into_inner().map_err(|e| io_err(e.into_error()))?;
        file.sync_all().map_err(io_err)?;
        drop(file);
        fs::rename(&self.temporary, &self.path).map_err(io_err)
    }
}

impl Drop for GeometryPacker {
    /// Remove the temporary file of an unfinished pack (a finished one has been moved into place).
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.temporary);
    }
}
