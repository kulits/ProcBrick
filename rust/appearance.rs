//! Compressed render appearance.

// A render's parts are coded part by part with rANS against a trained model (APPMODEL) given the parts' classes, one
// record per render in the framing of `trees`.
//
// Per part, a record codes its variant (when its class has several part ids) and its material. A class's first
// variant comes from the class's table; later ones are flagged as new (a flag per uses of the class, variants taken,
// and class size), a new one taken from the class's table with the variants already taken down-weighted, an old one
// by how often it was taken. A part's material is flagged as new (a flag per position, materials so far, uses of the
// part's class, and their materials), a new one taken from the material table with the render's materials
// down-weighted, an old one among the render's materials, each weighted by its count plus REUSE times its count in
// the part's class. The first material is always new.
//
// Appearance model (APPMODEL): the magic; the SHA-256 fingerprint of the catalog (u64 parts, then each part's
// class as u32); arrays of u32, each after its u64 length: the material and variant flag probabilities (out of 4096,
// by cell), the frequencies of the material table (2^16), then of each class of several part ids in class order its
// table of part ids in increasing order (2^16).

use crate::coding::{Encoder, Io, Reader, Records, Table, SCALE};
use crate::store::{bad, le, word};
use sha2::{Digest, Sha256};
use std::fs;
use std::io;
use std::path::Path;

/// Magic of an appearance model file.
pub(crate) const MODEL: &[u8; 8] = b"APPMODEL";
/// Weight of an old material: its count in the render plus REUSE times its count in the part's class.
pub(crate) const REUSE: u32 = 64;
/// Material flag cells: 27 position bins x 16 material counts x 4 class uses x 3 class material counts.
pub(crate) const MATERIAL_FLAGS: usize = 27 * 16 * 4 * 3;
/// Variant flag cells: 8 class uses x 4 variant counts x 10 class sizes.
pub(crate) const VARIANT_FLAGS: usize = 8 * 4 * 10;

/// Material flag cell: position i (27 bins), materials so far (to 15), class uses (to 3), and class materials (to 2).
pub(crate) fn material_cell(i: usize, distinct: usize, uses: usize, class_distinct: usize) -> usize {
    let position = if i < 16 { i } else { 16 + ((i - 16) / 8).min(10) };
    ((position * 16 + distinct.min(15)) * 4 + uses.min(3)) * 3 + class_distinct.min(2)
}

/// Variant flag cell: uses of the class (to 7), variants taken (to 3), and the class's part ids (to 9).
pub(crate) fn variant_cell(uses: usize, taken: usize, members: usize) -> usize {
    (uses.min(7) * 4 + taken.min(3)) * 10 + members.min(9)
}

/// Part ids of each class, in increasing order, and each part id's class and rank among them.
pub(crate) struct Catalog {
    pub class: Vec<u16>,
    pub rank: Vec<u16>,
    pub at: Vec<u32>,
    pub members: Vec<u32>,
    pub fingerprint: [u8; 32],
}

impl Catalog {
    /// Catalog of `class_ids`, the class of each part id.
    pub(crate) fn new(class_ids: &[i64]) -> io::Result<Self> {
        let count = class_ids.iter().copied().max().map_or(0, |c| c + 1) as usize;
        if class_ids.is_empty() || class_ids.iter().any(|&c| c < 0) || count > 1 << 16 || class_ids.len() > 1 << 32 {
            return Err(bad("the catalog has 1 to 65536 classes"));
        }
        let mut hash = Sha256::new();
        hash.update((class_ids.len() as u64).to_le_bytes());
        let mut at = vec![0u32; count + 1];
        for &c in class_ids {
            hash.update((c as u32).to_le_bytes());
            at[c as usize + 1] += 1;
        }
        for c in 0..count {
            at[c + 1] += at[c];
        }
        let (mut members, mut rank) = (vec![0u32; class_ids.len()], vec![0u16; class_ids.len()]);
        let mut next = at.clone();
        for (part, &c) in class_ids.iter().enumerate() {
            let slot = &mut next[c as usize];
            rank[part] = u16::try_from(*slot - at[c as usize]).map_err(|_| bad("a class has over 65536 part ids"))?;
            members[*slot as usize] = part as u32;
            *slot += 1;
        }
        let class = class_ids.iter().map(|&c| c as u16).collect();
        Ok(Catalog { class, rank, at, members, fingerprint: hash.finalize().into() })
    }

    /// Number of part ids of class `class`.
    pub(crate) fn size(&self, class: usize) -> usize {
        (self.at[class + 1] - self.at[class]) as usize
    }
}

/// Shared appearance model: flag probabilities, the material table, and the table of each class of several part ids.
pub(crate) struct Model {
    pub material_flag: Vec<u16>,
    pub variant_flag: Vec<u16>,
    pub material: Table,
    pub variant: Vec<Option<Table>>,
}

impl Model {
    /// Model of the APPMODEL bytes `raw`, which must be for this catalog.
    fn load(raw: &[u8], catalog: &Catalog) -> io::Result<Self> {
        if raw.len() < 40 || &raw[..8] != MODEL {
            return Err(bad("not an appearance model (APPMODEL)"));
        }
        if raw[8..40] != catalog.fingerprint {
            return Err(bad("the appearance model is for another catalog"));
        }
        let mut at = 40;
        let mut next = || -> io::Result<Vec<u32>> {
            let n = word(raw, at, 8)? as usize;
            let body = raw.get(at + 8..at + 8 + 4 * n).ok_or_else(|| bad("the appearance model is truncated"))?;
            at += 8 + 4 * n;
            Ok(body.chunks_exact(4).map(|b| le(b, 0, 4) as u32).collect())
        };
        let flags = |p: Vec<u32>, n: usize| -> io::Result<Vec<u16>> {
            match p.len() == n && p.iter().all(|&p| (1..4096).contains(&p)) {
                true => Ok(p.into_iter().map(|p| p as u16).collect()),
                false => Err(bad("the appearance model has invalid flag probabilities")),
            }
        };
        let (material_flag, variant_flag) = (flags(next()?, MATERIAL_FLAGS)?, flags(next()?, VARIANT_FLAGS)?);
        let mut table = |n: usize| -> io::Result<Table> {
            let t = Table::new(&next()?, SCALE)?;
            let fits = t.cum.len() == n + 1 || n == 0;
            fits.then_some(t).ok_or_else(|| bad("the appearance model's tables do not fit its catalog"))
        };
        let material = table(0)?;
        let classes = 0..catalog.at.len() - 1;
        let variant = classes.map(|c| (catalog.size(c) > 1).then(|| table(catalog.size(c))).transpose());
        let variant = variant.collect::<io::Result<Vec<_>>>()?;
        if at != raw.len() {
            return Err(bad("the appearance model has the wrong size"));
        }
        Ok(Model { material_flag, variant_flag, material, variant })
    }
}

/// (key, count) pairs in order of first use.
type Tally = Vec<(u16, u16)>;

/// Render being coded: each material's count and position among the render's materials (in order of first use and
/// sorted), and per class placed (by its slot) its (variant rank, count) and (material, count) pairs; reused between
/// renders.
pub(crate) struct Context {
    pub count: Vec<u16>,
    pub position: Vec<u16>,
    pub order: Vec<u16>,
    pub sorted: Vec<u16>,
    slot: Vec<u16>,
    classes: Vec<u16>,
    taken: Vec<(Tally, Tally)>,
    pub weights: Vec<u32>,
    pub used: Vec<u16>,
}

impl Context {
    /// Empty context for `materials` materials and the classes of `catalog`.
    pub(crate) fn new(catalog: &Catalog, materials: usize) -> Self {
        Context {
            count: vec![0; materials],
            position: vec![0; materials],
            order: Vec::new(),
            sorted: Vec::new(),
            slot: vec![u16::MAX; catalog.at.len() - 1],
            classes: Vec::new(),
            taken: Vec::new(),
            weights: Vec::new(),
            used: Vec::new(),
        }
    }

    /// Start a render, clearing the last one.
    pub(crate) fn start(&mut self) {
        for &m in &self.order {
            self.count[m as usize] = 0;
        }
        for (&c, (variants, materials)) in self.classes.iter().zip(&mut self.taken) {
            self.slot[c as usize] = u16::MAX;
            variants.clear();
            materials.clear();
        }
        self.order.clear();
        self.sorted.clear();
        self.classes.clear();
    }

    /// (variant rank, count) pairs of class c in the render so far.
    pub(crate) fn variants(&self, c: usize) -> &[(u16, u16)] {
        self.taken.get(self.slot[c] as usize).map_or(&[], |t| &t.0)
    }

    /// (material, count) pairs of class c in the render so far.
    pub(crate) fn materials(&self, c: usize) -> &[(u16, u16)] {
        self.taken.get(self.slot[c] as usize).map_or(&[], |t| &t.1)
    }

    /// Variant flag cell of a part of class c with `size` part ids; None for its first.
    pub(crate) fn variant_cell(&self, c: usize, size: usize) -> Option<usize> {
        let taken = self.variants(c);
        let uses = taken.iter().map(|t| t.1 as usize).sum();
        (!taken.is_empty()).then(|| variant_cell(uses, taken.len(), size))
    }

    /// Material flag cell of part i, of class c; None for the first part.
    pub(crate) fn material_cell(&self, i: usize, c: usize) -> Option<usize> {
        let class = self.materials(c);
        let uses = class.iter().map(|t| t.1 as usize).sum();
        (!self.order.is_empty()).then(|| material_cell(i, self.order.len(), uses, class.len()))
    }

    /// Record a part of class c with variant rank r and material m.
    pub(crate) fn add(&mut self, c: usize, r: usize, m: usize) {
        if self.slot[c] == u16::MAX {
            self.slot[c] = self.classes.len() as u16;
            self.classes.push(c as u16);
            if self.taken.len() < self.classes.len() {
                self.taken.push(Default::default());
            }
        }
        let (variants, materials) = &mut self.taken[self.slot[c] as usize];
        bump(variants, r as u16);
        bump(materials, m as u16);
        if self.count[m] == 0 {
            self.position[m] = self.order.len() as u16;
            self.order.push(m as u16);
            let at = self.sorted.partition_point(|&s| s < m as u16);
            self.sorted.insert(at, m as u16);
        }
        self.count[m] += 1;
    }
}

fn bump(pairs: &mut Tally, key: u16) {
    match pairs.iter_mut().find(|p| p.0 == key) {
        Some(p) => p.1 += 1,
        None => pairs.push((key, 1)),
    }
}

/// Appearance coding of renders: the catalog and the appearance model of the assets.
pub struct Appearance {
    pub(crate) catalog: Catalog,
    pub(crate) model: Model,
}

impl Appearance {
    /// Appearance coding of `assets/appearance_model`, for the classes `class_ids` of the part ids.
    pub fn open(assets: &Path, class_ids: &[i64]) -> io::Result<Self> {
        let catalog = Catalog::new(class_ids)?;
        let path = assets.join("appearance_model");
        let raw = fs::read(&path).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
        let model = Model::load(&raw, &catalog)?;
        Ok(Self { catalog, model })
    }

    /// Context for coding renders with this catalog and model.
    pub(crate) fn context(&self) -> Context {
        Context::new(&self.catalog, self.model.material.cum.len() - 1)
    }

    /// Code the variant ranks and materials of a render's parts of `classes`.
    pub(crate) fn code<I: Io>(
        &self,
        io: &mut I,
        classes: &[u16],
        ranks: &mut [u16],
        materials: &mut [u16],
        cx: &mut Context,
    ) -> io::Result<()> {
        let m = &self.model;
        cx.start();
        for (i, &c) in classes.iter().enumerate() {
            let c = c as usize;
            let size = self.catalog.size(c);
            if size == 0 {
                return Err(bad("a class has no part ids"));
            }
            let r = match (&m.variant[c], cx.variant_cell(c, size)) {
                (None, _) => 0,
                (Some(table), None) => io.table(table, ranks[i] as usize),
                (Some(table), Some(cell)) => {
                    let taken = cx.taken.get(cx.slot[c] as usize).map_or(&[][..], |t| &t.0);
                    let old = if I::ENCODE { taken.iter().position(|t| t.0 == ranks[i]) } else { None };
                    if io.bit(m.variant_flag[cell] as u32, old.is_none()) {
                        cx.used.clear();
                        cx.used.extend(taken.iter().map(|t| t.0));
                        cx.used.sort_unstable();
                        io.reweighted(table, &cx.used, ranks[i] as usize)
                    } else {
                        cx.weights.clear();
                        cx.weights.extend(taken.iter().map(|t| t.1 as u32));
                        taken[io.weighted(&cx.weights, old.unwrap_or(0))].0 as usize
                    }
                }
            };
            let material = materials[i] as usize;
            let new = match cx.material_cell(i, c) {
                None => true,
                Some(cell) => io.bit(m.material_flag[cell] as u32, I::ENCODE && cx.count[material] == 0),
            };
            let material = if new {
                io.reweighted(&m.material, &cx.sorted, material)
            } else {
                cx.weights.clear();
                cx.weights.extend(cx.order.iter().map(|&o| cx.count[o as usize] as u32));
                for &(o, n) in cx.taken.get(cx.slot[c] as usize).map_or(&[][..], |t| &t.1) {
                    cx.weights[cx.position[o as usize] as usize] += REUSE * n as u32;
                }
                let at = if I::ENCODE { cx.position[material] as usize } else { 0 };
                cx.order[io.weighted(&cx.weights, at)] as usize
            };
            if new == (cx.count[material] != 0) {
                return Err(bad("the appearance record is corrupt"));
            }
            (ranks[i], materials[i]) = (r as u16, material as u16);
            cx.add(c, r, material);
        }
        Ok(())
    }

    /// Decode the part ids and materials of renders `indices` of `set`, each showing the parts `part_ids` (from
    /// `starts[k]` to `starts[k + 1]`), into `variants` and `materials` at the same positions.
    pub fn read(
        &self,
        set: &Records,
        indices: &[usize],
        part_ids: &[u32],
        starts: &[usize],
        variants: &mut [u32],
        materials: &mut [u16],
    ) -> io::Result<()> {
        let (mut cx, mut classes, mut ranks) = (self.context(), Vec::new(), Vec::new());
        for (k, &t) in indices.iter().enumerate() {
            let (a, b) = (starts[k], starts[k + 1]);
            classes.clear();
            for &part in &part_ids[a..b] {
                classes.push(
                    *self.catalog.class.get(part as usize).ok_or_else(|| bad("a part id is not in the catalog"))?,
                );
            }
            ranks.resize(b - a, 0);
            let mut reader = Reader::new(set.record(t));
            self.code(&mut reader, &classes, &mut ranks, &mut materials[a..b], &mut cx)?;
            if !reader.finished() {
                return Err(bad("the appearance record is corrupt"));
            }
            for (v, (&c, &r)) in variants[a..b].iter_mut().zip(classes.iter().zip(&ranks)) {
                *v = self.catalog.members[(self.catalog.at[c as usize] + r as u32) as usize];
            }
        }
        Ok(())
    }
}

impl Appearance {
    /// Record of a render's appearance: its part ids with print variants and material ids; the record must decode to
    /// these given the parts' classes and be under 64 KiB.
    pub fn encode(&self, variants: &[u32], materials: &[u16]) -> io::Result<Vec<u8>> {
        let (catalog, count) = (&self.catalog, self.model.material.cum.len() - 1);
        if variants.len() != materials.len() {
            return Err(bad("a render has as many materials as parts"));
        }
        if variants.iter().any(|&v| v as usize >= catalog.class.len()) || materials.iter().any(|&m| m as usize >= count)
        {
            return Err(bad("a part id or material is outside the catalog or the model"));
        }
        let classes: Vec<u16> = variants.iter().map(|&v| catalog.class[v as usize]).collect();
        let mut ranks: Vec<u16> = variants.iter().map(|&v| catalog.rank[v as usize]).collect();
        let (mut coded, mut encoder, mut cx) = (materials.to_vec(), Encoder::default(), self.context());
        self.code(&mut encoder, &classes, &mut ranks, &mut coded, &mut cx)?;
        let record = encoder.finish();
        let (mut back, mut back_ranks) = (vec![0u16; materials.len()], vec![0u16; materials.len()]);
        let mut reader = Reader::new(&record);
        self.code(&mut reader, &classes, &mut back_ranks, &mut back, &mut cx)?;
        let same = back == materials && back_ranks.iter().zip(variants).all(|(&r, &v)| r == catalog.rank[v as usize]);
        if !reader.finished() || !same {
            return Err(bad("the render's appearance record does not decode to its appearance"));
        }
        if record.len() >= 1 << 16 {
            return Err(bad("the render's appearance record is 64 KiB or more"));
        }
        Ok(record)
    }
}
