//! Compressed trees and their pose dictionary.

// Each record codes a tree of 2 or more parts part by part against the assembly model (TWRMODEL), so a record
// decodes alone and its first k parts decode without the rest. Part i is placed at
// `pose[parent] @ F_a @ J(q) @ inverse(F_b)`, J(q) the joint at integer coordinates q. `encode` writes records.
//
// Per part after the first, a record codes its parent (a table per position below PARTS, uniform beyond), the
// parent's connector (a table per class, with the parent's used connectors down-weighted), whether its class was
// placed before among the placed classes that can attach to that connector's type (a flag per position, reuse count,
// and number of candidates, the first two clamped to PARTS - 1) and which (uniform), or else a new class (a table per
// connector type), the rank of its own connector among those compatible with the parent's, and the joint coordinates
// (tables per family). Encoder and decoder run the same `code`.
//
// A record is a 64-bit rANS state renormalized by 32-bit words: the final state in 4 + (record length mod 4)
// little-endian bytes, then the words in decoding order; a whole assembly decodes back to the initial state 2^31
// with every word read, and only a whole assembly does (every later part codes a parent of nonzero frequency below
// the total, which raises the encoder's state or emits a word), so a record ends its assembly. A value v uniform in
// n takes the slots [floor(v 2^s / n), floor((v + 1) 2^s / n)) of 2^s, with s = 16 for n <= 256, else 24. `index`
// holds per group of 64 records a u64, the offset of its first record in `data`, then 64 u16, each record's offset
// from it modulo 2^16 (0 past the last record); records are under 64 KiB.
//
// Assembly model (TWRMODEL): the magic; the SHA-256 fingerprint of its connectors (u64 classes, u64 types, then per
// class each connector's index and type as u16 and the count so far as u64); arrays of u32, each after its u64
// length: the reuse probabilities (out of 4096, by (i, reuses, candidates)), the class table of each connector type,
// the number of class tables, and the slide counts with a table; then the frequencies of each table: the first part's
// class (total 2^20), the parent at positions 1 to 99 (2^16), the connector of each class (2^16), the class tables
// (2^20), the stud, axle, ball x, y, and z, and hinge tables (2^16), and the slide tables (2^16).

use crate::coding::{Encoder, Io, Reader, Records, Table, SCALE};
use crate::parts::{has, Connector, Types};
use crate::store::{bad, le, word, Chart, Domain, AXLE, BALL, FIXED, HINGE, STUD};
use glamx::{DMat3 as Matrix3, DPose3 as Pose, DRot3 as Rotation, DVec3 as Vector};
use sha2::{Digest, Sha256};
use std::fs;
use std::io;
use std::path::Path;

/// Positions the assembly model fits: the parts of the released trees.
pub const PARTS: usize = 100;
/// Row-major 4 x 4 matrix of a pose.
pub fn matrix(p: Pose) -> [f64; 16] {
    let r = Matrix3::from_quat(p.rotation);
    let row = |i: usize| [r.x_axis[i], r.y_axis[i], r.z_axis[i], p.translation[i]];
    let [a, b, c] = [row(0), row(1), row(2)];
    [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], c[0], c[1], c[2], c[3], 0.0, 0.0, 0.0, 1.0]
}

/// Pose of a row-major rigid 4 x 4 matrix (the inverse of `matrix`), its rotation normalized.
pub fn pose(m: &[f64; 16]) -> Pose {
    let column = |j: usize| Vector::new(m[j], m[4 + j], m[8 + j]);
    let rotation = Rotation::from_mat3(&Matrix3::from_cols(column(0), column(1), column(2))).normalize();
    Pose::from_parts(column(3), rotation)
}

/// Joint J(q) of a family at integer coordinates q (degrees, LDU), from the rotations `axes` at each integer degree.
#[inline(always)] // as part of `PoseDictionary::attach`
pub fn joint(axes: &[Vec<Rotation>; 3], family: u8, q: [i32; 3]) -> Pose {
    let turn = |axis: usize, i: usize| axes[axis][q[i] as usize];
    let rotation = match family {
        FIXED => return Pose::IDENTITY,
        BALL => return Pose::from_rotation(turn(2, 2) * turn(1, 1) * turn(0, 0)),
        STUD => turn(1, 0),
        _ if q[0] != 0 => turn(1, 1) * Rotation::from_xyzw(1.0, 0.0, 0.0, 0.0),
        _ => turn(1, 1),
    };
    Pose::from_parts(Vector::new(0.0, if family == AXLE { -q[2] as f64 } else { 0.0 }, 0.0), rotation)
}

/// Pose dictionary: connector frames, rotations at integer degrees, and per (part, connector), in increasing order,
/// its frame.
pub struct PoseDictionary {
    pub frames: Vec<Pose>,
    pub(crate) connectors: Vec<(u32, u32, usize)>,
    axes: [Vec<Rotation>; 3],
}

impl PoseDictionary {
    /// Pose of a part attached by `j` to a part at `parent`: how the sampler places parts and the decoder places them
    /// again.
    #[inline(always)] // in the proposal loop and in `place`
    pub fn attach(&self, parent: Pose, j: &Joint) -> Pose {
        let joint = joint(&self.axes, j.family, j.q);
        parent * (self.frames[j.fa] * if j.reversed { joint.inverse() } else { joint } * self.frames[j.fb].inverse())
    }

    /// Pose dictionary of the assets directory `dir` (its file `pose_dictionary`).
    pub fn open(dir: &Path) -> io::Result<Self> {
        let path = dir.join("pose_dictionary");
        let raw = fs::read(&path).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
        Self::read(&raw)
    }

    /// Dictionary of the bytes of a POSEDICT file, checked against its trailing SHA-256 and its size.
    pub fn read(raw: &[u8]) -> io::Result<Self> {
        if raw.len() < 56
            || &raw[..8] != b"POSEDICT"
            || Sha256::digest(&raw[..raw.len() - 32])[..] != raw[raw.len() - 32..]
        {
            return Err(bad("the pose dictionary fails its checksum"));
        }
        let (frames, connectors) = (word(raw, 8, 8)? as usize, word(raw, 16, 8)? as usize);
        let expected = frames
            .checked_mul(56)
            .and_then(|n| n.checked_add(24 + 3 * 360 * 32))
            .and_then(|n| connectors.checked_mul(12).and_then(|m| n.checked_add(m + 32)));
        if expected != Some(raw.len()) {
            return Err(bad("the pose dictionary has the wrong size"));
        }
        let v = |at: usize, k: usize| f64::from_bits(le(raw, at + 8 * k, 8));
        let poses = (0..frames).map(|i| 24 + 56 * i).map(|at| {
            let rotation = Rotation::from_xyzw(v(at, 3), v(at, 4), v(at, 5), v(at, 6));
            Pose::from_parts(Vector::new(v(at, 0), v(at, 1), v(at, 2)), rotation)
        });
        let frames: Vec<Pose> = poses.collect();
        let first = 24 + 56 * frames.len();
        let axes: [Vec<Rotation>; 3] = std::array::from_fn(|a| {
            let rotation = |at: usize| Rotation::from_xyzw(v(at, 0), v(at, 1), v(at, 2), v(at, 3));
            (0..360).map(|i| rotation(first + 32 * (360 * a + i))).collect()
        });
        if frames.iter().any(|f| !f.translation.is_finite() || !f.rotation.is_finite())
            || axes.iter().flatten().any(|q| !q.is_finite())
        {
            return Err(bad("the pose dictionary has a non-finite frame or rotation"));
        }
        let rows = raw[first + 3 * 360 * 32..raw.len() - 32].chunks_exact(12);
        let row = |r: &[u8]| (le(r, 0, 4) as u32, le(r, 4, 4) as u32, le(r, 8, 4) as usize);
        let connectors: Vec<(u32, u32, usize)> = rows.map(row).collect();
        let increasing = connectors.windows(2).all(|w| (w[0].0, w[0].1) < (w[1].0, w[1].1));
        if !increasing || connectors.iter().any(|c| c.2 >= frames.len()) {
            return Err(bad("the pose dictionary's connectors are unsorted or have no frame"));
        }
        Ok(Self { frames, connectors, axes })
    }
}

/// Attachments of an assembly: the class of each part and, at entry i of the others, the parent of part i, the parent's
/// connector, its own connector, the joint state, and the joint; reused between assemblies.
#[derive(Clone, Default, PartialEq)]
pub(crate) struct Fields {
    pub class: Vec<u16>,
    pub parent: Vec<u32>,
    pub ca: Vec<u16>,
    pub cb: Vec<u16>,
    pub state: Vec<u32>,
    pub joints: Vec<Joint>,
}

impl Fields {
    /// Make room for assemblies of n parts.
    pub(crate) fn reserve(&mut self, n: usize) {
        if self.class.len() < n {
            self.class.resize(n, 0);
            self.parent.resize(n, 0);
            self.ca.resize(n, 0);
            self.cb.resize(n, 0);
            self.state.resize(n, 0);
            self.joints.resize(n, Joint::default());
        }
    }
}

/// Attachment of a part: `fa` the parent's connector frame, `fb` its own, the joint `family`, whether the chart stores
/// the connector pair `reversed`, and the joint coordinates `q`.
#[derive(Clone, Copy, Default, PartialEq)]
pub struct Joint {
    pub fa: usize,
    pub fb: usize,
    pub family: u8,
    pub reversed: bool,
    pub q: [i32; 3],
}

impl Joint {
    /// Joint coordinates from the parent to the child, as BrickNet's tree edges give them: the chart's, turned when the
    /// chart stores the pair child first.
    pub fn tree_q(&self) -> [i32; 3] {
        if self.reversed {
            turned(self.family, self.q)
        } else {
            self.q
        }
    }
}

/// Joint coordinates `q` of `family` read in the other direction: yaw and slide negated, unless flipped (a flipped
/// joint is its own inverse); ball coordinates, which the chart and BrickNet keep from the in side, unchanged.
pub fn turned(family: u8, q: [i32; 3]) -> [i32; 3] {
    let negated = |degrees: i32| (360 - degrees) % 360;
    match family {
        STUD => [negated(q[0]), q[1], q[2]],
        AXLE if q[0] == 0 => [q[0], negated(q[1]), -q[2]],
        HINGE if q[0] == 0 => [q[0], negated(q[1]), q[2]],
        _ => q,
    }
}

/// Magic of an assembly model file.
pub(crate) const MODEL: &[u8; 8] = b"TWRMODEL";
/// Total frequency 2^CLASS_SCALE of the class tables (the first part's class and new classes).
pub(crate) const CLASS_SCALE: u32 = 20;
/// Connectors of every class of a pose dictionary, in class order and by index within a class (class c's are
/// `list[at[c]..at[c + 1]]`), per class the connector types it can attach to (`attaches`, a bit set), and their SHA-256
/// `fingerprint`, which an assembly model must match.
pub(crate) struct Connectors {
    pub at: Vec<u32>,
    pub list: Vec<Connector>,
    pub types: Types,
    pub attaches: Vec<[u64; 3]>,
    pub fingerprint: [u8; 32],
}

impl Connectors {
    /// Connectors of `count` classes; `classes` is the class of each part id (usize::MAX: none).
    fn new(poses: &PoseDictionary, chart: &Chart, classes: &[usize], count: usize) -> io::Result<Self> {
        let types = Types::new(chart)?;
        let mut of_class = vec![Vec::new(); count];
        for &(part, connector, frame) in &poses.connectors {
            if let Some(&c) = classes.get(part as usize).filter(|&&c| c != usize::MAX) {
                of_class[c].push(types.connector(chart, connector as u16, frame));
            }
        }
        let mut hash = Sha256::new();
        hash.update((count as u64).to_le_bytes());
        hash.update((types.count() as u64).to_le_bytes());
        let (mut at, mut list) = (vec![0u32], Vec::new());
        for connectors in of_class {
            for c in &connectors {
                hash.update([c.id, c.ty].map(u16::to_le_bytes).concat());
            }
            list.extend(connectors);
            at.push(list.len() as u32);
            hash.update((list.len() as u64).to_le_bytes());
        }
        let mut attaches = vec![[0u64; 3]; count];
        for (c, bits) in attaches.iter_mut().enumerate() {
            for connector in &list[at[c] as usize..at[c + 1] as usize] {
                for (word, mask) in types.compatible[connector.ty as usize].iter().enumerate() {
                    bits[word] |= mask;
                }
            }
        }
        let fingerprint = hash.finalize().into();
        Ok(Connectors { at, list, types, attaches, fingerprint })
    }

    /// Connectors of class `class`, by index.
    pub(crate) fn of(&self, class: usize) -> &[Connector] {
        &self.list[self.at[class] as usize..self.at[class + 1] as usize]
    }

    /// Position of connector `id` in the list of class `class`; an error if the pose dictionary lacks it.
    pub(crate) fn position(&self, class: usize, id: u16) -> io::Result<usize> {
        let missing = || bad(&format!("connector {id} of class {class} is not in the pose dictionary"));
        self.of(class).binary_search_by_key(&id, |c| c.id).map_err(|_| missing())
    }
}

/// Flag cells: PARTS positions x PARTS reuse counts x 5 candidate bins.
pub(crate) const FLAGS: usize = PARTS * PARTS * 5;

/// Flag cell of position `i`, reuses so far (each clamped to PARTS - 1), and number of candidate classes (0, 1, 2-3,
/// 4-7, 8+).
pub(crate) fn flag_cell(i: usize, reuses: usize, candidates: usize) -> usize {
    let (i, reuses) = (i.min(PARTS - 1), reuses.min(PARTS - 1));
    (i * PARTS + reuses) * 5 + ((usize::BITS - candidates.leading_zeros()) as usize).min(4)
}

/// Shared assembly model: static tables of every symbol, for the connectors of one pose dictionary and chart set.
pub(crate) struct Model {
    pub root: Table,
    pub parent: Vec<Table>,
    pub connector: Vec<Option<Table>>,
    pub flag: Vec<u16>,
    pub new_of_type: Vec<u16>,
    pub new: Vec<Table>,
    pub stud: Table,
    pub axle: Table,
    pub slides: Vec<Option<Table>>,
    pub ball: [Table; 3],
    pub hinge: Table,
}

impl Model {
    /// Model of the TWRMODEL bytes `raw`, which must be for these connectors.
    pub(crate) fn load(raw: &[u8], cs: &Connectors) -> io::Result<Self> {
        if raw.len() < 40 || &raw[..8] != MODEL {
            return Err(bad("not an assembly model (TWRMODEL)"));
        }
        if raw[8..40] != cs.fingerprint {
            return Err(bad("the assembly model is for other connectors"));
        }
        let mut at = 40;
        let mut next = || -> io::Result<Vec<u32>> {
            let n = word(raw, at, 8)? as usize;
            let body = raw.get(at + 8..at + 8 + 4 * n).ok_or_else(|| bad("the assembly model is truncated"))?;
            at += 8 + 4 * n;
            Ok(body.chunks_exact(4).map(|b| le(b, 0, 4) as u32).collect())
        };
        let (flag, new_of_type, count, present) = (next()?, next()?, next()?, next()?);
        let classes = cs.at.len() - 1;
        let count = *count.first().ok_or_else(|| bad("the assembly model is truncated"))? as usize;
        let mut table = |n: usize, scale| -> io::Result<Table> {
            let t = Table::new(&next()?, scale)?;
            (t.cum.len() == n + 1)
                .then_some(t)
                .ok_or_else(|| bad("the assembly model's tables do not fit its connectors"))
        };
        let root = table(classes, CLASS_SCALE)?;
        let parent: Vec<Table> = (1..PARTS).map(|i| table(i, SCALE)).collect::<io::Result<_>>()?;
        if parent.iter().any(|t| t.cum.windows(2).any(|w| w[0] == w[1])) {
            return Err(bad("the assembly model has a parent of zero frequency"));
        }
        let connector = (0..classes)
            .map(|c| (!cs.of(c).is_empty()).then(|| table(cs.of(c).len(), SCALE)).transpose())
            .collect::<io::Result<_>>()?;
        let new = (0..count).map(|_| table(classes, CLASS_SCALE)).collect::<io::Result<Vec<_>>>()?;
        let (stud, axle) = (table(360, SCALE)?, table(720, SCALE)?);
        let (ball, hinge) = ([table(360, SCALE)?, table(360, SCALE)?, table(360, SCALE)?], table(720, SCALE)?);
        let mut slides: Vec<Option<Table>> = Vec::new();
        for s in present {
            slides.resize_with(s as usize + 1, || None);
            slides[s as usize] = Some(table(s as usize, SCALE)?);
        }
        let valid = flag.len() == FLAGS
            && flag.iter().all(|&p| (1..4096).contains(&p))
            && new_of_type.len() == cs.types.count()
            && new_of_type.iter().all(|&k| (k as usize) < new.len());
        if !valid {
            return Err(bad("the assembly model has invalid reuse probabilities or class tables"));
        }
        let (flag, new_of_type) = (flag.into_iter().map(|p| p as u16), new_of_type.into_iter().map(|k| k as u16));
        Ok(Model {
            root,
            parent,
            connector,
            flag: flag.collect(),
            new_of_type: new_of_type.collect(),
            new,
            stud,
            axle,
            slides,
            ball,
            hinge,
        })
    }
}

/// Assembly being coded: the placed classes (and per connector type those that can attach to it), the reuse count, each
/// part's used connector positions, and the poses placed; reused between assemblies.
pub(crate) struct Context {
    pub seen: Vec<u8>,
    pub distinct: Vec<u16>,
    pub reuses: usize,
    pub used: Vec<Vec<u16>>,
    pub by_type: Vec<Vec<u16>>,
    pub touched: Vec<u16>,
    pub compatible: Vec<u16>,
    pub poses: Vec<Pose>,
}

impl Context {
    /// Empty context for assemblies of up to PARTS parts (`reserve` makes room for more).
    pub(crate) fn new(cs: &Connectors) -> Self {
        Context {
            seen: vec![0; cs.at.len() - 1],
            distinct: Vec::with_capacity(PARTS),
            reuses: 0,
            used: vec![Vec::new(); PARTS],
            by_type: vec![Vec::new(); cs.types.count()],
            touched: Vec::new(),
            compatible: Vec::new(),
            poses: vec![Pose::IDENTITY; PARTS],
        }
    }

    /// Make room for assemblies of n parts.
    pub(crate) fn reserve(&mut self, n: usize) {
        if self.used.len() < n {
            self.used.resize(n, Vec::new());
            self.poses.resize(n, Pose::IDENTITY);
        }
    }

    /// Start an assembly of root class `root`, clearing the last one.
    pub(crate) fn start(&mut self, cs: &Connectors, root: usize) {
        for &d in &self.distinct {
            self.seen[d as usize] = 0;
        }
        for &t in &self.touched {
            self.by_type[t as usize].clear();
        }
        self.used[0].clear();
        self.distinct.clear();
        self.touched.clear();
        self.reuses = 0;
        self.place(cs, root);
    }

    /// Mark class c placed and list it under the types it can attach to.
    pub(crate) fn place(&mut self, cs: &Connectors, c: usize) {
        self.seen[c] = 1;
        self.distinct.push(c as u16);
        for (word, mut bits) in cs.attaches[c].into_iter().enumerate() {
            while bits != 0 {
                let t = word * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                if self.by_type[t].is_empty() {
                    self.touched.push(t as u16);
                }
                self.by_type[t].push(c as u16);
            }
        }
    }

    /// Record part i of class c, attached to part p at the parent's connector position ja by its own jb.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn add(&mut self, cs: &Connectors, i: usize, p: usize, ja: usize, c: usize, jb: usize, new: bool) {
        if new {
            self.place(cs, c);
        } else {
            self.reuses += 1;
        }
        self.used[i].clear();
        self.used[i].push(jb as u16);
        if let Err(at) = self.used[p].binary_search(&(ja as u16)) {
            self.used[p].insert(at, ja as u16);
        }
    }
}

/// Part counts, part ids, poses, and edge rows of decoded graphs, as `Coding::read` gives them.
pub type Decoded = (Vec<usize>, Vec<u8>, Vec<u8>, Vec<u8>);

/// `Coding::place` sink writing part ids (u32) and poses (float64) as little-endian bytes.
pub(crate) fn le_bytes<'a>(ids: &'a mut [u8], matrices: &'a mut [u8]) -> impl FnMut(usize, u32, [f64; 16]) + 'a {
    |i, id, matrix| {
        ids[4 * i..4 * i + 4].copy_from_slice(&id.to_le_bytes());
        for (bytes, value) in matrices[128 * i..128 * i + 128].chunks_exact_mut(8).zip(matrix) {
            bytes.copy_from_slice(&value.to_le_bytes());
        }
    }
}

/// Codec of the trees of an atlas: its pose dictionary, charts, and assembly model (which must be for its
/// connectors), the part id of each class (-1: none), and the class of each part.
pub struct Coding {
    pub(crate) poses: PoseDictionary,
    pub(crate) chart: Chart,
    pub(crate) representatives: Vec<i32>,
    pub(crate) classes: Vec<usize>,
    pub(crate) connectors: Connectors,
    pub(crate) model: Model,
}

impl Coding {
    /// Coding of the atlas in `assets` (its `pose_dictionary`, `charts`, and `assembly_model`); `representatives` gives
    /// each class's part id (-1: none).
    pub fn open(assets: &Path, representatives: Vec<i32>) -> io::Result<Self> {
        let chart = Chart::open(&assets.join("charts"))?;
        let poses = PoseDictionary::open(assets)?;
        if representatives.is_empty() || representatives.len() > 1 << 13 {
            return Err(bad("the catalog has 1 to 8192 part classes"));
        }
        let last = representatives.iter().copied().max().unwrap().max(0) as usize;
        let mut classes = vec![usize::MAX; last + 1];
        for (class, &part) in representatives.iter().enumerate() {
            if part < -1 || (part >= 0 && classes[part as usize] != usize::MAX) {
                return Err(bad("a class representative is invalid or repeated"));
            }
            if part >= 0 {
                classes[part as usize] = class;
            }
        }
        let connectors = Connectors::new(&poses, &chart, &classes, representatives.len())?;
        let model = Model::load(&fs::read(assets.join("assembly_model"))?, &connectors)?;
        Ok(Self { poses, chart, representatives, classes, connectors, model })
    }

    /// Part counts of the first `parts[n]` parts (None: all) of trees `indices[n]` of `set`, then their part ids
    /// (u32), row-major 4 x 4 poses (float64), and edge rows (int64, the sampler's: parent, parent's connector, own
    /// connector, family, and the joint coordinates from the parent), little-endian and concatenated.
    pub fn read(&self, set: &Records, indices: &[usize], parts: Option<&[usize]>) -> io::Result<Decoded> {
        let (mut f, mut cx) = (Fields::default(), self.context());
        let (mut sizes, mut ids, mut poses, mut edges) = (Vec::with_capacity(indices.len()), vec![], vec![], vec![]);
        for (n, &t) in indices.iter().enumerate() {
            let k = self.decode(set.record(t), parts.map(|p| p[n]), &mut f, &mut cx)?;
            let at = ids.len() / 4;
            ids.resize(4 * (at + k), 0);
            poses.resize(128 * (at + k), 0);
            self.place(&f, k, &mut cx, le_bytes(&mut ids[4 * at..], &mut poses[128 * at..]))?;
            for (i, j) in f.joints[1..k].iter().enumerate() {
                let q = j.tree_q().map(i64::from);
                let row = [f.parent[i + 1] as i64, f.ca[i + 1] as i64, f.cb[i + 1] as i64, j.family as i64];
                edges.extend(row.into_iter().chain(q).flat_map(i64::to_le_bytes));
            }
            sizes.push(k);
        }
        Ok((sizes, ids, poses, edges))
    }

    /// Part ids and row-major 4 x 4 poses of the first k parts of assembly t of `set`.
    pub fn poses(&self, set: &Records, t: usize, k: usize) -> io::Result<(Vec<u32>, Vec<[f64; 16]>)> {
        if t >= set.count {
            return Err(bad(&format!("assembly {t} is out of range")));
        }
        let (mut f, mut cx) = (Fields::default(), self.context());
        self.decode(set.record(t), Some(k), &mut f, &mut cx)?;
        let (mut ids, mut matrices) = (vec![0; k], vec![[0.0; 16]; k]);
        self.place(&f, k, &mut cx, |i, id, m| (ids[i], matrices[i]) = (id, m))?;
        Ok((ids, matrices))
    }

    /// Context for coding trees with these connectors.
    pub(crate) fn context(&self) -> Context {
        Context::new(&self.connectors)
    }

    /// Domain of connectors ca and cb, and whether the chart stores it reversed (two fixed ones: `Domain::FIXED`).
    pub(crate) fn domain(&self, ca: Connector, cb: Connector) -> io::Result<(Domain, bool)> {
        let fixed = self.connectors.types.fixed as u16;
        match (ca.ty == fixed, cb.ty == fixed) {
            (true, true) => Ok((Domain::FIXED, false)),
            (false, false) => match self.chart.domain(ca.frame as usize, cb.frame as usize) {
                Some((domain, reversed, _)) => Ok((domain, reversed)),
                None => Err(bad("the connectors cannot mate")),
            },
            _ => Err(bad("a fixed connector is attached to a charted one")),
        }
    }

    /// Decode the fields and joints of the first k parts of a record (None: the whole assembly) into `f`; returns the
    /// parts decoded.
    pub(crate) fn decode(
        &self,
        record: &[u8],
        k: Option<usize>,
        f: &mut Fields,
        cx: &mut Context,
    ) -> io::Result<usize> {
        if k == Some(0) {
            return Err(bad("assembly prefixes have 1 or more parts"));
        }
        // a part past PARTS codes a parent uniform in at least PARTS: over 6 bits of the record's 8 per byte
        let parts = k.unwrap_or(PARTS + 2 * record.len());
        f.reserve(parts);
        cx.reserve(parts);
        let mut reader = Reader::new(record);
        let coded = self.code(&mut reader, f, parts, k.is_none(), cx);
        // a prefix past the assembly's end reads past its record: the parent of the next part lowers the state below L
        if reader.overran() {
            return Err(k.map_or_else(
                || bad("the assembly's record is corrupt"),
                |k| bad(&format!("the assembly has fewer than {k} parts")),
            ));
        }
        coded
    }

    /// Pass the index, part id, and row-major 4 x 4 pose of each of the first k parts to `put`; part i is at
    /// `pose[parent] @ F_a @ J @ inverse(F_b)`.
    pub(crate) fn place(
        &self,
        f: &Fields,
        k: usize,
        cx: &mut Context,
        mut put: impl FnMut(usize, u32, [f64; 16]),
    ) -> io::Result<()> {
        let poses = &mut cx.poses[..k];
        for i in 0..k {
            let part = self.representatives.get(f.class[i] as usize).filter(|&&p| p >= 0);
            let part = *part.ok_or_else(|| bad("a class has no part"))? as u32;
            poses[i] = match i {
                0 => Pose::IDENTITY,
                _ => self.poses.attach(poses[f.parent[i] as usize], &f.joints[i]),
            };
            put(i, part, matrix(poses[i]));
        }
        Ok(())
    }

    /// Code the first k parts of an assembly (`whole`: the decoder stops at the assembly's end, before k); returns the
    /// parts coded. A whole assembly past k parts is an error.
    pub(crate) fn code<I: Io>(
        &self,
        io: &mut I,
        f: &mut Fields,
        k: usize,
        whole: bool,
        cx: &mut Context,
    ) -> io::Result<usize> {
        let (m, cs) = (&self.model, &self.connectors);
        // the buffers hold k parts (`reserve`); slices of them keep their bounds and pointers out of the loop
        let k = k.min(cx.used.len());
        let (classes, parents, cas, cbs) = (&mut f.class[..k], &mut f.parent[..k], &mut f.ca[..k], &mut f.cb[..k]);
        let (states, joints) = (&mut f.state[..k], &mut f.joints[..k]);
        classes[0] = io.table(&m.root, classes[0] as usize) as u16;
        cx.start(cs, classes[0] as usize);
        for i in 1..k {
            let p = match m.parent.get(i - 1) {
                Some(table) => io.table(table, parents[i] as usize),
                None => io.uniform(i, parents[i] as usize),
            };
            parents[i] = p as u32;
            let a = classes[p] as usize;
            let table = m.connector[a].as_ref().ok_or_else(|| bad("a parent has no connectors"))?;
            let ja = if I::ENCODE { cs.position(a, cas[i])? } else { 0 };
            let ja = io.reweighted(table, &cx.used[p], ja);
            let parent = cs.of(a);
            let ca = parent[ja];
            cas[i] = ca.id;
            let ta = ca.ty as usize;
            let candidates = cx.by_type[ta].len();
            let seen = candidates > 0
                && io.bit(
                    m.flag[flag_cell(i, cx.reuses, candidates)] as u32,
                    I::ENCODE && cx.seen[classes[i] as usize] != 0,
                );
            let c = if seen {
                let list = &cx.by_type[ta];
                let at = match I::ENCODE {
                    true => list
                        .iter()
                        .position(|&d| d == classes[i])
                        .ok_or_else(|| bad("a reused class was not placed"))?,
                    false => 0,
                };
                list[io.uniform(candidates, at)] as usize
            } else {
                io.table(&m.new[m.new_of_type[ta] as usize], classes[i] as usize)
            };
            classes[i] = c as u16;
            let child = cs.of(c);
            let mask = cs.types.compatible[ta];
            cx.compatible.resize(child.len(), 0);
            let mut n = 0;
            for (j, connector) in child.iter().enumerate() {
                cx.compatible[n] = j as u16;
                n += usize::from(has(&mask, connector.ty as usize));
            }
            if n == 0 {
                return Err(bad("a part has no compatible connector"));
            }
            let rank = if I::ENCODE {
                let jb = cs.position(c, cbs[i])? as u16;
                cx.compatible[..n].iter().position(|&j| j == jb).ok_or_else(|| bad("the connectors cannot mate"))?
            } else {
                0
            };
            let jb = cx.compatible[io.uniform(n, rank)] as usize;
            let cb = child[jb];
            cbs[i] = cb.id;
            let (domain, reversed) = self.domain(ca, cb)?;
            let q = if I::ENCODE { domain.coordinates(states[i])? } else { [0; 3] };
            let q = match domain.family {
                STUD => [io.table(&m.stud, q[0] as usize) as i32, 0, 0],
                AXLE => {
                    let turn = io.table(&m.axle, (q[0] * 360 + q[1]) as usize) as i32;
                    let index = (q[2] - domain.slide) as usize;
                    let index = match m.slides.get(domain.slides()).and_then(Option::as_ref) {
                        Some(t) => io.table(t, index),
                        None => io.uniform(domain.slides(), index),
                    };
                    [turn / 360, turn % 360, index as i32 + domain.slide]
                }
                BALL => std::array::from_fn(|k| io.table(&m.ball[k], q[k] as usize) as i32),
                HINGE => {
                    let turn = io.table(&m.hinge, (q[0] * 360 + q[1]) as usize) as i32;
                    [turn / 360, turn % 360, 0]
                }
                _ => [0; 3],
            };
            states[i] =
                domain.state(q).ok_or_else(|| bad("the joint coordinates are outside the chart of their connectors"))?
                    as u32;
            joints[i] = Joint { fa: ca.frame as usize, fb: cb.frame as usize, family: domain.family, reversed, q };
            cx.add(cs, i, p, ja, c, jb, !seen);
            if whole && io.ended() {
                return Ok(i + 1);
            }
        }
        match whole {
            true => Err(bad("the assembly's record is corrupt")),
            false => Ok(k),
        }
    }
}

impl Coding {
    /// Fields of a tree given as the sampler's rows: part ids and, per later part, (parent, parent's connector, own
    /// connector, family, and the joint coordinates from the parent).
    fn fields(&self, ids: &[u32], edges: &[[i64; 7]]) -> io::Result<Fields> {
        let mut f = Fields::default();
        f.reserve(ids.len());
        for (i, &part) in ids.iter().enumerate() {
            let class = self.classes.get(part as usize).copied().filter(|&c| c != usize::MAX);
            f.class[i] = class.ok_or_else(|| bad(&format!("part {part} is not a class representative")))? as u16;
        }
        let cs = &self.connectors;
        for (i, &[parent, ca, cb, family, q0, q1, q2]) in edges.iter().enumerate() {
            let parent = usize::try_from(parent)
                .ok()
                .filter(|&p| p <= i)
                .ok_or_else(|| bad("an edge's parent is not an earlier part"))?;
            let connector = |c: i64| u16::try_from(c).map_err(|_| bad("a connector index is out of range"));
            let (ca, cb) = (connector(ca)?, connector(cb)?);
            let (a, b) = (f.class[parent] as usize, f.class[i + 1] as usize);
            let (domain, stored) = self.domain(cs.of(a)[cs.position(a, ca)?], cs.of(b)[cs.position(b, cb)?])?;
            if i64::from(domain.family) != family {
                return Err(bad("an edge's family differs from the chart of its connectors"));
            }
            let q = [q0, q1, q2].map(|q| i32::try_from(q).unwrap_or(i32::MAX));
            let q = if stored { turned(domain.family, q) } else { q };
            let state = domain
                .state(q)
                .ok_or_else(|| bad("the joint coordinates are outside the chart of their connectors"))?;
            (f.parent[i + 1], f.ca[i + 1], f.cb[i + 1], f.state[i + 1]) = (parent as u32, ca, cb, state as u32);
        }
        Ok(f)
    }

    /// Record of a tree of 2 or more parts, given as the sampler's rows; the record must decode to these fields and, if
    /// given, these float64 poses (little-endian bytes), and be under 64 KiB.
    pub fn encode(&self, ids: &[u32], edges: &[[i64; 7]], poses: Option<&[u8]>) -> io::Result<Vec<u8>> {
        let n = ids.len();
        if n < 2 || edges.len() != n - 1 || poses.is_some_and(|poses| poses.len() != n * 128) {
            return Err(bad("a tree has 2 or more parts, an edge per part after the first, and a pose per part"));
        }
        let mut coded = self.fields(ids, edges)?;
        let (mut cx, mut encoder) = (self.context(), Encoder::default());
        cx.reserve(n);
        self.code(&mut encoder, &mut coded, n, false, &mut cx)?;
        let block = encoder.finish();
        let mut back = Fields::default();
        let parts = self.decode(&block, None, &mut back, &mut cx)?;
        let (mut decoded_ids, mut decoded) = (vec![0u8; 4 * n], vec![0u8; 128 * n]);
        if parts == n {
            self.place(&back, n, &mut cx, le_bytes(&mut decoded_ids, &mut decoded))?;
        }
        let expected_ids: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        let same = (0..n).all(|i| {
            (back.class[i], back.parent[i], back.ca[i], back.cb[i], back.state[i], back.joints[i])
                == (coded.class[i], coded.parent[i], coded.ca[i], coded.cb[i], coded.state[i], coded.joints[i])
        });
        if parts != n || !same || decoded_ids != expected_ids || poses.is_some_and(|poses| decoded[..] != *poses) {
            return Err(bad("the tree's record does not decode to its fields and poses"));
        }
        if block.len() >= 1 << 16 {
            return Err(bad("the tree's record is 64 KiB or more"));
        }
        Ok(block)
    }
}
