//! Connection atlas storage.

use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::ops::Range;
use std::path::Path;

use memmap2::Mmap;
use zstd::bulk::Decompressor;

/// Joint family codes.
pub const STUD: u8 = 0;
pub const AXLE: u8 = 1;
pub const BALL: u8 = 2;
pub const HINGE: u8 = 3;
pub const FIXED: u8 = 4; // a fixed connection, whose joint is the identity

/// Samples 0..=360 of a grid row, bit k = sample k.
pub type Row = [u64; 6];

const IDENTITY: usize = 32; // bytes; the atlas identity follows the charts' and index's dimensions and ends the tiles
const MASK_BYTES: usize = 45; // a stored row: samples 0..360 as bits
const FLAG_FRAME: usize = 8192; // bytes decoded per zstd frame of flags
const GRID: usize = 360 * 360; // ball grid rows (x, y)
const GROUP: usize = 1 << 16; // units per offset base

/// Little-endian unsigned integer of `n` bytes at `at`.
pub fn le(data: &[u8], at: usize, n: usize) -> u64 {
    let mut bytes = [0u8; 8];
    bytes[..n].copy_from_slice(&data[at..at + n]);
    u64::from_le_bytes(bytes)
}

/// `le`, checked: an error if `data` does not hold the `n` bytes at `at`.
pub fn word(data: &[u8], at: usize, n: usize) -> io::Result<u64> {
    match at.checked_add(n) {
        Some(end) if n <= 8 && end <= data.len() => Ok(le(data, at, n)),
        _ => Err(bad("truncated data")),
    }
}

/// Error for invalid data.
pub fn bad(message: &str) -> io::Error {
    io::Error::other(message.to_string())
}

/// Header dimensions of an atlas file: the N u64s after its 8-byte magic.
fn dims<const N: usize>(data: &[u8]) -> io::Result<[usize; N]> {
    let mut dims = [0; N];
    for (k, d) in dims.iter_mut().enumerate() {
        *d = word(data, 8 + 8 * k, 8).map_err(|_| truncated())? as usize;
    }
    Ok(dims)
}

/// End of `count` records of `size` bytes from `at`, which must lie within `len` bytes.
fn extent(at: usize, count: usize, size: usize, len: usize) -> io::Result<usize> {
    match count.checked_mul(size).and_then(|n| n.checked_add(at)) {
        Some(end) if end <= len => Ok(end),
        _ => Err(truncated()),
    }
}

/// How to replace atlas files that are truncated or of different releases.
const REFETCH: &str = "run `python -m procbrick fetch`";

/// Error for a truncated atlas file.
fn truncated() -> io::Error {
    bad(&format!("an atlas file is truncated; {REFETCH}"))
}

/// Read-only map of the file at `path`.
pub fn map(path: &Path) -> io::Result<Mmap> {
    // SAFETY: a read-only map of a published file, which is not modified while open
    unsafe { Mmap::map(&File::open(path)?) }
}

/// `map` of a file that starts with `magic` (else the error "not `what`").
pub fn mapped(path: &Path, magic: &[u8], what: &str) -> io::Result<Mmap> {
    let map = map(path)?;
    if !map.starts_with(magic) {
        return Err(bad(&format!("not {what}")));
    }
    Ok(map)
}

/// Joint domain of an ordered pair of connector profiles: its family, rows (`base`: of the first, in the index's
/// numbering), first slide, and table cell (entry, profile indices a and b).
#[derive(Clone, Copy, Default)]
pub struct Domain {
    /// Joint family (`STUD` .. `FIXED`).
    pub family: u8,
    /// Whether rows are by line: a row per (flip, slide) over the yaw, else 360 chart states per row.
    pub by_line: bool,
    pub(crate) base: usize,
    /// Number of stored rows.
    pub chunks: usize,
    /// First slide (LDU; axles).
    pub slide: i32,
    pub(crate) entry: usize,
    pub(crate) a: usize,
    pub(crate) b: usize,
}

impl Domain {
    /// Domain of a fixed connection: one state, the identity joint.
    pub const FIXED: Domain =
        Domain { family: FIXED, by_line: false, base: 0, chunks: 1, slide: 0, entry: 0, a: 0, b: 0 };

    /// Number of slide positions: for an axle (two rows, one per flip, per position) its rows / 2, else 1.
    pub fn slides(&self) -> usize {
        if self.family == AXLE {
            self.chunks / 2
        } else {
            1
        }
    }

    /// Largest |slide| (LDU) of an axle domain, else 0.
    pub fn reach(&self) -> f64 {
        if self.family != AXLE {
            return 0.0;
        }
        let last = self.slide + self.slides() as i32 - 1;
        f64::from(self.slide.abs().max(last.abs()))
    }

    /// Chart state of joint coordinates q (degrees, LDU), if they are in the chart: q0 the yaw (stud) or flip (axle,
    /// hinge), q1 the yaw, q2 the slide (axle); the ball's q is its Euler angles about x, y, and z.
    pub fn state(&self, q: [i32; 3]) -> Option<usize> {
        let (angle, flip) = (|v: i32| (0..360).contains(&v), |v: i32| (0..2).contains(&v));
        let (slide, slides) = (q[2] - self.slide, self.slides() as i32);
        let state = match self.family {
            STUD if angle(q[0]) && q[1] == 0 && q[2] == 0 => q[0],
            AXLE if flip(q[0]) && angle(q[1]) && (0..slides).contains(&slide) => (360 * q[0] + q[1]) * slides + slide,
            BALL if q.into_iter().all(angle) => (360 * q[0] + q[1]) * 360 + q[2],
            HINGE if flip(q[0]) && angle(q[1]) && q[2] == 0 => 360 * q[0] + q[1],
            FIXED if q == [0; 3] => 0,
            _ => return None,
        };
        Some(state as usize)
    }

    /// Joint coordinates of a chart state.
    pub(crate) fn coordinates(&self, state: u32) -> io::Result<[i32; 3]> {
        let (state, s) = (state as usize, self.slides());
        let q = match self.family {
            STUD if state < 360 => [state, 0, 0],
            AXLE if state < 720 * s => [state / s / 360, state / s % 360, state % s],
            BALL if state < 360 * 360 * 360 => [state / (360 * 360), state / 360 % 360, state % 360],
            HINGE if state < 720 => [state / 360, state % 360, 0],
            FIXED if state == 0 => [0; 3],
            _ => return Err(bad("a joint state is outside its chart")),
        };
        let slide = if self.family == AXLE { self.slide } else { 0 };
        Ok([q[0] as i32, q[1] as i32, q[2] as i32 + slide])
    }
}

/// Entry of the charts: the joint domains of a class pair (29 bytes: u64 position, u32 profiles of the second type,
/// u32 rows per domain, i32 first slide, u64 first flag byte, u8 family | 128 when rows are by line).
#[derive(Clone, Copy)]
struct Entry {
    position: usize,
    profiles: usize,
    rows: usize,
    slide: i32,
    flags: usize,
    family: u8,
}

/// Connector charts of a CHARTSET file, decoded: per profile its type, class, and index; the group of each type pair;
/// the groups; and the entries.
pub struct Chart {
    identity: [u8; IDENTITY],
    keys: Vec<[i32; 3]>,
    types: usize,
    pairs: Vec<i32>,
    groups: Vec<[usize; 2]>,
    entries: Vec<Entry>,
}

impl Chart {
    /// Charts of a CHARTSET file.
    pub fn open(path: &Path) -> io::Result<Chart> {
        let map = mapped(path, b"CHARTSET", "connector charts")?;
        let [profile_count, types, group_count, entry_count] = dims(&map)?;
        let profiles = extent(40, 1, IDENTITY, map.len())?;
        let pairs = extent(profiles, profile_count, 12, map.len())?;
        let groups = extent(pairs, types, extent(0, types, 4, usize::MAX)?, map.len())?;
        let entries = extent(groups, group_count, 16, map.len())?;
        let end = extent(entries, entry_count, 29, map.len())?;
        if map.len() != end {
            return Err(bad(&format!("the connector charts have the wrong size; {REFETCH}")));
        }
        let map: &[u8] = &map;
        let word = |at: usize| le(map, at, 4) as u32 as i32;
        let wide = |at: usize| le(map, at, 8) as usize;
        Ok(Chart {
            identity: map[40..profiles].try_into().unwrap(),
            keys: (profiles..pairs).step_by(12).map(|at| [word(at), word(at + 4), word(at + 8)]).collect(),
            types,
            pairs: (pairs..groups).step_by(4).map(word).collect(),
            groups: (groups..entries).step_by(16).map(|at| [wide(at), wide(at + 8)]).collect(),
            entries: (entries..end)
                .step_by(29)
                .map(|e| Entry {
                    position: wide(e),
                    profiles: word(e + 8) as u32 as usize,
                    rows: word(e + 12) as u32 as usize,
                    slide: word(e + 16),
                    flags: wide(e + 20),
                    family: map[e + 28],
                })
                .collect(),
        })
    }

    /// Number of connector profiles; profile p is connector frame p of the pose dictionary.
    pub(crate) fn profile_count(&self) -> usize {
        self.keys.len()
    }

    /// Type, class, and index of profile p (type -1: no chart).
    pub(crate) fn key(&self, p: usize) -> [i32; 3] {
        self.keys[p]
    }

    /// Group of the type pair (x, y), -1 if it is stored reversed or cannot mate.
    fn group(&self, x: usize, y: usize) -> i32 {
        self.pairs[x * self.types + y]
    }

    /// Number of profile types.
    pub(crate) fn types(&self) -> usize {
        self.types
    }

    /// Whether profiles of types x and y can mate: some domain pairs them, in either order.
    pub(crate) fn mates(&self, x: usize, y: usize) -> bool {
        self.group(x, y) >= 0 || self.group(y, x) >= 0
    }

    /// Per profile, the largest reach of the domains its type and class are part of (LDU).
    pub fn reaches(&self) -> Vec<f64> {
        let mut by_class = vec![Vec::new(); self.types];
        for (x, y) in (0..self.types).flat_map(|x| (0..self.types).map(move |y| (x, y))) {
            let Ok(g) = usize::try_from(self.group(x, y)) else {
                continue;
            };
            // group g's entries (class of x, class of y) run to the next group's
            let [first, stride] = self.groups[g];
            let end = self.groups.get(g + 1).map_or(self.entries.len(), |next| next[0]);
            for (k, e) in self.entries.iter().enumerate().take(end).skip(first) {
                let reach =
                    Domain { family: e.family & 127, slide: e.slide, chunks: e.rows, ..Domain::default() }.reach();
                for (t, class) in [(x, (k - first) / stride), (y, (k - first) % stride)] {
                    if by_class[t].len() <= class {
                        by_class[t].resize(class + 1, 0.0);
                    }
                    by_class[t][class] = f64::max(by_class[t][class], reach);
                }
            }
        }
        let reach = |[x, class, _]: [i32; 3]| by_class.get(x as usize)?.get(class as usize).copied();
        (0..self.profile_count()).map(|p| reach(self.key(p)).unwrap_or(0.0)).collect()
    }

    /// Domain of profiles pa and pb (pose dictionary frames), whether it is stored reversed, and its flag bit, if they
    /// can mate.
    pub fn domain(&self, pa: usize, pb: usize) -> Option<(Domain, bool, usize)> {
        let (a, b) = (*self.keys.get(pa)?, *self.keys.get(pb)?);
        if a[0] < 0 || b[0] < 0 {
            return None;
        }
        let reversed = self.group(a[0] as usize, b[0] as usize) < 0;
        let (a, b) = if reversed { (b, a) } else { (a, b) };
        let [first, stride] = self.groups[usize::try_from(self.group(a[0] as usize, b[0] as usize)).ok()?];
        let entry = first + a[1] as usize * stride + b[1] as usize;
        let e = self.entries[entry];
        let (a, b) = (a[2] as usize, b[2] as usize);
        let pair = a * e.profiles + b;
        let domain = Domain {
            family: e.family & 127,
            by_line: e.family & 128 != 0,
            base: e.position + pair * e.rows,
            chunks: e.rows,
            slide: e.slide,
            entry,
            a,
            b,
        };
        // the entry's flags start at its first flag byte, one bit per profile pair
        Some((domain, reversed, 8 * (e.flags + pair / 8) + pair % 8))
    }
}

/// Decoded tile: its frame (palette index planes from `at`, `all` indices), palette, and exceptions (first profile,
/// second profile, row id), sorted.
struct Tile {
    raw: Vec<u8>,
    at: usize,
    all: usize,
    size: usize,
    palette: Vec<u32>,
    exceptions: Vec<(u32, u32, u32)>,
}

impl Tile {
    /// Row id k of the tile (references, then exception rows).
    fn id(&self, k: usize) -> u32 {
        let index = (0..self.size).fold(0, |v, b| v | usize::from(self.raw[self.at + b * self.all + k]) << (8 * b));
        self.palette[index]
    }

    /// Row id of the tile's exception for profiles (a, b), if it lists one.
    fn exception(&self, a: usize, b: usize) -> Option<u32> {
        let key = (a as u32, b as u32);
        let found = self.exceptions.binary_search_by_key(&key, |e| (e.0, e.1)).ok()?;
        Some(self.exceptions[found].2)
    }
}

/// Decoded unit: a frame of bytes (flags or rows) or a tile.
enum Unit {
    Bytes(Vec<u8>),
    Tile(Tile),
}

impl Unit {
    /// Bytes the unit charges to the cache budget.
    fn size(&self) -> usize {
        match self {
            Unit::Bytes(b) => b.len(),
            Unit::Tile(t) => t.raw.len() + 4 * t.palette.len() + 12 * t.exceptions.len(),
        }
    }
}

/// Decoded units by key (flag frames, then pack units) up to `budget` bytes: a dense table of slots (slot + 1, 0 when
/// not held) and per slot its key, unit, and whether it was read since the clock hand passed.
#[derive(Default)]
struct Cache {
    slots: Vec<u32>,
    held: Vec<(usize, Unit, bool)>,
    bytes: usize,
    budget: usize,
    hand: usize,
}

impl Cache {
    /// Unit held for `key`, marked read.
    fn get(&mut self, key: usize) -> &Unit {
        let held = &mut self.held[self.slots[key] as usize - 1];
        held.2 = true;
        &held.1
    }

    /// Hold `unit` for `key`; while over budget, the units the clock hand finds unread since it last passed give way.
    fn insert(&mut self, key: usize, unit: Unit) {
        let size = unit.size();
        while self.bytes > 0 && self.bytes + size > self.budget {
            self.hand %= self.held.len();
            if std::mem::replace(&mut self.held[self.hand].2, false) {
                self.hand += 1;
                continue;
            }
            let (old, data, _) = self.held.swap_remove(self.hand);
            self.slots[old] = 0;
            self.bytes -= data.size();
            if let Some(&(key, _, _)) = self.held.get(self.hand) {
                self.slots[key] = self.hand as u32 + 1;
            }
        }
        self.bytes += size;
        self.held.push((key, unit, true));
        self.slots[key] = self.held.len() as u32;
    }
}

/// Atlas index and tile pack, with a cache of decoded units (hence `&mut self` to read rows).
pub struct Store {
    pub chart: Chart,
    index: Mmap,
    connectors: Range<usize>,
    explicit: Range<usize>,
    flag_bytes: usize,
    flags: Vec<usize>,
    ball_base: usize,
    ball_rows: usize,
    pack: Mmap,
    rows: usize,
    per_frame: usize,
    bases: usize,
    offsets: usize,
    entries: usize,
    sides: Vec<usize>,
    zstd: Decompressor<'static>,
    flag_zstd: Decompressor<'static>,
    cache: Cache,
}

/// Whether bit k of a row is set.
pub fn bit(r: &Row, k: usize) -> bool {
    r[k / 64] >> (k % 64) & 1 == 1
}

/// Row with sample 360 set where a run of samples 0..360 passes 359 (repeating sample 0).
pub fn wrap(mut r: Row) -> Row {
    r[5] &= (1 << 40) - 1;
    if bit(&r, 359) && bit(&r, 0) {
        r[5] |= 1 << 40;
    }
    r
}

/// Samples 0..360 of `r` turned: sample z of the result is sample z + shift (mod 360) of r; sample 360 dropped.
fn turned(r: &Row, shift: usize) -> Row {
    let at = shift % 360;
    if at == 0 {
        return *r;
    }
    // the samples of two turns (r, then r again from bit 360), read from bit `at` on
    let mut two = [0u64; 12];
    for (w, &x) in r.iter().enumerate() {
        two[w] |= x;
        two[w + 5] |= x << 40;
        two[w + 6] |= x >> 24;
    }
    let mut out: Row = std::array::from_fn(|i| {
        let (w, b) = ((at + 64 * i) / 64, (at + 64 * i) % 64);
        if b == 0 {
            two[w]
        } else {
            two[w] >> b | two[w + 1] << (64 - b)
        }
    });
    out[5] &= (1 << 40) - 1;
    out
}

/// (stored rank, shift) of the grid row (x, y) of a ball domain (sample z = stored sample z + shift), folded by the
/// Euler identities and the gimbal lines y = 90 and 270.
fn fold(x: usize, y: usize) -> (usize, usize) {
    let (cx, cy, shift) = match (x, y) {
        (_, 90) => (0, 90, (360 - x) % 360),
        (_, 270) => (0, 270, x),
        _ if x >= 180 => (x - 180, (540 - y) % 360, 180),
        _ => (x, y, 0),
    };
    let rank = if cx == 0 { cy } else { 360 + (cx - 1) * 358 + cy - usize::from(cy > 90) - usize::from(cy > 270) };
    (rank, shift)
}

impl Store {
    /// Whether the joint coordinates q of domain d are admitted.
    pub fn admits(&mut self, d: &Domain, q: [i32; 3]) -> io::Result<bool> {
        let state = d.state(q).ok_or_else(|| bad("pose is outside the sampled domain"))?;
        // the stored row holding q (of a ball, its grid row) and the sample in it
        let (row, z) = match d.family {
            AXLE if d.by_line => (q[0] as usize * d.slides() + (q[2] - d.slide) as usize, q[1] as usize),
            _ => (state / 360, state % 360),
        };
        let rows = if d.family == BALL { self.ball_rows(d, row..row + 1) } else { self.rows(d, row..row + 1) };
        Ok(bit(&rows[0], z))
    }

    /// Atlas in directory `root` (`charts`, `atlas.index`, `atlas.tiles`), decoded units cached up to `budget` bytes.
    pub fn open(root: &Path, budget: usize) -> io::Result<Store> {
        let chart = Chart::open(&root.join("charts"))?;
        let index = mapped(&root.join("atlas.index"), b"ATLASIDX", "an atlas index")?;
        let pack = mapped(&root.join("atlas.tiles"), b"ATLASTIL", "atlas tiles")?;
        let map: &[u8] = &index;
        let [connector_count, explicit_count, flag_bytes, ball_base, ball_rows] = dims(map)?;
        let [rows, units, entry_count, side_count, per_frame] = dims(&pack)?;
        let connectors = extent(48, 1, IDENTITY, map.len())?;
        // the end of the tiles, before the pack's identity
        let tiles = pack.len().checked_sub(IDENTITY).filter(|&end| end >= 48).ok_or_else(truncated)?;
        if map[48..connectors] != pack[tiles..] || chart.identity[..] != map[48..connectors] {
            return Err(bad(&format!("the atlas files are not of one atlas; {REFETCH}")));
        }
        let explicit = extent(connectors, connector_count, 12, map.len())?;
        let flag_table = extent(explicit, explicit_count, 12, map.len())?;
        // the table's offsets: of each flag frame, of the dictionary, and its end
        let flag_frames = flag_bytes.div_ceil(FLAG_FRAME);
        let flag_data = extent(flag_table, flag_frames + 2, 8, map.len())?;
        let flags: Vec<usize> =
            (0..flag_frames + 2).map(|f| flag_data.saturating_add(le(map, flag_table + 8 * f, 8) as usize)).collect();
        if flags.windows(2).any(|w| w[0] > w[1]) || flags[flag_frames + 1] > map.len() {
            return Err(truncated());
        }
        let bases = 48;
        let offsets = extent(bases, units / GROUP + 1, 8, tiles)?;
        let entries = extent(offsets, units + 1, 4, tiles)?;
        let sides = extent(entries, entry_count, 24, tiles)?;
        extent(sides, side_count, 8, tiles)?;
        if per_frame == 0 && rows > 0 {
            return Err(bad("invalid atlas tiles"));
        }
        let store = Store {
            connectors: connectors..explicit,
            explicit: explicit..flag_table,
            flag_bytes,
            flag_zstd: Decompressor::with_dictionary(&map[flags[flag_frames]..flags[flag_frames + 1]])?,
            flags,
            ball_base,
            ball_rows,
            rows,
            per_frame,
            bases,
            offsets,
            entries,
            sides: (0..side_count).map(|k| le(&pack, sides + 8 * k, 8) as usize).collect(),
            zstd: Decompressor::new()?,
            cache: Cache { slots: vec![0; flag_frames + units], budget, ..Default::default() },
            pack,
            chart,
            index,
        };
        // every unit ends within the tiles (one check: the units' offsets increase)
        if units > 0 && store.unit(units - 1).end > tiles {
            return Err(truncated());
        }
        Ok(store)
    }

    /// 12-byte records of an index section.
    fn records(&self, section: &Range<usize>) -> &[[u8; 12]] {
        self.index[section.clone()].as_chunks::<12>().0
    }

    /// Profile of connector `conn` of part `part`, if the atlas has it.
    pub fn profile(&self, part: usize, conn: usize) -> Option<usize> {
        let connectors = self.records(&self.connectors);
        let key = |r: &[u8; 12]| (le(r, 0, 4) as usize, le(r, 4, 4) as usize);
        let found = connectors.binary_search_by_key(&(part, conn), key).ok()?;
        Some(le(&connectors[found], 8, 4) as usize)
    }

    /// Unit cached for `key`, made by `make` if it is not held.
    fn cached(&mut self, key: usize, make: impl FnOnce(&mut Store) -> Unit) -> &Unit {
        if self.cache.slots[key] == 0 {
            let unit = make(self);
            self.cache.insert(key, unit);
        }
        self.cache.get(key)
    }

    /// Frame of bytes cached for `key`, decoded by `make` if it is not held.
    fn bytes(&mut self, key: usize, make: impl FnOnce(&mut Store) -> Vec<u8>) -> &[u8] {
        match self.cached(key, |s| Unit::Bytes(make(s))) {
            Unit::Bytes(bytes) => bytes,
            Unit::Tile(_) => unreachable!("a frame of bytes"),
        }
    }

    /// Tile cached for `key`, decoded by `make` if it is not held.
    fn tile(&mut self, key: usize, make: impl FnOnce(&mut Store) -> Tile) -> &Tile {
        match self.cached(key, |s| Unit::Tile(make(s))) {
            Unit::Tile(tile) => tile,
            Unit::Bytes(_) => unreachable!("a tile"),
        }
    }

    /// Whether flag bit k of the index is set: some sample of its domain is admitted.
    pub fn flag(&mut self, k: usize) -> bool {
        let (byte, f) = (k / 8, k / 8 / FLAG_FRAME);
        let size = FLAG_FRAME.min(self.flag_bytes - FLAG_FRAME * f);
        let frame = self.bytes(f, |s| {
            let raw = s.flag_zstd.decompress(&s.index[s.flags[f]..s.flags[f + 1]], size);
            raw.expect("corrupt atlas flag frame")
        });
        frame[byte % FLAG_FRAME] >> (k % 8) & 1 == 1
    }

    /// Cache key of pack unit u: after the flag frames'.
    fn key(&self, u: usize) -> usize {
        self.flags.len() - 2 + u
    }

    /// Byte range of pack unit u.
    fn unit(&self, u: usize) -> Range<usize> {
        let at = |u: usize| {
            let base = le(&self.pack, self.bases + 8 * (u / GROUP), 8) as usize;
            base + le(&self.pack, self.offsets + 4 * u, 4) as usize
        };
        at(u)..at(u + 1)
    }

    /// Decoded zstd frame of unit u after `skip` header bytes, `size` bytes long.
    fn unzstd(&mut self, u: usize, skip: usize, size: usize) -> Vec<u8> {
        let span = self.unit(u);
        self.zstd.decompress(&self.pack[span.start + skip..span.end], size).expect("corrupt atlas unit")
    }

    /// Samples of row `id` turned: sample z = stored sample z + shift (mod 360), wrapped.
    fn row(&mut self, id: u32, shift: usize) -> Row {
        let (f, at) = (id as usize / self.per_frame, MASK_BYTES * (id as usize % self.per_frame));
        let size = MASK_BYTES * self.per_frame.min(self.rows - f * self.per_frame);
        let frame = self.bytes(self.key(f), |s| s.unzstd(f, 0, size));
        let stored: Row = std::array::from_fn(|w| le(frame, at + 8 * w, 8.min(MASK_BYTES - 8 * w)));
        wrap(turned(&stored, shift))
    }

    /// Tile unit u of n references, decoded.
    fn decode_tile(&mut self, u: usize, n: usize) -> Tile {
        let head = &self.pack[self.unit(u)];
        let (size, unique, count) = (head[0] as usize, le(head, 1, 4) as usize, le(head, 5, 4) as usize);
        let all = n + count;
        let keys = 4 * unique + size * all;
        let raw = self.unzstd(u, 9, keys + 8 * count);
        let planes =
            |at: usize, len: usize, i: usize| (0..4).fold(0u32, |v, b| v | u32::from(raw[at + b * len + i]) << (8 * b));
        let palette: Vec<u32> = (0..unique)
            .scan(0u32, |id, i| {
                *id = id.wrapping_add(planes(0, unique, i));
                Some(*id)
            })
            .collect();
        // exception keys: first profile deltas, second profile deltas from 0 at each new first profile
        let mut key = (0u32, 0u32);
        let mut exceptions = Vec::with_capacity(count);
        for i in 0..count {
            let (da, db) = (planes(keys, count, i), planes(keys + 4 * count, count, i));
            key = if da == 0 { (key.0, key.1.wrapping_add(db)) } else { (key.0.wrapping_add(da), db) };
            exceptions.push((key.0, key.1, 0));
        }
        let mut tile = Tile { raw, at: 4 * unique, all, size, palette, exceptions };
        for i in 0..count {
            tile.exceptions[i].2 = tile.id(n + i);
        }
        tile
    }

    /// Position, root, and quarter turns of profile p on side s.
    fn side(&self, s: usize, p: usize) -> (usize, usize, usize) {
        let at = self.sides[s] + 8 * p;
        let root = le(&self.pack, at + 4, 4) as usize;
        (le(&self.pack, at, 4) as usize, root & ((1 << 30) - 1), root >> 30)
    }

    /// Row id and shift (sample z = stored sample z + shift) of the stored rows `ks` of domain d.
    fn refs(&mut self, d: &Domain, ks: impl Iterator<Item = usize>) -> Vec<(u32, usize)> {
        let e = self.entries + 24 * d.entry;
        let field = |k: usize| le(&self.pack, e + 8 + 4 * k, 4) as usize;
        let (first, edge, sides, height, width) =
            (le(&self.pack, e, 8) as usize, field(0), field(1), field(2), field(3));
        // the table cell of d, and with sides its roots and turns
        let (a, b, ra, ta, tb) = if sides == u32::MAX as usize {
            (d.a, d.b, d.a, 0, 0)
        } else {
            let (a, ra, ta) = self.side(sides >> 16, d.a);
            let (b, _, tb) = self.side(sides & 0xffff, d.b);
            (a, b, ra, ta, tb)
        };
        let (i, j) = (a / edge, b / edge);
        let wide = edge.min(width - j * edge);
        let tall = edge.min(height - i * edge);
        let u = first + i * width.div_ceil(edge) + j;
        let tile = self.tile(self.key(u), |s| s.decode_tile(u, tall * wide * d.chunks));
        let at = ((a % edge) * wide + b % edge) * d.chunks;
        if sides == u32::MAX as usize {
            return ks.map(|k| (tile.id(at + k), 0)).collect();
        }
        // turned by q quarter turns (a quarter turn takes sample z to z + 90)
        let quarter = |q: usize| (360 - 90 * (q % 4)) % 360;
        let found = match tile.exception(d.a, d.b) {
            Some(id) => (id, 0),
            None => match (d.a != ra).then(|| tile.exception(ra, d.b)).flatten() {
                Some(id) => (id, quarter(ta)),
                None => (tile.id(at), quarter(ta + tb)),
            },
        };
        ks.map(|_| found).collect()
    }

    /// Stored rows `rows` of a domain (of a ball domain: its stored rows, before folding).
    pub fn rows(&mut self, d: &Domain, rows: Range<usize>) -> Vec<Row> {
        self.refs(d, rows).into_iter().map(|(id, shift)| self.row(id, shift)).collect()
    }

    /// Grid rows (x, y) of a ball domain at `cells` (indices 360 x + y, increasing); explicit rows replace folded ones.
    pub fn ball_rows(&mut self, d: &Domain, cells: Range<usize>) -> Vec<Row> {
        let first = (d.base - self.ball_base) / self.ball_rows * GRID;
        let key = |r: &[u8; 12]| le(r, 0, 8) as usize;
        let records = self.records(&self.explicit);
        let from = records.partition_point(|r| key(r) < first + cells.start);
        let end = first + cells.end;
        let mut explicit = records[from..].iter().take_while(|r| key(r) < end).peekable();
        // per cell its explicit row, or its stored rank and shift
        let plan: Vec<(Option<u32>, usize, usize)> = cells
            .map(|cell| match explicit.next_if(|r| key(r) == first + cell) {
                Some(r) => (Some(le(r, 8, 4) as u32), 0, 0),
                None => {
                    let (rank, shift) = fold(cell / 360, cell % 360);
                    (None, rank, shift)
                }
            })
            .collect();
        let folded = self.refs(d, plan.iter().filter(|p| p.0.is_none()).map(|p| p.1));
        let mut folded = folded.iter();
        let mut memo = HashMap::new();
        let mut rows = Vec::with_capacity(plan.len());
        for (explicit, _, shift) in plan {
            let (id, shift) = match explicit {
                Some(id) => (id, 0),
                None => (folded.next().expect("a folded row").0, shift),
            };
            rows.push(*memo.entry((id, shift)).or_insert_with(|| self.row(id, shift)));
        }
        rows
    }
}
