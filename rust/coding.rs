//! Shared rANS coding and indexed records.

use crate::store::{bad, le, map};
use memmap2::Mmap;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::Path;

/// Records per index group; per group the index holds a u64 offset into `data`, then each record's u16 offset from it.
const GROUP: usize = 64;
const GROUP_BYTES: usize = 8 + 2 * GROUP;

/// Total frequency 2^SCALE of every other table.
pub(crate) const SCALE: u32 = 16;
/// Slots 2^WEIGHT_SCALE of a symbol of reweighted frequencies or of weights.
pub(crate) const WEIGHT_SCALE: u32 = 24;
/// Guide bits of a table of over 256 symbols (8 for smaller tables), at most its scale.
pub(crate) const GUIDE: u32 = 12;
/// Down-weighting of the parent's used connectors: each keeps 1 / 2^USED_SHIFT of its weight (at least 1).
pub(crate) const USED_SHIFT: u32 = 4;
/// Lower bound of the rANS state; the state stays in [L, 2^63) and is renormalized by 32-bit words.
pub(crate) const L: u64 = 1 << 31;

/// ceil(2^48 / n) for n < 4096: floor(a / n) = (a * MAGIC[n]) >> 48 exactly for a < 2^36.
pub(crate) static MAGIC: [u64; 4096] = {
    let mut table = [0u64; 4096];
    let mut n = 1;
    while n < 4096 {
        table[n] = (1u64 << 48).div_ceil(n as u64);
        n += 1;
    }
    table
};

/// floor((v << scale) / n) for v <= n, scale <= 24.
#[inline]
pub(crate) fn scaled(v: u32, scale: u32, n: u32) -> u64 {
    let a = (v as u64) << scale;
    if n < 4096 {
        ((a as u128 * MAGIC[n as usize] as u128) >> 48) as u64
    } else {
        a / n as u64
    }
}

/// Slots (start, frequency, scale) of v uniform in [0, n).
#[inline]
pub(crate) fn uniform_span(v: u32, n: u32) -> (u32, u32, u32) {
    let scale = if n <= 256 { 16 } else { 24 };
    let start = scaled(v, scale, n);
    (start as u32, (scaled(v + 1, scale, n) - start) as u32, scale)
}

/// floor(a / t) for a < 2^48 and t < 2^24, from the float reciprocal of t: the product is within 2^-28 of a / t,
/// whose fractional part is zero or at least 1 / t, so one correction makes it exact.
#[inline]
pub(crate) fn floor_div(a: u64, t: u64, inverse: f64) -> u64 {
    let q = (a as f64 * inverse) as u64;
    if q * t > a {
        q - 1
    } else if (q + 1) * t <= a {
        q + 1
    } else {
        q
    }
}

/// Slots (start, frequency) of weight w at cumulative weight `cum` of `total` (below 2^24) in 2^WEIGHT_SCALE.
#[inline]
pub(crate) fn weight_span(cum: u64, w: u64, total: u64) -> (u32, u32) {
    let inverse = 1.0 / total as f64;
    let start = floor_div(cum << WEIGHT_SCALE, total, inverse);
    (start as u32, (floor_div((cum + w) << WEIGHT_SCALE, total, inverse) - start) as u32)
}

/// Frequency f of a used connector, down-weighted (at least 1).
#[inline]
pub(crate) fn shrunk(f: u32) -> u32 {
    (f >> USED_SHIFT).max(1)
}

/// Running totals of `freqs` from 0, one more than `freqs`.
pub(crate) fn cumulative(freqs: &[u32]) -> Vec<u32> {
    std::iter::once(0)
        .chain(freqs.iter().scan(0, |total, &f| {
            *total += f;
            Some(*total)
        }))
        .collect()
}

/// Static distribution: cumulative frequencies of total 2^scale and, per range of 2^shift slots, its first symbol.
pub(crate) struct Table {
    pub scale: u32,
    pub shift: u32,
    pub cum: Vec<u32>,
    pub guide: Vec<u16>,
}

impl Table {
    /// Table of frequencies `freqs` of total 2^scale; an error if they are empty, over 2^16, or of another total.
    pub(crate) fn new(freqs: &[u32], scale: u32) -> io::Result<Self> {
        let cum = cumulative(freqs);
        if freqs.is_empty() || freqs.len() > 1 << 16 || cum[freqs.len()] != 1 << scale {
            return Err(bad("the assembly model has an invalid table"));
        }
        let bits = if freqs.len() <= 256 { 8 } else { GUIDE }.min(scale);
        let shift = scale - bits;
        let mut s = 0;
        let guide = (0..=1u32 << bits)
            .map(|k| {
                let slot = (k << shift).min((1 << scale) - 1);
                while cum[s + 1] <= slot {
                    s += 1;
                }
                s as u16
            })
            .collect();
        Ok(Table { scale, shift, cum, guide })
    }

    /// Symbol whose slots hold `slot`.
    #[inline]
    pub(crate) fn symbol(&self, slot: u32) -> usize {
        let mut s = self.guide[(slot >> self.shift) as usize] as usize;
        while self.cum[s + 1] <= slot {
            s += 1;
        }
        s
    }
}

/// Coding of one symbol: the encoder codes the given symbol and returns it; the decoder returns the decoded one.
pub(crate) trait Io {
    const ENCODE: bool;
    fn peek(&self, scale: u32) -> u32;
    fn advance(&mut self, start: u32, freq: u32, scale: u32);

    /// Symbol s of table t, or the decoded symbol.
    #[inline]
    fn table(&mut self, t: &Table, s: usize) -> usize {
        let s = if Self::ENCODE { s } else { t.symbol(self.peek(t.scale)) };
        self.advance(t.cum[s], t.cum[s + 1] - t.cum[s], t.scale);
        s
    }

    /// Table t with sorted, distinct used symbols down-weighted.
    #[inline]
    fn reweighted(&mut self, t: &Table, used: &[u16], s: usize) -> usize {
        let cum = &t.cum;
        let f = |j: usize| cum[j + 1] - cum[j];
        let removed: u64 = used.iter().map(|&u| (f(u as usize) - shrunk(f(u as usize))) as u64).sum();
        if removed == 0 {
            return self.table(t, s);
        }
        let total = (1 << t.scale) - removed;
        let value = ((self.peek(WEIGHT_SCALE) as u64 + 1) * total - 1) >> WEIGHT_SCALE;
        let mut cut = 0u64;
        let mut found = None;
        for &u in used {
            let u = u as usize;
            let start = cum[u] as u64 - cut;
            let before = if Self::ENCODE { s < u } else { value < start };
            if before {
                break;
            }
            let own = shrunk(f(u)) as u64;
            let selected = if Self::ENCODE { s == u } else { value < start + own };
            if selected {
                found = Some((u, start, own));
                break;
            }
            cut += f(u) as u64 - own;
        }
        let (s, start, own) = found.unwrap_or_else(|| {
            let s = if Self::ENCODE { s } else { t.symbol((value + cut) as u32) };
            (s, cum[s] as u64 - cut, f(s) as u64)
        });
        let (a, b) = weight_span(start, own, total);
        self.advance(a, b, WEIGHT_SCALE);
        s
    }

    /// Symbol s of weights w, or the decoded symbol.
    #[inline]
    fn weighted(&mut self, w: &[u32], s: usize) -> usize {
        let total: u64 = w.iter().map(|&x| x as u64).sum();
        let value = ((self.peek(WEIGHT_SCALE) as u64 + 1) * total - 1) >> WEIGHT_SCALE;
        let (mut at, mut cum) = (0, 0u64);
        while if Self::ENCODE { at < s } else { cum + w[at] as u64 <= value } {
            cum += w[at] as u64;
            at += 1;
        }
        let (a, b) = weight_span(cum, w[at] as u64, total);
        self.advance(a, b, WEIGHT_SCALE);
        at
    }

    /// Bit b with P(1) = p / 4096, or the decoded bit.
    #[inline]
    fn bit(&mut self, p: u32, b: bool) -> bool {
        let b = if Self::ENCODE { b } else { self.peek(12) < p };
        let (start, freq) = if b { (0, p) } else { (p, 4096 - p) };
        self.advance(start, freq, 12);
        b
    }

    /// Value v uniform in [0, n), or the decoded value.
    #[inline]
    fn uniform(&mut self, n: usize, v: usize) -> usize {
        if n <= 1 {
            return 0;
        }
        let scale = if n <= 256 { 16 } else { 24 };
        let v = if Self::ENCODE { v } else { (((self.peek(scale) as u64 + 1) * n as u64 - 1) >> scale) as usize };
        let (start, freq, _) = uniform_span(v as u32, n as u32);
        self.advance(start, freq, scale);
        v
    }

    /// Whether the record is consumed exactly; always false while encoding.
    fn ended(&self) -> bool {
        false
    }
}

/// rANS decoder of a record.
pub(crate) struct Reader<'a> {
    x: u64,
    data: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    /// Decoder of `data`: its state from the first 4 + len % 4 bytes, renormalized if below L.
    pub(crate) fn new(data: &'a [u8]) -> Self {
        let bytes = 4 + data.len() % 4;
        let mut raw = [0u8; 8];
        let n = bytes.min(data.len());
        raw[..n].copy_from_slice(&data[..n]);
        let mut reader = Reader { x: u64::from_le_bytes(raw), data, at: bytes };
        if reader.x < L {
            reader.x = (reader.x << 32) | reader.word() as u64;
        }
        reader
    }

    /// Next 32-bit word; 0 past the record's end, which `overran` reports.
    #[inline]
    fn word(&mut self) -> u32 {
        let w = self.data.get(self.at..self.at + 4).map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()));
        self.at += 4;
        w
    }

    /// Whether the record was consumed exactly: every word read and the initial state reached.
    pub(crate) fn finished(&self) -> bool {
        self.x == L && self.at == self.data.len()
    }

    /// Whether words past the record's end were read.
    pub(crate) fn overran(&self) -> bool {
        self.at > self.data.len()
    }
}

impl Io for Reader<'_> {
    const ENCODE: bool = false;

    fn ended(&self) -> bool {
        self.finished()
    }

    #[inline]
    fn peek(&self, scale: u32) -> u32 {
        (self.x & ((1u64 << scale) - 1)) as u32
    }

    /// Consume the symbol of slots [start, start + freq) of 2^scale, renormalizing below L.
    #[inline]
    fn advance(&mut self, start: u32, freq: u32, scale: u32) {
        self.x = freq as u64 * (self.x >> scale) + (self.x & ((1u64 << scale) - 1)) - start as u64;
        if self.x < L {
            self.x = (self.x << 32) | self.word() as u64;
        }
    }
}

/// rANS encoder: symbols are recorded in order and coded in reverse by `finish`.
#[derive(Default)]
pub(crate) struct Encoder {
    symbols: Vec<(u32, u32, u32)>,
}

impl Encoder {
    /// Record of the recorded symbols, which are cleared: the final state in 4 + len % 4 bytes, then the 32-bit words.
    /// A state of 2^56 or more is split into a word and a high part below L, which the decoder reads back with one
    /// renormalization.
    pub(crate) fn finish(&mut self) -> Vec<u8> {
        let mut x = L;
        let mut words = Vec::with_capacity(self.symbols.len() / 4);
        for &(start, freq, scale) in self.symbols.iter().rev() {
            let freq = freq as u64;
            if x >= ((L >> scale) << 32) * freq {
                words.push(x as u32);
                x >>= 32;
            }
            x = ((x / freq) << scale) + (x % freq) + start as u64;
        }
        if x >= 1 << 56 {
            words.push(x as u32);
            x >>= 32;
        }
        let bytes = if x < 1 << 32 { 4 } else { (64 - x.leading_zeros() as usize).div_ceil(8) };
        let mut out = x.to_le_bytes()[..bytes].to_vec();
        out.extend(words.iter().rev().flat_map(|w| w.to_le_bytes()));
        self.symbols.clear();
        out
    }
}

impl Io for Encoder {
    const ENCODE: bool = true;

    fn peek(&self, _: u32) -> u32 {
        0
    }

    fn advance(&mut self, start: u32, freq: u32, scale: u32) {
        self.symbols.push((start, freq, scale));
    }
}

/// File entry of `manifest.json`.
#[derive(Deserialize)]
struct Asset {
    bytes: usize,
    sha256: String,
}

/// `manifest.json`: the record count and the files.
#[derive(Deserialize)]
struct Manifest {
    count: usize,
    files: BTreeMap<String, Asset>,
}

impl Manifest {
    /// File `name` of the set in `root`, mapped, checked against its size and, with `verify`, its SHA-256.
    fn map(&self, root: &Path, name: &str, verify: bool) -> io::Result<Mmap> {
        let asset = self.files.get(name).ok_or_else(|| bad(&format!("manifest.json lists no file {name}")))?;
        let data = map(&root.join(name))?;
        if data.len() != asset.bytes || asset.bytes == 0 {
            return Err(bad(&format!("{name} has the wrong size")));
        }
        if verify {
            checked(&data, &asset.sha256, name)?;
        }
        Ok(data)
    }
}

/// Check file `name` against its SHA-256 (hex).
fn checked(data: &[u8], sha256: &str, name: &str) -> io::Result<()> {
    match format!("{:x}", Sha256::digest(data)) == sha256 {
        true => Ok(()),
        false => Err(bad(&format!("{name} fails its SHA-256"))),
    }
}

/// Whether some offset of an index wraps; None unless the index (per group a u64 offset, then its records' u16 offsets
/// from it modulo 2^16, the first 0) gives increasing record offsets from 0 within `data` bytes, each record under
/// 64 KiB.
fn groups_cover(index: &[u8], count: usize, data: usize) -> Option<bool> {
    if index.len() != count.div_ceil(GROUP) * GROUP_BYTES {
        return None;
    }
    // each record start above the one before by less than 2^16 (the first 0), checked streaming through the index
    let (mut last, mut wraps) = (None, false);
    for (g, group) in index.chunks_exact(GROUP_BYTES).enumerate() {
        let base = le(group, 0, 8);
        let step = last.map_or(base, |last| base.wrapping_sub(last));
        if group[8..10] != [0, 0] || (last.is_none() && base != 0) || (last.is_some() && !(1..1 << 16).contains(&step))
        {
            return None;
        }
        let (mut start, mut previous) = (base, 0);
        for offset in group[10..].chunks_exact(2).take(GROUP.min(count - g * GROUP) - 1).map(|b| le(b, 0, 2)) {
            let step = offset.wrapping_sub(previous) & 0xffff;
            if step == 0 {
                return None;
            }
            wraps |= offset < previous;
            (start, previous) = (start + step, offset);
        }
        last = Some(start);
    }
    last.is_some_and(|last| last < data as u64 && data as u64 - last < 1 << 16).then_some(wraps)
}

/// Directory of trees: `manifest.json` lists `count` and each file's `bytes` and `sha256`; `data` holds the
/// records and `index` their offsets.
pub struct Records {
    data: Mmap,
    index: Mmap,
    wraps: bool,
    pub count: usize,
}

impl Records {
    /// Trees in directory `root`, its manifest, index, and sizes checked.
    pub fn open(root: &Path) -> io::Result<Self> {
        let manifest: Manifest = serde_json::from_slice(&fs::read(root.join("manifest.json"))?)
            .map_err(|e| bad(&format!("invalid manifest.json: {e}")))?;
        let (data, index) = (manifest.map(root, "data", false)?, manifest.map(root, "index", true)?);
        let wraps = groups_cover(&index, manifest.count, data.len()).ok_or_else(|| bad("invalid index"))?;
        Ok(Self { data, index, wraps, count: manifest.count })
    }

    /// Record of assembly t: from its group's base plus its u16 offset from it and 2^16 per wrap of the offsets
    /// before it (in a set where some offset wraps), to the next record or the next group's base.
    pub(crate) fn record(&self, t: usize) -> &[u8] {
        let (at, k) = (t / GROUP * GROUP_BYTES, t % GROUP);
        let base = le(&self.index, at, 8) as usize;
        let offset = |j: usize| le(&self.index, at + 8 + 2 * j, 2) as usize;
        let wraps = |j: usize| match self.wraps {
            true => (1..=j).filter(|&i| offset(i) < offset(i - 1)).count() << 16,
            false => 0,
        };
        let stop = if t + 1 == self.count {
            self.data.len()
        } else if k + 1 < GROUP {
            base + wraps(k + 1) + offset(k + 1)
        } else {
            le(&self.index, at + GROUP_BYTES, 8) as usize
        };
        &self.data[base + wraps(k) + offset(k)..stop]
    }
}
