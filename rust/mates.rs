//! Mating parts of the sampling pool.

use crate::inputs::Part;
use crate::parts::Fixed;
use crate::store::{bad, mapped, word};
use std::collections::HashMap;
use std::io;
use std::path::Path;

/// Per part of the sampling pool, the bit set of the parts it can mate with, and how many there are.
pub(crate) struct Mates {
    rows: Vec<Vec<u64>>,
    pub counts: Vec<usize>,
}

impl Mates {
    /// Mates of the pool's `parts` from the atlas in `assets`.
    pub fn new(parts: &[Part], assets: &Path) -> io::Result<Self> {
        let mates = mapped(&assets.join("mates"), b"MATESBIT", "an atlas mate table")?;
        let covered = word(&mates, 8, 8)? as usize;
        if covered.checked_mul(4 + covered.div_ceil(8)).and_then(|n| n.checked_add(16)) != Some(mates.len()) {
            return Err(bad("invalid atlas mate table extent"));
        }
        let ids = mates[16..16 + 4 * covered].as_chunks::<4>().0;
        let rank = |part| {
            ids.binary_search_by_key(&part, |id| u32::from_le_bytes(*id))
                .map_err(|_| bad("part is outside the atlas's coverage"))
        };
        let can_mate = |a: usize, b: usize| {
            let row = 16 + 4 * covered + a * covered.div_ceil(8);
            mates[row + b / 8] & (1 << (b % 8)) != 0
        };
        let ranks = parts
            .iter()
            .map(|p| if p.covered { rank(p.id).map(Some) } else { Ok(None) })
            .collect::<io::Result<Vec<_>>>()?;
        let mut fixed = HashMap::<Fixed, Vec<usize>>::new();
        for (i, part) in parts.iter().enumerate() {
            for (_, label) in part.fixed_connectors() {
                fixed.entry(label).or_default().push(i);
            }
        }
        let mut rows = Vec::with_capacity(parts.len());
        for (i, part) in parts.iter().enumerate() {
            let mut row = vec![0u64; parts.len().div_ceil(64)];
            if let Some(left) = ranks[i] {
                for (j, rank) in ranks.iter().enumerate() {
                    if rank.is_some_and(|r| can_mate(left, r)) {
                        row[j / 64] |= 1 << (j % 64);
                    }
                }
            }
            for (_, label) in part.fixed_connectors() {
                if let Some(others) = fixed.get(&Fixed { polarity: !label.polarity, ..label }) {
                    for &j in others {
                        row[j / 64] |= 1 << (j % 64);
                    }
                }
            }
            rows.push(row);
        }
        Ok(Self { counts: rows.iter().map(|row| row.iter().map(|v| v.count_ones() as usize).sum()).collect(), rows })
    }

    /// Whether pool parts `a` and `b` can mate.
    pub fn contains(&self, a: usize, b: usize) -> bool {
        self.rows[a][b / 64] & (1 << (b % 64)) != 0
    }
}
