//! Atlas parts and connector types.

use crate::store::{bad, Chart, Domain, AXLE, BALL, HINGE, STUD};
use crate::trees::PoseDictionary;
use glamx::{DPose3 as Pose, DVec3 as Vector};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::Path;

/// Connector origins at most this far apart (LDU) can be joined.
pub const REACH: f64 = 3.0;
/// Least cosine between the axes of joined studs, and least absolute cosine for joined hinges.
const ALIGNED: f64 = 0.95;
/// Least absolute cosine between the axes of joined axles.
const AXLE_ALIGNED: f64 = 0.9;
/// Least length an axle counts as (LDU).
pub const AXLE_LENGTH: f64 = 4.0;
/// How far past its length an axle can join another (LDU).
pub const AXLE_SWEEP: f64 = 2.65;
/// Least overlap of joined axles (LDU), less the room the position threshold leaves along the axis.
const AXLE_OVERLAP: f64 = 4.0;
/// Last oriented fixed subtype: fixed connectors of subtypes up to it join only with parallel axes and nearly equal
/// frames.
const ORIENTED: u32 = 3;
/// Least cosine between the axes of joined oriented fixed connectors.
const FIXED_AXIS: f64 = 0.9999;
/// Least trace of the relative rotation of joined oriented fixed connectors (3 for equal frames).
const FIXED_TRACE: f64 = 2.99;

/// Subtype and polarity of a fixed connector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fixed {
    pub subtype: u32,
    pub polarity: bool,
}

impl Fixed {
    /// Whether two fixed connectors can mate: one subtype, opposite polarities.
    pub fn mates(self, other: Fixed) -> bool {
        self.subtype == other.subtype && self.polarity != other.polarity
    }
}

/// Fixed connector in parts.json: its index, subtype, and polarity.
type FixedConnector = (u32, u32, bool);

/// `parts.json`: per collider its id and the SHA-256 of BrickNet's OBJ file; per part its id, collider, and fixed
/// connectors; per connector profile its length (LDU); per collider (in `geometry`'s order) its box [lo, hi] (LDU, part
/// frame) as BrickNet's collision bounds it.
#[derive(Deserialize)]
pub struct Parts {
    pub geometry: Vec<(u32, String)>,
    pub parts: Vec<(u32, u32, Vec<FixedConnector>)>,
    pub lengths: Vec<f64>,
    pub bounds: Vec<[[f64; 3]; 2]>,
}

impl Parts {
    /// Parsed `parts.json` from its bytes; an error unless it has one box per collider.
    pub fn read(raw: &[u8]) -> io::Result<Self> {
        let parts: Parts = serde_json::from_slice(raw).map_err(|e| bad(&format!("invalid parts.json: {e}")))?;
        if parts.bounds.len() != parts.geometry.len() {
            return Err(bad("invalid parts.json: not one box per collider"));
        }
        Ok(parts)
    }

    /// Collider box [lo, hi] of each part, by part id (None: a part parts.json does not list).
    pub fn part_bounds(&self) -> io::Result<Vec<Option<[[f64; 3]; 2]>>> {
        let boxes: HashMap<u32, [[f64; 3]; 2]> =
            self.geometry.iter().map(|g| g.0).zip(self.bounds.iter().copied()).collect();
        let count = self.parts.iter().map(|p| p.0 as usize + 1).max().unwrap_or(0);
        let mut out = vec![None; count];
        for &(id, collider, _) in &self.parts {
            let found = boxes.get(&collider).ok_or_else(|| bad(&format!("part {id} has no collider {collider}")))?;
            out[id as usize] = Some(*found);
        }
        Ok(out)
    }
}

/// Connector of a part: its index, type (`Types`; fixed iff `Types::fixed`), and frame in the pose dictionary.
#[derive(Clone, Copy)]
pub struct Connector {
    pub id: u16,
    pub ty: u16,
    pub frame: u32,
}

/// Connector types: the charts' types, then fixed connectors (`fixed`), which join fixed connectors, then connectors
/// without a chart type (`none`), which join nothing; and which types are compatible, as bit sets.
pub struct Types {
    pub(crate) fixed: usize,
    none: usize,
    pub compatible: Vec<[u64; 3]>,
}

/// Whether bit t of a type set is set.
pub fn has(bits: &[u64; 3], t: usize) -> bool {
    (bits[t / 64] >> (t % 64)) & 1 != 0
}

impl Types {
    /// Connector types of the charts of an atlas.
    pub fn new(chart: &Chart) -> io::Result<Self> {
        let (fixed, none) = (chart.types(), chart.types() + 1);
        if none >= 192 {
            return Err(bad("an atlas has at most 190 connector types"));
        }
        let pair = |a: usize, b: usize| match (a == none || b == none, a == fixed || b == fixed) {
            (true, _) => false,
            (_, true) => a == b,
            _ => chart.mates(a, b),
        };
        let mut compatible = vec![[0u64; 3]; none + 1];
        for (a, bits) in compatible.iter_mut().enumerate() {
            for b in (0..=none).filter(|&b| pair(a, b)) {
                bits[b / 64] |= 1 << (b % 64);
            }
        }
        Ok(Types { fixed, none, compatible })
    }

    /// Connector of index `id` at pose dictionary frame `frame`: frames past the charts' profiles are fixed.
    pub fn connector(&self, chart: &Chart, id: u16, frame: usize) -> Connector {
        let key = (frame < chart.profile_count()).then(|| chart.key(frame));
        let ty = key.map_or(self.fixed, |key| usize::try_from(key[0]).unwrap_or(self.none));
        Connector { id, ty: ty as u16, frame: frame as u32 }
    }

    /// Number of connector types.
    pub fn count(&self) -> usize {
        self.none + 1
    }
}

/// Connector of a part as the assets list it: its `Connector`, frame in the part (from the pose dictionary), chart
/// profile (None: fixed), fixed label, and length (LDU; 0 for fixed connectors).
#[derive(Clone, Copy)]
pub struct PartConnector {
    pub connector: Connector,
    pub pose: Pose,
    pub profile: Option<usize>,
    pub fixed: Option<Fixed>,
    pub length: f64,
}

impl PartConnector {
    /// How the connector pairs: by its fixed label, or by its chart profile.
    pub fn kind(&self) -> Kind {
        match self.fixed {
            Some(fixed) => Kind::Fixed(fixed),
            None => Kind::Profiled(self.connector.frame as usize),
        }
    }
}

/// Parts of the atlas assets in a directory: the pose dictionary, the connector types of the atlas's charts,
/// parts.json (and its bytes), and the connectors of each part.
pub struct Assets {
    pub poses: PoseDictionary,
    pub types: Types,
    pub parts: Parts,
    pub raw: Vec<u8>,
    pub connectors: Vec<Vec<PartConnector>>,
}

/// Assets in directory `dir` (`pose_dictionary` and `parts.json`) of the atlas whose charts are `chart`.
pub fn open(dir: &Path, chart: &Chart) -> io::Result<Assets> {
    let poses = PoseDictionary::open(dir)?;
    let path = dir.join("parts.json");
    let raw = fs::read(&path).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
    let parts = Parts::read(&raw)?;
    let types = Types::new(chart)?;
    let connectors = part_connectors(&poses, chart, &types, &parts)?;
    Ok(Assets { poses, types, parts, raw, connectors })
}

/// Connectors of every part of the pose dictionary, indexed by part id, in connector order: exactly those past
/// the charts' profiles are fixed, labeled by `parts`.
pub fn part_connectors(
    poses: &PoseDictionary,
    chart: &Chart,
    types: &Types,
    parts: &Parts,
) -> io::Result<Vec<Vec<PartConnector>>> {
    if parts.lengths.len() != chart.profile_count() {
        return Err(bad("parts.json and the charts have different connectors"));
    }
    let mut labels = HashMap::new();
    for (id, _, fixed) in &parts.parts {
        for &(connector, subtype, polarity) in fixed {
            labels.insert((*id, connector), Fixed { subtype, polarity });
        }
    }
    let count = poses.connectors.last().map_or(0, |c| c.0 as usize + 1);
    let mut out = vec![Vec::new(); count];
    for &(part, id, frame) in &poses.connectors {
        let connector = types.connector(chart, id as u16, frame);
        let profile = (frame < chart.profile_count()).then_some(frame);
        let fixed = labels.get(&(part, id)).copied();
        if profile.is_some() == fixed.is_some() {
            return Err(bad("parts.json and the pose dictionary have different fixed connectors"));
        }
        let length = profile.map_or(0.0, |p| parts.lengths[p]);
        out[part as usize].push(PartConnector { connector, pose: poses.frames[frame], profile, fixed, length });
    }
    Ok(out)
}

/// Placed connector: its pose, axis (the frame's y), and length.
#[derive(Clone, Copy)]
pub struct Placed {
    pub pose: Pose,
    pub axis: Vector,
    pub length: f64,
}

impl Placed {
    /// Connector of length `length` placed at `pose`, which must be rigid.
    pub fn new(pose: Pose, length: f64) -> Self {
        Placed { pose, axis: pose.rotation * Vector::Y, length }
    }
}

/// Whether connectors a and b of compatible types, of joint family `family`, are joined; axles overlap along the axis
/// of the pair's first connector in the charts (BrickNet's: the axle, bar, or pin), a's unless `reversed`.
pub fn joined(family: u8, a: &Placed, b: &Placed, reversed: bool) -> bool {
    let delta = b.pose.translation - a.pose.translation;
    let dot = a.axis.dot(b.axis);
    let near = delta.length_squared() <= REACH * REACH;
    match family {
        STUD => near && dot >= ALIGNED,
        HINGE => near && dot.abs() >= ALIGNED,
        AXLE => {
            let axis = if reversed { b.axis } else { a.axis };
            let along = delta.dot(axis);
            let off = (delta - axis * along).length_squared();
            let overlap = a.length.max(AXLE_LENGTH) + b.length.max(AXLE_LENGTH) - along.abs();
            let room = (REACH * REACH - off).max(0.0).sqrt();
            dot.abs() >= AXLE_ALIGNED
                && off <= REACH * REACH
                && (overlap >= AXLE_OVERLAP || room >= AXLE_OVERLAP - overlap - 1e-4)
        }
        BALL => near,
        _ => false,
    }
}

/// How a connector pairs: by its fixed label, or by its chart profile (its frame in the pose dictionary).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    Fixed(Fixed),
    Profiled(usize),
}

/// How two connectors whose types mate are joined: fixed connectors of one subtype, or profiled ones by their chart
/// domain (stored reversed or not, with its flag bit for `Store::flag`).
#[derive(Clone, Copy)]
pub enum Pairing {
    Fixed(u32),
    Profiled { domain: Domain, reversed: bool, flag: usize },
}

/// How connectors a and b pair, if their types mate: fixed ones by `Fixed::mates`, profiled ones by `Chart::domain`,
/// never a fixed one with a profiled one.
pub fn pairing(chart: &Chart, a: Kind, b: Kind) -> Option<Pairing> {
    match (a, b) {
        (Kind::Fixed(x), Kind::Fixed(y)) => x.mates(y).then_some(Pairing::Fixed(x.subtype)),
        (Kind::Profiled(pa), Kind::Profiled(pb)) => {
            chart.domain(pa, pb).map(|(domain, reversed, flag)| Pairing::Profiled { domain, reversed, flag })
        }
        _ => None,
    }
}

impl Pairing {
    /// Whether the connectors placed at a and b are joined.
    pub fn joins(&self, a: &Placed, b: &Placed) -> bool {
        match *self {
            Pairing::Fixed(subtype) => fixed_joined(subtype, a.pose, b.pose),
            Pairing::Profiled { domain, reversed, .. } => joined(domain.family, a, b, reversed),
        }
    }
}

/// Whether fixed connectors of one subtype at poses a and b are joined.
pub fn fixed_joined(subtype: u32, a: Pose, b: Pose) -> bool {
    let dot = a.rotation.dot(b.rotation);
    (a.translation - b.translation).length_squared() <= REACH * REACH
        && (subtype > ORIENTED
            || (a.rotation * Vector::Y).dot(b.rotation * Vector::Y) >= FIXED_AXIS
                && 4.0 * dot * dot - 1.0 >= FIXED_TRACE)
}

/// Whether the origins are within `REACH`, or the axes are parallel and one origin is within `REACH` of the other's
/// axis line: a necessary condition of `joined` and `fixed_joined` that needs no types.
pub fn near(a: &Placed, b: &Placed) -> bool {
    let delta = b.pose.translation - a.pose.translation;
    if delta.length_squared() <= REACH * REACH {
        return true;
    }
    let (ax, bx) = (a.axis, b.axis);
    let off_line = |axis: Vector| (delta - axis * delta.dot(axis)).length_squared() <= REACH * REACH;
    ax.dot(bx).abs() >= AXLE_ALIGNED && (off_line(ax) || off_line(bx))
}
