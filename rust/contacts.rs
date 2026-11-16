//! Joins of placed connectors.

use crate::inputs::{Inputs, Part};
use crate::parts::{fixed_joined, near, pairing, Placed, AXLE_LENGTH, AXLE_SWEEP};
use crate::store::Store;
use parry3d_f64::bounding_volume::{Aabb, BoundingVolume};
use parry3d_f64::math::{Matrix3, Pose, Vector};

const AXIS: usize = 1; // the sweep axis: assemblies grow along y
const WIDTHS: [f64; 2] = [13.5, 32.0]; // the widest boxes of each band: those of connectors up to 4.1 and 13.3 long

/// Placed connector of a placed instance: its part, its index in the part's connectors, and its placement.
#[derive(Clone, Copy)]
struct Connector {
    instance: usize,
    part: usize,
    index: usize,
    ty: usize,
    placed: Placed,
}

/// Box of half width `radius` about `point`.
fn envelope(point: Vector, radius: f64) -> Aabb {
    Aabb::new(point - Vector::splat(radius), point + Vector::splat(radius))
}

impl Connector {
    /// Box within which a connector can join this one.
    fn bounds(&self) -> Aabb {
        envelope(self.placed.pose.translation, self.placed.length.max(AXLE_LENGTH) + AXLE_SWEEP)
    }
}

/// Connector boxes no wider than a width, sorted by their lower `AXIS` coordinate: a query box meets only boxes whose
/// lower coordinate lies between its own lower one less that width and its upper one.
#[derive(Default)]
struct Band {
    sweep: Vec<f64>,      // the lower `AXIS` coordinate of each box, ascending
    boxes: Vec<[f64; 6]>, // the lower corner, then the upper one
    index: Vec<usize>,
}

impl Band {
    fn insert(&mut self, bounds: &Aabb, index: usize) {
        let (lower, upper) = (bounds.mins.to_array(), bounds.maxs.to_array());
        let at = self.sweep.partition_point(|y| *y <= lower[AXIS]);
        self.sweep.insert(at, lower[AXIS]);
        self.boxes.insert(at, [lower[0], lower[1], lower[2], upper[0], upper[1], upper[2]]);
        self.index.insert(at, index);
    }

    fn remove(&mut self, bounds: &Aabb, index: usize) {
        let first = self.sweep.partition_point(|y| *y < bounds.mins[AXIS]);
        let at = (first..self.index.len()).find(|&at| self.index[at] == index).expect("a placed connector's box");
        self.sweep.remove(at);
        self.boxes.remove(at);
        self.index.remove(at);
    }

    /// Visit the indices of the boxes that meet the query box until `visit` returns true; whether it did.
    fn visit(&self, query: &Aabb, width: f64, visit: &mut impl FnMut(usize) -> bool) -> bool {
        let (lower, upper) = (query.mins.to_array(), query.maxs.to_array());
        let first = self.sweep.partition_point(|y| *y < lower[AXIS] - width);
        for (y, (b, index)) in self.sweep[first..].iter().zip(self.boxes[first..].iter().zip(&self.index[first..])) {
            if *y > upper[AXIS] {
                break;
            }
            // all six bounds compared, without branches
            let meets = (lower[0] <= b[3]) & (upper[0] >= b[0]) & (lower[1] <= b[4]) & (upper[1] >= b[1]);
            if meets & (lower[2] <= b[5]) & (upper[2] >= b[2]) && visit(*index) {
                return true;
            }
        }
        false
    }
}

/// Connector boxes in bands of `WIDTHS`, the wider ones (long axles) apart.
#[derive(Default)]
struct Boxes {
    bands: [Band; 2],
    long: Vec<(Aabb, usize)>,
}

impl Boxes {
    /// Band of boxes as wide as `bounds`, if any.
    fn band(bounds: &Aabb) -> Option<usize> {
        WIDTHS.iter().position(|width| bounds.maxs[AXIS] - bounds.mins[AXIS] <= *width)
    }

    fn insert(&mut self, bounds: Aabb, index: usize) {
        match Self::band(&bounds) {
            Some(band) => self.bands[band].insert(&bounds, index),
            None => self.long.push((bounds, index)),
        }
    }

    fn remove(&mut self, bounds: Aabb, index: usize) {
        match Self::band(&bounds) {
            Some(band) => self.bands[band].remove(&bounds, index),
            None => {
                let at = self.long.iter().position(|&(_, i)| i == index).expect("a placed connector's box");
                self.long.swap_remove(at);
            }
        }
    }

    /// Visit the indices of the boxes that meet `query` until `visit` returns true; whether it did.
    fn visit(&self, query: Aabb, mut visit: impl FnMut(usize) -> bool) -> bool {
        for (band, width) in self.bands.iter().zip(WIDTHS) {
            if !band.sweep.is_empty() && band.visit(&query, width, &mut visit) {
                return true;
            }
        }
        self.long.iter().any(|(bounds, index)| query.intersects(bounds) && visit(*index))
    }
}

/// Placed connectors that can join something, in placement order, and per connector type the boxes of the placed
/// connectors that type can join: a connector queries a single set of boxes.
pub(crate) struct Contacts {
    boxes: Vec<Boxes>, // per connector type
    connectors: Vec<Connector>,
}

/// Indices of the connectors of `part` that can join something.
fn joinable(inputs: &Inputs, part: usize) -> impl Iterator<Item = usize> + '_ {
    let connectors = &inputs.parts[part].connectors;
    (0..connectors.len()).filter(move |&k| !inputs.receivers[connectors[k].ty as usize].is_empty())
}

/// Connector `index` of `part` at `pose`, placed as `instance`.
fn connector(inputs: &Inputs, instance: usize, part: usize, pose: Pose, index: usize) -> Connector {
    let c = inputs.parts[part].connectors[index];
    let frame = c.frame as usize;
    let placed = Placed::new(pose * inputs.poses.frames[frame], inputs.lengths[frame]);
    Connector { instance, part, index, ty: c.ty as usize, placed }
}

/// Whether placed connectors `a` and `b` (of compatible types) are joined.
fn joins(inputs: &Inputs, atlas: &Store, a: &Connector, b: &Connector) -> bool {
    if !near(&a.placed, &b.placed) {
        return false;
    }
    let (ka, kb) = (inputs.parts[a.part].kinds[a.index], inputs.parts[b.part].kinds[b.index]);
    pairing(&atlas.chart, ka, kb).is_some_and(|pairing| pairing.joins(&a.placed, &b.placed))
}

/// Whether a fixed connector of part `a` at pose `ap` joins one of part `b` at pose `bp`.
pub(crate) fn fixed_contact(a: &Part, ap: Pose, b: &Part, bp: Pose, frames: &[Pose]) -> bool {
    let at = |part: &Part, pose: Pose, k: usize| pose * frames[part.connectors[k].frame as usize];
    a.fixed_connectors().any(|(i, x)| {
        b.fixed_connectors().any(|(j, y)| x.mates(y) && fixed_joined(x.subtype, at(a, ap, i), at(b, bp, j)))
    })
}

impl Contacts {
    /// Empty contacts for `types` connector types.
    pub fn new(types: usize) -> Self {
        Self { boxes: (0..types).map(|_| Boxes::default()).collect(), connectors: Vec::new() }
    }

    /// Add the connectors of `part` at `pose`, placed as `instance`, that can join something.
    pub fn add(&mut self, inputs: &Inputs, instance: usize, part: usize, pose: Pose) {
        for index in joinable(inputs, part) {
            let connector = connector(inputs, instance, part, pose, index);
            for &receiver in &inputs.receivers[connector.ty] {
                self.boxes[receiver].insert(connector.bounds(), self.connectors.len());
            }
            self.connectors.push(connector);
        }
    }

    /// Remove the connectors of the last placed instance.
    pub fn pop(&mut self, inputs: &Inputs, instance: usize) {
        while self.connectors.last().is_some_and(|c| c.instance == instance) {
            let connector = self.connectors.pop().unwrap();
            for &receiver in &inputs.receivers[connector.ty] {
                self.boxes[receiver].remove(connector.bounds(), self.connectors.len());
            }
        }
    }

    /// (joined connector pairs, distinct joined instances) of `part` at `pose`, multiple joins to the primary parent
    /// included, counted until `enough` holds for the counts so far. A connector's pose is composed only once its
    /// position meets some box.
    pub fn count_until(
        &self,
        inputs: &Inputs,
        atlas: &Store,
        part: usize,
        pose: Pose,
        mut enough: impl FnMut((usize, usize)) -> bool,
    ) -> (usize, usize) {
        let (mut pairs, mut neighbors) = (0, Vec::new());
        let rotation = Matrix3::from_quat(pose.rotation);
        for index in joinable(inputs, part) {
            let c = inputs.parts[part].connectors[index];
            let frame = c.frame as usize;
            // the translation of `pose * inputs.poses.frames[frame]`
            let at = pose.translation + rotation * inputs.poses.frames[frame].translation;
            let mut a = None;
            let done = self.boxes[c.ty as usize].visit(envelope(at, inputs.lengths[frame].max(AXLE_LENGTH)), |k| {
                let a = a.get_or_insert_with(|| connector(inputs, usize::MAX, part, pose, index));
                let b = &self.connectors[k];
                if !joins(inputs, atlas, a, b) {
                    return false;
                }
                pairs += 1;
                if !neighbors.contains(&b.instance) {
                    neighbors.push(b.instance);
                }
                enough((pairs, neighbors.len()))
            });
            if done {
                break;
            }
        }
        (pairs, neighbors.len())
    }
}
