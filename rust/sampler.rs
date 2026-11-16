//! Procedural tree sampler.

use crate::contacts::{fixed_contact, Contacts};
use crate::geometry::{pose, realized, Scene};
use crate::inputs::Inputs;
use crate::mates::Mates;
use crate::parts::{pairing, Pairing};
use crate::python::io_err;
use crate::store::Store;
use crate::store::{bad, Domain, AXLE, BALL, FIXED, HINGE, STUD};
use crate::trees::{matrix, Joint};
use crate::{FastLru, FastMap};
use parry3d_f64::math::Pose;
use pyo3::prelude::*;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Arc;

/// Cumulative distribution over indices, with a guide table for draws in constant expected time.
struct Cdf {
    values: Vec<f64>,
    guide: Vec<u32>, // guide[k]: how many values are at most threshold(k)
}

impl Cdf {
    fn new(values: Vec<f64>) -> Self {
        let mut guide = Vec::with_capacity(values.len());
        let mut count = 0;
        for k in 0..values.len() {
            let threshold = Self::threshold(k, values.len(), values[values.len() - 1]);
            while count < values.len() && values[count] <= threshold {
                count += 1;
            }
            guide.push(count as u32);
        }
        Self { values, guide }
    }

    /// Threshold k of `n`, equally spaced below `total`.
    fn threshold(k: usize, n: usize, total: f64) -> f64 {
        k as f64 / n as f64 * total
    }

    fn total(&self) -> f64 {
        self.values[self.values.len() - 1]
    }

    /// First index whose value exceeds `draw`: `values.partition_point(|v| *v <= draw)`.
    fn above(&self, draw: f64) -> usize {
        let n = self.values.len();
        let mut k = ((draw / self.total() * n as f64) as usize).min(n - 1);
        while k > 0 && Self::threshold(k, n, self.total()) > draw {
            k -= 1;
        }
        let mut index = self.guide[k] as usize;
        while index < n && self.values[index] <= draw {
            index += 1;
        }
        index
    }
}

/// SplitMix64 generator of the sampler's draws.
#[derive(Default)]
struct Random(u64);
impl Random {
    /// Index drawn by the weights of `cdf`; a draw at the total (by rounding) takes the first index that reaches it.
    fn weighted(&mut self, cdf: &Cdf) -> usize {
        let draw = self.index(1 << 53) as f64 / (1u64 << 53) as f64 * cdf.total();
        let chosen = cdf.above(draw);
        if chosen == cdf.values.len() {
            cdf.values.partition_point(|v| *v < cdf.total())
        } else {
            chosen
        }
    }

    /// Whether an event of `probability` happens, at a resolution of 2^-30.
    fn chance(&mut self, probability: f64) -> bool {
        probability > 0.0 && (self.index(1 << 30) as f64) < probability * (1u64 << 30) as f64
    }

    /// Angle in degrees: a multiple of 90 with probability `cardinal`, else uniform in [0, 360).
    fn angle(&mut self, cardinal: f64) -> i32 {
        if self.chance(cardinal) {
            90 * self.index(4) as i32
        } else {
            self.index(360) as i32
        }
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    /// Uniform index below `bound`: the high word of a 64 x 64-bit product, rejecting the few low words that
    /// would bias it (Lemire), so that a division is needed only rarely.
    fn index(&mut self, bound: usize) -> usize {
        let bound = bound as u64;
        let mut product = u128::from(self.next()) * u128::from(bound);
        if (product as u64) < bound {
            let threshold = bound.wrapping_neg() % bound;
            while (product as u64) < threshold {
                product = u128::from(self.next()) * u128::from(bound);
            }
        }
        (product >> 64) as usize
    }
}

/// Axle seat of a connector (the policy's `axle_seats`): the direction of a pin (None: a socket), whether the pin
/// has a collar, and the socket's recess depth on each side.
#[derive(Clone, Copy)]
struct AxleSeat {
    direction: Option<i64>,
    collar: bool,
    depths: [Option<f64>; 2],
}

/// Row of the policy's `axle_seats`: part id, connector index, then the `AxleSeat`.
type SeatRow = (u32, u32, Option<i64>, bool, [Option<f64>; 2]);

/// Slide (LDU) that seats a pin in a socket, if one of connectors `a` and `b` is a pin and the other a socket.
fn seated_slide(a: AxleSeat, b: AxleSeat, lengths: [f64; 2], flip: bool, reversed: bool) -> Option<i32> {
    let (direction, pin, socket, pin_child) = match (a.direction, b.direction) {
        (Some(direction), None) => (direction, a, b, false),
        (None, Some(direction)) => (direction, b, a, true),
        _ => return None,
    };
    let side = if flip { -1 } else { 1 };
    let face = side * direction;
    let [pin_length, socket_length] = if pin_child { [lengths[1], lengths[0]] } else { lengths };
    let recess = if pin.collar { socket.depths[((face + 1) / 2) as usize].unwrap_or(0.0) } else { 0.0 };
    let start = pin_length + recess;
    let slide =
        if pin_child { face as f64 * (socket_length - start) } else { direction as f64 * (start - socket_length) };
    Some((slide * if reversed { side as f64 } else { -1.0 }).round() as i32)
}

/// fraction^power, or 1 when `power` is not positive.
fn ramp(fraction: f64, power: f64) -> f64 {
    if power > 0.0 {
        fraction.powf(power)
    } else {
        1.0
    }
}

/// Probability of accepting a placement with `counts` (joined connector pairs, joined instances) against the
/// policy's `targets` of each, scaled by `growth`: `1 - bias` per missing join, at least `floor`.
fn connection_acceptance(targets: (f64, f64), counts: (usize, usize), growth: f64, bias: f64, floor: f64) -> f64 {
    let missing: f64 = [(targets.0, counts.0), (targets.1, counts.1)]
        .into_iter()
        .map(|(target, count)| (1.0 + (target - 1.0) * growth - count as f64).max(0.0))
        .sum();
    (1.0 - bias).powf(missing).max(floor)
}

/// Parts a retreat removes after `streak` retreats in a row: 2^(streak / 4), leaving at least one of `parts`.
fn retreat_depth(streak: usize, parts: usize) -> usize {
    2usize.saturating_pow((streak / 4).min(usize::BITS as usize) as u32).min(parts.saturating_sub(1))
}

/// Retreat streak after a retreat that leaves `remaining` parts: one more, or 0 once only the root remains.
fn next_retreat(streak: usize, remaining: usize) -> usize {
    if remaining == 1 {
        0
    } else {
        streak.saturating_add(1)
    }
}

/// Options of an assembly: the policy's parameters with the assembly's draw of its controls.
#[derive(FromPyObject)]
#[pyo3(from_item_all)]
struct Options {
    backtrack_after: usize,
    angle_biases: [f64; 4], // cardinal-angle probabilities of (stud, axle, ball, hinge) joints
    reuse_concentration: f64,
    instance_reuse_probability: f64,
    extra_connection_bias: f64,
    extra_connection_ramp: f64,
    connection_targets: (f64, f64),
    connector_prior: f64,
    envelope_penalty: f64,
    envelope_ramp: f64,
    envelope_offset: f64,
    envelope_floor: f64,
    mass_growth_penalty: f64,
    axle_seating_bias: f64,
}

/// Tree: part ids, row-major poses, and the edge attaching each part after the first (parent, parent's connector, own
/// connector, family, and the joint coordinates from the parent).
type Sample = (Vec<u32>, Vec<[f64; 16]>, Vec<[i64; 7]>);

/// Raise a pending Python signal (an interrupt) every 2^16 proposals.
fn interrupted(proposals: &mut u32) -> PyResult<()> {
    *proposals = proposals.wrapping_add(1);
    if *proposals % (1 << 16) == 0 {
        Python::attach(|py| py.check_signals())
    } else {
        Ok(())
    }
}

/// Cumulative weights of `weights` (by pool index) where `keep` holds.
fn cumulative(weights: &[f64], keep: impl Fn(usize) -> bool) -> Cdf {
    let mut sum = 0.0;
    Cdf::new(
        weights
            .iter()
            .enumerate()
            .map(|(p, w)| {
                if keep(p) {
                    sum += w;
                }
                sum
            })
            .collect(),
    )
}

/// Assembly sampler of an atlas, its parts, and the policy's weights and axle seats.
#[pyclass]
pub(crate) struct Sampler {
    atlas: Store,
    inputs: Inputs,
    mates: Mates,
    part_weights: Vec<f64>, // by pool index, divided by the largest
    root_cdf: Cdf,
    weighted_mates: FastLru<usize, Arc<Cdf>>, // per parent part, the weights of its mates
    axle_seats: FastMap<(u32, u32), AxleSeat>,
}

#[pymethods]
impl Sampler {
    /// Sampler of the atlas in `assets` (`cache_bytes` of its units cached) with the colliders of `geometry`
    /// (`collider_cache_bytes` of them cached), and the policy's part and root weights (by part id; the pool is the
    /// parts of positive weight) and axle seats.
    #[new]
    fn new(
        assets: &str,
        geometry: &str,
        cache_bytes: usize,
        collider_cache_bytes: usize,
        part_weights: Vec<f64>,
        root_part_weights: Vec<f64>,
        axle_seats: Vec<SeatRow>,
    ) -> PyResult<Self> {
        let atlas = Store::open(Path::new(assets), cache_bytes).map_err(io_err)?;
        let pool: Vec<u32> = (0..part_weights.len() as u32).filter(|&p| part_weights[p as usize] > 0.0).collect();
        let inputs =
            Inputs::new(&atlas, Path::new(assets), Path::new(geometry), &pool, collider_cache_bytes).map_err(io_err)?;
        let mates = Mates::new(&inputs.parts, Path::new(assets)).map_err(io_err)?;
        // the weights of the pool's parts, divided by the largest
        let normalize = |weights: &[f64]| {
            let pooled: Vec<f64> = inputs.parts.iter().map(|part| weights[part.id as usize]).collect();
            let maximum = pooled.iter().copied().fold(0.0, f64::max);
            pooled.iter().map(|w| w / maximum).collect::<Vec<_>>()
        };
        let part_weights = normalize(&part_weights);
        let root_cdf = cumulative(&normalize(&root_part_weights), |_| true);
        Ok(Self {
            atlas,
            inputs,
            mates,
            part_weights,
            root_cdf,
            weighted_mates: FastLru::with_hasher(NonZeroUsize::new(64).unwrap(), Default::default()),
            axle_seats: axle_seats
                .into_iter()
                .map(|(part, connector, direction, collar, depths)| {
                    ((part, connector), AxleSeat { direction, collar, depths })
                })
                .collect(),
        })
    }

    /// Assembly `seed` of `size` parts with `options` (see `Options`), sampled outside the GIL.
    fn sample(&mut self, py: Python<'_>, size: usize, seed: u64, options: Options) -> PyResult<Sample> {
        if size == 0 {
            return Err(io_err(bad("trees have at least one part")));
        }
        py.detach(|| self.run(size, seed, &options))
    }

    /// Whether parts `a` and `b` of the pool collide at row-major poses `ma` and `mb`.
    fn collides(&mut self, a: u32, ma: Vec<f64>, b: u32, mb: Vec<f64>) -> PyResult<bool> {
        let parts = &self.inputs.parts;
        let mesh = |id: u32| {
            let index = parts.binary_search_by_key(&id, |p| p.id);
            index.map(|i| parts[i].mesh).map_err(|_| io_err(bad("part is outside the sampling pool")))
        };
        let (ia, ib) = (mesh(a)?, mesh(b)?);
        let ga = self.inputs.geometry.get(ia)?;
        self.inputs.geometry.blocked(&ga, ia, pose(&ma)?, ib, pose(&mb)?)
    }
}

/// Proposed placement: the parent instance, the child's pool index, the parent's and the child's connector, the joint,
/// its domain (`Domain::FIXED` for fixed connectors), and the child's pose.
struct Proposal {
    parent: usize,
    child: usize,
    connectors: [u16; 2],
    joint: Joint,
    domain: Domain,
    world: Pose,
}

/// Assembly as it grows: its scene and placed connectors, the edges, per placed instance the weights of its part's
/// mates (computed when it is first a parent), per parent part the reuse choices, and the backtracking state.
struct Growth {
    scene: Scene,
    contacts: Contacts,
    edges: Vec<[i64; 7]>,
    laws: Vec<Option<Arc<Cdf>>>,
    reuse_choices: FastMap<usize, (Vec<usize>, Vec<usize>)>,
    stalled: usize,
    expected_failures: usize,
    retreats: usize,
}

impl Growth {
    /// Count a blocked placement; after enough in a row, remove the last parts, more after each retreat in a row.
    fn retreat(&mut self, inputs: &Inputs, backtrack_after: usize) {
        self.stalled += 1;
        let patience = backtrack_after.max(self.expected_failures.saturating_mul(4));
        if self.stalled >= patience && self.scene.placed.len() > 1 {
            let depth = retreat_depth(self.retreats, self.scene.placed.len());
            for _ in 0..depth {
                self.contacts.pop(inputs, self.scene.placed.len() - 1);
                self.scene.pop();
                self.edges.pop();
                self.laws.pop();
            }
            self.reuse_choices.clear();
            // a fresh suffix must not inherit an escalation that erases every later branch
            self.retreats = next_retreat(self.retreats, self.scene.placed.len());
            if depth > 1 {
                self.expected_failures = 0;
            }
            self.stalled = 0;
        }
    }
}

impl Sampler {
    /// Weights of the mates of `part`, cached per part.
    fn mate_weights(&mut self, part: usize) -> Arc<Cdf> {
        let (mates, weights) = (&self.mates, &self.part_weights);
        let law =
            self.weighted_mates.get_or_insert(part, || Arc::new(cumulative(weights, |p| mates.contains(part, p))));
        Arc::clone(law)
    }

    /// Placement drawn by the policy: a parent, a mate of it, a connector pair whose types mate and whose domain has
    /// an admitted sample, and joint coordinates; None when a draw is rejected.
    fn propose(&mut self, g: &mut Growth, random: &mut Random, o: &Options) -> Option<Proposal> {
        let n = g.scene.placed.len();
        let parent = random.index(n);
        let reuse = n as f64 / (o.reuse_concentration + n as f64);
        let part = g.scene.placed[parent].part;
        let mates = &self.mates;
        let scene = &g.scene;
        let (types, instances) = g.reuse_choices.entry(part).or_insert_with(|| {
            let (mut types, mut instances) = (Vec::new(), Vec::new());
            for placed in scene.placed.iter().filter(|p| mates.contains(part, p.part)) {
                if placed.first {
                    types.push(placed.part);
                }
                instances.push(placed.part);
            }
            (types, instances)
        });
        let law: &Cdf = g.laws[parent].get_or_insert_with(|| self.mate_weights(part));
        let type_mass = reuse * (1.0 - o.instance_reuse_probability) * f64::from(!types.is_empty());
        let instance_mass = reuse * o.instance_reuse_probability * f64::from(!instances.is_empty());
        let old_mass = type_mass + instance_mass;
        let new_mass = if law.total() > 0.0 { 1.0 - reuse } else { 0.0 };
        if old_mass + new_mass == 0.0 {
            return None;
        }
        let child = if random.chance(old_mass / (old_mass + new_mass)) {
            let pool = if random.chance(instance_mass / old_mass) { instances } else { types };
            pool[random.index(pool.len())]
        } else {
            random.weighted(law)
        };
        let (a, b) = (&self.inputs.parts[part], &self.inputs.parts[child]);
        if !random.chance(b.connectors.len() as f64 / (b.connectors.len() as f64 + o.connector_prior)) {
            return None;
        }
        let (ia, ib) = (random.index(a.connectors.len()), random.index(b.connectors.len()));
        let (ca, cb) = (a.connectors[ia], b.connectors[ib]);
        let (fa, fb) = (ca.frame as usize, cb.frame as usize);
        let (domain, reversed) = match pairing(&self.atlas.chart, a.kinds[ia], b.kinds[ib])? {
            Pairing::Fixed(_) => (Domain::FIXED, false),
            Pairing::Profiled { domain, reversed, flag } => {
                if !self.atlas.flag(flag) {
                    return None;
                }
                (domain, reversed)
            }
        };
        let cardinal = o.angle_biases.get(domain.family as usize).copied().unwrap_or(0.0);
        let mut q = match domain.family {
            STUD => [random.angle(cardinal), 0, 0],
            AXLE => {
                [random.index(2) as i32, random.angle(cardinal), domain.slide + random.index(domain.slides()) as i32]
            }
            BALL => [random.angle(cardinal), random.angle(cardinal), random.angle(cardinal)],
            HINGE => [random.index(2) as i32, random.angle(cardinal), 0],
            _ => [0; 3],
        };
        if domain.family == AXLE && random.chance(o.axle_seating_bias) {
            let lengths = [self.inputs.lengths[fa], self.inputs.lengths[fb]];
            let preferred = self
                .axle_seats
                .get(&(a.id, u32::from(ca.id)))
                .zip(self.axle_seats.get(&(b.id, u32::from(cb.id))))
                .and_then(|(&a, &b)| seated_slide(a, b, lengths, q[0] != 0, reversed))
                .unwrap_or_else(|| match random.index(3) {
                    0 => 0,
                    1 => (lengths[0] - lengths[1]).abs().round() as i32,
                    _ => -(lengths[0] - lengths[1]).abs().round() as i32,
                });
            if (domain.slide..domain.slide + domain.slides() as i32).contains(&preferred) {
                q[2] = preferred;
            }
        }
        let joint = Joint { fa, fb, family: domain.family, reversed, q };
        let world = self.inputs.poses.attach(g.scene.placed[parent].pose, &joint);
        Some(Proposal { parent, child, connectors: [ca.id, cb.id], joint, domain, world })
    }

    /// Assembly of `size` parts drawn with `seed` and `o`: a root, then accepted proposals, retreating when stalled.
    fn run(&mut self, size: usize, seed: u64, o: &Options) -> PyResult<Sample> {
        let mut random = Random(seed);
        let mut g = Growth {
            scene: Scene::default(),
            contacts: Contacts::new(self.inputs.receivers.len()),
            edges: Vec::with_capacity(size - 1),
            laws: Vec::with_capacity(size),
            reuse_choices: FastMap::default(),
            stalled: 0,
            expected_failures: 0,
            retreats: 0,
        };
        let mut proposals = 0;
        // a root with a mate in the pool (any root for an assembly of one part)
        loop {
            interrupted(&mut proposals)?;
            let root = random.weighted(&self.root_cdf);
            if size == 1 || self.mates.counts[root] > 0 {
                let mesh = self.inputs.geometry.get(self.inputs.parts[root].mesh)?;
                g.scene.add(root, self.inputs.parts[root].mesh, Pose::IDENTITY, mesh);
                g.contacts.add(&self.inputs, 0, root, Pose::IDENTITY);
                g.laws.push(None);
                break;
            }
        }
        let mut high_water = 1;
        // the growth ramps (powers of the filled fraction) at the scene size they were last computed for
        let mut ramps = (usize::MAX, [0.0; 2]);
        while g.scene.placed.len() < size {
            interrupted(&mut proposals)?;
            let n = g.scene.placed.len();
            if ramps.0 != n {
                let fraction = n as f64 / size as f64;
                ramps = (n, [ramp(fraction, o.envelope_ramp), ramp(fraction, o.extra_connection_ramp)]);
            }
            let [envelope_ramp, extra_ramp] = ramps.1;
            let Some(Proposal { parent, child, connectors, joint, domain, world }) =
                self.propose(&mut g, &mut random, o)
            else {
                continue;
            };
            let b = &self.inputs.parts[child];
            let bounds = match b.bounds.get() {
                Some(bounds) => *bounds,
                None => {
                    let bounds = self.inputs.geometry.bounds_of(b.mesh)?;
                    *b.bounds.get_or_init(|| bounds)
                }
            };
            let penalty = o.envelope_penalty * (n.saturating_sub(1) as f64 + o.envelope_offset) * envelope_ramp;
            let cost = penalty * g.scene.expansion(bounds, world) + o.mass_growth_penalty * g.scene.mass_growth(bounds);
            let acceptance = if cost == 0.0 { 1.0 } else { (-cost).exp() };
            if !random.chance(acceptance.max(o.envelope_floor)) {
                continue;
            }
            // the acceptance never falls as joins are found (the floor is positive), so draw first and stop counting
            // once it passes
            let draw = random.index(1 << 30) as f64;
            let passes = |counts| {
                let (targets, bias, floor) = (o.connection_targets, o.extra_connection_bias, o.envelope_floor);
                draw < connection_acceptance(targets, counts, extra_ramp, bias, floor) * (1u64 << 30) as f64
            };
            if !passes(g.contacts.count_until(&self.inputs, &self.atlas, child, world, passes)) {
                continue;
            }
            // independent gates, cheapest per rejection first: the atlas lookup rejects few placements for its cost
            if domain.family != FIXED && !self.atlas.admits(&domain, joint.q).map_err(io_err)? {
                continue;
            }
            let collider = self.inputs.geometry.get(b.mesh)?;
            let (parts, frames) = (&self.inputs.parts, &self.inputs.poses.frames);
            let blocked = g.scene.blocked(&collider, b.mesh, world, &mut self.inputs.geometry, |index, other| {
                index == parent || fixed_contact(&parts[child], world, &parts[other.part], other.pose, frames)
            })?;
            if blocked {
                g.retreat(&self.inputs, o.backtrack_after);
                continue;
            }
            // the parent rarely blocks a placement its atlas domain admits, so it is tested after the rest of the scene
            if domain.family != FIXED {
                let host = &g.scene.placed[parent];
                let geometry = &mut self.inputs.geometry;
                if geometry.blocked(&host.collider, host.mesh, host.realized, b.mesh, realized(world))? {
                    continue;
                }
            }
            g.expected_failures = (3 * g.expected_failures + g.stalled) / 4 + 1;
            g.stalled = 0;
            let [q0, q1, q2] = joint.tree_q().map(i64::from);
            let [ca, cb] = connectors.map(i64::from);
            g.edges.push([parent as i64, ca, cb, domain.family.into(), q0, q1, q2]);
            g.contacts.add(&self.inputs, g.scene.placed.len(), child, world);
            g.scene.add(child, b.mesh, world, collider);
            g.laws.push(None);
            if g.scene.placed.len() > high_water {
                high_water = g.scene.placed.len();
                g.retreats = 0;
            }
            g.reuse_choices.clear();
        }
        let ids = g.scene.placed.iter().map(|p| self.inputs.parts[p.part].id).collect();
        Ok((ids, g.scene.placed.iter().map(|p| matrix(p.pose)).collect(), g.edges))
    }
}
