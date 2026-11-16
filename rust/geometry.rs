//! Convex collision queries.

use crate::geometry_pack::Packed;
use crate::python::io_err;
use crate::store::{bad, le, word};
use crate::trees;
use crate::{FastLru, FastMap};
use parry3d_f64::bounding_volume::{Aabb, BoundingVolume};
use parry3d_f64::math::{Matrix3, Pose, Rotation, Vector};
use parry3d_f64::query::gjk::{self, CsoPoint, GJKResult, VoronoiSimplex};
use parry3d_f64::shape::SupportMap;
use parry3d_f64::transformation::try_convex_hull;
use pyo3::prelude::*;
use std::collections::HashSet;
use std::io;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, Weak};

const TAU: f64 = 1.0; // LDU, the penetration depth placed colliders may reach
const MARGIN: f64 = 1e-6; // beyond rounding: the fast tests settle only decisions this clear of TAU
const EXPANSIONS: usize = 24; // polytope expansions when bracketing a penetration depth
const GJK_ITERATIONS: usize = 32; // before GJK with a TAU margin leaves a hull pair to the general tests
const LEAF: usize = 8; // the hull boxes per leaf, and leaves per group, of a collider's hull hierarchy
const AXES: usize = 1 << 18; // separating axes remembered per generation of the memory of collider pairs
const BLOCKERS: usize = 1 << 14; // blocked placements remembered before the memory is cleared
const PAIRS: usize = 1 << 14; // collider pair decisions remembered, by exact relative pose

/// Feature bytes of the colliders the collider cache holds, shared by the colliders it decodes.
type Charge = Arc<AtomicU64>;

/// Pose of a rigid row-major 4 x 4 matrix, checked (for `Sampler::collides`).
pub(crate) fn pose(values: &[f64]) -> PyResult<Pose> {
    let values: &[f64; 16] =
        values.try_into().map_err(|_| io_err(bad("pose must be a finite row-major 4 x 4 matrix")))?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err(io_err(bad("pose must be a finite row-major 4 x 4 matrix")));
    }
    let column = |j: usize| Vector::new(values[j], values[4 + j], values[8 + j]);
    let rotation = Matrix3::from_cols(column(0), column(1), column(2));
    let gram = rotation.transpose() * rotation - Matrix3::IDENTITY;
    if gram.to_cols_array().iter().any(|v| v.abs() > 1e-10)
        || (rotation.determinant() - 1.0).abs() > 1e-10
        || values[12..15].iter().any(|v| v.abs() > 1e-12)
        || (values[15] - 1.0).abs() > 1e-12
    {
        return Err(io_err(bad("pose must be rigid")));
    }
    Ok(trees::pose(values))
}

/// Bits of a pose's translation and rotation: an exact key.
fn bits(p: Pose) -> [u64; 7] {
    let (t, r) = (p.translation.to_array(), p.rotation.to_array());
    std::array::from_fn(|i| if i < 3 { t[i].to_bits() } else { r[i - 3].to_bits() })
}

/// Pose as decoded from its row-major matrix, rounding included: what the collision tests take.
pub(crate) fn realized(p: Pose) -> Pose {
    trees::pose(&trees::matrix(p))
}

/// Volume of the intersection of two boxes.
fn overlap(a: &Aabb, b: &Aabb) -> f64 {
    let (lower, upper, other_lower, other_upper) = (a.mins, a.maxs, b.mins, b.maxs);
    (0..3).map(|k| (upper[k].min(other_upper[k]) - lower[k].max(other_lower[k])).max(0.0)).product()
}

/// Distance between two boxes (0 when they meet).
fn gap(a: Aabb, b: Aabb) -> f64 {
    (a.mins - b.maxs).max(b.mins - a.maxs).max(Vector::ZERO).length()
}

/// Convex component of a collider: its range of the collider's vertices, their bounds, and its separating-axis
/// features, built when first needed.
struct Hull {
    first: u32,
    end: u32,
    bounds: Aabb,
    features: OnceLock<Result<Features, String>>,
}

/// Hull with its collider's vertices and feature bytes: what the collision tests take.
#[derive(Clone, Copy)]
struct Shape<'a> {
    hull: &'a Hull,
    vertices: &'a [Vector],
    charge: &'a Charge,
    account: &'a Account,
}

/// Bytes of a collider's built features, and whether the collider cache holds (and so is charged for) them.
#[derive(Default)]
struct Account {
    features: AtomicU64,
    cached: AtomicBool,
}

/// Separating-axis features of a hull: its unique face normals and edge directions, and a ball inside it (`center`,
/// `radius`; radius 0 for a flat piece).
struct Features {
    normals: Vec<Vector>,
    edges: Vec<Vector>,
    center: Vector,
    radius: f64,
}

/// Vertex of `vertices` farthest along `dir`.
fn support(vertices: &[Vector], dir: Vector) -> Vector {
    let mut best = (vertices[0], vertices[0].dot(dir));
    for vertex in &vertices[1..] {
        let along = vertex.dot(dir);
        if along > best.1 {
            best = (*vertex, along);
        }
    }
    best.0
}

/// What GJK with a TAU margin finds of a CSO (the configuration-space obstacle: the Minkowski difference of the two
/// hulls), by its support function: a direction along which it reaches at most TAU past the origin (the shapes overlap
/// by at most TAU along it), or a tetrahedron of its points around the origin.
enum Gjk {
    Shallow(Vector),
    Enclosed([Vector; 4]),
    Unresolved,
}

/// GJK over the CSO of `support`, from direction `start`, stopping at the first direction along which it is shallow.
fn gjk(support: impl Fn(Vector) -> Vector, start: Vector) -> Gjk {
    let mut dir = if start.length_squared() > 0.0 { start } else { Vector::X };
    let mut simplex = [Vector::ZERO; 4];
    let mut count = 0;
    for _ in 0..GJK_ITERATIONS {
        let point = support(dir);
        if point.dot(dir) <= (TAU - MARGIN) * dir.length() {
            return Gjk::Shallow(dir);
        }
        simplex[count] = point;
        count += 1;
        // the face of the simplex nearest the origin and the direction from it towards the origin (newest point last)
        match count {
            1 => dir = -point,
            2 => {
                let (a, b) = (simplex[1], simplex[0]);
                let (ab, ao) = (b - a, -a);
                if ab.dot(ao) > 0.0 {
                    dir = ab.cross(ao).cross(ab);
                } else {
                    (simplex[0], count, dir) = (a, 1, ao);
                }
            }
            3 => (count, dir) = triangle(&mut simplex),
            _ => {
                let (a, b, c, d) = (simplex[3], simplex[2], simplex[1], simplex[0]);
                let ao = -a;
                let (abc, acd, adb) = ((b - a).cross(c - a), (c - a).cross(d - a), (d - a).cross(b - a));
                // each face normal oriented away from the vertex opposite it
                let outward =
                    |normal: Vector, opposite: Vector| if normal.dot(opposite - a) > 0.0 { -normal } else { normal };
                let faces = [(outward(abc, d), [c, b, a]), (outward(acd, b), [d, c, a]), (outward(adb, c), [b, d, a])];
                match faces.iter().find(|(normal, _)| normal.dot(ao) > 0.0) {
                    None => return Gjk::Enclosed(simplex),
                    Some(&(_, face)) => {
                        simplex[..3].copy_from_slice(&face);
                        (count, dir) = triangle(&mut simplex);
                    }
                }
            }
        }
        let length = dir.length_squared();
        if length.is_nan() || length <= 1e-24 {
            return Gjk::Unresolved;
        }
    }
    Gjk::Unresolved
}

/// Nearest feature to the origin of triangle simplex[..3] (newest point last), reduced in place: its point count
/// and the direction from it towards the origin.
fn triangle(simplex: &mut [Vector; 4]) -> (usize, Vector) {
    let (a, b, c) = (simplex[2], simplex[1], simplex[0]);
    let (ab, ac, ao) = (b - a, c - a, -a);
    let abc = ab.cross(ac);
    let edge = |simplex: &mut [Vector; 4], other: Vector, along: Vector| {
        if along.dot(ao) > 0.0 {
            (simplex[0], simplex[1]) = (other, a);
            (2, along.cross(ao).cross(along))
        } else {
            simplex[0] = a;
            (1, ao)
        }
    };
    if abc.cross(ac).dot(ao) > 0.0 {
        if ac.dot(ao) > 0.0 {
            (simplex[0], simplex[1]) = (c, a);
            (2, ac.cross(ao).cross(ac))
        } else {
            edge(simplex, b, ab)
        }
    } else if ab.cross(abc).dot(ao) > 0.0 {
        edge(simplex, b, ab)
    } else if abc.dot(ao) > 0.0 {
        (3, abc)
    } else {
        (simplex[0], simplex[1]) = (b, c);
        (3, -abc)
    }
}

/// Whether the penetration depth exceeds TAU, bracketed by expanding the CSO polytope from a GJK tetrahedron
/// around the origin (EPA): its closest face is a lower bound, the CSO support along that face an upper bound.
/// None when the bracket still holds TAU after a bounded expansion; the exact separating-axis test decides then.
fn penetrates(
    support: impl Fn(Vector) -> Vector,
    tetrahedron: &[Vector; 4],
    margin: f64,
    found: &mut Option<Vector>,
) -> Option<bool> {
    // fixed buffers, without allocation: a polytope of n points has at most 2n - 4 faces
    let mut points = [Vector::ZERO; 4 + EXPANSIONS];
    points[..4].copy_from_slice(tetrahedron);
    let mut count = 4;
    let mut faces = [([0usize; 3], Vector::ZERO, 0.0f64); 2 * (4 + EXPANSIONS)];
    let centroid = (points[0] + points[1] + points[2] + points[3]) / 4.0;
    for (face, [i, j, k]) in faces.iter_mut().zip([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]) {
        let normal = (points[j] - points[i]).cross(points[k] - points[i]);
        let (normal, order) =
            if normal.dot(centroid - points[i]) > 0.0 { (-normal, [i, k, j]) } else { (normal, [i, j, k]) };
        let length = normal.length();
        if length <= 0.0 {
            return None;
        }
        *face = (order, normal / length, normal.dot(points[i]) / length);
    }
    let mut faces_len = 4;
    if faces[..4].iter().any(|f| f.2 < 0.0) {
        return None;
    }
    let mut upper = f64::INFINITY;
    let mut best = Vector::ZERO;
    let mut visible = [false; 2 * (4 + EXPANSIONS)];
    let mut horizon = [(0usize, 0usize); 6 * (4 + EXPANSIONS)];
    for _ in 0..EXPANSIONS {
        let closest = (0..faces_len).min_by(|a, b| faces[*a].2.total_cmp(&faces[*b].2))?;
        let (_, normal, lower) = faces[closest];
        if lower > TAU + margin {
            return Some(true);
        }
        let support = support(normal);
        if support.dot(normal) < upper {
            (upper, best) = (support.dot(normal), normal);
        }
        if upper <= TAU - margin {
            *found = Some(best);
            return Some(false);
        }
        if support.dot(normal) - lower <= 1e-9 * (1.0 + lower) {
            return None;
        }
        // replace the faces that see the support point by a cone from it over their horizon
        for (seen, (f, n, _)) in visible.iter_mut().zip(&faces[..faces_len]) {
            *seen = n.dot(support - points[f[0]]) > 0.0;
        }
        let mut edges = 0;
        for (f, _, _) in faces[..faces_len].iter().zip(&visible).filter(|(_, v)| **v).map(|(f, _)| f) {
            for (a, b) in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
                let shared = faces[..faces_len]
                    .iter()
                    .zip(&visible)
                    .any(|((g, _, _), v)| *v && [(g[0], g[1]), (g[1], g[2]), (g[2], g[0])].contains(&(b, a)));
                if !shared {
                    horizon[edges] = (a, b);
                    edges += 1;
                }
            }
        }
        let mut kept = 0;
        for f in 0..faces_len {
            if !visible[f] {
                faces[kept] = faces[f];
                kept += 1;
            }
        }
        if kept + edges > faces.len() {
            return None;
        }
        points[count] = support;
        let apex = count;
        count += 1;
        for &(a, b) in &horizon[..edges] {
            let normal = (points[b] - points[a]).cross(points[apex] - points[a]);
            let length = normal.length();
            let distance = normal.dot(points[a]) / length;
            if length <= 0.0 || distance.is_nan() || distance < 0.0 {
                return None;
            }
            faces[kept] = ([a, b, apex], normal / length, distance);
            kept += 1;
        }
        faces_len = kept;
    }
    None
}

impl SupportMap for Shape<'_> {
    fn local_support_point(&self, dir: Vector) -> Vector {
        support(self.vertices, dir)
    }
}

/// Unit directions of the nonzero `values`, each signed so its first nonzero component is positive, without repeats.
fn directions(values: impl Iterator<Item = Vector>) -> Vec<Vector> {
    let mut seen = HashSet::new();
    values
        .filter_map(|v| {
            let length = v.length();
            if length == 0.0 {
                return None;
            }
            let mut n = v / length;
            if n.to_array().into_iter().find(|v| *v != 0.0).unwrap() < 0.0 {
                n = -n;
            }
            let key = n.to_array().map(|v| if v == 0.0 { 0 } else { v.to_bits() });
            seen.insert(key).then_some(n)
        })
        .collect()
}

fn project(vertices: &[Vector], axis: Vector) -> (f64, f64) {
    vertices.iter().map(|v| v.dot(axis)).fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| {
        (if p < lo { p } else { lo }, if p > hi { p } else { hi })
    })
}

/// Longest of `points` (relative to an origin among them), and the one spanning the largest area with it.
fn span(points: &[Vector]) -> (Vector, Vector) {
    let u = *points.iter().max_by(|a, b| a.length_squared().total_cmp(&b.length_squared())).unwrap();
    let v =
        *points.iter().max_by(|a, b| u.cross(**a).length_squared().total_cmp(&u.cross(**b).length_squared())).unwrap();
    (u, v)
}

/// Triangles of the hull of `points` (relative to an origin among them), None when they are flat.
fn topology(points: &[Vector]) -> io::Result<Option<Vec<[u32; 3]>>> {
    // an invertible affine basis preserves the hull while avoiding aspect-ratio loss
    let (u, v) = span(points);
    let normal = u.cross(v);
    let w = *points.iter().max_by(|a, b| normal.dot(**a).abs().total_cmp(&normal.dot(**b).abs())).unwrap();
    if normal.dot(w) == 0.0 {
        return Ok(None);
    }
    let inverse = Matrix3::from_cols(u, v, w).inverse();
    let conditioned: Vec<Vector> = points.iter().map(|p| inverse * *p).collect();
    if conditioned.iter().any(|p| !p.is_finite()) {
        return Err(bad("non-finite hull conditioning"));
    }
    let rotations = [
        Rotation::IDENTITY,
        Rotation::from_rotation_x(0.37) * Rotation::from_rotation_y(0.61),
        Rotation::from_rotation_z(0.83) * Rotation::from_rotation_x(0.29),
    ];
    let mut built = None;
    let mut failure = None;
    for rotation in rotations {
        let oriented: Vec<Vector> = conditioned.iter().map(|p| rotation * *p).collect();
        match try_convex_hull(&oriented) {
            Ok((hull, faces)) => {
                built = Some((oriented, hull, faces));
                break;
            }
            Err(error) => failure = Some(error),
        }
    }
    let (oriented, hull, faces) = built.ok_or_else(|| bad(&format!("invalid conditioned convex hull: {failure:?}")))?;
    let original: Vec<u32> = hull
        .iter()
        .map(|v| {
            oriented
                .iter()
                .position(|p| *p == *v)
                .map(|i| i as u32)
                .ok_or_else(|| bad("hull builder changed a source vertex"))
        })
        .collect::<io::Result<_>>()?;
    Ok(Some(faces.iter().map(|f| f.map(|i| original[i as usize])).collect()))
}

/// Whether boxes `a` and `b` overlap by at most `depth` along an axis. With `depth` TAU - MARGIN, shapes within them
/// cannot block each other, whatever the precise test decides.
fn shallow(a: Aabb, b: Aabb, depth: f64) -> bool {
    (a.maxs - b.mins).min(b.maxs - a.mins).min_element() <= depth
}

/// Whether the projection intervals `a` and `b` overlap by at most `depth`; not the intersection width, so that
/// containment counts as deep.
fn within(a: (f64, f64), b: (f64, f64), depth: f64) -> bool {
    (a.1 - b.0).min(b.1 - a.0) <= depth
}

/// Result of `query`, None also if it panics (parry's GJK on a degenerate simplex).
fn shortcut<T>(query: impl FnOnce() -> Option<T>) -> Option<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(query)).ok().flatten()
}

impl Features {
    /// Features of the hull of `vertices`, its ball centered on their mean.
    fn new(vertices: &[Vector]) -> io::Result<Self> {
        let centered: Vec<Vector> = vertices.iter().map(|v| *v - vertices[0]).collect();
        let indices = topology(&centered).map_err(|e| bad(&format!("collider feature topology: {e}")))?;
        let edge = |a: u32, b: u32| vertices[b as usize] - vertices[a as usize];
        let (edges, normals) = match &indices {
            Some(faces) => {
                let edges = faces.iter().flat_map(|f| [edge(f[0], f[1]), edge(f[1], f[2]), edge(f[2], f[0])]);
                (directions(edges), directions(faces.iter().map(|f| edge(f[0], f[1]).cross(edge(f[0], f[2])))))
            }
            None => {
                // a flat piece: its plane's normal and in-plane edge normals; a segment: the axes crossed with it
                let pairs =
                    vertices.iter().enumerate().flat_map(|(i, a)| vertices[i + 1..].iter().map(move |b| *b - *a));
                let edges = directions(pairs);
                let (u, v) = span(&centered);
                let plane = u.cross(v);
                let axes = if plane == Vector::ZERO { vec![Vector::X, Vector::Y, Vector::Z] } else { vec![plane] };
                let crosses = edges.iter().flat_map(|e| axes.iter().map(|axis| e.cross(*axis)));
                let normals = directions(axes.iter().copied().chain(crosses));
                (edges, normals)
            }
        };
        let center = vertices.iter().copied().sum::<Vector>() / vertices.len() as f64;
        let radius = if indices.is_none() {
            0.0
        } else {
            normals
                .iter()
                .map(|n| {
                    let (lo, hi) = project(vertices, *n);
                    let c = center.dot(*n);
                    (c - lo).min(hi - c)
                })
                .fold(f64::INFINITY, f64::min)
                .max(0.0)
        };
        Ok(Self { normals, edges, center, radius })
    }
}

impl Shape<'_> {
    /// Distance between the hulls (`other` at `relative`), infinity beyond `radius`. GJK settles it unless it
    /// stalls (on touching hulls, witnesses farther than `radius`) or gives up (or panics): then hulls without a
    /// separating axis meet, those apart by more than `radius` along one are beyond it, and GJK from other directions
    /// settles the rest.
    fn distance(&self, other: &Shape, relative: Pose, radius: f64) -> io::Result<f64> {
        let from = |start: Vector| {
            let found = shortcut(|| {
                let mut simplex = VoronoiSimplex::new();
                simplex.reset(CsoPoint::from_shapes(&relative, self, other, start));
                match gjk::closest_points(&relative, self, other, radius, true, &mut simplex) {
                    GJKResult::Intersection => Some(0.0),
                    GJKResult::ClosestPoints(p, q, _) => Some(p.distance(q)), // at least the distance
                    // apart by more than `radius` along `axis`
                    GJKResult::NoIntersection(axis) if axis != Vector::X => Some(f64::INFINITY),
                    _ => None, // GJK gave up (parry's NoIntersection along X after 100 iterations)
                }
            });
            found.unwrap_or(f64::NAN) // unsettled: gave up or panicked
        };
        let found = from(Vector::X);
        if found <= radius || found == f64::INFINITY {
            return Ok(found);
        }
        let separation = self.separation(other, relative)?;
        if separation <= MARGIN {
            return Ok(0.0);
        }
        if separation > radius {
            return Ok(f64::INFINITY);
        }
        let best =
            [Vector::Y, Vector::Z, -Vector::X, -Vector::Y, -Vector::Z].map(from).into_iter().fold(found, f64::min);
        Ok(if best <= radius { best } else { f64::INFINITY })
    }

    /// Largest gap between the hulls (`other` at `relative`) along a face normal or an edge cross product: a lower
    /// bound on their distance, positive iff they are apart.
    fn separation(&self, other: &Shape, relative: Pose) -> io::Result<f64> {
        let (a, b) = (self.features()?, other.features()?);
        let vertices: Vec<Vector> = other.vertices.iter().map(|v| relative.transform_point(*v)).collect();
        let apart = |axis: Vector| {
            let ((a0, a1), (b0, b1)) = (project(self.vertices, axis), project(&vertices, axis));
            (b0 - a1).max(a0 - b1) / axis.length()
        };
        let edges: Vec<Vector> = b.edges.iter().map(|e| relative.rotation * *e).collect();
        let faces = a.normals.iter().copied().chain(b.normals.iter().map(|n| relative.rotation * *n));
        let crosses = a.edges.iter().flat_map(|x| edges.iter().map(move |y| x.cross(*y)));
        Ok(faces.chain(crosses).filter(|axis| axis.length() > 0.0).map(apart).fold(f64::NEG_INFINITY, f64::max))
    }

    /// Whether the hulls overlap by at most `depth` along `axis`.
    fn shallow_along(&self, other: &Shape, relative: Pose, axis: Vector, depth: f64) -> bool {
        let length = axis.length();
        if length == 0.0 {
            return false;
        }
        let (b0, b1) = project(other.vertices, relative.rotation.inverse() * axis);
        let t = relative.translation.dot(axis);
        within(project(self.vertices, axis), (b0 + t, b1 + t), depth * length)
    }

    /// Cheap decision when one is at hand: a separating axis (the center line or GJK's), or a penetration bracketed
    /// beyond or within TAU.
    fn quick(&self, other: &Shape, relative: Pose, found: &mut Option<Vector>) -> Option<bool> {
        let delta = relative.transform_point(other.hull.bounds.center()) - self.hull.bounds.center();
        if self.shallow_along(other, relative, delta, TAU) {
            *found = Some(delta);
            return Some(false);
        }
        let mut simplex = VoronoiSimplex::new();
        simplex.reset(CsoPoint::from_shapes(&relative, self, other, delta.try_normalize().unwrap_or(Vector::X)));
        match gjk::closest_points(&relative, self, other, 0.0, false, &mut simplex) {
            GJKResult::ClosestPoints(_, _, axis) | GJKResult::Proximity(axis) | GJKResult::NoIntersection(axis) => {
                if self.shallow_along(other, relative, axis, TAU) {
                    *found = Some(axis);
                    return Some(false);
                }
            }
            GJKResult::Intersection if simplex.dimension() == 3 => {
                let p: [Vector; 4] = std::array::from_fn(|i| simplex.point(i).point);
                return penetrates(|dir| self.cso_support(other, relative, dir), &p, 0.0, found);
            }
            _ => {}
        }
        None
    }

    /// CSO support point of self - relative(other) in direction `dir`.
    fn cso_support(&self, other: &Shape, relative: Pose, dir: Vector) -> Vector {
        self.local_support_point(dir)
            - relative.transform_point(other.local_support_point(relative.rotation.inverse() * -dir))
    }

    /// Features of the hull, built on first use and charged to the collider's account (and to the cache, while it holds
    /// the collider).
    fn features(&self) -> io::Result<&Features> {
        match self.hull.features.get_or_init(|| {
            let result = Features::new(self.vertices).map_err(|e| e.to_string());
            if let Ok(features) = &result {
                let bytes = ((features.normals.capacity() + features.edges.capacity()) * 24) as u64;
                self.account.features.fetch_add(bytes, Ordering::Relaxed);
                if self.account.cached.load(Ordering::Relaxed) {
                    self.charge.fetch_add(bytes, Ordering::Relaxed);
                }
            }
            result
        }) {
            Ok(features) => Ok(features),
            Err(error) => Err(bad(error)),
        }
    }

    /// `blocked`, for hulls whose boxes overlap deeper than TAU; an axis found along which they are shallow is left in
    /// `found`.
    fn blocked_deep(&self, other: &Shape, relative: Pose, found: &mut Option<Vector>) -> io::Result<bool> {
        let decided = self.decide(other, relative, found).or_else(|| shortcut(|| self.quick(other, relative, found)));
        match decided {
            Some(blocked) => Ok(blocked),
            None => self.separated(other, relative, found).map(|separated| !separated),
        }
    }

    /// Decision with the other hull's vertices in this hull's frame, when one is at hand: shallow along an axis of
    /// either frame or a direction of GJK with a TAU margin, or a penetration bracketed beyond or within TAU.
    fn decide(&self, other: &Shape, relative: Pose, found: &mut Option<Vector>) -> Option<bool> {
        let rotation = Matrix3::from_quat(relative.rotation);
        let moved: Vec<Vector> = other.vertices.iter().map(|v| rotation * *v + relative.translation).collect();
        if shallow(self.hull.bounds, Aabb::from_points(moved.iter().copied()), TAU - MARGIN) {
            return Some(false);
        }
        let (inverse, back) = (rotation.transpose(), relative.inverse().translation);
        let back = Aabb::from_points(self.vertices.iter().map(|v| inverse * *v + back));
        if shallow(other.hull.bounds, back, TAU - MARGIN) {
            return Some(false);
        }
        let cso = |dir: Vector| support(self.vertices, dir) - support(&moved, -dir);
        let start = relative.transform_point(other.hull.bounds.center()) - self.hull.bounds.center();
        match gjk(cso, start) {
            Gjk::Shallow(axis) => {
                *found = Some(axis);
                Some(false)
            }
            Gjk::Enclosed(tetrahedron) => penetrates(cso, &tetrahedron, MARGIN, found),
            Gjk::Unresolved => None,
        }
    }

    /// Exact separating-axis test: whether the hulls are shallow along a face normal or edge cross product (the axis
    /// is left in `found`).
    fn separated(&self, other: &Shape, relative: Pose, found: &mut Option<Vector>) -> io::Result<bool> {
        *found = None;
        let a = self.features()?;
        let b = other.features()?;
        // included balls give a lower bound on the separating translation
        if a.radius + b.radius - TAU > a.center.distance(relative.transform_point(b.center)) {
            return Ok(false);
        }
        let vertices: Vec<Vector> = other.vertices.iter().map(|v| relative.transform_point(*v)).collect();
        let separates = |axis: Vector| {
            let length = axis.length();
            length != 0.0 && within(project(self.vertices, axis), project(&vertices, axis), TAU * length)
        };
        let mut faces = a.normals.iter().copied().chain(b.normals.iter().map(|n| relative.rotation * *n));
        *found = faces.find(|n| separates(*n));
        if found.is_none() {
            let edges: Vec<Vector> = b.edges.iter().map(|v| relative.rotation * *v).collect();
            *found = a.edges.iter().flat_map(|a| edges.iter().map(move |b| a.cross(*b))).find(|axis| separates(*axis));
        }
        Ok(found.is_some())
    }
}

/// Boxes of a collider's hulls in leaves of `LEAF`, the leaves in groups of `LEAF`, packed in sort-tile-recursive
/// order, with the box of each leaf and group.
struct HullTree {
    boxes: Vec<(Aabb, usize)>, // box, hull
    leaves: Vec<Aabb>,
    groups: Vec<Aabb>,
}

/// Bounds of each run of `LEAF` boxes.
fn merged(boxes: impl Iterator<Item = Aabb>) -> Vec<Aabb> {
    let mut boxes = boxes.peekable();
    let mut merged = Vec::new();
    while boxes.peek().is_some() {
        merged.push(boxes.by_ref().take(LEAF).reduce(|a, b| a.merged(&b)).unwrap());
    }
    merged
}

impl HullTree {
    fn new(hulls: &[Hull]) -> Self {
        let mut boxes: Vec<(Aabb, usize)> = hulls.iter().map(|h| h.bounds).zip(0..).collect();
        // slabs along x, each in slabs along y, each sorted along z: nearby boxes share leaves
        let slabs = (boxes.len().div_ceil(LEAF) as f64).cbrt().ceil().max(1.0) as usize;
        let center = |k: usize| move |a: &(Aabb, usize), b: &(Aabb, usize)| a.0.center()[k].total_cmp(&b.0.center()[k]);
        boxes.sort_by(center(0));
        let slab = boxes.len().div_ceil(slabs).div_ceil(LEAF) * LEAF;
        for x in boxes.chunks_mut(slab) {
            x.sort_by(center(1));
            for y in x.chunks_mut(slab.div_ceil(slabs).div_ceil(LEAF) * LEAF) {
                y.sort_by(center(2));
            }
        }
        let leaves = merged(boxes.iter().map(|b| b.0));
        let groups = merged(leaves.iter().copied());
        Self { boxes, leaves, groups }
    }

    /// Heap bytes of the hierarchy.
    fn bytes(&self) -> usize {
        let boxes = self.boxes.capacity() * std::mem::size_of::<(Aabb, usize)>();
        boxes + (self.leaves.capacity() + self.groups.capacity()) * std::mem::size_of::<Aabb>()
    }

    /// Hulls whose boxes overlap `query` deeper than TAU (not `shallow`): a box within a shallow one is shallow.
    fn deep(&self, query: Aabb) -> impl Iterator<Item = usize> + '_ {
        let groups = self.groups.iter().enumerate().filter(move |(_, g)| !shallow(**g, query, TAU));
        let leaves = groups.flat_map(move |(g, _)| {
            let first = g * LEAF;
            self.leaves[first..(first + LEAF).min(self.leaves.len())].iter().zip(first..)
        });
        let hit = leaves.filter(move |(l, _)| !shallow(**l, query, TAU));
        hit.flat_map(move |(_, l)| {
            let first = l * LEAF;
            self.boxes[first..(first + LEAF).min(self.boxes.len())]
                .iter()
                .filter(move |(b, _)| !shallow(*b, query, TAU))
        })
        .map(|&(_, hull)| hull)
    }
}

/// Convex components of a part: one array of their vertices, their hulls, and their hull hierarchy.
pub(crate) struct Collider {
    vertices: Vec<Vector>,
    hulls: Vec<Hull>,
    tree: HullTree,
    bounds: Aabb,
    bytes: usize, // the heap and struct bytes, without the features
    charge: Charge,
    account: Account,
}

impl Collider {
    /// Collider of an OBJ file: one convex component per `o` group, of the vertices its faces reference.
    pub(crate) fn from_obj(raw: &[u8]) -> io::Result<Self> {
        let text = std::str::from_utf8(raw).map_err(|e| bad(&e.to_string()))?;
        let mut vertices = Vec::new();
        let mut indices = Vec::<usize>::new();
        let mut components = Vec::new();
        let finish = |indices: &mut Vec<usize>, vertices: &[Vector], components: &mut Vec<Vec<Vector>>| {
            if !indices.is_empty() {
                indices.sort_unstable();
                indices.dedup();
                components.push(indices.iter().map(|&i| vertices[i]).collect());
                indices.clear();
            }
        };
        for line in text.lines() {
            let mut fields = line.split_whitespace();
            match fields.next() {
                Some("v") => {
                    let v: Vec<f64> = fields
                        .take(3)
                        .map(|s| s.parse::<f64>().map_err(|e| bad(&e.to_string())))
                        .collect::<io::Result<_>>()?;
                    if v.len() != 3 || v.iter().any(|v| !v.is_finite()) {
                        return Err(bad("invalid OBJ vertex"));
                    }
                    vertices.push(Vector::new(v[0], v[1], v[2]));
                }
                Some("o") => finish(&mut indices, &vertices, &mut components),
                Some("f") => {
                    let ids: Vec<usize> = fields
                        .map(|s| {
                            let i = s.split('/').next().unwrap().parse::<i64>().map_err(|e| bad(&e.to_string()))?;
                            if i == 0 || i.unsigned_abs() > vertices.len() as u64 {
                                return Err(bad("invalid OBJ vertex index"));
                            }
                            Ok(if i > 0 { i as usize - 1 } else { (vertices.len() as i64 + i) as usize })
                        })
                        .collect::<io::Result<_>>()?;
                    if ids.len() < 3 {
                        return Err(bad("invalid OBJ face"));
                    }
                    indices.extend(ids);
                }
                _ => {}
            }
        }
        finish(&mut indices, &vertices, &mut components);
        let ends = components.iter().scan(0, |end, c| {
            *end += c.len();
            Some(*end)
        });
        let ends = ends.collect();
        Self::assembled(components.concat(), ends, &Charge::default())
    }

    /// Collider of the hulls of `vertices` that end at `ends`.
    fn assembled(vertices: Vec<Vector>, ends: Vec<usize>, charge: &Charge) -> io::Result<Self> {
        if ends.is_empty() {
            return Err(bad("collider has no convex components"));
        }
        let hulls: Vec<Hull> = std::iter::once(0)
            .chain(ends.iter().copied())
            .zip(&ends)
            .map(|(first, &end)| Hull {
                first: first as u32,
                end: end as u32,
                bounds: Aabb::from_points(vertices[first..end].iter().copied()),
                features: OnceLock::new(),
            })
            .collect();
        let bounds = Aabb::from_points(hulls.iter().flat_map(|h| [h.bounds.mins, h.bounds.maxs]));
        let tree = HullTree::new(&hulls);
        let bytes = std::mem::size_of::<Self>()
            + vertices.capacity() * std::mem::size_of::<Vector>()
            + hulls.capacity() * std::mem::size_of::<Hull>()
            + tree.bytes();
        Ok(Self { vertices, hulls, tree, bounds, bytes, charge: Arc::clone(charge), account: Account::default() })
    }

    /// Hull `i` with this collider's vertices, charge, and account.
    fn shape(&self, i: usize) -> Shape<'_> {
        let hull = &self.hulls[i];
        let vertices = &self.vertices[hull.first as usize..hull.end as usize];
        Shape { hull, vertices, charge: &self.charge, account: &self.account }
    }

    /// Packed collider, little-endian: u32 hull count, then per hull a u32 vertex count and its vertices (f64 x, y, z).
    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&(self.hulls.len() as u32).to_le_bytes());
        for i in 0..self.hulls.len() {
            let vertices = self.shape(i).vertices;
            data.extend_from_slice(&(vertices.len() as u32).to_le_bytes());
            for v in vertices {
                for x in v.to_array() {
                    data.extend_from_slice(&x.to_le_bytes());
                }
            }
        }
        data
    }

    /// Vertices of the convex components of a packed collider, in one array, and where each component ends.
    fn components(data: &[u8]) -> io::Result<(Vec<Vector>, Vec<usize>)> {
        let mut offset = 4;
        let count = word(data, 0, 4)? as usize;
        let mut vertices = Vec::with_capacity(data.len() / 24);
        let mut ends = Vec::with_capacity(count.min(data.len() / 28));
        for _ in 0..count {
            let n = word(data, offset, 4)? as usize;
            offset += 4;
            let stop = offset
                .checked_add(n.checked_mul(24).ok_or_else(|| bad("invalid packed hull size"))?)
                .ok_or_else(|| bad("invalid packed hull size"))?;
            let raw = data.get(offset..stop).ok_or_else(|| bad("truncated packed geometry"))?;
            let first = vertices.len();
            let value = |k: usize| f64::from_bits(le(raw, 8 * k, 8));
            vertices.extend((0..n).map(|i| Vector::new(value(3 * i), value(3 * i + 1), value(3 * i + 2))));
            if vertices.len() == first || vertices[first..].iter().any(|v| !v.is_finite()) {
                return Err(bad("invalid packed vertices"));
            }
            ends.push(vertices.len());
            offset = stop;
        }
        if offset != data.len() {
            return Err(bad("packed geometry has trailing data"));
        }
        Ok((vertices, ends))
    }

    fn decode(data: &[u8], charge: &Charge) -> io::Result<Self> {
        let (vertices, ends) = Self::components(data)?;
        Self::assembled(vertices, ends, charge)
    }

    /// Bounds of a packed collider, without building it.
    fn bounds(data: &[u8]) -> io::Result<Aabb> {
        let (vertices, ends) = Self::components(data)?;
        if ends.is_empty() {
            return Err(bad("collider has no convex components"));
        }
        Ok(Aabb::from_points(vertices))
    }

    /// Distance from this collider to `other` at `relative` (its pose in this collider's frame): the least distance
    /// between their hulls, 0 when two hulls meet, infinity beyond `radius`.
    pub(crate) fn clearance(&self, other: &Collider, relative: Pose, radius: f64) -> io::Result<f64> {
        if gap(self.bounds, other.bounds.transform_by(&relative)) > radius {
            return Ok(f64::INFINITY);
        }
        let boxes: Vec<Aabb> = other.hulls.iter().map(|h| h.bounds.transform_by(&relative)).collect();
        let mut best = f64::INFINITY;
        for i in 0..self.hulls.len() {
            for (j, other_box) in boxes.iter().enumerate() {
                if gap(self.hulls[i].bounds, *other_box) <= best.min(radius) {
                    best = best.min(self.shape(i).distance(&other.shape(j), relative, radius)?);
                    if best == 0.0 {
                        return Ok(0.0);
                    }
                }
            }
        }
        Ok(best)
    }

    /// Pair of hulls (of self, of `other`) that blocks `other` at `relative`, if any, with what `memory` holds of
    /// this pair of colliders: the pair of hulls that blocked last is tested first, and the axis along which a pair of
    /// hulls was last shallow is tried before the narrow phase.
    fn blocking(&self, relative: Pose, other: &Collider, memory: &mut Memory) -> io::Result<Option<(usize, usize)>> {
        let Memory { likely, axes } = memory;
        let likely = *likely;
        // whether hulls i (of self) and j (of other), with boxes deeper than TAU, block
        let mut deep = |i: usize, j: usize| -> io::Result<bool> {
            let (x, y) = (other.shape(j), self.shape(i));
            let key = (i as u32, j as u32);
            if axes.get(&key).is_some_and(|axis| x.shallow_along(&y, relative, *axis, TAU - MARGIN)) {
                return Ok(false);
            }
            let mut found = None;
            let blocked = x.blocked_deep(&y, relative, &mut found)?;
            if let Some(axis) = found.filter(|_| !blocked) {
                axes.insert(key, axis);
            }
            Ok(blocked)
        };
        if let Some((i, j)) = likely {
            if !shallow(other.hulls[j].bounds, self.hulls[i].bounds.transform_by(&relative), TAU) && deep(i, j)? {
                return Ok(likely);
            }
        }
        // the hulls of self shallow within TAU - MARGIN against the bounds of other block none of its hulls
        let reach = other.bounds.transform_by(&relative.inverse());
        for (i, hull) in self.hulls.iter().enumerate() {
            if shallow(hull.bounds, reach, TAU - MARGIN) {
                continue;
            }
            for j in other.tree.deep(hull.bounds.transform_by(&relative)) {
                if deep(i, j)? {
                    return Ok(Some((i, j)));
                }
            }
        }
        Ok(None)
    }
}

/// Key of a collider pair decision: the two meshes and the bits of their relative pose.
type PairKey = (u32, u32, [u64; 7]);

/// What the narrow phase learned of a pair of colliders: the pair of their hulls that last blocked, and per pair of
/// hulls the axis along which they were last shallow.
#[derive(Default)]
struct Memory {
    likely: Option<(usize, usize)>,
    axes: FastMap<(u32, u32), Vector>,
}

/// Colliders of a geometry pack, decoded on demand into a cache of bounded bytes, with the memory of collider
/// pair decisions and separating axes.
pub(crate) struct Geometry {
    cache: FastLru<u32, Arc<Collider>>,
    live: FastMap<u32, Weak<Collider>>,
    bytes: usize, // of the cached colliders, without their features
    limit: usize,
    charge: Charge,
    packed: Packed,
    pairs: FastLru<PairKey, bool>,
    memory: FastMap<(u32, u32), Memory>, // per pair of colliders: those met since `older` was set aside
    older: FastMap<(u32, u32), Memory>,  // the generation before, dropped when `memory` holds AXES axes
    remembered: usize,                   // axes in `memory`
}

impl Geometry {
    /// Colliders of `packed`, cached up to `limit` bytes.
    pub fn new(packed: Packed, limit: usize) -> Self {
        Self {
            cache: FastLru::unbounded_with_hasher(Default::default()),
            live: FastMap::default(),
            bytes: 0,
            limit,
            charge: Charge::default(),
            packed,
            pairs: FastLru::with_hasher(std::num::NonZeroUsize::new(PAIRS).unwrap(), Default::default()),
            memory: FastMap::default(),
            older: FastMap::default(),
            remembered: 0,
        }
    }

    /// Whether collider `mesh` (mesh `id`) at `p` and mesh `other_id` at `q` block each other: their hulls overlap
    /// deeper than TAU. Decisions are remembered by exact relative pose.
    pub fn blocked(&mut self, mesh: &Collider, id: u32, p: Pose, other_id: u32, q: Pose) -> PyResult<bool> {
        let loaded = if id == other_id { None } else { Some(self.get(other_id)?) };
        let other = loaded.as_deref().unwrap_or(mesh);
        // keep boundary decisions and cache keys independent of caller operand order
        let (a, b, ia, ib, relative) = if (mesh.hulls.len(), id, bits(p)) > (other.hulls.len(), other_id, bits(q)) {
            (other, mesh, other_id, id, p.inv_mul(&q))
        } else {
            (mesh, other, id, other_id, q.inv_mul(&p))
        };
        let key = (ia, ib, bits(relative));
        if let Some(&blocked) = self.pairs.get(&key) {
            return Ok(blocked);
        }
        if self.remembered >= AXES {
            self.older = std::mem::take(&mut self.memory);
            self.remembered = 0;
        }
        let memory = self.memory.entry((ia, ib)).or_insert_with(|| {
            let older = self.older.remove(&(ia, ib)).unwrap_or_default();
            self.remembered += older.axes.len();
            older
        });
        let axes = memory.axes.len();
        let blocking = a.blocking(relative, b, memory).map_err(io_err)?;
        self.remembered += memory.axes.len() - axes;
        if blocking.is_some() {
            memory.likely = blocking;
        }
        self.pairs.put(key, blocking.is_some());
        Ok(blocking.is_some())
    }

    /// Bounds of collider `mesh`, read from the pack without building the collider.
    pub fn bounds_of(&self, mesh: u32) -> PyResult<Aabb> {
        self.packed.get(mesh).and_then(Collider::bounds).map_err(io_err)
    }

    /// Collider `mesh`: cached, still held elsewhere, or decoded from the pack (its pages then released from the
    /// mapping).
    pub fn get(&mut self, mesh: u32) -> PyResult<Arc<Collider>> {
        if let Some(value) = self.cache.get(&mesh) {
            return Ok(Arc::clone(value));
        }
        if let Some(value) = self.live.get(&mesh).and_then(Weak::upgrade) {
            return Ok(value);
        }
        let decoded = self.packed.get(mesh).and_then(|data| Collider::decode(data, &self.charge));
        let value = Arc::new(decoded.map_err(io_err)?);
        self.packed.release(mesh);
        self.live.insert(mesh, Arc::downgrade(&value));
        if value.bytes <= self.limit {
            // the cache holds colliders up to `limit` bytes with the features built since they were cached
            while self.bytes + self.charge.load(Ordering::Relaxed) as usize + value.bytes > self.limit {
                let (_, old) = self.cache.pop_lru().unwrap();
                old.account.cached.store(false, Ordering::Relaxed);
                self.charge.fetch_sub(old.account.features.load(Ordering::Relaxed), Ordering::Relaxed);
                self.bytes -= old.bytes;
            }
            value.account.cached.store(true, Ordering::Relaxed);
            self.bytes += value.bytes;
            self.cache.put(mesh, Arc::clone(&value));
        }
        Ok(value)
    }
}

/// Placed part: its pool index, collider, and pose.
pub(crate) struct Placed {
    pub part: usize,
    pub mesh: u32,
    pub pose: Pose,
    pub realized: Pose, // the pose as decoded from its row-major matrix, which the collision tests take
    pub first: bool,    // whether no earlier placement is of its part
    pub(crate) collider: Arc<Collider>,
    generation: u64, // the scene's placements up to this one: it tells this placement from a later one at its index
}

/// Placed parts of an assembly, their boxes, their growing envelopes, and the placements they blocked.
#[derive(Default)]
pub(crate) struct Scene {
    pub placed: Vec<Placed>,
    boxes: Vec<Aabb>, // of the placed parts, scanned whole: scenes are small
    envelopes: Vec<(Aabb, f64)>,
    generations: u64,
    blockers: FastMap<(u32, [u64; 7]), (usize, u64)>, // per blocked placement, the placed part that blocked it
    nearby: Vec<(f64, usize)>,
}

fn volume(bounds: Aabb) -> f64 {
    let size = bounds.maxs - bounds.mins;
    size.x * size.y * size.z
}

impl Scene {
    /// Place pool part `part` (mesh `mesh`) at `pose`.
    pub fn add(&mut self, part: usize, mesh: u32, pose: Pose, collider: Arc<Collider>) {
        let (realized, first) = (realized(pose), !self.placed.iter().any(|p| p.part == part));
        let bounds = collider.bounds.transform_by(&realized);
        let mass = volume(collider.bounds);
        self.envelopes.push(match self.envelopes.last() {
            Some((old, total)) => (old.merged(&bounds), total + mass),
            None => (bounds, mass),
        });
        self.boxes.push(bounds);
        self.generations += 1;
        self.placed.push(Placed { part, mesh, pose, realized, first, collider, generation: self.generations });
    }
    /// Remove the last placed part.
    pub fn pop(&mut self) {
        self.placed.pop();
        self.envelopes.pop();
        self.boxes.pop();
    }
    /// ln(1 + volume of `bounds` / the placed parts' summed box volume), 0 while that sum is 0.
    pub fn mass_growth(&self, bounds: Aabb) -> f64 {
        let mass = self.envelopes.last().unwrap().1;
        if mass > 0.0 {
            (volume(bounds) / mass).ln_1p()
        } else {
            0.0
        }
    }
    /// Envelope expansion by a part of box `bounds` (part frame) at `pose`: ln(envelope volume growth * old summed box
    /// volume / new), 0 when that ratio is at most 1 or a volume is not positive.
    pub fn expansion(&self, bounds: Aabb, pose: Pose) -> f64 {
        let (old, mass) = *self.envelopes.last().unwrap();
        let total = mass + volume(bounds);
        let before = volume(old);
        if mass <= 0.0 || total <= 0.0 || before <= 0.0 {
            return 0.0;
        }
        let merged = old.merged(&bounds.transform_by(&pose));
        let growth = volume(merged) / before * mass / total;
        if growth > 1.0 {
            growth.ln()
        } else {
            0.0
        }
    }
    /// Whether collider `mesh` (mesh `mesh_id`) at `pose` is blocked by a placed part that is not `exempt`.
    pub fn blocked(
        &mut self,
        mesh: &Collider,
        mesh_id: u32,
        pose: Pose,
        geometry: &mut Geometry,
        mut exempt: impl FnMut(usize, &Placed) -> bool,
    ) -> PyResult<bool> {
        let pose = realized(pose);
        // a placement tried before is blocked again by the part that blocked it, while that part stays placed
        let key = (mesh_id, bits(pose));
        if let Some(&(index, generation)) = self.blockers.get(&key) {
            if let Some(other) = self.placed.get(index).filter(|other| other.generation == generation) {
                if !exempt(index, other) {
                    return Ok(true);
                }
            }
        }
        let bounds = mesh.bounds.transform_by(&pose);
        // the parts whose boxes overlap the placement most are the likeliest to block it, so they are tested first
        let mut nearby = std::mem::take(&mut self.nearby);
        nearby.clear();
        nearby.extend(
            (self.boxes.iter().enumerate())
                .filter(|(_, other)| bounds.intersects(other))
                .map(|(index, other)| (overlap(&bounds, other), index)),
        );
        nearby.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        let mut blocker = None;
        for &(_, index) in &nearby {
            let other = &self.placed[index];
            if exempt(index, other) {
                continue;
            }
            if geometry.blocked(mesh, mesh_id, pose, other.mesh, other.realized)? {
                blocker = Some((index, other.generation));
                break;
            }
        }
        self.nearby = nearby;
        let Some(blocker) = blocker else {
            return Ok(false);
        };
        if self.blockers.len() >= BLOCKERS {
            self.blockers.clear();
        }
        self.blockers.insert(key, blocker);
        Ok(true)
    }
}
