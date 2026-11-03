//! Pruning the pairs a jump can cross between, by a lower bound on its arrival.
//!
//! A jump from a parent cell `p` onto a child cell `c` arrives at
//! `onset[p] + T(p, c)`, and the child starts at the earliest arrival over every pair
//! no farther apart than the reach. Reading the medium along every pair costs
//! `O(P C)`. A shear wave is never faster than the medium's largest shear speed, so
//! `T(p, c) >= |p - c| / v_max`, and
//!
//! ```text
//! bound(p, c) = onset[p] + |p - c| / v_max  <=  arrival(p, c).
//! ```
//!
//! The bound needs no medium, so it can prune before anything reads the rock. Any
//! real arrival `t` rules out every pair whose bound exceeds `t`. The caller reads
//! the medium along the pair with the least bound ([`least_bound`]) and gets a real
//! arrival from it. It then reads the medium only along the pairs whose bound is at or
//! below that arrival ([`pairs_below`]). The earliest arrival is among those pairs, so
//! the search is exact. This is best-first search with an admissible heuristic, as in
//! A*, except that a crossing is one straight line and not a path through a graph.
//!
//! Visiting parent cells in onset order makes the search cheap. A parent cell whose
//! onset alone exceeds the threshold cannot win, and the remaining ones only need
//! the child cells within `(threshold - onset) v_max`. An infinite `v_max` is a valid
//! bound that prunes by onset alone.

use rayon::prelude::*;

/// A pair within reach, or within a threshold, to a relative `1e-12`. That way a reach
/// set to exactly the nearest gap keeps that pair, however another distance routine
/// rounded the gap, and a threshold keeps the pair that set it.
const REACH_SLACK: f64 = 1.0e-12;

/// What the jump search refuses, in its own vocabulary.
#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
pub enum Error {
    /// Each parent cell needs an onset.
    #[error("{onsets} onsets for {cells} parent cells")]
    MismatchedOnsets { onsets: usize, cells: usize },
    /// A position array is flat `(x, y, z)` triples.
    #[error("a position array has {len} values, which is not a whole number of triples")]
    NotTriples { len: usize },
    /// A reach is a non-negative, finite distance.
    #[error("a reach of {reach_km} km; it must be non-negative and finite")]
    BadReach { reach_km: f64 },
    /// A speed bound is positive. Infinity is a valid, weak bound, and NaN is not one.
    #[error("a shear-speed bound of {max_speed_km_s} km/s; it must be positive")]
    BadSpeed { max_speed_km_s: f64 },
}

/// A uniform grid of bins over a point cloud, for ball queries.
///
/// The bins are cubes. `offsets` is CSR over the bins in `(x, y, z)` row-major order,
/// and `points` holds the indices of the points in each bin.
struct Bins<'a> {
    xyz: &'a [f64],
    origin: [f64; 3],
    size_km: f64,
    shape: [usize; 3],
    offsets: Vec<usize>,
    points: Vec<u32>,
}

impl<'a> Bins<'a> {
    /// Bin `xyz` into cubes of about `size_km`, with at most a few bins per point.
    fn new(xyz: &'a [f64], size_km: f64) -> Self {
        let n = xyz.len() / 3;
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for point in xyz.as_chunks::<3>().0 {
            for (axis, &value) in point.iter().enumerate() {
                lo[axis] = lo[axis].min(value);
                hi[axis] = hi[axis].max(value);
            }
        }
        // Bins small next to the extent multiply in three dimensions. Grow them until
        // the grid is no larger than a few bins per point.
        let budget = 4 * n + 64;
        let mut size = if size_km > 0.0 { size_km } else { 1.0 };
        let shape = loop {
            let shape = [0, 1, 2].map(|axis| ((hi[axis] - lo[axis]) / size) as usize + 1);
            if shape.iter().product::<usize>() <= budget {
                break shape;
            }
            size *= 1.5;
        };
        let mut bins = Self {
            xyz,
            origin: lo,
            size_km: size,
            shape,
            offsets: vec![0; shape.iter().product::<usize>() + 1],
            points: vec![0; n],
        };
        let keys: Vec<usize> = xyz.as_chunks::<3>().0.iter().map(|p| bins.key(p)).collect();
        for &key in &keys {
            bins.offsets[key + 1] += 1;
        }
        for bin in 0..bins.offsets.len() - 1 {
            bins.offsets[bin + 1] += bins.offsets[bin];
        }
        let mut fill = bins.offsets.clone();
        for (index, &key) in keys.iter().enumerate() {
            bins.points[fill[key]] = u32::try_from(index).expect("cell counts fit in u32");
            fill[key] += 1;
        }
        bins
    }

    fn coordinate(&self, axis: usize, value: f64) -> usize {
        (((value - self.origin[axis]) / self.size_km).max(0.0) as usize).min(self.shape[axis] - 1)
    }

    fn key(&self, point: &[f64]) -> usize {
        let [i, j, k] = [0, 1, 2].map(|axis| self.coordinate(axis, point[axis]));
        (i * self.shape[1] + j) * self.shape[2] + k
    }

    /// Call `visit(index, distance)` for every point within `radius_km` of `centre`.
    fn within(&self, centre: &[f64], radius_km: f64, mut visit: impl FnMut(usize, f64)) {
        let [lo, hi] = [-radius_km, radius_km]
            .map(|offset| [0, 1, 2].map(|axis| self.coordinate(axis, centre[axis] + offset)));
        // A centre beyond the bins on an axis by more than the radius has no points.
        let outside = (0..3).any(|axis| {
            let below = centre[axis] + radius_km < self.origin[axis];
            let above = centre[axis] - radius_km
                > self.origin[axis] + self.size_km * self.shape[axis] as f64;
            below || above
        });
        if outside {
            return;
        }
        let squared = radius_km * radius_km;
        for i in lo[0]..=hi[0] {
            for j in lo[1]..=hi[1] {
                let row = (i * self.shape[1] + j) * self.shape[2];
                let span = self.offsets[row + lo[2]]..self.offsets[row + hi[2] + 1];
                for &index in &self.points[span] {
                    let index = index as usize;
                    let point = &self.xyz[3 * index..3 * index + 3];
                    let d2 = (point[0] - centre[0]).powi(2)
                        + (point[1] - centre[1]).powi(2)
                        + (point[2] - centre[2]).powi(2);
                    if d2 <= squared {
                        visit(index, d2.sqrt());
                    }
                }
            }
        }
    }
}

/// The inputs to both searches, checked once.
pub struct Search<'a> {
    parent_xyz: &'a [f64],
    onset_s: &'a [f64],
    reach_km: f64,
    max_speed_km_s: f64,
    /// Parent cells in onset order, the earliest first. Ties keep index order, so the
    /// search is deterministic.
    order: Vec<usize>,
    child: Bins<'a>,
}

impl<'a> Search<'a> {
    /// Prepare a search between parent cells, with their onsets, and child cells.
    ///
    /// Positions are flat `(x, y, z)` triples in kilometres. `max_speed_km_s` bounds
    /// the shear speed anywhere a crossing can pass; infinity is always valid.
    ///
    /// # Errors
    ///
    /// [`Error`] if the arrays disagree, or the reach or speed bound is invalid.
    pub fn new(
        parent_xyz: &'a [f64],
        onset_s: &'a [f64],
        child_xyz: &'a [f64],
        reach_km: f64,
        max_speed_km_s: f64,
    ) -> Result<Self, Error> {
        for xyz in [parent_xyz, child_xyz] {
            if xyz.len() % 3 != 0 {
                return Err(Error::NotTriples { len: xyz.len() });
            }
        }
        if onset_s.len() != parent_xyz.len() / 3 {
            return Err(Error::MismatchedOnsets {
                onsets: onset_s.len(),
                cells: parent_xyz.len() / 3,
            });
        }
        if reach_km.is_nan() || reach_km < 0.0 || reach_km.is_infinite() {
            return Err(Error::BadReach { reach_km });
        }
        if max_speed_km_s.is_nan() || max_speed_km_s <= 0.0 {
            return Err(Error::BadSpeed { max_speed_km_s });
        }
        let mut order: Vec<usize> = (0..onset_s.len()).collect();
        order.sort_by(|&a, &b| onset_s[a].total_cmp(&onset_s[b]));
        let reach_km = reach_km * (1.0 + REACH_SLACK);
        Ok(Self {
            parent_xyz,
            onset_s,
            reach_km,
            max_speed_km_s,
            order,
            // A bin a few cells across keeps ball queries near the points they return.
            child: Bins::new(child_xyz, reach_km / 8.0),
        })
    }

    fn parent(&self, index: usize) -> &[f64] {
        &self.parent_xyz[3 * index..3 * index + 3]
    }

    /// The farthest a crossing from `parent` can go and still arrive by `threshold_s`.
    fn radius(&self, parent: usize, threshold_s: f64) -> f64 {
        let slack_s = threshold_s - self.onset_s[parent];
        if self.max_speed_km_s.is_finite() && slack_s.is_finite() {
            self.reach_km.min(slack_s * self.max_speed_km_s)
        } else {
            self.reach_km
        }
    }

    fn bound(&self, parent: usize, distance_km: f64) -> f64 {
        self.onset_s[parent] + distance_km / self.max_speed_km_s
    }

    /// The pair within reach with the least bound, as `(parent, child, bound)`, or
    /// `None` when no pair is within reach.
    ///
    /// # Panics
    ///
    /// If either chart has more than `u32::MAX` cells.
    #[must_use]
    pub fn least_bound(&self) -> Option<(usize, usize, f64)> {
        let mut best: Option<(usize, usize, f64)> = None;
        for &parent in &self.order {
            let threshold = best.map_or(f64::INFINITY, |(_, _, bound)| bound);
            if self.onset_s[parent] >= threshold {
                break;
            }
            self.child.within(
                self.parent(parent),
                self.radius(parent, threshold),
                |child, d| {
                    let bound = self.bound(parent, d);
                    if best.is_none_or(|(_, _, least)| bound < least) {
                        best = Some((parent, child, bound));
                    }
                },
            );
        }
        best
    }

    /// Every pair within reach whose bound is at or below `threshold_s`, as parallel
    /// `(parents, children)`. Parents come in onset order, and each parent's children
    /// in bin order, so the result is deterministic.
    ///
    /// # Panics
    ///
    /// If either chart has more than `u32::MAX` cells.
    #[must_use]
    pub fn pairs_below(&self, threshold_s: f64) -> (Vec<u32>, Vec<u32>) {
        // In a uniform medium a bound equals its arrival, so a rounding error could put
        // the pair that set the threshold above it.
        let threshold_s = threshold_s + REACH_SLACK * threshold_s.abs();
        let live = self
            .order
            .partition_point(|&parent| self.onset_s[parent] <= threshold_s);
        let per_parent: Vec<Vec<u32>> = self.order[..live]
            .par_iter()
            .map(|&parent| {
                let mut children = Vec::new();
                self.child.within(
                    self.parent(parent),
                    self.radius(parent, threshold_s),
                    |child, d| {
                        if self.bound(parent, d) <= threshold_s {
                            children.push(u32::try_from(child).expect("cell counts fit in u32"));
                        }
                    },
                );
                children
            })
            .collect();
        let mut parents = Vec::new();
        let mut children = Vec::new();
        for (&parent, found) in self.order[..live].iter().zip(per_parent) {
            let parent = u32::try_from(parent).expect("cell counts fit in u32");
            parents.extend(std::iter::repeat_n(parent, found.len()));
            children.extend(found);
        }
        (parents, children)
    }
}
