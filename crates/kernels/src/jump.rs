//! Pruning the pairs a jump can cross between, by a lower bound on its arrival.
//!
//! A jump from a parent cell `p` onto a child cell `c` arrives at
//! `onset[p] + T(p, c)`, and the child starts at the earliest arrival over every pair
//! no farther apart than `reach_km`. Reading the medium along every pair costs
//! `O(P C)`. A shear wave is never faster than the medium's largest shear speed, so
//! `T(p, c) >= |p - c| / v_max`, and
//!
//! ```text
//! bound(p, c) = onset[p] + |p - c| / v_max  <=  arrival(p, c).
//! ```
//!
//! The bound doesn't read the medium, so it can prune before anything reads the rock. Any
//! real arrival `t` rules out every pair whose bound exceeds `t`. The caller reads
//! the medium along the pair with the least bound ([`Search::least_bound`]) and gets a
//! real arrival from it. It then reads the medium only along the pairs whose bound is at
//! or below that arrival ([`Search::pairs_below`]). The earliest arrival is among
//! those pairs, so the search is exact. The search visits pairs in order of an admissible
//! heuristic, as A* does, except that a crossing is one straight line and not a path
//! through a graph.
//!
//! Visiting parent cells in onset order makes the search cheap. A parent cell whose
//! onset alone exceeds the threshold can't win, and the remaining ones only need the
//! child cells within `(threshold - onset) v_max`, which a k-d tree finds. An
//! infinite `v_max` is a valid bound that prunes by onset alone.

use kiddo::{Eytzinger, FlatVec, KdTree, SquaredEuclidean};
use rayon::prelude::*;

/// The child cells, as a k-d tree whose items index them.
type Tree = KdTree<f64, u32, Eytzinger, FlatVec<f64, u32, 3, 32>, 3, 32>;

/// How much wider than its radius a tree query runs, relatively. The tree's squared
/// distances can round differently from [`distance_km`], so a query that ran at
/// exactly the radius could drop a pair on its boundary. The test against the radius
/// itself is always [`distance_km`]'s.
const QUERY_MARGIN: f64 = 1.0e-9;

/// What the jump search refuses, in its own vocabulary.
#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
pub enum Error {
    /// Each parent cell needs an onset.
    #[error("{onsets} onsets for {cells} parent cells")]
    MismatchedOnsets { onsets: usize, cells: usize },
    /// Both charts need a cell to cross from or onto.
    #[error("a jump needs cells on both charts, and one has none")]
    NoCells,
    /// A speed bound is positive. Infinity is a valid, weak bound, and NaN isn't one.
    #[error("a shear-speed bound of {max_speed_km_s} km/s; it must be positive")]
    BadSpeed { max_speed_km_s: f64 },
}

/// The one distance routine, so the narrowest gap that sets `reach_km` and the test
/// against `reach_km` agree to the bit.
fn distance_km(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Parent cells in onset order, and a tree over child cells: built once per jump.
pub struct Search {
    parent_km: Vec<[f64; 3]>,
    onset_s: Vec<f64>,
    child_km: Vec<[f64; 3]>,
    max_speed_km_s: f64,
    /// Parent cells in onset order, the earliest first. Ties keep index order, so the
    /// search is deterministic.
    order: Vec<u32>,
    child: Tree,
}

impl Search {
    /// Prepare a search between parent cells, with their onsets, and child cells.
    ///
    /// Positions are in kilometres. `max_speed_km_s` bounds the shear speed anywhere
    /// a crossing can pass, and infinity is always valid.
    ///
    /// # Errors
    ///
    /// [`Error`] if the arrays disagree, a chart has no cells, or the speed bound is
    /// invalid.
    ///
    /// # Panics
    ///
    /// A chart with more than `u32::MAX` cells panics, since the indices are `u32`.
    pub fn new(
        parent_km: Vec<[f64; 3]>,
        onset_s: Vec<f64>,
        child_km: Vec<[f64; 3]>,
        max_speed_km_s: f64,
    ) -> Result<Self, Error> {
        if onset_s.len() != parent_km.len() {
            return Err(Error::MismatchedOnsets {
                onsets: onset_s.len(),
                cells: parent_km.len(),
            });
        }
        if parent_km.is_empty() || child_km.is_empty() {
            return Err(Error::NoCells);
        }
        if max_speed_km_s.is_nan() || max_speed_km_s <= 0.0 {
            return Err(Error::BadSpeed { max_speed_km_s });
        }
        let count = u32::try_from(parent_km.len()).expect("cell counts fit in u32");
        let mut order: Vec<u32> = (0..count).collect();
        order.sort_by(|&a, &b| onset_s[a as usize].total_cmp(&onset_s[b as usize]));
        let child = Tree::new_from_slice(&child_km).expect("finite cell centres build a tree");
        Ok(Self {
            parent_km,
            onset_s,
            child_km,
            max_speed_km_s,
            order,
            child,
        })
    }

    /// The narrowest gap between the charts: the least distance over all pairs.
    #[must_use]
    pub fn min_gap_km(&self) -> f64 {
        self.parent_km
            .par_iter()
            .map(|parent| {
                let nearest = self
                    .child
                    .query(parent)
                    .nearest_one::<SquaredEuclidean<f64>>()
                    .execute();
                distance_km(parent, &self.child_km[nearest.item as usize])
            })
            .reduce(|| f64::INFINITY, f64::min)
    }

    /// Call `visit(child, bound)` for each child cell within `reach_km` of `parent`
    /// whose bound is at or below `threshold_s`.
    fn visit(&self, parent: u32, reach_km: f64, threshold_s: f64, mut visit: impl FnMut(u32, f64)) {
        let (centre, onset_s) = (
            &self.parent_km[parent as usize],
            self.onset_s[parent as usize],
        );
        // With an infinite bound or threshold the product is infinite or NaN, and
        // `min` takes `reach_km` either way.
        let radius_km = reach_km.min((threshold_s - onset_s) * self.max_speed_km_s);
        let squared = (radius_km * radius_km) * (1.0 + QUERY_MARGIN);
        self.child
            .query(centre)
            .within::<SquaredEuclidean<f64>>(squared)
            .unsorted()
            .visit(|found| {
                let distance = distance_km(centre, &self.child_km[found.item as usize]);
                let bound = onset_s + distance / self.max_speed_km_s;
                if distance <= reach_km && bound <= threshold_s {
                    visit(found.item, bound);
                }
            });
    }

    /// The pair no farther apart than `reach_km` with the least bound, as
    /// `(parent, child)`, or `None` when no pair is that close.
    #[must_use]
    pub fn least_bound(&self, reach_km: f64) -> Option<(u32, u32)> {
        let mut best: Option<(u32, u32, f64)> = None;
        for &parent in &self.order {
            let threshold_s = best.map_or(f64::INFINITY, |(_, _, bound)| bound);
            if self.onset_s[parent as usize] >= threshold_s {
                break;
            }
            self.visit(parent, reach_km, threshold_s, |child, bound| {
                if best.is_none_or(|(_, _, least)| bound < least) {
                    best = Some((parent, child, bound));
                }
            });
        }
        best.map(|(parent, child, _)| (parent, child))
    }

    /// Every pair no farther apart than `reach_km` with a bound at or below
    /// `threshold_s`, as parallel `(parents, children)`, in onset order of the parent.
    #[must_use]
    pub fn pairs_below(&self, reach_km: f64, threshold_s: f64) -> (Vec<u32>, Vec<u32>) {
        let live = self
            .order
            .partition_point(|&parent| self.onset_s[parent as usize] <= threshold_s);
        self.order[..live]
            .par_iter()
            .flat_map_iter(|&parent| {
                let mut pairs = Vec::new();
                self.visit(parent, reach_km, threshold_s, |child, _| {
                    pairs.push((parent, child));
                });
                pairs
            })
            .unzip()
    }
}
