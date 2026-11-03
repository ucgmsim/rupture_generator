//! What the jump search must find, checked against reading every pair.
//!
//! The search prunes with `onset + distance / v_max`, so whatever it skips must be a
//! pair that brute force would also rule out. Every property compares the search with
//! the `O(P C)` loop it replaces, over random clouds of parent and child cells.

// The search and brute force share one distance routine and one bound formula, so
// they agree exactly, and a tolerance would hide a pair the search got wrong.
#![expect(
    clippy::float_cmp,
    reason = "the search must match brute force exactly"
)]

use _kernels::jump::Search;
use proptest::prelude::*;

/// Parent cells, their onsets, child cells, a speed bound, and how far `reach_km`
/// exceeds the gap.
type Inputs = (Vec<[f64; 3]>, Vec<f64>, Vec<[f64; 3]>, f64, f64);

fn distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn cloud(max: usize) -> impl Strategy<Value = Vec<[f64; 3]>> {
    prop::collection::vec(prop::array::uniform3(-10.0..10.0f64), 1..max)
}

fn inputs() -> impl Strategy<Value = Inputs> {
    (cloud(60), cloud(60), 1.0..6.0f64, 0.0..8.0f64).prop_flat_map(
        |(parent, child, speed, extra)| {
            let onsets = prop::collection::vec(0.0..10.0f64, parent.len());
            (Just(parent), onsets, Just(child), Just(speed), Just(extra))
        },
    )
}

fn search(parent: &[[f64; 3]], onset: &[f64], child: &[[f64; 3]], speed: f64) -> Search {
    Search::new(parent.to_vec(), onset.to_vec(), child.to_vec(), speed)
        .expect("generated inputs are valid")
}

proptest! {
    #[test]
    fn the_gap_is_the_least_distance((parent, onset, child, speed, _) in inputs()) {
        let gap = parent
            .iter()
            .flat_map(|p| child.iter().map(move |c| distance(p, c)))
            .fold(f64::INFINITY, f64::min);
        prop_assert_eq!(search(&parent, &onset, &child, speed).min_gap_km(), gap);
    }

    #[test]
    fn the_least_bound_and_the_pairs_below_it_match_brute_force(
        (parent, onset, child, speed, extra) in inputs()
    ) {
        let search = search(&parent, &onset, &child, speed);
        let reach = search.min_gap_km() + extra;
        let bound = |p: usize, c: usize| onset[p] + distance(&parent[p], &child[c]) / speed;
        let within: Vec<(usize, usize)> = (0..parent.len())
            .flat_map(|p| (0..child.len()).map(move |c| (p, c)))
            .filter(|&(p, c)| distance(&parent[p], &child[c]) <= reach)
            .collect();
        let least = within
            .iter()
            .map(|&(p, c)| bound(p, c))
            .fold(f64::INFINITY, f64::min);

        let (p, c) = search.least_bound(reach).expect("the reach covers the gap");
        prop_assert_eq!(bound(p as usize, c as usize), least);

        // Any threshold keeps exactly the pairs brute force keeps.
        let threshold = least + extra;
        let (parents, children) = search.pairs_below(reach, threshold);
        let mut found: Vec<(usize, usize)> = parents
            .iter()
            .zip(&children)
            .map(|(&p, &c)| (p as usize, c as usize))
            .collect();
        let mut expected: Vec<(usize, usize)> = within
            .into_iter()
            .filter(|&(p, c)| bound(p, c) <= threshold)
            .collect();
        found.sort_unstable();
        expected.sort_unstable();
        prop_assert_eq!(found, expected);
    }

    #[test]
    fn an_infinite_bound_prunes_by_onset_alone((parent, onset, child, _, extra) in inputs()) {
        let search = search(&parent, &onset, &child, f64::INFINITY);
        let reach = search.min_gap_km() + extra;
        let (p, _) = search.least_bound(reach).expect("the reach covers the gap");
        let earliest = (0..parent.len())
            .filter(|&p| child.iter().any(|c| distance(&parent[p], c) <= reach))
            .map(|p| onset[p])
            .fold(f64::INFINITY, f64::min);
        prop_assert_eq!(onset[p as usize], earliest);
    }
}
