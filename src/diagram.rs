//! Persistence diagrams and distance metrics.
//!
//! A persistence diagram is a multiset of points (b, d) in the plane where each
//! point represents a topological feature born at value b and dying at value d.
//!
//! # Distance Metrics
//!
//! The two fundamental distances between persistence diagrams are:
//!
//! - **Bottleneck distance**: d_B = inf_M sup_p ||p - M(p)||_∞
//!   The best possible matching between two diagrams, maximizing the minimum distance.
//!
//! - **Wasserstein distance**: W_p = (inf_M Σ ||p - M(p)||_∞^p)^(1/p)
//!   The best matching minimizing the total cost.
//!
//! # Stability Theorem
//!
//! The stability theorem guarantees:
//!
//! ```text
//! d_B(Dgm(f), Dgm(g)) ≤ ||f - g||_∞
//! ```
//!
//! Small perturbations in input → small perturbations in the diagram.

use serde::{Deserialize, Serialize};

/// A persistence diagram: a multiset of (birth, death) points.
///
/// Each point represents a topological feature. Points close to the diagonal
/// (birth ≈ death) are short-lived and likely noise. Points far from the diagonal
/// are significant features.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PersistenceDiagram {
    /// Points in the diagram: (birth, death, dimension).
    points: Vec<(f64, f64, usize)>,
}

impl PersistenceDiagram {
    /// Create an empty persistence diagram.
    pub fn new() -> Self {
        Self { points: vec![] }
    }

    /// Create from a barcode.
    pub fn from_barcode(barcode: &crate::barcode::Barcode) -> Self {
        let mut diagram = Self::new();
        for &(b, d, dim) in barcode.finite_bars() {
            diagram.add_point(b as f64, d as f64, dim);
        }
        // For infinite bars, use f64::INFINITY as death
        for &(b, dim) in barcode.infinite_bars() {
            diagram.add_point(b as f64, f64::INFINITY, dim);
        }
        diagram
    }

    /// Add a point to the diagram.
    pub fn add_point(&mut self, birth: f64, death: f64, dim: usize) {
        self.points.push((birth, death, dim));
    }

    /// Get all points as a slice.
    pub fn points(&self) -> &[(f64, f64, usize)] {
        &self.points
    }

    /// Get points of a specific dimension.
    pub fn points_in_dimension(&self, dim: usize) -> Vec<(f64, f64)> {
        self.points
            .iter()
            .filter(|(_, _, d)| *d == dim)
            .map(|(b, e, _)| (*b, *e))
            .collect()
    }

    /// Number of points in the diagram.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Check if the diagram is empty.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Compute the total persistence: Σ (death - birth)^p for all finite points.
    pub fn total_persistence(&self, p: f64) -> f64 {
        self.points
            .iter()
            .filter(|(_, d, _)| d.is_finite())
            .map(|(b, d, _)| (d - b).powf(p))
            .sum()
    }

    /// Compute persistence entropy.
    pub fn entropy(&self) -> f64 {
        let lifetimes: Vec<f64> = self
            .points
            .iter()
            .filter(|(_, d, _)| d.is_finite())
            .map(|(b, d, _)| d - b)
            .filter(|&l| l > 0.0)
            .collect();
        if lifetimes.is_empty() {
            return 0.0;
        }
        let total: f64 = lifetimes.iter().sum();
        if total == 0.0 {
            return 0.0;
        }
        lifetimes
            .iter()
            .map(|&l| {
                let p = l / total;
                -p * p.log2()
            })
            .sum()
    }

    /// Compute the bottleneck distance to another diagram.
    ///
    /// d_B(D₁, D₂) = max over all optimal matchings of the maximum L∞ distance.
    ///
    /// This implementation uses a greedy matching approach with diagonal projection
    /// for unmatched points. For exact computation, the Hungarian algorithm on the
    /// bipartite graph would be needed, but the greedy approach gives an upper bound.
    ///
    /// **Complexity**: O(n²) where n = max(|D₁|, |D₂|).
    pub fn bottleneck_distance(&self, other: &PersistenceDiagram) -> f64 {
        self.distance_with_matching(other, f64::INFINITY)
    }

    /// Compute the Wasserstein-p distance to another diagram.
    ///
    /// W_p(D₁, D₂) = (inf_M Σ ||a - M(a)||_∞^p)^(1/p)
    ///
    /// Uses a greedy matching with diagonal projection for unmatched points.
    pub fn wasserstein_distance(&self, other: &PersistenceDiagram, p: f64) -> f64 {
        let cost = self.distance_with_matching(other, p);
        if p.is_infinite() || p == 1.0 {
            cost
        } else {
            cost.powf(1.0 / p)
        }
    }

    /// Internal: compute matching cost using greedy nearest-neighbor.
    fn distance_with_matching(&self, other: &PersistenceDiagram, p: f64) -> f64 {
        let pts_a: Vec<(f64, f64)> = self
            .points
            .iter()
            .filter(|(_, d, _)| d.is_finite())
            .map(|(b, d, _)| (*b, *d))
            .collect();
        let pts_b: Vec<(f64, f64)> = other
            .points
            .iter()
            .filter(|(_, d, _)| d.is_finite())
            .map(|(b, d, _)| (*b, *d))
            .collect();

        if pts_a.is_empty() && pts_b.is_empty() {
            return 0.0;
        }

        // Compute L∞ distance from a point to the diagonal (b = d)
        let diag_dist = |b: f64, d: f64| (d - b).abs() / 2.0;

        // For each point in A, find nearest unmatched point in B (greedy)
        let mut matched_b: Vec<bool> = vec![false; pts_b.len()];
        let mut max_cost = 0.0_f64;

        for &(b_a, d_a) in &pts_a {
            let mut best_cost = f64::INFINITY;
            let mut best_j = None;

            for (j, &(b_b, d_b)) in pts_b.iter().enumerate() {
                if matched_b[j] {
                    continue;
                }
                let cost = ((b_a - b_b).abs()).max((d_a - d_b).abs());
                if cost < best_cost {
                    best_cost = cost;
                    best_j = Some(j);
                }
            }

            // Compare matching to B vs. projecting to diagonal
            let diag_cost = diag_dist(b_a, d_a);
            if let Some(j) = best_j {
                if best_cost <= diag_cost {
                    matched_b[j] = true;
                    if p.is_infinite() {
                        max_cost = max_cost.max(best_cost);
                    } else {
                        max_cost += best_cost.powf(p);
                    }
                } else {
                    if p.is_infinite() {
                        max_cost = max_cost.max(diag_cost);
                    } else {
                        max_cost += diag_cost.powf(p);
                    }
                }
            } else {
                if p.is_infinite() {
                    max_cost = max_cost.max(diag_cost);
                } else {
                    max_cost += diag_cost.powf(p);
                }
            }
        }

        // Unmatched points in B → project to diagonal
        for (j, &(b_b, d_b)) in pts_b.iter().enumerate() {
            if !matched_b[j] {
                let diag_cost = diag_dist(b_b, d_b);
                if p.is_infinite() {
                    max_cost = max_cost.max(diag_cost);
                } else {
                    max_cost += diag_cost.powf(p);
                }
            }
        }

        if p.is_infinite() {
            max_cost
        } else {
            // For p-norm, return the sum (caller takes the p-th root)
            max_cost
        }
    }
}

impl Default for PersistenceDiagram {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_diagram() {
        let d = PersistenceDiagram::new();
        assert!(d.is_empty());
        assert_eq!(d.total_persistence(1.0), 0.0);
    }

    #[test]
    fn test_add_points() {
        let mut d = PersistenceDiagram::new();
        d.add_point(0.0, 3.0, 0);
        d.add_point(1.0, 5.0, 1);
        assert_eq!(d.len(), 2);
        assert_eq!(d.points_in_dimension(0), vec![(0.0, 3.0)]);
        assert_eq!(d.points_in_dimension(1), vec![(1.0, 5.0)]);
    }

    #[test]
    fn test_total_persistence() {
        let mut d = PersistenceDiagram::new();
        d.add_point(0.0, 3.0, 0);
        d.add_point(1.0, 5.0, 0);
        assert_eq!(d.total_persistence(1.0), 7.0); // 3 + 4
        assert!((d.total_persistence(2.0) - (9.0 + 16.0)).abs() < 1e-10);
    }

    #[test]
    fn test_bottleneck_same_diagram() {
        let mut d1 = PersistenceDiagram::new();
        d1.add_point(0.0, 3.0, 0);
        d1.add_point(1.0, 5.0, 1);

        let d2 = d1.clone();
        assert_eq!(d1.bottleneck_distance(&d2), 0.0);
    }

    #[test]
    fn test_bottleneck_shifted() {
        let mut d1 = PersistenceDiagram::new();
        d1.add_point(0.0, 3.0, 0);

        let mut d2 = PersistenceDiagram::new();
        d2.add_point(0.5, 3.5, 0);

        let dist = d1.bottleneck_distance(&d2);
        assert!(dist > 0.0);
        assert!(dist <= 1.0);
    }

    #[test]
    fn test_wasserstein_same_diagram() {
        let mut d = PersistenceDiagram::new();
        d.add_point(0.0, 3.0, 0);
        let d2 = d.clone();
        assert_eq!(d.wasserstein_distance(&d2, 2.0), 0.0);
    }

    #[test]
    fn test_bottleneck_empty_diagrams() {
        let d1 = PersistenceDiagram::new();
        let d2 = PersistenceDiagram::new();
        assert_eq!(d1.bottleneck_distance(&d2), 0.0);
    }

    #[test]
    fn test_entropy() {
        let mut d = PersistenceDiagram::new();
        d.add_point(0.0, 5.0, 0);
        d.add_point(0.0, 5.0, 0);
        assert!((d.entropy() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_from_barcode() {
        let mut bc = crate::barcode::Barcode::new();
        bc.add_finite(0, 3, 0);
        bc.add_infinite(1, 0);
        let dgm = PersistenceDiagram::from_barcode(&bc);
        assert_eq!(dgm.len(), 2);
        // Finite bar → (0, 3)
        assert_eq!(dgm.points()[0], (0.0, 3.0, 0));
        // Infinite bar → (1, ∞)
        assert_eq!(dgm.points()[1], (1.0, f64::INFINITY, 0));
    }

    #[test]
    fn test_diagram_statistics() {
        let mut d = PersistenceDiagram::new();
        d.add_point(0.0, 10.0, 0);
        d.add_point(1.0, 2.0, 0);
        d.add_point(0.0, 5.0, 1);
        assert_eq!(d.total_persistence(1.0), 16.0); // 10 + 1 + 5
    }
}
