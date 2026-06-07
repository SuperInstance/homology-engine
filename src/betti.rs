//! Betti numbers and persistent Betti numbers.
//!
//! Betti numbers count the number of independent topological features in each dimension:
//!
//! - **β₀** = number of connected components
//! - **β₁** = number of loops (1-dimensional holes)
//! - **β₂** = number of voids (2-dimensional cavities)
//! - **βₖ** = number of k-dimensional holes
//!
//! Formally, βₖ = rank(Hₖ) = rank(ker(∂ₖ)) - rank(im(∂ₖ₊₁)).
//!
//! # Persistent Betti Numbers
//!
//! At a given threshold ε, the persistent Betti number βₖ(ε) counts features
//! that have been born but not yet died at that filtration value.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::reduction::ReducedMatrix;

/// Betti numbers for each homology dimension.
///
/// Computed from a reduced boundary matrix. The Betti number βₖ counts
/// the number of k-dimensional holes in the simplicial complex.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BettiNumbers {
    /// Map from dimension k to βₖ.
    betti: HashMap<usize, usize>,
    /// Maximum dimension encountered.
    max_dim: usize,
}

impl BettiNumbers {
    /// Create empty Betti numbers.
    pub fn new() -> Self {
        Self {
            betti: HashMap::new(),
            max_dim: 0,
        }
    }

    /// Compute Betti numbers from a reduced boundary matrix.
    ///
    /// βₖ = (number of unpaired k-simplices) - (number of paired k-simplices
    ///       where both birth and death are k-dimensional cycles that create/destroy
    ///       features of dimension k)
    ///
    /// More precisely: βₖ = (k-cycles that are born but never killed).
    /// In the reduction, this equals: (unpaired k-simplices acting as cycles)
    /// minus corrections from the pairing.
    pub fn from_reduced(reduced: &ReducedMatrix) -> Self {
        let mut result = Self::new();
        let boundary = reduced.reduced_matrix();

        // βₖ = #{unpaired simplices of dimension k}
        // Unpaired simplices are cycles that never die → they survive to the end
        // of the filtration, representing genuine homology classes.
        let max_dim = boundary
            .columns()
            .iter()
            .enumerate()
            .map(|(i, _)| boundary.simplex_dim(i))
            .max()
            .unwrap_or(0);

        result.max_dim = max_dim;
        for &col in reduced.unpaired() {
            let dim = boundary.simplex_dim(col);
            *result.betti.entry(dim).or_insert(0) += 1;
        }

        result
    }

    /// Get βₖ (Betti number in dimension k). Returns 0 if no features in dimension k.
    pub fn at(&self, k: usize) -> usize {
        self.betti.get(&k).copied().unwrap_or(0)
    }

    /// Get the total Betti number (sum over all dimensions).
    pub fn total(&self) -> usize {
        self.betti.values().sum()
    }

    /// Get the maximum dimension with non-zero Betti number.
    pub fn max_dimension(&self) -> usize {
        *self.betti.keys().max().unwrap_or(&0)
    }

    /// Get all Betti numbers as a vector, indexed by dimension.
    pub fn to_vec(&self) -> Vec<usize> {
        let max = self.max_dim;
        (0..=max).map(|k| self.at(k)).collect()
    }

    /// Compute Betti curve: Betti numbers as a function of filtration index.
    ///
    /// Returns a vector where `betti_curve[ε] = βₖ(at filtration step ε)`.
    /// At each step, a feature is born or killed, changing the Betti number.
    pub fn betti_curve(reduced: &ReducedMatrix, dim: usize) -> Vec<usize> {
        let boundary = reduced.reduced_matrix();
        let n = boundary.num_cols();
        let mut curve = vec![0usize; n + 1];

        // Track births and deaths by filtration index
        let mut births: Vec<usize> = vec![];
        let mut deaths: Vec<usize> = vec![];

        for &(birth, death) in reduced.pairs() {
            if boundary.simplex_dim(birth) == dim {
                births.push(birth);
                deaths.push(death);
            }
        }
        for &col in reduced.unpaired() {
            if boundary.simplex_dim(col) == dim {
                births.push(col);
            }
        }

        let mut current = 0usize;
        for step in 0..n {
            // Count births at this step
            current += births.iter().filter(|&&b| b == step).count();
            // Count deaths at this step
            current -= deaths.iter().filter(|&&d| d == step).count();
            curve[step + 1] = current;
        }

        curve
    }

    /// Compute persistent Betti number at a given filtration threshold.
    ///
    /// βₖ(ε) = number of features born before or at ε that haven't died yet.
    pub fn persistent_betti(reduced: &ReducedMatrix, dim: usize, threshold: usize) -> usize {
        let boundary = reduced.reduced_matrix();
        let mut count = 0usize;

        // Count infinite bars born ≤ threshold
        for &col in reduced.unpaired() {
            if boundary.simplex_dim(col) == dim && col <= threshold {
                count += 1;
            }
        }

        // Count finite bars where birth ≤ threshold < death
        for &(birth, death) in reduced.pairs() {
            if boundary.simplex_dim(birth) == dim && birth <= threshold && death > threshold {
                count += 1;
            }
        }

        count
    }
}

impl Default for BettiNumbers {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::BoundaryMatrix;

    fn make_reduced(simplices: &[Vec<usize>]) -> ReducedMatrix {
        let b = BoundaryMatrix::from_simplices(simplices);
        b.reduce()
    }

    #[test]
    fn test_single_vertex() {
        let r = make_reduced(&[vec![0]]);
        let betti = BettiNumbers::from_reduced(&r);
        assert_eq!(betti.at(0), 1); // One component
        assert_eq!(betti.at(1), 0);
    }

    #[test]
    fn test_single_edge() {
        let r = make_reduced(&[vec![0], vec![1], vec![0, 1]]);
        let betti = BettiNumbers::from_reduced(&r);
        assert_eq!(betti.at(0), 1); // Edge connects → 1 component
        assert_eq!(betti.at(1), 0);
    }

    #[test]
    fn test_triangle_hollow() {
        let r = make_reduced(&[
            vec![0],
            vec![1],
            vec![2],
            vec![0, 1],
            vec![0, 2],
            vec![1, 2],
        ]);
        let betti = BettiNumbers::from_reduced(&r);
        assert_eq!(betti.at(0), 1); // One component
        assert_eq!(betti.at(1), 1); // One loop!
    }

    #[test]
    fn test_triangle_filled() {
        let r = make_reduced(&[
            vec![0],
            vec![1],
            vec![2],
            vec![0, 1],
            vec![0, 2],
            vec![1, 2],
            vec![0, 1, 2],
        ]);
        let betti = BettiNumbers::from_reduced(&r);
        assert_eq!(betti.at(0), 1); // One component
        assert_eq!(betti.at(1), 0); // Loop is filled
    }

    #[test]
    fn test_two_disconnected_vertices() {
        let r = make_reduced(&[vec![0], vec![1]]);
        let betti = BettiNumbers::from_reduced(&r);
        assert_eq!(betti.at(0), 2); // Two components
    }

    #[test]
    fn test_empty_complex() {
        let r = BoundaryMatrix::new(0, 0).reduce();
        let betti = BettiNumbers::from_reduced(&r);
        assert_eq!(betti.total(), 0);
    }

    #[test]
    fn test_betti_curve() {
        let r = make_reduced(&[vec![0], vec![1], vec![0, 1]]);
        let curve = BettiNumbers::betti_curve(&r, 0);
        // Curve tracks β₀ over filtration steps
        assert!(!curve.is_empty());
    }

    #[test]
    fn test_persistent_betti() {
        let r = make_reduced(&[
            vec![0],
            vec![1],
            vec![2],
            vec![0, 1],
            vec![0, 2],
            vec![1, 2],
        ]);

        // At step 0: β₀ = 1 (first vertex)
        assert_eq!(BettiNumbers::persistent_betti(&r, 0, 0), 1);
        // At step 5 (all added): β₀ = 1, β₁ = 1
        assert_eq!(BettiNumbers::persistent_betti(&r, 0, 5), 1);
        assert_eq!(BettiNumbers::persistent_betti(&r, 1, 5), 1);
    }

    #[test]
    fn test_to_vec() {
        let r = make_reduced(&[
            vec![0],
            vec![1],
            vec![2],
            vec![0, 1],
            vec![0, 2],
            vec![1, 2],
        ]);
        let betti = BettiNumbers::from_reduced(&r);
        let v = betti.to_vec();
        assert_eq!(v[0], 1); // β₀
        assert_eq!(v[1], 1); // β₁
    }
}
