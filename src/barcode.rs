//! Persistence barcodes.
//!
//! A barcode is the canonical descriptor of persistent homology. It consists of a
//! collection of half-open intervals [birth, death) representing the lifetimes of
//! topological features across a filtration.
//!
//! # Interpretation
//!
//! - Each bar [b, d) represents a homology class that appears (is born) at
//!   filtration value b and disappears (dies) at filtration value d.
//! - Infinite bars [b, ∞) represent features that persist through the entire filtration.
//! - The length d - b measures the *persistence* of the feature — longer bars are
//!   more statistically significant.
//!
//! # Example
//!
//! ```
//! use homology_engine::barcode::Barcode;
//!
//! let mut barcode = Barcode::new();
//! barcode.add_finite(0, 3, 0);   // H₀ bar: born at 0, dies at 3
//! barcode.add_finite(0, 5, 0);   // H₀ bar: born at 0, dies at 5
//! barcode.add_infinite(0, 0);    // H₀ bar: born at 0, never dies
//! barcode.add_finite(1, 4, 1);   // H₁ bar: born at 1, dies at 4
//!
//! assert_eq!(barcode.len(), 4);
//! // Finite bars lifetimes: 3, 5, 3. Average = 11/3 ≈ 3.667
//! assert!((barcode.average_lifetime() - 11.0/3.0).abs() < 0.01);
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::reduction::ReducedMatrix;

/// A persistence barcode: collection of intervals [birth, death).
///
/// Each interval tracks the lifetime of a topological feature (connected component,
/// loop, void, etc.) across a filtration parameter.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Barcode {
    /// Finite bars: (birth, death, dimension).
    finite: Vec<(usize, usize, usize)>,
    /// Infinite bars: (birth, dimension).
    infinite: Vec<(usize, usize)>,
}

impl Barcode {
    /// Create an empty barcode.
    pub fn new() -> Self {
        Self {
            finite: vec![],
            infinite: vec![],
        }
    }

    /// Create a barcode from a reduced matrix.
    ///
    /// Extracts persistence pairs (finite bars) and unpaired columns (infinite bars)
    /// from the reduction result.
    pub fn from_reduced(reduced: &ReducedMatrix) -> Self {
        let mut barcode = Self::new();

        // Extract finite bars from pairs
        for &(birth, death) in reduced.pairs() {
            let dim = reduced.reduced_matrix().simplex_dim(birth);
            barcode.add_finite(birth, death, dim);
        }

        // Extract infinite bars from unpaired columns
        for &birth in reduced.unpaired() {
            let dim = reduced.reduced_matrix().simplex_dim(birth);
            barcode.add_infinite(birth, dim);
        }

        barcode
    }

    /// Add a finite bar [birth, death) in the given dimension.
    pub fn add_finite(&mut self, birth: usize, death: usize, dim: usize) {
        self.finite.push((birth, death, dim));
    }

    /// Add an infinite bar [birth, ∞) in the given dimension.
    pub fn add_infinite(&mut self, birth: usize, dim: usize) {
        self.infinite.push((birth, dim));
    }

    /// Total number of bars (finite + infinite).
    pub fn len(&self) -> usize {
        self.finite.len() + self.infinite.len()
    }

    /// Check if the barcode is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get finite bars as a slice of (birth, death, dimension).
    pub fn finite_bars(&self) -> &[(usize, usize, usize)] {
        &self.finite
    }

    /// Get infinite bars as a slice of (birth, dimension).
    pub fn infinite_bars(&self) -> &[(usize, usize)] {
        &self.infinite
    }

    /// Get bars of a specific dimension.
    pub fn bars_in_dimension(&self, dim: usize) -> (Vec<(usize, usize)>, Vec<usize>) {
        let finite: Vec<(usize, usize)> = self
            .finite
            .iter()
            .filter(|(_, _, d)| *d == dim)
            .map(|(b, e, _)| (*b, *e))
            .collect();
        let infinite: Vec<usize> = self
            .infinite
            .iter()
            .filter(|(_, d)| *d == dim)
            .map(|(b, _)| *b)
            .collect();
        (finite, infinite)
    }

    /// Compute the average lifetime of all finite bars.
    ///
    /// Lifetime of bar [b, d) = d - b. Infinite bars are excluded
    /// (they have no finite death value).
    pub fn average_lifetime(&self) -> f64 {
        if self.finite.is_empty() {
            return 0.0;
        }
        let total: usize = self.finite.iter().map(|(b, d, _)| d - b).sum();
        total as f64 / self.finite.len() as f64
    }

    /// Compute total persistence: sum of (death - birth)² for all finite bars.
    ///
    /// This is a common summary statistic used in stability arguments.
    pub fn total_persistence(&self, p: f64) -> f64 {
        self.finite
            .iter()
            .map(|(b, d, _)| ((d - b) as f64).powf(p))
            .sum()
    }

    /// Group bars by dimension, returning a map from dimension to (finite, infinite) counts.
    pub fn dimension_counts(&self) -> HashMap<usize, (usize, usize)> {
        let mut counts: HashMap<usize, (usize, usize)> = HashMap::new();
        for (_, _, dim) in &self.finite {
            counts.entry(*dim).or_insert((0, 0)).0 += 1;
        }
        for (_, dim) in &self.infinite {
            counts.entry(*dim).or_insert((0, 0)).1 += 1;
        }
        counts
    }

    /// Compute persistence entropy.
    ///
    /// Measures the information content of the barcode:
    /// E = -Σ pᵢ log(pᵢ) where pᵢ = |dᵢ - bᵢ| / Σ|dⱼ - bⱼ|
    pub fn entropy(&self) -> f64 {
        if self.finite.is_empty() {
            return 0.0;
        }
        let lifetimes: Vec<f64> = self.finite.iter().map(|(b, d, _)| (d - b) as f64).collect();
        let total: f64 = lifetimes.iter().sum();
        if total == 0.0 {
            return 0.0;
        }
        lifetimes
            .iter()
            .filter(|&&l| l > 0.0)
            .map(|&l| {
                let p = l / total;
                -p * p.log2()
            })
            .sum()
    }
}

impl Default for Barcode {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_barcode() {
        let bc = Barcode::new();
        assert!(bc.is_empty());
        assert_eq!(bc.len(), 0);
        assert_eq!(bc.average_lifetime(), 0.0);
    }

    #[test]
    fn test_finite_bars() {
        let mut bc = Barcode::new();
        bc.add_finite(0, 3, 0);
        bc.add_finite(1, 5, 1);
        assert_eq!(bc.len(), 2);
        assert_eq!(bc.finite_bars().len(), 2);
        assert_eq!(bc.average_lifetime(), (3.0 + 4.0) / 2.0);
    }

    #[test]
    fn test_infinite_bars() {
        let mut bc = Barcode::new();
        bc.add_infinite(0, 0);
        bc.add_infinite(2, 1);
        assert_eq!(bc.len(), 2);
        assert_eq!(bc.infinite_bars().len(), 2);
        assert_eq!(bc.average_lifetime(), 0.0); // no finite bars
    }

    #[test]
    fn test_mixed_bars() {
        let mut bc = Barcode::new();
        bc.add_finite(0, 3, 0);
        bc.add_infinite(0, 0);
        assert_eq!(bc.len(), 2);
        assert_eq!(bc.average_lifetime(), 3.0);
    }

    #[test]
    fn test_bars_in_dimension() {
        let mut bc = Barcode::new();
        bc.add_finite(0, 3, 0);
        bc.add_finite(1, 4, 1);
        bc.add_infinite(0, 0);
        bc.add_infinite(2, 1);

        let (f0, i0) = bc.bars_in_dimension(0);
        assert_eq!(f0.len(), 1);
        assert_eq!(i0.len(), 1);

        let (f1, i1) = bc.bars_in_dimension(1);
        assert_eq!(f1.len(), 1);
        assert_eq!(i1.len(), 1);
    }

    #[test]
    fn test_total_persistence() {
        let mut bc = Barcode::new();
        bc.add_finite(0, 3, 0); // lifetime 3
        bc.add_finite(1, 5, 0); // lifetime 4
        assert_eq!(bc.total_persistence(1.0), 7.0);
        assert!((bc.total_persistence(2.0) - (9.0 + 16.0)).abs() < 1e-10);
    }

    #[test]
    fn test_dimension_counts() {
        let mut bc = Barcode::new();
        bc.add_finite(0, 3, 0);
        bc.add_finite(1, 4, 0);
        bc.add_finite(2, 5, 1);
        bc.add_infinite(0, 0);

        let counts = bc.dimension_counts();
        assert_eq!(counts.get(&0), Some(&(2, 1)));
        assert_eq!(counts.get(&1), Some(&(1, 0)));
    }

    #[test]
    fn test_entropy() {
        let mut bc = Barcode::new();
        // Two bars with equal lifetime → entropy = log2(2) = 1
        bc.add_finite(0, 5, 0);
        bc.add_finite(0, 5, 0);
        assert!((bc.entropy() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_from_reduced() {
        let simplices: Vec<Vec<usize>> = vec![vec![0], vec![1], vec![0, 1]];
        let boundary = crate::boundary::BoundaryMatrix::from_simplices(&simplices);
        let reduced = boundary.reduce();
        let bc = Barcode::from_reduced(&reduced);

        // One pair + one unpaired
        assert_eq!(bc.len(), 2);
    }
}
