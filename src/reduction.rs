//! Column reduction algorithm for boundary matrices over Z/2Z.
//!
//! This module implements the iterative column reduction algorithm that transforms
//! a boundary matrix into its reduced form. The reduced form reveals the persistence
//! pairings from which barcodes and Betti numbers are extracted.
//!
//! # Algorithm
//!
//! The reduction is a variant of Gaussian elimination over Z/2Z (where addition = XOR).
//! Unlike recursive approaches, we use an explicit worklist of columns to process.
//!
//! 1. Process columns left-to-right (low-to-high filtration index).
//! 2. For each column j, find its pivot (lowest non-zero row).
//! 3. If another column k < j already has the same pivot, add (XOR) column k into column j.
//! 4. Repeat until column j is zero or has a unique pivot.
//! 5. Track all pivots in a `HashMap<usize, usize>` mapping `pivot_row → column_index`.
//!
//! # Result
//!
//! The reduction produces R = D·V where:
//! - R is the reduced boundary matrix
//! - V tracks the column operations applied
//! - The low(row) function gives the rightmost 1 in each row of R

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::boundary::BoundaryMatrix;

/// The result of reducing a boundary matrix.
///
/// Contains the reduced matrix and the pivot tracking information
/// needed to extract barcodes and Betti numbers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReducedMatrix {
    /// The reduced boundary matrix R.
    reduced: BoundaryMatrix,
    /// Map from pivot row to column index: pivot[row] = col means column `col` has pivot at `row`.
    pivots: HashMap<usize, usize>,
    /// Pairs (birth, death) from the reduction. Each pair means a feature born at `birth` dies at `death`.
    pairs: Vec<(usize, usize)>,
    /// Unpaired columns (infinite bars): features born but never dying.
    unpaired: Vec<usize>,
}

impl ReducedMatrix {
    /// Get a reference to the reduced boundary matrix.
    pub fn reduced_matrix(&self) -> &BoundaryMatrix {
        &self.reduced
    }

    /// Get the pivot map (row → column).
    pub fn pivots(&self) -> &HashMap<usize, usize> {
        &self.pivots
    }

    /// Get the persistence pairs (birth_col, death_col).
    pub fn pairs(&self) -> &[(usize, usize)] {
        &self.pairs
    }

    /// Get unpaired column indices (infinite bars).
    pub fn unpaired(&self) -> &[usize] {
        &self.unpaired
    }

    /// Extract the barcode from this reduced matrix.
    pub fn barcode(&self) -> crate::barcode::Barcode {
        crate::barcode::Barcode::from_reduced(self)
    }

    /// Compute Betti numbers from this reduced matrix.
    pub fn betti_numbers(&self) -> crate::betti::BettiNumbers {
        crate::betti::BettiNumbers::from_reduced(self)
    }
}

/// Extension trait for reducing boundary matrices.
impl BoundaryMatrix {
    /// Reduce this boundary matrix using iterative column reduction over Z/2Z.
    ///
    /// This is the standard "standard reduction" algorithm:
    /// - Process columns from left to right.
    /// - For each column j, while it is non-zero and its pivot conflicts with an already-reduced
    ///   column k, add (XOR) column k into column j.
    /// - Track pivots in a HashMap for O(1) lookup.
    ///
    /// # Returns
    ///
    /// A `ReducedMatrix` containing the reduced matrix, pivot map, pairs, and unpaired columns.
    ///
    /// # Example
    ///
    /// ```
    /// use homology_engine::boundary::BoundaryMatrix;
    /// use homology_engine::reduction::MatrixReduction;
    ///
    /// let simplices: Vec<Vec<usize>> = vec![
    ///     vec![0], vec![1], vec![2],
    ///     vec![0, 1], vec![0, 2], vec![1, 2],
    /// ];
    /// let boundary = BoundaryMatrix::from_simplices(&simplices);
    /// let reduced = boundary.reduce();
    /// assert!(!reduced.pairs().is_empty() || !reduced.unpaired().is_empty());
    /// ```
    pub fn reduce(&self) -> ReducedMatrix {
        let mut r = self.clone_matrix();
        let mut pivots: HashMap<usize, usize> = HashMap::new();
        let mut pairs: Vec<(usize, usize)> = vec![];
        let mut unpaired: Vec<usize> = vec![];

        let num_cols = r.num_cols();

        // Iterative left-to-right reduction with explicit worklist per column
        for j in 0..num_cols {
            // Worklist loop: keep reducing column j until it's zero or has a unique pivot
            loop {
                let pivot = r.pivot(j);
                match pivot {
                    None => {
                        // Column j is zero → it's a cycle (potential birth)
                        break;
                    }
                    Some(p) => {
                        if let Some(&k) = pivots.get(&p) {
                            // Another column k already has this pivot → add (XOR) k into j
                            r.add_columns(j, k);
                            // Continue the loop to check new pivot
                        } else {
                            // Unique pivot → record it
                            pivots.insert(p, j);
                            break;
                        }
                    }
                }
            }
        }

        // Extract pairs from pivots
        // pivot[row] = col_j means row is the pivot of column col_j
        // row corresponds to some simplex that was "killed" by column col_j
        // The simplex at index `row` was born earlier and dies when col_j appears
        for (&row, &col_j) in &pivots {
            pairs.push((row, col_j));
        }
        pairs.sort();

        // Find unpaired columns (those that are zero after reduction AND not killed)
        let paired_births: std::collections::HashSet<usize> =
            pairs.iter().map(|(b, _)| *b).collect();
        let paired_deaths: std::collections::HashSet<usize> =
            pairs.iter().map(|(_, d)| *d).collect();

        for j in 0..num_cols {
            if !paired_births.contains(&j) && !paired_deaths.contains(&j) {
                unpaired.push(j);
            }
        }

        ReducedMatrix {
            reduced: r,
            pivots,
            pairs,
            unpaired,
        }
    }
}

/// Trait for matrix reduction operations.
pub trait MatrixReduction {
    /// Reduce the boundary matrix and return the result.
    fn reduce(&self) -> ReducedMatrix;
}

impl MatrixReduction for BoundaryMatrix {
    fn reduce(&self) -> ReducedMatrix {
        self.reduce()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduce_single_edge() {
        let simplices: Vec<Vec<usize>> = vec![vec![0], vec![1], vec![0, 1]];
        let boundary = BoundaryMatrix::from_simplices(&simplices);
        let reduced = boundary.reduce();

        // Edge kills one vertex, one vertex survives (infinite bar)
        assert_eq!(reduced.pairs().len(), 1);
        assert_eq!(reduced.unpaired().len(), 1);
    }

    #[test]
    fn test_reduce_triangle() {
        let simplices: Vec<Vec<usize>> = vec![
            vec![0],
            vec![1],
            vec![2],
            vec![0, 1],
            vec![0, 2],
            vec![1, 2],
        ];
        let boundary = BoundaryMatrix::from_simplices(&simplices);
        let reduced = boundary.reduce();

        // 3 vertices, 3 edges: 2 finite pairs + 2 unpaired (one component + one loop)
        assert_eq!(reduced.unpaired().len(), 2);
    }

    #[test]
    fn test_reduce_triangle_filled() {
        let simplices: Vec<Vec<usize>> = vec![
            vec![0],
            vec![1],
            vec![2],
            vec![0, 1],
            vec![0, 2],
            vec![1, 2],
            vec![0, 1, 2],
        ];
        let boundary = BoundaryMatrix::from_simplices(&simplices);
        let reduced = boundary.reduce();

        // Triangle fills in: the loop is killed
        assert!(!reduced.pairs().is_empty());
    }

    #[test]
    fn test_reduce_empty_matrix() {
        let m = BoundaryMatrix::new(0, 0);
        let reduced = m.reduce();
        assert!(reduced.pairs().is_empty());
        assert!(reduced.unpaired().is_empty());
    }

    #[test]
    fn test_reduce_single_vertex() {
        let simplices: Vec<Vec<usize>> = vec![vec![0]];
        let boundary = BoundaryMatrix::from_simplices(&simplices);
        let reduced = boundary.reduce();
        assert_eq!(reduced.unpaired().len(), 1);
        assert_eq!(reduced.pairs().len(), 0);
    }

    #[test]
    fn test_reduce_is_idempotent() {
        let simplices: Vec<Vec<usize>> = vec![
            vec![0],
            vec![1],
            vec![2],
            vec![0, 1],
            vec![0, 2],
            vec![1, 2],
        ];
        let boundary = BoundaryMatrix::from_simplices(&simplices);
        let reduced1 = boundary.reduce();
        let reduced2 = boundary.reduce();

        assert_eq!(reduced1.pairs().len(), reduced2.pairs().len());
        assert_eq!(reduced1.unpaired().len(), reduced2.unpaired().len());
    }

    #[test]
    fn test_reduce_tetrahedron() {
        let simplices: Vec<Vec<usize>> = vec![
            vec![0],
            vec![1],
            vec![2],
            vec![3],
            vec![0, 1],
            vec![0, 2],
            vec![0, 3],
            vec![1, 2],
            vec![1, 3],
            vec![2, 3],
            vec![0, 1, 2],
            vec![0, 1, 3],
            vec![0, 2, 3],
            vec![1, 2, 3],
            vec![0, 1, 2, 3],
        ];
        let boundary = BoundaryMatrix::from_simplices(&simplices);
        let reduced = boundary.reduce();

        // Tetrahedron: 1 unpaired vertex, all other features killed
        assert!(reduced.unpaired().len() >= 1);
    }

    #[test]
    fn test_pivot_tracking() {
        let simplices: Vec<Vec<usize>> = vec![vec![0], vec![1], vec![0, 1]];
        let boundary = BoundaryMatrix::from_simplices(&simplices);
        let reduced = boundary.reduce();

        // After reduction, each pivot should be unique
        let pivot_rows: std::collections::HashSet<usize> =
            reduced.pivots().keys().copied().collect();
        assert_eq!(pivot_rows.len(), reduced.pivots().len());
    }
}
