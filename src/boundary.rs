//! Boundary matrices over Z/2Z.
//!
//! A boundary matrix represents the boundary operator ∂ in the chain complex
//! C_k → C_{k-1}. Over Z/2Z (integers mod 2), addition is XOR, which means
//! the boundary of a simplex is simply the symmetric difference of its faces.
//!
//! # Mathematical Background
//!
//! Given an n-simplex σ = [v₀, v₁, ..., vₙ], its boundary is:
//!
//! ```text
//! ∂σ = Σᵢ (-1)ⁱ [v₀, ..., v̂ᵢ, ..., vₙ]
//! ```
//!
//! Over Z/2Z, all signs become +1, so:
//!
//! ```text
//! ∂σ = Σᵢ [v₀, ..., v̂ᵢ, ..., vₙ]
//! ```
//!
//! This is the symmetric difference (XOR) of the (n-1)-faces.
//!
//! # Representation
//!
//! The boundary matrix is stored as a sparse column-oriented structure:
//! `Vec<Vec<usize>>` where each inner `Vec<usize>` contains the row indices
//! of non-zero entries (i.e., the 1s) in that column.

use serde::{Deserialize, Serialize};

/// A sparse boundary matrix over Z/2Z.
///
/// Each column is represented as a sorted `Vec<usize>` of row indices
/// where the entry is 1. All arithmetic is mod 2 (XOR).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BoundaryMatrix {
    /// Columns of the matrix. `columns[j]` contains sorted row indices of 1s.
    columns: Vec<Vec<usize>>,
    /// Number of rows.
    num_rows: usize,
    /// Dimension of each simplex (row index → simplex dimension).
    simplex_dims: Vec<usize>,
}

impl BoundaryMatrix {
    /// Create a new zero boundary matrix with `num_cols` columns and `num_rows` rows.
    pub fn new(num_rows: usize, num_cols: usize) -> Self {
        Self {
            columns: vec![vec![]; num_cols],
            num_rows,
            simplex_dims: vec![0; num_rows],
        }
    }

    /// Create a boundary matrix from a list of simplices.
    ///
    /// Each simplex is represented as a sorted `Vec<usize>` of vertex indices.
    /// The boundary of each simplex is computed as the symmetric difference
    /// of its (n-1)-faces (over Z/2Z).
    ///
    /// # Arguments
    ///
    /// * `simplices` - A list of simplices, where each simplex is a sorted list of vertex indices.
    ///   The simplices MUST be ordered so that all faces of simplex i appear before simplex i.
    ///
    /// # Returns
    ///
    /// A `BoundaryMatrix` where column j is the boundary of simplex j.
    pub fn from_simplices(simplices: &[Vec<usize>]) -> Self {
        let n = simplices.len();
        // Build vertex → simplex index mapping
        let mut simplex_index: std::collections::HashMap<Vec<usize>, usize> =
            std::collections::HashMap::new();
        for (i, s) in simplices.iter().enumerate() {
            simplex_index.insert(s.clone(), i);
        }

        let mut columns = vec![vec![]; n];
        let simplex_dims: Vec<usize> = simplices
            .iter()
            .map(|s| s.len().saturating_sub(1))
            .collect();

        for (col_idx, simplex) in simplices.iter().enumerate() {
            if simplex.len() <= 1 {
                // Vertices have empty boundary
                columns[col_idx] = vec![];
            } else {
                // Compute boundary faces: remove each vertex in turn
                let mut faces = vec![];
                for skip in 0..simplex.len() {
                    let face: Vec<usize> = simplex
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| *i != skip)
                        .map(|(_, &v)| v)
                        .collect();
                    if let Some(&face_idx) = simplex_index.get(&face) {
                        faces.push(face_idx);
                    }
                }
                faces.sort();
                faces.dedup();
                columns[col_idx] = faces;
            }
        }

        Self {
            columns,
            num_rows: n,
            simplex_dims,
        }
    }

    /// Get the number of rows.
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Get the number of columns.
    pub fn num_cols(&self) -> usize {
        self.columns.len()
    }

    /// Get a reference to column `j` (sorted list of row indices with 1s).
    pub fn column(&self, j: usize) -> &[usize] {
        &self.columns[j]
    }

    /// Set column `j` to the given sorted list of row indices.
    pub fn set_column(&mut self, j: usize, rows: Vec<usize>) {
        debug_assert!(
            rows.windows(2).all(|w| w[0] < w[1]),
            "column must be sorted"
        );
        self.columns[j] = rows;
    }

    /// Add (XOR) column `src` into column `dst` over Z/2Z.
    ///
    /// This is the fundamental operation for column reduction.
    /// The result is the symmetric difference of the two columns.
    pub fn add_columns(&mut self, dst: usize, src: usize) {
        let mut result = vec![];
        let (mut i, mut j) = (0, 0);
        let a = self.columns[src].clone();
        let b = &self.columns[dst];

        while i < a.len() && j < b.len() {
            if a[i] < b[j] {
                result.push(a[i]);
                i += 1;
            } else if a[i] > b[j] {
                result.push(b[j]);
                j += 1;
            } else {
                // XOR: cancel out
                i += 1;
                j += 1;
            }
        }
        while i < a.len() {
            result.push(a[i]);
            i += 1;
        }
        while j < b.len() {
            result.push(b[j]);
            j += 1;
        }
        self.columns[dst] = result;
    }

    /// Get the pivot (largest row index) of column `j`, if any.
    pub fn pivot(&self, j: usize) -> Option<usize> {
        self.columns[j].last().copied()
    }

    /// Check if column `j` is zero (empty).
    pub fn is_zero_column(&self, j: usize) -> bool {
        self.columns[j].is_empty()
    }

    /// Get the dimension of simplex at index `i`.
    pub fn simplex_dim(&self, i: usize) -> usize {
        self.simplex_dims[i]
    }

    /// Set the dimension of simplex at index `i`.
    pub fn set_simplex_dim(&mut self, i: usize, dim: usize) {
        self.simplex_dims[i] = dim;
    }

    /// Verify the chain complex property: ∂² = 0.
    ///
    /// For every column j (representing simplex σⱼ), computes ∂(∂σⱼ)
    /// and checks that it equals zero. Over Z/2Z, this means the XOR
    /// of all (n-2)-faces (counted with multiplicity) must cancel out.
    pub fn verify_chain_complex(&self) -> bool {
        for j in 0..self.columns.len() {
            // Compute ∂(∂σⱼ) = ∂(column j)
            let mut boundary_of_boundary = vec![];
            for &face_row in &self.columns[j] {
                for &face_face_row in &self.columns[face_row] {
                    // XOR: add or remove
                    if let Some(pos) = boundary_of_boundary
                        .iter()
                        .position(|&x| x == face_face_row)
                    {
                        boundary_of_boundary.remove(pos);
                    } else {
                        boundary_of_boundary.push(face_face_row);
                    }
                }
            }
            if !boundary_of_boundary.is_empty() {
                return false;
            }
        }
        true
    }

    /// Get all column data as a slice.
    pub fn columns(&self) -> &[Vec<usize>] {
        &self.columns
    }

    /// Create a deep copy of this boundary matrix.
    pub fn clone_matrix(&self) -> Self {
        Self {
            columns: self.columns.clone(),
            num_rows: self.num_rows,
            simplex_dims: self.simplex_dims.clone(),
        }
    }
}

/// Compute the boundary of a single simplex over Z/2Z.
///
/// Given simplex σ = [v₀, v₁, ..., vₙ], returns its (n-1)-faces
/// as a list of simplices (each with one vertex removed).
///
/// # Example
///
/// ```
/// use homology_engine::boundary::simplex_boundary;
///
/// let triangle = vec![0, 1, 2];
/// let faces = simplex_boundary(&triangle);
/// assert_eq!(faces.len(), 3);
/// for face in &faces {
///     assert_eq!(face.len(), 2);
/// }
/// ```
pub fn simplex_boundary(simplex: &[usize]) -> Vec<Vec<usize>> {
    if simplex.len() <= 1 {
        return vec![];
    }
    let mut faces = vec![];
    for skip in 0..simplex.len() {
        let face: Vec<usize> = simplex
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != skip)
            .map(|(_, &v)| v)
            .collect();
        faces.push(face);
    }
    faces
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_matrix() {
        let m = BoundaryMatrix::new(3, 2);
        assert_eq!(m.num_rows(), 3);
        assert_eq!(m.num_cols(), 2);
        assert!(m.is_zero_column(0));
        assert!(m.is_zero_column(1));
        assert_eq!(m.pivot(0), None);
    }

    #[test]
    fn test_single_edge_boundary() {
        // Two vertices [0], [1] and one edge [0,1]
        let simplices: Vec<Vec<usize>> = vec![
            vec![0],    // vertex 0 (index 0)
            vec![1],    // vertex 1 (index 1)
            vec![0, 1], // edge (index 2)
        ];
        let m = BoundaryMatrix::from_simplices(&simplices);
        assert!(m.is_zero_column(0)); // ∂(v₀) = 0
        assert!(m.is_zero_column(1)); // ∂(v₁) = 0
        assert_eq!(m.column(2), &[0, 1]); // ∂(e₀₁) = v₀ + v₁
        assert!(m.verify_chain_complex());
    }

    #[test]
    fn test_triangle_boundary() {
        let simplices: Vec<Vec<usize>> = vec![
            vec![0],       // v0
            vec![1],       // v1
            vec![2],       // v2
            vec![0, 1],    // e01
            vec![0, 2],    // e02
            vec![1, 2],    // e12
            vec![0, 1, 2], // triangle
        ];
        let m = BoundaryMatrix::from_simplices(&simplices);
        // Boundary of triangle = e01 + e02 + e12
        assert_eq!(m.column(6), &[3, 4, 5]);
        assert!(m.verify_chain_complex());
    }

    #[test]
    fn test_simplex_boundary_function() {
        let edge = vec![0, 1];
        let faces = simplex_boundary(&edge);
        assert_eq!(faces, vec![vec![1], vec![0]]);

        let triangle = vec![0, 1, 2];
        let faces = simplex_boundary(&triangle);
        assert_eq!(faces.len(), 3);
    }

    #[test]
    fn test_add_columns_xor() {
        let mut m = BoundaryMatrix::new(4, 2);
        m.set_column(0, vec![0, 2, 3]);
        m.set_column(1, vec![1, 2, 4]);
        m.add_columns(0, 1); // XOR: {0,2,3} ⊕ {1,2,4} = {0,1,3,4}
        assert_eq!(m.column(0), &[0, 1, 3, 4]);
    }

    #[test]
    fn test_add_columns_cancel() {
        let mut m = BoundaryMatrix::new(3, 2);
        m.set_column(0, vec![1, 2]);
        m.set_column(1, vec![1, 2]);
        m.add_columns(0, 1); // XOR: {1,2} ⊕ {1,2} = {}
        assert!(m.is_zero_column(0));
    }

    #[test]
    fn test_chain_complex_tetrahedron() {
        let simplices: Vec<Vec<usize>> = vec![
            vec![0],          // 0: v0
            vec![1],          // 1: v1
            vec![2],          // 2: v2
            vec![3],          // 3: v3
            vec![0, 1],       // 4: e01
            vec![0, 2],       // 5: e02
            vec![0, 3],       // 6: e03
            vec![1, 2],       // 7: e12
            vec![1, 3],       // 8: e13
            vec![2, 3],       // 9: e23
            vec![0, 1, 2],    // 10: f012
            vec![0, 1, 3],    // 11: f013
            vec![0, 2, 3],    // 12: f023
            vec![1, 2, 3],    // 13: f123
            vec![0, 1, 2, 3], // 14: tetrahedron
        ];
        let m = BoundaryMatrix::from_simplices(&simplices);
        assert!(m.verify_chain_complex());

        // Boundary of tetrahedron = f012 + f013 + f023 + f123
        assert_eq!(m.column(14), &[10, 11, 12, 13]);
    }

    #[test]
    fn test_pivot() {
        let mut m = BoundaryMatrix::new(5, 1);
        assert_eq!(m.pivot(0), None);
        m.set_column(0, vec![1, 3, 4]);
        assert_eq!(m.pivot(0), Some(4));
    }

    #[test]
    fn test_set_and_get_column() {
        let mut m = BoundaryMatrix::new(5, 3);
        m.set_column(1, vec![0, 2, 4]);
        assert_eq!(m.column(1), &[0, 2, 4]);
    }
}
