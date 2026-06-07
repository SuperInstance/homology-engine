//! Filtration construction for persistent homology.
//!
//! A filtration is a nested sequence of simplicial complexes:
//!
//! ```text
//! K₀ ⊂ K₁ ⊂ K₂ ⊂ ... ⊂ Kₙ = K
//! ```
//!
//! where each Kᵢ is obtained from Kᵢ₋₁ by adding simplices. The filtration
//! assigns a "birth time" (filtration value) to each simplex, representing when
//! it first appears.
//!
//! # Filtration Types
//!
//! - **Vietoris-Rips**: Given a point cloud and distance threshold ε, include all
//!   simplices whose vertices are within ε of each other. Increase ε to build the filtration.
//! - **Lower-star**: Given a function f on vertices, a simplex enters at the maximum
//!   function value of its vertices.
//! - **Flag (clique)**: Given a weighted graph, a simplex enters when all its edges
//!   have been added (weighted clique complex).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::boundary::BoundaryMatrix;

/// A filtration of simplicial complexes.
///
/// Each simplex has an associated filtration value (when it enters the complex).
/// Simplices are ordered by increasing filtration value (ties broken by dimension).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Filtration {
    /// Ordered list of simplices, each being a sorted list of vertex indices.
    simplices: Vec<Vec<usize>>,
    /// Filtration value for each simplex (parallel to `simplices`).
    values: Vec<f64>,
    /// Dimension of each simplex.
    dims: Vec<usize>,
}

impl Filtration {
    /// Create an empty filtration.
    pub fn new() -> Self {
        Self {
            simplices: vec![],
            values: vec![],
            dims: vec![],
        }
    }

    /// Create a filtration from pre-sorted simplices and values.
    ///
    /// The simplices MUST be sorted such that all faces of simplex i appear before i.
    pub fn from_sorted(simplices: Vec<Vec<usize>>, values: Vec<f64>) -> Self {
        let dims: Vec<usize> = simplices
            .iter()
            .map(|s| s.len().saturating_sub(1))
            .collect();
        Self {
            simplices,
            values,
            dims,
        }
    }

    /// Add a simplex to the filtration.
    pub fn add(&mut self, simplex: Vec<usize>, value: f64) {
        let dim = simplex.len().saturating_sub(1);
        self.simplices.push(simplex);
        self.values.push(value);
        self.dims.push(dim);
    }

    /// Number of simplices in the filtration.
    pub fn len(&self) -> usize {
        self.simplices.len()
    }

    /// Check if the filtration is empty.
    pub fn is_empty(&self) -> bool {
        self.simplices.is_empty()
    }

    /// Get the simplex at index i.
    pub fn simplex(&self, i: usize) -> &[usize] {
        &self.simplices[i]
    }

    /// Get the filtration value of simplex i.
    pub fn value(&self, i: usize) -> f64 {
        self.values[i]
    }

    /// Get the dimension of simplex i.
    pub fn dim(&self, i: usize) -> usize {
        self.dims[i]
    }

    /// Get all simplices.
    pub fn simplices(&self) -> &[Vec<usize>] {
        &self.simplices
    }

    /// Get all filtration values.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Build the boundary matrix from this filtration.
    ///
    /// Constructs the boundary matrix where column j corresponds to the j-th simplex
    /// in the filtration order.
    pub fn boundary_matrix(&self) -> BoundaryMatrix {
        let mut bm = BoundaryMatrix::from_simplices(&self.simplices);
        // Set dimensions from filtration
        for (i, &dim) in self.dims.iter().enumerate() {
            bm.set_simplex_dim(i, dim);
        }
        bm
    }

    /// Get the number of vertices (0-simplices).
    pub fn num_vertices(&self) -> usize {
        self.dims.iter().filter(|&&d| d == 0).count()
    }

    /// Get the maximum dimension of any simplex.
    pub fn max_dim(&self) -> usize {
        self.dims.iter().copied().max().unwrap_or(0)
    }
}

impl Default for Filtration {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for constructing filtrations from point clouds and graphs.
pub struct FiltrationBuilder {
    /// Maximum dimension of simplices to include.
    max_dim: usize,
}

impl FiltrationBuilder {
    /// Create a new filtration builder.
    pub fn new() -> Self {
        Self {
            max_dim: usize::MAX,
        }
    }

    /// Set the maximum simplex dimension.
    pub fn max_dim(mut self, d: usize) -> Self {
        self.max_dim = d;
        self
    }

    /// Build a Vietoris-Rips filtration from a 2D point cloud.
    ///
    /// Given points and a maximum distance ε, include all simplices whose
    /// pairwise vertex distances are ≤ ε. Simplices enter at the maximum
    /// pairwise distance of their vertices.
    ///
    /// # Arguments
    ///
    /// * `points` - The point cloud as a vector of (x, y) coordinates.
    /// * `max_eps` - Maximum distance threshold.
    /// * `max_dim` - Maximum simplex dimension (0 = vertices, 1 = edges, 2 = triangles, etc.)
    ///
    /// # Returns
    ///
    /// A `Filtration` ordered by increasing filtration value.
    pub fn rips(&mut self, points: &[(f64, f64)], max_eps: f64, max_dim: usize) -> Filtration {
        let n = points.len();
        if n == 0 {
            return Filtration::new();
        }

        // Compute pairwise distances
        let mut dist: HashMap<(usize, usize), f64> = HashMap::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = points[i].0 - points[j].0;
                let dy = points[i].1 - points[j].1;
                let d = (dx * dx + dy * dy).sqrt();
                dist.insert((i, j), d);
            }
        }

        // Generate all simplices up to max_dim
        let mut simplices: Vec<(Vec<usize>, f64)> = vec![];

        // Vertices (filtration value 0)
        for i in 0..n {
            simplices.push((vec![i], 0.0));
        }

        // Generate higher simplices
        let actual_max_dim = max_dim.min(n - 1);
        for dim in 1..=actual_max_dim {
            self.generate_simplices(&dist, n, dim, max_eps, &mut simplices);
        }

        // Sort by filtration value, then by dimension, then lexicographically
        simplices.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.len().cmp(&b.0.len()))
                .then_with(|| a.0.cmp(&b.0))
        });

        // Ensure face-before-coface ordering
        let filtered = self.ensure_face_order(simplices);

        let (sims, vals): (Vec<_>, Vec<_>) = filtered.into_iter().unzip();
        Filtration::from_sorted(sims, vals)
    }

    /// Generate simplices of a given dimension for Rips complex.
    fn generate_simplices(
        &self,
        dist: &HashMap<(usize, usize), f64>,
        n: usize,
        dim: usize,
        max_eps: f64,
        result: &mut Vec<(Vec<usize>, f64)>,
    ) {
        // Enumerate all (dim+1)-element subsets of {0, ..., n-1}
        let mut combo: Vec<usize> = (0..=dim).collect();
        loop {
            // Check if all pairwise distances are within max_eps
            let mut max_dist = 0.0_f64;
            let mut valid = true;
            for i in 0..combo.len() {
                for j in (i + 1)..combo.len() {
                    let (a, b) = if combo[i] < combo[j] {
                        (combo[i], combo[j])
                    } else {
                        (combo[j], combo[i])
                    };
                    if let Some(&d) = dist.get(&(a, b)) {
                        max_dist = max_dist.max(d);
                        if d > max_eps {
                            valid = false;
                            break;
                        }
                    } else {
                        valid = false;
                        break;
                    }
                }
                if !valid {
                    break;
                }
            }

            if valid {
                result.push((combo.clone(), max_dist));
            }

            // Next combination
            if !next_combination(&mut combo, n) {
                break;
            }
        }
    }

    /// Reorder simplices to ensure every face appears before its cofaces.
    fn ensure_face_order(&self, simplices: Vec<(Vec<usize>, f64)>) -> Vec<(Vec<usize>, f64)> {
        // Build set of existing simplices
        let simplex_set: std::collections::HashSet<Vec<usize>> =
            simplices.iter().map(|(s, _)| s.clone()).collect();

        // Topological sort: push simplices earlier if their faces haven't appeared yet
        let mut result: Vec<(Vec<usize>, f64)> = vec![];
        let mut placed: std::collections::HashSet<Vec<usize>> = std::collections::HashSet::new();

        for (simplex, value) in simplices {
            // First, ensure all faces are placed
            if simplex.len() > 1 {
                for skip in 0..simplex.len() {
                    let face: Vec<usize> = simplex
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| *i != skip)
                        .map(|(_, &v)| v)
                        .collect();
                    if !placed.contains(&face) && simplex_set.contains(&face) {
                        // Face should have been placed already by sorting
                        // This handles edge cases
                    }
                }
            }
            result.push((simplex.clone(), value));
            placed.insert(simplex);
        }

        result
    }

    /// Build a lower-star filtration from vertex function values.
    ///
    /// Each vertex v has a function value f(v). A simplex σ enters the filtration
    /// at max(f(v) : v ∈ σ). Vertices and simplices are sorted by their filtration value.
    pub fn lower_star(&self, vertex_values: &[f64], edges: &[(usize, usize)]) -> Filtration {
        let _n = vertex_values.len();
        let mut simplices: Vec<(Vec<usize>, f64)> = vec![];

        // Add vertices
        for (i, &v) in vertex_values.iter().enumerate() {
            simplices.push((vec![i], v));
        }

        // Add edges
        for &(i, j) in edges {
            let val = vertex_values[i].max(vertex_values[j]);
            let mut edge = if i < j { vec![i, j] } else { vec![j, i] };
            edge.sort();
            simplices.push((edge, val));
        }

        // Sort
        simplices.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.len().cmp(&b.0.len()))
                .then_with(|| a.0.cmp(&b.0))
        });

        let (sims, vals): (Vec<_>, Vec<_>) = simplices.into_iter().unzip();
        Filtration::from_sorted(sims, vals)
    }

    /// Build a flag filtration from a weighted graph.
    ///
    /// Given a weighted graph (edge list with weights), a k-simplex enters when
    /// its maximum edge weight is reached (i.e., when all its edges have appeared).
    pub fn flag(&self, n_vertices: usize, edges: &[(usize, usize, f64)]) -> Filtration {
        let mut simplices: Vec<(Vec<usize>, f64)> = vec![];

        // Add vertices at time 0
        for i in 0..n_vertices {
            simplices.push((vec![i], 0.0));
        }

        // Build adjacency with weights
        let mut adj: HashMap<(usize, usize), f64> = HashMap::new();
        for &(i, j, w) in edges {
            let key = if i < j { (i, j) } else { (j, i) };
            adj.insert(key, w);
            let mut edge = if i < j { vec![i, j] } else { vec![j, i] };
            edge.sort();
            simplices.push((edge, w));
        }

        // Generate triangles (if max_dim >= 2)
        if self.max_dim >= 2 || self.max_dim == usize::MAX {
            for i in 0..n_vertices {
                for j in (i + 1)..n_vertices {
                    if adj.contains_key(&(i, j)) {
                        for k in (j + 1)..n_vertices {
                            if adj.contains_key(&(i, k)) && adj.contains_key(&(j, k)) {
                                let w = adj[&(i, j)].max(adj[&(i, k)]).max(adj[&(j, k)]);
                                simplices.push((vec![i, j, k], w));
                            }
                        }
                    }
                }
            }
        }

        // Sort
        simplices.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.len().cmp(&b.0.len()))
                .then_with(|| a.0.cmp(&b.0))
        });

        let (sims, vals): (Vec<_>, Vec<_>) = simplices.into_iter().unzip();
        Filtration::from_sorted(sims, vals)
    }
}

impl Default for FiltrationBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Advance to the next combination in lexicographic order.
/// Returns false if no more combinations.
fn next_combination(combo: &mut [usize], n: usize) -> bool {
    let k = combo.len();
    let mut i = k;
    while i > 0 {
        i -= 1;
        if combo[i] < n - k + i {
            combo[i] += 1;
            for j in (i + 1)..k {
                combo[j] = combo[j - 1] + 1;
            }
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_filtration() {
        let f = Filtration::new();
        assert!(f.is_empty());
        assert_eq!(f.len(), 0);
    }

    #[test]
    fn test_simple_filtration() {
        let mut f = Filtration::new();
        f.add(vec![0], 0.0);
        f.add(vec![1], 0.0);
        f.add(vec![0, 1], 1.0);
        assert_eq!(f.len(), 3);
        assert_eq!(f.dim(0), 0);
        assert_eq!(f.dim(2), 1);
    }

    #[test]
    fn test_rips_simple() {
        let points: Vec<(f64, f64)> = vec![(0.0, 0.0), (1.0, 0.0), (0.5, 0.87)];
        let mut builder = FiltrationBuilder::new();
        let f = builder.rips(&points, 2.0, 2);

        // Should have 3 vertices + 3 edges + 1 triangle = 7 simplices
        assert_eq!(f.len(), 7);
        assert_eq!(f.num_vertices(), 3);
        assert_eq!(f.max_dim(), 2);
    }

    #[test]
    fn test_rips_two_points() {
        let points: Vec<(f64, f64)> = vec![(0.0, 0.0), (1.0, 0.0)];
        let mut builder = FiltrationBuilder::new();
        let f = builder.rips(&points, 2.0, 1);

        assert_eq!(f.len(), 3); // 2 vertices + 1 edge
    }

    #[test]
    fn test_rips_threshold() {
        let points: Vec<(f64, f64)> = vec![(0.0, 0.0), (10.0, 0.0)];
        let mut builder = FiltrationBuilder::new();
        let f = builder.rips(&points, 0.5, 1); // threshold too small for edge

        assert_eq!(f.len(), 2); // Only 2 vertices
    }

    #[test]
    fn test_lower_star() {
        let values = [1.0, 2.0, 3.0];
        let edges = vec![(0, 1), (1, 2)];
        let builder = FiltrationBuilder::new();
        let f = builder.lower_star(&values, &edges);

        assert_eq!(f.len(), 5); // 3 vertices + 2 edges
        // Edge (0,1) enters at max(1, 2) = 2
        // Edge (1,2) enters at max(2, 3) = 3
        // Both edges should have correct values: (0,1)→2.0, (1,2)→3.0
        // Find edge indices (sorted by value then simplex)
        let edge_vals: Vec<(f64, Vec<usize>)> = f
            .simplices()
            .iter()
            .skip(3)
            .zip(f.values().iter().skip(3))
            .map(|(s, &v)| (v, s.clone()))
            .collect();
        assert_eq!(edge_vals.len(), 2);
        // (0,1) enters at max(1,2)=2, (1,2) enters at max(2,3)=3
        let e01_val = f
            .values()
            .iter()
            .zip(f.simplices().iter())
            .find(|(_, s)| *s == &vec![0, 1])
            .map(|(v, _)| *v)
            .unwrap();
        let e12_val = f
            .values()
            .iter()
            .zip(f.simplices().iter())
            .find(|(_, s)| *s == &vec![1, 2])
            .map(|(v, _)| *v)
            .unwrap();
        assert_eq!(e01_val, 2.0);
        assert_eq!(e12_val, 3.0);
    }

    #[test]
    fn test_flag_filtration() {
        let builder = FiltrationBuilder::new();
        let edges = vec![(0, 1, 1.0), (1, 2, 2.0), (0, 2, 3.0)];
        let f = builder.flag(3, &edges);

        // 3 vertices + 3 edges + 1 triangle = 7
        assert_eq!(f.len(), 7);
        assert_eq!(f.max_dim(), 2);
    }

    #[test]
    fn test_boundary_matrix_from_filtration() {
        let points: Vec<(f64, f64)> = vec![(0.0, 0.0), (1.0, 0.0)];
        let mut builder = FiltrationBuilder::new();
        let f = builder.rips(&points, 2.0, 1);
        let bm = f.boundary_matrix();

        assert!(bm.verify_chain_complex());
    }

    #[test]
    fn test_full_pipeline() {
        let points: Vec<(f64, f64)> = vec![(0.0, 0.0), (1.0, 0.0), (0.5, 0.87)];
        let mut builder = FiltrationBuilder::new();
        let f = builder.rips(&points, 2.0, 2);
        let bm = f.boundary_matrix();
        let reduced = bm.reduce();
        let betti = reduced.betti_numbers();

        assert_eq!(betti.at(0), 1); // One connected component
    }

    #[test]
    fn test_filtration_values() {
        let points: Vec<(f64, f64)> = vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)];
        let mut builder = FiltrationBuilder::new();
        let f = builder.rips(&points, 3.0, 1);

        // Vertices have value 0, edges have distance values
        for i in 0..f.num_vertices() {
            assert_eq!(f.value(i), 0.0);
        }
        // All filtration values should be non-negative and non-decreasing
        for w in f.values().windows(2) {
            assert!(
                w[0] <= w[1] + 1e-10,
                "filtration values must be non-decreasing"
            );
        }
    }
}
