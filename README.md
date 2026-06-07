# homology-engine

**A persistent homology computation engine for topological data analysis, written in pure Rust.**

[![crates.io](https://img.shields.io/crates/v/homology-engine.svg)](https://crates.io/crates/homology-engine)
[![docs.rs](https://docs.rs/homology-engine/badge.svg)](https://docs.rs/homology-engine)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## Table of Contents

- [What Is This?](#what-is-this)
- [Architecture](#architecture)
- [The Mathematics](#the-mathematics)
  - [Simplicial Complexes](#simplicial-complexes)
  - [Chain Complexes and Boundary Operators](#chain-complexes-and-boundary-operators)
  - [Homology Groups](#homology-groups)
  - [Persistent Homology](#persistent-homology)
  - [Filtrations](#filtrations-1)
  - [Stability Theorems](#stability-theorems)
- [Quick Start](#quick-start)
- [Complete Examples](#complete-examples)
  - [Example 1: Boundary Matrices](#example-1-computing-boundary-matrices)
  - [Example 2: Reduction and Barcodes](#example-2-reducing-boundary-matrices-and-extracting-barcodes)
  - [Example 3: Rips Filtration](#example-3-building-a-rips-filtration-and-computing-persistent-homology)
  - [Example 4: Comparing Diagrams](#example-4-comparing-two-persistence-diagrams-with-bottleneck-distance)
- [Module Reference](#module-reference)
- [Design Decisions](#design-decisions)
- [Performance Analysis](#performance-analysis)
- [Comparison with Other Libraries](#comparison-with-other-libraries)
- [Practical Applications](#practical-applications)
- [API Stability](#api-stability)
- [References](#references)
- [License](#license)

---

## What Is This?

`homology-engine` is a Rust library for computing **persistent homology**, the core algorithm in **topological data analysis (TDA)**. Given data — point clouds, weighted graphs, scalar fields — it identifies and quantifies topological features: connected components, loops, voids, and higher-dimensional holes.

This crate is designed with **educational clarity** as a primary goal. Every module corresponds to a well-defined mathematical concept, and the code is structured to mirror the algebraic pipeline:

```
Data → Filtration → Boundary Matrix → Column Reduction → Barcode → Insight
```

If you're learning persistent homology, reading this code should help you understand the algorithm. If you're building a TDA application, this crate gives you a correct, dependency-light foundation.

### Key Features

- **Pure Rust** — no C bindings, no BLAS, no external dependencies beyond `serde`
- **Six focused modules** — each maps to a mathematical concept in the persistent homology pipeline
- **Z/2Z coefficients** — the standard setting for TDA: simple, efficient, and sufficient for most applications
- **Sparse representation** — boundary matrices stored as `Vec<Vec<usize>>`, column-oriented
- **Iterative reduction** — explicit worklist-based column reduction, not recursive Gaussian elimination
- **Full pipeline** — from point cloud to Betti numbers to persistence diagram distances
- **55+ tests** — comprehensive coverage of all modules including chain complex verification

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        homology-engine                              │
│                                                                     │
│  Point Cloud ──→ Filtration ──→ Boundary Matrix ──→ Reduction      │
│  (f64×f64)       (filtration)   (sparse Z/2Z)      (column ops)    │
│                                                                     │
│       ┌──────────────────────────────────────────────────┐          │
│       │              Reduced Matrix                      │          │
│       │   (pivots, pairs, unpaired columns)              │          │
│       └──────┬───────────────┬───────────────┬───────────┘          │
│              │               │               │                      │
│         Barcode        Betti Numbers   Persistence Diagram          │
│       [(b,d,dim)]    {dim → βₖ}      {(b,d,dim)}                   │
│              │               │               │                      │
│       lifetime stats   curves & curves   bottleneck &               │
│       persistence      persistent βₖ    Wasserstein distance        │
│       entropy                         stability checks              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Module Flow:
  filtration → boundary → reduction → { barcode, betti, diagram }
```

### ASCII Pipeline

```
  Point Cloud          Filtration           Boundary Matrix
 ┌───┐ ┌───┐        K₀ ⊂ K₁ ⊂ ... ⊂ Kₙ    Col j = ∂(σⱼ) over Z/2Z
 │ • │ │•• │   →    Each σ enters at εᵢ   →  Sparse: Vec<Vec<usize>>
 │•••│ │ • │         Sorted by birth time    Pivot = last nonzero row
 └───┘ └───┘
                                                    │
                                                    ▼
  Insight          Barcode             Column Reduction
 ┌──────────┐    [───) [──)         Iterative left-to-right
 │ β₀ = 1   │ ←  [──────────────)   XOR columns with same pivot
 │ β₁ = 2   │    Finite + infinite  Track pivots in HashMap
 │ Loop at ε│    bars               O(m³) worst, O(m²) typical
 └──────────┘
       │
       ▼
  Persistence Diagram
  Scatter plot of (birth, death)
  Bottleneck & Wasserstein distance
  d_B(Dgm(f), Dgm(g)) ≤ ‖f − g‖∞
```

---

## The Mathematics

This section provides the theoretical foundations. If you already know persistent homology, skip to [Quick Start](#quick-start). If you're here to learn, read on.

### Simplicial Complexes

A **simplicial complex** K is a collection of simplices (points, edges, triangles, tetrahedra, ...) that is closed under taking faces. Formally:

- A **k-simplex** is the convex hull of k+1 affinely independent points: σ = [v₀, v₁, ..., vₖ]
- A **face** of σ is obtained by removing one or more vertices
- K is a simplicial complex if every face of a simplex in K is also in K

**Examples:**
- 0-simplex = vertex: [v₀]
- 1-simplex = edge: [v₀, v₁]
- 2-simplex = triangle: [v₀, v₁, v₂]
- 3-simplex = tetrahedron: [v₀, v₁, v₂, v₃]

A simplicial complex is a combinatorial object — it doesn't need geometric coordinates. This makes it perfect for analyzing abstract data.

### Chain Complexes and Boundary Operators

A **chain complex** is a sequence of vector spaces (or modules) connected by boundary operators:

```
... → C_{k+1} →∂_{k+1} C_k →∂_k C_{k-1} → ... → C_1 →∂_1 C_0 → 0
```

where Cₖ is the vector space of k-chains (formal sums of k-simplices) over Z/2Z.

The **boundary operator** ∂ₖ maps a k-simplex to its (k-1)-dimensional boundary:

```
∂ₖ([v₀, v₁, ..., vₖ]) = Σᵢ₌₀ᵏ (-1)ⁱ [v₀, ..., v̂ᵢ, ..., vₖ]
```

Over **Z/2Z** (integers mod 2), all signs become +1:

```
∂ₖ([v₀, v₁, ..., vₖ]) = Σᵢ₌₀ᵏ [v₀, ..., v̂ᵢ, ..., vₖ]   (mod 2)
```

This is the **symmetric difference** (XOR) of the faces — the key simplification that makes Z/2Z arithmetic so clean.

**The fundamental property:** ∂² = 0. The boundary of a boundary is always zero:

```
∂_{k} ∘ ∂_{k+1} = 0   for all k
```

This is because each (k-1)-face of a (k+1)-simplex appears exactly twice in ∂(∂σ), and over Z/2Z, 2 = 0.

**Boundary matrix representation:** The boundary operator ∂ₖ is encoded as a matrix Dₖ over Z/2Z, where Dₖ[i][j] = 1 if the i-th (k-1)-simplex is in the boundary of the j-th k-simplex.

In this crate, we store the **full boundary matrix** as a single sparse matrix where column j is the boundary of simplex σⱼ, using `Vec<Vec<usize>>` (column → list of row indices with 1s).

### Homology Groups

The **k-th homology group** measures k-dimensional "holes" in the complex:

```
Hₖ(K) = ker(∂ₖ) / im(∂ₖ₊₁)
```

- **ker(∂ₖ)** = k-cycles: k-chains with zero boundary (things that "loop back")
- **im(∂ₖ₊₁)** = k-boundaries: k-chains that are boundaries of (k+1)-chains

Intuitively, Hₖ captures cycles that are NOT boundaries of anything — genuine holes.

**Betti numbers:** βₖ = rank(Hₖ) counts independent k-dimensional holes:

| Dimension | Feature | Example |
|-----------|---------|---------|
| β₀ | Connected components | 3 separate points → β₀ = 3 |
| β₁ | Loops | Circle → β₁ = 1 |
| β₂ | Voids (cavities) | Hollow sphere → β₂ = 1 |
| β₃ | 3D tunnels | Solid torus → β₃ = 0, β₁ = 1 |

**Example:** A hollow triangle (3 vertices, 3 edges, no filling triangle):
- β₀ = 1 (one connected component)
- β₁ = 1 (one loop — the triangle perimeter)
- β₂ = 0 (no voids)

### Persistent Homology

Classical homology gives a snapshot — it tells you the topology of a single complex. **Persistent homology** tracks how topology *evolves* across a filtration:

```
K₀ ⊂ K₁ ⊂ K₂ ⊂ ... ⊂ Kₙ = K
```

As we increase the filtration parameter ε, new simplices are added. Features (components, loops, voids) are **born** and **die**:

- A **birth** occurs when a new cycle is created (a new connected component forms, a new loop appears)
- A **death** occurs when a cycle becomes a boundary (two components merge, a loop is filled in)

The lifetime [birth, death) of a feature measures its **persistence** — long-lived features are statistically significant; short-lived ones are likely noise.

**The barcode** is the complete descriptor:

```
  ε: 0   1   2   3   4   5   6   7   8
     │   │   │   │   │   │   │   │   │
H₀:  ████████████████░░░░░               Component born at 0, dies at 4
     ████████████████████████████        Component born at 0, never dies
     │   │   │   │   │   │   │   │   │
H₁:          ██████████████              Loop born at 2, dies at 6
     │   │   │   │   │   │   │   │   │
```

**The persistence diagram** is the same information plotted as points (birth, death):

```
  death
   8 │
   7 │
   6 │         • (2,6) H₁
   5 │
   4 │  • (0,4) H₀
   3 │
   2 │
   1 │
   0 │──•──•────────────────── birth
     0  1  2  3  4  5  6  7  8
        ↑
     (0,∞) H₀ off-chart (infinite bar)
```

Points far from the diagonal (birth = death) represent persistent features. Points near the diagonal are noise.

### Filtrations

A **filtration** is the input to persistent homology. It defines when each simplex enters the complex:

**Vietoris-Rips filtration:** Given a point cloud with pairwise distances d(vᵢ, vⱼ), a simplex σ enters at:

```
f(σ) = max{d(vᵢ, vⱼ) : vᵢ, vⱼ ∈ σ}
```

As ε grows from 0, we add all simplices whose maximum pairwise distance is ≤ ε. This is the most common filtration for point cloud data.

**Lower-star filtration:** Given a scalar function g on vertices, a simplex σ enters at:

```
f(σ) = max{g(vᵢ) : vᵢ ∈ σ}
```

This is useful for analyzing scalar fields (e.g., elevation data, grayscale images).

**Flag (clique) filtration:** Given a weighted graph, a k-simplex enters when all its edges have been added (i.e., at the maximum edge weight in the simplex). This is efficient for graph data.

### Stability Theorems

A crucial theoretical result: **persistent homology is stable** under perturbations of the input.

**Bottleneck stability (Edelsbrunner & Harer 2010):**

```
d_B(Dgm(f), Dgm(g)) ≤ ‖f − g‖∞
```

If you perturb the input function by at most ε, the persistence diagram moves by at most ε in bottleneck distance.

**Wasserstein stability (Cohen-Steiner et al. 2007):**

```
W_p(Dgm(f), Dgm(g)) ≤ C · ‖f − g‖∞
```

This extends the stability result to the richer Wasserstein distances.

**Why this matters:** If your data has measurement noise (and it always does), the persistent features are robust — small noise causes only small changes in the diagram. This is what makes TDA statistically meaningful.

---

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
homology-engine = "0.1"
```

```rust
use homology_engine::filtration::{FiltrationBuilder};
use homology_engine::reduction::MatrixReduction;

fn main() {
    // A triangle of points in 2D
    let points: Vec<(f64, f64)> = vec![
        (0.0, 0.0),
        (1.0, 0.0),
        (0.5, 0.87),
    ];

    // Build Vietoris-Rips filtration up to dimension 2
    let mut builder = FiltrationBuilder::new();
    let filtration = builder.rips(&points, 2.0, 2);

    // Get boundary matrix and reduce
    let boundary = filtration.boundary_matrix();
    let reduced = boundary.reduce();

    // Extract results
    let barcode = reduced.barcode();
    let betti = reduced.betti_numbers();

    println!("Betti-0 (components): {}", betti.at(0));
    println!("Betti-1 (loops):      {}", betti.at(1));
    println!("Total bars:           {}", barcode.len());
}
```

---

## Complete Examples

### Example 1: Computing Boundary Matrices

This example shows how to construct boundary matrices directly from simplices and verify the chain complex property ∂² = 0.

```rust
use homology_engine::boundary::{BoundaryMatrix, simplex_boundary};

fn main() {
    // Define a simplicial complex: tetrahedron with 4 vertices
    let simplices: Vec<Vec<usize>> = vec![
        vec![0],          // 0: vertex v0
        vec![1],          // 1: vertex v1
        vec![2],          // 2: vertex v2
        vec![3],          // 3: vertex v3
        vec![0, 1],       // 4: edge e01
        vec![0, 2],       // 5: edge e02
        vec![0, 3],       // 6: edge e03
        vec![1, 2],       // 7: edge e12
        vec![1, 3],       // 8: edge e13
        vec![2, 3],       // 9: edge e23
        vec![0, 1, 2],    // 10: face f012
        vec![0, 1, 3],    // 11: face f013
        vec![0, 2, 3],    // 12: face f023
        vec![1, 2, 3],    // 13: face f123
        vec![0, 1, 2, 3], // 14: tetrahedron t0123
    ];

    // Build boundary matrix
    let bm = BoundaryMatrix::from_simplices(&simplices);

    // Inspect boundaries
    println!("Boundary of edge e01 (col 4): {:?}", bm.column(4));
    println!("Boundary of face f012 (col 10): {:?}", bm.column(10));
    println!("Boundary of tetrahedron (col 14): {:?}", bm.column(14));

    // Verify chain complex property: ∂² = 0
    assert!(bm.verify_chain_complex(), "∂² must be zero!");
    println!("✓ Chain complex verified: ∂² = 0");

    // Compute boundary of a single simplex
    let faces = simplex_boundary(&vec![0, 1, 2]);
    println!("Boundary of [0,1,2]: {:?}", faces);
    // Output: [[1, 2], [0, 2], [0, 1]]

    // Sparse XOR operations
    let mut m = BoundaryMatrix::new(5, 3);
    m.set_column(0, vec![0, 2, 4]);
    m.set_column(1, vec![1, 2, 3]);
    m.add_columns(0, 1); // XOR column 1 into column 0
    println!("After XOR: {:?}", m.column(0));
    // Output: [0, 1, 3, 4] (symmetric difference)
}
```

### Example 2: Reducing Boundary Matrices and Extracting Barcodes

This example demonstrates column reduction and barcode extraction for a triangle complex.

```rust
use homology_engine::boundary::BoundaryMatrix;
use homology_engine::reduction::MatrixReduction;
use homology_engine::barcode::Barcode;

fn main() {
    // Hollow triangle: 3 vertices + 3 edges (no filling triangle)
    let simplices: Vec<Vec<usize>> = vec![
        vec![0],     // vertex 0
        vec![1],     // vertex 1
        vec![2],     // vertex 2
        vec![0, 1],  // edge 0-1
        vec![0, 2],  // edge 0-2
        vec![1, 2],  // edge 1-2
    ];

    let boundary = BoundaryMatrix::from_simplices(&simplices);

    // Reduce the boundary matrix
    let reduced = boundary.reduce();

    // Inspect the reduction
    println!("Persistence pairs (birth, death):");
    for &(birth, death) in reduced.pairs() {
        println!("  simplex {} → simplex {}", birth, death);
    }

    println!("Unpaired simplices (infinite bars):");
    for &idx in reduced.unpaired() {
        println!("  simplex {} (dim {})", idx, reduced.reduced_matrix().simplex_dim(idx));
    }

    // Extract barcode
    let barcode = reduced.barcode();
    println!("\nBarcode:");
    let (h0_finite, h0_infinite) = barcode.bars_in_dimension(0);
    let (h1_finite, h1_infinite) = barcode.bars_in_dimension(1);

    println!("  H₀ finite bars: {:?}", h0_finite);
    println!("  H₀ infinite bars: {:?}", h0_infinite);
    println!("  H₁ finite bars: {:?}", h1_finite);
    println!("  H₁ infinite bars: {:?}", h1_infinite);

    // Barcode statistics
    println!("\nStatistics:");
    println!("  Average lifetime: {:.2}", barcode.average_lifetime());
    println!("  Total persistence (p=1): {:.2}", barcode.total_persistence(1.0));
    println!("  Entropy: {:.4}", barcode.entropy());

    // Betti numbers
    let betti = reduced.betti_numbers();
    println!("\nBetti numbers:");
    println!("  β₀ = {} (connected components)", betti.at(0));
    println!("  β₁ = {} (loops)", betti.at(1));
    println!("  β₂ = {} (voids)", betti.at(2));
}
```

### Example 3: Building a Rips Filtration and Computing Persistent Homology

This example builds a Vietoris-Rips filtration from a point cloud and computes persistent homology through the full pipeline.

```rust
use homology_engine::filtration::{Filtration, FiltrationBuilder};
use homology_engine::reduction::MatrixReduction;
use homology_engine::diagram::PersistenceDiagram;
use homology_engine::betti::BettiNumbers;

fn main() {
    // Point cloud: two clusters (triangle + nearby point)
    let points: Vec<(f64, f64)> = vec![
        (0.0, 0.0),   // Cluster 1: triangle
        (1.0, 0.0),
        (0.5, 0.87),
        (5.0, 0.0),   // Cluster 2: isolated point
    ];

    // Build Vietoris-Rips filtration with max epsilon = 6.0, up to dim 2
    let mut builder = FiltrationBuilder::new();
    let filtration = builder.rips(&points, 6.0, 2);

    println!("Filtration has {} simplices", filtration.len());
    println!("  Vertices: {}", filtration.num_vertices());
    println!("  Max dimension: {}", filtration.max_dim());

    // Print filtration order
    for i in 0..filtration.len() {
        let s = filtration.simplex(i);
        let v = filtration.value(i);
        println!("  ε={:.3}: {:?} (dim {})", v, s, filtration.dim(i));
    }

    // Compute persistent homology
    let boundary = filtration.boundary_matrix();
    let reduced = boundary.reduce();
    let betti = reduced.betti_numbers();

    println!("\nBetti numbers:");
    println!("  β₀ = {} (connected components)", betti.at(0));
    println!("  β₁ = {} (loops)", betti.at(1));

    // Compute persistent Betti numbers at different thresholds
    println!("\nPersistent Betti numbers:");
    for threshold in [0, 1, 2, 3, 4, 5] {
        let b0 = BettiNumbers::persistent_betti(&reduced, 0, threshold);
        let b1 = BettiNumbers::persistent_betti(&reduced, 1, threshold);
        println!("  ε={}: β₀={}, β₁={}", threshold, b0, b1);
    }

    // Extract persistence diagram
    let barcode = reduced.barcode();
    let diagram = PersistenceDiagram::from_barcode(&barcode);

    println!("\nPersistence diagram:");
    for (b, d, dim) in diagram.points() {
        if d.is_finite() {
            println!("  H{}: ({:.3}, {:.3}) persistence={:.3}", dim, b, d, d - b);
        } else {
            println!("  H{}: ({:.3}, ∞)", dim, b);
        }
    }

    println!("\nDiagram statistics:");
    println!("  Total persistence (p=2): {:.4}", diagram.total_persistence(2.0));
    println!("  Entropy: {:.4}", diagram.entropy());
}
```

### Example 4: Comparing Two Persistence Diagrams with Bottleneck Distance

This example demonstrates the stability framework by comparing persistence diagrams from two similar point clouds.

```rust
use homology_engine::filtration::FiltrationBuilder;
use homology_engine::reduction::MatrixReduction;
use homology_engine::diagram::PersistenceDiagram;

fn main() {
    // Original point cloud
    let points_a: Vec<(f64, f64)> = vec![
        (0.0, 0.0),
        (1.0, 0.0),
        (0.5, 0.87),
    ];

    // Perturbed point cloud (each point shifted by ~0.1)
    let points_b: Vec<(f64, f64)> = vec![
        (0.05, 0.08),
        (1.02, 0.03),
        (0.48, 0.91),
    ];

    let mut builder = FiltrationBuilder::new();

    // Compute diagrams for both
    let filt_a = builder.rips(&points_a, 2.0, 2);
    let filt_b = builder.rips(&points_b, 2.0, 2);

    let dgm_a = PersistenceDiagram::from_barcode(
        &filt_a.boundary_matrix().reduce().barcode()
    );
    let dgm_b = PersistenceDiagram::from_barcode(
        &filt_b.boundary_matrix().reduce().barcode()
    );

    println!("Diagram A:");
    for (b, d, dim) in dgm_a.points() {
        println!("  H{}: ({:.3}, {:.3})", dim, b, d);
    }

    println!("\nDiagram B:");
    for (b, d, dim) in dgm_b.points() {
        println!("  H{}: ({:.3}, {:.3})", dim, b, d);
    }

    // Compute distances
    let bottleneck = dgm_a.bottleneck_distance(&dgm_b);
    let wasserstein_1 = dgm_a.wasserstein_distance(&dgm_b, 1.0);
    let wasserstein_2 = dgm_a.wasserstein_distance(&dgm_b, 2.0);

    println!("\nDistances:");
    println!("  Bottleneck:     {:.6}", bottleneck);
    println!("  Wasserstein-1:  {:.6}", wasserstein_1);
    println!("  Wasserstein-2:  {:.6}", wasserstein_2);

    // Verify stability: small perturbation → small diagram distance
    assert!(bottleneck < 0.5, "Stability violated!");
    println!("\n✓ Stability check passed: d_B < 0.5");
}
```

---

## Module Reference

| Module | Description | Key Types | Example Usage |
|--------|-------------|-----------|---------------|
| [`boundary`] | Sparse boundary matrices over Z/2Z | `BoundaryMatrix` | `BoundaryMatrix::from_simplices(&simplices)` |
| [`reduction`] | Column reduction algorithm | `ReducedMatrix`, trait `MatrixReduction` | `boundary.reduce()` |
| [`barcode`] | Persistence barcodes: intervals [b, d) | `Barcode` | `reduced.barcode()` |
| [`betti`] | Betti numbers and persistent Betti | `BettiNumbers` | `reduced.betti_numbers()` |
| [`diagram`] | Persistence diagrams, distances | `PersistenceDiagram` | `PersistenceDiagram::from_barcode(&bc)` |
| [`filtration`] | Filtration construction | `Filtration`, `FiltrationBuilder` | `FiltrationBuilder::new().rips(&pts, 2.0, 2)` |

### `boundary` — Sparse Boundary Matrices

The foundation of the computation pipeline. Represents the boundary operator ∂ as a sparse column-oriented matrix over Z/2Z.

```rust
// Create from simplices
let bm = BoundaryMatrix::from_simplices(&[
    vec![0], vec![1], vec![2],      // vertices
    vec![0, 1], vec![0, 2], vec![1, 2],  // edges
]);

// Verify ∂² = 0
assert!(bm.verify_chain_complex());

// XOR columns (fundamental operation for reduction)
// bm.add_columns(dst, src);  // dst ^= src
```

### `reduction` — Column Reduction

Implements the iterative left-to-right column reduction algorithm with explicit worklist. Not recursive.

```rust
let reduced = boundary.reduce();
// reduced.pairs() → [(birth, death), ...]
// reduced.unpaired() → [idx, ...] (infinite bars)
```

### `barcode` — Persistence Barcodes

The primary output format: a collection of half-open intervals [birth, death).

```rust
let barcode = reduced.barcode();
let (h0_finite, h0_inf) = barcode.bars_in_dimension(0);
let avg = barcode.average_lifetime();
let entropy = barcode.entropy();
```

### `betti` — Betti Numbers

Computes Betti numbers βₖ for each dimension k, plus persistent Betti curves.

```rust
let betti = reduced.betti_numbers();
println!("β₀ = {}", betti.at(0));  // connected components
println!("β₁ = {}", betti.at(1));  // loops

// Persistent Betti at threshold
let b0 = BettiNumbers::persistent_betti(&reduced, 0, 3);

// Betti curve: β₀ as function of filtration step
let curve = BettiNumbers::betti_curve(&reduced, 0);
```

### `diagram` — Persistence Diagrams and Distances

Scatter-plot representation with bottleneck and Wasserstein distance metrics.

```rust
let dgm = PersistenceDiagram::from_barcode(&barcode);
let bottleneck = dgm.bottleneck_distance(&other_dgm);
let w2 = dgm.wasserstein_distance(&other_dgm, 2.0);
```

### `filtration` — Filtration Construction

Builds filtrations from point clouds (Rips), scalar fields (lower-star), and weighted graphs (flag).

```rust
let mut builder = FiltrationBuilder::new();

// Vietoris-Rips from point cloud
let rips = builder.rips(&points, max_epsilon, max_dim);

// Lower-star from vertex function values
let ls = builder.lower_star(&vertex_values, &edges);

// Flag from weighted graph
let flag = builder.flag(n_vertices, &weighted_edges);
```

---

## Design Decisions

### Why Z/2Z Coefficients?

Over Z/2Z, the boundary operator simplifies dramatically:

1. **No signs**: ∂σ = Σ faces (all +1, no alternating signs)
2. **XOR = addition**: Adding columns is symmetric difference — no carry, no borrow
3. **Sufficient for TDA**: Z/2Z coefficients capture all torsion-free homology, which covers the vast majority of TDA applications
4. **Efficient**: Bit operations are fast and cache-friendly

The trade-off: we cannot detect torsion (e.g., ℝP² has Z/2Z torsion that's invisible with Q coefficients). But for point cloud data, this is almost never an issue.

### Why Sparse Representation?

The boundary matrix is extremely sparse. For a filtration with m simplices:
- A k-simplex has exactly k+1 boundary faces
- The matrix has m columns, each with at most k+1 non-zero entries
- Dense storage would be O(m²), sparse storage is O(m · k_avg)

Using `Vec<Vec<usize>>` (column-oriented sparse) means:
- Column operations (XOR) are merge-like, O(k₁ + k₂)
- Pivot finding is O(1) — just look at the last element
- Memory efficient for typical simplicial complexes

### Why Iterative Reduction (Not Recursive)?

The standard reduction algorithm processes columns left-to-right, using an inner loop to resolve pivot conflicts:

```rust
for j in 0..num_cols {
    loop {
        match pivot(j) {
            None => break,                    // zero column
            Some(p) if pivots.contains(p) => {
                add_columns(j, pivots[p]);    // XOR to resolve conflict
            }
            Some(p) => {
                pivots.insert(p, j);          // unique pivot
                break;
            }
        }
    }
}
```

This is **iterative with an explicit inner loop**, not recursive Gaussian elimination. Benefits:
- No stack overflow risk for large matrices
- Clear control flow — easy to reason about correctness
- Pivot tracking in `HashMap<usize, usize>` for O(1) lookup
- Each column is modified in-place, no backtracking

### Why Edition 2024?

Rust 2024 edition brings improvements to the language that benefit mathematical code:
- Stricter borrow checking catches more bugs at compile time
- `if let` chains for cleaner pattern matching
- Better trait solver for generic math code

### Why No External Dependencies?

This crate depends only on `serde` for serialization. No BLAS, no LAPACK, no C libraries. This means:
- **Cross-compilation** works out of the box
- **WASM support** is straightforward
- **Build times** are fast
- **Audit surface** is minimal

For production use at scale, consider specialized libraries like Ripser (C++) or PHAT (C++). This crate prioritizes clarity and correctness over raw performance.

---

## Performance Analysis

### Complexity

| Operation | Worst Case | Typical Case |
|-----------|-----------|--------------|
| Boundary construction | O(m · k²) | O(m · k) |
| Column reduction | O(m³) | O(m²) |
| Barcode extraction | O(m) | O(m) |
| Betti numbers | O(m) | O(m) |
| Bottleneck distance | O(n²) | O(n · log n) |
| Rips filtration (d-dim) | O(n^(d+1)) | O(n^(d+1)) |

Where m = number of simplices, k = max simplex dimension, n = number of diagram points.

The column reduction is the bottleneck. Worst case O(m³) occurs when each column requires O(m) reductions, each of O(m) cost. In practice, with sparse matrices and geometric data, the typical complexity is closer to O(m²).

### Memory

| Structure | Size |
|-----------|------|
| Boundary matrix (sparse) | O(m · k_avg) where k_avg = average column fill |
| Boundary matrix (dense) | O(m²) — not used here |
| Pivot map | O(m) entries |
| Barcode | O(m) intervals |
| Filtration | O(m · d) where d = average simplex size |

For a point cloud with n points and max dimension D:
- Number of simplices m ≈ C(n, D+1) = n^(D+1) / (D+1)!
- A 100-point cloud with max dim 2 has ~166,000 simplices
- Sparse boundary matrix uses ~1MB for this case

### Benchmarks (Approximate)

| Input | Simplices | Reduction Time |
|-------|-----------|---------------|
| 10 points, dim 1 | ~55 | < 1ms |
| 50 points, dim 1 | ~1,275 | ~5ms |
| 100 points, dim 2 | ~166,000 | ~500ms |
| 100 points, dim 1 | ~5,050 | ~50ms |

These are rough estimates. For large inputs, use Ripser or PHAT which employ advanced optimizations (cohomology, clearing optimization, chunk-based parallelism).

---

## Comparison with Other Libraries

| Feature | homology-engine | Ripser | PHAT | Dionysus 2 | GUDHI |
|---------|----------------|--------|------|-------------|-------|
| Language | Rust | C++ | C++ | C++ | C++/Python |
| External deps | serde only | None | None | Boost | CGAL, Boost |
| Coefficients | Z/2Z | Z/pZ | Z/2Z | Z/2Z, Z/pZ | Z/2Z, Z/pZ |
| Rips support | ✓ | ✓ (optimized) | ✗ | ✓ | ✓ |
| Persistence pairs | ✓ | ✓ | ✓ | ✓ | ✓ |
| Distance metrics | ✓ (greedy) | ✗ | ✗ | ✓ | ✓ |
| Serde support | ✓ | ✗ | ✗ | ✗ | ✗ |
| Educational clarity | ★★★★★ | ★★☆ | ★★★ | ★★★ | ★★☆ |
| Performance | ★★☆ | ★★★★★ | ★★★★ | ★★★ | ★★★★ |
| WASM-ready | ✓ | ✗ | ✗ | ✗ | ✗ |

### What This Crate Does Differently

1. **Pure Rust, zero native deps** — compiles anywhere Rust does, including WASM
2. **Educational design** — each module is a self-contained mathematical concept
3. **Full pipeline in one crate** — filtration → reduction → barcode → diagram → distances
4. **Serde serialization** — all types derive `Serialize` + `Deserialize`
5. **No unsafe code** — 100% safe Rust

### When to Use Something Else

- **Million-point datasets** → Ripser (specialized cohomology algorithms, clearing optimization)
- **Custom reduction strategies** → PHAT (implements many algorithms: standard, chunk, sweep, twist)
- **Python ecosystem** → GUDHI, Ripser.py, scikit-tda
- **Z/pZ coefficients for p ≠ 2** → Ripser, Dionysus
- **Cohomological persistence** → Ripser (often 10x faster than homology)

---

## Practical Applications

### Shape Analysis

Classify 3D shapes by their topology. A sphere has (β₀, β₁, β₂) = (1, 0, 1), a torus has (1, 2, 1), and a double torus has (1, 4, 1). Persistence diagrams provide a more robust signature that's stable under deformation.

### Anomaly Detection

Points that create persistent features (long-lived bars) are structurally significant. Short-lived bars near the diagonal represent noise. By filtering bars by persistence, you can separate signal from noise without arbitrary thresholds.

### Sensor Network Coverage

In a sensor network, sensors cover regions. The topology of the coverage region determines if there are gaps. Persistent homology can identify coverage holes and track when they appear/disappear as sensors are added or removed.

### Collaboration Networks

In social/collaboration graphs, persistent loops reveal cyclic structures (A works with B, B with C, C with A). The persistence of these structures indicates their stability over time.

### Molecular Structure Analysis

Proteins and molecules can be analyzed as point clouds. Persistent homology identifies pockets, tunnels, and voids that correspond to binding sites and functional regions.

### Time Series Analysis

Using Takens' embedding theorem, a time series can be converted to a point cloud via delay embedding. Persistent homology of this point cloud reveals periodic structure, chaos, and regime changes.

---

## API Stability

This is version 0.1.x. The API may change between minor versions. The core types (`BoundaryMatrix`, `ReducedMatrix`, `Barcode`, `BettiNumbers`, `PersistenceDiagram`, `Filtration`) are stable, but method signatures may evolve.

Breaking changes will be documented in the changelog.

---

## References

1. **Edelsbrunner, H. & Harer, J.** (2010). *Computational Topology: An Introduction*. American Mathematical Society. — The foundational textbook for computational topology and persistent homology.

2. **Zomorodian, A. & Carlsson, G.** (2005). "Computing Persistent Homology." *Discrete & Computational Geometry*, 33(2), 249–274. — The original algorithm for computing persistent homology via matrix reduction.

3. **Carlsson, G.** (2009). "Topology and Data." *Bulletin of the American Mathematical Society*, 46(2), 255–308. — The seminal survey connecting topology to data analysis.

4. **Otter, N., Porter, M.A., Tillmann, U., Grindrod, P. & Harrington, H.A.** (2017). "A Roadmap for the Computation of Persistent Homology." *EPJ Data Science*, 6(17). — Comprehensive benchmarking and practical guide for persistent homology computation.

5. **Chazal, F., Fasy, B.T., Lecci, F., Rinaldo, A., & Wasserman, L.** (2014). "Stochastic Convergence of Persistence Landscapes and Silhouettes." *Journal of Computational Geometry*, 6(2), 140–161. — Statistical foundations for persistent homology.

6. **Cohen-Steiner, D., Edelsbrunner, H. & Harer, J.** (2007). "Stability of Persistence Diagrams." *Discrete & Computational Geometry*, 37(1), 103–120. — The stability theorem: d_B(Dgm(f), Dgm(g)) ≤ ‖f - g‖_∞.

7. **Ghrist, R.** (2014). *Elementary Applied Topology*. Createspace Independent Publishing. — An accessible introduction to applied topology with broad coverage.

8. **Chazal, F., de Silva, V., Glisse, M. & Oudot, S.** (2016). *The Structure and Stability of Persistence Modules*. Springer. — Rigorous treatment of persistence module theory and stability.

9. **Kerber, M., Morozov, D. & Nigmetov, A.** (2017). "Geometry Helps to Compare Persistence Diagrams." *Journal of Experimental Algorithmics*, 22. — Efficient algorithms for bottleneck and Wasserstein distances.

10. **Bauer, U.** (2021). "Ripser: Efficient Computation of Vietoris-Rips Persistence Barcodes." *Journal of Open Research Software*, 9(1). — The Ripser algorithm and implementation.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
