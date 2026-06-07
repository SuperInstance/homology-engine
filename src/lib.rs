//! # homology-engine
//!
//! A persistent homology computation engine for topological data analysis (TDA).
//!
//! This crate provides a pure-Rust implementation of the core algorithms used in
//! persistent homology, with no external dependencies beyond `serde`. It is designed
//! for educational clarity and correctness, making the mathematics of topological
//! data analysis accessible and transparent.
//!
//! ## Quick Start
//!
//! ```
//! use homology_engine::filtration::{Filtration, FiltrationBuilder};
//! use homology_engine::boundary::BoundaryMatrix;
//! use homology_engine::reduction::MatrixReduction;
//! use homology_engine::barcode::Barcode;
//! use homology_engine::betti::BettiNumbers;
//!
//! // Build a filtration from a point cloud (Rips complex)
//! let points: Vec<(f64, f64)> = vec![(0.0, 0.0), (1.0, 0.0), (0.5, 0.87), (0.5, 0.0)];
//! let mut builder = FiltrationBuilder::new();
//! let filtration = builder.rips(&points, 2.0, 2);
//!
//! // Get boundary matrix
//! let boundary = filtration.boundary_matrix();
//!
//! // Reduce to extract persistence information
//! let reduced = boundary.reduce();
//!
//! // Extract barcode and Betti numbers
//! let barcode = reduced.barcode();
//! let betti = reduced.betti_numbers();
//! println!("Betti-0: {} (connected components)", betti.at(0));
//! ```
//!
//! ## Modules
//!
//! - [`boundary`] — Sparse boundary matrices over Z/2Z
//! - [`reduction`] — Column reduction algorithm (Gaussian elimination over Z/2Z)
//! - [`barcode`] — Persistence barcodes: intervals tracking birth/death of features
//! - [`betti`] — Betti numbers: ranks of homology groups
//! - [`diagram`] — Persistence diagrams and distance metrics
//! - [`filtration`] — Filtration construction (Rips, lower-star, flag)

pub mod barcode;
pub mod betti;
pub mod boundary;
pub mod diagram;
pub mod filtration;
pub mod reduction;
