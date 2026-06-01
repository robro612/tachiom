//! Proximity Graph Clustering (PGC).
//!
//! Token-type-agnostic alternative to TAC for coarse centroid generation.
//! Iteratively refines random anchor vectors using HNSW-based nearest-neighbour
//! assignment and L2-normalised mean updates.

use half::f16;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use crate::graph::Graph;
use crate::hnsw::{HNSW, HNSWBuildConfiguration, HNSWSearchConfiguration};
use vectorium::core::index::Index;
use vectorium::core::vector::DenseVectorView;
use vectorium::distances::DotProduct;
use vectorium::{DenseDataset, PlainDenseQuantizer};

type AnchorDataset = DenseDataset<PlainDenseQuantizer<f16, DotProduct>>;
type AnchorHNSW = HNSW<AnchorDataset, Graph>;

// ============================================================================
// Public types
// ============================================================================

/// What to do when an anchor receives zero assignments in an iteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmptyAnchorStrategy {
    /// Reset the empty anchor to a randomly sampled corpus vector.
    Resample,
    /// Drop the empty anchor, permanently reducing the active centroid count.
    Remove,
    /// Reinitialise the empty anchor from a random member of the most-populated
    /// cluster.  Keeps the anchor count fixed (like Resample) while seeding the
    /// new position from a region that is demonstrably dense in the data.
    Split,
}

/// Result returned by [`ProximityGraphClustering::cluster`].
pub struct PgcResult {
    /// Flat f16 buffer of all centroids. Layout: `n_centroids × dim`.
    pub centroids: Vec<f16>,
    /// Per-vector assignment: `assignments[i]` is the centroid index for the i-th input vector.
    pub assignments: Vec<usize>,
    /// Vector dimension.
    pub dim: usize,
    /// Final number of centroids (may be < the requested count when using `Remove` strategy).
    pub n_centroids: usize,
}

// ============================================================================
// Builder
// ============================================================================

pub struct PgcBuilder {
    n_iter: usize,
    sample_multiplier: usize,
    empty_strategy: EmptyAnchorStrategy,
    iter_hnsw_m: usize,
    iter_ef_construction: usize,
    iter_ef_search: usize,
    seed: u64,
    verbose: bool,
}

impl Default for PgcBuilder {
    fn default() -> Self {
        PgcBuilder {
            n_iter: 10,
            sample_multiplier: 5,
            empty_strategy: EmptyAnchorStrategy::Resample,
            iter_hnsw_m: 16,
            iter_ef_construction: 200,
            iter_ef_search: 50,
            seed: 42,
            verbose: false,
        }
    }
}

impl PgcBuilder {
    pub fn new() -> Self {
        PgcBuilder::default()
    }

    pub fn n_iter(mut self, n_iter: usize) -> Self {
        self.n_iter = n_iter;
        self
    }

    /// Number of corpus vectors sampled per iteration = `n_centroids × sample_multiplier`.
    pub fn sample_multiplier(mut self, sample_multiplier: usize) -> Self {
        self.sample_multiplier = sample_multiplier;
        self
    }

    pub fn empty_strategy(mut self, empty_strategy: EmptyAnchorStrategy) -> Self {
        self.empty_strategy = empty_strategy;
        self
    }

    pub fn iter_hnsw_m(mut self, iter_hnsw_m: usize) -> Self {
        self.iter_hnsw_m = iter_hnsw_m;
        self
    }

    pub fn iter_ef_construction(mut self, iter_ef_construction: usize) -> Self {
        self.iter_ef_construction = iter_ef_construction;
        self
    }

    pub fn iter_ef_search(mut self, iter_ef_search: usize) -> Self {
        self.iter_ef_search = iter_ef_search;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    pub fn build(self) -> ProximityGraphClustering {
        ProximityGraphClustering {
            n_iter: self.n_iter,
            sample_multiplier: self.sample_multiplier,
            empty_strategy: self.empty_strategy,
            iter_hnsw_m: self.iter_hnsw_m,
            iter_ef_construction: self.iter_ef_construction,
            iter_ef_search: self.iter_ef_search,
            seed: self.seed,
            verbose: self.verbose,
        }
    }
}

// ============================================================================
// ProximityGraphClustering
// ============================================================================

pub struct ProximityGraphClustering {
    n_iter: usize,
    sample_multiplier: usize,
    empty_strategy: EmptyAnchorStrategy,
    iter_hnsw_m: usize,
    iter_ef_construction: usize,
    iter_ef_search: usize,
    seed: u64,
    verbose: bool,
}

impl Default for ProximityGraphClustering {
    fn default() -> Self {
        PgcBuilder::default().build()
    }
}

impl ProximityGraphClustering {
    pub fn builder() -> PgcBuilder {
        PgcBuilder::new()
    }

    /// Run PGC on a flat buffer of f16 vectors.
    ///
    /// # Arguments
    /// * `data`         – Flat f16 buffer; vector `i` occupies `data[i*dim .. (i+1)*dim]`.
    /// * `dim`          – Per-vector embedding dimension.
    /// * `n_centroids`  – Requested number of coarse centroids.
    ///
    /// # Panics
    /// Panics if `data.len() % dim != 0` or `n_centroids == 0`.
    pub fn cluster(&self, data: &[f16], dim: usize, n_centroids: usize) -> PgcResult {
        assert_eq!(data.len() % dim, 0, "data.len() must be divisible by dim");
        assert!(n_centroids > 0, "n_centroids must be > 0");

        let n_vectors = data.len() / dim;
        assert!(
            n_centroids <= n_vectors,
            "n_centroids ({}) must be <= n_vectors ({})",
            n_centroids,
            n_vectors
        );

        let total_start = Instant::now();
        let mut rng = StdRng::seed_from_u64(self.seed);

        if self.verbose {
            println!(
                "=== PGC: {} vectors × dim={}, {} centroids, {} iters, strategy={:?} ===",
                n_vectors, dim, n_centroids, self.n_iter, self.empty_strategy
            );
        }

        // ── Step 1: Initialise anchors by uniform random sampling ─────────────
        // Use sampling with replacement for O(n_centroids) memory cost.
        let mut anchors: Vec<f16> = Vec::with_capacity(n_centroids * dim);
        for _ in 0..n_centroids {
            let idx = rng.gen_range(0..n_vectors);
            anchors.extend_from_slice(&data[idx * dim..(idx + 1) * dim]);
        }
        let mut n_active = n_centroids;

        let search_config = HNSWSearchConfiguration::default().with_ef_search(self.iter_ef_search);

        // ── Step 2: Iterative refinement ──────────────────────────────────────
        for iter in 0..self.n_iter {
            if n_active == 0 {
                if self.verbose {
                    println!(
                        "  PGC: all anchors removed before iter {} — stopping early",
                        iter + 1
                    );
                }
                break;
            }

            let iter_start = Instant::now();

            let hnsw = build_anchor_hnsw(
                &anchors,
                n_active,
                dim,
                self.iter_hnsw_m,
                self.iter_ef_construction,
            );

            // Sample `n_active * sample_multiplier` random vector indices (with replacement).
            let sample_n = (n_active * self.sample_multiplier).min(n_vectors);
            let sampled: Vec<usize> = if sample_n >= n_vectors {
                (0..n_vectors).collect()
            } else {
                (0..sample_n).map(|_| rng.gen_range(0..n_vectors)).collect()
            };

            // Parallel: assign each sampled vector to its nearest anchor.
            let iter_assignments: Vec<usize> = sampled
                .par_iter()
                .map(|&vidx| {
                    let q: Vec<f32> = data[vidx * dim..(vidx + 1) * dim]
                        .iter()
                        .map(|x| x.to_f32())
                        .collect();
                    let hits = hnsw.search(DenseVectorView::new(&q), 1, &search_config);
                    if hits.is_empty() {
                        0
                    } else {
                        hits[0].vector as usize
                    }
                })
                .collect();

            // Serial: accumulate sum vectors and counts per anchor.
            let mut sums = vec![0.0f32; n_active * dim];
            let mut counts = vec![0usize; n_active];
            for (&vidx, &anchor_idx) in sampled.iter().zip(iter_assignments.iter()) {
                let a = anchor_idx.min(n_active - 1); // clamp to guard against HNSW approximate hits
                counts[a] += 1;
                let src = &data[vidx * dim..(vidx + 1) * dim];
                let dst = &mut sums[a * dim..(a + 1) * dim];
                for (d, s) in dst.iter_mut().zip(src.iter()) {
                    *d += s.to_f32();
                }
            }

            let n_empty = counts.iter().filter(|&&c| c == 0).count();

            // Update anchors based on means + empty-anchor strategy.
            match self.empty_strategy {
                EmptyAnchorStrategy::Resample => {
                    // Keep n_active unchanged; resample empty anchors.
                    for a in 0..n_active {
                        if counts[a] > 0 {
                            let mut mean: Vec<f32> = sums[a * dim..(a + 1) * dim]
                                .iter()
                                .map(|&x| x / counts[a] as f32)
                                .collect();
                            l2_normalize_f32(&mut mean);
                            for (j, &v) in mean.iter().enumerate() {
                                anchors[a * dim + j] = f16::from_f32(v);
                            }
                        } else {
                            let ridx = rng.gen_range(0..n_vectors);
                            anchors[a * dim..(a + 1) * dim]
                                .copy_from_slice(&data[ridx * dim..(ridx + 1) * dim]);
                        }
                    }
                }
                EmptyAnchorStrategy::Remove => {
                    // Compact anchors: keep only those with at least one assignment.
                    let mut new_anchors: Vec<f16> = Vec::with_capacity(n_active * dim);
                    for a in 0..n_active {
                        if counts[a] > 0 {
                            let mut mean: Vec<f32> = sums[a * dim..(a + 1) * dim]
                                .iter()
                                .map(|&x| x / counts[a] as f32)
                                .collect();
                            l2_normalize_f32(&mut mean);
                            for &v in &mean {
                                new_anchors.push(f16::from_f32(v));
                            }
                        }
                    }
                    n_active = new_anchors.len() / dim;
                    anchors = new_anchors;
                }
                EmptyAnchorStrategy::Split => {
                    // Find the most-populated anchor.
                    let max_anchor = counts
                        .iter()
                        .enumerate()
                        .max_by_key(|&(_, &c)| c)
                        .map(|(i, _)| i)
                        .unwrap_or(0);

                    // Collect the sampled vector indices assigned to that anchor.
                    let max_members: Vec<usize> = sampled
                        .iter()
                        .zip(iter_assignments.iter())
                        .filter_map(|(&vidx, &aidx)| {
                            if aidx.min(n_active - 1) == max_anchor {
                                Some(vidx)
                            } else {
                                None
                            }
                        })
                        .collect();

                    for a in 0..n_active {
                        if counts[a] > 0 {
                            let mut mean: Vec<f32> = sums[a * dim..(a + 1) * dim]
                                .iter()
                                .map(|&x| x / counts[a] as f32)
                                .collect();
                            l2_normalize_f32(&mut mean);
                            for (j, &v) in mean.iter().enumerate() {
                                anchors[a * dim + j] = f16::from_f32(v);
                            }
                        } else {
                            // Seed from a random member of the busiest cluster.
                            let ridx = if !max_members.is_empty() {
                                max_members[rng.gen_range(0..max_members.len())]
                            } else {
                                rng.gen_range(0..n_vectors) // degenerate fallback
                            };
                            anchors[a * dim..(a + 1) * dim]
                                .copy_from_slice(&data[ridx * dim..(ridx + 1) * dim]);
                        }
                    }
                    // n_active stays unchanged.
                }
            }

            if self.verbose {
                println!(
                    "  PGC iter {}/{}: {} active anchors, {} empty (strategy: {:?}) — {:.2?}",
                    iter + 1,
                    self.n_iter,
                    n_active,
                    n_empty,
                    self.empty_strategy,
                    iter_start.elapsed(),
                );
            }
        }

        if n_active == 0 {
            // Degenerate fallback: return first n_centroids vectors as centroids.
            let fallback_n = n_centroids.min(n_vectors);
            let centroids: Vec<f16> = data[..fallback_n * dim].to_vec();
            let assignments = vec![0usize; n_vectors];
            return PgcResult {
                centroids,
                assignments,
                dim,
                n_centroids: fallback_n,
            };
        }

        // ── Step 3: Final assignment over all corpus vectors ──────────────────
        if self.verbose {
            println!(
                "=== PGC: Final assignment ({} vectors → {} anchors) ===",
                n_vectors, n_active
            );
        }

        let final_hnsw = build_anchor_hnsw(
            &anchors,
            n_active,
            dim,
            self.iter_hnsw_m,
            self.iter_ef_construction,
        );

        let completed = AtomicUsize::new(0);
        let print_every = (n_vectors / 20).max(1);
        let verbose = self.verbose;

        let assignments: Vec<usize> = (0..n_vectors)
            .into_par_iter()
            .map(|vidx| {
                let q: Vec<f32> = data[vidx * dim..(vidx + 1) * dim]
                    .iter()
                    .map(|x| x.to_f32())
                    .collect();
                let hits = final_hnsw.search(DenseVectorView::new(&q), 1, &search_config);
                let anchor = if hits.is_empty() {
                    0
                } else {
                    hits[0].vector as usize
                };
                if verbose {
                    let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                    if done % print_every == 0 || done == n_vectors {
                        println!(
                            "  Final assign: {}/{} ({:.0}%)",
                            done,
                            n_vectors,
                            100.0 * done as f64 / n_vectors as f64,
                        );
                    }
                }
                anchor
            })
            .collect();

        if self.verbose {
            println!(
                "✓ PGC complete in {:.2?} — {} centroids",
                total_start.elapsed(),
                n_active
            );
        }

        PgcResult {
            centroids: anchors,
            assignments,
            dim,
            n_centroids: n_active,
        }
    }
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Build a lightweight HNSW index over `n_active` f16 anchor vectors.
fn build_anchor_hnsw(
    anchors: &[f16],
    n_active: usize,
    dim: usize,
    m: usize,
    ef_construction: usize,
) -> AnchorHNSW {
    let buf: Vec<f16> = anchors[..n_active * dim].to_vec();
    let dataset = AnchorDataset::from_raw(
        buf.into_boxed_slice(),
        n_active,
        PlainDenseQuantizer::<f16, DotProduct>::new(dim),
    );
    AnchorHNSW::build_index(
        dataset,
        &HNSWBuildConfiguration::default()
            .with_num_neighbors(m)
            .with_ef_construction(ef_construction),
    )
}

/// Normalise `v` in-place to unit L2 norm.  No-op if the norm is near zero.
fn l2_normalize_f32(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}
