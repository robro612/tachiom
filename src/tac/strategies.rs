//! Centroid allocation strategies for Token-Aware Clustering.
//!
//! Provides the damped-spread strategy for distributing a centroid budget across
//! token groups, plus helpers for computing the spread measure (sum of variances).

use half::f16;
use rayon::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

// ============================================================================
// Spread Measure
// ============================================================================

/// Compute the spread measure (W_i) for a token group: sum of per-dimension variances.
pub fn compute_spread_measure(data: &[f16], indices: &[usize], dim: usize) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }

    let n = indices.len();
    let inv_n_f32 = 1.0f32 / n as f32;

    // Pass 1: accumulate means (row-major traversal — cache-friendly).
    let mut means = vec![0.0f32; dim];
    for &idx in indices {
        let vec = unsafe { data.get_unchecked(idx * dim..(idx + 1) * dim) };
        for (m, &v) in means.iter_mut().zip(vec.iter()) {
            *m += v.to_f32();
        }
    }
    for m in means.iter_mut() {
        *m *= inv_n_f32;
    }

    // Pass 2: accumulate squared deviations (sequential, deterministic).
    let mut total_variance = 0.0f64;
    for &idx in indices {
        let vec = unsafe { data.get_unchecked(idx * dim..(idx + 1) * dim) };
        total_variance += compute_squared_diff(vec, &means);
    }

    total_variance / n as f64
}

#[inline]
fn compute_squared_diff(vector: &[f16], means: &[f32]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") {
            return unsafe { compute_squared_diff_avx2(vector, means) };
        }
    }
    compute_squared_diff_scalar(vector, means)
}

#[inline]
fn compute_squared_diff_scalar(vector: &[f16], means: &[f32]) -> f64 {
    vector
        .iter()
        .zip(means.iter())
        .map(|(&v, &m)| {
            let diff = v.to_f32() - m;
            (diff * diff) as f64
        })
        .sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn compute_squared_diff_avx2(vector: &[f16], means: &[f32]) -> f64 {
    use std::arch::x86_64::*;

    let len = vector.len();
    let chunks = len / 8;
    let mut sum = 0.0f64;

    if chunks > 0 {
        sum += unsafe {
            let mut acc = _mm256_setzero_ps();
            for i in 0..chunks {
                let offset = i * 8;
                let vec_f16_lo =
                    _mm_loadu_si128(vector.as_ptr().add(offset) as *const i16 as *const __m128i);
                let vec_f32 = _mm256_cvtph_ps(vec_f16_lo);
                let means_f32 = _mm256_loadu_ps(means.as_ptr().add(offset));
                let diff = _mm256_sub_ps(vec_f32, means_f32);
                acc = _mm256_add_ps(acc, _mm256_mul_ps(diff, diff));
            }
            let mut tmp = [0.0f32; 8];
            _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
            tmp.iter().map(|&x| x as f64).sum::<f64>()
        };
    }

    for i in (chunks * 8)..len {
        let diff = vector[i].to_f32() - means[i];
        sum += (diff * diff) as f64;
    }
    sum
}

// ============================================================================
// Damped Spread Strategy
// ============================================================================

/// Allocation hyper-parameters for [`allocate_centroids_damped_spread`].
///
/// Correspond to the paper's μ, τ, ε, θ respectively.
#[derive(Debug, Clone, Copy)]
pub struct TacAllocParams {
    /// μ — tokens with fewer than this many vectors get 1 centroid (micro tier).
    pub micro_threshold: usize,
    /// τ — tokens in [μ, τ) get 2 centroids (small tier).
    pub small_threshold: usize,
    /// ε — hard floor on centroids for active (≥ τ) tokens.
    pub hard_floor: usize,
    /// θ — cap formula: `max_centroids_for_token = n / θ`.
    /// Also used in the training-sample formula: `max(1M, 2θk, n/(2·n_iter))`.
    pub min_pts_per_centroid: usize,
}

impl Default for TacAllocParams {
    fn default() -> Self {
        TacAllocParams {
            micro_threshold: 128,
            small_threshold: 256,
            hard_floor: 4,
            min_pts_per_centroid: 39,
        }
    }
}

/// Allocate a centroid budget across token groups using the damped-spread strategy.
///
/// The strategy favors medium-frequency terms with high variance, penalises
/// high-frequency ones via sqrt-damping, and gives a fixed low-cost floor to
/// rare terms.
///
/// Returns a map `token_id → number of centroids`.
pub fn allocate_centroids_damped_spread(
    token_groups: &HashMap<usize, Vec<usize>>,
    data: &[f16],
    dim: usize,
    total_centroids: usize,
    verbose: bool,
    params: &TacAllocParams,
) -> HashMap<usize, usize> {
    let micro_threshold = params.micro_threshold;
    let small_threshold = params.small_threshold;
    let hard_floor = params.hard_floor;
    let min_pts_per_centroid = params.min_pts_per_centroid;

    let n_vectors: usize = token_groups.values().map(|v| v.len()).sum();

    eprintln!("\n=== Damped Spread Centroid Allocation ===");
    eprintln!("Total vectors: {}", n_vectors);
    eprintln!("Budget: {} centroids", total_centroids);
    eprintln!(
        "Thresholds: Micro < {}, Small < {}",
        micro_threshold, small_threshold
    );
    eprintln!(
        "Bounds: Floor = {}, Min points/centroid = {}",
        hard_floor, min_pts_per_centroid
    );

    // ── Phase 1: Tail handling ────────────────────────────────────────────────
    eprintln!("\n--- Phase 1: Tail Handling ---");

    let mut micro_tokens: Vec<usize> = Vec::new();
    let mut small_tokens: Vec<usize> = Vec::new();
    let mut active_tokens: Vec<(usize, usize)> = Vec::new();

    for (&token_id, indices) in token_groups {
        match indices.len() {
            n if n < micro_threshold => micro_tokens.push(token_id),
            n if n < small_threshold => small_tokens.push(token_id),
            n => active_tokens.push((token_id, n)),
        }
    }

    let micro_budget = micro_tokens.len();
    let small_budget = small_tokens.len() * 2;
    let tail_budget = micro_budget + small_budget;

    eprintln!(
        "Micro tokens (< {}): {} tokens → {} centroids",
        micro_threshold,
        micro_tokens.len(),
        micro_budget
    );
    eprintln!(
        "Small tokens ({}-{}): {} tokens → {} centroids",
        micro_threshold,
        small_threshold,
        small_tokens.len(),
        small_budget
    );
    eprintln!(
        "Active tokens (≥ {}): {} tokens",
        small_threshold,
        active_tokens.len()
    );
    eprintln!("Tail budget used: {}", tail_budget);

    let remaining_budget = total_centroids.saturating_sub(tail_budget);
    eprintln!("Remaining budget for active tokens: {}", remaining_budget);

    let mut allocation: HashMap<usize, usize> = HashMap::new();
    for &token_id in &micro_tokens {
        allocation.insert(token_id, 1);
    }
    for &token_id in &small_tokens {
        allocation.insert(token_id, 2);
    }

    if active_tokens.is_empty() || remaining_budget == 0 {
        eprintln!("\n=== Allocation Complete (no active tokens or budget) ===");
        eprintln!("Total allocated: {}", allocation.values().sum::<usize>());
        return allocation;
    }

    // ── Phase 2: Damped scoring ───────────────────────────────────────────────
    eprintln!("\n--- Phase 2: Damped Scoring ---");
    eprintln!(
        "Computing spread measures for {} active tokens...",
        active_tokens.len()
    );
    let spread_start = Instant::now();

    let spread_measures: Vec<(usize, usize, f64)> = active_tokens
        .par_iter()
        .map(|(token_id, count)| {
            let spread = compute_spread_measure(data, &token_groups[token_id], dim);
            (*token_id, *count, spread)
        })
        .collect();

    eprintln!("✓ Spread computation in {:.2?}", spread_start.elapsed());

    let damped_scores: Vec<(usize, usize, f64, f64)> = spread_measures
        .iter()
        .map(|&(token_id, count, spread)| {
            let score = (count as f64).sqrt() * spread;
            (token_id, count, spread, score)
        })
        .collect();

    let total_score: f64 = damped_scores.iter().map(|&(_, _, _, s)| s).sum();

    if verbose {
        eprintln!("\nTop 10 damped scores:");
        let mut sorted = damped_scores.clone();
        sorted.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
        for (i, (token_id, count, spread, score)) in sorted.iter().take(10).enumerate() {
            eprintln!(
                "  {}. Token {}: count={}, spread={:.4}, score={:.4}",
                i + 1,
                token_id,
                count,
                spread,
                score
            );
        }
    }

    let provisional: Vec<(usize, f64)> = if total_score > 0.0 {
        damped_scores
            .iter()
            .map(|&(tid, _, _, score)| (tid, score / total_score * remaining_budget as f64))
            .collect()
    } else {
        let equal = remaining_budget as f64 / active_tokens.len() as f64;
        damped_scores
            .iter()
            .map(|&(tid, _, _, _)| (tid, equal))
            .collect()
    };

    // ── Phase 3: Bounding (floor + cap) ──────────────────────────────────────
    eprintln!("\n--- Phase 3: Bounding ---");

    let mut bounded: Vec<(usize, usize, f64, bool)> = Vec::new();

    for (token_id, prov) in &provisional {
        let count = token_groups[token_id].len();
        let cap = (count / min_pts_per_centroid).max(hard_floor);
        let floored = prov.max(hard_floor as f64);
        let rounded = (floored.round() as usize).max(hard_floor);
        let final_alloc = rounded.min(cap);
        let is_capped = final_alloc == cap && rounded > cap;
        let frac = floored - floored.floor();
        bounded.push((*token_id, final_alloc, frac, is_capped));
    }

    eprintln!(
        "Tokens hitting floor ({}): {}",
        hard_floor,
        bounded
            .iter()
            .filter(|&&(_, a, _, _)| a == hard_floor)
            .count()
    );
    eprintln!(
        "Tokens hitting cap: {}",
        bounded.iter().filter(|&&(_, _, _, c)| c).count()
    );

    if verbose {
        let mut sorted_b = bounded.clone();
        sorted_b.sort_by_key(|&(_, a, _, _)| std::cmp::Reverse(a));
        eprintln!("\nTop 5 provisional allocations:");
        for (i, (token_id, alloc, _, is_capped)) in sorted_b.iter().take(5).enumerate() {
            let count = token_groups[token_id].len();
            eprintln!(
                "  {}. Token {}: {} vectors → {} centroids ({:.0} vecs/centroid){}",
                i + 1,
                token_id,
                count,
                alloc,
                count as f64 / *alloc as f64,
                if *is_capped { " (CAPPED)" } else { "" }
            );
        }
    }

    // ── Phase 4: Budget reconciliation ───────────────────────────────────────
    eprintln!("\n--- Phase 4: Budget Reconciliation ---");

    let active_sum: usize = bounded.iter().map(|&(_, a, _, _)| a).sum();
    let current_total = tail_budget + active_sum;
    let diff = total_centroids as i64 - current_total as i64;

    eprintln!(
        "Current: {} (tail: {}, active: {}), target: {}, diff: {}",
        current_total, tail_budget, active_sum, total_centroids, diff
    );

    if diff > 0 {
        let surplus = diff as usize;
        let mut non_capped: Vec<_> = bounded
            .iter()
            .filter(|&&(_, _, _, c)| !c)
            .cloned()
            .collect();
        non_capped.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

        let mut distributed = 0;
        while distributed < surplus && !non_capped.is_empty() {
            let mut made_progress = false;
            for (token_id, alloc, _, _) in non_capped.iter_mut() {
                if distributed >= surplus {
                    break;
                }
                let cap = (token_groups[token_id].len() / min_pts_per_centroid).max(hard_floor);
                if *alloc < cap {
                    *alloc += 1;
                    distributed += 1;
                    made_progress = true;
                }
            }
            if !made_progress {
                break;
            }
        }

        let update_map: HashMap<usize, usize> =
            non_capped.iter().map(|&(tid, a, _, _)| (tid, a)).collect();
        for (tid, alloc, _, _) in bounded.iter_mut() {
            if let Some(&new_a) = update_map.get(tid) {
                *alloc = new_a;
            }
        }

        let remaining_surplus = surplus - distributed;
        if remaining_surplus > 0 {
            eprintln!(
                "Still {} surplus after non-capped pass; distributing by fractional remainder",
                remaining_surplus
            );
            let mut candidates: Vec<_> = bounded.iter_mut().collect();
            candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
            let mut extra = 0;
            while extra < remaining_surplus && !candidates.is_empty() {
                for c in candidates.iter_mut() {
                    if extra >= remaining_surplus {
                        break;
                    }
                    c.1 += 1;
                    extra += 1;
                }
            }
        }

        eprintln!("Distributed {} surplus centroids", distributed);
    } else if diff < 0 {
        let deficit = (-diff) as usize;
        let mut removed = 0;
        bounded.sort_by_key(|&(_, a, _, _)| std::cmp::Reverse(a));
        while removed < deficit {
            let mut made_progress = false;
            for (_, alloc, _, _) in bounded.iter_mut() {
                if removed >= deficit {
                    break;
                }
                if *alloc > hard_floor {
                    *alloc -= 1;
                    removed += 1;
                    made_progress = true;
                }
            }
            if !made_progress {
                eprintln!("Warning: cannot remove more centroids without going below floor");
                break;
            }
        }
        eprintln!("Removed {} deficit centroids", removed);
    }

    for (token_id, alloc, _, _) in bounded {
        allocation.insert(token_id, alloc);
    }

    // ── Summary ───────────────────────────────────────────────────────────────
    let total_allocated: usize = allocation.values().sum();
    eprintln!("\n=== Damped Spread Allocation Summary ===");
    eprintln!("Total allocated: {}", total_allocated);
    if total_allocated != total_centroids {
        eprintln!(
            "⚠ Mismatch by {} centroids",
            (total_allocated as i64 - total_centroids as i64).abs()
        );
    } else {
        eprintln!("✓ Budget exactly matched");
    }

    if verbose {
        eprintln!("\nTop 20 allocations:");
        let mut sorted: Vec<_> = allocation.iter().collect();
        sorted.sort_by_key(|(_, k)| std::cmp::Reverse(*k));
        for (i, (token_id, k)) in sorted.iter().take(20).enumerate() {
            let n = token_groups[token_id].len();
            eprintln!(
                "  {}. Token {}: {} vectors → {} centroids ({:.2} vecs/centroid)",
                i + 1,
                token_id,
                n,
                **k,
                n as f64 / **k as f64
            );
        }
    }

    allocation
}
