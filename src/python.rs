//! Python bindings for Tachiom (PyO3).
//!
//! Compiled when the `python` feature is enabled.  Builds a `cdylib` exposing a
//! single `tachiom` module with one `Tachiom` class.

use crate::hnsw::HNSWBuildConfiguration;
use crate::pgc::{EmptyAnchorStrategy, PgcBuilder};
use crate::tac::{TacAllocParams, TacBuilder, TacResult};
use crate::tachiom::{Tachiom, TachiomBuildParams, TachiomInputDataset};
use vectorium::core::index::Index;
use vectorium::vector_encoder::{MultiVecEncoder, VectorEncoder};
use vectorium::{
    Dataset, DenseMultiVectorView, IndexSerializer, MultiVectorDataset, PlainMultiVecQuantizer,
};

use half::f16;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};

// PQ subspace count `M` is a compile-time const generic; the supported variants
// (32, 64) are dispatched at runtime via the `TachiomInner` enum below, selected
// by the `pq_subspaces` kwarg.  Default is 32 (see each constructor's signature).

/// Number of centroids per PQ subspace (vectorium's `KSUB`, fixed at 256 = 2^8).
/// K-means cannot cluster fewer than this many training points into this many
/// centroids, so the corpus must contain at least this many tokens in total.
const PQ_KSUB: usize = 256;

/// Minimum points per centroid — mirrors the TAC allocation strategy constant.
const MIN_PTS_PER_CENTROID: usize = 39;
/// Safety factor applied to the minimum TAC budget when computing the floor.
const TAC_FLOOR_FACTOR: f64 = 1.1;

// ============================================================================
// Auto build-param heuristic
// ============================================================================

/// Resolved TAC build parameters, returned by [`resolve_tac_params`].
struct ResolvedTacParams {
    total_centroids: usize,
    micro_threshold: usize,
    small_threshold: usize,
    hard_floor: usize,
    min_pts_per_centroid: usize,
    /// Saturation cap: adding more centroids beyond this is pointless.
    sat_cap: usize,
    /// Raw power-of-2 formula value, before floor / cap.
    formula: usize,
    /// True when an explicit `total_centroids` value was capped to `sat_cap`.
    was_capped: bool,
}

/// Compute recommended TAC build parameters from the token-ID distribution.
///
/// All `_override` arguments mirror the user-facing kwargs: `None` → auto-compute.
fn resolve_tac_params(
    token_ids: &[u32],
    total_centroids_override: Option<usize>,
    micro_override: Option<usize>,
    small_override: Option<usize>,
    hard_floor_override: Option<usize>,
    min_pts_per_centroid_override: Option<usize>,
) -> ResolvedTacParams {
    let n_tokens = token_ids.len().max(128);

    // Nearest power-of-2 to n_tokens / 128.
    let exp = (n_tokens as f64 / 128.0).log2().round() as u32;
    let formula = 1usize << exp;

    // Thresholds: nearest power-of-2 to n_tokens^(1/4), clamped to [32, 128].
    let micro_exp = (n_tokens as f64).powf(0.25).log2().round() as u32;
    let micro_auto = (1usize << micro_exp).clamp(32, 128);
    let small_auto = micro_auto * 2;
    let micro = micro_override.unwrap_or(micro_auto);
    let small = small_override.unwrap_or(small_auto);
    let hard_floor = hard_floor_override.unwrap_or(4);
    let min_pts_per_centroid = min_pts_per_centroid_override.unwrap_or(MIN_PTS_PER_CENTROID);

    // Token-type frequency histogram.
    let mut freq: HashMap<u32, usize> = HashMap::new();
    for &id in token_ids {
        *freq.entry(id).or_insert(0) += 1;
    }

    let mut n_micro_t: usize = 0;
    let mut n_small_t: usize = 0;
    let mut n_active_t: usize = 0;
    let mut total_active_tokens: usize = 0;
    for &c in freq.values() {
        if c < micro {
            n_micro_t += 1;
        } else if c < small {
            n_small_t += 1;
        } else {
            n_active_t += 1;
            total_active_tokens += c;
        }
    }

    let min_budget = n_micro_t + n_small_t * 2 + n_active_t * hard_floor;
    let sat_cap = n_micro_t + n_small_t * 2 + total_active_tokens / min_pts_per_centroid;

    // TAC floor: enough to run TAC with buffer, but never above sat_cap.
    let tac_floor = ((min_budget as f64 * TAC_FLOOR_FACTOR).ceil() as usize).min(sat_cap);

    let (total_centroids, was_capped) = match total_centroids_override {
        Some(tc) if tc > sat_cap => (sat_cap, true),
        Some(tc) => (tc, false),
        None => (formula.max(tac_floor), false),
    };

    ResolvedTacParams {
        total_centroids,
        micro_threshold: micro,
        small_threshold: small,
        hard_floor,
        min_pts_per_centroid,
        sat_cap,
        formula,
        was_capped,
    }
}

// ============================================================================
// Python-exposed auto_build_params function
// ============================================================================

/// Compute recommended TAC build parameters from a flat token-ID array.
///
/// Returns a dict with keys ``total_centroids``, ``tac_micro_threshold``,
/// ``tac_small_threshold``, ``tac_hard_floor``,
/// ``tac_min_pts_per_centroid``, ``sat_cap``, and ``formula``.
///
/// Any kwarg set to a non-``None`` value overrides the heuristic for that
/// parameter; ``None`` (default) triggers full auto-computation.
///
/// A ``UserWarning`` is emitted when an explicit ``total_centroids`` exceeds
/// the saturation cap.
#[pyfunction]
#[pyo3(signature = (
    token_ids,
    *,
    total_centroids = None,
    tac_micro_threshold = None,
    tac_small_threshold = None,
    tac_hard_floor = None,
    tac_min_pts_per_centroid = None,
))]
fn auto_build_params(
    py: Python<'_>,
    token_ids: PyReadonlyArray1<'_, u32>,
    total_centroids: Option<usize>,
    tac_micro_threshold: Option<usize>,
    tac_small_threshold: Option<usize>,
    tac_hard_floor: Option<usize>,
    tac_min_pts_per_centroid: Option<usize>,
) -> PyResult<Py<PyDict>> {
    let ids = token_ids
        .as_slice()
        .map_err(|_| PyValueError::new_err("token_ids must be C-contiguous"))?;

    let p = resolve_tac_params(
        ids,
        total_centroids,
        tac_micro_threshold,
        tac_small_threshold,
        tac_hard_floor,
        tac_min_pts_per_centroid,
    );

    if p.was_capped {
        let msg = format!(
            "total_centroids={} exceeds the saturation point ({}). \
             Capping to {}. Extra centroids beyond this point would be empty.",
            total_centroids.unwrap(),
            p.sat_cap,
            p.sat_cap,
        );
        let warnings = py.import("warnings")?;
        warnings.call_method1("warn", (msg,))?;
    }

    let dict = PyDict::new(py);
    dict.set_item("total_centroids", p.total_centroids)?;
    dict.set_item("tac_micro_threshold", p.micro_threshold)?;
    dict.set_item("tac_small_threshold", p.small_threshold)?;
    dict.set_item("tac_hard_floor", p.hard_floor)?;
    dict.set_item("tac_min_pts_per_centroid", p.min_pts_per_centroid)?;
    dict.set_item("sat_cap", p.sat_cap)?;
    dict.set_item("formula", p.formula)?;
    Ok(dict.into())
}

// ============================================================================
// Clustering-only timing helpers (benchmark / analysis use)
// ============================================================================
//
// These reproduce the *exact* coarse-clustering call made by the real build
// path (`build_from_arrays` for TAC, `build_with_pgc` for PGC) — same dataset
// construction, same resolved centroid budget, same params — but skip the
// PQ/HNSW stages entirely.  The numpy→Rust copy happens *outside* the timer,
// so the returned `elapsed_s` measures only the clustering routine.  Passing
// `total_centroids=None` reproduces the production centroid budget exactly
// (identical resolver as the index build).

/// Time ONLY the TAC coarse-clustering step on in-memory arrays.
#[pyfunction]
#[pyo3(signature = (
    vectors,
    token_ids,
    doclens,
    *,
    total_centroids = None,
    tac_n_iter = 10,
    tac_micro_threshold = None,
    tac_small_threshold = None,
    max_sample_size = None,
))]
#[allow(clippy::too_many_arguments)]
fn time_tac_clustering(
    py: Python<'_>,
    vectors: PyReadonlyArray2<'_, u16>,
    token_ids: PyReadonlyArray1<'_, u32>,
    doclens: PyReadonlyArray1<'_, i32>,
    total_centroids: Option<usize>,
    tac_n_iter: usize,
    tac_micro_threshold: Option<usize>,
    tac_small_threshold: Option<usize>,
    max_sample_size: Option<usize>,
) -> PyResult<Py<PyDict>> {
    let (dataset, token_ids_vec) = dataset_from_arrays(&vectors, &token_ids, &doclens)?;
    let ids_u32 = token_ids
        .as_slice()
        .map_err(|_| PyValueError::new_err("token_ids must be C-contiguous"))?;
    let resolved = resolve_tac_params(
        ids_u32,
        total_centroids,
        tac_micro_threshold,
        tac_small_threshold,
        None,
        None,
    );
    // Mirror build_from_arrays' TacBuilder construction exactly.
    let tac = TacBuilder::new()
        .n_iter(tac_n_iter)
        .verbose(false)
        .max_sample_size(max_sample_size)
        .alloc_params(TacAllocParams {
            micro_threshold: resolved.micro_threshold,
            small_threshold: resolved.small_threshold,
            hard_floor: resolved.hard_floor,
            min_pts_per_centroid: resolved.min_pts_per_centroid,
        })
        .build();
    let dim = dataset.encoder().input_dim();
    let budget = resolved.total_centroids;
    let n_tokens = dataset.values().len() / dim;

    let (elapsed_s, n_centroids) = py.allow_threads(|| {
        let start = std::time::Instant::now();
        let r: TacResult = tac.train(dataset.values(), dim, &token_ids_vec, budget);
        (start.elapsed().as_secs_f64(), r.n_centroids)
    });

    let dict = PyDict::new(py);
    dict.set_item("clustering", "tac")?;
    dict.set_item("elapsed_s", elapsed_s)?;
    dict.set_item("n_centroids", n_centroids)?;
    dict.set_item("requested_centroids", budget)?;
    dict.set_item("n_tokens", n_tokens)?;
    dict.set_item("dim", dim)?;
    dict.set_item("tac_n_iter", tac_n_iter)?;
    Ok(dict.into())
}

/// Time ONLY the PGC coarse-clustering step on in-memory arrays.
#[pyfunction]
#[pyo3(signature = (
    vectors,
    token_ids,
    doclens,
    *,
    total_centroids = None,
    pgc_n_iter = 10,
    pgc_sample_multiplier = 5,
    pgc_empty_strategy = "resample",
    pgc_iter_hnsw_m = 16,
    pgc_iter_ef_construction = 200,
    pgc_iter_ef_search = 50,
    pgc_iter_lambda = None,
    pgc_assign_topm = 1,
    pgc_assign_temp = 0.1,
    pgc_seed = 42,
))]
#[allow(clippy::too_many_arguments)]
fn time_pgc_clustering(
    py: Python<'_>,
    vectors: PyReadonlyArray2<'_, u16>,
    token_ids: PyReadonlyArray1<'_, u32>,
    doclens: PyReadonlyArray1<'_, i32>,
    total_centroids: Option<usize>,
    pgc_n_iter: usize,
    pgc_sample_multiplier: usize,
    pgc_empty_strategy: &str,
    pgc_iter_hnsw_m: usize,
    pgc_iter_ef_construction: usize,
    pgc_iter_ef_search: usize,
    pgc_iter_lambda: Option<f32>,
    pgc_assign_topm: usize,
    pgc_assign_temp: f32,
    pgc_seed: u64,
) -> PyResult<Py<PyDict>> {
    let empty_strategy = match pgc_empty_strategy {
        "resample" => EmptyAnchorStrategy::Resample,
        "remove" => EmptyAnchorStrategy::Remove,
        "split" => EmptyAnchorStrategy::Split,
        other => {
            return Err(PyValueError::new_err(format!(
                "pgc_empty_strategy must be \"resample\", \"remove\", or \"split\", got {:?}",
                other
            )));
        }
    };
    // token_ids feed the centroid-budget resolver only (PGC ignores token types);
    // this keeps the requested centroid count identical to the TAC/index path.
    let (dataset, _token_ids_vec) = dataset_from_arrays(&vectors, &token_ids, &doclens)?;
    let ids_u32 = token_ids
        .as_slice()
        .map_err(|_| PyValueError::new_err("token_ids must be C-contiguous"))?;
    let resolved = resolve_tac_params(ids_u32, total_centroids, None, None, None, None);

    // Mirror build_with_pgc's PgcBuilder construction exactly.
    let pgc = PgcBuilder::new()
        .n_iter(pgc_n_iter)
        .sample_multiplier(pgc_sample_multiplier)
        .empty_strategy(empty_strategy)
        .iter_hnsw_m(pgc_iter_hnsw_m)
        .iter_ef_construction(pgc_iter_ef_construction)
        .iter_ef_search(pgc_iter_ef_search)
        .iter_lambda(pgc_iter_lambda)
        .assign_topm(pgc_assign_topm)
        .assign_temp(pgc_assign_temp)
        .seed(pgc_seed)
        .verbose(false)
        .build();
    let dim = dataset.encoder().input_dim();
    let n_tokens = dataset.values().len() / dim;
    let n_req = resolved.total_centroids.min(n_tokens);

    let (elapsed_s, n_centroids) = py.allow_threads(|| {
        let start = std::time::Instant::now();
        let r = pgc.cluster(dataset.values(), dim, n_req);
        (start.elapsed().as_secs_f64(), r.n_centroids)
    });

    let dict = PyDict::new(py);
    dict.set_item("clustering", "pgc")?;
    dict.set_item("elapsed_s", elapsed_s)?;
    dict.set_item("n_centroids", n_centroids)?;
    dict.set_item("requested_centroids", n_req)?;
    dict.set_item("n_tokens", n_tokens)?;
    dict.set_item("dim", dim)?;
    dict.set_item("pgc_n_iter", pgc_n_iter)?;
    dict.set_item("pgc_iter_ef_search", pgc_iter_ef_search)?;
    dict.set_item("pgc_iter_lambda", pgc_iter_lambda)?;
    dict.set_item("pgc_assign_topm", pgc_assign_topm)?;
    dict.set_item("pgc_assign_temp", pgc_assign_temp)?;
    dict.set_item("pgc_sample_multiplier", pgc_sample_multiplier)?;
    Ok(dict.into())
}

// ============================================================================
// Standalone clustering: produce (centroids, assignments) for reuse.
// ============================================================================
//
// Decouples the *clustering* step from index building: run TAC/PGC once, save
// the centroids + per-token assignments, then build any number of downstream
// indexes (varying PQ / HNSW / search params) via
// `Tachiom.build_from_arrays_with_centroids` (clustering="external") WITHOUT
// re-clustering — clustering is the expensive part (PGC minutes-to-hours, TAC
// up to hours at high centroid budgets), so caching it saves most of the
// indexing time.  Returns (centroids [n_centroids, dim] f32, assignments
// [n_tokens] u32) in the same token order the build path uses.

/// Run TAC clustering only; return (centroids, assignments).
#[pyfunction]
#[pyo3(signature = (
    vectors, token_ids, doclens, *,
    total_centroids = None, tac_n_iter = 10,
    tac_micro_threshold = None, tac_small_threshold = None, max_sample_size = None,
    verbose = false,
))]
#[allow(clippy::too_many_arguments)]
fn cluster_tac(
    py: Python<'_>,
    vectors: PyReadonlyArray2<'_, u16>,
    token_ids: PyReadonlyArray1<'_, u32>,
    doclens: PyReadonlyArray1<'_, i32>,
    total_centroids: Option<usize>,
    tac_n_iter: usize,
    tac_micro_threshold: Option<usize>,
    tac_small_threshold: Option<usize>,
    max_sample_size: Option<usize>,
    verbose: bool,
) -> PyResult<(Py<PyArray2<f32>>, Py<PyArray1<u32>>)> {
    let (dataset, token_ids_vec) = dataset_from_arrays(&vectors, &token_ids, &doclens)?;
    let ids_u32 = token_ids
        .as_slice()
        .map_err(|_| PyValueError::new_err("token_ids must be C-contiguous"))?;
    let resolved = resolve_tac_params(
        ids_u32, total_centroids, tac_micro_threshold, tac_small_threshold, None, None,
    );
    let tac = TacBuilder::new()
        .n_iter(tac_n_iter)
        .verbose(verbose)
        .max_sample_size(max_sample_size)
        .alloc_params(TacAllocParams {
            micro_threshold: resolved.micro_threshold,
            small_threshold: resolved.small_threshold,
            hard_floor: resolved.hard_floor,
            min_pts_per_centroid: resolved.min_pts_per_centroid,
        })
        .build();
    let dim = dataset.encoder().input_dim();
    let budget = resolved.total_centroids;
    let result =
        py.allow_threads(|| tac.train(dataset.values(), dim, &token_ids_vec, budget));

    let centroids_f32: Vec<f32> = result.centroids.iter().map(|x| x.to_f32()).collect();
    let cen = ndarray::Array2::from_shape_vec((result.n_centroids, dim), centroids_f32)
        .map_err(|e| PyRuntimeError::new_err(format!("centroids reshape: {e}")))?;
    Ok((cen.into_pyarray(py).unbind(), result.assignments.into_pyarray(py).unbind()))
}

/// Run PGC clustering only; return (centroids, assignments).
#[pyfunction]
#[pyo3(signature = (
    vectors, token_ids, doclens, *,
    total_centroids = None, pgc_n_iter = 10, pgc_sample_multiplier = 5,
    pgc_empty_strategy = "resample", pgc_iter_hnsw_m = 16, pgc_iter_ef_construction = 200,
    pgc_iter_ef_search = 50, pgc_iter_lambda = None, pgc_assign_topm = 1,
    pgc_assign_temp = 0.1, pgc_seed = 42, verbose = false,
))]
#[allow(clippy::too_many_arguments)]
fn cluster_pgc(
    py: Python<'_>,
    vectors: PyReadonlyArray2<'_, u16>,
    token_ids: PyReadonlyArray1<'_, u32>,
    doclens: PyReadonlyArray1<'_, i32>,
    total_centroids: Option<usize>,
    pgc_n_iter: usize,
    pgc_sample_multiplier: usize,
    pgc_empty_strategy: &str,
    pgc_iter_hnsw_m: usize,
    pgc_iter_ef_construction: usize,
    pgc_iter_ef_search: usize,
    pgc_iter_lambda: Option<f32>,
    pgc_assign_topm: usize,
    pgc_assign_temp: f32,
    pgc_seed: u64,
    verbose: bool,
) -> PyResult<(Py<PyArray2<f32>>, Py<PyArray1<u32>>)> {
    let empty_strategy = match pgc_empty_strategy {
        "resample" => EmptyAnchorStrategy::Resample,
        "remove" => EmptyAnchorStrategy::Remove,
        "split" => EmptyAnchorStrategy::Split,
        other => {
            return Err(PyValueError::new_err(format!(
                "pgc_empty_strategy must be \"resample\", \"remove\", or \"split\", got {:?}",
                other
            )));
        }
    };
    let (dataset, _token_ids_vec) = dataset_from_arrays(&vectors, &token_ids, &doclens)?;
    let ids_u32 = token_ids
        .as_slice()
        .map_err(|_| PyValueError::new_err("token_ids must be C-contiguous"))?;
    let resolved = resolve_tac_params(ids_u32, total_centroids, None, None, None, None);
    let pgc = PgcBuilder::new()
        .n_iter(pgc_n_iter)
        .sample_multiplier(pgc_sample_multiplier)
        .empty_strategy(empty_strategy)
        .iter_hnsw_m(pgc_iter_hnsw_m)
        .iter_ef_construction(pgc_iter_ef_construction)
        .iter_ef_search(pgc_iter_ef_search)
        .iter_lambda(pgc_iter_lambda)
        .assign_topm(pgc_assign_topm)
        .assign_temp(pgc_assign_temp)
        .seed(pgc_seed)
        .verbose(verbose)
        .build();
    let dim = dataset.encoder().input_dim();
    let n_tokens = dataset.values().len() / dim;
    let n_req = resolved.total_centroids.min(n_tokens);

    let result = py.allow_threads(|| pgc.cluster(dataset.values(), dim, n_req));

    let centroids_f32: Vec<f32> = result.centroids.iter().map(|x| x.to_f32()).collect();
    let cen = ndarray::Array2::from_shape_vec((result.n_centroids, result.dim), centroids_f32)
        .map_err(|e| PyRuntimeError::new_err(format!("centroids reshape: {e}")))?;
    let asgn: Vec<u32> = result.assignments.iter().map(|&x| x as u32).collect();
    Ok((cen.into_pyarray(py).unbind(), asgn.into_pyarray(py).unbind()))
}

// ============================================================================
// Module
// ============================================================================

#[pymodule]
fn tachiom(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTachiom>()?;
    m.add_class::<PyTac>()?;
    m.add_function(wrap_pyfunction!(auto_build_params, m)?)?;
    m.add_function(wrap_pyfunction!(time_tac_clustering, m)?)?;
    m.add_function(wrap_pyfunction!(time_pgc_clustering, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_tac, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_pgc, m)?)?;
    Ok(())
}

// ============================================================================
// PyTachiom class
// ============================================================================

/// Runtime dispatch over the compiled PQ-subspace variants.  `M` is a const
/// generic (compile-time) for speed, so we hold one monomorphized `Tachiom<M>`
/// per supported `pq_subspaces` value and dispatch at runtime.
enum TachiomInner {
    M4(Tachiom<4>),
    M8(Tachiom<8>),
    M16(Tachiom<16>),
    M32(Tachiom<32>),
    M64(Tachiom<64>),
    M128(Tachiom<128>),
}

/// Run `$body` against the inner `Tachiom<M>` regardless of which variant it is.
macro_rules! with_inner {
    ($self:expr, $t:ident => $body:expr) => {
        match &$self.inner {
            TachiomInner::M4($t) => $body,
            TachiomInner::M8($t) => $body,
            TachiomInner::M16($t) => $body,
            TachiomInner::M32($t) => $body,
            TachiomInner::M64($t) => $body,
            TachiomInner::M128($t) => $body,
        }
    };
}

/// Dispatch a build call `$f::<M>($args...)` to the `TachiomInner` variant for the
/// runtime `pq_subspaces` value.  `$f` is a generic free fn over `const M: usize`.
/// `pq_subspaces` must be pre-validated to one of these (see `warn_pq_subspaces`).
macro_rules! dispatch_pq {
    ($pq:expr, $f:ident ( $($arg:expr),* $(,)? )) => {
        match $pq {
            4 => TachiomInner::M4($f::<4>($($arg),*)),
            8 => TachiomInner::M8($f::<8>($($arg),*)),
            16 => TachiomInner::M16($f::<16>($($arg),*)),
            32 => TachiomInner::M32($f::<32>($($arg),*)),
            64 => TachiomInner::M64($f::<64>($($arg),*)),
            128 => TachiomInner::M128($f::<128>($($arg),*)),
            _ => unreachable!("pq_subspaces validated by warn_pq_subspaces"),
        }
    };
}

/// Generic build helpers so `dispatch_pq!` can pick `M` at runtime.
fn build_index_m<const M: usize>(
    dataset: TachiomInputDataset,
    params: &TachiomBuildParams,
) -> Tachiom<M> {
    Tachiom::<M>::build_index(dataset, params)
}

fn build_from_tac_m<const M: usize>(
    centroids: Vec<f16>,
    n_centroids: usize,
    assignments: Vec<usize>,
    dataset: TachiomInputDataset,
    params: &TachiomBuildParams,
) -> Tachiom<M> {
    Tachiom::<M>::build_index_from_tac(centroids, n_centroids, assignments, dataset, params)
}

#[pyclass(name = "Tachiom", module = "tachiom", unsendable)]
pub struct PyTachiom {
    inner: TachiomInner,
}

#[pymethods]
impl PyTachiom {
    // ── Constructors ─────────────────────────────────────────────────────────

    /// Build a Tachiom index from raw .npy inputs (full pipeline: TAC → PQ → HNSW).
    #[classmethod]
    #[pyo3(signature = (
        vectors_path,
        token_ids_path,
        doclens_path,
        *,
        total_centroids = None,
        tac_n_iter = None,
        tac_micro_threshold = None,
        tac_small_threshold = None,
        tac_hard_floor = None,
        tac_min_pts_per_centroid = None,
        pq_sample_size = None,
        pq_n_iter = None,
        normalize = None,
        pq_seed = None,
        hnsw_m = None,
        ef_construction = None,
        pq_subspaces = 32,
        center_dataset = true,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn build(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        vectors_path: &str,
        token_ids_path: &str,
        doclens_path: &str,
        total_centroids: Option<usize>,
        tac_n_iter: Option<usize>,
        tac_micro_threshold: Option<usize>,
        tac_small_threshold: Option<usize>,
        tac_hard_floor: Option<usize>,
        tac_min_pts_per_centroid: Option<usize>,
        pq_sample_size: Option<usize>,
        pq_n_iter: Option<usize>,
        normalize: Option<bool>,
        pq_seed: Option<u64>,
        hnsw_m: Option<usize>,
        ef_construction: Option<usize>,
        pq_subspaces: usize,
        center_dataset: bool,
    ) -> PyResult<Self> {
        warn_pq_subspaces(py, pq_subspaces)?;
        let tac_n_iter = tac_n_iter.unwrap_or(10);
        let pq_sample_size = pq_sample_size.unwrap_or(10_000_000);
        let pq_n_iter = pq_n_iter.unwrap_or(10);
        let normalize = normalize.unwrap_or(true);
        let pq_seed = pq_seed.unwrap_or(42);
        let hnsw_m = hnsw_m.unwrap_or(32);
        let ef_construction = ef_construction.unwrap_or(1500);
        let (dataset, token_ids) = load_input_dataset(vectors_path, token_ids_path, doclens_path)?;
        require_min_tokens(token_ids.len())?;

        let token_ids_u32: Vec<u32> = token_ids.iter().map(|&x| x as u32).collect();
        let resolved = resolve_tac_params(
            &token_ids_u32,
            total_centroids,
            tac_micro_threshold,
            tac_small_threshold,
            tac_hard_floor,
            tac_min_pts_per_centroid,
        );
        if resolved.was_capped {
            warn_saturation_cap(py, total_centroids.unwrap(), resolved.sat_cap)?;
        }

        let params = TachiomBuildParams {
            token_ids,
            total_centroids: resolved.total_centroids,
            tac_n_iter,
            tac_alloc_params: TacAllocParams {
                micro_threshold: resolved.micro_threshold,
                small_threshold: resolved.small_threshold,
                hard_floor: resolved.hard_floor,
                min_pts_per_centroid: resolved.min_pts_per_centroid,
            },
            pq_sample_size,
            pq_n_iter,
            normalize,
            pq_seed: Some(pq_seed),
            hnsw_params: HNSWBuildConfiguration::default()
                .with_num_neighbors(hnsw_m)
                .with_ef_construction(ef_construction),
            center_dataset,
        };

        let inner = py.allow_threads(|| dispatch_pq!(pq_subspaces, build_index_m(dataset, &params)));
        Ok(PyTachiom { inner })
    }

    /// Build a Tachiom index from in-memory numpy arrays (full pipeline: TAC → PQ → HNSW).
    ///
    /// Equivalent to `build()` but accepts numpy arrays instead of file paths.
    /// Supports memory-mapped arrays (`np.load(..., mmap_mode='r')`) to minimise RAM
    /// usage during construction — data is read from the buffer with a single copy.
    ///
    /// `vectors` must be f16 and C-contiguous.  Cast with `.astype(np.float16)` if needed.
    #[classmethod]
    #[pyo3(signature = (
        vectors,
        token_ids,
        doclens,
        *,
        total_centroids = None,
        tac_n_iter = None,
        tac_micro_threshold = None,
        tac_small_threshold = None,
        tac_hard_floor = None,
        tac_min_pts_per_centroid = None,
        pq_sample_size = None,
        pq_n_iter = None,
        normalize = None,
        pq_seed = None,
        hnsw_m = None,
        ef_construction = None,
        pq_subspaces = 32,
        center_dataset = true,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn build_from_arrays(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        vectors: PyReadonlyArray2<'_, u16>,
        token_ids: PyReadonlyArray1<'_, u32>,
        doclens: PyReadonlyArray1<'_, i32>,
        total_centroids: Option<usize>,
        tac_n_iter: Option<usize>,
        tac_micro_threshold: Option<usize>,
        tac_small_threshold: Option<usize>,
        tac_hard_floor: Option<usize>,
        tac_min_pts_per_centroid: Option<usize>,
        pq_sample_size: Option<usize>,
        pq_n_iter: Option<usize>,
        normalize: Option<bool>,
        pq_seed: Option<u64>,
        hnsw_m: Option<usize>,
        ef_construction: Option<usize>,
        pq_subspaces: usize,
        center_dataset: bool,
    ) -> PyResult<Self> {
        warn_pq_subspaces(py, pq_subspaces)?;
        let tac_n_iter = tac_n_iter.unwrap_or(10);
        let pq_sample_size = pq_sample_size.unwrap_or(10_000_000);
        let pq_n_iter = pq_n_iter.unwrap_or(10);
        let normalize = normalize.unwrap_or(true);
        let pq_seed = pq_seed.unwrap_or(42);
        let hnsw_m = hnsw_m.unwrap_or(32);
        let ef_construction = ef_construction.unwrap_or(1500);
        require_min_tokens(token_ids.len())?;
        let (dataset, token_ids_vec) = dataset_from_arrays(&vectors, &token_ids, &doclens)?;

        let ids_u32 = token_ids
            .as_slice()
            .map_err(|_| PyValueError::new_err("token_ids must be C-contiguous"))?;
        let resolved = resolve_tac_params(
            ids_u32,
            total_centroids,
            tac_micro_threshold,
            tac_small_threshold,
            tac_hard_floor,
            tac_min_pts_per_centroid,
        );
        if resolved.was_capped {
            warn_saturation_cap(py, total_centroids.unwrap(), resolved.sat_cap)?;
        }

        let params = TachiomBuildParams {
            token_ids: token_ids_vec,
            total_centroids: resolved.total_centroids,
            tac_n_iter,
            tac_alloc_params: TacAllocParams {
                micro_threshold: resolved.micro_threshold,
                small_threshold: resolved.small_threshold,
                hard_floor: resolved.hard_floor,
                min_pts_per_centroid: resolved.min_pts_per_centroid,
            },
            pq_sample_size,
            pq_n_iter,
            normalize,
            pq_seed: Some(pq_seed),
            hnsw_params: HNSWBuildConfiguration::default()
                .with_num_neighbors(hnsw_m)
                .with_ef_construction(ef_construction),
            center_dataset,
        };

        let inner = py.allow_threads(|| dispatch_pq!(pq_subspaces, build_index_m(dataset, &params)));
        Ok(PyTachiom { inner })
    }

    /// Build a Tachiom index using Proximity Graph Clustering (PGC) instead of TAC.
    ///
    /// PGC is token-type-agnostic: it works directly on raw embeddings without
    /// needing vocabulary IDs.  All standard Tachiom params (PQ, HNSW) are forwarded
    /// unchanged; only the coarse-centroid step is replaced.
    #[classmethod]
    #[pyo3(signature = (
        vectors,
        token_ids,
        doclens,
        *,
        total_centroids = None,
        pgc_n_iter = 10,
        pgc_sample_multiplier = 5,
        pgc_empty_strategy = "resample",
        pgc_iter_hnsw_m = 16,
        pgc_iter_ef_construction = 200,
        pgc_iter_ef_search = 50,
        pgc_iter_lambda = None,
        pgc_assign_topm = 1,
        pgc_assign_temp = 0.1,
        pgc_seed = 42,
        pq_sample_size = None,
        pq_n_iter = None,
        normalize = None,
        pq_seed = None,
        hnsw_m = None,
        ef_construction = None,
        pq_subspaces = 32,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn build_with_pgc(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        vectors: PyReadonlyArray2<'_, u16>,
        token_ids: PyReadonlyArray1<'_, u32>,
        doclens: PyReadonlyArray1<'_, i32>,
        total_centroids: Option<usize>,
        pgc_n_iter: usize,
        pgc_sample_multiplier: usize,
        pgc_empty_strategy: &str,
        pgc_iter_hnsw_m: usize,
        pgc_iter_ef_construction: usize,
        pgc_iter_ef_search: usize,
        pgc_iter_lambda: Option<f32>,
        pgc_assign_topm: usize,
        pgc_assign_temp: f32,
        pgc_seed: u64,
        pq_sample_size: Option<usize>,
        pq_n_iter: Option<usize>,
        normalize: Option<bool>,
        pq_seed: Option<u64>,
        hnsw_m: Option<usize>,
        ef_construction: Option<usize>,
        pq_subspaces: usize,
    ) -> PyResult<Self> {
        warn_pq_subspaces(py, pq_subspaces)?;
        require_min_tokens(token_ids.len())?;
        let pq_sample_size = pq_sample_size.unwrap_or(10_000_000);
        let pq_n_iter = pq_n_iter.unwrap_or(10);
        let normalize = normalize.unwrap_or(true);
        let pq_seed = pq_seed.unwrap_or(42);
        let hnsw_m = hnsw_m.unwrap_or(32);
        let ef_construction = ef_construction.unwrap_or(1500);

        let empty_strategy = match pgc_empty_strategy {
            "resample" => EmptyAnchorStrategy::Resample,
            "remove" => EmptyAnchorStrategy::Remove,
            "split" => EmptyAnchorStrategy::Split,
            other => {
                return Err(PyValueError::new_err(format!(
                    "pgc_empty_strategy must be \"resample\", \"remove\", or \"split\", got {:?}",
                    other
                )));
            }
        };

        let (dataset, token_ids_vec) = dataset_from_arrays(&vectors, &token_ids, &doclens)?;
        let ids_u32 = token_ids
            .as_slice()
            .map_err(|_| PyValueError::new_err("token_ids must be C-contiguous"))?;
        let resolved = resolve_tac_params(ids_u32, total_centroids, None, None, None, None);
        if resolved.was_capped {
            warn_saturation_cap(py, total_centroids.unwrap(), resolved.sat_cap)?;
        }

        let pgc = PgcBuilder::new()
            .n_iter(pgc_n_iter)
            .sample_multiplier(pgc_sample_multiplier)
            .empty_strategy(empty_strategy)
            .iter_hnsw_m(pgc_iter_hnsw_m)
            .iter_ef_construction(pgc_iter_ef_construction)
            .iter_ef_search(pgc_iter_ef_search)
            .iter_lambda(pgc_iter_lambda)
            .assign_topm(pgc_assign_topm)
            .assign_temp(pgc_assign_temp)
            .seed(pgc_seed)
            .verbose(true)
            .build();

        let params = TachiomBuildParams {
            token_ids: token_ids_vec,
            total_centroids: resolved.total_centroids,
            tac_n_iter: 0,
            tac_alloc_params: Default::default(),
            pq_sample_size,
            pq_n_iter,
            normalize,
            pq_seed: Some(pq_seed),
            hnsw_params: HNSWBuildConfiguration::default()
                .with_num_neighbors(hnsw_m)
                .with_ef_construction(ef_construction),
            center_dataset: false,
        };

        // Run PGC outside the GIL, then hand centroids+assignments to build_index_from_tac.
        let inner = py.allow_threads(|| {
            let dim = dataset.encoder().input_dim();
            let n_tokens = dataset.values().len() / dim;
            let n_req = resolved.total_centroids.min(n_tokens);

            // Borrow dataset.values() for PGC; the borrow ends when cluster() returns,
            // so dataset can then be moved into build_index_from_tac without a clone.
            let pgc_result = pgc.cluster(dataset.values(), dim, n_req);
            let (c, nc, a) = (pgc_result.centroids, pgc_result.n_centroids, pgc_result.assignments);
            dispatch_pq!(pq_subspaces, build_from_tac_m(c, nc, a, dataset, &params))
        });
        Ok(PyTachiom { inner })
    }

    /// Build a Tachiom index from in-memory arrays using EXTERNALLY-computed
    /// coarse centroids and per-token assignments (e.g. from GPU k-means).
    ///
    /// Decouples clustering from indexing: the caller supplies `centroids`
    /// (`[K, dim]` f32) and `assignments` (`[n_tokens]` u32, token i -> centroid),
    /// and this runs the identical downstream (PQ -> HNSW -> IVF) as
    /// `build_with_pgc` / `build_from_arrays`.  Accepts the same flat shard
    /// buffers the Python loader already produces, so no corpus concatenation
    /// to disk is needed.  `center_dataset` is false (caller owns preprocessing).
    #[classmethod]
    #[pyo3(signature = (
        vectors,
        token_ids,
        doclens,
        centroids,
        assignments,
        *,
        pq_sample_size = None,
        pq_n_iter = None,
        normalize = None,
        pq_seed = None,
        hnsw_m = None,
        ef_construction = None,
        pq_subspaces = 32,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn build_from_arrays_with_centroids(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        vectors: PyReadonlyArray2<'_, u16>,
        token_ids: PyReadonlyArray1<'_, u32>,
        doclens: PyReadonlyArray1<'_, i32>,
        centroids: PyReadonlyArray2<'_, f32>,
        assignments: PyReadonlyArray1<'_, u32>,
        pq_sample_size: Option<usize>,
        pq_n_iter: Option<usize>,
        normalize: Option<bool>,
        pq_seed: Option<u64>,
        hnsw_m: Option<usize>,
        ef_construction: Option<usize>,
        pq_subspaces: usize,
    ) -> PyResult<Self> {
        warn_pq_subspaces(py, pq_subspaces)?;
        require_min_tokens(token_ids.len())?;
        let pq_sample_size = pq_sample_size.unwrap_or(10_000_000);
        let pq_n_iter = pq_n_iter.unwrap_or(10);
        let normalize = normalize.unwrap_or(true);
        let pq_seed = pq_seed.unwrap_or(42);
        let hnsw_m = hnsw_m.unwrap_or(32);
        let ef_construction = ef_construction.unwrap_or(1500);

        let (dataset, token_ids_vec) = dataset_from_arrays(&vectors, &token_ids, &doclens)?;
        let n_tokens = token_ids_vec.len();

        let cshape = centroids.shape();
        if cshape.len() != 2 {
            return Err(PyValueError::new_err("centroids must be a 2D array [K, dim]"));
        }
        let n_centroids = cshape[0];
        let centroids_f16: Vec<f16> = centroids
            .as_slice()
            .map_err(|_| PyValueError::new_err("centroids must be C-contiguous"))?
            .iter()
            .map(|&x| f16::from_f32(x))
            .collect();

        let asgn_slice = assignments
            .as_slice()
            .map_err(|_| PyValueError::new_err("assignments must be C-contiguous"))?;
        if asgn_slice.len() != n_tokens {
            return Err(PyValueError::new_err(format!(
                "assignments length ({}) != n_tokens ({})",
                asgn_slice.len(),
                n_tokens
            )));
        }
        let assignments_usize: Vec<usize> = asgn_slice.iter().map(|&x| x as usize).collect();

        let params = TachiomBuildParams {
            token_ids: token_ids_vec,
            total_centroids: n_centroids,
            tac_n_iter: 0,
            tac_alloc_params: Default::default(),
            pq_sample_size,
            pq_n_iter,
            normalize,
            pq_seed: Some(pq_seed),
            hnsw_params: HNSWBuildConfiguration::default()
                .with_num_neighbors(hnsw_m)
                .with_ef_construction(ef_construction),
            center_dataset: false,
        };

        let inner = py.allow_threads(|| {
            dispatch_pq!(pq_subspaces, build_from_tac_m(centroids_f16, n_centroids, assignments_usize, dataset, &params))
        });
        Ok(PyTachiom { inner })
    }

    /// Build a Tachiom index using pre-computed coarse centroids and assignments.
    /// Skips the TAC step.  Useful for isolating retrieval differences between
    /// clustering and residual encoding.
    #[classmethod]
    #[pyo3(signature = (
        vectors_path,
        token_ids_path,
        doclens_path,
        centroids_path,
        assignments_path,
        *,
        pq_sample_size = None,
        pq_n_iter = None,
        normalize = None,
        pq_seed = None,
        hnsw_m = None,
        ef_construction = None,
        pq_subspaces = 32,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn build_from_tac(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        vectors_path: &str,
        token_ids_path: &str,
        doclens_path: &str,
        centroids_path: &str,
        assignments_path: &str,
        pq_sample_size: Option<usize>,
        pq_n_iter: Option<usize>,
        normalize: Option<bool>,
        pq_seed: Option<u64>,
        hnsw_m: Option<usize>,
        ef_construction: Option<usize>,
        pq_subspaces: usize,
    ) -> PyResult<Self> {
        warn_pq_subspaces(py, pq_subspaces)?;
        let pq_sample_size = pq_sample_size.unwrap_or(10_000_000);
        let pq_n_iter = pq_n_iter.unwrap_or(10);
        let normalize = normalize.unwrap_or(true);
        let pq_seed = pq_seed.unwrap_or(42);
        let hnsw_m = hnsw_m.unwrap_or(32);
        let ef_construction = ef_construction.unwrap_or(1500);
        let (dataset, token_ids) = load_input_dataset(vectors_path, token_ids_path, doclens_path)?;
        let n_tokens = token_ids.len();
        require_min_tokens(n_tokens)?;

        let (centroids_f32, n_centroids, _dim) = read_f32_2d_npy(centroids_path)?;
        let centroids_f16: Vec<f16> = centroids_f32.iter().map(|&x| f16::from_f32(x)).collect();

        let assignments = read_assignments_npy(assignments_path, n_tokens)?;

        let params = TachiomBuildParams {
            token_ids,
            total_centroids: n_centroids, // unused by build_index_from_tac, required by struct
            tac_n_iter: 0,                // unused
            tac_alloc_params: Default::default(),
            pq_sample_size,
            pq_n_iter,
            normalize,
            pq_seed: Some(pq_seed),
            hnsw_params: HNSWBuildConfiguration::default()
                .with_num_neighbors(hnsw_m)
                .with_ef_construction(ef_construction),
            center_dataset: false, // externally-supplied centroids; caller controls preprocessing
        };

        let inner = py.allow_threads(|| {
            dispatch_pq!(pq_subspaces, build_from_tac_m(centroids_f16, n_centroids, assignments, dataset, &params))
        });
        Ok(PyTachiom { inner })
    }

    /// Load a previously-saved Tachiom index from disk.
    ///
    /// `pq_subspaces` must match the M the index was built with (the on-disk PQ
    /// format is M-specific); the caller is responsible for tracking it.
    #[classmethod]
    #[pyo3(signature = (path, *, pq_subspaces = 32))]
    fn load(_cls: &Bound<'_, PyType>, py: Python<'_>, path: &str, pq_subspaces: usize) -> PyResult<Self> {
        warn_pq_subspaces(py, pq_subspaces)?;
        let path_owned = path.to_owned();
        let inner = py
            .allow_threads(|| match pq_subspaces {
                4 => Tachiom::<4>::load_index(&path_owned).map(TachiomInner::M4),
                8 => Tachiom::<8>::load_index(&path_owned).map(TachiomInner::M8),
                16 => Tachiom::<16>::load_index(&path_owned).map(TachiomInner::M16),
                64 => Tachiom::<64>::load_index(&path_owned).map(TachiomInner::M64),
                128 => Tachiom::<128>::load_index(&path_owned).map(TachiomInner::M128),
                _ => Tachiom::<32>::load_index(&path_owned).map(TachiomInner::M32),
            })
            .map_err(|e| PyIOError::new_err(format!("Failed to load index: {e:?}")))?;
        Ok(PyTachiom { inner })
    }

    // ── Persistence ──────────────────────────────────────────────────────────

    /// Save the index to disk (bincode-flavoured serialization).
    fn save(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        let path_owned = path.to_owned();
        py.allow_threads(|| with_inner!(self, t => t.save_index(&path_owned)))
            .map_err(|e| PyIOError::new_err(format!("Failed to save index: {e:?}")))?;
        Ok(())
    }

    // ── Search ───────────────────────────────────────────────────────────────

    /// Search a single multivector query.
    ///
    /// `query` must be a 2D C-contiguous f32 array of shape `(n_tokens, dim)`.
    /// Returns `(scores, doc_ids)` as 1D ndarrays of length `k` (sentinel-padded
    /// when fewer than `k` results are produced).
    #[pyo3(signature = (
        query, k = 10, *,
        k_centroids = 20,
        k_docs_to_score = 500,
        ef_search = None,
        alpha = Some(0.45),
        beta = None,
        lambda_ = None,
        impute_missing = false,
        gap_relative = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn search<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray2<'py, f32>,
        k: usize,
        k_centroids: usize,
        k_docs_to_score: usize,
        ef_search: Option<usize>,
        alpha: Option<f32>,
        beta: Option<usize>,
        lambda_: Option<f32>,
        impute_missing: bool,
        gap_relative: bool,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<u32>>)> {
        let ef_search = ef_search.unwrap_or_else(|| ((k_centroids as f64) * 1.5).round() as usize);
        let dim = with_inner!(self, t => t.residuals.encoder().input_dim());
        let q_slice = require_contiguous_2d(&query, dim, "query")?;
        let q_view = DenseMultiVectorView::new(q_slice, dim);

        let result: Vec<(f32, u32)> = py.allow_threads(|| {
            with_inner!(self, t => t.search(
                q_view,
                k,
                k_centroids,
                k_docs_to_score,
                ef_search,
                alpha,
                beta,
                lambda_,
                impute_missing,
                gap_relative,
            ))
        });

        let (scores, doc_ids) = pad_result(result, k);
        Ok((
            scores.into_pyarray(py).unbind(),
            doc_ids.into_pyarray(py).unbind(),
        ))
    }

    /// Search a batch of multivector queries.
    ///
    /// `tokens` is a flat 2D C-contiguous f32 array of shape `(total_tokens, dim)` with
    /// all query token vectors concatenated in query order.  `n_queries` states the number
    /// of queries and is always required.
    ///
    /// **Uniform mode** (`offsets = None`): all queries are assumed to have the same token
    /// count.  `total_tokens` must divide evenly by `n_queries`; the per-query stride is
    /// `total_tokens / n_queries`.
    ///
    /// **Ragged mode** (`offsets` provided): a 1D u64 array of length `n_queries + 1`.
    /// `offsets[i]..offsets[i+1]` is the row range in `tokens` for query `i`.
    /// `n_queries` is validated against `len(offsets) - 1`.
    ///
    /// Returns `(scores, doc_ids)` — both 2D ndarrays of shape `(n_queries, k)`,
    /// sentinel-padded when fewer than `k` results are produced for a given query.
    ///
    /// `num_threads`:
    /// - `0` — rayon's default thread pool (typically all cores).
    /// - `1` — serial loop (reproducible single-thread benchmarks).
    /// - `n` — temporary rayon pool of size `n` for this call.
    #[pyo3(signature = (
        tokens, n_queries, k = 10, *,
        offsets = None,
        num_threads = 0,
        k_centroids = 20,
        k_docs_to_score = 500,
        ef_search = None,
        alpha = Some(0.45),
        beta = None,
        lambda_ = None,
        impute_missing = false,
        gap_relative = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn batch_search<'py>(
        &self,
        py: Python<'py>,
        tokens: PyReadonlyArray2<'py, f32>,
        n_queries: usize,
        k: usize,
        offsets: Option<PyReadonlyArray1<'py, u64>>,
        num_threads: usize,
        k_centroids: usize,
        k_docs_to_score: usize,
        ef_search: Option<usize>,
        alpha: Option<f32>,
        beta: Option<usize>,
        lambda_: Option<f32>,
        impute_missing: bool,
        gap_relative: bool,
    ) -> PyResult<(Py<PyArray2<f32>>, Py<PyArray2<u32>>)> {
        let ef_search = ef_search.unwrap_or_else(|| ((k_centroids as f64) * 1.5).round() as usize);
        let dim = with_inner!(self, t => t.residuals.encoder().input_dim());

        let tokens_shape = tokens.shape();
        if tokens_shape.len() != 2 || tokens_shape[1] != dim {
            return Err(PyValueError::new_err(format!(
                "tokens must have shape (total_tokens, {dim}); got {tokens_shape:?}"
            )));
        }
        if !tokens.is_c_contiguous() {
            return Err(PyValueError::new_err(
                "tokens must be C-contiguous; call np.ascontiguousarray(tokens) first",
            ));
        }
        if n_queries == 0 {
            return Err(PyValueError::new_err("n_queries must be > 0"));
        }
        let total_tokens = tokens_shape[0];
        let tokens_slice = tokens
            .as_slice()
            .map_err(|_| PyValueError::new_err("tokens could not be exposed as a slice"))?;

        let views: Vec<DenseMultiVectorView<f32>> = match offsets {
            None => {
                if total_tokens % n_queries != 0 {
                    return Err(PyValueError::new_err(format!(
                        "total_tokens={total_tokens} is not divisible by n_queries={n_queries}; \
                         provide an offsets array for variable-length queries"
                    )));
                }
                let stride = total_tokens / n_queries * dim;
                (0..n_queries)
                    .map(|i| {
                        DenseMultiVectorView::new(&tokens_slice[i * stride..(i + 1) * stride], dim)
                    })
                    .collect()
            }
            Some(off) => {
                if !off.is_c_contiguous() {
                    return Err(PyValueError::new_err("offsets must be C-contiguous"));
                }
                let off_slice = off.as_slice().map_err(|_| {
                    PyValueError::new_err("offsets could not be exposed as a slice")
                })?;
                if off_slice.len() != n_queries + 1 {
                    return Err(PyValueError::new_err(format!(
                        "offsets length ({}) must equal n_queries + 1 ({})",
                        off_slice.len(),
                        n_queries + 1
                    )));
                }
                if *off_slice.last().unwrap() as usize != total_tokens {
                    return Err(PyValueError::new_err(format!(
                        "offsets[-1]={} != total_tokens={total_tokens}",
                        off_slice.last().unwrap()
                    )));
                }
                let mut views = Vec::with_capacity(n_queries);
                for i in 0..n_queries {
                    let start = off_slice[i] as usize;
                    let end = off_slice[i + 1] as usize;
                    if start > end {
                        return Err(PyValueError::new_err(format!(
                            "offsets[{i}]={start} > offsets[{}]={end}: must be non-decreasing",
                            i + 1
                        )));
                    }
                    views.push(DenseMultiVectorView::new(
                        &tokens_slice[start * dim..end * dim],
                        dim,
                    ));
                }
                views
            }
        };

        let results: Vec<Vec<(f32, u32)>> = py.allow_threads(|| {
            with_inner!(self, t => t.batch_search(
                &views,
                k,
                k_centroids,
                k_docs_to_score,
                ef_search,
                alpha,
                beta,
                lambda_,
                impute_missing,
                gap_relative,
                num_threads,
            ))
        });

        let (scores_arr, doc_ids_arr) = pad_results_batch(results, n_queries, k);
        Ok((
            scores_arr.into_pyarray(py).unbind(),
            doc_ids_arr.into_pyarray(py).unbind(),
        ))
    }

    // ── Inspection ───────────────────────────────────────────────────────────

    /// Number of indexed documents.
    #[getter]
    fn len(&self) -> usize {
        with_inner!(self, t => t.n_elements())
    }

    /// Token vector dimensionality (before quantization).
    #[getter]
    fn dim(&self) -> usize {
        with_inner!(self, t => t.dim())
    }

    /// Total number of tokens across all documents.
    #[getter]
    fn n_tokens(&self) -> usize {
        with_inner!(self, t => {
            let dim = t.residuals.encoder().output_dim();
            if dim == 0 {
                return 0;
            }
            t.residuals.offsets().last().map(|&end| end / dim).unwrap_or(0)
        })
    }

    /// Number of coarse centroids in the IVF.
    #[getter]
    fn n_centroids(&self) -> usize {
        with_inner!(self, t => t.centroids.n_elements())
    }

    /// Print a per-component size breakdown of the index.
    fn print_space_usage(&self) {
        let (ch, il, off, res) = with_inner!(self, t => t.space_usage_components());
        let total = ch + il + off + res;
        let gb = |b: usize| b as f64 / 1_073_741_824.0;
        let pct = |b: usize| 100.0 * b as f64 / total as f64;
        println!("Index space usage:");
        println!(
            "  {:<20} {:6.2} GB  ({:5.1}%)",
            "centroids_hnsw",
            gb(ch),
            pct(ch)
        );
        println!(
            "  {:<20} {:6.2} GB  ({:5.1}%)",
            "inverted_lists",
            gb(il),
            pct(il)
        );
        println!(
            "  {:<20} {:6.2} GB  ({:5.1}%)",
            "offsets",
            gb(off),
            pct(off)
        );
        println!(
            "  {:<20} {:6.2} GB  ({:5.1}%)",
            "residuals",
            gb(res),
            pct(res)
        );
        println!("  {}", "─".repeat(38));
        println!("  {:<20} {:6.2} GB", "total", gb(total));
    }

    /// Reconstruct approximate token embeddings for a single document.
    ///
    /// Returns a 2D f32 array of shape `(n_tokens, dim)` obtained by decoding the
    /// stored PQ codes: `approx = coarse_centroid + norm * PQ_residual`.
    /// The result is approximate due to PQ lossy compression.
    ///
    /// Raises `ValueError` if `doc_id` is out of range.
    fn get_document_embeddings<'py>(
        &self,
        py: Python<'py>,
        doc_id: u32,
    ) -> PyResult<Py<PyArray2<f32>>> {
        let array = with_inner!(self, t => {
            let n_docs = t.residuals.len();
            if doc_id as usize >= n_docs {
                return Err(PyValueError::new_err(format!(
                    "doc_id {doc_id} is out of range (index has {n_docs} documents)"
                )));
            }
            let encoded = t.residuals.get(doc_id as u64);
            let decoded = t.residuals.encoder().decode_vector(encoded);
            let n_tokens = decoded.num_vecs();
            let dim = decoded.dim();
            let mut array = ndarray::Array2::from_shape_vec((n_tokens, dim), decoded.values().to_vec())
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            if let Some(mean) = &t.dataset_mean {
                for mut row in array.rows_mut() {
                    for (v, &m) in row.iter_mut().zip(mean.iter()) {
                        *v += m;
                    }
                }
            }
            array
        });
        Ok(array.into_pyarray(py).unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "<Tachiom: {} docs, dim={}, {} centroids>",
            self.len(),
            self.dim(),
            self.n_centroids()
        )
    }
}

// ============================================================================
// PyTac class
// ============================================================================

#[pyclass(name = "Tac", module = "tachiom", unsendable)]
pub struct PyTac {
    budget: Option<usize>,
    n_iter: usize,
    verbose: bool,
    max_sample_size: Option<usize>,
    micro_threshold: Option<usize>,
    small_threshold: Option<usize>,
    result: Option<TacResult>,
}

impl PyTac {
    fn require_trained(&self) -> PyResult<&TacResult> {
        self.result.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Tac has not been trained yet; call train() first")
        })
    }
}

#[pymethods]
impl PyTac {
    #[new]
    #[pyo3(signature = (n_centroids = None, *, n_iter = None, verbose = false, max_sample_size = None, micro_threshold = None, small_threshold = None))]
    fn new(
        n_centroids: Option<usize>,
        n_iter: Option<usize>,
        verbose: bool,
        max_sample_size: Option<usize>,
        micro_threshold: Option<usize>,
        small_threshold: Option<usize>,
    ) -> Self {
        PyTac {
            budget: n_centroids,
            n_iter: n_iter.unwrap_or(10),
            verbose,
            max_sample_size,
            micro_threshold,
            small_threshold,
            result: None,
        }
    }

    /// Run Token-Aware Clustering on the given .npy inputs.
    ///
    /// May be called multiple times; each call overwrites the previous result.
    fn train(&mut self, py: Python<'_>, vectors_path: &str, token_ids_path: &str) -> PyResult<()> {
        let (flat_f16, dim) = read_f16_npy(vectors_path)?;
        let token_ids = read_token_ids_npy(token_ids_path)?;
        let n_tokens = flat_f16.len() / dim;
        if token_ids.len() != n_tokens {
            return Err(PyValueError::new_err(format!(
                "token_ids length ({}) != n_tokens ({})",
                token_ids.len(),
                n_tokens
            )));
        }

        let token_ids_u32: Vec<u32> = token_ids.iter().map(|&x| x as u32).collect();
        let budget = self.budget.unwrap_or_else(|| {
            resolve_tac_params(
                &token_ids_u32,
                None,
                self.micro_threshold,
                self.small_threshold,
                None,
                None,
            )
            .total_centroids
        });

        let mut alloc_params = TacAllocParams::default();
        if let Some(v) = self.micro_threshold {
            alloc_params.micro_threshold = v;
        }
        if let Some(v) = self.small_threshold {
            alloc_params.small_threshold = v;
        }
        let tac = TacBuilder::new()
            .n_iter(self.n_iter)
            .verbose(self.verbose)
            .max_sample_size(self.max_sample_size)
            .alloc_params(alloc_params)
            .build();

        let result = py.allow_threads(|| tac.train(&flat_f16, dim, &token_ids, budget));
        self.result = Some(result);
        Ok(())
    }

    // ── Properties (available after train()) ────────────────────────────────

    /// Coarse centroids as f32, shape `[n_centroids, dim]`.
    #[getter]
    fn centroids<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray2<f32>>> {
        let r = self.require_trained()?;
        let f32_vec: Vec<f32> = r.centroids.iter().map(|x| x.to_f32()).collect();
        let arr = ndarray::Array2::from_shape_vec((r.n_centroids, r.dim), f32_vec)
            .map_err(|e| PyRuntimeError::new_err(format!("centroids reshape: {e}")))?;
        Ok(arr.into_pyarray(py).unbind())
    }

    /// Coarse centroids as f16 (raw), shape `[n_centroids, dim]`.
    #[getter]
    fn centroids_f16<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray2<f16>>> {
        let r = self.require_trained()?;
        let arr = ndarray::Array2::from_shape_vec((r.n_centroids, r.dim), r.centroids.clone())
            .map_err(|e| PyRuntimeError::new_err(format!("centroids_f16 reshape: {e}")))?;
        Ok(arr.into_pyarray(py).unbind())
    }

    /// Per-token centroid assignment, shape `[n_tokens]`.
    #[getter]
    fn assignments<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<u32>>> {
        let r = self.require_trained()?;
        Ok(r.assignments.clone().into_pyarray(py).unbind())
    }

    /// Actual number of centroids produced (equals the budget after reconciliation).
    #[getter]
    fn n_centroids(&self) -> PyResult<usize> {
        Ok(self.require_trained()?.n_centroids)
    }

    /// Token-vector dimensionality.
    #[getter]
    fn dim(&self) -> PyResult<usize> {
        Ok(self.require_trained()?.dim)
    }

    fn __repr__(&self) -> String {
        match &self.result {
            Some(r) => format!("<Tac: n_centroids={}, dim={}>", r.n_centroids, r.dim),
            None => match self.budget {
                Some(b) => format!("<Tac: budget={b}, not yet trained>"),
                None => "<Tac: budget=auto, not yet trained>".to_string(),
            },
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn warn_saturation_cap(py: Python<'_>, requested: usize, sat_cap: usize) -> PyResult<()> {
    let msg = format!(
        "total_centroids={requested} exceeds the saturation point ({sat_cap}). \
         Capping to {sat_cap}. Extra centroids beyond this point would be empty."
    );
    let warnings = py.import("warnings")?;
    warnings.call_method1("warn", (msg,))?;
    Ok(())
}

/// Validate that the corpus has enough tokens for PQ codebook training.
///
/// PQ trains `PQ_KSUB` (256) centroids per subspace via k-means, which requires
/// at least that many training points. Smaller corpora cause vectorium's
/// k-means to panic (and abort the process, since panics cannot cross the
/// PyO3 boundary), so we raise a clear `ValueError` instead.
fn require_min_tokens(n_tokens: usize) -> PyResult<()> {
    if n_tokens < PQ_KSUB {
        return Err(PyValueError::new_err(format!(
            "Corpus has only {n_tokens} token(s), but PQ codebook training \
             requires at least {PQ_KSUB} tokens. \
             Please provide a larger corpus."
        )));
    }
    Ok(())
}

/// Validate that `pq_subspaces` is one of the compiled variants ({32, 64}).
fn warn_pq_subspaces(_py: Python<'_>, pq_subspaces: usize) -> PyResult<()> {
    // M is a compile-time const generic, so only pre-compiled variants are
    // selectable.  These cover every permissible M for dim=128 (M | dim, M % 4 == 0).
    // The build itself asserts M | token_dim, so an M valid here but not for the
    // actual dim fails at build with a clear panic.  Quality is up to the caller;
    // this only gates on what's compiled.
    const COMPILED_M: [usize; 6] = [4, 8, 16, 32, 64, 128];
    if !COMPILED_M.contains(&pq_subspaces) {
        return Err(PyValueError::new_err(format!(
            "pq_subspaces={pq_subspaces} has no compiled variant; available: {COMPILED_M:?}. \
             (Add a TachiomInner variant + dispatch_pq arm and recompile to support more.)"
        )));
    }
    Ok(())
}

fn require_contiguous_2d<'py, 'a>(
    arr: &'a PyReadonlyArray2<'py, f32>,
    expected_dim: usize,
    arg_name: &str,
) -> PyResult<&'a [f32]>
where
    'py: 'a,
{
    let shape = arr.shape();
    if shape.len() != 2 || shape[1] != expected_dim {
        return Err(PyValueError::new_err(format!(
            "{arg_name} must have shape (n_tokens, {expected_dim}); got {shape:?}"
        )));
    }
    if !arr.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "{arg_name} must be C-contiguous; call np.ascontiguousarray({arg_name}) first"
        )));
    }
    arr.as_slice()
        .map_err(|_| PyValueError::new_err(format!("{arg_name} could not be exposed as a slice")))
}

/// Pad a single-query result vector to length `k` with sentinels.
fn pad_result(result: Vec<(f32, u32)>, k: usize) -> (Vec<f32>, Vec<u32>) {
    let mut scores = Vec::with_capacity(k);
    let mut doc_ids = Vec::with_capacity(k);
    for (s, d) in result.iter().take(k) {
        scores.push(*s);
        doc_ids.push(*d);
    }
    while scores.len() < k {
        scores.push(f32::NEG_INFINITY);
        doc_ids.push(u32::MAX);
    }
    (scores, doc_ids)
}

/// Pad a batch result into rectangular `(n_queries, k)` ndarrays.
fn pad_results_batch(
    results: Vec<Vec<(f32, u32)>>,
    n_queries: usize,
    k: usize,
) -> (ndarray::Array2<f32>, ndarray::Array2<u32>) {
    let mut scores = ndarray::Array2::<f32>::from_elem((n_queries, k), f32::NEG_INFINITY);
    let mut doc_ids = ndarray::Array2::<u32>::from_elem((n_queries, k), u32::MAX);
    for (i, row) in results.into_iter().enumerate() {
        for (j, (s, d)) in row.into_iter().take(k).enumerate() {
            scores[(i, j)] = s;
            doc_ids[(i, j)] = d;
        }
    }
    (scores, doc_ids)
}

// ============================================================================
// Array-based dataset construction
// ============================================================================

/// Build a `TachiomInputDataset` directly from numpy array slices.
///
/// `vectors` is a `u16` array whose bit patterns are the raw IEEE-754 f16 values —
/// the same reinterpretation that `read_f16_npy` performs when reading from disk.
/// Works transparently with memory-mapped numpy arrays.
fn dataset_from_arrays(
    vectors: &PyReadonlyArray2<'_, u16>,
    token_ids_arr: &PyReadonlyArray1<'_, u32>,
    doclens_arr: &PyReadonlyArray1<'_, i32>,
) -> PyResult<(TachiomInputDataset, Vec<usize>)> {
    if !vectors.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "vectors must be C-contiguous; call np.ascontiguousarray(vectors) first",
        ));
    }
    let shape = vectors.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("vectors must be a 2D array [N, dim]"));
    }
    let (n_tokens, dim) = (shape[0], shape[1]);
    let flat_f16: Vec<f16> = vectors
        .as_slice()
        .map_err(|_| PyValueError::new_err("vectors could not be exposed as a slice"))?
        .iter()
        .map(|&bits| f16::from_bits(bits))
        .collect();

    if !token_ids_arr.is_c_contiguous() {
        return Err(PyValueError::new_err("token_ids must be C-contiguous"));
    }
    let token_ids_slice = token_ids_arr
        .as_slice()
        .map_err(|_| PyValueError::new_err("token_ids could not be exposed as a slice"))?;
    if token_ids_slice.len() != n_tokens {
        return Err(PyValueError::new_err(format!(
            "token_ids length ({}) != n_tokens ({})",
            token_ids_slice.len(),
            n_tokens
        )));
    }
    let token_ids: Vec<usize> = token_ids_slice.iter().map(|&x| x as usize).collect();

    if !doclens_arr.is_c_contiguous() {
        return Err(PyValueError::new_err("doclens must be C-contiguous"));
    }
    let doclens_slice = doclens_arr
        .as_slice()
        .map_err(|_| PyValueError::new_err("doclens could not be exposed as a slice"))?;
    for &d in doclens_slice {
        if d < 0 {
            return Err(PyValueError::new_err("doclens must be non-negative"));
        }
    }
    let doclens: Vec<usize> = doclens_slice.iter().map(|&x| x as usize).collect();
    let total: usize = doclens.iter().sum();
    if total != n_tokens {
        return Err(PyValueError::new_err(format!(
            "sum(doclens)={total} != n_tokens={n_tokens}"
        )));
    }

    let encoder = PlainMultiVecQuantizer::<f16>::new(dim);
    let mut offsets: Vec<usize> = Vec::with_capacity(doclens.len() + 1);
    offsets.push(0);
    for &n_tok in &doclens {
        offsets.push(offsets.last().unwrap() + n_tok * dim);
    }

    let dataset = MultiVectorDataset::from_raw(
        flat_f16.into_boxed_slice(),
        offsets.into_boxed_slice(),
        encoder,
    );
    Ok((dataset, token_ids))
}

// ============================================================================
// Input loading
// ============================================================================

/// Load vectors + token_ids + doclens, validate cross-consistency, build the
/// `TachiomInputDataset`, and return it together with the token_ids vector.
fn load_input_dataset(
    vectors_path: &str,
    token_ids_path: &str,
    doclens_path: &str,
) -> PyResult<(TachiomInputDataset, Vec<usize>)> {
    let (flat_f16, dim) = read_f16_npy(vectors_path)?;
    let n_tokens = flat_f16.len() / dim;

    let token_ids = read_token_ids_npy(token_ids_path)?;
    if token_ids.len() != n_tokens {
        return Err(PyValueError::new_err(format!(
            "token_ids length ({}) != n_tokens ({})",
            token_ids.len(),
            n_tokens
        )));
    }

    let doclens = read_doclens_npy(doclens_path)?;
    let total: usize = doclens.iter().sum();
    if total != n_tokens {
        return Err(PyValueError::new_err(format!(
            "sum(doclens)={total} != n_tokens={n_tokens}"
        )));
    }

    let encoder = PlainMultiVecQuantizer::<f16>::new(dim);
    let mut offsets: Vec<usize> = Vec::with_capacity(doclens.len() + 1);
    offsets.push(0);
    for &n_tok in &doclens {
        offsets.push(offsets.last().unwrap() + n_tok * dim);
    }
    if *offsets.last().unwrap() != flat_f16.len() {
        return Err(PyValueError::new_err(format!(
            "sum(doclens)*dim={} != flat_f16.len()={}",
            offsets.last().unwrap(),
            flat_f16.len()
        )));
    }
    let dataset: TachiomInputDataset = MultiVectorDataset::from_raw(
        flat_f16.into_boxed_slice(),
        offsets.into_boxed_slice(),
        encoder,
    );
    Ok((dataset, token_ids))
}

// ── .npy parsing (mirrors the CLI binaries) ────────────────────────────────

fn read_f16_npy(path: &str) -> PyResult<(Vec<f16>, usize)> {
    let mut reader = open_buf(path)?;
    let (shape, elem_size, _) = parse_npy_header(&mut reader)?;
    if shape.len() != 2 {
        return Err(PyValueError::new_err("Expected 2D array for token vectors"));
    }
    if elem_size != 2 {
        return Err(PyValueError::new_err("Expected f16 (2-byte) dtype"));
    }
    let (n_vecs, dim) = (shape[0], shape[1]);
    let mut raw = vec![0u8; n_vecs * dim * 2];
    reader
        .read_exact(&mut raw)
        .map_err(|e| PyIOError::new_err(format!("read_f16_npy: {e}")))?;
    let data = raw
        .chunks_exact(2)
        .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    Ok((data, dim))
}

fn read_token_ids_npy(path: &str) -> PyResult<Vec<usize>> {
    let mut reader = open_buf(path)?;
    let (shape, elem_size, _) = parse_npy_header(&mut reader)?;
    if shape.len() != 1 {
        return Err(PyValueError::new_err("Expected 1D token-ID array"));
    }
    if elem_size != 4 && elem_size != 8 {
        return Err(PyValueError::new_err(format!(
            "Unsupported token-ID elem size: {elem_size}"
        )));
    }
    let n = shape[0];
    let mut raw = vec![0u8; n * elem_size];
    reader
        .read_exact(&mut raw)
        .map_err(|e| PyIOError::new_err(format!("read_token_ids_npy: {e}")))?;
    Ok(match elem_size {
        8 => raw
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as usize)
            .collect(),
        _ => raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()) as usize)
            .collect(),
    })
}

fn read_doclens_npy(path: &str) -> PyResult<Vec<usize>> {
    let mut reader = open_buf(path)?;
    let (shape, elem_size, _) = parse_npy_header(&mut reader)?;
    if shape.len() != 1 {
        return Err(PyValueError::new_err("Expected 1D doclens array"));
    }
    if elem_size != 4 && elem_size != 8 {
        return Err(PyValueError::new_err(format!(
            "Unsupported doclens elem size: {elem_size}"
        )));
    }
    let n = shape[0];
    let mut raw = vec![0u8; n * elem_size];
    reader
        .read_exact(&mut raw)
        .map_err(|e| PyIOError::new_err(format!("read_doclens_npy: {e}")))?;
    Ok(match elem_size {
        8 => raw
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as usize)
            .collect(),
        _ => raw
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as usize)
            .collect(),
    })
}

fn read_assignments_npy(path: &str, expected_len: usize) -> PyResult<Vec<usize>> {
    let mut reader = open_buf(path)?;
    let (shape, elem_size, _) = parse_npy_header(&mut reader)?;
    if shape.len() != 1 {
        return Err(PyValueError::new_err("assignments must be 1D"));
    }
    if shape[0] != expected_len {
        return Err(PyValueError::new_err(format!(
            "assignments length {} != n_tokens {}",
            shape[0], expected_len
        )));
    }
    let n = shape[0];
    let mut raw = vec![0u8; n * elem_size];
    reader
        .read_exact(&mut raw)
        .map_err(|e| PyIOError::new_err(format!("read_assignments_npy: {e}")))?;
    Ok(match elem_size {
        8 => raw
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()) as usize)
            .collect(),
        4 => raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()) as usize)
            .collect(),
        _ => {
            return Err(PyValueError::new_err(format!(
                "unsupported assignments dtype (elem_size={elem_size})"
            )));
        }
    })
}

fn read_f32_2d_npy(path: &str) -> PyResult<(Vec<f32>, usize, usize)> {
    use ndarray::Array2;
    use ndarray_npy::ReadNpyExt;
    let arr: Array2<f32> = Array2::read_npy(open_buf(path)?)
        .map_err(|e| PyIOError::new_err(format!("read_f32_2d_npy: {e}")))?;
    let (rows, cols) = arr.dim();
    Ok((arr.into_raw_vec_and_offset().0, rows, cols))
}

fn open_buf(path: &str) -> PyResult<BufReader<File>> {
    let file = File::open(path).map_err(|e| PyIOError::new_err(format!("{path}: {e}")))?;
    Ok(BufReader::new(file))
}

fn parse_npy_header(reader: &mut impl Read) -> PyResult<(Vec<usize>, usize, usize)> {
    let mut n = 0usize;
    let mut magic = [0u8; 6];
    reader
        .read_exact(&mut magic)
        .map_err(|e| PyIOError::new_err(format!("npy header: {e}")))?;
    n += 6;
    if &magic != b"\x93NUMPY" {
        return Err(PyValueError::new_err("Not a NumPy file"));
    }
    let mut ver = [0u8; 2];
    reader
        .read_exact(&mut ver)
        .map_err(|e| PyIOError::new_err(format!("npy version: {e}")))?;
    n += 2;
    let header_len: usize = if ver[0] == 1 {
        let mut hl = [0u8; 2];
        reader
            .read_exact(&mut hl)
            .map_err(|e| PyIOError::new_err(format!("npy header len: {e}")))?;
        n += 2;
        u16::from_le_bytes(hl) as usize
    } else {
        let mut hl = [0u8; 4];
        reader
            .read_exact(&mut hl)
            .map_err(|e| PyIOError::new_err(format!("npy header len: {e}")))?;
        n += 4;
        u32::from_le_bytes(hl) as usize
    };
    let mut hdr_bytes = vec![0u8; header_len];
    reader
        .read_exact(&mut hdr_bytes)
        .map_err(|e| PyIOError::new_err(format!("npy header body: {e}")))?;
    n += header_len;
    let hdr = String::from_utf8_lossy(&hdr_bytes);
    if hdr.contains("'fortran_order': True") {
        return Err(PyValueError::new_err("Fortran-order arrays not supported"));
    }
    let descr = {
        let prefix = "'descr': '";
        let start = hdr
            .find(prefix)
            .ok_or_else(|| PyValueError::new_err("'descr' not found in npy header"))?
            + prefix.len();
        let rest = &hdr[start..];
        let end = rest
            .find('\'')
            .ok_or_else(|| PyValueError::new_err("descr end quote not found"))?;
        rest[..end].to_string()
    };
    if descr.starts_with('>') {
        return Err(PyValueError::new_err("Big-endian dtype not supported"));
    }
    let elem_size = dtype_elem_size(descr.trim_start_matches(['<', '=', '|']))
        .ok_or_else(|| PyValueError::new_err(format!("Unrecognised dtype '{descr}'")))?;
    let tok = "'shape': (";
    let si = hdr
        .find(tok)
        .ok_or_else(|| PyValueError::new_err("'shape' not found in npy header"))?;
    let rest = &hdr[si + tok.len()..];
    let ei = rest
        .find(')')
        .ok_or_else(|| PyValueError::new_err("shape ')' not found"))?;
    let shape: Vec<usize> = rest[..ei]
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .map_err(|e| PyRuntimeError::new_err(format!("shape parse: {e}")))
        })
        .collect::<PyResult<_>>()?;
    Ok((shape, elem_size, n))
}

fn dtype_elem_size(code: &str) -> Option<usize> {
    match code {
        "i1" | "u1" | "b1" => Some(1),
        "i2" | "u2" | "f2" => Some(2),
        "i4" | "u4" | "f4" => Some(4),
        "i8" | "u8" | "f8" => Some(8),
        _ => None,
    }
}
