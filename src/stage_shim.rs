//! Compile-time gated stage timers for search.
//!
//! With `--features profile`, `stage!` records into `stage-profile`'s global
//! buffer. Without the feature, `stage!` expands to a no-op so `_stage.set(...)`
//! still type-checks at call sites.
//!
//! # Parallel batch caveat
//!
//! The buffer is process-global, so stages recorded on Rayon workers during a
//! multi-query `batch_search` *are* drained. They arrive as repeated samples per
//! stage in completion order, though, so per-query latency is not recoverable:
//! use one query per begin→`batch_search`→take (PyLate
//! `e2e_profile_batch_size=1`) for percentiles. See `stage-profile` README.

use std::collections::HashMap;

/// Span-compatible stage sample returned to PyO3.
#[derive(Debug, Clone, Default)]
pub struct StageSample {
    pub name: String,
    pub dur_ns: u64,
    pub meta: HashMap<String, f64>,
}

/// No-op guard used when the `profile` feature is off.
#[cfg(not(feature = "profile"))]
pub struct NoopGuard;

#[cfg(not(feature = "profile"))]
impl NoopGuard {
    #[inline]
    pub fn set(&mut self, _key: &'static str, _value: f64) -> &mut Self {
        self
    }
}

#[cfg(feature = "profile")]
macro_rules! stage {
    ($name:expr) => {
        ::stage_profile::StageGuard::new($name)
    };
}

#[cfg(not(feature = "profile"))]
macro_rules! stage {
    ($name:expr) => {
        $crate::stage_shim::NoopGuard
    };
}

/// Whether this build can record stages at all (i.e. `--features profile`).
///
/// The PyO3 hooks are exported unconditionally so callers can probe a build
/// rather than guess from `hasattr`; without the feature they are no-ops and
/// this returns `false`.
#[inline]
pub fn supported() -> bool {
    cfg!(feature = "profile")
}

#[inline]
pub fn begin() {
    #[cfg(feature = "profile")]
    stage_profile::begin();
}

#[inline]
pub fn take() -> Vec<StageSample> {
    #[cfg(feature = "profile")]
    {
        return stage_profile::take()
            .into_iter()
            .map(|s| StageSample {
                name: s.name,
                dur_ns: s.dur_ns,
                meta: s.meta,
            })
            .collect();
    }
    #[cfg(not(feature = "profile"))]
    Vec::new()
}
