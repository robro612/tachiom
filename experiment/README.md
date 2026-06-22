# full-coarse experiment (`coarse-mode-experiment` branch)

Measures the retrieval cost of tachiom's **probe-truncated** coarse score by adding an
**un-truncated** centroid-quantized MaxSim rescoring of the gathered candidate pool, before
the `k_docs_to_score` cut (`index.full_coarse` / `batch_search(full_coarse=True)`).

## Verdict — NOT worth it (do not merge)

LoTTE PGC M=32, EPYC-64, alpha=null. Full-coarse is **dominated on the recall↔QPS Pareto
frontier** (1.3–2.4× slower than truncated at every matched R@100 up to 0.828). Structural why:
- Within each `kc`, full-coarse **saturates instantly** (R@100 flat across all `kd`): the ceiling
  is set by the probe `kc`, not `kd`.
- Truncated **climbs with `kd` to the same ceiling** (e.g. kc20/kd40000 ties full's 0.817) at
  ~2× the QPS — PQ-reranking a deeper `kd` is cheaper than centroid-rescoring the whole pool.
- Full's only edge is a ~+0.0025 higher absolute ceiling (0.830 vs 0.828) at the slowest corner
  (kc80, 10–12 QPS) — marginal and likely beatable by truncated at `kd>40000`.

Full table + frontier: `pylate-pgc/results/scale_val/SWEEP_SUMMARY.md` and
`pylate-pgc/results/scale_val/lotte_fc_kc*_kd*_fc*.jsonl`.

## Components on this branch
- `src/tachiom.rs`, `src/python.rs`: `full_coarse` flag on `search`/`batch_search` +
  `recompute_coarse_full` (rescores the gathered pool, batched).
- `Cargo.toml` `[patch]`: redirects `vectorium` to the local fork `/exp/rjha/vectorium-coarse-fork`
  (branch `coarse-only`), which adds `compute_centroid_distance` + `compute_centroid_distance_batch`
  to `MultiVecTwoLevelPQQueryEvaluator`.
- `experiment/pylate-pgc-harness.patch`: pylate-pgc plumbing (wrapper `full_coarse` param,
  benchmark passthrough, `tachiom.yaml` key) — kept as a patch because it must NOT live in
  pylate-pgc main (it would break the base tachiom 0.3.0 in the shared venv).
- `experiment/run_full_coarse_experiment.sh`: the `kc × kd × {trunc,full}` grid runner.

## To re-run
1. `git -C /exp/rjha/tachiom checkout coarse-mode-experiment`
2. ensure `/exp/rjha/vectorium-coarse-fork` is on branch `coarse-only`
3. build wheel: `uv run --no-sync --with maturin maturin build --release --out target/wheels-coarse`
4. in pylate-pgc: `git apply /exp/rjha/tachiom/experiment/pylate-pgc-harness.patch`
5. run `experiment/run_full_coarse_experiment.sh` (overlays the wheel via `uv run --with`)
