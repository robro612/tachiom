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
- `Cargo.toml` `[patch]`: redirects `vectorium` to the pushed fork
  `github.com/robro612/vectorium` (branch `coarse-only`), which adds `compute_centroid_distance`
  + `compute_centroid_distance_batch` to `MultiVecTwoLevelPQQueryEvaluator`. `Cargo.lock` pins the
  exact commit, so a fresh clone resolves it from git — no local path required.
- `experiment/pylate-pgc-harness.patch`: pylate-pgc plumbing (wrapper `full_coarse` param,
  benchmark passthrough, `tachiom.yaml` key) — kept as a patch because it must NOT live in
  pylate-pgc main (it would break the base tachiom 0.3.0 in the shared venv).
- `experiment/vectorium-coarse-only.patch`: the vectorium diff alone, applicable onto a clean
  `TusKANNy/vectorium` checkout (belt-and-suspenders; the `[patch]` git source is the primary path).
- `experiment/run_full_coarse_experiment.sh`: the `kc × kd × {trunc,full}` grid runner (LoTTE PGC M=32).

## To re-pull and reproduce (from scratch, no local paths)
```bash
# 1. tachiom experiment branch
git clone -b coarse-mode-experiment https://github.com/robro612/tachiom.git
cd tachiom
# (the [patch] pulls robro612/vectorium@coarse-only from git automatically)

# 2. build the experimental wheel (enables index.full_coarse / batch_search(full_coarse=True))
uv run --no-sync --with maturin maturin build --release --out target/wheels-coarse

# 3. in the pylate-pgc checkout, apply the harness (wrapper/benchmark/yaml plumbing)
cd /path/to/pylate-pgc
git apply /path/to/tachiom/experiment/pylate-pgc-harness.patch

# 4. run the grid (overlays the wheel via uv --with; does NOT touch the shared venv)
WHEEL=/path/to/tachiom/target/wheels-coarse/tachiom-*.whl \
  bash /path/to/tachiom/experiment/run_full_coarse_experiment.sh
```
To abandon again afterwards: revert the pylate-pgc harness (`git checkout HEAD -- pylate/indexes/tachiom.py
scripts/benchmark_indexes.py conf/eval/index/tachiom.yaml`) so the base tachiom 0.3.0 keeps working.
