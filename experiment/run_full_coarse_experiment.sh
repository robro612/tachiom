#!/usr/bin/env bash
# EXPERIMENT (tachiom coarse-mode-experiment branch): measure the retrieval cost of the
# probe-truncated coarse score by comparing, at MATCHED (kc, kd), the current truncated
# scatter score vs the un-truncated centroid-quantized MaxSim (index.full_coarse).
#
# Candidate generation is identical in both arms (same kc probe -> same gathered pool);
# only the score used to rank/prune the pool before the kd cut changes. alpha=null so the
# ONLY pruning that depends on the coarse score is the top-kd cut -> clean isolation.
#
# Runs on the existing LoTTE PGC M=32 index (NO re-clustering, NO rebuild; stages=[retrieve]).
# Grid matches the canonical tachiom sweep (kc x kd) so the full_coarse=false arm reproduces
# the recorded truncated baseline AND sits next to the full_coarse=true arm on the SAME node
# (one srun = one node) -> clean within-run recall-vs-QPS read. RAYON_NUM_THREADS=64 + cpu
# partition (EPYC 7713) matches the conditions the SWEEP_SUMMARY QPS numbers were measured on.
#
# Uses the locally-built experimental tachiom wheel via a uv `--with` overlay, so the shared
# .venv is untouched. If the overlay fails to take, the base tachiom 0.3.0 lacks the
# full_coarse kwarg and batch_search raises TypeError -> the run fails LOUDLY (never silently
# falls back to truncated). To abandon: this is a no-op on the shared env; just delete outputs.
set -u
PROJ=/exp/rjha/pylate-pgc
DATASET="lotte/pooled/dev/search"
MODEL=lateon_regularized
CL="$PROJ/clusterings/lotte_pgc_m4t05"
WHEEL=$(ls "$PROJ"/../tachiom/target/wheels-coarse/tachiom-*.whl 2>/dev/null | head -1)
mkdir -p "$PROJ/logs/scale_val" "$PROJ/results/scale_val"

if [ -z "$WHEEL" ]; then echo "ERROR: experimental wheel not found under tachiom/target/wheels-coarse/"; exit 1; fi
echo "Using experimental wheel: $WHEEL"

IDXDIR="indexes/bench_lightonai_LateOn-regularized_lotte_pooled_dev_search_tachiom_external_lotte_pgc_m4t05_m32"
[ -d "$PROJ/$IDXDIR" ] || { echo "ERROR: lotte PGC M=32 index missing at $IDXDIR"; exit 1; }

LOG="$PROJ/logs/scale_val/lotte_full_coarse_experiment.log"
srun -u -p cpu -t 8:00:00 --cpus-per-task=64 --mem=240G -J "fc_exp" \
  bash -c "cd $PROJ && export RAYON_NUM_THREADS=64 && \
  for kc in 20 40 80; do \
    for kd in 5000 10000 20000 40000; do \
      for fc in false true; do \
        echo \"=== kc=\$kc kd=\$kd full_coarse=\$fc  \$(date) ===\"; \
        uv run --no-sync --with '$WHEEL' python scripts/benchmark_indexes.py \
          model=$MODEL index=tachiom index/clustering=external \
          index.clustering.centroids_path='$CL/centroids.npy' \
          index.clustering.assignments_path='$CL/assignments.npy' \
          index.pq_subspaces=32 index.k_centroids=\$kc \
          index.k_docs_to_score=\$kd index.alpha=null index.full_coarse=\$fc \
          stages='[retrieve]' datasets='[$DATASET]' \
          output.index_folder=indexes \
          output.results_file=results/scale_val/lotte_fc_kc\${kc}_kd\${kd}_fc\${fc}.jsonl \
          output.runs_dir=null; \
      done; \
    done; \
  done" \
  > "$LOG" 2>&1 &
echo "submitted full-coarse experiment; log: $LOG"
echo "results: results/scale_val/lotte_fc_kc{20,40}_kd{1000,2000,5000}_fc{false,true}.jsonl"
