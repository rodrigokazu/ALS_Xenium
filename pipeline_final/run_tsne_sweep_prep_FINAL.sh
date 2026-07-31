#!/bin/bash
#SBATCH --job-name=tsne_prep_FINAL
#SBATCH --partition=batch
#SBATCH --account=mpsnyder
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/logs/%x_%j.out
#SBATCH --error=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/logs/%x_%j.err
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/run_tsne_sweep_prep_FINAL.sh
#
# Small preparation job: 4 CPUs, 32G, one hour. Runs once and must finish before the sweep
# array starts, since all 24 tasks read the single subsample it writes.
# ========================================================================================


# ADAPTED FROM run_tsne_sweep_prep.sh. _FINAL deltas: BASE=astro_isolation_FINAL;
# oak logs; PYTHONDONTWRITEBYTECODE + caches -> $TMPDIR; python exit code propagated.
set -uo pipefail   # NOT -e: capture the python exit code explicitly
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

CACHE="${TMPDIR:-/tmp}/rk_tsneprep_FINAL_${SLURM_JOB_ID:-$$}"
export NUMBA_CACHE_DIR="$CACHE/numba"
export MPLCONFIGDIR="$CACHE/mpl"
export XDG_CACHE_HOME="$CACHE/xdg"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

BASE=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/astro_isolation_FINAL
CODE=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code
PY=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/pertpy_env/bin/python

echo "host: $(hostname)  start: $(date)  cache: $CACHE"
"$PY" "$CODE/tsne_sweep_prep_FINAL.py" \
    --ckpt "$BASE/clustered_full.h5ad" \
    --out  "$BASE/tsne_sweep/sweep_input.h5ad" \
    --n 200000
rc=$?
echo "prep done: $(date)  exit=$rc"
exit $rc
