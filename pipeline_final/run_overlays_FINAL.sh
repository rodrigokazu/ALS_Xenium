#!/bin/bash
#SBATCH --job-name=Xenium_overlays_FINAL
#SBATCH --output=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/logs/%x_%A_%a.out
#SBATCH --error=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/logs/%x_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch
#SBATCH --account=mpsnyder
#SBATCH --array=0-19%6
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/run_overlays_FINAL.sh
#
# Overlay rendering array: --array=0-19%6, 4 CPUs, 64G, 2 hours per sample.
#
# Modest resources because each task handles one section and the DAPI overview is downsampled.
# The full-resolution crop is small in area.
#
# Depends on the joint Novae run through afterok. Expect centroid-based maps rather than
# filled polygons until the boundaries are re-exported.
# ========================================================================================

# ^ 20 samples, one per array task; throttle to 6 concurrent. Indices map to
#   final_config.discover_samples() sorted labels via --array-index (run_overlays_batch_FINAL.py).
#   DO NOT SUBMIT until the Novae _FINAL per-sample h5ads exist (else every task cleanly SKIPs).

set -uo pipefail
CODE=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code
PY=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/xenium_vistools/bin/python

# genes for the transcript overlay (STMN2 + its cryptic exon = the vulnerability story).
GENES="${1:-STMN2,CE_STMN2}"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
# route all caches to node-local scratch (never $HOME -- quota is full)
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_cache"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/xdg"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

echo "Job ${SLURM_JOB_NAME} array task ${SLURM_ARRAY_TASK_ID} started: $(date) on $(hostname)"
echo "genes=${GENES}"
$PY "$CODE/run_overlays_batch_FINAL.py" --array-index "${SLURM_ARRAY_TASK_ID:-0}" --genes "$GENES"
rc=$?
echo "Job finished: $(date) (exit $rc)"
exit $rc
