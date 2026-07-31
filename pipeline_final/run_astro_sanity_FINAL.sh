#!/bin/bash
#SBATCH --job-name=astro_sanity_FINAL
#SBATCH --partition=batch
#SBATCH --account=mpsnyder
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/logs/%x_%j.out
#SBATCH --error=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/logs/%x_%j.err
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/run_astro_sanity_FINAL.sh
#
# Wrapper for the astrocyte isolation: 8 CPUs, 128G, 8 hours on batch.
#
# 128G is sized for clustering roughly 1.85M cells, and the plotting stage is what previously
# pushed it over. If it dies late, resubmit with --resume_from pointing at
# astro_isolation_FINAL/clustered_full.h5ad instead of starting again.
# ========================================================================================


# ADAPTED FROM /home/rodrigok/SLURM_jobs/Spatial/VH_isolation/run_astro_sanity.sh.
# _FINAL deltas: input = _FINAL combined h5ad; output = astro_isolation_FINAL; oak
# logs; PYTHONDONTWRITEBYTECODE + caches routed to $TMPDIR (HOME quota is full);
# python exit code propagated to SLURM. Proven params kept (8 cpu / 128G / 8h -- the
# _gausss run cleared the ~1.79M-cell preQC concat at this size).
set -uo pipefail   # NOT -e: capture the python exit code explicitly (see rc below)
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

# route all caches off the (full) HOME quota onto node-local scratch
CACHE="${TMPDIR:-/tmp}/rk_astro_FINAL_${SLURM_JOB_ID:-$$}"
export NUMBA_CACHE_DIR="$CACHE/numba"
export MPLCONFIGDIR="$CACHE/mpl"
export XDG_CACHE_HOME="$CACHE/xdg"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

IN=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/SC_MNcorrected_FINAL_h5ads/ALS_SCXenium_MNcorrected_FINAL_concatenated_offset_allsamples_preQC_20260719.h5ad
OUT=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/astro_isolation_FINAL
CODE=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code
PY=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/pertpy_env/bin/python

echo "host: $(hostname)  start: $(date)  cache: $CACHE"
"$PY" "$CODE/astro_sanity_FINAL.py" \
    --in_h5ad "$IN" --outdir "$OUT" --resolution 1.0
rc=$?
echo "end: $(date)  exit=$rc"
exit $rc
