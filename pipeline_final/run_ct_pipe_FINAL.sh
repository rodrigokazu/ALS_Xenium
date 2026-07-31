#!/bin/bash
#SBATCH --job-name=ct_pipe_FINAL
#SBATCH --output=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/logs/%x_%j.out
#SBATCH --error=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --partition=batch
#SBATCH --account=mpsnyder
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/run_ct_pipe_FINAL.sh
#
# The heavy one: 16 CPUs, 256G, 12 hours. Clustering 1.85M cells across 480 genes is what that
# memory is for, and the peak is in the neighbour graph rather than the PCA.
#
# Every other cell typing job depends on this finishing, so it sits at the head of that branch
# of the DAG.
# ========================================================================================

# Coarse cell-typing pipeline for the MN-corrected _FINAL cohort (celltype_pipeline_FINAL.py).
# Adapted from CellTyping_coarse/code/run_pipe.sh. F4 deltas: oak %x_%j logs; caches -> $TMPDIR
# (HOME quota is full); PYTHONDONTWRITEBYTECODE; exit-code propagated. Env: pertpy_env.
# DO NOT SUBMIT until cfg.COMBINED_H5AD (the _final concat) exists.
set -uo pipefail   # NOT -e: capture the python exit code explicitly
CODE=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code
PY=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/pertpy_env/bin/python

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=16
# route all caches to node-local scratch (never $HOME/OAK -- quota is full)
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_cache"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/xdg"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

echo "start $(date) $(hostname)"
$PY "$CODE/celltype_pipeline_FINAL.py"
rc=$?
echo "end $(date) exit $rc"
exit $rc
