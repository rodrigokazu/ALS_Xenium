#!/bin/bash
#SBATCH --job-name=ct_plots_FINAL
#SBATCH --output=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/logs/%x_%j.out
#SBATCH --error=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --partition=batch
#SBATCH --account=mpsnyder
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/run_ct_plots_FINAL.sh
#
# Plotting job for the combined cell typing figures: 8 CPUs, 200G, 4 hours.
#
# The memory looks excessive for drawing pictures. It is not. Scanpy's plotting helpers copy
# the expression matrix more freely than you would like, and this stage was the one that kept
# dying before the dotplot loop was pulled out of it.
# ========================================================================================

# Combined/general cell-type figures + composition for the _FINAL cohort (celltype_plots_FINAL.py).
# Adapted from CellTyping_coarse/code/run_plots.sh. F4 deltas: oak %x_%j logs; caches -> $TMPDIR;
# PYTHONDONTWRITEBYTECODE; exit-code propagated. Env: pertpy_env. Run AFTER the tsne job (afterok)
# -- it reads the celltyped h5ad + X_tsne.npy. Per-sample dotplots come from the SEPARATE
# celltype_dotplots_persample_FINAL job (F6: the fragile in-plots loop was removed).
set -uo pipefail   # NOT -e: capture the python exit code explicitly
CODE=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code
PY=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/pertpy_env/bin/python

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
# route all caches to node-local scratch (never $HOME/OAK -- quota is full)
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_cache"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/xdg"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

echo "start $(date) $(hostname)"
$PY "$CODE/celltype_plots_FINAL.py"
rc=$?
echo "end $(date) exit $rc"
exit $rc
