#!/bin/bash
#SBATCH --job-name=ct_dotpersample_FINAL
#SBATCH --output=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/logs/%x_%j.out
#SBATCH --error=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/logs/%x_%j.err
#SBATCH --time=01:30:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch
#SBATCH --account=mpsnyder
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/run_ct_dotpersample_FINAL.sh
#
# Runs the per-sample dotplots: 4 CPUs, 120G, 90 minutes. Separate from the main plots job
# precisely so its previous failures could not take the combined figures down with them.
# ========================================================================================

# Per-sample marker dotplots for the _FINAL cohort, dense-submatrix / scipy-int32-safe
# (celltype_dotplots_persample_FINAL.py). Adapted from CellTyping_coarse/code/run_dotpersample.sh.
# F4 deltas: oak %x_%j logs; caches -> $TMPDIR; PYTHONDONTWRITEBYTECODE; exit-code propagated.
# Env: pertpy_env. Run AFTER the pipeline (afterok) -- reads the celltyped h5ad.
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
$PY "$CODE/celltype_dotplots_persample_FINAL.py"
rc=$?
echo "end $(date) exit $rc"
exit $rc
