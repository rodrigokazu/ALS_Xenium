#!/bin/bash
#SBATCH --job-name=VH_concat_FINAL
#SBATCH --partition=batch
#SBATCH --account=mpsnyder
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/logs/%x_%j.out
#SBATCH --error=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/logs/%x_%j.err
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/run_VH_concatenate_FINAL.sh
#
# SLURM wrapper for the cohort concat. 8 CPUs, 128G, 8 hours on the batch partition under the
# mpsnyder account. Generous: the July run finished in under four minutes.
#
# The env dance matters. PYTHONNOUSERSITE=1 keeps a stale ~/.local anndata from shadowing the
# pinned stack. That was a real failure, not a theoretical one. Caches are pushed to /tmp
# because the SCG home quota is full, and logs go to the OAK staged code directory for the
# same reason.
#
# The wrapper captures the python return code and exits with it. Do not drop that. An earlier
# version ended on a bare echo. Echo returns 0, so a python crash got reported by SLURM as
# COMPLETED and we trusted an object that had never been written.
# ========================================================================================

#
# Concatenate Marcel's _final segmentation into the PRE-QC combined + per-sample h5ads.
# STAGING: do NOT submit until Marcel's _final MN annotation lands (the script degrades
# gracefully to is_MN=False today, but the whole point of the re-run is the MN join).
#
# To force is_MN=False even if a (human-confirmed) row-count-mismatched annotation is
# present, export MN_REQUIRE_FINAL_PROVENANCE=0 before sbatch (default 1 = HARD guard).

set -uo pipefail   # NOT -e: we capture and propagate the python exit code explicitly.

CODE=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code
PY=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/pertpy_env/bin/python

# env guards (pertpy_env scanpy import breaks via a stale ~/.local anndata)
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

# route all caches to node-local tmp (never $HOME -- quota is full)
export TMPDIR="${TMPDIR:-/tmp}"
export NUMBA_CACHE_DIR="$TMPDIR/nb_${SLURM_JOB_ID:-$$}"
export MPLCONFIGDIR="$TMPDIR/mpl_${SLURM_JOB_ID:-$$}"
export XDG_CACHE_HOME="$TMPDIR/xdg_${SLURM_JOB_ID:-$$}"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

echo "host=$(hostname)  start=$(date)"
echo "python=$PY"
echo "MN_REQUIRE_FINAL_PROVENANCE=${MN_REQUIRE_FINAL_PROVENANCE:-1}"

"$PY" "$CODE/VH_concatenate_FINAL.py"
rc=$?

echo "end=$(date)  rc=$rc"
exit $rc
