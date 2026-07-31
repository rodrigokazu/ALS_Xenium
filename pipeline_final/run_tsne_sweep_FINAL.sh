#!/bin/bash
#SBATCH --job-name=tsne_sweep_FINAL
#SBATCH --partition=batch
#SBATCH --account=mpsnyder
#SBATCH --gres=gpu:qrtx4000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --array=0-23%4
#SBATCH --output=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/logs/%x_%A_%a.out
#SBATCH --error=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/logs/%x_%A_%a.err
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/run_tsne_sweep_FINAL.sh
#
# The sweep array: --array=0-23%4, one qrtx4000 each, 8 CPUs, 48G, one hour per task.
#
# Capped at four concurrent because there are only three qrtx4000 nodes on SCG and
# monopolising them makes no friends. Needs LD_LIBRARY_PATH set to the environment's lib
# directory so the cuML GPU libraries resolve. The wrapper handles it.
# ========================================================================================


# ADAPTED FROM run_tsne_sweep.sh. _FINAL deltas: BASE=astro_isolation_FINAL; oak logs
# (array uses %A_%a so tasks never clobber); PYTHONDONTWRITEBYTECODE + caches ->
# $TMPDIR; python exit code propagated. ADDED LD_LIBRARY_PATH=$ENV/lib -- the proven
# gotcha for the opentsne_gpu env's direct python (cuML/GPU libs); the _gausss runner
# omitted it and relied on the openTSNE CPU fallback. qrtx4000 is the free GPU.
set -uo pipefail   # NOT -e: capture the python exit code explicitly
ENV=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/opentsne_gpu
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"

CACHE="${TMPDIR:-/tmp}/rk_tsnesweep_FINAL_${SLURM_ARRAY_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}"
export NUMBA_CACHE_DIR="$CACHE/numba"
export MPLCONFIGDIR="$CACHE/mpl"
export XDG_CACHE_HOME="$CACHE/xdg"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

BASE=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/astro_isolation_FINAL
CODE=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code
PY="$ENV/bin/python"

echo "host $(hostname)  task ${SLURM_ARRAY_TASK_ID}  gpu ${CUDA_VISIBLE_DEVICES:-none}  $(date)  cache: $CACHE"
nvidia-smi -L || true
"$PY" "$CODE/tsne_sweep_run_FINAL.py" \
    --in_h5ad "$BASE/tsne_sweep/sweep_input.h5ad" \
    --outdir  "$BASE/tsne_sweep"
rc=$?
echo "task ${SLURM_ARRAY_TASK_ID} done: $(date)  exit=$rc"
exit $rc
