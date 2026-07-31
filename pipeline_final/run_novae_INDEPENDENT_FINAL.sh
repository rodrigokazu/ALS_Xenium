#!/bin/bash
#SBATCH --job-name=Novae_INDEP_FINAL
#SBATCH --output=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/logs/%x_%A_%a.out
#SBATCH --error=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/logs/%x_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --partition=batch
#SBATCH --account=mpsnyder
#SBATCH --gres=gpu:qrtx4000:1
#SBATCH --array=0-19%5
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/run_novae_INDEPENDENT_FINAL.sh
#
# Array wrapper for the per-sample Novae fits: --array=0-19%5, so 20 tasks with at most five
# resident at once. One qrtx4000 each, 96G, 8 hours.
#
# The %5 throttle is there to be a good citizen on a shared GPU node and not because the work
# needs serialising. Widen it if the node is quiet and you are in a hurry.
#
# Verifies torch, CUDA and the novae stack per task before doing anything expensive. There is
# a matching aggregate step in the footer wired afterany, so a single failed sample still lets
# the roll-up run over the ones that worked.
# ========================================================================================

# SLURM JOB ARRAY: ONE sample per task (SLURM_ARRAY_TASK_ID = 0..19), a FRESH Novae model
# each. %5 caps concurrency; note only ~3 qrtx4000 GPUs exist on SCG, so effective
# concurrency is GPU-bound (extra tasks just queue). Each sample is small (~50-200k cells)
# so 8GB qrtx4000 + 8h is ample; mem=96G is generous and also covers the (exceptional)
# combined-subset fallback that loads the full concat when the 20 per-sample files are
# absent. cpus-per-task=8 matches NUM_WORKERS=8 (Novae starves the GPU with 0 workers).
# Log naming uses %A_%a (array-job id + task id) so each task has its own log in the
# required .../final_rerun_code/logs/ dir. Per-task failures are isolated (the array
# continues); the aggregate step lists any missing sample as "failed".
#
# *** STAGING ONLY -- DO NOT SUBMIT until the per-sample (or combined) _FINAL h5ads exist.
#     is_MN degrades gracefully (all-False -> placeholder MN outputs) so the niche/smear
#     QC maps are valid even before Marcel's _final MN annotation lands. ***

source /home/rodrigok/.bashrc

module load python/3.11.1 gcc/13.3.0
source /oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/novae_env/bin/activate

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

# Route all caches to node-local /tmp (HOME quota full). Use a per-ARRAY-TASK dir so
# concurrent tasks never share a cache dir.
export TMPDIR="${TMPDIR:-/tmp/${USER}/${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}}"
mkdir -p "$TMPDIR"
export NUMBA_CACHE_DIR="$TMPDIR/numba"
export MPLCONFIGDIR="$TMPDIR/mpl"
export XDG_CACHE_HOME="$TMPDIR/xdg"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

echo "Job started: $(date)"
echo "Node: $(hostname)  ARRAY_JOB=${SLURM_ARRAY_JOB_ID}  TASK=${SLURM_ARRAY_TASK_ID}"
echo "=== GPU ==="
nvidia-smi || echo "nvidia-smi unavailable"
echo "=== which python ==="
which python
python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"
python -c "import novae, scanpy, anndata; print('novae', novae.__version__, '| scanpy', scanpy.__version__, '| anndata', anndata.__version__)"

# Run from the staged code dir so 'import final_config' resolves (the .py also inserts it).
cd /oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code
python /oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/novae_INDEPENDENT_FINAL.py "$SLURM_ARRAY_TASK_ID"
rc=$?

echo "Job finished: $(date) (python exit code = $rc)"
exit $rc

# ============================================================================
# TINY AGGREGATE NOTE (run ONCE, AFTER the whole array finishes)
# ----------------------------------------------------------------------------
# Each array task writes only its own per-sample files (per_sample/<LABEL>__niches_
# independent.h5ad, plots/<LABEL>__niches_n{4,6,8,10}.png, summary/persample_rows/
# <LABEL>__rows.csv, summary/persample/<LABEL>__manifest.json) -- no shared-file races.
# Stitch them into summary/persample_independent_summary.csv + summary/run_manifest.json
# (lists any missing/failed sample) with ONE lightweight, GPU-free command, e.g.:
#
#   # as a dependent job after the array (recommended):
#   sbatch --dependency=afterany:<ARRAY_JOBID> --job-name=Novae_INDEP_FINAL_agg \
#          --output=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/logs/%x_%j.out \
#          --error=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/logs/%x_%j.err \
#          --time=00:20:00 --mem=8G --cpus-per-task=1 --partition=batch --account=mpsnyder \
#          --wrap 'module load python/3.11.1 gcc/13.3.0; source /oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/novae_env/bin/activate; export PYTHONNOUSERSITE=1; cd /oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code; python novae_INDEPENDENT_FINAL.py aggregate'
#
#   # or interactively on a login/compute node (seconds, no GPU):
#   cd /oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code
#   python novae_INDEPENDENT_FINAL.py aggregate
#
# (DO NOT submit anything now -- this is staging only.)
# ============================================================================
