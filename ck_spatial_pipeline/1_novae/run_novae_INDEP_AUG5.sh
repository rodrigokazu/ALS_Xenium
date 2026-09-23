#!/bin/bash
#SBATCH --job-name=Novae_INDEP_AUG5
#SBATCH --output=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/logs/%x_%A_%a.out
#SBATCH --error=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/logs/%x_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --partition=batch
#SBATCH --account=mpsnyder
#SBATCH --gres=gpu:qrtx4000:1
#SBATCH --array=0-19%3
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/1_novae/run_novae_INDEP_AUG5.sh
#
# Array 0-19 for the AUG5 driver, one sample and one fresh model per task. Same GPU and
# memory request as run_novae_INDEPENDENT_FINAL.sh. Report-only lineage.
# ========================================================================================
# One sample per array task, a fresh Novae model each. Same resources as the _FINAL runner.
# %3 rather than %5: only three qrtx4000 GPUs exist on SCG, so a higher cap just parks tasks
# in the queue holding a 96G reservation.
#
# The AUG5 h5ads have no counts layer and store X as CSC. Both are handled downstream:
# process_one() sets layers['counts'] = X.copy() before Novae normalises X in place, and
# ps_io_FINAL_fix dispatches on the on-disk encoding-type when the report reads it back.

source /home/rodrigok/.bashrc

module load python/3.11.1 gcc/13.3.0
source /oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/novae_env/bin/activate

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

# Per-array-task cache dirs on node-local /tmp so concurrent tasks never share one.
export TMPDIR="${TMPDIR:-/tmp/${USER}/${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}}"
mkdir -p "$TMPDIR"
export NUMBA_CACHE_DIR="$TMPDIR/numba"
export MPLCONFIGDIR="$TMPDIR/mpl"
export XDG_CACHE_HOME="$TMPDIR/xdg"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

echo "Job started: $(date)"
echo "Node: $(hostname)  ARRAY_JOB=${SLURM_ARRAY_JOB_ID}  TASK=${SLURM_ARRAY_TASK_ID}"
nvidia-smi || echo "nvidia-smi unavailable"
which python
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import novae, scanpy, anndata; print('novae', novae.__version__, '| scanpy', scanpy.__version__, '| anndata', anndata.__version__)"

cd /oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code
python novae_INDEP_AUG5.py "$SLURM_ARRAY_TASK_ID"
rc=$?

echo "Job finished: $(date) (python exit code = $rc)"
exit $rc
