#!/bin/bash
# run_tx_array.sh -- transcript-level QC (%QV>=20, %assigned, SNR, unassigned map)
#SBATCH --job-name=txqc
#SBATCH --partition=batch
#SBATCH --account=mpsnyder
#SBATCH --array=0-19%8
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/rodrigok/SLURM_jobs/Spatial/dualpass_rerun/logs/tx_%A_%a.out
#SBATCH --error=/home/rodrigok/SLURM_jobs/Spatial/dualpass_rerun/logs/tx_%A_%a.err
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/2_qc/dual_pass/run_tx_array.sh
#
# Array wrapper for tx_metrics.py, one sample per task, on the curio_spatial env.
# ========================================================================================

echo "host=$(hostname) task=${SLURM_ARRAY_TASK_ID} start=$(date)"
source /home/rodrigok/.bashrc
set -u
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/tmp
export MPLCONFIGDIR=/tmp/mpltx_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export XDG_CACHE_HOME=/tmp/xdgtx_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

SAMPLES=(SD00614_BG SD01015_BG SD01115_BG SD01320_BG SD01413_BA SD01616_BI \
         SD01620_BI SD01623_BI SD01915_BG SD01922_BI SD01923_BI SD02022_BI \
         SD02622_BI SD02818_BG SD02913_BA SD03522_BG SD03614_BG SD03914_BG \
         SD04219_BI SD05413_BG)
S=${SAMPLES[${SLURM_ARRAY_TASK_ID}]}

CODE=/home/rodrigok/SLURM_jobs/Spatial/dualpass_rerun
PY=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/curio_spatial/bin/python
cd "$CODE"
$PY -W ignore tx_metrics.py "$S"; rc=$?
echo "done task=${SLURM_ARRAY_TASK_ID} sample=$S end=$(date) rc=$rc"
exit $rc
