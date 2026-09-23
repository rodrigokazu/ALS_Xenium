#!/bin/bash
# Execute the 20 v6-relabelled per-sample dual-pass QC notebooks, top to bottom.
# One array task per sample.
#
# Outputs go to runs_v6, NOT runs_srcfixed. runs_srcfixed holds the 2026-08-18
# 02:00 run, which was built on the v2 annotation and is superseded; keeping it
# intact means the v2-vs-v6 label change can be diffed per sample instead of
# being overwritten by the thing that replaced it.
#
# SUBMIT: sbatch run_dualpass_v6_array.sh
#SBATCH --job-name=dualpass_v6
#SBATCH --partition=batch
#SBATCH --account=mpsnyder
#SBATCH --array=0-19%7
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=05:00:00
#SBATCH --output=/home/rodrigok/SLURM_jobs/Spatial/dualpass_rerun/logs/v6_%A_%a.out
#SBATCH --error=/home/rodrigok/SLURM_jobs/Spatial/dualpass_rerun/logs/v6_%A_%a.err
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/2_qc/dual_pass/run_dualpass_v6_array.sh
#
# The run that produced the canonical QC: array 0-19%7, 8 CPUs, 90G, 5 hours per sample,
# job 52381454 on 2026-08-18. Outputs go to Novae_IND_QC/runs_v6. runs_srcfixed holds an
# earlier run on the v2 annotation and is superseded.
#
# The funnel from the 20 traces: 1,694,362 cells, 1,128,650 after the depth cut, 1,103,563
# after the neighbour cut, 1,097,669 after the domain verdicts.
# ========================================================================================

echo "host=$(hostname) task=${SLURM_ARRAY_TASK_ID} start=$(date)"
source /home/rodrigok/.bashrc          # must precede set -u (unbound vars in bashrc)
set -u

export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/tmp
export NUMBA_CACHE_DIR=/tmp/nb_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export MPLCONFIGDIR=/tmp/mpl_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export XDG_CACHE_HOME=/tmp/xdg_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8

SAMPLES=(SD00614_BG SD01015_BG SD01115_BG SD01320_BG SD01413_BA SD01616_BI \
         SD01620_BI SD01623_BI SD01915_BG SD01922_BI SD01923_BI SD02022_BI \
         SD02622_BI SD02818_BG SD02913_BA SD03522_BG SD03614_BG SD03914_BG \
         SD04219_BI SD05413_BG)
S=${SAMPLES[${SLURM_ARRAY_TASK_ID}]}
TAG=${S//_/}

SPATIAL=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial
STAGE=$SPATIAL/Novae_IND_QC/runs_v6
mkdir -p "$STAGE/stage" "$STAGE/figures/$S"

export DUALPASS_RUN_DIR=$STAGE
export DUALPASS_TRACE_DIR=$STAGE
export DUALPASS_PASS1_H5AD=$STAGE/stage/ALS_SCXenium_${TAG}_postQC.h5ad
export DUALPASS_PASS2_H5AD=$STAGE/stage/ALS_SCXenium_${TAG}_pass2.h5ad
export DUALPASS_FIGDIR=$STAGE/figures/$S

CODE=/home/rodrigok/SLURM_jobs/Spatial/dualpass_rerun
PY=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/curio_spatial/bin/python

cd "$CODE"
echo "sample=$S tag=$TAG"
echo "pass1=$DUALPASS_PASS1_H5AD"
echo "pass2=$DUALPASS_PASS2_H5AD"
echo "relabel=v6 snapshot=neuron_classification_tdp_v6__snapshot_20260818_1649.parquet"
$PY -W ignore run_dualpass_src.py "$S"; rc=$?
echo "done task=${SLURM_ARRAY_TASK_ID} sample=$S end=$(date) rc=$rc"
exit $rc
