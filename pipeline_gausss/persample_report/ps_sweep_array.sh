#!/bin/bash
#SBATCH --job-name=ps_sweep
#SBATCH --partition=batch
#SBATCH --account=mpsnyder
#SBATCH --array=0-19%10
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/rodrigok/SLURM_jobs/Spatial/VH_isolation/persample_report/logs/ps_sweep_%A_%a.out
#SBATCH --error=/home/rodrigok/SLURM_jobs/Spatial/VH_isolation/persample_report/logs/ps_sweep_%A_%a.err
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_gausss/persample_report/ps_sweep_array.sh
#
# Runs the sweep report for all 20 samples: --array=0-19%10, 6 CPUs, 64G, one hour each.
#
# Uses the full stack module load plus the conda environment, since it needs both scanpy and
# scikit-learn for the adjusted Rand index.
# ========================================================================================


echo "host=$(hostname) array_task=${SLURM_ARRAY_TASK_ID} start=$(date)"
source /home/rodrigok/.bashrc
module load python/3.11.1 gcc/13.3.0            # full stack; xenium_vistools=conda-no-activate, so we skip it
set -eo pipefail
cd /home/rodrigok/SLURM_jobs/Spatial/VH_isolation/persample_report
python ps_sweep.py --idx "${SLURM_ARRAY_TASK_ID}"
echo "done task=${SLURM_ARRAY_TASK_ID} end=$(date)"
