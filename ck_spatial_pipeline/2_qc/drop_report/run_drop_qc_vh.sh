#!/bin/bash
#SBATCH --job-name=drop_qc_vh
#SBATCH --account=mpsnyder
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/rodrigok/SLURM_jobs/Spatial/logs/drop_qc_vh_%j.out
#SBATCH --error=/home/rodrigok/SLURM_jobs/Spatial/logs/drop_qc_vh_%j.err
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/2_qc/drop_report/run_drop_qc_vh.sh
#
# Wrapper for drop_qc_vh.py, 4 CPUs, 64G, 1 hour, nichecompass_env. Job 52579301.
# ========================================================================================
set -euo pipefail
module load python/3.11.1 gcc/13.3.0
export PYTHONNOUSERSITE=1
cd /home/rodrigok/SLURM_jobs/Spatial
/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/nichecompass_env/bin/python drop_qc_vh.py
